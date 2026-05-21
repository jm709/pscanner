"""Subgraph-driven copy-trade collector (#152).

Polls the Polymarket V2 subgraph for ``orderFilledEvents`` involving any
watchlisted wallet and emits a ``subgraph_copy`` :class:`Alert` for every
position-increasing trade. Booking happens downstream via
:class:`pscanner.strategies.evaluators.subgraph_copy.SubgraphCopyEvaluator`.

Lifecycle is owned by the daemon scheduler — restart-on-crash, shared
``stop_event`` for clean shutdown.
"""

from __future__ import annotations

import asyncio  # noqa: F401
import json
import os  # noqa: F401
import sqlite3  # noqa: F401
import time  # noqa: F401
from typing import Any, Final

import structlog

from pscanner.alerts.models import Alert  # noqa: F401
from pscanner.alerts.protocol import IAlertSink  # noqa: F401
from pscanner.collectors.watchlist import WatchlistRegistry  # noqa: F401
from pscanner.config import SubgraphTradeCollectorConfig  # noqa: F401
from pscanner.corpus.repos import AssetIndexRepo  # noqa: F401
from pscanner.poly.gamma import GammaClient  # noqa: F401
from pscanner.poly.ids import AssetId  # noqa: F401
from pscanner.poly.subgraph import SubgraphClient  # noqa: F401
from pscanner.poly.token_resolver import resolve_token  # noqa: F401
from pscanner.store.repo import (  # noqa: F401
    MarketCacheRepo,
    SubgraphWatchStateRepo,
)
from pscanner.util.clock import Clock, RealClock  # noqa: F401

_LOG = structlog.get_logger(__name__)

SUBGRAPH_ID: Final[str] = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
DETECTOR_TAG: Final[str] = "subgraph_copy"
ALERT_KEY_PREFIX: Final[str] = "subgraph"
PAGE_SIZE: Final[int] = 1000

_GRAPHQL_QUERY: Final[str] = f"""
{{
  orderFilledEvents(
    where: $where
    first: {PAGE_SIZE}
    orderBy: timestamp
    orderDirection: asc
  ) {{
    transactionHash
    timestamp
    maker {{ id }}
    taker {{ id }}
    market {{ id }}
    tokenId
    side
    price
    size
  }}
  _meta {{ block {{ number timestamp }} }}
}}
"""


def _compute_copy_direction(
    maker: str, taker: str, side: int, watchlist: set[str]
) -> str:
    """Return ``"BUY"`` iff the watchlist wallet's position increases.

    Subgraph ``side``: 0=BUY, 1=SELL on the order's direction.
    Maker placed the resting order; taker hit it from the opposite side.

    - watchlist == maker AND side == 0 -> maker accumulates -> BUY
    - watchlist == maker AND side == 1 -> maker reduces      -> SKIP
    - watchlist == taker AND side == 0 -> taker sold         -> SKIP
    - watchlist == taker AND side == 1 -> taker bought       -> BUY
    """
    maker_l = maker.lower()
    taker_l = taker.lower()
    if maker_l in watchlist and side == 0:
        return "BUY"
    if taker_l in watchlist and side == 1:
        return "BUY"
    return "SKIP"


def _serialize_where_inline(where: dict[str, Any]) -> str:
    """Render ``where:`` as a GraphQL object literal (NOT JSON).

    GraphQL object literals do not quote keys. We hand-emit a minimal
    serializer to avoid pulling in a full GraphQL client.
    """

    def render(v: Any) -> str:
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ",".join(render(x) for x in v) + "]"
        if isinstance(v, dict):
            inner = ",".join(f"{k}:{render(val)}" for k, val in v.items())
            return "{" + inner + "}"
        raise TypeError(f"unsupported where value: {v!r}")

    return render(where)


def _build_where_clause(addrs: list[str], last_seen_ts: int) -> dict[str, Any]:
    """Build the ``where:`` argument for ``orderFilledEvents``.

    TheGraph rejects ``or`` mixed with same-level column filters, so the
    timestamp predicate must be repeated inside each ``or`` branch.
    ``timestamp_gte`` (not ``_gt``) plus a within-cycle ``tx_hash`` dedupe
    gives strict no-loss boundary behaviour.
    """
    ts_str = str(last_seen_ts)
    return {
        "or": [
            {"timestamp_gte": ts_str, "maker_in": addrs},
            {"timestamp_gte": ts_str, "taker_in": addrs},
        ],
    }
