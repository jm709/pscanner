r"""Watch the Polymarket subgraph for trades by watchlisted wallets and book paper copies.

Standalone research script — see
``docs/superpowers/specs/2026-05-20-subgraph-watcher-copy-design.md`` for the
design and ``docs/superpowers/plans/2026-05-20-subgraph-watcher-copy.md``
for the implementation plan.

Reads the watchlist from the daemon DB's ``WatchlistRepo``. Queries the
current Polymarket Orderbook subgraph (id ``B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR``)
for ``orderFilledEvents`` since the last checkpoint with server-side
``maker_in`` / ``taker_in`` filter. For each watchlist hit whose direction
is a position-increase (BUY-on-maker or SELL-on-taker), looks up an
outcome name + fill price and books an entry row into ``paper_trades``
under ``triggering_alert_detector='subgraph_copy'``.

Coexists with the daemon's smart_money paper trader — distinct detector
tag plus distinct ``triggering_alert_key`` prefix keep both sets parallel
in the ledger.

Usage::

    uv run python scripts/watch_subgraph_copy.py --once --since-hours 1
"""

# ruff: noqa: T201, RUF100  # T201: prints added in Tasks 4-8; RUF100: suppresses premature-unused warning

from __future__ import annotations

import argparse
import asyncio
import json  # noqa: F401 — used in Tasks 4-8
import os  # noqa: F401 — used in Tasks 4-8
import sqlite3  # noqa: F401 — used in Tasks 4-8
import sys  # noqa: F401 — used in Tasks 4-8
import time  # noqa: F401 — used in Tasks 4-8
from pathlib import Path
from typing import Any, Final  # noqa: F401 — Any used in Tasks 4-8

import structlog

from pscanner.config import Config  # noqa: F401 — used in Tasks 4-8
from pscanner.poly.data import DataClient  # noqa: F401 — used in Tasks 4-8
from pscanner.poly.gamma import GammaClient  # noqa: F401 — used in Tasks 4-8
from pscanner.poly.subgraph import SubgraphClient  # noqa: F401 — used in Tasks 4-8

_LOG = structlog.get_logger(__name__)

SUBGRAPH_ID: Final[str] = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
DETECTOR_TAG: Final[str] = "subgraph_copy"
ALERT_KEY_PREFIX: Final[str] = "subgraph"
DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 10.0
DEFAULT_RPM: Final[int] = 60
PAGE_SIZE: Final[int] = 1000
DEFAULT_CHECKPOINT_PATH: Final[Path] = Path("data/subgraph_watch_state.json")
INDEXER_LAG_WARN_SECONDS: Final[int] = 60
INDEXER_LAG_ERROR_SECONDS: Final[int] = 600


def _compute_copy_direction(
    maker: str,
    taker: str,
    side: int,
    watchlist: set[str],
) -> str:
    """Return ``"BUY"`` iff the watchlist wallet's position in ``tokenId`` increases.

    The subgraph's ``side`` field is the order's direction (0=BUY, 1=SELL).
    Maker placed the resting order; taker hit it from the opposite side.
    So:

    - watchlist == maker AND side == 0 -> maker accumulates -> BUY
    - watchlist == maker AND side == 1 -> maker reduces -> SKIP
    - watchlist == taker AND side == 0 -> taker sold (hit a buy order) -> SKIP
    - watchlist == taker AND side == 1 -> taker bought (hit a sell order) -> BUY

    See the copy-direction table in the design spec for the full derivation.
    """
    maker_lower = maker.lower()
    taker_lower = taker.lower()
    if maker_lower in watchlist and side == 0:
        return "BUY"
    if taker_lower in watchlist and side == 1:
        return "BUY"
    return "SKIP"


def _build_where_clause(addrs: list[str], last_seen_ts: int) -> dict[str, Any]:
    """Build the ``where:`` argument for ``orderFilledEvents``.

    TheGraph rejects ``or`` mixed with same-level column filters, so the
    timestamp predicate must be repeated inside each ``or`` branch.
    ``timestamp_gte`` (not ``_gt``) plus a within-cycle tx_hash dedupe
    in the pagination loop gives strict no-loss boundary behaviour.

    Returns a dict ready to pass to :class:`SubgraphClient.query`.
    """
    ts_str = str(last_seen_ts)
    return {
        "or": [
            {"timestamp_gte": ts_str, "maker_in": addrs},
            {"timestamp_gte": ts_str, "taker_in": addrs},
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=str, default="data/pscanner.sqlite3",
                        help="Daemon SQLite path (default: data/pscanner.sqlite3)")
    parser.add_argument("--corpus-db", type=str, default="data/corpus.sqlite3",
                        help="Corpus SQLite path for AssetIndexRepo (default: data/corpus.sqlite3)")
    parser.add_argument("--subgraph-id", type=str, default=SUBGRAPH_ID,
                        help=f"Subgraph ID (default: {SUBGRAPH_ID})")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Seconds between poll cycles (default: {DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                        help=f"Subgraph queries per minute (default: {DEFAULT_RPM})")
    parser.add_argument("--since-hours", type=float, default=None,
                        help="Optional cold-start backfill window in hours; "
                             "overrides the checkpoint if set.")
    parser.add_argument("--once", action="store_true",
                        help="Single poll pass then exit (for testing).")
    parser.add_argument("--position-fraction-override", type=float, default=None,
                        help="Override paper-trader position_fraction (default: from config).")
    parser.add_argument("--bankroll-override", type=float, default=None,
                        help="Override paper-trader starting_bankroll_usd (default: from config).")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH),
                        help=f"Checkpoint JSON path (default: {DEFAULT_CHECKPOINT_PATH})")
    return parser.parse_args()


async def main() -> int:
    """Run the subgraph watcher poll loop."""
    args = _parse_args()
    _LOG.info(
        "subgraph_watch.startup",
        db=args.db,
        corpus_db=args.corpus_db,
        subgraph_id=args.subgraph_id,
        poll_interval_seconds=args.poll_interval_seconds,
        rpm=args.rpm,
        since_hours=args.since_hours,
        once=args.once,
    )
    # Full implementation lands in Tasks 4-8.
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
