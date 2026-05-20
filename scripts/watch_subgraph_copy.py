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
import json
import os  # noqa: F401 — used in Tasks 4-8
import sqlite3  # noqa: F401 — used in Tasks 4-8
import sys  # noqa: F401 — used in Tasks 4-8
import time
from pathlib import Path
from typing import Any, Final

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


async def _fetch_events_since(
    client: Any,
    *,
    addrs: list[str],
    last_seen_ts: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Drain the subgraph for all events newer than ``last_seen_ts``.

    Watermark pagination: each page advances ``ts`` to the most recent
    event seen. Loop terminates when a page returns fewer than
    ``PAGE_SIZE`` events. Within-cycle tx_hash dedupe catches boundary
    events re-fetched by ``timestamp_gte``.

    Returns the list of unique events (asc ts ordering) and the
    indexer's ``_meta.block.timestamp`` from the last page (used by
    the caller for indexing-lag detection).
    """
    events: list[dict[str, Any]] = []
    seen_tx: set[str] = set()
    ts = last_seen_ts
    indexer_ts: int | None = None
    while True:
        where = _build_where_clause(addrs, ts)
        # SubgraphClient.query takes (graphql, variables); we hand-emit the
        # `where` clause as a GraphQL object literal inside the query body
        # because The Graph rejects column filters mixed with `or` at the
        # variables level and our simplest workaround is inline substitution.
        graphql = _GRAPHQL_QUERY.replace("$where", _serialize_where_inline(where))
        data = await client.query(graphql, {})
        page = data.get("orderFilledEvents") or []
        for e in page:
            tx = e["transactionHash"]
            if tx in seen_tx:
                continue
            seen_tx.add(tx)
            events.append(e)
        meta_block = (data.get("_meta") or {}).get("block") or {}
        meta_ts_raw = meta_block.get("timestamp")
        if meta_ts_raw is not None:
            indexer_ts = int(meta_ts_raw)
        if len(page) < PAGE_SIZE:
            break
        ts = max(int(e["timestamp"]) for e in page)
    return events, indexer_ts


def _serialize_where_inline(where: dict[str, Any]) -> str:
    """Render ``where:`` as a GraphQL object literal (not JSON).

    GraphQL object literals don't quote keys. We hand-emit a minimal
    serializer instead of pulling in a full GraphQL client.
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


def _load_checkpoint(path: Path, since_hours_override: float | None) -> int:
    """Return the timestamp to resume from.

    ``--since-hours`` always wins. Otherwise read the checkpoint file;
    if it's missing or corrupt, default to ``now()``.
    """
    if since_hours_override is not None:
        return int(time.time() - 3600.0 * since_hours_override)
    if not path.exists():
        _LOG.info("subgraph_watch.checkpoint_missing", path=str(path))
        return int(time.time())
    try:
        payload = json.loads(path.read_text())
        return int(payload["last_seen_ts"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _LOG.warning(
            "subgraph_watch.checkpoint_corrupt",
            path=str(path),
            exc=str(exc),
        )
        return int(time.time())


def _save_checkpoint(path: Path, last_seen_ts: int) -> None:
    """Atomically write the checkpoint via tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"last_seen_ts": int(last_seen_ts)}))
    tmp.replace(path)


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
