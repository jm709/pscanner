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

# ruff: noqa: T201, RUF100  # T201: print used for operator feedback; RUF100: suppresses premature-unused warning

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import structlog

from pscanner.config import Config
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.subgraph import SubgraphClient
from pscanner.store.repo import MarketCacheRepo

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


@dataclass(frozen=True, slots=True)
class _ResolvedToken:
    condition_id: ConditionId
    asset_id: AssetId
    outcome_name: str
    outcome_index: int


def _resolve_token(
    token_id: str,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
) -> _ResolvedToken | None:
    """Resolve a subgraph ``tokenId`` to ``(condition_id, outcome_name, ...)``.

    Returns ``None`` when neither the corpus ``asset_index`` table nor
    the daemon's ``market_cache`` has the asset registered. Caller logs
    ``subgraph_watch.tokenid_unresolved`` and skips the event.
    """
    entry = asset_index.get(token_id)
    if entry is None:
        _LOG.warning("subgraph_watch.tokenid_unresolved_asset_index", token_id=token_id)
        return None
    condition_id = ConditionId(entry.condition_id)
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None:
        _LOG.warning(
            "subgraph_watch.tokenid_unresolved_market_cache",
            token_id=token_id,
            condition_id=condition_id,
        )
        return None
    # Find the asset_id's position in the cached market's parallel
    # outcomes / asset_ids lists.
    asset_id = AssetId(token_id)
    try:
        idx = cached.asset_ids.index(asset_id)
    except ValueError:
        _LOG.warning(
            "subgraph_watch.tokenid_not_in_cache",
            token_id=token_id,
            condition_id=condition_id,
        )
        return None
    return _ResolvedToken(
        condition_id=condition_id,
        asset_id=asset_id,
        outcome_name=cached.outcomes[idx],
        outcome_index=idx,
    )


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


async def _run_one_cycle(
    *,
    subgraph_client: Any,
    watchlist_repo: Any,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    market_ticks: Any,
    paper_trades: Any,
    last_seen_ts: int,
    bankroll: float,
    position_fraction: float,
    min_position_cost: float,
) -> dict[str, Any]:
    """Run a single poll cycle. Returns counts + new last_seen_ts."""
    # WatchlistRepo exposes list_active() returning list[WatchlistEntry]; pull addresses.
    entries = watchlist_repo.list_active()
    addrs_raw = sorted({e.address for e in entries})
    if not addrs_raw:
        _LOG.warning("subgraph_watch.empty_watchlist")
        return {
            "events_seen": 0, "events_copied": 0, "events_skipped": 0,
            "new_last_seen_ts": last_seen_ts,
        }
    addrs = [a.lower() for a in addrs_raw]
    watchlist_set = set(addrs)

    _LOG.info("subgraph_watch.poll_start", addrs=len(addrs), last_seen_ts=last_seen_ts)
    events, indexer_ts = await _fetch_events_since(
        subgraph_client, addrs=addrs, last_seen_ts=last_seen_ts,
    )

    if indexer_ts is not None:
        lag = int(time.time()) - indexer_ts
        if lag >= INDEXER_LAG_ERROR_SECONDS:
            _LOG.error("subgraph_watch.indexer_lag", lag_seconds=lag)
        elif lag >= INDEXER_LAG_WARN_SECONDS:
            _LOG.warning("subgraph_watch.indexer_lag", lag_seconds=lag)

    copied = 0
    skipped = 0
    new_last_seen_ts = last_seen_ts
    for ev in events:
        ev_ts = int(ev["timestamp"])
        new_last_seen_ts = max(new_last_seen_ts, ev_ts)
        try:
            booked = _try_copy_event(
                ev=ev,
                watchlist_set=watchlist_set,
                asset_index=asset_index,
                market_cache=market_cache,
                market_ticks=market_ticks,
                paper_trades=paper_trades,
                bankroll=bankroll,
                position_fraction=position_fraction,
                min_position_cost=min_position_cost,
            )
        except Exception:
            _LOG.exception("subgraph_watch.copy_event_failed", tx=ev.get("transactionHash"))
            skipped += 1
            continue
        if booked:
            copied += 1
        else:
            skipped += 1

    return {
        "events_seen": len(events),
        "events_copied": copied,
        "events_skipped": skipped,
        "new_last_seen_ts": new_last_seen_ts,
    }


@dataclass(frozen=True, slots=True)
class _BookingParams:
    source_wallet: str
    resolved: _ResolvedToken
    fill_price: float
    cost: float
    shares: float


def _resolve_event_booking(
    *,
    ev: dict[str, Any],
    watchlist_set: set[str],
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    market_ticks: Any,
    bankroll: float,
    position_fraction: float,
    min_position_cost: float,
) -> _BookingParams | None:
    """Pre-flight: validate direction, resolve token, check price/size. Returns None to skip."""
    from pscanner.strategies.paper_trader import lookup_fill_price  # local import  # noqa: PLC0415

    maker = ev["maker"]["id"]
    taker = ev["taker"]["id"]
    side = int(ev["side"])
    if _compute_copy_direction(maker, taker, side, watchlist_set) != "BUY":
        return None

    # Whose side is on the watchlist? Use that address as source_wallet.
    if maker.lower() in watchlist_set:
        source_wallet = maker
    elif taker.lower() in watchlist_set:
        source_wallet = taker
    else:  # unreachable in practice; defensive for misclassified events
        return None

    resolved = _resolve_token(ev["tokenId"], asset_index, market_cache)
    if resolved is None:
        return None

    fill_price = lookup_fill_price(
        market_cache, market_ticks, resolved.condition_id, resolved.asset_id,
    )
    if fill_price is None:
        _LOG.warning("subgraph_watch.no_fill_price",
                     condition_id=resolved.condition_id, asset_id=resolved.asset_id)
        return None

    cost = bankroll * position_fraction
    if cost < min_position_cost or not (0.0 < fill_price < 1.0):
        _LOG.debug("subgraph_watch.size_or_price_invalid",
                   cost=cost, min=min_position_cost, fill_price=fill_price)
        return None

    return _BookingParams(
        source_wallet=source_wallet,
        resolved=resolved,
        fill_price=fill_price,
        cost=cost,
        shares=cost / fill_price,
    )


def _try_copy_event(
    *,
    ev: dict[str, Any],
    watchlist_set: set[str],
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    market_ticks: Any,
    paper_trades: Any,
    bankroll: float,
    position_fraction: float,
    min_position_cost: float,
) -> bool:
    """Attempt to book a paper copy for one event. Returns True iff inserted."""
    params = _resolve_event_booking(
        ev=ev,
        watchlist_set=watchlist_set,
        asset_index=asset_index,
        market_cache=market_cache,
        market_ticks=market_ticks,
        bankroll=bankroll,
        position_fraction=position_fraction,
        min_position_cost=min_position_cost,
    )
    if params is None:
        return False

    nav = paper_trades.compute_cost_basis_nav(starting_bankroll=bankroll)
    alert_key = f"{ALERT_KEY_PREFIX}:{ev['transactionHash']}:{params.resolved.outcome_name}"
    try:
        paper_trades.insert_entry(
            triggering_alert_key=alert_key,
            triggering_alert_detector=DETECTOR_TAG,
            rule_variant=None,
            source_wallet=params.source_wallet,
            condition_id=params.resolved.condition_id,
            asset_id=params.resolved.asset_id,
            outcome=params.resolved.outcome_name,
            shares=params.shares,
            fill_price=params.fill_price,
            cost_usd=params.cost,
            nav_after_usd=nav,
            ts=int(ev["timestamp"]),
        )
    except sqlite3.IntegrityError:
        _LOG.debug("subgraph_watch.duplicate_alert", alert_key=alert_key)
        return False

    _LOG.info(
        "subgraph_watch.copy_inserted",
        wallet=params.source_wallet,
        condition_id=params.resolved.condition_id,
        outcome=params.resolved.outcome_name,
        fill_price=params.fill_price,
        shares=round(params.shares, 4),
        cost_usd=round(params.cost, 2),
    )
    cid_short = params.resolved.condition_id[:10]
    print(
        f"COPY {params.source_wallet[:14]}.. {params.resolved.outcome_name}"
        f" @ {params.fill_price:.3f} shares={params.shares:.2f}"
        f" cost=${params.cost:.2f} cid={cid_short}.."
    )
    return True


async def main() -> int:
    """Entry point: wire deps and run the poll loop until SIGINT."""
    args = _parse_args()
    config = Config.load()

    bankroll = args.bankroll_override or config.paper_trading.starting_bankroll_usd
    position_fraction = (
        args.position_fraction_override
        or config.paper_trading.evaluators.gate_model.position_fraction
    )
    min_position_cost = config.paper_trading.min_position_cost_usd

    _LOG.info(
        "subgraph_watch.startup",
        db=args.db,
        corpus_db=args.corpus_db,
        subgraph_id=args.subgraph_id,
        poll_interval_seconds=args.poll_interval_seconds,
        rpm=args.rpm,
        since_hours=args.since_hours,
        once=args.once,
        bankroll=bankroll,
        position_fraction=position_fraction,
        min_position_cost=min_position_cost,
    )

    api_key = os.environ.get("GRAPH_API_KEY")
    if not api_key:
        _LOG.error("subgraph_watch.missing_graph_api_key")
        return 2
    subgraph_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{args.subgraph_id}"

    # Daemon DB — read/write for paper_trades, read for watchlist/market_cache/ticks.
    daemon_conn = sqlite3.connect(args.db)
    daemon_conn.row_factory = sqlite3.Row
    daemon_conn.execute("PRAGMA busy_timeout=5000")

    # Corpus DB — read-only for asset_index.
    corpus_uri = f"file:{args.corpus_db}?mode=ro"
    corpus_conn = sqlite3.connect(corpus_uri, uri=True)
    corpus_conn.row_factory = sqlite3.Row
    corpus_conn.execute("PRAGMA busy_timeout=5000")

    from pscanner.store.repo import (  # noqa: PLC0415
        MarketTicksRepo,
        PaperTradesRepo,
        WatchlistRepo,
    )

    watchlist_repo = WatchlistRepo(daemon_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    market_ticks = MarketTicksRepo(daemon_conn)
    paper_trades = PaperTradesRepo(daemon_conn)
    asset_index = AssetIndexRepo(corpus_conn)

    subgraph_client = SubgraphClient(url=subgraph_url, rpm=args.rpm)
    data_client = DataClient(rpm=50)
    gamma_client = GammaClient(rpm=50)

    checkpoint_path = Path(args.checkpoint)
    last_seen_ts = _load_checkpoint(checkpoint_path, args.since_hours)
    _LOG.info("subgraph_watch.checkpoint_loaded", last_seen_ts=last_seen_ts)

    try:
        while True:
            cycle_start = time.time()
            stats = await _run_one_cycle(
                subgraph_client=subgraph_client,
                watchlist_repo=watchlist_repo,
                asset_index=asset_index,
                market_cache=market_cache,
                market_ticks=market_ticks,
                paper_trades=paper_trades,
                last_seen_ts=last_seen_ts,
                bankroll=bankroll,
                position_fraction=position_fraction,
                min_position_cost=min_position_cost,
            )
            last_seen_ts = stats["new_last_seen_ts"]
            _save_checkpoint(checkpoint_path, last_seen_ts)
            _LOG.info(
                "subgraph_watch.poll_done",
                events_seen=stats["events_seen"],
                events_copied=stats["events_copied"],
                events_skipped=stats["events_skipped"],
                wall_seconds=round(time.time() - cycle_start, 2),
                new_last_seen_ts=last_seen_ts,
            )
            if args.once:
                return 0
            await asyncio.sleep(args.poll_interval_seconds)
    except (KeyboardInterrupt, asyncio.CancelledError):
        _LOG.info("subgraph_watch.shutdown")
        _save_checkpoint(checkpoint_path, last_seen_ts)
        return 0
    finally:
        await subgraph_client.aclose()
        await data_client.aclose()
        await gamma_client.aclose()
        daemon_conn.close()
        corpus_conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
