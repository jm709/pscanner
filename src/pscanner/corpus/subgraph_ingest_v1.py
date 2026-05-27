"""V1 subgraph adapter for `corpus_trades` backfill (issue #193).

The V1 Polymarket Orderbook subgraph (`7fu2DWYK…`) emits the pre-#151
`OrderFilledEvent` schema: flat `maker`/`taker` hex addresses,
`makerAssetId`/`takerAssetId` (one is `"0"` for USDC), and
`makerAmountFilled`/`takerAmountFilled` in 6-decimal base units —
identical to V2's amount conventions. This module owns the V1-specific
row adapter and paginator. The orchestrator lands in a later task.

The adapter emits the same `OrderFilledEvent` dataclass V2 produces so
the downstream `event_to_corpus_trade` insert path is shared verbatim.

Verified against `tests/corpus/fixtures/v1_v2_overlap.json` (Stage 1).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade, CorpusTradesRepo
from pscanner.poly.onchain import OrderFilledEvent
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)
from pscanner.poly.subgraph import SubgraphClient

# Earliest Unix-second timestamp the V1 Polymarket Orderbook subgraph
# (`7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`) has ever indexed an
# OrderFilledEvent for. Verified by Stage 0 investigation (commit
# `e430e54`). Markets that closed before this date have no V1 history
# the subgraph can serve, so backfilling them is wasted gamma quota
# AND wall time (The Graph's indexer is slow to return empty results
# on wide id_gt scans — ~60-90s per market observed in the 2026-05-27
# production run before this filter was added).
_V1_SUBGRAPH_EARLIEST_TS: Final[int] = 1744013119  # 2025-04-07 UTC

_LOG = structlog.get_logger(__name__)


def _parse_int(key: str, raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} could not be parsed as int: {raw!r}") from exc
    raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")


def _parse_str(key: str, raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be str, got {type(raw).__name__}")
    return raw


def subgraph_v1_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one V1 `orderFilledEvents` row to the existing `OrderFilledEvent`.

    V1 stores `makerAssetId` and `takerAssetId` directly as decimal-string
    CTF token ids (one will be `"0"` for the USDC side). `maker` and
    `taker` are flat hex strings (lowercased on output). Amount fields are
    in 6-decimal base units, matching V2 exactly.

    V1's `side` (string) and `price` (decimal string) fields are present
    but ignored — the downstream `event_to_corpus_trade` derives maker-POV
    BUY/SELL from `(maker_asset_id, taker_asset_id)` alone.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list.

    Returns:
        ``OrderFilledEvent`` (block_number=0, log_index=0).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable, a string field has
            the wrong type, or both/neither asset id is zero.
    """
    maker_asset = _parse_int("makerAssetId", row["makerAssetId"])
    taker_asset = _parse_int("takerAssetId", row["takerAssetId"])
    if (maker_asset == 0) == (taker_asset == 0):
        raise ValueError(
            f"both-zero or both-non-zero asset ids: maker={maker_asset}, taker={taker_asset}"
        )

    return OrderFilledEvent(
        order_hash=_parse_str("orderHash", row["orderHash"]),
        maker=_parse_str("maker", row["maker"]).lower(),
        taker=_parse_str("taker", row["taker"]).lower(),
        maker_asset_id=maker_asset,
        taker_asset_id=taker_asset,
        making=_parse_int("makerAmountFilled", row["makerAmountFilled"]),
        taking=_parse_int("takerAmountFilled", row["takerAmountFilled"]),
        fee=_parse_int("fee", row["fee"]),
        tx_hash=_parse_str("transactionHash", row["transactionHash"]),
        block_number=0,
        log_index=0,
    )


# ---------------------------------------------------------------------------
# Paginator + iterator
# ---------------------------------------------------------------------------

_MAX_PAGE_SIZE = 1000

_V1_QUERY_MAKER_SIDE = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee
  }
}
"""

_V1_QUERY_TAKER_SIDE = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee
  }
}
"""


async def _paginate_v1_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from one V1 query (maker or taker side).

    Args:
        client: Open ``SubgraphClient``.
        graphql: GraphQL query string (``_V1_QUERY_MAKER_SIDE`` or
            ``_V1_QUERY_TAKER_SIDE``).
        asset_ids: CTF token ids (as decimal strings) to filter on.
        page_size: Rows per query page (≤ ``_MAX_PAGE_SIZE``).

    Yields:
        ``(event, ts)`` tuples where ``ts`` is the Unix timestamp integer
        from the subgraph ``timestamp`` field.
    """
    cursor = ""
    while True:
        result = await client.query(
            graphql,
            {"assets": list(asset_ids), "cursor": cursor, "first": page_size},
        )
        rows: list[dict[str, Any]] = result.get("orderFilledEvents") or []
        if not rows:
            return
        for row in rows:
            event = subgraph_v1_row_to_event(row)
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            return
        cursor = str(rows[-1]["id"])


async def iter_v1_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every V1 fill involving any of ``asset_ids``.

    V1's schema has no ``_or`` operator, so two separate paginated passes
    are required: one on ``makerAssetId_in`` (catches SELL rows where the
    maker held a CTF token) and one on ``takerAssetId_in`` (catches BUY
    rows where the taker held a CTF token). The two passes are disjoint for
    valid CTF↔USDC fills; the downstream ``CorpusTradesRepo.insert_batch``
    ``INSERT OR IGNORE`` absorbs any accidental overlap.

    Empty ``asset_ids`` short-circuits to an empty iterator (no query).

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as bare decimal strings) belonging to one
            condition. Pass both YES and NO token ids for a binary market.
        page_size: Rows per query, capped at ``_MAX_PAGE_SIZE`` (1000).
            Reduce for lower memory pressure during tests.

    Yields:
        ``(event, ts)`` tuples.

    Raises:
        ValueError: ``page_size`` is out of the ``1.._MAX_PAGE_SIZE`` range.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return
    async for ev, ts in _paginate_v1_side(
        client=client,
        graphql=_V1_QUERY_MAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
    async for ev, ts in _paginate_v1_side(
        client=client,
        graphql=_V1_QUERY_TAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V1SubgraphRunSummary:
    """Aggregate counts returned by ``run_v1_subgraph_backfill``.

    ``markets_no_new_trades`` covers any market where the drain produced
    zero inserted rows. That includes both genuine zero-event responses
    AND markets where every decoded event was filtered out (sub-floor
    notional, dup against existing ``corpus_trades``, or
    ``UnsupportedFill``/``UnresolvableAsset`` per row). The sentinel is
    not written in any of these cases so re-runs pick up the market again.
    """

    markets_processed: int
    markets_failed: int
    markets_no_new_trades: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    dups_dropped: int


@dataclass(frozen=True)
class _PendingV1Market:
    condition_id: str
    market_slug: str
    total_volume_usd: float


def _load_pending_v1_markets(
    conn: sqlite3.Connection, *, limit: int | None
) -> list[_PendingV1Market]:
    """Return v1_history_pending markets not yet processed, ordered by volume desc.

    Markets that closed before the V1 subgraph's earliest indexed event
    (``_V1_SUBGRAPH_EARLIEST_TS``) are excluded — V1 has nothing for
    them and confirming that round-trips slow indexer queries. They
    keep their ``v1_history_pending=1`` flag (still pending in a
    literal sense — there is just no upstream that can serve them).
    """
    sql = """
        SELECT condition_id,
               COALESCE(market_slug, '') AS market_slug,
               total_volume_usd
        FROM corpus_markets
        WHERE platform = 'polymarket'
          AND v1_history_pending = 1
          AND onchain_v1_processed_at IS NULL
          AND closed_at >= ?
        ORDER BY total_volume_usd DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, (_V1_SUBGRAPH_EARLIEST_TS,)).fetchall()
    return [
        _PendingV1Market(
            condition_id=r["condition_id"],
            market_slug=r["market_slug"],
            total_volume_usd=float(r["total_volume_usd"]),
        )
        for r in rows
    ]


def _count_pre_v1_skipped(conn: sqlite3.Connection) -> int:
    """Count v1_history_pending markets excluded by the pre-V1-coverage filter."""
    row = conn.execute(
        """
        SELECT COUNT(*) FROM corpus_markets
        WHERE platform = 'polymarket'
          AND v1_history_pending = 1
          AND onchain_v1_processed_at IS NULL
          AND closed_at < ?
        """,
        (_V1_SUBGRAPH_EARLIEST_TS,),
    ).fetchone()
    return int(row[0])


def _load_asset_ids_for_market(conn: sqlite3.Connection, condition_id: str) -> list[str]:
    """Return every asset_id mapped to condition_id in asset_index."""
    rows = conn.execute(
        "SELECT asset_id FROM asset_index WHERE condition_id = ?",
        (condition_id,),
    ).fetchall()
    return [r["asset_id"] for r in rows]


def _mark_v1_processed(conn: sqlite3.Connection, condition_id: str, *, now_ts: int) -> None:
    """Stamp onchain_v1_processed_at and clear v1_history_pending for one market."""
    conn.execute(
        """
        UPDATE corpus_markets
        SET onchain_v1_processed_at = ?,
            v1_history_pending = 0
        WHERE platform = 'polymarket' AND condition_id = ?
        """,
        (now_ts, condition_id),
    )
    conn.commit()


async def _backfill_one_v1_market(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    condition_id: str,
    page_size: int,
) -> tuple[int, int, int, int, int]:
    """Drain one V1-pending market. Returns (events, inserted, unsup, unres, dups).

    Returns (0, 0, 0, 0, 0) without querying the subgraph when asset_index
    has no rows for this condition_id.
    """
    asset_ids = _load_asset_ids_for_market(conn, condition_id)
    if not asset_ids:
        _LOG.warning("subgraph.v1.no_asset_index", condition_id=condition_id)
        return 0, 0, 0, 0, 0

    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    events_decoded = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    pending: list[CorpusTrade] = []

    async for event, ts in iter_v1_market_trades(
        client=client,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        events_decoded += 1
        try:
            trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=ts)
        except UnsupportedFill:
            skipped_unsupported += 1
            continue
        except UnresolvableAsset:
            skipped_unresolvable += 1
            continue
        if trade.condition_id != condition_id:
            # Stale asset_index mapping: drop silently (mirrors V2 orchestrator).
            continue
        pending.append(trade)

    if not pending:
        return events_decoded, 0, skipped_unsupported, skipped_unresolvable, 0

    inserted = trades_repo.insert_batch(pending)
    dups = len(pending) - inserted
    return events_decoded, inserted, skipped_unsupported, skipped_unresolvable, dups


async def run_v1_subgraph_backfill(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    page_size: int = _MAX_PAGE_SIZE,
    limit: int | None = None,
    now_ts: int | None = None,
) -> V1SubgraphRunSummary:
    """Process every v1_history_pending market via the V1 subgraph.

    Stamps ``onchain_v1_processed_at`` and clears ``v1_history_pending``
    only when ≥1 trade row was inserted. Zero-event drains and per-market
    exceptions leave both sentinel columns unchanged so re-runs pick up
    the market again.

    Args:
        conn: Open corpus DB connection.
        client: Open ``SubgraphClient`` pointed at the V1 subgraph.
        page_size: GraphQL ``first:`` per query page (max 1000).
        limit: Process at most ``N`` markets per run.
        now_ts: Timestamp to stamp on processed markets. Defaults to
            ``int(time.time())`` at each market's completion.
    """
    pending = _load_pending_v1_markets(conn, limit=limit)
    skipped_pre_v1 = _count_pre_v1_skipped(conn)
    _LOG.info(
        "subgraph.v1.start",
        markets=len(pending),
        skipped_pre_v1=skipped_pre_v1,
        v1_earliest_ts=_V1_SUBGRAPH_EARLIEST_TS,
    )

    processed = 0
    failed = 0
    no_new_trades = 0
    total_events = 0
    total_inserted = 0
    total_unsupported = 0
    total_unresolvable = 0
    total_dups = 0

    for i, market in enumerate(pending, start=1):
        try:
            events, inserted, unsup, unres, dups = await _backfill_one_v1_market(
                conn=conn,
                client=client,
                condition_id=market.condition_id,
                page_size=page_size,
            )
        except Exception as exc:
            failed += 1
            _LOG.error(
                "subgraph.v1.market_failed",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                error=repr(exc),
            )
            continue

        total_events += events
        total_inserted += inserted
        total_unsupported += unsup
        total_unresolvable += unres
        total_dups += dups

        if inserted == 0:
            no_new_trades += 1
            _LOG.info(
                "subgraph.v1.no_new_trades",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                slug=market.market_slug[:50],
            )
            continue

        _mark_v1_processed(
            conn, market.condition_id, now_ts=now_ts if now_ts is not None else int(time.time())
        )
        processed += 1
        _LOG.info(
            "subgraph.v1.market_complete",
            idx=i,
            of=len(pending),
            condition_id=market.condition_id[:14] + "...",
            slug=market.market_slug[:50],
            events_decoded=events,
            trades_inserted=inserted,
            dups_dropped=dups,
        )

    summary = V1SubgraphRunSummary(
        markets_processed=processed,
        markets_failed=failed,
        markets_no_new_trades=no_new_trades,
        events_decoded=total_events,
        trades_inserted=total_inserted,
        skipped_unsupported=total_unsupported,
        skipped_unresolvable=total_unresolvable,
        dups_dropped=total_dups,
    )
    _LOG.info("subgraph.v1.run_done", **summary.__dict__)
    return summary
