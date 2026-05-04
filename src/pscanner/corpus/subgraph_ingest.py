"""Subgraph-driven backfill of `corpus_trades` (Phase 3).

Adapter, paginator, and orchestrator that replace the eth_getLogs path
in ``pscanner.corpus.onchain_targeted``. Reuses the Phase 2 decoder
output type (``OrderFilledEvent``) and ``event_to_corpus_trade`` so the
maker-POV BUY/SELL semantics stay identical.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from pscanner.corpus.onchain_backfill import clear_truncation_flags
from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade, CorpusTradesRepo
from pscanner.poly.onchain import OrderFilledEvent
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)
from pscanner.poly.subgraph import SubgraphClient

_LOG = structlog.get_logger(__name__)

_REQUIRED_KEYS = (
    "id",  # consumed by _paginate_side (Task 4) cursor logic, not by the adapter
    "transactionHash",
    "timestamp",  # consumed by iter_market_trades (Task 4), not by the adapter
    "orderHash",
    "maker",
    "taker",
    "makerAssetId",
    "takerAssetId",
    "makerAmountFilled",
    "takerAmountFilled",
    "fee",
)


def _parse_int_field(key: str, raw: object) -> int:
    """Parse a GraphQL BigInt field (string or native int) to Python int."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} could not be parsed as int: {raw!r}") from exc
    raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")


def _parse_str_field(key: str, raw: object) -> str:
    """Validate that a GraphQL field is a plain string."""
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be str, got {type(raw).__name__}")
    return raw


def subgraph_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one GraphQL ``OrderFilledEvent`` row to a Phase 2 dataclass.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list. Must
            carry every key in ``_REQUIRED_KEYS``.

    Returns:
        ``OrderFilledEvent`` with ``block_number=0`` and ``log_index=0``
        (subgraph payloads do not include these; downstream
        ``event_to_corpus_trade`` does not read those fields).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable as int, or a string
            field has the wrong type.
    """
    for key in _REQUIRED_KEYS:
        if key not in row:
            raise KeyError(key)

    def as_int(key: str) -> int:
        return _parse_int_field(key, row[key])

    def as_str(key: str) -> str:
        return _parse_str_field(key, row[key])

    return OrderFilledEvent(
        order_hash=as_str("orderHash"),
        maker=as_str("maker"),
        taker=as_str("taker"),
        maker_asset_id=as_int("makerAssetId"),
        taker_asset_id=as_int("takerAssetId"),
        making=as_int("makerAmountFilled"),
        taking=as_int("takerAmountFilled"),
        fee=as_int("fee"),
        tx_hash=as_str("transactionHash"),
        block_number=0,
        log_index=0,
    )


# ---------------------------------------------------------------------------
# Paginator
# ---------------------------------------------------------------------------

# The Graph's hard cap on a single page of results.
_MAX_PAGE_SIZE = 1000

# Two separate query constants rather than one template: the filter field
# name (makerAssetId_in vs takerAssetId_in) is a structural part of the
# GraphQL document, not a $variable, so it can't be parameterised. Keeping
# them as separate string literals avoids any string-formatting on GraphQL
# document text — which would be a query-injection foot-gun.
_TRADES_QUERY_MAKER_SIDE = """
query MarketTradesMakerSide($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    orderBy: id
    first: $first
  ) {
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }
}
""".strip()

_TRADES_QUERY_TAKER_SIDE = """
query MarketTradesTakerSide($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $assets, id_gt: $cursor }
    orderBy: id
    first: $first
  ) {
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }
}
""".strip()


async def _paginate_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from one filter side, paginated by id_gt.

    Args:
        client: Open ``SubgraphClient``.
        graphql: One of ``_TRADES_QUERY_MAKER_SIDE`` or ``_TRADES_QUERY_TAKER_SIDE``.
        asset_ids: CTF token ids (as strings) to filter on.
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
            event = subgraph_row_to_event(row)
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            # Short page guarantees no more rows exist for this cursor range.
            return
        cursor = str(rows[-1]["id"])


async def iter_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every ``OrderFilledEvent`` whose maker- or taker-side asset is in ``asset_ids``.

    Runs maker-side queries to exhaustion, then taker-side queries.  Each side
    uses ``id_gt`` cursor pagination so restarts are safe (no duplicates on
    resume, only forward progress).

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as decimal strings) belonging to one condition.
            Pass both YES and NO token ids for a binary market.
        page_size: Rows per query, capped at ``_MAX_PAGE_SIZE`` (1000) by
            The Graph.  Reduce for lower memory pressure during tests.

    Yields:
        ``(event, ts)`` tuples — same shape as ``iter_order_filled_logs``
        from Phase 2 so the orchestrator can mirror its loop body.

    Raises:
        ValueError: ``page_size`` is out of the ``1.._MAX_PAGE_SIZE`` range.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY_MAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY_TAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphRunSummary:
    """Aggregate counts returned by ``run_subgraph_backfill``."""

    markets_processed: int
    markets_failed: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    truncation_flags_cleared: int


@dataclass(frozen=True)
class _PendingMarket:
    condition_id: str
    market_slug: str
    total_volume_usd: float


def _load_pending_markets(conn: sqlite3.Connection, *, limit: int | None) -> list[_PendingMarket]:
    sql = """
        SELECT condition_id,
               COALESCE(market_slug, '') AS market_slug,
               total_volume_usd
        FROM corpus_markets
        WHERE truncated_at_offset_cap = 1
          AND onchain_processed_at IS NULL
        ORDER BY total_volume_usd DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [
        _PendingMarket(
            condition_id=r["condition_id"],
            market_slug=r["market_slug"],
            total_volume_usd=float(r["total_volume_usd"]),
        )
        for r in rows
    ]


def _load_market_asset_ids(conn: sqlite3.Connection, condition_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT asset_id FROM asset_index WHERE condition_id = ?", (condition_id,)
    ).fetchall()
    return [row["asset_id"] for row in rows]


def _mark_processed(
    conn: sqlite3.Connection,
    condition_id: str,
    *,
    now_ts: int,
) -> int:
    """Persist post-backfill state; returns the new on-chain trade count."""
    count = int(
        conn.execute(
            "SELECT COUNT(*) FROM corpus_trades WHERE condition_id = ?", (condition_id,)
        ).fetchone()[0]
    )
    conn.execute(
        """
        UPDATE corpus_markets
        SET onchain_processed_at = ?,
            onchain_trades_count = ?
        WHERE condition_id = ?
        """,
        (now_ts, count, condition_id),
    )
    conn.commit()
    return count


async def _backfill_one_market(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    condition_id: str,
    page_size: int,
) -> tuple[int, int, int, int]:
    """Return (events_decoded, trades_inserted, skipped_unsupported, skipped_unresolvable)."""
    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    asset_ids = _load_market_asset_ids(conn, condition_id)
    if not asset_ids:
        return 0, 0, 0, 0

    events_decoded = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    pending: list[CorpusTrade] = []

    async for event, ts in iter_market_trades(
        client=client, asset_ids=asset_ids, page_size=page_size
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
            continue
        pending.append(trade)

    inserted = trades_repo.insert_batch(pending) if pending else 0
    return events_decoded, inserted, skipped_unsupported, skipped_unresolvable


async def run_subgraph_backfill(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    page_size: int = _MAX_PAGE_SIZE,
    limit: int | None = None,
) -> SubgraphRunSummary:
    """Process every truncated, unprocessed market via the subgraph.

    Args:
        conn: Open corpus DB connection.
        client: Open ``SubgraphClient``.
        page_size: GraphQL ``first:`` per query (max 1000).
        limit: Process at most ``N`` markets in this run.
    """
    pending = _load_pending_markets(conn, limit=limit)
    _LOG.info("subgraph.start", markets=len(pending))

    processed = 0
    failed = 0
    total_events = 0
    total_inserted = 0
    total_unsupported = 0
    total_unresolvable = 0

    for i, market in enumerate(pending, start=1):
        try:
            events, inserted, unsup, unres = await _backfill_one_market(
                conn=conn,
                client=client,
                condition_id=market.condition_id,
                page_size=page_size,
            )
            total_events += events
            total_inserted += inserted
            total_unsupported += unsup
            total_unresolvable += unres
            count = _mark_processed(conn, market.condition_id, now_ts=int(time.time()))
            processed += 1
            _LOG.info(
                "subgraph.market_done",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id[:14] + "...",
                slug=market.market_slug[:50],
                events_decoded=events,
                trades_inserted=inserted,
                trade_count=count,
            )
        except Exception as exc:
            failed += 1
            _LOG.error(
                "subgraph.market_failed",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                error=str(exc),
            )

    cleared = clear_truncation_flags(conn=conn) if processed > 0 else 0

    summary = SubgraphRunSummary(
        markets_processed=processed,
        markets_failed=failed,
        events_decoded=total_events,
        trades_inserted=total_inserted,
        skipped_unsupported=total_unsupported,
        skipped_unresolvable=total_unresolvable,
        truncation_flags_cleared=cleared,
    )
    _LOG.info("subgraph.run_done", **summary.__dict__)
    return summary
