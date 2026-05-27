"""Subgraph-driven backfill of `corpus_trades`.

Adapter, paginator, and orchestrator that walk Polymarket's Orderbook
subgraph for `OrderFilledEvent` rows and write to `corpus_trades`. Reuses
the ``OrderFilledEvent`` dataclass and ``event_to_corpus_trade`` helper
from ``pscanner.poly.onchain_ingest`` so the maker-POV BUY/SELL semantics
match the eth_getLogs decoder that this module replaced.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade, CorpusTradesRepo
from pscanner.poly.onchain import OrderFilledEvent
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)
from pscanner.poly.subgraph import SubgraphClient

_LOG = structlog.get_logger(__name__)


def _clear_truncation_flags(conn: sqlite3.Connection, *, threshold: int = 3000) -> int:
    """Refresh `corpus_markets.onchain_trades_count` and clear truncation flags.

    For every market where `truncated_at_offset_cap = 1`, count its rows in
    `corpus_trades`, persist that as `onchain_trades_count`, and clear the
    truncation flag iff the count is at or above `threshold` (default
    3000 = the REST `/trades` offset cap).

    Inlined from the deleted ``pscanner.corpus.onchain_backfill`` module.
    Called by ``run_subgraph_backfill`` and by
    ``pscanner.corpus.subgraph_dispatch.run_subgraph_backfill_dispatched``.
    """
    rows = conn.execute(
        """
        SELECT m.condition_id, COUNT(t.tx_hash) AS row_count
        FROM corpus_markets m
        LEFT JOIN corpus_trades t USING (condition_id)
        WHERE m.truncated_at_offset_cap = 1
        GROUP BY m.condition_id
        """
    ).fetchall()
    cleared = 0
    for row in rows:
        cid = row["condition_id"]
        count = int(row["row_count"])
        new_flag = 0 if count >= threshold else 1
        conn.execute(
            """
            UPDATE corpus_markets
            SET onchain_trades_count = ?,
                truncated_at_offset_cap = ?
            WHERE condition_id = ?
            """,
            (count, new_flag, cid),
        )
        if new_flag == 0:
            cleared += 1
    conn.commit()
    _LOG.info(
        "subgraph.truncation_clearance_done",
        markets_examined=len(rows),
        cleared=cleared,
        threshold=threshold,
    )
    return cleared


_REQUIRED_KEYS = (
    "id",  # consumed by _paginate_side cursor logic, not by the adapter
    "transactionHash",
    "timestamp",  # consumed by iter_market_trades, not by the adapter
    "orderHash",
    "maker",  # value is the nested {"id": "0x..."} object, see _parse_account_id below
    "taker",  # same
    "tokenId",
    "side",
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


def _parse_account_id(key: str, raw: object) -> str:
    """Extract ``.id`` from a nested ``Account`` object.

    The new subgraph schema returns ``maker`` and ``taker`` as nested
    objects ``{"id": "0x..."}``. The old schema returned them as bare
    Bytes strings. This helper unwraps and validates.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a nested object, got {type(raw).__name__}")
    account = cast(Mapping[str, object], raw)
    inner = account.get("id")
    if not isinstance(inner, str):
        raise ValueError(f"{key}.id must be str, got {type(inner).__name__}")
    return inner


def subgraph_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one GraphQL ``OrderFilledEvent`` row to the on-chain dataclass.

    The new subgraph schema collapses ``makerAssetId`` / ``takerAssetId``
    into ``tokenId`` (= ``Market.id``, the conditional token traded) +
    ``side`` (Int: 0=BUY, 1=SELL, indicating the maker's order direction).
    The amount fields ``makerAmountFilled`` / ``takerAmountFilled`` follow
    the same maker/taker convention as the old schema and flow through
    directly.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list. Must
            carry every key in ``_REQUIRED_KEYS``; ``maker`` and ``taker``
            are nested ``Account`` objects with an ``id`` field.

    Returns:
        ``OrderFilledEvent`` with ``block_number=0`` and ``log_index=0``
        (subgraph payloads do not include these; downstream
        ``event_to_corpus_trade`` does not read those fields).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable, a string field has
            the wrong type, or ``side`` is not 0 or 1.
    """
    for key in _REQUIRED_KEYS:
        if key not in row:
            raise KeyError(key)

    def as_int(key: str) -> int:
        return _parse_int_field(key, row[key])

    def as_str(key: str) -> str:
        return _parse_str_field(key, row[key])

    side = as_int("side")
    token_id = as_int("tokenId")
    if side == 0:
        # Maker placed a BUY order: gave USDC, took conditional tokens.
        maker_asset_id = 0
        taker_asset_id = token_id
    elif side == 1:
        # Maker placed a SELL order: gave conditional tokens, took USDC.
        maker_asset_id = token_id
        taker_asset_id = 0
    else:
        raise ValueError(f"unexpected side: {side}")

    return OrderFilledEvent(
        order_hash=as_str("orderHash"),
        maker=_parse_account_id("maker", row["maker"]),
        taker=_parse_account_id("taker", row["taker"]),
        maker_asset_id=maker_asset_id,
        taker_asset_id=taker_asset_id,
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

# Single query — the new subgraph's market_in filter catches every fill
# involving any of the listed tokens (maker side or taker side), so the
# old maker/taker two-query split is no longer needed.
_TRADES_QUERY = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { market_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id
    orderHash
    transactionHash
    timestamp
    maker { id }
    taker { id }
    market { id }
    tokenId
    side
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
"""


async def _paginate_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from a single query, paginated by id_gt.

    Args:
        client: Open ``SubgraphClient``.
        graphql: GraphQL query string (e.g. ``_TRADES_QUERY``).
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
    """Yield every ``OrderFilledEvent`` involving any asset in ``asset_ids``.

    Uses the new subgraph's ``market_in`` filter so a single paginated
    query catches every fill on the listed tokens (no maker/taker split
    needed). Cursor-paginated via ``id_gt`` so restarts are safe (no
    duplicates on resume, only forward progress).

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as decimal strings) belonging to one condition.
            Pass both YES and NO token ids for a binary market.
        page_size: Rows per query, capped at ``_MAX_PAGE_SIZE`` (1000) by
            The Graph. Reduce for lower memory pressure during tests.

    Yields:
        ``(event, ts)`` tuples.

    Raises:
        ValueError: ``page_size`` is out of the ``1.._MAX_PAGE_SIZE`` range.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY,
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
    """Return truncated, unprocessed markets ordered by descending volume."""
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
    """Return every ``asset_id`` mapped to ``condition_id`` in ``asset_index``."""
    rows = conn.execute(
        "SELECT asset_id FROM asset_index WHERE condition_id = ?", (condition_id,)
    ).fetchall()
    return [row["asset_id"] for row in rows]


def _mark_processed(
    conn: sqlite3.Connection,
    condition_id: str,
    *,
    now_ts: int,
    truncation_threshold: int,
) -> int:
    """Persist post-backfill state for one market. Returns updated trade count.

    Updates ``onchain_processed_at``, ``onchain_trades_count``, and
    ``truncated_at_offset_cap`` in one atomic write. The flag is cleared
    iff ``count >= truncation_threshold``. Mirrors the per-market
    write in ``onchain_targeted._mark_processed`` so a mid-run crash
    loses at most the one market in flight.
    """
    count = int(
        conn.execute(
            "SELECT COUNT(*) FROM corpus_trades WHERE condition_id = ?", (condition_id,)
        ).fetchone()[0]
    )
    new_flag = 0 if count >= truncation_threshold else 1
    conn.execute(
        """
        UPDATE corpus_markets
        SET onchain_processed_at = ?,
            onchain_trades_count = ?,
            truncated_at_offset_cap = ?
        WHERE condition_id = ?
        """,
        (now_ts, count, new_flag, condition_id),
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
            # Defensive: iter_market_trades filters by asset_ids and each asset_id
            # maps to exactly one condition_id via asset_index, so a mismatch only
            # occurs if the asset_index has a stale (asset_id → condition_id) row.
            # Drop silently — that's an asset_index integrity issue, not ours.
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
    truncation_threshold: int = 3000,
) -> SubgraphRunSummary:
    """Process every truncated, unprocessed market via the subgraph.

    Args:
        conn: Open corpus DB connection.
        client: Open ``SubgraphClient``.
        page_size: GraphQL ``first:`` per query (max 1000).
        limit: Process at most ``N`` markets in this run.
        truncation_threshold: Trade count at or above which
            ``truncated_at_offset_cap`` is cleared for a market. Mirrors
            ``_clear_truncation_flags``'s default of 3000.
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
            count = _mark_processed(
                conn,
                market.condition_id,
                now_ts=int(time.time()),
                truncation_threshold=truncation_threshold,
            )
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

    cleared = (
        _clear_truncation_flags(conn, threshold=truncation_threshold) if processed > 0 else 0
    )

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
