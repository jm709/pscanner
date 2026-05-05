"""Per-market targeted on-chain backfill (resumable).

Walks each truncated market's known trade-time window on Polygon, decodes
``OrderFilled`` events whose asset_id maps to that market, and inserts the
resulting trades. The naive whole-history walk in
``pscanner.corpus.onchain_backfill`` is infeasible at scale (~3B events
across the deployment-to-head range); this module narrows each fetch to a
single market's window.

Resume semantics: a market is "processed" iff
``corpus_markets.onchain_processed_at IS NOT NULL``. The orchestrator
picks up where it left off on every run.

Block-time resolution: rather than bisect per market (~30 RPC calls each
across 2K+ markets), we sample timestamps every ``ANCHOR_STEP_BLOCKS`` blocks
across the contract's deployment-to-head range and linear-interpolate.
Polygon's ~2 sec block time keeps this accurate; we add a configurable
slack to absorb drift.
"""

from __future__ import annotations

import bisect
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import (
    AssetIndexRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
    iter_order_filled_logs,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient

_LOG = structlog.get_logger(__name__)

# CTF Exchange deployment block (verified via Blockscout).
DEPLOYMENT_BLOCK: int = 33_605_403
# Anchor sample stride for the block↔timestamp interpolator.
ANCHOR_STEP_BLOCKS: int = 200_000


@dataclass(frozen=True)
class TargetedRunSummary:
    """Aggregate counts returned by ``run_targeted_backfill``."""

    markets_processed: int
    markets_failed: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int


@dataclass(frozen=True)
class _PendingMarket:
    condition_id: str
    market_slug: str
    total_volume_usd: float
    min_ts: int
    max_ts: int


async def build_block_time_anchors(
    rpc: OnchainRpcClient, *, step: int = ANCHOR_STEP_BLOCKS
) -> list[tuple[int, int]]:
    """Return a sorted list of ``(block, timestamp)`` anchors.

    Sampled every ``step`` blocks from ``DEPLOYMENT_BLOCK`` to current head.
    Used by ``ts_to_block`` to interpolate block heights from timestamps
    without per-call binary searches.
    """
    head = await rpc.get_block_number()
    blocks = list(range(DEPLOYMENT_BLOCK, head + 1, step))
    if not blocks or blocks[-1] != head:
        blocks.append(head)
    anchors: list[tuple[int, int]] = []
    for b in blocks:
        ts = await rpc.get_block_timestamp(b)
        anchors.append((b, ts))
    return anchors


def ts_to_block(anchors: Sequence[tuple[int, int]], target_ts: int) -> int:
    """Linear-interpolate the block height at ``target_ts`` from ``anchors``.

    Args:
        anchors: Sorted list of ``(block, timestamp)`` tuples.
        target_ts: Unix-second timestamp to resolve.

    Returns:
        Estimated block height. Clamped to the anchor range when ``target_ts``
        falls outside.

    Raises:
        ValueError: If ``anchors`` is empty.
    """
    if not anchors:
        raise ValueError("anchors must be non-empty")
    timestamps = [a[1] for a in anchors]
    idx = bisect.bisect_left(timestamps, target_ts)
    if idx == 0:
        return anchors[0][0]
    if idx >= len(anchors):
        return anchors[-1][0]
    b0, t0 = anchors[idx - 1]
    b1, t1 = anchors[idx]
    if t1 == t0:
        return b0
    frac = (target_ts - t0) / (t1 - t0)
    return int(b0 + frac * (b1 - b0))


def _load_pending_markets(conn: sqlite3.Connection, *, limit: int | None) -> list[_PendingMarket]:
    """Return truncated, unprocessed markets ordered by descending volume."""
    sql = """
        SELECT m.condition_id,
               COALESCE(m.market_slug, '') AS market_slug,
               m.total_volume_usd,
               (SELECT MIN(ts) FROM corpus_trades t
                WHERE t.condition_id = m.condition_id) AS min_ts,
               (SELECT MAX(ts) FROM corpus_trades t
                WHERE t.condition_id = m.condition_id) AS max_ts
        FROM corpus_markets m
        WHERE m.truncated_at_offset_cap = 1
          AND m.onchain_processed_at IS NULL
        ORDER BY m.total_volume_usd DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [
        _PendingMarket(
            condition_id=r["condition_id"],
            market_slug=r["market_slug"],
            total_volume_usd=float(r["total_volume_usd"]),
            min_ts=int(r["min_ts"]) if r["min_ts"] is not None else 0,
            max_ts=int(r["max_ts"]) if r["max_ts"] is not None else 0,
        )
        for r in rows
        if r["min_ts"] is not None and r["max_ts"] is not None
    ]


def _load_market_asset_ids(conn: sqlite3.Connection, condition_id: str) -> set[str]:
    """Return every ``asset_id`` mapped to ``condition_id`` in ``asset_index``."""
    rows = conn.execute(
        "SELECT asset_id FROM asset_index WHERE condition_id = ?", (condition_id,)
    ).fetchall()
    return {row["asset_id"] for row in rows}


async def backfill_market(
    *,
    conn: sqlite3.Connection,
    rpc: OnchainRpcClient,
    condition_id: str,
    from_block: int,
    to_block: int,
    chunk_size: int,
) -> tuple[int, int, int, int]:
    """Walk ``[from_block, to_block]`` and ingest events for ``condition_id`` only.

    Returns:
        ``(events_decoded, trades_inserted, skipped_unsupported,
        skipped_unresolvable)`` accumulated across the run.
    """
    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    market_assets = _load_market_asset_ids(conn, condition_id)
    if not market_assets:
        # No asset_index entries for this market; nothing to filter against.
        return 0, 0, 0, 0

    events_decoded = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    pending: list[CorpusTrade] = []

    async for event, ts in iter_order_filled_logs(
        rpc=rpc, from_block=from_block, to_block=to_block, chunk_size=chunk_size
    ):
        events_decoded += 1
        # Fast pre-filter: skip events whose CTF asset id isn't ours. Saves the
        # AssetIndexRepo.get round-trip for the ~99% of events on other markets.
        if (
            str(event.maker_asset_id) not in market_assets
            and str(event.taker_asset_id) not in market_assets
        ):
            continue
        try:
            trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=ts)
        except UnsupportedFill:
            skipped_unsupported += 1
            continue
        except UnresolvableAsset:
            skipped_unresolvable += 1
            continue
        if trade.condition_id != condition_id:
            # Defensive: shouldn't happen since the asset filter above is by
            # this condition's assets, but keep as a guard.
            continue
        pending.append(trade)

    trades_inserted = trades_repo.insert_batch(pending) if pending else 0
    return events_decoded, trades_inserted, skipped_unsupported, skipped_unresolvable


def _mark_processed(
    conn: sqlite3.Connection,
    condition_id: str,
    *,
    truncation_threshold: int,
    now_ts: int,
) -> int:
    """Persist post-backfill state for one market. Returns updated trade count."""
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


async def run_targeted_backfill(
    *,
    conn: sqlite3.Connection,
    rpc: OnchainRpcClient,
    chunk_size: int = 500,
    block_slack: int = 5_000,
    truncation_threshold: int = 3_000,
    limit: int | None = None,
) -> TargetedRunSummary:
    """Process every truncated, unprocessed market.

    Builds the block-time anchor table once, then iterates pending markets
    largest-volume-first. Each market's success is committed before moving on
    so a crash mid-run loses at most the one market in flight.

    Args:
        conn: Open corpus DB connection (must have the corpus schema applied).
        rpc: Async RPC client (caller owns lifecycle).
        chunk_size: Blocks per ``eth_getLogs`` call.
        block_slack: Extra blocks padded around each market's window to
            absorb anchor-interpolation drift.
        truncation_threshold: Threshold for clearing
            ``truncated_at_offset_cap``. Mirrors
            ``onchain_backfill.clear_truncation_flags``.
        limit: Process at most ``N`` markets in this run.
    """
    pending = _load_pending_markets(conn, limit=limit)
    _LOG.info("onchain_targeted.start", markets=len(pending))

    if not pending:
        return TargetedRunSummary(0, 0, 0, 0, 0, 0)

    anchors = await build_block_time_anchors(rpc)
    _LOG.info(
        "onchain_targeted.anchors_built",
        anchors=len(anchors),
        first_block=anchors[0][0],
        last_block=anchors[-1][0],
    )

    total_events = 0
    total_inserted = 0
    total_unsupported = 0
    total_unresolvable = 0
    processed = 0
    failed = 0

    for i, market in enumerate(pending, start=1):
        from_block = max(DEPLOYMENT_BLOCK, ts_to_block(anchors, market.min_ts) - block_slack)
        to_block = ts_to_block(anchors, market.max_ts) + block_slack
        try:
            events, inserted, unsup, unres = await backfill_market(
                conn=conn,
                rpc=rpc,
                condition_id=market.condition_id,
                from_block=from_block,
                to_block=to_block,
                chunk_size=chunk_size,
            )
            total_events += events
            total_inserted += inserted
            total_unsupported += unsup
            total_unresolvable += unres
            count = _mark_processed(
                conn,
                market.condition_id,
                truncation_threshold=truncation_threshold,
                now_ts=int(time.time()),
            )
            processed += 1
            _LOG.info(
                "onchain_targeted.market_done",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id[:14] + "...",
                slug=market.market_slug[:50],
                from_block=from_block,
                to_block=to_block,
                blocks=to_block - from_block + 1,
                events_decoded=events,
                trades_inserted=inserted,
                trade_count=count,
            )
        except Exception as exc:
            failed += 1
            _LOG.error(
                "onchain_targeted.market_failed",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                error=str(exc),
            )

    summary = TargetedRunSummary(
        markets_processed=processed,
        markets_failed=failed,
        events_decoded=total_events,
        trades_inserted=total_inserted,
        skipped_unsupported=total_unsupported,
        skipped_unresolvable=total_unresolvable,
    )
    _LOG.info("onchain_targeted.run_done", **summary.__dict__)
    return summary
