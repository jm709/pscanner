"""Orchestrate on-chain `OrderFilled` ingest into `corpus_trades`.

Composes `iter_order_filled_logs` with `event_to_corpus_trade` and writes
through `CorpusTradesRepo`. Tracks the chunk cursor in
`corpus_state['onchain_last_block']` so partial runs are resumable.
"""

from __future__ import annotations

import sqlite3
import time

import structlog

from pscanner.corpus.repos import (
    AssetIndexRepo,
    CorpusStateRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.poly.onchain_ingest import (
    IngestRunSummary,
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
    iter_order_filled_logs,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient

# Public re-export so callers can do
# `from pscanner.corpus.onchain_backfill import IngestRunSummary`.
# IngestRunSummary is defined in pscanner.poly.onchain_ingest (the conversion
# layer); re-exporting here keeps the public API of this orchestration module
# self-contained.
__all__ = ["IngestRunSummary", "clear_truncation_flags", "run_onchain_backfill"]

_LOG = structlog.get_logger(__name__)
_STATE_KEY = "onchain_last_block"


async def run_onchain_backfill(
    *,
    conn: sqlite3.Connection,
    rpc: OnchainRpcClient,
    from_block: int,
    to_block: int,
    chunk_size: int = 5_000,
) -> IngestRunSummary:
    """Walk [from_block, to_block], decode events, insert trades, persist cursor.

    Args:
        conn: Open corpus DB connection (must have the corpus schema applied).
        rpc: Async RPC client (caller owns lifecycle).
        from_block: First block (inclusive).
        to_block: Last block (inclusive).
        chunk_size: Blocks per `eth_getLogs` call.

    Returns:
        `IngestRunSummary` with per-run counts.
    """
    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    state_repo = CorpusStateRepo(conn)

    events_decoded = 0
    trades_inserted = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    chunks_processed = 0
    pending: list[CorpusTrade] = []
    chunk_boundary = from_block + chunk_size - 1

    async for event, ts in iter_order_filled_logs(
        rpc=rpc,
        from_block=from_block,
        to_block=to_block,
        chunk_size=chunk_size,
    ):
        events_decoded += 1
        # Flush + advance cursor when a chunk boundary is crossed.
        while event.block_number > chunk_boundary:
            inserted_count = trades_repo.insert_batch(pending)
            trades_inserted += inserted_count
            pending = []
            state_repo.set(_STATE_KEY, str(chunk_boundary), updated_at=int(time.time()))
            chunks_processed += 1
            chunk_boundary = min(chunk_boundary + chunk_size, to_block)

        try:
            trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=ts)
        except UnsupportedFill as exc:
            skipped_unsupported += 1
            _LOG.debug("onchain.skip_unsupported", reason=str(exc))
            continue
        except UnresolvableAsset as exc:
            skipped_unresolvable += 1
            _LOG.debug("onchain.skip_unresolvable", asset_id=str(exc))
            continue
        pending.append(trade)

    if pending:
        trades_inserted += trades_repo.insert_batch(pending)
    state_repo.set(_STATE_KEY, str(to_block), updated_at=int(time.time()))
    chunks_processed += 1

    summary = IngestRunSummary(
        chunks_processed=chunks_processed,
        events_decoded=events_decoded,
        trades_inserted=trades_inserted,
        skipped_unsupported=skipped_unsupported,
        skipped_unresolvable=skipped_unresolvable,
        last_block=to_block,
    )
    _LOG.info(
        "onchain.backfill_done",
        chunks=summary.chunks_processed,
        events=summary.events_decoded,
        inserted=summary.trades_inserted,
        skipped_unsupported=summary.skipped_unsupported,
        skipped_unresolvable=summary.skipped_unresolvable,
        last_block=summary.last_block,
    )
    return summary


def clear_truncation_flags(*, conn: sqlite3.Connection, threshold: int = 3000) -> int:
    """Refresh `corpus_markets.onchain_trades_count` and clear truncation flags.

    For every market where `truncated_at_offset_cap = 1`, count its rows in
    `corpus_trades`, persist that as `onchain_trades_count`, and clear the
    truncation flag iff the count is at or above `threshold` (default
    3000 = the REST `/trades` offset cap).

    Args:
        conn: Open corpus DB connection.
        threshold: Minimum `corpus_trades` row count required to clear the flag.

    Returns:
        Number of markets whose flag was cleared this call.
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
        "onchain.truncation_clearance_done",
        markets_examined=len(rows),
        cleared=cleared,
        threshold=threshold,
    )
    return cleared
