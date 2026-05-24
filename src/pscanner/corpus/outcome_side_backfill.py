"""Backfill incorrect ``outcome_side`` values introduced by the pre-#166 bug.

Issue #167. Spec: ``docs/superpowers/specs/2026-05-22-issue-167-outcome-side-backfill-design.md``.

``market_walker._parse_trade`` used to collapse every non-``yes`` outcome
label to ``NO`` (#159), writing both legs of binary sports/esports markets
as ``outcome_side=NO`` in ``corpus_trades`` and downstream ``asset_index``.
PR #166 forward-fixed the parser; this module rewrites the historical rows.
"""

from __future__ import annotations

import sqlite3
import time

import structlog

from pscanner.corpus.outcome_resolver import resolve_binary_outcome_map
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient


def find_buggy_markets(conn: sqlite3.Connection) -> list[str]:
    """Return the ``condition_id``s of binary markets stored as NO+NO.

    A market is buggy when ``asset_index`` has exactly 2 rows for it,
    both with ``outcome_side='NO'``. Excludes markets already marked
    backfilled via ``corpus_markets.outcome_side_backfilled_at``.

    Markets with no matching ``corpus_markets`` row still surface — the
    backfill should reach them too (they get a sentinel row created later).

    Args:
        conn: Open SQLite connection to the corpus database.

    Returns:
        Sorted list of condition_id strings representing buggy markets.
    """
    rows = conn.execute(
        """
        SELECT ai.condition_id
          FROM asset_index ai
          LEFT JOIN corpus_markets cm
                 ON cm.condition_id = ai.condition_id
                 AND cm.platform = ai.platform
         WHERE ai.platform = 'polymarket'
           AND (cm.outcome_side_backfilled_at IS NULL OR cm.condition_id IS NULL)
         GROUP BY ai.condition_id
         HAVING COUNT(*) = 2
            AND COUNT(DISTINCT ai.outcome_side) = 1
            AND MIN(ai.outcome_side) = 'NO'
         ORDER BY ai.condition_id
        """,
    ).fetchall()
    return [row[0] for row in rows]


_log = structlog.get_logger(__name__)


async def resolve_correct_mapping(
    condition_id: str,
    *,
    data: DataClient,
    gamma: GammaClient,
) -> dict[str, tuple[str, int]] | None:
    """Thin shim over :func:`pscanner.corpus.outcome_resolver.resolve_binary_outcome_map`.

    Kept as a module-local alias so callers that grep for
    ``resolve_correct_mapping`` (and the existing tests that patch it on
    this module) keep working. The body lives in the shared
    ``pscanner.corpus.outcome_resolver`` module — see that module's
    docstring for the lookup contract and log-event prefixes.
    """
    return await resolve_binary_outcome_map(condition_id, data=data, gamma=gamma)


def apply_market_backfill(
    conn: sqlite3.Connection,
    condition_id: str,
    mapping: dict[str, tuple[str, int]],
    *,
    now_ts: int,
) -> None:
    """Rewrite ``asset_index`` + ``corpus_trades`` for one market in one transaction.

    Args:
        conn: Corpus DB connection (caller owns lifecycle).
        condition_id: Market identifier to backfill.
        mapping: ``{token_id: (outcome_side, outcome_index)}`` from
            :func:`resolve_correct_mapping`. Always 2 entries.
        now_ts: Sentinel timestamp written to ``corpus_markets``.

    Transaction shape (auto-commit on success, rollback on raise):
      1. 2 x UPDATE asset_index (one per leg, defensive idempotent on correct legs).
      2. 2 x UPDATE corpus_trades (index-seek on ``idx_corpus_trades_market_ts``).
      3. 1 x UPDATE corpus_markets (sentinel). No-op when no row exists.

    The ``corpus_trades`` UPDATE rewrites the ``outcome_side`` portion of
    the composite PK. SQLite handles this as delete-then-insert at the
    storage layer; uniqueness is preserved because ``tx_hash`` is in the
    PK and globally unique per trade.
    """
    with conn:
        for token_id, (side, idx) in mapping.items():
            conn.execute(
                """
                UPDATE asset_index
                   SET outcome_side = ?, outcome_index = ?
                 WHERE platform = 'polymarket'
                   AND asset_id = ?
                """,
                (side, idx, token_id),
            )
            conn.execute(
                """
                UPDATE corpus_trades
                   SET outcome_side = ?
                 WHERE platform = 'polymarket'
                   AND condition_id = ?
                   AND asset_id = ?
                """,
                (side, condition_id, token_id),
            )
        conn.execute(
            """
            UPDATE corpus_markets
               SET outcome_side_backfilled_at = ?
             WHERE platform = 'polymarket'
               AND condition_id = ?
            """,
            (now_ts, condition_id),
        )


def validate_backfill_state(conn: sqlite3.Connection) -> int:
    """Return the count of binary markets still stored as NO+NO in asset_index.

    A clean post-backfill state returns ``0``. Non-zero indicates markets
    that failed resolution (gamma missing, non-binary, etc.); operator
    should re-run.

    Unlike :func:`find_buggy_markets`, this ignores the sentinel column
    so it reports the true asset_index health, not the work-queue size.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT condition_id
            FROM asset_index
           WHERE platform = 'polymarket'
           GROUP BY condition_id
          HAVING COUNT(*) = 2
             AND COUNT(DISTINCT outcome_side) = 1
             AND MIN(outcome_side) = 'NO'
        )
        """,
    ).fetchone()
    return int(row[0])


async def run_backfill(
    conn: sqlite3.Connection,
    *,
    data: DataClient,
    gamma: GammaClient,
    dry_run: bool,
    limit: int | None,
) -> dict[str, int]:
    """Walk every buggy market, resolve the correct mapping, apply UPDATEs.

    Returns a stats dict: ``processed``, ``resolved``, ``skipped``,
    ``remaining`` (post-run count of NO+NO markets still in asset_index).

    Per-market failures (gamma missing, non-binary) increment ``skipped``
    and do NOT touch the DB; the market stays in the work queue for a
    future re-run.
    """
    buggy = find_buggy_markets(conn)
    if limit is not None:
        buggy = buggy[:limit]

    processed = 0
    resolved = 0
    skipped = 0
    for condition_id in buggy:
        mapping = await resolve_correct_mapping(condition_id, data=data, gamma=gamma)
        processed += 1
        if mapping is None:
            skipped += 1
            continue
        resolved += 1
        if dry_run:
            _log.info(
                "corpus.backfill_outcome_side.dry_run_resolve",
                condition_id=condition_id,
                mapping={k: v[0] for k, v in mapping.items()},
            )
            continue
        apply_market_backfill(conn, condition_id, mapping, now_ts=int(time.time()))
        if processed % 100 == 0:
            _log.info(
                "corpus.backfill_outcome_side.progress",
                processed=processed,
                resolved=resolved,
                skipped=skipped,
            )

    remaining = validate_backfill_state(conn)
    _log.info(
        "corpus.backfill_outcome_side.done",
        processed=processed,
        resolved=resolved,
        skipped=skipped,
        remaining=remaining,
        dry_run=dry_run,
    )
    return {
        "processed": processed,
        "resolved": resolved,
        "skipped": skipped,
        "remaining": remaining,
    }
