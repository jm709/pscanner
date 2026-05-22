"""Backfill incorrect ``outcome_side`` values introduced by the pre-#166 bug.

Issue #167. Spec: ``docs/superpowers/specs/2026-05-22-issue-167-outcome-side-backfill-design.md``.

``market_walker._parse_trade`` used to collapse every non-``yes`` outcome
label to ``NO`` (#159), writing both legs of binary sports/esports markets
as ``outcome_side=NO`` in ``corpus_trades`` and downstream ``asset_index``.
PR #166 forward-fixed the parser; this module rewrites the historical rows.
"""

from __future__ import annotations

import sqlite3

import structlog

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

_BINARY_MARKET_OUTCOME_COUNT = 2


async def resolve_correct_mapping(
    condition_id: str,
    *,
    data: DataClient,
    gamma: GammaClient,
) -> dict[str, tuple[str, int]] | None:
    """Return ``{token_id: (outcome_side, outcome_index)}`` for ``condition_id``.

    Uses the established ``data.get_market_slug_by_condition_id`` →
    ``gamma.get_market_by_slug`` chain (the same one ``PaperTrader._backfill_market_cache``
    and ``market_walker.walk_market`` use post-#166).

    Returns ``None`` when:
    - either client raises
    - the slug lookup returns ``None``
    - the gamma market lookup returns ``None``
    - the market has ``len(clob_token_ids) != 2`` (non-binary)

    The caller treats ``None`` as "skip this market, no sentinel written".
    """
    try:
        slug = await data.get_market_slug_by_condition_id(condition_id)
    except Exception:
        _log.warning("corpus.backfill_outcome_side.slug_lookup_failed", condition_id=condition_id)
        return None
    if slug is None:
        return None
    try:
        market = await gamma.get_market_by_slug(slug)
    except Exception:
        _log.warning(
            "corpus.backfill_outcome_side.gamma_lookup_failed",
            condition_id=condition_id,
            slug=slug,
        )
        return None
    if market is None:
        return None
    if len(market.clob_token_ids) != _BINARY_MARKET_OUTCOME_COUNT:
        _log.info(
            "corpus.backfill_outcome_side.not_binary",
            condition_id=condition_id,
            n_outcomes=len(market.clob_token_ids),
        )
        return None
    return {
        str(market.clob_token_ids[0]): ("YES", 0),
        str(market.clob_token_ids[1]): ("NO", 1),
    }


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
