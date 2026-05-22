"""Backfill incorrect ``outcome_side`` values introduced by the pre-#166 bug.

Issue #167. Spec: ``docs/superpowers/specs/2026-05-22-issue-167-outcome-side-backfill-design.md``.

``market_walker._parse_trade`` used to collapse every non-``yes`` outcome
label to ``NO`` (#159), writing both legs of binary sports/esports markets
as ``outcome_side=NO`` in ``corpus_trades`` and downstream ``asset_index``.
PR #166 forward-fixed the parser; this module rewrites the historical rows.
"""

from __future__ import annotations

import sqlite3


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
