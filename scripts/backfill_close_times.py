"""One-time fix for issue #40: rewrite `closed_at`/`resolved_at` from observed trades.

`corpus_markets.closed_at` was previously written as the enumerator's run
timestamp (placeholder), and `record_resolutions` propagated that into
`market_resolutions.resolved_at`. Result: only two distinct values across
the entire corpus, which collapses `temporal_split` into a hash split by
`condition_id`.

This script applies the same correction the live pipeline now uses (in
`CorpusMarketsRepo.mark_complete`) to existing rows: set
`corpus_markets.closed_at = MAX(corpus_trades.ts) per condition_id`, then
propagate the corrected `closed_at` into `market_resolutions.resolved_at`.

Idempotent. Safe to re-run after subsequent corpus rebuilds. Logs the
distinct-value count of `resolved_at` before and after so the fix is
visibly verified.
"""

# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import argparse
import sqlite3


def _distinct_resolved_at_count(conn: sqlite3.Connection) -> int:
    """Return the count of distinct `resolved_at` values in `market_resolutions`."""
    return int(
        conn.execute("SELECT COUNT(DISTINCT resolved_at) FROM market_resolutions").fetchone()[0]
    )


def _distinct_closed_at_count(conn: sqlite3.Connection) -> int:
    """Return the count of distinct `closed_at` values in `corpus_markets`."""
    return int(conn.execute("SELECT COUNT(DISTINCT closed_at) FROM corpus_markets").fetchone()[0])


def backfill(conn: sqlite3.Connection) -> tuple[int, int]:
    """Apply the close-time fix in-place and return (rows_corpus, rows_resolutions).

    Step 1: rewrite `corpus_markets.closed_at` to `MAX(corpus_trades.ts)` per
    `condition_id`. Markets with no trades retain their placeholder value
    (shouldn't happen given the $1M volume gate, but guarded).

    Step 2: propagate the corrected `closed_at` into
    `market_resolutions.resolved_at` for every row that has a corresponding
    market.
    """
    cur = conn.execute(
        """
        UPDATE corpus_markets
        SET closed_at = (
            SELECT MAX(ts) FROM corpus_trades
            WHERE corpus_trades.condition_id = corpus_markets.condition_id
        )
        WHERE EXISTS (
            SELECT 1 FROM corpus_trades
            WHERE corpus_trades.condition_id = corpus_markets.condition_id
        )
        """
    )
    rows_corpus = cur.rowcount

    cur = conn.execute(
        """
        UPDATE market_resolutions
        SET resolved_at = (
            SELECT closed_at FROM corpus_markets
            WHERE corpus_markets.condition_id = market_resolutions.condition_id
        )
        WHERE EXISTS (
            SELECT 1 FROM corpus_markets
            WHERE corpus_markets.condition_id = market_resolutions.condition_id
        )
        """
    )
    rows_resolutions = cur.rowcount

    conn.commit()
    return rows_corpus, rows_resolutions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=str,
        default="data/corpus.sqlite3",
        help="Path to the corpus SQLite database",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: connect, log before/after distincts, apply backfill."""
    args = _parse_args()
    conn = sqlite3.connect(args.db)
    try:
        before_corpus = _distinct_closed_at_count(conn)
        before_resolutions = _distinct_resolved_at_count(conn)
        print(f"before: corpus_markets distinct closed_at = {before_corpus}")
        print(f"before: market_resolutions distinct resolved_at = {before_resolutions}")

        rows_corpus, rows_resolutions = backfill(conn)
        print(f"updated {rows_corpus:,} corpus_markets rows")
        print(f"updated {rows_resolutions:,} market_resolutions rows")

        after_corpus = _distinct_closed_at_count(conn)
        after_resolutions = _distinct_resolved_at_count(conn)
        print(f"after:  corpus_markets distinct closed_at = {after_corpus}")
        print(f"after:  market_resolutions distinct resolved_at = {after_resolutions}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
