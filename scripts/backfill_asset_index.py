"""One-shot backfill of `asset_index` from existing `corpus_trades` data.

Phase 1 of #42 on-chain backfill. Populates the `asset_index` lookup
table from the (asset_id, condition_id, outcome_side) tuples already
present in `corpus_trades`. Idempotent: re-runs are a no-op for rows
already indexed.

Usage:
    uv run python scripts/backfill_asset_index.py
    uv run python scripts/backfill_asset_index.py --db data/corpus.sqlite3
"""

# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo


def _row_count(conn: sqlite3.Connection) -> int:
    """Return total rows currently in `asset_index`."""
    return int(conn.execute("SELECT COUNT(*) FROM asset_index").fetchone()[0])


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
    """Entry point: connect, log before/after counts, run backfill."""
    args = _parse_args()
    conn = init_corpus_db(Path(args.db))
    try:
        before = _row_count(conn)
        print(f"before: asset_index rows = {before:,}")

        repo = AssetIndexRepo(conn)
        inserted = repo.backfill_from_corpus_trades()
        print(f"inserted: {inserted:,} new asset_index rows")

        after = _row_count(conn)
        print(f"after:  asset_index rows = {after:,}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
