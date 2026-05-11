"""DuckDB-based engine for ``pscanner corpus build-features``.

Pure SQL pipeline that produces ``training_examples`` rows bit-equivalent
(within ``rtol=1e-9``) to the Python ``StreamingHistoryProvider`` fold,
in 5-25 min vs 6h. See ``docs/superpowers/plans/2026-05-11-issue-116-duckdb-engine.md``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Final

import duckdb
import structlog

_log = structlog.get_logger(__name__)

_V2_TABLE: Final[str] = "training_examples_v2"
_V2_INDEX_PREFIX: Final[str] = "idx_te_v2_"


def build_features_duckdb(
    *,
    db_path: Path,
    platform: str,
    now_ts: int,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> int:
    """Rebuild ``training_examples`` for ``platform`` via DuckDB.

    Args:
        db_path: Path to ``corpus.sqlite3``.
        platform: Single platform to rebuild (``polymarket``/``manifold``/etc).
        now_ts: Value written to every row's ``built_at`` column.
        memory_limit: DuckDB ``memory_limit`` PRAGMA value (e.g. ``"6GB"``).
        temp_dir: Spill directory for partitions that don't fit in memory.
        threads: ``threads`` PRAGMA value.

    Returns:
        Number of rows in the new ``training_examples`` table.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # Pre-create v2 via stdlib sqlite3 BEFORE DuckDB opens the file.
    # DuckDB's attached-SQLite CREATE TABLE rewrites types and strips
    # DEFAULT / CHECK clauses, so we must own schema creation here.
    _create_v2_via_sqlite3(db_path=db_path)

    duck = duckdb.connect(":memory:")
    try:
        _configure_duckdb(duck, memory_limit=memory_limit, temp_dir=temp_dir, threads=threads)
        _attach_corpus(duck, db_path=db_path)
        _materialize_trades(duck, platform=platform)
        _build_training_examples_v2(duck, platform=platform, now_ts=now_ts)
        n_rows = _count_v2(duck)
        corpus_path = _detach_corpus(duck)
        _atomic_swap(corpus_path)
        _log.info(
            "corpus.build_features_duckdb_done",
            rows=n_rows,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        return n_rows
    finally:
        duck.close()


def _configure_duckdb(
    duck: duckdb.DuckDBPyConnection,
    *,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> None:
    duck.execute(f"SET memory_limit = '{memory_limit}'")
    duck.execute(f"SET temp_directory = '{temp_dir}'")
    duck.execute(f"SET threads = {threads}")
    duck.execute("INSTALL sqlite")
    duck.execute("LOAD sqlite")


def _attach_corpus(duck: duckdb.DuckDBPyConnection, *, db_path: Path) -> None:
    duck.execute(f"ATTACH '{db_path}' AS corpus (TYPE sqlite)")


def _create_v2_via_sqlite3(*, db_path: Path) -> None:
    """Create training_examples_v2 with the canonical SQLite DDL.

    Run via stdlib sqlite3 (NOT DuckDB) because DuckDB's attached-SQLite
    CREATE TABLE rewrites types and strips ``DEFAULT`` / ``CHECK`` clauses.
    Pre-creating v2 here means DuckDB's INSERT can target the canonical
    schema, and the post-swap production table keeps its constraints.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Drop any stale artifacts from a prior crashed run.
        conn.execute(f"DROP TABLE IF EXISTS {_V2_TABLE}")
        for suffix in ("condition", "wallet", "label"):
            conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}{suffix}")

        # Canonical training_examples DDL with table+index names suffixed.
        # Keep in sync with src/pscanner/corpus/db.py:_SCHEMA_STATEMENTS.
        conn.execute(
            f"""
            CREATE TABLE {_V2_TABLE} (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              tx_hash TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              wallet_address TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              trade_ts INTEGER NOT NULL,
              built_at INTEGER NOT NULL,
              prior_trades_count INTEGER NOT NULL,
              prior_buys_count INTEGER NOT NULL,
              prior_resolved_buys INTEGER NOT NULL,
              prior_wins INTEGER NOT NULL,
              prior_losses INTEGER NOT NULL,
              win_rate REAL,
              avg_implied_prob_paid REAL,
              realized_edge_pp REAL,
              prior_realized_pnl_usd REAL NOT NULL DEFAULT 0,
              avg_bet_size_usd REAL,
              median_bet_size_usd REAL,
              wallet_age_days REAL NOT NULL,
              seconds_since_last_trade INTEGER,
              prior_trades_30d INTEGER NOT NULL,
              top_category TEXT,
              category_diversity INTEGER NOT NULL,
              bet_size_usd REAL NOT NULL,
              bet_size_rel_to_avg REAL,
              edge_confidence_weighted REAL NOT NULL DEFAULT 0,
              win_rate_confidence_weighted REAL NOT NULL DEFAULT 0,
              is_high_quality_wallet INTEGER NOT NULL DEFAULT 0,
              bet_size_relative_to_history REAL NOT NULL DEFAULT 1,
              side TEXT NOT NULL,
              implied_prob_at_buy REAL NOT NULL,
              market_category TEXT NOT NULL,
              market_volume_so_far_usd REAL NOT NULL,
              market_unique_traders_so_far INTEGER NOT NULL,
              market_age_seconds INTEGER NOT NULL,
              time_to_resolution_seconds INTEGER,
              last_trade_price REAL,
              price_volatility_recent REAL,
              label_won INTEGER NOT NULL,
              UNIQUE (platform, tx_hash, asset_id, wallet_address)
            )
            """
        )
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}condition ON {_V2_TABLE}(condition_id)")
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}wallet ON {_V2_TABLE}(wallet_address)")
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}label ON {_V2_TABLE}(label_won)")
        conn.commit()
    finally:
        conn.close()


def _materialize_trades(duck: duckdb.DuckDBPyConnection, *, platform: str) -> None:
    """Pull corpus_trades + corpus_markets + market_resolutions into DuckDB TEMP."""
    duck.execute(
        f"""
        CREATE TEMP TABLE trades AS
        SELECT
            t.tx_hash, t.asset_id, t.wallet_address, t.condition_id,
            t.outcome_side, t.bs, t.price, t.size, t.notional_usd, t.ts,
            m.category, m.closed_at, m.enumerated_at
        FROM corpus.corpus_trades t
        JOIN corpus.corpus_markets m
          ON m.platform = t.platform AND m.condition_id = t.condition_id
        WHERE t.platform = '{platform}' AND m.platform = '{platform}'
        """  # noqa: S608 — platform is caller-controlled, not user input
    )
    duck.execute(
        f"""
        CREATE TEMP TABLE resolutions AS
        SELECT condition_id, resolved_at, outcome_yes_won
        FROM corpus.market_resolutions
        WHERE platform = '{platform}'
        """  # noqa: S608 — platform is caller-controlled, not user input
    )


def _build_training_examples_v2(
    duck: duckdb.DuckDBPyConnection, *, platform: str, now_ts: int
) -> None:
    """Stub: v2 is pre-created via stdlib sqlite3; this leaves it empty.

    Tasks 7-11 will replace this with the real CTE chain that INSERTs rows
    into the pre-existing v2 table.
    """
    # v2 table already exists (created by _create_v2_via_sqlite3 with the
    # canonical DDL including DEFAULT 'polymarket' and CHECK constraints).
    # Tasks 7-11 will populate it. Skeleton keeps it empty so the parity
    # test fails on row count, not on a crash.
    del platform, now_ts  # unused in skeleton


def _count_v2(duck: duckdb.DuckDBPyConnection) -> int:
    row = duck.execute(f"SELECT COUNT(*) FROM corpus.{_V2_TABLE}").fetchone()  # noqa: S608 — _V2_TABLE is a module-level literal
    return int(row[0]) if row else 0


def _detach_corpus(duck: duckdb.DuckDBPyConnection) -> str:
    """Retrieve the corpus DB path, then detach to release file locks.

    Returns:
        Absolute path to the SQLite file.
    """
    row = duck.execute("FROM duckdb_databases() WHERE database_name = 'corpus'").fetchone()
    if row is None:
        raise RuntimeError("corpus database not attached")
    # duckdb_databases() columns: database_name, database_oid, path, ...
    corpus_path = str(row[2])
    duck.execute("DETACH corpus")
    return corpus_path


def _atomic_swap(corpus_path: str) -> None:
    """Swap training_examples_v2 → training_examples inside one transaction.

    SQLite does not support ``ALTER INDEX ... RENAME TO``, so the swap drops
    the old table's named indexes first (before renaming, so DROP INDEX sees
    the canonical names), then renames v2 into place, then recreates the
    canonical indexes on the new table. This is safe because DuckDB has been
    detached before this function is called.
    """
    swap_conn = sqlite3.connect(corpus_path, isolation_level=None)
    try:
        swap_conn.execute("BEGIN IMMEDIATE")
        # Drop old named indexes before renaming (SQLite tracks indexes by name,
        # not by table, so they survive the rename and would collide on recreate).
        swap_conn.execute("DROP INDEX IF EXISTS idx_training_examples_condition")
        swap_conn.execute("DROP INDEX IF EXISTS idx_training_examples_wallet")
        swap_conn.execute("DROP INDEX IF EXISTS idx_training_examples_label")
        # Also drop any stale v2 indexes (shouldn't exist here, but be safe).
        swap_conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}condition")
        swap_conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}wallet")
        swap_conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}label")
        swap_conn.execute("ALTER TABLE training_examples RENAME TO training_examples_old")
        swap_conn.execute(f"ALTER TABLE {_V2_TABLE} RENAME TO training_examples")
        # Recreate canonical indexes on the new table.
        swap_conn.execute(
            "CREATE INDEX idx_training_examples_condition ON training_examples(condition_id)"
        )
        swap_conn.execute(
            "CREATE INDEX idx_training_examples_wallet ON training_examples(wallet_address)"
        )
        swap_conn.execute(
            "CREATE INDEX idx_training_examples_label ON training_examples(label_won)"
        )
        # Drop the old table (its auto-index is dropped automatically).
        swap_conn.execute("DROP TABLE training_examples_old")
        swap_conn.execute("DELETE FROM corpus_state WHERE key = 'build_features_in_progress'")
        swap_conn.execute("COMMIT")
    except Exception:
        swap_conn.execute("ROLLBACK")
        raise
    finally:
        swap_conn.close()
