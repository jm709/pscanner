"""SQLite bootstrap for the corpus subsystem.

Creates ``data/corpus.sqlite3`` (idempotently), applies WAL pragmas, and
sets ``row_factory = sqlite3.Row``. The schema is deliberately separate
from ``pscanner.store.db`` — corpus tables never live in the live DB,
and the live DB never holds corpus tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS corpus_markets (
      condition_id TEXT PRIMARY KEY,
      event_slug TEXT NOT NULL,
      category TEXT,
      closed_at INTEGER NOT NULL,
      total_volume_usd REAL NOT NULL,
      backfill_state TEXT NOT NULL,
      last_offset_seen INTEGER,
      trades_pulled_count INTEGER NOT NULL DEFAULT 0,
      truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
      error_message TEXT,
      enumerated_at INTEGER NOT NULL,
      backfill_started_at INTEGER,
      backfill_completed_at INTEGER,
      market_slug TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corpus_markets_state ON corpus_markets(backfill_state)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_markets_volume ON corpus_markets(total_volume_usd DESC)",
    """
    CREATE TABLE IF NOT EXISTS corpus_trades (
      tx_hash TEXT NOT NULL,
      asset_id TEXT NOT NULL,
      wallet_address TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      outcome_side TEXT NOT NULL,
      bs TEXT NOT NULL,
      price REAL NOT NULL,
      size REAL NOT NULL,
      notional_usd REAL NOT NULL,
      ts INTEGER NOT NULL,
      UNIQUE(tx_hash, asset_id, wallet_address)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_market_ts ON corpus_trades(condition_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_wallet_ts ON corpus_trades(wallet_address, ts)",
    # Composite covers chronological keyset pagination
    # (``CorpusTradesRepo.iter_chronological``) without a temp B-tree sort.
    # The leading ``ts`` column also satisfies any ts-prefix query the old
    # single-column index served.
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_ts_tx_asset "
    "ON corpus_trades(ts, tx_hash, asset_id)",
    """
    CREATE TABLE IF NOT EXISTS market_resolutions (
      condition_id TEXT PRIMARY KEY,
      winning_outcome_index INTEGER NOT NULL,
      outcome_yes_won INTEGER NOT NULL,
      resolved_at INTEGER NOT NULL,
      source TEXT NOT NULL,
      recorded_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_examples (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
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
      UNIQUE(tx_hash, asset_id, wallet_address)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_training_examples_condition ON training_examples(condition_id)",
    "CREATE INDEX IF NOT EXISTS idx_training_examples_wallet ON training_examples(wallet_address)",
    "CREATE INDEX IF NOT EXISTS idx_training_examples_label ON training_examples(label_won)",
    """
    CREATE TABLE IF NOT EXISTS corpus_state (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS asset_index (
      asset_id TEXT PRIMARY KEY,
      condition_id TEXT NOT NULL,
      outcome_side TEXT NOT NULL,
      outcome_index INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_asset_index_condition ON asset_index(condition_id)",
)

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)

_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE corpus_markets ADD COLUMN market_slug TEXT",
    # Superseded by ``idx_corpus_trades_ts_tx_asset``, which covers ts-prefix queries.
    "DROP INDEX IF EXISTS idx_corpus_trades_ts",
    "ALTER TABLE corpus_markets ADD COLUMN onchain_trades_count INTEGER",
    "ALTER TABLE corpus_markets ADD COLUMN onchain_processed_at INTEGER",
    # Wallet-quality x confidence interaction features (issue #44).
    "ALTER TABLE training_examples ADD COLUMN edge_confidence_weighted REAL NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN win_rate_confidence_weighted REAL NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN is_high_quality_wallet INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN bet_size_relative_to_history REAL NOT NULL DEFAULT 1",
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations. Idempotent.

    SQLite has no IF NOT EXISTS for ADD COLUMN, so each migration is
    wrapped to swallow ``duplicate column name`` errors. Mirrors the
    pattern in :mod:`pscanner.store.db`.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "duplicate column name" in msg or "no such column" in msg:
                continue
            raise
    conn.commit()


def init_corpus_db(path: Path) -> sqlite3.Connection:
    """Open the corpus SQLite database, creating dirs/schema as needed.

    Idempotent: every CREATE statement uses ``IF NOT EXISTS``. The returned
    connection has ``row_factory = sqlite3.Row`` and is in WAL mode.

    Args:
        path: Filesystem path to the corpus database, or ``":memory:"``.
            Parent directories are created for non-memory paths.

    Returns:
        An open ``sqlite3.Connection``. Caller owns the lifecycle.

    Raises:
        sqlite3.DatabaseError: If pragma application or schema creation fails.
    """
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        for pragma in _PRAGMAS:
            conn.execute(pragma)
        with conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
        _apply_migrations(conn)
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn
