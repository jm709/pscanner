"""SQLite bootstrap for pscanner.

Creates (idempotently) the project's tables plus their indexes, applies
pragmas for WAL + foreign keys, and sets ``row_factory = sqlite3.Row`` so
callers can index columns by name.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tracked_wallets (
      address TEXT PRIMARY KEY,
      closed_position_count INTEGER NOT NULL,
      closed_position_wins INTEGER NOT NULL,
      winrate REAL NOT NULL,
      leaderboard_pnl REAL,
      last_refreshed_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wallet_position_snapshots (
      address TEXT NOT NULL,
      market_id TEXT NOT NULL,
      side TEXT NOT NULL,
      size REAL NOT NULL,
      avg_price REAL NOT NULL,
      snapshot_at INTEGER NOT NULL,
      PRIMARY KEY (address, market_id, side)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wallet_first_seen (
      address TEXT PRIMARY KEY,
      first_activity_at INTEGER,
      total_trades INTEGER,
      cached_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_cache (
      market_id TEXT PRIMARY KEY,
      event_id TEXT,
      title TEXT,
      liquidity_usd REAL,
      volume_usd REAL,
      outcome_prices_json TEXT,
      active INTEGER NOT NULL,
      cached_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
      alert_key TEXT PRIMARY KEY,
      detector TEXT NOT NULL,
      severity TEXT NOT NULL,
      title TEXT NOT NULL,
      body_json TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS wallet_watchlist (
      address TEXT PRIMARY KEY,
      source TEXT NOT NULL,
      reason TEXT,
      added_at INTEGER NOT NULL,
      active INTEGER NOT NULL DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_watchlist_active ON wallet_watchlist(active, address)",
    """
    CREATE TABLE IF NOT EXISTS wallet_trades (
      transaction_hash TEXT NOT NULL,
      asset_id TEXT NOT NULL,
      side TEXT NOT NULL,
      wallet TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      size REAL NOT NULL,
      price REAL NOT NULL,
      usd_value REAL NOT NULL,
      status TEXT NOT NULL,
      source TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      recorded_at INTEGER NOT NULL,
      PRIMARY KEY (transaction_hash, asset_id, side)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wallet_trades_wallet_ts "
    "ON wallet_trades(wallet, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_wallet_trades_market_ts "
    "ON wallet_trades(condition_id, timestamp DESC)",
    """
    CREATE TABLE IF NOT EXISTS wallet_positions_history (
      wallet TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      outcome TEXT NOT NULL,
      size REAL NOT NULL,
      avg_price REAL NOT NULL,
      current_value REAL,
      cash_pnl REAL,
      realized_pnl REAL,
      redeemable INTEGER,
      snapshot_at INTEGER NOT NULL,
      PRIMARY KEY (wallet, condition_id, outcome, snapshot_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wph_wallet_ts "
    "ON wallet_positions_history(wallet, snapshot_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS wallet_activity_events (
      wallet TEXT NOT NULL,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      recorded_at INTEGER NOT NULL,
      source TEXT NOT NULL,
      PRIMARY KEY (wallet, timestamp, event_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wae_wallet_ts "
    "ON wallet_activity_events(wallet, timestamp DESC)",
)

_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE tracked_wallets ADD COLUMN mean_edge REAL",
    "ALTER TABLE tracked_wallets ADD COLUMN weighted_edge REAL",
    "ALTER TABLE tracked_wallets ADD COLUMN excess_pnl_usd REAL",
    "ALTER TABLE tracked_wallets ADD COLUMN total_stake_usd REAL",
)

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations. Idempotent.

    SQLite has no IF NOT EXISTS for ADD COLUMN, so each migration is wrapped
    in a try/except that swallows the "duplicate column name" OperationalError.
    Other DatabaseErrors propagate.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                continue
            raise
    conn.commit()


def init_db(path: Path) -> sqlite3.Connection:
    """Open the pscanner SQLite database, creating dirs/schema as needed.

    The function is idempotent: every CREATE statement uses ``IF NOT EXISTS``,
    so repeated calls are safe. The returned connection has ``row_factory``
    set to ``sqlite3.Row`` and is in WAL mode with foreign keys enabled.

    Args:
        path: Filesystem path to the database file. Use ``":memory:"`` for an
            in-process db (typically in tests). Parent directories are created
            for non-memory paths.

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
