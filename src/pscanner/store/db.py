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
      cached_at INTEGER NOT NULL,
      condition_id TEXT,
      event_slug TEXT
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
    """
    CREATE TABLE IF NOT EXISTS market_snapshots (
      market_id TEXT NOT NULL,
      event_id TEXT,
      outcome_prices_json TEXT NOT NULL,
      liquidity_usd REAL,
      volume_usd REAL,
      active INTEGER NOT NULL,
      snapshot_at INTEGER NOT NULL,
      PRIMARY KEY (market_id, snapshot_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_ts ON market_snapshots(snapshot_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_market_snapshots_market_ts "
    "ON market_snapshots(market_id, snapshot_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS event_snapshots (
      event_id TEXT NOT NULL,
      title TEXT NOT NULL,
      slug TEXT NOT NULL,
      liquidity_usd REAL,
      volume_usd REAL,
      active INTEGER NOT NULL,
      closed INTEGER NOT NULL,
      market_count INTEGER NOT NULL,
      snapshot_at INTEGER NOT NULL,
      PRIMARY KEY (event_id, snapshot_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_event_snapshots_ts ON event_snapshots(snapshot_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS event_outcome_sum_history (
      event_id TEXT NOT NULL,
      market_count INTEGER NOT NULL,
      price_sum REAL NOT NULL,
      deviation REAL NOT NULL,
      snapshot_at INTEGER NOT NULL,
      PRIMARY KEY (event_id, snapshot_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eosh_ts ON event_outcome_sum_history(snapshot_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_eosh_deviation ON event_outcome_sum_history(deviation)",
    """
    CREATE TABLE IF NOT EXISTS tracked_wallet_categories (
      wallet TEXT NOT NULL,
      category TEXT NOT NULL,
      position_count INTEGER NOT NULL,
      win_count INTEGER NOT NULL,
      mean_edge REAL,
      weighted_edge REAL,
      excess_pnl_usd REAL,
      total_stake_usd REAL,
      last_refreshed_at INTEGER NOT NULL,
      PRIMARY KEY (wallet, category)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_twc_category ON tracked_wallet_categories(category)",
    """
    CREATE TABLE IF NOT EXISTS event_tag_cache (
      event_slug TEXT PRIMARY KEY,
      tags_json TEXT NOT NULL,
      cached_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_market_cache_condition ON market_cache(condition_id)",
    "CREATE INDEX IF NOT EXISTS idx_market_cache_event_slug ON market_cache(event_slug)",
    """
    CREATE TABLE IF NOT EXISTS market_ticks (
      asset_id TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      snapshot_at INTEGER NOT NULL,
      mid_price REAL,
      best_bid REAL,
      best_ask REAL,
      spread REAL,
      bid_depth_top5 REAL,
      ask_depth_top5 REAL,
      last_trade_price REAL,
      PRIMARY KEY (asset_id, snapshot_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_market_ticks_asset_ts "
    "ON market_ticks(asset_id, snapshot_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_market_ticks_ts ON market_ticks(snapshot_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS wallet_clusters (
      cluster_id TEXT PRIMARY KEY,
      member_count INTEGER NOT NULL,
      first_member_created_at INTEGER NOT NULL,
      last_member_created_at INTEGER NOT NULL,
      shared_market_count INTEGER NOT NULL,
      behavior_tag TEXT,
      detection_score INTEGER NOT NULL,
      first_detected_at INTEGER NOT NULL,
      last_active_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS wallet_cluster_members (
      cluster_id TEXT NOT NULL,
      wallet TEXT NOT NULL,
      PRIMARY KEY (cluster_id, wallet)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_wcm_wallet ON wallet_cluster_members(wallet)",
)

_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE tracked_wallets ADD COLUMN mean_edge REAL",
    "ALTER TABLE tracked_wallets ADD COLUMN weighted_edge REAL",
    "ALTER TABLE tracked_wallets ADD COLUMN excess_pnl_usd REAL",
    "ALTER TABLE tracked_wallets ADD COLUMN total_stake_usd REAL",
    "ALTER TABLE market_cache ADD COLUMN condition_id TEXT",
    "ALTER TABLE market_cache ADD COLUMN event_slug TEXT",
    # Issue 18 / 21: the column historically named ``event_id`` always stored
    # event slugs, never numeric ids. Rename it so the column type matches its
    # contents (and the typed ``EventTagCacheRepo`` API).
    "ALTER TABLE event_tag_cache RENAME COLUMN event_id TO event_slug",
)

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations. Idempotent.

    SQLite has no IF NOT EXISTS for ADD COLUMN, so each ``ADD COLUMN``
    migration is wrapped to swallow ``duplicate column name`` errors. For
    ``RENAME COLUMN`` migrations applied a second time (or against a fresh
    schema where the new column name already exists), SQLite reports
    ``no such column`` on the source side; we treat that the same way.
    Other DatabaseErrors propagate.
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
