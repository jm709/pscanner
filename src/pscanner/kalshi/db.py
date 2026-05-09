"""SQLite table definitions for Kalshi data.

These statements are registered into the daemon DB via ``init_db`` in
``pscanner.store.db``. All CREATE statements use ``IF NOT EXISTS`` so
calling them on an existing database is safe.

Tables:
    kalshi_markets: Latest snapshot of each Kalshi market by ticker.
    kalshi_trades: Executed trades fetched from the Kalshi REST API.
    kalshi_orderbook_snapshots: Point-in-time orderbook snapshots.
"""

from __future__ import annotations

import sqlite3

KALSHI_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS kalshi_markets (
      ticker                    TEXT PRIMARY KEY,
      event_ticker              TEXT NOT NULL,
      title                     TEXT NOT NULL,
      status                    TEXT NOT NULL,
      market_type               TEXT NOT NULL DEFAULT '',
      open_time                 TEXT NOT NULL DEFAULT '',
      close_time                TEXT NOT NULL DEFAULT '',
      expected_expiration_time  TEXT NOT NULL DEFAULT '',
      yes_sub_title             TEXT NOT NULL DEFAULT '',
      no_sub_title              TEXT NOT NULL DEFAULT '',
      result                    TEXT,
      last_price_cents          INTEGER NOT NULL DEFAULT 0,
      yes_bid_cents             INTEGER NOT NULL DEFAULT 0,
      yes_ask_cents             INTEGER NOT NULL DEFAULT 0,
      no_bid_cents              INTEGER NOT NULL DEFAULT 0,
      no_ask_cents              INTEGER NOT NULL DEFAULT 0,
      volume_fp                 REAL NOT NULL DEFAULT 0.0,
      volume_24h_fp             REAL NOT NULL DEFAULT 0.0,
      open_interest_fp          REAL NOT NULL DEFAULT 0.0,
      cached_at                 INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kalshi_markets_event ON kalshi_markets(event_ticker)",
    "CREATE INDEX IF NOT EXISTS idx_kalshi_markets_status ON kalshi_markets(status)",
    """
    CREATE TABLE IF NOT EXISTS kalshi_trades (
      trade_id      TEXT PRIMARY KEY,
      ticker        TEXT NOT NULL,
      taker_side    TEXT NOT NULL,
      yes_price_cents INTEGER NOT NULL,
      no_price_cents  INTEGER NOT NULL,
      count_fp      REAL NOT NULL,
      created_time  TEXT NOT NULL,
      recorded_at   INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kalshi_trades_ticker ON kalshi_trades(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_kalshi_trades_created ON kalshi_trades(created_time DESC)",
    """
    CREATE TABLE IF NOT EXISTS kalshi_orderbook_snapshots (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker        TEXT NOT NULL,
      ts            INTEGER NOT NULL,
      yes_bids_json TEXT NOT NULL,
      no_bids_json  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_kalshi_ob_ticker_ts "
    "ON kalshi_orderbook_snapshots(ticker, ts DESC)",
)


_MIGRATIONS: tuple[str, ...] = ("ALTER TABLE kalshi_markets ADD COLUMN result TEXT",)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations. Idempotent.

    Each migration is wrapped to swallow ``duplicate column name`` errors
    so repeated calls on already-migrated DBs are no-ops. Mirrors
    ``pscanner.manifold.db._apply_migrations``.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                continue
            raise
    conn.commit()


def init_kalshi_tables(conn: sqlite3.Connection) -> None:
    """Apply all Kalshi schema statements + migrations to ``conn``.

    Idempotent — safe to call on an already-initialised database. Mirrors
    ``pscanner.manifold.db.init_manifold_tables``.

    Note: the daemon's ``pscanner.store.db.init_db`` already concatenates
    ``KALSHI_SCHEMA_STATEMENTS`` into its own schema. This standalone helper
    exists so tests and any code path that wants Kalshi schema in isolation
    has a single entry point that includes migrations.
    """
    for statement in KALSHI_SCHEMA_STATEMENTS:
        conn.execute(statement)
    _apply_migrations(conn)
    conn.commit()
