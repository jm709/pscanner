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
