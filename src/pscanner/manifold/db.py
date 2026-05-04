"""SQLite schema for Manifold Markets daemon tables.

All ``CREATE TABLE`` statements are idempotent (``IF NOT EXISTS``). These tables
are Manifold-specific and parallel the Polymarket daemon tables without sharing
them — Manifold market IDs are opaque hash strings incompatible with Polymarket
CLOB asset IDs or Kalshi tickers.

Call :func:`init_manifold_tables` on an open ``sqlite3.Connection`` to apply
the schema. This is intentionally separate from ``pscanner.store.db.init_db``
so the tables are added lazily when the Manifold module is first used.
"""

from __future__ import annotations

import sqlite3

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS manifold_markets (
      id TEXT PRIMARY KEY,
      creator_id TEXT NOT NULL,
      question TEXT NOT NULL,
      outcome_type TEXT NOT NULL,
      mechanism TEXT NOT NULL,
      prob_at_last_seen REAL,
      volume REAL NOT NULL DEFAULT 0.0,
      total_liquidity REAL NOT NULL DEFAULT 0.0,
      is_resolved INTEGER NOT NULL DEFAULT 0,
      resolution_time INTEGER,
      close_time INTEGER,
      raw_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS manifold_bets (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      contract_id TEXT NOT NULL,
      outcome TEXT NOT NULL,
      amount REAL NOT NULL,
      prob_before REAL NOT NULL,
      prob_after REAL NOT NULL,
      created_time INTEGER NOT NULL,
      is_filled INTEGER,
      is_cancelled INTEGER,
      limit_prob REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS manifold_users (
      id TEXT PRIMARY KEY,
      username TEXT NOT NULL,
      name TEXT NOT NULL,
      created_time INTEGER NOT NULL,
      raw_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_manifold_bets_contract ON manifold_bets(contract_id)",
    "CREATE INDEX IF NOT EXISTS idx_manifold_bets_user ON manifold_bets(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_manifold_bets_time ON manifold_bets(created_time)",
    "CREATE INDEX IF NOT EXISTS idx_manifold_markets_resolved ON manifold_markets(is_resolved)",
)


def init_manifold_tables(conn: sqlite3.Connection) -> None:
    """Apply all Manifold schema statements to ``conn``.

    Idempotent — safe to call on an already-initialised database.

    Args:
        conn: Open ``sqlite3.Connection`` with WAL mode already set.
    """
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.commit()
