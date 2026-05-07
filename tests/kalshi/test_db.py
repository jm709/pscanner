"""Tests for ``pscanner.kalshi.db`` schema and migrations."""

from __future__ import annotations

import sqlite3

from pscanner.kalshi.db import init_kalshi_tables


def test_kalshi_markets_has_result_column() -> None:
    """`kalshi_markets` exposes a nullable TEXT `result` column."""
    conn = sqlite3.connect(":memory:")
    try:
        init_kalshi_tables(conn)
        info = conn.execute("PRAGMA table_info(kalshi_markets)").fetchall()
        cols = {row[1]: row for row in info}
        assert "result" in cols
        assert cols["result"][2].upper() == "TEXT"
        assert cols["result"][3] == 0, "result must be nullable"
    finally:
        conn.close()


def test_init_kalshi_tables_idempotent_on_result_column() -> None:
    """Calling init_kalshi_tables twice leaves the result column intact."""
    conn = sqlite3.connect(":memory:")
    try:
        init_kalshi_tables(conn)
        init_kalshi_tables(conn)
        info = conn.execute("PRAGMA table_info(kalshi_markets)").fetchall()
        cols = {row[1] for row in info}
        assert "result" in cols
    finally:
        conn.close()
