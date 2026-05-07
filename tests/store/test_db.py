"""Tests for the daemon SQLite bootstrap in ``pscanner.store.db``."""

from __future__ import annotations

from pathlib import Path

from pscanner.store.db import init_db


def test_init_db_creates_manifold_tables(tmp_path: Path) -> None:
    """`init_db` creates the Manifold daemon tables alongside Polymarket and Kalshi."""
    db_path = tmp_path / "test.sqlite3"
    conn = init_db(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {row["name"] for row in rows}
        assert {"manifold_markets", "manifold_bets", "manifold_users"}.issubset(names)
    finally:
        conn.close()
