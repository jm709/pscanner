"""Tests for ``pscanner.manifold.db`` schema and migrations."""

from __future__ import annotations

import sqlite3

from pscanner.manifold.db import init_manifold_tables


def test_manifold_markets_has_resolution_column() -> None:
    """`manifold_markets` exposes a nullable TEXT `resolution` column."""
    conn = sqlite3.connect(":memory:")
    try:
        init_manifold_tables(conn)
        info = conn.execute("PRAGMA table_info(manifold_markets)").fetchall()
        cols = {row[1]: row for row in info}
        assert "resolution" in cols
        assert cols["resolution"][2].upper() == "TEXT"
        assert cols["resolution"][3] == 0, "resolution must be nullable"
    finally:
        conn.close()


def test_init_manifold_tables_idempotent_on_resolution_column() -> None:
    """Calling init_manifold_tables twice leaves the resolution column intact."""
    conn = sqlite3.connect(":memory:")
    try:
        init_manifold_tables(conn)
        init_manifold_tables(conn)
        info = conn.execute("PRAGMA table_info(manifold_markets)").fetchall()
        cols = {row[1] for row in info}
        assert "resolution" in cols
    finally:
        conn.close()
