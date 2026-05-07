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


def test_init_db_creates_wallet_state_live_table() -> None:
    conn = init_db(Path(":memory:"))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(wallet_state_live)")}
    finally:
        conn.close()
    assert {
        "wallet_address",
        "first_seen_ts",
        "prior_trades_count",
        "prior_buys_count",
        "prior_resolved_buys",
        "prior_wins",
        "prior_losses",
        "cumulative_buy_price_sum",
        "cumulative_buy_count",
        "realized_pnl_usd",
        "last_trade_ts",
        "bet_size_sum",
        "bet_size_count",
        "recent_30d_trades_json",
        "category_counts_json",
        "unresolved_buys_json",
    }.issubset(cols)


def test_init_db_creates_market_state_live_table() -> None:
    conn = init_db(Path(":memory:"))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(market_state_live)")}
    finally:
        conn.close()
    assert {
        "condition_id",
        "market_age_start_ts",
        "volume_so_far_usd",
        "unique_traders_count",
        "last_trade_price",
        "recent_prices_json",
        "traders_json",
    }.issubset(cols)
