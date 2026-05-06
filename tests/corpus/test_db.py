"""Tests for ``pscanner.corpus.db`` schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.db import _SCHEMA_STATEMENTS, init_corpus_db

_EXPECTED_TABLES = {
    "corpus_markets",
    "corpus_trades",
    "market_resolutions",
    "training_examples",
    "corpus_state",
}


def test_init_corpus_db_creates_all_tables() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {row["name"] for row in rows}
        assert _EXPECTED_TABLES.issubset(names)
    finally:
        conn.close()


def test_init_corpus_db_is_idempotent() -> None:
    conn1 = init_corpus_db(Path(":memory:"))
    conn1.close()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        for _ in range(2):
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def test_init_corpus_db_sets_row_factory() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        assert conn.row_factory is sqlite3.Row
    finally:
        conn.close()


def test_init_corpus_db_creates_asset_index_table() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(asset_index)").fetchall()
        cols = {row[1] for row in info}
        assert cols == {"platform", "asset_id", "condition_id", "outcome_side", "outcome_index"}
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["asset_id", "platform"]
    finally:
        conn.close()


def test_init_corpus_db_corpus_markets_has_platform_pk() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(corpus_markets)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["condition_id", "platform"]
    finally:
        conn.close()


def test_init_corpus_db_corpus_trades_has_platform_pk() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(corpus_trades)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["asset_id", "platform", "tx_hash", "wallet_address"]
    finally:
        conn.close()


def test_init_corpus_db_market_resolutions_has_platform_pk() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(market_resolutions)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["condition_id", "platform"]
    finally:
        conn.close()


def test_init_corpus_db_training_examples_has_platform_pk() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(training_examples)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        assert "id" not in cols, "legacy id column must be dropped"
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["asset_id", "platform", "tx_hash", "wallet_address"]
    finally:
        conn.close()


def test_init_corpus_db_asset_index_has_platform_pk() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(asset_index)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["asset_id", "platform"]
    finally:
        conn.close()


def test_training_examples_has_wallet_quality_interaction_columns() -> None:
    """Issue #44 migration: 4 new columns present and have correct types/defaults."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        cols = {row["name"]: row for row in conn.execute("PRAGMA table_info(training_examples)")}
        for col in (
            "edge_confidence_weighted",
            "win_rate_confidence_weighted",
            "is_high_quality_wallet",
            "bet_size_relative_to_history",
        ):
            assert col in cols, f"missing column: {col}"
            assert cols[col]["notnull"] == 1, f"{col} should be NOT NULL"
        # Type checks
        assert cols["edge_confidence_weighted"]["type"].upper() == "REAL"
        assert cols["win_rate_confidence_weighted"]["type"].upper() == "REAL"
        assert cols["is_high_quality_wallet"]["type"].upper() == "INTEGER"
        assert cols["bet_size_relative_to_history"]["type"].upper() == "REAL"
    finally:
        conn.close()


def test_corpus_trades_unique_key() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        conn.execute(
            """
            INSERT INTO corpus_trades
              (platform, tx_hash, asset_id, wallet_address, condition_id,
               outcome_side, bs, price, size, notional_usd, ts)
            VALUES ('polymarket', '0xtx', 'asset1', '0xw', 'cond1',
                    'YES', 'BUY', 0.5, 100.0, 50.0, 1000)
            """
        )
        conn.commit()
        try:
            conn.execute(
                """
                INSERT INTO corpus_trades
                  (platform, tx_hash, asset_id, wallet_address, condition_id,
                   outcome_side, bs, price, size, notional_usd, ts)
                VALUES ('polymarket', '0xtx', 'asset1', '0xw', 'cond1',
                        'YES', 'BUY', 0.5, 100.0, 50.0, 1000)
                """
            )
            conn.commit()
            raise AssertionError("expected UNIQUE constraint failure")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


def test_corpus_markets_has_onchain_trades_count_column() -> None:
    """Phase 2 migration: corpus_markets.onchain_trades_count is present and nullable."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        cols = {row["name"]: row for row in conn.execute("PRAGMA table_info(corpus_markets)")}
        assert "onchain_trades_count" in cols
        assert cols["onchain_trades_count"]["type"].upper() == "INTEGER"
        assert cols["onchain_trades_count"]["notnull"] == 0
    finally:
        conn.close()
