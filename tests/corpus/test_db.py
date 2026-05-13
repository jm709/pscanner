"""Tests for ``pscanner.corpus.db`` schema bootstrap."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from pscanner.corpus.db import (
    _SCHEMA_STATEMENTS,
    TRAINING_EXAMPLES_COLUMNS,
    _apply_migrations,
    init_corpus_db,
    training_examples_ddl,
)

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


def test_init_corpus_db_training_examples_has_platform_unique() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(training_examples)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        assert "id" in cols, "legacy id column is preserved for streaming.py ORDER BY"
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["id"], (
            "id stays the PK; cross-platform uniqueness is on the UNIQUE index"
        )
        # Verify the UNIQUE constraint covers (platform, tx_hash, asset_id, wallet_address).
        idx_rows = conn.execute("PRAGMA index_list(training_examples)").fetchall()
        unique_idx = next(row for row in idx_rows if row["unique"] == 1 and row["origin"] == "u")
        unique_cols = sorted(
            row[2] for row in conn.execute(f"PRAGMA index_info({unique_idx['name']})").fetchall()
        )
        assert unique_cols == ["asset_id", "platform", "tx_hash", "wallet_address"]
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


def _assert_corpus_markets_round_trip(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT condition_id, event_slug, closed_at, total_volume_usd, "
        "backfill_state, enumerated_at FROM corpus_markets"
    ).fetchone()
    assert row["condition_id"] == "cond1"
    assert row["event_slug"] == "slug1"
    assert row["closed_at"] == 1000
    assert row["total_volume_usd"] == 5_000_000.0
    assert row["backfill_state"] == "complete"
    assert row["enumerated_at"] == 999


def _assert_corpus_trades_round_trip(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT tx_hash, asset_id, wallet_address, condition_id, outcome_side, "
        "bs, price, size, notional_usd, ts FROM corpus_trades"
    ).fetchone()
    assert row["tx_hash"] == "0xtx"
    assert row["asset_id"] == "asset1"
    assert row["wallet_address"] == "0xw"
    assert row["condition_id"] == "cond1"
    assert row["outcome_side"] == "YES"
    assert row["bs"] == "BUY"
    assert row["price"] == 0.5
    assert row["size"] == 100.0
    assert row["notional_usd"] == 50.0
    assert row["ts"] == 1000


def _assert_market_resolutions_round_trip(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT condition_id, winning_outcome_index, outcome_yes_won, "
        "resolved_at, source, recorded_at FROM market_resolutions"
    ).fetchone()
    assert row["condition_id"] == "cond1"
    assert row["winning_outcome_index"] == 0
    assert row["outcome_yes_won"] == 1
    assert row["resolved_at"] == 1500
    assert row["source"] == "gamma"
    assert row["recorded_at"] == 1500


def _assert_asset_index_round_trip(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT asset_id, condition_id, outcome_side, outcome_index FROM asset_index"
    ).fetchone()
    assert row["asset_id"] == "asset1"
    assert row["condition_id"] == "cond1"
    assert row["outcome_side"] == "YES"
    assert row["outcome_index"] == 0


def _assert_training_examples_round_trip(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at, "
        "prior_trades_count, prior_buys_count, prior_resolved_buys, "
        "prior_wins, prior_losses, wallet_age_days, prior_trades_30d, "
        "category_diversity, bet_size_usd, "
        "side, implied_prob_at_buy, market_category, market_volume_so_far_usd, "
        "market_unique_traders_so_far, market_age_seconds, label_won, "
        "prior_realized_pnl_usd, edge_confidence_weighted, "
        "win_rate_confidence_weighted, is_high_quality_wallet, "
        "bet_size_relative_to_history FROM training_examples"
    ).fetchone()
    assert row["tx_hash"] == "0xtx"
    assert row["asset_id"] == "asset1"
    assert row["wallet_address"] == "0xw"
    assert row["condition_id"] == "cond1"
    assert row["trade_ts"] == 1000
    assert row["built_at"] == 1500
    assert row["prior_trades_count"] == 0
    assert row["prior_buys_count"] == 0
    assert row["prior_resolved_buys"] == 0
    assert row["prior_wins"] == 0
    assert row["prior_losses"] == 0
    assert row["wallet_age_days"] == 1.0
    assert row["prior_trades_30d"] == 0
    assert row["category_diversity"] == 1
    assert row["bet_size_usd"] == 50.0
    assert row["side"] == "YES"
    assert row["implied_prob_at_buy"] == 0.5
    assert row["market_category"] == "sports"
    assert row["market_volume_so_far_usd"] == 1000.0
    assert row["market_unique_traders_so_far"] == 1
    assert row["market_age_seconds"] == 3600
    assert row["label_won"] == 1
    # Schema defaults preserved through the copy.
    assert row["prior_realized_pnl_usd"] == 0
    assert row["edge_confidence_weighted"] == 0
    assert row["win_rate_confidence_weighted"] == 0
    assert row["is_high_quality_wallet"] == 0
    assert row["bet_size_relative_to_history"] == 1


def test_apply_migrations_adds_platform_to_existing_corpus() -> None:
    """A pre-existing on-disk corpus (old schema) gets migrated in place
    with every existing row backfilled to platform='polymarket'."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "old.sqlite3"
        # Build a pre-PR-A DB by hand using the OLD schema.
        old_conn = sqlite3.connect(str(db_path))
        old_conn.row_factory = sqlite3.Row
        old_conn.executescript(
            """
            CREATE TABLE corpus_markets (
              condition_id TEXT PRIMARY KEY,
              event_slug TEXT NOT NULL,
              category TEXT,
              closed_at INTEGER NOT NULL,
              total_volume_usd REAL NOT NULL,
              backfill_state TEXT NOT NULL,
              last_offset_seen INTEGER,
              trades_pulled_count INTEGER NOT NULL DEFAULT 0,
              truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
              error_message TEXT,
              enumerated_at INTEGER NOT NULL,
              backfill_started_at INTEGER,
              backfill_completed_at INTEGER,
              market_slug TEXT,
              onchain_trades_count INTEGER,
              onchain_processed_at INTEGER
            );
            CREATE TABLE corpus_trades (
              tx_hash TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              wallet_address TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              bs TEXT NOT NULL,
              price REAL NOT NULL,
              size REAL NOT NULL,
              notional_usd REAL NOT NULL,
              ts INTEGER NOT NULL,
              UNIQUE(tx_hash, asset_id, wallet_address)
            );
            CREATE TABLE market_resolutions (
              condition_id TEXT PRIMARY KEY,
              winning_outcome_index INTEGER NOT NULL,
              outcome_yes_won INTEGER NOT NULL,
              resolved_at INTEGER NOT NULL,
              source TEXT NOT NULL,
              recorded_at INTEGER NOT NULL
            );
            CREATE TABLE asset_index (
              asset_id TEXT PRIMARY KEY,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              outcome_index INTEGER NOT NULL
            );
            CREATE TABLE training_examples (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tx_hash TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              wallet_address TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              trade_ts INTEGER NOT NULL,
              built_at INTEGER NOT NULL,
              prior_trades_count INTEGER NOT NULL,
              prior_buys_count INTEGER NOT NULL,
              prior_resolved_buys INTEGER NOT NULL,
              prior_wins INTEGER NOT NULL,
              prior_losses INTEGER NOT NULL,
              win_rate REAL,
              avg_implied_prob_paid REAL,
              realized_edge_pp REAL,
              prior_realized_pnl_usd REAL NOT NULL DEFAULT 0,
              avg_bet_size_usd REAL,
              median_bet_size_usd REAL,
              wallet_age_days REAL NOT NULL,
              seconds_since_last_trade INTEGER,
              prior_trades_30d INTEGER NOT NULL,
              top_category TEXT,
              category_diversity INTEGER NOT NULL,
              bet_size_usd REAL NOT NULL,
              bet_size_rel_to_avg REAL,
              edge_confidence_weighted REAL NOT NULL DEFAULT 0,
              win_rate_confidence_weighted REAL NOT NULL DEFAULT 0,
              is_high_quality_wallet INTEGER NOT NULL DEFAULT 0,
              bet_size_relative_to_history REAL NOT NULL DEFAULT 1,
              side TEXT NOT NULL,
              implied_prob_at_buy REAL NOT NULL,
              market_category TEXT NOT NULL,
              market_volume_so_far_usd REAL NOT NULL,
              market_unique_traders_so_far INTEGER NOT NULL,
              market_age_seconds INTEGER NOT NULL,
              time_to_resolution_seconds INTEGER,
              last_trade_price REAL,
              price_volatility_recent REAL,
              label_won INTEGER NOT NULL,
              UNIQUE(tx_hash, asset_id, wallet_address)
            );
            INSERT INTO corpus_markets(condition_id, event_slug, closed_at, total_volume_usd,
                                       backfill_state, enumerated_at)
              VALUES ('cond1', 'slug1', 1000, 5000000.0, 'complete', 999);
            INSERT INTO corpus_trades VALUES
              ('0xtx', 'asset1', '0xw', 'cond1', 'YES', 'BUY', 0.5, 100.0, 50.0, 1000);
            INSERT INTO market_resolutions VALUES ('cond1', 0, 1, 1500, 'gamma', 1500);
            INSERT INTO asset_index VALUES ('asset1', 'cond1', 'YES', 0);
            INSERT INTO training_examples (
              tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
              prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, wallet_age_days, prior_trades_30d,
              category_diversity, bet_size_usd,
              side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds, label_won
            ) VALUES (
              '0xtx', 'asset1', '0xw', 'cond1', 1000, 1500,
              0, 0, 0, 0, 0, 1.0, 0,
              1, 50.0,
              'YES', 0.5, 'sports', 1000.0,
              1, 3600, 1
            );
            """
        )
        old_conn.commit()
        old_conn.close()

        # init_corpus_db should detect the missing `platform` column and migrate.
        conn = init_corpus_db(db_path)
        try:
            for table in (
                "corpus_markets",
                "corpus_trades",
                "market_resolutions",
                "asset_index",
                "training_examples",
            ):
                rows = conn.execute(f"SELECT platform FROM {table}").fetchall()  # noqa: S608
                assert len(rows) == 1, f"{table}: expected 1 row, got {len(rows)}"
                assert rows[0]["platform"] == "polymarket"
            # PK is composite
            info = conn.execute("PRAGMA table_info(corpus_markets)").fetchall()
            pk_cols = sorted([r[1] for r in info if r[5] > 0])
            assert pk_cols == ["condition_id", "platform"]
            # training_examples preserves its legacy `id` AUTOINCREMENT PK
            # (used by ``pscanner.ml.streaming`` for chunk-iteration ORDER BY).
            # Cross-platform uniqueness is enforced via a UNIQUE index instead.
            te_info = conn.execute("PRAGMA table_info(training_examples)").fetchall()
            te_cols = {r[1] for r in te_info}
            assert "id" in te_cols, "legacy id column must survive migration"
            te_pk = sorted([r[1] for r in te_info if r[5] > 0])
            assert te_pk == ["id"], "id stays the PK after migration"

            # Round-trip data assertions: every original column value must
            # survive the table-copy. Catches column-list typos in the
            # migration's INSERT/SELECT pairs that the platform-only check
            # would miss.
            _assert_corpus_markets_round_trip(conn)
            _assert_corpus_trades_round_trip(conn)
            _assert_market_resolutions_round_trip(conn)
            _assert_asset_index_round_trip(conn)
            _assert_training_examples_round_trip(conn)
        finally:
            conn.close()


def test_apply_migrations_is_idempotent_on_already_migrated_db() -> None:
    """Calling init_corpus_db twice on a migrated DB is a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "fresh.sqlite3"
        conn1 = init_corpus_db(db_path)
        conn1.execute(
            """
            INSERT INTO corpus_markets(platform, condition_id, event_slug, closed_at,
                                       total_volume_usd, backfill_state, enumerated_at)
              VALUES ('polymarket', 'cond1', 'slug1', 1000, 5000000.0, 'complete', 999)
            """
        )
        conn1.commit()
        conn1.close()
        # Second init must not raise and must preserve the row.
        conn2 = init_corpus_db(db_path)
        try:
            count = conn2.execute("SELECT COUNT(*) AS n FROM corpus_markets").fetchone()["n"]
            assert count == 1
        finally:
            conn2.close()


def test_check_constraint_rejects_invalid_platform() -> None:
    """A row with platform NOT IN ('polymarket','kalshi','manifold') is rejected."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        try:
            conn.execute(
                """
                INSERT INTO corpus_markets(platform, condition_id, event_slug, closed_at,
                                           total_volume_usd, backfill_state, enumerated_at)
                  VALUES ('nonsense', 'cond1', 'slug1', 1000, 5000000.0, 'complete', 999)
                """
            )
            conn.commit()
            raise AssertionError("expected CHECK constraint to reject")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


def test_init_corpus_db_creates_covering_index(tmp_path: Path) -> None:
    """The chronological scan index covers every build-features SELECT column (#114)."""
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='corpus_trades' "
            "AND name='idx_corpus_trades_chrono_covering'"
        ).fetchall()
        assert len(rows) == 1, "covering index missing"

        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' "
            "AND name='idx_corpus_trades_platform_ts_tx_asset'"
        ).fetchall()
        assert len(rows) == 0, "old non-covering index still present"

        info = conn.execute("PRAGMA index_info(idx_corpus_trades_chrono_covering)").fetchall()
        cols_in_index = {row["name"] for row in info}
        required = {
            "platform",
            "ts",
            "tx_hash",
            "asset_id",
            "wallet_address",
            "condition_id",
            "outcome_side",
            "bs",
            "price",
            "size",
            "notional_usd",
        }
        assert required.issubset(cols_in_index), (
            f"covering index missing columns: {required - cols_in_index}"
        )
    finally:
        conn.close()


def test_corpus_markets_has_tags_json_column() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(corpus_markets)").fetchall()
        cols = {row["name"] for row in info}
        assert "tags_json" in cols
        assert "categories_json" in cols
    finally:
        conn.close()


def test_corpus_markets_tags_json_defaults_to_empty_list() -> None:
    """New rows that omit tags_json / categories_json get the '[]' default."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        conn.execute(
            """
            INSERT INTO corpus_markets (
              platform, condition_id, event_slug, category, closed_at,
              total_volume_usd, backfill_state, enumerated_at
            ) VALUES ('polymarket', '0xc1', 'test', 'thesis', 0, 0.0, 'pending', 0)
            """
        )
        row = conn.execute(
            "SELECT tags_json, categories_json FROM corpus_markets WHERE condition_id = '0xc1'"
        ).fetchone()
        assert row["tags_json"] == "[]"
        assert row["categories_json"] == "[]"
    finally:
        conn.close()


def test_apply_migrations_is_idempotent_for_new_columns() -> None:
    """Re-running ``_apply_migrations`` on a populated DB is a no-op."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        _apply_migrations(conn)  # re-run; should swallow "duplicate column name"
        info = conn.execute("PRAGMA table_info(corpus_markets)").fetchall()
        cols = {row["name"] for row in info}
        assert "tags_json" in cols
        assert "categories_json" in cols
    finally:
        conn.close()


def test_training_examples_has_cat_columns() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(training_examples)").fetchall()
        cols = {row["name"] for row in info}
        expected = {
            "cat_sports",
            "cat_esports",
            "cat_thesis",
            "cat_macro",
            "cat_elections",
            "cat_crypto",
            "cat_geopolitics",
            "cat_tech",
            "cat_culture",
        }
        assert expected.issubset(cols)
    finally:
        conn.close()


def test_training_examples_cat_columns_default_to_zero() -> None:
    """A row inserted without specifying cat_* columns gets 0 defaults."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        conn.execute(
            """
            INSERT INTO training_examples (
              platform, tx_hash, asset_id, wallet_address, condition_id, trade_ts,
              built_at, prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, prior_realized_pnl_usd,
              wallet_age_days, prior_trades_30d, category_diversity, bet_size_usd,
              side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds, label_won
            ) VALUES (
              'polymarket', '0xa', '0xb', '0xc', '0xc1', 0,
              0, 0, 0, 0, 0, 0, 0,
              0.0, 0, 0, 1.0,
              'YES', 0.5, 'sports', 0.0, 0, 0, 0
            )
            """
        )
        row = conn.execute(
            "SELECT cat_sports, cat_esports, cat_thesis, cat_macro, cat_elections, "
            "cat_crypto, cat_geopolitics, cat_tech, cat_culture FROM training_examples "
            "WHERE tx_hash = '0xa'"
        ).fetchone()
        for col in row:
            assert row[col] == 0, f"{col} should default to 0"
    finally:
        conn.close()


def test_training_examples_ddl_parametrized_table_name(tmp_path: Path) -> None:
    """training_examples_ddl(name) returns a CREATE TABLE statement that
    builds a table with the same shape as the canonical schema."""
    sql = training_examples_ddl("training_examples_x")
    assert "CREATE TABLE training_examples_x" in sql
    assert "PRIMARY KEY AUTOINCREMENT" in sql
    assert "UNIQUE (platform, tx_hash, asset_id, wallet_address)" in sql


def test_training_examples_columns_tuple_matches_ddl(tmp_path: Path) -> None:
    """TRAINING_EXAMPLES_COLUMNS lists every non-identity column in
    insertion order, matching what the DDL declares."""
    db_path = tmp_path / "x.sqlite3"
    conn = init_corpus_db(db_path)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(training_examples)").fetchall()]
    finally:
        conn.close()
    # id + built_at are identity-ish; the tuple lists the columns the
    # engine INSERTs into.
    expected_inserted = [c for c in cols if c != "id"]
    assert list(TRAINING_EXAMPLES_COLUMNS) == expected_inserted


def test_cross_platform_rows_coexist() -> None:
    """Same condition_id under different platforms = two distinct rows."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        for platform in ("polymarket", "kalshi", "manifold"):
            conn.execute(
                """
                INSERT INTO corpus_markets(platform, condition_id, event_slug, closed_at,
                                           total_volume_usd, backfill_state, enumerated_at)
                  VALUES (?, 'shared-id', 'slug', 1000, 1.0, 'pending', 999)
                """,
                (platform,),
            )
        conn.commit()
        rows = conn.execute("SELECT platform FROM corpus_markets ORDER BY platform").fetchall()
        assert [r["platform"] for r in rows] == ["kalshi", "manifold", "polymarket"]
    finally:
        conn.close()
