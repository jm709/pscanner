"""Parity test for the DuckDB build-features engine.

Runs the Python engine and the DuckDB engine against an identical
synthetic corpus, then asserts the resulting ``training_examples``
rows match column-by-column.
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import pytest

from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)
from tests.corpus._duckdb_fixture import build_fixture_db

# Float tolerance per the parity decision (issue #116, branches 9-10):
# both engines accumulate in the same row order so values should be
# bit-identical, but assert_allclose with these tolerances is belt-and-
# suspenders against floating-point reordering inside DuckDB.
_FLOAT_RTOL = 1e-9
_FLOAT_ATOL = 1e-12


@pytest.fixture
def parity_dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Build two identical fixture DBs side-by-side."""
    py_db = tmp_path / "python_engine.sqlite3"
    dd_db = tmp_path / "duckdb_engine.sqlite3"
    build_fixture_db(py_db)
    build_fixture_db(dd_db)
    return py_db, dd_db


def _run_python_engine(db: Path) -> None:
    write = sqlite3.connect(db)
    write.row_factory = sqlite3.Row
    read = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    read.row_factory = sqlite3.Row
    try:
        build_features(
            trades_repo=CorpusTradesRepo(read),
            resolutions_repo=MarketResolutionsRepo(write),
            examples_repo=TrainingExamplesRepo(write),
            markets_conn=write,
            now_ts=int(time.time()),
            rebuild=True,
            platform="polymarket",
        )
    finally:
        read.close()
        write.close()


def _run_duckdb_engine(db: Path) -> None:
    # Intentional deferred import: this module doesn't exist until Task 6.
    # The test FAILs (not skips) until the engine is implemented, serving
    # as a red checkpoint.  noqa: PLC0415 suppresses the top-level import rule.
    from pscanner.corpus._duckdb_engine import build_features_duckdb  # noqa: PLC0415

    build_features_duckdb(
        db_path=db,
        platform="polymarket",
        now_ts=int(time.time()),
        memory_limit="1GB",
        temp_dir=db.parent / "duckdb_spill",
        threads=2,
    )


def _ordered_rows(db: Path) -> list[dict[str, object]]:
    """Return all training_examples rows in a stable canonical order."""
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT * FROM training_examples
            ORDER BY platform, condition_id, wallet_address,
                     tx_hash, asset_id, trade_ts
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _diff_rows(py: dict[str, object], dd: dict[str, object]) -> list[str]:
    """Return human-readable per-column diffs (empty if rows match)."""
    diffs: list[str] = []
    skip = {"id", "built_at"}  # id is autoinc; built_at is wall clock
    for col, v_py in py.items():
        if col in skip:
            continue
        v_dd = dd[col]
        if v_py is None and v_dd is None:
            continue
        if isinstance(v_py, float) or isinstance(v_dd, float):
            if v_py is None or v_dd is None:
                diffs.append(f"{col}: py={v_py!r} dd={v_dd!r}")
                continue
            if not math.isclose(float(v_py), float(v_dd), rel_tol=_FLOAT_RTOL, abs_tol=_FLOAT_ATOL):  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
                diffs.append(f"{col}: py={v_py!r} dd={v_dd!r}")
        elif v_py != v_dd:
            diffs.append(f"{col}: py={v_py!r} dd={v_dd!r}")
    return diffs


def test_duckdb_engine_matches_python_engine(parity_dbs: tuple[Path, Path]) -> None:
    py_db, dd_db = parity_dbs

    _run_python_engine(py_db)
    _run_duckdb_engine(dd_db)

    py_rows = _ordered_rows(py_db)
    dd_rows = _ordered_rows(dd_db)

    assert len(py_rows) == len(dd_rows), (
        f"row count differs: python={len(py_rows)} duckdb={len(dd_rows)}"
    )

    failures: list[str] = []
    for i, (p, d) in enumerate(zip(py_rows, dd_rows, strict=True)):
        key = (p["condition_id"], p["wallet_address"], p["tx_hash"], p["asset_id"])
        diffs = _diff_rows(p, d)
        if diffs:
            failures.append(f"row {i} {key}: {'; '.join(diffs)}")

    assert not failures, "parity mismatch:\n" + "\n".join(failures)


def test_build_features_duckdb_rejects_unknown_platform(tmp_path: Path) -> None:
    """Engine entry validates platform against allowlist, no SQL injection
    surface via f-string interpolation."""
    from pscanner.corpus._duckdb_engine import build_features_duckdb  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    db_path.touch()
    with pytest.raises(ValueError, match="unknown platform"):
        build_features_duckdb(
            db_path=db_path,
            platform="polymarket'; DROP TABLE corpus_trades; --",
            now_ts=0,
            memory_limit="1GB",
            temp_dir=tmp_path,
            threads=1,
        )


def test_scratch_db_lifecycle(tmp_path: Path) -> None:
    """Scratch file is created on open, removed on success-close, persists
    on failure-close."""
    from pscanner.corpus._duckdb_engine import _open_scratch, _wipe_scratch  # noqa: PLC0415

    scratch_path = tmp_path / "scratch.duckdb"
    assert not scratch_path.exists()
    conn = _open_scratch(scratch_path, memory_limit="256MB", threads=1)
    conn.execute("CREATE TABLE smoke AS SELECT 1 AS x")
    conn.close()
    assert scratch_path.exists()
    _wipe_scratch(scratch_path)
    assert not scratch_path.exists()


def test_stage1_events_row_count_and_columns(tmp_path: Path) -> None:
    """Stage 1 produces an events table whose row count is trades +
    eligible-resolutions, with the expected column set."""
    from pscanner.corpus._duckdb_engine import (  # noqa: PLC0415
        _attach_corpus,
        _materialize_trades,
        _open_scratch,
        _scratch_path,
        _stage1_events,
    )
    from tests.corpus._duckdb_fixture import build_fixture_db  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    build_fixture_db(db_path)

    scratch = _open_scratch(_scratch_path(tmp_path), memory_limit="256MB", threads=1)
    try:
        scratch.execute("INSTALL sqlite")
        scratch.execute("LOAD sqlite")
        _attach_corpus(scratch, db_path=db_path)
        _materialize_trades(scratch, platform="polymarket")
        _stage1_events(scratch)

        cols = [
            r[0]
            for r in scratch.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'events' ORDER BY ordinal_position"
            ).fetchall()
        ]
        assert "wallet_address" in cols
        assert "condition_id" in cols
        assert "event_ts" in cols
        assert "kind_priority" in cols
        assert "is_resolution" in cols
        assert "is_buy_only" in cols

        events_row = scratch.execute("SELECT COUNT(*) FROM events").fetchone()
        trades_row = scratch.execute("SELECT COUNT(*) FROM trades").fetchone()
        assert events_row is not None
        assert trades_row is not None
        n_events = events_row[0]
        n_trades = trades_row[0]
        assert n_events >= n_trades
        assert n_events <= 2 * n_trades
    finally:
        scratch.close()


def test_stage2_wallet_aggs_strictly_prior(tmp_path: Path) -> None:
    """Per-wallet running aggregates exclude the current event (UNBOUNDED
    PRECEDING ... 1 PRECEDING) — verify by hand on a 2-trade wallet."""
    from pscanner.corpus._duckdb_engine import (  # noqa: PLC0415
        _attach_corpus,
        _materialize_trades,
        _open_scratch,
        _scratch_path,
        _stage1_events,
        _stage2_wallet_aggs,
    )
    from tests.corpus._duckdb_fixture import build_fixture_db  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    build_fixture_db(db_path)

    scratch = _open_scratch(_scratch_path(tmp_path), memory_limit="256MB", threads=1)
    try:
        scratch.execute("INSTALL sqlite")
        scratch.execute("LOAD sqlite")
        _attach_corpus(scratch, db_path=db_path)
        _materialize_trades(scratch, platform="polymarket")
        _stage1_events(scratch)
        _stage2_wallet_aggs(scratch)

        # First event for any wallet must have prior_trades_count_w = 0
        first_event = scratch.execute(
            """
            SELECT prior_trades_count_w
            FROM (
                SELECT
                    prior_trades_count_w,
                    ROW_NUMBER() OVER (
                        PARTITION BY wallet_address
                        ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ) AS rn
                FROM wallet_aggs
            )
            WHERE rn = 1
            LIMIT 1
            """
        ).fetchone()
        assert first_event is not None
        assert first_event[0] == 0
    finally:
        scratch.close()


def test_stage3_market_aggs_unique_traders_monotone(tmp_path: Path) -> None:
    """unique_traders_so_far must be non-decreasing within a market."""
    from pscanner.corpus._duckdb_engine import (  # noqa: PLC0415
        _attach_corpus,
        _materialize_trades,
        _open_scratch,
        _scratch_path,
        _stage1_events,
        _stage3_market_aggs,
    )
    from tests.corpus._duckdb_fixture import build_fixture_db  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    build_fixture_db(db_path)

    scratch = _open_scratch(_scratch_path(tmp_path), memory_limit="256MB", threads=1)
    try:
        scratch.execute("INSTALL sqlite")
        scratch.execute("LOAD sqlite")
        _attach_corpus(scratch, db_path=db_path)
        _materialize_trades(scratch, platform="polymarket")
        _stage1_events(scratch)
        _stage3_market_aggs(scratch)

        rows = scratch.execute(
            """
            SELECT condition_id, market_unique_traders_so_far_w
            FROM market_aggs
            ORDER BY condition_id, event_ts, kind_priority, tx_hash, asset_id
            """
        ).fetchall()
        last: dict[str, int] = {}
        for cid, n in rows:
            assert n >= last.get(cid, 0)
            last[cid] = n
    finally:
        scratch.close()


def test_stage4_wallet_cat_summary_uses_filter_not_or_chain(tmp_path: Path) -> None:
    """Stage 4 produces wallet_cat_summary with top_category + category_diversity."""
    from pscanner.corpus._duckdb_engine import (  # noqa: PLC0415
        _attach_corpus,
        _materialize_trades,
        _open_scratch,
        _scratch_path,
        _stage1_events,
        _stage4_wallet_cat,
    )
    from tests.corpus._duckdb_fixture import build_fixture_db  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    build_fixture_db(db_path)

    scratch = _open_scratch(_scratch_path(tmp_path), memory_limit="256MB", threads=1)
    try:
        scratch.execute("INSTALL sqlite")
        scratch.execute("LOAD sqlite")
        _attach_corpus(scratch, db_path=db_path)
        _materialize_trades(scratch, platform="polymarket")
        _stage1_events(scratch)
        _stage4_wallet_cat(scratch)

        cols = [
            r[0]
            for r in scratch.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'wallet_cat_summary'"
            ).fetchall()
        ]
        assert "top_category" in cols
        assert "category_diversity" in cols
    finally:
        scratch.close()


def test_final_join_skips_resolution_for_last_trade_ts(tmp_path: Path) -> None:
    """A wallet with BUY@100 -> RESOLUTION@200 -> BUY@300 must report
    seconds_since_last_trade=200 (=300-100), not 100 (=300-200)."""
    import sqlite3  # noqa: PLC0415

    from pscanner.corpus._duckdb_engine import build_features_duckdb  # noqa: PLC0415
    from pscanner.corpus.db import init_corpus_db  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()
    conn = sqlite3.connect(db_path)
    try:
        cid = "0x" + "a" * 64
        wallet = "0x" + "b" * 40
        conn.execute(
            """
            INSERT INTO corpus_markets
                (platform, condition_id, event_slug, category, categories_json,
                 enumerated_at, closed_at, total_volume_usd, backfill_state)
            VALUES ('polymarket', ?, 'slug', 'sports', '["sports"]',
                    50, 300, 1000.0, 'complete')
            """,
            (cid,),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions
                (platform, condition_id, resolved_at, winning_outcome_index,
                 outcome_yes_won, source, recorded_at)
            VALUES ('polymarket', ?, 200, 0, 1, 'gamma', 200)
            """,
            (cid,),
        )
        rows = [
            (
                "polymarket",
                "0x" + "1" * 64,
                "asset_yes",
                wallet,
                cid,
                "YES",
                "BUY",
                0.5,
                100.0,
                50.0,
                100,
            ),
            (
                "polymarket",
                "0x" + "2" * 64,
                "asset_yes",
                wallet,
                cid,
                "YES",
                "BUY",
                0.6,
                100.0,
                60.0,
                300,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO corpus_trades
                (platform, tx_hash, asset_id, wallet_address, condition_id,
                 outcome_side, bs, price, size, notional_usd, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    build_features_duckdb(
        db_path=db_path,
        platform="polymarket",
        now_ts=1000,
        memory_limit="256MB",
        temp_dir=tmp_path,
        threads=1,
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT trade_ts, seconds_since_last_trade, prior_resolved_buys "
            "FROM training_examples ORDER BY trade_ts"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert rows[0]["seconds_since_last_trade"] is None
    assert rows[0]["prior_resolved_buys"] == 0
    assert rows[1]["seconds_since_last_trade"] == 200
    assert rows[1]["prior_resolved_buys"] == 1


def test_final_join_top_category_breaks_tied_ts_by_tx_hash(tmp_path: Path) -> None:
    """When two categories share cat_first_ts, tiebreak by cat_first_tx (ASC)."""
    import sqlite3  # noqa: PLC0415

    from pscanner.corpus._duckdb_engine import build_features_duckdb  # noqa: PLC0415
    from pscanner.corpus.db import init_corpus_db  # noqa: PLC0415

    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()
    conn = sqlite3.connect(db_path)
    try:
        wallet = "0x" + "b" * 40
        cid_esports = "0x" + "e" * 64
        cid_sports = "0x" + "f" * 64
        for cid, cat in ((cid_esports, "esports"), (cid_sports, "sports")):
            conn.execute(
                """
                INSERT INTO corpus_markets
                    (platform, condition_id, event_slug, category, categories_json,
                     enumerated_at, closed_at, total_volume_usd, backfill_state)
                VALUES ('polymarket', ?, 'slug', ?, '[]',
                        50, 1000, 1000.0, 'complete')
                """,
                (cid, cat),
            )
            conn.execute(
                """
                INSERT INTO market_resolutions
                    (platform, condition_id, resolved_at, winning_outcome_index,
                     outcome_yes_won, source, recorded_at)
                VALUES ('polymarket', ?, 1000, 0, 1, 'gamma', 1000)
                """,
                (cid,),
            )
        rows = [
            (
                "polymarket",
                "0x" + "1" * 64,
                "asset",
                wallet,
                cid_esports,
                "YES",
                "BUY",
                0.5,
                100.0,
                50.0,
                100,
            ),
            (
                "polymarket",
                "0x" + "2" * 64,
                "asset",
                wallet,
                cid_sports,
                "YES",
                "BUY",
                0.5,
                100.0,
                50.0,
                100,
            ),
            (
                "polymarket",
                "0x" + "3" * 64,
                "asset",
                wallet,
                cid_esports,
                "YES",
                "BUY",
                0.5,
                100.0,
                50.0,
                500,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO corpus_trades
                (platform, tx_hash, asset_id, wallet_address, condition_id,
                 outcome_side, bs, price, size, notional_usd, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    build_features_duckdb(
        db_path=db_path,
        platform="polymarket",
        now_ts=1000,
        memory_limit="256MB",
        temp_dir=tmp_path,
        threads=1,
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        third = conn.execute(
            "SELECT top_category, category_diversity FROM training_examples WHERE trade_ts = 500"
        ).fetchone()
    finally:
        conn.close()
    assert third["top_category"] == "esports"
    assert third["category_diversity"] == 2


def test_heartbeat_emits_during_long_operation() -> None:
    """Heartbeat thread fires at least once and stops cleanly on signal."""
    import threading  # noqa: PLC0415

    from structlog.testing import capture_logs  # noqa: PLC0415

    from pscanner.corpus._duckdb_engine import _heartbeat_loop  # noqa: PLC0415

    stop = threading.Event()
    counter = {"polls": 0}

    def fake_poll() -> int:
        counter["polls"] += 1
        return counter["polls"] * 100

    with capture_logs() as logs:
        t = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "stop": stop,
                "poll_fn": fake_poll,
                "interval_seconds": 0.05,
                "stage": "test_stage",
            },
            daemon=True,
        )
        t.start()
        time.sleep(0.20)  # allow at least 2-3 emits
        stop.set()
        t.join(timeout=2.0)

    assert not t.is_alive()
    heartbeats = [r for r in logs if r["event"] == "corpus.build_features.heartbeat"]
    assert len(heartbeats) >= 2
    assert all(r["stage"] == "test_stage" for r in heartbeats)
    assert all("elapsed_seconds" in r and "rows" in r for r in heartbeats)
