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
