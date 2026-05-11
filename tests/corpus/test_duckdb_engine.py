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
            if not math.isclose(float(v_py), float(v_dd), rel_tol=_FLOAT_RTOL, abs_tol=_FLOAT_ATOL):
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
