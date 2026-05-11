#!/usr/bin/env python3
# ruff: noqa: T201  # script prints progress/results to stdout by design
r"""Compare Python and DuckDB build-features engines on a real corpus.

Usage:
    uv run python scripts/parity_build_features.py \
        --source data/corpus.sqlite3 \
        --workdir /tmp/parity \
        --platform polymarket

Workflow:
    1. Copies the source corpus to two scratch DBs (python.sqlite3, duckdb.sqlite3).
    2. Runs the Python engine into python.sqlite3.
    3. Runs the DuckDB engine into duckdb.sqlite3.
    4. Streams both training_examples tables in the same canonical order.
    5. Diffs row-by-row and reports column-level mismatches with tolerance.
    6. On success, writes corpus_state['build_features_parity_passed_at']
       to the SOURCE db so PR-B's check can read it.

Wall time on production corpus: hours (dominated by the Python engine).
"""

from __future__ import annotations

import argparse
import math
import shutil
import sqlite3
import sys
import time
from pathlib import Path

_FLOAT_RTOL = 1e-9
_FLOAT_ATOL = 1e-12
_SKIP_COLS = {"id", "built_at"}
_MAX_MISMATCH_LINES = 50


def main() -> int:
    """Run both engines against copies of the source corpus and diff results."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--source", required=True, type=Path, help="Source corpus.sqlite3")
    ap.add_argument("--workdir", required=True, type=Path, help="Scratch dir for copies")
    ap.add_argument("--platform", default="polymarket")
    ap.add_argument("--duckdb-memory", default="6GB")
    ap.add_argument("--duckdb-threads", type=int, default=8)
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    py_db = args.workdir / "python_engine.sqlite3"
    dd_db = args.workdir / "duckdb_engine.sqlite3"

    print(f"[parity] copying source to {py_db}", flush=True)
    shutil.copy2(args.source, py_db)
    print(f"[parity] copying source to {dd_db}", flush=True)
    shutil.copy2(args.source, dd_db)

    print(f"[parity] running python engine on {py_db}", flush=True)
    t0 = time.monotonic()
    _run_python(py_db, platform=args.platform)
    print(f"[parity] python engine done in {time.monotonic() - t0:.1f}s", flush=True)

    print(f"[parity] running duckdb engine on {dd_db}", flush=True)
    t0 = time.monotonic()
    _run_duckdb(
        dd_db,
        platform=args.platform,
        memory=args.duckdb_memory,
        threads=args.duckdb_threads,
    )
    print(f"[parity] duckdb engine done in {time.monotonic() - t0:.1f}s", flush=True)

    print("[parity] diffing...", flush=True)
    ok = _diff(py_db, dd_db)
    if not ok:
        print("[parity] FAIL", file=sys.stderr)
        return 1

    print("[parity] PASS — writing build_features_parity_passed_at", flush=True)
    src_conn = sqlite3.connect(args.source)
    try:
        src_conn.execute(
            """
            INSERT INTO corpus_state (key, value, updated_at)
            VALUES ('build_features_parity_passed_at', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (str(int(time.time())), int(time.time())),
        )
        src_conn.commit()
    finally:
        src_conn.close()
    return 0


def _run_python(db: Path, *, platform: str) -> None:
    """Run the Python streaming engine into ``db``, rebuilding training_examples."""
    from pscanner.corpus.examples import build_features  # noqa: PLC0415
    from pscanner.corpus.repos import (  # noqa: PLC0415
        CorpusTradesRepo,
        MarketResolutionsRepo,
        TrainingExamplesRepo,
    )

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
            platform=platform,
        )
    finally:
        read.close()
        write.close()


def _run_duckdb(db: Path, *, platform: str, memory: str, threads: int) -> None:
    """Run the DuckDB engine into ``db``, rebuilding training_examples."""
    from pscanner.corpus._duckdb_engine import build_features_duckdb  # noqa: PLC0415

    build_features_duckdb(
        db_path=db,
        platform=platform,
        now_ts=int(time.time()),
        memory_limit=memory,
        temp_dir=db.parent / "duckdb_spill",
        threads=threads,
    )


def _check_col(
    col: str,
    a: object,
    b: object,
    *,
    row_num: int,
    py_row: sqlite3.Row,
    mismatches: int,
) -> bool:
    """Return True if the column values differ (mismatch detected).

    Logs the first ``_MAX_MISMATCH_LINES`` mismatches to stdout.
    """
    if a is None and b is None:
        return False
    is_mismatch = False
    if isinstance(a, float) or isinstance(b, float):
        if (
            a is None
            or b is None
            or not math.isclose(
                float(a),
                float(b),
                rel_tol=_FLOAT_RTOL,
                abs_tol=_FLOAT_ATOL,  # type: ignore[arg-type]
            )
        ):
            is_mismatch = True
    elif a != b:
        is_mismatch = True

    if is_mismatch and mismatches < _MAX_MISMATCH_LINES:
        key = (py_row["condition_id"], py_row["wallet_address"])
        print(f"[parity] row{row_num} {key} {col}: py={a!r} dd={b!r}", flush=True)
    return is_mismatch


def _diff(py_db: Path, dd_db: Path) -> bool:
    """Stream both training_examples tables and diff row-by-row."""
    order_clause = "ORDER BY platform, condition_id, wallet_address, tx_hash, asset_id, trade_ts"

    py_conn = sqlite3.connect(py_db)
    py_conn.row_factory = sqlite3.Row
    dd_conn = sqlite3.connect(dd_db)
    dd_conn.row_factory = sqlite3.Row

    n_py = py_conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    n_dd = dd_conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    print(f"[parity] python rows: {n_py:,}  duckdb rows: {n_dd:,}", flush=True)
    if n_py != n_dd:
        print("[parity] ROW COUNT MISMATCH", flush=True)
        return False

    py_cur = py_conn.execute(f"SELECT * FROM training_examples {order_clause}")  # noqa: S608
    dd_cur = dd_conn.execute(f"SELECT * FROM training_examples {order_clause}")  # noqa: S608

    mismatches = 0
    total = 0
    for py_row, dd_row in zip(py_cur, dd_cur, strict=True):
        total += 1
        for col in py_row.keys():  # noqa: SIM118 — sqlite3.Row.keys() not equivalent to __iter__
            if col in _SKIP_COLS:
                continue
            if _check_col(
                col,
                py_row[col],
                dd_row[col],
                row_num=total,
                py_row=py_row,
                mismatches=mismatches,
            ):
                mismatches += 1
        if total % 100_000 == 0:
            print(
                f"[parity] checked {total:,} rows, {mismatches} mismatches so far",
                flush=True,
            )

    print(f"[parity] total rows={total:,} mismatches={mismatches}", flush=True)
    return mismatches == 0


if __name__ == "__main__":
    raise SystemExit(main())
