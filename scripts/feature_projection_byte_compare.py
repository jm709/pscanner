r"""One-shot byte-compare gate for the feature-projection switch (#145).

Reads a real corpus DB, runs build-features under both the python and
duckdb engines into two copies of training_examples, and reports any
divergence at row granularity. Run before merging the DuckDB switch.

Usage:
    uv run python scripts/feature_projection_byte_compare.py \
        --corpus ./data/corpus.sqlite3 \
        --sample-size 100000

The script copies the source corpus to two tmp DBs, runs build-features
on each, and diffs training_examples row-by-row. Exits non-zero on any
divergence; prints up to 20 example divergences to stdout.

This is a manual gate — the result should be pasted into the PR
description as evidence before merging.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

_FLOAT_REL_TOL = 1e-5
_FLOAT_ABS_TOL = 1e-7
_MAX_EXAMPLES_SHOWN = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="path to corpus.sqlite3")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="cap on number of training_examples rows compared (default 100000)",
    )
    parser.add_argument(
        "--duckdb-memory",
        default="6GB",
        help="DuckDB memory limit (default 6GB)",
    )
    return parser.parse_args()


def _floats_close(a: object, b: object) -> bool:
    """Check if two float values are numerically close."""
    if a is None and b is None:
        return True
    if not isinstance(a, float) or not isinstance(b, float):
        return False
    return abs(a - b) < max(_FLOAT_ABS_TOL, _FLOAT_REL_TOL * max(abs(a), abs(b)))


def _read_rows(db_path: Path, limit: int) -> Iterator[dict[str, object]]:
    """Read training_examples rows from a DB in deterministic order."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(
            "SELECT * FROM training_examples "
            "ORDER BY trade_ts, wallet_address, condition_id, asset_id "
            "LIMIT ?",
            (limit,),
        ):
            yield dict(row)
    finally:
        conn.close()


def _run_build_features(db_path: Path, engine: str, duckdb_memory: str) -> None:
    """Run build-features with specified engine via subprocess."""
    cmd = [
        "uv",
        "run",
        "pscanner",
        "corpus",
        "build-features",
        "--engine",
        engine,
        "--rebuild",
        "--db",
        str(db_path),
    ]
    if engine == "duckdb":
        cmd.extend(["--duckdb-memory", duckdb_memory])
    subprocess.run(cmd, check=True)  # noqa: S603


def _get_row_diff(p: dict[str, object], d: dict[str, object]) -> list[tuple[str, object, object]]:
    """Find columns where two rows diverge."""
    diff = []
    for col in sorted(p.keys() | d.keys()):
        pv, dv = p.get(col), d.get(col)
        if isinstance(pv, float) or isinstance(dv, float):
            if not _floats_close(pv, dv):
                diff.append((col, pv, dv))
        elif pv != dv:
            diff.append((col, pv, dv))
    return diff


def _print_row_example(
    i: int, p: dict[str, object], diff: list[tuple[str, object, object]]
) -> None:
    """Print a single divergence example."""
    print(  # noqa: T201
        f"\n  ROW {i} (tx_hash={p.get('tx_hash')}):"
    )
    for col, pv, dv in diff:
        print(  # noqa: T201
            f"    {col}: py={pv!r}  duck={dv!r}"
        )


def _compare_rows(py_db: Path, duck_db: Path, sample_size: int) -> int:
    """Compare rows between two DBs, return count of divergences."""
    divergences = 0
    examples_shown = 0
    # Defensive row-count check before per-row comparison. A divergence
    # in row count is itself a parity failure, not just a column-level diff.
    py_count = sum(1 for _ in _read_rows(py_db, sample_size))
    duck_count = sum(1 for _ in _read_rows(duck_db, sample_size))
    if py_count != duck_count:
        print(  # noqa: T201
            f"\nROW COUNT MISMATCH: python={py_count} duckdb={duck_count}",
            file=sys.stderr,
        )
        return 1
    for i, (p, d) in enumerate(
        zip(
            _read_rows(py_db, sample_size),
            _read_rows(duck_db, sample_size),
            strict=True,
        )
    ):
        row_diff = _get_row_diff(p, d)
        if row_diff:
            divergences += 1
            if examples_shown < _MAX_EXAMPLES_SHOWN:
                _print_row_example(i, p, row_diff)
                examples_shown += 1
    return divergences


def main() -> int:
    """Run byte-compare gate between python and duckdb engines."""
    args = parse_args()
    if not args.corpus.exists():
        print(f"ERROR: corpus DB not found at {args.corpus}", file=sys.stderr)  # noqa: T201
        return 2

    with tempfile.TemporaryDirectory(prefix="feature-projection-bytecompare-") as tmpdir:
        tmp = Path(tmpdir)
        py_db = tmp / "py.sqlite3"
        duck_db = tmp / "duck.sqlite3"

        print(  # noqa: T201
            f"[1/4] Copying corpus to {py_db} and {duck_db} ..."
        )
        shutil.copy(args.corpus, py_db)
        shutil.copy(args.corpus, duck_db)

        # Clear training_examples in both
        for path in (py_db, duck_db):
            conn = sqlite3.connect(path)
            try:
                conn.execute("DELETE FROM training_examples")
                conn.commit()
            finally:
                conn.close()

        print("[2/4] Running build-features --engine python ...")  # noqa: T201
        _run_build_features(py_db, "python", "")

        print(  # noqa: T201
            f"[3/4] Running build-features --engine duckdb (memory={args.duckdb_memory}) ..."
        )
        _run_build_features(duck_db, "duckdb", args.duckdb_memory)

        print(  # noqa: T201
            f"[4/4] Comparing up to {args.sample_size} rows ..."
        )
        divergences = _compare_rows(py_db, duck_db, args.sample_size)

        if divergences:
            print(  # noqa: T201
                f"\nRESULT: {divergences} diverging rows out of ~{args.sample_size} sampled."
            )
            return 1
        print(  # noqa: T201
            "\nRESULT: all sampled rows agree byte-for-byte."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
