"""CLI dispatch tests for `pscanner corpus build-features`."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus._build_features_sentinel import SENTINEL_KEY, SentinelAlreadySetError
from pscanner.corpus.cli import (
    _cmd_build_features,
    _default_duckdb_memory,
    build_corpus_parser,
    run_corpus_command,
)
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import CorpusStateRepo
from tests.corpus._duckdb_fixture import build_fixture_db


def _set_sentinel(db_path: Path, ts: int = 1_700_000_000) -> None:
    """Mirror what check_and_set_sentinel does on first run."""
    conn = init_corpus_db(db_path)
    try:
        CorpusStateRepo(conn).set(SENTINEL_KEY, str(ts), updated_at=ts)
    finally:
        conn.close()


def test_build_features_parser_accepts_engine_flag() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(
        ["build-features", "--engine", "duckdb", "--force", "--db", "x.sqlite3"]
    )
    assert args.engine == "duckdb"
    assert args.force is True


def test_build_features_parser_defaults_to_python_engine() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--db", "x.sqlite3"])
    assert args.engine == "python"
    assert args.force is False


@pytest.mark.asyncio
async def test_build_features_refuses_with_existing_sentinel(tmp_path: Path) -> None:
    """End-to-end: sentinel set → dispatch raises SentinelAlreadySetError."""
    db = tmp_path / "corpus.sqlite3"
    build_fixture_db(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        CorpusStateRepo(conn).set(SENTINEL_KEY, "1700000000", updated_at=1_700_000_000)
    finally:
        conn.close()

    args = argparse.Namespace(
        db=str(db),
        platform="polymarket",
        rebuild=True,
        engine="python",
        force=False,
        duckdb_memory="1GB",
        duckdb_threads=2,
    )
    with pytest.raises(SentinelAlreadySetError):
        await _cmd_build_features(args)


@pytest.mark.asyncio
async def test_build_features_refuses_with_stale_sentinel_and_no_force(tmp_path: Path) -> None:
    """When build_features_in_progress is set, the CLI refuses with
    SentinelAlreadySetError unless --force or --reset-scratch is passed."""
    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()
    _set_sentinel(db_path)

    with pytest.raises(SentinelAlreadySetError):
        await run_corpus_command(["build-features", "--db", str(db_path), "--engine", "duckdb"])


@pytest.mark.asyncio
async def test_build_features_reset_scratch_overrides_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--reset-scratch implies --force on the sentinel AND wipes the
    scratch DuckDB file from a prior crashed run."""
    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()
    _set_sentinel(db_path)

    # Drop a stale scratch file to verify it gets wiped.
    spill_dir = db_path.parent / "duckdb_spill"
    spill_dir.mkdir()
    stale = spill_dir / "build_scratch.duckdb"
    stale.write_bytes(b"stale")
    assert stale.exists()

    # Short-circuit the actual build so the test stays fast — we only
    # care that the sentinel + scratch wipe ran. Replace the engine
    # function with a stub that returns 0.
    monkeypatch.setattr(
        "pscanner.corpus.cli.build_features_duckdb",
        lambda **_: 0,
    )

    rc = await run_corpus_command(
        [
            "build-features",
            "--db",
            str(db_path),
            "--engine",
            "duckdb",
            "--reset-scratch",
        ]
    )
    assert rc == 0
    assert not stale.exists()


def test_default_duckdb_memory_brackets(monkeypatch):
    """Auto-detected budget falls within the documented bracket per host RAM."""

    class _FakeVM:
        def __init__(self, total_gb: float):
            self.total = int(total_gb * 1024**3)

    cases = [
        (32.0, "8GB"),
        (16.0, "8GB"),
        (12.0, "6GB"),
        (8.0, "3GB"),
        (5.5, None),  # refuse: below floor
    ]
    for total_gb, expected in cases:
        monkeypatch.setattr(
            "pscanner.corpus.cli.psutil.virtual_memory",
            lambda total=total_gb: _FakeVM(total),
        )
        if expected is None:
            with pytest.raises(RuntimeError, match="insufficient"):
                _default_duckdb_memory()
        else:
            assert _default_duckdb_memory() == expected
