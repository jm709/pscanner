"""CLI dispatch tests for `pscanner corpus build-features`."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus._build_features_sentinel import SENTINEL_KEY, SentinelAlreadySetError
from pscanner.corpus.cli import _cmd_build_features, build_corpus_parser
from pscanner.corpus.repos import CorpusStateRepo
from tests.corpus._duckdb_fixture import build_fixture_db


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


def test_build_features_refuses_with_existing_sentinel(tmp_path: Path) -> None:
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
        asyncio.run(_cmd_build_features(args))
