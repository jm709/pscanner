"""Opt-in memory + parity regression test for the DuckDB engine.

Skipped by default; run with ``uv run pytest -m slow``.

Asserts at ~2M trades / 1K markets / 5K wallets:
  - DuckDB engine completes within wall-time budget
  - peak RSS stays under the spill-safety threshold even at a tight
    ``memory_limit`` (catches regressions where intermediate state
    isn't released between stages)
  - row count and a sampled-column parity check vs the python engine
"""

from __future__ import annotations

import math
import resource
import sqlite3
import time

import pytest

from pscanner.corpus._duckdb_engine import build_features_duckdb
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)
from tests.corpus._duckdb_scale_fixture import build_scale_fixture_db

pytestmark = pytest.mark.slow


def _peak_rss_mb() -> float:
    """Peak RSS in MB for this process so far.

    On Linux ``ru_maxrss`` is in KB; on macOS it's in bytes. The test
    only runs in CI / on dev hosts (Linux WSL2), so we assume KB.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0


def test_duckdb_engine_at_scale_memory_and_parity(tmp_path):
    db_path = tmp_path / "corpus.sqlite3"
    # 2M trades is the lower bound where the prod corpus's OR-chain
    # blowup shape kicks in. 500K fit comfortably in a 512MB budget and
    # don't actually exercise spill.
    stats = build_scale_fixture_db(db_path, target_trades=2_000_000)
    assert stats["trades"] >= 1_500_000

    # Run DuckDB engine at a tight memory budget — forces spill behavior
    # and surfaces regressions where stages don't release state.
    started_rss = _peak_rss_mb()
    started_wall = time.monotonic()
    n_duck = build_features_duckdb(
        db_path=db_path,
        platform="polymarket",
        now_ts=int(time.time()),
        memory_limit="512MB",
        temp_dir=tmp_path,
        threads=1,
    )
    duck_wall = time.monotonic() - started_wall
    duck_rss = _peak_rss_mb() - started_rss

    # RSS guard: if a stage doesn't release intermediate state, peak
    # RSS will balloon past the 512MB memory_limit. <2GB headroom is
    # the regression signal.
    assert duck_rss < 2048, (
        f"duckdb engine peak RSS {duck_rss:.0f}MB exceeds budget; "
        "stages aren't releasing intermediate state"
    )

    # Wall-time guard: 2M trades at 512MB on 1 thread is a stress test.
    # 20 min is generous — if hitting this, something has regressed.
    assert duck_wall < 1200, f"duckdb engine too slow: {duck_wall:.0f}s"

    # Read the DuckDB output, then truncate and run python engine for
    # a sampled-column parity check.
    duck_conn = sqlite3.connect(db_path)
    duck_conn.row_factory = sqlite3.Row
    try:
        duck_sample = duck_conn.execute(
            """
            SELECT tx_hash, asset_id, wallet_address,
                   prior_trades_count, win_rate, top_category, label_won
            FROM training_examples
            ORDER BY tx_hash
            LIMIT 1000
            """
        ).fetchall()
    finally:
        duck_conn.close()

    # Run python engine
    rebuild_conn = init_corpus_db(db_path)
    read_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    read_conn.row_factory = sqlite3.Row
    try:
        n_py = build_features(
            trades_repo=CorpusTradesRepo(read_conn),
            resolutions_repo=MarketResolutionsRepo(rebuild_conn),
            examples_repo=TrainingExamplesRepo(rebuild_conn),
            markets_conn=rebuild_conn,
            now_ts=int(time.time()),
            rebuild=True,
            platform="polymarket",
        )
    finally:
        read_conn.close()
        rebuild_conn.close()

    assert n_duck == n_py, f"row count mismatch: duck={n_duck}, py={n_py}"

    py_conn = sqlite3.connect(db_path)
    py_conn.row_factory = sqlite3.Row
    try:
        py_sample = py_conn.execute(
            """
            SELECT tx_hash, asset_id, wallet_address,
                   prior_trades_count, win_rate, top_category, label_won
            FROM training_examples
            ORDER BY tx_hash
            LIMIT 1000
            """
        ).fetchall()
    finally:
        py_conn.close()

    # Column-by-column compare on the sample
    for d, p in zip(duck_sample, py_sample, strict=True):
        assert d["tx_hash"] == p["tx_hash"]
        assert d["prior_trades_count"] == p["prior_trades_count"]
        assert d["label_won"] == p["label_won"]
        if d["win_rate"] is None:
            assert p["win_rate"] is None
        else:
            assert math.isclose(d["win_rate"], p["win_rate"], rel_tol=1e-9)
        assert d["top_category"] == p["top_category"]
