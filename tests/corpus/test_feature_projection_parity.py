"""Engine-vs-engine parity test for project_row vs project_sql (#145).

Drives a Hypothesis-generated trade stream through both the Python engine
(via build_features) and the DuckDB engine (via build_features_duckdb),
then compares the resulting training_examples rows.

Initially FAILS — passes once Task 14 lands (when _final_join_to_v2 is
switched to project_sql).
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)

# --- Strategies ----------------------------------------------------------


@st.composite
def _trade_factory(
    draw: st.DrawFn,
    *,
    wallets: list[str],
    markets: list[str],
    base_ts: int,
    index: int,
) -> CorpusTrade:
    """Generate one CorpusTrade. ts is monotonic via the ``index`` arg."""
    wallet = draw(st.sampled_from(wallets))
    market = draw(st.sampled_from(markets))
    side = draw(st.sampled_from(("YES", "NO")))
    bs = draw(st.sampled_from(("BUY", "SELL")))
    price = draw(st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    size = draw(st.floats(min_value=100.0, max_value=500.0, allow_nan=False))
    notional = round(price * size, 4)
    return CorpusTrade(
        tx_hash=f"tx{index:05d}",
        asset_id=f"{market}-{side}",
        wallet_address=wallet,
        condition_id=market,
        outcome_side=side,
        bs=bs,
        price=round(price, 4),
        size=round(size, 2),
        notional_usd=notional,
        ts=base_ts + index * 60,
    )


@st.composite
def _trade_stream(draw: st.DrawFn, min_size: int = 20, max_size: int = 100) -> list[CorpusTrade]:
    """Generate a chronologically-ordered stream of trades."""
    n_wallets = draw(st.integers(2, 6))
    n_markets = draw(st.integers(2, 4))
    wallets = [f"0xw{i:02d}" for i in range(n_wallets)]
    markets = [f"0xm{i:02d}" for i in range(n_markets)]
    n = draw(st.integers(min_size, max_size))
    base_ts = 1_700_000_000
    return [
        draw(_trade_factory(wallets=wallets, markets=markets, base_ts=base_ts, index=i))
        for i in range(n)
    ]


# --- Harness ------------------------------------------------------------


def _materialize_stream_to_corpus(stream: list[CorpusTrade], db_path: Path) -> None:
    """Write ``stream`` into corpus_trades + corpus_markets + market_resolutions.

    Mirrors what ``pscanner corpus backfill`` produces. Resolves every market
    at the latest trade's ts so feature labels have non-null label_won.
    All markets are categorised as ``sports`` with a generous $100K total
    volume so they pass the corpus volume gate.
    """
    if not stream:
        return

    conn = init_corpus_db(db_path)
    try:
        markets = {t.condition_id for t in stream}
        last_ts = max(t.ts for t in stream)

        # Insert corpus_markets (one per unique condition_id).
        conn.executemany(
            """
            INSERT OR IGNORE INTO corpus_markets (
              platform, condition_id, event_slug, category, categories_json,
              closed_at, total_volume_usd, backfill_state, enumerated_at
            ) VALUES ('polymarket', ?, ?, 'sports', '["sports"]', ?, 100000.0, 'complete', ?)
            """,
            [(cid, cid, last_ts, last_ts) for cid in markets],
        )

        # Insert market_resolutions (YES wins for simplicity).
        conn.executemany(
            """
            INSERT OR IGNORE INTO market_resolutions (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES ('polymarket', ?, 0, 1, ?, 'test', ?)
            """,
            [(cid, last_ts, last_ts) for cid in markets],
        )

        conn.commit()

        # Insert corpus_trades via the repo (applies the $10 notional floor).
        trades_repo = CorpusTradesRepo(conn)
        trades_repo.insert_batch(stream)
        conn.commit()
    finally:
        conn.close()


def _run_python_engine(db_path: Path) -> None:
    """Run build_features (Python engine) against ``db_path``."""
    write_conn = init_corpus_db(db_path)
    read_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    read_conn.row_factory = sqlite3.Row
    try:
        build_features(
            trades_repo=CorpusTradesRepo(read_conn),
            resolutions_repo=MarketResolutionsRepo(write_conn),
            examples_repo=TrainingExamplesRepo(write_conn),
            markets_conn=write_conn,
            now_ts=int(time.time()),
            rebuild=True,
            platform="polymarket",
        )
    finally:
        read_conn.close()
        write_conn.close()


def _run_duckdb_engine(db_path: Path, scratch_dir: Path) -> None:
    """Run build_features_duckdb (DuckDB engine) against ``db_path``."""
    from pscanner.corpus._duckdb_engine import build_features_duckdb  # noqa: PLC0415

    build_features_duckdb(
        db_path=db_path,
        platform="polymarket",
        now_ts=int(time.time()),
        memory_limit="2GB",
        temp_dir=scratch_dir,
        threads=2,
    )


def _read_training_examples(db_path: Path) -> list[dict[str, object]]:
    """Return all training_examples rows in a stable canonical order."""
    conn = sqlite3.connect(db_path)
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


def _diff_row(py: dict[str, object], duck: dict[str, object]) -> list[str]:
    """Return per-column diffs between one Python row and one DuckDB row."""
    skip = {"id", "built_at"}
    diffs: list[str] = []
    all_cols = sorted(py.keys() | duck.keys())
    for col in all_cols:
        if col in skip:
            continue
        pv, dv = py.get(col), duck.get(col)
        if pv is None and dv is None:
            continue
        if isinstance(pv, float) or isinstance(dv, float):
            if pv is None or dv is None:
                diffs.append(f"  {col}: py={pv!r} duck={dv!r}")
                continue
            if not math.isclose(float(pv), float(dv), rel_tol=1e-9, abs_tol=1e-12):  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
                diffs.append(f"  {col}: py={pv!r} duck={dv!r}")
        elif pv != dv:
            diffs.append(f"  {col}: py={pv!r} duck={dv!r}")
    return diffs


# --- The property test --------------------------------------------------


@given(stream=_trade_stream(min_size=20, max_size=80))
@settings(
    max_examples=20,  # bump to 200+ once green on CI
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_python_duckdb_engines_produce_identical_feature_rows(
    stream: list[CorpusTrade], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """build-features in python engine and duckdb engine produce row-identical output."""
    tmp_path = tmp_path_factory.mktemp("parity")
    py_db = tmp_path / "py.sqlite3"
    duck_db = tmp_path / "duck.sqlite3"

    _materialize_stream_to_corpus(stream, py_db)
    _materialize_stream_to_corpus(stream, duck_db)

    _run_python_engine(py_db)
    _run_duckdb_engine(duck_db, scratch_dir=tmp_path / "scratch")

    py_rows = _read_training_examples(py_db)
    duck_rows = _read_training_examples(duck_db)

    assert len(py_rows) == len(duck_rows), (
        f"row count diverges: python={len(py_rows)} duckdb={len(duck_rows)}"
    )

    failures: list[str] = []
    for i, (p, d) in enumerate(zip(py_rows, duck_rows, strict=True)):
        key = (p.get("condition_id"), p.get("wallet_address"), p.get("tx_hash"))
        diffs = _diff_row(p, d)
        if diffs:
            failures.append(f"row {i} {key}:\n" + "\n".join(diffs))

    assert not failures, (
        f"{len(failures)} row(s) diverge between python and duckdb engines:\n"
        + "\n\n".join(failures[:5])  # show up to 5 to keep output readable
    )
