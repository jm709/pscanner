# Issue #116 — DuckDB Engine for `corpus build-features` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the row-by-row Python fold in `pscanner corpus build-features` with a pure-DuckDB SQL pipeline that produces a bit-equivalent `training_examples` table in 5-25 min instead of 6 hours.

**Architecture:** ATTACH `corpus.sqlite3` to DuckDB (READ_ONLY for the source side, R/W for the sink), materialize trades into a DuckDB TEMP TABLE once, then a CTE chain that UNIONs trades with synthetic resolution events to produce all 36 feature columns including the prior_wins/losses ledger that previously required a Python heap. Write rows into `training_examples_v2`, build suffixed indexes, then atomic swap inside a single `BEGIN IMMEDIATE` transaction that also clears the in-progress sentinel.

**Tech Stack:** Python 3.13, DuckDB ≥1.3.0 (prefer 1.4.4 LTS), SQLite via `sqlite3` stdlib, structlog, pytest.

**Scope:** PR-A only (ship DuckDB engine alongside Python engine, gated by `--engine` flag). PR-B (delete Python engine) is a follow-up after full-corpus parity passes on the production training box.

**Path A retrospective:** Path A landed via #114 but rebuild wall was 6 hours, not the predicted 1.5-2h. Path B's 12-72× projected speedup is therefore high-leverage, not marginal — the cadence-gating language in the original issue #116 is moot.

---

## File Map

**New files:**
- `src/pscanner/corpus/_duckdb_engine.py` — `build_features_duckdb()` entry point + SQL templates + heartbeat helper
- `src/pscanner/corpus/_build_features_sentinel.py` — sentinel set/check/clear helpers (shared by both engines)
- `tests/corpus/test_duckdb_engine.py` — synthetic-fixture parity tests
- `tests/corpus/test_build_features_sentinel.py` — sentinel behavior tests
- `scripts/parity_build_features.py` — operator-facing full-corpus parity runner

**Modified files:**
- `pyproject.toml` — add `duckdb>=1.3.0,<2.0`
- `src/pscanner/corpus/repos.py:237-268` — add `CorpusStateRepo.delete(key)`
- `src/pscanner/corpus/cli.py:115-124,496-518` — add CLI flags + dispatch
- `src/pscanner/corpus/examples.py:167-251` — wrap with sentinel; preserve Python-engine path
- `tests/corpus/test_repos.py` — `CorpusStateRepo.delete` test
- `CLAUDE.md` — document the new flags + parity-passed-at criterion

---

## Architecture Notes (read before starting)

**Why the heap dies.** The Python engine's per-wallet resolution heap (`features.py:apply_resolution_to_state`) increments `prior_resolved_buys/wins/losses/realized_pnl_usd` once per (BUY, resolved-market) pair, fired at the resolution's `resolved_at`. That's expressible as a UNION ALL of `corpus_trades` BUYs with synthetic "RESOLUTION" events ordered by `(event_ts, kind_priority, tx_hash, asset_id)` and aggregated with a windowed SUM strictly-preceding the current row. `kind_priority` puts RESOLUTIONs before BUYs at the same ts so a BUY at the same ts as its market's resolution sees the resolution in its `prior_*`. (This matches Python heap-drain semantics: `wallet_state(W, as_of_ts=T)` drains entries with `resolution_ts < T`.)

**Why pure SQL, not hybrid.** Grilling resolved that all 36 columns map to either raw fields, SQL window functions, or simple inline arithmetic in the final SELECT. Polars adds nothing once the heap is gone. Avoiding it removes the float-determinism question entirely.

**Why ATTACH + TEMP TABLE, not Parquet intermediate.** DuckDB's `sqlite_scanner` extension streams rows; pushdown handles the `WHERE platform = ?` filter at scan time. Materializing into a DuckDB columnar TEMP TABLE avoids the planner re-scanning SQLite for each CTE branch. Parquet intermediate is strictly worse — TEMP storage is already columnar+compressed.

**Tiebreak ordering matches Python.** Both engines order trades by `(ts, tx_hash, asset_id)`. The Python engine's `CorpusTradesRepo.iter_chronological` uses this order (see `repos.py` and `corpus/db.py:69-70`'s composite index). The DuckDB window `ORDER BY` must match exactly.

**Parity gate.** `assert_allclose(rtol=1e-9, atol=1e-12)` for REAL columns, exact `==` for INT/TEXT. In practice the float values should be bit-identical because both engines accumulate in the same row order; the tolerance is belt-and-suspenders.

**Sentinel must clear inside the swap transaction.** Otherwise a crash between swap-commit and sentinel-clear leaves the v2 table live with the sentinel still set, and a next-run `--force` would drop the live table.

---

## Task 1: Add DuckDB dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Look up latest stable DuckDB**

Run: `uv pip index versions duckdb 2>&1 | head -5`
Expected: latest 1.4.x release. Pin range `>=1.3.0,<2.0` per CLAUDE.md "pin exact versions" intent and the issue's regression note (1.2.0 has a 4× window-function regression).

- [ ] **Step 2: Add to pyproject.toml**

Open `pyproject.toml`, find the `dependencies = [` list, append:

```toml
    "duckdb>=1.3.0,<2.0",
```

Maintain alphabetical order with surrounding entries.

- [ ] **Step 3: Sync and verify import**

Run: `uv sync && uv run python -c "import duckdb; print(duckdb.__version__)"`
Expected: prints a version string ≥ 1.3.0.

- [ ] **Step 4: Quick verify**

Run: `uv run ruff check pyproject.toml`
Expected: no lint errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
build(deps): add duckdb for build-features path B engine (#116)
EOF
)"
```

---

## Task 2: Add `CorpusStateRepo.delete(key)` (TDD)

**Files:**
- Modify: `src/pscanner/corpus/repos.py:237-268`
- Test: `tests/corpus/test_repos.py`

- [ ] **Step 1: Write the failing test**

Find the existing `CorpusStateRepo` tests in `tests/corpus/test_repos.py` (search for `CorpusStateRepo` or `corpus_state`). Append:

```python
def test_corpus_state_delete_removes_key(tmp_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_db)
    repo.set("transient_flag", "1", updated_at=1_000_000)
    assert repo.get("transient_flag") == "1"

    repo.delete("transient_flag")

    assert repo.get("transient_flag") is None


def test_corpus_state_delete_unknown_key_is_noop(tmp_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_db)
    repo.delete("never_set")  # must not raise

    assert repo.get("never_set") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/corpus/test_repos.py -k "corpus_state_delete" -v`
Expected: FAIL with `AttributeError: 'CorpusStateRepo' object has no attribute 'delete'`.

- [ ] **Step 3: Implement**

Open `src/pscanner/corpus/repos.py`, find the `set` method at line 257-268, append after it (before the class closes at line 269):

```python
    def delete(self, key: str) -> None:
        """Remove ``key`` if present. No-op if absent."""
        self._conn.execute("DELETE FROM corpus_state WHERE key = ?", (key,))
        self._conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_repos.py -k "corpus_state_delete" -v`
Expected: 2 PASS.

- [ ] **Step 5: Run full repo verifier**

Run: `uv run ruff check src/pscanner/corpus/repos.py tests/corpus/test_repos.py && uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos.py
git commit -m "$(cat <<'EOF'
feat(corpus): add CorpusStateRepo.delete for sentinel cleanup (#116)
EOF
)"
```

---

## Task 3: Sentinel helpers (TDD)

**Files:**
- Create: `src/pscanner/corpus/_build_features_sentinel.py`
- Test: `tests/corpus/test_build_features_sentinel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/corpus/test_build_features_sentinel.py`:

```python
"""Tests for the build-features in-progress sentinel."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus._build_features_sentinel import (
    SENTINEL_KEY,
    SentinelAlreadySetError,
    check_and_set_sentinel,
    clear_sentinel,
)
from pscanner.corpus.repos import CorpusStateRepo


def test_check_and_set_writes_sentinel(tmp_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_db)
    check_and_set_sentinel(repo, now_ts=1_700_000_000, force=False)
    assert repo.get(SENTINEL_KEY) == "1700000000"


def test_check_and_set_refuses_when_present(tmp_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_db)
    repo.set(SENTINEL_KEY, "1699900000", updated_at=1_699_900_000)

    with pytest.raises(SentinelAlreadySetError) as exc:
        check_and_set_sentinel(repo, now_ts=1_700_000_000, force=False)

    assert "1699900000" in str(exc.value)


def test_check_and_set_force_overrides(tmp_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_db)
    repo.set(SENTINEL_KEY, "1699900000", updated_at=1_699_900_000)

    check_and_set_sentinel(repo, now_ts=1_700_000_000, force=True)

    assert repo.get(SENTINEL_KEY) == "1700000000"


def test_clear_sentinel_removes_key(tmp_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_db)
    repo.set(SENTINEL_KEY, "1700000000", updated_at=1_700_000_000)

    clear_sentinel(repo)

    assert repo.get(SENTINEL_KEY) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/corpus/test_build_features_sentinel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pscanner.corpus._build_features_sentinel'`.

- [ ] **Step 3: Implement the sentinel module**

Create `src/pscanner/corpus/_build_features_sentinel.py`:

```python
"""Crash-recovery sentinel for ``pscanner corpus build-features``.

Writes ``corpus_state['build_features_in_progress']`` at run start, clears
on success. A prior crashed run leaves the key set; the CLI refuses to
proceed without ``--force``.
"""

from __future__ import annotations

from typing import Final

from pscanner.corpus.repos import CorpusStateRepo

SENTINEL_KEY: Final[str] = "build_features_in_progress"


class SentinelAlreadySetError(RuntimeError):
    """Raised when the sentinel is set and ``--force`` was not supplied."""


def check_and_set_sentinel(
    repo: CorpusStateRepo, *, now_ts: int, force: bool
) -> None:
    """Write the sentinel; raise if already set unless ``force`` is True.

    The stored value is the run-start Unix timestamp as a string, so a
    ``--force`` recovery prints when the stuck run started.
    """
    existing = repo.get(SENTINEL_KEY)
    if existing is not None and not force:
        raise SentinelAlreadySetError(
            f"build_features in-progress sentinel is set "
            f"(started at ts={existing}). A prior run crashed mid-rebuild. "
            f"Inspect ./data/corpus.sqlite3 and re-run with --force to override."
        )
    repo.set(SENTINEL_KEY, str(now_ts), updated_at=now_ts)


def clear_sentinel(repo: CorpusStateRepo) -> None:
    """Remove the sentinel after a successful run."""
    repo.delete(SENTINEL_KEY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_build_features_sentinel.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck**

Run: `uv run ruff check src/pscanner/corpus/_build_features_sentinel.py tests/corpus/test_build_features_sentinel.py && uv run ty check`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/_build_features_sentinel.py tests/corpus/test_build_features_sentinel.py
git commit -m "$(cat <<'EOF'
feat(corpus): add build_features crash-recovery sentinel (#116)
EOF
)"
```

---

## Task 4: Synthetic-fixture builder for parity tests

**Files:**
- Create: `tests/corpus/_duckdb_fixture.py`

A reusable fixture builder used by Tasks 6-12. Lives in `tests/corpus/` so pytest's import machinery picks it up; underscore-prefixed name so it isn't auto-collected as a test module.

- [ ] **Step 1: Write the fixture builder**

Create `tests/corpus/_duckdb_fixture.py`:

```python
"""Synthetic-corpus fixture for build-features parity tests.

Produces a small SQLite file (~30 trades, 4 markets, 3 wallets) that
exercises the edge cases the DuckDB engine must handle bit-equivalent
to the Python engine:

- Multiple trades by same wallet at same ts (tiebreak on tx_hash, asset_id)
- A market resolution at the same ts as a buy on that market (RESOLUTION
  vs BUY tiebreak)
- Wallets with multi-category trades (top_category, category_diversity)
- A wallet with both wins and losses across markets
- BUYs on markets that never resolve (must not produce training_examples)
- SELLs interleaved with BUYs (must update recency but not BUY aggregates)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolutionsRepo,
)


def build_fixture_db(path: Path) -> None:
    """Build a deterministic corpus DB at ``path``. Overwrites if it exists."""
    if path.exists():
        path.unlink()
    conn = init_corpus_db(path)
    try:
        _insert_markets(conn)
        _insert_trades(conn)
        _insert_resolutions(conn)
        conn.commit()
    finally:
        conn.close()


def _insert_markets(conn: sqlite3.Connection) -> None:
    """Four markets: 3 resolved (sports/esports/politics), 1 unresolved."""
    rows = [
        ("polymarket", "MKT_A", "ev-a", "sports",   2_000_000, 5000.0, "complete", None, 0, 0, None, 1_700_000_000, None, None, None, None, None),
        ("polymarket", "MKT_B", "ev-b", "esports",  2_000_500, 7000.0, "complete", None, 0, 0, None, 1_700_000_000, None, None, None, None, None),
        ("polymarket", "MKT_C", "ev-c", "politics", 2_001_000, 3000.0, "complete", None, 0, 0, None, 1_700_000_000, None, None, None, None, None),
        ("polymarket", "MKT_D", "ev-d", "sports",   0,         1000.0, "complete", None, 0, 0, None, 1_700_000_000, None, None, None, None, None),
    ]
    conn.executemany(
        """
        INSERT INTO corpus_markets (
          platform, condition_id, event_slug, category, closed_at,
          total_volume_usd, backfill_state, last_offset_seen,
          trades_pulled_count, truncated_at_offset_cap, error_message,
          enumerated_at, backfill_started_at, backfill_completed_at,
          market_slug, onchain_trades_count, onchain_processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_trades(conn: sqlite3.Connection) -> None:
    """Hand-crafted trades exercising every parity edge case.

    Layout (ts in seconds, ordered chronologically):
      1_000_000 W1 BUY  MKT_A  YES @ 0.40  size=100 notional=40
      1_000_000 W1 BUY  MKT_B  YES @ 0.30  size=100 notional=30   (same ts; tiebreak by tx_hash)
      1_000_500 W2 BUY  MKT_A  NO  @ 0.55  size=50  notional=27.5
      1_001_000 W1 SELL MKT_A  YES @ 0.45  size=10  notional=4.5  (SELL bumps recency only)
      1_500_000 W3 BUY  MKT_C  YES @ 0.20  size=200 notional=40
      1_900_000 W1 BUY  MKT_C  NO  @ 0.85  size=50  notional=42.5
      2_000_000 (RESOLUTION MKT_A yes_won=1)                       (same ts as next BUY)
      2_000_000 W2 BUY  MKT_B  NO  @ 0.65  size=80  notional=52
      2_000_500 (RESOLUTION MKT_B yes_won=0)
      2_001_000 (RESOLUTION MKT_C yes_won=1)
      2_100_000 W1 BUY  MKT_D  YES @ 0.50  size=40  notional=20    (MKT_D never resolves)
    """
    trades = [
        CorpusTrade("tx_a", "asset_a_yes", "0xw1", "MKT_A", "YES", "BUY",  0.40, 100.0, 40.0,  1_000_000),
        CorpusTrade("tx_b", "asset_b_yes", "0xw1", "MKT_B", "YES", "BUY",  0.30, 100.0, 30.0,  1_000_000),
        CorpusTrade("tx_c", "asset_a_no",  "0xw2", "MKT_A", "NO",  "BUY",  0.55,  50.0, 27.5,  1_000_500),
        CorpusTrade("tx_d", "asset_a_yes", "0xw1", "MKT_A", "YES", "SELL", 0.45,  10.0,  4.5,  1_001_000),
        CorpusTrade("tx_e", "asset_c_yes", "0xw3", "MKT_C", "YES", "BUY",  0.20, 200.0, 40.0,  1_500_000),
        CorpusTrade("tx_f", "asset_c_no",  "0xw1", "MKT_C", "NO",  "BUY",  0.85,  50.0, 42.5,  1_900_000),
        CorpusTrade("tx_g", "asset_b_no",  "0xw2", "MKT_B", "NO",  "BUY",  0.65,  80.0, 52.0,  2_000_000),
        CorpusTrade("tx_h", "asset_d_yes", "0xw1", "MKT_D", "YES", "BUY",  0.50,  40.0, 20.0,  2_100_000),
    ]
    repo = CorpusTradesRepo(conn)
    repo.insert_batch(trades)


def _insert_resolutions(conn: sqlite3.Connection) -> None:
    """Three markets resolve; MKT_D does not."""
    repo = MarketResolutionsRepo(conn)
    repo.record(condition_id="MKT_A", winning_outcome_index=0, outcome_yes_won=1, resolved_at=2_000_000, source="test", now_ts=2_000_100)
    repo.record(condition_id="MKT_B", winning_outcome_index=1, outcome_yes_won=0, resolved_at=2_000_500, source="test", now_ts=2_000_600)
    repo.record(condition_id="MKT_C", winning_outcome_index=0, outcome_yes_won=1, resolved_at=2_001_000, source="test", now_ts=2_001_100)
```

> **Note on `MarketResolutionsRepo.record`:** Verify the actual method name by running `rg "class MarketResolutionsRepo" -A 30 src/pscanner/corpus/repos.py`. If the signature differs, adjust the calls above before moving on. Common alternatives: `insert`, `upsert`.

- [ ] **Step 2: Verify the fixture builds cleanly**

Create a one-shot smoke test at the bottom of `_duckdb_fixture.py`:

```python
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fixture.sqlite3"
        build_fixture_db(p)
        with sqlite3.connect(p) as c:
            n_trades = c.execute("SELECT COUNT(*) FROM corpus_trades").fetchone()[0]
            n_markets = c.execute("SELECT COUNT(*) FROM corpus_markets").fetchone()[0]
            n_res = c.execute("SELECT COUNT(*) FROM market_resolutions").fetchone()[0]
            print(f"trades={n_trades} markets={n_markets} resolutions={n_res}")
```

Run: `uv run python tests/corpus/_duckdb_fixture.py`
Expected: `trades=8 markets=4 resolutions=3`.

Remove the `if __name__ == "__main__"` block after verifying.

- [ ] **Step 3: Lint**

Run: `uv run ruff check tests/corpus/_duckdb_fixture.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/corpus/_duckdb_fixture.py
git commit -m "$(cat <<'EOF'
test(corpus): synthetic fixture for build-features parity (#116)
EOF
)"
```

---

## Task 5: Parity test harness (red)

**Files:**
- Create: `tests/corpus/test_duckdb_engine.py`

This is the failing test that Tasks 6-11 will drive to green. It runs the Python engine and the DuckDB engine on the same fixture and compares row-by-row.

- [ ] **Step 1: Write the parity test**

Create `tests/corpus/test_duckdb_engine.py`:

```python
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
    from pscanner.corpus._duckdb_engine import build_features_duckdb

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
    for col in py.keys():
        if col in skip:
            continue
        v_py, v_dd = py[col], dd[col]
        if v_py is None and v_dd is None:
            continue
        if isinstance(v_py, float) or isinstance(v_dd, float):
            if v_py is None or v_dd is None:
                diffs.append(f"{col}: py={v_py!r} dd={v_dd!r}")
                continue
            if not math.isclose(
                float(v_py), float(v_dd), rel_tol=_FLOAT_RTOL, abs_tol=_FLOAT_ATOL
            ):
                diffs.append(f"{col}: py={v_py!r} dd={v_dd!r}")
        else:
            if v_py != v_dd:
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pscanner.corpus._duckdb_engine'`.

- [ ] **Step 3: Commit (test red is the checkpoint)**

```bash
git add tests/corpus/test_duckdb_engine.py
git commit -m "$(cat <<'EOF'
test(corpus): failing parity harness for duckdb engine (#116)
EOF
)"
```

---

## Task 6: DuckDB engine skeleton — ATTACH + write empty `training_examples_v2`

**Files:**
- Create: `src/pscanner/corpus/_duckdb_engine.py`

Start with a no-op engine that just creates the v2 table empty + swaps it in. The parity test will fail on row count (0 vs 6), proving the swap mechanics work before we tackle SQL.

- [ ] **Step 1: Write the skeleton module**

Create `src/pscanner/corpus/_duckdb_engine.py`:

```python
"""DuckDB-based engine for ``pscanner corpus build-features``.

Pure SQL pipeline that produces ``training_examples`` rows bit-equivalent
(within ``rtol=1e-9``) to the Python ``StreamingHistoryProvider`` fold,
in 5-25 min vs 6h. See ``docs/superpowers/plans/2026-05-11-issue-116-duckdb-engine.md``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Final

import duckdb
import structlog

_log = structlog.get_logger(__name__)

_V2_TABLE: Final[str] = "training_examples_v2"
_V2_INDEX_PREFIX: Final[str] = "idx_te_v2_"


def build_features_duckdb(
    *,
    db_path: Path,
    platform: str,
    now_ts: int,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> int:
    """Rebuild ``training_examples`` for ``platform`` via DuckDB.

    Args:
        db_path: Path to ``corpus.sqlite3``.
        platform: Single platform to rebuild (``polymarket``/``manifold``/etc).
        now_ts: Value written to every row's ``built_at`` column.
        memory_limit: DuckDB ``memory_limit`` PRAGMA value (e.g. ``"6GB"``).
        temp_dir: Spill directory for partitions that don't fit in memory.
        threads: ``threads`` PRAGMA value.

    Returns:
        Number of rows in the new ``training_examples`` table.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    duck = duckdb.connect(":memory:")
    try:
        _configure_duckdb(duck, memory_limit=memory_limit, temp_dir=temp_dir, threads=threads)
        _attach_corpus(duck, db_path=db_path)
        _drop_stale_v2(duck)
        _materialize_trades(duck, platform=platform)
        _build_training_examples_v2(duck, platform=platform, now_ts=now_ts)
        n_rows = _count_v2(duck)
        _build_v2_indexes(duck)
        _atomic_swap(duck)
        _log.info(
            "corpus.build_features_duckdb_done",
            rows=n_rows,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        return n_rows
    finally:
        duck.close()


def _configure_duckdb(
    duck: duckdb.DuckDBPyConnection,
    *,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> None:
    duck.execute(f"SET memory_limit = '{memory_limit}'")
    duck.execute(f"SET temp_directory = '{temp_dir}'")
    duck.execute(f"SET threads = {threads}")
    duck.execute("INSTALL sqlite")
    duck.execute("LOAD sqlite")


def _attach_corpus(duck: duckdb.DuckDBPyConnection, *, db_path: Path) -> None:
    duck.execute(f"ATTACH '{db_path}' AS corpus (TYPE sqlite)")


def _drop_stale_v2(duck: duckdb.DuckDBPyConnection) -> None:
    """Clean up any v2 leftovers from a prior crashed run."""
    duck.execute(f"DROP TABLE IF EXISTS corpus.{_V2_TABLE}")
    # Drop suffixed indexes if present (best-effort; SQLite doesn't error
    # on unknown index drops with IF EXISTS).
    for col in ("condition", "wallet", "label"):
        duck.execute(f"DROP INDEX IF EXISTS corpus.{_V2_INDEX_PREFIX}{col}")


def _materialize_trades(duck: duckdb.DuckDBPyConnection, *, platform: str) -> None:
    """Pull corpus_trades + corpus_markets + market_resolutions into DuckDB TEMP."""
    duck.execute(
        f"""
        CREATE TEMP TABLE trades AS
        SELECT
            t.tx_hash, t.asset_id, t.wallet_address, t.condition_id,
            t.outcome_side, t.bs, t.price, t.size, t.notional_usd, t.ts,
            m.category, m.closed_at, m.enumerated_at
        FROM corpus.corpus_trades t
        JOIN corpus.corpus_markets m
          ON m.platform = t.platform AND m.condition_id = t.condition_id
        WHERE t.platform = '{platform}' AND m.platform = '{platform}'
        """
    )
    duck.execute(
        f"""
        CREATE TEMP TABLE resolutions AS
        SELECT condition_id, resolved_at, outcome_yes_won
        FROM corpus.market_resolutions
        WHERE platform = '{platform}'
        """
    )


def _build_training_examples_v2(
    duck: duckdb.DuckDBPyConnection, *, platform: str, now_ts: int
) -> None:
    """Stub: create v2 with the same schema as v1 but no rows.

    Tasks 7-11 will replace this with the real CTE chain.
    """
    duck.execute(
        f"""
        CREATE TABLE corpus.{_V2_TABLE} (
            id INTEGER PRIMARY KEY,
            platform TEXT NOT NULL,
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
            UNIQUE (platform, tx_hash, asset_id, wallet_address)
        )
        """
    )


def _count_v2(duck: duckdb.DuckDBPyConnection) -> int:
    row = duck.execute(f"SELECT COUNT(*) FROM corpus.{_V2_TABLE}").fetchone()
    return int(row[0]) if row else 0


def _build_v2_indexes(duck: duckdb.DuckDBPyConnection) -> None:
    duck.execute(
        f"CREATE INDEX {_V2_INDEX_PREFIX}condition ON corpus.{_V2_TABLE}(condition_id)"
    )
    duck.execute(
        f"CREATE INDEX {_V2_INDEX_PREFIX}wallet ON corpus.{_V2_TABLE}(wallet_address)"
    )
    duck.execute(
        f"CREATE INDEX {_V2_INDEX_PREFIX}label ON corpus.{_V2_TABLE}(label_won)"
    )


def _atomic_swap(duck: duckdb.DuckDBPyConnection) -> None:
    """Detach DuckDB and run the swap via raw sqlite3 in one transaction.

    DuckDB's SQLite extension doesn't expose ``BEGIN IMMEDIATE``, so the
    swap runs through stdlib sqlite3. The DuckDB connection is detached
    first to release any locks on the source file.
    """
    # Capture the corpus DB path from DuckDB's catalog before detaching.
    row = duck.execute(
        "SELECT path FROM duckdb_databases WHERE database_name = 'corpus'"
    ).fetchone()
    if row is None:
        raise RuntimeError("corpus database not attached")
    corpus_path = str(row[0])
    duck.execute("DETACH corpus")

    swap_conn = sqlite3.connect(corpus_path, isolation_level=None)
    try:
        swap_conn.execute("BEGIN IMMEDIATE")
        swap_conn.execute("ALTER TABLE training_examples RENAME TO training_examples_old")
        swap_conn.execute(f"ALTER TABLE {_V2_TABLE} RENAME TO training_examples")
        swap_conn.execute(
            f"ALTER INDEX {_V2_INDEX_PREFIX}condition RENAME TO idx_training_examples_condition"
        )
        swap_conn.execute(
            f"ALTER INDEX {_V2_INDEX_PREFIX}wallet RENAME TO idx_training_examples_wallet"
        )
        swap_conn.execute(
            f"ALTER INDEX {_V2_INDEX_PREFIX}label RENAME TO idx_training_examples_label"
        )
        swap_conn.execute("DROP TABLE training_examples_old")
        swap_conn.execute(
            "DELETE FROM corpus_state WHERE key = 'build_features_in_progress'"
        )
        swap_conn.execute("COMMIT")
    except Exception:
        swap_conn.execute("ROLLBACK")
        raise
    finally:
        swap_conn.close()
```

- [ ] **Step 2: Run the parity test (expect row-count failure)**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v`
Expected: FAIL with `row count differs: python=6 duckdb=0` (or similar non-zero python count, 0 duckdb count). The skeleton must NOT crash — if it does, fix before moving on.

> **If `_atomic_swap` errors on "duckdb_databases":** older DuckDB versions name this `duckdb_settings` or expose only `pragma_database_list()`. Adapt to whichever the installed version provides.

> **If `ALTER INDEX ... RENAME TO` errors:** SQLite ≥3.25 supports it but some bundled SQLite versions lag. Check the installed version: `uv run python -c "import sqlite3; print(sqlite3.sqlite_version)"`. If <3.25 the plan needs a fallback (drop indexes, swap tables, rebuild indexes with final names). 3.50.4 is expected per the grilling.

- [ ] **Step 3: Lint + typecheck**

Run: `uv run ruff check src/pscanner/corpus/_duckdb_engine.py && uv run ty check`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): duckdb engine skeleton with atomic swap (#116)
EOF
)"
```

---

## Task 7: SQL CTE — raw + per-wallet running aggregates

This task fills in `_build_training_examples_v2` with the heap-replacement UNION and the per-wallet/per-market running aggregates. The parity test will go from "row count 0 vs 6" to "row count matches; values differ on some columns" — that's expected progress.

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`

- [ ] **Step 1: Replace `_build_training_examples_v2` with the real CTE**

Open `src/pscanner/corpus/_duckdb_engine.py`. Replace the entire `_build_training_examples_v2` function (the stub from Task 6) with:

```python
def _build_training_examples_v2(
    duck: duckdb.DuckDBPyConnection, *, platform: str, now_ts: int
) -> None:
    """Build training_examples_v2 via a single CTE chain.

    Stages:
      1. ``events`` — UNION of BUYs + synthetic RESOLUTION events. RESOLUTIONs
         carry the original BUY's notional/size/side so realized_pnl can sum
         in the same window as the wins/losses counters. ``kind_priority`` is
         0 for RESOLUTION and 1 for BUY so same-ts ties put resolutions first
         (matching Python heap-drain semantics: ``wallet_state(W, as_of_ts=T)``
         drains heap entries with ``resolution_ts < T`` BEFORE folding in
         the current BUY).
      2. ``wallet_acc`` — windowed running aggregates per wallet over events.
      3. ``market_acc`` — windowed running aggregates per market over BUYs+SELLs.
      4. ``cat_acc`` — running argmax + cardinality of (wallet × category).
      5. Final SELECT — joins everything, computes interaction features,
         filters to BUYs whose market has resolved (i.e. has a row in
         ``resolutions``).
    """
    duck.execute(
        f"""
        CREATE TABLE corpus.{_V2_TABLE} AS
        WITH
        -- Per-wallet first-seen ts (any kind of trade). Drives wallet_age_days.
        wallet_first_seen AS (
            SELECT wallet_address, MIN(ts) AS first_seen_ts
            FROM trades
            GROUP BY wallet_address
        ),
        -- Per-market first-seen ts (any trade). Drives market_age_seconds.
        market_first_seen AS (
            SELECT condition_id, MIN(ts) AS market_age_start_ts
            FROM trades
            GROUP BY condition_id
        ),
        -- BUY events. ``payout_pnl`` is 0 for BUYs (PnL increments only on RESOLUTION).
        buy_events AS (
            SELECT
                t.wallet_address, t.condition_id, t.ts AS event_ts,
                t.tx_hash, t.asset_id, t.bs, t.outcome_side,
                t.price, t.size, t.notional_usd, t.category,
                t.closed_at, t.enumerated_at,
                CAST(1 AS INTEGER) AS kind_priority,  -- buys come after resolutions at same ts
                CAST(0 AS INTEGER) AS is_resolution,
                CAST(NULL AS INTEGER) AS res_won_for_this_buy,
                CAST(0.0 AS DOUBLE) AS payout_pnl_increment,
                CAST(1 AS INTEGER) AS is_buy,
                CAST(CASE WHEN t.bs = 'BUY' THEN 1 ELSE 0 END AS INTEGER) AS is_buy_only,
                CAST(CASE WHEN t.bs = 'BUY' THEN t.price ELSE NULL END AS DOUBLE) AS buy_price,
                CAST(CASE WHEN t.bs = 'BUY' THEN t.notional_usd ELSE NULL END AS DOUBLE) AS buy_notional
            FROM trades t
        ),
        -- Synthetic RESOLUTION events: one per (BUY, market_resolution) pair.
        -- The wallet bought into market M; when M resolves, the wallet's
        -- prior_resolved_buys / prior_wins / prior_losses / realized_pnl bump.
        -- Matches ``apply_resolution_to_state(state, won, notional_usd, payout_usd)``
        -- in features.py:219-238.
        resolution_events AS (
            SELECT
                t.wallet_address, t.condition_id, r.resolved_at AS event_ts,
                t.tx_hash, t.asset_id,
                CAST(NULL AS VARCHAR) AS bs,
                CAST(NULL AS VARCHAR) AS outcome_side,
                CAST(NULL AS DOUBLE) AS price,
                CAST(NULL AS DOUBLE) AS size,
                CAST(NULL AS DOUBLE) AS notional_usd,
                CAST(NULL AS VARCHAR) AS category,
                CAST(NULL AS INTEGER) AS closed_at,
                CAST(NULL AS INTEGER) AS enumerated_at,
                CAST(0 AS INTEGER) AS kind_priority,
                CAST(1 AS INTEGER) AS is_resolution,
                CAST(
                    CASE
                        WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES')
                          OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO')
                        THEN 1 ELSE 0
                    END AS INTEGER
                ) AS res_won_for_this_buy,
                CAST(
                    CASE
                        WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES')
                          OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO')
                        THEN t.size - t.notional_usd  -- payout - notional
                        ELSE -t.notional_usd
                    END AS DOUBLE
                ) AS payout_pnl_increment,
                CAST(0 AS INTEGER) AS is_buy,
                CAST(0 AS INTEGER) AS is_buy_only,
                CAST(NULL AS DOUBLE) AS buy_price,
                CAST(NULL AS DOUBLE) AS buy_notional
            FROM trades t
            JOIN resolutions r USING (condition_id)
            WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at
        ),
        events AS (
            SELECT * FROM buy_events
            UNION ALL
            SELECT * FROM resolution_events
        ),
        -- Per-wallet running aggregates, strict-< (exclude current row).
        -- All windows share the same ORDER BY so DuckDB reuses one sort.
        wallet_acc AS (
            SELECT
                e.*,
                -- prior_trades_count = count of all PRIOR events (BUY+SELL) for this wallet,
                -- excluding RESOLUTION events (those don't bump trade count in Python engine).
                COALESCE(SUM(is_buy) OVER w_strict, 0) AS prior_trades_count_w,
                -- prior_buys_count = count of prior BUYs only (excludes SELLs).
                COALESCE(SUM(is_buy_only) OVER w_strict, 0) AS prior_buys_count_w,
                COALESCE(SUM(is_resolution) OVER w_strict, 0) AS prior_resolved_buys_w,
                COALESCE(SUM(res_won_for_this_buy) OVER w_strict, 0) AS prior_wins_w,
                COALESCE(
                    SUM(is_resolution) OVER w_strict
                    - SUM(res_won_for_this_buy) OVER w_strict, 0
                ) AS prior_losses_w,
                COALESCE(SUM(payout_pnl_increment) OVER w_strict, 0.0) AS prior_realized_pnl_w,
                SUM(buy_price) OVER w_strict AS cumulative_buy_price_sum_w,
                SUM(buy_notional) OVER w_strict AS bet_size_sum_w,
                SUM(is_buy_only) OVER w_strict AS bet_size_count_w,
                -- last_trade_ts = MAX ts of prior BUY+SELL events for this wallet
                MAX(CASE WHEN is_buy = 1 THEN event_ts END) OVER w_strict AS last_trade_ts_w,
                -- recent_30d trades: BUYs+SELLs in [event_ts - 30d, event_ts), strict-<
                COUNT(*) FILTER (WHERE is_buy = 1) OVER w_range_30d AS prior_trades_30d_w
            FROM events e
            WINDOW
                w_strict AS (
                    PARTITION BY wallet_address
                    ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                w_range_30d AS (
                    PARTITION BY wallet_address
                    ORDER BY event_ts
                    RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW
                    EXCLUDE CURRENT ROW
                )
        )
        -- Filter to the BUY rows: those are the ones that become training_examples.
        -- Final SELECT shape will be expanded in later tasks; for now, emit
        -- a row per BUY whose market resolved.
        SELECT
            ROW_NUMBER() OVER () AS id,
            '{platform}' AS platform,
            wa.tx_hash, wa.asset_id, wa.wallet_address, wa.condition_id,
            wa.event_ts AS trade_ts,
            {now_ts} AS built_at,
            wa.prior_trades_count_w AS prior_trades_count,
            wa.prior_buys_count_w AS prior_buys_count,
            wa.prior_resolved_buys_w AS prior_resolved_buys,
            wa.prior_wins_w AS prior_wins,
            wa.prior_losses_w AS prior_losses,
            CASE WHEN wa.prior_resolved_buys_w > 0
                 THEN CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w
                 ELSE NULL END AS win_rate,
            CASE WHEN COALESCE(wa.bet_size_count_w, 0) > 0
                 THEN wa.cumulative_buy_price_sum_w
                      / (SELECT MAX(1)) -- placeholder; recomputed below
                 ELSE NULL END AS avg_implied_prob_paid,
            CAST(NULL AS DOUBLE) AS realized_edge_pp,
            wa.prior_realized_pnl_w AS prior_realized_pnl_usd,
            CASE WHEN COALESCE(wa.bet_size_count_w, 0) > 0
                 THEN wa.bet_size_sum_w / wa.bet_size_count_w
                 ELSE NULL END AS avg_bet_size_usd,
            CAST(NULL AS DOUBLE) AS median_bet_size_usd,
            GREATEST(0.0, (wa.event_ts - wfs.first_seen_ts) / 86400.0) AS wallet_age_days,
            CASE WHEN wa.last_trade_ts_w IS NOT NULL
                 THEN wa.event_ts - wa.last_trade_ts_w
                 ELSE NULL END AS seconds_since_last_trade,
            COALESCE(wa.prior_trades_30d_w, 0) AS prior_trades_30d,
            CAST(NULL AS VARCHAR) AS top_category,
            CAST(0 AS INTEGER) AS category_diversity,
            wa.notional_usd AS bet_size_usd,
            CAST(NULL AS DOUBLE) AS bet_size_rel_to_avg,
            CAST(0.0 AS DOUBLE) AS edge_confidence_weighted,
            CAST(0.0 AS DOUBLE) AS win_rate_confidence_weighted,
            CAST(0 AS INTEGER) AS is_high_quality_wallet,
            CAST(1.0 AS DOUBLE) AS bet_size_relative_to_history,
            wa.outcome_side AS side,
            wa.price AS implied_prob_at_buy,
            wa.category AS market_category,
            CAST(0.0 AS DOUBLE) AS market_volume_so_far_usd,
            CAST(0 AS INTEGER) AS market_unique_traders_so_far,
            (wa.event_ts - mfs.market_age_start_ts) AS market_age_seconds,
            (wa.closed_at - wa.event_ts) AS time_to_resolution_seconds,
            CAST(NULL AS DOUBLE) AS last_trade_price,
            CAST(NULL AS DOUBLE) AS price_volatility_recent,
            CASE
                WHEN (r.outcome_yes_won = 1 AND wa.outcome_side = 'YES')
                  OR (r.outcome_yes_won = 0 AND wa.outcome_side = 'NO')
                THEN 1 ELSE 0
            END AS label_won
        FROM wallet_acc wa
        JOIN wallet_first_seen wfs USING (wallet_address)
        JOIN market_first_seen mfs USING (condition_id)
        JOIN resolutions r USING (condition_id)
        WHERE wa.is_buy_only = 1
        """
    )
```

> **Known shortcut in this task:** `avg_implied_prob_paid` is written via a placeholder formula that won't match Python. Task 8 fixes it. The parity test column-diff will show this as a known failure on `avg_implied_prob_paid`, `realized_edge_pp`, `top_category`, `category_diversity`, market columns, and interaction features — those are Tasks 8-11.

- [ ] **Step 2: Run parity test, expect partial progress**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v 2>&1 | head -60`
Expected: FAIL with column-level diffs. Verify these columns now MATCH:
- `prior_trades_count`, `prior_buys_count`, `prior_resolved_buys`, `prior_wins`, `prior_losses`
- `win_rate`, `prior_realized_pnl_usd`, `avg_bet_size_usd`
- `wallet_age_days`, `seconds_since_last_trade`, `prior_trades_30d`
- `bet_size_usd`, `side`, `implied_prob_at_buy`, `market_category`
- `market_age_seconds`, `time_to_resolution_seconds`, `label_won`

If any of those still mismatch, that's a bug in the CTE — debug before moving on. Common culprits:
- `kind_priority` reversed (RESOLUTIONs must come BEFORE same-ts BUYs at the wallet level)
- `ts <= resolved_at` vs `ts < resolved_at` — Python heap fires drain on `resolution_ts < T`, so a BUY at T's resolution should see the resolution in its `prior_*`. The `ts <= r.resolved_at` filter ensures the resolution-event exists; the `kind_priority=0` ordering ensures the window sees it before the BUY at the same ts.

- [ ] **Step 3: Commit progress**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): duckdb engine wallet+heap aggregates via union (#116)
EOF
)"
```

---

## Task 8: SQL CTE — `avg_implied_prob_paid` + `realized_edge_pp`

The placeholder in Task 7 won't match Python. This task fixes it.

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`

- [ ] **Step 1: Fix the formulas**

In `_build_training_examples_v2`, find the `avg_implied_prob_paid` line and replace the entire `avg_implied_prob_paid` and `realized_edge_pp` projections with:

```python
            CASE WHEN COALESCE(wa.bet_size_count_w, 0) > 0
                 THEN wa.cumulative_buy_price_sum_w / wa.bet_size_count_w
                 ELSE NULL END AS avg_implied_prob_paid,
            CASE
                WHEN wa.prior_resolved_buys_w > 0
                 AND COALESCE(wa.bet_size_count_w, 0) > 0
                THEN (CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w)
                     - (wa.cumulative_buy_price_sum_w / wa.bet_size_count_w)
                ELSE NULL
            END AS realized_edge_pp,
```

The Python formula (`features.py:343-351`):
```python
win_rate = wallet.prior_wins / wallet.prior_resolved_buys if wallet.prior_resolved_buys > 0 else None
avg_prob = wallet.cumulative_buy_price_sum / wallet.cumulative_buy_count if wallet.cumulative_buy_count > 0 else None
edge = win_rate - avg_prob if win_rate is not None and avg_prob is not None else None
```

`cumulative_buy_count` in Python is `bet_size_count_w` here (both are running count of BUYs).

- [ ] **Step 2: Run parity test**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v 2>&1 | head -60`
Expected: `avg_implied_prob_paid` and `realized_edge_pp` now match. Remaining diffs: `top_category`, `category_diversity`, market columns, interaction features.

- [ ] **Step 3: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): duckdb engine avg-prob + realized-edge columns (#116)
EOF
)"
```

---

## Task 9: SQL CTE — category accumulator (`top_category`, `category_diversity`)

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`

- [ ] **Step 1: Add the category CTE**

In `_build_training_examples_v2`, BEFORE the final SELECT (so right after the `wallet_acc AS (...)` block but inside the same `WITH`), add:

```python
        ,
        -- Per-(wallet, category) running BUY count, strict-<.
        wallet_cat_running AS (
            SELECT
                e.wallet_address, e.event_ts, e.kind_priority,
                e.tx_hash, e.asset_id, e.category,
                COALESCE(SUM(e.is_buy_only) OVER (
                    PARTITION BY e.wallet_address, e.category
                    ORDER BY e.event_ts, e.kind_priority, e.tx_hash, e.asset_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS cat_count_prior
            FROM events e
            WHERE e.is_buy = 1  -- categories only relevant on BUY/SELL rows
        ),
        -- Per (wallet, event row), pivot the wallet's category counter map to
        -- (top_category, category_diversity).
        wallet_cat_summary AS (
            SELECT
                wallet_address, event_ts, kind_priority, tx_hash, asset_id,
                -- top_category := argmax over category by cat_count_prior, ties
                -- broken by category text ascending (Python's max(dict.items(),
                -- key=...) returns first-inserted on ties; we approximate via
                -- alphabetical since insertion order isn't stable across engines).
                ARG_MAX(category, cat_count_prior * 1000000 - LENGTH(category))
                    FILTER (WHERE cat_count_prior > 0) AS top_category,
                COUNT(DISTINCT CASE WHEN cat_count_prior > 0 THEN category END)
                    AS category_diversity
            FROM wallet_cat_running
            GROUP BY wallet_address, event_ts, kind_priority, tx_hash, asset_id
        )
```

> **Tie-break warning on `top_category`:** the Python engine returns `max(dict.items(), key=lambda kv: kv[1])[0]`. On a tie, Python returns the first-iterated key — which in Python 3.7+ dicts is the insertion order. Insertion order = first-seen-category order per wallet. The DuckDB version above uses `ARG_MAX(category, count * 1e6 - len(category))` as an alphabetical-fallback tiebreak that's deterministic but won't always match Python.
>
> **If parity fails on `top_category` due to ties:** the fix is either (a) match insertion order by joining wallet_cat_running back to its first-occurrence ts for that category, or (b) document the tie-break as an accepted divergence in the parity test. The synthetic fixture is small enough that ties may not happen — verify before doing extra work.

In the final SELECT, replace `CAST(NULL AS VARCHAR) AS top_category` and `CAST(0 AS INTEGER) AS category_diversity` with:

```python
            wcs.top_category,
            wcs.category_diversity,
```

And add to the FROM-JOIN list:

```python
        LEFT JOIN wallet_cat_summary wcs USING (wallet_address, event_ts, kind_priority, tx_hash, asset_id)
```

- [ ] **Step 2: Run parity test**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v 2>&1 | head -60`
Expected: `category_diversity` matches; `top_category` either matches or shows the documented tie-break divergence.

If a real divergence remains, debug per the tie-break warning above.

- [ ] **Step 3: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): duckdb engine top_category + diversity (#116)
EOF
)"
```

---

## Task 10: SQL CTE — market-side aggregates

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`

The market-side columns:
- `market_volume_so_far_usd` = SUM(notional) prior to this trade, partitioned by market
- `market_unique_traders_so_far` = count of distinct wallets that traded the market before this trade
- `last_trade_price` = price of the most recent prior trade on this market
- `price_volatility_recent` = stddev_pop of the last 20 prior trade prices (NULL if <2)

- [ ] **Step 1: Add the market CTE**

In `_build_training_examples_v2`, append to the `WITH` chain (after `wallet_cat_summary`):

```python
        ,
        -- Per-market running aggregates over events that are BUYs or SELLs
        -- (RESOLUTION events don't update market state in the Python engine).
        market_acc AS (
            SELECT
                e.condition_id, e.event_ts, e.kind_priority,
                e.tx_hash, e.asset_id,
                COALESCE(SUM(e.notional_usd) OVER w_strict, 0.0) AS market_volume_so_far_w,
                -- Last trade price prior to this row: use ARRAY_AGG with strict-<
                -- and take the last element.
                LAST_VALUE(e.price IGNORE NULLS) OVER w_strict AS last_trade_price_w,
                -- 20-row volatility window (matches Python's tuple[-20:]).
                STDDEV_POP(e.price) OVER w_strict_20 AS price_volatility_w,
                COUNT(e.price) OVER w_strict_20 AS price_count_20
            FROM events e
            WHERE e.is_buy = 1  -- includes BUYs and SELLs
            WINDOW
                w_strict AS (
                    PARTITION BY condition_id
                    ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                w_strict_20 AS (
                    PARTITION BY condition_id
                    ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                )
        ),
        -- Unique traders so far in the market. Use ROW_NUMBER over
        -- (market, wallet) and count rows where it's the wallet's first
        -- trade in this market.
        market_first_trade AS (
            SELECT
                e.condition_id, e.wallet_address, e.event_ts,
                e.kind_priority, e.tx_hash, e.asset_id,
                CAST(
                    ROW_NUMBER() OVER (
                        PARTITION BY condition_id, wallet_address
                        ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ) = 1
                    AS INTEGER
                ) AS is_first_trade_in_market
            FROM events e
            WHERE e.is_buy = 1
        ),
        market_unique_acc AS (
            SELECT
                m.condition_id, m.event_ts, m.kind_priority,
                m.tx_hash, m.asset_id,
                COALESCE(SUM(m.is_first_trade_in_market) OVER (
                    PARTITION BY m.condition_id
                    ORDER BY m.event_ts, m.kind_priority, m.tx_hash, m.asset_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS market_unique_traders_so_far_w
            FROM market_first_trade m
        )
```

In the final SELECT, replace the market-column placeholders:

```python
            ma.market_volume_so_far_w AS market_volume_so_far_usd,
            mua.market_unique_traders_so_far_w AS market_unique_traders_so_far,
            ma.last_trade_price_w AS last_trade_price,
            CASE WHEN ma.price_count_20 >= 2 THEN ma.price_volatility_w ELSE NULL END
                AS price_volatility_recent,
```

And add to the JOIN chain:

```python
        LEFT JOIN market_acc ma USING (condition_id, event_ts, kind_priority, tx_hash, asset_id)
        LEFT JOIN market_unique_acc mua USING (condition_id, event_ts, kind_priority, tx_hash, asset_id)
```

- [ ] **Step 2: Run parity test**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v 2>&1 | head -60`
Expected: market columns match. Remaining diffs: interaction features only.

- [ ] **Step 3: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): duckdb engine market-side running aggregates (#116)
EOF
)"
```

---

## Task 11: Interaction features in final SELECT (green)

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`

The four interaction features from `features.py:356-367`:
- `bet_size_rel_to_avg = notional / avg_bet if avg_bet > 0 else None`
- `edge_confidence_weighted = (edge * min(1, prior_resolved_buys/20)) if edge is not None else 0.0`
- `win_rate_confidence_weighted = ((win_rate - 0.5) * min(1, prior_resolved_buys/20)) if win_rate is not None else 0.0`
- `is_high_quality_wallet = int(prior_resolved_buys >= 20 AND win_rate is not None AND win_rate > 0.55)`
- `bet_size_relative_to_history = 1.0` (always — median_bet is always None in v1)

- [ ] **Step 1: Replace the interaction placeholders**

In the final SELECT, replace these projections:

```python
            CAST(NULL AS DOUBLE) AS bet_size_rel_to_avg,
            CAST(0.0 AS DOUBLE) AS edge_confidence_weighted,
            CAST(0.0 AS DOUBLE) AS win_rate_confidence_weighted,
            CAST(0 AS INTEGER) AS is_high_quality_wallet,
            CAST(1.0 AS DOUBLE) AS bet_size_relative_to_history,
```

with:

```python
            CASE
                WHEN COALESCE(wa.bet_size_count_w, 0) > 0
                 AND wa.bet_size_sum_w > 0
                THEN wa.notional_usd / (wa.bet_size_sum_w / wa.bet_size_count_w)
                ELSE NULL
            END AS bet_size_rel_to_avg,
            CASE
                WHEN wa.prior_resolved_buys_w > 0
                 AND COALESCE(wa.bet_size_count_w, 0) > 0
                THEN (
                    (CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w
                     - wa.cumulative_buy_price_sum_w / wa.bet_size_count_w)
                    * LEAST(1.0, CAST(wa.prior_resolved_buys_w AS DOUBLE) / 20.0)
                )
                ELSE 0.0
            END AS edge_confidence_weighted,
            CASE
                WHEN wa.prior_resolved_buys_w > 0
                THEN (
                    (CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w - 0.5)
                    * LEAST(1.0, CAST(wa.prior_resolved_buys_w AS DOUBLE) / 20.0)
                )
                ELSE 0.0
            END AS win_rate_confidence_weighted,
            CAST(
                CASE
                    WHEN wa.prior_resolved_buys_w >= 20
                     AND (CAST(wa.prior_wins_w AS DOUBLE) / NULLIF(wa.prior_resolved_buys_w, 0)) > 0.55
                    THEN 1 ELSE 0
                END AS INTEGER
            ) AS is_high_quality_wallet,
            CAST(1.0 AS DOUBLE) AS bet_size_relative_to_history,
```

- [ ] **Step 2: Run the parity test — expect GREEN**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v`
Expected: PASS.

If failures remain, the diff output tells you which column. Common causes:
- `is_high_quality_wallet` returning 1 in DuckDB but 0 in Python: check the `> 0.55` strict inequality (Python uses strict `>`).
- Float diffs > `rtol=1e-9`: indicates a real reordering of additions. The window ORDER BY must exactly match Python's iter_chronological order.

- [ ] **Step 3: Lint + typecheck + full test run**

Run: `uv run ruff check src/pscanner/corpus/_duckdb_engine.py && uv run ty check && uv run pytest tests/corpus/ -q`
Expected: clean, all corpus tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): duckdb engine interaction features (parity green) (#116)
EOF
)"
```

---

## Task 12: Heartbeat thread

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`
- Test: `tests/corpus/test_duckdb_engine.py`

The pre-pass + INSERT takes 5-25 min on production scale. Add an observability heartbeat that polls the in-flight row count every 30s.

- [ ] **Step 1: Write the failing test**

Append to `tests/corpus/test_duckdb_engine.py`:

```python
def test_heartbeat_emits_during_long_operation() -> None:
    """Heartbeat thread fires at least once and stops cleanly on signal."""
    import threading
    from structlog.testing import capture_logs

    from pscanner.corpus._duckdb_engine import _heartbeat_loop

    stop = threading.Event()
    # Use a tiny interval and a poll fn that doesn't touch the DB.
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
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py::test_heartbeat_emits_during_long_operation -v`
Expected: FAIL with `ImportError: cannot import name '_heartbeat_loop'`.

- [ ] **Step 3: Implement**

In `src/pscanner/corpus/_duckdb_engine.py`, add the heartbeat helper near the top (after imports, before `build_features_duckdb`):

```python
import threading
from collections.abc import Callable


def _heartbeat_loop(
    *,
    stop: threading.Event,
    poll_fn: Callable[[], int],
    interval_seconds: float,
    stage: str,
) -> None:
    """Emit a heartbeat every ``interval_seconds`` until ``stop`` is set.

    ``poll_fn`` returns a snapshot of "how many rows landed so far" (or
    any integer progress signal). Errors in ``poll_fn`` are caught so a
    transient hiccup doesn't kill observability.
    """
    started = time.monotonic()
    while not stop.wait(interval_seconds):
        try:
            n = poll_fn()
        except Exception as exc:  # noqa: BLE001 — observability must not propagate
            _log.warning("corpus.build_features.heartbeat_poll_failed", error=str(exc))
            continue
        _log.info(
            "corpus.build_features.heartbeat",
            stage=stage,
            elapsed_seconds=round(time.monotonic() - started, 1),
            rows=n,
        )
```

Then wire it into `build_features_duckdb` — wrap the `_build_training_examples_v2` call in a heartbeat:

```python
        _materialize_trades(duck, platform=platform)

        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "stop": stop,
                "poll_fn": lambda: _count_v2_safe(duck),
                "interval_seconds": 30.0,
                "stage": "duckdb_build_v2",
            },
            daemon=True,
            name="build_features_heartbeat",
        )
        heartbeat.start()
        try:
            _build_training_examples_v2(duck, platform=platform, now_ts=now_ts)
        finally:
            stop.set()
            heartbeat.join(timeout=5.0)

        n_rows = _count_v2(duck)
```

Add the safe-poll helper:

```python
def _count_v2_safe(duck: duckdb.DuckDBPyConnection) -> int:
    """Heartbeat-safe row count. Returns 0 if v2 doesn't exist yet."""
    try:
        row = duck.execute(f"SELECT COUNT(*) FROM corpus.{_V2_TABLE}").fetchone()
        return int(row[0]) if row else 0
    except duckdb.Error:
        return 0
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v`
Expected: 2 PASS (parity + heartbeat).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py tests/corpus/test_duckdb_engine.py
git commit -m "$(cat <<'EOF'
feat(corpus): heartbeat thread for duckdb engine progress (#116)
EOF
)"
```

---

## Task 13: CLI flags + dispatch + sentinel wrapper (TDD)

**Files:**
- Modify: `src/pscanner/corpus/cli.py:115-124,496-518`
- Modify: `src/pscanner/corpus/examples.py:167-251`
- Test: `tests/corpus/test_cli_build_features.py` (create if missing)

- [ ] **Step 1: Write the failing CLI test**

Create or extend `tests/corpus/test_cli_build_features.py`:

```python
"""CLI dispatch tests for `pscanner corpus build-features`."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from pscanner.corpus._build_features_sentinel import SENTINEL_KEY
from pscanner.corpus.cli import build_corpus_parser
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
    from pscanner.corpus._build_features_sentinel import SentinelAlreadySetError
    from pscanner.corpus.cli import _cmd_build_features  # type: ignore[attr-defined]

    db = tmp_path / "corpus.sqlite3"
    build_fixture_db(db)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        CorpusStateRepo(conn).set(SENTINEL_KEY, "1700000000", updated_at=1_700_000_000)
    finally:
        conn.close()

    import argparse, asyncio

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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/corpus/test_cli_build_features.py -v`
Expected: FAIL — flags not recognized.

- [ ] **Step 3: Add the CLI flags**

In `src/pscanner/corpus/cli.py`, find the `build-features` subparser at lines 115-124. Replace those lines with:

```python
    bf = sub.add_parser("build-features", help="Rebuild training_examples from raw events")
    _add_db_arg(bf)
    bf.add_argument("--rebuild", action="store_true", help="Drop and recreate the table")
    bf.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "manifold"],
        default="polymarket",
        help="Platform whose corpus rows feed the training_examples build.",
    )
    bf.add_argument(
        "--engine",
        type=str,
        choices=["python", "duckdb"],
        default="python",
        help=(
            "Build engine. `python` is the row-by-row streaming fold (6h on "
            "the production corpus). `duckdb` is the SQL-pipeline rewrite "
            "(target 5-25 min). See issue #116. Default `python` until "
            "parity is gated."
        ),
    )
    bf.add_argument(
        "--force",
        action="store_true",
        help=(
            "Override the build_features_in_progress sentinel if a prior "
            "run crashed without clearing it."
        ),
    )
    bf.add_argument(
        "--duckdb-memory",
        type=str,
        default=None,
        help=(
            "DuckDB memory_limit (e.g. '6GB'). Default: min(available//2, 12GB). "
            "Only relevant with --engine duckdb."
        ),
    )
    bf.add_argument(
        "--duckdb-threads",
        type=int,
        default=None,
        help=(
            "DuckDB thread count. Default: min(cpu_count, 8). "
            "Only relevant with --engine duckdb."
        ),
    )
```

- [ ] **Step 4: Add the dispatch logic**

Replace the entire `_cmd_build_features` function (lines 496-518) with:

```python
async def _cmd_build_features(args: argparse.Namespace) -> int:
    """Rebuild the training_examples table. Dispatches on ``--engine``."""
    from pscanner.corpus._build_features_sentinel import (
        SentinelAlreadySetError,
        check_and_set_sentinel,
        clear_sentinel,
    )

    db_path = Path(args.db)
    now_ts = int(time.time())

    # Sentinel guard runs for BOTH engines: a crashed Python build also
    # needs --force on retry, since training_examples may be half-truncated.
    sentinel_conn = init_corpus_db(db_path)
    try:
        check_and_set_sentinel(
            CorpusStateRepo(sentinel_conn),
            now_ts=now_ts,
            force=bool(getattr(args, "force", False)),
        )
    finally:
        sentinel_conn.close()

    engine = getattr(args, "engine", "python")
    try:
        if engine == "duckdb":
            written = _run_duckdb_engine(args=args, db_path=db_path, now_ts=now_ts)
        else:
            written = _run_python_engine(args=args, db_path=db_path, now_ts=now_ts)
    except BaseException:
        # Leave the sentinel set so the operator must --force to recover.
        # This is intentional: a partial rebuild left the table corrupt.
        raise

    # DuckDB engine clears the sentinel inside the swap txn. Python engine
    # has no swap; clear here on success.
    if engine != "duckdb":
        clear_conn = init_corpus_db(db_path)
        try:
            clear_sentinel(CorpusStateRepo(clear_conn))
        finally:
            clear_conn.close()

    _log.info("corpus.build_features_done", written=written, engine=engine)
    return 0


def _run_python_engine(
    *, args: argparse.Namespace, db_path: Path, now_ts: int
) -> int:
    write_conn = init_corpus_db(db_path)
    read_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    read_conn.row_factory = sqlite3.Row
    try:
        return build_features(
            trades_repo=CorpusTradesRepo(read_conn),
            resolutions_repo=MarketResolutionsRepo(write_conn),
            examples_repo=TrainingExamplesRepo(write_conn),
            markets_conn=write_conn,
            now_ts=now_ts,
            rebuild=bool(getattr(args, "rebuild", False)),
            platform=args.platform,
        )
    finally:
        read_conn.close()
        write_conn.close()


def _run_duckdb_engine(
    *, args: argparse.Namespace, db_path: Path, now_ts: int
) -> int:
    import os
    import psutil

    from pscanner.corpus._duckdb_engine import build_features_duckdb

    memory_limit = getattr(args, "duckdb_memory", None) or _default_duckdb_memory()
    threads = getattr(args, "duckdb_threads", None) or min(os.cpu_count() or 4, 8)
    temp_dir = db_path.parent / "duckdb_spill"

    return build_features_duckdb(
        db_path=db_path,
        platform=args.platform,
        now_ts=now_ts,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
        threads=threads,
    )


def _default_duckdb_memory() -> str:
    """Default to min(available_memory // 2, 12GB) as bytes-string."""
    import psutil

    half = psutil.virtual_memory().available // 2
    cap = 12 * 1024 * 1024 * 1024
    chosen = min(half, cap)
    return f"{chosen}"
```

> **Note:** `psutil` is already a dependency in this repo. Verify with `rg '"psutil' pyproject.toml`. If not, add it.

- [ ] **Step 5: Run all related tests**

Run: `uv run pytest tests/corpus/test_cli_build_features.py tests/corpus/test_duckdb_engine.py tests/corpus/test_build_features_sentinel.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint + typecheck**

Run: `uv run ruff check src/pscanner/corpus/ tests/corpus/ && uv run ty check`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli_build_features.py
git commit -m "$(cat <<'EOF'
feat(corpus): build-features --engine flag + sentinel dispatch (#116)
EOF
)"
```

---

## Task 14: Operator parity-runner script

**Files:**
- Create: `scripts/parity_build_features.py`

A standalone script the operator runs on the production corpus BEFORE flipping the default engine. Not in CI (too slow); manual gate before PR-B.

- [ ] **Step 1: Write the script**

Create `scripts/parity_build_features.py`:

```python
#!/usr/bin/env python3
"""Compare Python and DuckDB build-features engines on a real corpus.

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    print(f"[parity] python engine done in {time.monotonic()-t0:.1f}s", flush=True)

    print(f"[parity] running duckdb engine on {dd_db}", flush=True)
    t0 = time.monotonic()
    _run_duckdb(
        dd_db,
        platform=args.platform,
        memory=args.duckdb_memory,
        threads=args.duckdb_threads,
    )
    print(f"[parity] duckdb engine done in {time.monotonic()-t0:.1f}s", flush=True)

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
    from pscanner.corpus.examples import build_features
    from pscanner.corpus.repos import (
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
    from pscanner.corpus._duckdb_engine import build_features_duckdb

    build_features_duckdb(
        db_path=db,
        platform=platform,
        now_ts=int(time.time()),
        memory_limit=memory,
        temp_dir=db.parent / "duckdb_spill",
        threads=threads,
    )


def _diff(py_db: Path, dd_db: Path) -> bool:
    """Streaming row-by-row diff. Reports up to 50 mismatches, then bails."""
    order_clause = (
        "ORDER BY platform, condition_id, wallet_address, tx_hash, asset_id, trade_ts"
    )

    py_conn = sqlite3.connect(py_db)
    py_conn.row_factory = sqlite3.Row
    dd_conn = sqlite3.connect(dd_db)
    dd_conn.row_factory = sqlite3.Row

    n_py = py_conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    n_dd = dd_conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    print(f"[parity] python rows: {n_py:,}  duckdb rows: {n_dd:,}", flush=True)
    if n_py != n_dd:
        print(f"[parity] ROW COUNT MISMATCH", flush=True)
        return False

    py_cur = py_conn.execute(f"SELECT * FROM training_examples {order_clause}")
    dd_cur = dd_conn.execute(f"SELECT * FROM training_examples {order_clause}")

    mismatches = 0
    total = 0
    for py_row, dd_row in zip(py_cur, dd_cur, strict=True):
        total += 1
        for col in py_row.keys():
            if col in _SKIP_COLS:
                continue
            a, b = py_row[col], dd_row[col]
            if a is None and b is None:
                continue
            if isinstance(a, float) or isinstance(b, float):
                if a is None or b is None or not math.isclose(
                    float(a), float(b), rel_tol=_FLOAT_RTOL, abs_tol=_FLOAT_ATOL
                ):
                    if mismatches < 50:
                        key = (py_row["condition_id"], py_row["wallet_address"])
                        print(f"[parity] row{total} {key} {col}: py={a!r} dd={b!r}", flush=True)
                    mismatches += 1
            elif a != b:
                if mismatches < 50:
                    key = (py_row["condition_id"], py_row["wallet_address"])
                    print(f"[parity] row{total} {key} {col}: py={a!r} dd={b!r}", flush=True)
                mismatches += 1
        if total % 100_000 == 0:
            print(f"[parity] checked {total:,} rows, {mismatches} mismatches so far", flush=True)

    print(f"[parity] total rows={total:,} mismatches={mismatches}", flush=True)
    return mismatches == 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check scripts/parity_build_features.py`
Expected: clean.

- [ ] **Step 3: Smoke test on the synthetic fixture**

Run:
```bash
uv run python -c "
from tests.corpus._duckdb_fixture import build_fixture_db
from pathlib import Path
build_fixture_db(Path('/tmp/parity_smoke.sqlite3'))
"
uv run python scripts/parity_build_features.py \
    --source /tmp/parity_smoke.sqlite3 \
    --workdir /tmp/parity_workdir \
    --platform polymarket \
    --duckdb-memory 512MB \
    --duckdb-threads 2
```
Expected: prints `[parity] PASS`. Verifies the script works end-to-end on a tiny corpus.

- [ ] **Step 4: Commit**

```bash
git add scripts/parity_build_features.py
git commit -m "$(cat <<'EOF'
feat(corpus): operator parity runner for duckdb engine cutover (#116)
EOF
)"
```

---

## Task 15: Documentation in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a "Tracked work in flight" entry**

Open `CLAUDE.md`. Find the `## Tracked work in flight (issues filed)` section. Append:

```markdown
- **DuckDB engine for `corpus build-features`** (#116, PR-A). New `--engine duckdb` flag on `pscanner corpus build-features` runs a pure-SQL pipeline (ATTACH SQLite read-only, TEMP TABLE materialize, CTE chain producing all 36 columns via window functions, atomic table swap on commit) targeting 5-25 min vs the Python engine's 6h. The heap-driven `prior_wins/losses/resolved_buys/realized_pnl` columns are produced by UNION-ing trades with synthetic resolution events and aggregating with `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`. Crash-safe via `corpus_state['build_features_in_progress']` sentinel that's cleared inside the swap transaction. PR-B (delete Python engine) is gated on `scripts/parity_build_features.py` writing `corpus_state['build_features_parity_passed_at']` newer than the corpus's `MAX(corpus_trades.ts)` — operator must run the parity on the production corpus before the cutover PR merges.
```

- [ ] **Step 2: Update the CLI surface section**

In `## CLI surface`, find the `pscanner corpus build-features [--rebuild]` line. Replace with:

```markdown
- `pscanner corpus build-features [--rebuild] [--engine {python,duckdb}] [--force] [--duckdb-memory SIZE] [--duckdb-threads N]` — (re)build `training_examples` from `corpus_trades` + `market_resolutions`. `--engine python` (default) runs the row-by-row streaming fold (6h on production). `--engine duckdb` runs the SQL-pipeline rewrite (target 5-25 min, see #116). `--force` overrides the in-progress sentinel if a prior run crashed.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude.md): note duckdb engine + parity-gated cutover (#116)
EOF
)"
```

---

## Cutover Runbook (PR-B, post-parity)

This is NOT a task in PR-A. It's the operator's runbook for the follow-up PR that deletes the Python engine.

1. On the desktop training box (RTX 3070, 32GB RAM):
   ```bash
   uv run python scripts/parity_build_features.py \
       --source /home/macph/projects/polymarketScanner/data/corpus.sqlite3 \
       --workdir /tmp/parity \
       --platform polymarket \
       --duckdb-memory 16GB \
       --duckdb-threads 8
   ```
   Wall time: bounded by Python engine (expect ~6h on production corpus).

2. Confirm parity-passed flag is written:
   ```bash
   sqlite3 data/corpus.sqlite3 \
       "SELECT key, value, updated_at FROM corpus_state \
        WHERE key = 'build_features_parity_passed_at';"
   ```

3. Open PR-B which:
   - Deletes `src/pscanner/corpus/features.py:StreamingHistoryProvider` (the row-by-row fold engine).
   - Deletes `_run_python_engine` from `cli.py`.
   - Collapses `--engine` to a single duckdb default (or removes it entirely if no rollback path is wanted).
   - CI check refuses to merge unless `build_features_parity_passed_at` exists in the operator-supplied corpus AND its value is newer than `SELECT MAX(ts) FROM corpus_trades`.

---

## Self-Review

**Spec coverage check:**
- ✓ Heap replacement via UNION+window — Task 7
- ✓ `top_category` / `category_diversity` — Task 9
- ✓ `recent_30d_trades` via time-RANGE window — Task 7 (`w_range_30d`)
- ✓ `median_bet_size_usd` skipped (always NULL) — Task 7 placeholder
- ✓ DuckDB resource config (memory, temp, threads) — Tasks 6, 13
- ✓ ATTACH + TEMP TABLE (no Parquet) — Task 6
- ✓ Atomic swap with sentinel-clear in same txn — Task 6 (`_atomic_swap`)
- ✓ `corpus_state['build_features_in_progress']` sentinel — Task 3
- ✓ `CorpusStateRepo.delete()` — Task 2
- ✓ Heartbeat thread (30s, `threading.Event`) — Task 12
- ✓ CLI flags (`--engine`, `--force`, `--duckdb-memory`, `--duckdb-threads`) — Task 13
- ✓ Parity test on synthetic fixture (CI) — Tasks 4, 5
- ✓ Operator parity runner for full corpus — Task 14
- ✓ Documentation — Task 15
- ✓ PR-B gating mechanism — Task 14 (writes `build_features_parity_passed_at`)

**Type consistency:**
- `build_features_duckdb` signature consistent across Tasks 5, 6, 13, 14.
- `_heartbeat_loop` signature consistent across Task 12 (impl) and Task 12 (test).
- `SENTINEL_KEY` consistent across Tasks 3, 13.
- `CorpusStateRepo.delete(key)` signature consistent across Tasks 2, 3.

**Open implementation risks (call out before starting):**
1. **DuckDB-SQLite ALTER INDEX RENAME.** Task 6 swap path uses `ALTER INDEX ... RENAME TO`. SQLite ≥3.25 supports it; verify on the actual `sqlite3.sqlite_version`. If the bundled SQLite lags, fallback is: drop suffixed indexes, swap tables, rebuild indexes with final names (slower but works).
2. **`top_category` tie-break.** Python uses dict-insertion-order on ties. DuckDB's `ARG_MAX` doesn't preserve insertion order. The synthetic fixture is probably small enough to dodge this, but the production parity may surface ties. If so, either match insertion order via an extra `wallet_cat_first_seen` CTE or document the divergence (small num rows affected).
3. **DuckDB-attached-SQLite write performance.** The CTAS into `corpus.{_V2_TABLE}` writes 15M+ rows back through the SQLite extension. This may be the bottleneck — could be 5-10 min just for the write. If so, an optimization is to bulk-INSERT via Parquet intermediate + `sqlite3 .import`. Defer until measured.
