# Issue #39: Corpus Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate full-corpus Polars DataFrame materialization from `pscanner ml train` so training runs end-to-end on the desktop's 32 GB corpus inside a 12 GB WSL2 instance with peak RSS ≤ 10 GB.

**Architecture:** Two-pass streaming. New `pscanner.ml.streaming` module exposes `open_dataset(db_path) -> StreamingDataset` (context manager). The pre-pass runs three small SQL queries at `__enter__` (split partition, encoder fit, row counts) using per-connection `TEMP TABLE`s for split-membership lookups. Per-split chunked SELECTs feed `xgb.QuantileDMatrix(SplitDataIter)` for train/val and a pre-allocated numpy accumulator for test. `load_dataset` and `temporal_split` are deleted; `analyze_model.py` migrates to the streaming API.

**Tech Stack:** Python 3.13, sqlite3 (stdlib), Polars, NumPy, XGBoost 3.x (`DataIter`, `QuantileDMatrix`), pytest. Run quick verify with `cd /home/macph/projects/pscanner-worktrees/issue-39 && uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q tests/ml/`.

---

## Scope notes

- **Branch / worktree:** `feat/issue-39-corpus-streaming` at `/home/macph/projects/pscanner-worktrees/issue-39`, off the merged `main` (which contains #67).
- **Spec:** `docs/superpowers/specs/2026-05-06-ml-streaming-corpus-load.md`. Read it before starting; this plan implements that spec end-to-end.
- **Determinism canary:** the eager path's `test_run_study_is_deterministic_under_same_seed` survives — same seed must produce identical `test_edge` across runs.
- **Streaming-vs-eager parity:** Task 10 captures an eager-path baseline to JSON; Task 12 asserts the post-streaming pipeline matches within ≤ 0.001 absolute. The baseline JSON is checked in alongside the test.
- **Per-task green CI:** every task ends with `uv run pytest -q tests/ml/` clean. The eager path stays alive until Task 14 deletes it. Tasks 11–13 migrate one caller at a time so CI stays green at every commit.
- **`label_won` dtype is int32** (per existing `_INT32_COLS`), not int8 as the spec's TestSplit comment suggested. The plan uses int32 throughout.

---

## File map

| Path | Disposition |
|---|---|
| `src/pscanner/ml/streaming.py` | Create. Owns `open_dataset`, `StreamingDataset`, `TestSplit`, `_SplitIter`, `SplitDataIter`. ~300 LOC target. |
| `src/pscanner/ml/preprocessing.py` | Modify. Tasks 14 deletes `load_dataset`, `temporal_split`, `Split`. Other helpers (`OneHotEncoder`, `drop_leakage_cols`, `build_feature_matrix`, the cast-tuple constants) survive. |
| `src/pscanner/ml/training.py` | Modify (Task 11). `run_study` signature: `df: pl.DataFrame` → `db_path: Path`, accepts `chunk_size`. Preprocessing block replaced by `with open_dataset(...) as ds`. |
| `src/pscanner/ml/cli.py` | Modify (Task 11). Drop `df = load_dataset(db_path)`; pass `db_path` to `run_study`. Add `--chunk-size` arg, default 100_000. |
| `scripts/analyze_model.py` | Modify (Task 13). Replace `load_dataset` + `drop_leakage_cols` + `temporal_split` block with `with open_dataset(db_path) as ds: test = ds.materialize_test()`. |
| `tests/ml/conftest.py` | Modify (Task 1). Add `make_synthetic_examples_db` fixture returning a `Path` to a populated SQLite. Move `_seed_db_from_synthetic` from `test_preprocessing.py` here. |
| `tests/ml/test_streaming.py` | Create (Tasks 2–9, 12). Seven streaming tests + the parity test. |
| `tests/ml/test_preprocessing.py` | Modify (Task 14). Remove the four `test_load_dataset_*` cases + any `test_temporal_split_*`. Helpers `_seed_db_from_synthetic` get moved to conftest in Task 1. |
| `tests/ml/test_training.py` | Modify (Task 11). `test_run_study_*` cases switch from `df=` to `db_path=`. |
| `tests/ml/data/eager_baseline.json` | Create (Task 10). Snapshot of `test_edge`/`test_accuracy`/`test_logloss` from a pre-streaming run on the synthetic fixture. |

---

## Task 1: Add `make_synthetic_examples_db` fixture

Foundation for every test downstream. Wraps the existing `make_synthetic_examples` Polars frame into a populated SQLite that the streaming code can read.

**Files:**
- Modify: `tests/ml/conftest.py`

- [ ] **Step 1: Move `_seed_db_from_synthetic` helper from `test_preprocessing.py` into `conftest.py`**

In `tests/ml/test_preprocessing.py`, locate `_seed_db_from_synthetic` (around line 195-231). Cut the function. In `tests/ml/conftest.py`, paste it under the imports. Add the necessary imports at the top of `conftest.py`:

```python
import sqlite3

from pscanner.corpus.db import init_corpus_db
```

The helper signature stays:

```python
def _seed_db_from_synthetic(conn: sqlite3.Connection, df: pl.DataFrame) -> None:
    """Populate corpus_markets, market_resolutions, training_examples from
    a synthetic-examples Polars frame so load_dataset / open_dataset see
    matching rows."""
    markets = df.select(["condition_id", "resolved_at"]).unique()
    for row in markets.iter_rows(named=True):
        conn.execute(
            """
            INSERT INTO corpus_markets (
              condition_id, event_slug, category, closed_at,
              total_volume_usd, market_slug, backfill_state, enumerated_at
            ) VALUES (?, '', 'sports', ?, 1000.0, '', 'complete', ?)
            """,
            (row["condition_id"], int(row["resolved_at"]), int(row["resolved_at"]) - 1),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions (
              condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, 0, 1, ?, ?)
            """,
            (row["condition_id"], int(row["resolved_at"]), int(row["resolved_at"])),
        )
    examples = df.drop("resolved_at")
    for row in examples.iter_rows(named=True):
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO training_examples ({cols}) VALUES ({placeholders})",  # noqa: S608 -- column names are statically derived from synthetic frame
            tuple(row.values()),
        )
    conn.commit()
```

Update `test_preprocessing.py` imports and existing test calls to import `_seed_db_from_synthetic` from `tests.ml.conftest`.

- [ ] **Step 2: Add the new fixture in `conftest.py`**

After `make_synthetic_examples`:

```python
@pytest.fixture
def make_synthetic_examples_db(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> Callable[..., Path]:
    """Return a builder: (n_markets, rows_per_market, seed) -> Path to populated SQLite."""

    def _build(
        *,
        n_markets: int = 30,
        rows_per_market: int = 20,
        seed: int = 0,
    ) -> Path:
        df = make_synthetic_examples(
            n_markets=n_markets, rows_per_market=rows_per_market, seed=seed
        )
        db_path = tmp_path / f"corpus_n{n_markets}_r{rows_per_market}_s{seed}.sqlite3"
        conn = init_corpus_db(db_path)
        try:
            _seed_db_from_synthetic(conn, df)
        finally:
            conn.close()
        return db_path

    return _build
```

Add to imports at top of `conftest.py`:

```python
from pathlib import Path
```

- [ ] **Step 3: Run the existing test suite to verify the helper move didn't break anything**

```bash
cd /home/macph/projects/pscanner-worktrees/issue-39
uv run pytest -q tests/ml/
```

Expected: all 44+ ml tests pass. The four `test_load_dataset_*` and `test_run_study_*` cases still consume the old fixture; they should be unaffected.

- [ ] **Step 4: Lint + type-check**

```bash
uv run ruff check tests/ml/conftest.py tests/ml/test_preprocessing.py
uv run ruff format --check tests/ml/conftest.py tests/ml/test_preprocessing.py
uv run ty check
```

Expected: clean (modulo pre-existing ty diagnostics in `tests/corpus/`).

- [ ] **Step 5: Commit**

```bash
git add tests/ml/conftest.py tests/ml/test_preprocessing.py
git commit -m "$(cat <<'EOF'
test(ml): hoist _seed_db_from_synthetic + add db-path fixture

Foundation for the upcoming streaming tests. Moves the existing
helper from test_preprocessing.py into the shared conftest.py and
adds a make_synthetic_examples_db fixture that returns a Path to
a populated SQLite (parallel to the existing pl.DataFrame fixture).
EOF
)"
```

---

## Task 2: P1 — Split partition + `open_dataset` skeleton

First slice of `streaming.py`. Implements P1 (temporal split partition via `market_resolutions` SELECT) and the context-manager scaffolding.

**Files:**
- Create: `src/pscanner/ml/streaming.py`
- Create: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test for split partition**

Create `tests/ml/test_streaming.py`:

```python
"""Tests for ml.streaming."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pscanner.ml.streaming import open_dataset


def test_open_dataset_partitions_markets_by_resolved_at(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """Markets are partitioned 60/20/20 by resolved_at, sorted ascending."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        train = ds._train_markets
        val = ds._val_markets
        test = ds._test_markets

    # 20 markets at 60/20/20 = 12/4/4
    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4

    # Disjoint
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    # Synthetic markets are named 0xmarket{idx:03d} with monotonically
    # increasing resolved_at, so train must contain idx 0-11, val 12-15,
    # test 16-19.
    assert "0xmarket000" in train
    assert "0xmarket011" in train
    assert "0xmarket012" in val
    assert "0xmarket015" in val
    assert "0xmarket016" in test
    assert "0xmarket019" in test


def test_open_dataset_closes_pre_pass_connection_on_exit(
    monkeypatch: pytest.MonkeyPatch,
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """The pre-pass sqlite connection is closed when the context exits."""
    import sqlite3 as _sqlite3

    db_path = make_synthetic_examples_db(n_markets=10, rows_per_market=5, seed=0)
    real_connect = _sqlite3.connect
    captured: list[_sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr("pscanner.ml.streaming.sqlite3.connect", tracking_connect)

    with open_dataset(db_path) as ds:
        assert ds._train_markets  # touch attr to ensure pre-pass ran

    # The pre-pass opens exactly one connection; __exit__ closes it.
    assert len(captured) == 1, f"expected 1 connection, got {len(captured)}"
    pre_pass_conn = captured[0]
    with pytest.raises(_sqlite3.ProgrammingError):
        pre_pass_conn.execute("SELECT 1")
```

- [ ] **Step 2: Run the test, see it fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_open_dataset_partitions_markets_by_resolved_at -v
```

Expected: `ModuleNotFoundError: No module named 'pscanner.ml.streaming'`.

- [ ] **Step 3: Create `streaming.py` with P1 + skeleton**

Create `src/pscanner/ml/streaming.py`:

```python
"""Streaming corpus load for the training pipeline.

Replaces the eager ``preprocessing.load_dataset`` / ``temporal_split`` path
with a two-pass architecture: a small pre-pass at ``__enter__`` computes
the temporal split partition, the encoder fit, and per-split row counts;
per-split chunked SELECTs are then fed into ``xgb.QuantileDMatrix`` via
``xgb.DataIter``.

Public API:

* :func:`open_dataset` -- context manager returning a :class:`StreamingDataset`.
* :class:`StreamingDataset` -- exposes ``dtrain``/``dval``/``val_aux``/``materialize_test``.
* :class:`TestSplit` -- materialized test split for ``evaluate_on_test``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

_TRAIN_FRAC = 0.6
_VAL_FRAC = 0.2


@dataclass
class StreamingDataset:
    """Two-pass streaming view over training_examples.

    Constructed by :func:`open_dataset`. Pre-scan results live on the
    instance; per-split chunked reads are deferred until ``dtrain`` /
    ``dval`` / ``val_aux`` / ``materialize_test`` is called.
    """

    _db_path: Path
    _chunk_size: int
    _train_markets: frozenset[str] = field(default_factory=frozenset)
    _val_markets: frozenset[str] = field(default_factory=frozenset)
    _test_markets: frozenset[str] = field(default_factory=frozenset)


def _partition_markets(
    conn: sqlite3.Connection,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Run P1: SELECT condition_id, resolved_at FROM market_resolutions ORDER BY...

    Slice the sorted list at 60% / 80% into train, val, test.
    """
    rows = conn.execute(
        "SELECT condition_id, resolved_at FROM market_resolutions "
        "ORDER BY resolved_at, condition_id"
    ).fetchall()
    n = len(rows)
    n_train = round(_TRAIN_FRAC * n)
    n_val = round(_VAL_FRAC * n)
    train = frozenset(r[0] for r in rows[:n_train])
    val = frozenset(r[0] for r in rows[n_train : n_train + n_val])
    test = frozenset(r[0] for r in rows[n_train + n_val :])
    return train, val, test


@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[StreamingDataset]:
    """Open the corpus for streaming training.

    Args:
        db_path: Path to the corpus SQLite database.
        chunk_size: Rows per chunk fed into xgboost's DataIter. Default
            100_000; see Issue #39 for the memory / overhead trade-off.

    Yields:
        A :class:`StreamingDataset` whose pre-scan has completed.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        train, val, test = _partition_markets(conn)
        ds = StreamingDataset(
            _db_path=db_path,
            _chunk_size=chunk_size,
            _train_markets=train,
            _val_markets=val,
            _test_markets=test,
        )
        yield ds
    finally:
        conn.close()
```

- [ ] **Step 4: Run the test, see it pass**

```bash
uv run pytest tests/ml/test_streaming.py::test_open_dataset_partitions_markets_by_resolved_at -v
```

Expected: PASS.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean; full ml suite green.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — open_dataset + P1 split partition (#39)

First slice of the streaming pipeline. open_dataset is a context
manager that opens the corpus, runs P1 (SELECT condition_id,
resolved_at FROM market_resolutions ORDER BY resolved_at), and
slices the sorted markets 60/20/20 into three frozenset[str].
EOF
)"
```

---

## Task 3: P2 — Encoder fit via per-connection temp table

Adds the encoder-fit pre-pass. Uses a `TEMP TABLE` populated with the train markets to bypass SQLite's parameter limit (pre-mitigation for risk #3 from the spec).

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test for encoder fit**

Append to `tests/ml/test_streaming.py`:

```python
def test_encoder_fits_on_train_levels_only(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """OneHotEncoder.levels reflects only train-split categorical levels."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        # Synthetic encoder always fits 'side', 'top_category', 'market_category'
        assert "side" in ds.encoder.levels
        assert "top_category" in ds.encoder.levels
        assert "market_category" in ds.encoder.levels

        # Side is ('YES', 'NO'); both levels show up given enough rows
        assert set(ds.encoder.levels["side"]).issubset({"YES", "NO"})

        # Encoder.levels values are tuples of strings (deterministic order)
        for col, lvls in ds.encoder.levels.items():
            assert isinstance(lvls, tuple)
            assert all(isinstance(v, str) for v in lvls)


def test_open_dataset_uses_temp_table_for_split_filter(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """The encoder-fit query joins on a per-connection temp table.

    Confirmed by checking we don't hit SQLite's parameter limit for huge
    splits — synthesize 5,000 markets and assert no OperationalError.
    """
    db_path = make_synthetic_examples_db(n_markets=5_000, rows_per_market=1, seed=0)
    with open_dataset(db_path) as ds:
        # Just touching .encoder forces P2 to have run.
        _ = ds.encoder
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_encoder_fits_on_train_levels_only tests/ml/test_streaming.py::test_open_dataset_uses_temp_table_for_split_filter -v
```

Expected: FAIL with `AttributeError: 'StreamingDataset' object has no attribute 'encoder'`.

- [ ] **Step 3: Add P2 + encoder field**

In `src/pscanner/ml/streaming.py`, add imports at top:

```python
import polars as pl

from pscanner.ml.preprocessing import (
    CATEGORICAL_COLS,
    OneHotEncoder,
)
```

Add the `encoder` field to `StreamingDataset`:

```python
@dataclass
class StreamingDataset:
    _db_path: Path
    _chunk_size: int
    _train_markets: frozenset[str] = field(default_factory=frozenset)
    _val_markets: frozenset[str] = field(default_factory=frozenset)
    _test_markets: frozenset[str] = field(default_factory=frozenset)
    encoder: OneHotEncoder | None = None
```

Add a temp-table helper and the P2 query function:

```python
def _populate_temp_table(
    conn: sqlite3.Connection,
    table_name: str,
    condition_ids: frozenset[str],
) -> None:
    """Create + populate a per-connection TEMP TABLE for split-membership joins.

    Used in place of an ``IN (?, ?, ...)`` parameterized query so the
    SQLite ``SQLITE_MAX_VARIABLE_NUMBER`` limit (32766 in 3.32+) doesn't
    bite as the corpus grows. Cost: ~10K INSERTs at startup, < 100 ms.
    """
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")  # noqa: S608 -- table_name is a literal at every call site
    conn.execute(
        f"CREATE TEMP TABLE {table_name} (condition_id TEXT PRIMARY KEY)"  # noqa: S608 -- table_name is a literal
    )
    conn.executemany(
        f"INSERT INTO {table_name} VALUES (?)",  # noqa: S608 -- table_name is a literal
        [(cid,) for cid in condition_ids],
    )


def _fit_encoder_on_train(
    conn: sqlite3.Connection,
    train_markets: frozenset[str],
) -> OneHotEncoder:
    """Run P2: SELECT DISTINCT side, top_category, market_category FROM training_examples
    WHERE condition_id IN train_markets. Fit OneHotEncoder on the result.
    """
    _populate_temp_table(conn, "_p2_train", train_markets)
    rows = conn.execute(
        "SELECT DISTINCT side, top_category, market_category "
        "FROM training_examples te "
        "JOIN _p2_train tm USING (condition_id)"
    ).fetchall()
    df = pl.DataFrame(
        rows,
        schema={
            "side": pl.String,
            "top_category": pl.String,
            "market_category": pl.String,
        },
        orient="row",
    )
    return OneHotEncoder.fit(df, columns=CATEGORICAL_COLS)
```

Update `open_dataset` to call P2:

```python
@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[StreamingDataset]:
    conn = sqlite3.connect(str(db_path))
    try:
        train, val, test = _partition_markets(conn)
        encoder = _fit_encoder_on_train(conn, train)
        ds = StreamingDataset(
            _db_path=db_path,
            _chunk_size=chunk_size,
            _train_markets=train,
            _val_markets=val,
            _test_markets=test,
            encoder=encoder,
        )
        yield ds
    finally:
        conn.close()
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 4 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — P2 encoder fit via temp table (#39)

Fits the OneHotEncoder by SELECTing DISTINCT side/top_category/
market_category from training_examples joined on a per-connection
TEMP TABLE populated with the train split's condition_ids. Temp
table avoids SQLite's parameter limit as the corpus grows.
EOF
)"
```

---

## Task 4: P3 — Per-split row counts + `feature_names`

Adds row counts and the post-encoding feature-name tuple. After this task, the pre-pass is feature-complete.

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ml/test_streaming.py`:

```python
def test_open_dataset_reports_row_counts(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """n_train_rows, n_val_rows, n_test_rows match SUM of per-split rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        # 20 markets × 5 rows = 100 total. 60/20/20 split = 60/20/20.
        assert ds.n_train_rows == 60
        assert ds.n_val_rows == 20
        assert ds.n_test_rows == 20


def test_feature_names_excludes_carriers_and_label(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """ds.feature_names is the post-encoding column list, less carriers + label."""
    db_path = make_synthetic_examples_db(n_markets=10, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        names = ds.feature_names
        # Sentinel exclusions
        assert "condition_id" not in names
        assert "trade_ts" not in names
        assert "resolved_at" not in names
        assert "label_won" not in names
        # Cat columns are gone, replaced by indicators
        assert "side" not in names
        assert "side__YES" in names or "side__NO" in names
        # Non-cat numeric column survives
        assert "implied_prob_at_buy" in names
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_open_dataset_reports_row_counts tests/ml/test_streaming.py::test_feature_names_excludes_carriers_and_label -v
```

Expected: FAIL with `AttributeError: 'StreamingDataset' object has no attribute 'n_train_rows'`.

- [ ] **Step 3: Add the P3 query + feature-name derivation**

In `src/pscanner/ml/streaming.py`, add to imports:

```python
from pscanner.ml.preprocessing import (
    CARRIER_COLS,
    CATEGORICAL_COLS,
    OneHotEncoder,
    _NEVER_LOAD_COLS,
)
```

(`_NEVER_LOAD_COLS` is the existing private constant in `preprocessing.py`. Re-export it from `preprocessing.py` if `ty` complains about the leading underscore — change the import to `from pscanner.ml.preprocessing import _NEVER_LOAD_COLS as NEVER_LOAD_COLS` and use the renamed alias inside `streaming.py`.)

Add the row-count query and feature-name derivation:

```python
def _count_split_rows(
    conn: sqlite3.Connection,
    train: frozenset[str],
    val: frozenset[str],
    test: frozenset[str],
) -> tuple[int, int, int]:
    """Run P3: COUNT(*) per split via temp tables."""
    counts = []
    for label, markets in (("_p3_train", train), ("_p3_val", val), ("_p3_test", test)):
        _populate_temp_table(conn, label, markets)
        (n,) = conn.execute(
            f"SELECT COUNT(*) FROM training_examples te "  # noqa: S608 -- label is a literal
            f"JOIN {label} sm USING (condition_id)"
        ).fetchone()
        counts.append(int(n))
    return counts[0], counts[1], counts[2]


def _kept_columns(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return training_examples columns minus _NEVER_LOAD_COLS, in PRAGMA order.

    Equivalent to the SELECT-list construction in the deleted load_dataset.
    """
    rows = conn.execute("PRAGMA table_info(training_examples)").fetchall()
    return tuple(r[1] for r in rows if r[1] not in _NEVER_LOAD_COLS)


def _derive_feature_names(
    kept_cols: tuple[str, ...],
    encoder: OneHotEncoder,
) -> tuple[str, ...]:
    """Compute the post-encoding column list, less carriers and label.

    Mirrors the deleted ``temporal_split`` + ``encoder.transform`` +
    ``build_feature_matrix`` pipeline analytically. Encoder.transform appends
    ``{col}__{level}`` indicators for each categorical column and drops the
    original; non-categorical columns keep their original SELECT order.
    """
    excluded = {*CARRIER_COLS, "label_won"}
    non_cat = [c for c in kept_cols if c not in encoder.levels]
    # resolved_at gets joined in by the SELECT (not in PRAGMA).
    if "resolved_at" not in non_cat:
        non_cat.append("resolved_at")
    indicators = [
        f"{col}__{lvl}" for col, lvls in encoder.levels.items() for lvl in lvls
    ]
    return tuple(c for c in [*non_cat, *indicators] if c not in excluded)
```

Update `StreamingDataset` and `open_dataset`:

```python
@dataclass
class StreamingDataset:
    _db_path: Path
    _chunk_size: int
    _train_markets: frozenset[str] = field(default_factory=frozenset)
    _val_markets: frozenset[str] = field(default_factory=frozenset)
    _test_markets: frozenset[str] = field(default_factory=frozenset)
    encoder: OneHotEncoder | None = None
    feature_names: tuple[str, ...] = ()
    _kept_cols: tuple[str, ...] = ()
    n_train_rows: int = 0
    n_val_rows: int = 0
    n_test_rows: int = 0


@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[StreamingDataset]:
    conn = sqlite3.connect(str(db_path))
    try:
        train, val, test = _partition_markets(conn)
        encoder = _fit_encoder_on_train(conn, train)
        n_train, n_val, n_test = _count_split_rows(conn, train, val, test)
        kept = _kept_columns(conn)
        ds = StreamingDataset(
            _db_path=db_path,
            _chunk_size=chunk_size,
            _train_markets=train,
            _val_markets=val,
            _test_markets=test,
            encoder=encoder,
            feature_names=_derive_feature_names(kept, encoder),
            _kept_cols=kept,
            n_train_rows=n_train,
            n_val_rows=n_val,
            n_test_rows=n_test,
        )
        yield ds
    finally:
        conn.close()
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 6 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — P3 row counts + feature_names derivation (#39)

Completes the pre-pass: per-split COUNT(*) queries (sized via the
same temp-table mechanism as P2) and the post-encoding feature-name
tuple derived analytically from PRAGMA table_info + encoder.levels.
EOF
)"
```

---

## Task 5: `_SplitIter` — chunked numpy iterator

The chunked SELECT-and-encode loop. After this task, we can iterate (X, y, implied) numpy chunks for any split.

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_streaming.py`:

```python
def test_split_iter_yields_expected_chunk_count(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """chunk_size=50 over 60 train rows yields 2 chunks (50 + 10)."""
    from pscanner.ml.streaming import _SplitIter

    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        assert ds.encoder is not None  # narrow for ty
        it = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=50,
        )
        chunks = list(iter(it))

    assert len(chunks) == 2  # 60 train rows / 50 = 2 chunks (50 + 10)
    x0, y0, implied0 = chunks[0]
    assert x0.shape[0] == 50
    assert x0.dtype.name == "float32"
    assert y0.shape == (50,)
    assert implied0.shape == (50,)

    x1, _, _ = chunks[1]
    assert x1.shape[0] == 10  # final partial chunk


def test_split_iter_x_columns_match_feature_names(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """The numpy x matrix has exactly len(feature_names) columns."""
    from pscanner.ml.streaming import _SplitIter

    db_path = make_synthetic_examples_db(n_markets=10, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=100) as ds:
        assert ds.encoder is not None
        it = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=100,
        )
        x, _, _ = next(iter(it))

    assert x.shape[1] == len(ds.feature_names)
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_split_iter_yields_expected_chunk_count -v
```

Expected: FAIL with `ImportError: cannot import name '_SplitIter'`.

- [ ] **Step 3: Implement `_SplitIter`**

In `src/pscanner/ml/streaming.py`, add to imports:

```python
import numpy as np

from pscanner.ml.preprocessing import (
    _CATEGORICAL_CAST_COLS,
    _FLOAT32_COLS,
    _INT32_COLS,
    build_feature_matrix,
    drop_leakage_cols,
)
```

(If `ty` flags the leading-underscore imports, expose them as public re-exports in `preprocessing.py`. The simplest fix: rename the constants in `preprocessing.py` by removing the leading underscore — they're documented as part of the module's surface in the existing docstring.)

Add the iterator class:

```python
@dataclass
class _SplitIter:
    """Yields (x, y, implied) numpy tuples per chunk for one split.

    Each iter() opens a fresh sqlite3.Connection (XGBoost's DataIter may
    iterate from worker threads; sqlite3 connections aren't thread-safe).
    The connection's TEMP TABLE is populated from condition_ids on first
    iteration; the connection closes when iteration finishes or raises.
    """

    db_path: Path
    condition_ids: frozenset[str]
    encoder: OneHotEncoder
    kept_cols: tuple[str, ...]
    chunk_size: int

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        select_list = ", ".join(f"te.{c}" for c in self.kept_cols)
        sql = (
            f"SELECT {select_list}, mr.resolved_at "  # noqa: S608 -- kept_cols derived from PRAGMA
            "FROM training_examples te "
            "JOIN market_resolutions mr USING (condition_id) "
            "JOIN _split_markets sm USING (condition_id) "
            "ORDER BY te.id"
        )
        col_names = (*self.kept_cols, "resolved_at")
        conn = sqlite3.connect(str(self.db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self.condition_ids)
            cursor = conn.execute(sql)
            while True:
                rows = cursor.fetchmany(self.chunk_size)
                if not rows:
                    return
                yield self._encode_chunk(rows, col_names)
        finally:
            conn.close()

    def _encode_chunk(
        self,
        rows: list[tuple],
        col_names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = pl.DataFrame(rows, schema=list(col_names), orient="row")
        # Mirror load_dataset's dtype casting (preserved from preprocessing.py).
        cast_exprs = [
            *[pl.col(c).cast(pl.Categorical) for c in _CATEGORICAL_CAST_COLS if c in df.columns],
            *[pl.col(c).cast(pl.Int32) for c in _INT32_COLS if c in df.columns],
            *[pl.col(c).cast(pl.Float32) for c in _FLOAT32_COLS if c in df.columns],
        ]
        df = df.with_columns(cast_exprs)
        df = drop_leakage_cols(df)  # idempotent; no-op when SELECT already excluded them
        df = self.encoder.transform(df)
        return build_feature_matrix(df)
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 8 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — _SplitIter chunked numpy iterator (#39)

Per-split chunked SELECT through encoder.transform + build_feature_matrix,
yielding (x, y, implied) numpy tuples. Each iter() opens a fresh
sqlite3.Connection (worker-thread safety) and populates a TEMP TABLE
for the split's condition_ids. ORDER BY te.id keeps chunk order
deterministic across runs.
EOF
)"
```

---

## Task 6: `SplitDataIter` — XGBoost adapter

Wraps `_SplitIter` in xgboost's `DataIter` protocol. Used by `dtrain` / `dval` in Task 7.

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_streaming.py`:

```python
def test_split_data_iter_passes_chunks_to_input_data(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """SplitDataIter feeds each chunk into the input_data callback once."""
    from pscanner.ml.streaming import SplitDataIter, _SplitIter

    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        assert ds.encoder is not None
        source = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=50,
        )
        adapter = SplitDataIter(source)

        captured_chunks = []

        def fake_input_data(*, data, label):
            captured_chunks.append((data.shape[0], label.shape[0]))

        # Drive the iterator until it returns False.
        while adapter.next(fake_input_data):
            pass

        assert len(captured_chunks) == 2  # 50 + 10 over 60 train rows
        assert captured_chunks[0] == (50, 50)
        assert captured_chunks[1] == (10, 10)


def test_split_data_iter_reset_re_iterates(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """reset() lets next() iterate the same SplitIter from the start again."""
    from pscanner.ml.streaming import SplitDataIter, _SplitIter

    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        assert ds.encoder is not None
        source = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=50,
        )
        adapter = SplitDataIter(source)

        first_pass = []
        while adapter.next(lambda *, data, label: first_pass.append(data.shape[0])):
            pass

        adapter.reset()

        second_pass = []
        while adapter.next(lambda *, data, label: second_pass.append(data.shape[0])):
            pass

        assert first_pass == second_pass
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_split_data_iter_passes_chunks_to_input_data -v
```

Expected: FAIL with `ImportError: cannot import name 'SplitDataIter'`.

- [ ] **Step 3: Implement `SplitDataIter`**

In `src/pscanner/ml/streaming.py`, add to imports:

```python
from collections.abc import Callable as _Callable

import xgboost as xgb
```

Add the class:

```python
class SplitDataIter(xgb.DataIter):
    """XGBoost ``DataIter`` adapter over a :class:`_SplitIter`.

    ``release_data=True`` lets XGBoost free each chunk after ingestion;
    in-flight working set per split is one chunk + XGBoost's quantization
    buffer instead of the full numpy matrix.
    """

    def __init__(self, source: _SplitIter) -> None:
        super().__init__(release_data=True)
        self._source = source
        self._iter: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    def next(self, input_data: _Callable[..., None]) -> bool:
        """Pull one chunk; return False when exhausted."""
        if self._iter is None:
            self._iter = iter(self._source)
        try:
            x, y, _implied = next(self._iter)
        except StopIteration:
            self._iter = None
            return False
        input_data(data=x, label=y)
        return True

    def reset(self) -> None:
        """Drop the iterator so the next ``next()`` reopens the cursor."""
        self._iter = None
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 10 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — SplitDataIter xgboost adapter (#39)

Wraps _SplitIter in xgb.DataIter with release_data=True. next()
pulls one chunk from the underlying iterator and hands it to
input_data; reset() drops the iter so the next pass reopens the
cursor.
EOF
)"
```

---

## Task 7: `dtrain()` / `dval()` returning `QuantileDMatrix`

Wires `_SplitIter` + `SplitDataIter` into XGBoost's quantized DMatrix. Public API for the training caller.

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_streaming.py`:

```python
def test_dtrain_returns_quantile_dmatrix_with_expected_shape(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """dtrain() returns a QuantileDMatrix with num_row=n_train_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        dtrain = ds.dtrain(device="cpu")

    import xgboost as xgb
    assert isinstance(dtrain, xgb.QuantileDMatrix)
    assert dtrain.num_row() == ds.n_train_rows == 60
    assert dtrain.num_col() == len(ds.feature_names)


def test_dval_returns_quantile_dmatrix_with_expected_shape(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        dval = ds.dval(device="cpu")

    import xgboost as xgb
    assert isinstance(dval, xgb.QuantileDMatrix)
    assert dval.num_row() == ds.n_val_rows == 20
    assert dval.num_col() == len(ds.feature_names)
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_dtrain_returns_quantile_dmatrix_with_expected_shape -v
```

Expected: FAIL with `AttributeError: 'StreamingDataset' object has no attribute 'dtrain'`.

- [ ] **Step 3: Add the methods**

In `src/pscanner/ml/streaming.py`, expand `StreamingDataset`:

```python
@dataclass
class StreamingDataset:
    _db_path: Path
    _chunk_size: int
    _train_markets: frozenset[str] = field(default_factory=frozenset)
    _val_markets: frozenset[str] = field(default_factory=frozenset)
    _test_markets: frozenset[str] = field(default_factory=frozenset)
    encoder: OneHotEncoder | None = None
    feature_names: tuple[str, ...] = ()
    _kept_cols: tuple[str, ...] = ()
    n_train_rows: int = 0
    n_val_rows: int = 0
    n_test_rows: int = 0

    def dtrain(self, *, device: str) -> xgb.QuantileDMatrix:
        """Build a QuantileDMatrix for the train split."""
        return self._build_dmatrix(self._train_markets, device=device)

    def dval(self, *, device: str) -> xgb.QuantileDMatrix:
        """Build a QuantileDMatrix for the val split."""
        return self._build_dmatrix(self._val_markets, device=device)

    def _build_dmatrix(
        self,
        condition_ids: frozenset[str],
        *,
        device: str,
    ) -> xgb.QuantileDMatrix:
        if self.encoder is None:
            raise RuntimeError("StreamingDataset.encoder is None; was open_dataset used?")
        source = _SplitIter(
            db_path=self._db_path,
            condition_ids=condition_ids,
            encoder=self.encoder,
            kept_cols=self._kept_cols,
            chunk_size=self._chunk_size,
        )
        return xgb.QuantileDMatrix(
            SplitDataIter(source),
            max_bin=256,
            feature_names=list(self.feature_names),
            device=device,
        )
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 12 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — dtrain/dval return QuantileDMatrix (#39)

Wires _SplitIter + SplitDataIter into xgb.QuantileDMatrix at
max_bin=256 with feature_names. Works on both device='cpu' and
device='cuda' since QuantileDMatrix is the unified hist-tree path.
EOF
)"
```

---

## Task 8: `val_aux()` — y_val + implied_prob_val without features

Pulls only the two columns the edge metric closure needs. Skips feature-matrix construction entirely.

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_streaming.py`:

```python
def test_val_aux_returns_y_and_implied_prob_arrays(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """val_aux() returns (y_val, implied_prob_val) of length n_val_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        y_val, implied_val = ds.val_aux()

    assert y_val.shape == (20,)
    assert implied_val.shape == (20,)
    # Labels are 0/1 ints
    assert set(y_val.tolist()).issubset({0, 1})
    # Implied probabilities are in [0, 1]
    assert (implied_val >= 0.0).all()
    assert (implied_val <= 1.0).all()
```

- [ ] **Step 2: Run the test, see it fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_val_aux_returns_y_and_implied_prob_arrays -v
```

Expected: FAIL with `AttributeError: 'StreamingDataset' object has no attribute 'val_aux'`.

- [ ] **Step 3: Add `val_aux()`**

Add to `StreamingDataset`:

```python
    def val_aux(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_val, implied_prob_val)`` numpy arrays.

        Streamed via the same chunked path as ``dval`` but pulls only
        the two columns the edge-metric closure needs, into pre-allocated
        arrays sized from the P3 row count. No feature matrix is built.
        """
        y = np.empty(self.n_val_rows, dtype=np.int32)
        implied = np.empty(self.n_val_rows, dtype=np.float32)

        sql = (
            "SELECT te.label_won, te.implied_prob_at_buy "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._val_markets)
            cursor = conn.execute(sql)
            offset = 0
            while True:
                rows = cursor.fetchmany(self._chunk_size)
                if not rows:
                    break
                end = offset + len(rows)
                for i, (label, prob) in enumerate(rows):
                    y[offset + i] = int(label)
                    implied[offset + i] = float(prob)
                offset = end
        finally:
            conn.close()

        return y, implied
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 13 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — val_aux returns y_val + implied (#39)

Streams y_val and implied_prob_val into pre-allocated numpy arrays
sized from P3's count. No feature matrix is built — these are the
only two arrays the edge-metric closure consumes.
EOF
)"
```

---

## Task 9: `materialize_test()` — TestSplit with unencoded `top_categories`

Materializes the test split into numpy arrays. Streamed in via `_SplitIter` for X/y/implied; a parallel small SELECT pulls the unencoded `top_category` column for `TestSplit.top_categories`.

**Files:**
- Modify: `src/pscanner/ml/streaming.py`
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_streaming.py`:

```python
def test_materialize_test_returns_unencoded_top_categories(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """TestSplit.top_categories is the raw category strings, parallel to .y."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        test = ds.materialize_test()

    assert test.x.shape == (ds.n_test_rows, len(ds.feature_names))
    assert test.x.dtype.name == "float32"
    assert test.y.shape == (ds.n_test_rows,)
    assert test.implied_prob.shape == (ds.n_test_rows,)
    assert test.top_categories.shape == (ds.n_test_rows,)
    # top_categories is unencoded — strings like "sports" / "esports" / "thesis"
    # (or empty string for nulls, mirroring _extract_top_category's fill_null).
    assert test.top_categories.dtype == object
    valid = {"sports", "esports", "thesis", ""}
    assert all(v in valid for v in test.top_categories.tolist())
```

- [ ] **Step 2: Run the test, see it fail**

```bash
uv run pytest tests/ml/test_streaming.py::test_materialize_test_returns_unencoded_top_categories -v
```

Expected: FAIL with `AttributeError: 'StreamingDataset' object has no attribute 'materialize_test'`.

- [ ] **Step 3: Add `TestSplit` + `materialize_test()`**

In `src/pscanner/ml/streaming.py`, add the dataclass:

```python
@dataclass(frozen=True)
class TestSplit:
    """Materialized test split for ``evaluate_on_test`` + ``analyze_model.py``."""

    x: np.ndarray              # float32, shape (n_test_rows, n_features)
    y: np.ndarray              # int32 labels
    implied_prob: np.ndarray   # float32
    top_categories: np.ndarray # object (str), unencoded — for per-category breakdowns
```

Add to `StreamingDataset`:

```python
    def materialize_test(self) -> TestSplit:
        """Stream the test split into pre-allocated numpy arrays.

        Uses the same _SplitIter as dtrain/dval for X/y/implied; pulls
        the unencoded ``top_category`` column via a parallel SELECT for
        per-category metrics in evaluate_on_test.
        """
        if self.encoder is None:
            raise RuntimeError("StreamingDataset.encoder is None; was open_dataset used?")

        x = np.empty((self.n_test_rows, len(self.feature_names)), dtype=np.float32)
        y = np.empty(self.n_test_rows, dtype=np.int32)
        implied = np.empty(self.n_test_rows, dtype=np.float32)

        source = _SplitIter(
            db_path=self._db_path,
            condition_ids=self._test_markets,
            encoder=self.encoder,
            kept_cols=self._kept_cols,
            chunk_size=self._chunk_size,
        )
        offset = 0
        for x_chunk, y_chunk, implied_chunk in iter(source):
            end = offset + x_chunk.shape[0]
            x[offset:end] = x_chunk
            y[offset:end] = y_chunk
            implied[offset:end] = implied_chunk
            offset = end

        # Parallel small SELECT for unencoded top_category strings.
        # Mirrors the deleted _extract_top_category: nulls become "".
        top_categories = np.empty(self.n_test_rows, dtype=object)
        sql = (
            "SELECT COALESCE(te.top_category, '') "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._test_markets)
            for i, (cat,) in enumerate(conn.execute(sql)):
                top_categories[i] = cat
        finally:
            conn.close()

        return TestSplit(x=x, y=y, implied_prob=implied, top_categories=top_categories)
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
uv run pytest tests/ml/test_streaming.py -v
```

Expected: all 14 streaming tests pass.

- [ ] **Step 5: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ruff format --check src/pscanner/ml/streaming.py tests/ml/test_streaming.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
feat(ml): streaming.py — materialize_test returns TestSplit (#39)

The test split is small enough (~1-2M rows) to materialize into
pre-allocated numpy arrays. _SplitIter feeds X/y/implied; a parallel
small SELECT pulls the unencoded top_category strings for per-category
metrics in evaluate_on_test. Mirrors the deleted _extract_top_category's
null-to-"" fill.
EOF
)"
```

---

## Task 10: Capture eager-path baseline JSON

Run the still-alive eager path on the synthetic fixture, save `test_edge` / `test_accuracy` / `test_logloss` to `tests/ml/data/eager_baseline.json`. Used by Task 12's parity test after the streaming migration in Task 11 lands.

**Files:**
- Create: `tests/ml/data/eager_baseline.json`
- Create: `tests/ml/_capture_eager_baseline.py` (script-style; deleted in Task 14 along with `load_dataset`)

- [ ] **Step 1: Write the capture script**

Create `tests/ml/_capture_eager_baseline.py`:

```python
"""One-off capture of eager-path metrics for the streaming-vs-eager parity test.

Runs ``run_study`` with the EAGER pipeline (load_dataset + temporal_split)
on a deterministic synthetic fixture. Saves test_edge / test_accuracy /
test_logloss to tests/ml/data/eager_baseline.json.

This script + the JSON file get deleted in the same commit that deletes
load_dataset (Task 14). Until then, the JSON is the regression baseline
the streaming path must match within ≤ 0.001 absolute.

Run:
    uv run python -m tests.ml._capture_eager_baseline
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from pscanner.ml.training import run_study
from tests.ml.conftest import _make_synthetic_examples


def main() -> None:
    np.random.seed(42)
    df = _make_synthetic_examples(n_markets=30, rows_per_market=20, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        run_study(
            df=df,
            output_dir=out,
            n_trials=3,
            n_jobs=1,
            n_min=5,
            seed=42,
        )
        metrics = json.loads((out / "metrics.json").read_text())

    baseline = {
        "fixture": {"n_markets": 30, "rows_per_market": 20, "seed": 3},
        "study": {"n_trials": 3, "n_jobs": 1, "n_min": 5, "seed": 42},
        "test_edge": metrics["test_edge"],
        "test_accuracy": metrics["test_accuracy"],
        "test_logloss": metrics["test_logloss"],
    }
    target = Path(__file__).parent / "data" / "eager_baseline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the capture script**

```bash
cd /home/macph/projects/pscanner-worktrees/issue-39
uv run python -m tests.ml._capture_eager_baseline
```

Expected: prints `Wrote .../tests/ml/data/eager_baseline.json`. The JSON has three numeric metrics. If the run errors, the eager pipeline regressed — investigate before continuing.

- [ ] **Step 3: Sanity-check the JSON shape**

```bash
cat tests/ml/data/eager_baseline.json
```

Expected: a JSON object with `fixture`, `study`, `test_edge`, `test_accuracy`, `test_logloss` keys. Numeric values are floats.

- [ ] **Step 4: Commit**

```bash
git add tests/ml/_capture_eager_baseline.py tests/ml/data/eager_baseline.json
git commit -m "$(cat <<'EOF'
test(ml): capture eager-path baseline for streaming parity test (#39)

One-off snapshot of test_edge/test_accuracy/test_logloss from the
current eager run_study path on a deterministic synthetic fixture.
Used by the upcoming streaming-vs-eager parity test as a regression
baseline. Both the capture script and the JSON get deleted when
load_dataset is removed.
EOF
)"
```

---

## Task 11: Migrate `run_study` + `cli.py`

Replace the preprocessing block in `run_study` with `with open_dataset(db_path) as ds`. Update CLI to pass `db_path` directly and add `--chunk-size`.

**Files:**
- Modify: `src/pscanner/ml/training.py`
- Modify: `src/pscanner/ml/cli.py`
- Modify: `tests/ml/test_training.py`

- [ ] **Step 1: Update `test_run_study_*` cases to pass `db_path` instead of `df`**

In `tests/ml/test_training.py`, find each `test_run_study_*` case (~lines 199, 238, 353, 372, 394). Each currently does:

```python
df = make_synthetic_examples(n_markets=..., rows_per_market=..., seed=...)
run_study(df=df, output_dir=..., ...)
```

Change to:

```python
db_path = make_synthetic_examples_db(n_markets=..., rows_per_market=..., seed=...)
run_study(db_path=db_path, output_dir=..., ...)
```

Update each test signature to accept `make_synthetic_examples_db: Callable[..., Path]` instead of `make_synthetic_examples: Callable[..., pl.DataFrame]`.

Five test sites to update:
- `test_run_study_writes_all_artifacts`
- `test_run_study_n_jobs_2_completes_without_lock_errors`
- `test_run_study_writes_accepted_categories_to_preprocessor_json`
- `test_run_study_writes_test_edge_filtered_to_metrics_json`
- `test_run_study_is_deterministic_under_same_seed`

- [ ] **Step 2: Run the changed tests, see them fail**

```bash
uv run pytest tests/ml/test_training.py::test_run_study_writes_all_artifacts -v
```

Expected: FAIL with `TypeError: run_study() got an unexpected keyword argument 'db_path'` or similar.

- [ ] **Step 3: Update `run_study` signature + body**

In `src/pscanner/ml/training.py`, find `run_study` (~line 356). Replace the entire function body with:

```python
def run_study(
    db_path: Path,
    output_dir: Path,
    n_trials: int,
    n_jobs: int,
    n_min: int,
    seed: int,
    device: str = "cpu",
    chunk_size: int = 100_000,
    accepted_categories: tuple[str, ...] | None = None,
) -> None:
    """End-to-end study: stream corpus, run Optuna, refit, evaluate, dump.

    Mutates ``output_dir`` (created if missing). Writes ``model.json``,
    ``preprocessor.json``, ``metrics.json``.

    Args:
        db_path: Path to the corpus SQLite database.
        output_dir: Per-run artifact directory.
        n_trials: Optuna trial budget.
        n_jobs: Parallel trials. Must be >=1.
        n_min: Edge-metric anti-overfit guard threshold.
        seed: Master RNG seed.
        device: XGBoost device, ``"cpu"`` or ``"cuda"``.
        chunk_size: Rows per chunk fed into xgboost's DataIter. Default
            100_000.
        accepted_categories: Category strings to gate on at inference
            time. Written to ``preprocessor.json`` as metadata; does NOT
            filter training data.
    """
    resolved_categories = (
        accepted_categories if accepted_categories is not None else _DEFAULT_ACCEPTED_CATEGORIES
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    _log.info("ml.mem", phase="run_study_entry", rss_mb=_rss_mb())

    with open_dataset(db_path, chunk_size=chunk_size) as ds:
        encoder = ds.encoder
        if encoder is None:
            raise RuntimeError("open_dataset did not fit the encoder")
        _log.info("ml.mem", phase="post_pre_pass", rss_mb=_rss_mb())

        dtrain = ds.dtrain(device=device)
        dval = ds.dval(device=device)
        y_val, implied_val = ds.val_aux()
        _log.info("ml.mem", phase="post_dmatrix", rss_mb=_rss_mb())

        rates = {
            "train": _label_rate_from_dmatrix(dtrain),
            "val": float(np.asarray(y_val).mean()),
            "test": -1.0,  # populated after materialize_test below
        }

        best_iteration, best_params, best_value = _run_optimization_phase(
            dtrain=dtrain,
            dval=dval,
            y_val=y_val,
            implied_val=implied_val,
            n_trials=n_trials,
            n_jobs=n_jobs,
            n_min=n_min,
            seed=seed,
            device=device,
        )

        del dval, y_val, implied_val
        gc.collect()
        _log.info("ml.mem", phase="post_optuna", rss_mb=_rss_mb())

        booster = fit_winning_model(
            best_params=best_params,
            best_iteration=best_iteration,
            dtrain=dtrain,
            seed=seed,
            device=device,
        )
        del dtrain
        gc.collect()
        _log.info("ml.mem", phase="post_fit_winning", rss_mb=_rss_mb())

        test = ds.materialize_test()
        rates["test"] = float(test.y.mean())

    _log.info("ml.split_label_won_rate", **rates)

    test_metrics = evaluate_on_test(
        booster=booster,
        X_test=test.x,
        y_test=test.y,
        implied_prob_test=test.implied_prob,
        n_min=n_min,
        top_category_test=test.top_categories,
        accepted_categories=resolved_categories,
    )

    metrics: dict[str, object] = {
        "best_params": best_params,
        "best_iteration": best_iteration,
        "best_val_edge": best_value,
        "test_edge": test_metrics["edge"],
        "test_accuracy": test_metrics["accuracy"],
        "test_logloss": test_metrics["logloss"],
        "test_per_decile": test_metrics["per_decile"],
        "split_label_won_rate": rates,
        "seed": seed,
        "accepted_categories": list(resolved_categories),
    }
    if "edge_filtered" in test_metrics:
        metrics["test_edge_filtered"] = test_metrics["edge_filtered"]
    _dump_artifacts(output_dir, booster, encoder, metrics, resolved_categories)
    _log.info(
        "ml.study_complete",
        best_val_edge=metrics["best_val_edge"],
        test_edge=metrics["test_edge"],
        n_trials=n_trials,
    )
```

Add a helper above `run_study` for the train-side label rate:

```python
def _label_rate_from_dmatrix(d: xgb.QuantileDMatrix) -> float:
    """Read labels off a QuantileDMatrix without re-reading the source.

    XGBoost stores labels internally; ``get_label()`` is cheap.
    """
    return float(np.asarray(d.get_label()).mean())
```

Add the imports near the existing ones in `training.py`:

```python
from pscanner.ml.streaming import open_dataset
```

- [ ] **Step 4: Update `cli.py` to pass `db_path` and accept `--chunk-size`**

In `src/pscanner/ml/cli.py`, find the `train` subcommand parser. Add:

```python
train.add_argument(
    "--chunk-size",
    type=int,
    default=100_000,
    help="Rows per chunk fed into xgboost's DataIter (default 100_000)",
)
```

Find the call to `run_study` (~line 70-90) and the `df = load_dataset(...)` line. Remove the `df = load_dataset(db_path)` line. Change the `run_study` call to:

```python
run_study(
    db_path=db_path,
    output_dir=...,
    n_trials=args.n_trials,
    n_jobs=args.n_jobs,
    n_min=args.n_min,
    seed=args.seed,
    device=args.device,
    chunk_size=args.chunk_size,
    accepted_categories=resolved_categories,
)
```

Drop the `from pscanner.ml.preprocessing import load_dataset` import — it's unused after this change.

- [ ] **Step 5: Run the migrated tests, see them pass**

```bash
uv run pytest tests/ml/test_training.py -v
```

Expected: all `test_run_study_*` cases pass on the streaming pipeline. The deterministic case still passes (chunk order is `ORDER BY te.id` — stable across runs).

- [ ] **Step 6: Lint + type-check + full ml suite**

```bash
uv run ruff check src/pscanner/ml/training.py src/pscanner/ml/cli.py tests/ml/test_training.py
uv run ruff format --check src/pscanner/ml/training.py src/pscanner/ml/cli.py tests/ml/test_training.py
uv run ty check
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/ml/training.py src/pscanner/ml/cli.py tests/ml/test_training.py
git commit -m "$(cat <<'EOF'
refactor(ml): run_study + cli use open_dataset (#39)

run_study(df: pl.DataFrame, ...) -> run_study(db_path: Path, *, chunk_size, ...).
The preprocessing block is replaced by `with open_dataset(db_path) as ds:`.
The lifetime trims from #67 are preserved (del dval/y_val/implied_val
after Optuna; del dtrain after fit_winning_model). The CLI gains a
--chunk-size flag (default 100_000).

A new ml.mem phase post_pre_pass is added so we can observe the post-
pre-pass RSS independently from post_dmatrix; the eager-only phases
post_split_and_encode and post_polars_release are gone.
EOF
)"
```

---

## Task 12: Streaming-vs-eager parity test

Asserts the streaming pipeline's `test_edge` matches the JSON baseline within ≤ 0.001 absolute.

**Files:**
- Modify: `tests/ml/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_streaming.py`:

```python
def test_streaming_pipeline_matches_eager_baseline(
    tmp_path: Path,
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """test_edge from the streaming run matches the eager-path snapshot.

    Tolerance: 0.001 absolute (per #39 DoD). The eager baseline is
    captured at tests/ml/data/eager_baseline.json — see
    tests/ml/_capture_eager_baseline.py.
    """
    import json

    from pscanner.ml.training import run_study

    baseline_path = Path(__file__).parent / "data" / "eager_baseline.json"
    baseline = json.loads(baseline_path.read_text())

    db_path = make_synthetic_examples_db(
        n_markets=baseline["fixture"]["n_markets"],
        rows_per_market=baseline["fixture"]["rows_per_market"],
        seed=baseline["fixture"]["seed"],
    )
    output_dir = tmp_path / "streaming_run"
    run_study(
        db_path=db_path,
        output_dir=output_dir,
        n_trials=baseline["study"]["n_trials"],
        n_jobs=baseline["study"]["n_jobs"],
        n_min=baseline["study"]["n_min"],
        seed=baseline["study"]["seed"],
    )

    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert abs(metrics["test_edge"] - baseline["test_edge"]) < 0.001, (
        f"test_edge {metrics['test_edge']} drifted from "
        f"eager baseline {baseline['test_edge']} by more than 0.001"
    )
    # Looser tolerances on the calibration metrics — they shift more under
    # quantization changes but are bounded.
    assert abs(metrics["test_accuracy"] - baseline["test_accuracy"]) < 0.05
    assert abs(metrics["test_logloss"] - baseline["test_logloss"]) < 0.10
```

- [ ] **Step 2: Run the test, see it pass (or fix if it fails)**

```bash
uv run pytest tests/ml/test_streaming.py::test_streaming_pipeline_matches_eager_baseline -v
```

Expected: PASS. The streaming path uses `xgb.QuantileDMatrix(max_bin=256)` which matches the eager path's default 256-bin hist quantization, so `test_edge` should be within 0.001.

If it FAILS by a small margin (e.g. 0.002), the most likely cause is float32 vs. float64 quantization round-off in chunked accumulation. Investigate:
- Confirm `_SplitIter` casts numerics to float32 BEFORE the encoder. If casting happens after, accumulating int columns at int64 precision drifts.
- Confirm `ORDER BY te.id` is on every chunked SELECT.
- Consider widening tolerance to 0.005 if the drift is bounded and not growing.

- [ ] **Step 3: Lint + type-check + full ml suite**

```bash
uv run ruff check tests/ml/test_streaming.py
uv run ruff format --check tests/ml/test_streaming.py
uv run pytest -q tests/ml/
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/ml/test_streaming.py
git commit -m "$(cat <<'EOF'
test(ml): streaming pipeline matches eager baseline (#39)

Asserts post-streaming run_study produces test_edge within 0.001
of the eager-path snapshot in eager_baseline.json. Looser bounds
on test_accuracy (0.05) and test_logloss (0.10) since calibration
shifts more than edge under quantization changes. test_edge is
the metric the project optimizes; tightening it pins the shipped
contract.
EOF
)"
```

---

## Task 13: Migrate `scripts/analyze_model.py`

Single-file migration. The script's flow stays the same; only the data-loading block changes.

**Files:**
- Modify: `scripts/analyze_model.py`

- [ ] **Step 1: Replace the data-loading block**

In `scripts/analyze_model.py`, find `analyze` (~line 136). Replace lines 144-157 (from `print(f"Loading corpus from {db_path}")` through `print(f"Feature matrix: {x_test.shape}")`) with:

```python
    print(f"Loading corpus from {db_path}")
    with open_dataset(db_path) as ds:
        if ds.encoder is None:
            raise RuntimeError("open_dataset did not fit the encoder")
        # Sanity-check: encoder fit on this corpus should match the
        # encoder serialized into preprocessor.json. A mismatch implies
        # the corpus drifted since the model was trained.
        if ds.encoder.levels != encoder.levels:
            print(
                "WARN: encoder levels in corpus differ from preprocessor.json — "
                "model may be stale relative to the current corpus."
            )
        feature_cols = list(ds.feature_names)
        test = ds.materialize_test()

    x_test = test.x
    y_test = test.y
    implied_test = test.implied_prob
    top_categories = test.top_categories.tolist()
    print(f"Test split: {x_test.shape[0]:,} rows, {x_test.shape[1]} columns")
    print(f"Feature matrix: {x_test.shape}")
```

Update imports at the top of `scripts/analyze_model.py`:

```python
from pscanner.ml.preprocessing import OneHotEncoder
from pscanner.ml.streaming import open_dataset
```

Drop the now-unused imports: `load_dataset`, `drop_leakage_cols`, `temporal_split`, `CARRIER_COLS`, `build_feature_matrix` (these are no longer called in the script).

- [ ] **Step 2: Smoke-test the script (manual)**

This script doesn't have unit tests. Run it against any model artifact + the synthetic-fixture DB (manually, not as an automated test). Skip if no artifact handy.

```bash
# Example — adjust to your local model dir
uv run python scripts/analyze_model.py --model models/last-run --db data/corpus.sqlite3 --top-k 5
```

Expected: the script runs and prints metrics. It SHOULD warn about encoder drift if the corpus has new categorical levels since the model was trained — that warning is informational, not fatal.

- [ ] **Step 3: Lint + type-check**

```bash
uv run ruff check scripts/analyze_model.py
uv run ruff format --check scripts/analyze_model.py
uv run ty check
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_model.py
git commit -m "$(cat <<'EOF'
refactor(scripts): analyze_model uses open_dataset (#39)

Replaces the load_dataset + drop_leakage_cols + temporal_split block
with `with open_dataset(db_path) as ds: test = ds.materialize_test()`.
Adds a sanity-check warning when the corpus's encoder levels diverge
from the model's preprocessor.json (model staleness signal).
EOF
)"
```

---

## Task 14: Delete `load_dataset` + `temporal_split` + migrate the old tests

Cleanup commit. After this, the eager pipeline is gone.

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py`
- Modify: `tests/ml/test_preprocessing.py`
- Delete: `tests/ml/_capture_eager_baseline.py`

- [ ] **Step 1: Remove the four `test_load_dataset_*` cases from `test_preprocessing.py`**

In `tests/ml/test_preprocessing.py`, find and delete:
- `test_load_dataset_joins_resolved_at`
- `test_load_dataset_excludes_leakage_cols`
- `test_load_dataset_casts_low_cardinality_columns_to_categorical`
- `test_load_dataset_casts_numeric_columns_to_int32_float32`

Plus any `test_temporal_split_*` cases.

Drop the `from pscanner.ml.preprocessing import load_dataset, temporal_split, Split` imports (or trim down whichever symbols are gone).

- [ ] **Step 2: Run the trimmed test suite, see it stay green**

```bash
uv run pytest -q tests/ml/
```

Expected: all remaining ml tests pass; the four `test_load_dataset_*` cases are gone.

- [ ] **Step 3: Delete `load_dataset`, `temporal_split`, `Split` from `preprocessing.py`**

In `src/pscanner/ml/preprocessing.py`, delete:
- `load_dataset` (function + its docstring)
- `temporal_split` (function + its docstring)
- `Split` (the dataclass)

Keep:
- `OneHotEncoder` class (still used by streaming.py)
- `drop_leakage_cols` (still used by `_SplitIter`)
- `build_feature_matrix` (still used by `_SplitIter`)
- The constants: `LEAKAGE_COLS`, `CARRIER_COLS`, `CATEGORICAL_COLS`, `_NEVER_LOAD_COLS`, `_INT32_COLS`, `_FLOAT32_COLS`, `_CATEGORICAL_CAST_COLS`, `_NONE_TOKEN`.

Update the module docstring at the top of `preprocessing.py` to drop references to `load_dataset` and `temporal_split`. The exports list should now mention `OneHotEncoder`, `drop_leakage_cols`, `build_feature_matrix`, and the constants only.

- [ ] **Step 4: Delete the eager-capture script**

```bash
rm tests/ml/_capture_eager_baseline.py
```

The `eager_baseline.json` file STAYS — `test_streaming_pipeline_matches_eager_baseline` reads from it. The script that generates it is gone (the eager path is gone), and that's fine: the JSON is now a frozen contract.

- [ ] **Step 5: Run the full ml suite**

```bash
uv run pytest -q tests/ml/
```

Expected: all green. The streaming-vs-eager parity test still passes (it reads the JSON, not the eager code).

- [ ] **Step 6: Lint + type-check + full project verify**

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```

Expected: clean. No project-wide regressions; ml tests all pass; the rest of the suite is unaffected.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git rm tests/ml/_capture_eager_baseline.py
git commit -m "$(cat <<'EOF'
refactor(ml): delete load_dataset + temporal_split + Split (#39)

The eager full-corpus loader is replaced by streaming.open_dataset.
analyze_model.py + run_study + the four test_load_dataset_* cases
already migrated; this commit removes the dead code.

The eager_baseline.json snapshot stays — the streaming-vs-eager
parity test reads it as a frozen contract. The capture script
(which depended on the now-deleted load_dataset) is removed.
EOF
)"
```

---

## Verification checklist (before opening the PR)

- [ ] `uv run ruff check .` clean
- [ ] `uv run ruff format --check .` clean
- [ ] `uv run ty check` — same 13 pre-existing diagnostics in `tests/corpus/` as `main`; nothing new
- [ ] `uv run pytest -q` — full suite green (1072+ tests passing)
- [ ] All 15 streaming tests pass: `uv run pytest -q tests/ml/test_streaming.py`
- [ ] `tests/ml/data/eager_baseline.json` exists and is checked in
- [ ] Operational verification on the desktop: `pscanner ml train --device cuda --n-jobs 1 --n-trials 5 --db data/corpus.sqlite3 --output-dir /tmp/ml-streaming-after` completes WITHOUT OOM. `grep ml.mem /tmp/ml-streaming-after.log` shows the new `post_pre_pass` phase plus all phases from #67. Capture this for the PR description.
- [ ] `scripts/analyze_model.py` runs against a previously-trained model and produces the same headline metrics as the pre-streaming version (modulo the warning if encoder levels drifted).

---

## PR shape

Single PR with 14 commits (Task 1 through Task 14) + the plan-doc commit + the spec-doc commit. Title: `feat(ml): streaming corpus load + DataIter (#39)`. Body cites the operational result from the desktop verification, the new `ml.mem` phase shape, and the parity-test tolerance. References #67 as the predecessor lifetime fix.
