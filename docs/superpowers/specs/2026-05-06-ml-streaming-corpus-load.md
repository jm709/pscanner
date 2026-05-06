# ML Streaming Corpus Load — Design

**Status:** Approved (brainstormed 2026-05-06).
**Tracking issue:** [#39](https://github.com/jm709/pscanner/issues/39).
**Predecessor:** #67 (training-pipeline lifetime trims). #67 reduced Optuna-phase RSS but operational verification on the desktop's 32 GB corpus showed the dominant peak is `pl.read_database` at preprocessing — before any `_log.info("ml.mem", ...)` fires. This spec replaces the eager full-corpus load with a streaming + per-split chunked architecture so training can run end-to-end on the full corpus inside a 12 GB host.

## Motivation

`pscanner ml train` currently calls `pscanner.ml.preprocessing.load_dataset(db_path)` which executes a single `pl.read_database` over the full `training_examples ⨝ market_resolutions` join. The result is a Polars DataFrame holding every row in memory at once. Operational measurement on `feat/issue-67-training-memory` against the desktop's 32 GB corpus produced an OOM kill at anon-rss 11.8 GB inside a 12 GB WSL2 instance — before any of #67's lifetime trims could fire. The eager pipeline is incompatible with corpora larger than ~5 M rows on a 12 GB host and ~10 M rows on a 16 GB host.

Issue #39 originally proposed wrapping `xgb.DataIter` around the existing in-memory feature matrices. That addresses the DMatrix copy but not the upstream materialization, which is the actual bottleneck. This spec expands #39's scope to cover the full data path from SQLite → DMatrix.

## Goals

1. Eliminate the full-corpus Polars DataFrame from the training pipeline. Peak RSS during `run_study` stays ≤ 10 GB on the desktop's 12 GB WSL2 instance, leaving ≥ 2 GB of headroom for the kernel + concurrent processes.
2. Preserve the existing temporal split semantics (markets sorted by `resolved_at`, sliced into train/val/test percentiles) and the existing `OneHotEncoder.fit` semantics (fit on train levels only).
3. Preserve the existing determinism guarantees: identical seeds produce identical `test_edge` within ≤ 0.001 absolute on the synthetic test fixture (per #39's stated DoD).
4. Keep `evaluate_on_test` unchanged so analysis tooling (`scripts/analyze_model.py`) and metrics computation continue to consume materialized numpy arrays.

## Non-goals

- Changing the schema of `training_examples` or `market_resolutions`. Pre-computing split assignments into a column was considered and rejected — it would couple feature build to training and force `pscanner corpus build-features` reruns whenever the split fraction changes.
- Improving training speed. The streaming path's per-chunk Polars conversion adds modest CPU overhead. This is acceptable; memory headroom is the constraint.
- Cross-platform support. The streaming reads target Polymarket's `training_examples` table only. Multi-platform corpora (Kalshi/Manifold) are out of scope per the multi-platform RFC's current Stage 1 framing.
- Resume / checkpoint training. Each `pscanner ml train` invocation rebuilds DMatrices from scratch; chunked iterators do not persist state across invocations.

## Architecture

### Public API surface

A new module `pscanner.ml.streaming` exposes one entry point and one handle:

```python
@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[StreamingDataset]: ...

@dataclass(frozen=True)
class TestSplit:
    """Materialized test split for evaluate_on_test + analyze_model.py."""
    x: np.ndarray              # float32 feature matrix, shape (n_test_rows, n_features)
    y: np.ndarray              # int8 labels, shape (n_test_rows,)
    implied_prob: np.ndarray   # float32, shape (n_test_rows,)
    top_categories: np.ndarray # object (str), unencoded — for per-category breakdowns


class StreamingDataset:
    """Two-pass streaming view over training_examples.

    Constructed via ``open_dataset``. Pre-scans (split partition + encoder
    fit + per-split row counts) execute at ``__enter__``. Per-split chunked
    reads are deferred until ``dtrain``/``dval``/``materialize_test`` is
    called. SQLite connections opened by the pre-pass and by each split
    iterator are closed at ``__exit__``.
    """

    encoder: OneHotEncoder              # fit on train levels during __enter__
    feature_names: tuple[str, ...]      # post-encoding column order, drops CARRIER_COLS + label_won
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int

    def dtrain(self, *, device: str) -> xgb.QuantileDMatrix: ...
    def dval(self, *, device: str) -> xgb.QuantileDMatrix: ...
    def val_aux(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_val, implied_prob_val) — only the auxiliary arrays the
        edge metric closure needs. No feature matrix is materialized."""
    def materialize_test(self) -> TestSplit: ...
```

Caller pattern in `run_study`:

```python
with open_dataset(db_path, chunk_size=chunk_size) as ds:
    encoder = ds.encoder
    dtrain = ds.dtrain(device=device)
    dval = ds.dval(device=device)
    y_val, implied_val = ds.val_aux()
    _log.info("ml.mem", phase="post_dmatrix", rss_mb=_rss_mb())

    best_iteration, best_params, best_value = _run_optimization_phase(
        dtrain=dtrain, dval=dval, y_val=y_val, implied_val=implied_val, ...
    )
    booster = fit_winning_model(
        best_params=best_params, best_iteration=best_iteration, dtrain=dtrain, ...
    )
    del dtrain, dval

    test = ds.materialize_test()
    test_metrics = evaluate_on_test(
        booster, test.x, test.y, test.implied_prob, n_min,
        top_category_test=test.top_categories,
        accepted_categories=resolved_categories,
    )
```

Caller pattern in `scripts/analyze_model.py`:

```python
with open_dataset(db_path) as ds:
    test = ds.materialize_test()
    # uses ds.encoder, ds.feature_names, test.x/y/implied_prob/top_categories
```

### Two-pass internals

Pre-pass executes three queries inside `open_dataset.__enter__`:

**P1 — Temporal split partition.** Replaces `temporal_split(df)`:

```sql
SELECT condition_id, resolved_at
FROM market_resolutions
ORDER BY resolved_at, condition_id
```

Returns one row per resolved market (~1 K – 10 K rows). Sorted in Python, sliced into train (60%), val (20%), test (20%) by row index. Returns three `frozenset[str]` of `condition_id`s. Negligible memory cost.

**P2 — Encoder fit on train levels.** Replaces `OneHotEncoder.fit(splits.train, columns=CATEGORICAL_COLS)`:

```sql
SELECT DISTINCT side, top_category, market_category
FROM training_examples te
JOIN train_markets tm USING (condition_id)
```

Where `train_markets` is a `TEMP TABLE` populated with the train split's `condition_id`s (see "IN-clause mitigation" below). Returns ~50 distinct rows. Levels per column are extracted via Polars + `OneHotEncoder.fit` on the resulting tiny DataFrame. Same fit semantics as the eager path — unseen test/val levels still map to all-zeros at transform time.

**P3 — Per-split row counts.** Three small queries indexed by `idx_training_examples_condition`:

```sql
SELECT COUNT(*) FROM training_examples te
JOIN <split>_markets sm USING (condition_id)
```

Used to size pre-allocated numpy arrays in `val_aux()` and `materialize_test()` (`np.empty(n_rows, dtype=...)` rather than growing on append).

After P1–P3, the dataset handle stores: the three split frozensets, the fitted encoder, the feature-names tuple, the three row counts, and the path/handle state needed for the per-split iterators.

### Per-split chunked read

Each chunk iterator is a private class:

```python
class _SplitIter:
    """Yields (x_chunk, y_chunk, implied_chunk) numpy tuples per chunk.

    Each chunk is ~chunk_size rows × n_features float32. At chunk_size=100_000
    and n_features≈60, that's ~24 MB per chunk on the wire and ~24 MB peak
    in flight (XGBoost's release_data=True frees each chunk after ingestion).
    """

    def __init__(
        self,
        db_path: Path,
        condition_ids: frozenset[str],
        encoder: OneHotEncoder,
        feature_names: tuple[str, ...],
        chunk_size: int,
    ) -> None: ...

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]: ...
```

Iteration body:

1. Open a fresh `sqlite3.Connection`. XGBoost's `DataIter` may iterate from worker threads; Python's `sqlite3.Connection` is not safe to share across threads, so each iterator owns its own.
2. Create and populate a per-connection temp table:
   ```sql
   CREATE TEMP TABLE split_markets (condition_id TEXT PRIMARY KEY)
   ```
   Bulk-insert via `executemany`. Cost: ~10 K INSERTs at iter start, < 100 ms one-time.
3. Issue the chunked SELECT joining on the temp table:
   ```sql
   SELECT <kept_cols>, mr.resolved_at
   FROM training_examples te
   JOIN market_resolutions mr USING (condition_id)
   JOIN split_markets sm USING (condition_id)
   ORDER BY te.id
   ```
   The `<kept_cols>` list is computed once from `PRAGMA table_info(training_examples)` minus `_NEVER_LOAD_COLS` (preserved from the existing `load_dataset`). The `ORDER BY te.id` clause anchors chunk order so determinism survives across runs.
4. `cursor.fetchmany(chunk_size)` per chunk → build a small Polars DataFrame from the rows → apply dtype casting (`_INT32_COLS`, `_FLOAT32_COLS`, `_CATEGORICAL_CAST_COLS`) → `drop_leakage_cols()` (idempotent, drops only what's present) → `encoder.transform()` → `build_feature_matrix()` → emit the `(x, y, implied)` numpy tuple.
5. After the last chunk, close the cursor and the connection.

### `SplitDataIter` — XGBoost adapter

Wraps `_SplitIter` with the `xgb.DataIter` protocol:

```python
class SplitDataIter(xgb.DataIter):
    def __init__(self, source: _SplitIter) -> None:
        super().__init__(release_data=True)
        self._source = source
        self._iter: Iterator | None = None

    def next(self, input_data: Callable[..., None]) -> bool:
        if self._iter is None:
            self._iter = iter(self._source)
        try:
            x, y, _implied = next(self._iter)
        except StopIteration:
            return False
        input_data(data=x, label=y)
        return True

    def reset(self) -> None:
        self._iter = None  # next call to next() reopens the cursor
```

`release_data=True` is the linchpin — XGBoost frees each ingested chunk before the next one is requested. Peak in-flight working set per split is one chunk plus XGBoost's internal quantization buffer.

### DMatrix construction

Both `dtrain()` and `dval()` return `xgb.QuantileDMatrix`:

```python
def dtrain(self, *, device: str) -> xgb.QuantileDMatrix:
    source = _SplitIter(
        self._db_path, self._train_markets, self.encoder,
        self.feature_names, self._chunk_size,
    )
    return xgb.QuantileDMatrix(
        SplitDataIter(source),
        max_bin=256,
        feature_names=list(self.feature_names),
        device=device,
    )
```

`QuantileDMatrix` is GPU-native (no host-side dense array materialization on `device="cuda"`) and works on CPU `tree_method=hist` (XGBoost 3.x default). Quantization is fixed at 256 bins to match the eager path's default.

`val_aux()` uses a different chunked SELECT that pulls only `label_won` and `implied_prob_at_buy` columns. It pre-allocates `np.empty(n_val_rows, dtype=...)` arrays from P3's count and fills them per chunk.

`materialize_test()` runs the same `_SplitIter` as `dtrain`/`dval` but accumulates chunks into pre-allocated `np.empty((n_test_rows, n_features), dtype=np.float32)` arrays. Also pulls the unencoded `top_category` column via a parallel small SELECT for `TestSplit.top_categories`.

## Determinism

The eager path used `xgb.DMatrix(numpy_array, ...)` with default 256-bin quantization at training time (`tree_method=hist` is XGBoost 3.x's default). The new path uses `xgb.QuantileDMatrix(SplitDataIter(...), max_bin=256, ...)` — same quantization, just performed at construction instead of per-tree.

Three sources of determinism:

1. **Split partition.** P1's `ORDER BY resolved_at, condition_id` produces the same row order from the same DB state.
2. **Encoder fit.** P2's `SELECT DISTINCT` produces the same level set from the same train markets.
3. **Chunk order.** Each `_SplitIter`'s SELECT uses `ORDER BY te.id`. Since `id` is `INTEGER PRIMARY KEY AUTOINCREMENT`, chunk arrival order is stable across runs given a fixed corpus.

The tested invariant per #39's DoD: `test_edge` matches the eager path within ≤ 0.001 absolute on the synthetic test fixture.

## Migration scope

**Files modified:**

- `src/pscanner/ml/preprocessing.py` — delete `load_dataset`, `temporal_split`, `Split`. Keep `OneHotEncoder`, `drop_leakage_cols`, `build_feature_matrix`, `LEAKAGE_COLS`, `CARRIER_COLS`, `CATEGORICAL_COLS`, `_NEVER_LOAD_COLS`, the dtype-cast tuples (`_INT32_COLS`, `_FLOAT32_COLS`, `_CATEGORICAL_CAST_COLS`).
- `src/pscanner/ml/streaming.py` — new file. Contains `open_dataset`, `StreamingDataset`, `TestSplit`, `_SplitIter`, `SplitDataIter`.
- `src/pscanner/ml/training.py` — `run_study` signature changes from `df: pl.DataFrame` to `db_path: Path`. The preprocessing block is replaced by `with open_dataset(db_path, chunk_size=chunk_size) as ds:`. Drop `_extract_top_category` (replaced by `TestSplit.top_categories`). Preserve the `del dtrain, dval` and `gc.collect()` lifetime trims from #67.
- `src/pscanner/ml/cli.py` — add `--chunk-size` argument (default `100_000`). `run_study` is called with `db_path` directly; the `df = load_dataset(db_path)` line is removed.
- `scripts/analyze_model.py` — replace `load_dataset` + `drop_leakage_cols` + `temporal_split` block with `with open_dataset(db_path) as ds: test = ds.materialize_test()`. Use `ds.encoder`, `ds.feature_names`, and the four `TestSplit` fields.
- `tests/ml/test_streaming.py` — new file. Contains the seven streaming tests (see "Test strategy").
- `tests/ml/test_preprocessing.py` — remove the four `test_load_dataset_*` tests (migrated to `test_streaming.py`) and any `test_temporal_split_*` tests. Tests of `OneHotEncoder`, `drop_leakage_cols`, `build_feature_matrix` stay.
- `tests/ml/test_training.py` — `test_run_study_*` cases switch from `df=make_synthetic_examples(...)` to `db_path=...`.
- `tests/ml/conftest.py` — add `make_synthetic_examples_db(tmp_path)` fixture returning a `Path` to a populated SQLite file. The existing `pl.DataFrame` fixture stays for per-component tests.

**No backward-compat shim.** Per the project rule "Replace, don't deprecate" (CLAUDE.md), `load_dataset`, `temporal_split`, and `Split` are deleted in the same PR that lands `streaming.py`. All call sites migrate in one diff.

## Test strategy

Seven new tests in `tests/ml/test_streaming.py`:

1. **`test_open_dataset_partitions_markets_by_resolved_at`** — synthetic 20-market DB. Compute the expected partition in-test by sorting markets on `resolved_at` and slicing at 60% / 80%. Assert `ds._train_markets`, `ds._val_markets`, `ds._test_markets` (exposed via test-only accessors or read directly) match. Regression guard for the migration off `temporal_split`.
2. **`test_encoder_fits_on_train_levels_only`** — synthetic DB where `top_category` has level X only in val/test; assert `ds.encoder.levels["top_category"]` does not contain X (matches current `OneHotEncoder.fit` semantics).
3. **`test_dtrain_returns_quantile_dmatrix_with_expected_shape`** — synthetic DB; build dtrain on CPU; assert `dtrain.num_row() == ds.n_train_rows` and `dtrain.num_col() == len(ds.feature_names)`.
4. **`test_chunk_size_respected`** — instrumented `_SplitIter` to count chunks; chunk_size=50 over 200 rows yields exactly 4 chunks. Confirms iterator boundaries.
5. **`test_streaming_matches_eager_path`** — run training on a small synthetic corpus with the new streaming path. Compare `test_edge` against a snapshot taken from the pre-streaming code path on the same fixture. Assert within `0.001` absolute (per #39's DoD). Snapshot is checked in alongside the test.
6. **`test_materialize_test_returns_unencoded_top_categories`** — assert `test.top_categories` is the raw category strings (not one-hot encoded) and parallel to `test.y`. Required for `analyze_model.py`'s per-category breakdown.
7. **`test_open_dataset_closes_connection_on_exit`** — `__exit__` closes the SQLite connection used by the pre-pass; iterator connections close on iterator exhaustion or `__exit__`.

Existing integration tests carry over without modification (other than the `df` → `db_path` swap):

- `test_run_study_writes_all_artifacts` — full-pipeline smoke against the synthetic DB.
- `test_run_study_is_deterministic_under_same_seed` — determinism canary; DataIter must produce stable chunk ordering.

## Observability

`run_study`'s `ml.mem` log phases shift slightly:

| Phase | Pre-streaming meaning | Post-streaming meaning |
|---|---|---|
| `run_study_entry` | Same | Same |
| `post_pre_pass` | (didn't exist) | New — RSS after `open_dataset.__enter__` finishes (split partition + encoder fit + counts). Should be ~baseline + a few MB. |
| `post_encoder_fit` | After `OneHotEncoder.fit` on full train DataFrame | Removed — now part of `post_pre_pass`. |
| `post_build_feature_matrix` | After all 3 numpy matrices built | Removed — there is no eager 3-matrix moment. |
| `pre_optuna` | After source-array release, before Optuna | After `dtrain`/`dval` `QuantileDMatrix` construction. The `release_data=True` flag means each chunk is freed by XGBoost during construction; peak should be one chunk + XGBoost's quantization buffer. |
| `post_optuna` | After Optuna returns | Same |
| `post_fit_winning` | After `fit_winning_model` | Same |

The new `post_pre_pass` phase is the canary for "did streaming actually work?" If RSS at `post_pre_pass` is multi-GB, the pre-pass query did not stream and we have a regression.

## Risks

1. **`release_data=True` hidden state.** XGBoost frees each chunk after ingestion, so any consumer that re-iterates without `reset()` would fail. Covered by integration test #5 (streaming matches eager) end-to-end.
2. **SQLite connection leak in worker threads.** XGBoost's DataIter may be called from worker threads. Each chunk iterator opens its own connection; we close it on iterator exhaustion. If XGBoost crashes mid-iteration, the connection leaks until Python shutdown. The `with open_dataset(...)` `__exit__` closes any tracked-but-not-yet-closed iterator connections as a last-resort cleanup. Acceptable failure mode — leaks file descriptors transiently, no data corruption.
3. **`IN`-clause parameter limit.** ~~Mitigated~~ pre-mitigated. The design uses a per-connection `TEMP TABLE` populated via `executemany` instead of an `IN (?, ?, ...)` parameterized query. Removes SQLite's `SQLITE_MAX_VARIABLE_NUMBER` ceiling from the design entirely.
4. **Polars per-chunk overhead.** Converting each `chunk_size`-row `sqlite3` batch into Polars and through `encoder.transform()` + `build_feature_matrix()` costs CPU per chunk. At `chunk_size=100_000` over a 10 M-row corpus, that's ~100 chunks per training run — bounded overhead. The `--chunk-size` CLI flag is the lever if profiling later shows it matters.
5. **`run_study` signature change.** `run_study(df: pl.DataFrame, ...)` → `run_study(db_path: Path, ...)` breaks any external caller. The codebase has one caller (`pscanner.ml.cli`); blast radius is local. `scripts/analyze_model.py` uses `open_dataset` directly, not `run_study`.

## Definition of done

- `pscanner ml train --device cuda --n-jobs 1` runs end-to-end on a corpus that previously OOMed at load. Operational target: the desktop's 32 GB SQLite at < 10 GB peak RSS on the 12 GB WSL2 instance.
- `test_edge` from a streaming run matches the eager-path snapshot within ≤ 0.001 absolute on `tests/ml/test_streaming.py::test_streaming_matches_eager_path`.
- All seven new streaming tests pass; the four `test_load_dataset_*` cases are removed; the two `test_run_study_*` cases pass with the `db_path` migration.
- `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q` clean.
- `scripts/analyze_model.py` runs against the merged-with-#39 codebase and produces the same headline metrics as the pre-streaming version on the same model artifact.
- `--chunk-size` exposed on the `pscanner ml train` CLI with default `100_000`.
