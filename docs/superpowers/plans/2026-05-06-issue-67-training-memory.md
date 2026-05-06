# Issue #67: Training Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce peak RSS during `pscanner ml train` so the laptop dev host (7.6 GB) stops OOMing and the desktop's 8 GB VRAM is no longer brushed by avoidable working-set overhead.

**Architecture:** Pure lifetime-tightening refactor of `src/pscanner/ml/training.py` and adjacent tests. No new modules, no new dependencies. Three independent fixes converge on the same end-state: (a) splits are torn down one at a time during preprocessing; (b) DMatrices are constructed in `run_study` so the source numpy arrays can be released before the 100-trial Optuna phase; (c) Optuna's per-trial RDB reload is replaced with in-memory storage. `fit_winning_model` shifts to a DMatrix input so the training arrays don't need to survive across the optimization phase.

**Tech Stack:** Python 3.13, Polars, XGBoost, Optuna, pytest. Verify locally with `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q tests/ml/`.

---

## Scope notes

- **Branch / worktree:** This plan should run in a dedicated worktree off `main`. The current laptop checkout is on `feat/issue-46-phase3-subgraph-backfill` (an unrelated in-flight branch). Suggested setup: `wt switch -c feat/issue-67-training-memory` (or `git worktree add ../pscanner-worktrees/issue-67 -b feat/issue-67-training-memory main`).
- **Desktop is off-limits:** `build-features` is running on the desktop via SSH. Do NOT rsync code to it during this work or trigger remote training. All verification stays on the laptop.
- **T4 was misidentified.** Issue #67 lists "T4. Dead `wallet_address` reference in `_CATEGORICAL_CAST_COLS`" — verified false. `_CATEGORICAL_CAST_COLS` (`src/pscanner/ml/preprocessing.py:31-36`) contains only `condition_id, top_category, market_category, side`. `wallet_address` lives in `LEAKAGE_COLS` (`:80-86`) and is correctly excluded via `_NEVER_LOAD_COLS` at SELECT time. Task 6 includes a correction comment on the issue. **No code change needed for T4.**
- **T2's framing in the issue is partially wrong.** The issue suggests `del x_train, x_val, y_train` immediately after `dval = xgb.DMatrix(...)` inside `_run_optimization_phase`. That `del` only drops the local function-frame reference; the outer `run_study` frame still holds the same arrays alive (it needs `x_train`/`y_train` for `fit_winning_model` at line 443). The real fix requires moving DMatrix construction to `run_study` and refactoring `fit_winning_model` to accept `dtrain` directly. Task 3 + Task 4 implement this correctly.
- **Test environment:** all tests below run on `tmp_path` with synthetic data (`make_synthetic_examples` fixture in `tests/ml/conftest.py`). No GPU required; CPU XGBoost works for every test. The existing `test_run_study_writes_all_artifacts` (`tests/ml/test_training.py:199`) is the end-to-end smoke that catches regressions in `run_study`.

---

## File map

| Path | Change |
|---|---|
| `src/pscanner/ml/training.py` | Modify: `fit_winning_model` signature, `run_study` body, `_run_optimization_phase` signature + body |
| `tests/ml/test_training.py` | Modify: `test_fit_winning_model_*`, `test_evaluate_on_test_*`, `_toy_booster`, `test_run_study_writes_all_artifacts` (drop `study.db` assertion) |
| (none) | No new files. |

---

## Task 1: Switch Optuna to InMemoryStorage (T3)

Smallest, most isolated change. Land first to keep the diff narrow.

**Files:**
- Modify: `src/pscanner/ml/training.py:296-332`
- Modify: `tests/ml/test_training.py:215` (drop the `study.db` existence assertion; the file no longer exists on disk)

- [ ] **Step 1: Update the existing artifact-existence test to reflect the new (no `study.db`) reality**

In `tests/ml/test_training.py`, find `test_run_study_writes_all_artifacts` (~line 199). Remove the line:

```python
assert (output_dir / "study.db").exists()
```

Add an explicit assertion that `study.db` is *not* written:

```python
assert not (output_dir / "study.db").exists(), (
    "study.db must not be written — InMemoryStorage replaced RDBStorage to "
    "avoid the per-trial reload of full study state from SQLite"
)
```

Also update `test_run_study_n_jobs_2_completes_without_lock_errors` (~line 238) — it currently asserts `study.db` exists; replace with an assertion that `metrics.json` exists (already there, just remove the `study.db` line).

- [ ] **Step 2: Run the changed tests, see them fail**

```bash
uv run pytest tests/ml/test_training.py::test_run_study_writes_all_artifacts tests/ml/test_training.py::test_run_study_n_jobs_2_completes_without_lock_errors -v
```

Expected: both fail with `AssertionError: study.db must not be written` because the production code still uses RDBStorage, which DOES write `study.db`.

- [ ] **Step 3: Replace RDBStorage with InMemoryStorage**

In `src/pscanner/ml/training.py`, replace lines 296-332 of `_run_optimization_phase`:

```python
    storage_url = f"sqlite:///{output_dir / 'study.db'}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    # Silence optuna's per-trial chatter on stderr while the test suite
    # is running with filterwarnings=error.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(),
            storage=storage,
            study_name="copy_trade_gate",
        )
        study.optimize(
            lambda t: run_single_trial(
                trial=t,
                dtrain=dtrain,
                dval=dval,
                y_val=y_val,
                implied_prob_val=implied_val,
                n_min=n_min,
                seed=seed,
                device=device,
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        best_iteration = int(study.best_trial.user_attrs["best_iteration"])
        best_params = dict(study.best_params)
        best_value = float(study.best_value)
    finally:
        # Release SQLAlchemy connection pool so the SQLite file isn't
        # left open across pytest teardown (would trip filterwarnings=error
        # via ResourceWarning -> PytestUnraisableExceptionWarning).
        storage.remove_session()
        storage.engine.dispose()

    return best_iteration, best_params, best_value
```

with:

```python
    # InMemoryStorage avoids the per-trial reload of the full study history
    # that RDBStorage(SQLite) does on every TPESampler.sample(). Each `run`
    # uses a fresh output_dir; resume isn't a documented feature here, so
    # there's no reason to leave a study.db artifact on disk.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
        storage=optuna.storages.InMemoryStorage(),
        study_name="copy_trade_gate",
    )
    study.optimize(
        lambda t: run_single_trial(
            trial=t,
            dtrain=dtrain,
            dval=dval,
            y_val=y_val,
            implied_prob_val=implied_val,
            n_min=n_min,
            seed=seed,
            device=device,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_iteration = int(study.best_trial.user_attrs["best_iteration"])
    best_params = dict(study.best_params)
    best_value = float(study.best_value)
    return best_iteration, best_params, best_value
```

- [ ] **Step 4: Run the changed tests, see them pass**

```bash
uv run pytest tests/ml/test_training.py -v
```

Expected: all green. Pay attention to `test_run_study_is_deterministic_under_same_seed` — the determinism guarantee relies on `TPESampler(seed=seed)`, which is independent of storage backend, so it must still pass.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/pscanner/ml/training.py tests/ml/test_training.py
uv run ruff format --check src/pscanner/ml/training.py tests/ml/test_training.py
uv run ty check
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/training.py tests/ml/test_training.py
git commit -m "$(cat <<'EOF'
perf(ml): use Optuna InMemoryStorage to skip per-trial SQLite reload

RDBStorage(SQLite) reloads the full study history on every TPE sample.
The training command writes to a fresh output_dir per run and never
resumes a prior study, so the on-disk study.db was pure overhead.
Drops the SQLAlchemy session-cleanup branch with it.
EOF
)"
```

---

## Task 2: Per-split frame teardown (T1)

Refactor the preprocessing block in `run_study` so each split's encoded DataFrame is released as soon as its numpy matrices are extracted. Eliminates the simultaneous coexistence of three Polars frames + three numpy matrices.

**Files:**
- Modify: `src/pscanner/ml/training.py:395-424` (the preprocessing block of `run_study`)

- [ ] **Step 1: No new test — existing `test_run_study_writes_all_artifacts` covers this end-to-end**

The existing test (`tests/ml/test_training.py:199`) builds a synthetic 20-market × 15-row DataFrame, runs the full pipeline, and checks every output artifact. If the per-split refactor breaks split semantics (rates per split, encoder fit, etc.), this test fails. Verify the test runs green against the current code first:

```bash
uv run pytest tests/ml/test_training.py::test_run_study_writes_all_artifacts tests/ml/test_training.py::test_run_study_is_deterministic_under_same_seed -v
```

Expected: both pass on the post-Task-1 codebase.

- [ ] **Step 2: Replace the preprocessing block**

In `src/pscanner/ml/training.py`, replace lines 395-424 (the block from `df = drop_leakage_cols(df)` through `_log.info("ml.mem", phase="post_polars_release", ...)`):

```python
    df = drop_leakage_cols(df)
    splits = temporal_split(df)
    encoder = OneHotEncoder.fit(splits.train, columns=CATEGORICAL_COLS)
    train_df = encoder.transform(splits.train)
    val_df = encoder.transform(splits.val)
    test_df = encoder.transform(splits.test)
    _log.info("ml.mem", phase="post_split_and_encode", rss_mb=_rss_mb())

    x_train, y_train, _ = build_feature_matrix(train_df)
    x_val, y_val, implied_val = build_feature_matrix(val_df)
    x_test, y_test, implied_test = build_feature_matrix(test_df)
    top_category_test = _extract_top_category(splits.test)
    _log.info("ml.mem", phase="post_build_feature_matrix", rss_mb=_rss_mb())

    rates = {
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    }
    _log.info("ml.split_label_won_rate", **rates)

    # Polars frames are no longer needed once the numpy matrices are
    # extracted; release ~3-4 GB before the optuna phase allocates
    # DMatrix copies.  Explicit gc.collect() because Python's cyclic
    # collector doesn't always reclaim Polars/Arrow buffers promptly on
    # its own — post_polars_release was previously identical to
    # post_build_feature_matrix without it.
    del df, splits, train_df, val_df, test_df
    gc.collect()
    _log.info("ml.mem", phase="post_polars_release", rss_mb=_rss_mb())
```

with:

```python
    df = drop_leakage_cols(df)
    splits = temporal_split(df)
    encoder = OneHotEncoder.fit(splits.train, columns=CATEGORICAL_COLS)
    _log.info("ml.mem", phase="post_encoder_fit", rss_mb=_rss_mb())

    # Process splits one at a time. Each encoded Polars frame is released
    # as soon as its numpy matrices are extracted, so we never hold all
    # three encoded frames + all three numpy matrices simultaneously.
    train_df = encoder.transform(splits.train)
    x_train, y_train, _ = build_feature_matrix(train_df)
    del train_df
    gc.collect()

    val_df = encoder.transform(splits.val)
    x_val, y_val, implied_val = build_feature_matrix(val_df)
    del val_df
    gc.collect()

    test_df = encoder.transform(splits.test)
    x_test, y_test, implied_test = build_feature_matrix(test_df)
    top_category_test = _extract_top_category(splits.test)
    del test_df, df, splits
    gc.collect()
    _log.info("ml.mem", phase="post_build_feature_matrix", rss_mb=_rss_mb())

    rates = {
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    }
    _log.info("ml.split_label_won_rate", **rates)
```

Note: the `post_polars_release` log line is removed — it now coincides with `post_build_feature_matrix` since the frame teardown happens inline. The `post_encoder_fit` log is added so the encoder-fit RSS is observable separately from the per-split work.

- [ ] **Step 3: Run the test suite**

```bash
uv run pytest tests/ml/ -v
```

Expected: all green. Pay particular attention to `test_run_study_is_deterministic_under_same_seed` — the per-split GC introduces no determinism risk (allocator non-determinism doesn't affect numpy values), but it's the canary.

- [ ] **Step 4: Lint + type-check**

```bash
uv run ruff check src/pscanner/ml/training.py
uv run ruff format --check src/pscanner/ml/training.py
uv run ty check
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/training.py
git commit -m "$(cat <<'EOF'
perf(ml): tear down each encoded split as soon as its arrays are extracted

The previous run_study held train_df, val_df, test_df, and all six
numpy arrays simultaneously between build_feature_matrix and the
trailing del. With ~70 cols × float32 over a 16M-row corpus that's
several GB of avoidable peak RSS on top of the steady state.

Process splits one at a time; del + gc.collect() between each. The
post_polars_release log line collapses into post_build_feature_matrix
now that the teardown happens inline; a new post_encoder_fit log
isolates the encoder-fit RSS from the per-split work.
EOF
)"
```

---

## Task 3: Refactor `fit_winning_model` to accept `dtrain`

Change `fit_winning_model`'s signature from `(X_train, y_train, ...)` to `(dtrain, ...)` so the winning-model refit can reuse the DMatrix built for Optuna instead of building a new one from raw numpy arrays. Cascading test updates land in the same commit.

**Files:**
- Modify: `src/pscanner/ml/training.py:157-195` (`fit_winning_model` definition)
- Modify: `src/pscanner/ml/training.py:440-447` (the `fit_winning_model` call inside `run_study` — this becomes part of Task 4 but the call site is updated here for type consistency)
- Modify: `tests/ml/test_training.py` — five sites: `test_fit_winning_model_returns_booster_with_expected_iterations` (~148), `test_evaluate_on_test_returns_metric_dict` (~171), `_toy_booster` (~256), `test_evaluate_on_test_returns_edge_filtered_when_categories_provided` (~282), `test_evaluate_on_test_omits_edge_filtered_when_only_one_kwarg_set` (~422)

- [ ] **Step 1: Update the test sites first (failing tests)**

In `tests/ml/test_training.py`:

Replace `test_fit_winning_model_returns_booster_with_expected_iterations` (~line 148):

```python
def test_fit_winning_model_returns_booster_with_expected_iterations() -> None:
    X_train, y_train, _, _, _ = _toy_problem()  # noqa: N806 -- ML matrix convention
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = fit_winning_model(
        best_params=params,
        best_iteration=10,
        dtrain=dtrain,
        seed=42,
    )
    # 11 trees corresponds to best_iteration + 1.
    assert booster.num_boosted_rounds() == 11
```

Replace `test_evaluate_on_test_returns_metric_dict` (~line 171) — change the `fit_winning_model` call so it builds a DMatrix:

```python
def test_evaluate_on_test_returns_metric_dict() -> None:
    X_train, y_train, X_val, y_val, _ = _toy_problem()  # noqa: N806 -- ML matrix convention
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }
    booster = fit_winning_model(
        best_params=params,
        best_iteration=20,
        dtrain=xgb.DMatrix(X_train, label=y_train),
        seed=42,
    )
    implied_test = np.full(len(y_val), 0.5)
    result = evaluate_on_test(booster, X_val, y_val, implied_test, n_min=5)
    assert "edge" in result
    assert "accuracy" in result
    assert "logloss" in result
    assert "per_decile" in result
    assert isinstance(result["per_decile"], dict)
```

Replace `_toy_booster` (~line 256):

```python
def _toy_booster(
    seed: int = 42,
) -> tuple[xgb.Booster, np.ndarray, np.ndarray, np.ndarray]:
    """Build a minimal booster + test arrays for evaluate_on_test tests."""
    X_train, y_train, X_val, y_val, _ = _toy_problem(seed=seed)  # noqa: N806 -- ML matrix convention
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }
    booster = fit_winning_model(
        best_params=params,
        best_iteration=20,
        dtrain=xgb.DMatrix(X_train, label=y_train),
        seed=seed,
    )
    implied_test = np.full(len(y_val), 0.5)
    return booster, X_val, y_val, implied_test
```

Replace `test_evaluate_on_test_returns_edge_filtered_when_categories_provided` (~line 282) — same pattern, the `fit_winning_model` call becomes:

```python
    X_train, y_train, _, _, _ = _toy_problem()  # noqa: N806
    booster = fit_winning_model(
        best_params={
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_child_weight": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.1,
        },
        best_iteration=10,
        dtrain=xgb.DMatrix(X_train, label=y_train),
        seed=42,
    )
```

Replace `test_evaluate_on_test_omits_edge_filtered_when_only_one_kwarg_set` (~line 422) — same pattern:

```python
    X_train, y_train, X_val, y_val, implied_val = _toy_problem()  # noqa: N806
    booster = fit_winning_model(
        best_params={
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_child_weight": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.1,
        },
        best_iteration=10,
        dtrain=xgb.DMatrix(X_train, label=y_train),
        seed=42,
    )
```

- [ ] **Step 2: Run the changed tests, see them fail**

```bash
uv run pytest tests/ml/test_training.py -v
```

Expected: all five updated tests fail with `TypeError: fit_winning_model() got an unexpected keyword argument 'dtrain'` (or similar — the production code still wants `X_train`/`y_train`).

- [ ] **Step 3: Update `fit_winning_model` signature**

In `src/pscanner/ml/training.py`, replace lines 157-195:

```python
def fit_winning_model(
    best_params: Mapping[str, object],
    best_iteration: int,
    X_train: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_train: np.ndarray,
    seed: int,
    device: str = "cpu",
) -> xgb.Booster:
    """Refit the winning hyperparams on train alone for ``best_iteration+1`` rounds.

    Avoids retraining on ``train + val`` (per the spec): the val set
    has already been used for model selection. Determinism is preserved
    by the shared ``seed`` + ``nthread=1``; this gives the same booster
    the winning trial produced.

    Args:
        best_params: Optuna's ``study.best_params`` dict.
        best_iteration: From the winning trial's user attrs.
        X_train: Training feature matrix.
        y_train: Training labels.
        seed: XGBoost RNG seed.
        device: XGBoost device, ``"cpu"`` or ``"cuda"``.

    Returns:
        The fitted XGBoost booster.
    """
    params: dict[str, object] = dict(best_params)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": device,
            "nthread": 1,
            "seed": seed,
            "verbosity": 0,
        }
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(params, dtrain, num_boost_round=best_iteration + 1)
```

with:

```python
def fit_winning_model(
    best_params: Mapping[str, object],
    best_iteration: int,
    dtrain: xgb.DMatrix,
    seed: int,
    device: str = "cpu",
) -> xgb.Booster:
    """Refit the winning hyperparams on ``dtrain`` for ``best_iteration+1`` rounds.

    Avoids retraining on ``train + val`` (per the spec): the val set
    has already been used for model selection. Determinism is preserved
    by the shared ``seed`` + ``nthread=1``; this gives the same booster
    the winning trial produced.

    Takes a pre-built ``dtrain`` so the winning-model refit reuses the
    DMatrix built for the Optuna phase. Callers that hold the source
    numpy arrays can release them between optimization and refit; the
    DMatrix carries XGBoost's quantized internal copy.

    Args:
        best_params: Optuna's ``study.best_params`` dict.
        best_iteration: From the winning trial's user attrs.
        dtrain: Pre-built training DMatrix (typically the one Optuna used).
        seed: XGBoost RNG seed.
        device: XGBoost device, ``"cpu"`` or ``"cuda"``.

    Returns:
        The fitted XGBoost booster.
    """
    params: dict[str, object] = dict(best_params)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": device,
            "nthread": 1,
            "seed": seed,
            "verbosity": 0,
        }
    )
    return xgb.train(params, dtrain, num_boost_round=best_iteration + 1)
```

- [ ] **Step 4: Update the `fit_winning_model` call site in `run_study`**

In `src/pscanner/ml/training.py`, find the `fit_winning_model(...)` call (~line 440):

```python
    booster = fit_winning_model(
        best_params=best_params,
        best_iteration=best_iteration,
        X_train=x_train,
        y_train=y_train,
        seed=seed,
        device=device,
    )
```

Replace with a temporary stand-in that builds a DMatrix at the call site so this task lands green. (Task 4 will lift this DMatrix construction up the function and release the source arrays.)

```python
    booster = fit_winning_model(
        best_params=best_params,
        best_iteration=best_iteration,
        dtrain=xgb.DMatrix(x_train, label=y_train),
        seed=seed,
        device=device,
    )
```

This is intentionally not the final shape — it builds a third DMatrix copy temporarily. Task 4 fixes that. Doing it this way means each task lands behaviorally complete on its own and the test suite stays green between tasks.

- [ ] **Step 5: Run the updated tests, see them pass**

```bash
uv run pytest tests/ml/test_training.py -v
```

Expected: all green.

- [ ] **Step 6: Lint + type-check**

```bash
uv run ruff check src/pscanner/ml/training.py tests/ml/test_training.py
uv run ruff format --check src/pscanner/ml/training.py tests/ml/test_training.py
uv run ty check
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/ml/training.py tests/ml/test_training.py
git commit -m "$(cat <<'EOF'
refactor(ml): fit_winning_model takes dtrain DMatrix instead of X_train/y_train

Lets callers reuse the DMatrix built for the Optuna phase, so the
source numpy arrays can be released between optimization and refit
in a follow-up commit. The internal xgb.train call is unchanged;
this is a signature-only refactor with cascading test updates.
EOF
)"
```

---

## Task 4: Lift DMatrix construction into `run_study`, release source arrays before Optuna

This is the core T2 fix. Move DMatrix construction out of `_run_optimization_phase` and into `run_study`. Release the source numpy arrays for train+val before the 100-trial loop. Reuse `dtrain` for `fit_winning_model`. Release `dval`/`y_val`/`implied_val` after Optuna returns.

**Files:**
- Modify: `src/pscanner/ml/training.py:272-334` (`_run_optimization_phase` signature + body)
- Modify: `src/pscanner/ml/training.py:425-447` (the `run_study` block from `_run_optimization_phase` call through `fit_winning_model` call)

- [ ] **Step 1: No new test — same end-to-end coverage as Task 2 applies**

`test_run_study_writes_all_artifacts` and `test_run_study_is_deterministic_under_same_seed` (both in `tests/ml/test_training.py`) already exercise the full pipeline. If the DMatrix-lifting refactor breaks anything, they fail.

Verify they're green on the post-Task-3 codebase:

```bash
uv run pytest tests/ml/test_training.py::test_run_study_writes_all_artifacts tests/ml/test_training.py::test_run_study_is_deterministic_under_same_seed -v
```

- [ ] **Step 2: Update `_run_optimization_phase` to take DMatrices**

In `src/pscanner/ml/training.py`, replace lines 272-334 (the entire `_run_optimization_phase` function):

```python
def _run_optimization_phase(
    output_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    implied_val: np.ndarray,
    n_trials: int,
    n_jobs: int,
    n_min: int,
    seed: int,
    device: str,
) -> tuple[int, dict[str, object], float]:
    """Run the Optuna study and return ``(best_iteration, best_params, best_value)``.

    Built as a separate function so the shared ``dtrain``/``dval``
    DMatrices (each a copy of the train/val data) are released back
    to the allocator at function return, before ``fit_winning_model``
    and ``evaluate_on_test`` build their own DMatrices.
    """
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    _log.info("ml.mem", phase="post_dmatrix", rss_mb=_rss_mb())

    # InMemoryStorage avoids the per-trial reload of the full study history
    # that RDBStorage(SQLite) does on every TPESampler.sample(). Each `run`
    # uses a fresh output_dir; resume isn't a documented feature here, so
    # there's no reason to leave a study.db artifact on disk.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
        storage=optuna.storages.InMemoryStorage(),
        study_name="copy_trade_gate",
    )
    study.optimize(
        lambda t: run_single_trial(
            trial=t,
            dtrain=dtrain,
            dval=dval,
            y_val=y_val,
            implied_prob_val=implied_val,
            n_min=n_min,
            seed=seed,
            device=device,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_iteration = int(study.best_trial.user_attrs["best_iteration"])
    best_params = dict(study.best_params)
    best_value = float(study.best_value)
    return best_iteration, best_params, best_value
```

with:

```python
def _run_optimization_phase(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    y_val: np.ndarray,
    implied_val: np.ndarray,
    n_trials: int,
    n_jobs: int,
    n_min: int,
    seed: int,
    device: str,
) -> tuple[int, dict[str, object], float]:
    """Run the Optuna study and return ``(best_iteration, best_params, best_value)``.

    DMatrices are constructed by the caller and passed in so the source
    numpy arrays can be released before this function runs — Optuna's
    100-trial loop is the longest phase and benefits most from minimum
    resident working set.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
        storage=optuna.storages.InMemoryStorage(),
        study_name="copy_trade_gate",
    )
    study.optimize(
        lambda t: run_single_trial(
            trial=t,
            dtrain=dtrain,
            dval=dval,
            y_val=y_val,
            implied_prob_val=implied_val,
            n_min=n_min,
            seed=seed,
            device=device,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_iteration = int(study.best_trial.user_attrs["best_iteration"])
    best_params = dict(study.best_params)
    best_value = float(study.best_value)
    return best_iteration, best_params, best_value
```

- [ ] **Step 3: Update the `run_study` body to build DMatrices, release source arrays, then call optimization**

In `src/pscanner/ml/training.py`, find the block in `run_study` from `best_iteration, best_params, best_value = _run_optimization_phase(...)` through the temporary `fit_winning_model(...)` call from Task 3. Replace this block:

```python
    best_iteration, best_params, best_value = _run_optimization_phase(
        output_dir=output_dir,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        implied_val=implied_val,
        n_trials=n_trials,
        n_jobs=n_jobs,
        n_min=n_min,
        seed=seed,
        device=device,
    )

    booster = fit_winning_model(
        best_params=best_params,
        best_iteration=best_iteration,
        dtrain=xgb.DMatrix(x_train, label=y_train),
        seed=seed,
        device=device,
    )
```

with:

```python
    # Build DMatrices up-front so the source numpy arrays can be released
    # before the 100-trial Optuna phase. XGBoost's DMatrix carries a
    # quantized internal copy; the float32 source arrays are dead from
    # this point until evaluate_on_test (which uses x_test, not x_train).
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    del x_train, y_train, x_val
    gc.collect()
    _log.info("ml.mem", phase="pre_optuna", rss_mb=_rss_mb())

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

    # Val arrays + dval are dead after optimization; only dtrain survives
    # for the winning-model refit.
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
```

Note: the `output_dir` argument is no longer passed to `_run_optimization_phase` — `study.db` isn't written anymore, so the function doesn't need the directory.

- [ ] **Step 4: Run the test suite**

```bash
uv run pytest tests/ml/ -v
```

Expected: all green. The determinism test is the most important canary — DMatrix construction order is stable, and `gc.collect()` doesn't perturb numpy values, so determinism must hold.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/pscanner/ml/training.py
uv run ruff format --check src/pscanner/ml/training.py
uv run ty check
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/training.py
git commit -m "$(cat <<'EOF'
perf(ml): build DMatrices in run_study, release source arrays before Optuna

The 100-trial Optuna loop is the longest training phase. Previously
x_train/y_train/x_val all stayed pinned in run_study's frame for the
duration even though XGBoost copies them into DMatrix at trial start.

Lift DMatrix construction up into run_study, del the source arrays
before _run_optimization_phase, and reuse dtrain for fit_winning_model
(via the new dtrain-taking signature from the previous commit). After
Optuna returns, release dval + val arrays before the winning-model
refit. New ml.mem log phases pre_optuna / post_optuna / post_fit_winning
make the lifetime trims observable.
EOF
)"
```

---

## Task 5: Operational verification on the laptop

The training command is too slow to run end-to-end in CI, so verification is operational. We compare RSS log lines on a sampled corpus before/after the changes.

**Files:** none modified. This task only runs the command and compares logs.

- [ ] **Step 1: Stash a baseline log from before the refactor**

If you didn't capture a baseline before starting Task 1, regenerate one by checking out `main` in a sibling worktree and running:

```bash
# In a separate worktree at main (NOT this branch)
uv run pscanner ml train --device cpu --n-jobs 1 --n-trials 5 \
    --db ./data/corpus-sample.sqlite3 \
    --output-dir /tmp/ml-baseline \
    > /tmp/ml-baseline.log 2>&1
grep '"ml.mem"' /tmp/ml-baseline.log
```

(If `corpus-sample.sqlite3` doesn't exist, you can sample one row-limited subset of the full corpus or skip this step and rely on the desktop's after-run RSS log inspection — see step 3.)

- [ ] **Step 2: Run on this branch, capture log**

```bash
uv run pscanner ml train --device cpu --n-jobs 1 --n-trials 5 \
    --db ./data/corpus-sample.sqlite3 \
    --output-dir /tmp/ml-after \
    > /tmp/ml-after.log 2>&1
grep '"ml.mem"' /tmp/ml-after.log
```

Expected: every `ml.mem` phase emits an RSS value. New phases `post_encoder_fit`, `pre_optuna`, `post_optuna`, `post_fit_winning` are present.

- [ ] **Step 3: Compare phases**

Diff the two logs side by side. The expected pattern:

| Phase | Before | After |
|---|---|---|
| `run_study_entry` | identical | identical |
| `post_encoder_fit` | absent | present (new) |
| `post_split_and_encode` | present | absent (collapsed into per-split flow) |
| `post_build_feature_matrix` | high (3 frames + 6 arrays + encoder) | lower (only arrays + encoder) |
| `post_polars_release` | drops to ~arrays-only | absent (collapsed) |
| `post_dmatrix` | present | absent (moved to `pre_optuna` shape) |
| `pre_optuna` | absent | present, lower than old `post_dmatrix` |
| `post_optuna` | absent | present, lower than `pre_optuna` |
| `post_fit_winning` | absent | present, lower than `post_optuna` |

Confirm `post_build_feature_matrix` is meaningfully lower (target: at least 1 GB drop on a full corpus run, proportionally less on a sampled one).

- [ ] **Step 4: Confirm no OOM on the laptop with a larger sample**

If the dev host has the full corpus available (~10 GB), try a slightly larger sample with `--n-trials 5` to confirm the laptop completes the run without OOM. The full 100-trial run is desktop-only; this is just a smoke check.

```bash
uv run pscanner ml train --device cpu --n-jobs 1 --n-trials 5 \
    --db ./data/corpus.sqlite3 \
    --output-dir /tmp/ml-smoke \
    > /tmp/ml-smoke.log 2>&1
echo exit=$?
grep '"ml.mem"' /tmp/ml-smoke.log | tail -10
```

Expected: exit=0, RSS stays well below the 7.4 GB OOM ceiling.

- [ ] **Step 5: No commit — operational task only**

Append the comparison numbers to a working note on the PR description when you open it.

---

## Task 6: Update related issues (T5 + T4 correction)

Pure GitHub issue-comment work. No code changes.

- [ ] **Step 1: Comment on issue #39 that the lifetime fixes here landed first (T5)**

```bash
gh issue comment 39 --body "$(cat <<'EOF'
Heads up — issue #67 landed first and recovered most of the same RSS
via lifetime trims (DMatrix lift + per-split teardown + InMemoryStorage)
without changing the training contract. The DataIter streaming work
proposed here should be re-evaluated against the new RSS baseline before
investing in it; #67's `ml.mem` log phases (`pre_optuna`, `post_optuna`,
`post_fit_winning`) make the new ceiling observable.
EOF
)"
```

- [ ] **Step 2: Correct the T4 framing on issue #67**

```bash
gh issue comment 67 --body "$(cat <<'EOF'
Correction on T4 — verified false. \`_CATEGORICAL_CAST_COLS\`
(\`src/pscanner/ml/preprocessing.py:31-36\`) contains only
\`condition_id\`, \`top_category\`, \`market_category\`, \`side\`.
\`wallet_address\` lives in \`LEAKAGE_COLS\` (\`:80-86\`) and is
correctly excluded from load via \`_NEVER_LOAD_COLS\` at the SQL
SELECT boundary. No code change needed. The audit conflated the two
constants.

T1, T2, T3, T5 are the real items and have all landed.
EOF
)"
```

- [ ] **Step 3: Update issue #67's T4 status**

After the PR for this plan merges, edit issue #67's body to strike through T4 with the correction note (or close the issue and note T4 separately if it was the only remaining item — but T1/T2/T3 are the meat, so closing the issue on PR merge is the natural outcome).

---

## Verification checklist (run before opening the PR)

- [ ] `uv run ruff check .` — clean
- [ ] `uv run ruff format --check .` — clean
- [ ] `uv run ty check` — clean
- [ ] `uv run pytest -q tests/ml/` — all green
- [ ] `uv run pytest -q` — full suite green (training changes shouldn't ripple, but verify)
- [ ] Operational RSS comparison from Task 5 captured for the PR description
- [ ] Issue #67 + #39 comments posted

---

## PR shape

Single PR with the 4 commits from Tasks 1–4. Title: `perf(ml): training memory — lifetime trims + Optuna InMemoryStorage (#67)`. Body cites the RSS log comparison from Task 5 and links #67 + #39.
