# XGBoost copy-trade gate — design

Date: 2026-04-30
Status: design, awaiting implementation plan

## Problem

The historical corpus (`data/corpus.sqlite3`, ~3.79 GB; 5.16M trades,
4,364 markets, 360→3,346 resolutions in flight) materializes a
`training_examples` table — one row per qualifying BUY in a resolved
market, with ~30 point-in-time features and a binary `label_won` target.

We want to train an XGBoost classifier on this table whose live use case
is **copy-trade gating**: given a fresh BUY signal from another wallet,
score it; if the score implies positive expected edge over the trade's
implied probability, copy it.

Two design constraints follow from that use case:

1. The features and split scheme must mirror live deployment — at
   inference time we have only the trade itself plus everything observed
   strictly before it.
2. Model selection must optimize **realized edge per copied bet**, not
   accuracy. A model that wins 70% of the time on bets at 60% implied
   probability is strictly better than a model that wins 90% of the time
   on bets at 90% implied probability.

This spec defines a new `pscanner.ml` package that consumes
`training_examples` and produces a versioned model artifact. Inference
(live scoring of incoming trades) is out of scope for v1.

## Goals (v1)

1. Reproducible training pipeline from `training_examples` → versioned
   model artifact, runnable as `pscanner ml train`.
2. Temporal train/val/test split that eliminates market-level leakage
   and mirrors live deployment.
3. Drop columns with leakage or identity risk; one-hot encode the three
   low-cardinality categoricals; otherwise feed XGBoost raw values.
4. Optuna-driven hyperparameter search over a wide space, optimizing a
   custom realized-edge metric on val with a minimum-bet-count guard.
5. Final test-set evaluation on the Optuna-winning model with no
   retraining (val set is already used for model selection).
6. Persisted artifacts: model, preprocessor (one-hot mapping + dropped
   cols), Optuna study DB, metrics JSON including per-decile-of-implied
   breakdown.

## Non-goals (deferred)

- Live inference / scoring incoming trades. Separate concern; the
  preprocessor artifact persists the column transforms so a future
  scorer can apply them, but no live path is built here.
- Online learning / model updates from new data. Each `train` run is a
  full retrain.
- Stacked / ensembled models. Single XGBoost classifier.
- Cost-of-trading model (fees, slippage, depth). The edge metric is
  pre-cost; live deployment will need to layer those on.
- Calibration adjustments (Platt, isotonic). XGBoost's `binary:logistic`
  outputs are roughly calibrated; we rely on that for `model_prob`
  vs `implied_prob_at_buy` comparisons. If calibration drifts on
  diagnostics we revisit.
- A scaler or log transforms. XGBoost is invariant to monotonic
  transformations of input features; both add deployment surface for
  zero accuracy benefit.

## Architecture

### Module layout

```
src/pscanner/ml/
    __init__.py
    preprocessing.py   # column drops, one-hot encoding, temporal split
    metrics.py         # realized_edge_metric + per-decile breakdown
    training.py        # Optuna study, single-trial xgb fit, artifact dump
    cli.py             # `pscanner ml train` Click command
```

```
tests/ml/
    test_preprocessing.py
    test_metrics.py
    test_training.py
```

The package is parallel to `pscanner.corpus`. It consumes
`data/corpus.sqlite3` via Polars and writes nothing back. No shared
runtime state with the live daemon.

### Data flow

```
data/corpus.sqlite3 (training_examples + market_resolutions)
    |
    | polars.scan_database (lazy)
    v
join market_resolutions.resolved_at on condition_id
    |
    v
drop leakage cols (keeping condition_id and resolved_at as carriers)
    |
    v
temporal split by market resolved_at (60/20/20)
    |
    v
fit one-hot encoder on train; transform train/val/test
    |
    v
to numpy → xgb.DMatrix (in-memory, all three splits)
    |
    v
Optuna study (sqlite-backed, n_trials=100, n_jobs=10)
    |  — per trial: xgb.train(early_stopping=50) → realized_edge on val
    v
winning trial's xgb model evaluated once on test
    |
    v
dump model + preprocessor + study + metrics
```

## Preprocessing

### Lazy load

`training_examples` does not carry `resolved_at`; join from
`market_resolutions` on `condition_id`:

```python
df = pl.scan_database(
    """
    SELECT te.*, mr.resolved_at
    FROM training_examples te
    JOIN market_resolutions mr USING (condition_id)
    """,
    "sqlite:///data/corpus.sqlite3",
)
```

Predicate pushdown is preserved through the column-drop step. Inner
join is correct: `build-features` only emits rows for markets with a
`market_resolutions` row, so the join is row-preserving.

### Column drops

Drop these from features entirely:

| col | reason |
|---|---|
| `tx_hash` | identity; per-fill unique |
| `asset_id` | per-outcome identifier (would memorize labels) |
| `wallet_address` | identity; risk of fingerprinting via raw address |
| `condition_id` (as feature) | identity; market would be memorizable |
| `built_at` | meta-timestamp; not a real feature |
| `time_to_resolution_seconds` | uses post-hoc `closed_at` from `corpus_markets`, recorded at enumeration after resolution. Mild future leakage. Reinstating requires storing the trader-visible scheduled `endDate` separately at enumeration time — out of scope. |

Keep these as **carrier columns** through preprocessing — used for the
split and per-market eval, dropped before fitting:

| col | use |
|---|---|
| `condition_id` | grouping for diagnostics; never a feature |
| `trade_ts` | available if we ever want sanity checks; resolved_at is the actual split key |
| `resolved_at` | join from `market_resolutions`; the split key |

Keep `implied_prob_at_buy` both as a feature **and** as the baseline
input to the edge metric. It is not a leakage column; the trader knows
it at trade time.

### One-hot encoding

Three categoricals, all low-cardinality:

| col | levels |
|---|---|
| `side` | YES, NO |
| `top_category` | sports, esports, thesis, `__none__` (when wallet has no priors) |
| `market_category` | sports, esports, thesis, unknown |

Encode `top_category=None` as the explicit `__none__` level — first-time
wallets are a real signal, not missing data. Use Polars
`to_dummies(columns=...)` with the train-fit category set; for val/test,
align columns and emit zero-vectors for any unseen level (defensive,
should not occur in practice).

### Temporal split

Sort markets by their `resolved_at` (joined in from `market_resolutions`
on `condition_id`). Compute the 60th and 80th percentiles of distinct
`resolved_at` values. Assign each `condition_id` (and therefore all its
training rows) to one of `{train, val, test}` based on which percentile
band its `resolved_at` falls into. This guarantees:

- No market appears in two splits (no market leakage).
- Train precedes val precedes test in wall-clock time (mirrors
  deployment).
- Wallet leakage is unavoidable but does not violate point-in-time
  correctness: at inference, a wallet's `prior_*` stats would be
  computed from data preceding the trade, exactly as in train.

Tie-break on duplicate `resolved_at`: sort by `condition_id` lexically
(deterministic, irrelevant which side ties land on as long as it's
stable across runs).

Print `label_won` rate per split as a one-line diagnostic when splits
are constructed. Sharp deviation between train and test is informative,
not a fix needed.

### Missing values

Leave NaNs intact in numeric columns. XGBoost's missing-direction-on-split
rule routes them as a learned signal. Polars → numpy preserves nulls as
`np.nan`. Do not impute.

### No scaler, no log transforms

XGBoost is invariant to monotonic transformations of features.
Standardization and log-transforms add deployment surface (must apply at
inference) for zero accuracy benefit. Skip both.

## Edge metric

The single quantity Optuna optimizes on val:

```python
def realized_edge_metric(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
    n_min: int = 20,
) -> float:
    """Mean realized edge over bets the model would copy.

    A bet is "copied" iff y_pred_proba > implied_prob (model predicts
    positive expected edge). Realized edge per copied bet is
    label_won - implied_prob_at_buy. Returns -1.0 if fewer than n_min
    bets pass the gate, to penalize trial configurations that overfit
    to a tiny lucky subset.
    """
    take = y_pred_proba > implied_prob
    if int(take.sum()) < n_min:
        return -1.0
    return float((y_true[take] - implied_prob[take]).mean())
```

`n_min=20` is the relaxed initial guard. Tunable via CLI flag if it
proves under- or over-restrictive.

### Per-decile breakdown (diagnostic, not optimized)

Alongside the optimized metric, report mean realized edge stratified by
decile of `implied_prob_at_buy` over copied bets. Prints once for the
test-set evaluation. Reveals whether edge is concentrated in cheap-side
bets ("longshot finder") or distributed across implied-prob buckets
("genuine mispricing detector"). Not an optimization target.

## Training pipeline

### Class imbalance

Do not set `scale_pos_weight`. The edge metric depends on `model_prob`
being directly comparable to `implied_prob_at_buy`. Reweighting distorts
calibration and breaks that comparison. Use raw `binary:logistic`.

### Hyperparameter search

Optuna with TPE sampler + median pruner.

```
learning_rate:    log-uniform [0.01, 0.3]
max_depth:        int [3, 10]
min_child_weight: log-uniform [1, 100]
subsample:        uniform [0.5, 1.0]
colsample_bytree: uniform [0.5, 1.0]
reg_alpha:        log-uniform [1e-3, 10]
reg_lambda:       log-uniform [1e-3, 10]
gamma:            log-uniform [1e-3, 1]
n_estimators:     2000 (ceiling) with early_stopping_rounds=50 on val
                  (effective trees per trial = best_iteration + 1)
```

Each trial:
1. Build `xgb.DMatrix` for train and val (in memory).
2. `xgb.train(params, dtrain, num_boost_round=2000,
   evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)`.
3. Predict probabilities on val with `iteration_range=(0, best_iteration+1)`.
4. Compute `realized_edge_metric(y_val, p_val, implied_prob_val)`.
5. Return the metric. Pruner cuts trials whose mid-training val log-loss
   trends below the running median.

### Parallelism

Optuna `study.optimize(..., n_jobs=10)` — 10 trials in parallel. Inside
each trial, `xgb` runs single-threaded (`nthread=1`) to avoid
oversubscription. With CPU count `N`, cap at `min(10, N-1)`.

### Optuna storage

SQLite-backed, per-run: `models/<run-name>/study.db` via
`optuna.storages.RDBStorage`. Run survives kills; can be inspected
post-hoc.

### Final test evaluation

After the study finishes, take the Optuna-winning model (trained during
its trial; do **not** retrain on `train + val`). Predict probabilities
on test, compute:

- `realized_edge_metric` (optimized metric on test)
- Accuracy (informational)
- Log-loss (informational)
- Per-decile breakdown (diagnostic)

Dump everything to `metrics.json`.

## Output artifacts

Each `pscanner ml train` invocation writes to
`models/<YYYY-MM-DD>-<study-name>/`:

| file | format | content |
|---|---|---|
| `model.json` | XGBoost native JSON | The winning trial's booster |
| `preprocessor.json` | hand-rolled JSON | Column drop list + per-categorical level set, for inference-time consistency |
| `study.db` | SQLite (Optuna) | All trial parameters and values, resumable |
| `metrics.json` | JSON | val + test realized edge, accuracy, log-loss, per-decile breakdown, label_won rate per split, winning hyperparameters, seed |

## Reproducibility

Single `--seed` flag (default 42) propagated to:
- `numpy.random.seed`
- `optuna.samplers.TPESampler(seed=...)`
- XGBoost `seed` param per trial

`uv.lock` pins all deps; the dump captures the exact hyperparameter set;
`study.db` captures the trial sequence. Two runs with the same seed and
the same `corpus.sqlite3` should produce identical models.

## CLI surface

```
pscanner ml train [--n-trials 100] [--seed 42] [--output-dir models/]
                  [--n-min 20] [--n-jobs 10] [--db-path data/corpus.sqlite3]
```

Wires into the existing top-level `pscanner` Click group as a sibling
of `corpus`. No daemon hooks; one-shot.

## Testing

| test file | covers |
|---|---|
| `test_preprocessing.py` | column drops are exactly the documented set; one-hot encoder is fit on train levels and transforms val/test consistently; temporal split assigns no `condition_id` to two splits; `label_won` rate is logged |
| `test_metrics.py` | `realized_edge_metric` returns -1 below `n_min`; computes correct mean over the take-mask; per-decile breakdown handles empty deciles |
| `test_training.py` | single Optuna trial runs end-to-end on a tiny synthetic dataset; artifacts land at the documented paths; metrics JSON is well-formed; seed determinism (two runs same seed → same metrics) |

Use `polars` and a tiny synthetic in-memory `training_examples` for the
preprocessing and training tests; no need to touch real `corpus.sqlite3`.

`pyproject.toml` enforces `filterwarnings = ["error"]`; XGBoost and
Optuna are warning-noisy on edge cases — integration tests need the
`-W error::DeprecationWarning` style filter overrides only if
genuinely unfixable, with justification.

## Open follow-ups (not blocking v1)

- **Trader-visible `endDate` for non-leaky `time_to_resolution_seconds`.**
  Requires `pscanner.corpus.enumerator` to record the gamma `endDate`
  field separately from `closed_at`, then a `build-features` rerun.
- **Live inference path** (`pscanner.ml.scoring` + a live detector that
  consumes the model). The persisted preprocessor is built so this is
  drop-in.
- **Cost-of-trading model.** The edge metric is gross of fees and
  slippage. Live use will need to subtract these.
- **Calibration check.** Reliability diagram on val to confirm
  `model_prob` ≈ `P(label_won)`. If calibration is poor, the edge
  metric's threshold (`y_pred_proba > implied_prob`) is biased.
- **Feature ablations.** Once a baseline trains, try dropping
  `prior_*`-only, `market_*`-only, etc. to estimate per-group
  contribution.

## Dependencies

New packages to add to `pyproject.toml`:

- `polars` — lazy DataFrame I/O against SQLite
- `xgboost` — the model
- `optuna` — hyperparameter search

Each is a substantial dependency. Justification: polars handles the
SQLite scan + column ops cleanly without sqlalchemy; XGBoost is the
chosen learner; Optuna replaces grid/random search with TPE+pruner for
~10× efficiency at a small API surface.

`uv add polars xgboost optuna` and re-pin.
