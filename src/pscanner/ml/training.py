"""Optuna-driven XGBoost training for the copy-trade gate model.

Single-trial fitting, study orchestration, winning-model refit, test
evaluation, and artifact dump. The optimization target is the custom
``realized_edge_metric``; ``binary:logistic`` keeps ``model_prob``
calibrated against ``implied_prob_at_buy``.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import structlog
import xgboost as xgb

from pscanner.categories import Category
from pscanner.ml.metrics import per_decile_edge_breakdown, realized_edge_metric
from pscanner.ml.preprocessing import (
    CARRIER_COLS,
    CATEGORICAL_COLS,
    LEAKAGE_COLS,
    OneHotEncoder,
    build_feature_matrix,
    drop_leakage_cols,
    temporal_split,
)

_log = structlog.get_logger(__name__)

_NUM_BOOST_ROUND = 2000
_EARLY_STOPPING_ROUNDS = 50
_BINARY_DECISION_THRESHOLD = 0.5
_DEFAULT_ACCEPTED_CATEGORIES: tuple[str, ...] = (Category.SPORTS, Category.ESPORTS)


def _rss_mb() -> int:
    """Return current resident set size in MB (Linux /proc/self/statm)."""
    with Path("/proc/self/statm").open() as f:
        rss_pages = int(f.read().split()[1])
    return (rss_pages * os.sysconf("SC_PAGE_SIZE")) // (1024 * 1024)


def _make_edge_eval_metric(
    y_val: np.ndarray,
    implied_prob_val: np.ndarray,
    n_min: int,
) -> Callable[[np.ndarray, xgb.DMatrix], tuple[str, float]]:
    """Build an xgboost ``custom_metric`` closure that tracks realized edge.

    Returned tuple is ``("realized_edge", -edge)`` so xgboost's default
    minimization treats higher edge as better — early stopping then picks
    the round whose val edge is highest, not the round whose val log-loss
    is lowest. Logloss minima and edge maxima can diverge sharply (logloss
    penalizes overconfidence on losers; edge cares whether ``pred >
    implied`` predicts winners), so aligning the inner loop with what the
    outer Optuna loop actually grades on closes a real selection bias.

    The flat ``-1.0`` region returned by ``realized_edge_metric`` when
    ``n_taken < n_min`` becomes ``+1.0`` after negation. xgboost sees a
    plateau at the worst possible value early in training and continues
    boosting until enough bets pass the gate — no false-positive stop.
    """
    captured_y = y_val
    captured_implied = implied_prob_val
    captured_n_min = n_min

    def _metric(predt: np.ndarray, _dmatrix: xgb.DMatrix) -> tuple[str, float]:
        edge = realized_edge_metric(captured_y, predt, captured_implied, n_min=captured_n_min)
        return ("realized_edge", -edge)

    return _metric


def run_single_trial(
    trial: optuna.Trial,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    y_val: np.ndarray,
    implied_prob_val: np.ndarray,
    n_min: int,
    seed: int,
    device: str = "cpu",
) -> float:
    """Fit one XGBoost trial and return its validation realized edge.

    Sampled hyperparameters: ``learning_rate``, ``max_depth``,
    ``min_child_weight``, ``subsample``, ``colsample_bytree``,
    ``reg_alpha``, ``reg_lambda``, ``gamma``. Boosting rounds are
    capped at 2000 with 50-round early stopping on the custom
    ``realized_edge`` metric (negated so xgboost's default minimization
    selects the edge-maximizing round). ``logloss`` is still reported
    via ``eval_metric`` as a calibration sanity baseline in train logs.
    The actual rounds used for prediction is ``best_iteration + 1``.
    The chosen ``best_iteration`` is recorded on the trial's user attrs
    so the winning model can be refit later without re-running the study.

    ``dtrain`` and ``dval`` are expected to be shared across trials --
    XGBoost's ``train()`` treats them as read-only, so a single pair
    serves any number of joblib-thread workers without per-trial
    DMatrix copies.

    Args:
        trial: Optuna trial object for parameter suggestion.
        dtrain: Pre-built training DMatrix (shared across trials).
        dval: Pre-built validation DMatrix (shared across trials).
        y_val: Validation labels (for the edge metric).
        implied_prob_val: Implied probability per validation row.
        n_min: Minimum copied bets for the edge metric guard.
        seed: XGBoost RNG seed.
        device: XGBoost device, ``"cpu"`` or ``"cuda"``.

    Returns:
        The trial's realized edge on val (or ``-1.0`` if too few bets).
    """
    params: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        "device": device,
        "nthread": 1,
        "seed": seed,
        "verbosity": 0,
    }
    edge_metric = _make_edge_eval_metric(
        y_val=y_val, implied_prob_val=implied_prob_val, n_min=n_min
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=_NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        custom_metric=edge_metric,
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = booster.best_iteration
    p_val = booster.predict(dval, iteration_range=(0, best_iter + 1))
    edge = realized_edge_metric(y_val, p_val, implied_prob_val, n_min=n_min)
    trial.set_user_attr("best_iteration", int(best_iter))
    return edge


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


def evaluate_on_test(
    booster: xgb.Booster,
    X_test: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_test: np.ndarray,
    implied_prob_test: np.ndarray,
    n_min: int,
    top_category_test: np.ndarray | None = None,
    accepted_categories: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Score the booster on the held-out test split.

    Args:
        booster: Fitted XGBoost booster.
        X_test: Test feature matrix.
        y_test: Test labels.
        implied_prob_test: Implied probabilities per test row.
        n_min: Anti-overfit guard threshold for ``realized_edge_metric``.
        top_category_test: Optional string array (parallel to ``y_test``)
            of per-row ``top_category`` values. When provided together
            with ``accepted_categories``, an ``edge_filtered`` metric is
            computed over the accepted-category subset of taken bets.
        accepted_categories: Category strings to include in the filtered
            edge computation. Ignored when ``top_category_test`` is None.

    Returns:
        Dict with keys ``"edge"``, ``"accuracy"``, ``"logloss"``,
        ``"per_decile"``. When both ``top_category_test`` and
        ``accepted_categories`` are supplied, also includes
        ``"edge_filtered"``.
    """
    dtest = xgb.DMatrix(X_test)
    p_test = booster.predict(dtest)
    edge = realized_edge_metric(y_test, p_test, implied_prob_test, n_min=n_min)
    accuracy = float(((p_test >= _BINARY_DECISION_THRESHOLD).astype(int) == y_test).mean())
    eps = 1e-9
    logloss = float(
        -(y_test * np.log(p_test + eps) + (1 - y_test) * np.log(1 - p_test + eps)).mean()
    )
    decile = per_decile_edge_breakdown(y_test, p_test, implied_prob_test)
    result: dict[str, object] = {
        "edge": edge,
        "accuracy": accuracy,
        "logloss": logloss,
        "per_decile": decile,
    }
    if top_category_test is not None and accepted_categories is not None:
        cat_mask = np.isin(top_category_test, accepted_categories)
        # Apply category mask first, then let realized_edge_metric handle the
        # take-mask + n_min sentinel uniformly with the unfiltered branch.
        result["edge_filtered"] = realized_edge_metric(
            y_test[cat_mask],
            p_test[cat_mask],
            implied_prob_test[cat_mask],
            n_min=n_min,
        )
    return result


def _extract_top_category(df: pl.DataFrame) -> np.ndarray:
    """Return ``top_category`` values as a numpy string array.

    Null entries become the empty string ``""`` so ``np.isin`` comparisons
    against real category names always return ``False`` for them.

    Args:
        df: A Polars DataFrame that still has the ``top_category`` column
            (i.e. before the leakage-drop or one-hot encoding step).

    Returns:
        1D numpy array of dtype ``object`` (Python str), one entry per row.
    """
    return df["top_category"].fill_null("").to_numpy()


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


def _dump_artifacts(
    output_dir: Path,
    booster: xgb.Booster,
    encoder: OneHotEncoder,
    metrics: dict[str, object],
    accepted_categories: tuple[str, ...],
) -> None:
    """Write model, preprocessor, and metrics to ``output_dir``."""
    booster.save_model(str(output_dir / "model.json"))
    preprocessor = {
        "leakage_cols": list(LEAKAGE_COLS),
        "carrier_cols": list(CARRIER_COLS),
        "encoder": encoder.to_json(),
        "accepted_categories": list(accepted_categories),
    }
    (output_dir / "preprocessor.json").write_text(json.dumps(preprocessor, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def run_study(
    df: pl.DataFrame,
    output_dir: Path,
    n_trials: int,
    n_jobs: int,
    n_min: int,
    seed: int,
    device: str = "cpu",
    accepted_categories: tuple[str, ...] | None = None,
) -> None:
    """End-to-end study: preprocess, run Optuna, refit, evaluate, dump.

    Mutates ``output_dir`` (created if missing). Writes ``model.json``,
    ``preprocessor.json``, ``metrics.json``.

    Args:
        df: Output of ``load_dataset``.
        output_dir: Per-run artifact directory.
        n_trials: Optuna trial budget.
        n_jobs: Parallel trials. Must be >=1.
        n_min: Edge-metric anti-overfit guard threshold.
        seed: Master RNG seed.
        device: XGBoost device, ``"cpu"`` or ``"cuda"``. CPU is the
            default so the test suite stays runnable on hosts without
            an NVIDIA GPU.
        accepted_categories: Category strings to gate on at inference
            time. Written to ``preprocessor.json`` as metadata; does
            NOT filter training data. Defaults to
            ``_DEFAULT_ACCEPTED_CATEGORIES`` when ``None``.
    """
    resolved_categories = (
        accepted_categories if accepted_categories is not None else _DEFAULT_ACCEPTED_CATEGORIES
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    _log.info("ml.mem", phase="run_study_entry", rss_mb=_rss_mb())

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
    del test_df
    del df, splits  # raw frames no longer needed after all splits processed
    gc.collect()
    _log.info("ml.mem", phase="post_build_feature_matrix", rss_mb=_rss_mb())

    rates = {
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    }
    _log.info("ml.split_label_won_rate", **rates)

    # Build DMatrices up-front so the train/val feature arrays can be
    # released before the 100-trial Optuna phase. XGBoost's DMatrix carries
    # a quantized internal copy. y_val and implied_val survive into Optuna
    # (the edge metric closure needs them) and are released after the study
    # returns. x_test stays live for evaluate_on_test.
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
    test_metrics = evaluate_on_test(
        booster=booster,
        X_test=x_test,
        y_test=y_test,
        implied_prob_test=implied_test,
        n_min=n_min,
        top_category_test=top_category_test,
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
