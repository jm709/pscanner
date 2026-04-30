"""Optuna-driven XGBoost training for the copy-trade gate model.

Single-trial fitting, study orchestration, winning-model refit, test
evaluation, and artifact dump. The optimization target is the custom
``realized_edge_metric``; ``binary:logistic`` keeps ``model_prob``
calibrated against ``implied_prob_at_buy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import structlog
import xgboost as xgb

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


def run_single_trial(
    trial: optuna.Trial,
    X_train: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_train: np.ndarray,
    X_val: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_val: np.ndarray,
    implied_prob_val: np.ndarray,
    n_min: int,
    seed: int,
) -> float:
    """Fit one XGBoost trial and return its validation realized edge.

    Sampled hyperparameters: ``learning_rate``, ``max_depth``,
    ``min_child_weight``, ``subsample``, ``colsample_bytree``,
    ``reg_alpha``, ``reg_lambda``, ``gamma``. Boosting rounds are
    capped at 2000 with 50-round early stopping on val log-loss; the
    actual rounds used for prediction is ``best_iteration + 1``. The
    chosen ``best_iteration`` is recorded on the trial's user attrs so
    the winning model can be refit later without re-running the study.

    Args:
        trial: Optuna trial object for parameter suggestion.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        implied_prob_val: Implied probability per validation row.
        n_min: Minimum copied bets for the edge metric guard.
        seed: XGBoost RNG seed.

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
        "nthread": 1,
        "seed": seed,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=_NUM_BOOST_ROUND,
        evals=[(dval, "val")],
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
    X_train: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_train: np.ndarray,
    seed: int,
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

    Returns:
        The fitted XGBoost booster.
    """
    params: dict[str, object] = dict(best_params)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "nthread": 1,
            "seed": seed,
            "verbosity": 0,
        }
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(params, dtrain, num_boost_round=best_iteration + 1)


def evaluate_on_test(
    booster: xgb.Booster,
    X_test: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_test: np.ndarray,
    implied_prob_test: np.ndarray,
    n_min: int,
) -> dict[str, object]:
    """Score the booster on the held-out test split.

    Returns:
        ``{"edge": float, "accuracy": float, "logloss": float,
        "per_decile": {decile_label: {"n": float, "mean_edge": float}}}``.
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
    return {
        "edge": edge,
        "accuracy": accuracy,
        "logloss": logloss,
        "per_decile": decile,
    }


def _dump_artifacts(
    output_dir: Path,
    booster: xgb.Booster,
    encoder: OneHotEncoder,
    metrics: dict[str, object],
) -> None:
    """Write model, preprocessor, and metrics to ``output_dir``."""
    booster.save_model(str(output_dir / "model.json"))
    preprocessor = {
        "leakage_cols": list(LEAKAGE_COLS),
        "carrier_cols": list(CARRIER_COLS),
        "encoder": encoder.to_json(),
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
) -> None:
    """End-to-end study: preprocess, run Optuna, refit, evaluate, dump.

    Mutates ``output_dir`` (created if missing). Writes ``model.json``,
    ``preprocessor.json``, ``study.db``, ``metrics.json``.

    Args:
        df: Output of ``load_dataset``.
        output_dir: Per-run artifact directory.
        n_trials: Optuna trial budget.
        n_jobs: Parallel trials. Must be >=1.
        n_min: Edge-metric anti-overfit guard threshold.
        seed: Master RNG seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    df = drop_leakage_cols(df)
    splits = temporal_split(df)
    encoder = OneHotEncoder.fit(splits.train, columns=CATEGORICAL_COLS)
    train_df = encoder.transform(splits.train)
    val_df = encoder.transform(splits.val)
    test_df = encoder.transform(splits.test)

    x_train, y_train, _ = build_feature_matrix(train_df)
    x_val, y_val, implied_val = build_feature_matrix(val_df)
    x_test, y_test, implied_test = build_feature_matrix(test_df)

    rates = {
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    }
    _log.info("ml.split_label_won_rate", **rates)

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
                X_train=x_train,
                y_train=y_train,
                X_val=x_val,
                y_val=y_val,
                implied_prob_val=implied_val,
                n_min=n_min,
                seed=seed,
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

    booster = fit_winning_model(
        best_params=best_params,
        best_iteration=best_iteration,
        X_train=x_train,
        y_train=y_train,
        seed=seed,
    )
    test_metrics = evaluate_on_test(
        booster=booster,
        X_test=x_test,
        y_test=y_test,
        implied_prob_test=implied_test,
        n_min=n_min,
    )

    metrics = {
        "best_params": best_params,
        "best_iteration": best_iteration,
        "best_val_edge": best_value,
        "test_edge": test_metrics["edge"],
        "test_accuracy": test_metrics["accuracy"],
        "test_logloss": test_metrics["logloss"],
        "test_per_decile": test_metrics["per_decile"],
        "split_label_won_rate": rates,
        "seed": seed,
    }
    _dump_artifacts(output_dir, booster, encoder, metrics)
    _log.info(
        "ml.study_complete",
        best_val_edge=metrics["best_val_edge"],
        test_edge=metrics["test_edge"],
        n_trials=n_trials,
    )
