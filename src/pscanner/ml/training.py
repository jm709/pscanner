"""Optuna-driven XGBoost training for the copy-trade gate model.

Single-trial fitting, study orchestration, winning-model refit, test
evaluation, and artifact dump. The optimization target is the custom
``realized_edge_metric``; ``binary:logistic`` keeps ``model_prob``
calibrated against ``implied_prob_at_buy``.
"""

from __future__ import annotations

import numpy as np
import optuna
import xgboost as xgb

from pscanner.ml.metrics import realized_edge_metric

_NUM_BOOST_ROUND = 2000
_EARLY_STOPPING_ROUNDS = 50


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
