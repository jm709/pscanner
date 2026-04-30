"""Tests for ml.training."""

from __future__ import annotations

import numpy as np
import optuna

from pscanner.ml.training import run_single_trial


def _toy_problem(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 200
    # X is a single feature that is correlated with y.
    X = rng.normal(size=(n, 3))  # noqa: N806 -- ML matrix convention
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    implied = np.full(n, 0.5)  # All bets at 50% implied prob.
    X_train, X_val = X[:150], X[150:]  # noqa: N806 -- ML matrix convention
    y_train, y_val = y[:150], y[150:]
    implied_val = implied[150:]
    return X_train, y_train, X_val, y_val, implied_val


def test_run_single_trial_returns_finite_edge() -> None:
    X_train, y_train, X_val, y_val, implied_val = _toy_problem()  # noqa: N806 -- ML matrix convention
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        return run_single_trial(
            trial=trial,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            implied_prob_val=implied_val,
            n_min=5,
            seed=42,
        )

    study.optimize(objective, n_trials=2)
    assert len(study.trials) == 2
    # Each trial must record best_iteration as a user attr.
    for trial in study.trials:
        assert "best_iteration" in trial.user_attrs
        assert isinstance(trial.user_attrs["best_iteration"], int)


def test_run_single_trial_is_deterministic_under_same_seed() -> None:
    X_train, y_train, X_val, y_val, implied_val = _toy_problem()  # noqa: N806 -- ML matrix convention

    def study_value() -> float:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=7),
        )
        study.optimize(
            lambda t: run_single_trial(
                trial=t,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                implied_prob_val=implied_val,
                n_min=5,
                seed=7,
            ),
            n_trials=2,
        )
        return study.best_value

    assert study_value() == study_value()
