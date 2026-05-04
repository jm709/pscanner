"""Tests for ml.training."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

from pscanner.ml.training import (
    evaluate_on_test,
    fit_winning_model,
    run_single_trial,
    run_study,
)


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
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        return run_single_trial(
            trial=trial,
            dtrain=dtrain,
            dval=dval,
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
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def study_value() -> float:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=7),
        )
        study.optimize(
            lambda t: run_single_trial(
                trial=t,
                dtrain=dtrain,
                dval=dval,
                y_val=y_val,
                implied_prob_val=implied_val,
                n_min=5,
                seed=7,
            ),
            n_trials=2,
        )
        return study.best_value

    assert study_value() == study_value()


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
    booster = fit_winning_model(
        best_params=params,
        best_iteration=10,
        X_train=X_train,
        y_train=y_train,
        seed=42,
    )
    # 11 trees corresponds to best_iteration + 1.
    assert booster.num_boosted_rounds() == 11


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
        X_train=X_train,
        y_train=y_train,
        seed=42,
    )
    implied_test = np.full(len(y_val), 0.5)
    result = evaluate_on_test(booster, X_val, y_val, implied_test, n_min=5)
    assert "edge" in result
    assert "accuracy" in result
    assert "logloss" in result
    assert "per_decile" in result
    assert isinstance(result["per_decile"], dict)


def test_run_study_writes_all_artifacts(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=20, rows_per_market=15, seed=3)
    output_dir = tmp_path / "run"
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=3,
        n_jobs=1,
        n_min=5,
        seed=42,
    )
    assert (output_dir / "model.json").exists()
    assert (output_dir / "preprocessor.json").exists()
    assert (output_dir / "study.db").exists()
    assert (output_dir / "metrics.json").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "best_params" in metrics
    assert "best_iteration" in metrics
    assert "best_val_edge" in metrics
    assert "test_edge" in metrics
    assert "test_accuracy" in metrics
    assert "test_logloss" in metrics
    assert "test_per_decile" in metrics
    assert "split_label_won_rate" in metrics
    rates = metrics["split_label_won_rate"]
    assert {"train", "val", "test"} == set(rates.keys())
    assert metrics["seed"] == 42
    preprocessor = json.loads((output_dir / "preprocessor.json").read_text())
    assert "leakage_cols" in preprocessor
    assert "carrier_cols" in preprocessor
    assert "encoder" in preprocessor
    assert "tx_hash" in preprocessor["leakage_cols"]
    assert "condition_id" in preprocessor["carrier_cols"]
    assert "side" in preprocessor["encoder"]["levels"]


def test_run_study_n_jobs_2_completes_without_lock_errors(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=20, rows_per_market=15, seed=3)
    output_dir = tmp_path / "run_parallel"
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=2,
        n_jobs=2,
        n_min=5,
        seed=42,
    )
    assert (output_dir / "study.db").exists()
    assert (output_dir / "metrics.json").exists()


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
        X_train=X_train,
        y_train=y_train,
        seed=seed,
    )
    implied_test = np.full(len(y_val), 0.5)
    return booster, X_val, y_val, implied_test


def test_evaluate_on_test_returns_edge_filtered_when_categories_provided() -> None:
    booster, X_val, y_val, implied_test = _toy_booster()  # noqa: N806 -- ML matrix convention
    n = len(y_val)
    # Assign half the rows to accepted categories, the other half to "thesis".
    top_cat = np.array(["sports" if i % 2 == 0 else "thesis" for i in range(n)])
    accepted = ("sports", "esports")

    result = evaluate_on_test(
        booster,
        X_val,
        y_val,
        implied_test,
        n_min=1,
        top_category_test=top_cat,
        accepted_categories=accepted,
    )

    assert "edge_filtered" in result
    assert "edge" in result
    # Verify edge_filtered is independently computable from raw arrays.
    p_test = booster.predict(xgb.DMatrix(X_val))
    cat_mask = np.isin(top_cat, accepted)
    take_mask = p_test > implied_test
    combined = cat_mask & take_mask
    if combined.sum() >= 1:
        expected_filtered = float((y_val[combined] - implied_test[combined]).mean())
        assert result["edge_filtered"] == expected_filtered
    # The overall edge uses all taken bets, so the two metrics differ when
    # there are taken bets outside the accepted categories.
    assert result["edge"] != result["edge_filtered"] or not (take_mask & ~cat_mask).any(), (
        "edge == edge_filtered implies no out-of-category taken bets"
    )


def test_evaluate_on_test_omits_edge_filtered_when_categories_none() -> None:
    booster, X_val, y_val, implied_test = _toy_booster()  # noqa: N806 -- ML matrix convention
    result = evaluate_on_test(
        booster,
        X_val,
        y_val,
        implied_test,
        n_min=5,
        top_category_test=None,
        accepted_categories=None,
    )
    assert "edge_filtered" not in result


def test_run_study_writes_accepted_categories_to_preprocessor_json(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=20, rows_per_market=15, seed=3)
    output_dir = tmp_path / "run_cats"
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=2,
        n_jobs=1,
        n_min=5,
        seed=42,
    )
    preprocessor = json.loads((output_dir / "preprocessor.json").read_text())
    assert "accepted_categories" in preprocessor
    assert preprocessor["accepted_categories"] == ["sports", "esports"]


def test_run_study_writes_test_edge_filtered_to_metrics_json(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=20, rows_per_market=15, seed=3)
    output_dir = tmp_path / "run_filtered"
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=2,
        n_jobs=1,
        n_min=5,
        seed=42,
    )
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "test_edge_filtered" in metrics
    assert "accepted_categories" in metrics
    assert metrics["accepted_categories"] == ["sports", "esports"]


def test_run_study_is_deterministic_under_same_seed(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=20, rows_per_market=15, seed=3)

    def run_once(name: str) -> dict[str, object]:
        out = tmp_path / name
        run_study(
            df=df,
            output_dir=out,
            n_trials=3,
            n_jobs=1,  # n_jobs=1 for strict determinism
            n_min=5,
            seed=42,
        )
        return json.loads((out / "metrics.json").read_text())

    a = run_once("a")
    b = run_once("b")
    assert a["best_params"] == b["best_params"]
    assert a["best_iteration"] == b["best_iteration"]
    assert a["best_val_edge"] == b["best_val_edge"]
    assert a["test_edge"] == b["test_edge"]
    assert a["test_accuracy"] == b["test_accuracy"]
    assert a["test_logloss"] == b["test_logloss"]
