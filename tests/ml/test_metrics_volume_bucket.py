"""Unit tests for per_volume_bucket_edge_breakdown (#109)."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from pscanner.ml.metrics import per_volume_bucket_edge_breakdown
from pscanner.ml.training import evaluate_on_test


def test_buckets_emitted_only_when_taken_bets_present() -> None:
    """Buckets with no taken bets are omitted from the result."""
    y_true = np.array([1, 0])
    y_pred = np.array([0.6, 0.4])
    implied = np.array([0.5, 0.5])  # only first row has y_pred > implied
    volume = np.array([2_000_000.0, 50_000.0])

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    assert list(result.keys()) == ["1M-5M"]
    assert result["1M-5M"]["n"] == 1.0
    assert result["1M-5M"]["mean_edge"] == 0.5  # (1 - 0.5)


def test_volume_bucket_boundaries() -> None:
    """Boundary values land in the lower-bound-inclusive bucket."""
    # 5 rows, all taken (y_pred > implied), one in each bucket
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    implied = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    volume = np.array(
        [
            100_000.0,  # <250K
            250_000.0,  # 250K-1M
            1_000_000.0,  # 1M-5M
            5_000_000.0,  # 5M-25M
            25_000_000.0,  # >=25M
        ]
    )

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    assert sorted(result.keys()) == ["1M-5M", "250K-1M", "25M+", "5M-25M", "<250K"]
    for bucket in result.values():
        assert bucket["n"] == 1.0
        assert bucket["mean_edge"] == 0.5


def test_only_taken_bets_counted() -> None:
    """Bets where y_pred <= implied are excluded from the breakdown."""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0.6, 0.4, 0.6])
    implied = np.array([0.5, 0.5, 0.5])
    volume = np.array([2_000_000.0, 2_000_000.0, 2_000_000.0])

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    # Only rows 0 and 2 are taken (y_pred=0.6 > implied=0.5).
    assert result["1M-5M"]["n"] == 2.0


def test_empty_inputs_return_empty_dict() -> None:
    """No rows yields an empty dict, not an error."""
    y_true = np.array([], dtype=np.int32)
    y_pred = np.array([], dtype=np.float32)
    implied = np.array([], dtype=np.float32)
    volume = np.array([], dtype=np.float32)

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    assert result == {}


def test_evaluate_on_test_includes_per_volume_bucket() -> None:
    """`evaluate_on_test` returns ``per_volume_bucket`` when volume array is given."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(int)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "max_depth": 2,
            "tree_method": "hist",
            "verbosity": 0,
        },
        dtrain=xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    implied = np.full(n, 0.4, dtype=np.float32)
    volume = np.full(n, 2_000_000.0, dtype=np.float32)

    result = evaluate_on_test(
        booster=booster,
        X_test=x,
        y_test=y,
        implied_prob_test=implied,
        n_min=20,
        total_volume_usd_test=volume,
    )

    assert "per_volume_bucket" in result
    assert "1M-5M" in result["per_volume_bucket"]
