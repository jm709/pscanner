"""Tests for ml.preprocessing."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import polars as pl

from pscanner.ml.preprocessing import (
    _INT32_COLS,
    CARRIER_COLS,
    CATEGORICAL_COLS,
    LEAKAGE_COLS,
    OneHotEncoder,
    build_feature_matrix,
    drop_leakage_cols,
)


def test_leakage_cols_lists_documented_drops() -> None:
    expected = {
        "tx_hash",
        "asset_id",
        "wallet_address",
        "built_at",
        "time_to_resolution_seconds",
    }
    assert set(LEAKAGE_COLS) == expected


def test_carrier_cols_lists_documented_carriers() -> None:
    assert set(CARRIER_COLS) == {"condition_id", "trade_ts", "resolved_at"}


def test_categorical_cols_excludes_market_category() -> None:
    """``market_category`` is no longer one-hot-encoded — it's replaced by
    9 binary ``cat_*`` columns populated by build-features (#122).
    """
    assert "market_category" not in CATEGORICAL_COLS
    assert set(CATEGORICAL_COLS) == {"side", "top_category"}


def test_int32_cols_includes_cat_indicators() -> None:
    """Multi-label category indicators are int32 (0/1)."""
    expected = {
        "cat_sports",
        "cat_esports",
        "cat_thesis",
        "cat_macro",
        "cat_elections",
        "cat_crypto",
        "cat_geopolitics",
        "cat_tech",
        "cat_culture",
    }
    assert expected.issubset(set(_INT32_COLS))


def test_drop_leakage_cols_removes_each_documented_col(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=3, rows_per_market=2)
    out = drop_leakage_cols(df)
    for col in LEAKAGE_COLS:
        assert col not in out.columns
    # Carrier cols must survive the drop.
    for col in CARRIER_COLS:
        assert col in out.columns
    # Categorical cols and label must survive.
    assert "label_won" in out.columns
    for col in CATEGORICAL_COLS:
        assert col in out.columns


def test_drop_leakage_cols_is_idempotent(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=3, rows_per_market=2)
    once = drop_leakage_cols(df)
    twice = drop_leakage_cols(once)
    assert once.columns == twice.columns


def test_one_hot_encoder_fits_on_train_levels() -> None:
    df = pl.DataFrame(
        {
            "side": ["YES", "NO", "YES"],
            "top_category": ["sports", None, "thesis"],
            "market_category": ["sports", "esports", "thesis"],
            "implied_prob_at_buy": [0.5, 0.5, 0.5],
            "label_won": [1, 0, 1],
        }
    )
    enc = OneHotEncoder.fit(df, columns=("side", "top_category", "market_category"))
    assert enc.levels["side"] == ("NO", "YES")
    assert enc.levels["top_category"] == ("__none__", "sports", "thesis")
    assert enc.levels["market_category"] == ("esports", "sports", "thesis")


def test_one_hot_encoder_transform_emits_indicator_columns() -> None:
    df = pl.DataFrame(
        {
            "side": ["YES", "NO", "YES"],
            "top_category": ["sports", None, "thesis"],
            "market_category": ["sports", "esports", "thesis"],
            "implied_prob_at_buy": [0.5, 0.5, 0.5],
            "label_won": [1, 0, 1],
        }
    )
    enc = OneHotEncoder.fit(df, columns=("side", "top_category", "market_category"))
    out = enc.transform(df)
    # Original categoricals dropped.
    for col in ("side", "top_category", "market_category"):
        assert col not in out.columns
    # New indicator columns present.
    assert "side__YES" in out.columns
    assert "side__NO" in out.columns
    assert "top_category____none__" in out.columns
    # Indicators carry correct values for the first row (YES, sports, sports).
    assert out["side__YES"][0] == 1
    assert out["side__NO"][0] == 0
    assert out["top_category__sports"][0] == 1
    assert out["top_category____none__"][0] == 0
    # Second row had top_category=None -> __none__.
    assert out["top_category____none__"][1] == 1


def test_one_hot_encoder_handles_unseen_levels_at_transform() -> None:
    train = pl.DataFrame({"side": ["YES", "NO"]})
    val = pl.DataFrame({"side": ["YES", "DRAW"]})  # DRAW not seen at fit
    enc = OneHotEncoder.fit(train, columns=("side",))
    out = enc.transform(val)
    # Both fit-time levels exist on the output.
    assert "side__YES" in out.columns
    assert "side__NO" in out.columns
    # Unseen value gets all zeros across known levels.
    assert out["side__YES"][1] == 0
    assert out["side__NO"][1] == 0


def test_one_hot_encoder_round_trips_through_json() -> None:
    df = pl.DataFrame({"side": ["YES", "NO"], "top_category": ["sports", None]})
    enc = OneHotEncoder.fit(df, columns=("side", "top_category"))
    payload = enc.to_json()
    rendered = json.dumps(payload)
    parsed = json.loads(rendered)
    enc2 = OneHotEncoder.from_json(parsed)
    assert enc2.levels == enc.levels


def test_build_feature_matrix_extracts_arrays() -> None:
    df = pl.DataFrame(
        {
            "condition_id": ["a", "b"],
            "trade_ts": [1, 2],
            "resolved_at": [10, 20],
            "implied_prob_at_buy": [0.4, 0.7],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            "label_won": [1, 0],
        }
    )
    X, y, implied = build_feature_matrix(df)  # noqa: N806 -- ML matrix convention
    assert X.shape == (2, 3)  # implied_prob_at_buy, feature_a, feature_b
    assert y.tolist() == [1, 0]
    assert implied.tolist() == [0.4, 0.7]


def test_build_feature_matrix_preserves_nan() -> None:
    df = pl.DataFrame(
        {
            "condition_id": ["a"],
            "trade_ts": [1],
            "resolved_at": [10],
            "implied_prob_at_buy": [0.5],
            "win_rate": [None],
            "label_won": [1],
        }
    )
    X, _, _ = build_feature_matrix(df)  # noqa: N806 -- ML matrix convention
    # Polars null -> numpy nan.
    assert np.isnan(X[0, 1])
