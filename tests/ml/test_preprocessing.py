"""Tests for ml.preprocessing."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from pscanner.ml.preprocessing import (
    CARRIER_COLS,
    CATEGORICAL_COLS,
    LEAKAGE_COLS,
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


def test_categorical_cols_lists_documented_categoricals() -> None:
    assert set(CATEGORICAL_COLS) == {"side", "top_category", "market_category"}


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
