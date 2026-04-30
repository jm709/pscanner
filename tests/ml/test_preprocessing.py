"""Tests for ml.preprocessing."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import polars as pl

from pscanner.corpus.db import init_corpus_db
from pscanner.ml.preprocessing import (
    CARRIER_COLS,
    CATEGORICAL_COLS,
    LEAKAGE_COLS,
    OneHotEncoder,
    Split,
    drop_leakage_cols,
    load_dataset,
    temporal_split,
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


def test_temporal_split_partitions_by_resolved_at_percentiles(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    assert isinstance(split, Split)
    # All rows must be assigned exactly once.
    total = split.train.height + split.val.height + split.test.height
    assert total == df.height


def test_temporal_split_no_market_in_two_splits(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    train_markets = set(split.train["condition_id"].unique().to_list())
    val_markets = set(split.val["condition_id"].unique().to_list())
    test_markets = set(split.test["condition_id"].unique().to_list())
    assert train_markets.isdisjoint(val_markets)
    assert train_markets.isdisjoint(test_markets)
    assert val_markets.isdisjoint(test_markets)


def test_temporal_split_train_precedes_val_precedes_test(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    train_max = split.train["resolved_at"].max()
    val_min = split.val["resolved_at"].min()
    val_max = split.val["resolved_at"].max()
    test_min = split.test["resolved_at"].min()
    assert train_max <= val_min
    assert val_max <= test_min


def test_temporal_split_60_20_20_proportion(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    # 30 markets at 60/20/20 -> 18 / 6 / 6 markets.
    assert split.train["condition_id"].n_unique() == 18
    assert split.val["condition_id"].n_unique() == 6
    assert split.test["condition_id"].n_unique() == 6


def _seed_db_from_synthetic(
    conn: sqlite3.Connection,
    df: pl.DataFrame,
) -> None:
    # Populate corpus_markets, market_resolutions, training_examples
    # from a synthetic Polars frame so load_dataset has matching rows.
    markets = df.select(["condition_id", "resolved_at"]).unique()
    for row in markets.iter_rows(named=True):
        conn.execute(
            """
            INSERT INTO corpus_markets (
              condition_id, event_slug, category, closed_at,
              total_volume_usd, market_slug, backfill_state, enumerated_at
            ) VALUES (?, '', 'sports', ?, 1000.0, '', 'complete', ?)
            """,
            (row["condition_id"], int(row["resolved_at"]), int(row["resolved_at"]) - 1),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions (
              condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, 0, 1, ?, 'gamma', ?)
            """,
            (row["condition_id"], int(row["resolved_at"]), int(row["resolved_at"])),
        )
    # Drop resolved_at from the example rows (it lives on market_resolutions).
    examples = df.drop("resolved_at")
    for row in examples.iter_rows(named=True):
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO training_examples ({cols}) VALUES ({placeholders})",  # noqa: S608 -- column names are statically derived from synthetic frame
            tuple(row.values()),
        )
    conn.commit()


def test_load_dataset_joins_resolved_at(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    try:
        synthetic = make_synthetic_examples(n_markets=4, rows_per_market=3)
        _seed_db_from_synthetic(conn, synthetic)
    finally:
        conn.close()
    out = load_dataset(db_path)
    assert out.height == 12
    assert "resolved_at" in out.columns
    assert "label_won" in out.columns
    # Inner join: every row has a resolved_at.
    assert out["resolved_at"].null_count() == 0
