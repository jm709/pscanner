"""Preprocessing for the copy-trade gate training pipeline.

Exposes:

* ``LEAKAGE_COLS`` / ``CARRIER_COLS`` / ``CATEGORICAL_COLS`` -- column
  membership constants documented in the design spec.
* ``drop_leakage_cols`` -- pure column removal.
* ``OneHotEncoder`` -- fit-on-train, transform-each-split. Handles
  the ``__none__`` level for nullable categoricals.
* ``temporal_split`` -- assigns each ``condition_id`` to one of
  ``{train, val, test}`` by ``resolved_at`` percentile.
* ``load_dataset`` -- sqlite3 -> Polars DataFrame, joining
  ``training_examples`` with ``market_resolutions.resolved_at``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Low-cardinality string columns cast to Categorical at load time.
# condition_id has ~50K unique values; top_category/market_category/side
# each have single-digit cardinality.  Categorical storage avoids
# materialising the full 66-char hex strings (condition_id) or repeated
# string literals across millions of rows.
_CATEGORICAL_CAST_COLS: tuple[str, ...] = (
    "condition_id",
    "top_category",
    "market_category",
    "side",
)

# Integer columns safe to narrow from Int64 → Int32.  All values are
# counts / durations that fit comfortably in 32 bits; the resulting numpy
# matrix is float32 anyway, so the narrowing costs nothing in precision.
_INT32_COLS: tuple[str, ...] = (
    "prior_trades_count",
    "prior_buys_count",
    "prior_resolved_buys",
    "prior_wins",
    "prior_losses",
    "seconds_since_last_trade",
    "prior_trades_30d",
    "category_diversity",
    "market_unique_traders_so_far",
    "market_age_seconds",
    "time_to_resolution_seconds",
    "label_won",
)

# Float columns narrowed from Float64 → Float32.  XGBoost and
# build_feature_matrix both produce float32 output, so keeping float64
# inputs is pure overhead.
_FLOAT32_COLS: tuple[str, ...] = (
    "win_rate",
    "avg_implied_prob_paid",
    "realized_edge_pp",
    "prior_realized_pnl_usd",
    "avg_bet_size_usd",
    "median_bet_size_usd",
    "wallet_age_days",
    "bet_size_usd",
    "bet_size_rel_to_avg",
    "implied_prob_at_buy",
    "market_volume_so_far_usd",
    "last_trade_price",
    "price_volatility_recent",
)

_NONE_TOKEN = "__none__"  # noqa: S105 -- explicit null sentinel level, not a credential

LEAKAGE_COLS: tuple[str, ...] = (
    "tx_hash",
    "asset_id",
    "wallet_address",
    "built_at",
    "time_to_resolution_seconds",
)

CARRIER_COLS: tuple[str, ...] = ("condition_id", "trade_ts", "resolved_at")

CATEGORICAL_COLS: tuple[str, ...] = ("side", "top_category", "market_category")


def drop_leakage_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that risk identity leakage or future-information leakage.

    See the design spec for per-column reasoning. Idempotent -- drops only
    columns that exist on the input frame.
    """
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    return df.drop(to_drop)


@dataclass(frozen=True)
class OneHotEncoder:
    """Fit-once, transform-many one-hot encoder.

    Only encodes the columns passed to ``fit``. At ``fit`` time, nulls
    are mapped to the explicit ``"__none__"`` level so first-time
    wallets become a learnable signal rather than a missing value.
    Levels are stored sorted for deterministic column ordering.
    """

    levels: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, df: pl.DataFrame, columns: Iterable[str]) -> OneHotEncoder:
        """Discover the level set per column on a (training) DataFrame."""
        levels: dict[str, tuple[str, ...]] = {}
        for col in columns:
            uniq = (
                df.select(pl.col(col).fill_null(_NONE_TOKEN)).to_series().unique().sort().to_list()
            )
            levels[col] = tuple(str(v) for v in uniq)
        return cls(levels=levels)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace fit columns with per-level int8 indicator columns.

        Each fit column becomes one ``{col}__{level}`` indicator per
        known level, dropping the original column. Unseen levels at
        transform time are silently mapped to all-zeros across the
        known indicators.
        """
        out = df
        for col, lvls in self.levels.items():
            if col not in out.columns:
                continue
            filled = out.with_columns(pl.col(col).fill_null(_NONE_TOKEN).alias(col))
            indicator_exprs = [
                (pl.col(col) == lvl).cast(pl.Int8).alias(f"{col}__{lvl}") for lvl in lvls
            ]
            out = filled.with_columns(indicator_exprs).drop(col)
        return out

    def to_json(self) -> dict[str, dict[str, list[str]]]:
        """Serialise level state to a JSON-safe dict."""
        return {"levels": {k: list(v) for k, v in self.levels.items()}}

    @classmethod
    def from_json(cls, payload: dict[str, dict[str, list[str]]]) -> OneHotEncoder:
        """Rebuild an encoder from ``to_json`` output."""
        return cls(levels={k: tuple(v) for k, v in payload["levels"].items()})


@dataclass(frozen=True)
class Split:
    """Three-way temporal split of a training-examples DataFrame."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def temporal_split(
    df: pl.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Split:
    """Split rows into train/val/test by ``resolved_at`` market percentiles.

    Each ``condition_id`` lands in exactly one split; trades for a market
    cannot leak across splits. The split key is the market's
    ``resolved_at``, sorted ascending. Tie-break on ``condition_id``
    lexically for a stable order.

    Args:
        df: Polars DataFrame with at least ``condition_id`` and
            ``resolved_at`` columns.
        train_frac: Fraction of distinct markets in train.
        val_frac: Fraction of distinct markets in val. ``test_frac`` is
            ``1 - train_frac - val_frac``.

    Returns:
        A ``Split`` with three disjoint DataFrames.
    """
    markets = (
        df.select(["condition_id", "resolved_at"]).unique().sort(["resolved_at", "condition_id"])
    )
    n = markets.height
    n_train = round(train_frac * n)
    n_val = round(val_frac * n)
    train_ids = set(markets["condition_id"].slice(0, n_train).to_list())
    val_ids = set(markets["condition_id"].slice(n_train, n_val).to_list())
    test_ids = set(markets["condition_id"].slice(n_train + n_val, n - n_train - n_val).to_list())
    # Drop condition_id after assignment — downstream pipeline (encoder,
    # feature matrix) doesn't need it, and three corpus-scale copies of a
    # 66-char hex column are significant memory overhead.
    return Split(
        train=df.filter(pl.col("condition_id").is_in(train_ids)).drop("condition_id"),
        val=df.filter(pl.col("condition_id").is_in(val_ids)).drop("condition_id"),
        test=df.filter(pl.col("condition_id").is_in(test_ids)).drop("condition_id"),
    )


# Columns excluded at SELECT time so the full-fat DataFrame is never
# materialized. ``id`` is the autoincrement primary key, useless for
# training. ``LEAKAGE_COLS`` are dropped downstream by
# ``drop_leakage_cols`` anyway -- excluding them at the SQL boundary
# avoids loading ~1+ GB of hex-string columns (tx_hash, asset_id,
# wallet_address are 42-66 chars x 5M rows each) only to drop them.
_NEVER_LOAD_COLS: frozenset[str] = frozenset({"id", *LEAKAGE_COLS})


def load_dataset(db_path: Path) -> pl.DataFrame:
    """Load ``training_examples`` joined with ``market_resolutions.resolved_at``.

    Inner join is correct: ``build-features`` only emits rows for
    markets with a resolutions entry, so the join is row-preserving.

    Excludes ``LEAKAGE_COLS`` (and the internal ``id`` PK) at the SQL
    SELECT boundary so they are never materialized. At ~5M rows the
    three hex-string leakage cols (``tx_hash``, ``asset_id``,
    ``wallet_address``) account for ~1 GB+ of Python/Arrow allocations
    on their own, which is enough to OOM a 7.6 GB host during load.

    Args:
        db_path: Path to the corpus SQLite file.

    Returns:
        A Polars DataFrame with the surviving ``training_examples``
        columns plus ``resolved_at``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        all_cols = [r[1] for r in conn.execute("PRAGMA table_info(training_examples)").fetchall()]
        keep_cols = [c for c in all_cols if c not in _NEVER_LOAD_COLS]
        select_list = ", ".join(f"te.{c}" for c in keep_cols)
        df = pl.read_database(
            query=(
                f"SELECT {select_list}, mr.resolved_at "  # noqa: S608 -- col names are derived from PRAGMA, not user input
                "FROM training_examples te "
                "JOIN market_resolutions mr USING (condition_id)"
            ),
            connection=conn,
            batch_size=100_000,
        )
    finally:
        conn.close()

    cast_exprs = [
        *[pl.col(c).cast(pl.Categorical) for c in _CATEGORICAL_CAST_COLS if c in df.columns],
        *[pl.col(c).cast(pl.Int32) for c in _INT32_COLS if c in df.columns],
        *[pl.col(c).cast(pl.Float32) for c in _FLOAT32_COLS if c in df.columns],
    ]
    return df.with_columns(cast_exprs)


def build_feature_matrix(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(X, y, implied_prob)`` numpy arrays from a preprocessed split.

    Drops carrier columns and the ``label_won`` column from ``X``. The
    feature column ordering is the surviving column order on ``df``.
    Polars nulls become ``np.nan`` in float columns -- XGBoost's
    missing-direction rule handles them at split time.

    ``X`` is returned as ``float32`` to halve DMatrix and numpy memory
    versus the polars-default ``float64``; XGBoost converts to float32
    internally regardless.

    Args:
        df: A preprocessed Polars DataFrame (post-drop, post-encoding).

    Returns:
        ``(X, y, implied_prob)`` tuple.
    """
    feature_cols = [c for c in df.columns if c not in (*CARRIER_COLS, "label_won")]
    x_matrix = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    y = df["label_won"].to_numpy()
    implied = df["implied_prob_at_buy"].to_numpy()
    return x_matrix, y, implied
