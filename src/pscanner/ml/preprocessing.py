"""Preprocessing for the copy-trade gate training pipeline.

Exposes:

* ``LEAKAGE_COLS`` / ``CARRIER_COLS`` / ``CATEGORICAL_COLS`` -- column
  membership constants documented in the design spec.
* ``_CATEGORICAL_CAST_COLS`` / ``_INT32_COLS`` / ``_FLOAT32_COLS`` /
  ``_NEVER_LOAD_COLS`` / ``_NONE_TOKEN`` -- dtype-narrowing constants
  used by the streaming pipeline.
* ``drop_leakage_cols`` -- pure column removal.
* ``build_feature_matrix`` -- extract ``(X, y, implied_prob)`` numpy
  arrays from a preprocessed split DataFrame.
* ``OneHotEncoder`` -- fit-on-train, transform-each-split. Handles
  the ``__none__`` level for nullable categoricals.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
    "is_high_quality_wallet",
    "market_unique_traders_so_far",
    "market_age_seconds",
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
    "edge_confidence_weighted",
    "win_rate_confidence_weighted",
    "bet_size_relative_to_history",
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


# Columns excluded at SELECT time so the full-fat DataFrame is never
# materialized. ``id`` is the autoincrement primary key, useless for
# training. ``LEAKAGE_COLS`` are dropped downstream by
# ``drop_leakage_cols`` anyway -- excluding them at the SQL boundary
# avoids loading ~1+ GB of hex-string columns (tx_hash, asset_id,
# wallet_address are 42-66 chars x 5M rows each) only to drop them.
_NEVER_LOAD_COLS: frozenset[str] = frozenset({"id", *LEAKAGE_COLS})


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
