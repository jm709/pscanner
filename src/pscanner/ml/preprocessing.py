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

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

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
                df.select(pl.col(col).fill_null(_NONE_TOKEN))
                .to_series()
                .unique()
                .sort()
                .to_list()
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
        df.select(["condition_id", "resolved_at"])
        .unique()
        .sort(["resolved_at", "condition_id"])
    )
    n = markets.height
    n_train = round(train_frac * n)
    n_val = round(val_frac * n)
    train_ids = set(markets["condition_id"].slice(0, n_train).to_list())
    val_ids = set(markets["condition_id"].slice(n_train, n_val).to_list())
    test_ids = set(
        markets["condition_id"].slice(n_train + n_val, n - n_train - n_val).to_list()
    )
    return Split(
        train=df.filter(pl.col("condition_id").is_in(train_ids)),
        val=df.filter(pl.col("condition_id").is_in(val_ids)),
        test=df.filter(pl.col("condition_id").is_in(test_ids)),
    )
