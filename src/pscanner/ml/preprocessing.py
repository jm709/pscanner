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

import polars as pl

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
