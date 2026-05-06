"""Streaming corpus load for the training pipeline.

Replaces the eager ``preprocessing.load_dataset`` / ``temporal_split`` path
with a two-pass architecture: a small pre-pass at ``__enter__`` computes
the temporal split partition, the encoder fit, and per-split row counts;
per-split chunked SELECTs are then fed into ``xgb.QuantileDMatrix`` via
``xgb.DataIter``.

Public API:

* :func:`open_dataset` -- context manager returning a :class:`StreamingDataset`.
* :class:`StreamingDataset` -- exposes ``dtrain``/``dval``/``val_aux``/``materialize_test``.
* :class:`TestSplit` -- materialized test split for ``evaluate_on_test``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from pscanner.ml.preprocessing import (
    _CATEGORICAL_CAST_COLS,
    _FLOAT32_COLS,
    _INT32_COLS,
    _NEVER_LOAD_COLS,
    CARRIER_COLS,
    CATEGORICAL_COLS,
    OneHotEncoder,
    build_feature_matrix,
    drop_leakage_cols,
)

_TRAIN_FRAC = 0.6
_VAL_FRAC = 0.2


@dataclass
class StreamingDataset:
    """Two-pass streaming view over training_examples.

    Constructed by :func:`open_dataset`. Pre-scan results live on the
    instance; per-split chunked reads are deferred until ``dtrain`` /
    ``dval`` / ``val_aux`` / ``materialize_test`` is called.
    """

    _db_path: Path
    _chunk_size: int
    _train_markets: frozenset[str] = field(default_factory=frozenset)
    _val_markets: frozenset[str] = field(default_factory=frozenset)
    _test_markets: frozenset[str] = field(default_factory=frozenset)
    encoder: OneHotEncoder | None = None
    feature_names: tuple[str, ...] = ()
    _kept_cols: tuple[str, ...] = ()
    n_train_rows: int = 0
    n_val_rows: int = 0
    n_test_rows: int = 0


def _partition_markets(
    conn: sqlite3.Connection,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Run P1: SELECT condition_id, resolved_at FROM market_resolutions ORDER BY...

    Slice the sorted list at 60% / 80% into train, val, test.
    """
    rows = conn.execute(
        "SELECT condition_id, resolved_at FROM market_resolutions "
        "ORDER BY resolved_at, condition_id"
    ).fetchall()
    n = len(rows)
    n_train = round(_TRAIN_FRAC * n)
    n_val = round(_VAL_FRAC * n)
    train = frozenset(r[0] for r in rows[:n_train])
    val = frozenset(r[0] for r in rows[n_train : n_train + n_val])
    test = frozenset(r[0] for r in rows[n_train + n_val :])
    return train, val, test


def _populate_temp_table(
    conn: sqlite3.Connection,
    table_name: str,
    condition_ids: frozenset[str],
) -> None:
    """Create + populate a per-connection TEMP TABLE for split-membership joins.

    Used in place of an ``IN (?, ?, ...)`` parameterized query so the
    SQLite ``SQLITE_MAX_VARIABLE_NUMBER`` limit (32766 in 3.32+) doesn't
    bite as the corpus grows. Cost: ~10K INSERTs at startup, < 100 ms.
    """
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TEMP TABLE {table_name} (condition_id TEXT PRIMARY KEY)")
    conn.executemany(
        f"INSERT INTO {table_name} VALUES (?)",  # noqa: S608 -- table_name is a module-internal literal
        [(cid,) for cid in condition_ids],
    )


def _fit_encoder_on_train(
    conn: sqlite3.Connection,
    train_markets: frozenset[str],
) -> OneHotEncoder:
    """Run P2: fit a OneHotEncoder on the train split's categorical levels.

    SELECTs DISTINCT side, top_category, market_category from training_examples
    joined on the _p2_train temp table.
    """
    _populate_temp_table(conn, "_p2_train", train_markets)
    rows = conn.execute(
        "SELECT DISTINCT side, top_category, market_category "
        "FROM training_examples te "
        "JOIN _p2_train tm USING (condition_id)"
    ).fetchall()
    df = pl.DataFrame(
        rows,
        schema={
            "side": pl.String,
            "top_category": pl.String,
            "market_category": pl.String,
        },
        orient="row",
    )
    return OneHotEncoder.fit(df, columns=CATEGORICAL_COLS)


def _count_split_rows(
    conn: sqlite3.Connection,
    train: frozenset[str],
    val: frozenset[str],
    test: frozenset[str],
) -> tuple[int, int, int]:
    """Run P3: COUNT(*) per split via temp tables."""
    counts = []
    for label, markets in (("_p3_train", train), ("_p3_val", val), ("_p3_test", test)):
        _populate_temp_table(conn, label, markets)
        (n,) = conn.execute(
            f"SELECT COUNT(*) FROM training_examples te "  # noqa: S608 -- label is a literal
            f"JOIN {label} sm USING (condition_id)"
        ).fetchone()
        counts.append(int(n))
    return counts[0], counts[1], counts[2]


def _kept_columns(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return training_examples columns minus _NEVER_LOAD_COLS, in PRAGMA order.

    Equivalent to the SELECT-list construction in the deleted load_dataset.
    """
    rows = conn.execute("PRAGMA table_info(training_examples)").fetchall()
    return tuple(r[1] for r in rows if r[1] not in _NEVER_LOAD_COLS)


def _derive_feature_names(
    kept_cols: tuple[str, ...],
    encoder: OneHotEncoder,
) -> tuple[str, ...]:
    """Compute the post-encoding column list, less carriers and label.

    Mirrors the deleted ``temporal_split`` + ``encoder.transform`` +
    ``build_feature_matrix`` pipeline analytically. Encoder.transform appends
    ``{col}__{level}`` indicators for each categorical column and drops the
    original; non-categorical columns keep their original SELECT order.
    """
    excluded = {*CARRIER_COLS, "label_won"}
    non_cat = [c for c in kept_cols if c not in encoder.levels]
    # resolved_at gets joined in by the SELECT (not in PRAGMA).
    if "resolved_at" not in non_cat:
        non_cat.append("resolved_at")
    indicators = [f"{col}__{lvl}" for col, lvls in encoder.levels.items() for lvl in lvls]
    return tuple(c for c in [*non_cat, *indicators] if c not in excluded)


@dataclass
class _SplitIter:
    """Yields (x, y, implied) numpy tuples per chunk for one split.

    Each iter() opens a fresh sqlite3.Connection (XGBoost's DataIter may
    iterate from worker threads; sqlite3 connections aren't thread-safe).
    The connection's TEMP TABLE is populated from condition_ids on first
    iteration; the connection closes when iteration finishes or raises.
    """

    db_path: Path
    condition_ids: frozenset[str]
    encoder: OneHotEncoder
    kept_cols: tuple[str, ...]
    chunk_size: int

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        select_list = ", ".join(f"te.{c}" for c in self.kept_cols)
        sql = (
            f"SELECT {select_list}, mr.resolved_at "  # noqa: S608 -- kept_cols derived from PRAGMA
            "FROM training_examples te "
            "JOIN market_resolutions mr USING (condition_id) "
            "JOIN _split_markets sm USING (condition_id) "
            "ORDER BY te.id"
        )
        col_names = (*self.kept_cols, "resolved_at")
        conn = sqlite3.connect(str(self.db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self.condition_ids)
            cursor = conn.execute(sql)
            while True:
                rows = cursor.fetchmany(self.chunk_size)
                if not rows:
                    return
                yield self._encode_chunk(rows, col_names)
        finally:
            conn.close()

    def _encode_chunk(
        self,
        rows: list[tuple[object, ...]],
        col_names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = pl.DataFrame(rows, schema=list(col_names), orient="row")
        # Mirror load_dataset's dtype casting (preserved from preprocessing.py).
        cast_exprs = [
            *[pl.col(c).cast(pl.Categorical) for c in _CATEGORICAL_CAST_COLS if c in df.columns],
            *[pl.col(c).cast(pl.Int32) for c in _INT32_COLS if c in df.columns],
            *[pl.col(c).cast(pl.Float32) for c in _FLOAT32_COLS if c in df.columns],
        ]
        df = df.with_columns(cast_exprs)
        df = drop_leakage_cols(df)  # idempotent; no-op when SELECT already excluded them
        df = self.encoder.transform(df)
        return build_feature_matrix(df)


@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
) -> Iterator[StreamingDataset]:
    """Open the corpus for streaming training.

    Args:
        db_path: Path to the corpus SQLite database.
        chunk_size: Rows per chunk fed into xgboost's DataIter. Default
            100_000; see Issue #39 for the memory / overhead trade-off.

    Yields:
        A :class:`StreamingDataset` whose pre-scan has completed.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        train, val, test = _partition_markets(conn)
        encoder = _fit_encoder_on_train(conn, train)
        n_train, n_val, n_test = _count_split_rows(conn, train, val, test)
        kept = _kept_columns(conn)
        ds = StreamingDataset(
            _db_path=db_path,
            _chunk_size=chunk_size,
            _train_markets=train,
            _val_markets=val,
            _test_markets=test,
            encoder=encoder,
            feature_names=_derive_feature_names(kept, encoder),
            _kept_cols=kept,
            n_train_rows=n_train,
            n_val_rows=n_val,
            n_test_rows=n_test,
        )
        yield ds
    finally:
        conn.close()
