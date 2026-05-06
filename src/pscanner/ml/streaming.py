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
        ds = StreamingDataset(
            _db_path=db_path,
            _chunk_size=chunk_size,
            _train_markets=train,
            _val_markets=val,
            _test_markets=test,
        )
        yield ds
    finally:
        conn.close()
