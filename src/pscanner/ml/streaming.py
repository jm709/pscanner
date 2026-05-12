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
from collections.abc import Callable as _Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

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

_TRAIN_FRAC = 0.7
_VAL_FRAC = 0.15


@dataclass(frozen=True)
class TestSplit:
    """Materialized test split for ``evaluate_on_test`` + ``analyze_model.py``."""

    x: np.ndarray  # float32, shape (n_test_rows, n_features)
    y: np.ndarray  # int32 labels
    implied_prob: np.ndarray  # float32
    top_categories: np.ndarray  # object (str), unencoded — for per-category breakdowns
    total_volume_usd: np.ndarray  # float32 (n,) — for per-volume-bucket breakdowns (#109)


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
    _platform: str = "polymarket"

    def dtrain(self, *, device: str) -> xgb.QuantileDMatrix:
        """Build a QuantileDMatrix for the train split."""
        return self._build_dmatrix(self._train_markets, device=device, ref=None)

    def dval(
        self,
        *,
        device: str,
        ref: xgb.QuantileDMatrix | None = None,
    ) -> xgb.QuantileDMatrix:
        """Build a QuantileDMatrix for the val split.

        Args:
            device: XGBoost device string (forwarded to booster, not DMatrix).
            ref: Optional reference DMatrix (typically ``dtrain``). When
                supplied, XGBoost reuses the quantile cuts from the reference
                so train and val share the same bin boundaries — required when
                passing a ``QuantileDMatrix`` as an eval set to ``xgb.train``.
        """
        return self._build_dmatrix(self._val_markets, device=device, ref=ref)

    def val_aux(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_val, implied_prob_val)`` numpy arrays.

        Streamed via the same chunked path as ``dval`` but pulls only
        the two columns the edge-metric closure needs, into pre-allocated
        arrays sized from the P3 row count. No feature matrix is built.
        """
        y = np.empty(self.n_val_rows, dtype=np.int32)
        implied = np.empty(self.n_val_rows, dtype=np.float32)

        sql = (
            "SELECT te.label_won, te.implied_prob_at_buy "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._val_markets)
            cursor = conn.execute(sql, (self._platform,))
            offset = 0
            while True:
                rows = cursor.fetchmany(self._chunk_size)
                if not rows:
                    break
                end = offset + len(rows)
                for i, (label, prob) in enumerate(rows):
                    y[offset + i] = int(label)
                    implied[offset + i] = float(prob)
                offset = end
        finally:
            conn.close()

        return y, implied

    def materialize_test(self) -> TestSplit:
        """Stream the test split into pre-allocated numpy arrays.

        Uses the same _SplitIter as dtrain/dval for X/y/implied; pulls
        the unencoded ``top_category`` column via a parallel SELECT for
        per-category metrics in evaluate_on_test.
        """
        if self.encoder is None:
            raise RuntimeError("StreamingDataset.encoder is None; was open_dataset used?")

        x = np.empty((self.n_test_rows, len(self.feature_names)), dtype=np.float32)
        y = np.empty(self.n_test_rows, dtype=np.int32)
        implied = np.empty(self.n_test_rows, dtype=np.float32)

        source = _SplitIter(
            db_path=self._db_path,
            condition_ids=self._test_markets,
            encoder=self.encoder,
            kept_cols=self._kept_cols,
            chunk_size=self._chunk_size,
            platform=self._platform,
        )
        offset = 0
        for x_chunk, y_chunk, implied_chunk in iter(source):
            end = offset + x_chunk.shape[0]
            x[offset:end] = x_chunk
            y[offset:end] = y_chunk
            implied[offset:end] = implied_chunk
            offset = end

        # Parallel small SELECT for unencoded top_category strings.
        # Mirrors the deleted _extract_top_category: nulls become "".
        # Bulk fetchall + np.array is multiple orders of magnitude faster than
        # the per-row loop pattern at corpus scale (~2M rows takes seconds vs
        # ~10+ minutes for the row-at-a-time numpy assignment).
        sql = (
            "SELECT COALESCE(te.top_category, '') "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._test_markets)
            rows = conn.execute(sql, (self._platform,)).fetchall()
        finally:
            conn.close()
        top_categories = np.array([r[0] for r in rows], dtype=object)

        # Parallel small SELECT for total_volume_usd, JOINed to corpus_markets
        # via (platform, condition_id). Used by per_volume_bucket_edge_breakdown
        # (#109) to stratify the test edge by market lifetime volume.
        sql_volume = (
            "SELECT COALESCE(cm.total_volume_usd, 0.0) "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "LEFT JOIN corpus_markets cm "
            "  ON cm.condition_id = te.condition_id "
            " AND cm.platform = te.platform "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._test_markets)
            volume_rows = conn.execute(sql_volume, (self._platform,)).fetchall()
        finally:
            conn.close()
        total_volume_usd = np.array([r[0] for r in volume_rows], dtype=np.float32)

        return TestSplit(
            x=x,
            y=y,
            implied_prob=implied,
            top_categories=top_categories,
            total_volume_usd=total_volume_usd,
        )

    def _build_dmatrix(
        self,
        condition_ids: frozenset[str],
        *,
        device: str,  # forwarded to booster via xgb.train params, not DMatrix constructor
        ref: xgb.QuantileDMatrix | None,
    ) -> xgb.QuantileDMatrix:
        if self.encoder is None:
            raise RuntimeError("StreamingDataset.encoder is None; was open_dataset used?")
        source = _SplitIter(
            db_path=self._db_path,
            condition_ids=condition_ids,
            encoder=self.encoder,
            kept_cols=self._kept_cols,
            chunk_size=self._chunk_size,
            platform=self._platform,
        )
        kwargs: dict[str, object] = {"max_bin": 256}
        if ref is not None:
            kwargs["ref"] = ref
        return xgb.QuantileDMatrix(SplitDataIter(source), **kwargs)


def _partition_markets(
    conn: sqlite3.Connection,
    *,
    platform: str = "polymarket",
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Run P1: SELECT condition_id, resolved_at FROM market_resolutions ORDER BY...

    Slice the sorted list at 70% / 85% into train, val, test. Filters
    ``market_resolutions`` by ``platform`` so each split is platform-scoped;
    after RFC #35 PR A, ``condition_id`` is no longer unique across
    platforms (composite PK), so the filter is required for correctness.
    """
    rows = conn.execute(
        "SELECT condition_id, resolved_at FROM market_resolutions "
        "WHERE platform = ? "
        "ORDER BY resolved_at, condition_id",
        (platform,),
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

    Single-platform note: the table stores ``condition_id`` only because
    each training run is single-platform. Multi-platform aggregation
    (future follow-up) needs ``(platform, condition_id)`` tuples here and
    a corresponding JOIN tightening to ``USING (platform, condition_id)``
    in ``_SplitIter`` / ``val_aux`` / ``materialize_test``.
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
    *,
    platform: str = "polymarket",
) -> OneHotEncoder:
    """Run P2: fit a OneHotEncoder on the train split's categorical levels.

    SELECTs DISTINCT side, top_category from training_examples joined on the
    _p2_train temp table. ``market_category`` was dropped from the categorical
    set in #122 — it's replaced by 9 pre-binarized ``cat_*`` indicator columns
    that bypass the encoder entirely. Belt-and-suspenders ``WHERE platform = ?``
    is added because after RFC #35 PR A, two platforms can share the same
    ``condition_id`` string; the temp-table partition isolates split membership,
    the platform predicate isolates platform membership.
    """
    _populate_temp_table(conn, "_p2_train", train_markets)
    rows = conn.execute(
        "SELECT DISTINCT side, top_category "
        "FROM training_examples te "
        "JOIN _p2_train tm USING (condition_id) "
        "WHERE te.platform = ?",
        (platform,),
    ).fetchall()
    df = pl.DataFrame(
        rows,
        schema={
            "side": pl.String,
            "top_category": pl.String,
        },
        orient="row",
    )
    return OneHotEncoder.fit(df, columns=CATEGORICAL_COLS)


def _count_split_rows(
    conn: sqlite3.Connection,
    train: frozenset[str],
    val: frozenset[str],
    test: frozenset[str],
    *,
    platform: str = "polymarket",
) -> tuple[int, int, int]:
    """Run P3: COUNT(*) per split via temp tables.

    ``WHERE te.platform = ?`` is belt-and-suspenders against cross-platform
    ``condition_id`` collisions (see ``_fit_encoder_on_train`` docstring).
    """
    counts = []
    for label, markets in (("_p3_train", train), ("_p3_val", val), ("_p3_test", test)):
        _populate_temp_table(conn, label, markets)
        (n,) = conn.execute(
            f"SELECT COUNT(*) FROM training_examples te "  # noqa: S608 -- label is a literal
            f"JOIN {label} sm USING (condition_id) "
            "WHERE te.platform = ?",
            (platform,),
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

    The ``platform`` field gates both the JOIN to ``market_resolutions``
    (composite key after RFC #35 PR A) and the ``WHERE te.platform = ?``
    predicate on ``training_examples``. Defaults to ``"polymarket"`` to
    preserve existing test fixtures that construct ``_SplitIter`` directly.
    """

    db_path: Path
    condition_ids: frozenset[str]
    encoder: OneHotEncoder
    kept_cols: tuple[str, ...]
    chunk_size: int
    platform: str = "polymarket"

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        select_list = ", ".join(f"te.{c}" for c in self.kept_cols)
        sql = (
            f"SELECT {select_list}, mr.resolved_at "  # noqa: S608 -- kept_cols derived from PRAGMA
            "FROM training_examples te "
            "JOIN market_resolutions mr "
            "  ON mr.platform = te.platform AND mr.condition_id = te.condition_id "
            "JOIN _split_markets sm USING (condition_id) "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        col_names = (*self.kept_cols, "resolved_at")
        conn = sqlite3.connect(str(self.db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self.condition_ids)
            cursor = conn.execute(sql, (self.platform,))
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
        df = pl.DataFrame(rows, schema=_chunk_schema(col_names), orient="row")
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


def _chunk_schema(col_names: tuple[str, ...]) -> dict[str, pl.DataType]:
    """Build an explicit Polars schema for a chunked SELECT result.

    Polars's default ``infer_schema_length=100`` mis-types nullable numeric
    columns as ``pl.Null`` when the first 100 rows of a chunk are all
    ``None``, then raises ``ComputeError`` on subsequent real values. An
    explicit schema removes the dependency on row content. Categories
    mirror preprocessing.py's dtype-cast tuples; ``_encode_chunk``'s
    ``cast_exprs`` then narrow Int64 → Int32 and Float64 → Float32.
    """
    schema: dict[str, pl.DataType] = {}
    for c in col_names:
        if c in _INT32_COLS:
            schema[c] = pl.Int64()
        elif c in _FLOAT32_COLS:
            schema[c] = pl.Float64()
        elif c in _CATEGORICAL_CAST_COLS:
            schema[c] = pl.String()
        elif c in ("trade_ts", "resolved_at"):
            schema[c] = pl.Int64()  # SQLite INTEGER unix timestamps
        else:
            # condition_id and any other carrier strings (LEAKAGE_COLS are
            # already excluded at SELECT time via _NEVER_LOAD_COLS).
            schema[c] = pl.String()
    return schema


class SplitDataIter(xgb.DataIter):
    """XGBoost ``DataIter`` adapter over a :class:`_SplitIter`.

    ``release_data=True`` lets XGBoost free each chunk after ingestion;
    in-flight working set per split is one chunk + XGBoost's quantization
    buffer instead of the full numpy matrix.
    """

    def __init__(self, source: _SplitIter) -> None:
        """Wrap *source* in xgboost's DataIter protocol."""
        super().__init__(release_data=True)
        self._source = source
        self._iter: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    def next(self, input_data: _Callable[..., None]) -> bool:
        """Pull one chunk; return False when exhausted."""
        if self._iter is None:
            self._iter = iter(self._source)
        try:
            x, y, _implied = next(self._iter)
        except StopIteration:
            self._iter = None
            return False
        input_data(data=x, label=y)
        return True

    def reset(self) -> None:
        """Drop the iterator so the next ``next()`` reopens the cursor."""
        self._iter = None


@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
    platform: str = "polymarket",
) -> Iterator[StreamingDataset]:
    """Open the corpus for streaming training.

    Args:
        db_path: Path to the corpus SQLite database.
        chunk_size: Rows per chunk fed into xgboost's DataIter. Default
            100_000; see Issue #39 for the memory / overhead trade-off.
        platform: RFC #35 PR A platform tag. Filters every SELECT against
            ``training_examples`` and ``market_resolutions`` to rows with
            this tag. Defaults to ``"polymarket"`` so existing callers
            (tests, ``scripts/analyze_model.py``, internal callers) see
            no behavior change. Single-platform-per-run by design;
            multi-platform aggregation widens this to a sequence type.

    Yields:
        A :class:`StreamingDataset` whose pre-scan has completed.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        train, val, test = _partition_markets(conn, platform=platform)
        encoder = _fit_encoder_on_train(conn, train, platform=platform)
        n_train, n_val, n_test = _count_split_rows(conn, train, val, test, platform=platform)
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
            _platform=platform,
        )
        yield ds
    finally:
        conn.close()
