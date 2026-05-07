"""Tests for ml.streaming."""

from __future__ import annotations

import json
import sqlite3 as _sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
import xgboost as xgb

from pscanner.ml.streaming import (
    SplitDataIter,
    _fit_encoder_on_train,
    _partition_markets,
    _SplitIter,
    open_dataset,
)
from pscanner.ml.training import run_study


def test_open_dataset_partitions_markets_by_resolved_at(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """Markets are partitioned 70/15/15 by resolved_at, sorted ascending."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        train = ds._train_markets
        val = ds._val_markets
        test = ds._test_markets

    # 20 markets at 70/15/15 = 14/3/3
    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3

    # Disjoint
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    # Synthetic markets are named 0xmarket{idx:03d} with monotonically
    # increasing resolved_at, so train must contain idx 0-13, val 14-16,
    # test 17-19.
    assert "0xmarket000" in train
    assert "0xmarket013" in train
    assert "0xmarket014" in val
    assert "0xmarket016" in val
    assert "0xmarket017" in test
    assert "0xmarket019" in test


def test_partition_markets_filters_by_platform(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """`_partition_markets` returns only the requested platform's condition_ids.

    Without the WHERE-platform filter the totals would be 8 (4 + 4); the
    per-platform count == 4 proves the filter is in effect. (The synthetic
    fixture reuses ``0xmarket{idx:03d}`` names regardless of seed so the
    polymarket and kalshi condition_id strings overlap — that's a fixture
    artifact, not a real-world condition; in production ``condition_id`` is
    unique per platform but the composite PK ``(platform, condition_id)``
    is what makes the filter mandatory.)
    """
    poly_db = make_synthetic_examples_db(n_markets=4, rows_per_market=2, seed=0)
    # Layer kalshi rows on top of the same DB.
    make_synthetic_examples_db(
        n_markets=4, rows_per_market=2, seed=1, platform="kalshi", db_path=poly_db
    )
    conn = _sqlite3.connect(str(poly_db))
    try:
        train_p, val_p, test_p = _partition_markets(conn, platform="polymarket")
        train_k, val_k, test_k = _partition_markets(conn, platform="kalshi")
    finally:
        conn.close()
    poly_total = len(train_p) + len(val_p) + len(test_p)
    kalshi_total = len(train_k) + len(val_k) + len(test_k)
    assert poly_total == 4, "polymarket has 4 markets"
    assert kalshi_total == 4, "kalshi has 4 markets"


def test_fit_encoder_on_train_filters_by_platform(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """`_fit_encoder_on_train` only sees rows with the requested platform."""
    poly_db = make_synthetic_examples_db(n_markets=4, rows_per_market=2, seed=0)
    make_synthetic_examples_db(
        n_markets=4, rows_per_market=2, seed=1, platform="kalshi", db_path=poly_db
    )
    conn = _sqlite3.connect(str(poly_db))
    try:
        train_poly, _, _ = _partition_markets(conn, platform="polymarket")
        encoder = _fit_encoder_on_train(conn, train_poly, platform="polymarket")
    finally:
        conn.close()
    # The encoder fits over the categorical levels of training_examples joined to
    # the train condition_ids. Even seeding two platforms, the train markets are
    # platform-scoped — encoder.levels reflects exactly the polymarket train rows.
    assert "side" in encoder.levels
    assert set(encoder.levels["side"]).issubset({"YES", "NO"})


def test_open_dataset_closes_pre_pass_connection_on_exit(
    monkeypatch: pytest.MonkeyPatch,
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """The pre-pass sqlite connection is closed when the context exits."""
    db_path = make_synthetic_examples_db(n_markets=10, rows_per_market=5, seed=0)
    real_connect = _sqlite3.connect
    captured: list[_sqlite3.Connection] = []

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    monkeypatch.setattr("pscanner.ml.streaming.sqlite3.connect", tracking_connect)

    with open_dataset(db_path) as ds:
        assert ds._train_markets  # touch attr to ensure pre-pass ran

    # The pre-pass opens exactly one connection; __exit__ closes it.
    assert len(captured) == 1, f"expected 1 connection, got {len(captured)}"
    pre_pass_conn = captured[0]
    with pytest.raises(_sqlite3.ProgrammingError):
        pre_pass_conn.execute("SELECT 1")


def test_encoder_fits_on_train_levels_only(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """OneHotEncoder.levels reflects only train-split categorical levels."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        assert ds.encoder is not None

        # Synthetic encoder always fits 'side', 'top_category', 'market_category'
        assert "side" in ds.encoder.levels
        assert "top_category" in ds.encoder.levels
        assert "market_category" in ds.encoder.levels

        # Side is ('YES', 'NO'); both levels show up given enough rows
        assert set(ds.encoder.levels["side"]).issubset({"YES", "NO"})

        # Encoder.levels values are tuples of strings (deterministic order)
        for _col, lvls in ds.encoder.levels.items():
            assert isinstance(lvls, tuple)
            assert all(isinstance(v, str) for v in lvls)


def test_open_dataset_uses_temp_table_for_split_filter(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """The encoder-fit query joins on a per-connection temp table.

    Confirmed by checking we don't hit SQLite's parameter limit for huge
    splits — synthesize 5,000 markets and assert no OperationalError.
    """
    db_path = make_synthetic_examples_db(n_markets=5_000, rows_per_market=1, seed=0)
    with open_dataset(db_path) as ds:
        # Just touching .encoder forces P2 to have run.
        _ = ds.encoder


def test_open_dataset_reports_row_counts(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """n_train_rows, n_val_rows, n_test_rows match SUM of per-split rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        # 20 markets x 5 rows = 100 total. 70/15/15 split = 70/15/15.
        assert ds.n_train_rows == 70
        assert ds.n_val_rows == 15
        assert ds.n_test_rows == 15


def test_feature_names_excludes_carriers_and_label(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """ds.feature_names is the post-encoding column list, less carriers + label."""
    db_path = make_synthetic_examples_db(n_markets=10, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        names = ds.feature_names
        # Sentinel exclusions
        assert "condition_id" not in names
        assert "trade_ts" not in names
        assert "resolved_at" not in names
        assert "label_won" not in names
        # Cat columns are gone, replaced by indicators
        assert "side" not in names
        assert "side__YES" in names or "side__NO" in names
        # Non-cat numeric column survives
        assert "implied_prob_at_buy" in names


def test_split_iter_yields_expected_chunk_count(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """chunk_size=50 over 70 train rows yields 2 chunks (50 + 20)."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        assert ds.encoder is not None  # narrow for ty
        it = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=50,
        )
        chunks = list(iter(it))

    assert len(chunks) == 2  # 70 train rows / 50 = 2 chunks (50 + 20)
    x0, y0, implied0 = chunks[0]
    assert x0.shape[0] == 50
    assert x0.dtype.name == "float32"
    assert y0.shape == (50,)
    assert implied0.shape == (50,)

    x1, _, _ = chunks[1]
    assert x1.shape[0] == 20  # final partial chunk


def test_split_iter_x_columns_match_feature_names(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """The numpy x matrix has exactly len(feature_names) columns."""
    db_path = make_synthetic_examples_db(n_markets=10, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=100) as ds:
        assert ds.encoder is not None
        it = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=100,
        )
        x, _, _ = next(iter(it))

    assert x.shape[1] == len(ds.feature_names)


def test_split_data_iter_passes_chunks_to_input_data(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """SplitDataIter feeds each chunk into the input_data callback once."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        assert ds.encoder is not None
        source = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=50,
        )
        adapter = SplitDataIter(source)

        captured_chunks = []

        def fake_input_data(*, data, label):
            captured_chunks.append((data.shape[0], label.shape[0]))

        # Drive the iterator until it returns False.
        while adapter.next(fake_input_data):
            pass

        assert len(captured_chunks) == 2  # 50 + 20 over 70 train rows
        assert captured_chunks[0] == (50, 50)
        assert captured_chunks[1] == (20, 20)


def test_split_data_iter_reset_re_iterates(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """reset() lets next() iterate the same SplitIter from the start again."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        assert ds.encoder is not None
        source = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=50,
        )
        adapter = SplitDataIter(source)

        first_pass = []
        while adapter.next(lambda *, data, label: first_pass.append(data.shape[0])):
            pass

        adapter.reset()

        second_pass = []
        while adapter.next(lambda *, data, label: second_pass.append(data.shape[0])):
            pass

        assert first_pass == second_pass


def test_dtrain_returns_quantile_dmatrix_with_expected_shape(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """dtrain() returns a QuantileDMatrix with num_row=n_train_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        dtrain = ds.dtrain(device="cpu")

    assert isinstance(dtrain, xgb.QuantileDMatrix)
    assert dtrain.num_row() == ds.n_train_rows == 70
    assert dtrain.num_col() == len(ds.feature_names)


def test_dval_returns_quantile_dmatrix_with_expected_shape(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """dval() returns a QuantileDMatrix with num_row=n_val_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        dval = ds.dval(device="cpu")

    assert isinstance(dval, xgb.QuantileDMatrix)
    assert dval.num_row() == ds.n_val_rows == 15
    assert dval.num_col() == len(ds.feature_names)


def test_val_aux_returns_y_and_implied_prob_arrays(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """val_aux() returns (y_val, implied_prob_val) of length n_val_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        y_val, implied_val = ds.val_aux()

    assert y_val.shape == (15,)
    assert implied_val.shape == (15,)
    # Labels are 0/1 ints
    assert set(y_val.tolist()).issubset({0, 1})
    # Implied probabilities are in [0, 1]
    assert (implied_val >= 0.0).all()
    assert (implied_val <= 1.0).all()


def test_materialize_test_returns_unencoded_top_categories(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """TestSplit.top_categories is the raw category strings, parallel to .y."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        test = ds.materialize_test()

    assert test.x.shape == (ds.n_test_rows, len(ds.feature_names))
    assert test.x.dtype.name == "float32"
    assert test.y.shape == (ds.n_test_rows,)
    assert test.implied_prob.shape == (ds.n_test_rows,)
    assert test.top_categories.shape == (ds.n_test_rows,)
    # top_categories is unencoded — strings like "sports" / "esports" / "thesis"
    # (or empty string for nulls, mirroring _extract_top_category's fill_null).
    assert test.top_categories.dtype == object
    valid = {"sports", "esports", "thesis", ""}
    assert all(v in valid for v in test.top_categories.tolist())


def test_streaming_pipeline_matches_eager_baseline(
    tmp_path: Path,
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """test_edge from the streaming run matches the eager-path snapshot.

    Tolerance: 0.001 absolute (per #39 DoD). The eager baseline is
    captured at tests/ml/data/eager_baseline.json — see
    tests/ml/_capture_eager_baseline.py.
    """
    baseline_path = Path(__file__).parent / "data" / "eager_baseline.json"
    baseline = json.loads(baseline_path.read_text())

    db_path = make_synthetic_examples_db(
        n_markets=baseline["fixture"]["n_markets"],
        rows_per_market=baseline["fixture"]["rows_per_market"],
        seed=baseline["fixture"]["seed"],
    )
    output_dir = tmp_path / "streaming_run"
    run_study(
        db_path=db_path,
        output_dir=output_dir,
        n_trials=baseline["study"]["n_trials"],
        n_jobs=baseline["study"]["n_jobs"],
        n_min=baseline["study"]["n_min"],
        seed=baseline["study"]["seed"],
    )

    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert abs(metrics["test_edge"] - baseline["test_edge"]) < 0.001, (
        f"test_edge {metrics['test_edge']} drifted from "
        f"eager baseline {baseline['test_edge']} by more than 0.001"
    )
    # Looser tolerances on the calibration metrics — they shift more under
    # quantization changes but are bounded.
    assert abs(metrics["test_accuracy"] - baseline["test_accuracy"]) < 0.05
    assert abs(metrics["test_logloss"] - baseline["test_logloss"]) < 0.10


def test_split_iter_handles_null_to_float_transition_within_chunk(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """Regression: nullable column with all-None leading rows + a real float
    later in the same chunk used to crash ``_encode_chunk``.

    Polars's default ``infer_schema_length=100`` typed an all-null leading
    section as ``pl.Null``, then raised ``ComputeError`` on the first real
    float. The fix passes an explicit schema to ``pl.DataFrame``. Synthetic
    fixtures with ``chunk_size <= 100`` don't trip the boundary; production
    runs at chunk_size=100_000 hit it the moment a chunk had >100 leading
    NULLs in a nullable column.

    100 markets x 2 rows places te.id 1-140 in train (markets 0-69).
    UPDATEs force the mixed-type pattern within a single chunk.
    """
    db_path = make_synthetic_examples_db(n_markets=100, rows_per_market=2, seed=0)

    conn = _sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE training_examples SET win_rate = NULL WHERE id BETWEEN 1 AND 100")
        conn.execute("UPDATE training_examples SET win_rate = 0.5 WHERE id BETWEEN 101 AND 120")
        conn.commit()
    finally:
        conn.close()

    with open_dataset(db_path, chunk_size=200) as ds:
        assert ds.encoder is not None
        it = _SplitIter(
            db_path=ds._db_path,
            condition_ids=ds._train_markets,
            encoder=ds.encoder,
            kept_cols=ds._kept_cols,
            chunk_size=200,
        )
        chunks = list(iter(it))

    assert len(chunks) == 1
    x, _y, _implied = chunks[0]
    assert x.shape[0] == 140
    assert x.dtype.name == "float32"
