"""Tests for ml.streaming."""

from __future__ import annotations

import sqlite3 as _sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
import xgboost as xgb

from pscanner.ml.streaming import SplitDataIter, _SplitIter, open_dataset


def test_open_dataset_partitions_markets_by_resolved_at(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """Markets are partitioned 60/20/20 by resolved_at, sorted ascending."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path) as ds:
        train = ds._train_markets
        val = ds._val_markets
        test = ds._test_markets

    # 20 markets at 60/20/20 = 12/4/4
    assert len(train) == 12
    assert len(val) == 4
    assert len(test) == 4

    # Disjoint
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    # Synthetic markets are named 0xmarket{idx:03d} with monotonically
    # increasing resolved_at, so train must contain idx 0-11, val 12-15,
    # test 16-19.
    assert "0xmarket000" in train
    assert "0xmarket011" in train
    assert "0xmarket012" in val
    assert "0xmarket015" in val
    assert "0xmarket016" in test
    assert "0xmarket019" in test


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
        # 20 markets x 5 rows = 100 total. 60/20/20 split = 60/20/20.
        assert ds.n_train_rows == 60
        assert ds.n_val_rows == 20
        assert ds.n_test_rows == 20


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
    """chunk_size=50 over 60 train rows yields 2 chunks (50 + 10)."""
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

    assert len(chunks) == 2  # 60 train rows / 50 = 2 chunks (50 + 10)
    x0, y0, implied0 = chunks[0]
    assert x0.shape[0] == 50
    assert x0.dtype.name == "float32"
    assert y0.shape == (50,)
    assert implied0.shape == (50,)

    x1, _, _ = chunks[1]
    assert x1.shape[0] == 10  # final partial chunk


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

        assert len(captured_chunks) == 2  # 50 + 10 over 60 train rows
        assert captured_chunks[0] == (50, 50)
        assert captured_chunks[1] == (10, 10)


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
    assert dtrain.num_row() == ds.n_train_rows == 60
    assert dtrain.num_col() == len(ds.feature_names)


def test_dval_returns_quantile_dmatrix_with_expected_shape(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """dval() returns a QuantileDMatrix with num_row=n_val_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        dval = ds.dval(device="cpu")

    assert isinstance(dval, xgb.QuantileDMatrix)
    assert dval.num_row() == ds.n_val_rows == 20
    assert dval.num_col() == len(ds.feature_names)


def test_val_aux_returns_y_and_implied_prob_arrays(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """val_aux() returns (y_val, implied_prob_val) of length n_val_rows."""
    db_path = make_synthetic_examples_db(n_markets=20, rows_per_market=5, seed=0)

    with open_dataset(db_path, chunk_size=50) as ds:
        y_val, implied_val = ds.val_aux()

    assert y_val.shape == (20,)
    assert implied_val.shape == (20,)
    # Labels are 0/1 ints
    assert set(y_val.tolist()).issubset({0, 1})
    # Implied probabilities are in [0, 1]
    assert (implied_val >= 0.0).all()
    assert (implied_val <= 1.0).all()
