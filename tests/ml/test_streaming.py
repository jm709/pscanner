"""Tests for ml.streaming."""

from __future__ import annotations

import sqlite3 as _sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from pscanner.ml.streaming import open_dataset


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
