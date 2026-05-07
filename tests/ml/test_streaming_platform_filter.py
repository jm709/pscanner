"""Behavioral tests for the streaming platform filter (RFC #35 follow-up).

Each test seeds a corpus with both polymarket and kalshi rows, then verifies
that ``open_dataset`` and the underlying ``_SplitIter`` honor the platform
parameter — no cross-platform leakage in either direction.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pscanner.ml.streaming import open_dataset


def _build_two_platform_db(
    builder: Callable[..., Path],
    *,
    n_markets: int = 10,
    rows_per_market: int = 4,
) -> Path:
    """Seed n_markets * 2 markets — half polymarket, half kalshi."""
    db = builder(
        n_markets=n_markets,
        rows_per_market=rows_per_market,
        seed=0,
        platform="polymarket",
    )
    builder(
        n_markets=n_markets,
        rows_per_market=rows_per_market,
        seed=1,
        platform="kalshi",
        db_path=db,
    )
    return db


def test_open_dataset_defaults_to_polymarket(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """No platform kwarg => row counts equal the polymarket subset only."""
    db = _build_two_platform_db(make_synthetic_examples_db, n_markets=10, rows_per_market=4)
    with open_dataset(db) as ds:
        total = ds.n_train_rows + ds.n_val_rows + ds.n_test_rows
    # Polymarket has 10 markets x 4 rows = 40 rows.
    assert total == 40


def test_open_dataset_filters_to_kalshi(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """`platform='kalshi'` => row counts equal the kalshi subset only."""
    db = _build_two_platform_db(make_synthetic_examples_db, n_markets=10, rows_per_market=4)
    with open_dataset(db, platform="kalshi") as ds:
        total = ds.n_train_rows + ds.n_val_rows + ds.n_test_rows
    assert total == 40


def test_split_iter_does_not_leak_other_platform(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """Iterating dtrain on a polymarket-opened dataset never yields kalshi rows.

    Because the synthetic frame uses different ``condition_id`` namespaces per
    seed (`0xmarket000` etc. for both, but with platform-disjoint membership
    in market_resolutions), the row-count equality is the proxy for no-leak:
    if the WHERE clause leaked, the materialized X would have more rows than
    `n_train_rows`.
    """
    db = _build_two_platform_db(make_synthetic_examples_db, n_markets=10, rows_per_market=4)
    with open_dataset(db, platform="polymarket") as ds:
        dtrain = ds.dtrain(device="cpu")
    # Polymarket has 10 markets x 4 rows = 40 rows; train fraction is 0.7.
    assert dtrain.num_row() == ds.n_train_rows
    expected = round(0.7 * 40)
    assert dtrain.num_row() == expected
