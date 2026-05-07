"""Shared test fixtures for the ml suite.

``make_synthetic_examples`` builds a Polars DataFrame whose schema
matches a join of ``training_examples`` + ``market_resolutions.resolved_at``.
Used by preprocessing, training, and CLI tests.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pscanner.corpus.db import init_corpus_db

_BASE_RESOLVED_AT = 1_700_000_000
_DAY_SECONDS = 86_400


def _seed_db_from_synthetic(
    conn: sqlite3.Connection,
    df: pl.DataFrame,
    *,
    platform: str = "polymarket",
) -> None:
    """Populate corpus_markets, market_resolutions, training_examples from
    a synthetic-examples Polars frame so load_dataset / open_dataset see
    matching rows."""
    markets = df.select(["condition_id", "resolved_at"]).unique()
    for row in markets.iter_rows(named=True):
        conn.execute(
            """
            INSERT INTO corpus_markets (
              platform, condition_id, event_slug, category, closed_at,
              total_volume_usd, market_slug, backfill_state, enumerated_at
            ) VALUES (?, ?, '', 'sports', ?, 1000.0, '', 'complete', ?)
            """,
            (
                platform,
                row["condition_id"],
                int(row["resolved_at"]),
                int(row["resolved_at"]) - 1,
            ),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, ?, 0, 1, ?, 'gamma', ?)
            """,
            (
                platform,
                row["condition_id"],
                int(row["resolved_at"]),
                int(row["resolved_at"]),
            ),
        )
    examples = df.drop("resolved_at")
    for raw_row in examples.iter_rows(named=True):
        # Inject platform if the synthetic frame didn't set it (most tests).
        row = raw_row if "platform" in raw_row else {"platform": platform, **raw_row}
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO training_examples ({cols}) VALUES ({placeholders})",  # noqa: S608 -- column names are statically derived from synthetic frame
            tuple(row.values()),
        )
    conn.commit()


def _make_synthetic_examples(
    n_markets: int = 30,
    rows_per_market: int = 20,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a synthetic ``training_examples`` DataFrame for tests.

    Markets resolve at evenly spaced timestamps over ~60 days so the
    temporal split has clean cutoffs. ``label_won`` is computed
    consistently with ``side`` and the market's outcome to keep the
    edge metric meaningful.

    Args:
        n_markets: Distinct ``condition_id`` values to generate.
        rows_per_market: BUYs per market.
        seed: Numpy RNG seed for reproducibility.

    Returns:
        Polars DataFrame with all 38 ``training_examples`` columns plus
        ``resolved_at`` (joined from ``market_resolutions``).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for m_idx in range(n_markets):
        cond_id = f"0xmarket{m_idx:03d}"
        resolved_at = _BASE_RESOLVED_AT + (m_idx * 60 * _DAY_SECONDS // n_markets)
        outcome_yes = bool(rng.integers(0, 2))
        for i in range(rows_per_market):
            implied_prob = float(rng.uniform(0.1, 0.95))
            side = "YES" if rng.random() < 0.6 else "NO"
            won_yes_market = side == "YES"
            won = won_yes_market if outcome_yes else (side == "NO")
            top_cat_idx = int(rng.integers(0, 4))
            top_cat_options: tuple[str | None, ...] = ("sports", "esports", "thesis", None)
            top_cat = top_cat_options[top_cat_idx]
            rows.append(
                {
                    "tx_hash": f"0xtx{m_idx:03d}{i:02d}",
                    "asset_id": f"asset{m_idx}{i}",
                    "wallet_address": f"0xwallet{int(rng.integers(0, 50)):03d}",
                    "condition_id": cond_id,
                    "trade_ts": resolved_at - int(rng.integers(_DAY_SECONDS, _DAY_SECONDS * 30)),
                    "built_at": _BASE_RESOLVED_AT,
                    "prior_trades_count": int(rng.integers(0, 100)),
                    "prior_buys_count": int(rng.integers(0, 80)),
                    "prior_resolved_buys": int(rng.integers(0, 50)),
                    "prior_wins": int(rng.integers(0, 30)),
                    "prior_losses": int(rng.integers(0, 20)),
                    "win_rate": (float(rng.uniform(0, 1)) if rng.random() < 0.8 else None),
                    "avg_implied_prob_paid": (
                        float(rng.uniform(0.3, 0.9)) if rng.random() < 0.8 else None
                    ),
                    "realized_edge_pp": (
                        float(rng.uniform(-0.2, 0.3)) if rng.random() < 0.6 else None
                    ),
                    "prior_realized_pnl_usd": float(rng.normal(0, 1000)),
                    "avg_bet_size_usd": (
                        float(rng.uniform(20, 1000)) if rng.random() < 0.8 else None
                    ),
                    "median_bet_size_usd": (
                        float(rng.uniform(20, 800)) if rng.random() < 0.8 else None
                    ),
                    "wallet_age_days": float(rng.uniform(0, 365)),
                    "seconds_since_last_trade": (
                        int(rng.integers(0, _DAY_SECONDS)) if rng.random() < 0.9 else None
                    ),
                    "prior_trades_30d": int(rng.integers(0, 30)),
                    "top_category": top_cat,
                    "category_diversity": int(rng.integers(0, 4)),
                    "bet_size_usd": float(rng.uniform(10, 500)),
                    "bet_size_rel_to_avg": (
                        float(rng.uniform(0.5, 3.0)) if rng.random() < 0.8 else None
                    ),
                    "edge_confidence_weighted": float(rng.uniform(0, 1)),
                    "win_rate_confidence_weighted": float(rng.uniform(0, 1)),
                    "is_high_quality_wallet": int(rng.integers(0, 2)),
                    "bet_size_relative_to_history": float(rng.uniform(0.5, 2.0)),
                    "side": side,
                    "implied_prob_at_buy": implied_prob,
                    "market_category": str(rng.choice(["sports", "esports", "thesis", "unknown"])),
                    "market_volume_so_far_usd": float(rng.uniform(1000, 1e6)),
                    "market_unique_traders_so_far": int(rng.integers(1, 500)),
                    "market_age_seconds": int(rng.integers(60, _DAY_SECONDS * 30)),
                    "time_to_resolution_seconds": int(rng.integers(60, _DAY_SECONDS * 30)),
                    "last_trade_price": (
                        float(rng.uniform(0.05, 0.95)) if rng.random() < 0.95 else None
                    ),
                    "price_volatility_recent": (
                        float(rng.uniform(0, 0.1)) if rng.random() < 0.95 else None
                    ),
                    "label_won": int(won),
                    "resolved_at": resolved_at,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def make_synthetic_examples() -> Callable[..., pl.DataFrame]:
    """Return the synthetic-examples builder."""
    return _make_synthetic_examples


@pytest.fixture
def make_synthetic_examples_db(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> Callable[..., Path]:
    """Return a builder: (n_markets, rows_per_market, seed) -> Path to populated SQLite."""

    def _build(
        *,
        n_markets: int = 30,
        rows_per_market: int = 20,
        seed: int = 0,
        platform: str = "polymarket",
        db_path: Path | None = None,
    ) -> Path:
        df = make_synthetic_examples(
            n_markets=n_markets, rows_per_market=rows_per_market, seed=seed
        )
        if db_path is None:
            db_path = (
                tmp_path / f"corpus_n{n_markets}_r{rows_per_market}_s{seed}_{platform}.sqlite3"
            )
        # init_corpus_db is idempotent; re-opening lets a caller layer a second
        # platform's rows onto a corpus the helper already created.
        conn = init_corpus_db(db_path)
        try:
            _seed_db_from_synthetic(conn, df, platform=platform)
        finally:
            conn.close()
        return db_path

    return _build
