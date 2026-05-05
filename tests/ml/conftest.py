"""Shared test fixtures for the ml suite.

``make_synthetic_examples`` builds a Polars DataFrame whose schema
matches a join of ``training_examples`` + ``market_resolutions.resolved_at``.
Used by preprocessing, training, and CLI tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

_BASE_RESOLVED_AT = 1_700_000_000
_DAY_SECONDS = 86_400


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
