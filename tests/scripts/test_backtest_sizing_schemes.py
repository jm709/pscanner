"""Unit tests for backtest sizing schemes."""

from __future__ import annotations

import pytest
from scripts.backtest_copy_sizing import ConcentrationCapped, EqualWeight, Trade


def _trade(
    *,
    wallet: str = "0xwallet",
    condition_id: str = "0xcond",
    outcome_side: str = "YES",
    price: float = 0.5,
    notional_usd: float = 1000.0,
    ts: int = 1_700_000_000,
) -> Trade:
    return Trade(
        wallet=wallet,
        condition_id=condition_id,
        outcome_side=outcome_side,
        price=price,
        notional_usd=notional_usd,
        ts=ts,
    )


def test_equal_weight_returns_constant_cost() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    cost = scheme.compute(_trade(), bankroll=10_000.0)
    assert cost == 100.0


def test_equal_weight_ignores_trade_details() -> None:
    scheme = EqualWeight(position_fraction=0.02)
    cost_a = scheme.compute(_trade(price=0.1, notional_usd=50.0), bankroll=5_000.0)
    cost_b = scheme.compute(_trade(price=0.9, notional_usd=500.0), bankroll=5_000.0)
    assert cost_a == cost_b == 100.0


def test_concentration_capped_first_trade_per_wallet_is_unit_multiplier() -> None:
    scheme = ConcentrationCapped(
        position_fraction=0.01, min_multiplier=0.10, watchlist_size=10
    )
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    assert cost == 100.0


def test_concentration_capped_decays_with_repeat_offender() -> None:
    scheme = ConcentrationCapped(
        position_fraction=0.01, min_multiplier=0.10, watchlist_size=10
    )
    scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    scheme.compute(_trade(wallet="0xB"), bankroll=10_000.0)
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    assert cost == pytest.approx(10_000.0 * 0.01 * 0.15)


def test_concentration_capped_floors_at_min_multiplier() -> None:
    scheme = ConcentrationCapped(
        position_fraction=0.01, min_multiplier=0.25, watchlist_size=100
    )
    for _ in range(50):
        scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    assert cost == pytest.approx(10_000.0 * 0.01 * 0.25)
