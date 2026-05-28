"""Integration tests for the backtest event-walk simulator."""

from __future__ import annotations

from scripts.backtest_copy_sizing import (
    EdgeWeightedCausal,
    EqualWeight,
    Resolution,
    Simulator,
    Trade,
)


def test_simulator_initializes_per_scheme_state() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert state.resolved_trades == []
    assert state.cumulative_pnl == 0.0
    assert state.nav_series == []


def test_simulator_books_pnl_on_resolution() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert len(state.resolved_trades) == 1
    rec = state.resolved_trades[0]
    assert rec.payout == 1.0
    assert rec.proceeds == 200.0
    assert rec.pnl == 100.0
    assert state.cumulative_pnl == 100.0
    assert state.nav_series == [(200, 100.0)]


def test_simulator_books_zero_payout_on_losing_outcome() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="NO", resolved_at=200)
    )
    rec = sim.state_for(scheme).resolved_trades[0]
    assert rec.payout == 0.0
    assert rec.pnl == -100.0


def test_simulator_temporal_correctness_with_edge_weighted() -> None:
    """Resolutions must surface to EdgeWeightedCausal before any later trade.

    Scenario: 2 prior wins for wallet A at price 0.5 (edge=+0.5/trade),
    then a 3rd trade for A. After the 2 resolutions, the 3rd trade
    must be sized at the boosted multiplier.
    """
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=2,
    )
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM2",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=200,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=300)
    )
    sim.on_resolution(
        Resolution(condition_id="0xM2", winning_side="YES", resolved_at=400)
    )
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM3",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=500,
        )
    )
    state = sim.state_for(scheme)
    third = state.open_positions[next(iter(state.open_positions))]
    assert third.cost == 300.0


def test_simulator_resolves_multiple_open_positions_on_same_market() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xB",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=150,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert len(state.resolved_trades) == 2


def test_simulator_unresolved_trade_stays_open() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 1
    assert state.resolved_trades == []
