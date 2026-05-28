"""Integration tests for the backtest event-walk simulator."""

from __future__ import annotations

from scripts.backtest_copy_sizing import (
    EqualWeight,
    Simulator,
)


def test_simulator_initializes_per_scheme_state() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert state.resolved_trades == []
    assert state.cumulative_pnl == 0.0
    assert state.nav_series == []
