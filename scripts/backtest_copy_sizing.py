"""Backtest harness for copy-trading sizing schemes.

Reads the daemon's current watchlist, pulls every position-increasing
(BUY) trade by those wallets from `corpus_trades`, joins with
`market_resolutions`, and walks the chronological event stream
through four sizing schemes. Outputs a markdown report comparing
their realized PnL, ROI, win rate, drawdown, and quarterly P&L.

Spec: docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    """One BUY trade by a watchlist wallet."""

    wallet: str
    condition_id: str
    outcome_side: str  # "YES" or "NO"
    price: float
    notional_usd: float
    ts: int


class EqualWeight:
    """cost = bankroll * position_fraction. No state, no scaling."""

    name = "equal_weight"

    def __init__(self, *, position_fraction: float) -> None:
        """Initialize equal-weight sizing scheme."""
        self._position_fraction = position_fraction

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Return constant position cost regardless of trade details."""
        del trade
        return bankroll * self._position_fraction

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Record resolution outcome (no-op for stateless scheme)."""
        del trade, payout
