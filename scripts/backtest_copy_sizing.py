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


class ConcentrationCapped:
    """Production sizing: cost = bankroll * position_fraction * mult.

    Mirrors ``pscanner.strategies.evaluators.subgraph_copy
    .SubgraphCopyEvaluator._concentration_multiplier``. The running
    count is incremented AFTER the size is computed so the first trade
    per wallet always gets a clean 1.0 multiplier.
    """

    name = "concentration_capped"

    def __init__(
        self,
        *,
        position_fraction: float,
        min_multiplier: float,
        watchlist_size: int,
    ) -> None:
        """Initialize concentration-capped sizing scheme."""
        self._position_fraction = position_fraction
        self._min_multiplier = min_multiplier
        self._watchlist_size = max(1, watchlist_size)
        self._counts: dict[str, int] = {}

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Size the trade, decaying the multiplier as wallet share rises."""
        total = sum(self._counts.values())
        share = self._counts.get(trade.wallet, 0) / total if total else 0.0
        target_share = 1.0 / self._watchlist_size
        raw = min(1.0, target_share / max(share, target_share))
        mult = max(raw, self._min_multiplier)
        self._counts[trade.wallet] = self._counts.get(trade.wallet, 0) + 1
        return bankroll * self._position_fraction * mult

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Record resolution outcome (no-op; concentration ignores outcomes)."""
        del trade, payout


class FollowSeedSize:
    """cost = min(trade.notional_usd * scale_factor, max_cost_per_trade)."""

    name = "follow_seed_size"

    def __init__(self, *, scale_factor: float, max_cost_per_trade: float) -> None:
        """Initialize follow-seed-size sizing scheme."""
        self._scale_factor = scale_factor
        self._max_cost = max_cost_per_trade

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Return notional * scale, capped at max_cost_per_trade."""
        del bankroll
        return min(trade.notional_usd * self._scale_factor, self._max_cost)

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Record resolution outcome (no-op; stateless scheme)."""
        del trade, payout


class EdgeWeightedCausal:
    """Edge-weighted sizing with strict no-look-ahead.

    ``compute`` uses ONLY trades passed via :meth:`observe_resolution`
    BEFORE this call. The event-walk caller is responsible for
    processing resolutions in ``ts`` order before any subsequent trade
    with ``ts >= resolution.ts``.
    """

    name = "edge_weighted_causal"

    def __init__(
        self,
        *,
        position_fraction: float,
        edge_scale: float,
        min_multiplier: float,
        max_multiplier: float,
        min_trades_for_edge: int,
    ) -> None:
        """Initialize edge-weighted causal sizing scheme."""
        self._position_fraction = position_fraction
        self._edge_scale = edge_scale
        self._min_multiplier = min_multiplier
        self._max_multiplier = max_multiplier
        self._min_trades_for_edge = min_trades_for_edge
        self._resolved: dict[str, list[tuple[float, float]]] = {}

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Size based on wallet's rolling realized edge from prior resolved trades."""
        prior = self._resolved.get(trade.wallet, [])
        if len(prior) < self._min_trades_for_edge:
            mult = 1.0
        else:
            edge = sum(p - imp for p, imp in prior) / len(prior)
            raw = 1.0 + self._edge_scale * edge
            mult = max(self._min_multiplier, min(self._max_multiplier, raw))
        return bankroll * self._position_fraction * mult

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Append (payout, fill price) to this wallet's resolved-trade history."""
        self._resolved.setdefault(trade.wallet, []).append((payout, trade.price))
