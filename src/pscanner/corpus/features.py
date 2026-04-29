"""Pure feature-computation primitives for the corpus pipeline.

This module is the heart of the live/historical parity guarantee: every
function here is pure, taking a ``Trade`` plus a ``HistoryProvider`` (or
plain ``WalletState`` / ``MarketState``) and returning frozen dataclasses.
The same functions run inside ``StreamingHistoryProvider`` (v1, walking
``corpus_trades`` for ``build-features``) and inside
``LiveHistoryProvider`` (v2, fed by the live trade stream).

No DB handles, no network, no clocks. All non-determinism enters via
``HistoryProvider``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Trade:
    """One BUY or SELL fill, the input to feature extraction.

    The same shape covers both historical (``corpus_trades``) and live
    (websocket / activity stream) trade events.
    """

    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    outcome_side: str
    bs: str
    price: float
    size: float
    notional_usd: float
    ts: int
    category: str


@dataclass(frozen=True)
class WalletState:
    """Running per-wallet aggregate at some point in time.

    Holds enough state to derive every trader feature in
    ``training_examples``. Updated by ``apply_*_to_state`` functions.
    """

    first_seen_ts: int
    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    cumulative_buy_price_sum: float
    cumulative_buy_count: int
    realized_pnl_usd: float
    last_trade_ts: int | None
    recent_30d_trades: tuple[int, ...]
    bet_sizes: tuple[float, ...]
    category_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    """Running per-market aggregate at some point in time."""

    market_age_start_ts: int
    volume_so_far_usd: float
    unique_traders_so_far: tuple[str, ...]
    last_trade_price: float | None
    recent_prices: tuple[float, ...]


@dataclass(frozen=True)
class MarketMetadata:
    """Static per-market metadata. Does not change with time."""

    condition_id: str
    category: str
    closed_at: int
    opened_at: int


def empty_wallet_state(*, first_seen_ts: int) -> WalletState:
    """Construct an initial WalletState for a wallet's first seen ts."""
    return WalletState(
        first_seen_ts=first_seen_ts,
        prior_trades_count=0,
        prior_buys_count=0,
        prior_resolved_buys=0,
        prior_wins=0,
        prior_losses=0,
        cumulative_buy_price_sum=0.0,
        cumulative_buy_count=0,
        realized_pnl_usd=0.0,
        last_trade_ts=None,
        recent_30d_trades=(),
        bet_sizes=(),
        category_counts={},
    )


def empty_market_state(*, market_age_start_ts: int) -> MarketState:
    """Construct an initial MarketState for a market's first seen trade."""
    return MarketState(
        market_age_start_ts=market_age_start_ts,
        volume_so_far_usd=0.0,
        unique_traders_so_far=(),
        last_trade_price=None,
        recent_prices=(),
    )


def apply_buy_to_state(state: WalletState, trade: Trade) -> WalletState:
    """Apply a BUY fill to wallet state. Returns a new WalletState."""
    new_categories = dict(state.category_counts)
    new_categories[trade.category] = new_categories.get(trade.category, 0) + 1
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        prior_buys_count=state.prior_buys_count + 1,
        cumulative_buy_price_sum=state.cumulative_buy_price_sum + trade.price,
        cumulative_buy_count=state.cumulative_buy_count + 1,
        last_trade_ts=trade.ts,
        recent_30d_trades=(*state.recent_30d_trades, trade.ts),
        bet_sizes=(*state.bet_sizes, trade.notional_usd),
        category_counts=new_categories,
    )


def apply_sell_to_state(state: WalletState, trade: Trade) -> WalletState:
    """Apply a SELL fill to wallet state. Returns a new WalletState.

    Sells contribute to total trade count and recency but not to BUY
    aggregates (avg price paid, bet sizes, win/loss ledger).
    """
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        last_trade_ts=trade.ts,
        recent_30d_trades=(*state.recent_30d_trades, trade.ts),
    )


def apply_resolution_to_state(
    state: WalletState,
    *,
    won: bool,
    notional_usd: float,
    payout_usd: float,
) -> WalletState:
    """Fold a resolved prior buy into wallet state.

    ``payout_usd`` is the dollar amount returned at resolution
    (``size * 1.0`` if won, ``0.0`` if lost). Realized PnL increments by
    ``payout_usd - notional_usd``.
    """
    return replace(
        state,
        prior_resolved_buys=state.prior_resolved_buys + 1,
        prior_wins=state.prior_wins + (1 if won else 0),
        prior_losses=state.prior_losses + (0 if won else 1),
        realized_pnl_usd=state.realized_pnl_usd + (payout_usd - notional_usd),
    )


def apply_trade_to_market(state: MarketState, trade: Trade) -> MarketState:
    """Apply a fill to market state (per-market running aggregates)."""
    new_traders: tuple[str, ...]
    if trade.wallet_address in state.unique_traders_so_far:
        new_traders = state.unique_traders_so_far
    else:
        new_traders = (*state.unique_traders_so_far, trade.wallet_address)
    return replace(
        state,
        volume_so_far_usd=state.volume_so_far_usd + trade.notional_usd,
        unique_traders_so_far=new_traders,
        last_trade_price=trade.price,
        recent_prices=(*state.recent_prices, trade.price)[-20:],
    )
