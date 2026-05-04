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

import heapq
import statistics
from dataclasses import dataclass, field, replace
from typing import Protocol


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
    # Running totals for avg_bet_size_usd. Storing the raw bet_sizes
    # tuple would cost O(N) per fold and O(N) per feature read on
    # heavy-hitter wallets — a streaming sum/count keeps both at O(1).
    # ``median_bet_size_usd`` is no longer derived (always None in
    # FeatureRow) — accepted v1 cost; could be revived via a bounded
    # rolling window if a model needs it.
    bet_size_sum: float
    bet_size_count: int
    category_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    """Running per-market aggregate at some point in time.

    ``unique_traders_count`` is the number of distinct wallet addresses
    that have traded the market so far. The set itself is held in the
    streaming provider's mutable bookkeeping (``StreamingHistoryProvider``)
    so per-market state stays O(1) per fold; storing the set in the
    immutable state would require an O(N) tuple/frozenset rebuild on
    every trade for large markets.
    """

    market_age_start_ts: int
    volume_so_far_usd: float
    unique_traders_count: int
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
        bet_size_sum=0.0,
        bet_size_count=0,
        category_counts={},
    )


def empty_market_state(*, market_age_start_ts: int) -> MarketState:
    """Construct an initial MarketState for a market's first seen trade."""
    return MarketState(
        market_age_start_ts=market_age_start_ts,
        volume_so_far_usd=0.0,
        unique_traders_count=0,
        last_trade_price=None,
        recent_prices=(),
    )


# Rolling-window for `recent_30d_trades` storage. The tuple holds only
# trades within this many seconds of the most recent fold, so the
# accumulator's per-wallet memory stays bounded for very-active wallets
# (without a trim, hot wallets create O(N^2) work over the streaming
# walk: scanning a growing tuple per call). The window matches what
# `compute_features` reads (30 days), so the trimmed entries are
# exactly the ones a feature query would have discarded anyway.
_RECENT_WINDOW_SECONDS = 30 * 86_400


def _trim_recent_trades(recent: tuple[int, ...], current_ts: int) -> tuple[int, ...]:
    """Drop entries older than ``current_ts - _RECENT_WINDOW_SECONDS``."""
    cutoff = current_ts - _RECENT_WINDOW_SECONDS
    return tuple(ts for ts in recent if ts >= cutoff)


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
        recent_30d_trades=(*_trim_recent_trades(state.recent_30d_trades, trade.ts), trade.ts),
        bet_size_sum=state.bet_size_sum + trade.notional_usd,
        bet_size_count=state.bet_size_count + 1,
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
        recent_30d_trades=(*_trim_recent_trades(state.recent_30d_trades, trade.ts), trade.ts),
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


def apply_trade_to_market(state: MarketState, trade: Trade, *, is_new_trader: bool) -> MarketState:
    """Apply a fill to market state (per-market running aggregates).

    ``is_new_trader`` is computed by the caller against its own membership
    set — keeping the set out of the immutable state lets per-market
    folds stay O(1) instead of O(N) on the trader count.
    """
    return replace(
        state,
        volume_so_far_usd=state.volume_so_far_usd + trade.notional_usd,
        unique_traders_count=state.unique_traders_count + (1 if is_new_trader else 0),
        last_trade_price=trade.price,
        recent_prices=(*state.recent_prices, trade.price)[-20:],
    )


@dataclass(frozen=True)
class FeatureRow:
    """All features computed for a single trade.

    Mirrors the columns of ``training_examples`` (sans identity columns
    and ``built_at``).
    """

    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    win_rate: float | None
    avg_implied_prob_paid: float | None
    realized_edge_pp: float | None
    prior_realized_pnl_usd: float
    avg_bet_size_usd: float | None
    median_bet_size_usd: float | None
    wallet_age_days: float
    seconds_since_last_trade: int | None
    prior_trades_30d: int
    top_category: str | None
    category_diversity: int
    bet_size_usd: float
    bet_size_rel_to_avg: float | None
    edge_confidence_weighted: float
    win_rate_confidence_weighted: float
    is_high_quality_wallet: int
    # Always 1.0 in v1; the streaming feature provider doesn't maintain a running
    # median (see WalletState.median_bet_size_usd). A future v2 provider will fill
    # this in without schema changes.
    bet_size_relative_to_history: float
    side: str
    implied_prob_at_buy: float
    market_category: str
    market_volume_so_far_usd: float
    market_unique_traders_so_far: int
    market_age_seconds: int
    time_to_resolution_seconds: int | None
    last_trade_price: float | None
    price_volatility_recent: float | None


class HistoryProvider(Protocol):
    """Looks up wallet/market state at a point in time.

    Two implementations: ``StreamingHistoryProvider`` (corpus, walks
    ``corpus_trades`` chronologically) and ``LiveHistoryProvider`` (v2,
    fed by live trade stream). Both must return state computed strictly
    from events with ``ts < as_of_ts``.
    """

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        """Return wallet state strictly before ``as_of_ts``."""
        ...

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        """Return market state strictly before ``as_of_ts``."""
        ...

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        """Return static market metadata (not time-varying)."""
        ...


_SECONDS_PER_DAY = 86_400
_MIN_PRICES_FOR_VOLATILITY = 2
# Minimum resolved buys for full confidence in per-wallet edge estimates.
# Below this threshold features are linearly discounted toward zero.
_CONFIDENCE_N_MIN = 20
_HIGH_QUALITY_WIN_RATE_THRESHOLD = 0.55


def compute_features(trade: Trade, history: HistoryProvider) -> FeatureRow:
    """Compute the full feature row for a trade, point-in-time correct.

    Pure function: takes only ``trade`` and ``history``. All
    non-determinism enters via the provider.
    """
    wallet = history.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
    market = history.market_state(trade.condition_id, as_of_ts=trade.ts)
    meta = history.market_metadata(trade.condition_id)

    win_rate = (
        wallet.prior_wins / wallet.prior_resolved_buys if wallet.prior_resolved_buys > 0 else None
    )
    avg_prob = (
        wallet.cumulative_buy_price_sum / wallet.cumulative_buy_count
        if wallet.cumulative_buy_count > 0
        else None
    )
    edge = win_rate - avg_prob if win_rate is not None and avg_prob is not None else None
    avg_bet = wallet.bet_size_sum / wallet.bet_size_count if wallet.bet_size_count > 0 else None
    # median_bet is no longer maintained — would require a bounded
    # rolling window or a streaming estimator. v1 always emits None.
    median_bet: float | None = None
    rel_to_avg = trade.notional_usd / avg_bet if avg_bet is not None and avg_bet > 0 else None
    confidence = min(1.0, wallet.prior_resolved_buys / _CONFIDENCE_N_MIN)
    edge_conf = (edge * confidence) if edge is not None else 0.0
    wr_conf = ((win_rate - 0.5) * confidence) if win_rate is not None else 0.0
    is_high_quality = int(
        wallet.prior_resolved_buys >= _CONFIDENCE_N_MIN
        and win_rate is not None
        and win_rate > _HIGH_QUALITY_WIN_RATE_THRESHOLD
    )
    rel_to_median = (
        trade.notional_usd / median_bet if median_bet is not None and median_bet > 0 else 1.0
    )
    seconds_since_last = (
        trade.ts - wallet.last_trade_ts if wallet.last_trade_ts is not None else None
    )
    wallet_age_days = max(0.0, (trade.ts - wallet.first_seen_ts) / _SECONDS_PER_DAY)
    cutoff = trade.ts - 30 * _SECONDS_PER_DAY
    recent_30d = sum(1 for ts in wallet.recent_30d_trades if ts >= cutoff)
    top_cat = (
        max(wallet.category_counts.items(), key=lambda kv: kv[1])[0]
        if wallet.category_counts
        else None
    )
    diversity = len(wallet.category_counts)

    implied_prob = trade.price

    volatility = (
        statistics.pstdev(market.recent_prices)
        if len(market.recent_prices) >= _MIN_PRICES_FOR_VOLATILITY
        else None
    )
    time_to_resolution = meta.closed_at - trade.ts

    return FeatureRow(
        prior_trades_count=wallet.prior_trades_count,
        prior_buys_count=wallet.prior_buys_count,
        prior_resolved_buys=wallet.prior_resolved_buys,
        prior_wins=wallet.prior_wins,
        prior_losses=wallet.prior_losses,
        win_rate=win_rate,
        avg_implied_prob_paid=avg_prob,
        realized_edge_pp=edge,
        prior_realized_pnl_usd=wallet.realized_pnl_usd,
        avg_bet_size_usd=avg_bet,
        median_bet_size_usd=median_bet,
        wallet_age_days=wallet_age_days,
        seconds_since_last_trade=seconds_since_last,
        prior_trades_30d=recent_30d,
        top_category=top_cat,
        category_diversity=diversity,
        bet_size_usd=trade.notional_usd,
        bet_size_rel_to_avg=rel_to_avg,
        edge_confidence_weighted=edge_conf,
        win_rate_confidence_weighted=wr_conf,
        is_high_quality_wallet=is_high_quality,
        bet_size_relative_to_history=rel_to_median,
        side=trade.outcome_side,
        implied_prob_at_buy=implied_prob,
        market_category=meta.category,
        market_volume_so_far_usd=market.volume_so_far_usd,
        market_unique_traders_so_far=market.unique_traders_count,
        market_age_seconds=trade.ts - market.market_age_start_ts,
        time_to_resolution_seconds=time_to_resolution,
        last_trade_price=market.last_trade_price,
        price_volatility_recent=volatility,
    )


@dataclass(frozen=True)
class _UnresolvedBuy:
    """A BUY fill waiting for its market's resolution to be registered."""

    seq: int
    condition_id: str
    notional_usd: float
    size: float
    side_yes: bool


@dataclass
class _WalletAccumulator:
    """Mutable wrapper around WalletState for streaming updates."""

    state: WalletState
    # Heap of (resolution_ts, seq, _UnresolvedBuy) — only entries whose
    # resolution is already known.
    heap: list[tuple[int, int, _UnresolvedBuy]]
    # Buys whose market has not yet had register_resolution() called.
    unscheduled: list[_UnresolvedBuy]


class StreamingHistoryProvider:
    """In-memory provider that walks events chronologically.

    Used inside ``build-features``: the orchestrator calls
    ``wallet_state(...)`` and ``market_state(...)`` BEFORE folding each
    trade in via ``observe(...)``. Resolutions are registered up-front
    or after-the-fact via ``register_resolution(...)`` and applied lazily
    when the next ``wallet_state`` query crosses their ``resolution_ts``.
    """

    def __init__(self, metadata: dict[str, MarketMetadata]) -> None:
        """Create an empty provider seeded with per-market metadata."""
        self._metadata = metadata
        self._wallets: dict[str, _WalletAccumulator] = {}
        self._markets: dict[str, MarketState] = {}
        self._market_traders: dict[str, set[str]] = {}
        self._resolutions: dict[str, tuple[int, int]] = {}  # cond_id -> (resolved_at, yes_won)
        self._seq = 0

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        """Return static metadata for ``condition_id``; raises KeyError if unknown."""
        return self._metadata[condition_id]

    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None:
        """Record a market's resolution.

        Any unscheduled buys on this market across all wallets are moved
        onto each wallet's resolution heap so they drain at the correct ts.
        """
        self._resolutions[condition_id] = (resolved_at, outcome_yes_won)
        for accum in self._wallets.values():
            remaining: list[_UnresolvedBuy] = []
            for buy in accum.unscheduled:
                if buy.condition_id == condition_id:
                    heapq.heappush(accum.heap, (resolved_at, buy.seq, buy))
                else:
                    remaining.append(buy)
            accum.unscheduled = remaining

    def observe(self, trade: Trade) -> None:
        """Fold a trade into running wallet + market state."""
        accum = self._wallets.get(trade.wallet_address)
        if accum is None:
            accum = _WalletAccumulator(
                state=empty_wallet_state(first_seen_ts=trade.ts),
                heap=[],
                unscheduled=[],
            )
            self._wallets[trade.wallet_address] = accum

        if trade.bs == "BUY":
            accum.state = apply_buy_to_state(accum.state, trade)
            self._seq += 1
            buy = _UnresolvedBuy(
                seq=self._seq,
                condition_id=trade.condition_id,
                notional_usd=trade.notional_usd,
                size=trade.size,
                side_yes=trade.outcome_side == "YES",
            )
            resolution = self._resolutions.get(trade.condition_id)
            if resolution is not None:
                resolved_at, _ = resolution
                heapq.heappush(accum.heap, (resolved_at, buy.seq, buy))
            else:
                accum.unscheduled.append(buy)
        elif trade.bs == "SELL":
            accum.state = apply_sell_to_state(accum.state, trade)

        market = self._markets.get(trade.condition_id)
        if market is None:
            market = empty_market_state(market_age_start_ts=trade.ts)
        traders = self._market_traders.setdefault(trade.condition_id, set())
        is_new_trader = trade.wallet_address not in traders
        if is_new_trader:
            traders.add(trade.wallet_address)
        self._markets[trade.condition_id] = apply_trade_to_market(
            market, trade, is_new_trader=is_new_trader
        )

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        """Return the wallet's state at ``as_of_ts``, draining ready resolutions."""
        accum = self._wallets.get(wallet_address)
        if accum is None:
            return empty_wallet_state(first_seen_ts=as_of_ts)
        while accum.heap and accum.heap[0][0] < as_of_ts:
            _, _, buy = heapq.heappop(accum.heap)
            resolution = self._resolutions.get(buy.condition_id)
            if resolution is None:
                continue
            _, yes_won = resolution
            won = (yes_won == 1) if buy.side_yes else (yes_won == 0)
            payout = buy.size if won else 0.0
            accum.state = apply_resolution_to_state(
                accum.state,
                won=won,
                notional_usd=buy.notional_usd,
                payout_usd=payout,
            )
        return accum.state

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        """Return per-market running state.

        ``as_of_ts`` is unused — caller must query before observing the
        next event for the same market.
        """
        del as_of_ts
        return self._markets.get(
            condition_id,
            empty_market_state(market_age_start_ts=0),
        )
