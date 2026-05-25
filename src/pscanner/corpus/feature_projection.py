"""Single source of truth for ML feature dataclasses and the projection registry (#145).

This module holds three things:

1. **Shared dataclasses** (``Trade``, ``WalletState``, ``MarketState``,
   ``MarketMetadata``, ``FeatureRow``, ``_TradeFields``) that describe the
   inputs and outputs of feature projection.
2. **Pure helpers** (``apply_buy_to_state``, ``apply_sell_to_state``,
   ``apply_resolution_to_state``, ``apply_trade_to_market``,
   ``empty_wallet_state``, ``empty_market_state``, ``compute_features``,
   and the ``HistoryProvider`` Protocol).
3. **The canonical ``FEATURES`` registry** used by both the Python
   ``compute_features`` path and the DuckDB ``_final_join_to_v2`` SQL
   builder.

See ``docs/superpowers/plans/2026-05-19-issue-145-feature-projection.md``
for the architectural rationale. In short: the same FeatureRow used to be
computed by three code paths (Python streaming via ``compute_features``,
live daemon via a now-deleted ``LiveHistoryProvider``, DuckDB batch via
``_final_join_to_v2``). The live daemon path has been removed; the two
surviving paths (``compute_features`` here and the DuckDB engine) share
this registry.
"""

from __future__ import annotations

import heapq
import statistics
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


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


class _TradeFields(Protocol):
    """Structural shape of the fields ``observe`` reads on the SELL path.

    Both ``Trade`` and ``pscanner.corpus.repos.CorpusTrade`` satisfy this.
    Used so the ``build-features`` loop can hand ``CorpusTrade`` straight
    to ``observe`` for SELL rows without rebuilding a ``Trade`` (which
    would also force allocating the unused ``category`` field).
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


@dataclass(frozen=True)
class WalletState:
    """Running per-wallet aggregate at some point in time.

    Holds enough state to derive every trader feature in
    ``training_examples``. Updated by ``apply_*_to_state`` functions.

    ``recent_30d_trades`` is mutated in place by the apply_* functions
    (see issue #110 — the previous immutable-tuple rebuild was O(N) per
    trade and dominated the build-features wall time on heavy wallets).
    The dataclass stays frozen — only the deque's contents change, not
    the field reference.
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
    recent_30d_trades: deque[int]
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

    ``recent_prices`` is a bounded deque (maxlen=20) mutated in place by
    ``apply_trade_to_market`` — the previous tuple-rebuild was O(20) per
    fold and dominated wall time on the build-features hot loop (#114).
    The dataclass stays frozen — only the deque's contents change, not
    the field reference.
    """

    market_age_start_ts: int
    volume_so_far_usd: float
    unique_traders_count: int
    last_trade_price: float | None
    recent_prices: deque[float]


@dataclass(frozen=True)
class MarketMetadata:
    """Static per-market metadata. Does not change with time.

    ``categories`` is the multi-label set of every taxonomy category that
    matches the market's gamma tags (see :func:`pscanner.categories.categorize_tags`).
    Defaults to ``()`` so callers that only know the primary string
    ``category`` keep working unchanged; consumers that need multi-label
    behaviour read ``categories or (category,)`` as the fallback.
    """

    condition_id: str
    category: str
    closed_at: int
    opened_at: int
    categories: tuple[str, ...] = ()


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
    market_categories: tuple[str, ...]
    cat_sports: int
    cat_esports: int
    cat_thesis: int
    cat_macro: int
    cat_elections: int
    cat_crypto: int
    cat_geopolitics: int
    cat_tech: int
    cat_culture: int
    market_volume_so_far_usd: float
    market_unique_traders_so_far: int
    market_age_seconds: int
    time_to_resolution_seconds: int | None
    last_trade_price: float | None
    price_volatility_recent: float | None


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


# Rolling-window for `recent_30d_trades` storage. The deque holds only
# trades within this many seconds of the most recent fold, so the
# accumulator's per-wallet memory stays bounded for very-active wallets.
# The window matches what `compute_features` reads (30 days), so trimmed
# entries are exactly the ones a feature query would have discarded.
_RECENT_WINDOW_SECONDS = 30 * 86_400

# Bounded rolling-window for recent_prices. Kept as a deque(maxlen) so
# appends are O(1) without manual trimming. The window matches what
# compute_features reads (last N prices for volatility).
_RECENT_PRICES_MAX = 20


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
        recent_30d_trades=deque(),
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
        recent_prices=deque(maxlen=_RECENT_PRICES_MAX),
    )


def _trim_and_append(window: deque[int], current_ts: int) -> None:
    """Drop entries older than ``current_ts - _RECENT_WINDOW_SECONDS`` and append.

    Mutates ``window`` in place. O(1) amortized per call (popleft + append),
    versus O(N) for the old tuple rebuild — the change that drives most of
    issue #110's wall-time reduction.
    """
    cutoff = current_ts - _RECENT_WINDOW_SECONDS
    while window and window[0] < cutoff:
        window.popleft()
    window.append(current_ts)


def apply_buy_to_state(state: WalletState, trade: Trade) -> WalletState:
    """Apply a BUY fill to wallet state. Returns a new WalletState.

    Mutates ``state.recent_30d_trades`` and ``state.category_counts`` in
    place — see :class:`WalletState` for why frozen+mutate is safe.
    """
    state.category_counts[trade.category] = state.category_counts.get(trade.category, 0) + 1
    _trim_and_append(state.recent_30d_trades, trade.ts)
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        prior_buys_count=state.prior_buys_count + 1,
        cumulative_buy_price_sum=state.cumulative_buy_price_sum + trade.price,
        cumulative_buy_count=state.cumulative_buy_count + 1,
        last_trade_ts=trade.ts,
        bet_size_sum=state.bet_size_sum + trade.notional_usd,
        bet_size_count=state.bet_size_count + 1,
    )


def apply_sell_to_state(state: WalletState, trade: _TradeFields) -> WalletState:
    """Apply a SELL fill to wallet state. Returns a new WalletState.

    Sells contribute to total trade count and recency but not to BUY
    aggregates (avg price paid, bet sizes, win/loss ledger). Accepts any
    object with the SELL-relevant fields so callers can pass either
    ``Trade`` or the bare repo ``CorpusTrade`` without rebuilding.
    Mutates ``state.recent_30d_trades`` in place.
    """
    _trim_and_append(state.recent_30d_trades, trade.ts)
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        last_trade_ts=trade.ts,
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


def apply_trade_to_market(
    state: MarketState, trade: _TradeFields, *, is_new_trader: bool
) -> MarketState:
    """Apply a fill to market state (per-market running aggregates).

    ``is_new_trader`` is computed by the caller against its own membership
    set — keeping the set out of the immutable state lets per-market
    folds stay O(1) instead of O(N) on the trader count.

    Mutates ``state.recent_prices`` in place — see :class:`MarketState`
    for why frozen+mutate is safe (#114).
    """
    state.recent_prices.append(trade.price)
    return replace(
        state,
        volume_so_far_usd=state.volume_so_far_usd + trade.notional_usd,
        unique_traders_count=state.unique_traders_count + (1 if is_new_trader else 0),
        last_trade_price=trade.price,
    )


# ---------------------------------------------------------------------------
# HistoryProvider Protocol + compute_features
# ---------------------------------------------------------------------------


class HistoryProvider(Protocol):
    """Looks up wallet/market state at a point in time.

    The corpus-side ``StreamingHistoryProvider`` (walks ``corpus_trades``
    chronologically) implements this Protocol. The Protocol stays here
    so future implementations have a stable contract.
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


# ---------------------------------------------------------------------------
# StreamingHistoryProvider (Python build-features fold)
# ---------------------------------------------------------------------------


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

    def get_resolution(self, condition_id: str) -> tuple[int, int] | None:
        """Return ``(resolved_at, outcome_yes_won)`` for a market, or None if unresolved.

        Reads from the in-memory map seeded by ``register_resolution`` so the
        ``build-features`` hot path can answer "is this market resolved?"
        without a per-trade SQLite SELECT against ``market_resolutions``.
        """
        return self._resolutions.get(condition_id)

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
        """Fold a trade into running wallet + market state.

        For BUY rows the caller must pass a full ``Trade`` (the
        ``category`` field feeds ``WalletState.category_counts``). For
        SELL rows ``observe_sell`` is preferred — it accepts the bare
        repo dataclass and skips an unnecessary ``Trade`` rebuild.
        """
        accum = self._ensure_accumulator(trade)

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

        self._fold_market_state(trade)

    def observe_sell(self, trade: _TradeFields) -> None:
        """Fold a SELL fill into wallet + market state.

        Accepts any object with the trade fields used by the SELL path
        (no ``category`` required). Lets the ``build-features`` loop hand
        ``CorpusTrade`` directly without rebuilding a ``Trade``.

        Caller must guarantee ``trade.bs == "SELL"`` — BUYs would skip
        the heap-bookkeeping and silently lose label coverage.
        """
        accum = self._ensure_accumulator(trade)
        accum.state = apply_sell_to_state(accum.state, trade)
        self._fold_market_state(trade)

    def _ensure_accumulator(self, trade: _TradeFields) -> _WalletAccumulator:
        accum = self._wallets.get(trade.wallet_address)
        if accum is None:
            accum = _WalletAccumulator(
                state=empty_wallet_state(first_seen_ts=trade.ts),
                heap=[],
                unscheduled=[],
            )
            self._wallets[trade.wallet_address] = accum
        return accum

    def _fold_market_state(self, trade: _TradeFields) -> None:
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

    def iter_wallet_states(self) -> Iterator[tuple[str, WalletState]]:
        """Yield ``(wallet_address, WalletState)`` for every observed wallet.

        Iterates the provider's currently-resolved wallet state — does
        NOT drain pending resolutions for any wallet. Callers that need
        point-in-time state for a specific ts should use
        :meth:`wallet_state` instead.
        """
        for wallet_address, accum in self._wallets.items():
            yield wallet_address, accum.state

    def iter_market_states(self) -> Iterator[tuple[str, MarketState]]:
        """Yield ``(condition_id, MarketState)`` for every observed market."""
        yield from self._markets.items()


# ---------------------------------------------------------------------------
# Feature projection registry
# ---------------------------------------------------------------------------


# Magic numbers. Both engines import these; never inline a literal that
# duplicates one of these.
CONFIDENCE_N_MIN = 20
HIGH_QUALITY_WIN_RATE_THRESHOLD = 0.55
RECENT_TRADES_WINDOW_DAYS = 30
SECONDS_PER_DAY = 86_400
MIN_PRICES_FOR_VOLATILITY = 2

# Multi-label category universe. The DuckDB engine refuses to start if a
# corpus row has a category outside this tuple (see _duckdb_engine.py
# _assert_no_unknown_categories); the registry's cat_* indicators are
# generated by looping over this tuple.
KNOWN_CATEGORIES: tuple[str, ...] = (
    "sports",
    "esports",
    "thesis",
    "macro",
    "elections",
    "crypto",
    "geopolitics",
    "tech",
    "culture",
)

FeatureDType = Literal["float", "int", "str", "tuple_str"]


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    """Bundle of state passed to a feature's Python evaluator.

    ``wallet`` and ``market`` are point-in-time-correct (computed strictly
    from events with ``ts < trade.ts`` by the caller). ``meta`` is static
    per market. ``trade`` is the current row being projected.
    """

    wallet: WalletState
    market: MarketState
    meta: MarketMetadata
    trade: Trade


@dataclass(frozen=True, slots=True)
class FeatureFormula:
    """One column in ``training_examples_v2``.

    ``py`` and ``sql`` MUST compute the same value on the same input. The
    parity test in ``tests/corpus/test_feature_projection_parity.py``
    asserts this with Hypothesis.

    The ``sql`` string is a template — placeholders of the form
    ``{w.field}``, ``{m.field}``, ``{meta.field}``, ``{t.field}`` are
    replaced by ``render_sql_fragment`` using ``SQL_BINDINGS`` below.
    """

    name: str
    dtype: FeatureDType
    nullable: bool
    py: Callable[[FeatureInputs], object]
    sql: str
    # If False, project_sql() skips this formula. Used for compute-only
    # features (e.g. market_categories) that exist on FeatureRow as a
    # transient input but are not persisted to training_examples_v2.
    project_to_sql: bool = True
    docs: str = ""


# Maps {scope.field} placeholder keys to the SQL column references used by
# pscanner.corpus._duckdb_engine._final_join_to_v2. The names on the SQL
# side are the column aliases that wallet_aggs (wa), market_aggs (ma), and
# wallet_cat_summary (wcs) expose at the final-join stage.
#
# Key naming: "<scope>.<feature-py-attr>". The same logical field can have
# a different name on each engine (e.g. WalletState.cumulative_buy_count
# == wallet_aggs.bet_size_count_w by construction); the binding hides
# that asymmetry.
SQL_BINDINGS: Mapping[str, str] = {
    # WalletState fields
    "w.prior_trades_count": "wa.prior_trades_count_w",
    "w.prior_buys_count": "wa.prior_buys_count_w",
    "w.prior_resolved_buys": "wa.prior_resolved_buys_w",
    "w.prior_wins": "wa.prior_wins_w",
    "w.prior_losses": "wa.prior_losses_w",
    "w.cumulative_buy_price_sum": "wa.cum_buy_price_sum_w",
    # WalletState.cumulative_buy_count tracks the same count as bet_size_count
    # by construction; the SQL side uses the bet_size_count_w column for both.
    "w.cumulative_buy_count": "wa.bet_size_count_w",
    "w.bet_size_sum": "wa.bet_size_sum_w",
    "w.bet_size_count": "wa.bet_size_count_w",
    "w.realized_pnl_usd": "wa.prior_realized_pnl_usd_w",
    "w.last_trade_ts": "wa.last_trade_ts_w",
    "w.first_seen_ts": "wa.first_seen_ts",
    "w.prior_trades_30d": "wa.prior_trades_30d_w",
    # MarketState fields
    "m.volume_so_far_usd": "ma.market_volume_so_far_w",
    "m.unique_traders_count": "ma.market_unique_traders_so_far_w",
    "m.market_age_start_ts": "ma.market_first_prior_ts_w",
    "m.last_trade_price": "ma.last_trade_price_w",
    "m.price_volatility": "ma.price_volatility_w",
    "m.price_count_20": "ma.price_count_20",
    # MarketMetadata fields
    "meta.category": "wa.category",
    "meta.categories_json": "wa.categories_json",
    "meta.closed_at": "wa.closed_at",
    # Trade fields (the current row being projected)
    "t.notional_usd": "wa.notional_usd",
    "t.price": "wa.price",
    "t.outcome_side": "wa.outcome_side",
    "t.ts": "wa.event_ts",
    # Already-computed columns (from wallet_cat_summary subquery)
    "wcs.top_category": "wcs.top_category",
    "wcs.category_diversity": "COALESCE(wcs.category_diversity, 0)",
}


def render_sql_fragment(template: str, bindings: Mapping[str, str] = SQL_BINDINGS) -> str:
    """Resolve ``{scope.field}`` placeholders against ``bindings``.

    Raises ``KeyError`` if the template references an unbound placeholder
    (e.g. ``{w.bogus_field}``) — this catches typos at module-load time
    instead of producing malformed SQL at query time.
    """
    rendered = template
    for key, value in bindings.items():
        rendered = rendered.replace("{" + key + "}", value)
    # Belt-and-braces: if any "{...}" placeholder survives the loop, it
    # didn't match a binding key. Surface that as a clear error rather
    # than letting DuckDB choke on the literal braces.
    if "{" in rendered:
        # Find the first unresolved placeholder for the error message.
        start = rendered.index("{")
        end = rendered.index("}", start)
        raise KeyError(f"feature_projection: unbound SQL placeholder {rendered[start : end + 1]!r}")
    return rendered


def _cat_indicator_formula(category: str) -> FeatureFormula:
    """Generate one cat_<category> indicator formula.

    The 9 indicators all share a shape: int(category in meta.categories).
    Generated programmatically to avoid 9 copies of the same Python lambda
    + SQL fragment.
    """

    def _py(i: FeatureInputs) -> int:
        categories = i.meta.categories if i.meta.categories else (i.meta.category,)
        return int(category in set(categories))

    sql = (
        "CAST(CASE "
        "WHEN json_array_length(COALESCE({meta.categories_json}, '[]')) > 0 "
        "THEN list_contains("
        "CAST(json_extract({meta.categories_json}, '$') AS VARCHAR[]), "
        f"'{category}') "
        f"ELSE {{meta.category}} = '{category}' "
        "END AS INTEGER)"
    )
    return FeatureFormula(
        name=f"cat_{category}",
        dtype="int",
        nullable=False,
        py=_py,
        sql=sql,
    )


# The canonical registry. Order is intentional but not load-bearing —
# project_row uses keyword construction (FeatureRow(**values)) so any
# permutation produces the same FeatureRow. project_sql emits columns
# in this order, which becomes the SELECT-list order in
# _final_join_to_v2; the _copy_to_sqlite step then SELECTs by name, so
# the column ordering doesn't have to match the SQLite schema either.
FEATURES: tuple[FeatureFormula, ...] = (
    # ----- Passthrough wallet aggregates -----
    FeatureFormula(
        name="prior_trades_count",
        dtype="int",
        nullable=False,
        py=lambda i: i.wallet.prior_trades_count,
        sql="{w.prior_trades_count}",
    ),
    FeatureFormula(
        name="prior_buys_count",
        dtype="int",
        nullable=False,
        py=lambda i: i.wallet.prior_buys_count,
        sql="{w.prior_buys_count}",
    ),
    FeatureFormula(
        name="prior_resolved_buys",
        dtype="int",
        nullable=False,
        py=lambda i: i.wallet.prior_resolved_buys,
        sql="{w.prior_resolved_buys}",
    ),
    FeatureFormula(
        name="prior_wins",
        dtype="int",
        nullable=False,
        py=lambda i: i.wallet.prior_wins,
        sql="{w.prior_wins}",
    ),
    FeatureFormula(
        name="prior_losses",
        dtype="int",
        nullable=False,
        py=lambda i: i.wallet.prior_losses,
        sql="{w.prior_losses}",
    ),
    FeatureFormula(
        name="prior_realized_pnl_usd",
        dtype="float",
        nullable=False,
        py=lambda i: i.wallet.realized_pnl_usd,
        sql="{w.realized_pnl_usd}",
    ),
    # prior_trades_30d: Python computes from recent_30d_trades deque;
    # DuckDB stage 2 pre-computes the count as wa.prior_trades_30d_w.
    FeatureFormula(
        name="prior_trades_30d",
        dtype="int",
        nullable=False,
        py=lambda i: sum(
            1
            for ts in i.wallet.recent_30d_trades
            if ts >= i.trade.ts - RECENT_TRADES_WINDOW_DAYS * SECONDS_PER_DAY
        ),
        sql="{w.prior_trades_30d}",
    ),
    # ----- Trade-row passthroughs -----
    FeatureFormula(
        name="bet_size_usd",
        dtype="float",
        nullable=False,
        py=lambda i: i.trade.notional_usd,
        sql="{t.notional_usd}",
    ),
    FeatureFormula(
        name="side",
        dtype="str",
        nullable=False,
        py=lambda i: i.trade.outcome_side,
        sql="{t.outcome_side}",
    ),
    FeatureFormula(
        name="implied_prob_at_buy",
        dtype="float",
        nullable=False,
        py=lambda i: i.trade.price,
        sql="{t.price}",
    ),
    # ----- Market-state passthroughs -----
    FeatureFormula(
        name="market_volume_so_far_usd",
        dtype="float",
        nullable=False,
        py=lambda i: i.market.volume_so_far_usd,
        sql="COALESCE({m.volume_so_far_usd}, 0.0)",
    ),
    FeatureFormula(
        name="market_unique_traders_so_far",
        dtype="int",
        nullable=False,
        py=lambda i: i.market.unique_traders_count,
        sql="CAST(COALESCE({m.unique_traders_count}, 0) AS INTEGER)",
    ),
    FeatureFormula(
        name="last_trade_price",
        dtype="float",
        nullable=True,
        py=lambda i: i.market.last_trade_price,
        sql="{m.last_trade_price}",
    ),
    # ----- Market metadata passthroughs -----
    FeatureFormula(
        name="market_category",
        dtype="str",
        nullable=False,
        py=lambda i: i.meta.category,
        sql="{meta.category}",
    ),
    # ----- Nullable divisions (denominator == 0 → None) -----
    FeatureFormula(
        name="win_rate",
        dtype="float",
        nullable=True,
        py=lambda i: (
            i.wallet.prior_wins / i.wallet.prior_resolved_buys
            if i.wallet.prior_resolved_buys > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 "
            "THEN CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="avg_implied_prob_paid",
        dtype="float",
        nullable=True,
        py=lambda i: (
            i.wallet.cumulative_buy_price_sum / i.wallet.cumulative_buy_count
            if i.wallet.cumulative_buy_count > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.bet_size_count} > 0 "
            "THEN {w.cumulative_buy_price_sum} / {w.bet_size_count} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="realized_edge_pp",
        dtype="float",
        nullable=True,
        py=lambda i: (
            (i.wallet.prior_wins / i.wallet.prior_resolved_buys)
            - (i.wallet.cumulative_buy_price_sum / i.wallet.cumulative_buy_count)
            if i.wallet.prior_resolved_buys > 0 and i.wallet.cumulative_buy_count > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 AND {w.bet_size_count} > 0 "
            "THEN (CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys}) "
            "- ({w.cumulative_buy_price_sum} / {w.bet_size_count}) "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="avg_bet_size_usd",
        dtype="float",
        nullable=True,
        py=lambda i: (
            i.wallet.bet_size_sum / i.wallet.bet_size_count if i.wallet.bet_size_count > 0 else None
        ),
        sql=(
            "CASE WHEN {w.bet_size_count} > 0 "
            "THEN {w.bet_size_sum} / {w.bet_size_count} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="median_bet_size_usd",
        dtype="float",
        nullable=True,
        # v1: not maintained — compute_features always emits None.
        # The running-median requires a bounded rolling window the streaming
        # provider doesn't yet maintain; could be revived in a v2 provider.
        py=lambda _i: None,
        sql="CAST(NULL AS DOUBLE)",
    ),
    FeatureFormula(
        name="bet_size_rel_to_avg",
        dtype="float",
        nullable=True,
        py=lambda i: (
            i.trade.notional_usd / (i.wallet.bet_size_sum / i.wallet.bet_size_count)
            if i.wallet.bet_size_count > 0 and i.wallet.bet_size_sum > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.bet_size_count} > 0 AND {w.bet_size_sum} > 0 "
            "THEN {t.notional_usd} / ({w.bet_size_sum} / {w.bet_size_count}) "
            "ELSE NULL END"
        ),
    ),
    # ----- Wallet-quality interaction features (#44) -----
    FeatureFormula(
        name="edge_confidence_weighted",
        dtype="float",
        nullable=False,
        py=lambda i: (
            (
                (i.wallet.prior_wins / i.wallet.prior_resolved_buys)
                - (i.wallet.cumulative_buy_price_sum / i.wallet.cumulative_buy_count)
            )
            * min(1.0, i.wallet.prior_resolved_buys / CONFIDENCE_N_MIN)
            if i.wallet.prior_resolved_buys > 0 and i.wallet.cumulative_buy_count > 0
            else 0.0
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 AND {w.bet_size_count} > 0 "
            "THEN ((CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys}) "
            "- ({w.cumulative_buy_price_sum} / {w.bet_size_count})) "
            f"* LEAST(1.0, CAST({{w.prior_resolved_buys}} AS DOUBLE) / {CONFIDENCE_N_MIN}.0) "
            "ELSE 0.0 END"
        ),
    ),
    FeatureFormula(
        name="win_rate_confidence_weighted",
        dtype="float",
        nullable=False,
        py=lambda i: (
            ((i.wallet.prior_wins / i.wallet.prior_resolved_buys) - 0.5)
            * min(1.0, i.wallet.prior_resolved_buys / CONFIDENCE_N_MIN)
            if i.wallet.prior_resolved_buys > 0
            else 0.0
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 "
            "THEN ((CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys}) - 0.5) "
            f"* LEAST(1.0, CAST({{w.prior_resolved_buys}} AS DOUBLE) / {CONFIDENCE_N_MIN}.0) "
            "ELSE 0.0 END"
        ),
    ),
    FeatureFormula(
        name="is_high_quality_wallet",
        dtype="int",
        nullable=False,
        py=lambda i: int(
            i.wallet.prior_resolved_buys >= CONFIDENCE_N_MIN
            and i.wallet.prior_resolved_buys > 0
            and (i.wallet.prior_wins / i.wallet.prior_resolved_buys)
            > HIGH_QUALITY_WIN_RATE_THRESHOLD
        ),
        sql=(
            f"CASE WHEN {{w.prior_resolved_buys}} >= {CONFIDENCE_N_MIN} "
            "AND (CAST({w.prior_wins} AS DOUBLE) "
            f"/ NULLIF({{w.prior_resolved_buys}}, 0)) > {HIGH_QUALITY_WIN_RATE_THRESHOLD} "
            "THEN 1 ELSE 0 END"
        ),
    ),
    FeatureFormula(
        name="bet_size_relative_to_history",
        dtype="float",
        nullable=False,
        # v1: median_bet_size_usd is never maintained, so the ratio is
        # always 1.0.
        py=lambda _i: 1.0,
        sql="CAST(1.0 AS DOUBLE)",
    ),
    # ----- Temporal features -----
    FeatureFormula(
        name="wallet_age_days",
        dtype="float",
        nullable=False,
        py=lambda i: max(0.0, (i.trade.ts - i.wallet.first_seen_ts) / SECONDS_PER_DAY),
        sql=f"GREATEST(0.0, ({{t.ts}} - {{w.first_seen_ts}}) / {SECONDS_PER_DAY}.0)",
    ),
    FeatureFormula(
        name="seconds_since_last_trade",
        dtype="int",
        nullable=True,
        py=lambda i: (
            i.trade.ts - i.wallet.last_trade_ts if i.wallet.last_trade_ts is not None else None
        ),
        sql=(
            "CASE WHEN {w.last_trade_ts} IS NOT NULL THEN {t.ts} - {w.last_trade_ts} ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="market_age_seconds",
        dtype="int",
        nullable=False,
        # When the market has not yet been observed, compute_features reads
        # market_state's default empty state (market_age_start_ts=0), so
        # market_age_seconds = trade.ts on first sighting. The SQL mirrors
        # this with COALESCE-to-0 (see _duckdb_engine.py:798-801 comment).
        py=lambda i: i.trade.ts - i.market.market_age_start_ts,
        sql="CAST({t.ts} - COALESCE({m.market_age_start_ts}, 0) AS INTEGER)",
    ),
    FeatureFormula(
        name="time_to_resolution_seconds",
        dtype="int",
        nullable=True,
        py=lambda i: i.meta.closed_at - i.trade.ts,
        sql="CAST({meta.closed_at} - {t.ts} AS INTEGER)",
    ),
    # ----- Category counts (from wallet_cat_summary subquery on the SQL side) -----
    FeatureFormula(
        name="top_category",
        dtype="str",
        nullable=True,
        py=lambda i: (
            max(i.wallet.category_counts.items(), key=lambda kv: kv[1])[0]
            if i.wallet.category_counts
            else None
        ),
        sql="{wcs.top_category}",
    ),
    FeatureFormula(
        name="category_diversity",
        dtype="int",
        nullable=False,
        py=lambda i: len(i.wallet.category_counts),
        sql="{wcs.category_diversity}",
    ),
    # ----- Price volatility -----
    FeatureFormula(
        name="price_volatility_recent",
        dtype="float",
        nullable=True,
        py=lambda i: (
            statistics.pstdev(i.market.recent_prices)
            if len(i.market.recent_prices) >= MIN_PRICES_FOR_VOLATILITY
            else None
        ),
        sql=(
            f"CASE WHEN {{m.price_count_20}} >= {MIN_PRICES_FOR_VOLATILITY} "
            "THEN {m.price_volatility} ELSE NULL END"
        ),
    ),
    # ----- Market categories (tuple-valued; sql returns a list expression) -----
    FeatureFormula(
        name="market_categories",
        dtype="tuple_str",
        nullable=False,
        project_to_sql=False,
        py=lambda i: i.meta.categories if i.meta.categories else (i.meta.category,),
        sql=(
            "CASE "
            "WHEN json_array_length(COALESCE({meta.categories_json}, '[]')) > 0 "
            "THEN CAST(json_extract({meta.categories_json}, '$') AS VARCHAR[]) "
            "ELSE [{meta.category}] "
            "END"
        ),
    ),
)

# Extend FEATURES with the generated cat_* indicators. Defined as a
# separate statement (rather than expanding inside the FEATURES literal)
# because the closure capture inside _cat_indicator_formula requires a
# helper function — see issue #145 for the parity rationale.
FEATURES = FEATURES + tuple(_cat_indicator_formula(cat) for cat in KNOWN_CATEGORIES)


def project_sql(*, bindings: Mapping[str, str] = SQL_BINDINGS) -> str:
    """Emit the SELECT-list column expressions for ``training_examples_v2``.

    Skips formulas where ``project_to_sql=False`` (compute-only features
    that exist on FeatureRow but aren't persisted to the SQLite schema).

    Returns a comma-separated string of ``<expression> AS <column_name>``
    lines, ready to splice into the ``_final_join_to_v2`` SELECT. The
    caller is responsible for the surrounding SELECT scaffolding
    (platform, tx_hash, label_won, JOIN clauses, WHERE).
    """
    parts = []
    for formula in FEATURES:
        if not formula.project_to_sql:
            continue
        rendered = render_sql_fragment(formula.sql, bindings)
        parts.append(f"{rendered} AS {formula.name}")
    return ",\n    ".join(parts)


def project_row(
    *,
    trade: Trade,
    wallet: WalletState,
    market: MarketState,
    meta: MarketMetadata,
) -> FeatureRow:
    """Compute a FeatureRow from point-in-time state.

    Walks ``FEATURES``, evaluates each formula's ``py`` against
    ``FeatureInputs``, and packages the result into a FeatureRow.
    """
    inputs = FeatureInputs(wallet=wallet, market=market, meta=meta, trade=trade)
    values = {formula.name: formula.py(inputs) for formula in FEATURES}
    return FeatureRow(**values)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]


def compute_features(trade: Trade, history: HistoryProvider) -> FeatureRow:
    """Compute the full feature row for a trade, point-in-time correct.

    Thin wrapper around :func:`project_row`; the canonical formulas live
    in the ``FEATURES`` registry above.

    Pure function: takes only ``trade`` and ``history``. All
    non-determinism enters via the provider.
    """
    return project_row(
        trade=trade,
        wallet=history.wallet_state(trade.wallet_address, as_of_ts=trade.ts),
        market=history.market_state(trade.condition_id, as_of_ts=trade.ts),
        meta=history.market_metadata(trade.condition_id),
    )
