"""SQLite-backed history provider for the live daemon (#78).

Mirrors ``StreamingHistoryProvider`` from ``pscanner.corpus.features`` but
persists state to ``wallet_state_live`` + ``market_state_live`` so daemon
restarts are O(1) instead of O(corpus). Implements the ``HistoryProvider``
Protocol so ``compute_features`` can consume it unchanged.

The accumulator semantics are point-for-point identical to the streaming
provider — the only difference is storage. The same pure
``apply_*_to_state`` functions in ``pscanner.corpus.features`` drive both
providers, which is the parity contract validated by
``tests/daemon/test_live_history_parity.py``.
"""

from __future__ import annotations

import heapq
import json
import sqlite3
from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

from pscanner.corpus.features import (
    MarketMetadata,
    MarketState,
    Trade,
    WalletState,
    _TradeFields,  # re-using the structural Protocol from features
    apply_buy_to_state,
    apply_resolution_to_state,
    apply_sell_to_state,
    apply_trade_to_market,
    empty_market_state,
    empty_wallet_state,
)
from pscanner.daemon._state_persistence import (
    MARKET_STATE_UPSERT_SQL,
    WALLET_STATE_UPSERT_SQL,
    market_state_from_row,
    market_state_to_row,
    market_traders_from_row,
    wallet_state_from_row,
    wallet_state_to_row,
)


@runtime_checkable
class BootstrapPosition(Protocol):
    """Structural shape of one closed position used by ``bootstrap_wallet``.

    Both Polymarket's ``ClosedPosition`` API model (wrapped) and test fakes
    satisfy this Protocol.
    """

    condition_id: str
    side: str  # "YES" | "NO"
    avg_price: float
    size: float
    notional_usd: float
    opened_at: int
    closed_at: int
    won: bool


class BootstrapDataClient(Protocol):
    """Subset of ``DataClient`` needed to bootstrap an unseen wallet."""

    async def get_closed_positions_for_bootstrap(
        self, address: str, *, limit: int = 500
    ) -> Sequence[BootstrapPosition]:
        """Fetch closed positions for the given wallet address.

        Return type is ``Sequence`` (covariant) rather than ``list`` so
        structurally-compatible position dataclasses fit without an
        explicit cast at call sites.
        """
        ...


class LiveHistoryProvider:
    """Persistent ``HistoryProvider`` backed by SQLite.

    The provider holds an open ``sqlite3.Connection`` for the daemon DB
    (the same one returned by ``init_db``). All reads/writes happen on
    that connection — caller owns the connection lifecycle.

    Args:
        conn: Open daemon-DB connection (schema applied). Must remain
            open for the lifetime of the provider.
        metadata: Map of ``condition_id -> MarketMetadata`` used by
            ``market_metadata`` and the time-to-resolution feature. The
            daemon refreshes this dict from ``market_resolutions`` on a
            cadence (out of scope for this class).
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        metadata: dict[str, MarketMetadata],
    ) -> None:
        """Bind to an open daemon-DB connection and a metadata mapping."""
        self._conn = conn
        self._metadata = metadata
        self._resolutions: dict[str, tuple[int, int]] = {}

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        """Return static metadata for ``condition_id``; raises KeyError if unknown."""
        return self._metadata[condition_id]

    def set_market_metadata(self, condition_id: str, metadata: MarketMetadata) -> None:
        """Insert or overwrite metadata for ``condition_id``.

        Used by :class:`MarketScopedTradeCollector` to push metadata for
        currently-open markets that aren't yet in ``corpus_markets`` — the
        boot-time corpus load only covers resolved markets, so live trading
        targets need this runtime injection (issue #102).
        """
        self._metadata[condition_id] = metadata

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        """Return the wallet's state at ``as_of_ts``, draining ready resolutions."""
        row = self._conn.execute(
            "SELECT * FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if row is None:
            return empty_wallet_state(first_seen_ts=as_of_ts)
        state = wallet_state_from_row(row)
        unresolved = list(json.loads(row["unresolved_buys_json"]))
        new_state, remaining = self._drain_resolved_buys(state, unresolved, as_of_ts)
        if remaining is not unresolved:
            self._persist_wallet(wallet_address, new_state, remaining)
        return new_state

    def _drain_resolved_buys(
        self,
        state: WalletState,
        unresolved: list[dict[str, object]],
        as_of_ts: int,
    ) -> tuple[WalletState, list[dict[str, object]]]:
        """Apply resolutions whose ``resolved_at < as_of_ts`` to ``state``.

        Returns the (possibly mutated) state plus the list of buys still
        waiting for resolution (or resolved past the ``as_of_ts`` cutoff).
        """
        ready: list[tuple[int, int, dict[str, object]]] = []
        deferred: list[dict[str, object]] = []
        for idx, buy in enumerate(unresolved):
            cond_id = cast(str, buy["condition_id"])
            resolution = self._resolutions.get(cond_id)
            if resolution is None:
                deferred.append(buy)
                continue
            resolved_at, _ = resolution
            heapq.heappush(ready, (resolved_at, idx, buy))
        if not ready:
            return state, unresolved
        leftover: list[dict[str, object]] = list(deferred)
        while ready:
            resolved_at, _, buy = heapq.heappop(ready)
            if resolved_at >= as_of_ts:
                leftover.append(buy)
                continue
            cond_id = cast(str, buy["condition_id"])
            _, yes_won = self._resolutions[cond_id]
            side_yes = bool(buy["side_yes"])
            won = (yes_won == 1) if side_yes else (yes_won == 0)
            size = float(buy["size"])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            notional = float(buy["notional_usd"])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            payout = size if won else 0.0
            state = apply_resolution_to_state(
                state,
                won=won,
                notional_usd=notional,
                payout_usd=payout,
            )
        return state, leftover

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        """Return per-market running state.

        ``as_of_ts`` is unused — caller must query before observing the
        next event for the same market (parity with streaming provider).
        """
        del as_of_ts
        row = self._conn.execute(
            "SELECT * FROM market_state_live WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None:
            return empty_market_state(market_age_start_ts=0)
        return market_state_from_row(row)

    def observe(self, trade: Trade) -> None:
        """Fold a trade into running wallet + market state.

        BUY rows update the wallet's running aggregates AND append an
        unresolved-buy entry to the wallet's serialized list so it can be
        drained when ``register_resolution`` fires (Task 4).
        """
        wallet = self.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
        unresolved = self._load_unresolved(trade.wallet_address)
        if trade.bs == "BUY":
            new_state = apply_buy_to_state(wallet, trade)
            unresolved.append(
                {
                    "condition_id": trade.condition_id,
                    "notional_usd": trade.notional_usd,
                    "size": trade.size,
                    "side_yes": trade.outcome_side == "YES",
                    "ts": trade.ts,
                }
            )
        elif trade.bs == "SELL":
            new_state = apply_sell_to_state(wallet, trade)
        else:
            return
        self._persist_wallet(trade.wallet_address, new_state, unresolved)
        self._observe_market(trade)

    def observe_sell(self, trade: _TradeFields) -> None:
        """Fold a SELL fill (no ``category`` required) into wallet + market state."""
        wallet = self.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
        new_state = apply_sell_to_state(wallet, trade)
        unresolved = self._load_unresolved(trade.wallet_address)
        self._persist_wallet(trade.wallet_address, new_state, unresolved)
        self._observe_market(trade)

    def _observe_market(self, trade: _TradeFields) -> None:
        market_row = self._conn.execute(
            "SELECT * FROM market_state_live WHERE condition_id = ?",
            (trade.condition_id,),
        ).fetchone()
        if market_row is None:
            current = empty_market_state(market_age_start_ts=trade.ts)
            traders: set[str] = set()
        else:
            current = market_state_from_row(market_row)
            traders = market_traders_from_row(market_row)
        is_new_trader = trade.wallet_address not in traders
        if is_new_trader:
            traders.add(trade.wallet_address)
        new_state = apply_trade_to_market(current, trade, is_new_trader=is_new_trader)
        self._persist_market(trade.condition_id, new_state, traders)

    def _load_unresolved(self, wallet_address: str) -> list[dict[str, object]]:
        row = self._conn.execute(
            "SELECT unresolved_buys_json FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if row is None:
            return []
        return list(json.loads(row["unresolved_buys_json"]))

    def _persist_wallet(
        self,
        wallet_address: str,
        state: WalletState,
        unresolved: list[dict[str, object]],
    ) -> None:
        self._conn.execute(
            WALLET_STATE_UPSERT_SQL,
            wallet_state_to_row(
                wallet_address,
                state,
                unresolved_buys_json=json.dumps(unresolved),
            ),
        )
        self._conn.commit()

    def _persist_market(self, condition_id: str, state: MarketState, traders: set[str]) -> None:
        self._conn.execute(
            MARKET_STATE_UPSERT_SQL,
            market_state_to_row(condition_id, state, traders=traders),
        )
        self._conn.commit()

    async def bootstrap_wallet(
        self,
        wallet_address: str,
        *,
        data_client: BootstrapDataClient,
        limit: int = 500,
    ) -> None:
        """Pre-warm wallet state from ``/positions?user=X&closed=true``.

        Folds historical closed positions into the wallet's running state so
        subsequent feature reads have prior_resolved_buys / win_rate /
        realized_pnl populated for previously-unseen wallets.

        Idempotent: no-ops if the wallet already has a row in
        ``wallet_state_live``.

        Note: positions are folded with ``category=""`` because the offline
        bootstrap path does not know the market category.  The ``""`` key will
        appear in ``category_counts`` for bootstrapped wallets; this is
        acceptable for v1 and the gate-model feature builder ignores the raw
        category map (it uses ``top_category`` derived from it).
        """
        existing = self._conn.execute(
            "SELECT 1 FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if existing is not None:
            return
        positions = await data_client.get_closed_positions_for_bootstrap(
            wallet_address, limit=limit
        )
        positions_sorted = sorted(positions, key=lambda p: p.opened_at)
        first_seen = positions_sorted[0].opened_at if positions_sorted else 0
        state = empty_wallet_state(first_seen_ts=first_seen)
        for position in positions_sorted:
            buy_trade = Trade(
                tx_hash=f"bootstrap:{wallet_address}:{position.condition_id}",
                asset_id=f"{position.condition_id}-{position.side}",
                wallet_address=wallet_address,
                condition_id=position.condition_id,
                outcome_side=position.side,
                bs="BUY",
                price=position.avg_price,
                size=position.size,
                notional_usd=position.notional_usd,
                ts=position.opened_at,
                category="",
            )
            state = apply_buy_to_state(state, buy_trade)
            payout = position.size if position.won else 0.0
            state = apply_resolution_to_state(
                state,
                won=position.won,
                notional_usd=position.notional_usd,
                payout_usd=payout,
            )
        self._persist_wallet(wallet_address, state, [])

    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None:
        """Record a market's resolution.

        Subsequent ``wallet_state`` queries past ``resolved_at`` drain
        any unresolved buys against this market for the queried wallet.
        """
        self._resolutions[condition_id] = (resolved_at, outcome_yes_won)

    def get_resolution(self, condition_id: str) -> tuple[int, int] | None:
        """Return ``(resolved_at, outcome_yes_won)`` if known, else ``None``."""
        return self._resolutions.get(condition_id)
