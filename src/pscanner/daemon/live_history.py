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

import json
import sqlite3
from typing import TYPE_CHECKING

from pscanner.corpus.features import (
    MarketMetadata,
    MarketState,
    WalletState,
    empty_market_state,
    empty_wallet_state,
)

if TYPE_CHECKING:
    from pscanner.corpus.features import Trade, _TradeFields


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

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        """Return static metadata for ``condition_id``; raises KeyError if unknown."""
        return self._metadata[condition_id]

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        """Return the wallet's state at ``as_of_ts``.

        ``as_of_ts`` is used only as the seed ``first_seen_ts`` for an
        unknown wallet (matching ``StreamingHistoryProvider`` semantics).
        Resolution drain is implemented in Task 4.
        """
        row = self._conn.execute(
            "SELECT * FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if row is None:
            return empty_wallet_state(first_seen_ts=as_of_ts)
        return WalletState(
            first_seen_ts=row["first_seen_ts"],
            prior_trades_count=row["prior_trades_count"],
            prior_buys_count=row["prior_buys_count"],
            prior_resolved_buys=row["prior_resolved_buys"],
            prior_wins=row["prior_wins"],
            prior_losses=row["prior_losses"],
            cumulative_buy_price_sum=row["cumulative_buy_price_sum"],
            cumulative_buy_count=row["cumulative_buy_count"],
            realized_pnl_usd=row["realized_pnl_usd"],
            last_trade_ts=row["last_trade_ts"],
            recent_30d_trades=tuple(json.loads(row["recent_30d_trades_json"])),
            bet_size_sum=row["bet_size_sum"],
            bet_size_count=row["bet_size_count"],
            category_counts=dict(json.loads(row["category_counts_json"])),
        )

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
        return MarketState(
            market_age_start_ts=row["market_age_start_ts"],
            volume_so_far_usd=row["volume_so_far_usd"],
            unique_traders_count=row["unique_traders_count"],
            last_trade_price=row["last_trade_price"],
            recent_prices=tuple(json.loads(row["recent_prices_json"])),
        )

    def observe(self, trade: Trade) -> None:
        """Fold a trade into running wallet + market state. Implemented in Task 3."""
        raise NotImplementedError

    def observe_sell(self, trade: _TradeFields) -> None:
        """Fold a SELL fill into wallet + market state. Implemented in Task 3."""
        raise NotImplementedError

    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None:
        """Record a market's resolution. Implemented in Task 4."""
        raise NotImplementedError
