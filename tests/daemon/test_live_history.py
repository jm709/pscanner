"""Unit tests for LiveHistoryProvider (#78)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus.features import Trade
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


def _new_conn() -> sqlite3.Connection:
    return init_db(Path(":memory:"))


def _make_trade(
    *,
    bs: str = "BUY",
    wallet: str = "0xabc",
    condition_id: str = "0xcond",
    side: str = "YES",
    price: float = 0.42,
    size: float = 100.0,
    notional_usd: float = 42.0,
    ts: int = 1_700_000_000,
    category: str = "esports",
) -> Trade:
    return Trade(
        tx_hash=f"tx-{ts}-{bs}",
        asset_id="0xasset",
        wallet_address=wallet,
        condition_id=condition_id,
        outcome_side=side,
        bs=bs,
        price=price,
        size=size,
        notional_usd=notional_usd,
        ts=ts,
        category=category,
    )


def test_wallet_state_returns_empty_for_unknown_wallet() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        state = provider.wallet_state("0xabc", as_of_ts=1_700_000_000)
    finally:
        conn.close()
    assert state.first_seen_ts == 1_700_000_000
    assert state.prior_trades_count == 0
    assert state.prior_buys_count == 0
    assert state.prior_wins == 0
    assert state.recent_30d_trades == ()
    assert state.category_counts == {}


def test_observe_buy_persists_wallet_and_market_state() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        trade = _make_trade(bs="BUY", price=0.42, notional_usd=42.0)
        provider.observe(trade)
        wallet = provider.wallet_state("0xabc", as_of_ts=trade.ts + 1)
        market = provider.market_state("0xcond", as_of_ts=trade.ts + 1)
    finally:
        conn.close()
    assert wallet.prior_trades_count == 1
    assert wallet.prior_buys_count == 1
    assert wallet.cumulative_buy_count == 1
    assert wallet.cumulative_buy_price_sum == pytest.approx(0.42)
    assert wallet.bet_size_count == 1
    assert wallet.bet_size_sum == pytest.approx(42.0)
    assert wallet.category_counts == {"esports": 1}
    assert market.unique_traders_count == 1
    assert market.last_trade_price == pytest.approx(0.42)
    assert market.volume_so_far_usd == pytest.approx(42.0)


def test_observe_sell_updates_wallet_and_market_state() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        provider.observe(_make_trade(bs="BUY", ts=1_700_000_000))
        provider.observe(_make_trade(bs="SELL", ts=1_700_000_100, price=0.55))
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_000_200)
        market = provider.market_state("0xcond", as_of_ts=1_700_000_200)
    finally:
        conn.close()
    assert wallet.prior_trades_count == 2
    assert wallet.prior_buys_count == 1  # SELL doesn't increment buys
    assert market.last_trade_price == pytest.approx(0.55)


def test_register_resolution_drains_buy_to_win() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        provider.observe(
            _make_trade(
                bs="BUY",
                side="YES",
                price=0.40,
                size=100.0,
                notional_usd=40.0,
                ts=1_700_000_000,
            )
        )
        provider.register_resolution(
            condition_id="0xcond",
            resolved_at=1_700_001_000,
            outcome_yes_won=1,
        )
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_001_500)
    finally:
        conn.close()
    assert wallet.prior_resolved_buys == 1
    assert wallet.prior_wins == 1
    assert wallet.prior_losses == 0
    assert wallet.realized_pnl_usd == pytest.approx(60.0)  # 100 - 40
