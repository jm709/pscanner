"""Unit tests for LiveHistoryProvider (#78)."""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus.features import Trade
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


@dataclasses.dataclass(frozen=True)
class _FakePosition:
    condition_id: str
    side: str  # YES | NO
    avg_price: float  # implied prob paid
    size: float  # # shares bought
    notional_usd: float
    opened_at: int
    closed_at: int
    won: bool


class _FakeDataClient:
    def __init__(self, positions: list[_FakePosition]) -> None:
        self._positions = positions

    async def get_closed_positions_for_bootstrap(
        self, address: str, *, limit: int = 500
    ) -> list[_FakePosition]:
        del address, limit
        return list(self._positions)


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


def test_restart_preserves_wallet_state(tmp_path: Path) -> None:
    db_path = tmp_path / "daemon.sqlite3"
    conn1 = init_db(db_path)
    try:
        provider1 = LiveHistoryProvider(conn=conn1, metadata={})
        provider1.observe(_make_trade(bs="BUY", ts=1_700_000_000, price=0.40))
        provider1.observe(
            _make_trade(bs="BUY", ts=1_700_000_100, price=0.45, condition_id="0xcond2")
        )
        before = provider1.wallet_state("0xabc", as_of_ts=1_700_000_200)
    finally:
        conn1.close()
    conn2 = init_db(db_path)
    try:
        provider2 = LiveHistoryProvider(conn=conn2, metadata={})
        after = provider2.wallet_state("0xabc", as_of_ts=1_700_000_200)
    finally:
        conn2.close()
    assert before == after


@pytest.mark.asyncio
async def test_bootstrap_wallet_folds_closed_positions() -> None:
    conn = _new_conn()
    try:
        positions = [
            _FakePosition(
                condition_id="0xc1",
                side="YES",
                avg_price=0.40,
                size=100.0,
                notional_usd=40.0,
                opened_at=1_699_000_000,
                closed_at=1_699_500_000,
                won=True,
            ),
            _FakePosition(
                condition_id="0xc2",
                side="NO",
                avg_price=0.20,
                size=50.0,
                notional_usd=10.0,
                opened_at=1_699_000_500,
                closed_at=1_699_500_500,
                won=False,
            ),
        ]
        provider = LiveHistoryProvider(conn=conn, metadata={})
        await provider.bootstrap_wallet("0xabc", data_client=_FakeDataClient(positions))
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_000_000)
    finally:
        conn.close()
    assert wallet.prior_buys_count == 2
    assert wallet.prior_resolved_buys == 2
    assert wallet.prior_wins == 1
    assert wallet.prior_losses == 1
    assert wallet.realized_pnl_usd == pytest.approx(60.0 - 10.0)


@pytest.mark.asyncio
async def test_bootstrap_wallet_is_idempotent_for_known_wallet() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        # Seed with a real observed BUY first.
        provider.observe(_make_trade(bs="BUY", wallet="0xabc", ts=1_700_000_000))
        before = provider.wallet_state("0xabc", as_of_ts=1_700_000_500)
        # Now bootstrap should NO-OP (wallet already exists).
        positions = [
            _FakePosition(
                condition_id="0xother",
                side="YES",
                avg_price=0.30,
                size=10.0,
                notional_usd=3.0,
                opened_at=1_699_000_000,
                closed_at=1_699_500_000,
                won=True,
            )
        ]
        await provider.bootstrap_wallet("0xabc", data_client=_FakeDataClient(positions))
        after = provider.wallet_state("0xabc", as_of_ts=1_700_000_500)
    finally:
        conn.close()
    assert before == after
