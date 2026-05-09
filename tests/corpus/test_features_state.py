"""Tests for the pure state-update functions in ``corpus.features``."""

from __future__ import annotations

import math
from collections import deque

import pytest

from pscanner.corpus.features import (
    Trade,
    apply_buy_to_state,
    apply_resolution_to_state,
    apply_sell_to_state,
    empty_wallet_state,
)


def _trade(**kwargs: object) -> Trade:
    base = {
        "tx_hash": "0x" + str(kwargs.get("tx_hash", "a")),
        "asset_id": "a1",
        "wallet_address": "0xw",
        "condition_id": kwargs.get("condition_id", "cond1"),
        "outcome_side": "YES",
        "bs": kwargs.get("bs", "BUY"),
        "price": float(kwargs.get("price", 0.4)),  # type: ignore[arg-type]
        "size": float(kwargs.get("size", 100.0)),  # type: ignore[arg-type]
        "notional_usd": float(kwargs.get("notional_usd", 40.0)),  # type: ignore[arg-type]
        "ts": int(kwargs.get("ts", 1_000)),  # type: ignore[arg-type]
        "category": kwargs.get("category", "crypto"),
    }
    return Trade(**base)  # type: ignore[arg-type]


def test_empty_wallet_state_has_zero_counts() -> None:
    state = empty_wallet_state(first_seen_ts=500)
    assert state.first_seen_ts == 500
    assert state.prior_trades_count == 0
    assert state.prior_buys_count == 0
    assert state.prior_resolved_buys == 0
    assert state.prior_wins == 0
    assert state.prior_losses == 0
    assert state.cumulative_buy_price_sum == 0.0
    assert state.cumulative_buy_count == 0
    assert state.realized_pnl_usd == 0.0
    assert state.last_trade_ts is None
    assert state.recent_30d_trades == deque()
    assert state.bet_size_sum == 0.0
    assert state.bet_size_count == 0
    assert state.category_counts == {}


def test_apply_buy_increments_counts_and_records_price() -> None:
    state = empty_wallet_state(first_seen_ts=500)
    new_state = apply_buy_to_state(state, _trade(price=0.4, notional_usd=40.0))
    assert new_state.prior_trades_count == 1
    assert new_state.prior_buys_count == 1
    assert new_state.cumulative_buy_count == 1
    assert new_state.cumulative_buy_price_sum == pytest.approx(0.4)
    assert new_state.last_trade_ts == 1_000
    assert new_state.bet_size_sum == pytest.approx(40.0)
    assert new_state.bet_size_count == 1
    assert new_state.category_counts == {"crypto": 1}


def test_apply_sell_increments_total_but_not_buy() -> None:
    state = empty_wallet_state(first_seen_ts=500)
    new_state = apply_sell_to_state(state, _trade(bs="SELL"))
    assert new_state.prior_trades_count == 1
    assert new_state.prior_buys_count == 0


def test_apply_buy_appends_recent_window() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = apply_buy_to_state(state, _trade(tx_hash="a", ts=1_000))
    state = apply_buy_to_state(state, _trade(tx_hash="b", ts=2_000))
    assert state.recent_30d_trades == deque([1_000, 2_000])


def test_apply_resolution_records_win() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = apply_buy_to_state(state, _trade(price=0.4, notional_usd=40.0))
    state = apply_resolution_to_state(state, won=True, notional_usd=40.0, payout_usd=100.0)
    assert state.prior_resolved_buys == 1
    assert state.prior_wins == 1
    assert state.prior_losses == 0
    assert state.realized_pnl_usd == pytest.approx(60.0)


def test_apply_resolution_records_loss() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = apply_buy_to_state(state, _trade(notional_usd=40.0))
    state = apply_resolution_to_state(state, won=False, notional_usd=40.0, payout_usd=0.0)
    assert state.prior_resolved_buys == 1
    assert state.prior_wins == 0
    assert state.prior_losses == 1
    assert state.realized_pnl_usd == pytest.approx(-40.0)


def test_state_immutability() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state2 = apply_buy_to_state(state, _trade())
    assert state.prior_trades_count == 0
    assert state2.prior_trades_count == 1
    assert math.isclose(state.cumulative_buy_price_sum, 0.0)
