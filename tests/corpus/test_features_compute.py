"""Tests for ``compute_features`` and the ``HistoryProvider`` Protocol."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from pscanner.corpus.features import (
    HistoryProvider,
    MarketMetadata,
    MarketState,
    Trade,
    WalletState,
    compute_features,
    empty_market_state,
    empty_wallet_state,
)


@dataclass
class _StubHistory:
    wallet: WalletState
    market: MarketState
    meta: MarketMetadata

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        del wallet_address, as_of_ts
        return self.wallet

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        del condition_id, as_of_ts
        return self.market

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        del condition_id
        return self.meta


def _trade(**kwargs: object) -> Trade:
    base = {
        "tx_hash": "0xa",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "outcome_side": "YES",
        "bs": "BUY",
        "price": 0.4,
        "size": 100.0,
        "notional_usd": 40.0,
        "ts": 1_000_000,
        "category": "crypto",
    }
    base.update(kwargs)
    return Trade(**base)  # type: ignore[arg-type]


def _meta(**kwargs: object) -> MarketMetadata:
    base = {
        "condition_id": "cond1",
        "category": "crypto",
        "closed_at": 2_000_000,
        "opened_at": 500_000,
    }
    base.update(kwargs)
    return MarketMetadata(**base)  # type: ignore[arg-type]


def test_compute_features_no_prior_history_yields_nulls() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=1_000_000),
        market=empty_market_state(market_age_start_ts=500_000),
        meta=_meta(),
    )
    features = compute_features(_trade(), history)
    assert features.prior_trades_count == 0
    assert features.win_rate is None
    assert features.avg_implied_prob_paid is None
    assert features.realized_edge_pp is None
    assert features.bet_size_rel_to_avg is None
    assert features.seconds_since_last_trade is None
    assert features.top_category is None
    assert features.last_trade_price is None
    assert features.price_volatility_recent is None
    assert features.bet_size_usd == pytest.approx(40.0)
    assert features.implied_prob_at_buy == pytest.approx(0.4)


def test_compute_features_with_one_resolved_buy() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = state.__class__(
        first_seen_ts=0,
        prior_trades_count=1,
        prior_buys_count=1,
        prior_resolved_buys=1,
        prior_wins=1,
        prior_losses=0,
        cumulative_buy_price_sum=0.3,
        cumulative_buy_count=1,
        realized_pnl_usd=70.0,
        last_trade_ts=900_000,
        recent_30d_trades=deque([900_000]),
        bet_size_sum=30.0,
        bet_size_count=1,
        category_counts={"crypto": 1},
    )
    history: HistoryProvider = _StubHistory(
        wallet=state,
        market=empty_market_state(market_age_start_ts=500_000),
        meta=_meta(),
    )
    features = compute_features(_trade(notional_usd=60.0), history)
    assert features.win_rate == pytest.approx(1.0)
    assert features.avg_implied_prob_paid == pytest.approx(0.3)
    assert features.realized_edge_pp == pytest.approx(0.7)
    assert features.avg_bet_size_usd == pytest.approx(30.0)
    assert features.bet_size_rel_to_avg == pytest.approx(60.0 / 30.0)
    assert features.seconds_since_last_trade == 100_000
    assert features.top_category == "crypto"
    assert features.category_diversity == 1
    assert features.prior_realized_pnl_usd == pytest.approx(70.0)


def test_compute_features_market_features() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=1_000_000),
        market=MarketState(
            market_age_start_ts=500_000,
            volume_so_far_usd=12_345.0,
            unique_traders_count=3,
            last_trade_price=0.45,
            recent_prices=(0.4, 0.42, 0.45),
        ),
        meta=_meta(),
    )
    features = compute_features(_trade(ts=900_000), history)
    assert features.market_volume_so_far_usd == pytest.approx(12_345.0)
    assert features.market_unique_traders_so_far == 3
    assert features.market_age_seconds == 400_000
    assert features.time_to_resolution_seconds == 1_100_000
    assert features.last_trade_price == pytest.approx(0.45)
    assert features.price_volatility_recent is not None


def test_compute_features_implied_prob_for_no_side() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=0),
        market=empty_market_state(market_age_start_ts=0),
        meta=_meta(),
    )
    features = compute_features(_trade(outcome_side="NO", price=0.7), history)
    assert features.implied_prob_at_buy == pytest.approx(0.7)
    assert features.side == "NO"


def test_compute_features_volatility_null_with_few_prices() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=0),
        market=MarketState(
            market_age_start_ts=0,
            volume_so_far_usd=100.0,
            unique_traders_count=1,
            last_trade_price=0.5,
            recent_prices=(0.5,),
        ),
        meta=_meta(),
    )
    features = compute_features(_trade(), history)
    assert features.price_volatility_recent is None
