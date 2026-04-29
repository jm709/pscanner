"""Tests for ``StreamingHistoryProvider``."""

from __future__ import annotations

import pytest

from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
)


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
        "ts": 1_000,
        "category": "crypto",
    }
    base.update(kwargs)
    return Trade(**base)  # type: ignore[arg-type]


def _meta(condition_id: str = "cond1") -> MarketMetadata:
    return MarketMetadata(
        condition_id=condition_id,
        category="crypto",
        closed_at=10_000,
        opened_at=500,
    )


def test_observe_buy_then_query_returns_buy_state() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000))
    state = provider.wallet_state("0xw", as_of_ts=2_000)
    assert state.prior_buys_count == 1


def test_observe_then_query_at_same_ts_excludes_event() -> None:
    """Querying at the SAME ts as an observed event should EXCLUDE the event,
    because features are computed BEFORE the event is folded into state.
    """
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    state_before = provider.wallet_state("0xw", as_of_ts=1_000)
    provider.observe(_trade(tx_hash="0xa", ts=1_000))
    assert state_before.prior_buys_count == 0


def test_resolutions_register_resolves_pending_buys() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000, notional_usd=40.0, price=0.4, size=100.0))
    provider.register_resolution(condition_id="cond1", resolved_at=5_000, outcome_yes_won=1)
    state = provider.wallet_state("0xw", as_of_ts=6_000)
    assert state.prior_resolved_buys == 1
    assert state.prior_wins == 1
    assert state.realized_pnl_usd == pytest.approx(60.0)


def test_resolution_pre_query_only_counts_if_resolution_ts_lt_query_ts() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000, size=100.0, price=0.4, notional_usd=40.0))
    provider.register_resolution(condition_id="cond1", resolved_at=5_000, outcome_yes_won=1)
    state_before = provider.wallet_state("0xw", as_of_ts=4_000)
    state_after = provider.wallet_state("0xw", as_of_ts=6_000)
    assert state_before.prior_resolved_buys == 0
    assert state_after.prior_resolved_buys == 1


def test_market_state_tracks_price_history() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000, price=0.4))
    provider.observe(_trade(tx_hash="0xb", ts=2_000, price=0.5, wallet_address="0xv"))
    market = provider.market_state("cond1", as_of_ts=3_000)
    assert market.last_trade_price == pytest.approx(0.5)
    assert len(market.unique_traders_so_far) == 2
    assert market.volume_so_far_usd > 0


def test_unknown_market_metadata_raises_keyerror() -> None:
    provider = StreamingHistoryProvider(metadata={})
    with pytest.raises(KeyError):
        provider.market_metadata("unknown")


def test_resolution_heap_drains_in_order() -> None:
    """Property-style: random interleaving of trades + resolutions; the
    invariant ``prior_wins + prior_losses`` always equals the count of
    prior buys whose resolution_ts < query_ts.
    """
    provider = StreamingHistoryProvider(
        metadata={
            "c1": _meta("c1"),
            "c2": _meta("c2"),
            "c3": _meta("c3"),
        }
    )
    provider.observe(
        _trade(
            tx_hash="0xa",
            ts=100,
            condition_id="c1",
            size=100.0,
            price=0.5,
            notional_usd=50.0,
        )
    )
    provider.observe(
        _trade(
            tx_hash="0xb",
            ts=200,
            condition_id="c2",
            size=100.0,
            price=0.5,
            notional_usd=50.0,
        )
    )
    provider.observe(
        _trade(
            tx_hash="0xc",
            ts=300,
            condition_id="c3",
            size=100.0,
            price=0.5,
            notional_usd=50.0,
        )
    )
    provider.register_resolution(condition_id="c1", resolved_at=1_000, outcome_yes_won=1)
    provider.register_resolution(condition_id="c2", resolved_at=5_000, outcome_yes_won=0)
    provider.register_resolution(condition_id="c3", resolved_at=3_000, outcome_yes_won=1)

    state_2k = provider.wallet_state("0xw", as_of_ts=2_000)
    assert state_2k.prior_resolved_buys == 1
    state_4k = provider.wallet_state("0xw", as_of_ts=4_000)
    assert state_4k.prior_resolved_buys == 2
    state_10k = provider.wallet_state("0xw", as_of_ts=10_000)
    assert state_10k.prior_resolved_buys == 3
    assert state_10k.prior_wins == 2
    assert state_10k.prior_losses == 1
