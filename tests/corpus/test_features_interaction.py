"""Tests for wallet-quality x confidence interaction features.

Covers ``edge_confidence_weighted``, ``win_rate_confidence_weighted``,
``is_high_quality_wallet``, and ``bet_size_relative_to_history`` as
computed by ``compute_features`` and stored in ``training_examples``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from pscanner.corpus.examples import build_features
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
from pscanner.corpus.repos import (
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def _meta() -> MarketMetadata:
    return MarketMetadata(
        condition_id="cond1",
        category="crypto",
        closed_at=2_000_000,
        opened_at=500_000,
    )


def _trade(**kwargs: object) -> Trade:
    base: dict[str, object] = {
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


def _wallet_with_history(
    *,
    prior_resolved_buys: int,
    prior_wins: int,
    realized_edge_pp: float | None = None,
    bet_size_sum: float = 0.0,
    bet_size_count: int = 0,
) -> WalletState:
    """Build a WalletState with specific resolved-buy history."""
    cumulative_buy_price_sum = 0.0
    cumulative_buy_count = prior_resolved_buys
    # Back-calculate: set a fixed avg_prob so edge = win_rate - avg_prob
    # Use avg_prob = 0.5 as a neutral baseline so win_rate is the driver.
    if prior_resolved_buys > 0:
        cumulative_buy_price_sum = 0.5 * prior_resolved_buys

    return WalletState(
        first_seen_ts=0,
        prior_trades_count=prior_resolved_buys,
        prior_buys_count=prior_resolved_buys,
        prior_resolved_buys=prior_resolved_buys,
        prior_wins=prior_wins,
        prior_losses=prior_resolved_buys - prior_wins,
        cumulative_buy_price_sum=cumulative_buy_price_sum,
        cumulative_buy_count=cumulative_buy_count,
        realized_pnl_usd=0.0,
        last_trade_ts=None,
        recent_30d_trades=(),
        bet_size_sum=bet_size_sum,
        bet_size_count=bet_size_count,
        category_counts={},
    )


def _history(wallet: WalletState) -> HistoryProvider:
    return _StubHistory(  # type: ignore[return-value]
        wallet=wallet,
        market=empty_market_state(market_age_start_ts=0),
        meta=_meta(),
    )


# ---------------------------------------------------------------------------
# edge_confidence_weighted
# ---------------------------------------------------------------------------


def test_edge_confidence_weighted_full_credit_at_high_n() -> None:
    """n=50 (>=20): edge_conf = realized_edge_pp * 1.0."""
    wallet = _wallet_with_history(prior_resolved_buys=50, prior_wins=52)
    # 52/50 is impossible but let's use 50 wins out of 50 for clean math
    wallet = _wallet_with_history(prior_resolved_buys=50, prior_wins=50)
    # win_rate = 50/50 = 1.0, avg_prob = 0.5, edge = 0.5
    features = compute_features(_trade(), _history(wallet))
    assert features.edge_confidence_weighted == pytest.approx(0.5)


def test_edge_confidence_weighted_partial_credit_at_n10() -> None:
    """n=10: edge_conf = realized_edge_pp * (10/20) = edge * 0.5."""
    # prior_wins=6 out of 10 → win_rate=0.6, avg_prob=0.5, edge=0.1
    wallet = _wallet_with_history(prior_resolved_buys=10, prior_wins=6)
    features = compute_features(_trade(), _history(wallet))
    assert features.realized_edge_pp == pytest.approx(0.1)
    assert features.edge_confidence_weighted == pytest.approx(0.1 * 10 / 20)


def test_edge_confidence_weighted_zero_at_n0() -> None:
    """n=0: no edge info, edge_conf = 0.0."""
    wallet = empty_wallet_state(first_seen_ts=0)
    features = compute_features(_trade(), _history(wallet))
    assert features.realized_edge_pp is None
    assert features.edge_confidence_weighted == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# win_rate_confidence_weighted
# ---------------------------------------------------------------------------


def test_win_rate_confidence_weighted_full_credit_at_high_n() -> None:
    """n=50, win_rate=0.6 → (0.6 - 0.5) * 1.0 = 0.1."""
    # 30 wins out of 50 → win_rate=0.6
    wallet = _wallet_with_history(prior_resolved_buys=50, prior_wins=30)
    features = compute_features(_trade(), _history(wallet))
    assert features.win_rate_confidence_weighted == pytest.approx(0.1)


def test_win_rate_confidence_weighted_partial_credit_at_n10() -> None:
    """n=10, win_rate=0.6 → (0.6 - 0.5) * (10/20) = 0.05."""
    wallet = _wallet_with_history(prior_resolved_buys=10, prior_wins=6)
    features = compute_features(_trade(), _history(wallet))
    assert features.win_rate_confidence_weighted == pytest.approx(0.1 * 10 / 20)


def test_win_rate_confidence_weighted_zero_at_n0() -> None:
    """n=0: no win_rate, wr_conf = 0.0."""
    wallet = empty_wallet_state(first_seen_ts=0)
    features = compute_features(_trade(), _history(wallet))
    assert features.win_rate is None
    assert features.win_rate_confidence_weighted == pytest.approx(0.0)


def test_win_rate_confidence_weighted_negative_for_below_chance() -> None:
    """win_rate=0.3 below-chance → negative centered value."""
    # 15 wins out of 50 → win_rate=0.3
    wallet = _wallet_with_history(prior_resolved_buys=50, prior_wins=15)
    features = compute_features(_trade(), _history(wallet))
    assert features.win_rate == pytest.approx(0.3)
    assert features.win_rate_confidence_weighted == pytest.approx((0.3 - 0.5) * 1.0)


# ---------------------------------------------------------------------------
# is_high_quality_wallet
# ---------------------------------------------------------------------------


def test_is_high_quality_wallet_true_when_n_and_win_rate_pass() -> None:
    """n=25 (>=20), win_rate=0.6 (>0.55) → 1."""
    # 15 wins out of 25 → win_rate=0.6
    wallet = _wallet_with_history(prior_resolved_buys=25, prior_wins=15)
    features = compute_features(_trade(), _history(wallet))
    assert features.is_high_quality_wallet == 1


def test_is_high_quality_wallet_false_when_win_rate_too_low() -> None:
    """n=25 but win_rate=0.5 (not >0.55) → 0."""
    # 12 or 13 wins out of 25: 12/25=0.48, 13/25=0.52 — both ≤0.55
    wallet = _wallet_with_history(prior_resolved_buys=25, prior_wins=13)
    features = compute_features(_trade(), _history(wallet))
    assert features.win_rate is not None
    assert features.win_rate <= 0.55
    assert features.is_high_quality_wallet == 0


def test_is_high_quality_wallet_false_when_n_too_low() -> None:
    """n=15 (<20), win_rate=0.65 → 0 (fails sample-size gate)."""
    # 10 wins out of 15 → win_rate ≈ 0.667
    wallet = _wallet_with_history(prior_resolved_buys=15, prior_wins=10)
    features = compute_features(_trade(), _history(wallet))
    assert features.is_high_quality_wallet == 0


def test_is_high_quality_wallet_false_at_boundary_n20_win_rate_exactly_0_55() -> None:
    """n=20, win_rate exactly 0.55 → 0 (strict >0.55 required)."""
    # 11 wins out of 20 → win_rate=0.55
    wallet = _wallet_with_history(prior_resolved_buys=20, prior_wins=11)
    features = compute_features(_trade(), _history(wallet))
    assert features.win_rate == pytest.approx(0.55)
    assert features.is_high_quality_wallet == 0


# ---------------------------------------------------------------------------
# bet_size_relative_to_history
# ---------------------------------------------------------------------------


def test_bet_size_relative_to_history_default_when_no_prior_bets() -> None:
    """No prior bets → median=None → default 1.0."""
    wallet = empty_wallet_state(first_seen_ts=0)
    features = compute_features(_trade(notional_usd=100.0), _history(wallet))
    assert features.bet_size_relative_to_history == pytest.approx(1.0)


def test_bet_size_relative_to_history_default_note() -> None:
    """``median_bet_size_usd`` is None in v1 → always defaults to 1.0."""
    # Even with prior bet history, the v1 streaming provider doesn't maintain
    # a rolling median. The feature defaults to 1.0 as documented.
    wallet = _wallet_with_history(
        prior_resolved_buys=5,
        prior_wins=3,
        bet_size_sum=500.0,
        bet_size_count=5,
    )
    features = compute_features(_trade(notional_usd=200.0), _history(wallet))
    assert features.median_bet_size_usd is None
    assert features.bet_size_relative_to_history == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: all 4 columns land in training_examples via build_features
# ---------------------------------------------------------------------------


def _seed_market_metadata(conn: sqlite3.Connection, condition_id: str) -> None:
    conn.execute(
        """
        INSERT INTO corpus_markets (condition_id, event_slug, category, closed_at,
                                    total_volume_usd, backfill_state, enumerated_at)
        VALUES (?, ?, ?, ?, ?, 'complete', ?)
        """,
        (condition_id, "evt", "crypto", 10_000, 50_000.0, 0),
    )
    conn.commit()


def test_build_features_new_columns_present_and_typed(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Integration: all 4 new columns are written to training_examples."""
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)

    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xabc",
                asset_id="asset1",
                wallet_address="0xw",
                condition_id="cond1",
                outcome_side="YES",
                bs="BUY",
                price=0.4,
                size=100.0,
                notional_usd=40.0,
                ts=1_000,
            )
        ]
    )
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=5_000,
            source="gamma",
        ),
        recorded_at=5_001,
    )

    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=10_000,
    )
    assert written == 1

    row = tmp_corpus_db.execute(
        """
        SELECT edge_confidence_weighted, win_rate_confidence_weighted,
               is_high_quality_wallet, bet_size_relative_to_history
        FROM training_examples
        """
    ).fetchone()
    assert row is not None
    # First-time buyer: no prior history, so all confidence features default
    assert row["edge_confidence_weighted"] == pytest.approx(0.0)
    assert row["win_rate_confidence_weighted"] == pytest.approx(0.0)
    assert row["is_high_quality_wallet"] == 0
    assert row["bet_size_relative_to_history"] == pytest.approx(1.0)
