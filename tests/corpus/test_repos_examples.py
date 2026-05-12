"""Tests for ``TrainingExamplesRepo``."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import TrainingExample, TrainingExamplesRepo


def _example(**kwargs: object) -> TrainingExample:
    base = {
        "tx_hash": "0xtx",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "trade_ts": 1_000,
        "built_at": 2_000,
        "prior_trades_count": 0,
        "prior_buys_count": 0,
        "prior_resolved_buys": 0,
        "prior_wins": 0,
        "prior_losses": 0,
        "win_rate": None,
        "avg_implied_prob_paid": None,
        "realized_edge_pp": None,
        "prior_realized_pnl_usd": 0.0,
        "avg_bet_size_usd": None,
        "median_bet_size_usd": None,
        "wallet_age_days": 0.0,
        "seconds_since_last_trade": None,
        "prior_trades_30d": 0,
        "top_category": None,
        "category_diversity": 0,
        "bet_size_usd": 50.0,
        "bet_size_rel_to_avg": None,
        "edge_confidence_weighted": 0.0,
        "win_rate_confidence_weighted": 0.0,
        "is_high_quality_wallet": 0,
        "bet_size_relative_to_history": 1.0,
        "side": "YES",
        "implied_prob_at_buy": 0.5,
        "market_category": "crypto",
        "market_volume_so_far_usd": 0.0,
        "market_unique_traders_so_far": 0,
        "market_age_seconds": 100,
        "time_to_resolution_seconds": 86_400,
        "last_trade_price": None,
        "price_volatility_recent": None,
        "label_won": 1,
    }
    base.update(kwargs)
    return TrainingExample(**base)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]


def test_insert_or_ignore_persists_row(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    inserted = repo.insert_or_ignore([_example(tx_hash="0xa")])
    assert inserted == 1
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
    assert count == 1


def test_insert_or_ignore_skips_duplicates(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([_example(tx_hash="0xa")])
    inserted = repo.insert_or_ignore([_example(tx_hash="0xa"), _example(tx_hash="0xb")])
    assert inserted == 1


def test_truncate_deletes_all_rows(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([_example(tx_hash="0xa"), _example(tx_hash="0xb")])
    repo.truncate()
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
    assert count == 0


def test_existing_keys_returns_set_of_seen_unique_keys(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([_example(tx_hash="0xa", asset_id="A1", wallet_address="0xw")])
    keys = repo.existing_keys()
    assert ("0xa", "A1", "0xw") in keys


def test_examples_repo_isolates_platforms(tmp_corpus_db: sqlite3.Connection) -> None:
    base_kwargs: dict[str, object] = {
        "condition_id": "cond1",
        "trade_ts": 1000,
        "built_at": 1500,
        "prior_trades_count": 10,
        "prior_buys_count": 8,
        "prior_resolved_buys": 5,
        "prior_wins": 3,
        "prior_losses": 2,
        "win_rate": 0.6,
        "avg_implied_prob_paid": 0.5,
        "realized_edge_pp": 0.05,
        "prior_realized_pnl_usd": 10.0,
        "avg_bet_size_usd": 50.0,
        "median_bet_size_usd": 40.0,
        "wallet_age_days": 30.0,
        "seconds_since_last_trade": 3600,
        "prior_trades_30d": 5,
        "top_category": "sports",
        "category_diversity": 2,
        "bet_size_usd": 100.0,
        "bet_size_rel_to_avg": 2.0,
        "edge_confidence_weighted": 0.025,
        "win_rate_confidence_weighted": 0.05,
        "is_high_quality_wallet": 0,
        "bet_size_relative_to_history": 2.5,
        "side": "YES",
        "implied_prob_at_buy": 0.5,
        "market_category": "sports",
        "market_volume_so_far_usd": 1_000_000.0,
        "market_unique_traders_so_far": 500,
        "market_age_seconds": 86_400,
        "time_to_resolution_seconds": 86_400,
        "last_trade_price": 0.51,
        "price_volatility_recent": 0.02,
        "label_won": 1,
    }
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore(
        [
            TrainingExample(
                tx_hash="0xtx",
                asset_id="a1",
                wallet_address="0xw",
                **base_kwargs,  # type: ignore[arg-type]
                platform="polymarket",
            ),
            TrainingExample(
                tx_hash="kx-1",
                asset_id="KX-Y",
                wallet_address="anon",
                **base_kwargs,  # type: ignore[arg-type]
                platform="kalshi",
            ),
        ]
    )
    poly_keys = repo.existing_keys(platform="polymarket")
    kalshi_keys = repo.existing_keys(platform="kalshi")
    assert ("0xtx", "a1", "0xw") in poly_keys
    assert ("0xtx", "a1", "0xw") not in kalshi_keys
    assert ("kx-1", "KX-Y", "anon") in kalshi_keys

    repo.truncate(platform="kalshi")
    assert repo.existing_keys(platform="kalshi") == set()
    assert repo.existing_keys(platform="polymarket") == {("0xtx", "a1", "0xw")}


def test_insert_or_ignore_round_trips_cat_columns(tmp_corpus_db: sqlite3.Connection) -> None:
    """Cat indicators round-trip through INSERT and SELECT."""
    repo = TrainingExamplesRepo(tmp_corpus_db)
    ex = _example(
        condition_id="0xc1",
        market_category="macro",
        cat_sports=0,
        cat_esports=0,
        cat_thesis=0,
        cat_macro=1,
        cat_elections=1,
        cat_crypto=0,
        cat_geopolitics=0,
        cat_tech=0,
        cat_culture=0,
    )
    repo.insert_or_ignore([ex])
    row = tmp_corpus_db.execute(
        "SELECT cat_macro, cat_elections, cat_sports FROM training_examples "
        "WHERE condition_id = '0xc1'"
    ).fetchone()
    assert row["cat_macro"] == 1
    assert row["cat_elections"] == 1
    assert row["cat_sports"] == 0
