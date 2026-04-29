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
    return TrainingExample(**base)  # type: ignore[arg-type]


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
