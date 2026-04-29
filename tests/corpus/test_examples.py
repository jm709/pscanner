"""Tests for the build-features orchestrator."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)


def _seed_market_metadata(conn: sqlite3.Connection, condition_id: str, **kwargs: object) -> None:
    """Insert a corpus_markets row so build-features has metadata to read."""
    conn.execute(
        """
        INSERT INTO corpus_markets (condition_id, event_slug, category, closed_at,
                                    total_volume_usd, backfill_state, enumerated_at)
        VALUES (?, ?, ?, ?, ?, 'complete', ?)
        """,
        (
            condition_id,
            kwargs.get("event_slug", "evt"),
            kwargs.get("category", "crypto"),
            kwargs.get("closed_at", 10_000),
            kwargs.get("total_volume_usd", 50_000.0),
            kwargs.get("enumerated_at", 0),
        ),
    )
    conn.commit()


def _trade(**kwargs: object) -> CorpusTrade:
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
    }
    base.update(kwargs)
    return CorpusTrade(**base)  # type: ignore[arg-type]


def test_build_features_skips_when_no_resolution(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade()])

    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=2_000,
    )
    assert written == 0
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
    assert count == 0


def test_build_features_writes_row_for_resolved_buy(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(notional_usd=40.0, price=0.4, size=100.0)])
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
        "SELECT label_won, prior_buys_count FROM training_examples"
    ).fetchone()
    assert row["label_won"] == 1
    assert row["prior_buys_count"] == 0


def test_build_features_label_zero_for_losing_buy(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(outcome_side="YES", price=0.4, size=100.0, notional_usd=40.0)])
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=5_000,
            source="gamma",
        ),
        recorded_at=5_001,
    )

    build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=10_000,
    )
    row = tmp_corpus_db.execute("SELECT label_won FROM training_examples").fetchone()
    assert row["label_won"] == 0


def test_build_features_skips_sells(tmp_corpus_db: sqlite3.Connection) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(bs="SELL")])
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
    assert written == 0


def test_build_features_is_incremental(tmp_corpus_db: sqlite3.Connection) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(tx_hash="0xa")])
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
    build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=10_000,
    )
    trades.insert_batch([_trade(tx_hash="0xb", ts=2_000)])
    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=11_000,
    )
    assert written == 1
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
    assert count == 2
