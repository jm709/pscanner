"""Tests for the build-features orchestrator."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

from pscanner.corpus import examples as examples_module
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


def test_build_features_does_not_query_resolutions_repo_per_row(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """The hot loop must read resolutions from the in-memory provider map.

    Pre-fix, every row in ``corpus_trades`` triggered a SELECT-by-PK on
    ``market_resolutions``. Post-fix, only ``_register_resolutions`` reads
    from the repo (once at startup), and the per-row check goes through
    ``StreamingHistoryProvider.get_resolution`` against an in-memory dict.
    """
    _seed_market_metadata(tmp_corpus_db, "cond1")
    _seed_market_metadata(tmp_corpus_db, "cond2")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)

    trades.insert_batch(
        [
            _trade(tx_hash="0xa", condition_id="cond1", ts=1_000),
            _trade(tx_hash="0xb", condition_id="cond1", bs="SELL", ts=1_500),
            _trade(tx_hash="0xc", condition_id="cond2", ts=2_000),
            _trade(tx_hash="0xd", condition_id="cond2", ts=2_500),
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

    with patch.object(
        MarketResolutionsRepo, "get", autospec=True, side_effect=MarketResolutionsRepo.get
    ) as get_spy:
        build_features(
            trades_repo=trades,
            resolutions_repo=resolutions,
            examples_repo=examples,
            markets_conn=tmp_corpus_db,
            now_ts=10_000,
        )
    assert get_spy.call_count == 0, (
        f"MarketResolutionsRepo.get must not be called per-row "
        f"during build_features (got {get_spy.call_count} calls). "
        "Resolutions are seeded once via _register_resolutions."
    )


def test_build_features_skips_maybe_make_example_for_sell_rows(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """SELL rows must short-circuit before _maybe_make_example.

    The function is a no-op for SELLs, so calling it is wasted work
    plus a wasted Trade dataclass allocation. The loop now routes SELLs
    straight to ``provider.observe_sell`` and never invokes the
    BUY-only example pipeline.
    """
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch(
        [
            _trade(tx_hash="0xa", bs="BUY", ts=1_000),
            _trade(tx_hash="0xb", bs="SELL", ts=2_000),
            _trade(tx_hash="0xc", bs="SELL", ts=3_000),
            _trade(tx_hash="0xd", bs="SELL", ts=4_000),
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

    with patch(
        "pscanner.corpus.examples._maybe_make_example",
        wraps=examples_module._maybe_make_example,
    ) as spy:
        build_features(
            trades_repo=trades,
            resolutions_repo=resolutions,
            examples_repo=examples,
            markets_conn=tmp_corpus_db,
            now_ts=10_000,
        )

    assert spy.call_count == 1, (
        f"Expected _maybe_make_example called only once (for the BUY); "
        f"got {spy.call_count}. SELLs must short-circuit."
    )


def test_build_features_writes_same_examples_with_mixed_buy_sell(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """End-to-end no-regression: BUY+SELL mix produces the expected example set.

    Post-fix the SELL rows go through ``observe_sell`` (no Trade rebuild,
    no resolution lookup), but the materialised ``training_examples``
    rows must be unchanged: same row count, same labels, same
    prior_trades_count progression.
    """
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)

    # Wallet 0xw: BUY -> SELL -> BUY -> SELL on the same market.
    # Each BUY whose market has resolved should yield one training row.
    trades.insert_batch(
        [
            _trade(tx_hash="0xa", wallet_address="0xw", bs="BUY", ts=1_000),
            _trade(tx_hash="0xb", wallet_address="0xw", bs="SELL", ts=1_500),
            _trade(tx_hash="0xc", wallet_address="0xw", bs="BUY", ts=2_000),
            _trade(tx_hash="0xd", wallet_address="0xw", bs="SELL", ts=2_500),
        ]
    )
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=8_000,
            source="gamma",
        ),
        recorded_at=8_001,
    )

    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=10_000,
    )
    assert written == 2

    rows = tmp_corpus_db.execute(
        """
        SELECT tx_hash, label_won, prior_trades_count, prior_buys_count
        FROM training_examples
        ORDER BY trade_ts
        """
    ).fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xa", "0xc"]
    # Both YES BUYs on a YES-winning market -> label_won == 1.
    assert [r["label_won"] for r in rows] == [1, 1]
    # The first BUY sees an empty wallet history; the second sees one
    # prior BUY + one prior SELL = 2 prior trades, 1 prior buy.
    assert [r["prior_trades_count"] for r in rows] == [0, 2]
    assert [r["prior_buys_count"] for r in rows] == [0, 1]


def test_build_features_threads_platform_to_training_examples(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """build_features writes training_examples rows tagged with the requested platform."""
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
        platform="polymarket",
    )
    assert written == 1
    rows = tmp_corpus_db.execute("SELECT platform FROM training_examples").fetchall()
    assert [r["platform"] for r in rows] == ["polymarket"]


def test_build_features_platform_filter_excludes_other_platform_data(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Calling build_features with a platform that has no seeded rows produces 0 examples.

    Seeds polymarket trades + market + resolution, then runs build_features
    with platform='kalshi'. The platform parameter must scope the trade
    iteration (and any market/resolution lookups) so no rows are emitted.
    """
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
        platform="kalshi",
    )
    assert written == 0
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
    assert count == 0


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
