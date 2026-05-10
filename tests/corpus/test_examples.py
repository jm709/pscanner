"""Tests for the build-features orchestrator."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

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
    return CorpusTrade(**base)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]


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


def test_build_features_rebuild_drops_and_recreates_secondary_indexes(
    tmp_corpus_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--rebuild` drops the 3 secondary indexes pre-walk and recreates them post-walk (#114)."""
    drops: list[str] = []
    creates: list[str] = []
    real_drop = TrainingExamplesRepo.drop_secondary_indexes
    real_recreate = TrainingExamplesRepo.recreate_secondary_indexes

    def _spy_drop(self: TrainingExamplesRepo) -> None:
        drops.append("drop")
        real_drop(self)

    def _spy_recreate(self: TrainingExamplesRepo) -> None:
        creates.append("recreate")
        real_recreate(self)

    monkeypatch.setattr(TrainingExamplesRepo, "drop_secondary_indexes", _spy_drop)
    monkeypatch.setattr(TrainingExamplesRepo, "recreate_secondary_indexes", _spy_recreate)

    build_features(
        trades_repo=CorpusTradesRepo(tmp_corpus_db),
        resolutions_repo=MarketResolutionsRepo(tmp_corpus_db),
        examples_repo=TrainingExamplesRepo(tmp_corpus_db),
        markets_conn=tmp_corpus_db,
        now_ts=0,
        rebuild=True,
        platform="polymarket",
    )

    assert drops == ["drop"]
    assert creates == ["recreate"]


def test_build_features_incremental_does_not_drop_indexes(
    tmp_corpus_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-rebuild build_features leaves secondary indexes alone (#114)."""
    drops: list[str] = []
    monkeypatch.setattr(
        TrainingExamplesRepo,
        "drop_secondary_indexes",
        lambda self: drops.append("drop"),
    )

    build_features(
        trades_repo=CorpusTradesRepo(tmp_corpus_db),
        resolutions_repo=MarketResolutionsRepo(tmp_corpus_db),
        examples_repo=TrainingExamplesRepo(tmp_corpus_db),
        markets_conn=tmp_corpus_db,
        now_ts=0,
        rebuild=False,
        platform="polymarket",
    )

    assert drops == []


def test_build_features_emits_progress_events(
    tmp_corpus_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_features emits structlog progress events every N batches (#114)."""
    # Tiny batches: batch_size=2, progress_every=2 → every 2nd batch fires.
    # 6 BUYs → 3 batches (batch_idx=1,2,3) → progress fires at batch_idx=2.
    monkeypatch.setattr(examples_module, "_BATCH_SIZE", 2)
    monkeypatch.setattr(examples_module, "_PROGRESS_EVERY_N_BATCHES", 2)

    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)

    # 6 BUY trades each ≥$10 notional (floor guard in insert_batch).
    trades.insert_batch(
        [_trade(tx_hash=f"0x{i}", ts=i * 1_000, notional_usd=40.0) for i in range(1, 7)]
    )
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=10_000,
            source="gamma",
        ),
        recorded_at=10_001,
    )

    with capture_logs() as logs:  # type: ignore[no-untyped-call]
        build_features(
            trades_repo=trades,
            resolutions_repo=resolutions,
            examples_repo=examples,
            markets_conn=tmp_corpus_db,
            now_ts=20_000,
            rebuild=True,
            platform="polymarket",
        )

    progress_events = [log for log in logs if log.get("event") == "corpus.build_features_progress"]
    assert len(progress_events) >= 1, f"expected >=1 progress event, got {len(progress_events)}"
    sample = progress_events[0]
    assert "rows_emitted" in sample
    assert "batch_idx" in sample
    assert "last_trade_ts" in sample
    assert "elapsed_seconds" in sample
    # Strengthened (#114): values must be meaningful, not all-zero. The
    # fixture inserts 6 BUYs at ts=1000..6000; with _BATCH_SIZE=2 and
    # _PROGRESS_EVERY_N_BATCHES=2 the first event fires at batch_idx=2
    # after rows with ts up through the 4th BUY have flushed.
    assert sample["rows_emitted"] > 0
    assert sample["batch_idx"] == 2
    assert sample["last_trade_ts"] >= 1_000
    assert sample["elapsed_seconds"] >= 0


def test_build_features_checkpoint_fires_on_threshold(
    tmp_corpus_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """checkpoint_wal fires once per N batches during the rebuild walk (#114)."""
    # Force tiny batches so the test corpus only needs ~5 resolved BUYs
    # to trigger one checkpoint: batch_size=2, checkpoint_every=2 means
    # every 2nd batch fires → need >4 rows emitted → 5 BUYs suffice.
    monkeypatch.setattr(examples_module, "_BATCH_SIZE", 2)
    monkeypatch.setattr(examples_module, "_CHECKPOINT_EVERY_N_BATCHES", 2)

    checkpoints: list[None] = []
    monkeypatch.setattr(
        TrainingExamplesRepo,
        "checkpoint_wal",
        lambda self: checkpoints.append(None),  # type: ignore[no-untyped-def]
    )

    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)

    # Insert 6 BUY trades each ≥$10 notional so none are filtered by the
    # _NOTIONAL_FLOOR_USD guard in insert_batch. 6 rows → 3 batches of 2
    # → checkpoint fires at batch_idx=2.
    trades.insert_batch(
        [_trade(tx_hash=f"0x{i}", ts=i * 1_000, notional_usd=40.0) for i in range(1, 7)]
    )
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=10_000,
            source="gamma",
        ),
        recorded_at=10_001,
    )

    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=20_000,
        rebuild=True,
        platform="polymarket",
    )

    assert written >= 1
    assert len(checkpoints) >= 1, (
        f"expected >=1 checkpoint, got {len(checkpoints)} for written={written}"
    )
