"""Tests for ``CorpusMarketsRepo`` and ``CorpusStateRepo``."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusStateRepo,
    CorpusTrade,
    CorpusTradesRepo,
)


def _insert_market(repo: CorpusMarketsRepo, condition_id: str, **kwargs: object) -> None:
    base = CorpusMarket(
        condition_id=condition_id,
        event_slug=kwargs.get("event_slug", "evt"),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        category=kwargs.get("category", "crypto"),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        closed_at=int(kwargs.get("closed_at", 1_000)),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        total_volume_usd=float(kwargs.get("total_volume_usd", 50_000.0)),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        enumerated_at=int(kwargs.get("enumerated_at", 500)),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_slug=str(kwargs.get("market_slug", "slug-" + condition_id)),  # type: ignore[arg-type]
    )
    repo.insert_pending(base)


def test_insert_pending_persists_market(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    row = tmp_corpus_db.execute(
        "SELECT * FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row is not None
    assert row["backfill_state"] == "pending"
    assert row["trades_pulled_count"] == 0
    assert row["truncated_at_offset_cap"] == 0


def test_insert_pending_is_idempotent(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1", total_volume_usd=10_000.0)
    _insert_market(repo, "cond1", total_volume_usd=99_999.0)
    row = tmp_corpus_db.execute(
        "SELECT total_volume_usd FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["total_volume_usd"] == pytest.approx(10_000.0)


def test_pending_largest_first(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "small", total_volume_usd=12_000.0, closed_at=1_000)
    _insert_market(repo, "huge", total_volume_usd=200_000.0, closed_at=900)
    _insert_market(repo, "mid", total_volume_usd=50_000.0, closed_at=950)
    queue = repo.next_pending(limit=10)
    assert [m.condition_id for m in queue] == ["huge", "mid", "small"]


def test_mark_in_progress_updates_state(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_in_progress("cond1", started_at=1_500)
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, backfill_started_at FROM corpus_markets "
        "WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["backfill_state"] == "in_progress"
    assert row["backfill_started_at"] == 1_500


def test_record_progress_updates_offset_and_count(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.record_progress("cond1", last_offset=500, inserted_delta=400)
    repo.record_progress("cond1", last_offset=1000, inserted_delta=350)
    row = tmp_corpus_db.execute(
        "SELECT last_offset_seen, trades_pulled_count "
        "FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["last_offset_seen"] == 1000
    assert row["trades_pulled_count"] == 750


def test_mark_complete_sets_state_and_timestamp(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_complete("cond1", completed_at=2_000, truncated=False)
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, backfill_completed_at, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["backfill_state"] == "complete"
    assert row["backfill_completed_at"] == 2_000
    assert row["truncated_at_offset_cap"] == 0


def test_mark_complete_with_truncation_sets_flag(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_complete("cond1", completed_at=2_000, truncated=True)
    row = tmp_corpus_db.execute(
        "SELECT truncated_at_offset_cap FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["truncated_at_offset_cap"] == 1


def _insert_trade(trades_repo: CorpusTradesRepo, condition_id: str, ts: int) -> None:
    """Insert a minimal corpus_trades row at the given timestamp."""
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash=f"0x{condition_id}-{ts}",
                asset_id="asset1",
                wallet_address="0xw",
                condition_id=condition_id,
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=100.0,
                notional_usd=50.0,
                ts=ts,
            )
        ]
    )


def test_mark_complete_rewrites_closed_at_to_max_trade_ts(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """`mark_complete` should overwrite the enumerator placeholder with MAX(trade_ts)."""
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _insert_market(markets, "cond1", closed_at=10_000)  # placeholder from enumerator
    _insert_trade(trades, "cond1", ts=5_555)
    _insert_trade(trades, "cond1", ts=7_777)  # latest
    _insert_trade(trades, "cond1", ts=6_000)
    markets.mark_complete("cond1", completed_at=20_000, truncated=False)
    row = tmp_corpus_db.execute(
        "SELECT closed_at FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["closed_at"] == 7_777


def test_mark_complete_preserves_closed_at_when_no_trades(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Without observed trades, `mark_complete` keeps the placeholder rather than NULL-ing it."""
    markets = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(markets, "cond1", closed_at=10_000)
    markets.mark_complete("cond1", completed_at=20_000, truncated=False)
    row = tmp_corpus_db.execute(
        "SELECT closed_at FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["closed_at"] == 10_000


def test_mark_complete_is_idempotent_on_closed_at(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Re-running `mark_complete` after the same set of trades should not drift the value."""
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _insert_market(markets, "cond1", closed_at=10_000)
    _insert_trade(trades, "cond1", ts=8_000)
    markets.mark_complete("cond1", completed_at=20_000, truncated=False)
    markets.mark_complete("cond1", completed_at=21_000, truncated=False)
    row = tmp_corpus_db.execute(
        "SELECT closed_at FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["closed_at"] == 8_000


def test_mark_failed_records_error(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_failed("cond1", error_message="HTTP 500 after 3 retries")
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, error_message FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["backfill_state"] == "failed"
    assert row["error_message"] == "HTTP 500 after 3 retries"


def test_resume_in_progress_returned_in_pending_queue(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_in_progress("cond1", started_at=1_000)
    queue = repo.next_pending(limit=10)
    assert [m.condition_id for m in queue] == ["cond1"]


def test_complete_markets_excluded_from_queue(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "done", total_volume_usd=99_999.0)
    repo.mark_complete("done", completed_at=2_000, truncated=False)
    _insert_market(repo, "todo", total_volume_usd=20_000.0)
    queue = repo.next_pending(limit=10)
    assert [m.condition_id for m in queue] == ["todo"]


def test_state_repo_get_set_roundtrip(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set("last_gamma_sweep_ts", "1700000000", updated_at=1_700_000_001)
    assert repo.get("last_gamma_sweep_ts") == "1700000000"
    repo.set("last_gamma_sweep_ts", "1700001000", updated_at=1_700_001_001)
    assert repo.get("last_gamma_sweep_ts") == "1700001000"


def test_state_repo_get_missing_returns_none(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    assert repo.get("never_set") is None


def test_state_repo_get_int(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set("last_gamma_sweep_ts", "1700000000", updated_at=1_700_000_001)
    assert repo.get_int("last_gamma_sweep_ts") == 1_700_000_000
    assert repo.get_int("missing") is None


def test_insert_pending_backfills_market_slug_on_existing_row(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """A re-enumeration after the schema migration should populate
    market_slug on rows that pre-date it (where market_slug was NULL).
    """
    repo = CorpusMarketsRepo(tmp_corpus_db)
    # Simulate a row inserted before the migration: market_slug NULL.
    tmp_corpus_db.execute(
        """
        INSERT INTO corpus_markets (
          condition_id, event_slug, category, closed_at, total_volume_usd,
          backfill_state, enumerated_at
        ) VALUES ('cond1', 'evt', 'crypto', 1000, 50000.0, 'complete', 500)
        """
    )
    tmp_corpus_db.commit()
    # Now re-enumerate the same market with a slug.
    inserted = repo.insert_pending(
        CorpusMarket(
            condition_id="cond1",
            event_slug="evt",
            category="crypto",
            closed_at=1000,
            total_volume_usd=50000.0,
            market_slug="cond1-slug",
            enumerated_at=500,
        )
    )
    assert inserted == 0  # row was already present, INSERT OR IGNORE no-op
    # But market_slug should now be populated on the existing row.
    row = tmp_corpus_db.execute(
        "SELECT market_slug, backfill_state FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["market_slug"] == "cond1-slug"
    assert row["backfill_state"] == "complete"  # state preserved


def test_repo_isolates_polymarket_and_kalshi(tmp_corpus_db: sqlite3.Connection) -> None:
    """Inserting markets with different platforms keeps them isolated by `platform` arg."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    repo.insert_pending(
        CorpusMarket(
            condition_id="0xpoly",
            event_slug="poly-event",
            category=None,
            closed_at=1000,
            total_volume_usd=1_000_000.0,
            enumerated_at=900,
            market_slug="poly-slug",
            platform="polymarket",
        )
    )
    repo.insert_pending(
        CorpusMarket(
            condition_id="KX-1",
            event_slug="kx-event",
            category=None,
            closed_at=1100,
            total_volume_usd=2_000_000.0,
            enumerated_at=950,
            market_slug="kx-slug",
            platform="kalshi",
        )
    )
    poly = repo.next_pending(limit=10, platform="polymarket")
    kalshi = repo.next_pending(limit=10, platform="kalshi")
    assert [m.condition_id for m in poly] == ["0xpoly"]
    assert [m.condition_id for m in kalshi] == ["KX-1"]
    assert all(m.platform == "polymarket" for m in poly)
    assert all(m.platform == "kalshi" for m in kalshi)


def test_insert_pending_backfill_does_not_overwrite_existing_slug(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """If market_slug is already set, re-enumeration must not stomp it."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    repo.insert_pending(
        CorpusMarket(
            condition_id="cond1",
            event_slug="evt",
            category="crypto",
            closed_at=1000,
            total_volume_usd=50000.0,
            market_slug="original-slug",
            enumerated_at=500,
        )
    )
    repo.insert_pending(
        CorpusMarket(
            condition_id="cond1",
            event_slug="evt",
            category="crypto",
            closed_at=1000,
            total_volume_usd=50000.0,
            market_slug="different-slug",
            enumerated_at=500,
        )
    )
    row = tmp_corpus_db.execute(
        "SELECT market_slug FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["market_slug"] == "original-slug"


def test_corpus_market_default_tags_json_is_empty_list() -> None:
    market = CorpusMarket(
        condition_id="0xc1",
        event_slug="test",
        category="thesis",
        closed_at=0,
        total_volume_usd=0.0,
        enumerated_at=0,
        market_slug="",
    )
    assert market.tags_json == "[]"
    assert market.categories_json == "[]"


def test_corpus_market_accepts_explicit_tags_json() -> None:
    market = CorpusMarket(
        condition_id="0xc1",
        event_slug="test",
        category="thesis",
        closed_at=0,
        total_volume_usd=0.0,
        enumerated_at=0,
        market_slug="",
        tags_json='["Sports", "NBA"]',
        categories_json='["sports"]',
    )
    assert market.tags_json == '["Sports", "NBA"]'
    assert market.categories_json == '["sports"]'


def test_insert_pending_writes_tags_and_categories_json(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    market = CorpusMarket(
        condition_id="0xc1",
        event_slug="nba-final",
        category="sports",
        closed_at=0,
        total_volume_usd=1.0,
        enumerated_at=0,
        market_slug="nba-final-okc",
        tags_json='["Sports", "NBA"]',
        categories_json='["sports"]',
    )
    repo.insert_pending(market)
    row = tmp_corpus_db.execute(
        "SELECT tags_json, categories_json FROM corpus_markets WHERE condition_id = '0xc1'"
    ).fetchone()
    assert row["tags_json"] == '["Sports", "NBA"]'
    assert row["categories_json"] == '["sports"]'


def test_insert_pending_defaults_keep_existing_callers_working(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """A market constructed without tag fields lands with '[]' defaults."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    market = CorpusMarket(
        condition_id="0xc2",
        event_slug="ev",
        category="thesis",
        closed_at=0,
        total_volume_usd=1.0,
        enumerated_at=0,
        market_slug="",
    )
    repo.insert_pending(market)
    row = tmp_corpus_db.execute(
        "SELECT tags_json, categories_json FROM corpus_markets WHERE condition_id = '0xc2'"
    ).fetchone()
    assert row["tags_json"] == "[]"
    assert row["categories_json"] == "[]"


def test_iter_unbackfilled_tags_returns_only_default_rows(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Rows with non-default tags_json are skipped."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    unbackfilled = CorpusMarket(
        condition_id="0xun",
        event_slug="unbackfilled",
        category="thesis",
        closed_at=0,
        total_volume_usd=1.0,
        enumerated_at=0,
        market_slug="",
    )
    already_done = CorpusMarket(
        condition_id="0xdone",
        event_slug="done",
        category="sports",
        closed_at=0,
        total_volume_usd=1.0,
        enumerated_at=0,
        market_slug="",
        tags_json='["Sports"]',
        categories_json='["sports"]',
    )
    repo.insert_pending(unbackfilled)
    repo.insert_pending(already_done)
    slugs = [m.event_slug for m in repo.iter_unbackfilled_tags(limit=100)]
    assert slugs == ["unbackfilled"]


def test_iter_unbackfilled_tags_skips_error_sentinel(tmp_corpus_db: sqlite3.Connection) -> None:
    """Rows quarantined with tags_json='__ERROR__' are NOT returned."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    repo.insert_pending(
        CorpusMarket(
            condition_id="0xer",
            event_slug="err",
            category="thesis",
            closed_at=0,
            total_volume_usd=1.0,
            enumerated_at=0,
            market_slug="",
            tags_json="__ERROR__",
        )
    )
    rows = list(repo.iter_unbackfilled_tags(limit=100))
    assert rows == []


def test_iter_unbackfilled_tags_honors_limit(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    for i in range(5):
        repo.insert_pending(
            CorpusMarket(
                condition_id=f"0x{i}",
                event_slug=f"e{i}",
                category="thesis",
                closed_at=0,
                total_volume_usd=float(i),
                enumerated_at=0,
                market_slug="",
            )
        )
    rows = list(repo.iter_unbackfilled_tags(limit=3))
    assert len(rows) == 3
