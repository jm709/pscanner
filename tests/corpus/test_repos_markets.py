"""Tests for ``CorpusMarketsRepo`` and ``CorpusStateRepo``."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusStateRepo,
)


def _insert_market(repo: CorpusMarketsRepo, condition_id: str, **kwargs: object) -> None:
    base = CorpusMarket(
        condition_id=condition_id,
        event_slug=kwargs.get("event_slug", "evt"),  # type: ignore[arg-type]
        category=kwargs.get("category", "crypto"),  # type: ignore[arg-type]
        closed_at=int(kwargs.get("closed_at", 1_000)),  # type: ignore[arg-type]
        total_volume_usd=float(kwargs.get("total_volume_usd", 50_000.0)),  # type: ignore[arg-type]
        enumerated_at=int(kwargs.get("enumerated_at", 500)),  # type: ignore[arg-type]
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
