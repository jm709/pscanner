"""Tests for ``MarketResolutionsRepo``."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import MarketResolution, MarketResolutionsRepo


def test_upsert_inserts_new(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    row = tmp_corpus_db.execute(
        "SELECT * FROM market_resolutions WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["winning_outcome_index"] == 0
    assert row["outcome_yes_won"] == 1
    assert row["resolved_at"] == 2_000
    assert row["recorded_at"] == 2_001


def test_upsert_updates_existing(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=2_500,
            source="gamma",
        ),
        recorded_at=2_501,
    )
    row = tmp_corpus_db.execute(
        "SELECT winning_outcome_index, outcome_yes_won FROM market_resolutions "
        "WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["winning_outcome_index"] == 1
    assert row["outcome_yes_won"] == 0


def test_get_returns_none_for_missing(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    assert repo.get("missing") is None


def test_get_returns_resolution(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    res = repo.get("cond1")
    assert res is not None
    assert res.outcome_yes_won == 0


def test_missing_for_returns_unresolved_condition_ids(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="resolved",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    missing = repo.missing_for(["resolved", "unresolved1", "unresolved2"])
    assert set(missing) == {"unresolved1", "unresolved2"}
