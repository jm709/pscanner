"""Unit tests for the outcome_side backfill (#167)."""

from __future__ import annotations

import sqlite3
import time

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.outcome_side_backfill import find_buggy_markets


@pytest.fixture
def conn(tmp_path):  # type: ignore[no-untyped-def]
    """Temporary corpus DB with schema initialized."""
    db_path = tmp_path / "corpus.sqlite3"
    db = init_corpus_db(db_path)
    yield db
    db.close()


def _seed_asset(
    conn: sqlite3.Connection,
    *,
    condition_id: str,
    asset_id: str,
    outcome_side: str,
    outcome_index: int,
) -> None:
    """Insert an asset_index row."""
    conn.execute(
        "INSERT INTO asset_index (platform, asset_id, condition_id, outcome_side, outcome_index) "
        "VALUES ('polymarket', ?, ?, ?, ?)",
        (asset_id, condition_id, outcome_side, outcome_index),
    )
    conn.commit()


def _seed_corpus_market(conn: sqlite3.Connection, condition_id: str) -> None:
    """Insert a corpus_markets row."""
    conn.execute(
        "INSERT INTO corpus_markets (platform, condition_id, event_slug, market_slug, "
        " category, closed_at, total_volume_usd, backfill_state, enumerated_at) "
        "VALUES ('polymarket', ?, 'evt', ?, 'sports', ?, 0, 'complete', ?)",
        (condition_id, f"slug-{condition_id}", int(time.time()), int(time.time())),
    )
    conn.commit()


def test_find_buggy_markets_returns_only_no_no_pairs(
    conn: sqlite3.Connection,
) -> None:
    """Buggy markets with NO+NO pair are discovered."""
    # Both legs NO — this is the buggy case
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t2", outcome_side="NO", outcome_index=1)

    # YES + NO pair — this is correct, should be skipped
    _seed_corpus_market(conn, "correct1")
    _seed_asset(conn, condition_id="correct1", asset_id="t3", outcome_side="YES", outcome_index=0)
    _seed_asset(conn, condition_id="correct1", asset_id="t4", outcome_side="NO", outcome_index=1)

    # Single asset only — should be skipped
    _seed_corpus_market(conn, "single1")
    _seed_asset(conn, condition_id="single1", asset_id="t5", outcome_side="NO", outcome_index=1)

    buggy = find_buggy_markets(conn)
    assert buggy == ["buggy1"]


def test_find_buggy_markets_excludes_already_backfilled(
    conn: sqlite3.Connection,
) -> None:
    """Markets with outcome_side_backfilled_at set are excluded."""
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t2", outcome_side="NO", outcome_index=1)
    conn.execute(
        "UPDATE corpus_markets SET outcome_side_backfilled_at = ? WHERE condition_id = ?",
        (1_700_000_000, "buggy1"),
    )
    conn.commit()

    assert find_buggy_markets(conn) == []


def test_find_buggy_markets_includes_markets_with_no_corpus_markets_row(
    conn: sqlite3.Connection,
) -> None:
    """Markets with no corpus_markets row still surface if NO+NO."""
    # Some asset_index entries exist without a matching corpus_markets row
    # (e.g. populated via the live token_resolver). They should still surface
    # if they're NO+NO so the operator gets full coverage.
    _seed_asset(conn, condition_id="orphan1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="orphan1", asset_id="t2", outcome_side="NO", outcome_index=1)
    assert find_buggy_markets(conn) == ["orphan1"]
