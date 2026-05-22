"""Unit tests for the outcome_side backfill (#167)."""

from __future__ import annotations

import sqlite3
import time
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.outcome_side_backfill import (
    apply_market_backfill,
    find_buggy_markets,
    resolve_correct_mapping,
)
from pscanner.poly.models import Market


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


def _make_market(
    *, condition_id: str, outcomes: tuple[str, str], tokens: tuple[str, str]
) -> Market:
    """Create a test Market with the given condition_id, outcomes, and token IDs."""
    return Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": f"slug-{condition_id}",
            "conditionId": condition_id,
            "outcomes": list(outcomes),
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": list(tokens),
            "active": True,
            "closed": False,
        }
    )


def _fake_data(slug: str | None) -> AsyncMock:
    """Create a mock DataClient that returns the given slug."""
    data = AsyncMock()
    data.get_market_slug_by_condition_id = AsyncMock(return_value=slug)
    return data


def _fake_gamma(market: Market | None) -> AsyncMock:
    """Create a mock GammaClient that returns the given market."""
    gamma = AsyncMock()
    gamma.get_market_by_slug = AsyncMock(return_value=market)
    return gamma


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_yes_no_dict() -> None:
    """resolve_correct_mapping returns {token_id: (outcome_side, index)} for binary market."""
    data = _fake_data("slug-cond1")
    market = _make_market(
        condition_id="cond1",
        outcomes=("Cavaliers", "Knicks"),
        tokens=("token-cavs", "token-knicks"),
    )
    gamma = _fake_gamma(market)
    mapping = await resolve_correct_mapping("cond1", data=data, gamma=gamma)
    assert mapping == {"token-cavs": ("YES", 0), "token-knicks": ("NO", 1)}


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_missing_slug() -> None:
    """resolve_correct_mapping returns None when slug lookup returns None."""
    data = _fake_data(None)
    gamma = _fake_gamma(None)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None
    gamma.get_market_by_slug.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_missing_market() -> None:
    """resolve_correct_mapping returns None when gamma market lookup returns None."""
    data = _fake_data("slug-cond1")
    gamma = _fake_gamma(None)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_non_binary() -> None:
    """resolve_correct_mapping returns None for non-binary markets."""
    data = _fake_data("slug-cond1")
    three_outcome = Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": "slug-cond1",
            "outcomes": ["A", "B", "C"],
            "outcomePrices": ["0.33", "0.33", "0.34"],
            "clobTokenIds": ["t-a", "t-b", "t-c"],
            "active": True,
            "closed": False,
        }
    )
    gamma = _fake_gamma(three_outcome)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_data_exception() -> None:
    """resolve_correct_mapping returns None when data client raises."""
    data = AsyncMock()
    data.get_market_slug_by_condition_id = AsyncMock(side_effect=RuntimeError("boom"))
    gamma = _fake_gamma(None)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_gamma_exception() -> None:
    """resolve_correct_mapping returns None when gamma client raises."""
    data = _fake_data("slug-cond1")
    gamma = AsyncMock()
    gamma.get_market_by_slug = AsyncMock(side_effect=RuntimeError("boom"))
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


# Tests for apply_market_backfill


def _seed_trade(
    conn: sqlite3.Connection,
    *,
    condition_id: str,
    asset_id: str,
    outcome_side: str,
    tx_hash: str,
) -> None:
    """Insert a corpus_trades row."""
    conn.execute(
        "INSERT INTO corpus_trades (platform, tx_hash, asset_id, wallet_address, "
        " condition_id, outcome_side, bs, price, size, notional_usd, ts) "
        "VALUES ('polymarket', ?, ?, '0xWALLET', ?, ?, 'BUY', 0.5, 100.0, 50.0, 1)",
        (tx_hash, asset_id, condition_id, outcome_side),
    )
    conn.commit()


def test_apply_market_backfill_rewrites_asset_index_and_corpus_trades(
    conn: sqlite3.Connection,
) -> None:
    """apply_market_backfill rewrites asset_index and corpus_trades."""
    _seed_corpus_market(conn, "cond1")
    _seed_asset(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="cond1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    _seed_trade(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", tx_hash="0xA")
    _seed_trade(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", tx_hash="0xB")
    _seed_trade(conn, condition_id="cond1", asset_id="t-no", outcome_side="NO", tx_hash="0xC")

    mapping = {"t-yes": ("YES", 0), "t-no": ("NO", 1)}
    apply_market_backfill(conn, "cond1", mapping, now_ts=1_700_000_500)

    # asset_index updated to YES + NO
    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='cond1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "YES", "t-no": "NO"}

    # corpus_trades updated: YES on t-yes rows, NO on t-no rows
    yes_rows = [
        r[0]
        for r in conn.execute(
            "SELECT tx_hash FROM corpus_trades"
            " WHERE condition_id='cond1' AND outcome_side='YES' ORDER BY tx_hash"
        )
    ]
    assert yes_rows == ["0xA", "0xB"]
    no_rows = [
        r[0]
        for r in conn.execute(
            "SELECT tx_hash FROM corpus_trades"
            " WHERE condition_id='cond1' AND outcome_side='NO' ORDER BY tx_hash"
        )
    ]
    assert no_rows == ["0xC"]

    # Sentinel set
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()[0]
    assert sentinel == 1_700_000_500


def test_apply_market_backfill_idempotent(conn: sqlite3.Connection) -> None:
    """apply_market_backfill is idempotent; last write wins on sentinel."""
    _seed_corpus_market(conn, "cond1")
    _seed_asset(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="cond1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    _seed_trade(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", tx_hash="0xA")
    mapping = {"t-yes": ("YES", 0), "t-no": ("NO", 1)}

    apply_market_backfill(conn, "cond1", mapping, now_ts=1_700_000_500)
    apply_market_backfill(conn, "cond1", mapping, now_ts=1_700_000_999)

    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='cond1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "YES", "t-no": "NO"}
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()[0]
    assert sentinel == 1_700_000_999  # last write wins


def test_apply_market_backfill_creates_sentinel_row_when_missing(
    conn: sqlite3.Connection,
) -> None:
    """apply_market_backfill handles orphan asset_index with no corpus_markets row."""
    # No corpus_markets row at all (orphan asset_index entries from token_resolver)
    _seed_asset(conn, condition_id="orphan1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="orphan1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    mapping = {"t-yes": ("YES", 0), "t-no": ("NO", 1)}

    apply_market_backfill(conn, "orphan1", mapping, now_ts=1_700_000_500)

    # We DON'T fabricate a corpus_markets row — the sentinel UPDATE is a no-op
    # when no row exists. find_buggy_markets gates on asset_index leg-sides,
    # so once the asset_index is corrected the market drops out regardless.
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='orphan1'"
    ).fetchone()
    assert sentinel is None

    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='orphan1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "YES", "t-no": "NO"}
