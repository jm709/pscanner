"""Tests for ``pscanner.manifold.repos.*`` CRUD operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from pscanner.manifold.db import init_manifold_tables
from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId
from pscanner.manifold.models import ManifoldBet, ManifoldMarket, ManifoldUser
from pscanner.manifold.repos import ManifoldBetsRepo, ManifoldMarketsRepo, ManifoldUsersRepo


@pytest.fixture
def mdb() -> Iterator[sqlite3.Connection]:
    """In-memory SQLite connection with Manifold schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    init_manifold_tables(conn)
    try:
        yield conn
    finally:
        conn.close()


def _market(market_id: str = "mkABC") -> ManifoldMarket:
    return ManifoldMarket.model_validate(
        {
            "id": market_id,
            "creatorId": "userXYZ",
            "question": f"Will {market_id} resolve YES?",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "prob": 0.5,
            "volume": 1000.0,
            "totalLiquidity": 100.0,
            "isResolved": False,
            "resolutionTime": None,
            "closeTime": 1_800_000_000_000,
        }
    )


def _bet(bet_id: str = "betXYZ", *, market_id: str = "mkABC", ts: int = 1_000) -> ManifoldBet:
    return ManifoldBet.model_validate(
        {
            "id": bet_id,
            "userId": "userXYZ",
            "contractId": market_id,
            "outcome": "YES",
            "amount": 10.0,
            "probBefore": 0.49,
            "probAfter": 0.50,
            "createdTime": ts,
            "isFilled": True,
            "isCancelled": False,
            "limitProb": None,
        }
    )


def _user(user_id: str = "userXYZ") -> ManifoldUser:
    return ManifoldUser.model_validate(
        {
            "id": user_id,
            "username": "alice",
            "name": "Alice Wonderland",
            "createdTime": 1_700_000_000_000,
            "balance": 500.0,
        }
    )


# ---- ManifoldMarketsRepo ----


def test_markets_insert_and_get(mdb: sqlite3.Connection) -> None:
    repo = ManifoldMarketsRepo(mdb)
    m = _market()
    repo.insert_or_replace(m)
    result = repo.get_by_id(ManifoldMarketId("mkABC"))
    assert result is not None
    assert result.id == "mkABC"
    assert result.question == "Will mkABC resolve YES?"


def test_markets_get_missing_returns_none(mdb: sqlite3.Connection) -> None:
    repo = ManifoldMarketsRepo(mdb)
    result = repo.get_by_id(ManifoldMarketId("nonexistent"))
    assert result is None


def test_markets_insert_or_replace_updates_row(mdb: sqlite3.Connection) -> None:
    repo = ManifoldMarketsRepo(mdb)
    m1 = _market()
    repo.insert_or_replace(m1)
    # Simulate an update: same ID, different prob.
    updated = ManifoldMarket.model_validate(
        {
            **m1.model_dump(by_alias=True),
            "prob": 0.75,
        }
    )
    repo.insert_or_replace(updated)
    result = repo.get_by_id(ManifoldMarketId("mkABC"))
    assert result is not None
    assert result.prob == pytest.approx(0.75)


def test_markets_iter_chronological_empty(mdb: sqlite3.Connection) -> None:
    repo = ManifoldMarketsRepo(mdb)
    assert list(repo.iter_chronological()) == []


def test_markets_iter_chronological_order(mdb: sqlite3.Connection) -> None:
    repo = ManifoldMarketsRepo(mdb)
    m1 = ManifoldMarket.model_validate(
        {
            "id": "early",
            "creatorId": "u",
            "question": "Early market",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "volume": 100.0,
            "totalLiquidity": 10.0,
            "isResolved": False,
            "closeTime": 1_000_000,
        }
    )
    m2 = ManifoldMarket.model_validate(
        {
            "id": "late",
            "creatorId": "u",
            "question": "Late market",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "volume": 200.0,
            "totalLiquidity": 20.0,
            "isResolved": False,
            "closeTime": 2_000_000,
        }
    )
    repo.insert_or_replace(m2)
    repo.insert_or_replace(m1)
    markets = list(repo.iter_chronological())
    assert [m.id for m in markets] == ["early", "late"]


def test_markets_null_close_time_last(mdb: sqlite3.Connection) -> None:
    repo = ManifoldMarketsRepo(mdb)
    m_with_time = ManifoldMarket.model_validate(
        {
            "id": "timed",
            "creatorId": "u",
            "question": "Q",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "volume": 0.0,
            "totalLiquidity": 0.0,
            "isResolved": False,
            "closeTime": 1_000,
        }
    )
    m_no_time = ManifoldMarket.model_validate(
        {
            "id": "notimed",
            "creatorId": "u",
            "question": "Q2",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "volume": 0.0,
            "totalLiquidity": 0.0,
            "isResolved": False,
        }
    )
    repo.insert_or_replace(m_no_time)
    repo.insert_or_replace(m_with_time)
    ids = [m.id for m in repo.iter_chronological()]
    assert ids.index(ManifoldMarketId("timed")) < ids.index(ManifoldMarketId("notimed"))


# ---- ManifoldBetsRepo ----


def test_bets_insert_and_get(mdb: sqlite3.Connection) -> None:
    repo = ManifoldBetsRepo(mdb)
    b = _bet()
    repo.insert_or_replace(b)
    result = repo.get_by_id("betXYZ")
    assert result is not None
    assert result.id == "betXYZ"
    assert result.amount == 10.0
    assert result.outcome == "YES"


def test_bets_get_missing_returns_none(mdb: sqlite3.Connection) -> None:
    repo = ManifoldBetsRepo(mdb)
    assert repo.get_by_id("nonexistent") is None


def test_bets_insert_or_replace_updates(mdb: sqlite3.Connection) -> None:
    repo = ManifoldBetsRepo(mdb)
    b = _bet()
    repo.insert_or_replace(b)
    updated = ManifoldBet.model_validate(
        {
            **b.model_dump(by_alias=True),
            "amount": 99.0,
        }
    )
    repo.insert_or_replace(updated)
    result = repo.get_by_id("betXYZ")
    assert result is not None
    assert result.amount == 99.0


def test_bets_iter_chronological_all(mdb: sqlite3.Connection) -> None:
    repo = ManifoldBetsRepo(mdb)
    repo.insert_or_replace(_bet("b2", ts=2000))
    repo.insert_or_replace(_bet("b1", ts=1000))
    bets = list(repo.iter_chronological())
    assert [b.id for b in bets] == ["b1", "b2"]


def test_bets_iter_chronological_market_filter(mdb: sqlite3.Connection) -> None:
    repo = ManifoldBetsRepo(mdb)
    repo.insert_or_replace(_bet("b1", market_id="mkt1", ts=1000))
    repo.insert_or_replace(_bet("b2", market_id="mkt2", ts=2000))
    bets = list(repo.iter_chronological(market_id=ManifoldMarketId("mkt1")))
    assert [b.id for b in bets] == ["b1"]


def test_bets_limit_prob_round_trips(mdb: sqlite3.Connection) -> None:
    repo = ManifoldBetsRepo(mdb)
    limit_bet = ManifoldBet.model_validate(
        {
            "id": "limit1",
            "userId": "u",
            "contractId": "m",
            "outcome": "NO",
            "amount": 5.0,
            "probBefore": 0.6,
            "probAfter": 0.59,
            "createdTime": 100,
            "isFilled": False,
            "isCancelled": False,
            "limitProb": 0.55,
        }
    )
    repo.insert_or_replace(limit_bet)
    result = repo.get_by_id("limit1")
    assert result is not None
    assert result.limit_prob == pytest.approx(0.55)
    assert result.is_filled is False


# ---- ManifoldUsersRepo ----


def test_users_insert_and_get(mdb: sqlite3.Connection) -> None:
    repo = ManifoldUsersRepo(mdb)
    u = _user()
    repo.insert_or_replace(u)
    result = repo.get_by_id(ManifoldUserId("userXYZ"))
    assert result is not None
    assert result.username == "alice"
    assert result.name == "Alice Wonderland"


def test_users_get_missing_returns_none(mdb: sqlite3.Connection) -> None:
    repo = ManifoldUsersRepo(mdb)
    assert repo.get_by_id(ManifoldUserId("nobody")) is None


def test_users_insert_or_replace_updates(mdb: sqlite3.Connection) -> None:
    repo = ManifoldUsersRepo(mdb)
    u = _user()
    repo.insert_or_replace(u)
    updated = ManifoldUser.model_validate(
        {
            **u.model_dump(by_alias=True),
            "name": "Alice Updated",
        }
    )
    repo.insert_or_replace(updated)
    result = repo.get_by_id(ManifoldUserId("userXYZ"))
    assert result is not None
    assert result.name == "Alice Updated"


def test_users_iter_chronological(mdb: sqlite3.Connection) -> None:
    repo = ManifoldUsersRepo(mdb)
    u1 = ManifoldUser.model_validate(
        {
            "id": "u1",
            "username": "a",
            "name": "A",
            "createdTime": 1000,
        }
    )
    u2 = ManifoldUser.model_validate(
        {
            "id": "u2",
            "username": "b",
            "name": "B",
            "createdTime": 2000,
        }
    )
    repo.insert_or_replace(u2)
    repo.insert_or_replace(u1)
    users = list(repo.iter_chronological())
    assert [u.id for u in users] == ["u1", "u2"]


def test_init_manifold_tables_idempotent(mdb: sqlite3.Connection) -> None:
    """Calling init_manifold_tables twice must not raise."""
    init_manifold_tables(mdb)
    init_manifold_tables(mdb)
