"""Tests for PaperTradesRepo."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    OpenPaperPosition,
    PaperSummary,
    PaperTradesRepo,
)

_NOW = 1700000000


def _entry(repo: PaperTradesRepo, **overrides: Any) -> int:
    """Insert an entry row with sensible defaults; return its trade_id."""
    args: dict[str, Any] = {
        "triggering_alert_key": "smart:0xw1:0xc1:yes:20260427",
        "source_wallet": "0xwallet1",
        "condition_id": ConditionId("0xcond-1"),
        "asset_id": AssetId("asset-yes"),
        "outcome": "yes",
        "shares": 20.0,
        "fill_price": 0.5,
        "cost_usd": 10.0,
        "nav_after_usd": 990.0,
        "ts": _NOW,
    }
    args.update(overrides)
    return repo.insert_entry(**args)


def test_insert_entry_returns_trade_id(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    tid = _entry(repo)
    assert tid >= 1


def test_insert_entry_unique_alert_key(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    _entry(repo, triggering_alert_key="smart:0xa:0xc:yes:1")
    with pytest.raises(sqlite3.IntegrityError):
        _entry(repo, triggering_alert_key="smart:0xa:0xc:yes:1")


def test_insert_entry_null_alert_key_allowed(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    _entry(repo, triggering_alert_key=None)
    _entry(repo, triggering_alert_key=None)


def test_insert_exit_links_parent(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    parent = _entry(repo)
    repo.insert_exit(
        parent_trade_id=parent,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,
        nav_after_usd=1010.0,
        ts=_NOW + 100,
    )
    assert repo.list_open_positions() == []


def test_list_open_positions_returns_unmatched_entries(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    p1 = _entry(repo, triggering_alert_key="a1")
    p2 = _entry(repo, triggering_alert_key="a2")
    repo.insert_exit(
        parent_trade_id=p1,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,
        nav_after_usd=1010.0,
        ts=_NOW + 100,
    )
    open_positions = repo.list_open_positions()
    assert [p.trade_id for p in open_positions] == [p2]
    assert isinstance(open_positions[0], OpenPaperPosition)


def test_compute_cost_basis_nav_empty(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 1000.0


def test_compute_cost_basis_nav_open_only(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    _entry(repo)
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 1000.0


def test_compute_cost_basis_nav_one_winning_exit(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    parent = _entry(repo, cost_usd=10.0, shares=20.0, fill_price=0.5)
    repo.insert_exit(
        parent_trade_id=parent,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,
        nav_after_usd=1010.0,
        ts=_NOW + 100,
    )
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 1010.0


def test_compute_cost_basis_nav_one_losing_exit(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    parent = _entry(repo, cost_usd=10.0)
    repo.insert_exit(
        parent_trade_id=parent,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=0.0,
        cost_usd=0.0,
        nav_after_usd=990.0,
        ts=_NOW + 100,
    )
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 990.0


def test_compute_cost_basis_nav_mixed(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    p1 = _entry(repo, cost_usd=10.0, triggering_alert_key="a1")
    p2 = _entry(repo, cost_usd=20.0, triggering_alert_key="a2")
    _entry(repo, cost_usd=15.0, triggering_alert_key="a3")
    repo.insert_exit(
        parent_trade_id=p1,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a"),
        outcome="yes",
        shares=1,
        fill_price=1,
        cost_usd=15.0,
        nav_after_usd=0,
        ts=_NOW + 1,
    )
    repo.insert_exit(
        parent_trade_id=p2,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a"),
        outcome="yes",
        shares=1,
        fill_price=0,
        cost_usd=0.0,
        nav_after_usd=0,
        ts=_NOW + 2,
    )
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 985.0


def test_summary_stats(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    p1 = _entry(repo, cost_usd=10.0, triggering_alert_key="a1", source_wallet="0xw1")
    _entry(repo, cost_usd=20.0, triggering_alert_key="a2", source_wallet="0xw2")
    repo.insert_exit(
        parent_trade_id=p1,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a"),
        outcome="yes",
        shares=1,
        fill_price=1,
        cost_usd=15.0,
        nav_after_usd=0,
        ts=_NOW + 1,
    )
    summary: PaperSummary = repo.summary_stats(starting_bankroll=1000.0)
    assert summary.starting_bankroll == 1000.0
    assert summary.current_nav == 1005.0
    assert summary.realized_pnl == 5.0
    assert summary.open_positions == 1
    assert summary.closed_positions == 1
