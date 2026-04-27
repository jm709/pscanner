"""Tests for PaperTradesRepo."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.db import _apply_migrations
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
        "triggering_alert_detector": "smart_money",
        "rule_variant": None,
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


def test_insert_entry_records_detector_and_variant(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    trade_id = repo.insert_entry(
        triggering_alert_key="vel:0xa:1",
        triggering_alert_detector="velocity",
        rule_variant="follow",
        source_wallet=None,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0,
        fill_price=0.5,
        cost_usd=2.5,
        nav_after_usd=1000.0,
        ts=1700000000,
    )
    assert trade_id > 0

    rows = [
        tuple(r)
        for r in tmp_db.execute(
            "SELECT triggering_alert_detector, rule_variant FROM paper_trades WHERE trade_id = ?",
            (trade_id,),
        )
    ]
    assert rows == [("velocity", "follow")]


def test_unique_entry_index_allows_paired_velocity(tmp_db: sqlite3.Connection) -> None:
    """Two velocity entries with the same alert_key but different rule_variants
    both succeed; same key + same variant raises IntegrityError.
    """
    repo = PaperTradesRepo(tmp_db)
    repo.insert_entry(
        triggering_alert_key="vel:0xa:1",
        triggering_alert_detector="velocity",
        rule_variant="follow",
        source_wallet=None,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0,
        fill_price=0.5,
        cost_usd=2.5,
        nav_after_usd=1000.0,
        ts=1700000000,
    )
    repo.insert_entry(
        triggering_alert_key="vel:0xa:1",
        triggering_alert_detector="velocity",
        rule_variant="fade",
        source_wallet=None,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a-n"),
        outcome="no",
        shares=20.0,
        fill_price=0.5,
        cost_usd=2.5,
        nav_after_usd=1000.0,
        ts=1700000000,
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_entry(
            triggering_alert_key="vel:0xa:1",
            triggering_alert_detector="velocity",
            rule_variant="follow",  # duplicate of the first
            source_wallet=None,
            condition_id=ConditionId("0xc"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=0.5,
            cost_usd=2.5,
            nav_after_usd=1000.0,
            ts=1700000000,
        )


def test_existing_rows_backfilled_to_smart_money(tmp_db: sqlite3.Connection) -> None:
    """A row inserted with NULL triggering_alert_detector gets backfilled to
    'smart_money' on next migration apply (simulates upgrade from old schema).
    """
    tmp_db.execute(
        """
        INSERT INTO paper_trades (
          trade_kind, triggering_alert_key, source_wallet, condition_id,
          asset_id, outcome, shares, fill_price, cost_usd, nav_after_usd, ts,
          triggering_alert_detector
        ) VALUES ('entry', 'smart:0xa:1', '0xa', '0xc', 'a-y', 'yes',
                  20.0, 0.5, 10.0, 990.0, 1700000000, NULL)
        """,
    )
    tmp_db.commit()
    _apply_migrations(tmp_db)

    detector = tmp_db.execute(
        "SELECT triggering_alert_detector FROM paper_trades "
        "WHERE triggering_alert_key = 'smart:0xa:1'"
    ).fetchone()[0]
    assert detector == "smart_money"


def test_summary_by_source_groups_correctly(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    e1 = repo.insert_entry(
        triggering_alert_key="smart:0xa:1",
        triggering_alert_detector="smart_money",
        rule_variant=None,
        source_wallet="0xa",
        condition_id=ConditionId("0xc1"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0,
        fill_price=0.5,
        cost_usd=10.0,
        nav_after_usd=1000.0,
        ts=1700000000,
    )
    e2 = repo.insert_entry(
        triggering_alert_key="vel:0xb:1",
        triggering_alert_detector="velocity",
        rule_variant="follow",
        source_wallet=None,
        condition_id=ConditionId("0xc2"),
        asset_id=AssetId("b-y"),
        outcome="yes",
        shares=10.0,
        fill_price=0.25,
        cost_usd=2.5,
        nav_after_usd=1000.0,
        ts=1700000010,
    )
    e3 = repo.insert_entry(
        triggering_alert_key="vel:0xb:1",
        triggering_alert_detector="velocity",
        rule_variant="fade",
        source_wallet=None,
        condition_id=ConditionId("0xc2"),
        asset_id=AssetId("b-n"),
        outcome="no",
        shares=10.0,
        fill_price=0.25,
        cost_usd=2.5,
        nav_after_usd=1000.0,
        ts=1700000010,
    )
    repo.insert_exit(
        parent_trade_id=e1,
        condition_id=ConditionId("0xc1"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,
        nav_after_usd=1010.0,
        ts=1700000100,
    )
    repo.insert_exit(
        parent_trade_id=e2,
        condition_id=ConditionId("0xc2"),
        asset_id=AssetId("b-y"),
        outcome="yes",
        shares=10.0,
        fill_price=1.0,
        cost_usd=10.0,
        nav_after_usd=1017.5,
        ts=1700000200,
    )
    repo.insert_exit(
        parent_trade_id=e3,
        condition_id=ConditionId("0xc2"),
        asset_id=AssetId("b-n"),
        outcome="no",
        shares=10.0,
        fill_price=0.0,
        cost_usd=0.0,
        nav_after_usd=1015.0,
        ts=1700000200,
    )

    rows = repo.summary_by_source()
    by_key = {(r.detector, r.rule_variant): r for r in rows}

    assert by_key[("smart_money", None)].resolved_count == 1
    assert by_key[("smart_money", None)].realized_pnl == pytest.approx(10.0)
    assert by_key[("smart_money", None)].win_rate == pytest.approx(1.0)

    assert by_key[("velocity", "follow")].resolved_count == 1
    assert by_key[("velocity", "follow")].realized_pnl == pytest.approx(7.5)
    assert by_key[("velocity", "follow")].win_rate == pytest.approx(1.0)

    assert by_key[("velocity", "fade")].resolved_count == 1
    assert by_key[("velocity", "fade")].realized_pnl == pytest.approx(-2.5)
    assert by_key[("velocity", "fade")].win_rate == pytest.approx(0.0)
