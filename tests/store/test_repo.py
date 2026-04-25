"""Tests for the SQLite repositories in ``pscanner.store.repo``."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from pscanner.alerts.models import Alert
from pscanner.poly.models import Market
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    PositionSnapshotsRepo,
    TrackedWalletsRepo,
    WalletFirstSeenRepo,
)


def test_tracked_wallets_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        closed_position_count=30,
        closed_position_wins=21,
        winrate=0.7,
        leaderboard_pnl=1234.5,
    )

    wallets = repo.list_all()
    assert len(wallets) == 1
    wallet = wallets[0]
    assert wallet.address == "0xabc"
    assert wallet.closed_position_count == 30
    assert wallet.closed_position_wins == 21
    assert wallet.winrate == 0.7
    assert wallet.leaderboard_pnl == 1234.5
    assert wallet.last_refreshed_at >= int(time.time()) - 5


def test_tracked_wallets_second_upsert_updates_in_place(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        closed_position_count=10,
        closed_position_wins=5,
        winrate=0.5,
        leaderboard_pnl=100.0,
    )
    first_refresh = repo.list_all()[0].last_refreshed_at

    # Force the timestamp to advance by at least one whole second.
    time.sleep(1.1)
    repo.upsert(
        address="0xabc",
        closed_position_count=20,
        closed_position_wins=15,
        winrate=0.75,
        leaderboard_pnl=None,
    )

    wallets = repo.list_all()
    assert len(wallets) == 1
    updated = wallets[0]
    assert updated.closed_position_count == 20
    assert updated.closed_position_wins == 15
    assert updated.winrate == 0.75
    assert updated.leaderboard_pnl is None
    assert updated.last_refreshed_at > first_refresh


def test_tracked_wallets_list_active_filters_and_orders(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    # Below winrate threshold.
    repo.upsert(
        address="0xlow",
        closed_position_count=30,
        closed_position_wins=12,
        winrate=0.4,
        leaderboard_pnl=10.0,
    )
    # Below resolved threshold.
    repo.upsert(
        address="0xfew",
        closed_position_count=5,
        closed_position_wins=5,
        winrate=1.0,
        leaderboard_pnl=10.0,
    )
    # Two passing — different winrates so we can assert ordering.
    repo.upsert(
        address="0xmid",
        closed_position_count=25,
        closed_position_wins=18,
        winrate=0.72,
        leaderboard_pnl=10.0,
    )
    repo.upsert(
        address="0xhi",
        closed_position_count=40,
        closed_position_wins=36,
        winrate=0.9,
        leaderboard_pnl=10.0,
    )

    active = repo.list_active(min_winrate=0.65, min_resolved=20)
    assert [w.address for w in active] == ["0xhi", "0xmid"]


def test_tracked_wallets_list_active_empty_when_no_matches(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xlow",
        closed_position_count=10,
        closed_position_wins=2,
        winrate=0.2,
        leaderboard_pnl=None,
    )
    assert repo.list_active(min_winrate=0.65, min_resolved=20) == []


def test_position_snapshot_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        market_id="m1",
        side="YES",
        size=100.0,
        avg_price=0.42,
    )

    snaps = repo.get_for_wallet("0xabc")
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.market_id == "m1"
    assert snap.side == "YES"
    assert snap.size == 100.0
    assert snap.avg_price == 0.42
    assert snap.snapshot_at >= int(time.time()) - 5


def test_position_snapshot_second_upsert_updates(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    repo.upsert(address="0xabc", market_id="m1", side="YES", size=100.0, avg_price=0.42)
    repo.upsert(address="0xabc", market_id="m1", side="YES", size=250.0, avg_price=0.50)

    snaps = repo.get_for_wallet("0xabc")
    assert len(snaps) == 1
    assert snaps[0].size == 250.0
    assert snaps[0].avg_price == 0.50


def test_position_snapshot_distinct_sides_kept(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    repo.upsert(address="0xabc", market_id="m1", side="YES", size=100.0, avg_price=0.42)
    repo.upsert(address="0xabc", market_id="m1", side="NO", size=50.0, avg_price=0.55)
    repo.upsert(address="0xabc", market_id="m2", side="YES", size=10.0, avg_price=0.10)

    snaps = repo.get_for_wallet("0xabc")
    assert len(snaps) == 3
    assert {(s.market_id, s.side) for s in snaps} == {
        ("m1", "YES"),
        ("m1", "NO"),
        ("m2", "YES"),
    }


def test_position_snapshot_get_for_wallet_isolates_addresses(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    repo.upsert(address="0xabc", market_id="m1", side="YES", size=1.0, avg_price=0.1)
    repo.upsert(address="0xdef", market_id="m1", side="YES", size=2.0, avg_price=0.2)

    assert len(repo.get_for_wallet("0xabc")) == 1
    assert len(repo.get_for_wallet("0xdef")) == 1
    assert repo.get_for_wallet("0xnone") == []


def test_position_snapshot_previous_size(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    assert repo.previous_size("0xabc", "m1", "YES") is None

    repo.upsert(address="0xabc", market_id="m1", side="YES", size=42.5, avg_price=0.3)
    assert repo.previous_size("0xabc", "m1", "YES") == 42.5
    assert repo.previous_size("0xabc", "m1", "NO") is None
    assert repo.previous_size("0xother", "m1", "YES") is None


def test_wallet_first_seen_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = WalletFirstSeenRepo(tmp_db)
    assert repo.get("0xabc") is None

    repo.upsert(address="0xabc", first_activity_at=1_700_000_000, total_trades=42)
    row = repo.get("0xabc")
    assert row is not None
    assert row.first_activity_at == 1_700_000_000
    assert row.total_trades == 42
    assert row.cached_at >= int(time.time()) - 5


def test_wallet_first_seen_upsert_overwrites(tmp_db: sqlite3.Connection) -> None:
    repo = WalletFirstSeenRepo(tmp_db)
    repo.upsert(address="0xabc", first_activity_at=1_700_000_000, total_trades=10)
    first = repo.get("0xabc")
    assert first is not None

    time.sleep(1.1)
    repo.upsert(address="0xabc", first_activity_at=None, total_trades=None)
    second = repo.get("0xabc")
    assert second is not None
    assert second.first_activity_at is None
    assert second.total_trades is None
    assert second.cached_at > first.cached_at


def test_market_cache_upsert_round_trip(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    market = Market.model_validate(sample_market_json)
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(market)

    cached = repo.get(market.id)
    assert cached is not None
    assert cached.market_id == market.id
    assert cached.title == market.question
    assert cached.liquidity_usd == market.liquidity
    assert cached.volume_usd == market.volume
    assert cached.outcome_prices == market.outcome_prices
    assert cached.outcome_prices  # non-empty list of floats
    assert all(isinstance(p, float) for p in cached.outcome_prices)
    assert cached.active is True
    assert cached.cached_at >= int(time.time()) - 5


def test_market_cache_upsert_updates_in_place(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    market = Market.model_validate(sample_market_json)
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(market)

    mutated = sample_market_json | {"active": False, "liquidity": "9999.0"}
    market_v2 = Market.model_validate(mutated)
    repo.upsert(market_v2)

    cached = repo.get(market.id)
    assert cached is not None
    assert cached.active is False
    assert cached.liquidity_usd == 9999.0


def test_market_cache_get_unknown_returns_none(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    assert repo.get("does-not-exist") is None


def test_market_cache_list_active_filters(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    repo = MarketCacheRepo(tmp_db)
    active_market = Market.model_validate(sample_market_json)
    inactive_payload = sample_market_json | {"id": "inactive-id", "active": False}
    inactive_market = Market.model_validate(inactive_payload)
    repo.upsert(active_market)
    repo.upsert(inactive_market)

    rows = repo.list_active()
    assert len(rows) == 1
    assert rows[0].market_id == active_market.id


def test_market_cache_handles_empty_outcome_prices(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    payload = sample_market_json | {"outcomePrices": "[]"}
    market = Market.model_validate(payload)
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(market)

    cached = repo.get(market.id)
    assert cached is not None
    assert cached.outcome_prices == []


def _make_alert(
    *,
    key: str = "k1",
    detector: Any = "smart_money",
    created_at: int = 1_700_000_000,
    body: dict[str, Any] | None = None,
) -> Alert:
    return Alert(
        detector=detector,
        alert_key=key,
        severity="med",
        title="t",
        body=body if body is not None else {"x": 1},
        created_at=created_at,
    )


def test_alerts_insert_if_new_returns_true_then_false(tmp_db: sqlite3.Connection) -> None:
    repo = AlertsRepo(tmp_db)
    alert = _make_alert(key="dup-key")

    assert repo.insert_if_new(alert) is True
    assert repo.insert_if_new(alert) is False


def test_alerts_recent_orders_by_created_at_desc(tmp_db: sqlite3.Connection) -> None:
    repo = AlertsRepo(tmp_db)
    repo.insert_if_new(_make_alert(key="a", created_at=100))
    repo.insert_if_new(_make_alert(key="b", created_at=300))
    repo.insert_if_new(_make_alert(key="c", created_at=200))

    keys = [a.alert_key for a in repo.recent()]
    assert keys == ["b", "c", "a"]


def test_alerts_recent_filters_by_detector(tmp_db: sqlite3.Connection) -> None:
    repo = AlertsRepo(tmp_db)
    repo.insert_if_new(_make_alert(key="sm", detector="smart_money", created_at=100))
    repo.insert_if_new(_make_alert(key="mp", detector="mispricing", created_at=200))
    repo.insert_if_new(_make_alert(key="wh", detector="whales", created_at=300))

    sm_only = repo.recent(detector="smart_money")
    assert [a.alert_key for a in sm_only] == ["sm"]

    none_match = repo.recent(detector="smart_money", limit=0)
    assert none_match == []


def test_alerts_recent_respects_limit(tmp_db: sqlite3.Connection) -> None:
    repo = AlertsRepo(tmp_db)
    for i in range(5):
        repo.insert_if_new(_make_alert(key=f"k{i}", created_at=100 + i))

    assert len(repo.recent(limit=3)) == 3


def test_alerts_recent_round_trips_body_json(tmp_db: sqlite3.Connection) -> None:
    repo = AlertsRepo(tmp_db)
    body = {"wallet": "0xabc", "size": 1.5, "nested": {"yes": True}}
    repo.insert_if_new(_make_alert(key="rt", body=body))

    got = repo.recent()
    assert len(got) == 1
    assert got[0].body == body
    # Verify the stored column actually contains JSON, not the dict repr.
    raw = tmp_db.execute(
        "SELECT body_json FROM alerts WHERE alert_key = ?",
        ("rt",),
    ).fetchone()
    assert json.loads(raw["body_json"]) == body
