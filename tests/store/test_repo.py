"""Tests for the SQLite repositories in ``pscanner.store.repo``."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from pscanner.alerts.models import Alert
from pscanner.poly.ids import AssetId, ConditionId, EventId, EventSlug, MarketId
from pscanner.poly.models import Market
from pscanner.store.repo import (
    AlertsRepo,
    EventOutcomeSumRepo,
    EventOutcomeSumRow,
    EventSnapshot,
    EventSnapshotsRepo,
    EventTagCacheRepo,
    MarketCacheRepo,
    MarketSnapshot,
    MarketSnapshotsRepo,
    MarketTick,
    MarketTicksRepo,
    PositionSnapshotsRepo,
    TrackedWalletCategoriesRepo,
    TrackedWalletsRepo,
    WalletActivityEvent,
    WalletActivityEventsRepo,
    WalletFirstSeenRepo,
    WalletPositionsHistoryRepo,
    WalletPositionsHistoryRow,
    WalletTrade,
    WalletTradesRepo,
    WatchlistRepo,
)


def test_tracked_wallets_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        closed_position_count=30,
        closed_position_wins=21,
        winrate=0.7,
        leaderboard_pnl=1234.5,
        mean_edge=0.12,
        weighted_edge=0.15,
        excess_pnl_usd=2500.0,
        total_stake_usd=8000.0,
    )

    wallets = repo.list_all()
    assert len(wallets) == 1
    wallet = wallets[0]
    assert wallet.address == "0xabc"
    assert wallet.closed_position_count == 30
    assert wallet.closed_position_wins == 21
    assert wallet.winrate == 0.7
    assert wallet.leaderboard_pnl == 1234.5
    assert wallet.mean_edge == 0.12
    assert wallet.weighted_edge == 0.15
    assert wallet.excess_pnl_usd == 2500.0
    assert wallet.total_stake_usd == 8000.0
    assert wallet.last_refreshed_at >= int(time.time()) - 5


def test_tracked_wallets_upsert_round_trip_optional_metrics_default_none(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        closed_position_count=30,
        closed_position_wins=21,
        winrate=0.7,
    )

    wallet = repo.list_all()[0]
    assert wallet.leaderboard_pnl is None
    assert wallet.mean_edge is None
    assert wallet.weighted_edge is None
    assert wallet.excess_pnl_usd is None
    assert wallet.total_stake_usd is None


def test_tracked_wallets_second_upsert_updates_in_place(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        closed_position_count=10,
        closed_position_wins=5,
        winrate=0.5,
        leaderboard_pnl=100.0,
        mean_edge=0.05,
        weighted_edge=0.06,
        excess_pnl_usd=500.0,
        total_stake_usd=2000.0,
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
        mean_edge=0.20,
        weighted_edge=0.22,
        excess_pnl_usd=3000.0,
        total_stake_usd=5000.0,
    )

    wallets = repo.list_all()
    assert len(wallets) == 1
    updated = wallets[0]
    assert updated.closed_position_count == 20
    assert updated.closed_position_wins == 15
    assert updated.winrate == 0.75
    assert updated.leaderboard_pnl is None
    assert updated.mean_edge == 0.20
    assert updated.weighted_edge == 0.22
    assert updated.excess_pnl_usd == 3000.0
    assert updated.total_stake_usd == 5000.0
    assert updated.last_refreshed_at > first_refresh


def test_tracked_wallets_list_active_filters_by_edge_and_excess_pnl(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    # Below resolved threshold (would otherwise pass).
    repo.upsert(
        address="0xfew",
        closed_position_count=5,
        closed_position_wins=5,
        winrate=1.0,
        leaderboard_pnl=10.0,
        mean_edge=0.30,
        weighted_edge=0.30,
        excess_pnl_usd=5000.0,
        total_stake_usd=8000.0,
    )
    # Below mean_edge threshold.
    repo.upsert(
        address="0xlow_edge",
        closed_position_count=30,
        closed_position_wins=10,
        winrate=0.33,
        leaderboard_pnl=10.0,
        mean_edge=0.01,
        weighted_edge=0.01,
        excess_pnl_usd=5000.0,
        total_stake_usd=8000.0,
    )
    # Below excess_pnl_usd threshold.
    repo.upsert(
        address="0xlow_pnl",
        closed_position_count=30,
        closed_position_wins=20,
        winrate=0.66,
        leaderboard_pnl=10.0,
        mean_edge=0.10,
        weighted_edge=0.12,
        excess_pnl_usd=200.0,
        total_stake_usd=8000.0,
    )
    # NULL edge metrics — must be excluded.
    repo.upsert(
        address="0xnull",
        closed_position_count=30,
        closed_position_wins=20,
        winrate=0.66,
        leaderboard_pnl=10.0,
    )
    # Two passing — assert ordering by excess_pnl_usd desc.
    repo.upsert(
        address="0xmid",
        closed_position_count=25,
        closed_position_wins=18,
        winrate=0.72,
        leaderboard_pnl=10.0,
        mean_edge=0.10,
        weighted_edge=0.11,
        excess_pnl_usd=2500.0,
        total_stake_usd=8000.0,
    )
    repo.upsert(
        address="0xhi",
        closed_position_count=40,
        closed_position_wins=36,
        winrate=0.9,
        leaderboard_pnl=10.0,
        mean_edge=0.20,
        weighted_edge=0.22,
        excess_pnl_usd=9000.0,
        total_stake_usd=15000.0,
    )

    active = repo.list_active(
        min_edge=0.05,
        min_excess_pnl_usd=1000.0,
        min_resolved=20,
    )
    assert [w.address for w in active] == ["0xhi", "0xmid"]


def test_tracked_wallets_list_active_empty_when_no_matches(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletsRepo(tmp_db)
    repo.upsert(
        address="0xlow",
        closed_position_count=10,
        closed_position_wins=2,
        winrate=0.2,
        leaderboard_pnl=None,
        mean_edge=0.01,
        weighted_edge=0.01,
        excess_pnl_usd=10.0,
        total_stake_usd=100.0,
    )
    assert repo.list_active(min_edge=0.05, min_excess_pnl_usd=1000.0, min_resolved=20) == []


def test_position_snapshot_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    repo.upsert(
        address="0xabc",
        market_id=ConditionId("m1"),
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
    cond_m1 = ConditionId("m1")
    repo.upsert(address="0xabc", market_id=cond_m1, side="YES", size=100.0, avg_price=0.42)
    repo.upsert(address="0xabc", market_id=cond_m1, side="YES", size=250.0, avg_price=0.50)

    snaps = repo.get_for_wallet("0xabc")
    assert len(snaps) == 1
    assert snaps[0].size == 250.0
    assert snaps[0].avg_price == 0.50


def test_position_snapshot_distinct_sides_kept(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    cond_m1 = ConditionId("m1")
    cond_m2 = ConditionId("m2")
    repo.upsert(address="0xabc", market_id=cond_m1, side="YES", size=100.0, avg_price=0.42)
    repo.upsert(address="0xabc", market_id=cond_m1, side="NO", size=50.0, avg_price=0.55)
    repo.upsert(address="0xabc", market_id=cond_m2, side="YES", size=10.0, avg_price=0.10)

    snaps = repo.get_for_wallet("0xabc")
    assert len(snaps) == 3
    assert {(s.market_id, s.side) for s in snaps} == {
        ("m1", "YES"),
        ("m1", "NO"),
        ("m2", "YES"),
    }


def test_position_snapshot_get_for_wallet_isolates_addresses(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    repo.upsert(address="0xabc", market_id=ConditionId("m1"), side="YES", size=1.0, avg_price=0.1)
    repo.upsert(address="0xdef", market_id=ConditionId("m1"), side="YES", size=2.0, avg_price=0.2)

    assert len(repo.get_for_wallet("0xabc")) == 1
    assert len(repo.get_for_wallet("0xdef")) == 1
    assert repo.get_for_wallet("0xnone") == []


def test_position_snapshot_previous_size(tmp_db: sqlite3.Connection) -> None:
    repo = PositionSnapshotsRepo(tmp_db)
    cond_m1 = ConditionId("m1")
    assert repo.previous_size("0xabc", cond_m1, "YES") is None

    repo.upsert(address="0xabc", market_id=cond_m1, side="YES", size=42.5, avg_price=0.3)
    assert repo.previous_size("0xabc", cond_m1, "YES") == 42.5
    assert repo.previous_size("0xabc", cond_m1, "NO") is None
    assert repo.previous_size("0xother", cond_m1, "YES") is None


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
    assert repo.get(MarketId("does-not-exist")) is None


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


def test_market_cache_round_trips_condition_id_and_event_slug(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    market = Market.model_validate(sample_market_json)
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(market)

    cached = repo.get(market.id)
    assert cached is not None
    assert cached.condition_id == market.condition_id
    assert cached.condition_id is not None
    assert cached.event_slug == market.event_slug
    assert cached.event_slug is not None


def test_market_cache_get_by_condition_id_returns_match(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    market = Market.model_validate(sample_market_json)
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(market)

    assert market.condition_id is not None
    fetched = repo.get_by_condition_id(market.condition_id)
    assert fetched is not None
    assert fetched.market_id == market.id
    assert fetched.condition_id == market.condition_id


def test_market_cache_get_by_condition_id_unknown_returns_none(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketCacheRepo(tmp_db)
    assert repo.get_by_condition_id(ConditionId("0xunknown")) is None


def test_market_cache_get_by_condition_id_handles_null_condition(
    tmp_db: sqlite3.Connection,
    sample_market_json: dict[str, Any],
) -> None:
    payload = dict(sample_market_json)
    payload.pop("conditionId", None)
    market = Market.model_validate(payload)
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(market)
    # An empty-string lookup must not match the NULL row.
    assert repo.get_by_condition_id(ConditionId("")) is None


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


def test_watchlist_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    inserted = repo.upsert(address="0xabc", source="manual", reason="cli-add")
    assert inserted is True

    got = repo.get("0xabc")
    assert got is not None
    assert got.address == "0xabc"
    assert got.source == "manual"
    assert got.reason == "cli-add"
    assert got.active is True
    assert got.added_at >= int(time.time()) - 5


def test_watchlist_second_upsert_preserves_source_and_reason(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    repo.upsert(address="0xabc", source="smart_money", reason="winrate>0.7")
    second = repo.upsert(address="0xabc", source="manual", reason="overwrite-attempt")
    assert second is False

    got = repo.get("0xabc")
    assert got is not None
    assert got.source == "smart_money"
    assert got.reason == "winrate>0.7"


def test_watchlist_set_active_excludes_from_active_list(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    repo.upsert(address="0xabc", source="manual")
    repo.upsert(address="0xdef", source="whale_alert", reason="usd>50k")

    repo.set_active("0xabc", active=False)

    active = repo.list_active()
    assert [e.address for e in active] == ["0xdef"]

    deactivated = repo.get("0xabc")
    assert deactivated is not None
    assert deactivated.active is False

    all_rows = repo.list_all()
    assert {e.address for e in all_rows} == {"0xabc", "0xdef"}


def test_watchlist_get_unknown_returns_none(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    assert repo.get("0xnope") is None


def test_watchlist_list_active_only_returns_active(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    repo.upsert(address="0xa", source="manual")
    repo.upsert(address="0xb", source="manual")
    repo.set_active("0xa", active=False)

    active = repo.list_active()
    assert [e.address for e in active] == ["0xb"]


def _make_trade(
    *,
    txn: str = "0xtx1",
    asset_id: str = "asset-1",
    side: str = "BUY",
    wallet: str = "0xabc",
    condition_id: str = "cond-1",
    timestamp: int = 1_700_000_000,
) -> WalletTrade:
    return WalletTrade(
        transaction_hash=txn,
        asset_id=AssetId(asset_id),
        side=side,
        wallet=wallet,
        condition_id=ConditionId(condition_id),
        size=10.0,
        price=0.42,
        usd_value=4.2,
        status="CONFIRMED",
        source="ws",
        timestamp=timestamp,
        recorded_at=timestamp + 1,
    )


def test_wallet_trades_insert_returns_true_then_false(tmp_db: sqlite3.Connection) -> None:
    repo = WalletTradesRepo(tmp_db)
    trade = _make_trade()
    assert repo.insert(trade) is True
    assert repo.insert(trade) is False


def test_wallet_trades_insert_round_trips_all_fields(tmp_db: sqlite3.Connection) -> None:
    repo = WalletTradesRepo(tmp_db)
    trade = WalletTrade(
        transaction_hash="0xtx9",
        asset_id=AssetId("asset-9"),
        side="SELL",
        wallet="0xWALLET",
        condition_id=ConditionId("cond-9"),
        size=123.5,
        price=0.6125,
        usd_value=75.64,
        status="CONFIRMED",
        source="activity_api",
        timestamp=1_701_000_000,
        recorded_at=1_701_000_005,
    )
    assert repo.insert(trade) is True

    rows = repo.recent_for_wallet("0xWALLET")
    assert len(rows) == 1
    got = rows[0]
    assert got == trade


def test_wallet_trades_recent_for_wallet_orders_desc_and_limits(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletTradesRepo(tmp_db)
    repo.insert(_make_trade(txn="0xa", timestamp=100))
    repo.insert(_make_trade(txn="0xb", timestamp=300))
    repo.insert(_make_trade(txn="0xc", timestamp=200))

    rows = repo.recent_for_wallet("0xabc")
    assert [r.transaction_hash for r in rows] == ["0xb", "0xc", "0xa"]

    limited = repo.recent_for_wallet("0xabc", limit=2)
    assert [r.transaction_hash for r in limited] == ["0xb", "0xc"]


def test_wallet_trades_distinct_side_is_separate_row(tmp_db: sqlite3.Connection) -> None:
    repo = WalletTradesRepo(tmp_db)
    assert repo.insert(_make_trade(txn="0xtx", side="BUY")) is True
    assert repo.insert(_make_trade(txn="0xtx", side="SELL")) is True

    counts = repo.count_by_wallet()
    assert counts == {"0xabc": 2}


def test_wallet_trades_count_by_wallet_groups_correctly(tmp_db: sqlite3.Connection) -> None:
    repo = WalletTradesRepo(tmp_db)
    repo.insert(_make_trade(txn="0x1", wallet="0xa"))
    repo.insert(_make_trade(txn="0x2", wallet="0xa"))
    repo.insert(_make_trade(txn="0x3", wallet="0xb"))

    assert repo.count_by_wallet() == {"0xa": 2, "0xb": 1}


def test_distinct_wallets_for_condition_returns_unique_set(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletTradesRepo(tmp_db)
    repo.insert(_make_trade(txn="0x1", wallet="0xa", condition_id="cond-1", timestamp=100))
    repo.insert(_make_trade(txn="0x2", wallet="0xa", condition_id="cond-1", timestamp=200))
    repo.insert(_make_trade(txn="0x3", wallet="0xb", condition_id="cond-1", timestamp=150))
    repo.insert(_make_trade(txn="0x4", wallet="0xc", condition_id="cond-2", timestamp=200))

    wallets = repo.distinct_wallets_for_condition(ConditionId("cond-1"), since=0)
    assert wallets == {"0xa", "0xb"}


def test_distinct_wallets_for_condition_respects_since_filter(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletTradesRepo(tmp_db)
    repo.insert(_make_trade(txn="0x1", wallet="0xold", condition_id="cond-1", timestamp=50))
    repo.insert(_make_trade(txn="0x2", wallet="0xnew", condition_id="cond-1", timestamp=200))

    assert repo.distinct_wallets_for_condition(ConditionId("cond-1"), since=100) == {"0xnew"}
    assert repo.distinct_wallets_for_condition(ConditionId("cond-1"), since=0) == {"0xold", "0xnew"}
    assert repo.distinct_wallets_for_condition(ConditionId("cond-1"), since=300) == set()


def test_distinct_wallets_for_unknown_condition_returns_empty_set(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletTradesRepo(tmp_db)
    repo.insert(_make_trade(txn="0x1", wallet="0xa", condition_id="cond-1", timestamp=100))
    assert repo.distinct_wallets_for_condition(ConditionId("cond-other"), since=0) == set()


def _make_history_row(
    *,
    wallet: str = "0xabc",
    condition_id: str = "cond-1",
    outcome: str = "Yes",
    snapshot_at: int = 1_700_000_000,
    size: float = 100.0,
    avg_price: float = 0.42,
    current_value: float | None = 50.0,
    cash_pnl: float | None = 8.0,
    realized_pnl: float | None = 2.0,
    redeemable: bool | None = False,
) -> WalletPositionsHistoryRow:
    return WalletPositionsHistoryRow(
        wallet=wallet,
        condition_id=ConditionId(condition_id),
        outcome=outcome,
        size=size,
        avg_price=avg_price,
        current_value=current_value,
        cash_pnl=cash_pnl,
        realized_pnl=realized_pnl,
        redeemable=redeemable,
        snapshot_at=snapshot_at,
    )


def test_positions_history_insert_round_trips_all_fields(tmp_db: sqlite3.Connection) -> None:
    repo = WalletPositionsHistoryRepo(tmp_db)
    row = _make_history_row(redeemable=True)
    assert repo.insert(row) is True

    rows = repo.recent_for_wallet("0xabc")
    assert len(rows) == 1
    got = rows[0]
    assert got == row


def test_positions_history_insert_round_trips_none_optional_fields(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletPositionsHistoryRepo(tmp_db)
    row = _make_history_row(
        current_value=None,
        cash_pnl=None,
        realized_pnl=None,
        redeemable=None,
    )
    assert repo.insert(row) is True

    got = repo.recent_for_wallet("0xabc")[0]
    assert got.current_value is None
    assert got.cash_pnl is None
    assert got.realized_pnl is None
    assert got.redeemable is None


def test_positions_history_insert_pk_collision_returns_false(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletPositionsHistoryRepo(tmp_db)
    row = _make_history_row()
    assert repo.insert(row) is True
    assert repo.insert(row) is False


def test_positions_history_two_snapshots_kept_and_ordered_desc(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletPositionsHistoryRepo(tmp_db)
    repo.insert(_make_history_row(snapshot_at=100, size=10.0))
    repo.insert(_make_history_row(snapshot_at=300, size=30.0))
    repo.insert(_make_history_row(snapshot_at=200, size=20.0))

    rows = repo.recent_for_wallet("0xabc")
    assert [r.snapshot_at for r in rows] == [300, 200, 100]

    limited = repo.recent_for_wallet("0xabc", limit=2)
    assert [r.snapshot_at for r in limited] == [300, 200]


def test_positions_history_count_by_wallet_groups_correctly(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletPositionsHistoryRepo(tmp_db)
    repo.insert(_make_history_row(wallet="0xa", snapshot_at=100))
    repo.insert(_make_history_row(wallet="0xa", snapshot_at=200))
    repo.insert(_make_history_row(wallet="0xb", snapshot_at=300))

    assert repo.count_by_wallet() == {"0xa": 2, "0xb": 1}


def _make_activity_event(
    *,
    wallet: str = "0xabc",
    event_type: str = "TRADE",
    timestamp: int = 1_700_000_000,
    payload: dict[str, Any] | None = None,
    source: str = "activity_api",
) -> WalletActivityEvent:
    body = payload if payload is not None else {"hash": "0xtx", "size": 1.0}
    return WalletActivityEvent(
        wallet=wallet,
        event_type=event_type,
        payload_json=json.dumps(body),
        timestamp=timestamp,
        recorded_at=timestamp + 5,
        source=source,
    )


def test_activity_events_insert_round_trips_all_fields(tmp_db: sqlite3.Connection) -> None:
    repo = WalletActivityEventsRepo(tmp_db)
    event = _make_activity_event(payload={"k": "v", "n": 1})
    assert repo.insert(event) is True

    rows = repo.recent_for_wallet("0xabc")
    assert len(rows) == 1
    got = rows[0]
    assert got == event
    assert json.loads(got.payload_json) == {"k": "v", "n": 1}


def test_activity_events_insert_pk_collision_returns_false(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletActivityEventsRepo(tmp_db)
    event = _make_activity_event()
    assert repo.insert(event) is True
    assert repo.insert(event) is False


def test_activity_events_recent_filters_by_event_type(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletActivityEventsRepo(tmp_db)
    repo.insert(_make_activity_event(event_type="TRADE", timestamp=100))
    repo.insert(_make_activity_event(event_type="REDEEM", timestamp=200))
    repo.insert(_make_activity_event(event_type="SPLIT", timestamp=300))

    trades = repo.recent_for_wallet("0xabc", event_type="TRADE")
    assert [e.event_type for e in trades] == ["TRADE"]

    all_events = repo.recent_for_wallet("0xabc")
    assert [e.timestamp for e in all_events] == [300, 200, 100]


def test_activity_events_count_by_wallet_groups_correctly(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WalletActivityEventsRepo(tmp_db)
    repo.insert(_make_activity_event(wallet="0xa", timestamp=100))
    repo.insert(_make_activity_event(wallet="0xa", timestamp=200))
    repo.insert(_make_activity_event(wallet="0xb", timestamp=300))

    assert repo.count_by_wallet() == {"0xa": 2, "0xb": 1}


def _make_market_snapshot(
    *,
    market_id: str = "m1",
    event_id: str | None = "evt-1",
    outcome_prices_json: str = "[0.42, 0.58]",
    liquidity_usd: float | None = 5000.0,
    volume_usd: float | None = 9000.0,
    active: bool = True,
    snapshot_at: int = 1_700_000_000,
) -> MarketSnapshot:
    return MarketSnapshot(
        market_id=MarketId(market_id),
        event_id=EventId(event_id) if event_id is not None else None,
        outcome_prices_json=outcome_prices_json,
        liquidity_usd=liquidity_usd,
        volume_usd=volume_usd,
        active=active,
        snapshot_at=snapshot_at,
    )


def test_market_snapshots_insert_round_trips_all_fields(tmp_db: sqlite3.Connection) -> None:
    repo = MarketSnapshotsRepo(tmp_db)
    snap = _make_market_snapshot()
    assert repo.insert(snap) is True

    rows = repo.recent_for_market(MarketId("m1"))
    assert len(rows) == 1
    assert rows[0] == snap


def test_market_snapshots_insert_round_trips_none_optionals(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketSnapshotsRepo(tmp_db)
    snap = _make_market_snapshot(
        event_id=None,
        liquidity_usd=None,
        volume_usd=None,
        active=False,
    )
    assert repo.insert(snap) is True

    got = repo.recent_for_market(MarketId("m1"))[0]
    assert got.event_id is None
    assert got.liquidity_usd is None
    assert got.volume_usd is None
    assert got.active is False


def test_market_snapshots_pk_collision_returns_false(tmp_db: sqlite3.Connection) -> None:
    repo = MarketSnapshotsRepo(tmp_db)
    snap = _make_market_snapshot()
    assert repo.insert(snap) is True
    assert repo.insert(snap) is False


def test_market_snapshots_two_snapshots_kept_and_ordered_desc(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketSnapshotsRepo(tmp_db)
    repo.insert(_make_market_snapshot(snapshot_at=100))
    repo.insert(_make_market_snapshot(snapshot_at=300))
    repo.insert(_make_market_snapshot(snapshot_at=200))

    rows = repo.recent_for_market(MarketId("m1"))
    assert [r.snapshot_at for r in rows] == [300, 200, 100]

    limited = repo.recent_for_market(MarketId("m1"), limit=2)
    assert [r.snapshot_at for r in limited] == [300, 200]


def test_market_snapshots_distinct_count_and_count_by_market(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketSnapshotsRepo(tmp_db)
    repo.insert(_make_market_snapshot(market_id="m1", snapshot_at=100))
    repo.insert(_make_market_snapshot(market_id="m2", snapshot_at=100))
    repo.insert(_make_market_snapshot(market_id="m1", snapshot_at=200))

    assert repo.distinct_snapshot_count() == 2
    assert repo.count_by_market() == {"m1": 2, "m2": 1}


def _make_event_snapshot(
    *,
    event_id: str = "evt-1",
    title: str = "Test event",
    slug: str = "test-event",
    liquidity_usd: float | None = 25000.0,
    volume_usd: float | None = 80000.0,
    active: bool = True,
    closed: bool = False,
    market_count: int = 3,
    snapshot_at: int = 1_700_000_000,
) -> EventSnapshot:
    return EventSnapshot(
        event_id=EventId(event_id),
        title=title,
        slug=EventSlug(slug),
        liquidity_usd=liquidity_usd,
        volume_usd=volume_usd,
        active=active,
        closed=closed,
        market_count=market_count,
        snapshot_at=snapshot_at,
    )


def test_event_snapshots_insert_round_trips_all_fields(tmp_db: sqlite3.Connection) -> None:
    repo = EventSnapshotsRepo(tmp_db)
    snap = _make_event_snapshot()
    assert repo.insert(snap) is True

    rows = repo.recent_for_event(EventId("evt-1"))
    assert len(rows) == 1
    assert rows[0] == snap


def test_event_snapshots_insert_round_trips_none_optionals_and_closed(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = EventSnapshotsRepo(tmp_db)
    snap = _make_event_snapshot(
        liquidity_usd=None,
        volume_usd=None,
        active=False,
        closed=True,
    )
    assert repo.insert(snap) is True

    got = repo.recent_for_event(EventId("evt-1"))[0]
    assert got.liquidity_usd is None
    assert got.volume_usd is None
    assert got.active is False
    assert got.closed is True


def test_event_snapshots_pk_collision_returns_false(tmp_db: sqlite3.Connection) -> None:
    repo = EventSnapshotsRepo(tmp_db)
    snap = _make_event_snapshot()
    assert repo.insert(snap) is True
    assert repo.insert(snap) is False


def test_event_snapshots_recent_for_event_orders_desc_and_limits(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = EventSnapshotsRepo(tmp_db)
    repo.insert(_make_event_snapshot(snapshot_at=100))
    repo.insert(_make_event_snapshot(snapshot_at=300))
    repo.insert(_make_event_snapshot(snapshot_at=200))

    rows = repo.recent_for_event(EventId("evt-1"))
    assert [r.snapshot_at for r in rows] == [300, 200, 100]

    limited = repo.recent_for_event(EventId("evt-1"), limit=2)
    assert [r.snapshot_at for r in limited] == [300, 200]


def test_event_snapshots_distinct_count_and_count_by_event(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = EventSnapshotsRepo(tmp_db)
    repo.insert(_make_event_snapshot(event_id="evt-1", snapshot_at=100))
    repo.insert(_make_event_snapshot(event_id="evt-2", snapshot_at=100))
    repo.insert(_make_event_snapshot(event_id="evt-1", snapshot_at=200))

    assert repo.distinct_snapshot_count() == 2
    assert repo.count_by_event() == {"evt-1": 2, "evt-2": 1}


def _make_outcome_sum_row(
    *,
    event_id: str = "evt-1",
    market_count: int = 3,
    price_sum: float = 1.05,
    deviation: float = 0.05,
    snapshot_at: int = 1_700_000_000,
) -> EventOutcomeSumRow:
    return EventOutcomeSumRow(
        event_id=EventId(event_id),
        market_count=market_count,
        price_sum=price_sum,
        deviation=deviation,
        snapshot_at=snapshot_at,
    )


def test_event_outcome_sum_insert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = EventOutcomeSumRepo(tmp_db)
    row = _make_outcome_sum_row()
    assert repo.insert(row) is True

    rows = repo.recent()
    assert len(rows) == 1
    assert rows[0] == row


def test_event_outcome_sum_pk_collision_returns_false(tmp_db: sqlite3.Connection) -> None:
    repo = EventOutcomeSumRepo(tmp_db)
    row = _make_outcome_sum_row()
    assert repo.insert(row) is True
    assert repo.insert(row) is False


def test_event_outcome_sum_recent_orders_desc_and_limits(tmp_db: sqlite3.Connection) -> None:
    repo = EventOutcomeSumRepo(tmp_db)
    repo.insert(_make_outcome_sum_row(event_id="evt-a", snapshot_at=100))
    repo.insert(_make_outcome_sum_row(event_id="evt-b", snapshot_at=300))
    repo.insert(_make_outcome_sum_row(event_id="evt-c", snapshot_at=200))

    rows = repo.recent()
    assert [r.snapshot_at for r in rows] == [300, 200, 100]

    limited = repo.recent(limit=2)
    assert [r.snapshot_at for r in limited] == [300, 200]


def test_event_outcome_sum_by_event_id_isolates(tmp_db: sqlite3.Connection) -> None:
    repo = EventOutcomeSumRepo(tmp_db)
    repo.insert(_make_outcome_sum_row(event_id="evt-1", snapshot_at=100))
    repo.insert(_make_outcome_sum_row(event_id="evt-1", snapshot_at=200))
    repo.insert(_make_outcome_sum_row(event_id="evt-2", snapshot_at=150))

    rows = repo.by_event_id(EventId("evt-1"))
    assert [r.snapshot_at for r in rows] == [200, 100]
    assert all(r.event_id == "evt-1" for r in rows)

    assert repo.by_event_id(EventId("evt-missing")) == []


def test_event_outcome_sum_with_high_deviation_filters_and_orders(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = EventOutcomeSumRepo(tmp_db)
    repo.insert(
        _make_outcome_sum_row(event_id="small", deviation=0.04, snapshot_at=100),
    )
    repo.insert(
        _make_outcome_sum_row(event_id="medium", deviation=-2.0, snapshot_at=200),
    )
    repo.insert(
        _make_outcome_sum_row(event_id="huge", deviation=15.5, snapshot_at=300),
    )
    repo.insert(
        _make_outcome_sum_row(event_id="big_neg", deviation=-7.5, snapshot_at=400),
    )

    rows = repo.with_high_deviation(min_abs_deviation=1.0)
    # Ordered by ABS(deviation) DESC: 15.5, 7.5, 2.0; 0.04 filtered out.
    assert [r.event_id for r in rows] == ["huge", "big_neg", "medium"]

    limited = repo.with_high_deviation(min_abs_deviation=1.0, limit=2)
    assert [r.event_id for r in limited] == ["huge", "big_neg"]

    # Threshold above the largest magnitude returns empty.
    assert repo.with_high_deviation(min_abs_deviation=100.0) == []


def test_event_tag_cache_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = EventTagCacheRepo(tmp_db)
    repo.upsert(EventSlug("evt-1"), ["Sports", "NFL"])

    tags = repo.get(EventSlug("evt-1"))
    assert tags == ["Sports", "NFL"]


def test_event_tag_cache_upsert_overwrites_and_bumps_cached_at(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = EventTagCacheRepo(tmp_db)
    repo.upsert(EventSlug("evt-1"), ["Sports"])
    first_row = tmp_db.execute(
        "SELECT cached_at FROM event_tag_cache WHERE event_slug = ?",
        ("evt-1",),
    ).fetchone()
    assert first_row is not None
    first_cached_at = int(first_row["cached_at"])

    time.sleep(1.1)
    repo.upsert(EventSlug("evt-1"), ["Politics", "Elections"])

    assert repo.get(EventSlug("evt-1")) == ["Politics", "Elections"]
    second_row = tmp_db.execute(
        "SELECT cached_at FROM event_tag_cache WHERE event_slug = ?",
        ("evt-1",),
    ).fetchone()
    assert second_row is not None
    assert int(second_row["cached_at"]) > first_cached_at


def test_event_tag_cache_get_returns_none_when_unknown(tmp_db: sqlite3.Connection) -> None:
    repo = EventTagCacheRepo(tmp_db)
    assert repo.get(EventSlug("does-not-exist")) is None


def test_event_tag_cache_get_supports_empty_tag_list(tmp_db: sqlite3.Connection) -> None:
    repo = EventTagCacheRepo(tmp_db)
    repo.upsert(EventSlug("evt-1"), [])
    assert repo.get(EventSlug("evt-1")) == []


def _twc_upsert(
    repo: TrackedWalletCategoriesRepo,
    *,
    wallet: str,
    category: str,
    position_count: int = 10,
    win_count: int = 7,
    mean_edge: float | None = 0.10,
    weighted_edge: float | None = 0.12,
    excess_pnl_usd: float | None = 1500.0,
    total_stake_usd: float | None = 5000.0,
) -> None:
    """Convenience wrapper for ``TrackedWalletCategoriesRepo.upsert`` with defaults."""
    repo.upsert(
        wallet=wallet,
        category=category,
        position_count=position_count,
        win_count=win_count,
        mean_edge=mean_edge,
        weighted_edge=weighted_edge,
        excess_pnl_usd=excess_pnl_usd,
        total_stake_usd=total_stake_usd,
    )


def test_tracked_wallet_categories_upsert_round_trip(tmp_db: sqlite3.Connection) -> None:
    repo = TrackedWalletCategoriesRepo(tmp_db)
    _twc_upsert(repo, wallet="0xabc", category="thesis")

    rows = repo.list_for_wallet("0xabc")
    assert len(rows) == 1
    row = rows[0]
    assert row.wallet == "0xabc"
    assert row.category == "thesis"
    assert row.position_count == 10
    assert row.win_count == 7
    assert row.mean_edge == 0.10
    assert row.weighted_edge == 0.12
    assert row.excess_pnl_usd == 1500.0
    assert row.total_stake_usd == 5000.0
    assert row.last_refreshed_at >= int(time.time()) - 5


def test_tracked_wallet_categories_upsert_updates_in_place(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletCategoriesRepo(tmp_db)
    _twc_upsert(repo, wallet="0xabc", category="sports", position_count=10, win_count=5)
    first = repo.list_for_wallet("0xabc")[0]

    time.sleep(1.1)
    _twc_upsert(
        repo,
        wallet="0xabc",
        category="sports",
        position_count=20,
        win_count=15,
        mean_edge=0.20,
        excess_pnl_usd=4000.0,
    )

    rows = repo.list_for_wallet("0xabc")
    assert len(rows) == 1
    updated = rows[0]
    assert updated.position_count == 20
    assert updated.win_count == 15
    assert updated.mean_edge == 0.20
    assert updated.excess_pnl_usd == 4000.0
    assert updated.last_refreshed_at > first.last_refreshed_at


def test_tracked_wallet_categories_distinct_categories_per_wallet(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletCategoriesRepo(tmp_db)
    _twc_upsert(repo, wallet="0xabc", category="thesis")
    _twc_upsert(repo, wallet="0xabc", category="sports")
    _twc_upsert(repo, wallet="0xabc", category="esports")

    rows = repo.list_for_wallet("0xabc")
    assert {r.category for r in rows} == {"thesis", "sports", "esports"}


def test_tracked_wallet_categories_list_by_category_filters(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletCategoriesRepo(tmp_db)
    # Different category — must not appear.
    _twc_upsert(repo, wallet="0xother", category="thesis")
    # Below resolved threshold.
    _twc_upsert(
        repo,
        wallet="0xfew",
        category="sports",
        position_count=3,
        win_count=2,
        mean_edge=0.30,
        excess_pnl_usd=5000.0,
    )
    # Below mean_edge threshold.
    _twc_upsert(
        repo,
        wallet="0xlow_edge",
        category="sports",
        position_count=20,
        win_count=10,
        mean_edge=0.02,
        excess_pnl_usd=5000.0,
    )
    # Below excess_pnl_usd threshold.
    _twc_upsert(
        repo,
        wallet="0xlow_pnl",
        category="sports",
        position_count=20,
        win_count=14,
        mean_edge=0.20,
        excess_pnl_usd=200.0,
    )
    # NULL metrics — must be excluded.
    _twc_upsert(
        repo,
        wallet="0xnull",
        category="sports",
        position_count=20,
        win_count=10,
        mean_edge=None,
        excess_pnl_usd=None,
    )
    # Two passing — assert ordering by excess_pnl_usd DESC.
    _twc_upsert(
        repo,
        wallet="0xmid",
        category="sports",
        position_count=20,
        win_count=14,
        mean_edge=0.15,
        excess_pnl_usd=2500.0,
    )
    _twc_upsert(
        repo,
        wallet="0xhi",
        category="sports",
        position_count=30,
        win_count=22,
        mean_edge=0.25,
        excess_pnl_usd=9000.0,
    )

    rows = repo.list_by_category(
        "sports",
        min_edge=0.10,
        min_excess_pnl_usd=1000.0,
        min_resolved=10,
    )
    assert [r.wallet for r in rows] == ["0xhi", "0xmid"]


def test_tracked_wallet_categories_list_all_returns_everything(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletCategoriesRepo(tmp_db)
    _twc_upsert(repo, wallet="0xabc", category="thesis")
    _twc_upsert(repo, wallet="0xabc", category="sports")
    _twc_upsert(repo, wallet="0xdef", category="esports")

    rows = repo.list_all()
    assert {(r.wallet, r.category) for r in rows} == {
        ("0xabc", "thesis"),
        ("0xabc", "sports"),
        ("0xdef", "esports"),
    }


def test_tracked_wallet_categories_list_for_unknown_wallet_returns_empty(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = TrackedWalletCategoriesRepo(tmp_db)
    assert repo.list_for_wallet("0xnone") == []


def _make_market_tick(
    *,
    asset_id: str = "asset-1",
    condition_id: str = "cond-1",
    snapshot_at: int = 1_700_000_000,
    mid_price: float | None = 0.5,
    best_bid: float | None = 0.49,
    best_ask: float | None = 0.51,
    spread: float | None = 0.02,
    bid_depth_top5: float | None = 1500.0,
    ask_depth_top5: float | None = 1700.0,
    last_trade_price: float | None = 0.495,
) -> MarketTick:
    return MarketTick(
        asset_id=AssetId(asset_id),
        condition_id=ConditionId(condition_id),
        snapshot_at=snapshot_at,
        mid_price=mid_price,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        bid_depth_top5=bid_depth_top5,
        ask_depth_top5=ask_depth_top5,
        last_trade_price=last_trade_price,
    )


def test_market_ticks_insert_round_trips_all_fields(tmp_db: sqlite3.Connection) -> None:
    repo = MarketTicksRepo(tmp_db)
    tick = _make_market_tick()
    assert repo.insert(tick) is True

    rows = repo.recent_for_asset(AssetId("asset-1"))
    assert len(rows) == 1
    assert rows[0] == tick


def test_market_ticks_insert_round_trips_none_optionals(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    tick = _make_market_tick(
        mid_price=None,
        best_bid=None,
        best_ask=None,
        spread=None,
        bid_depth_top5=None,
        ask_depth_top5=None,
        last_trade_price=None,
    )
    assert repo.insert(tick) is True

    got = repo.recent_for_asset(AssetId("asset-1"))[0]
    assert got.mid_price is None
    assert got.best_bid is None
    assert got.best_ask is None
    assert got.spread is None
    assert got.bid_depth_top5 is None
    assert got.ask_depth_top5 is None
    assert got.last_trade_price is None


def test_market_ticks_insert_pk_collision_returns_false(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    tick = _make_market_tick()
    assert repo.insert(tick) is True
    assert repo.insert(tick) is False


def test_market_ticks_recent_for_asset_orders_desc_and_limits(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    repo.insert(_make_market_tick(snapshot_at=100))
    repo.insert(_make_market_tick(snapshot_at=300))
    repo.insert(_make_market_tick(snapshot_at=200))

    rows = repo.recent_for_asset(AssetId("asset-1"))
    assert [r.snapshot_at for r in rows] == [300, 200, 100]

    limited = repo.recent_for_asset(AssetId("asset-1"), limit=2)
    assert [r.snapshot_at for r in limited] == [300, 200]


def test_market_ticks_recent_for_asset_isolates_assets(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    repo.insert(_make_market_tick(asset_id="asset-1", snapshot_at=100))
    repo.insert(_make_market_tick(asset_id="asset-2", snapshot_at=200))

    assert [r.asset_id for r in repo.recent_for_asset(AssetId("asset-1"))] == ["asset-1"]
    assert [r.asset_id for r in repo.recent_for_asset(AssetId("asset-2"))] == ["asset-2"]
    assert repo.recent_for_asset(AssetId("asset-missing")) == []


def test_market_ticks_recent_mids_in_window_filters_time_and_orders_asc(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    now = 1_700_000_000
    # Outside window (older than now - 60).
    repo.insert(_make_market_tick(snapshot_at=now - 120, mid_price=0.30))
    # Inside window — three points, inserted out of order.
    repo.insert(_make_market_tick(snapshot_at=now - 30, mid_price=0.42))
    repo.insert(_make_market_tick(snapshot_at=now - 10, mid_price=0.55))
    repo.insert(_make_market_tick(snapshot_at=now - 50, mid_price=0.40))
    # Future point past now (must be excluded by upper bound).
    repo.insert(_make_market_tick(snapshot_at=now + 5, mid_price=0.60))

    pairs = repo.recent_mids_in_window(
        AssetId("asset-1"),
        window_seconds=60,
        now_ts=now,
    )
    assert pairs == [(now - 50, 0.40), (now - 30, 0.42), (now - 10, 0.55)]


def test_market_ticks_recent_mids_in_window_drops_null_mid_rows(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    now = 1_700_000_000
    repo.insert(_make_market_tick(snapshot_at=now - 30, mid_price=None))
    repo.insert(_make_market_tick(snapshot_at=now - 20, mid_price=0.42))
    repo.insert(_make_market_tick(snapshot_at=now - 10, mid_price=None))

    pairs = repo.recent_mids_in_window(
        AssetId("asset-1"),
        window_seconds=60,
        now_ts=now,
    )
    assert pairs == [(now - 20, 0.42)]


def test_market_ticks_recent_mids_in_window_isolates_assets(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    now = 1_700_000_000
    repo.insert(_make_market_tick(asset_id="asset-1", snapshot_at=now - 10, mid_price=0.5))
    repo.insert(_make_market_tick(asset_id="asset-2", snapshot_at=now - 10, mid_price=0.7))

    one = repo.recent_mids_in_window(AssetId("asset-1"), window_seconds=60, now_ts=now)
    two = repo.recent_mids_in_window(AssetId("asset-2"), window_seconds=60, now_ts=now)
    assert one == [(now - 10, 0.5)]
    assert two == [(now - 10, 0.7)]


def test_market_ticks_recent_mids_in_window_empty_when_no_match(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    now = 1_700_000_000
    repo.insert(_make_market_tick(snapshot_at=now - 1000, mid_price=0.5))

    assert repo.recent_mids_in_window(AssetId("asset-1"), window_seconds=60, now_ts=now) == []
    assert repo.recent_mids_in_window(AssetId("asset-missing"), window_seconds=60, now_ts=now) == []


def test_market_ticks_distinct_count_and_count_by_asset(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = MarketTicksRepo(tmp_db)
    repo.insert(_make_market_tick(asset_id="asset-1", snapshot_at=100))
    repo.insert(_make_market_tick(asset_id="asset-2", snapshot_at=100))
    repo.insert(_make_market_tick(asset_id="asset-1", snapshot_at=200))

    assert repo.distinct_snapshot_count() == 2
    assert repo.count_by_asset() == {"asset-1": 2, "asset-2": 1}
