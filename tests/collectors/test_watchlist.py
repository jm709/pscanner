"""Tests for ``WatchlistRegistry`` and ``WatchlistSyncer``.

Exercises the registry against an in-memory SQLite-backed repo (no mocks for
storage) and the syncer with a ``MagicMock`` for ``AlertSink`` since the sink
is the only abstract collaborator the syncer cares about.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.collectors.watchlist import WatchlistRegistry, WatchlistSyncer
from pscanner.store.repo import TrackedWalletsRepo, WatchlistRepo

_NOW = 1_700_000_000


def _make_alert(
    *,
    detector: str = "whales",
    body: dict[str, Any] | None = None,
    alert_key: str = "whale:0xtx1",
) -> Alert:
    """Build an ``Alert`` with sensible test defaults.

    Note: ``detector`` is typed as ``DetectorName`` literal in the dataclass,
    but the tests need to construct non-whale alerts too; we cast at the call
    site.
    """
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key=alert_key,
        severity="med",
        title="title",
        body=body if body is not None else {"wallet": "0xwhale1"},
        created_at=_NOW,
    )


def test_registry_construction_reloads_from_db(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    repo.upsert(address="0xseed", source="manual", reason="prepop")

    registry = WatchlistRegistry(repo)

    assert "0xseed" in registry
    assert registry.addresses() == {"0xseed"}


def test_registry_add_returns_true_then_false_on_duplicate(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    seen: list[str] = []
    registry.subscribe(seen.append)

    assert registry.add(address="0xabc", source="manual", reason="r") is True
    assert registry.add(address="0xabc", source="manual", reason="r") is False

    assert "0xabc" in registry
    assert registry.addresses() == {"0xabc"}
    assert seen == ["0xabc"]  # callback fires exactly once


def test_registry_add_reactivates_inactive_entry(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    repo.upsert(address="0xrev", source="smart_money", reason="historic")
    repo.set_active("0xrev", active=False)

    registry = WatchlistRegistry(repo)
    seen: list[str] = []
    registry.subscribe(seen.append)

    # The reload at construction excludes inactive rows.
    assert "0xrev" not in registry

    inserted = registry.add(address="0xrev", source="manual", reason="re-add")

    assert inserted is False  # the row already existed
    assert "0xrev" in registry
    row = repo.get("0xrev")
    assert row is not None
    assert row.active is True
    # Reason and source remain those of the first insert (repo preserves provenance).
    assert row.source == "smart_money"
    assert seen == ["0xrev"]


def test_registry_add_invalid_source_raises(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)

    with pytest.raises(ValueError, match="invalid watchlist source"):
        registry.add(address="0xabc", source="not_a_real_source")


def test_registry_deactivate_removes_from_set_but_keeps_row(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    seen: list[str] = []
    registry.subscribe(seen.append)

    registry.add(address="0xabc", source="manual")
    seen.clear()  # ignore the add callback
    registry.deactivate("0xabc")

    assert "0xabc" not in registry
    assert registry.addresses() == set()
    row = repo.get("0xabc")
    assert row is not None
    assert row.active is False
    assert seen == []  # deactivate does not fire callbacks


def test_registry_subscribe_multiple_callbacks_all_fire(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    seen_a: list[str] = []
    seen_b: list[str] = []
    registry.subscribe(seen_a.append)
    registry.subscribe(seen_b.append)

    registry.add(address="0xabc", source="manual")

    assert seen_a == ["0xabc"]
    assert seen_b == ["0xabc"]


def test_registry_addresses_returns_a_copy(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    registry.add(address="0xabc", source="manual")

    snapshot = registry.addresses()
    snapshot.add("0xfake")

    assert "0xfake" not in registry


def test_syncer_constructor_subscribes_to_sink(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    tracked = TrackedWalletsRepo(tmp_db)
    sink = MagicMock()

    syncer = WatchlistSyncer(registry=registry, tracked_repo=tracked, sink=sink)

    sink.subscribe.assert_called_once_with(syncer._on_alert)


def test_syncer_on_alert_adds_whale_wallet(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    tracked = TrackedWalletsRepo(tmp_db)
    sink = MagicMock()
    syncer = WatchlistSyncer(registry=registry, tracked_repo=tracked, sink=sink)

    syncer._on_alert(_make_alert(body={"wallet": "0xwhale"}, alert_key="whale:0xtx7"))

    assert "0xwhale" in registry
    row = repo.get("0xwhale")
    assert row is not None
    assert row.source == "whale_alert"
    assert row.reason == "whale alert whale:0xtx7"


def test_syncer_on_alert_ignores_non_whale_detector(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    tracked = TrackedWalletsRepo(tmp_db)
    sink = MagicMock()
    syncer = WatchlistSyncer(registry=registry, tracked_repo=tracked, sink=sink)

    syncer._on_alert(
        _make_alert(detector="smart_money", body={"wallet": "0xnotwhale"}),
    )

    assert registry.addresses() == set()


def test_syncer_on_alert_swallows_malformed_body(tmp_db: sqlite3.Connection) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    tracked = TrackedWalletsRepo(tmp_db)
    sink = MagicMock()
    syncer = WatchlistSyncer(registry=registry, tracked_repo=tracked, sink=sink)

    # No 'wallet' key in body — must not raise.
    syncer._on_alert(_make_alert(body={"something_else": True}))

    assert registry.addresses() == set()


@pytest.mark.asyncio
async def test_syncer_sync_smart_money_mirrors_all_tracked_wallets(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    tracked = TrackedWalletsRepo(tmp_db)
    tracked.upsert(
        address="0xa",
        closed_position_count=10,
        closed_position_wins=8,
        winrate=0.80,
        leaderboard_pnl=1000.0,
    )
    tracked.upsert(
        address="0xb",
        closed_position_count=20,
        closed_position_wins=15,
        winrate=0.75,
        leaderboard_pnl=2000.0,
    )
    sink = MagicMock()
    syncer = WatchlistSyncer(registry=registry, tracked_repo=tracked, sink=sink)

    await syncer._sync_smart_money()

    assert registry.addresses() == {"0xa", "0xb"}
    row_a = repo.get("0xa")
    row_b = repo.get("0xb")
    assert row_a is not None
    assert row_b is not None
    assert row_a.source == "smart_money"
    assert row_b.source == "smart_money"
    assert row_a.reason == "winrate 0.80"
    assert row_b.reason == "winrate 0.75"


@pytest.mark.asyncio
async def test_syncer_run_loops_and_exits_on_stop_event(
    tmp_db: sqlite3.Connection,
) -> None:
    repo = WatchlistRepo(tmp_db)
    registry = WatchlistRegistry(repo)
    tracked = TrackedWalletsRepo(tmp_db)
    sink = MagicMock()
    syncer = WatchlistSyncer(
        registry=registry,
        tracked_repo=tracked,
        sink=sink,
        sync_interval_seconds=0.01,
    )

    call_count = 0
    original = syncer._sync_smart_money

    async def counting_sync() -> None:
        nonlocal call_count
        call_count += 1
        await original()

    syncer._sync_smart_money = counting_sync  # type: ignore[method-assign]  # ty:ignore[invalid-assignment]

    stop_event = asyncio.Event()
    task = asyncio.create_task(syncer.run(stop_event))
    # Let several iterations run.
    await asyncio.sleep(0.05)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert call_count >= 2
