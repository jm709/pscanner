"""Tests for ``ActivityCollector`` (REST polling against ``/activity``).

The watchlist registry and data client are mocked. The
:class:`WalletActivityEventsRepo` runs against an in-memory SQLite to verify
end-to-end persistence, dedupe via composite PK, and filter behaviour.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.activity import ActivityCollector
from pscanner.store.repo import WalletActivityEventsRepo

_WALLET = "0xwatched"
_OTHER_WALLET = "0xother"
_BASE_TS = 1_700_000_000


def _make_event(
    *,
    event_type: str = "TRADE",
    timestamp: int = _BASE_TS,
    **extras: Any,
) -> dict[str, Any]:
    """Build a synthetic ``/activity`` event dict."""
    base: dict[str, Any] = {"type": event_type, "timestamp": timestamp}
    base.update(extras)
    return base


def _make_registry(addresses: set[str]) -> MagicMock:
    """Build a ``WatchlistRegistry`` mock returning a snapshot of addresses."""
    addrs = set(addresses)
    registry = MagicMock()
    registry.addresses = MagicMock(side_effect=lambda: set(addrs))
    return registry


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    registry: MagicMock,
    data_client: MagicMock,
    poll_interval_seconds: float = 60.0,
    activity_page_limit: int = 200,
    max_pages: int = 10,
    dup_lookback: int = 50,
) -> tuple[ActivityCollector, WalletActivityEventsRepo]:
    """Wire up a collector with the supplied mocks plus a real repo."""
    repo = WalletActivityEventsRepo(tmp_db)
    collector = ActivityCollector(
        registry=registry,  # type: ignore[arg-type]
        data_client=data_client,  # type: ignore[arg-type]
        activity_repo=repo,
        poll_interval_seconds=poll_interval_seconds,
        activity_page_limit=activity_page_limit,
        max_pages=max_pages,
        dup_lookback=dup_lookback,
    )
    return collector, repo


@pytest.mark.asyncio
async def test_happy_path_single_wallet_mixed_event_types(
    tmp_db: sqlite3.Connection,
) -> None:
    """One wallet, four mixed-type events: all four persisted, ordered desc."""
    events = [
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 1, asset="a"),
        _make_event(event_type="SPLIT", timestamp=_BASE_TS + 2, conditionId="c"),
        _make_event(event_type="MERGE", timestamp=_BASE_TS + 3, conditionId="c"),
        _make_event(event_type="REDEEM", timestamp=_BASE_TS + 4, conditionId="c"),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 4
    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 4
    assert [row.timestamp for row in rows] == [
        _BASE_TS + 4,
        _BASE_TS + 3,
        _BASE_TS + 2,
        _BASE_TS + 1,
    ]
    assert [row.event_type for row in rows] == ["REDEEM", "MERGE", "SPLIT", "TRADE"]
    assert all(row.source == "activity_api" for row in rows)
    assert all(row.wallet == _WALLET for row in rows)
    decoded = json.loads(rows[0].payload_json)
    assert decoded["type"] == "REDEEM"
    assert decoded["timestamp"] == _BASE_TS + 4


@pytest.mark.asyncio
async def test_multiple_wallets_each_with_events(tmp_db: sqlite3.Connection) -> None:
    """Two wallets with disjoint events: both polled, counted by wallet."""
    wallet_a = "0xaaa"
    wallet_b = "0xbbb"
    events_a = [
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 1),
        _make_event(event_type="SPLIT", timestamp=_BASE_TS + 2),
    ]
    events_b = [
        _make_event(event_type="MERGE", timestamp=_BASE_TS + 10),
        _make_event(event_type="REDEEM", timestamp=_BASE_TS + 11),
        _make_event(event_type="CONVERT", timestamp=_BASE_TS + 12),
    ]
    data = MagicMock()

    async def _get(address: str, **_kwargs: Any) -> list[dict[str, Any]]:
        return events_a if address == wallet_a else events_b

    data.get_activity = AsyncMock(side_effect=_get)
    registry = _make_registry({wallet_a, wallet_b})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 5
    counts = repo.count_by_wallet()
    assert counts == {wallet_a: 2, wallet_b: 3}


@pytest.mark.asyncio
async def test_repolling_dedupes_via_composite_pk(tmp_db: sqlite3.Connection) -> None:
    """Second poll inserts only the new event; overlapping rows are skipped."""
    first_page = [
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 1),
        _make_event(event_type="SPLIT", timestamp=_BASE_TS + 2),
        _make_event(event_type="MERGE", timestamp=_BASE_TS + 3),
        _make_event(event_type="REDEEM", timestamp=_BASE_TS + 4),
    ]
    second_page = [*first_page, _make_event(event_type="CONVERT", timestamp=_BASE_TS + 5)]
    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=[first_page, second_page])
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    first_inserted = await collector.poll_all_wallets()
    second_inserted = await collector.poll_all_wallets()

    assert first_inserted == 4
    assert second_inserted == 1
    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 5
    assert rows[0].event_type == "CONVERT"


@pytest.mark.asyncio
async def test_event_missing_type_is_skipped(tmp_db: sqlite3.Connection) -> None:
    """Events without a ``type`` field are dropped silently, others persist."""
    events = [
        {"timestamp": _BASE_TS + 1},
        {"type": "", "timestamp": _BASE_TS + 2},
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 3),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 1
    assert rows[0].event_type == "TRADE"


@pytest.mark.asyncio
async def test_event_missing_timestamp_is_skipped(tmp_db: sqlite3.Connection) -> None:
    """Events without a ``timestamp`` are dropped; valid events still persist."""
    events = [
        {"type": "TRADE"},
        _make_event(event_type="SPLIT", timestamp=_BASE_TS + 9),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 1
    assert rows[0].event_type == "SPLIT"


@pytest.mark.asyncio
async def test_get_activity_failure_isolated_to_one_wallet(
    tmp_db: sqlite3.Connection,
) -> None:
    """A failing ``get_activity`` for one wallet does not stop the others."""
    addrs = {"0xa", "0xfail", "0xc"}

    async def _get(address: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if address == "0xfail":
            msg = "boom"
            raise RuntimeError(msg)
        return [_make_event(event_type="TRADE", timestamp=_BASE_TS + 1)]

    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=_get)
    registry = _make_registry(addrs)
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 2
    assert repo.recent_for_wallet("0xfail") == []
    assert len(repo.recent_for_wallet("0xa")) == 1
    assert len(repo.recent_for_wallet("0xc")) == 1


@pytest.mark.asyncio
async def test_recent_for_wallet_event_type_filter(tmp_db: sqlite3.Connection) -> None:
    """The repo's ``event_type`` filter returns only matching rows."""
    events = [
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 1),
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 2),
        _make_event(event_type="SPLIT", timestamp=_BASE_TS + 3),
        _make_event(event_type="REDEEM", timestamp=_BASE_TS + 4),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    await collector.poll_all_wallets()
    trades = repo.recent_for_wallet(_WALLET, event_type="TRADE")

    assert len(trades) == 2
    assert {row.event_type for row in trades} == {"TRADE"}
    assert [row.timestamp for row in trades] == [_BASE_TS + 2, _BASE_TS + 1]


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` exits within a second after the stop event is set."""
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=[])
    registry = _make_registry({_WALLET})
    collector, _repo = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        poll_interval_seconds=0.05,
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.1)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=1.0,
    )


@pytest.mark.asyncio
async def test_non_serialisable_event_is_skipped(tmp_db: sqlite3.Connection) -> None:
    """A payload containing a non-JSON value is skipped; siblings still persist."""
    events = [
        _make_event(event_type="TRADE", timestamp=_BASE_TS + 1, weird={"a", "b"}),
        _make_event(event_type="SPLIT", timestamp=_BASE_TS + 2),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(tmp_db=tmp_db, registry=registry, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 1
    assert rows[0].event_type == "SPLIT"
    assert _OTHER_WALLET not in repo.count_by_wallet()


def _page_of(*, count: int, start_ts: int, event_type: str = "TRADE") -> list[dict[str, Any]]:
    """Build a page of ``count`` events with strictly-increasing timestamps."""
    return [_make_event(event_type=event_type, timestamp=start_ts + i, idx=i) for i in range(count)]


@pytest.mark.asyncio
async def test_pagination_walks_until_short_page(tmp_db: sqlite3.Connection) -> None:
    """Three pages of distinct events are fully fetched and persisted."""
    page_size = 200
    page_a = _page_of(count=page_size, start_ts=_BASE_TS)
    page_b = _page_of(count=page_size, start_ts=_BASE_TS + page_size)
    page_c = _page_of(count=50, start_ts=_BASE_TS + 2 * page_size)
    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=[page_a, page_b, page_c])
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        activity_page_limit=page_size,
    )

    inserted = await collector.poll_all_wallets()

    assert inserted == 450
    assert data.get_activity.await_count == 3
    offsets = [call.kwargs["offset"] for call in data.get_activity.await_args_list]
    assert offsets == [0, page_size, 2 * page_size]
    assert repo.count_by_wallet() == {_WALLET: 450}


@pytest.mark.asyncio
async def test_pagination_stops_on_dup_streak(tmp_db: sqlite3.Connection) -> None:
    """A second page of pure PK-collisions trips the dup-streak break."""
    page_size = 200
    fresh = _page_of(count=page_size, start_ts=_BASE_TS)
    duplicates = list(fresh)  # exact same wallet/timestamp/event_type → all dupes.
    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=[fresh, duplicates])
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        activity_page_limit=page_size,
        dup_lookback=50,
    )

    inserted = await collector.poll_all_wallets()

    assert inserted == page_size
    # First page (full, all new) + second page (full, all dupes) = 2 calls.
    assert data.get_activity.await_count == 2
    assert repo.count_by_wallet() == {_WALLET: page_size}


@pytest.mark.asyncio
async def test_pagination_stops_on_short_page(tmp_db: sqlite3.Connection) -> None:
    """A short second page ends pagination; only two pages are fetched."""
    page_size = 200
    page_a = _page_of(count=page_size, start_ts=_BASE_TS)
    page_b = _page_of(count=100, start_ts=_BASE_TS + page_size)
    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=[page_a, page_b])
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        activity_page_limit=page_size,
    )

    inserted = await collector.poll_all_wallets()

    assert inserted == 300
    assert data.get_activity.await_count == 2
    assert repo.count_by_wallet() == {_WALLET: 300}


@pytest.mark.asyncio
async def test_pagination_stops_at_max_pages(tmp_db: sqlite3.Connection) -> None:
    """A wallet that only emits full pages of fresh events is capped at max_pages."""
    page_size = 200
    max_pages = 4

    def _page(call_idx: int) -> list[dict[str, Any]]:
        return _page_of(count=page_size, start_ts=_BASE_TS + call_idx * page_size)

    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=[_page(i) for i in range(max_pages + 2)])
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        activity_page_limit=page_size,
        max_pages=max_pages,
        dup_lookback=10_000,  # disable the dup-streak break for this test.
    )

    inserted = await collector.poll_all_wallets()

    assert inserted == page_size * max_pages
    assert data.get_activity.await_count == max_pages
    assert repo.count_by_wallet() == {_WALLET: page_size * max_pages}


@pytest.mark.asyncio
async def test_page_exception_returns_partial_count(tmp_db: sqlite3.Connection) -> None:
    """A failure on page 2 keeps page 1's rows and reports the partial count."""
    page_size = 200
    page_a = _page_of(count=page_size, start_ts=_BASE_TS)

    async def _get(_address: str, **kwargs: Any) -> list[dict[str, Any]]:
        if kwargs["offset"] == 0:
            return page_a
        msg = "boom"
        raise RuntimeError(msg)

    data = MagicMock()
    data.get_activity = AsyncMock(side_effect=_get)
    registry = _make_registry({_WALLET})
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        activity_page_limit=page_size,
    )

    inserted = await collector.poll_all_wallets()

    assert inserted == page_size
    assert data.get_activity.await_count == 2
    assert repo.count_by_wallet() == {_WALLET: page_size}
