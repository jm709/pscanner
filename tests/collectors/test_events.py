"""Tests for ``EventCollector`` (gamma ``/events`` pagination + snapshotting).

The :class:`GammaClient` is mocked via ``AsyncMock``; a real
:class:`EventSnapshotsRepo` runs against an in-memory SQLite (``tmp_db``)
to verify end-to-end persistence and append-only semantics.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.events import EventCollector
from pscanner.poly.ids import EventId, EventSlug
from pscanner.poly.models import Event
from pscanner.store.repo import EventSnapshotsRepo, EventTagCacheRepo

_BASE_TS = 1_700_000_000


def _async_iter(items: list[Event]) -> AsyncIterator[Event]:
    """Wrap a list as an async iterator matching ``GammaClient.iter_events``."""

    async def _gen() -> AsyncIterator[Event]:
        for event in items:
            yield event

    return _gen()


def _make_market(idx: int) -> dict[str, Any]:
    """Build a minimal but realistic gamma market payload."""
    return {
        "id": f"market-{idx}",
        "conditionId": f"0xcond{idx:064x}",
        "question": f"Will outcome {idx} happen?",
        "slug": f"will-outcome-{idx}-happen",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.42", "0.58"]',
        "liquidity": "1234.5",
        "volume": "9876.5",
        "active": True,
        "closed": False,
    }


def _make_event(
    *,
    event_id: str,
    title: str = "Some Event",
    slug: str = "some-event",
    market_count: int = 2,
    liquidity: float | None = 5_000.0,
    volume: float | None = 25_000.0,
    active: bool = True,
    closed: bool = False,
    tags: list[str] | None = None,
) -> Event:
    """Build a validated ``Event`` with ``market_count`` synthetic markets."""
    payload: dict[str, Any] = {
        "id": event_id,
        "title": title,
        "slug": slug,
        "markets": [_make_market(i) for i in range(market_count)],
        "liquidity": liquidity,
        "volume": volume,
        "active": active,
        "closed": closed,
        "tags": tags if tags is not None else [],
    }
    return Event.model_validate(payload)


def _make_gamma(events: list[Event]) -> AsyncMock:
    """Build a mock ``GammaClient`` whose ``iter_events`` yields ``events``."""
    gamma = AsyncMock()
    gamma.iter_events = lambda **_kwargs: _async_iter(events)
    return gamma


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    events: list[Event],
    snapshot_max: int = 2000,
    snapshot_interval_seconds: float = 900.0,
) -> tuple[EventCollector, EventSnapshotsRepo, EventTagCacheRepo]:
    """Wire up a collector with a mocked gamma client and a real repo."""
    repo = EventSnapshotsRepo(tmp_db)
    tag_cache = EventTagCacheRepo(tmp_db)
    gamma = _make_gamma(events)
    collector = EventCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        events_repo=repo,
        event_tag_cache=tag_cache,
        snapshot_interval_seconds=snapshot_interval_seconds,
        snapshot_max=snapshot_max,
    )
    return collector, repo, tag_cache


@pytest.mark.asyncio
async def test_happy_path_four_events_persisted(tmp_db: sqlite3.Connection) -> None:
    """Four events with 2-3 markets each persist as four rows."""
    events = [
        _make_event(event_id="e1", title="Event 1", slug="event-1", market_count=2),
        _make_event(event_id="e2", title="Event 2", slug="event-2", market_count=3),
        _make_event(event_id="e3", title="Event 3", slug="event-3", market_count=2),
        _make_event(event_id="e4", title="Event 4", slug="event-4", market_count=3),
    ]
    collector, repo, _tag_cache = _make_collector(tmp_db=tmp_db, events=events)

    inserted = await collector.snapshot_all_events()

    assert inserted == 4
    counts = repo.count_by_event()
    assert counts == {"e1": 1, "e2": 1, "e3": 1, "e4": 1}
    rows_e2 = repo.recent_for_event(EventId("e2"))
    assert len(rows_e2) == 1
    assert rows_e2[0].title == "Event 2"
    assert rows_e2[0].slug == "event-2"
    assert rows_e2[0].market_count == 3
    assert rows_e2[0].active is True
    assert rows_e2[0].closed is False
    assert rows_e2[0].liquidity_usd == pytest.approx(5_000.0)
    assert rows_e2[0].volume_usd == pytest.approx(25_000.0)


@pytest.mark.asyncio
async def test_two_consecutive_snapshots_yield_two_rows_per_event(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sweeps with distinct ``snapshot_at`` values produce two rows each."""
    events = [
        _make_event(event_id="e1", market_count=1),
        _make_event(event_id="e2", market_count=2),
    ]
    collector, repo, _tag_cache = _make_collector(tmp_db=tmp_db, events=events)

    sweep = {"index": 0}
    timestamps = [_BASE_TS, _BASE_TS + 60]
    monkeypatch.setattr(
        "pscanner.collectors.events.time.time",
        lambda: timestamps[sweep["index"]],
    )

    first = await collector.snapshot_all_events()
    sweep["index"] = 1
    second = await collector.snapshot_all_events()

    assert first == 2
    assert second == 2
    assert repo.distinct_snapshot_count() == 2
    assert repo.count_by_event() == {"e1": 2, "e2": 2}


@pytest.mark.asyncio
async def test_snapshot_max_caps_inserts(tmp_db: sqlite3.Connection) -> None:
    """Cap of 5 over 50 input events stops iteration after 5 inserts."""
    events = [_make_event(event_id=f"e{i}", market_count=1) for i in range(50)]
    collector, repo, _tag_cache = _make_collector(
        tmp_db=tmp_db,
        events=events,
        snapshot_max=5,
    )

    inserted = await collector.snapshot_all_events()

    assert inserted == 5
    counts = repo.count_by_event()
    assert len(counts) == 5
    assert sum(counts.values()) == 5


@pytest.mark.asyncio
async def test_events_with_empty_id_are_skipped(tmp_db: sqlite3.Connection) -> None:
    """Events whose ``id`` is empty/whitespace are dropped silently."""
    events = [
        _make_event(event_id="e1", market_count=1),
        _make_event(event_id="", market_count=1),
        _make_event(event_id="e3", market_count=2),
    ]
    collector, repo, _tag_cache = _make_collector(tmp_db=tmp_db, events=events)

    inserted = await collector.snapshot_all_events()

    assert inserted == 2
    assert repo.count_by_event() == {"e1": 1, "e3": 1}


@pytest.mark.asyncio
async def test_event_with_no_markets_serialises_market_count_zero(
    tmp_db: sqlite3.Connection,
) -> None:
    """An event with ``markets=[]`` writes ``market_count = 0``."""
    events = [_make_event(event_id="e1", market_count=0)]
    collector, repo, _tag_cache = _make_collector(tmp_db=tmp_db, events=events)

    inserted = await collector.snapshot_all_events()

    assert inserted == 1
    rows = repo.recent_for_event(EventId("e1"))
    assert len(rows) == 1
    assert rows[0].market_count == 0


@pytest.mark.asyncio
async def test_per_row_insert_exception_does_not_break_sweep() -> None:
    """A single ``insert`` raising mid-sweep is logged; the loop continues."""
    events = [
        _make_event(event_id="e1", market_count=1),
        _make_event(event_id="e2", market_count=1),
        _make_event(event_id="e3", market_count=1),
    ]
    repo = MagicMock()
    repo.insert.side_effect = [True, RuntimeError("disk error"), True]
    tag_cache = MagicMock()
    gamma = _make_gamma(events)
    collector = EventCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        events_repo=repo,
        event_tag_cache=tag_cache,
        snapshot_interval_seconds=900.0,
        snapshot_max=2000,
    )

    inserted = await collector.snapshot_all_events()

    assert inserted == 2
    assert repo.insert.call_count == 3


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` exits within a second after ``stop_event.set()`` is called."""
    events = [_make_event(event_id="e1", market_count=1)]
    collector, repo, _tag_cache = _make_collector(
        tmp_db=tmp_db,
        events=events,
        snapshot_interval_seconds=0.05,
    )
    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.1)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=1.0,
    )

    assert repo.distinct_snapshot_count() >= 1


@pytest.mark.asyncio
async def test_run_swallows_per_iteration_exceptions(
    tmp_db: sqlite3.Connection,
) -> None:
    """A failing ``snapshot_all_events`` doesn't kill the loop."""
    repo = EventSnapshotsRepo(tmp_db)
    calls = {"n": 0}

    def _make_iter(**_kwargs: Any) -> AsyncIterator[Event]:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "transient upstream error"
            raise RuntimeError(msg)
        return _async_iter([_make_event(event_id="e1", market_count=1)])

    gamma = AsyncMock()
    gamma.iter_events = _make_iter
    tag_cache = EventTagCacheRepo(tmp_db)
    collector = EventCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        events_repo=repo,
        event_tag_cache=tag_cache,
        snapshot_interval_seconds=0.05,
        snapshot_max=2000,
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.2)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=1.0,
    )

    assert calls["n"] >= 2
    assert repo.distinct_snapshot_count() >= 1


@pytest.mark.asyncio
async def test_event_tag_cache_populated_per_event(tmp_db: sqlite3.Connection) -> None:
    """Each event's tags are upserted to the cache (keyed on slug) during the sweep."""
    events = [
        _make_event(event_id="e1", slug="evt-1", market_count=1, tags=["Sports", "NFL"]),
        _make_event(event_id="e2", slug="evt-2", market_count=1, tags=["Esports"]),
        _make_event(event_id="e3", slug="evt-3", market_count=1, tags=[]),
    ]
    collector, _repo, tag_cache = _make_collector(tmp_db=tmp_db, events=events)

    await collector.snapshot_all_events()

    assert tag_cache.get(EventSlug("evt-1")) == ["Sports", "NFL"]
    assert tag_cache.get(EventSlug("evt-2")) == ["Esports"]
    assert tag_cache.get(EventSlug("evt-3")) == []


@pytest.mark.asyncio
async def test_event_tag_cache_upsert_called_with_mock(tmp_db: sqlite3.Connection) -> None:
    """The collector calls ``tag_cache.upsert`` once per event keyed on slug."""
    events = [
        _make_event(event_id="e1", slug="evt-1", market_count=1, tags=["Sports"]),
        _make_event(event_id="e2", slug="evt-2", market_count=1, tags=["Politics"]),
    ]
    repo = EventSnapshotsRepo(tmp_db)
    tag_cache = MagicMock()
    gamma = _make_gamma(events)
    collector = EventCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        events_repo=repo,
        event_tag_cache=tag_cache,
        snapshot_interval_seconds=900.0,
        snapshot_max=2000,
    )

    await collector.snapshot_all_events()

    assert tag_cache.upsert.call_count == 2
    call_args = [call.args for call in tag_cache.upsert.call_args_list]
    assert call_args == [("evt-1", ["Sports"]), ("evt-2", ["Politics"])]


@pytest.mark.asyncio
async def test_event_tag_cache_upsert_failure_does_not_break_sweep(
    tmp_db: sqlite3.Connection,
) -> None:
    """A tag-cache upsert raising must not abort the snapshot loop."""
    events = [
        _make_event(event_id="e1", market_count=1, tags=["Sports"]),
        _make_event(event_id="e2", market_count=1, tags=["Politics"]),
    ]
    repo = EventSnapshotsRepo(tmp_db)
    tag_cache = MagicMock()
    tag_cache.upsert.side_effect = [RuntimeError("disk full"), None]
    gamma = _make_gamma(events)
    collector = EventCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        events_repo=repo,
        event_tag_cache=tag_cache,
        snapshot_interval_seconds=900.0,
        snapshot_max=2000,
    )

    inserted = await collector.snapshot_all_events()

    assert inserted == 2
    assert tag_cache.upsert.call_count == 2
