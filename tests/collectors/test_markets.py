"""Tests for ``MarketCollector`` (REST polling against gamma ``/markets``).

The :class:`GammaClient` is mocked via ``MagicMock``/``AsyncMock`` (its
``iter_markets`` is patched to yield a synthetic list). The
:class:`MarketSnapshotsRepo` runs against an in-memory SQLite via the
``tmp_db`` conftest fixture so end-to-end persistence is verified.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.markets import MarketCollector
from pscanner.poly.ids import MarketId
from pscanner.poly.models import Market
from pscanner.store.repo import MarketSnapshotsRepo

_BASE_TS = 1_700_000_000


_DEFAULT_PRICES: list[float] = [0.5, 0.5]


def _make_market(
    market_id: str,
    *,
    event_id: str | None = "evt-1",
    outcome_prices: list[float] | None = None,
    liquidity: float | None = 100.0,
    volume: float | None = 1000.0,
    active: bool = True,
) -> Market:
    """Build a synthetic ``Market`` with realistic gamma fields."""
    prices = _DEFAULT_PRICES if outcome_prices is None else outcome_prices
    payload: dict[str, Any] = {
        "id": market_id,
        "question": f"Question for {market_id}",
        "slug": f"market-{market_id}",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [str(p) for p in prices],
        "liquidity": str(liquidity) if liquidity is not None else None,
        "volume": str(volume) if volume is not None else None,
        "active": active,
        "closed": False,
        "enableOrderBook": True,
        "event_id": event_id,
    }
    return Market.model_validate(payload)


def _async_iter(items: list[Market]) -> AsyncIterator[Market]:
    """Wrap a list in an async iterator matching ``iter_markets`` shape."""

    async def _gen() -> AsyncIterator[Market]:
        for item in items:
            yield item

    return _gen()


def _make_gamma(markets: list[Market]) -> AsyncMock:
    """Build a ``GammaClient`` mock whose ``iter_markets`` yields ``markets``."""
    gamma = AsyncMock()
    gamma.iter_markets = lambda **_kwargs: _async_iter(markets)
    return gamma


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    gamma: AsyncMock,
    snapshot_max: int = 5000,
    snapshot_interval_seconds: float = 300.0,
) -> tuple[MarketCollector, MarketSnapshotsRepo]:
    """Wire up a collector with a real repo plus the supplied gamma mock."""
    repo = MarketSnapshotsRepo(tmp_db)
    collector = MarketCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        markets_repo=repo,
        snapshot_interval_seconds=snapshot_interval_seconds,
        snapshot_max=snapshot_max,
    )
    return collector, repo


@pytest.mark.asyncio
async def test_happy_path_five_markets_persisted(tmp_db: sqlite3.Connection) -> None:
    """Five markets snapshotted, all five rows persisted with correct fields."""
    markets = [
        _make_market(
            f"m{i}",
            event_id=f"evt-{i}",
            outcome_prices=[0.4 + i * 0.01, 0.6 - i * 0.01],
            liquidity=1000.0 + i,
            volume=5000.0 + i,
            active=True,
        )
        for i in range(5)
    ]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(tmp_db=tmp_db, gamma=gamma)

    inserted = await collector.snapshot_all_markets()

    assert inserted == 5
    for i in range(5):
        rows = repo.recent_for_market(MarketId(f"m{i}"))
        assert len(rows) == 1
        row = rows[0]
        assert row.market_id == f"m{i}"
        assert row.event_id == f"evt-{i}"
        assert row.liquidity_usd == pytest.approx(1000.0 + i)
        assert row.volume_usd == pytest.approx(5000.0 + i)
        assert row.active is True
        prices = json.loads(row.outcome_prices_json)
        assert prices == pytest.approx([0.4 + i * 0.01, 0.6 - i * 0.01])


@pytest.mark.asyncio
async def test_two_consecutive_snapshots_produce_two_rows(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two snapshots at distinct timestamps produce two rows per market."""
    markets = [_make_market(f"m{i}") for i in range(3)]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(tmp_db=tmp_db, gamma=gamma)

    timestamps = iter([_BASE_TS, _BASE_TS + 60])
    monkeypatch.setattr(
        "pscanner.collectors.markets.time.time",
        lambda: next(timestamps),
    )

    first = await collector.snapshot_all_markets()
    second = await collector.snapshot_all_markets()

    assert first == 3
    assert second == 3
    counts = repo.count_by_market()
    assert counts == {"m0": 2, "m1": 2, "m2": 2}
    assert repo.distinct_snapshot_count() == 2


@pytest.mark.asyncio
async def test_snapshot_max_cap_respected(tmp_db: sqlite3.Connection) -> None:
    """Hard cap stops iteration after ``snapshot_max`` inserts."""
    markets = [_make_market(f"m{i:03d}") for i in range(100)]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(tmp_db=tmp_db, gamma=gamma, snapshot_max=10)

    inserted = await collector.snapshot_all_markets()

    assert inserted == 10
    counts = repo.count_by_market()
    assert len(counts) == 10
    assert sum(counts.values()) == 10


@pytest.mark.asyncio
async def test_markets_with_missing_id_are_skipped(tmp_db: sqlite3.Connection) -> None:
    """Markets whose ``id`` is empty are silently skipped."""
    markets = [
        _make_market("m1"),
        _make_market(""),
        _make_market("m3"),
    ]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(tmp_db=tmp_db, gamma=gamma)

    inserted = await collector.snapshot_all_markets()

    assert inserted == 2
    counts = repo.count_by_market()
    assert counts == {"m1": 1, "m3": 1}


@pytest.mark.asyncio
async def test_empty_outcome_prices_serialise_to_empty_list(
    tmp_db: sqlite3.Connection,
) -> None:
    """A market with empty ``outcome_prices`` round-trips as ``"[]"``."""
    markets = [_make_market("m1", outcome_prices=[])]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(tmp_db=tmp_db, gamma=gamma)

    inserted = await collector.snapshot_all_markets()

    assert inserted == 1
    rows = repo.recent_for_market(MarketId("m1"))
    assert len(rows) == 1
    assert rows[0].outcome_prices_json == "[]"
    assert json.loads(rows[0].outcome_prices_json) == []


@pytest.mark.asyncio
async def test_per_row_insert_exception_does_not_break_cycle() -> None:
    """A repo.insert raising mid-cycle is logged; other rows still persist."""
    markets = [_make_market("m1"), _make_market("m2"), _make_market("m3")]
    gamma = _make_gamma(markets)
    repo = MagicMock()
    repo.insert.side_effect = [True, RuntimeError("disk error"), True]

    collector = MarketCollector(
        gamma_client=gamma,  # type: ignore[arg-type]
        markets_repo=repo,
        snapshot_interval_seconds=300.0,
        snapshot_max=5000,
    )

    inserted = await collector.snapshot_all_markets()

    assert inserted == 2
    assert repo.insert.call_count == 3


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` exits within a second after the stop event is set."""
    markets = [_make_market("m1"), _make_market("m2")]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        gamma=gamma,
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
async def test_run_survives_per_iteration_exception(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised by one ``snapshot_all_markets`` call doesn't kill ``run``."""
    markets = [_make_market("m1")]
    gamma = _make_gamma(markets)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        gamma=gamma,
        snapshot_interval_seconds=0.05,
    )

    calls: list[int] = []
    real_snapshot = collector.snapshot_all_markets

    async def _flaky(_self: MarketCollector) -> int:
        calls.append(1)
        if len(calls) == 1:
            msg = "transient upstream"
            raise RuntimeError(msg)
        return await real_snapshot()

    monkeypatch.setattr(MarketCollector, "snapshot_all_markets", _flaky)

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.25)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=1.0,
    )

    assert len(calls) >= 2
    assert repo.distinct_snapshot_count() >= 1
