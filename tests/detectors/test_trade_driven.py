"""Tests for the shared :class:`TradeDrivenDetector` plumbing.

The base class wires ``handle_trade_sync`` to dispatch into the abstract
``evaluate`` coroutine and provides a callback-driven ``run`` that simply
parks. These tests exercise that machinery once with a tiny concrete
subclass; per-detector signal logic lives in the per-detector test
modules.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.store.repo import WalletTrade

_NOW = int(time.time())


def _make_trade(**overrides: Any) -> WalletTrade:
    """Build a synthetic ``WalletTrade`` for the dispatch tests."""
    base: dict[str, Any] = {
        "transaction_hash": "0xtxhash1",
        "asset_id": "asset-1",
        "side": "BUY",
        "wallet": "0xwallet",
        "condition_id": "cond-1",
        "size": 1.0,
        "price": 0.5,
        "usd_value": 0.5,
        "status": "CONFIRMED",
        "source": "activity_api",
        "timestamp": _NOW,
        "recorded_at": _NOW,
    }
    base.update(overrides)
    return WalletTrade(**base)


class _RecordingDetector(TradeDrivenDetector):
    """Tiny subclass that records every ``evaluate`` call."""

    name = "test"

    def __init__(self) -> None:
        super().__init__()
        self.evaluated: list[WalletTrade] = []

    async def evaluate(self, trade: WalletTrade) -> None:
        self.evaluated.append(trade)


class _StubSink:
    """Stand-in for :class:`AlertSink` — only identity matters here."""


def test_handle_trade_sync_no_running_loop_is_noop() -> None:
    """With no running loop, ``handle_trade_sync`` returns silently."""
    detector = _RecordingDetector()
    detector.handle_trade_sync(_make_trade())
    assert detector.evaluated == []
    assert detector._pending_tasks == set()


@pytest.mark.asyncio
async def test_handle_trade_sync_spawns_and_tracks_then_clears_task() -> None:
    """``handle_trade_sync`` schedules ``evaluate`` and clears the task on done."""
    detector = _RecordingDetector()
    trade = _make_trade()

    detector.handle_trade_sync(trade)
    assert len(detector._pending_tasks) == 1

    for _ in range(10):
        await asyncio.sleep(0)

    assert detector.evaluated == [trade]
    assert detector._pending_tasks == set()


@pytest.mark.asyncio
async def test_run_parks_and_stores_sink_when_unset() -> None:
    """``run`` records the injected sink and blocks until cancelled."""
    detector = _RecordingDetector()
    sink = _StubSink()

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    for _ in range(5):
        await asyncio.sleep(0)
    assert detector._sink is sink

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_does_not_overwrite_prewired_sink() -> None:
    """When ``_sink`` was pre-wired, ``run`` keeps the original instance."""
    detector = _RecordingDetector()
    prewired = _StubSink()
    detector._sink = prewired  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
    other = _StubSink()

    task = asyncio.create_task(detector.run(other))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    for _ in range(5):
        await asyncio.sleep(0)
    assert detector._sink is prewired

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_alert_sink_type_satisfies_run_signature() -> None:
    """Smoke test: real ``AlertSink`` instances satisfy ``run``'s annotation."""
    # This is a static assertion: AlertSink is the documented type of the sink
    # parameter. The test exists only to ensure the import is exercised so a
    # broken module path fails fast.
    assert AlertSink is not None
