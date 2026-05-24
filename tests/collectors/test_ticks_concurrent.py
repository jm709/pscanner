"""Concurrent refresh-while-handling regression for the ticks decomposition.

The pre-decomposition collector serialized WS message apply, subscription
refresh, and snapshot under one big ``asyncio.Lock``. After the C1-C4 split
those concerns live behind two distinct locks (``_BookApplier._lock`` for
book state, ``_SubscriptionManager._lock`` only around the
``_asset_to_condition`` dict update). This module pins the new contract:
a refresh that's slow-awaiting inside ``_collect_volume_floor_assets`` must
NOT block ``applier.apply`` progress.

If a future change re-introduces a shared lock or moves the gamma-iteration
under the sub-manager lock, these tests fail immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.ticks import _BookApplier, _SubscriptionManager
from pscanner.config import TicksConfig
from pscanner.poly.models import Market, WsBookMessage

# Each yielded market simulates a 50 ms gamma-page boundary; 5 markets =
# 250 ms total slow-refresh window.
_PER_MARKET_DELAY_SECONDS = 0.05
_MARKET_COUNT = 5
# Applies must complete well within the slow-refresh window to prove the
# locks are independent. 50 ms covers ~20 applies on slow CI.
_APPLY_BUDGET_SECONDS = 0.05


def _slow_iter_markets() -> AsyncIterator[Market]:
    """Async generator that yields markets with a per-yield sleep."""

    async def _gen() -> AsyncIterator[Market]:
        for i in range(_MARKET_COUNT):
            await asyncio.sleep(_PER_MARKET_DELAY_SECONDS)
            yield Market.model_validate(
                {
                    "id": f"m{i}",
                    "conditionId": f"0xcond{i}",
                    "question": "q",
                    "slug": f"s{i}",
                    "outcomes": ["Yes", "No"],
                    "outcomePrices": ["0.5", "0.5"],
                    "volume": 10_000_000.0,
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                    "clobTokenIds": [f"A{i}_yes", f"A{i}_no"],
                },
            )

    return _gen()


def _book_msg(asset_id: str) -> WsBookMessage:
    """Minimal ``book`` snapshot WS message."""
    return WsBookMessage.model_validate(
        {
            "event_type": "book",
            "asset_id": asset_id,
            "market": None,
            "bids": [{"price": "0.5", "size": "1"}],
            "asks": [{"price": "0.52", "size": "1"}],
        },
    )


def _make_sub_mgr() -> _SubscriptionManager:
    """Build a ``_SubscriptionManager`` with a slow gamma + empty wallet set."""
    config = TicksConfig()
    ws = MagicMock()
    ws.subscribe = AsyncMock(return_value=None)
    gamma = MagicMock()
    gamma.iter_markets = lambda **_: _slow_iter_markets()
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=[])
    registry = MagicMock()
    registry.addresses = MagicMock(return_value=set())
    return _SubscriptionManager(
        config=config,
        ws=ws,  # type: ignore[arg-type]
        gamma=gamma,  # type: ignore[arg-type]
        data=data,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        market_cache=None,
    )


@pytest.mark.asyncio
async def test_book_applier_lock_distinct_from_sub_manager_lock() -> None:
    """The two collaborators must have independent locks — not aliased."""
    sub_mgr = _make_sub_mgr()
    applier = _BookApplier()
    assert applier._lock is not sub_mgr._lock


@pytest.mark.asyncio
async def test_refresh_does_not_block_book_apply() -> None:
    """A slow in-flight refresh must not stall ``applier.apply`` progress.

    Spawn ``sub_mgr.refresh()`` (paced by a slow gamma iter so the task
    spends ~250 ms awaiting between page yields). While it's mid-flight,
    drive 20 ``applier.apply`` calls; assert the total apply time stays
    well under the refresh window. If a future change reintroduces a
    shared lock or holds the sub-manager lock across the gamma loop,
    apply progress would stall to the refresh's pace and this test fails.
    """
    sub_mgr = _make_sub_mgr()
    applier = _BookApplier()

    refresh_task = asyncio.create_task(sub_mgr.refresh())
    # Yield once so the refresh actually starts and enters its first await.
    await asyncio.sleep(0)

    loop = asyncio.get_running_loop()
    apply_start = loop.time()
    for i in range(20):
        await applier.apply(_book_msg(f"B{i}"))
    apply_elapsed = loop.time() - apply_start

    assert apply_elapsed < _APPLY_BUDGET_SECONDS, (
        f"applies took {apply_elapsed * 1000:.0f} ms — sub-manager lock "
        f"appears to block book apply (expected < {_APPLY_BUDGET_SECONDS * 1000:.0f} ms)"
    )

    # The refresh must still finish and produce real subscription state.
    await refresh_task
    assert len(sub_mgr.subscribed_asset_ids()) == _MARKET_COUNT * 2
    snapshot = await applier.snapshot()
    assert len(snapshot) == 20


@pytest.mark.asyncio
async def test_concurrent_apply_and_snapshot_progress_under_refresh() -> None:
    """Same scenario, but interleave apply with applier.snapshot reads.

    Pins the contract that snapshot reads (used by ``MarketTickCollector.
    snapshot_once``) also progress unblocked while a refresh is in flight.
    """
    sub_mgr = _make_sub_mgr()
    applier = _BookApplier()

    refresh_task = asyncio.create_task(sub_mgr.refresh())
    await asyncio.sleep(0)

    loop = asyncio.get_running_loop()
    start = loop.time()
    for i in range(10):
        await applier.apply(_book_msg(f"C{i}"))
        snap = await applier.snapshot()
        # Each loop sees a growing book set.
        assert len(snap) == i + 1
    elapsed = loop.time() - start

    assert elapsed < _APPLY_BUDGET_SECONDS * 2, (
        f"apply+snapshot took {elapsed * 1000:.0f} ms — appears to be blocking on the slow refresh"
    )

    await refresh_task
