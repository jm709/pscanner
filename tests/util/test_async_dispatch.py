"""Tests for :class:`pscanner.util.async_dispatch.AsyncDispatcher`.

The dispatcher's contract is exercised end-to-end via the detector
test modules (``test_trade_driven.py``, ``test_cluster.py``,
``test_move_attribution.py``). These focused unit tests pin the
contract directly so future callers find the spec.
"""

from __future__ import annotations

import asyncio

import pytest

from pscanner.util.async_dispatch import AsyncDispatcher


async def _make_coro(out: list[int], value: int) -> None:
    out.append(value)


def test_spawn_without_running_loop_drops_silently() -> None:
    """No event loop → coro closed (not awaited), no pending tasks, no raise."""
    dispatcher = AsyncDispatcher(log_event_no_loop="test.no_event_loop")
    out: list[int] = []
    coro = _make_coro(out, 42)
    dispatcher.spawn(coro, source="t1")
    assert dispatcher.pending == set()
    assert out == []
    # The coroutine must be closed so the test process doesn't leak a warning.
    with pytest.raises(RuntimeError, match=r"cannot reuse already awaited|closed"):
        coro.send(None)


@pytest.mark.asyncio
async def test_spawn_runs_and_clears_on_done() -> None:
    """spawn(coro) schedules + tracks; tasks self-discard on completion."""
    dispatcher = AsyncDispatcher(log_event_no_loop="test.no_event_loop")
    out: list[int] = []
    dispatcher.spawn(_make_coro(out, 1))
    dispatcher.spawn(_make_coro(out, 2))
    assert len(dispatcher.pending) == 2
    for _ in range(10):
        await asyncio.sleep(0)
    assert sorted(out) == [1, 2]
    assert dispatcher.pending == set()


@pytest.mark.asyncio
async def test_pending_is_live_view() -> None:
    """The .pending property mutates as tasks come and go."""
    dispatcher = AsyncDispatcher(log_event_no_loop="test.no_event_loop")
    snapshot_before = dispatcher.pending
    dispatcher.spawn(_make_coro([], 1))
    assert snapshot_before is dispatcher.pending
    assert len(dispatcher.pending) == 1
    for _ in range(5):
        await asyncio.sleep(0)
    assert dispatcher.pending == set()
