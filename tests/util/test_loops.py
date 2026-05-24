"""Tests for :func:`pscanner.util.loops.run_periodic`.

The contract is exercised end-to-end by every periodic detector's test
module (PollingDetector subclasses, cluster, whales, smart_money). These
focused unit tests pin the contract directly.
"""

from __future__ import annotations

import asyncio

import pytest

from pscanner.util.clock import FakeClock
from pscanner.util.loops import run_periodic


@pytest.mark.asyncio
async def test_run_periodic_cancellation_propagates() -> None:
    """CancelledError from the surrounding task tears the loop down cleanly."""
    clock = FakeClock(start=0.0)
    iterations = 0

    async def work() -> None:
        nonlocal iterations
        iterations += 1

    task = asyncio.create_task(
        run_periodic(
            work,
            interval_seconds=1.0,
            clock=clock,
            log_event="test.failed",
        ),
    )
    # let one iteration land before cancelling
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert iterations >= 1


@pytest.mark.asyncio
async def test_run_periodic_swallows_exception_and_continues() -> None:
    """work() raising Exception is logged + loop continues to next iteration."""
    clock = FakeClock(start=0.0)
    calls: list[str] = []

    async def work() -> None:
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("boom")

    task = asyncio.create_task(
        run_periodic(
            work,
            interval_seconds=1.0,
            clock=clock,
            log_event="test.failed",
            log_fields={"detector": "test"},
        ),
    )
    await asyncio.sleep(0)
    assert calls == ["call"]
    await clock.advance(1.0)
    await asyncio.sleep(0)
    assert calls == ["call", "call"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_periodic_sleeps_for_interval() -> None:
    """clock.sleep(interval) is awaited between iterations."""
    clock = FakeClock(start=0.0)
    calls: list[float] = []

    async def work() -> None:
        calls.append(clock.now())

    task = asyncio.create_task(
        run_periodic(
            work,
            interval_seconds=5.0,
            clock=clock,
            log_event="test.failed",
        ),
    )
    await asyncio.sleep(0)
    assert calls == [0.0]
    await clock.advance(5.0)
    await asyncio.sleep(0)
    assert calls == [0.0, 5.0]
    await clock.advance(5.0)
    await asyncio.sleep(0)
    assert calls == [0.0, 5.0, 10.0]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
