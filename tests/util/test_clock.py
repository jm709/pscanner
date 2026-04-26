"""Tests for :mod:`pscanner.util.clock`."""

from __future__ import annotations

import asyncio

import pytest

from pscanner.util.clock import FakeClock, RealClock


async def test_fake_clock_now_starts_at_zero_by_default() -> None:
    """Default ``start`` is ``0.0``; ``now`` reflects it before any advance."""
    clock = FakeClock()
    assert clock.now() == 0.0


async def test_fake_clock_now_advances_with_advance() -> None:
    """``advance`` shifts the simulated wall-clock forward."""
    clock = FakeClock(start=100.0)
    await clock.advance(5.5)
    assert clock.now() == pytest.approx(105.5)


async def test_fake_clock_sleep_blocks_until_advance() -> None:
    """A pending ``sleep`` only resolves once ``advance`` reaches its deadline."""
    clock = FakeClock()
    sleeper_done = False

    async def _sleeper() -> None:
        nonlocal sleeper_done
        await clock.sleep(10.0)
        sleeper_done = True

    task = asyncio.create_task(_sleeper())
    # Yield enough times for the task to park on the future.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not sleeper_done

    await clock.advance(5.0)
    assert not sleeper_done

    await clock.advance(5.0)
    assert sleeper_done
    await task


async def test_fake_clock_releases_sleepers_in_deadline_order() -> None:
    """Sleepers wake in deadline order, regardless of scheduling order."""
    clock = FakeClock()
    order: list[str] = []

    async def _sleeper(label: str, seconds: float) -> None:
        await clock.sleep(seconds)
        order.append(label)

    tasks = [
        asyncio.create_task(_sleeper("c", 30.0)),
        asyncio.create_task(_sleeper("a", 10.0)),
        asyncio.create_task(_sleeper("b", 20.0)),
    ]
    for _ in range(5):
        await asyncio.sleep(0)

    await clock.advance(35.0)
    for task in tasks:
        await task
    assert order == ["a", "b", "c"]


async def test_fake_clock_cancelled_sleeper_is_removed_from_heap() -> None:
    """Cancelling a sleeper drops it from the heap; advance does not crash."""
    clock = FakeClock()

    async def _sleeper() -> None:
        await clock.sleep(60.0)

    task = asyncio.create_task(_sleeper())
    for _ in range(5):
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await clock.advance(120.0)
    assert clock.now() == pytest.approx(120.0)


async def test_fake_clock_zero_sleep_yields_cooperatively() -> None:
    """``sleep(0)`` does not queue; it just yields to the event loop."""
    clock = FakeClock()
    other_ran = False

    async def _other() -> None:
        nonlocal other_ran
        other_ran = True

    other_task = asyncio.create_task(_other())
    await clock.sleep(0)
    assert other_ran
    await other_task
    # No sleeper queued by the zero-sleep call.
    await clock.advance(0)


async def test_fake_clock_negative_sleep_yields_cooperatively() -> None:
    """Negative durations are treated as cooperative yields, not queued."""
    clock = FakeClock()
    await clock.sleep(-5.0)
    # If ``sleep`` had queued, advance(0) would not release; this just
    # confirms the heap is empty so a subsequent advance is a no-op.
    await clock.advance(0)
    assert clock.now() == 0.0


async def test_fake_clock_advance_zero_releases_nothing() -> None:
    """``advance(0)`` is a no-op when no sleeper has a zero-or-past deadline."""
    clock = FakeClock()
    sleeper_done = False

    async def _sleeper() -> None:
        nonlocal sleeper_done
        await clock.sleep(1.0)
        sleeper_done = True

    task = asyncio.create_task(_sleeper())
    for _ in range(5):
        await asyncio.sleep(0)

    await clock.advance(0.0)
    assert not sleeper_done

    await clock.advance(1.0)
    assert sleeper_done
    await task


async def test_real_clock_now_returns_a_float() -> None:
    """:meth:`RealClock.now` exposes ``time.time``."""
    clock = RealClock()
    assert isinstance(clock.now(), float)


async def test_real_clock_sleep_zero_returns_immediately() -> None:
    """:meth:`RealClock.sleep` accepts zero and returns without delay."""
    clock = RealClock()
    await clock.sleep(0)
