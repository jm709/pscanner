"""Tests for the shared :class:`PollingDetector` plumbing.

The base class wires a fixed-interval ``_scan`` loop with logging and
cancellation behaviour. These tests exercise that machinery once with a tiny
concrete subclass; per-detector signal logic lives in the per-detector test
modules.
"""

from __future__ import annotations

import asyncio

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.detectors.polling import PollingDetector
from pscanner.util.clock import FakeClock


class _CountingDetector(PollingDetector):
    """Subclass that increments a counter on every ``_scan`` call.

    ``raise_on`` is a set of 1-indexed iteration numbers on which ``_scan``
    raises ``RuntimeError`` (to exercise the loop's swallow-and-continue
    behaviour). ``cancel_on`` is the iteration on which ``_scan`` raises
    ``asyncio.CancelledError`` instead.
    """

    name = "counting"

    def __init__(
        self,
        *,
        clock: FakeClock,
        interval: float = 1.0,
        raise_on: set[int] | None = None,
        cancel_on: int | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._interval = interval
        self.count = 0
        self._raise_on = raise_on or set()
        self._cancel_on = cancel_on

    def _interval_seconds(self) -> float:
        return self._interval

    async def _scan(self, sink: AlertSink) -> None:
        del sink  # unused; the test exercises the loop, not emission
        self.count += 1
        if self.count == self._cancel_on:
            raise asyncio.CancelledError
        if self.count in self._raise_on:
            msg = f"transient failure on iteration {self.count}"
            raise RuntimeError(msg)


class _StubSink:
    """Stand-in for :class:`AlertSink` — only identity matters here."""


async def _drain() -> None:
    """Yield repeatedly so any pending coroutines get a chance to run."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_run_invokes_scan_on_each_interval(fake_clock: FakeClock) -> None:
    """``run`` calls ``_scan`` once immediately and once per advance."""
    detector = _CountingDetector(clock=fake_clock, interval=10.0)
    sink = _StubSink()

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    await _drain()
    assert detector.count == 1

    await fake_clock.advance(10.0)
    assert detector.count == 2

    await fake_clock.advance(10.0)
    assert detector.count == 3

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_swallows_scan_exception_and_continues(fake_clock: FakeClock) -> None:
    """A ``_scan`` raising ``Exception`` is logged and the loop continues."""
    detector = _CountingDetector(
        clock=fake_clock,
        interval=10.0,
        raise_on={1},
    )
    sink = _StubSink()

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    await _drain()
    assert detector.count == 1

    await fake_clock.advance(10.0)
    assert detector.count == 2

    await fake_clock.advance(10.0)
    assert detector.count == 3

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_propagates_cancelled_error(fake_clock: FakeClock) -> None:
    """``_scan`` raising ``CancelledError`` exits the loop cleanly."""
    detector = _CountingDetector(
        clock=fake_clock,
        interval=10.0,
        cancel_on=1,
    )
    sink = _StubSink()

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    with pytest.raises(asyncio.CancelledError):
        await task
    assert detector.count == 1


class _VaryingIntervalDetector(PollingDetector):
    """Subclass whose ``_interval_seconds`` returns a different value each call."""

    name = "varying"

    def __init__(self, *, clock: FakeClock, intervals: list[float]) -> None:
        super().__init__(clock=clock)
        self._intervals = list(intervals)
        self.interval_history: list[float] = []
        self.scans = 0

    def _interval_seconds(self) -> float:
        # Always return the next unconsumed interval; once exhausted, repeat
        # the last value forever to keep the loop progressing.
        current_idx = len(self.interval_history)
        value = self._intervals[min(current_idx, len(self._intervals) - 1)]
        self.interval_history.append(value)
        return value

    async def _scan(self, sink: AlertSink) -> None:
        del sink  # unused; the test exercises the loop, not emission
        self.scans += 1


@pytest.mark.asyncio
async def test_interval_seconds_is_called_each_cycle(fake_clock: FakeClock) -> None:
    """Each iteration re-reads the interval, so dynamic config takes effect."""
    detector = _VaryingIntervalDetector(clock=fake_clock, intervals=[5.0, 30.0, 60.0])
    sink = _StubSink()

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    await _drain()
    assert detector.scans == 1
    assert detector.interval_history == [5.0]

    # Advancing by less than the next interval should NOT release the sleeper.
    await fake_clock.advance(4.0)
    assert detector.scans == 1

    # Crossing the 5.0 boundary releases the first sleeper; the second cycle
    # then queries _interval_seconds again and parks for 30.0.
    await fake_clock.advance(1.0)
    assert detector.scans == 2
    assert detector.interval_history == [5.0, 30.0]

    # The 30.0 sleeper outlasts a 10.0 advance.
    await fake_clock.advance(10.0)
    assert detector.scans == 2

    # Finish the 30.0 sleeper to confirm it was the active interval.
    await fake_clock.advance(20.0)
    assert detector.scans == 3
    assert detector.interval_history == [5.0, 30.0, 60.0]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
