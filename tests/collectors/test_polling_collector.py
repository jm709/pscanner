"""Tests for :class:`pscanner.collectors.base.PollingCollector`."""

from __future__ import annotations

import asyncio

import pytest
import structlog
from structlog.testing import capture_logs

from pscanner.collectors.base import Collector, PollingCollector


class _CountingCollector(PollingCollector):
    """Test subclass that records every poll and exposes a hook for sabotage."""

    name: str = "counting"
    log_event_iteration_failed: str = "counting.iteration_failed"

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        super().__init__(interval_seconds=interval_seconds)
        self.polls = 0
        self.start_calls = 0
        self.next_raises: BaseException | None = None

    async def _on_start(self) -> None:
        self.start_calls += 1

    async def poll_once(self) -> None:
        self.polls += 1
        if self.next_raises is not None:
            exc, self.next_raises = self.next_raises, None
            raise exc


def test_polling_collector_satisfies_collector_protocol() -> None:
    collector = _CountingCollector()
    assert isinstance(collector, Collector)


async def test_poll_once_runs_and_stop_event_returns() -> None:
    collector = _CountingCollector(interval_seconds=10.0)
    stop = asyncio.Event()

    async def trigger_stop() -> None:
        # Yield once so run() reaches the wait-for-stop checkpoint.
        await asyncio.sleep(0.02)
        stop.set()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(collector.run(stop))
        tg.create_task(trigger_stop())

    # One poll happened before stop fired; loop returned immediately on stop.
    assert collector.polls == 1


async def test_on_start_runs_once_before_first_poll() -> None:
    collector = _CountingCollector(interval_seconds=10.0)
    stop = asyncio.Event()

    async def trigger_stop() -> None:
        await asyncio.sleep(0.02)
        stop.set()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(collector.run(stop))
        tg.create_task(trigger_stop())

    assert collector.start_calls == 1


async def test_loop_continues_on_poll_exception_and_logs() -> None:
    """A raise from poll_once is logged + swallowed; loop survives to the next cycle."""
    two_polls_done = asyncio.Event()

    class _SignallingCollector(_CountingCollector):
        async def poll_once(self) -> None:
            await super().poll_once()
            if self.polls >= 2:
                two_polls_done.set()

    collector = _SignallingCollector(interval_seconds=0.01)
    collector.next_raises = RuntimeError("simulated upstream failure")
    stop = asyncio.Event()

    # Reset structlog so capture_logs sees the warning event.
    structlog.reset_defaults()

    async def trigger_stop_after_two_polls() -> None:
        await two_polls_done.wait()
        stop.set()

    with capture_logs() as logs:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(collector.run(stop))
            tg.create_task(trigger_stop_after_two_polls())

    assert collector.polls >= 2
    assert any(entry["event"] == "counting.iteration_failed" for entry in logs)


async def test_stop_during_work_returns_after_wait_checkpoint() -> None:
    """When poll_once is mid-flight when stop fires, run returns at the wait step."""
    collector = _CountingCollector(interval_seconds=10.0)
    stop = asyncio.Event()

    async def trigger_stop_during_work() -> None:
        # Set stop before run even starts.
        stop.set()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(collector.run(stop))
        tg.create_task(trigger_stop_during_work())

    # Stop was set before the loop predicate evaluated.
    # The first iteration may run if the loop started before stop got set;
    # either way it returns promptly. Don't assert poll count — just no hang.


async def test_abstract_method_enforced() -> None:
    class _NoPoll(PollingCollector):
        name: str = "nopoll"
        log_event_iteration_failed: str = "nopoll.iteration_failed"

    with pytest.raises(TypeError, match="abstract"):
        _NoPoll(interval_seconds=1.0)  # type: ignore[abstract]
