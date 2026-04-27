"""Behavioural tests for :class:`WorkerSink`.

All tests use :class:`FakeClock` and a stub :class:`IAlertSink` that
records calls. No real DB needed. Log assertions use
``structlog.testing.capture_logs`` because pscanner pipes structlog
through ``PrintLoggerFactory`` (see ``src/pscanner/cli.py``) — stdlib
``caplog`` never sees structlog events.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from structlog.testing import capture_logs

import pscanner.alerts.worker_sink as ws_mod
from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.alerts.protocol import IAlertSink
from pscanner.alerts.sink import AlertSink
from pscanner.alerts.worker_sink import WorkerSink
from pscanner.util.clock import FakeClock


def test_alert_sink_satisfies_ialertsink_structurally() -> None:
    """``AlertSink`` already conforms to ``IAlertSink`` without inheritance.

    This guards against accidental signature drift on either side. The
    assertion runs at type-check time via ``ty``; this test is the
    runtime mirror.
    """
    sink: IAlertSink = AlertSink.__new__(AlertSink)
    assert callable(sink.emit)
    assert inspect.iscoroutinefunction(AlertSink.emit)


# ---- shared test helpers ----------------------------------------------------


def _alert(
    key: str = "k1",
    detector: DetectorName = "velocity",
    severity: Severity = "med",
) -> Alert:
    """Minimal Alert for tests; mirrors what velocity emits."""
    return Alert(
        detector=detector,
        alert_key=key,
        severity=severity,
        title="t",
        body={},
        created_at=0,
    )


@dataclass
class StubInnerSink:
    """Records every emit call. Optional ``gate`` blocks emits until set."""

    received: list[Alert] = field(default_factory=list)
    gate: asyncio.Event | None = None
    raise_on: Callable[[Alert], bool] = field(default=lambda _alert: False)
    raise_with: type[Exception] = RuntimeError

    async def emit(self, alert: Alert) -> bool:
        if self.gate is not None:
            await self.gate.wait()
        if self.raise_on(alert):
            raise self.raise_with("stub-inner-sink-raise")
        self.received.append(alert)
        return True


# ---- phase A tests ----------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_returns_immediately_on_non_full_queue() -> None:
    inner = StubInnerSink()
    worker = WorkerSink(inner, maxsize=8, name="t", clock=FakeClock(start=0.0))

    result = await worker.emit(_alert("k1"))

    assert result is True
    # Drain task hasn't been started yet, so the inner sink hasn't run.
    assert inner.received == []
    assert worker._queue.qsize() == 1


@pytest.mark.asyncio
async def test_drain_delivers_alert_to_inner_sink() -> None:
    inner = StubInnerSink()
    worker = WorkerSink(inner, maxsize=8, name="t", clock=FakeClock(start=0.0))
    await worker.start()
    try:
        await worker.emit(_alert("k1"))
        # Yield twice: once for the drain task to wake, once for emit to settle.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert inner.received == [_alert("k1")]
    finally:
        await worker.aclose()


@pytest.mark.asyncio
async def test_drain_preserves_fifo_order() -> None:
    inner = StubInnerSink()
    worker = WorkerSink(inner, maxsize=64, name="t", clock=FakeClock(start=0.0))
    await worker.start()
    try:
        for i in range(50):
            await worker.emit(_alert(f"k{i}"))
        for _ in range(10):
            await asyncio.sleep(0)
        assert [a.alert_key for a in inner.received] == [f"k{i}" for i in range(50)]
    finally:
        await worker.aclose()


@pytest.mark.asyncio
async def test_inner_sink_failure_logs_and_continues() -> None:
    inner = StubInnerSink(raise_on=lambda a: a.alert_key == "k3")
    worker = WorkerSink(inner, maxsize=8, name="t", clock=FakeClock(start=0.0))
    await worker.start()
    try:
        with capture_logs() as logs:
            for i in range(1, 6):
                await worker.emit(_alert(f"k{i}"))
            for _ in range(20):
                await asyncio.sleep(0)
        assert [a.alert_key for a in inner.received] == ["k1", "k2", "k4", "k5"]
        assert any(log["event"] == "worker_sink.drain_failed" for log in logs)
    finally:
        await worker.aclose()


# ---- phase B test -----------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_full_warns_and_blocks() -> None:
    """maxsize=1; hold inner via gate; emit 3 → 3rd blocks until released.

    Sequence: drain dequeues k1 and stalls on gate; queue accepts k2 (cap=1);
    third emit hits ``QueueFull`` on ``put_nowait`` and falls through to
    awaiting ``put``, which blocks until the drain consumes k2.
    """
    gate = asyncio.Event()
    inner = StubInnerSink(gate=gate)
    worker = WorkerSink(inner, maxsize=1, name="t", clock=FakeClock(start=0.0))
    await worker.start()
    try:
        with capture_logs() as logs:
            await worker.emit(_alert("k1"))
            # Pump so drain task picks up k1 and gets stuck on the gate.
            for _ in range(5):
                await asyncio.sleep(0)
            await worker.emit(_alert("k2"))
            # Queue is now {k2}; drain holds k1; 3rd emit fills queue and blocks.
            third = asyncio.create_task(worker.emit(_alert("k3")))
            await asyncio.sleep(0.05)  # let the put_nowait → QueueFull → put run
            assert not third.done()
            assert any(log["event"] == "worker_sink.queue_full" for log in logs)

            gate.set()  # release inner
            await third
            # Drain rest.
            for _ in range(20):
                await asyncio.sleep(0)
        assert [a.alert_key for a in inner.received] == ["k1", "k2", "k3"]
    finally:
        gate.set()
        await worker.aclose()


# ---- phase C tests ----------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_event_fires_on_cadence() -> None:
    inner = StubInnerSink()
    clock = FakeClock(start=0.0)
    worker = WorkerSink(
        inner,
        maxsize=16,
        name="velocity",
        stats_interval_seconds=60.0,
        clock=clock,
    )
    await worker.start()
    try:
        with capture_logs() as logs:
            for i in range(5):
                await worker.emit(_alert(f"k{i}"))
            # Drain everything.
            for _ in range(10):
                await asyncio.sleep(0)
            # Advance to fire one stats interval.
            await clock.advance(60.0)
            await asyncio.sleep(0)

        stats = [log for log in logs if log["event"] == "worker_sink.stats"]
        assert len(stats) >= 1
        first = stats[0]
        assert first["drain_count"] == 5
        assert first["blocking_emit_count"] == 0
        assert first["name"] == "velocity"
    finally:
        await worker.aclose()


@pytest.mark.asyncio
async def test_stats_counters_reset_between_intervals() -> None:
    inner = StubInnerSink()
    clock = FakeClock(start=0.0)
    worker = WorkerSink(
        inner,
        maxsize=16,
        name="t",
        stats_interval_seconds=60.0,
        clock=clock,
    )
    await worker.start()
    try:
        with capture_logs() as logs:
            for i in range(3):
                await worker.emit(_alert(f"a{i}"))
            for _ in range(10):
                await asyncio.sleep(0)
            await clock.advance(60.0)
            await asyncio.sleep(0)

            for i in range(2):
                await worker.emit(_alert(f"b{i}"))
            for _ in range(10):
                await asyncio.sleep(0)
            await clock.advance(60.0)
            await asyncio.sleep(0)

        stats = [log for log in logs if log["event"] == "worker_sink.stats"]
        assert len(stats) >= 2
        assert stats[0]["drain_count"] == 3
        assert stats[1]["drain_count"] == 2
    finally:
        await worker.aclose()


# ---- phase D tests ----------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_drains_pending_then_exits() -> None:
    """Emits queued at aclose time still reach the inner sink before exit."""
    gate = asyncio.Event()
    inner = StubInnerSink(gate=gate)
    worker = WorkerSink(inner, maxsize=16, name="t", clock=FakeClock(start=0.0))
    await worker.start()
    try:
        for i in range(5):
            await worker.emit(_alert(f"k{i}"))
        gate.set()  # let drain proceed
        await worker.aclose()
        assert len(inner.received) == 5
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_aclose_blocks_new_emits() -> None:
    inner = StubInnerSink()
    worker = WorkerSink(inner, maxsize=16, name="t", clock=FakeClock(start=0.0))
    await worker.start()
    await worker.aclose()

    with capture_logs() as logs:
        result = await worker.emit(_alert("after-close"))
    assert result is False
    assert any(log["event"] == "worker_sink.closed_drop" for log in logs)
    assert inner.received == []  # closed_drop never enqueued


@pytest.mark.asyncio
async def test_aclose_timeout_when_inner_stalls() -> None:
    """Inner stalls forever; aclose times out, logs remaining count, returns."""
    gate = asyncio.Event()  # never set
    inner = StubInnerSink(gate=gate)
    # Override the module-level timeout to 0.1s for fast test.
    original = ws_mod._SHUTDOWN_DRAIN_TIMEOUT_SECONDS
    ws_mod._SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 0.1
    try:
        worker = WorkerSink(inner, maxsize=16, name="t", clock=FakeClock(start=0.0))
        await worker.start()
        for i in range(3):
            await worker.emit(_alert(f"k{i}"))
        with capture_logs() as logs:
            await worker.aclose()
        timeout_logs = [log for log in logs if log["event"] == "worker_sink.shutdown_drain_timeout"]
        assert len(timeout_logs) == 1
        assert "remaining" in timeout_logs[0]
    finally:
        ws_mod._SHUTDOWN_DRAIN_TIMEOUT_SECONDS = original
        gate.set()
