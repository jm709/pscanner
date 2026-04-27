# WorkerSink Alert-Emit Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offload velocity's per-tick alert-emit path off the consumer hot loop using a reusable, per-detector `WorkerSink` that wraps any `IAlertSink` and drains alerts to it via a background task. Eliminates the 498/h `tick_stream.subscriber_queue_full` warnings observed in the 2026-04-27 1h smoke and provides a primitive future tick-driven detectors can adopt by Scheduler wire-up alone.

**Architecture:** A new `IAlertSink` Protocol captures the public surface tick-driven detectors actually depend on (`async def emit(alert) -> bool`). `AlertSink` stays unchanged and structurally satisfies it. New `WorkerSink(inner: IAlertSink, *, maxsize, name, stats_interval_seconds, clock)` enqueues on `emit()` and drains via a background task that calls `await self._inner.emit(alert)`. Scheduler builds one `WorkerSink` per tick-driven detector (velocity in v1) and routes that detector's `run(sink)` to the wrapped sink.

**Tech Stack:** Python 3.13, `asyncio` (Queue + tasks), `structlog`, `pytest` + `pytest-asyncio`. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-04-27-alert-emitter-design.md`

---

## File Structure

**Create:**
- `src/pscanner/alerts/protocol.py` — single-method `IAlertSink` Protocol.
- `src/pscanner/alerts/worker_sink.py` — `WorkerSink` class + helpers.
- `tests/alerts/test_worker_sink.py` — 10 unit tests + Protocol-conformance check.

**Modify:**
- `src/pscanner/config.py` — add `WorkerSinkConfig` section + register on `Config`.
- `src/pscanner/scheduler.py` — build `WorkerSink` for velocity, route via per-detector sink map, manage lifecycle in `run()` + `aclose()`.
- `src/pscanner/detectors/velocity.py` — annotation change `sink: AlertSink` → `sink: IAlertSink` on `run()` and `evaluate()`.
- `tests/test_config.py` — assert defaults of new `WorkerSinkConfig` fields.

No restructure of existing files; all changes are additive.

## Task ordering

T1 and T2 are independent — they could parallelize, but each is small enough that sequential execution is faster than worktree-overhead. Run sequentially.

| # | Task | Touches | Depends on |
|---|------|---------|------------|
| 1 | `IAlertSink` Protocol | `alerts/protocol.py`, `tests/alerts/test_worker_sink.py` (single conformance test) | — |
| 2 | `WorkerSinkConfig` section | `config.py`, `tests/test_config.py` | — |
| 3 | `WorkerSink` class + 10 unit tests | `alerts/worker_sink.py`, `tests/alerts/test_worker_sink.py` | T1 |
| 4 | Scheduler wiring + velocity annotation | `scheduler.py`, `detectors/velocity.py` | T1, T2, T3 |
| 5 | 2h smoke validation | runtime artifact only | T4 |

---

## Task 1: `IAlertSink` Protocol

Tiny isolated change. Adds the Protocol that `WorkerSink` and detectors will depend on, and a structural-conformance test that locks in the contract.

**Files:**
- Create: `src/pscanner/alerts/protocol.py`
- Modify: `tests/alerts/test_worker_sink.py` (create file with one test)

- [ ] **Step 1.1: Create `src/pscanner/alerts/protocol.py`**

```python
"""``IAlertSink`` Protocol — the surface tick-driven detectors depend on.

Both :class:`AlertSink` (synchronous in-process delivery) and
:class:`WorkerSink` (queue-deferred delivery) implement this Protocol so
detectors can accept either without code changes.
"""

from __future__ import annotations

from typing import Protocol

from pscanner.alerts.models import Alert


class IAlertSink(Protocol):
    """The single method tick-driven detectors call to publish an alert."""

    async def emit(self, alert: Alert) -> bool:
        """Publish ``alert``. Return ``True`` if accepted (newly inserted or
        successfully enqueued), ``False`` if rejected (dedupe hit on the
        synchronous path, or worker closed on the deferred path)."""
        ...
```

- [ ] **Step 1.2: Create `tests/alerts/test_worker_sink.py` with the conformance test**

```python
"""Behavioural tests for :class:`WorkerSink`.

All tests use :class:`FakeClock` and a stub :class:`IAlertSink` that
records calls. No real DB needed. Log assertions use
``structlog.testing.capture_logs`` because pscanner pipes structlog
through ``PrintLoggerFactory`` (see ``src/pscanner/cli.py``) — stdlib
``caplog`` never sees structlog events.
"""

from __future__ import annotations

import asyncio

import pytest
import structlog
from structlog.testing import capture_logs

from pscanner.alerts.models import Alert
from pscanner.alerts.protocol import IAlertSink
from pscanner.alerts.sink import AlertSink


def test_alert_sink_satisfies_iAlertSink_structurally() -> None:
    """``AlertSink`` already conforms to ``IAlertSink`` without inheritance.

    This guards against accidental signature drift on either side. The
    assertion runs at type-check time via ``ty``; this test is the
    runtime mirror.
    """
    sink: IAlertSink = AlertSink.__new__(AlertSink)
    assert callable(sink.emit)
```

(Detail: we use `AlertSink.__new__` to avoid constructing a real one — we
only care that the type system accepts the assignment. Construction
requires an `AlertsRepo`, which isn't part of this test's concern.)

- [ ] **Step 1.3: Run the test, verify it passes**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v -k satisfies_iAlertSink
```

Expected: PASS.

- [ ] **Step 1.4: Type-check both files**

```bash
uv run ty check src/pscanner/alerts/protocol.py tests/alerts/test_worker_sink.py
```

Expected: clean. If `ty` complains that `AlertSink` doesn't satisfy
`IAlertSink`, inspect the signature of `AlertSink.emit` — it should
already match exactly. (Verified at spec time: both are
`async def emit(self, alert: Alert) -> bool`.)

- [ ] **Step 1.5: Lint / format**

```bash
uv run ruff check src/pscanner/alerts/protocol.py tests/alerts/test_worker_sink.py
uv run ruff format --check src/pscanner/alerts/protocol.py tests/alerts/test_worker_sink.py
```

- [ ] **Step 1.6: Commit**

```bash
git add src/pscanner/alerts/protocol.py tests/alerts/test_worker_sink.py
git commit -m "feat(alerts): add IAlertSink Protocol

Captures the surface tick-driven detectors depend on. AlertSink already
conforms structurally; new WorkerSink (next task) will too. Protocol-only
change; zero behaviour change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `WorkerSinkConfig` section

Adds the two tunables exposed to operators: `velocity_maxsize` (per-detector queue depth) and `stats_interval_seconds` (cadence for the periodic `worker_sink.stats` log event). Defaults match the spec.

**Files:**
- Modify: `src/pscanner/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 2.1: Add the failing test in `tests/test_config.py`**

Append to `tests/test_config.py`:

```python
def test_worker_sink_config_defaults() -> None:
    from pscanner.config import Config, WorkerSinkConfig

    cfg = WorkerSinkConfig()
    assert cfg.velocity_maxsize == 4096
    assert cfg.stats_interval_seconds == 60

    root = Config()
    assert root.worker_sink == cfg
```

- [ ] **Step 2.2: Run, verify it fails**

```bash
uv run pytest tests/test_config.py -v -k worker_sink_config
```

Expected: FAIL — `ImportError` for `WorkerSinkConfig` and/or `AttributeError`
for `Config.worker_sink`.

- [ ] **Step 2.3: Add `WorkerSinkConfig` and register on `Config`**

In `src/pscanner/config.py`, place the new section after `PaperTradingConfig`
(line 282-296 region) and before `class Config(BaseModel)` (line 299):

```python
class WorkerSinkConfig(_Section):
    """Tunables for the per-detector :class:`WorkerSink` that offloads
    alert-emit work off tick-driven detectors' hot paths.

    Set ``velocity_maxsize`` higher if ``worker_sink.stats`` reports
    sustained nonzero ``blocking_emit_count`` — a sign the queue is
    chronically full and the inner sink is the bottleneck.
    """

    velocity_maxsize: int = 4096
    stats_interval_seconds: int = 60
```

Then in the `Config` class (around line 299-318), add a field. Place it
after `paper_trading` for visual grouping:

```python
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    worker_sink: WorkerSinkConfig = Field(default_factory=WorkerSinkConfig)
```

- [ ] **Step 2.4: Run, verify it passes**

```bash
uv run pytest tests/test_config.py -v -k worker_sink_config
```

Expected: PASS.

- [ ] **Step 2.5: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/config.py tests/test_config.py
uv run ruff format --check src/pscanner/config.py tests/test_config.py
uv run ty check src/pscanner/config.py
```

- [ ] **Step 2.6: Commit**

```bash
git add src/pscanner/config.py tests/test_config.py
git commit -m "feat(config): add WorkerSinkConfig section

Two operator tunables for the upcoming WorkerSink primitive:
- velocity_maxsize=4096: per-detector queue depth.
- stats_interval_seconds=60: cadence for the periodic worker_sink.stats
  event.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `WorkerSink` class + 10 unit tests

The core of the work. Implemented via TDD in five phases inside one task: skeleton, `emit + drain`, queue-full path, `aclose` lifecycle, stats loop. Each phase ends with passing tests so we always have a working partial implementation.

**Files:**
- Create: `src/pscanner/alerts/worker_sink.py`
- Modify: `tests/alerts/test_worker_sink.py` (append phase-by-phase)

### Phase A — skeleton + emit + drain (tests 1-4)

- [ ] **Step 3.1: Write the skeleton in `src/pscanner/alerts/worker_sink.py`**

```python
"""Per-detector ``WorkerSink`` — defers alert-emit work off the hot path.

Wraps any :class:`IAlertSink`. ``emit()`` enqueues onto a bounded
``asyncio.Queue`` and returns immediately; a background drain task pulls
alerts off the queue and invokes the wrapped inner sink. A separate
stats task emits a periodic ``worker_sink.stats`` event with queue
health counters.

Goal: keep tick-driven detectors' per-tick processing pure-CPU even when
the inner sink (SQLite write + renderer + sync subscribers) is briefly
slow. See ``docs/superpowers/specs/2026-04-27-alert-emitter-design.md``.
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.protocol import IAlertSink
from pscanner.util.clock import Clock, SystemClock

_LOG = structlog.get_logger(__name__)
_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 5.0


class WorkerSink:
    """Queue-deferred :class:`IAlertSink` implementation."""

    def __init__(
        self,
        inner: IAlertSink,
        *,
        maxsize: int,
        name: str,
        stats_interval_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        self._inner = inner
        self._name = name
        self._stats_interval = stats_interval_seconds
        self._clock = clock if clock is not None else SystemClock()
        self._queue: asyncio.Queue[Alert] = asyncio.Queue(maxsize=maxsize)
        self._drain_task: asyncio.Task[None] | None = None
        self._stats_task: asyncio.Task[None] | None = None
        self._closing = False
        self._drained = 0
        self._blocking_emits = 0
        self._depth_max = 0

    async def emit(self, alert: Alert) -> bool:
        raise NotImplementedError  # phase A step 3.3 fills this in

    async def start(self) -> None:
        raise NotImplementedError  # phase A step 3.3 fills this in

    async def aclose(self) -> None:
        raise NotImplementedError  # phase D fills this in
```

(Yes, `NotImplementedError` placeholders here are fine because the next
step writes failing tests against them, then phase steps implement.
Skeleton compiles; nothing imports `WorkerSink` yet.)

- [ ] **Step 3.2: Append phase-A tests to `tests/alerts/test_worker_sink.py`**

Append after the conformance test from Task 1:

```python
# ---- shared test helpers ----------------------------------------------------

from collections.abc import Callable
from dataclasses import dataclass, field

from pscanner.alerts.protocol import IAlertSink
from pscanner.alerts.worker_sink import WorkerSink
from pscanner.util.clock import FakeClock


def _alert(key: str = "k1", detector: str = "velocity", severity: str = "med") -> Alert:
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
    """Records every emit call. Optional `gate` blocks emits until set."""

    received: list[Alert] = field(default_factory=list)
    gate: asyncio.Event | None = None
    raise_on: Callable[[Alert], bool] = lambda _alert: False
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
    worker = WorkerSink(inner, maxsize=8, name="t", clock=FakeClock(start_time=0.0))

    result = await worker.emit(_alert("k1"))

    assert result is True
    # Drain task hasn't been started yet, so the inner sink hasn't run.
    assert inner.received == []
    assert worker._queue.qsize() == 1


@pytest.mark.asyncio
async def test_drain_delivers_alert_to_inner_sink() -> None:
    inner = StubInnerSink()
    worker = WorkerSink(inner, maxsize=8, name="t", clock=FakeClock(start_time=0.0))
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
    worker = WorkerSink(inner, maxsize=64, name="t", clock=FakeClock(start_time=0.0))
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
    worker = WorkerSink(inner, maxsize=8, name="t", clock=FakeClock(start_time=0.0))
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
```

- [ ] **Step 3.3: Run phase-A tests, verify they fail with NotImplementedError**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v -k "emit_returns_immediately or drain_delivers or drain_preserves_fifo or inner_sink_failure"
```

Expected: 4 fails (NotImplementedError on emit / start / aclose).

- [ ] **Step 3.4: Implement phase A in `worker_sink.py`**

Replace the three `raise NotImplementedError` lines from step 3.1 with:

```python
    async def emit(self, alert: Alert) -> bool:
        """Enqueue ``alert`` for deferred delivery; return when queue accepts.

        Returns ``True`` once the alert is on the queue. Returns ``False``
        only when ``aclose()`` has been called — see :meth:`aclose`. The
        return is *not* the dedupe bool of the inner sink (which we only
        learn after the drain task runs); see the design spec for the
        rationale (no current tick-driven caller uses the bool).
        """
        if self._closing:
            _LOG.warning("worker_sink.closed_drop", name=self._name, alert_key=alert.alert_key)
            return False
        try:
            self._queue.put_nowait(alert)
        except asyncio.QueueFull:
            _LOG.warning("worker_sink.queue_full", name=self._name, alert_key=alert.alert_key)
            self._blocking_emits += 1
            await self._queue.put(alert)
        depth = self._queue.qsize()
        if depth > self._depth_max:
            self._depth_max = depth
        return True

    async def start(self) -> None:
        """Launch the drain and stats background tasks. Idempotent."""
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(
                self._drain_loop(), name=f"worker_sink.drain[{self._name}]",
            )
        if self._stats_task is None:
            self._stats_task = asyncio.create_task(
                self._stats_loop(), name=f"worker_sink.stats[{self._name}]",
            )

    async def aclose(self) -> None:
        """Stop accepting emits, drain remaining queue, cancel tasks."""
        self._closing = True
        if self._drain_task is not None:
            try:
                await asyncio.wait_for(
                    self._queue.join(),
                    timeout=_SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                _LOG.warning(
                    "worker_sink.shutdown_drain_timeout",
                    name=self._name,
                    remaining=self._queue.qsize(),
                )
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drain_task
            self._drain_task = None
        if self._stats_task is not None:
            self._stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stats_task
            self._stats_task = None

    async def _drain_loop(self) -> None:
        while True:
            alert = await self._queue.get()
            try:
                try:
                    await self._inner.emit(alert)
                    self._drained += 1
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _LOG.exception(
                        "worker_sink.drain_failed",
                        name=self._name,
                        alert_key=alert.alert_key,
                    )
            finally:
                self._queue.task_done()

    async def _stats_loop(self) -> None:
        while True:
            await self._clock.sleep(self._stats_interval)
            try:
                _LOG.info(
                    "worker_sink.stats",
                    name=self._name,
                    queue_depth_max=self._depth_max,
                    blocking_emit_count=self._blocking_emits,
                    drain_count=self._drained,
                )
            except Exception:
                pass
            self._depth_max = 0
            self._blocking_emits = 0
            self._drained = 0
```

(Notes on subtleties:
- `put_nowait` raises `QueueFull`. We use it as the fast path.
- `_queue.task_done()` MUST be called in `finally:` so the `_queue.join()`
  in `aclose()` can complete even when the inner sink raises.
- `aclose()` uses real `asyncio.wait_for` for the drain timeout — that's
  loop-time, not `Clock` time. Test #10 uses a tiny real-time timeout to
  exercise this path.)

- [ ] **Step 3.5: Run phase-A tests, verify they pass**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v -k "emit_returns_immediately or drain_delivers or drain_preserves_fifo or inner_sink_failure"
```

Expected: 4 passed.

### Phase B — queue-full warn-and-block (test 5)

- [ ] **Step 3.6: Append phase-B test**

Append to `tests/alerts/test_worker_sink.py`:

```python
@pytest.mark.asyncio
async def test_queue_full_warns_and_blocks() -> None:
    """maxsize=2; hold inner via gate; emit 3 → 3rd blocks until released."""
    gate = asyncio.Event()
    inner = StubInnerSink(gate=gate)
    worker = WorkerSink(inner, maxsize=2, name="t", clock=FakeClock(start_time=0.0))
    await worker.start()
    try:
        with capture_logs() as logs:
            await worker.emit(_alert("k1"))
            await worker.emit(_alert("k2"))
            # Pump so drain task picks up k1 and gets stuck on the gate.
            for _ in range(5):
                await asyncio.sleep(0)
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
```

- [ ] **Step 3.7: Run, verify it passes**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v -k queue_full_warns_and_blocks
```

Expected: PASS. (The phase-A `emit()` already handles `QueueFull` per the
implementation in step 3.4.)

### Phase C — stats loop (tests 6-7)

- [ ] **Step 3.8: Append phase-C tests**

```python
@pytest.mark.asyncio
async def test_stats_event_fires_on_cadence() -> None:
    inner = StubInnerSink()
    clock = FakeClock(start_time=0.0)
    worker = WorkerSink(
        inner, maxsize=16, name="velocity", stats_interval_seconds=60.0, clock=clock,
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
    clock = FakeClock(start_time=0.0)
    worker = WorkerSink(
        inner, maxsize=16, name="t", stats_interval_seconds=60.0, clock=clock,
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
```

- [ ] **Step 3.9: Run, verify they pass**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v -k "stats_event_fires or stats_counters_reset"
```

Expected: 2 passed. (The stats loop and counter resets are already
implemented in step 3.4.)

### Phase D — aclose lifecycle (tests 8-10)

- [ ] **Step 3.10: Append phase-D tests**

```python
@pytest.mark.asyncio
async def test_aclose_drains_pending_then_exits() -> None:
    """Emits queued at aclose time still reach the inner sink before exit."""
    gate = asyncio.Event()
    inner = StubInnerSink(gate=gate)
    worker = WorkerSink(inner, maxsize=16, name="t", clock=FakeClock(start_time=0.0))
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
    worker = WorkerSink(inner, maxsize=16, name="t", clock=FakeClock(start_time=0.0))
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
    import pscanner.alerts.worker_sink as ws_mod

    original = ws_mod._SHUTDOWN_DRAIN_TIMEOUT_SECONDS
    ws_mod._SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 0.1
    try:
        worker = WorkerSink(inner, maxsize=16, name="t", clock=FakeClock(start_time=0.0))
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
```

- [ ] **Step 3.11: Run, verify they pass**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v -k "aclose_drains or aclose_blocks or aclose_timeout"
```

Expected: 3 passed.

### Phase E — full-suite green + ship

- [ ] **Step 3.12: Run the full new test file, then the full repo suite**

```bash
uv run pytest tests/alerts/test_worker_sink.py -v
uv run pytest -q
```

Expected: 11 tests in `test_worker_sink.py` (1 conformance + 10 behaviour);
full suite still 620+ green (test counts may shift by one or two due to
the conformance test's import side-effects).

- [ ] **Step 3.13: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/alerts/worker_sink.py tests/alerts/test_worker_sink.py
uv run ruff format --check src/pscanner/alerts/worker_sink.py tests/alerts/test_worker_sink.py
uv run ty check src/pscanner/alerts/worker_sink.py
```

Expected: clean. If `ruff` flags `_drain_loop` complexity (>8) split out
`_handle_one(alert)` and call it from the loop body. If `ty` flags
`Clock` Protocol structural issues, double-check the import.

- [ ] **Step 3.14: Commit**

```bash
git add src/pscanner/alerts/worker_sink.py tests/alerts/test_worker_sink.py
git commit -m "feat(alerts): add WorkerSink for deferred alert emission

Per-detector primitive that wraps any IAlertSink. emit() puts the alert
on a bounded asyncio.Queue and returns immediately; a drain task pulls
alerts off the queue and invokes the inner sink. A stats task emits a
periodic worker_sink.stats event.

Behaviour covered by 10 unit tests + structural Protocol-conformance
test against AlertSink. Decouples velocity's per-tick alert path from
the consumer hot loop. Not yet wired in (next task).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Scheduler wiring + velocity annotation

The integration step. Builds a `WorkerSink` for velocity, threads it through `_supervise_detector`, and updates lifecycle. Also flips velocity's type annotation.

**Files:**
- Modify: `src/pscanner/scheduler.py`
- Modify: `src/pscanner/detectors/velocity.py`

- [ ] **Step 4.1: Update velocity annotations**

In `src/pscanner/detectors/velocity.py`:

1. At the top, replace the `AlertSink` import with `IAlertSink`:
   ```python
   from pscanner.alerts.protocol import IAlertSink
   ```
   (If `from pscanner.alerts.sink import AlertSink` was present, remove
   it. If `AlertSink` is referenced anywhere else in the file, leave that
   import alone; otherwise delete it.)

2. In `run`'s signature:
   ```python
   async def run(self, sink: IAlertSink) -> None:
   ```

3. In `evaluate`'s signature:
   ```python
   async def evaluate(self, tick: TickEvent, sink: IAlertSink) -> None:
   ```

No internal call-site changes — `await sink.emit(alert)` is unchanged.

- [ ] **Step 4.2: Type-check velocity, run its tests**

```bash
uv run ty check src/pscanner/detectors/velocity.py
uv run pytest tests/detectors/test_velocity.py -v
```

Expected: clean type-check; tests pass unchanged. If a test fixture
explicitly annotates `sink: AlertSink`, change it to `sink: IAlertSink`
or drop the annotation — `AlertSink` instances structurally satisfy
both.

- [ ] **Step 4.3: Add the per-detector sink map + workers list to Scheduler**

In `src/pscanner/scheduler.py`, in `Scheduler.__init__` (find the body
that constructs `self._sink = AlertSink(self._alerts_repo, renderer=...)`
around line 154):

Below that line, add:

```python
        self._detector_sinks: dict[str, IAlertSink] = {}
        self._workers: list[WorkerSink] = []
```

Add the imports at the top of `scheduler.py` (with the other
`pscanner.alerts.*` imports near line 32):

```python
from pscanner.alerts.protocol import IAlertSink
from pscanner.alerts.worker_sink import WorkerSink
```

- [ ] **Step 4.4: Build a `WorkerSink` for velocity in `_maybe_attach_velocity_detector`**

Find `_maybe_attach_velocity_detector` (currently around line 364-381).
Replace its body so it both builds the detector AND wires the worker:

```python
    def _maybe_attach_velocity_detector(self, detectors: dict[str, Any]) -> None:
        """Attach the velocity detector if both ticks and velocity are enabled.

        Velocity goes through a per-detector :class:`WorkerSink` that
        defers the alert-emit work off its tick-consume hot loop. The
        worker's lifecycle is owned by the scheduler.
        """
        if not self._config.velocity.enabled:
            return
        tick_collector = self._collectors.get("tick_collector")
        if not isinstance(tick_collector, MarketTickCollector):
            return
        detectors["velocity"] = PriceVelocityDetector(
            config=self._config.velocity,
            tick_stream=self._tick_stream,
            market_cache=self._market_cache_repo,
            clock=self._clock,
        )
        worker = WorkerSink(
            self._sink,
            maxsize=self._config.worker_sink.velocity_maxsize,
            name="velocity",
            stats_interval_seconds=self._config.worker_sink.stats_interval_seconds,
            clock=self._clock,
        )
        self._detector_sinks["velocity"] = worker
        self._workers.append(worker)
```

- [ ] **Step 4.5: Route per-detector sinks through `_supervise_detector`**

In `_supervise_detector` (currently around line 422-434), replace the
`await run_fn(self._sink)` call. The current shape is:

```python
    async def _supervise_detector(
        self,
        name: str,
        run_fn: Callable[[AlertSink], Awaitable[None]],
    ) -> None:
        """Restart a detector on unexpected return/exception, up to a cap."""
        restarts: list[float] = []
        while True:
            try:
                await run_fn(self._sink)
                _LOG.warning("scanner.detector.returned", detector=name)
            except asyncio.CancelledError:
                raise
            ...
```

Update the type hint and call to consult the per-detector map:

```python
    async def _supervise_detector(
        self,
        name: str,
        run_fn: Callable[[IAlertSink], Awaitable[None]],
    ) -> None:
        """Restart a detector on unexpected return/exception, up to a cap."""
        restarts: list[float] = []
        sink: IAlertSink = self._detector_sinks.get(name, self._sink)
        while True:
            try:
                await run_fn(sink)
                _LOG.warning("scanner.detector.returned", detector=name)
            except asyncio.CancelledError:
                raise
            ...
```

(Leave the rest of the method body — restart counting, etc. — unchanged.)

- [ ] **Step 4.6: Start workers before TaskGroup, close in `aclose`**

Find the `run` method (around line 395-420). Just before the
`async with asyncio.TaskGroup() as tg:` line (around line 405), add:

```python
        for worker in self._workers:
            await worker.start()
```

Now find `Scheduler.aclose()` at line 709-721. Insert worker shutdown
**immediately after** `self._collectors_stop.set()` and **before**
`await self._renderer.stop()`. Rationale: by the time `aclose()` runs
the TaskGroup has already cancelled all detectors (so velocity is no
longer emitting); we want workers to drain remaining queued alerts
through the inner sink (which still needs the renderer + DB); only
then can we tear those down.

Final shape of `aclose`:

```python
    async def aclose(self) -> None:
        """Tear down sockets, HTTP clients, renderer, and DB. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._collectors_stop.set()
        for worker in self._workers:
            with contextlib.suppress(Exception):
                await worker.aclose()
        with contextlib.suppress(Exception):
            await self._renderer.stop()
        if self._owns_clients:
            await self._close_owned_clients()
        with contextlib.suppress(sqlite3.Error):
            self._db.close()
        _LOG.info("scanner.shutdown.complete")
```

- [ ] **Step 4.7: Smoke `pscanner run --once`**

```bash
rm -f data/pscanner.sqlite3
timeout 30 uv run pscanner run --once > /tmp/wiring-smoke.log 2>&1
echo "exit=$?"
grep -E "worker_sink\.|alert\.emitted" /tmp/wiring-smoke.log | head -20
```

Expected: `exit=0`. At least one `worker_sink.stats` line **may** appear
if the run lasted long enough (default `stats_interval_seconds=60`
won't fire in a 30s smoke; that's OK). At minimum verify there are no
exceptions about `WorkerSink` and that velocity alerts (if any) appear
under `alert.emitted` as before.

- [ ] **Step 4.8: Run the full test suite**

```bash
uv run pytest -q
```

Expected: all tests pass (likely 631+ now: 620 baseline + 11 new).

- [ ] **Step 4.9: Lint / format / type-check the whole repo**

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

- [ ] **Step 4.10: Commit**

```bash
git add src/pscanner/scheduler.py src/pscanner/detectors/velocity.py
git commit -m "feat(scheduler): wire WorkerSink for velocity detector

Velocity now receives a per-detector WorkerSink instead of the raw
AlertSink, decoupling its per-tick alert-emit work from the consumer
hot loop. Worker lifecycle (start before TaskGroup, aclose during
shutdown) is owned by the scheduler. Other detectors keep the raw sink.

Velocity's run/evaluate signatures change AlertSink -> IAlertSink. No
internal call-site changes — AlertSink and WorkerSink both satisfy the
Protocol structurally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 2h smoke validation

Manual validation step. Confirms the architectural fix landed against the
2026-04-27 baseline. Not a code task — just a procedure to follow and
metrics to record.

**No files modified.** This task has no commit at the end (unless the
smoke surfaces issues that need fixing).

- [ ] **Step 5.1: Reset DB and re-add Cavill watchlist**

```bash
rm -f data/pscanner.sqlite3
for a in 0x5cbd326a7f9dfac9855b9a23caee48fc097eabb0 0x53daff4663382b86808feb77e4fcaffd94e57cc8 \
         0x13b775f8a46762d031cbf9a6a478fe90a81e0aaf 0x7bfbc1e83ffb9203b29f653e5367acd3a580f6f8 \
         0xd5983aab43ef59620fda70599e30e693fd93c659 0x43d621fc31491eec23d9f696dcfb7e8923cd8ac9 \
         0xcbd11366479deef70576a4c7c0f6eda1bc6aed42 0xf04e089482c1349d3556a36951b033094731b79b \
         0x5266edffc8f4737c2b9d0fa959ecae2c7b55c8cb; do
  uv run pscanner watch "$a" --reason cavill-cluster-feb2026
done
```

- [ ] **Step 5.2: Run a 2h smoke (background)**

```bash
timeout 7200 uv run pscanner run > /tmp/smoke-2hr.log 2>&1; echo "exit=$?" >> /tmp/smoke-2hr.log
```

(Run in background or under a separate terminal. Will exit with 124 on
timeout, which is the expected normal completion.)

- [ ] **Step 5.3: Post-run analysis — primary metrics**

```bash
echo "=== queue-full warnings ==="
grep -c "tick_stream.subscriber_queue_full" /tmp/smoke-2hr.log

echo "=== worker_sink.stats events ==="
grep -c "worker_sink.stats" /tmp/smoke-2hr.log
grep "worker_sink.stats" /tmp/smoke-2hr.log | tail -5

echo "=== worker_sink.queue_full warnings ==="
grep -c "worker_sink.queue_full" /tmp/smoke-2hr.log

echo "=== exceptions ==="
grep -c "^Traceback" /tmp/smoke-2hr.log
```

Acceptance thresholds (from spec):
- `tick_stream.subscriber_queue_full`: **≤10/h** (was 498/h baseline → expect near-zero).
- `worker_sink.stats` events: ~120 over 2h (one per 60s).
- `worker_sink.queue_full`: **0** under default `velocity_maxsize=4096`.
  Non-zero is the new signal that the inner sink is the bottleneck — not
  a failure of this task, but worth flagging as a follow-up.
- `Traceback`: **0**.

- [ ] **Step 5.4: Post-run analysis — secondary metrics**

```bash
uv run python -c "
import sqlite3
con = sqlite3.connect('data/pscanner.sqlite3')
cur = con.cursor()
print('=== alerts by detector / severity ===')
for row in cur.execute('SELECT detector, severity, COUNT(*) FROM alerts GROUP BY detector, severity ORDER BY detector, severity'):
    print(f'{row[0]:25s} {row[1]:8s} {row[2]:>5d}')
"
```

Acceptance: alert counts on non-velocity detectors (cluster, smart_money,
mispricing, convergence, move_attribution, whales) should be in the same
range as the 2026-04-27 1h run scaled up by 2× — they're not on the new
path. Velocity's count should be similar to the 1h baseline scaled by 2×
(roughly 500 alerts).

- [ ] **Step 5.5: If thresholds met, push and close out**

```bash
git push origin main
```

Report a one-paragraph summary: queue-full delta, worker_sink stats
shape, anomalies if any. If thresholds are not met, file the issue
inline (don't fix in this task — the spec called this out as scope
for a follow-up).

---

## Self-review (notes to the executor)

### Spec coverage

- **`IAlertSink` Protocol** — Task 1.
- **`WorkerSink` class** — Task 3.
- **`AlertSink` unchanged** — verified at Task 1 step 1.4 (ty check).
- **Velocity opts in (annotation flip)** — Task 4 step 4.1.
- **Scheduler wires `WorkerSink` per detector** — Task 4 steps 4.3-4.6.
- **Per-detector sink map + workers list** — Task 4 step 4.3.
- **Lifecycle: start before TaskGroup, aclose in shutdown** — Task 4 step 4.6.
- **Backpressure: warn-then-block on full** — Task 3 step 3.4 + test 3.6.
- **Periodic `worker_sink.stats` event** — Task 3 step 3.4 + tests 3.8.
- **Counter reset between intervals** — Task 3 step 3.4 (`_drained = 0` at end of `_stats_loop` body) + test 3.8.
- **`emit` after `aclose` returns False with `closed_drop` log** — Task 3 step 3.4 + test 3.10.
- **`aclose` 5s drain timeout** — Task 3 step 3.4 + test 3.10.
- **Failed-drain logs with `worker_sink.drain_failed`, drain continues** — Task 3 step 3.4 + test 3.2.
- **Configurable `velocity_maxsize`, `stats_interval_seconds`** — Task 2.
- **2h smoke validation** — Task 5.

### Type / signature consistency

- `IAlertSink.emit(self, alert: Alert) -> bool` — Task 1, used in Tasks 3, 4.
- `WorkerSink(inner: IAlertSink, *, maxsize: int, name: str, stats_interval_seconds: float = 60.0, clock: Clock | None = None)` — Task 3, called in Task 4.
- `WorkerSink.start()`, `WorkerSink.aclose()` — Task 3, called in Task 4.
- `Scheduler._detector_sinks: dict[str, IAlertSink]`, `Scheduler._workers: list[WorkerSink]` — Task 4 step 4.3.
- Velocity's `run/evaluate` use `sink: IAlertSink` — Task 4 step 4.1.

### Placeholder scan

No `TBD` / `TODO` / "fill in details" anywhere. Every step has runnable
commands or complete code. Test code is concrete.

### Commit cadence

5 commits — one per task (Task 5 is a manual validation, no commit unless
issues surface).

---

## Out-of-plan follow-ups (not blocking)

- **Pace `snapshot_once` publisher.** Not in scope here (per spec
  non-goals). If `worker_sink.queue_full` shows up despite a 4096-deep
  queue, the publisher-side fix becomes worth doing.
- **Profile per-tick velocity work.** If `worker_sink.stats` reports
  sustained nonzero `blocking_emit_count`, the inner sink itself is
  slow — a velocity-internal profile (record_ms / window_ms / sink_ms
  histograms) is the next diagnostic step.
- **Wrap polling-driven detectors.** None today show burst-driven
  backpressure. Revisit only if a polling detector starts emitting at
  burst rates.
- **Per-detector `stats_interval`.** Single shared default suffices.
  Add per-detector tuning when a real divergence appears.
