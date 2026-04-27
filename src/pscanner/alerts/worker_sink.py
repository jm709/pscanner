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
from pscanner.util.clock import Clock, RealClock

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
        """Build a queue-deferred sink wrapping ``inner``.

        Args:
            inner: The downstream :class:`IAlertSink` invoked by the drain
                task. Failures inside ``inner.emit`` are logged and
                swallowed so the drain loop survives transient errors.
            maxsize: Bounded queue capacity. ``emit`` falls through to a
                blocking ``put`` (with a ``worker_sink.queue_full`` warning)
                when the queue is full — a back-pressure signal rather
                than a silent drop.
            name: Stable identifier used in log fields and asyncio task
                names; usually the wrapped detector's name.
            stats_interval_seconds: Cadence for the periodic
                ``worker_sink.stats`` log event. Counters reset at each tick.
            clock: Injected :class:`Clock`; defaults to :class:`RealClock`.
        """
        self._inner = inner
        self._name = name
        self._stats_interval = stats_interval_seconds
        self._clock = clock if clock is not None else RealClock()
        self._queue: asyncio.Queue[Alert] = asyncio.Queue(maxsize=maxsize)
        self._drain_task: asyncio.Task[None] | None = None
        self._stats_task: asyncio.Task[None] | None = None
        self._closing = False
        self._drained = 0
        self._blocking_emits = 0
        self._depth_max = 0

    async def emit(self, alert: Alert) -> bool:
        """Enqueue ``alert`` for deferred delivery; return when queue accepts.

        Returns ``True`` once the alert is on the queue. Returns ``False``
        only when ``aclose()`` has been called — see :meth:`aclose`. The
        return is *not* the dedupe bool of the inner sink (which we only
        learn after the drain task runs); see the design spec for the
        rationale (no current tick-driven caller uses the bool).
        """
        if self._closing:
            _LOG.warning(
                "worker_sink.closed_drop",
                name=self._name,
                alert_key=alert.alert_key,
            )
            return False
        try:
            self._queue.put_nowait(alert)
        except asyncio.QueueFull:
            _LOG.warning(
                "worker_sink.queue_full",
                name=self._name,
                alert_key=alert.alert_key,
            )
            self._blocking_emits += 1
            await self._queue.put(alert)
        self._depth_max = max(self._depth_max, self._queue.qsize())
        return True

    async def start(self) -> None:
        """Launch the drain and stats background tasks. Idempotent."""
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(
                self._drain_loop(),
                name=f"worker_sink.drain[{self._name}]",
            )
        if self._stats_task is None:
            self._stats_task = asyncio.create_task(
                self._stats_loop(),
                name=f"worker_sink.stats[{self._name}]",
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
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None
        if self._stats_task is not None:
            self._stats_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stats_task
            self._stats_task = None

    async def _drain_loop(self) -> None:
        while True:
            alert = await self._queue.get()
            try:
                await self._handle_one(alert)
            finally:
                self._queue.task_done()

    async def _handle_one(self, alert: Alert) -> None:
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

    async def _stats_loop(self) -> None:
        while True:
            await self._clock.sleep(self._stats_interval)
            _LOG.info(
                "worker_sink.stats",
                name=self._name,
                queue_depth_max=self._depth_max,
                blocking_emit_count=self._blocking_emits,
                drain_count=self._drained,
            )
            self._depth_max = 0
            self._blocking_emits = 0
            self._drained = 0
