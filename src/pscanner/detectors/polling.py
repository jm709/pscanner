"""Base class for periodic-scan detectors (e.g. mispricing).

These detectors run a fixed-interval loop where each iteration calls a
subclass-defined ``_scan(sink)`` method and then sleeps. Failures inside
``_scan`` are logged and the loop continues; ``CancelledError`` propagates
cleanly. The interval is read from a subclass-defined hook so existing
detectors don't need a uniform config shape.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)


class PollingDetector(ABC):
    """Base for detectors that loop ``_scan`` on a fixed interval.

    Subclasses implement:

    * ``name: str`` — stable identifier used for logging.
    * ``async def _scan(self, sink: AlertSink) -> None`` — one iteration of
      the detector's work.
    * ``def _interval_seconds(self) -> float`` — sleep duration between
      iterations. Reads from per-detector config so different detectors can
      use different config shapes.

    The :meth:`run` loop catches and logs every non-cancellation exception so
    a transient API failure inside ``_scan`` doesn't tear down the detector.
    ``asyncio.CancelledError`` is re-raised so the orchestrating
    ``asyncio.TaskGroup`` shuts down cleanly.
    """

    name: str = ""

    def __init__(self, *, clock: Clock | None = None) -> None:
        """Store the injected clock (or a :class:`RealClock` default).

        Args:
            clock: Injectable :class:`Clock`. Defaults to :class:`RealClock`
                so production wiring needs no changes.
        """
        self._clock: Clock = clock if clock is not None else RealClock()

    @abstractmethod
    async def _scan(self, sink: AlertSink) -> None:
        """Run one iteration of the detector's work.

        Args:
            sink: Shared alert sink that the iteration emits to.
        """

    @abstractmethod
    def _interval_seconds(self) -> float:
        """Return the sleep duration between iterations, in seconds."""

    async def run(self, sink: AlertSink) -> None:
        """Loop ``_scan`` forever; logs and continues on any non-cancel error.

        Args:
            sink: Shared alert sink every iteration emits to.
        """
        while True:
            try:
                await self._scan(sink)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("polling.scan_failed", detector=self.name)
            await self._clock.sleep(self._interval_seconds())
