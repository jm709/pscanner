"""Collector protocol and concrete polling base.

Collectors mirror :class:`pscanner.detectors.base.Detector` in spirit but do
not emit alerts; their sole job is to persist data into the database. The
scheduler drives them inside an ``asyncio.TaskGroup`` and signals shutdown
via an ``asyncio.Event`` so each collector can flush state and exit cleanly.

:class:`Collector` is the structural Protocol every collector satisfies.
:class:`PollingCollector` is the concrete ABC for the dominant pattern:
``poll_once()`` on a fixed cadence with stop-aware sleep and per-iteration
exception swallowing. Subclass it and implement ``poll_once`` rather than
hand-rolling the loop.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import structlog

_LOG = structlog.get_logger(__name__)


@runtime_checkable
class Collector(Protocol):
    """A long-running coroutine that writes data and watches a stop event.

    Unlike :class:`pscanner.detectors.base.Detector`, collectors do not take
    an :class:`AlertSink`; they perform pure data persistence. Shutdown is
    cooperative: the scheduler sets ``stop_event`` and the implementation is
    expected to drain in-flight work and return.

    Attributes:
        name: Stable identifier used for logging and supervised restart.
    """

    name: str

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run until ``stop_event`` is set, persisting data as it arrives."""
        ...


class PollingCollector(ABC):
    """Concrete base for the periodic-poll-loop pattern.

    Subclasses set ``name`` and ``log_event_iteration_failed`` as class
    attributes, then implement :meth:`poll_once`. The base's :meth:`run`
    provides the loop, the per-iteration ``try/except``, and a stop-event-
    aware sleep between cycles. Pre-loop side-effects (e.g. registry
    subscription) go in :meth:`_on_start`, called once before the first poll.

    Wraps:
        - ``while not stop_event.is_set(): try: poll_once / except: log /
          wait_for(stop_event.wait(), timeout=interval)`` — return on stop,
          continue on timeout.

    See :class:`Collector` for the external contract this satisfies.
    """

    name: str
    log_event_iteration_failed: str

    def __init__(self, *, interval_seconds: float) -> None:
        """Bind the poll cadence.

        Args:
            interval_seconds: Wall-clock seconds between successive
                :meth:`poll_once` invocations.
        """
        self._interval_seconds = interval_seconds

    @abstractmethod
    async def poll_once(self) -> None:
        """One unit of work — read sources, persist results.

        Implementations should be idempotent so a retry on transient failure
        does not corrupt state.
        """
        ...

    async def _on_start(self) -> None:  # noqa: B027 — intentional default no-op
        """Pre-loop side-effects. Default no-op; override when needed."""

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run :meth:`poll_once` on the configured cadence until ``stop_event``.

        Per-iteration exceptions from :meth:`poll_once` are logged under
        ``log_event_iteration_failed`` and swallowed so a transient upstream
        hiccup does not kill the loop.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        await self._on_start()
        while not stop_event.is_set():
            try:
                await self.poll_once()
            except Exception:
                _LOG.exception(self.log_event_iteration_failed)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                continue
            return
