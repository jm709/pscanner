"""Detector protocol that the scheduler runs concurrently."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pscanner.alerts.sink import AlertSink


@runtime_checkable
class Detector(Protocol):
    """A long-running coroutine that emits :class:`Alert` instances to a sink.

    Implementers are responsible for their own scheduling cadence; the
    orchestrator simply ``await``s :meth:`run` inside an ``asyncio.TaskGroup``.

    Attributes:
        name: Stable identifier used for logging and supervised restart.
    """

    name: str

    async def run(self, sink: AlertSink) -> None:
        """Run forever (or until cancelled) and publish alerts to ``sink``."""
        ...
