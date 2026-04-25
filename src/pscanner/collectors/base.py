"""Collector protocol — long-running data-persistence loops.

Collectors mirror :class:`pscanner.detectors.base.Detector` in spirit but do
not emit alerts; their sole job is to persist data into the database. The
scheduler drives them inside an ``asyncio.TaskGroup`` and signals shutdown
via an ``asyncio.Event`` so each collector can flush state and exit cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable


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
