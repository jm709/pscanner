"""Clock dependency for testable time-based loops.

Production code uses :class:`RealClock` (delegates to :func:`asyncio.sleep`
and :func:`time.time`). Tests inject :class:`FakeClock` and call
:meth:`FakeClock.advance` to release pending sleepers without mutating the
global :mod:`asyncio` module — which historically deadlocked the suite when
sibling detectors ran ``while True: await asyncio.sleep(...)`` loops.
"""

from __future__ import annotations

import asyncio
import heapq
import time as _time
from typing import Protocol


class Clock(Protocol):
    """Time and sleep interface used by every long-running loop.

    Implementations only need a ``sleep`` coroutine and a sync ``now``
    accessor. The :class:`Detector` and supervisor wiring is kept narrow
    on purpose so a future implementation (e.g. ``trio``-backed) can be
    swapped in without touching call sites.
    """

    async def sleep(self, seconds: float) -> None:
        """Suspend the calling coroutine for at least ``seconds`` seconds."""
        ...

    def now(self) -> float:
        """Return the current wall-clock time in seconds since the epoch."""
        ...


class RealClock:
    """Production :class:`Clock` — delegates to :mod:`asyncio` and :mod:`time`."""

    async def sleep(self, seconds: float) -> None:
        """Delegate to :func:`asyncio.sleep`.

        Args:
            seconds: Suspension duration in seconds. Negative or zero values
                yield to the event loop without scheduling a timer.
        """
        await asyncio.sleep(seconds)

    def now(self) -> float:
        """Return :func:`time.time` — wall-clock seconds since the epoch."""
        return _time.time()


class FakeClock:
    """In-memory :class:`Clock` for tests; :meth:`advance` releases sleepers.

    Sleepers are parked on a min-heap keyed by their wake-up deadline. A
    monotonic sequence counter breaks ties so the heap stays totally
    ordered even when two sleepers share a deadline. :meth:`advance` resolves
    every sleeper whose deadline is reached and yields cooperatively a few
    times so the released coroutines actually execute before control
    returns to the caller.
    """

    def __init__(self, *, start: float = 0.0) -> None:
        """Build the fake clock.

        Args:
            start: Initial value returned by :meth:`now`. Defaults to ``0.0``.
        """
        self._now = start
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0

    def now(self) -> float:
        """Return the simulated current time."""
        return self._now

    async def sleep(self, seconds: float) -> None:
        """Suspend until :meth:`advance` reaches the wake-up deadline.

        Non-positive durations yield cooperatively without queueing — this
        matches the ``await asyncio.sleep(0)`` "yield to event loop" idiom
        we rely on across the codebase.

        Args:
            seconds: Duration to sleep, in simulated seconds.
        """
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        wake_at = self._now + seconds
        self._seq += 1
        heapq.heappush(self._sleepers, (wake_at, self._seq, future))
        try:
            await future
        except asyncio.CancelledError:
            self._sleepers = [s for s in self._sleepers if s[2] is not future]
            heapq.heapify(self._sleepers)
            raise

    async def advance(self, seconds: float) -> None:
        """Move time forward; resolve any sleepers whose deadline is reached.

        After resolving, yield repeatedly with ``asyncio.sleep(0)`` so the
        released coroutines actually run before this method returns.

        Args:
            seconds: Non-negative simulated duration to advance.
        """
        target = self._now + seconds
        while self._sleepers and self._sleepers[0][0] <= target:
            _wake, _seq, future = heapq.heappop(self._sleepers)
            if not future.done():
                future.set_result(None)
        self._now = target
        for _ in range(8):
            await asyncio.sleep(0)
