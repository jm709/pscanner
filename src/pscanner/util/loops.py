"""Shared periodic-loop helper used by polling-shaped loops.

The "while True / try work / except CancelledError raise / except Exception
log / sleep" pattern lives here as the single shared implementation;
:class:`PollingDetector.run` is the canonical caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import structlog

from pscanner.util.clock import Clock

_LOG = structlog.get_logger(__name__)


async def run_periodic(
    work: Callable[[], Awaitable[object]],
    *,
    interval_seconds: float | Callable[[], float],
    clock: Clock,
    log_event: str,
    log_fields: Mapping[str, object] | None = None,
) -> None:
    """Run ``work`` forever on ``interval_seconds``, surviving exceptions.

    ``asyncio.CancelledError`` is re-raised so the orchestrating
    :class:`asyncio.TaskGroup` can shut down cleanly. Every other
    exception is logged via ``_LOG.exception(log_event, **log_fields)``
    and the loop continues — one bad iteration cannot tear down the
    detector.

    Args:
        work: Zero-arg coroutine factory invoked once per iteration.
            Wrap closure state in a ``lambda`` when ``work`` needs
            arguments.
        interval_seconds: Sleep duration between iterations. Pass a
            ``float`` for a fixed interval, or a ``Callable[[], float]``
            (re-evaluated per cycle) when the interval may change at
            runtime — e.g. :meth:`PollingDetector._interval_seconds`.
        clock: Injected :class:`Clock` used for the sleep.
        log_event: structlog event name emitted when ``work`` raises
            a non-cancellation exception. Preserved per-caller for
            telemetry continuity.
        log_fields: Optional extra fields attached to the
            ``log_event`` exception log line.
    """
    fields = dict(log_fields) if log_fields is not None else {}
    while True:
        try:
            await work()
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOG.exception(log_event, **fields)
        if isinstance(interval_seconds, int | float):
            seconds = float(interval_seconds)
        else:
            seconds = interval_seconds()
        await clock.sleep(seconds)


__all__ = ["run_periodic"]
