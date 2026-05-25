"""Shared sync→async dispatch helper for callback-driven detectors.

Detectors with synchronous callback entry points (e.g. trade-collector
``subscribe_new_trade`` callbacks, alert ``subscribe`` subscribers) need
to spawn coroutines as fire-and-forget tasks while still holding a
reference to each task so the GC doesn't collect it mid-flight.

The same three-step idiom (``try get_running_loop``, ``create_task``,
``add + add_done_callback``) is shared by every callback-driven
consumer; this module hosts the one implementation so consumers
compose an :class:`AsyncDispatcher` instead of re-implementing it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog

_LOG = structlog.get_logger(__name__)


class AsyncDispatcher:
    """Spawn coroutines as tracked fire-and-forget tasks.

    Each dispatcher owns its own pending-task set. When there's no
    running event loop (e.g. test setup that hasn't started one yet),
    :meth:`spawn` logs the configured ``log_event_no_loop`` event at
    ``DEBUG`` and returns silently — the un-awaited coroutine is closed
    so the call doesn't leak.
    """

    def __init__(self, *, log_event_no_loop: str) -> None:
        """Build a dispatcher with a stable no-loop log event name.

        Args:
            log_event_no_loop: structlog event name emitted when
                :meth:`spawn` is called without a running event loop.
                Preserved per-caller for telemetry continuity.
        """
        self._tasks: set[asyncio.Task[None]] = set()
        self._log_event_no_loop = log_event_no_loop

    def spawn(
        self,
        coro: Coroutine[Any, Any, None],
        /,
        **log_fields: object,
    ) -> None:
        """Schedule ``coro`` on the running event loop.

        Args:
            coro: The coroutine to spawn. Closed (not awaited) when
                no event loop is running.
            **log_fields: Additional fields attached to the
                ``log_event_no_loop`` debug event when the no-loop
                branch fires.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug(self._log_event_no_loop, **log_fields)
            coro.close()
            return
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @property
    def pending(self) -> set[asyncio.Task[None]]:
        """Live view of the in-flight task set (for tests and aclose())."""
        return self._tasks


__all__ = ["AsyncDispatcher"]
