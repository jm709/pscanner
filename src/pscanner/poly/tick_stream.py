"""Live tick-event publish/subscribe port for tick-consuming detectors.

The market tick collector publishes ``TickEvent`` objects to a ``TickStream``
on every successful tick write. Consumers subscribe via async iterator and
process each event. The default in-memory adapter (``BroadcastTickStream``)
fans out to every active subscriber via per-subscriber async queues; tests
inject a fake stream that yields canned events without involving the
collector at all.

Designed so a future second tick-consuming detector (e.g. depth-shock,
spread-widening) drops in alongside velocity by calling
``stream.subscribe()`` rather than reaching into collector internals.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import structlog

from pscanner.poly.ids import AssetId, ConditionId, EventSlug, MarketId

_LOG = structlog.get_logger(__name__)
_QUEUE_DEFAULT_MAXSIZE = 1024


@dataclass(frozen=True, slots=True)
class TickEvent:
    """One tick observation with optional market metadata enrichment.

    Carries the same numeric fields as :class:`MarketTick` plus the
    pre-resolved market metadata (title, condition_id, event_slug) the
    publisher had on hand at insert time. Detectors consume this single
    structure rather than calling back into collector helpers to enrich.
    """

    asset_id: AssetId
    snapshot_at: int
    mid_price: float | None
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    bid_depth_top5: float | None
    ask_depth_top5: float | None
    last_trade_price: float | None
    market_id: MarketId | None
    condition_id: ConditionId | None
    market_title: str | None
    event_slug: EventSlug | None


class TickStream(Protocol):
    """Subscribe-once-iterate-forever interface."""

    def subscribe(self) -> AsyncIterator[TickEvent]:
        """Return an async iterator yielding TickEvents until the consumer stops."""
        ...


class BroadcastTickStream:
    """In-memory fan-out stream. Each ``subscribe()`` returns a fresh async iterator.

    Slow subscribers see drops (queue full) rather than back-pressure into
    the publisher. ``maxsize`` per subscriber is configurable.
    """

    def __init__(self, *, maxsize: int = _QUEUE_DEFAULT_MAXSIZE) -> None:
        """Build an empty broadcaster.

        Args:
            maxsize: Per-subscriber queue capacity. When a subscriber's queue
                is full, ``publish`` drops the event for that subscriber and
                logs a warning instead of blocking.
        """
        self._maxsize = maxsize
        self._subs: list[asyncio.Queue[TickEvent]] = []

    async def publish(self, event: TickEvent) -> None:
        """Push ``event`` to every active subscriber (drop if its queue is full).

        Args:
            event: The tick event to fan out.
        """
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                _LOG.warning(
                    "tick_stream.subscriber_queue_full",
                    asset_id=event.asset_id,
                )

    def subscribe(self) -> AsyncGenerator[TickEvent]:
        """Return an async generator yielding TickEvents until the consumer stops.

        The subscriber's queue is registered synchronously before this method
        returns so any subsequent ``publish`` call sees the new subscriber,
        regardless of whether the caller has begun iterating.
        """
        q: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=self._maxsize)
        self._subs.append(q)
        return self._iterate(q)

    async def _iterate(self, q: asyncio.Queue[TickEvent]) -> AsyncGenerator[TickEvent]:
        """Yield events from ``q`` until the consumer stops; deregister on exit."""
        try:
            while True:
                event = await q.get()
                yield event
        finally:
            with suppress(ValueError):
                self._subs.remove(q)

    @property
    def subscriber_count(self) -> int:
        """Return the current number of active subscribers (for tests/metrics)."""
        return len(self._subs)
