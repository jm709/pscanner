"""Tests for the publish/subscribe tick-stream primitives."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pscanner.poly.ids import AssetId
from pscanner.poly.tick_stream import BroadcastTickStream, TickEvent


def _event(*, asset: str = "A1", snapshot_at: int = 100, mid: float = 0.5) -> TickEvent:
    """Build a minimal ``TickEvent`` for tests."""
    return TickEvent(
        asset_id=AssetId(asset),
        snapshot_at=snapshot_at,
        mid_price=mid,
        best_bid=mid - 0.01,
        best_ask=mid + 0.01,
        spread=0.02,
        bid_depth_top5=1000.0,
        ask_depth_top5=1000.0,
        last_trade_price=mid,
        market_id=None,
        condition_id=None,
        market_title=None,
        event_slug=None,
    )


async def _collect_iter(iterator: AsyncIterator[TickEvent], count: int) -> list[TickEvent]:
    """Drain ``count`` events from a pre-subscribed iterator."""
    out: list[TickEvent] = []
    async for event in iterator:
        out.append(event)
        if len(out) >= count:
            break
    return out


async def test_single_subscriber_receives_events_in_order() -> None:
    stream = BroadcastTickStream()
    events = [_event(snapshot_at=i, mid=0.40 + i * 0.01) for i in range(3)]

    iterator = stream.subscribe()  # registers the subscriber synchronously
    for event in events:
        await stream.publish(event)
    received: list[TickEvent] = []
    for _ in range(3):
        received.append(await asyncio.wait_for(iterator.__anext__(), timeout=1.0))
    await iterator.aclose()

    assert [e.snapshot_at for e in received] == [0, 1, 2]
    assert [e.mid_price for e in received] == [pytest.approx(0.40 + i * 0.01) for i in range(3)]


async def test_multiple_subscribers_each_get_full_stream() -> None:
    stream = BroadcastTickStream()
    events = [_event(snapshot_at=i) for i in range(3)]

    # Register all three subscribers before publishing.
    iterators = [stream.subscribe() for _ in range(3)]
    assert stream.subscriber_count == 3
    consumers = [asyncio.create_task(_collect_iter(it, 3)) for it in iterators]
    for event in events:
        await stream.publish(event)

    results = await asyncio.wait_for(asyncio.gather(*consumers), timeout=1.0)
    assert len(results) == 3
    for received in results:
        assert [e.snapshot_at for e in received] == [0, 1, 2]


async def test_late_subscriber_misses_prior_events() -> None:
    stream = BroadcastTickStream()
    early = _event(snapshot_at=1)
    await stream.publish(early)  # No subscribers yet — event is dropped.

    iterator = stream.subscribe()
    later = _event(snapshot_at=2)
    await stream.publish(later)
    received = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
    await iterator.aclose()

    assert received.snapshot_at == 2


async def test_slow_subscriber_drops_overflow_without_blocking_publisher() -> None:
    stream = BroadcastTickStream(maxsize=2)
    iterator = stream.subscribe()
    assert stream.subscriber_count == 1
    # Publisher fills the queue (maxsize=2) and then overflows; should not raise.
    await stream.publish(_event(snapshot_at=0))
    await stream.publish(_event(snapshot_at=1))
    await stream.publish(_event(snapshot_at=2))  # dropped (queue full)
    await stream.publish(_event(snapshot_at=3))  # dropped (queue full)

    first = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
    second = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
    assert first.snapshot_at == 0
    assert second.snapshot_at == 1
    # No third event is queued.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(iterator.__anext__(), timeout=0.05)
    await iterator.aclose()


async def test_cancelled_subscriber_is_removed_cleanly() -> None:
    stream = BroadcastTickStream()
    iterator = stream.subscribe()
    assert stream.subscriber_count == 1
    task = asyncio.create_task(iterator.__anext__())
    # Yield once so the task starts awaiting the queue.
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await iterator.aclose()
    assert stream.subscriber_count == 0
