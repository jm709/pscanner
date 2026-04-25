"""Behavior tests for ``pscanner.poly.clob_ws.MarketWebSocket``.

A real ``websockets.serve`` is spun up per test on a random localhost port so
we exercise the actual library code paths (handshake, frames, close codes).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.models import WsBookMessage, WsTradeMessage

ServerHandler = Callable[[ServerConnection], Awaitable[None]]


class ServerState:
    """Shared in-memory state captured by tests' websocket handlers."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.connections: int = 0
        self.outbound: asyncio.Queue[str | None] = asyncio.Queue()
        self.close_after_first: bool = False
        self.first_handler_finished: asyncio.Event = asyncio.Event()


async def _drain_inbound(ws: ServerConnection, state: ServerState) -> None:
    """Append every text frame the client sends into ``state.received``."""
    try:
        async for message in ws:
            text = message.decode() if isinstance(message, bytes) else message
            state.received.append(text)
    except Exception:
        return


async def _pump_outbound(ws: ServerConnection, state: ServerState) -> None:
    """Forward queued outbound payloads to the connected client."""
    while True:
        item = await state.outbound.get()
        if item is None:
            return
        try:
            await ws.send(item)
        except Exception:
            return


def _make_handler(state: ServerState) -> ServerHandler:
    """Construct a websocket handler bound to ``state``."""

    async def handler(ws: ServerConnection) -> None:
        state.connections += 1
        is_first = state.connections == 1
        inbound = asyncio.create_task(_drain_inbound(ws, state))
        outbound = asyncio.create_task(_pump_outbound(ws, state))
        try:
            if is_first and state.close_after_first:
                await asyncio.sleep(0.1)
                await ws.close(code=1011, reason="server closing for test")
                return
            await asyncio.gather(inbound, outbound)
        finally:
            inbound.cancel()
            outbound.cancel()
            if is_first:
                state.first_handler_finished.set()

    return handler


@pytest.fixture
async def server() -> AsyncIterator[tuple[str, ServerState]]:
    """Start a websockets server on a random port and yield ``(url, state)``."""
    state = ServerState()
    handler = _make_handler(state)
    async with serve(handler, "127.0.0.1", 0) as server_obj:
        sock = next(iter(server_obj.sockets))
        port = sock.getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        yield url, state
        await state.outbound.put(None)


async def _next_message(
    client: MarketWebSocket,
    deadline_seconds: float = 2.0,
) -> WsTradeMessage | WsBookMessage:
    """Pull the next typed message off the client with a deadline."""
    iterator = client.messages()
    return await asyncio.wait_for(iterator.__anext__(), timeout=deadline_seconds)


async def _send_and_receive(
    client: MarketWebSocket,
    state: ServerState,
    payload: dict[str, Any],
) -> WsTradeMessage | WsBookMessage:
    """Push ``payload`` from the server, return the next yielded client message."""
    await state.outbound.put(json.dumps(payload))
    return await _next_message(client)


async def test_subscribe_sends_expected_payload(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    client = MarketWebSocket(url=url, ping_interval_seconds=10.0)
    async with client:
        await client.subscribe(["a", "b"])
        await asyncio.sleep(0.1)
    subscriptions = [r for r in state.received if r != "PING"]
    assert subscriptions, "expected at least one subscription frame"
    decoded = json.loads(subscriptions[0])
    assert decoded["type"] == "market"
    assert sorted(decoded["assets_ids"]) == ["a", "b"]


async def test_ping_sent_at_configured_interval(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    client = MarketWebSocket(url=url, ping_interval_seconds=0.1)
    async with client:
        await asyncio.sleep(0.5)
    pings = [r for r in state.received if r == "PING"]
    assert len(pings) >= 3


async def test_matched_trade_dropped_confirmed_yielded(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    client = MarketWebSocket(url=url, ping_interval_seconds=10.0)
    matched = {
        "event_type": "trade",
        "condition_id": "0xc",
        "asset_id": "tok",
        "side": "BUY",
        "size": 10.0,
        "price": 0.5,
        "taker_proxy": "0xabc",
        "status": "MATCHED",
        "transaction_hash": "0xdead",
        "timestamp": 1,
    }
    confirmed = {**matched, "status": "CONFIRMED"}
    async with client:
        await state.outbound.put(json.dumps(matched))
        await state.outbound.put(json.dumps(confirmed))
        msg = await _next_message(client)
    assert isinstance(msg, WsTradeMessage)
    assert msg.status == "CONFIRMED"
    assert msg.transaction_hash == "0xdead"


async def test_book_message_yielded_as_book(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    client = MarketWebSocket(url=url, ping_interval_seconds=10.0)
    book_payload = {
        "event_type": "book",
        "asset_id": "tok",
        "data": {"bids": [["0.41", "10"]], "asks": [["0.43", "5"]]},
    }
    async with client:
        msg = await _send_and_receive(client, state, book_payload)
    assert isinstance(msg, WsBookMessage)
    assert msg.event_type == "book"
    assert msg.data["bids"] == [["0.41", "10"]]


async def test_reconnect_restores_subscription(
    server: tuple[str, ServerState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, state = server
    state.close_after_first = True

    async def fast_sleep(seconds: float) -> None:
        await asyncio.sleep(min(seconds, 0.05))

    monkeypatch.setattr(MarketWebSocket, "_sleep_backoff", staticmethod(fast_sleep))

    client = MarketWebSocket(url=url, ping_interval_seconds=10.0)
    await client.connect()
    await client.subscribe(["a", "b"])
    try:
        book = {
            "event_type": "book",
            "asset_id": "tok",
            "data": {},
        }

        async def push_after_reconnect() -> None:
            await state.first_handler_finished.wait()
            for _ in range(50):
                if state.connections >= 2:
                    break
                await asyncio.sleep(0.05)
            await state.outbound.put(json.dumps(book))

        pusher = asyncio.create_task(push_after_reconnect())
        msg = await asyncio.wait_for(client.messages().__anext__(), timeout=5.0)
        await pusher
    finally:
        await client.close()

    assert isinstance(msg, WsBookMessage)
    assert state.connections >= 2
    subscriptions = [
        json.loads(r) for r in state.received if r and r.startswith("{") and "type" in r
    ]
    assert any(
        sub.get("type") == "market" and sorted(sub.get("assets_ids", [])) == ["a", "b"]
        for sub in subscriptions[1:]
    ), "expected subscription replayed after reconnect"


async def test_close_cancels_ping_task_cleanly(
    server: tuple[str, ServerState],
) -> None:
    url, _ = server
    client = MarketWebSocket(url=url, ping_interval_seconds=0.05)
    await client.connect()
    await asyncio.sleep(0.15)
    ping_task = client._ping_task
    assert ping_task is not None
    await client.close()
    assert ping_task.done()
    assert client._ping_task is None
    # Second close must be idempotent and not raise.
    await client.close()


async def test_async_context_manager_connects_and_closes(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    client = MarketWebSocket(url=url, ping_interval_seconds=10.0)
    async with client as entered:
        assert entered is client
        await client.subscribe(["only"])
        await asyncio.sleep(0.1)
    assert state.connections == 1
    assert any(r != "PING" for r in state.received)
