"""Tests for ``pscanner.manifold.ws.ManifoldStream``.

Uses a real ``websockets.asyncio.server.serve`` on a random localhost port —
same pattern as ``tests/poly/test_clob_ws.py`` — so we exercise the actual
library code paths.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from websockets.asyncio.server import ServerConnection, serve

from pscanner.manifold.ws import ManifoldStream, ManifoldStreamError

ServerHandler = Callable[[ServerConnection], Awaitable[None]]

_BET_DATA = {
    "id": "betXYZ",
    "userId": "userXYZ",
    "contractId": "mkABC123",
    "outcome": "YES",
    "amount": 25.0,
    "probBefore": 0.49,
    "probAfter": 0.51,
    "createdTime": 1_714_000_000_000,
    "isFilled": True,
    "isCancelled": False,
    "limitProb": None,
}


class ServerState:
    """Shared in-memory state captured by test server handlers."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.outbound: asyncio.Queue[str | None] = asyncio.Queue()
        self.connections: int = 0


async def _drain_inbound(ws: ServerConnection, state: ServerState) -> None:
    try:
        async for message in ws:
            text = message.decode() if isinstance(message, bytes) else message
            state.received.append(text)
    except Exception:
        return


async def _pump_outbound(ws: ServerConnection, state: ServerState) -> None:
    while True:
        item = await state.outbound.get()
        if item is None:
            return
        try:
            await ws.send(item)
        except Exception:
            return


def _make_handler(state: ServerState) -> ServerHandler:
    async def handler(ws: ServerConnection) -> None:
        state.connections += 1
        inbound = asyncio.create_task(_drain_inbound(ws, state))
        outbound = asyncio.create_task(_pump_outbound(ws, state))
        try:
            await asyncio.gather(inbound, outbound)
        finally:
            inbound.cancel()
            outbound.cancel()

    return handler


@pytest.fixture
async def server() -> AsyncIterator[tuple[str, ServerState]]:
    """Start a websockets server on a random port, yield ``(url, state)``."""
    state = ServerState()
    handler = _make_handler(state)
    # Use small ping params for the server side to not interfere with client
    async with serve(handler, "127.0.0.1", 0) as server_obj:
        sock = next(iter(server_obj.sockets))
        port = sock.getsockname()[1]
        url = f"ws://127.0.0.1:{port}"
        yield url, state
        await state.outbound.put(None)


def _new_bet_frame(data: dict) -> str:
    """Wrap a bet dict in a ``global/new-bet`` frame."""
    return json.dumps({"topic": "global/new-bet", "data": data})


async def test_stream_yields_bet_from_new_bet_frame(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    await state.outbound.put(_new_bet_frame(_BET_DATA))
    async with stream:
        bet = await asyncio.wait_for(stream.__aiter__().__anext__(), timeout=2.0)
    assert bet.id == "betXYZ"
    assert bet.outcome == "YES"
    assert bet.amount == 25.0


async def test_stream_sends_subscription_on_connect(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    async with stream:
        await asyncio.sleep(0.1)
    assert any("global/new-bet" in msg for msg in state.received)


async def test_stream_skips_unknown_topic(
    server: tuple[str, ServerState],
) -> None:
    """Frames for other topics must not be yielded."""
    url, state = server
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    other_frame = json.dumps({"topic": "global/new-contract", "data": {"id": "c1"}})
    valid_frame = _new_bet_frame(_BET_DATA)
    await state.outbound.put(other_frame)
    await state.outbound.put(valid_frame)
    async with stream:
        bet = await asyncio.wait_for(stream.__aiter__().__anext__(), timeout=2.0)
    # The unknown topic frame was skipped; we got the valid bet.
    assert bet.id == "betXYZ"


async def test_stream_skips_invalid_json(
    server: tuple[str, ServerState],
) -> None:
    """Non-JSON frames must be silently skipped."""
    url, state = server
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    await state.outbound.put("not-json!!!")
    await state.outbound.put(_new_bet_frame(_BET_DATA))
    async with stream:
        bet = await asyncio.wait_for(stream.__aiter__().__anext__(), timeout=2.0)
    assert bet.id == "betXYZ"


async def test_stream_skips_frame_missing_data(
    server: tuple[str, ServerState],
) -> None:
    """A ``global/new-bet`` frame without ``data`` key must be skipped."""
    url, state = server
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    bad_frame = json.dumps({"topic": "global/new-bet"})
    await state.outbound.put(bad_frame)
    await state.outbound.put(_new_bet_frame(_BET_DATA))
    async with stream:
        bet = await asyncio.wait_for(stream.__aiter__().__anext__(), timeout=2.0)
    assert bet.id == "betXYZ"


async def test_stream_multiple_bets(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    bet2 = {**_BET_DATA, "id": "betABC", "amount": 50.0}
    await state.outbound.put(_new_bet_frame(_BET_DATA))
    await state.outbound.put(_new_bet_frame(bet2))
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    collected: list[str] = []
    async with stream:
        it = stream.__aiter__()
        b1 = await asyncio.wait_for(it.__anext__(), timeout=2.0)
        b2 = await asyncio.wait_for(it.__anext__(), timeout=2.0)
        collected.extend([b1.id, b2.id])
    assert set(collected) == {"betXYZ", "betABC"}


async def test_stream_raises_manifold_stream_error_on_server_close(
    server: tuple[str, ServerState],
) -> None:
    """Server closing the connection mid-stream raises ``ManifoldStreamError``."""
    _url, state = server

    async def closing_handler(ws: ServerConnection) -> None:
        state.connections += 1
        await asyncio.sleep(0.05)
        await ws.close(code=1011, reason="server going away")

    # Override: spin up a second server that immediately closes.
    async with serve(closing_handler, "127.0.0.1", 0) as s:
        sock2 = next(iter(s.sockets))
        port2 = sock2.getsockname()[1]
        close_url = f"ws://127.0.0.1:{port2}"

        stream = ManifoldStream(url=close_url, ping_interval=600.0, ping_timeout=30.0)
        with pytest.raises(ManifoldStreamError):
            async with stream:
                await asyncio.wait_for(stream.__aiter__().__anext__(), timeout=2.0)


async def test_stream_used_outside_context_manager_raises() -> None:
    stream = ManifoldStream(url="ws://127.0.0.1:9999")
    with pytest.raises(ManifoldStreamError, match="context manager"):
        await stream.__aiter__().__anext__()


async def test_stream_connection_count(
    server: tuple[str, ServerState],
) -> None:
    url, state = server
    stream = ManifoldStream(url=url, ping_interval=600.0, ping_timeout=30.0)
    async with stream:
        await asyncio.sleep(0.05)
    assert state.connections == 1
