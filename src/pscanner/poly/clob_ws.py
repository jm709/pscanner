"""Async websocket subscriber for ``wss://ws-subscriptions-clob.polymarket.com/ws/market``.

Implements connection management, ping keep-alive, additive subscription,
exponential-backoff reconnect, and the MATCHED→CONFIRMED dedupe so consumers
only see CONFIRMED trades.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterable
from types import TracebackType
from typing import Any, Self

import structlog
from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from pscanner.poly.models import WsBookMessage, WsTradeMessage

DEFAULT_CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_PING_TEXT = "PING"
_SUBSCRIBE_TYPE = "market"
_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_BOOK_EVENT_TYPES = {"book", "price_change", "tick_size_change", "last_trade_price"}

_log = structlog.get_logger(__name__)


class MarketWebSocket:
    """Async client for the Polymarket CLOB market-data websocket.

    Use as an async context manager. ``subscribe`` is additive — calling it
    multiple times accumulates the asset-id set; the implementation re-sends
    the full subscription message on each reconnect.
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_CLOB_WS_URL,
        ping_interval_seconds: float = 10.0,
    ) -> None:
        """Configure the client without opening the socket.

        Args:
            url: Websocket URL (override only for testing).
            ping_interval_seconds: How often to send PING frames.
        """
        self._url = url
        self._ping_interval_seconds = ping_interval_seconds
        self._subscribed: set[str] = set()
        self._ws: ClientConnection | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the websocket. Idempotent — safe to call after a reconnect."""
        async with self._lock:
            if self._ws is not None:
                return
            self._ws = await connect(self._url)
            _log.info("clob_ws.connected", url=self._url)
            if self._subscribed:
                await self._send_subscription(sorted(self._subscribed))
            self._start_ping_task()

    async def subscribe(self, asset_ids: Iterable[str]) -> None:
        """Add ``asset_ids`` to the active subscription set.

        Calls accumulate: the second call with a new set of ids unions them
        with the previously-subscribed ids.

        Args:
            asset_ids: Iterable of CLOB token ids.
        """
        new_ids = {aid for aid in asset_ids if aid not in self._subscribed}
        if not new_ids:
            return
        self._subscribed.update(new_ids)
        if self._ws is not None:
            await self._send_subscription(sorted(new_ids))

    async def messages(self) -> AsyncIterator[WsTradeMessage | WsBookMessage]:
        """Yield typed messages from the socket until ``close()`` is called.

        Trades are filtered to ``CONFIRMED`` only — ``MATCHED`` events are
        dropped so consumers never see duplicates per fill. Book/price-change
        messages are passed through unchanged.

        Yields:
            ``WsTradeMessage`` or ``WsBookMessage`` instances.
        """
        backoff = _INITIAL_BACKOFF_SECONDS
        while True:
            if self._ws is None:
                await self.connect()
            assert self._ws is not None  # noqa: S101 -- type narrowing for type checker
            try:
                async for raw in self._ws:
                    parsed = _parse_message(raw)
                    if parsed is None:
                        continue
                    yield parsed
                    backoff = _INITIAL_BACKOFF_SECONDS
                return
            except ConnectionClosed as exc:
                rcvd = exc.rcvd
                _log.warning(
                    "clob_ws.disconnected",
                    code=rcvd.code if rcvd is not None else None,
                    reason=rcvd.reason if rcvd is not None else None,
                )
                await self._teardown_connection()
                await self._sleep_backoff(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_SECONDS)

    async def close(self) -> None:
        """Close the websocket and cancel the keepalive task. Idempotent."""
        await self._teardown_connection()

    async def __aenter__(self) -> Self:
        """Async context-manager entry — calls :meth:`connect`."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context-manager exit — calls :meth:`close`."""
        await self.close()

    async def _send_subscription(self, asset_ids: list[str]) -> None:
        """Send a subscription frame for ``asset_ids`` on the open socket."""
        if self._ws is None:
            return
        payload = json.dumps({"type": _SUBSCRIBE_TYPE, "assets_ids": asset_ids})
        await self._ws.send(payload)
        _log.debug("clob_ws.subscribe", count=len(asset_ids))

    def _start_ping_task(self) -> None:
        """Spawn the keep-alive ping task if it isn't already running."""
        if self._ping_task is not None and not self._ping_task.done():
            return
        self._ping_task = asyncio.create_task(self._ping_loop())

    async def _ping_loop(self) -> None:
        """Send raw ``PING`` text every ``ping_interval_seconds`` until closed."""
        while True:
            await self._sleep_backoff(self._ping_interval_seconds)
            ws = self._ws
            if ws is None:
                return
            try:
                await ws.send(_PING_TEXT)
            except ConnectionClosed:
                return

    async def _teardown_connection(self) -> None:
        """Cancel the ping task and close the socket. Safe to call repeatedly."""
        async with self._lock:
            ping_task = self._ping_task
            self._ping_task = None
            ws = self._ws
            self._ws = None
        if ping_task is not None and not ping_task.done():
            ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ping_task
        if ws is not None:
            with contextlib.suppress(ConnectionClosed, OSError):
                await ws.close()

    @staticmethod
    async def _sleep_backoff(seconds: float) -> None:
        """Indirection point so tests can monkey-patch sleep durations."""
        await asyncio.sleep(seconds)


def _parse_message(raw: str | bytes) -> WsTradeMessage | WsBookMessage | None:
    """Parse one wire frame into a typed message, dropping MATCHED trades.

    Returns ``None`` for frames that aren't valid JSON, aren't dicts, or are
    ``MATCHED`` trade events (deduped — only ``CONFIRMED`` is yielded).

    Args:
        raw: Bytes or text frame received from the socket.

    Returns:
        Typed message or ``None`` to skip.
    """
    text = raw.decode() if isinstance(raw, bytes) else raw
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    return _route_event(decoded)


def _route_event(decoded: dict[str, Any]) -> WsTradeMessage | WsBookMessage | None:
    """Dispatch a decoded JSON dict to the correct typed model."""
    event_type = decoded.get("event_type")
    if event_type == "trade":
        if decoded.get("status") != "CONFIRMED":
            return None
        try:
            return WsTradeMessage.model_validate(decoded)
        except ValidationError as exc:
            _log.warning("clob_ws.trade_validation_failed", error=str(exc))
            return None
    if event_type in _BOOK_EVENT_TYPES:
        try:
            return WsBookMessage.model_validate(decoded)
        except ValidationError as exc:
            _log.warning("clob_ws.book_validation_failed", error=str(exc))
            return None
    return None
