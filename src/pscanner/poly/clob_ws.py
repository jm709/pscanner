"""Async websocket subscriber for ``wss://ws-subscriptions-clob.polymarket.com/ws/market``.

Wave 1 freezes only the public shape. Wave 2's ``clob-ws-client`` agent
implements: connection, ping every 10s, exponential-backoff reconnect, additive
``subscribe()``, and the MATCHED→CONFIRMED dedupe so consumers only see
CONFIRMED trades.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from types import TracebackType
from typing import Self

from pscanner.poly.models import WsBookMessage, WsTradeMessage

DEFAULT_CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


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
        raise NotImplementedError("Wave 2: clob-ws-client")

    async def connect(self) -> None:
        """Open the websocket. Idempotent — safe to call after a reconnect."""
        raise NotImplementedError("Wave 2: clob-ws-client")

    async def subscribe(self, asset_ids: Iterable[str]) -> None:
        """Add ``asset_ids`` to the active subscription set.

        Calls accumulate: the second call with a new set of ids unions them
        with the previously-subscribed ids.

        Args:
            asset_ids: Iterable of CLOB token ids.
        """
        raise NotImplementedError("Wave 2: clob-ws-client")

    async def messages(self) -> AsyncIterator[WsTradeMessage | WsBookMessage]:
        """Yield typed messages from the socket until ``close()`` is called.

        Trades are deduped MATCHED→CONFIRMED; only ``CONFIRMED`` trade messages
        are yielded. Book/price-change messages are passed through unchanged.

        Yields:
            ``WsTradeMessage`` or ``WsBookMessage`` instances.
        """
        raise NotImplementedError("Wave 2: clob-ws-client")

    async def close(self) -> None:
        """Close the websocket and cancel the keepalive task."""
        raise NotImplementedError("Wave 2: clob-ws-client")

    async def __aenter__(self) -> Self:
        """Async context-manager entry — calls :meth:`connect`."""
        raise NotImplementedError("Wave 2: clob-ws-client")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context-manager exit — calls :meth:`close`."""
        raise NotImplementedError("Wave 2: clob-ws-client")
