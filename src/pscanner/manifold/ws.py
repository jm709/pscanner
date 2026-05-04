"""Async WebSocket consumer for the Manifold Markets global bet firehose.

Subscribes to ``global/new-bet`` and yields ``ManifoldBet`` objects as they
arrive. The ``websockets`` library handles ping/pong keepalive natively via
``ping_interval`` and ``ping_timeout`` constructor arguments.

Unknown topics (e.g. ``global/new-contract``) that arrive on the same connection
are silently skipped — the firehose may deliver events for other topics the
server decides to push without an explicit subscription request.

Example::

    async with ManifoldStream() as stream:
        async for bet in stream:
            print(bet.id, bet.amount)
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import structlog
from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from pscanner.manifold.models import ManifoldBet

_WS_URL = "wss://api.manifold.markets/ws"
_PING_INTERVAL_SECONDS = 45.0
_PING_TIMEOUT_SECONDS = 20.0
_GLOBAL_NEW_BET_TOPIC = "global/new-bet"

_LOG = structlog.get_logger(__name__)


class ManifoldStreamError(Exception):
    """Raised on fatal protocol-level errors from the Manifold WebSocket."""


class ManifoldStream:
    """Async context manager yielding ``ManifoldBet`` from the global firehose.

    Subscribes to ``global/new-bet`` on connect. The ``websockets`` library
    sends WebSocket-level ping frames every ``ping_interval`` seconds (default
    45 s) and raises ``ConnectionClosed`` if a pong isn't received within
    ``ping_timeout`` seconds (default 20 s).

    Unknown-topic messages that arrive on the socket are silently dropped,
    so callers don't need to handle unexpected event types.

    On connection loss the iteration raises ``ManifoldStreamError``; callers
    are responsible for wrapping the stream in their own reconnect loop. This
    is a deliberate divergence from the auto-reconnecting ``poly.clob_ws``
    pattern — the Manifold WS protocol's frame-loss-on-reconnect semantics
    make blind retry problematic, so we surface drops to let callers decide.
    """

    def __init__(
        self,
        *,
        url: str = _WS_URL,
        ping_interval: float = _PING_INTERVAL_SECONDS,
        ping_timeout: float = _PING_TIMEOUT_SECONDS,
    ) -> None:
        """Configure without opening a connection.

        Args:
            url: WebSocket URL (override for testing).
            ping_interval: Seconds between WebSocket-level pings.
            ping_timeout: Seconds to wait for a pong before treating the
                connection as dead.
        """
        self._url = url
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._ws: ClientConnection | None = None

    async def _open(self) -> None:
        """Open the WebSocket connection and send the subscription frame."""
        self._ws = await connect(
            self._url,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        await self._subscribe()
        _LOG.info("manifold_ws.connected", url=self._url)

    async def _subscribe(self) -> None:
        """Send the ``global/new-bet`` subscription frame."""
        if self._ws is None:
            return
        payload = json.dumps({"type": "subscribe", "channel": _GLOBAL_NEW_BET_TOPIC})
        await self._ws.send(payload)
        _LOG.debug("manifold_ws.subscribed", channel=_GLOBAL_NEW_BET_TOPIC)

    async def _close(self) -> None:
        """Close the WebSocket connection gracefully."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(ConnectionClosed, OSError):
                await ws.close()

    def __aiter__(self) -> AsyncIterator[ManifoldBet]:
        """Return self — this object IS the async iterator."""
        return self._iter_bets()

    async def _iter_bets(self) -> AsyncIterator[ManifoldBet]:
        """Yield ``ManifoldBet`` objects from the open WebSocket until closed."""
        if self._ws is None:
            raise ManifoldStreamError("ManifoldStream must be used as an async context manager")
        try:
            async for raw in self._ws:
                bet = _parse_bet_frame(raw)
                if bet is not None:
                    yield bet
        except ConnectionClosed as exc:
            rcvd = exc.rcvd
            raise ManifoldStreamError(
                f"manifold WebSocket closed unexpectedly: "
                f"code={rcvd.code if rcvd else None}, "
                f"reason={rcvd.reason if rcvd else None}"
            ) from exc

    async def __aenter__(self) -> Self:
        """Open the connection and return self."""
        await self._open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the connection."""
        await self._close()


def _parse_bet_frame(raw: str | bytes) -> ManifoldBet | None:
    """Parse one WebSocket frame into a ``ManifoldBet`` or ``None`` to skip.

    Frames that are not JSON, not dicts, not ``global/new-bet`` topic, or fail
    pydantic validation are silently dropped.

    Args:
        raw: Text or bytes frame from the WebSocket.

    Returns:
        A ``ManifoldBet`` if the frame carries a new-bet payload, else ``None``.
    """
    text = raw.decode() if isinstance(raw, bytes) else raw
    try:
        decoded: Any = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    topic = decoded.get("topic") or decoded.get("channel")
    if topic != _GLOBAL_NEW_BET_TOPIC:
        return None
    data = decoded.get("data")
    if not isinstance(data, dict):
        return None
    try:
        return ManifoldBet.model_validate(data)
    except ValidationError as exc:
        _LOG.warning("manifold_ws.bet_validation_failed", error=str(exc))
        return None
