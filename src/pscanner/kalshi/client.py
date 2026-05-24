"""Async REST client for the Kalshi public API.

Wraps :class:`pscanner.util.http_client.RateLimitedHttpClient` for the
token-bucket + retry + Retry-After plumbing; the Kalshi-specific endpoint
methods (markets, orderbook, trades) are the public surface.

Public REST endpoints require no authentication. WebSocket streaming with
RSA-signed auth is deferred to Stage 2.

Base URL: ``https://api.elections.kalshi.com/trade-api/v2``
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Final, Self

from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import (
    KalshiMarket,
    KalshiMarketsPage,
    KalshiOrderbook,
    KalshiTrade,
    KalshiTradesPage,
)
from pscanner.util.http_client import RateLimitedHttpClient

_BASE_URL: Final[str] = "https://api.elections.kalshi.com/trade-api/v2"
_LOG_EVENT: Final[str] = "kalshi_http_retry"
_DEFAULT_RPM: Final[int] = 60
_DEFAULT_TIMEOUT: Final[float] = 30.0


class KalshiClient:
    """Async client for the Kalshi public REST API.

    The client is a long-lived singleton — open once, share across collectors,
    close on shutdown. Underlying httpx client + token bucket are created
    lazily on first use.

    Attributes:
        rpm: Requests-per-minute ceiling enforced by the token bucket.
        timeout_seconds: Per-request timeout passed to :mod:`httpx`.
    """

    def __init__(
        self,
        *,
        rpm: int = _DEFAULT_RPM,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        base_url: str = _BASE_URL,
    ) -> None:
        """Store config without opening any sockets.

        Args:
            rpm: Requests-per-minute budget (default 60).
            timeout_seconds: Default per-request timeout (default 30 s).
            base_url: Override the base URL (useful in tests).
        """
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._base_url = base_url
        self._inner = RateLimitedHttpClient(
            rpm=rpm,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            log_event=_LOG_EVENT,
        )

    async def _get(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """GET ``base_url + path`` with rate limiting and retries.

        Args:
            path: Path-only URL fragment (must start with ``/``).
            params: Optional query-string parameters.

        Returns:
            Parsed JSON object (dict).

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx, or after retries on 429/5xx.
            TypeError: If the response body is not a JSON object.
        """
        response = await self._inner.get(path, params=params)
        payload = response.json()
        if not isinstance(payload, dict):
            msg = f"expected JSON object from {path}, got {type(payload).__name__}"
            raise TypeError(msg)
        return payload  # type: ignore[return-value]

    async def get_markets(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiMarketsPage:
        """Fetch a page of markets from ``GET /markets``.

        Args:
            status: Filter by market status (e.g. ``"active"``, ``"closed"``).
            limit: Maximum markets to return (1-200, default 100).
            cursor: Pagination cursor from a previous response.

        Returns:
            A page of markets and the next cursor (empty string when exhausted).
        """
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        payload = await self._get("/markets", params=params)
        return KalshiMarketsPage.model_validate(payload)

    async def get_market(self, ticker: KalshiMarketTicker) -> KalshiMarket:
        """Fetch a single market by ticker from ``GET /markets/{ticker}``.

        Args:
            ticker: Kalshi market ticker (e.g. ``"KXELONMARS-99"``).

        Returns:
            The market detail.

        Raises:
            httpx.HTTPStatusError: On 404 or other non-retryable errors.
        """
        payload = await self._get(f"/markets/{ticker}")
        market_data = payload.get("market", payload)
        return KalshiMarket.model_validate(market_data)

    async def get_orderbook(self, ticker: KalshiMarketTicker) -> KalshiOrderbook:
        """Fetch the current orderbook from ``GET /markets/{ticker}/orderbook``.

        Args:
            ticker: Kalshi market ticker.

        Returns:
            The orderbook snapshot with YES and NO bid levels.

        Raises:
            httpx.HTTPStatusError: On 404 or other non-retryable errors.
        """
        payload = await self._get(f"/markets/{ticker}/orderbook")
        return KalshiOrderbook.model_validate(payload)

    async def get_market_trades(
        self,
        ticker: KalshiMarketTicker,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiTradesPage:
        """Fetch a page of trades from ``GET /markets/trades``.

        Args:
            ticker: Kalshi market ticker to filter by.
            limit: Maximum trades to return (default 100).
            cursor: Pagination cursor from a previous response.

        Returns:
            A page of trades and the next cursor.

        Note:
            The live Kalshi API returns trades via ``/markets/trades?ticker=TICKER``
            (global trades endpoint with a ticker filter), not via
            ``/markets/{ticker}/trades`` which returns 404.
        """
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        payload = await self._get("/markets/trades", params=params)
        return KalshiTradesPage.model_validate(payload)

    async def get_single_trade(self, trade_id: str) -> KalshiTrade:
        """Fetch a single trade by trade ID from ``GET /trades/{trade_id}``.

        Args:
            trade_id: UUID of the trade.

        Returns:
            The trade detail.
        """
        payload = await self._get(f"/trades/{trade_id}")
        trade_data = payload.get("trade", payload)
        return KalshiTrade.model_validate(trade_data)

    async def aclose(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient` and release sockets."""
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        """Async context-manager entry — returns ``self`` for the with-block."""
        await self._inner.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context-manager exit — calls :meth:`aclose`."""
        await self._inner.__aexit__(exc_type, exc, tb)
