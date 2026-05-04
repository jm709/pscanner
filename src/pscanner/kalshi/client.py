"""Async REST client for the Kalshi public API.

Wraps ``httpx.AsyncClient`` with a token-bucket rate limiter and ``tenacity``
retries honouring ``Retry-After`` on 429/5xx, mirroring the shape of
:class:`pscanner.poly.http.PolyHttpClient`.

Public REST endpoints require no authentication. WebSocket streaming with
RSA-signed auth is deferred to Stage 2.

Base URL: ``https://api.elections.kalshi.com/trade-api/v2``
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Final, Self

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import (
    KalshiMarket,
    KalshiMarketsPage,
    KalshiOrderbook,
    KalshiTrade,
    KalshiTradesPage,
)

_LOG = structlog.get_logger(__name__)

_BASE_URL: Final[str] = "https://api.elections.kalshi.com/trade-api/v2"
_USER_AGENT: Final[str] = "pscanner/0.1"

_STATUS_TOO_MANY_REQUESTS: Final[int] = 429
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({_STATUS_TOO_MANY_REQUESTS, 502, 503, 504})
_RETRYABLE_TRANSPORT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
_MAX_ATTEMPTS: Final[int] = 5
_BACKOFF_MIN_SECONDS: Final[float] = 1.0
_BACKOFF_MAX_SECONDS: Final[float] = 30.0

_DEFAULT_RPM: Final[int] = 60
_DEFAULT_TIMEOUT: Final[float] = 30.0


class _TokenBucket:
    """Async token bucket: capacity tokens, refilled at ``rate`` per second."""

    def __init__(self, *, capacity: int, rate_per_second: float) -> None:
        self._capacity = float(capacity)
        self._rate = rate_per_second
        self._tokens = float(capacity)
        self._last_refill = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            while True:
                now = loop.time()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                    self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait_seconds = deficit / self._rate
                await asyncio.sleep(wait_seconds)


class _RetryableStatusError(Exception):
    """Raised internally to trigger tenacity retry on retryable status codes."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"retryable status {response.status_code}")
        self.response = response


def _parse_retry_after(value: str) -> float | None:
    """Parse a ``Retry-After`` header into a non-negative seconds delay.

    Args:
        value: Raw header value (integer seconds or HTTP-date string).

    Returns:
        Seconds to wait, or ``None`` if the header is unparseable.
    """
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(tz=UTC)).total_seconds())


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read and parse the ``Retry-After`` header off ``response``."""
    raw = response.headers.get("Retry-After")
    return None if raw is None else _parse_retry_after(raw)


def _is_retryable(exc: BaseException) -> bool:
    """Predicate for tenacity: retry on retryable status or transient transport error."""
    if isinstance(exc, _RetryableStatusError):
        return True
    return isinstance(exc, _RETRYABLE_TRANSPORT_EXC)


def _before_sleep_log(retry_state: RetryCallState) -> None:
    """Tenacity hook: log a warning before each retry sleep."""
    outcome = retry_state.outcome
    if outcome is None:
        return
    exc = outcome.exception()
    if not isinstance(exc, _RetryableStatusError):
        return
    response = exc.response
    _LOG.warning(
        "kalshi_http_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        url=str(response.request.url) if response.request else None,
        retry_after=response.headers.get("Retry-After"),
    )


class KalshiClient:
    """Async client for the Kalshi public REST API.

    The client is a long-lived singleton — open once, share across collectors,
    close on shutdown. The httpx client and token bucket are created lazily on
    first use.

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
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._bucket: _TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, _TokenBucket]:
        """Lazily create the shared httpx client and token bucket."""
        if self._closed:
            raise RuntimeError("KalshiClient is closed")
        if self._client is not None and self._bucket is not None:
            return self._client, self._bucket
        async with self._init_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self.timeout_seconds),
                    headers={"User-Agent": _USER_AGENT},
                )
            if self._bucket is None:
                self._bucket = _TokenBucket(
                    capacity=self.rpm,
                    rate_per_second=self.rpm / 60.0,
                )
            return self._client, self._bucket

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
        client, bucket = await self._ensure_ready()
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=1.0,
                min=_BACKOFF_MIN_SECONDS,
                max=_BACKOFF_MAX_SECONDS,
            ),
            before_sleep=_before_sleep_log,
            reraise=True,
        )
        response: httpx.Response | None = None
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._send_once(client, bucket, path, params)
        except _RetryableStatusError as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover - raise_for_status always raises
        if response is None:  # pragma: no cover - tenacity guarantees one attempt
            raise RuntimeError("retry loop produced no response")
        payload = response.json()
        if not isinstance(payload, dict):
            msg = f"expected JSON object from {path}, got {type(payload).__name__}"
            raise TypeError(msg)
        return payload  # type: ignore[return-value]

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        bucket: _TokenBucket,
        path: str,
        params: Mapping[str, Any] | None,
    ) -> httpx.Response:
        """Single request attempt with token-bucket gating and Retry-After support."""
        await bucket.acquire()
        response = await client.get(path, params=dict(params) if params else None)
        if response.status_code in _RETRYABLE_STATUS:
            if response.status_code == _STATUS_TOO_MANY_REQUESTS:
                wait = _retry_after_seconds(response)
                if wait is not None and wait > 0:
                    await asyncio.sleep(wait)
            raise _RetryableStatusError(response)
        response.raise_for_status()
        return response

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
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> Self:
        """Async context-manager entry — returns ``self`` for the with-block."""
        await self._ensure_ready()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context-manager exit — calls :meth:`aclose`."""
        await self.aclose()
