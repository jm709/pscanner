"""Async HTTP client for the Manifold Markets REST API.

Public REST endpoints require no authentication. The IP-shared 500-req/min
rate limit applies globally across all endpoints — a single ``_TokenBucket``
instance enforces this for all concurrent callers sharing a ``ManifoldClient``.

Tenacity retries on 429 and 5xx transient failures with exponential backoff.

Example::

    async with ManifoldClient() as client:
        markets = await client.get_markets(limit=100)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId
from pscanner.manifold.models import ManifoldBet, ManifoldMarket

_BASE_URL = "https://api.manifold.markets"
_USER_AGENT = "pscanner/0.1"

# Manifold's global IP-shared rate limit.
_RPM_LIMIT = 500
_RATE_PER_SECOND = _RPM_LIMIT / 60.0

_STATUS_TOO_MANY_REQUESTS = 429
_RETRYABLE_STATUS = frozenset({_STATUS_TOO_MANY_REQUESTS, 502, 503, 504})
_RETRYABLE_TRANSPORT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
_MAX_ATTEMPTS = 5
_BACKOFF_MIN_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0

_LOG = structlog.get_logger(__name__)


class _TokenBucket:
    """Async token bucket: capacity tokens, refilled at ``rate`` per second."""

    def __init__(self, *, capacity: int, rate_per_second: float) -> None:
        self._capacity = float(capacity)
        self._rate = rate_per_second
        self._tokens = float(capacity)
        self._last_refill = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> float:
        """Current token count (without refill)."""
        return self._tokens

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
    """Raised internally to trigger tenacity retry on retryable HTTP status codes."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"retryable status {response.status_code}")
        self.response = response


def _parse_retry_after(value: str) -> float | None:
    """Parse a ``Retry-After`` header into a non-negative wait in seconds.

    Args:
        value: Raw header value (integer seconds or HTTP-date).

    Returns:
        Seconds to wait, or ``None`` if unparseable.
    """
    stripped = value.strip()
    if not stripped:
        return None
    try:
        seconds = float(stripped)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)
    try:
        when = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - datetime.now(tz=UTC)).total_seconds()
    return max(0.0, delta)


def _is_retryable(exc: BaseException) -> bool:
    """Predicate for tenacity: True for retryable status or transient transport error."""
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
        "manifold_http_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        url=str(response.request.url) if response.request else None,
        retry_after=response.headers.get("Retry-After"),
    )


class ManifoldClient:
    """Async HTTP client for the public Manifold Markets REST API.

    Enforces the IP-shared 500-req/min budget via an internal ``_TokenBucket``
    and retries 429/5xx with tenacity exponential backoff.

    The client is a long-lived singleton. Open once, share across callers,
    close on shutdown. Both context-manager and explicit ``aclose()`` patterns
    are supported.
    """

    def __init__(
        self,
        *,
        base_url: str = _BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Configure the client without opening any connections.

        Args:
            base_url: Manifold API base URL (override for testing).
            timeout_seconds: Per-request timeout.
        """
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._http: httpx.AsyncClient | None = None
        self._bucket: _TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, _TokenBucket]:
        """Lazily create the shared httpx client and token bucket."""
        if self._closed:
            raise RuntimeError("ManifoldClient is closed")
        if self._http is not None and self._bucket is not None:
            return self._http, self._bucket
        async with self._init_lock:
            if self._http is None:
                self._http = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout_seconds),
                    headers={"User-Agent": _USER_AGENT},
                )
            if self._bucket is None:
                self._bucket = _TokenBucket(
                    capacity=_RPM_LIMIT,
                    rate_per_second=_RATE_PER_SECOND,
                )
            return self._http, self._bucket

    async def _get_raw(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """GET ``base_url + path`` with rate-limiting and retries.

        Args:
            path: Path-only fragment (must start with ``/``).
            params: Optional query-string parameters.

        Returns:
            Parsed JSON: either a dict or list.

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx or exhausted retries.
            httpx.HTTPError: On transport-level failures.
        """
        http, bucket = await self._ensure_ready()
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
                    response = await self._send_once(http, bucket, path, params)
        except _RetryableStatusError as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover
        if response is None:  # pragma: no cover
            raise RuntimeError("retry loop produced no response")
        return response.json()  # type: ignore[no-any-return]

    async def _send_once(
        self,
        http: httpx.AsyncClient,
        bucket: _TokenBucket,
        path: str,
        params: Mapping[str, Any] | None,
    ) -> httpx.Response:
        """Single request attempt with token-bucket gating and Retry-After honour."""
        await bucket.acquire()
        response = await http.get(path, params=dict(params) if params else None)
        if response.status_code in _RETRYABLE_STATUS:
            if response.status_code == _STATUS_TOO_MANY_REQUESTS:
                raw = response.headers.get("Retry-After")
                if raw is not None:
                    wait = _parse_retry_after(raw)
                    if wait is not None and wait > 0:
                        await asyncio.sleep(wait)
            raise _RetryableStatusError(response)
        response.raise_for_status()
        return response

    async def get_markets(
        self,
        *,
        limit: int = 1000,
        before: str | None = None,
    ) -> list[ManifoldMarket]:
        """Fetch one page of markets using Manifold's ``before`` cursor.

        Args:
            limit: Maximum markets to return (server default is 500; capped at 1000).
            before: Opaque cursor — the ``id`` of the last market from the previous
                page. Pass ``None`` to start from the most recent.

        Returns:
            List of ``ManifoldMarket`` models.
        """
        params: dict[str, Any] = {"limit": limit}
        if before is not None:
            params["before"] = before
        payload = await self._get_raw("/v0/markets", params=params)
        if not isinstance(payload, list):
            return []
        return [ManifoldMarket.model_validate(item) for item in payload]

    async def get_market(self, market_id: ManifoldMarketId) -> ManifoldMarket:
        """Fetch a single market by its opaque hash ID.

        Args:
            market_id: Manifold market ID (not the slug).

        Returns:
            ``ManifoldMarket`` model.
        """
        payload = await self._get_raw(f"/v0/market/{market_id}")
        return ManifoldMarket.model_validate(payload)

    async def search_markets(
        self,
        query: str,
        *,
        limit: int = 100,
    ) -> list[ManifoldMarket]:
        """Search markets by text query.

        Args:
            query: Full-text search string.
            limit: Maximum results to return.

        Returns:
            List of ``ManifoldMarket`` models.
        """
        payload = await self._get_raw("/v0/search-markets", params={"term": query, "limit": limit})
        if not isinstance(payload, list):
            return []
        return [ManifoldMarket.model_validate(item) for item in payload]

    async def get_bets(
        self,
        *,
        market_id: ManifoldMarketId | None = None,
        user_id: ManifoldUserId | None = None,
        limit: int = 1000,
        before: str | None = None,
    ) -> list[ManifoldBet]:
        """Fetch bets, optionally scoped to a market or user.

        Pass ``kinds="open-limit"`` queries aren't exposed as a typed parameter here;
        call ``_get_raw`` directly if you need open-limit-order filtering.

        Args:
            market_id: Filter to bets on a specific market.
            user_id: Filter to bets by a specific user.
            limit: Maximum bets to return.
            before: Opaque cursor (bet ``id``) for pagination.

        Returns:
            List of ``ManifoldBet`` models.
        """
        params: dict[str, Any] = {"limit": limit}
        if market_id is not None:
            params["contractId"] = market_id
        if user_id is not None:
            params["userId"] = user_id
        if before is not None:
            params["before"] = before
        payload = await self._get_raw("/v0/bets", params=params)
        if not isinstance(payload, list):
            return []
        return [ManifoldBet.model_validate(item) for item in payload]

    async def aclose(self) -> None:
        """Close the underlying httpx client and release connections."""
        self._closed = True
        http = self._http
        self._http = None
        if http is not None:
            await http.aclose()

    async def __aenter__(self) -> Self:
        """Async context-manager entry — ensures the client is initialised."""
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
