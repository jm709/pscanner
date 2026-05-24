"""Shared async HTTP client with token-bucket rate limit + tenacity retries.

Single source of truth for the rate-limited HTTP / JSON-RPC / GraphQL
transport plumbing previously duplicated across ``pscanner.poly.http``,
``pscanner.poly.onchain_rpc``, ``pscanner.poly.subgraph``,
``pscanner.kalshi.client``, and ``pscanner.manifold.client``.

:class:`RateLimitedHttpClient` lazily opens one :class:`httpx.AsyncClient` and
one :class:`TokenBucket` on first request. Each request acquires a token,
issues the call, and retries on 429/5xx or transient transport errors via
:mod:`tenacity`. ``Retry-After`` (numeric seconds or HTTP-date) is honoured
on 429 before tenacity's exponential backoff is consulted.

The ``log_event`` constructor argument names the structlog event emitted on
every retry sleep — preserved per-platform so existing log consumers keep
working.
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

_USER_AGENT_DEFAULT = "pscanner/0.1"

_STATUS_TOO_MANY_REQUESTS = 429
_RETRYABLE_STATUS = frozenset({_STATUS_TOO_MANY_REQUESTS, 502, 503, 504})
# Transport-level errors treated as transient. Connection drops, read/write
# timeouts, and protocol errors mid-response all benefit from exponential
# backoff. ``UnsupportedProtocol`` and ``ProxyError`` are configuration bugs
# and intentionally excluded.
_RETRYABLE_TRANSPORT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
_MAX_ATTEMPTS = 5
_BACKOFF_MIN_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0

_LOG = structlog.get_logger(__name__)


class TokenBucket:
    """Async token bucket: capacity tokens, refilled at ``rate_per_second``."""

    def __init__(self, *, capacity: int, rate_per_second: float) -> None:
        """Initialise a full bucket.

        Args:
            capacity: Maximum tokens the bucket can hold.
            rate_per_second: Refill rate (tokens per second).
        """
        self._capacity = float(capacity)
        self._rate = rate_per_second
        self._tokens = float(capacity)
        self._last_refill = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> float:
        """Current token count (without applying pending refill)."""
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
                await asyncio.sleep(deficit / self._rate)


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


def _is_retryable(exc: BaseException) -> bool:
    """True for retryable status or transient transport error."""
    return isinstance(exc, (_RetryableStatusError, *_RETRYABLE_TRANSPORT_EXC))


class RateLimitedHttpClient:
    """Async HTTP client: lazy httpx + TokenBucket + 429/5xx tenacity retry.

    The client is a long-lived singleton — open once, share across callers,
    close on shutdown. Both context-manager and explicit ``aclose()`` patterns
    are supported. No sockets are opened until the first request, so
    construction is cheap.

    When ``base_url`` is None (the default), :meth:`request` accepts absolute
    URLs in the ``path`` argument — the use case for JSON-RPC and GraphQL
    clients that target a single full endpoint URL.
    """

    def __init__(
        self,
        *,
        rpm: int,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        log_event: str = "http_retry",
        user_agent: str = _USER_AGENT_DEFAULT,
    ) -> None:
        """Store config without opening any sockets.

        Args:
            rpm: Requests-per-minute budget enforced by the token bucket.
            base_url: Optional host (with scheme) prefixed to relative paths.
                When omitted, callers pass absolute URLs to :meth:`request`.
            timeout_seconds: Default per-request timeout.
            log_event: Structlog event name used on retry sleep — distinct per
                platform so existing log consumers keep working.
            user_agent: ``User-Agent`` header value sent on every request.

        Raises:
            ValueError: If ``rpm`` or ``timeout_seconds`` is non-positive.
        """
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.rpm = rpm
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._log_event = log_event
        self._user_agent = user_agent
        self._client: httpx.AsyncClient | None = None
        self._bucket: TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, TokenBucket]:
        """Lazily create the shared httpx client and token bucket."""
        if self._closed:
            raise RuntimeError("RateLimitedHttpClient is closed")
        if self._client is not None and self._bucket is not None:
            return self._client, self._bucket
        async with self._init_lock:
            if self._client is None:
                kwargs: dict[str, Any] = {
                    "timeout": httpx.Timeout(self.timeout_seconds),
                    "headers": {"User-Agent": self._user_agent},
                }
                if self.base_url is not None:
                    kwargs["base_url"] = self.base_url
                self._client = httpx.AsyncClient(**kwargs)
            if self._bucket is None:
                self._bucket = TokenBucket(
                    capacity=self.rpm,
                    rate_per_second=self.rpm / 60.0,
                )
            return self._client, self._bucket

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
    ) -> httpx.Response:
        """Issue one request with token-bucket gating and tenacity retry.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, etc.).
            path: Path-only URL fragment (when ``base_url`` was set) or full
                absolute URL (when ``base_url`` was ``None``).
            params: Optional query-string parameters.
            json: Optional JSON-encodable body. When set, httpx auto-applies
                ``Content-Type: application/json``.

        Returns:
            The :class:`httpx.Response` after a successful (or retried) call.

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx, or after retries are
                exhausted on 429/5xx.
            httpx.HTTPError: On transport-level failures exceeding retry budget.
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
            before_sleep=self._before_sleep_log,
            reraise=True,
        )
        response: httpx.Response | None = None
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._send_once(
                        client,
                        bucket,
                        method,
                        path,
                        params=params,
                        json=json,
                    )
        except _RetryableStatusError as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover - raise_for_status always raises
        if response is None:  # pragma: no cover - tenacity guarantees one attempt
            raise RuntimeError("retry loop produced no response")
        return response

    async def get(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        """Convenience wrapper for ``GET``."""
        return await self.request("GET", path, params=params)

    async def post(
        self,
        path: str,
        *,
        json: Any = None,
    ) -> httpx.Response:
        """Convenience wrapper for ``POST``."""
        return await self.request("POST", path, json=json)

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        bucket: TokenBucket,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None,
        json: Any,
    ) -> httpx.Response:
        """Single request attempt with token-bucket gating and Retry-After honour."""
        await bucket.acquire()
        response = await client.request(
            method,
            path,
            params=dict(params) if params else None,
            json=json,
        )
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

    def _before_sleep_log(self, retry_state: RetryCallState) -> None:
        """Tenacity hook: log a warning before each retry sleep."""
        outcome = retry_state.outcome
        if outcome is None:
            return
        exc = outcome.exception()
        if not isinstance(exc, _RetryableStatusError):
            return
        response = exc.response
        _LOG.warning(
            self._log_event,
            attempt=retry_state.attempt_number,
            status_code=response.status_code,
            url=str(response.request.url) if response.request else None,
            retry_after=response.headers.get("Retry-After"),
        )

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
