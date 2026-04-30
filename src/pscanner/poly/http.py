"""Shared HTTP client base for Polymarket REST endpoints.

Wraps :class:`httpx.AsyncClient` with a token-bucket rate limiter and
:mod:`tenacity` retries that honour ``Retry-After`` on 429/5xx responses.
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

_LOG = structlog.get_logger(__name__)

_USER_AGENT = "pscanner/0.1"
_STATUS_TOO_MANY_REQUESTS = 429
_RETRYABLE_STATUS = frozenset({_STATUS_TOO_MANY_REQUESTS, 502, 503, 504})
# Transport-level errors we treat as transient. Connection drops, read/write
# timeouts, and protocol errors mid-response all benefit from exponential
# backoff. ``UnsupportedProtocol`` and ``ProxyError`` are configuration bugs
# and intentionally excluded.
_RETRYABLE_TRANSPORT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,  # ReadTimeout, ConnectTimeout, WriteTimeout, PoolTimeout
    httpx.NetworkError,  # ConnectError, ReadError, WriteError, CloseError
    httpx.RemoteProtocolError,  # connection broken before response ended
)
_MAX_ATTEMPTS = 5
_BACKOFF_MIN_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0


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
    """Raised internally to trigger tenacity retry on retryable status codes."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"retryable status {response.status_code}")
        self.response = response


def _parse_retry_after(value: str) -> float | None:
    """Parse a ``Retry-After`` header value into a non-negative seconds delay.

    Args:
        value: Raw header value (integer seconds or HTTP-date).

    Returns:
        Number of seconds to wait, or ``None`` if the header is unparseable.
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


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read and parse the ``Retry-After`` header off ``response``."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    return _parse_retry_after(raw)


def _is_retryable(exc: BaseException) -> bool:
    """Predicate for tenacity: retry on retryable status or transient transport error."""
    if isinstance(exc, _RetryableStatusError):
        return True
    return isinstance(exc, _RETRYABLE_TRANSPORT_EXC)


class PolyHttpClient:
    """Async HTTP client for Polymarket REST hosts.

    The client is a long-lived singleton — open once, share across detectors,
    close on shutdown. The token bucket is constructed lazily on first use so
    instantiation is cheap.

    Attributes:
        base_url: Host (with scheme) for relative ``get`` calls.
        rpm: Requests-per-minute ceiling enforced by the token bucket.
        timeout_seconds: Per-request timeout passed to :mod:`httpx`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        rpm: int,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Store config without opening any sockets.

        Args:
            base_url: Host base URL (e.g. ``https://gamma-api.polymarket.com``).
            rpm: Requests-per-minute budget.
            timeout_seconds: Default per-request timeout.
        """
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.base_url = base_url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._bucket: _TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, _TokenBucket]:
        """Lazily create the shared httpx client and token bucket."""
        if self._closed:
            raise RuntimeError("PolyHttpClient is closed")
        if self._client is not None and self._bucket is not None:
            return self._client, self._bucket
        async with self._init_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout_seconds),
                    headers={"User-Agent": _USER_AGENT},
                )
            if self._bucket is None:
                self._bucket = _TokenBucket(
                    capacity=self.rpm,
                    rate_per_second=self.rpm / 60.0,
                )
            return self._client, self._bucket

    async def get(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """GET ``base_url + path`` with retries and rate limiting.

        Args:
            path: Path-only URL fragment (must start with ``/``).
            params: Optional query-string parameters.

        Returns:
            Parsed JSON: either an object (``dict``) or array (``list``).

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx, or after retries are
                exhausted on 429/5xx.
            httpx.HTTPError: On transport-level failures.
        """
        response = await self._send_with_retry(path, params=params)
        return response.json()

    async def _send_with_retry(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None,
    ) -> httpx.Response:
        """Issue the request with token-bucket gating and tenacity retry."""
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
        return response

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        bucket: _TokenBucket,
        path: str,
        params: Mapping[str, Any] | None,
    ) -> httpx.Response:
        """Single request attempt with token-bucket + Retry-After honouring."""
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
        "polymarket_http_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        url=str(response.request.url) if response.request else None,
        retry_after=response.headers.get("Retry-After"),
    )
