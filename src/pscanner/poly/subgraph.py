"""Async GraphQL client for The Graph's hosted gateway.

Mirrors the rate-limit + retry shape of ``pscanner.poly.http``.
The single public surface is ``query(graphql, variables)`` returning the
``data`` payload; GraphQL ``errors`` arrays surface as RuntimeError.
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
_RETRYABLE_TRANSPORT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
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
    """Predicate for tenacity: retry on retryable status or transient transport error."""
    return isinstance(exc, (_RetryableStatusError, *_RETRYABLE_TRANSPORT_EXC))


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
        "subgraph_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        url=str(response.request.url) if response.request else None,
        retry_after=response.headers.get("Retry-After"),
    )


class SubgraphClient:
    """Async GraphQL client targeting a single subgraph endpoint.

    The client is lazy — no sockets are opened until the first ``query()``
    call. Use as an async context manager or call ``aclose()`` explicitly.

    Attributes:
        url: Full subgraph endpoint URL.
        rpm: Requests-per-minute ceiling enforced by the token bucket.
        timeout_seconds: Per-request timeout passed to :mod:`httpx`.
    """

    def __init__(self, *, url: str, rpm: int, timeout_seconds: float = 30.0) -> None:
        """Store config without opening any sockets.

        Args:
            url: Full subgraph endpoint URL (e.g. the Graph gateway URL).
            rpm: Requests-per-minute budget.
            timeout_seconds: Default per-request timeout.
        """
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.url = url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._bucket: _TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, _TokenBucket]:
        """Lazily create the shared httpx client and token bucket."""
        if self._closed:
            raise RuntimeError("SubgraphClient is closed")
        if self._client is not None and self._bucket is not None:
            return self._client, self._bucket
        async with self._init_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Content-Type": "application/json",
                    },
                )
            if self._bucket is None:
                self._bucket = _TokenBucket(
                    capacity=self.rpm,
                    rate_per_second=self.rpm / 60.0,
                )
            return self._client, self._bucket

    async def query(self, graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one GraphQL query, returning the ``data`` payload.

        Args:
            graphql: GraphQL query string.
            variables: Query variables to pass alongside the query.

        Returns:
            The ``data`` object from the GraphQL response.

        Raises:
            RuntimeError: If the response contains a non-empty ``errors`` array.
            httpx.HTTPStatusError: On non-2xx after retries are exhausted.
        """
        client, bucket = await self._ensure_ready()
        body = {"query": graphql, "variables": dict(variables)}
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
                    response = await self._send_once(client, bucket, body)
        except _RetryableStatusError as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover
        if response is None:  # pragma: no cover
            raise RuntimeError("retry loop produced no response")
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"GraphQL response missing 'data' object: {payload!r}")
        return data

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        bucket: _TokenBucket,
        body: dict[str, Any],
    ) -> httpx.Response:
        """Single request attempt with token-bucket + Retry-After honouring."""
        await bucket.acquire()
        response = await client.post(self.url, json=body)
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
