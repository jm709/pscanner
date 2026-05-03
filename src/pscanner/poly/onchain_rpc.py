"""Async JSON-RPC client for Polygon mainnet (eth_* methods).

Targets any EVM-compatible RPC endpoint. Default is Polygon Foundation's
public RPC (``https://polygon-rpc.com/``), free and unauthenticated.
Override via constructor for Alchemy or other providers.

Mirrors the rate-limiting + retry pattern in ``pscanner.poly.http`` but
speaks JSON-RPC POST instead of REST GET.
"""

from __future__ import annotations

import asyncio
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
_BLOCK_TIMESTAMP_CACHE_SIZE = 4096


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
    return max(0.0, (when - datetime.now(tz=UTC)).total_seconds())


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
        "polygon_rpc_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        url=str(response.request.url) if response.request else None,
        retry_after=response.headers.get("Retry-After"),
    )


class OnchainRpcClient:
    """Async JSON-RPC client for ``eth_*`` calls against any Polygon RPC.

    Long-lived: open once, reuse across an ingest run, close on shutdown.
    The httpx client is created lazily on first use so construction is cheap.
    """

    def __init__(
        self,
        *,
        rpc_url: str,
        rpm: int,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Store config without opening any sockets.

        Args:
            rpc_url: Full RPC endpoint URL (e.g. ``https://polygon-rpc.com/``).
            rpm: Requests-per-minute budget (informational; not token-bucketed).
            timeout_seconds: Default per-request timeout.
        """
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.rpc_url = rpc_url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._bucket: _TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False
        self._next_id = 1
        self._ts_cache: dict[int, int] = {}

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, _TokenBucket]:
        """Lazily create the shared httpx client and token bucket."""
        if self._closed:
            raise RuntimeError("OnchainRpcClient is closed")
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

    async def _call(self, method: str, params: list[Any]) -> Any:
        """Issue a single JSON-RPC call with tenacity retry."""
        client, bucket = await self._ensure_ready()
        request_id = self._next_id
        self._next_id += 1
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

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
            raise  # pragma: no cover - raise_for_status always raises
        if response is None:  # pragma: no cover - tenacity guarantees one attempt
            raise RuntimeError("retry loop produced no response")
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"RPC error from {method}: {payload['error']}")
        return payload["result"]

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        bucket: _TokenBucket,
        body: dict[str, Any],
    ) -> httpx.Response:
        """Single POST attempt with token-bucket gating and Retry-After honouring."""
        await bucket.acquire()
        response = await client.post(self.rpc_url, json=body)
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

    async def get_block_number(self) -> int:
        """Return the current Polygon head block number."""
        result = await self._call("eth_blockNumber", [])
        return int(result, 16)

    async def get_logs(
        self,
        *,
        address: str,
        topics: list[str],
        from_block: int,
        to_block: int,
    ) -> list[dict[str, Any]]:
        """Fetch logs matching ``address`` and ``topics`` between two block bounds.

        Args:
            address: Contract address (lowercase or checksummed; RPC accepts both).
            topics: Topic filter; ``topics[0]`` is the event signature hash.
            from_block: First block in the inclusive range.
            to_block: Last block in the inclusive range.

        Returns:
            List of raw log dicts as returned by the RPC.

        Raises:
            RuntimeError: If the RPC returns a JSON-RPC error.
            httpx.HTTPStatusError: On non-2xx HTTP status.
        """
        params: list[Any] = [
            {
                "address": address,
                "topics": topics,
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
            }
        ]
        result = await self._call("eth_getLogs", params)
        if not isinstance(result, list):
            raise RuntimeError(f"eth_getLogs returned non-list result: {result!r}")
        return result

    async def get_block_timestamp(self, block_number: int) -> int:
        """Return the Unix-second timestamp of the given Polygon block.

        Caches the ``(block_number -> timestamp)`` mapping in-memory; capped to
        ``_BLOCK_TIMESTAMP_CACHE_SIZE`` entries. When the cap is hit the oldest
        insertion is evicted (FIFO — Polygon walk is forward-monotonic so older
        blocks rarely re-appear).

        Args:
            block_number: Polygon block height to look up.

        Returns:
            Unix timestamp in seconds.
        """
        cached = self._ts_cache.get(block_number)
        if cached is not None:
            return cached
        result = await self._call("eth_getBlockByNumber", [hex(block_number), False])
        if not isinstance(result, dict) or "timestamp" not in result:
            raise RuntimeError(
                f"eth_getBlockByNumber({block_number}) returned malformed payload: {result!r}"
            )
        ts = int(result["timestamp"], 16)
        if len(self._ts_cache) >= _BLOCK_TIMESTAMP_CACHE_SIZE:
            oldest = next(iter(self._ts_cache))
            del self._ts_cache[oldest]
        self._ts_cache[block_number] = ts
        return ts

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
