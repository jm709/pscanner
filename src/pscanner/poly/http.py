"""Shared HTTP client base for Polymarket REST endpoints.

Thin adapter over :class:`pscanner.util.http_client.RateLimitedHttpClient`.
Preserves the public surface (``get(path, *, params) -> dict | list``) used by
``GammaClient`` and ``DataClient``. The structlog retry-event name is
``polymarket_http_retry`` (passed through to the shared client).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

from pscanner.util.http_client import RateLimitedHttpClient

_LOG_EVENT = "polymarket_http_retry"


class PolyHttpClient:
    """Async HTTP client for Polymarket REST hosts.

    The client is a long-lived singleton — open once, share across detectors,
    close on shutdown. The underlying httpx client and token bucket are
    constructed lazily on first use so instantiation is cheap.

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
        self.base_url = base_url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._inner = RateLimitedHttpClient(
            rpm=rpm,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            log_event=_LOG_EVENT,
        )

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
        response = await self._inner.get(path, params=params)
        return response.json()  # type: ignore[no-any-return]

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
