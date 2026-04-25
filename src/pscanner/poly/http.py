"""Shared HTTP client base for Polymarket REST endpoints.

Wraps ``httpx.AsyncClient`` with a token-bucket rate limiter and ``tenacity``
retries that honour ``Retry-After`` on 429/5xx. Wave 2 (``http-client`` agent)
implements the body; Wave 1 freezes only the public shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self


class PolyHttpClient:
    """Async HTTP client for Polymarket REST hosts.

    The client is a long-lived singleton — open once, share across detectors,
    close on shutdown. The token bucket is constructed lazily on first use so
    instantiation is cheap.

    Attributes:
        base_url: Host (with scheme) for relative ``get`` calls.
        rpm: Requests-per-minute ceiling enforced by the token bucket.
        timeout_seconds: Per-request timeout passed to ``httpx``.
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
        raise NotImplementedError("Wave 2: http-client")

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
            httpx.HTTPStatusError: On 4xx after retries have been exhausted.
            httpx.HTTPError: On transport-level failures.
        """
        raise NotImplementedError("Wave 2: http-client")

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` and release sockets."""
        raise NotImplementedError("Wave 2: http-client")

    async def __aenter__(self) -> Self:
        """Async context-manager entry — returns ``self`` for the with-block."""
        raise NotImplementedError("Wave 2: http-client")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context-manager exit — must call :meth:`aclose`."""
        raise NotImplementedError("Wave 2: http-client")
