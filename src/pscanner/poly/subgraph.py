"""Async GraphQL client for The Graph's hosted gateway.

Wraps :class:`pscanner.util.http_client.RateLimitedHttpClient` for the
token-bucket + retry + Retry-After plumbing; the GraphQL-specific
``query(graphql, variables)`` layer is the single public surface here.
``errors`` arrays in responses surface as ``RuntimeError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

from pscanner.util.http_client import RateLimitedHttpClient

_LOG_EVENT = "subgraph_retry"


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
        self.url = url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._inner = RateLimitedHttpClient(
            rpm=rpm,
            timeout_seconds=timeout_seconds,
            log_event=_LOG_EVENT,
        )

    async def query(self, graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one GraphQL query, returning the ``data`` payload.

        Args:
            graphql: GraphQL query string.
            variables: Query variables to pass alongside the query.

        Returns:
            The ``data`` object from the GraphQL response.

        Raises:
            RuntimeError: If the response contains a non-empty ``errors`` array
                or no ``data`` object.
            httpx.HTTPStatusError: On non-2xx after retries are exhausted.
        """
        body = {"query": graphql, "variables": dict(variables)}
        response = await self._inner.post(self.url, json=body)
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"GraphQL response missing 'data' object: {payload!r}")
        return data

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
