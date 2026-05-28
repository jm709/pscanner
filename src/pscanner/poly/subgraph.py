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

import structlog

from pscanner.util.http_client import RateLimitedHttpClient

_LOG_EVENT = "subgraph_retry"
_LOG = structlog.get_logger()


_QUOTA_ERROR_MARKER = "payment required"


def _is_quota_exhausted(errors: list[dict[str, Any]]) -> bool:
    """Return True if any GraphQL error message indicates an exhausted key.

    The gateway returns ``{"message": "auth error: payment required for
    subsequent requests for this API key"}`` once the monthly free-tier
    quota runs out. Substring match keeps this resilient to minor wording
    changes from The Graph.
    """
    return any(_QUOTA_ERROR_MARKER in str(err.get("message", "")).lower() for err in errors)


class SubgraphQuotaExhaustedError(RuntimeError):
    """Raised when the gateway rejects requests with `payment required`.

    Distinct subclass of :class:`RuntimeError` so callers' broad
    ``except RuntimeError`` still catch it, while a narrower
    ``except SubgraphQuotaExhaustedError`` can branch on quota errors
    specifically (used by the V1/V2 dispatchers to short-circuit and by
    the CLI to print a clear remediation message).
    """


class SubgraphClient:
    """Async GraphQL client targeting a single subgraph endpoint.

    The client is lazy — no sockets are opened until the first ``query()``
    call. Use as an async context manager or call ``aclose()`` explicitly.

    Attributes:
        url: Full subgraph endpoint URL.
        rpm: Requests-per-minute ceiling enforced by the token bucket.
        timeout_seconds: Per-request timeout passed to :mod:`httpx`.
    """

    def __init__(
        self,
        *,
        url: str,
        rpm: int,
        timeout_seconds: float = 30.0,
        fallback_url: str | None = None,
    ) -> None:
        """Store config without opening any sockets.

        Args:
            url: Full subgraph endpoint URL (e.g. the Graph gateway URL).
            rpm: Requests-per-minute budget.
            timeout_seconds: Default per-request timeout.
            fallback_url: Optional secondary endpoint URL (typically the
                same subgraph id with a backup API key baked in). On the
                first :class:`SubgraphQuotaExhaustedError` from the
                primary URL, the client swaps ``self.url`` to this value
                and retries the in-flight query. A second quota error
                (now from the fallback) propagates.
        """
        self.url = url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._fallback_url = fallback_url
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
        try:
            return await self._query_once(body)
        except SubgraphQuotaExhaustedError:
            if self._fallback_url is None:
                raise
            _LOG.warning(
                "subgraph.api_key_rotated",
                reason="primary_quota_exhausted",
                primary_url=self.url,
                fallback_url=self._fallback_url,
            )
            self.url = self._fallback_url
            self._fallback_url = None
            return await self._query_once(body)

    async def _query_once(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a single GraphQL POST against ``self.url`` with no retries on quota errors."""
        response = await self._inner.post(self.url, json=body)
        payload = response.json()
        if payload.get("errors"):
            if _is_quota_exhausted(payload["errors"]):
                raise SubgraphQuotaExhaustedError(f"GraphQL quota exhausted: {payload['errors']}")
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
