"""Async HTTP client for the Manifold Markets REST API.

Wraps :class:`pscanner.util.http_client.RateLimitedHttpClient` for the
token-bucket + retry + Retry-After plumbing; the Manifold-specific endpoint
methods (markets, bets, search) are the public surface.

Public REST endpoints require no authentication. The IP-shared 500-req/min
rate limit applies globally across all endpoints — a single ``TokenBucket``
instance inside the shared client enforces this for all concurrent callers
sharing a ``ManifoldClient``.

Example::

    async with ManifoldClient() as client:
        markets = await client.get_markets(limit=100)
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId
from pscanner.manifold.models import ManifoldBet, ManifoldMarket
from pscanner.util.http_client import RateLimitedHttpClient

_BASE_URL = "https://api.manifold.markets"
_LOG_EVENT = "manifold_http_retry"

# Manifold's global IP-shared rate limit (500 req/min, applied across all
# endpoints; multi-IP rotation is prohibited per Manifold ToS).
_RPM_LIMIT = 500


class ManifoldClient:
    """Async HTTP client for the public Manifold Markets REST API.

    Enforces the IP-shared 500-req/min budget via the shared RateLimited
    HttpClient's internal ``TokenBucket`` and retries 429/5xx with tenacity
    exponential backoff.

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
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._inner = RateLimitedHttpClient(
            rpm=_RPM_LIMIT,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            log_event=_LOG_EVENT,
        )

    async def _get_raw(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
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
        response = await self._inner.get(path, params=params)
        return response.json()  # type: ignore[no-any-return]

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
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        """Async context-manager entry — ensures the client is initialised."""
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
