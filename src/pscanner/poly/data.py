"""Async client for Polymarket's data and leaderboard REST hosts.

The data API (``https://data-api.polymarket.com``) covers a wallet's open
positions, closed positions, and activity stream. The leaderboard lives on a
separate host (``https://lb-api.polymarket.com``). This client multiplexes a
single :class:`DataClient` over both, owning a second :class:`PolyHttpClient`
internally for the leaderboard host.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import structlog

from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position

_DATA_API_BASE_URL: Final[str] = "https://data-api.polymarket.com"
_LB_API_BASE_URL: Final[str] = "https://lb-api.polymarket.com"

_PERIOD_TO_WINDOW: Final[dict[str, str]] = {"day": "1d", "week": "7d", "all": "all"}

_ACTIVITY_PAGE_SIZE: Final[int] = 500

_log = structlog.get_logger(__name__)


def _ensure_list(payload: Any, *, endpoint: str) -> list[Any]:
    """Return ``payload`` as a list or raise an explicit ``TypeError``.

    Args:
        payload: Decoded JSON body returned by ``PolyHttpClient.get``.
        endpoint: Endpoint name, included in the error for context.

    Returns:
        The payload as a list of items.

    Raises:
        TypeError: If ``payload`` is not a list.
    """
    if not isinstance(payload, list):
        msg = f"expected list response from {endpoint}, got {type(payload).__name__}"
        raise TypeError(msg)
    return payload


class DataClient:
    """Typed wrapper over the data and leaderboard REST endpoints."""

    def __init__(self, http: PolyHttpClient | None = None, *, rpm: int = 50) -> None:
        """Build a client over a shared or freshly-created ``PolyHttpClient``.

        The data API and the leaderboard are hosted on different domains, so
        the client always uses two underlying HTTP clients. When ``http`` is
        passed in, it is reused for the data-api host and a second client is
        instantiated internally for the leaderboard host. When ``http`` is
        ``None`` both are owned and closed by :meth:`aclose`.

        Args:
            http: An existing ``PolyHttpClient`` bound to the data-api host.
                If ``None``, the client constructs its own.
            rpm: Per-host rate limit applied to any client this object owns.
        """
        if http is None:
            self._data_http: PolyHttpClient = PolyHttpClient(
                base_url=_DATA_API_BASE_URL,
                rpm=rpm,
            )
            self._owns_data_http = True
        else:
            self._data_http = http
            self._owns_data_http = False
        self._lb_http: PolyHttpClient = PolyHttpClient(
            base_url=_LB_API_BASE_URL,
            rpm=rpm,
        )
        self._owns_lb_http = True
        self._closed = False

    async def get_positions(
        self,
        address: str,
        *,
        size_threshold: float = 0.0,
    ) -> list[Position]:
        """Return a wallet's currently-open positions.

        Args:
            address: 0x-prefixed proxy wallet address.
            size_threshold: Minimum position size to include (shares).

        Returns:
            A list of ``Position`` models.
        """
        params: dict[str, Any] = {"user": address, "sizeThreshold": size_threshold}
        payload = await self._data_http.get("/positions", params=params)
        items = _ensure_list(payload, endpoint="/positions")
        return [Position.model_validate(item) for item in items]

    async def get_closed_positions(
        self,
        address: str,
        *,
        limit: int = 500,
    ) -> list[ClosedPosition]:
        """Return a wallet's resolved (closed) positions for winrate computation.

        Hits ``/v1/closed-positions`` on the data API. The legacy
        ``/closed-positions`` path still works but is marked deprecated by the
        server (``Deprecation: true``), so we use the v1 path directly.

        Args:
            address: 0x-prefixed proxy wallet address.
            limit: Max number of positions to fetch.

        Returns:
            A list of ``ClosedPosition`` models, newest-first.
        """
        params: dict[str, Any] = {"user": address, "limit": limit}
        payload = await self._data_http.get("/v1/closed-positions", params=params)
        items = _ensure_list(payload, endpoint="/v1/closed-positions")
        return [ClosedPosition.model_validate(item) for item in items]

    async def get_activity(
        self,
        address: str,
        *,
        limit: int = 500,
        offset: int = 0,
        type: str | None = None,  # noqa: A002  # mirrors the API query param name
    ) -> list[dict[str, Any]]:
        """Return raw activity events for a wallet (heterogeneous shape).

        The data API returns mixed event types (TRADE, REWARD, SPLIT, MERGE,
        etc.) with shape that varies by type, so the return value is left as
        a list of dicts. Callers that need a typed view should re-parse
        downstream.

        Args:
            address: 0x-prefixed proxy wallet address.
            limit: Max number of events to fetch.
            offset: Zero-based pagination offset; ``0`` returns the first page.
            type: Optional filter for a single activity type.

        Returns:
            A list of raw JSON event dicts.
        """
        params: dict[str, Any] = {"user": address, "limit": limit}
        if offset:
            params["offset"] = offset
        if type is not None:
            params["type"] = type
        payload = await self._data_http.get("/activity", params=params)
        items = _ensure_list(payload, endpoint="/activity")
        return [item for item in items if isinstance(item, dict)]

    async def get_first_activity_timestamp(self, address: str) -> int | None:
        """Return the unix-seconds timestamp of a wallet's earliest activity.

        Pages forward through ``/activity`` (newest-first, by ``offset``) until
        the server returns a short page, then takes the smallest ``timestamp``
        seen.

        Args:
            address: 0x-prefixed proxy wallet address.

        Returns:
            Unix seconds of the earliest event, or ``None`` if the wallet has
            no activity.
        """
        offset = 0
        earliest: int | None = None
        while True:
            page = await self._fetch_activity_page(address, offset=offset)
            if not page:
                break
            page_min = self._page_min_timestamp(page)
            if page_min is not None:
                earliest = page_min if earliest is None else min(earliest, page_min)
            if len(page) < _ACTIVITY_PAGE_SIZE:
                break
            offset += _ACTIVITY_PAGE_SIZE
        return earliest

    async def _fetch_activity_page(
        self,
        address: str,
        *,
        offset: int,
    ) -> list[dict[str, Any]]:
        """Fetch one ``_ACTIVITY_PAGE_SIZE`` page of activity at ``offset``."""
        params: dict[str, Any] = {
            "user": address,
            "limit": _ACTIVITY_PAGE_SIZE,
            "offset": offset,
        }
        payload = await self._data_http.get("/activity", params=params)
        items = _ensure_list(payload, endpoint="/activity")
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _page_min_timestamp(page: list[dict[str, Any]]) -> int | None:
        """Return the smallest int ``timestamp`` field on a page, or None."""
        timestamps = [item["timestamp"] for item in page if isinstance(item.get("timestamp"), int)]
        return min(timestamps) if timestamps else None

    async def get_leaderboard(
        self,
        *,
        period: Literal["day", "week", "all"] = "all",
        limit: int = 200,
    ) -> list[LeaderboardEntry]:
        """Return the top-N traders by realised PnL over the chosen period.

        Hits ``GET https://lb-api.polymarket.com/profit?window={window}&limit={limit}``
        where ``window`` is the wire-format value (``1d``/``7d``/``all``)
        derived from ``period``.

        The wire response does not include the ``period`` field, so we inject
        it from the request before validation so each ``LeaderboardEntry``
        carries the originating window.

        Args:
            period: Leaderboard window — ``day``, ``week``, or ``all``-time.
            limit: How many entries to fetch (server-capped).

        Returns:
            A list of ``LeaderboardEntry`` models, ordered best-first.
        """
        window = _PERIOD_TO_WINDOW[period]
        params: dict[str, Any] = {"window": window, "limit": limit}
        payload = await self._lb_http.get("/profit", params=params)
        items = _ensure_list(payload, endpoint="/profit")
        entries: list[LeaderboardEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            enriched = {**item, "period": period}
            entries.append(LeaderboardEntry.model_validate(enriched))
        return entries

    async def aclose(self) -> None:
        """Close every underlying HTTP client this object owns."""
        if self._closed:
            return
        self._closed = True
        if self._owns_data_http:
            await self._data_http.aclose()
        if self._owns_lb_http:
            await self._lb_http.aclose()
        _log.debug("data_client.closed")
