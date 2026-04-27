"""Async client for Polymarket's data and leaderboard REST hosts.

The data API (``https://data-api.polymarket.com``) covers a wallet's open
positions, closed positions, and activity stream. The leaderboard lives on a
separate host (``https://lb-api.polymarket.com``). This client multiplexes a
single :class:`DataClient` over both, owning a second :class:`PolyHttpClient`
internally for the leaderboard host.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import httpx
import structlog

from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position

_DATA_API_BASE_URL: Final[str] = "https://data-api.polymarket.com"
_LB_API_BASE_URL: Final[str] = "https://lb-api.polymarket.com"

_PERIOD_TO_WINDOW: Final[dict[str, str]] = {"day": "1d", "week": "7d", "all": "all"}

_ACTIVITY_PAGE_SIZE: Final[int] = 500

_TRADES_PAGE_SIZE: Final[int] = 500
_TRADES_PAGE_CAP: Final[int] = 30  # 15k trades per condition_id maximum

# Polymarket caps offset-based pagination on /activity and /trades at this
# offset, returning HTTP 400 once the requested offset reaches it. We treat
# such 400s as end-of-data; other 400s (e.g. malformed address at offset=0)
# still propagate.
_POLYMARKET_OFFSET_CAP: Final[int] = 3500
_HTTP_BAD_REQUEST: Final[int] = 400

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

    async def get_settled_positions(
        self,
        address: str,
        *,
        limit: int = 500,
    ) -> list[ClosedPosition]:
        """Return ALL of a wallet's settled positions (wins + losses).

        Hits ``GET /positions?user={address}&closed=true&limit={limit}`` on the
        data API. Returns ALL settled positions (wins + losses), unlike the
        legacy ``/v1/closed-positions`` which only returns redeemed winners
        (and is hard-capped at 50 rows server-side regardless of ``limit``).

        Args:
            address: 0x-prefixed proxy wallet address.
            limit: Max number of positions to fetch.

        Returns:
            A list of ``ClosedPosition`` models. The response shape matches the
            standard ``/positions`` payload, so the existing model parses it.
        """
        params: dict[str, Any] = {"user": address, "closed": "true", "limit": limit}
        payload = await self._data_http.get("/positions", params=params)
        items = _ensure_list(payload, endpoint="/positions?closed=true")
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

    async def get_market_trades(
        self,
        condition_id: str,
        *,
        since_ts: int,
        until_ts: int,
    ) -> list[dict[str, Any]]:
        """Return all CONFIRMED trades on a market within ``[since_ts, until_ts]``.

        Paginates ``/trades?market=`` newest-first. Stops as soon as the newest
        timestamp on a page is older than ``since_ts`` or a short page is
        returned. Hard-capped at ``_TRADES_PAGE_CAP`` pages (15k trades) so a
        runaway market cannot exhaust the rate budget on a single call.

        Args:
            condition_id: 0x-prefixed market condition_id.
            since_ts: Inclusive lower bound on trade ``timestamp`` (unix seconds).
            until_ts: Inclusive upper bound on trade ``timestamp`` (unix seconds).

        Returns:
            A list of raw JSON trade dicts whose timestamps fall inside the window.
            Heterogeneous shape — callers re-parse downstream.
        """
        out: list[dict[str, Any]] = []
        offset = 0
        for _ in range(_TRADES_PAGE_CAP):
            page = await self._fetch_market_trades_page(condition_id, offset=offset)
            if not page:
                break
            page_max_ts = max(
                (t.get("timestamp", 0) for t in page if isinstance(t, dict)),
                default=0,
            )
            for item in page:
                ts = item.get("timestamp")
                if isinstance(ts, int) and since_ts <= ts <= until_ts:
                    out.append(item)
            if page_max_ts < since_ts or len(page) < _TRADES_PAGE_SIZE:
                break
            offset += _TRADES_PAGE_SIZE
        return out

    async def _fetch_market_trades_page(
        self,
        condition_id: str,
        *,
        offset: int,
    ) -> list[dict[str, Any]]:
        """Fetch one ``_TRADES_PAGE_SIZE`` page of market trades at ``offset``.

        Returns ``[]`` (which the caller treats as end-of-data) when Polymarket
        returns 400 at ``offset >= _POLYMARKET_OFFSET_CAP`` — the API caps
        offset-based pagination there. Other 400s propagate.
        """
        params: dict[str, Any] = {
            "market": condition_id,
            "limit": _TRADES_PAGE_SIZE,
            "offset": offset,
        }
        try:
            payload = await self._data_http.get("/trades", params=params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == _HTTP_BAD_REQUEST and offset >= _POLYMARKET_OFFSET_CAP:
                _log.info(
                    "data_client.trades_offset_capped",
                    condition_id=condition_id,
                    offset=offset,
                )
                return []
            raise
        items = _ensure_list(payload, endpoint="/trades")
        return [item for item in items if isinstance(item, dict)]

    async def get_market_slug_by_condition_id(self, condition_id: str) -> str | None:
        """Return a market's ``slug`` by querying its first trade.

        Polymarket's ``/trades?market=<conditionId>&limit=1`` returns recent
        trade rows with full market metadata (slug, title, outcome, asset_id).
        We pull the slug so the caller can fetch the full ``Market`` from
        gamma, since gamma's ``/markets`` endpoint does not filter by
        ``condition_id`` directly.

        Args:
            condition_id: 0x-prefixed market condition_id.

        Returns:
            The market's ``slug``, or ``None`` if the market has no trades or
            the response is malformed.
        """
        payload = await self._data_http.get(
            "/trades",
            params={"market": condition_id, "limit": 1},
        )
        items = _ensure_list(payload, endpoint="/trades")
        if not items or not isinstance(items[0], dict):
            return None
        slug = items[0].get("slug")
        return slug if isinstance(slug, str) and slug else None

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
        """Fetch one ``_ACTIVITY_PAGE_SIZE`` page of activity at ``offset``.

        Returns ``[]`` (which callers treat as end-of-data) when Polymarket
        returns 400 at ``offset >= _POLYMARKET_OFFSET_CAP`` — the API caps
        offset-based pagination there. Other 400s propagate so genuine
        validation errors at ``offset == 0`` are not silently swallowed.
        """
        params: dict[str, Any] = {
            "user": address,
            "limit": _ACTIVITY_PAGE_SIZE,
            "offset": offset,
        }
        try:
            payload = await self._data_http.get("/activity", params=params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == _HTTP_BAD_REQUEST and offset >= _POLYMARKET_OFFSET_CAP:
                _log.info(
                    "data_client.activity_offset_capped",
                    address=address,
                    offset=offset,
                )
                return []
            raise
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
