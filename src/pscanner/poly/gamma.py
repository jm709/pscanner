"""Async client for ``gamma-api.polymarket.com`` (events, markets catalogue)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import Event, Market

_BASE_URL = "https://gamma-api.polymarket.com"


def _bool_param(value: bool) -> str:
    """Render a boolean for Polymarket's lower-case query-string convention."""
    return "true" if value else "false"


class GammaClient:
    """Typed wrapper over the gamma REST endpoints.

    Use :meth:`iter_events` / :meth:`iter_markets` for full-catalogue scans;
    they paginate transparently. Use :meth:`list_events` / :meth:`list_markets`
    for one-shot bounded fetches.
    """

    def __init__(self, http: PolyHttpClient | None = None, *, rpm: int = 50) -> None:
        """Build a client over a shared or freshly-created ``PolyHttpClient``.

        Args:
            http: An existing ``PolyHttpClient`` instance. If ``None``, the
                client constructs its own bound to ``gamma-api.polymarket.com``.
            rpm: Per-host rate limit (only used when ``http`` is ``None``).
        """
        if http is None:
            self._http: PolyHttpClient = PolyHttpClient(base_url=_BASE_URL, rpm=rpm)
            self._owns_http: bool = True
        else:
            self._http = http
            self._owns_http = False

    async def list_events(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        """Fetch one page of events matching the filters.

        Args:
            active: Restrict to currently-active events.
            closed: Include closed events.
            limit: Page size (server-capped at 500).
            offset: Pagination offset.

        Returns:
            A list of validated ``Event`` models (possibly empty).
        """
        params = {
            "active": _bool_param(active),
            "closed": _bool_param(closed),
            "limit": limit,
            "offset": offset,
        }
        payload = await self._http.get("/events", params=params)
        return _parse_list(payload, Event)

    async def iter_events(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[Event]:
        """Async-iterate every event matching the filters across all pages.

        Args:
            active: Restrict to currently-active events.
            closed: Include closed events.
            page_size: Page size sent to the server per request.

        Yields:
            Each ``Event`` exactly once until the catalogue is exhausted.
        """
        offset = 0
        while True:
            page = await self.list_events(
                active=active,
                closed=closed,
                limit=page_size,
                offset=offset,
            )
            if not page:
                return
            for event in page:
                yield event
            if len(page) < page_size:
                return
            offset += page_size

    async def get_event(self, event_id: str) -> Event:
        """Fetch a single event by ID.

        Args:
            event_id: Polymarket event identifier.

        Returns:
            The event.

        Raises:
            httpx.HTTPStatusError: On 404 / other HTTP errors.
        """
        payload = await self._http.get(f"/events/{event_id}")
        if not isinstance(payload, dict):
            msg = f"expected JSON object for event {event_id}, got {type(payload).__name__}"
            raise TypeError(msg)
        return Event.model_validate(payload)

    async def get_event_by_slug(self, slug: str) -> Event | None:
        """Fetch a single event by slug, or ``None`` if no event matches.

        Polymarket's ``/events`` endpoint accepts a ``slug`` query parameter
        and returns a (possibly empty) list. ``/events/{slug}`` only resolves
        numeric ids on gamma, so callers with a slug must use this method.

        Args:
            slug: Event slug (e.g. ``"nba-okc-phx-2026-04-25"``).

        Returns:
            The matching event, or ``None`` if the slug is unknown to gamma.

        Raises:
            httpx.HTTPStatusError: On non-404 HTTP errors.
        """
        payload = await self._http.get("/events", params={"slug": slug})
        if not isinstance(payload, list):
            msg = f"expected JSON array for /events?slug={slug}, got {type(payload).__name__}"
            raise TypeError(msg)
        if not payload:
            return None
        return Event.model_validate(payload[0])

    async def list_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Fetch one page of markets matching the filters.

        Args:
            active: Restrict to currently-active markets.
            closed: Include closed markets.
            limit: Page size (server-capped).
            offset: Pagination offset.

        Returns:
            A list of validated ``Market`` models (possibly empty).
        """
        params = {
            "active": _bool_param(active),
            "closed": _bool_param(closed),
            "limit": limit,
            "offset": offset,
        }
        payload = await self._http.get("/markets", params=params)
        return _parse_list(payload, Market)

    async def iter_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[Market]:
        """Async-iterate every market matching the filters across all pages.

        Args:
            active: Restrict to currently-active markets.
            closed: Include closed markets.
            page_size: Page size sent to the server per request.

        Yields:
            Each ``Market`` exactly once until the catalogue is exhausted.
        """
        offset = 0
        while True:
            page = await self.list_markets(
                active=active,
                closed=closed,
                limit=page_size,
                offset=offset,
            )
            if not page:
                return
            for market in page:
                yield market
            if len(page) < page_size:
                return
            offset += page_size

    async def aclose(self) -> None:
        """Close the underlying HTTP client (no-op if shared)."""
        if self._owns_http:
            await self._http.aclose()


def _parse_list[T: (Event, Market)](
    payload: dict[str, Any] | list[Any],
    model: type[T],
) -> list[T]:
    """Validate a JSON array payload into a list of ``model`` instances.

    Args:
        payload: Decoded JSON from the gamma endpoint.
        model: The pydantic class to validate each item against.

    Returns:
        A list of validated models (empty if ``payload`` was empty).

    Raises:
        TypeError: If ``payload`` is not a JSON array.
    """
    if not isinstance(payload, list):
        msg = f"expected JSON array for {model.__name__} listing, got {type(payload).__name__}"
        raise TypeError(msg)
    return [model.model_validate(item) for item in payload]
