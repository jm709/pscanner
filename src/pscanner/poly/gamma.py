"""Async client for ``gamma-api.polymarket.com`` (events, markets catalogue).

Wave 1 freezes only the public method signatures and docstrings; Wave 2's
``gamma-client`` agent implements the bodies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import Event, Market


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
        raise NotImplementedError("Wave 2: gamma-client")

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
        raise NotImplementedError("Wave 2: gamma-client")

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
        raise NotImplementedError("Wave 2: gamma-client")

    async def get_event(self, event_id: str) -> Event:
        """Fetch a single event by ID.

        Args:
            event_id: Polymarket event identifier.

        Returns:
            The event.

        Raises:
            httpx.HTTPStatusError: On 404 / other HTTP errors.
        """
        raise NotImplementedError("Wave 2: gamma-client")

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
        raise NotImplementedError("Wave 2: gamma-client")

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
        raise NotImplementedError("Wave 2: gamma-client")

    async def aclose(self) -> None:
        """Close the underlying HTTP client (no-op if shared)."""
        raise NotImplementedError("Wave 2: gamma-client")
