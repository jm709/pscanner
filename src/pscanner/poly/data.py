"""Async client for ``data-api.polymarket.com`` (positions, activity, leaderboard).

Wave 1 freezes only the public method signatures; Wave 2's ``data-client`` agent
implements the bodies.
"""

from __future__ import annotations

from typing import Any, Literal

from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position


class DataClient:
    """Typed wrapper over the data REST endpoints."""

    def __init__(self, http: PolyHttpClient | None = None, *, rpm: int = 50) -> None:
        """Build a client over a shared or freshly-created ``PolyHttpClient``.

        Args:
            http: An existing ``PolyHttpClient`` instance. If ``None``, the
                client constructs its own bound to ``data-api.polymarket.com``.
            rpm: Per-host rate limit (only used when ``http`` is ``None``).
        """
        raise NotImplementedError("Wave 2: data-client")

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
        raise NotImplementedError("Wave 2: data-client")

    async def get_closed_positions(
        self,
        address: str,
        *,
        limit: int = 500,
    ) -> list[ClosedPosition]:
        """Return a wallet's resolved (closed) positions for winrate computation.

        Args:
            address: 0x-prefixed proxy wallet address.
            limit: Max number of positions to fetch.

        Returns:
            A list of ``ClosedPosition`` models, newest-first.
        """
        raise NotImplementedError("Wave 2: data-client")

    async def get_activity(
        self,
        address: str,
        *,
        limit: int = 500,
        type: str | None = None,  # noqa: A002  # mirrors the API query param name
    ) -> list[dict[str, Any]]:
        """Return raw activity events for a wallet (heterogeneous shape).

        The data API returns mixed event types (TRADE, REWARD, SPLIT, MERGE, etc.)
        with shape that varies by type, so the return value is left as a list of
        dicts. Callers that need a typed view should re-parse downstream.

        Args:
            address: 0x-prefixed proxy wallet address.
            limit: Max number of events to fetch.
            type: Optional filter for a single activity type.

        Returns:
            A list of raw JSON event dicts.
        """
        raise NotImplementedError("Wave 2: data-client")

    async def get_first_activity_timestamp(self, address: str) -> int | None:
        """Return the unix-seconds timestamp of a wallet's earliest activity.

        Pages backwards through the activity stream to the oldest event.

        Args:
            address: 0x-prefixed proxy wallet address.

        Returns:
            Unix seconds of the earliest event, or ``None`` if the wallet has no
            activity (or pagination fails irrecoverably).
        """
        raise NotImplementedError("Wave 2: data-client")

    async def get_leaderboard(
        self,
        *,
        period: Literal["day", "week", "all"] = "all",
        limit: int = 200,
    ) -> list[LeaderboardEntry]:
        """Return the top-N traders by realised PnL over the chosen period.

        Args:
            period: Leaderboard window — ``day``, ``week``, or ``all``-time.
            limit: How many entries to fetch (server-capped).

        Returns:
            A list of ``LeaderboardEntry`` models, ordered best-first.
        """
        raise NotImplementedError("Wave 2: data-client")

    async def aclose(self) -> None:
        """Close the underlying HTTP client (no-op if shared)."""
        raise NotImplementedError("Wave 2: data-client")
