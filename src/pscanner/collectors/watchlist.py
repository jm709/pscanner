"""Watchlist registry and syncer — DC-1 Wave A contract stubs.

This module freezes the public API for the watchlist layer; Wave B fills in
the bodies. The registry is the in-memory source of truth for which wallet
addresses are recorded by :mod:`pscanner.collectors.trades`. The syncer keeps
the registry aligned with smart-money entries (mirrored from
``tracked_wallets``) and whale-alert events (subscribed via the alert sink).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pscanner.alerts.sink import AlertSink
from pscanner.store.repo import TrackedWalletsRepo, WatchlistRepo


class WatchlistRegistry:
    """In-memory watchlist with DB-backed persistence and a change-listener API.

    Source of truth for which wallet addresses we record data for. Updated by
    ``WatchlistSyncer`` (smart-money + whale-alert mirroring) and via the CLI
    ``pscanner watch``/``unwatch`` commands.
    """

    def __init__(self, repo: WatchlistRepo) -> None:
        """Bind the registry to a repo; defer load to ``reload``."""
        # wave b: hydrate self._addresses from repo.list_active().
        raise NotImplementedError("DC-1 Wave B: watchlist")

    def reload(self) -> None:
        """Re-read the active set from the DB into memory."""
        raise NotImplementedError("DC-1 Wave B: watchlist")

    def __contains__(self, address: str) -> bool:
        """Return whether ``address`` is currently active."""
        raise NotImplementedError("DC-1 Wave B: watchlist")

    def addresses(self) -> set[str]:
        """Return a copy of the active address set."""
        raise NotImplementedError("DC-1 Wave B: watchlist")

    def add(
        self,
        *,
        address: str,
        source: str,
        reason: str | None = None,
    ) -> bool:
        """Persist an entry, refresh the in-memory set, and notify subscribers.

        Returns ``True`` if a new row was inserted, ``False`` if the address
        was already present.
        """
        raise NotImplementedError("DC-1 Wave B: watchlist")

    def deactivate(self, address: str) -> None:
        """Set ``active=0`` for ``address`` and drop it from the live set."""
        raise NotImplementedError("DC-1 Wave B: watchlist")

    def subscribe(self, callback: Callable[[str], None]) -> None:
        """Register a listener fired with the new address on each ``add``.

        The callback receives the address string when an entry is added.
        """
        raise NotImplementedError("DC-1 Wave B: watchlist")


class WatchlistSyncer:
    """Keeps the WatchlistRegistry in sync with smart-money + whale-alert sources.

    Subscribes to the alert sink for whale alerts, and periodically mirrors
    the tracked_wallets table for smart-money entries.
    """

    name: str = "watchlist_sync"

    def __init__(
        self,
        *,
        registry: WatchlistRegistry,
        tracked_repo: TrackedWalletsRepo,
        sink: AlertSink,
        sync_interval_seconds: float = 60.0,
    ) -> None:
        """Wire the syncer to the registry, the smart-money repo, and the sink."""
        # wave b: store deps + register a sink.subscribe handler for whale alerts.
        raise NotImplementedError("DC-1 Wave B: watchlist")

    async def run(self, stop_event: asyncio.Event) -> None:
        """Mirror loop. Returns when ``stop_event`` is set."""
        raise NotImplementedError("DC-1 Wave B: watchlist")
