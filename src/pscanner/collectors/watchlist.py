"""Watchlist registry and syncer — DC-1 Wave B implementation.

The registry is the in-memory source of truth for which wallet addresses are
recorded by :mod:`pscanner.collectors.trades`. The syncer keeps the registry
aligned with smart-money entries (mirrored from ``tracked_wallets``) and
whale-alert events (subscribed via the alert sink).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.store.repo import TrackedWalletsRepo, WatchlistRepo

_LOG = structlog.get_logger(__name__)
_VALID_SOURCES: frozenset[str] = frozenset({"smart_money", "whale_alert", "manual"})


class WatchlistRegistry:
    """In-memory watchlist with DB-backed persistence and a change-listener API.

    Source of truth for which wallet addresses we record data for. Updated by
    ``WatchlistSyncer`` (smart-money + whale-alert mirroring) and via the CLI
    ``pscanner watch``/``unwatch`` commands.

    Thread-safe: the in-memory set and callback list are guarded by a lock so
    the trade-collector's WS handler can read concurrently with writes from
    the syncer or the CLI.
    """

    def __init__(self, repo: WatchlistRepo) -> None:
        """Bind the registry to a repo and hydrate from active rows.

        Args:
            repo: SQLite-backed watchlist repository.
        """
        self._repo = repo
        self._addresses: set[str] = set()
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        """Re-read the active set from the DB into memory.

        Subscribed callbacks are NOT fired on reload — this is a snapshot
        refresh, not an "added" event.
        """
        rows = self._repo.list_active()
        with self._lock:
            self._addresses = {entry.address for entry in rows}

    def __contains__(self, address: str) -> bool:
        """Return whether ``address`` is currently active (thread-safe)."""
        with self._lock:
            return address in self._addresses

    def addresses(self) -> set[str]:
        """Return a copy of the active address set (thread-safe)."""
        with self._lock:
            return set(self._addresses)

    def add(
        self,
        *,
        address: str,
        source: str,
        reason: str | None = None,
    ) -> bool:
        """Persist an entry, refresh the in-memory set, and notify subscribers.

        If the row already exists in the DB but is inactive, it is reactivated
        (via ``repo.set_active(address, True)``) and treated as a newly-added
        event for callback purposes.

        Args:
            address: 0x-prefixed proxy wallet address.
            source: One of ``smart_money``, ``whale_alert``, or ``manual``.
            reason: Optional free-form provenance note.

        Returns:
            ``True`` if a new row was inserted, ``False`` if the address was
            already present (and thus a re-add was a no-op).

        Raises:
            ValueError: If ``source`` is not one of the supported values.
        """
        if source not in _VALID_SOURCES:
            msg = f"invalid watchlist source {source!r}; expected one of {sorted(_VALID_SOURCES)}"
            raise ValueError(msg)

        inserted = self._repo.upsert(address=address, source=source, reason=reason)
        if inserted:
            self._record_addition(address)
            return True

        existing = self._repo.get(address)
        if existing is not None and not existing.active:
            self._repo.set_active(address, active=True)
            self._record_addition(address)
        return False

    def deactivate(self, address: str) -> None:
        """Set ``active=0`` for ``address`` and drop it from the live set.

        No callbacks fire — deactivation is not an "added" event.
        """
        self._repo.set_active(address, active=False)
        with self._lock:
            self._addresses.discard(address)

    def subscribe(self, callback: Callable[[str], None]) -> None:
        """Register a listener fired with the new address on each ``add``.

        Args:
            callback: Synchronous callable invoked with the address whenever a
                new (or reactivated) entry is added. Must not block.
        """
        with self._lock:
            self._callbacks.append(callback)

    def _record_addition(self, address: str) -> None:
        """Add ``address`` to the live set and fire callbacks outside the lock."""
        with self._lock:
            self._addresses.add(address)
            callbacks = list(self._callbacks)
        for callback in callbacks:
            callback(address)


class WatchlistSyncer:
    """Mirrors smart-money + whale-alert sources into the WatchlistRegistry.

    Subscribes to the alert sink for whale alerts at construction, and
    periodically mirrors the ``tracked_wallets`` table for smart-money
    entries.
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
        """Wire the syncer to the registry, the smart-money repo, and the sink.

        Args:
            registry: In-memory watchlist this syncer mirrors into.
            tracked_repo: Source of smart-money wallet rows.
            sink: Alert sink the syncer subscribes to for whale events.
            sync_interval_seconds: Cadence for mirroring smart-money entries.
        """
        self._registry = registry
        self._tracked_repo = tracked_repo
        self._sink = sink
        self._sync_interval_seconds = sync_interval_seconds
        sink.subscribe(self._on_alert)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Mirror loop. Returns when ``stop_event`` is set.

        On each iteration the smart-money mirror runs, then the loop sleeps
        for ``sync_interval_seconds`` (or returns early if the stop event
        fires). Exceptions raised by the sync step are logged and swallowed
        so a transient DB hiccup does not kill the loop.
        """
        while not stop_event.is_set():
            try:
                await self.sync_smart_money()
            except Exception:
                _LOG.exception("watchlist_sync.iteration_failed")
            if await self._wait_or_stop(stop_event, self._sync_interval_seconds):
                return

    async def sync_smart_money(self) -> None:
        """Mirror every tracked wallet into the registry as ``smart_money``.

        Public delegating wrapper invoked by the scheduler's single-shot
        path and by :meth:`run`. Idempotent.
        """
        await self._sync_smart_money()

    async def _sync_smart_money(self) -> None:
        """Mirror every tracked wallet into the registry as ``smart_money``.

        Idempotent: re-adding an already-watched wallet is a no-op.
        """
        wallets = self._tracked_repo.list_all()
        for wallet in wallets:
            self._registry.add(
                address=wallet.address,
                source="smart_money",
                reason=f"winrate {wallet.winrate:.2f}",
            )

    def _on_alert(self, alert: Alert) -> None:
        """Add a whale-alert wallet to the registry.

        Non-whale alerts are ignored. Malformed payloads (missing ``wallet``
        key) are logged and swallowed so a single bad alert never breaks the
        sink fan-out.
        """
        if alert.detector != "whales":
            return
        try:
            wallet = alert.body["wallet"]
            self._registry.add(
                address=wallet,
                source="whale_alert",
                reason=f"whale alert {alert.alert_key}",
            )
        except KeyError:
            _LOG.warning("watchlist_sync.alert_missing_wallet", alert_key=alert.alert_key)
        except Exception:
            _LOG.exception("watchlist_sync.alert_handler_failed", alert_key=alert.alert_key)

    @staticmethod
    async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
        """Wait up to ``seconds`` for the stop event.

        Returns:
            ``True`` if the stop event was set during the wait, ``False`` if
            the timeout elapsed first.
        """
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True
