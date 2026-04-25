"""Activity-stream collector — append-only event log per watched wallet.

Polls the public ``/activity`` endpoint for every watched wallet on a fixed
cadence and persists every event (TRADE, SPLIT, MERGE, REDEEM, CONVERT, ...)
into :class:`WalletActivityEventsRepo`. The composite primary key
``(wallet, timestamp, event_type)`` dedupes overlapping pages across polls
so the loop is naturally idempotent.

Persistence keeps the original JSON payload verbatim in ``payload_json`` so
downstream consumers can re-parse type-specific fields without forcing this
collector to know every shape upfront.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.poly.data import DataClient
from pscanner.store.repo import WalletActivityEvent, WalletActivityEventsRepo

_LOG = structlog.get_logger(__name__)
_SOURCE = "activity_api"


class ActivityCollector:
    """Periodically polls ``/activity`` for every watched wallet.

    Fetches every event type for each address in the registry on a fixed
    cadence and inserts the raw row into :class:`WalletActivityEventsRepo`.
    The composite primary key on ``(wallet, timestamp, event_type)`` dedupes
    overlapping pages.
    """

    name: str = "activity_collector"

    def __init__(
        self,
        *,
        registry: WatchlistRegistry,
        data_client: DataClient,
        activity_repo: WalletActivityEventsRepo,
        poll_interval_seconds: float = 300.0,
        activity_page_limit: int = 200,
    ) -> None:
        """Build the collector.

        Args:
            registry: In-memory watchlist of wallets to poll.
            data_client: REST client for ``/activity`` queries.
            activity_repo: Append-only, dedupe-on-PK repo for events.
            poll_interval_seconds: Cadence for full-watchlist polling cycles.
            activity_page_limit: Per-wallet ``/activity`` page size.
        """
        self._registry = registry
        self._data_client = data_client
        self._activity_repo = activity_repo
        self._poll_interval_seconds = poll_interval_seconds
        self._activity_page_limit = activity_page_limit

    async def run(self, stop_event: asyncio.Event) -> None:
        """Loop: poll all wallets every interval until ``stop_event`` is set.

        Per-iteration exceptions from :meth:`poll_all_wallets` are logged and
        swallowed so a transient upstream hiccup does not kill the loop.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        while not stop_event.is_set():
            try:
                await self.poll_all_wallets()
            except Exception:
                _LOG.exception("activity.poll_iteration_failed")
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except TimeoutError:
                continue
            return

    async def poll_all_wallets(self) -> int:
        """Poll every watched wallet once.

        Iterates a snapshot of ``registry.addresses()`` taken at the start of
        the cycle so additions made mid-cycle are picked up next time. Per-
        wallet exceptions are caught inside :meth:`_poll_wallet` so a single
        failing wallet cannot break the broader cycle.

        Returns:
            Number of newly-inserted activity rows across every wallet.
        """
        watched = sorted(self._registry.addresses())
        total = 0
        for wallet in watched:
            total += await self._poll_wallet(wallet)
        _LOG.info(
            "activity.poll.completed",
            watched_wallets=len(watched),
            inserted=total,
        )
        return total

    async def _poll_wallet(self, address: str) -> int:
        """Poll one wallet's ``/activity`` feed and persist new events.

        Args:
            address: 0x-prefixed proxy wallet to poll.

        Returns:
            Number of newly-inserted rows for this wallet (0 on error).
        """
        try:
            events = await self._data_client.get_activity(
                address,
                limit=self._activity_page_limit,
            )
        except Exception:
            _LOG.exception("activity.get_activity_failed", wallet=address)
            return 0
        recorded_at = int(time.time())
        inserted = 0
        for event in events:
            if self._persist_event(event, address=address, recorded_at=recorded_at):
                inserted += 1
        _LOG.debug(
            "activity.poll",
            wallet=address,
            events_returned=len(events),
            inserted=inserted,
        )
        return inserted

    def _persist_event(
        self,
        event: dict[str, Any],
        *,
        address: str,
        recorded_at: int,
    ) -> bool:
        """Persist one ``/activity`` event; return True on insert.

        Drops events missing ``type`` or ``timestamp`` (debug-logged) and
        skips events whose payload cannot be JSON-encoded (e.g. contains a
        ``set`` value) so a single malformed entry cannot poison the page.

        Args:
            event: Raw decoded JSON dict for one activity entry.
            address: Wallet address we polled — written to ``wallet`` column.
            recorded_at: Shared poll-cycle timestamp (unix seconds).

        Returns:
            ``True`` if the row was newly inserted, ``False`` otherwise.
        """
        event_type = event.get("type")
        if not event_type:
            _LOG.debug("activity.event.missing_type", wallet=address)
            return False
        timestamp_raw = event.get("timestamp")
        if timestamp_raw is None:
            _LOG.debug("activity.event.missing_timestamp", wallet=address)
            return False
        try:
            payload_json = json.dumps(event)
        except TypeError:
            _LOG.exception(
                "activity.event.payload_not_serialisable",
                wallet=address,
                event_type=event_type,
            )
            return False
        row = WalletActivityEvent(
            wallet=address,
            event_type=str(event_type),
            payload_json=payload_json,
            timestamp=int(timestamp_raw),
            recorded_at=recorded_at,
            source=_SOURCE,
        )
        try:
            return self._activity_repo.insert(row)
        except Exception:
            _LOG.exception(
                "activity.insert_failed",
                wallet=address,
                event_type=event_type,
            )
            return False
