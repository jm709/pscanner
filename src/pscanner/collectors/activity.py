"""Activity-stream collector — append-only event log per watched wallet.

Wave 1 contract; Wave 2 fills in the implementation. The collector is wired
into the scheduler so the orchestration plumbing — config flag, repo, run
loop registration, and ``run_once`` invocation — can be exercised before the
real polling logic lands. Construction succeeds and stores the dependencies;
the async methods raise :class:`NotImplementedError` until Wave 2 lands.
"""

from __future__ import annotations

import asyncio

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.poly.data import DataClient
from pscanner.store.repo import WalletActivityEventsRepo


class ActivityCollector:
    """Periodically polls ``/activity`` for every watched wallet.

    Wave 2 will fetch every TRADE / SPLIT / MERGE / REDEEM / CONVERT event for
    each address in the registry on a fixed cadence and insert each event
    into :class:`WalletActivityEventsRepo`. The composite primary key on
    ``(wallet, timestamp, event_type)`` dedupes overlapping pages.
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
        """Loop: poll all wallets every interval until ``stop_event`` is set."""
        raise NotImplementedError("DC-2 Wave 2: activity")

    async def poll_all_wallets(self) -> int:
        """Poll every watched wallet once.

        Returns:
            Number of newly-inserted activity rows across every wallet.
        """
        raise NotImplementedError("DC-2 Wave 2: activity")
