"""Position-snapshot collector — append-only history of watched-wallet positions.

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
from pscanner.store.repo import WalletPositionsHistoryRepo


class PositionCollector:
    """Periodically snapshots every watched wallet's open positions.

    Wave 2 will fetch ``data_client.get_positions(...)`` for each address in
    the registry on a fixed cadence and append every row to
    :class:`WalletPositionsHistoryRepo`. The composite primary key on
    ``(wallet, condition_id, outcome, snapshot_at)`` provides idempotency for
    overlapping polls.
    """

    name: str = "position_collector"

    def __init__(
        self,
        *,
        registry: WatchlistRegistry,
        data_client: DataClient,
        positions_repo: WalletPositionsHistoryRepo,
        snapshot_interval_seconds: float = 300.0,
    ) -> None:
        """Build the collector.

        Args:
            registry: In-memory watchlist of wallets to snapshot.
            data_client: REST client for ``/positions`` queries.
            positions_repo: Append-only repo for history rows.
            snapshot_interval_seconds: Cadence for full-watchlist snapshots.
        """
        self._registry = registry
        self._data_client = data_client
        self._positions_repo = positions_repo
        self._snapshot_interval_seconds = snapshot_interval_seconds

    async def run(self, stop_event: asyncio.Event) -> None:
        """Loop: snapshot all wallets every interval until ``stop_event`` is set."""
        raise NotImplementedError("DC-2 Wave 2: positions")

    async def snapshot_all_wallets(self) -> int:
        """Snapshot every watched wallet once.

        Returns:
            Number of newly-inserted history rows across every wallet.
        """
        raise NotImplementedError("DC-2 Wave 2: positions")
