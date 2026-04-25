"""Market snapshot collector — append-only per-market history.

Wave 1 contract; Wave 2 fills in the implementation. The collector paginates
``gamma /markets`` bounded by ``snapshot_max`` and writes one row per market
to :class:`MarketSnapshotsRepo` with a single shared ``snapshot_at`` so the
full sweep can be reconstructed as a point-in-time view.
"""

from __future__ import annotations

import asyncio

from pscanner.poly.gamma import GammaClient
from pscanner.store.repo import MarketSnapshotsRepo


class MarketCollector:
    """Periodically snapshots every active market's state.

    Paginates gamma ``/markets`` bounded by ``snapshot_max`` to keep one
    cycle's work tractable. Each call writes one row per market with a single
    ``snapshot_at`` value, dedupe-on-PK in the repo collapses re-snapshots
    within the same second.
    """

    name: str = "market_collector"

    def __init__(
        self,
        *,
        gamma_client: GammaClient,
        markets_repo: MarketSnapshotsRepo,
        snapshot_interval_seconds: float = 300.0,
        snapshot_max: int = 5000,
    ) -> None:
        """Build the collector. Wave 2 fills in the body.

        Args:
            gamma_client: Gamma REST client for ``/markets`` queries.
            markets_repo: Append-only, dedupe-on-PK repo for snapshots.
            snapshot_interval_seconds: Cadence between full sweeps.
            snapshot_max: Hard cap on markets fetched per sweep.
        """
        self._gamma = gamma_client
        self._repo = markets_repo
        self._interval = snapshot_interval_seconds
        self._max = snapshot_max

    async def run(self, stop_event: asyncio.Event) -> None:
        """Loop: snapshot all active markets every interval until stopped.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        raise NotImplementedError("DC-3 Wave 2: markets")

    async def snapshot_all_markets(self) -> int:
        """Snapshot every active market once.

        Returns:
            Number of newly-inserted snapshot rows for this sweep.
        """
        raise NotImplementedError("DC-3 Wave 2: markets")
