"""Event snapshot collector — append-only per-event history.

Wave 1 contract; Wave 2 fills in the implementation. The collector paginates
``gamma /events`` bounded by ``snapshot_max`` and writes one row per event
to :class:`EventSnapshotsRepo` with a single shared ``snapshot_at`` so the
full sweep can be reconstructed as a point-in-time view.
"""

from __future__ import annotations

import asyncio

from pscanner.poly.gamma import GammaClient
from pscanner.store.repo import EventSnapshotsRepo


class EventCollector:
    """Periodically snapshots every active event's metadata.

    Paginates gamma ``/events`` bounded by ``snapshot_max`` to keep one
    cycle's work tractable. Each call writes one row per event with a single
    ``snapshot_at`` value; dedupe-on-PK in the repo collapses re-snapshots
    within the same second.
    """

    name: str = "event_collector"

    def __init__(
        self,
        *,
        gamma_client: GammaClient,
        events_repo: EventSnapshotsRepo,
        snapshot_interval_seconds: float = 900.0,
        snapshot_max: int = 2000,
    ) -> None:
        """Build the collector. Wave 2 fills in the body.

        Args:
            gamma_client: Gamma REST client for ``/events`` queries.
            events_repo: Append-only, dedupe-on-PK repo for snapshots.
            snapshot_interval_seconds: Cadence between full sweeps.
            snapshot_max: Hard cap on events fetched per sweep.
        """
        self._gamma = gamma_client
        self._repo = events_repo
        self._interval = snapshot_interval_seconds
        self._max = snapshot_max

    async def run(self, stop_event: asyncio.Event) -> None:
        """Loop: snapshot all active events every interval until stopped.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        raise NotImplementedError("DC-3 Wave 2: events")

    async def snapshot_all_events(self) -> int:
        """Snapshot every active event once.

        Returns:
            Number of newly-inserted snapshot rows for this sweep.
        """
        raise NotImplementedError("DC-3 Wave 2: events")
