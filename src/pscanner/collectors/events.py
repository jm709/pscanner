"""Event snapshot collector — append-only per-event history.

Wave 1 contract; Wave 2 fills in the implementation. The collector paginates
``gamma /events`` bounded by ``snapshot_max`` and writes one row per event
to :class:`EventSnapshotsRepo` with a single shared ``snapshot_at`` so the
full sweep can be reconstructed as a point-in-time view.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import EventSlug
from pscanner.poly.models import Event
from pscanner.store.repo import EventSnapshot, EventSnapshotsRepo, EventTagCacheRepo

_LOG = structlog.get_logger(__name__)


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
        event_tag_cache: EventTagCacheRepo,
        snapshot_interval_seconds: float = 900.0,
        snapshot_max: int = 2000,
    ) -> None:
        """Build the collector.

        Args:
            gamma_client: Gamma REST client for ``/events`` queries.
            events_repo: Append-only, dedupe-on-PK repo for snapshots.
            event_tag_cache: Tag cache; updated alongside each snapshot so
                downstream detectors can categorise events without hitting
                gamma.
            snapshot_interval_seconds: Cadence between full sweeps.
            snapshot_max: Hard cap on events fetched per sweep.
        """
        self._gamma = gamma_client
        self._repo = events_repo
        self._event_tag_cache = event_tag_cache
        self._interval = snapshot_interval_seconds
        self._max = snapshot_max

    async def run(self, stop_event: asyncio.Event) -> None:
        """Loop: snapshot all active events every interval until stopped.

        Per-iteration exceptions from :meth:`snapshot_all_events` are logged
        and swallowed so a transient upstream hiccup does not kill the loop.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        while not stop_event.is_set():
            try:
                await self.snapshot_all_events()
            except Exception:
                _LOG.exception("events.snapshot_iteration_failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval)
            except TimeoutError:
                continue
            return

    async def snapshot_all_events(self) -> int:
        """Snapshot every active event once, bounded by ``snapshot_max``.

        Iterates ``gamma.iter_events(active=True, closed=False)``, builds an
        :class:`EventSnapshot` per row using a shared ``snapshot_at``, and
        inserts via the repo. Per-row insert failures are logged and skipped
        so a single broken row cannot break the sweep.

        Returns:
            Number of newly-inserted snapshot rows for this sweep.
        """
        snapshot_at = int(time.time())
        inserted = 0
        async for event in self._gamma.iter_events(active=True, closed=False):
            if inserted >= self._max:
                break
            if not event.id:
                continue
            snapshot = _build_snapshot(event, snapshot_at=snapshot_at)
            if self._try_insert(snapshot):
                inserted += 1
            self._cache_tags(event)
        _LOG.info(
            "events.snapshot_complete",
            inserted=inserted,
            snapshot_at=snapshot_at,
        )
        return inserted

    def _try_insert(self, snapshot: EventSnapshot) -> bool:
        """Insert one snapshot, swallowing per-row exceptions.

        Args:
            snapshot: Snapshot row to persist.

        Returns:
            ``True`` iff the repo reports the row was newly inserted.
        """
        try:
            return self._repo.insert(snapshot)
        except Exception:
            _LOG.exception(
                "events.insert_failed",
                event_id=snapshot.event_id,
                snapshot_at=snapshot.snapshot_at,
            )
            return False

    def _cache_tags(self, event: Event) -> None:
        """Upsert ``event.tags`` into the tag cache, swallowing failures.

        The cache is keyed on ``event.slug`` because closed-position payloads
        expose ``eventSlug`` rather than a numeric event id; using slugs
        means the smart-money categoriser can hit the cache without a
        secondary id-lookup step. When the gamma payload is missing a slug
        we fall back to the numeric event id so a partial row still
        populates the cache; the cache key remains an :class:`EventSlug`
        for type uniformity.

        Args:
            event: Validated gamma event whose tags should be cached.
        """
        key = event.slug or (EventSlug(event.id) if event.id else None)
        if not key:
            return
        try:
            self._event_tag_cache.upsert(key, list(event.tags))
        except Exception:
            _LOG.exception("events.tag_cache_upsert_failed", event_slug=key)


def _build_snapshot(event: Event, *, snapshot_at: int) -> EventSnapshot:
    """Project a gamma ``Event`` into an :class:`EventSnapshot`.

    Args:
        event: Validated gamma event model.
        snapshot_at: Shared sweep timestamp (unix seconds).

    Returns:
        A populated ``EventSnapshot`` ready for insertion.
    """
    return EventSnapshot(
        event_id=event.id,
        title=event.title or "",
        slug=event.slug if event.slug else EventSlug(""),
        liquidity_usd=event.liquidity,
        volume_usd=event.volume,
        active=bool(event.active),
        closed=bool(event.closed),
        market_count=len(event.markets or []),
        snapshot_at=snapshot_at,
    )
