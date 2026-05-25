"""Periodic backfill of market_cache + event_tag_cache for paper_trades.

Closes the structural blind spot where ``MarketCollector`` and
``EventCollector`` both sweep ``active=true, closed=false`` from gamma:
condition_ids that were already-resolved when our daemon first saw them
never land in our local caches, leaving copy-trading analysis blind.

The collector is bounded by the open + resolved paper-trade set (small
relative to the full Polymarket catalogue) and rate-limited by the
shared gamma + data RPM token buckets, so the per-cycle cost is
predictable. Two phases per ``poll_once``:

1. Backfill ``market_cache`` for every paper_trade ``condition_id`` that
   either lacks a row or has a row with ``event_slug IS NULL`` (the
   refresh has a chance to fill ``event_slug`` if gamma's response now
   carries the event nesting).
2. Backfill ``event_tag_cache`` for every ``event_slug`` referenced by a
   paper-trade-linked ``market_cache`` row that has no tag-cache entry.

Idempotent: rows already populated are skipped on subsequent cycles.
"""

from __future__ import annotations

import sqlite3

import structlog

from pscanner.collectors.base import PollingCollector
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import ConditionId, EventSlug
from pscanner.store.repo import EventTagCacheRepo, MarketCacheRepo
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row

_LOG = structlog.get_logger(__name__)


class PaperTradesMetadataCollector(PollingCollector):
    """Keeps market_cache + event_tag_cache truthful for paper-trade markets.

    Reads gap-detection queries directly from the daemon DB connection.
    Per-row failures (gamma slug miss, transient exception) are logged
    and skipped — one bad row never blocks the rest of the cycle.
    """

    name: str = "paper_trades_metadata"
    log_event_iteration_failed: str = "paper_trades_metadata.poll_failed"

    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        market_cache: MarketCacheRepo,
        event_tag_cache: EventTagCacheRepo,
        data_client: DataClient,
        gamma_client: GammaClient,
        interval_seconds: float = 300.0,
    ) -> None:
        """Bind dependencies and configure the poll cadence.

        Args:
            db: Daemon SQLite connection for gap-detection queries.
            market_cache: Where refreshed market rows are written.
            event_tag_cache: Where backfilled event tags are written.
            data_client: Used by the refresh helper's first hop.
            gamma_client: Used by the refresh helper and for
                ``get_event_by_slug`` in phase 2.
            interval_seconds: Cadence between cycles. Default 300s
                mirrors PaperResolver.
        """
        super().__init__(interval_seconds=interval_seconds)
        self._db = db
        self._market_cache = market_cache
        self._event_tag_cache = event_tag_cache
        self._data_client = data_client
        self._gamma_client = gamma_client

    async def poll_once(self) -> None:
        """Run both backfill phases once. See module docstring for shape."""
        market_ok, market_fail = await self._backfill_market_cache_phase()
        tags_ok, tags_fail = await self._backfill_event_tag_cache_phase()
        if market_ok or market_fail or tags_ok or tags_fail:
            _LOG.info(
                "paper_trades_metadata.cycle_completed",
                market_cache_ok=market_ok,
                market_cache_fail=market_fail,
                event_tag_cache_ok=tags_ok,
                event_tag_cache_fail=tags_fail,
            )

    async def _backfill_market_cache_phase(self) -> tuple[int, int]:
        """Phase 1: refresh market_cache rows that are missing or slug-less.

        Returns ``(ok, fail)`` count for this cycle.
        """
        cond_ids = self._cids_needing_market_cache_refresh()
        ok = 0
        fail = 0
        for cid in cond_ids:
            result = await refresh_market_cache_row(
                data_client=self._data_client,
                gamma_client=self._gamma_client,
                market_cache=self._market_cache,
                condition_id=cid,
            )
            if result:
                ok += 1
            else:
                fail += 1
        return ok, fail

    async def _backfill_event_tag_cache_phase(self) -> tuple[int, int]:
        """Phase 2: populate event_tag_cache for missing slugs. Returns ``(ok, fail)``."""
        slugs = self._slugs_needing_tag_cache()
        ok = 0
        fail = 0
        for slug in slugs:
            try:
                event = await self._gamma_client.get_event_by_slug(slug)
            except Exception:
                _LOG.warning(
                    "paper_trades_metadata.event_fetch_failed",
                    event_slug=slug,
                    exc_info=True,
                )
                fail += 1
                continue
            if event is None:
                fail += 1
                continue
            try:
                self._event_tag_cache.upsert(slug, list(event.tags))
            except Exception:
                _LOG.warning(
                    "paper_trades_metadata.tag_cache_upsert_failed",
                    event_slug=slug,
                    exc_info=True,
                )
                fail += 1
                continue
            ok += 1
        return ok, fail

    def _cids_needing_market_cache_refresh(self) -> list[ConditionId]:
        """Distinct paper_trades condition_ids with no row or no event_slug."""
        rows = self._db.execute(
            """
            SELECT DISTINCT e.condition_id
              FROM paper_trades e
              LEFT JOIN market_cache mc ON mc.condition_id = e.condition_id
             WHERE e.trade_kind = 'entry'
               AND (mc.condition_id IS NULL OR mc.event_slug IS NULL)
            """,
        ).fetchall()
        return [ConditionId(r[0]) for r in rows]

    def _slugs_needing_tag_cache(self) -> list[EventSlug]:
        """Distinct event_slugs referenced by paper-trade markets, not yet cached."""
        rows = self._db.execute(
            """
            SELECT DISTINCT mc.event_slug
              FROM market_cache mc
              JOIN paper_trades e ON e.condition_id = mc.condition_id
              LEFT JOIN event_tag_cache etc ON etc.event_slug = mc.event_slug
             WHERE e.trade_kind = 'entry'
               AND mc.event_slug IS NOT NULL
               AND etc.event_slug IS NULL
            """,
        ).fetchall()
        return [EventSlug(r[0]) for r in rows]


__all__ = ["PaperTradesMetadataCollector"]
