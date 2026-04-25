"""Market snapshot collector — append-only per-market history.

The collector paginates ``gamma /markets`` bounded by ``snapshot_max`` and
writes one row per market to :class:`MarketSnapshotsRepo` with a single
shared ``snapshot_at`` so the full sweep can be reconstructed as a
point-in-time view.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Market
from pscanner.store.repo import MarketSnapshot, MarketSnapshotsRepo

_LOG = structlog.get_logger(__name__)


class MarketCollector:
    """Periodically snapshots every active market's state.

    Paginates gamma ``/markets`` bounded by ``snapshot_max`` to keep one
    cycle's work tractable. Each call writes one row per market with a single
    ``snapshot_at`` value; dedupe-on-PK in the repo collapses re-snapshots
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
        """Build the collector.

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

        Per-iteration exceptions from :meth:`snapshot_all_markets` are logged
        and swallowed so a transient upstream hiccup does not kill the loop.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        while not stop_event.is_set():
            try:
                await self.snapshot_all_markets()
            except Exception:
                _LOG.exception("markets.snapshot_iteration_failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval)
            except TimeoutError:
                continue
            return

    async def snapshot_all_markets(self) -> int:
        """Snapshot every active market once, capped by ``snapshot_max``.

        Returns:
            Number of newly-inserted snapshot rows for this sweep.
        """
        snapshot_at = int(time.time())
        inserted = 0
        async for market in self._gamma.iter_markets(active=True, closed=False):
            if inserted >= self._max:
                break
            if not market.id:
                continue
            if self._persist_market(market, snapshot_at=snapshot_at):
                inserted += 1
        _LOG.info(
            "markets.snapshot_complete",
            inserted=inserted,
            snapshot_at=snapshot_at,
        )
        return inserted

    def _persist_market(self, market: Market, *, snapshot_at: int) -> bool:
        """Persist one market snapshot row; return True on insert.

        Per-row exceptions (insert errors, JSON encoding failures) are logged
        and swallowed so a single bad market cannot break the whole sweep.

        Args:
            market: Source-of-truth market model from gamma.
            snapshot_at: Shared sweep timestamp (unix seconds).

        Returns:
            ``True`` if the row was newly inserted, ``False`` otherwise.
        """
        snapshot = MarketSnapshot(
            market_id=market.id,
            event_id=market.event_id,
            outcome_prices_json=json.dumps(list(market.outcome_prices or [])),
            liquidity_usd=market.liquidity,
            volume_usd=market.volume,
            active=bool(market.active),
            snapshot_at=snapshot_at,
        )
        try:
            return self._repo.insert(snapshot)
        except Exception:
            _LOG.exception("markets.insert_failed", market_id=market.id)
            return False
