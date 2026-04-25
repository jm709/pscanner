"""Smart-money detector — track high-winrate wallets and alert on new positions.

The detector runs two cooperating loops inside a single :class:`Detector` task:

* ``_refresh_loop`` — periodically pulls the leaderboard, recomputes each
  candidate wallet's winrate from their closed positions, and persists the
  qualifying wallets via :class:`TrackedWalletsRepo`.
* ``_poll_loop`` — periodically polls the open positions of every tracked
  wallet, diffs them against the last :class:`PositionSnapshotsRepo` snapshot,
  and emits an :class:`Alert` whenever a position grows by enough USD to clear
  ``new_position_min_usd``.

Bootstrap semantics: the first observation of any ``(wallet, market, side)``
tuple is recorded silently to the snapshots repo. Alerts only fire on later
observations whose USD growth meets the threshold. This avoids a flood of
"new position" alerts on cold start when every existing position would
otherwise look brand new.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import SmartMoneyConfig
from pscanner.poly.data import DataClient
from pscanner.poly.models import LeaderboardEntry, Position
from pscanner.store.repo import (
    PositionSnapshotsRepo,
    TrackedWallet,
    TrackedWalletsRepo,
)

_LOG = structlog.get_logger(__name__)


class SmartMoneyDetector:
    """Smart-money signal detector.

    Implements the :class:`pscanner.detectors.base.Detector` protocol. Wires
    leaderboard refresh and position polling together against the data API,
    the tracked-wallets repo, the position-snapshots repo, and the shared
    alert sink.
    """

    name = "smart_money"

    def __init__(
        self,
        *,
        config: SmartMoneyConfig,
        data_client: DataClient,
        tracked_repo: TrackedWalletsRepo,
        snapshots_repo: PositionSnapshotsRepo,
    ) -> None:
        """Build a detector wired to its collaborators.

        Args:
            config: Smart-money thresholds (winrate, USD floor, intervals).
            data_client: Async client for ``data-api.polymarket.com``.
            tracked_repo: Repo holding qualified wallets.
            snapshots_repo: Repo holding the last-seen size per (wallet, market, side).
        """
        self._config = config
        self._data_client = data_client
        self._tracked_repo = tracked_repo
        self._snapshots_repo = snapshots_repo

    async def run(self, sink: AlertSink) -> None:
        """Run the refresh and poll loops concurrently until cancelled.

        Args:
            sink: The shared alert sink that all detectors emit to.
        """
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._refresh_loop())
            tg.create_task(self._poll_loop(sink))

    async def _refresh_loop(self) -> None:
        """Drive :meth:`_refresh_tracked_wallets` on the configured cadence."""
        interval = self._config.refresh_interval_seconds
        while True:
            try:
                await self._refresh_tracked_wallets()
            except Exception:
                _LOG.exception("smart_money.refresh_failed")
            await asyncio.sleep(interval)

    async def _poll_loop(self, sink: AlertSink) -> None:
        """Drive :meth:`poll_positions` on the configured cadence."""
        interval = self._config.position_poll_interval_seconds
        while True:
            try:
                await self.poll_positions(sink)
            except Exception:
                _LOG.exception("smart_money.poll_failed")
            await asyncio.sleep(interval)

    async def _refresh_tracked_wallets(self) -> None:
        """Recompute and persist tracked wallets from the live leaderboard."""
        entries = await self._data_client.get_leaderboard(
            period="all",
            limit=self._config.leaderboard_top_n,
        )
        for entry in entries:
            await self.refresh_one_wallet(entry)

    async def refresh_one_wallet(self, entry: LeaderboardEntry) -> None:
        """Evaluate a single leaderboard entry and upsert if it qualifies.

        Args:
            entry: Leaderboard row to score against the configured thresholds.
        """
        closed = await self._data_client.get_closed_positions(entry.proxy_wallet)
        count = len(closed)
        if count < self._config.min_resolved_positions:
            return
        wins = sum(1 for position in closed if position.won)
        winrate = wins / count
        if winrate < self._config.min_winrate:
            return
        self._tracked_repo.upsert(
            address=entry.proxy_wallet,
            closed_position_count=count,
            closed_position_wins=wins,
            winrate=winrate,
            leaderboard_pnl=entry.pnl,
        )

    async def poll_positions(self, sink: AlertSink) -> None:
        """Diff every tracked wallet's open positions and emit alerts.

        Args:
            sink: Shared alert sink that receives any new-position alerts.
        """
        tracked = self._tracked_repo.list_active(
            self._config.min_winrate,
            self._config.min_resolved_positions,
        )
        for wallet in tracked:
            try:
                positions = await self._data_client.get_positions(wallet.address)
            except Exception:
                _LOG.exception("smart_money.get_positions_failed", wallet=wallet.address)
                continue
            for position in positions:
                await self._handle_position(sink, wallet, position)

    async def _handle_position(
        self,
        sink: AlertSink,
        wallet: TrackedWallet,
        position: Position,
    ) -> None:
        """Diff a single position against its snapshot and emit if it grew.

        On the first observation of a ``(wallet, market, side)`` tuple
        (``prev is None``) the snapshot is upserted silently — no alert is
        emitted, since cold-start data isn't a real "new position" event.
        """
        side = position.outcome.lower()
        condition_id = position.condition_id
        prev = self._snapshots_repo.previous_size(wallet.address, condition_id, side)
        delta_size = position.size - (prev or 0.0)
        delta_usd = delta_size * position.avg_price
        if prev is not None and delta_usd >= self._config.new_position_min_usd:
            alert = _build_alert(wallet, position, side, prev, delta_usd)
            await sink.emit(alert)
        self._snapshots_repo.upsert(
            address=wallet.address,
            market_id=condition_id,
            side=side,
            size=position.size,
            avg_price=position.avg_price,
        )


def _build_alert(
    wallet: TrackedWallet,
    position: Position,
    side: str,
    prev: float | None,
    delta_usd: float,
) -> Alert:
    """Construct the smart-money :class:`Alert` for a new/grown position."""
    now = int(time.time())
    day = time.strftime("%Y%m%d", time.gmtime())
    alert_key = f"smart:{wallet.address}:{position.condition_id}:{side}:{day}"
    title = f"smart-money {wallet.address[:8]} +{side} {position.title or position.condition_id}"
    body = {
        "wallet": wallet.address,
        "market_title": position.title,
        "condition_id": position.condition_id,
        "side": side,
        "new_size": position.size,
        "prev_size": prev or 0.0,
        "delta_usd": delta_usd,
        "winrate": wallet.winrate,
        "closed_position_count": wallet.closed_position_count,
    }
    return Alert(
        detector="smart_money",
        alert_key=alert_key,
        severity="med",
        title=title,
        body=body,
        created_at=now,
    )
