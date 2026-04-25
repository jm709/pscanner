"""Smart-money detector — track edge-positive wallets and alert on new positions.

The detector runs two cooperating loops inside a single :class:`Detector` task:

* ``_refresh_loop`` — periodically pulls the leaderboard, recomputes each
  candidate wallet's edge metrics from their closed positions, and persists
  the qualifying wallets via :class:`TrackedWalletsRepo`.
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
from dataclasses import dataclass

import structlog

from pscanner.alerts.models import Alert, Severity
from pscanner.alerts.sink import AlertSink
from pscanner.config import SmartMoneyConfig
from pscanner.poly.data import DataClient
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position
from pscanner.store.repo import (
    PositionSnapshotsRepo,
    TrackedWallet,
    TrackedWalletsRepo,
)

_LOG = structlog.get_logger(__name__)

# Severity thresholds for the score (delta_usd / $10k * mean_edge).
_SEVERITY_HIGH_THRESHOLD = 0.5
_SEVERITY_MED_THRESHOLD = 0.1


@dataclass(frozen=True, slots=True)
class _WalletMetrics:
    """Aggregated edge metrics for a wallet's resolved positions."""

    count: int
    wins: int
    winrate: float
    mean_edge: float
    weighted_edge: float
    excess_pnl_usd: float
    total_stake_usd: float


def _compute_metrics(closed: list[ClosedPosition]) -> _WalletMetrics:
    """Compute edge-based skill metrics from a wallet's closed positions.

    Skips degenerate positions where avg_price <= 0 or >= 1 (no information).
    excess_pnl_usd is realized PnL summed: market-rate expected PnL is 0 by
    construction, so realized PnL IS the dollar alpha.
    """
    edges: list[float] = []
    weighted_edge_sum = 0.0
    weight_sum = 0.0
    realized_pnl_sum = 0.0
    wins = 0
    for p in closed:
        if p.avg_price <= 0 or p.avg_price >= 1:
            continue
        outcome = 1.0 if p.won else 0.0
        edge = outcome - p.avg_price
        stake_usd = p.size * p.avg_price
        edges.append(edge)
        weighted_edge_sum += edge * stake_usd
        weight_sum += stake_usd
        realized_pnl_sum += p.realized_pnl or 0.0
        if p.won:
            wins += 1
    count = len(edges)
    if count == 0:
        return _WalletMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean_edge = sum(edges) / count
    weighted_edge = weighted_edge_sum / weight_sum if weight_sum > 0 else mean_edge
    return _WalletMetrics(
        count=count,
        wins=wins,
        winrate=wins / count,
        mean_edge=mean_edge,
        weighted_edge=weighted_edge,
        excess_pnl_usd=realized_pnl_sum,
        total_stake_usd=weight_sum,
    )


def _severity(delta_usd: float, mean_edge: float | None) -> Severity:
    """Grade a smart-money alert by combined size and skill.

    score = (delta_usd / 10_000) * max(mean_edge or 0, 0)
      score >= 0.5 -> high (e.g., $10k position from a 5%+ edge wallet)
      score >= 0.1 -> med
      else         -> low
    """
    edge = 0.0 if mean_edge is None or mean_edge < 0 else mean_edge
    score = (delta_usd / 10_000.0) * edge
    if score >= _SEVERITY_HIGH_THRESHOLD:
        return "high"
    if score >= _SEVERITY_MED_THRESHOLD:
        return "med"
    return "low"


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
            config: Smart-money thresholds (edge, USD floor, intervals).
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
        metrics = _compute_metrics(closed)
        if metrics.count < self._config.min_resolved_positions:
            return
        if metrics.mean_edge < self._config.min_edge:
            return
        if metrics.excess_pnl_usd < self._config.min_excess_pnl_usd:
            return
        self._tracked_repo.upsert(
            address=entry.proxy_wallet,
            closed_position_count=metrics.count,
            closed_position_wins=metrics.wins,
            winrate=metrics.winrate,
            leaderboard_pnl=entry.pnl,
            mean_edge=metrics.mean_edge,
            weighted_edge=metrics.weighted_edge,
            excess_pnl_usd=metrics.excess_pnl_usd,
            total_stake_usd=metrics.total_stake_usd,
        )

    async def poll_positions(self, sink: AlertSink) -> None:
        """Diff every tracked wallet's open positions and emit alerts.

        Args:
            sink: Shared alert sink that receives any new-position alerts.
        """
        tracked = self._tracked_repo.list_active(
            min_edge=self._config.min_edge,
            min_excess_pnl_usd=self._config.min_excess_pnl_usd,
            min_resolved=self._config.min_resolved_positions,
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
        "mean_edge": wallet.mean_edge,
        "excess_pnl_usd": wallet.excess_pnl_usd,
        "closed_position_count": wallet.closed_position_count,
    }
    return Alert(
        detector="smart_money",
        alert_key=alert_key,
        severity=_severity(delta_usd, wallet.mean_edge),
        title=title,
        body=body,
        created_at=now,
    )
