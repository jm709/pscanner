"""Price velocity detector — fires on rapid mid-price moves.

Polls :class:`MarketTickCollector` for recent mid-price history per asset and
emits an :class:`~pscanner.alerts.models.Alert` whenever the percent move over
``velocity_window_seconds`` exceeds ``velocity_threshold_pct`` in either
direction.

Alerts are suppressed when the latest tick has lopsided book depth (one side
a wall, the other a whisper) or insufficient USD depth — those moves are
sweep artifacts on illiquid micro-cap markets, not real signal. The alert
body is enriched with ``market_title`` and ``condition_id`` (when known) so
terminal panel alerts are actionable without a manual lookup.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import structlog

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.alerts.sink import AlertSink
from pscanner.collectors.ticks import MarketTickCollector
from pscanner.config import VelocityConfig
from pscanner.store.repo import MarketCacheRepo, MarketTick
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)
_DEDUP_BUCKET_SECONDS = 60
_HIGH_SEVERITY_MULTIPLE = 2
_MIN_MIDS_FOR_DELTA = 2


class PriceVelocityDetector:
    """Polls the tick collector for recent mid-price history; alerts on big moves.

    Reads via ``MarketTickCollector.get_recent_mids(asset_id, window_seconds)``.
    Computes ``(end_price - start_price) / start_price`` over the window. If
    the absolute move exceeds ``velocity_threshold_pct``, emits an alert with
    severity bucketed by magnitude.
    """

    name: str = "velocity"

    def __init__(
        self,
        *,
        config: VelocityConfig,
        ticks_collector: MarketTickCollector,
        market_cache: MarketCacheRepo,
        clock: Clock | None = None,
    ) -> None:
        """Build the detector.

        Args:
            config: Threshold + cadence settings (see ``VelocityConfig``).
            ticks_collector: Source of recent mid-price snapshots. Accessed
                via the frozen public API ``get_recent_mids`` and
                ``subscribed_asset_ids``.
            market_cache: Held for symmetry with other detectors; alert-body
                enrichment now reads from the tick collector's
                ``get_market_for_asset`` map, which is itself populated from
                this same repo at subscription-refresh time.
            clock: Injectable :class:`Clock`. Defaults to :class:`RealClock`
                so production wiring needs no changes.
        """
        self._config = config
        self._ticks = ticks_collector
        self._market_cache = market_cache
        self._clock: Clock = clock if clock is not None else RealClock()

    async def run(self, sink: AlertSink) -> None:
        """Poll every subscribed asset on a fixed cadence until cancelled.

        The scheduler's TaskGroup drives shutdown; on cancellation we
        propagate ``CancelledError`` cleanly. Any other exception during a
        single sweep is logged and the loop continues.

        Args:
            sink: Shared alert sink every detector publishes to.
        """
        while True:
            try:
                for asset_id in self._ticks.subscribed_asset_ids():
                    await self.evaluate_asset(asset_id, sink)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("velocity.poll_failed")
            await self._clock.sleep(self._config.poll_interval_seconds)

    async def evaluate_asset(self, asset_id: str, sink: AlertSink) -> None:
        """Evaluate one asset's velocity and emit an alert if the threshold trips.

        Public so it can be driven from tests and from ``run_once``.

        Args:
            asset_id: CLOB token id to evaluate.
            sink: Sink to emit alerts to when the threshold is exceeded.
        """
        window = self._config.velocity_window_seconds
        mids = self._ticks.get_recent_mids(asset_id, window_seconds=window)
        if len(mids) < _MIN_MIDS_FOR_DELTA:
            return
        start_ts, start_price = mids[0]
        end_ts, end_price = mids[-1]
        if start_price <= 0:
            return
        change_pct = (end_price - start_price) / start_price
        threshold = self._config.velocity_threshold_pct
        if abs(change_pct) < threshold:
            return
        recent_ticks = self._ticks.get_recent_ticks(asset_id, window_seconds=window)
        if not recent_ticks:
            return
        if not self._passes_liquidity_filters(recent_ticks[-1]):
            return
        severity: Severity = (
            "high" if abs(change_pct) > _HIGH_SEVERITY_MULTIPLE * threshold else "med"
        )
        market = self._ticks.get_market_for_asset(asset_id)
        alert = _build_alert(
            asset_id=asset_id,
            start_ts=start_ts,
            end_ts=end_ts,
            start_price=start_price,
            end_price=end_price,
            change_pct=change_pct,
            severity=severity,
            samples=len(mids),
            market_title=market.title if market else None,
            condition_id=market.condition_id if market else None,
        )
        await sink.emit(alert)

    def _passes_liquidity_filters(self, tick: MarketTick) -> bool:
        """Return ``True`` when the latest tick has balanced, liquid book depth.

        Suppresses two failure modes seen on illiquid micro-cap markets:

        1. Lopsided depth — one side a wall, the other a whisper. A sweep of
           the whisper side moves the mid by tens of percent without any
           real-money flow. ``min/max`` ratio below ``depth_asymmetry_floor``
           is rejected.
        2. Insufficient USD on both sides. ``min(bid_depth, ask_depth) * mid``
           must clear ``min_mid_liquidity_usd``.
        """
        bid = tick.bid_depth_top5
        ask = tick.ask_depth_top5
        if bid is None or ask is None:
            return False
        denom = max(bid, ask)
        if denom <= 0:
            return False
        ratio = min(bid, ask) / denom
        if ratio < self._config.depth_asymmetry_floor:
            return False
        mid = tick.mid_price
        if mid is None or mid <= 0:
            return False
        return min(bid, ask) * mid >= self._config.min_mid_liquidity_usd


def _build_alert(
    *,
    asset_id: str,
    start_ts: int,
    end_ts: int,
    start_price: float,
    end_price: float,
    change_pct: float,
    severity: Severity,
    samples: int,
    market_title: str | None,
    condition_id: str | None,
) -> Alert:
    """Construct the Alert payload for a velocity event.

    Body includes ``market_title`` and ``condition_id`` (or ``None`` if the
    asset's market is not in the cached map) so the terminal panel renders an
    actionable alert without a manual ``asset_id`` → market lookup.
    """
    bucket = end_ts // _DEDUP_BUCKET_SECONDS
    alert_key = f"velocity:{asset_id}:{bucket}"
    title = f"Velocity: asset {asset_id[:14]} {change_pct:+.1%} in {end_ts - start_ts}s"
    body: dict[str, Any] = {
        "asset_id": asset_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "start_price": start_price,
        "end_price": end_price,
        "change_pct": change_pct,
        "window_seconds": end_ts - start_ts,
        "samples_in_window": samples,
        "market_title": market_title,
        "condition_id": condition_id,
    }
    return Alert(
        detector=cast(DetectorName, "velocity"),
        alert_key=alert_key,
        severity=severity,
        title=title,
        body=body,
        created_at=int(time.time()),
    )
