"""Price velocity detector — fires on rapid mid-price moves.

Polls :class:`MarketTickCollector` for recent mid-price history per asset and
emits an :class:`~pscanner.alerts.models.Alert` whenever the percent move over
``velocity_window_seconds`` exceeds ``velocity_threshold_pct`` in either
direction.

The market-cache repo is wired in for future enrichment (looking up the
condition_id / market title from a CLOB token id), but v1 keeps the alert body
keyed on ``asset_id`` only — the collector's frozen API does not expose the
condition_id and a full ``list_active`` scan per evaluation would be too slow.
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
from pscanner.store.repo import MarketCacheRepo
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
            market_cache: Reserved for v2 enrichment of the alert body with
                market title / condition_id. Currently stored but unused.
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
        mids = self._ticks.get_recent_mids(
            asset_id,
            window_seconds=self._config.velocity_window_seconds,
        )
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
        severity: Severity = (
            "high" if abs(change_pct) > _HIGH_SEVERITY_MULTIPLE * threshold else "med"
        )
        alert = _build_alert(
            asset_id=asset_id,
            start_ts=start_ts,
            end_ts=end_ts,
            start_price=start_price,
            end_price=end_price,
            change_pct=change_pct,
            severity=severity,
            samples=len(mids),
        )
        await sink.emit(alert)


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
) -> Alert:
    """Construct the Alert payload for a velocity event.

    The alert body deliberately uses ``asset_id`` rather than a market title /
    condition_id; the tick collector's frozen API does not expose the
    condition_id, and reverse-lookup via ``MarketCacheRepo.list_active`` is too
    slow per evaluation. v2 enhancement: enrich body with market metadata.
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
    }
    return Alert(
        detector=cast(DetectorName, "velocity"),
        alert_key=alert_key,
        severity=severity,
        title=title,
        body=body,
        created_at=int(time.time()),
    )
