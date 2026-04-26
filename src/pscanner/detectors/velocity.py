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

Alerts are keyed by ``condition_id`` (when known) rather than ``asset_id`` so
binary YES/NO pairs collapse to one row in the alerts table. Moves dominated
by quote consolidation — a market-maker tightening a stale wide spread around
the new fair value — are demoted to ``low`` severity with ``consolidation =
True`` in the body, keeping the data queryable without inflating high-priority
counts.
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
from pscanner.poly.ids import AssetId, ConditionId
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

    async def evaluate_asset(self, asset_id: AssetId, sink: AlertSink) -> None:
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
        consolidation = _classify_consolidation(recent_ticks, self._config.spread_compression_floor)
        severity = _severity_for(change_pct, threshold, consolidation is not None)
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
            consolidation=consolidation,
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


def _classify_consolidation(
    recent_ticks: list[MarketTick], floor: float
) -> dict[str, float] | None:
    """Return consolidation metadata when the spread compressed past ``floor``.

    A quote-consolidation event is a market-maker tightening a stale wide
    spread around the new fair value: the mid swings dramatically without a
    real trade. Detect it by comparing the spread on the first tick in the
    window to the spread on the last tick. If
    ``first.spread / last.spread > floor``, treat the move as consolidation
    rather than price discovery.

    Args:
        recent_ticks: Tick history for the window, oldest first.
        floor: Minimum compression ratio to qualify as consolidation.

    Returns:
        A dict with ``spread_ratio``, ``spread_before``, and ``spread_after``
        when the move qualifies as consolidation, otherwise ``None``. Returns
        ``None`` whenever the ratio cannot be computed (fewer than two ticks,
        missing or non-positive spreads).
    """
    if len(recent_ticks) < _MIN_MIDS_FOR_DELTA:
        return None
    first = recent_ticks[0]
    last = recent_ticks[-1]
    if first.spread is None or first.spread <= 0:
        return None
    if last.spread is None or last.spread <= 0:
        return None
    ratio = first.spread / last.spread
    if ratio <= floor:
        return None
    return {
        "spread_ratio": round(ratio, 1),
        "spread_before": first.spread,
        "spread_after": last.spread,
    }


def _severity_for(change_pct: float, threshold: float, is_consolidation: bool) -> Severity:
    """Bucket severity by magnitude unless the move is quote consolidation."""
    if is_consolidation:
        return "low"
    if abs(change_pct) > _HIGH_SEVERITY_MULTIPLE * threshold:
        return "high"
    return "med"


def _alert_key_for(asset_id: AssetId, condition_id: ConditionId | None, end_ts: int) -> str:
    """Build the alert_key, preferring ``condition_id`` to dedupe binary YES/NO pairs.

    Binary Polymarket markets have two complementary asset_ids. A velocity
    move on the YES side mirrors an opposite move on NO, so keying alerts by
    ``condition_id`` collapses both sides into one row in the alerts table.
    Falls back to the asset-keyed format when the market cache has not yet
    populated for this asset, so the alert isn't dropped silently.
    """
    bucket = end_ts // _DEDUP_BUCKET_SECONDS
    if condition_id:
        return f"velocity:{condition_id}:{bucket}"
    return f"velocity:{asset_id}:{bucket}"


def _build_alert(
    *,
    asset_id: AssetId,
    start_ts: int,
    end_ts: int,
    start_price: float,
    end_price: float,
    change_pct: float,
    severity: Severity,
    samples: int,
    market_title: str | None,
    condition_id: ConditionId | None,
    consolidation: dict[str, float] | None,
) -> Alert:
    """Construct the Alert payload for a velocity event.

    Body includes ``market_title`` and ``condition_id`` (or ``None`` if the
    asset's market is not in the cached map) so the terminal panel renders an
    actionable alert without a manual ``asset_id`` → market lookup. The
    ``consolidation`` flag is always present: ``True`` when the move was
    dominated by a market-maker tightening a stale spread, otherwise ``False``.
    Consolidation alerts also carry ``spread_ratio``, ``spread_before``, and
    ``spread_after`` for downstream analysis.
    """
    alert_key = _alert_key_for(asset_id, condition_id, end_ts)
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
        "consolidation": consolidation is not None,
    }
    if consolidation is not None:
        body.update(consolidation)
    return Alert(
        detector=cast(DetectorName, "velocity"),
        alert_key=alert_key,
        severity=severity,
        title=title,
        body=body,
        created_at=int(time.time()),
    )
