"""Price velocity detector — fires on rapid mid-price moves.

Wave 1 contract; Wave 2 fills in the implementation.
"""

from __future__ import annotations

from pscanner.alerts.sink import AlertSink
from pscanner.collectors.ticks import MarketTickCollector
from pscanner.config import VelocityConfig
from pscanner.store.repo import MarketCacheRepo


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
    ) -> None:
        """Build the detector. Wave 2 fills in the body."""
        self._config = config
        self._ticks = ticks_collector
        self._market_cache = market_cache

    async def run(self, sink: AlertSink) -> None:
        """Long-running loop: poll all subscribed assets every poll_interval."""
        raise NotImplementedError("DC-4 Wave 2: velocity")

    async def evaluate_asset(self, asset_id: str, sink: AlertSink) -> None:
        """Evaluate one asset's velocity and emit an alert if the threshold trips.

        Public so it can be driven from tests and from ``run_once``.

        Args:
            asset_id: CLOB token id to evaluate.
            sink: Sink to emit alerts to when the threshold is exceeded.
        """
        raise NotImplementedError("DC-4 Wave 2: velocity")
