"""Mispricing detector — flags events whose YES outcome prices don't sum to 1.

For mutex-outcome events on Polymarket the sum of the YES legs across every
market in the event must equal ``1.0`` at no-arbitrage. A persistent deviation
hints at either an arbitrage opportunity or a stale book; either way it is
worth a human eyeball.

Non-mutex layouts (date-range or threshold buckets, e.g. "Measles cases in
2026 above N") have markets with a non-empty ``groupItemTitle`` and are
skipped: their outcomes are independent, so the sum-to-1 invariant doesn't
apply.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from pscanner.alerts.models import Alert, Severity
from pscanner.alerts.sink import AlertSink
from pscanner.config import MispricingConfig
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event, Market

_LOGGER = structlog.get_logger(__name__)

_HIGH_SEVERITY_DEVIATION = 0.10
_MED_SEVERITY_DEVIATION = 0.05
_MIN_VALID_MARKETS = 2


class MispricingDetector:
    """Detector that scans events for outcome-price sums that drift from 1.0.

    The detector iterates the gamma ``/events`` catalogue (active, open) and
    flags any event whose YES-leg prices sum more than
    ``config.sum_deviation_threshold`` away from ``1.0``. Severity is bucketed
    by deviation magnitude so the renderer can prioritise the worst offenders.
    """

    name: str = "mispricing"

    def __init__(
        self,
        *,
        config: MispricingConfig,
        gamma_client: GammaClient,
    ) -> None:
        """Build a detector bound to a config and gamma client.

        Args:
            config: Threshold + cadence settings (see ``MispricingConfig``).
            gamma_client: Async gamma-api client used to enumerate events.
        """
        self._config = config
        self._gamma = gamma_client

    async def run(self, sink: AlertSink) -> None:
        """Loop forever: scan, sleep, repeat. Logs and continues on error.

        Args:
            sink: Shared alert sink every detector publishes to.
        """
        while True:
            try:
                await self._scan(sink)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("mispricing scan failed", detector=self.name)
            await asyncio.sleep(self._config.scan_interval_seconds)

    async def _scan(self, sink: AlertSink) -> None:
        """Run a single pass over the active-event catalogue.

        Args:
            sink: Sink to publish alerts to.
        """
        events = self._gamma.iter_events(active=True, closed=False)
        async for event in events:
            await self.evaluate_event(event, sink)

    async def evaluate_event(self, event: Event, sink: AlertSink) -> None:
        """Evaluate a single event and emit an alert if it mispricies.

        Args:
            event: Event to evaluate.
            sink: Sink to publish the alert to (when warranted).
        """
        if not self._is_eligible(event):
            return
        price_sum, count = self._sum_outcome_prices(event)
        if count < _MIN_VALID_MARKETS:
            return
        deviation = price_sum - 1.0
        if abs(deviation) <= self._config.sum_deviation_threshold:
            return
        alert = self._build_alert(event, price_sum, count)
        await sink.emit(alert)

    def _is_eligible(self, event: Event) -> bool:
        """Apply pre-filters that don't depend on outcome prices.

        Skips events that aren't true mutex layouts: a non-empty
        ``groupItemTitle`` on any market signals a date-range or
        threshold-bucket layout where outcomes are independent and the
        no-arbitrage sum-to-1 invariant does not apply.
        """
        if len(event.markets) < _MIN_VALID_MARKETS:
            return False
        if any(not market.enable_order_book for market in event.markets):
            return False
        if any(market.group_item_title for market in event.markets):
            return False
        return not (
            event.liquidity is None or event.liquidity < self._config.min_event_liquidity_usd
        )

    def _sum_outcome_prices(self, event: Event) -> tuple[float, int]:
        """Sum YES-leg prices across the event's markets.

        Args:
            event: The event whose markets to aggregate.

        Returns:
            ``(price_sum, valid_market_count)`` where valid markets are those
            with a non-empty ``outcome_prices`` list.
        """
        price_sum = 0.0
        count = 0
        for market in event.markets:
            if not market.outcome_prices:
                continue
            price_sum += market.outcome_prices[0]
            count += 1
        return price_sum, count

    def _build_alert(self, event: Event, price_sum: float, count: int) -> Alert:
        """Construct the Alert payload for a mispriced event."""
        deviation = price_sum - 1.0
        return Alert(
            detector="mispricing",
            alert_key=f"mispricing:{event.id}:{round(price_sum, 2)}",
            severity=_severity_for(deviation),
            title=f"{event.title} — Σ outcomes = {price_sum:.3f}",
            body=_build_body(event, price_sum, deviation, count),
            created_at=int(time.time()),
        )


def _severity_for(deviation: float) -> Severity:
    """Map signed deviation to a triage bucket."""
    magnitude = abs(deviation)
    if magnitude > _HIGH_SEVERITY_DEVIATION:
        return "high"
    if magnitude > _MED_SEVERITY_DEVIATION:
        return "med"
    return "low"


def _build_body(
    event: Event,
    price_sum: float,
    deviation: float,
    count: int,
) -> dict[str, Any]:
    """Build the JSON-serialisable body the alert sink will persist."""
    return {
        "event_id": event.id,
        "event_title": event.title,
        "price_sum": price_sum,
        "deviation": deviation,
        "market_count": count,
        "markets": [_market_summary(market) for market in event.markets],
    }


def _market_summary(market: Market) -> dict[str, Any]:
    """Compact per-market summary attached to the alert body."""
    yes_price = market.outcome_prices[0] if market.outcome_prices else None
    return {"id": market.id, "question": market.question, "yes_price": yes_price}
