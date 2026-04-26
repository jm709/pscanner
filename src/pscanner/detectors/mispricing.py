"""Mispricing detector — flags events whose YES outcome prices don't sum to 1.

For mutex-outcome events on Polymarket the sum of the YES legs across every
market in the event must equal ``1.0`` at no-arbitrage. A persistent deviation
hints at either an arbitrage opportunity or a stale book; either way it is
worth a human eyeball.

Non-mutex layouts (date-range or threshold buckets, e.g. "Measles cases in
2026 above N") have markets whose ``groupItemTitle`` looks like a date or
numeric threshold and are skipped: their outcomes are independent, so the
sum-to-1 invariant doesn't apply.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import structlog

from pscanner.alerts.models import Alert, Severity
from pscanner.alerts.sink import AlertSink
from pscanner.categories import categorize_event, settings_for
from pscanner.config import MispricingConfig
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event, Market
from pscanner.store.repo import EventOutcomeSumRepo, EventOutcomeSumRow
from pscanner.util.clock import Clock, RealClock

_LOGGER = structlog.get_logger(__name__)

_HIGH_SEVERITY_DEVIATION = 0.10
_MED_SEVERITY_DEVIATION = 0.05
_MIN_VALID_MARKETS = 2

_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d+(?:,\s*\d{4})?$"),
)
_NUMERIC_THRESHOLD_PATTERN = re.compile(r"^[<>]?=?\s*\$?\d")
_RANGE_KEYWORD_PATTERN = re.compile(
    r"^(?:above|below|over|under|at\s+least|at\s+most|more\s+than|less\s+than|>=|<=|≥|≤)\s+\$?\d",
    re.IGNORECASE,
)


def _looks_like_bucket_label(label: str | None) -> bool:
    """Return True if a market's groupItemTitle looks like a date or numeric bucket.

    Used to detect range/threshold-style events (where outcomes overlap and the
    sum-to-1 invariant does not apply) vs candidate-style mutex events (where
    titles are arbitrary names). Range-keyword prefixes (``Above $300M``,
    ``At least 5``, ``≥ 10``) are also treated as bucket labels.
    """
    if not label:
        return False
    text = label.strip()
    if not text:
        return False
    return (
        any(p.match(text) for p in _DATE_PATTERNS)
        or bool(_NUMERIC_THRESHOLD_PATTERN.match(text))
        or bool(_RANGE_KEYWORD_PATTERN.match(text))
    )


def _should_skip_by_category(event: Event) -> bool:
    """Return True if the event's category is configured for mispricing skip.

    Sports/esports events are tournament aggregations rather than mutex
    outcomes, so the sum-to-1 invariant doesn't apply. The taxonomy in
    :data:`pscanner.categories.DEFAULT_TAXONOMY` controls which categories
    skip via the ``mispricing_skip`` flag on each ``CategorySettings``.
    """
    return settings_for(categorize_event(event)).mispricing_skip


def _is_range_bucket_event(event: Event) -> bool:
    """Return True if every market in the event has a date-or-numeric title.

    Pure mutex (candidate-style) events have None or arbitrary string titles
    and produce False. Mixed events (some buckets, some non-buckets) produce
    False — we err toward eligibility, since a single non-bucket market means
    the event is not a clean threshold layout.
    """
    if not event.markets:
        return False
    return all(_looks_like_bucket_label(m.group_item_title) for m in event.markets)


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
        sum_history_repo: EventOutcomeSumRepo,
        clock: Clock | None = None,
    ) -> None:
        """Build a detector bound to a config, gamma client, and history repo.

        Args:
            config: Threshold + cadence settings (see ``MispricingConfig``).
            gamma_client: Async gamma-api client used to enumerate events.
            sum_history_repo: Repo that persists every eligible event's Σ-of-
                outcomes regardless of whether an alert fires, building the
                research dataset that backs ``ABS(deviation) > 5`` queries.
            clock: Injectable :class:`Clock`. Defaults to :class:`RealClock`
                so production wiring needs no changes.
        """
        self._config = config
        self._gamma = gamma_client
        self._sum_repo = sum_history_repo
        self._clock: Clock = clock if clock is not None else RealClock()

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
            await self._clock.sleep(self._config.scan_interval_seconds)

    async def _scan(self, sink: AlertSink) -> None:
        """Run a single pass over the active-event catalogue.

        Args:
            sink: Sink to publish alerts to.
        """
        events = self._gamma.iter_events(active=True, closed=False)
        async for event in events:
            await self.evaluate_event(event, sink)

    async def evaluate_event(self, event: Event, sink: AlertSink) -> None:
        """Evaluate a single event, record Σ-history, and alert if in band.

        Every eligible event is captured in ``event_outcome_sum_history`` —
        even when no alert fires — so analysts can later study high-Σ
        multi-outcome layouts. An alert is emitted only when
        ``sum_deviation_threshold < |deviation| <= alert_max_deviation``.

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
        self._sum_repo.insert(
            EventOutcomeSumRow(
                event_id=event.id,
                market_count=count,
                price_sum=price_sum,
                deviation=deviation,
                snapshot_at=int(time.time()),
            ),
        )
        abs_dev = abs(deviation)
        if abs_dev <= self._config.sum_deviation_threshold:
            return
        if abs_dev > self._config.alert_max_deviation:
            return
        alert = self._build_alert(event, price_sum, count)
        await sink.emit(alert)

    def _is_eligible(self, event: Event) -> bool:
        """Apply the cheap pre-filters that don't depend on outcome prices.

        Skips events:

        * with fewer than two markets, or any market with the order book
          disabled;
        * whose category is configured for mispricing skip in
          :data:`pscanner.categories.DEFAULT_TAXONOMY` (e.g. ``Sports`` or
          ``Esports``) — those are tournament aggregations, not mutex
          outcomes;
        * whose markets all carry date-like or numeric-threshold
          ``groupItemTitle`` values — bucket layouts where outcomes overlap;
        * with insufficient event-level liquidity;
        * containing any market with per-market liquidity below
          ``min_market_liquidity_usd`` (when > 0). Drops noise-floor markets
          where individual fills can swing prices by >10%.

        Candidate-style mutex events (Trump/Harris/Other) leave
        ``groupItemTitle`` arbitrary and remain eligible.
        """
        if len(event.markets) < _MIN_VALID_MARKETS:
            return False
        if any(not market.enable_order_book for market in event.markets):
            return False
        if _should_skip_by_category(event):
            return False
        if _is_range_bucket_event(event):
            return False
        if event.liquidity is None or event.liquidity < self._config.min_event_liquidity_usd:
            return False
        return not self._has_below_floor_market(event)

    def _has_below_floor_market(self, event: Event) -> bool:
        """Return True if any market falls below ``min_market_liquidity_usd``.

        When the configured floor is 0.0, returns False (filter disabled).
        Markets with NULL ``liquidity`` are treated as below the floor.
        """
        floor = self._config.min_market_liquidity_usd
        if floor <= 0:
            return False
        return any(m.liquidity is None or m.liquidity < floor for m in event.markets)

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
