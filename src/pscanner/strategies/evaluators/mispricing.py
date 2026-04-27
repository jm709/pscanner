"""``MispricingEvaluator`` — trade the most-extreme leg of a mispriced event.

Reads the detector-emitted target_* fields from the alert body. Quality
gate is the magnitude of the gap between fair and current price.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import MispricingEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class MispricingEvaluator:
    """Trade the most-overpriced/most-underpriced YES leg of a mispriced event."""

    def __init__(self, *, config: MispricingEvaluatorConfig) -> None:
        """Bind dependencies for the mispricing evaluator.

        Args:
            config: Tunables (position_fraction, min_edge_dollars).
        """
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert was emitted by the mispricing detector."""
        return alert.detector == "mispricing"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull target_* fields from the alert body; skip legacy pre-T4 alerts."""
        body = alert.body if isinstance(alert.body, dict) else {}
        cond = body.get("target_condition_id")
        side = body.get("target_side")
        current = body.get("target_current_price")
        fair = body.get("target_fair_price")
        if not (
            isinstance(cond, str)
            and isinstance(side, str)
            and isinstance(current, int | float)
            and isinstance(fair, int | float)
        ):
            _LOG.debug("mispricing_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(cond),
                side=side,
                rule_variant=None,
                metadata={"current": float(current), "fair": float(fair)},
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject signals whose ``|fair - current|`` is below the edge floor."""
        current = parsed.metadata.get("current")
        fair = parsed.metadata.get("fair")
        if not (isinstance(current, int | float) and isinstance(fair, int | float)):
            return False
        edge = abs(float(fair) - float(current))
        return edge >= self._config.min_edge_dollars

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return a constant ``bankroll * position_fraction`` per signal."""
        del parsed
        return bankroll * self._config.position_fraction
