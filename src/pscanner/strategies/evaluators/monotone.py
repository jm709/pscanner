"""``MonotoneEvaluator`` — paired-trade NO-strict + YES-loose per alert.

Each monotone alert names the two condition_ids of the violating adjacent
pair. The evaluator emits two ParsedSignals (``strict_no`` + ``loose_yes``)
sized at the per-leg ``position_fraction``. Quality gate is the ``gap``
field on the alert body — when below ``min_edge_dollars`` no signals fire.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import MonotoneEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class MonotoneEvaluator:
    """Two-leg paired-trade evaluator for monotone-arb alerts."""

    def __init__(self, *, config: MonotoneEvaluatorConfig) -> None:
        """Bind dependencies for the monotone evaluator."""
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert was emitted by the monotone detector."""
        return alert.detector == "monotone"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull pair fields from the alert body and emit two ParsedSignals."""
        body = alert.body if isinstance(alert.body, dict) else {}
        strict = body.get("strict_condition_id")
        loose = body.get("loose_condition_id")
        gap = body.get("gap")
        if not (
            isinstance(strict, str) and isinstance(loose, str) and isinstance(gap, int | float)
        ):
            _LOG.debug("monotone_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        meta = {"gap": float(gap)}
        return [
            ParsedSignal(
                condition_id=ConditionId(strict),
                side="NO",
                rule_variant="strict_no",
                metadata=meta,
            ),
            ParsedSignal(
                condition_id=ConditionId(loose),
                side="YES",
                rule_variant="loose_yes",
                metadata=meta,
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject signals whose gap is below the edge floor."""
        gap = parsed.metadata.get("gap")
        if not isinstance(gap, int | float):
            return False
        return float(gap) >= self._config.min_edge_dollars

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return constant ``bankroll * position_fraction`` per leg."""
        del parsed
        return bankroll * self._config.position_fraction
