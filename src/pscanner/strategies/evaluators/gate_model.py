"""``GateModelEvaluator`` — paper-trade ``gate_buy`` alerts (#80).

Single-leg evaluator: one :class:`ParsedSignal` per alert. No twin/paired
logic (that's velocity's domain). Sizing is constant
``bankroll * position_fraction`` matching the project's "constant sizing,
infinite paper bankroll" research config per CLAUDE.md.

The evaluator does NOT call the booster — the prediction comes
pre-computed in the alert body, written by
:class:`pscanner.detectors.gate_model.GateModelDetector`. ``quality_passes``
re-checks the edge floor as a defensive double-check, catching any operator
config drift between detector ``min_edge_pct`` and evaluator ``min_edge_pct``.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import GateModelEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class GateModelEvaluator:
    """Paper-trades alerts emitted by the gate-model detector."""

    def __init__(self, *, config: GateModelEvaluatorConfig) -> None:
        """Bind tunables for the gate-model evaluator.

        Args:
            config: Tunables (``min_edge_pct`` quality gate floor and
                ``position_fraction`` sizing factor).
        """
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert came from the gate-model detector."""
        return alert.detector == "gate_buy"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull condition_id/side/pred/implied from the alert body."""
        body = alert.body if isinstance(alert.body, dict) else {}
        condition_id = body.get("condition_id")
        side = body.get("side")
        pred = body.get("pred")
        implied = body.get("implied_prob_at_buy")
        if not (
            isinstance(condition_id, str)
            and side in ("YES", "NO")
            and isinstance(pred, int | float)
            and isinstance(implied, int | float)
        ):
            _LOG.debug("gate_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(condition_id),
                side=side,
                rule_variant=None,
                metadata={
                    "pred": float(pred),
                    "implied": float(implied),
                    "edge": float(pred) - float(implied),
                },
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject signals whose edge falls below ``min_edge_pct``."""
        edge = parsed.metadata.get("edge")
        if not isinstance(edge, int | float):
            return False
        return float(edge) >= self._config.min_edge_pct

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return constant ``bankroll * position_fraction`` per signal."""
        del parsed  # uniform sizing across alerts by design
        return bankroll * self._config.position_fraction
