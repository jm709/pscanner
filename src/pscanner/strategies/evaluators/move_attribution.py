"""``MoveAttributionEvaluator`` — burst-driven copy-trade evaluator.

Trades the (outcome, side) pair surfaced by MoveAttributionDetector when
≥ ``min_wallets`` distinct wallets converged on a market in the burst
window.

Note: the alert body's ``side`` field is a taker action ("BUY"/"SELL"),
NOT an outcome name. We use ``outcome`` as the cache lookup key (the
actual outcome name like "Anastasia Potapova" or "yes").
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import SEVERITY_RANK, Alert
from pscanner.config import MoveAttributionEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class MoveAttributionEvaluator:
    """Trade move-attribution bursts named by upstream detectors."""

    def __init__(self, *, config: MoveAttributionEvaluatorConfig) -> None:
        """Bind dependencies for the move-attribution evaluator.

        Args:
            config: Tunables (position_fraction, min_severity, min_wallets).
        """
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert was emitted by the move_attribution detector."""
        return alert.detector == "move_attribution"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull condition_id + outcome from the alert body, stash severity/n_wallets."""
        body = alert.body if isinstance(alert.body, dict) else {}
        condition_id_str = body.get("condition_id")
        outcome = body.get("outcome")
        n_wallets = body.get("n_wallets")
        if not (isinstance(condition_id_str, str) and isinstance(outcome, str)):
            _LOG.debug("move_attribution_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(condition_id_str),
                side=outcome,
                rule_variant=None,
                metadata={
                    "severity": alert.severity,
                    "n_wallets": int(n_wallets) if isinstance(n_wallets, int) else 0,
                },
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject below the severity floor or below the wallet-count floor."""
        severity = parsed.metadata.get("severity")
        n_wallets = parsed.metadata.get("n_wallets")
        if not isinstance(severity, str) or not isinstance(n_wallets, int):
            return False
        if SEVERITY_RANK.get(severity, -1) < SEVERITY_RANK[self._config.min_severity]:
            return False
        return n_wallets >= self._config.min_wallets

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return a constant ``bankroll * position_fraction`` per signal."""
        del parsed
        return bankroll * self._config.position_fraction
