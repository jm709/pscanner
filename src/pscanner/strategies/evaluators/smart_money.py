"""``SmartMoneyEvaluator`` — copy-trade smart_money alerts.

Today's PaperTrader behaviour, lifted into the per-detector evaluator
shape. Quality gate is wallet ``weighted_edge``; sizing is constant
``bankroll * position_fraction``.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import SmartMoneyEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import TrackedWalletsRepo
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class SmartMoneyEvaluator:
    """Smart-money copy-trade evaluator."""

    def __init__(
        self,
        *,
        config: SmartMoneyEvaluatorConfig,
        tracked_wallets: TrackedWalletsRepo,
    ) -> None:
        """Bind dependencies for the smart-money evaluator.

        Args:
            config: Tunables (position_fraction, min_weighted_edge).
            tracked_wallets: Lookup for the source wallet's edge metadata.
        """
        self._config = config
        self._tracked_wallets = tracked_wallets

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert was emitted by the smart_money detector."""
        return alert.detector == "smart_money"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull wallet/condition_id/side out of the alert body."""
        body = alert.body if isinstance(alert.body, dict) else {}
        wallet = body.get("wallet")
        condition_id_str = body.get("condition_id")
        side = body.get("side")
        if not (
            isinstance(wallet, str) and isinstance(condition_id_str, str) and isinstance(side, str)
        ):
            _LOG.debug("smart_money_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(condition_id_str),
                side=side,
                rule_variant=None,
                metadata={"wallet": wallet},
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject signals from unknown wallets or wallets below the edge floor."""
        wallet = parsed.metadata.get("wallet")
        if not isinstance(wallet, str):
            return False
        tracked = self._tracked_wallets.get(wallet)
        if tracked is None:
            _LOG.debug("smart_money_evaluator.no_edge", wallet=wallet)
            return False
        edge = tracked.weighted_edge
        if edge is None or edge <= self._config.min_weighted_edge:
            _LOG.debug("smart_money_evaluator.below_edge", wallet=wallet, edge=edge)
            return False
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return a constant ``bankroll * position_fraction`` per signal."""
        del parsed  # SmartMoney sizes uniformly across alerts
        return bankroll * self._config.position_fraction
