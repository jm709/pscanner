"""``VelocityEvaluator`` — twin-trade follow + fade per velocity alert.

Each velocity alert spawns two ParsedSignals at half-size: one with
rule_variant='follow' (buying the moving side, i.e. the asset_id named
in the alert) and one with rule_variant='fade' (buying the opposing
outcome on the same condition_id). Resolving the opposing-side outcome
requires a MarketCacheRepo lookup. Cache misses yield no signals — we
can't trade either side without knowing the outcome names.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import SEVERITY_RANK, Alert
from pscanner.config import VelocityEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import MarketCacheRepo
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class VelocityEvaluator:
    """Twin-trade evaluator for velocity alerts."""

    def __init__(
        self,
        *,
        config: VelocityEvaluatorConfig,
        market_cache: MarketCacheRepo,
    ) -> None:
        """Bind config + market cache; ready to evaluate alerts."""
        self._config = config
        self._market_cache = market_cache

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert was emitted by the velocity detector."""
        return alert.detector == "velocity"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Resolve the alert's asset_id to follow + fade outcome names.

        For binary markets, returns 2 ParsedSignals (follow + fade). For
        markets with more than two outcomes, the fade is assigned to the
        FIRST non-matching outcome in cache insertion order; remaining
        outcomes are ignored. For 1-outcome markets, returns just the
        follow signal. Returns ``[]`` on missing required fields, cache
        miss, or alert asset_id not in the cached market.
        """
        body = alert.body if isinstance(alert.body, dict) else {}
        condition_id_str = body.get("condition_id")
        asset_id_str = body.get("asset_id")
        consolidation = bool(body.get("consolidation", False))
        if not (isinstance(condition_id_str, str) and isinstance(asset_id_str, str)):
            _LOG.debug("velocity_evaluator.bad_body", alert_key=alert.alert_key)
            return []

        condition_id = ConditionId(condition_id_str)
        cached = self._market_cache.get_by_condition_id(condition_id)
        if cached is None or not cached.outcomes or not cached.asset_ids:
            _LOG.debug(
                "velocity_evaluator.cache_miss",
                alert_key=alert.alert_key,
                condition_id=condition_id_str,
            )
            return []

        follow_outcome: str | None = None
        fade_outcome: str | None = None
        for outcome_name, oid in zip(cached.outcomes, cached.asset_ids, strict=False):
            if oid == asset_id_str:
                follow_outcome = outcome_name
            elif fade_outcome is None:
                fade_outcome = outcome_name
        if follow_outcome is None:
            _LOG.debug(
                "velocity_evaluator.unknown_asset",
                alert_key=alert.alert_key,
                asset_id=asset_id_str,
            )
            return []

        meta = {"severity": alert.severity, "consolidation": consolidation}
        signals = [
            ParsedSignal(
                condition_id=condition_id,
                side=follow_outcome,
                rule_variant="follow",
                metadata=meta,
            ),
        ]
        if fade_outcome is not None:
            signals.append(
                ParsedSignal(
                    condition_id=condition_id,
                    side=fade_outcome,
                    rule_variant="fade",
                    metadata=meta,
                ),
            )
        return signals

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject below the severity floor or on consolidation when disallowed."""
        severity = parsed.metadata.get("severity")
        consolidation = parsed.metadata.get("consolidation")
        if not isinstance(severity, str):
            return False
        if SEVERITY_RANK.get(severity, -1) < SEVERITY_RANK[self._config.min_severity]:
            return False
        return not (bool(consolidation) and not self._config.allow_consolidation)

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return constant ``bankroll * position_fraction`` per entry side."""
        del parsed
        return bankroll * self._config.position_fraction
