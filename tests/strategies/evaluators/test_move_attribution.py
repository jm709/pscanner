"""Unit tests for MoveAttributionEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MoveAttributionEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.move_attribution import MoveAttributionEvaluator


def _alert(
    *,
    body: dict[str, Any],
    detector: str = "move_attribution",
    severity: str = "med",
) -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key="k1",
        severity=severity,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        title="t",
        body=body,
        created_at=0,
    )


def _evaluator(
    *,
    position_fraction: float = 0.01,
    min_severity: str = "med",
    min_wallets: int = 3,
) -> MoveAttributionEvaluator:
    cfg = MoveAttributionEvaluatorConfig(
        position_fraction=position_fraction,
        min_severity=min_severity,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        min_wallets=min_wallets,
    )
    return MoveAttributionEvaluator(config=cfg)


def test_accepts_only_move_attribution() -> None:
    ev = _evaluator()
    assert ev.accepts(_alert(body={}, detector="move_attribution")) is True
    assert ev.accepts(_alert(body={}, detector="smart_money")) is False


def test_parse_extracts_outcome_as_side() -> None:
    """The outcome name (e.g. 'Anastasia Potapova') becomes the ParsedSignal side
    for cache lookup; the body's 'side' field (BUY/SELL taker) is ignored.
    """
    ev = _evaluator()
    body = {
        "condition_id": "0xc1",
        "outcome": "Anastasia Potapova",
        "side": "BUY",
        "n_wallets": 4,
    }
    [parsed] = ev.parse(_alert(body=body))
    assert parsed.condition_id == ConditionId("0xc1")
    assert parsed.side == "Anastasia Potapova"
    assert parsed.rule_variant is None


def test_parse_returns_empty_on_missing_outcome() -> None:
    ev = _evaluator()
    body = {"condition_id": "0xc1", "side": "BUY", "n_wallets": 4}
    assert ev.parse(_alert(body=body)) == []


def test_quality_passes_below_severity_rejected() -> None:
    """Default min_severity='med' rejects 'low'."""
    ev = _evaluator()
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "low", "n_wallets": 10},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_below_min_wallets_rejected() -> None:
    ev = _evaluator(min_wallets=5)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "high", "n_wallets": 3},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_high_severity_and_n_wallets() -> None:
    ev = _evaluator()
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "med", "n_wallets": 4},
    )
    assert ev.quality_passes(parsed) is True


def test_size_returns_bankroll_times_fraction() -> None:
    ev = _evaluator(position_fraction=0.01)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "med", "n_wallets": 4},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(10.0)
