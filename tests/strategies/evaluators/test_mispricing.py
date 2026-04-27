"""Unit tests for MispricingEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MispricingEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.mispricing import MispricingEvaluator


def _alert(*, body: dict[str, Any], detector: str = "mispricing") -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key="k1",
        severity="med",
        title="t",
        body=body,
        created_at=0,
    )


def _evaluator(
    *,
    position_fraction: float = 0.01,
    min_edge_dollars: float = 0.05,
) -> MispricingEvaluator:
    return MispricingEvaluator(
        config=MispricingEvaluatorConfig(
            position_fraction=position_fraction,
            min_edge_dollars=min_edge_dollars,
        ),
    )


def test_accepts_only_mispricing() -> None:
    ev = _evaluator()
    assert ev.accepts(_alert(body={}, detector="mispricing")) is True
    assert ev.accepts(_alert(body={}, detector="smart_money")) is False


def test_parse_extracts_target_fields() -> None:
    ev = _evaluator()
    body = {
        "target_condition_id": "0xc1",
        "target_side": "NO",
        "target_current_price": 0.5,
        "target_fair_price": 0.643,
    }
    [parsed] = ev.parse(_alert(body=body))
    assert parsed.condition_id == ConditionId("0xc1")
    assert parsed.side == "NO"
    assert parsed.metadata == {"current": 0.5, "fair": 0.643}


def test_parse_returns_empty_when_target_fields_missing() -> None:
    """Legacy alerts without target_* (pre-T4) are silently skipped."""
    ev = _evaluator()
    body = {"event_id": "ev1", "deviation": 0.4}  # no target_*
    assert ev.parse(_alert(body=body)) == []


def test_quality_passes_above_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.05)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="NO",
        metadata={"current": 0.5, "fair": 0.643},  # edge = 0.143
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_below_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.10)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="NO",
        metadata={"current": 0.5, "fair": 0.55},  # edge = 0.05 < 0.10
    )
    assert ev.quality_passes(parsed) is False


def test_size_returns_bankroll_times_fraction() -> None:
    ev = _evaluator(position_fraction=0.01)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="NO",
        metadata={"current": 0.5, "fair": 0.643},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(10.0)
