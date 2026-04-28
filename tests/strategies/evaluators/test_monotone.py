"""Unit tests for MonotoneEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MonotoneEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.monotone import MonotoneEvaluator


def _alert(*, body: dict[str, Any], detector: str = "monotone") -> Alert:
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
    position_fraction: float = 0.005,
    min_edge_dollars: float = 0.02,
) -> MonotoneEvaluator:
    return MonotoneEvaluator(
        config=MonotoneEvaluatorConfig(
            position_fraction=position_fraction,
            min_edge_dollars=min_edge_dollars,
        ),
    )


def _good_body() -> dict[str, Any]:
    return {
        "strict_condition_id": "0xstrict",
        "loose_condition_id": "0xloose",
        "strict_yes_price": 0.40,
        "loose_yes_price": 0.30,
        "gap": 0.10,
    }


def test_accepts_only_monotone() -> None:
    ev = _evaluator()
    assert ev.accepts(_alert(body={}, detector="monotone")) is True
    assert ev.accepts(_alert(body={}, detector="mispricing")) is False


def test_parse_emits_two_legs() -> None:
    ev = _evaluator()
    signals = ev.parse(_alert(body=_good_body()))
    assert len(signals) == 2
    by_variant = {s.rule_variant: s for s in signals}
    strict = by_variant["strict_no"]
    loose = by_variant["loose_yes"]
    assert strict.condition_id == ConditionId("0xstrict")
    assert strict.side == "NO"
    assert strict.metadata["gap"] == 0.10
    assert loose.condition_id == ConditionId("0xloose")
    assert loose.side == "YES"


def test_parse_returns_empty_when_required_fields_missing() -> None:
    ev = _evaluator()
    bad = {"strict_condition_id": "0xstrict"}
    assert ev.parse(_alert(body=bad)) == []


def test_quality_passes_above_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.05)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xstrict"),
        side="NO",
        rule_variant="strict_no",
        metadata={"gap": 0.10},
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_below_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.05)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xstrict"),
        side="NO",
        rule_variant="strict_no",
        metadata={"gap": 0.03},
    )
    assert ev.quality_passes(parsed) is False


def test_size_returns_per_leg_fraction() -> None:
    ev = _evaluator(position_fraction=0.005)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xstrict"),
        side="NO",
        rule_variant="strict_no",
        metadata={"gap": 0.10},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(5.0)
