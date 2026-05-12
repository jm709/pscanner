"""Unit tests for GateModelEvaluator (#80)."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import GateModelEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
from pscanner.strategies.evaluators.protocol import ParsedSignal

_NOW = 1_700_000_000


def _alert(*, body: Any, detector: str = "gate_buy") -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key=f"gate:tx:{detector}",
        severity="med",
        title="t",
        body=body,
        created_at=_NOW,
    )


def _good_body(
    *,
    condition_id: str = "0xc1",
    side: str = "YES",
    pred: float = 0.85,
    implied: float = 0.40,
) -> dict[str, Any]:
    return {
        "wallet": "0xabc",
        "condition_id": condition_id,
        "side": side,
        "implied_prob_at_buy": implied,
        "pred": pred,
        "edge": pred - implied,
        "market_category": "esports",
        "model_version": "abc123",
        "trade_ts": _NOW,
        "bet_size_usd": 42.0,
    }


def test_accepts_only_gate_buy() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    assert evaluator.accepts(_alert(body=_good_body(), detector="gate_buy")) is True
    assert evaluator.accepts(_alert(body=_good_body(), detector="smart_money")) is False
    assert evaluator.accepts(_alert(body=_good_body(), detector="velocity")) is False


def test_parse_returns_one_signal_with_metadata() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    signals = evaluator.parse(
        _alert(body=_good_body(condition_id="0xc1", side="YES", pred=0.85, implied=0.40))
    )
    assert len(signals) == 1
    sig = signals[0]
    assert sig.condition_id == ConditionId("0xc1")
    assert sig.side == "YES"
    assert sig.rule_variant is None
    assert sig.metadata["pred"] == pytest.approx(0.85)
    assert sig.metadata["implied"] == pytest.approx(0.40)
    assert sig.metadata["edge"] == pytest.approx(0.45)


def test_parse_returns_empty_on_missing_fields() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    assert evaluator.parse(_alert(body={"missing": "fields"})) == []


def test_parse_rejects_non_dict_body() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    assert evaluator.parse(_alert(body="not-a-dict")) == []


def test_parse_rejects_invalid_side() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    body = _good_body(side="MAYBE")
    assert evaluator.parse(_alert(body=body)) == []


def test_quality_passes_above_floor() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig(min_edge_pct=0.01))
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.85, "implied": 0.40, "edge": 0.45},
    )
    assert evaluator.quality_passes(parsed) is True


def test_quality_passes_rejects_below_floor() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig(min_edge_pct=0.10))
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.45, "implied": 0.40, "edge": 0.05},
    )
    assert evaluator.quality_passes(parsed) is False


def test_quality_passes_rejects_missing_edge() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.85},
    )
    assert evaluator.quality_passes(parsed) is False


def test_size_is_constant_bankroll_fraction() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig(position_fraction=0.005))
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.85, "implied": 0.40, "edge": 0.45},
    )
    assert evaluator.size(bankroll=10_000.0, parsed=parsed) == pytest.approx(50.0)
    assert evaluator.size(bankroll=1_000.0, parsed=parsed) == pytest.approx(5.0)


def test_size_independent_of_metadata() -> None:
    """Sizing is uniform across alerts — no Kelly, no edge-scaled sizing."""
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig(position_fraction=0.01))
    a = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.55, "implied": 0.50, "edge": 0.05},
    )
    b = ParsedSignal(
        condition_id=ConditionId("0xc2"),
        side="NO",
        rule_variant=None,
        metadata={"pred": 0.95, "implied": 0.10, "edge": 0.85},
    )
    assert evaluator.size(bankroll=1_000.0, parsed=a) == evaluator.size(bankroll=1_000.0, parsed=b)
