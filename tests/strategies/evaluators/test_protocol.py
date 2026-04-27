"""Smoke tests for the Protocol surface (import + dataclass roundtrip)."""

from __future__ import annotations

from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal, SignalEvaluator


def test_parsed_signal_roundtrip() -> None:
    """ParsedSignal stores its fields and exposes them by attribute."""
    sig = ParsedSignal(
        condition_id=ConditionId("0xabc"),
        side="yes",
        rule_variant="follow",
        metadata={"wallet": "0x123"},
    )
    assert sig.condition_id == "0xabc"
    assert sig.side == "yes"
    assert sig.rule_variant == "follow"
    assert sig.metadata == {"wallet": "0x123"}


def test_parsed_signal_default_rule_variant_is_none() -> None:
    """Single-entry sources omit rule_variant; defaults to None."""
    sig = ParsedSignal(condition_id=ConditionId("0xabc"), side="no")
    assert sig.rule_variant is None
    assert sig.metadata == {}


def test_signal_evaluator_protocol_importable() -> None:
    """Protocol type is importable; structural-only — no runtime check."""
    # Ensures the import path is stable for downstream evaluators.
    assert SignalEvaluator is not None
