"""Unit tests for SmartMoneyEvaluator."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import SmartMoneyEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import TrackedWalletsRepo
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.smart_money import SmartMoneyEvaluator

_NOW = 1_700_000_000


def _alert(*, body: dict[str, Any], detector: str = "smart_money") -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key="k1",
        severity="med",
        title="t",
        body=body,
        created_at=_NOW,
    )


def _seed_wallet(repo: TrackedWalletsRepo, *, address: str, weighted_edge: float) -> None:
    """Seed a tracked-wallet row with the given weighted_edge."""
    repo.upsert(
        address=address,
        closed_position_count=10,
        closed_position_wins=6,
        winrate=0.6,
        mean_edge=weighted_edge,
        weighted_edge=weighted_edge,
        excess_pnl_usd=1500.0,
        total_stake_usd=10000.0,
    )


def _evaluator(
    tmp_db: sqlite3.Connection,
    *,
    position_fraction: float = 0.01,
    min_weighted_edge: float = 0.0,
) -> tuple[SmartMoneyEvaluator, TrackedWalletsRepo]:
    cfg = SmartMoneyEvaluatorConfig(
        position_fraction=position_fraction,
        min_weighted_edge=min_weighted_edge,
    )
    repo = TrackedWalletsRepo(tmp_db)
    return SmartMoneyEvaluator(config=cfg, tracked_wallets=repo), repo


def test_accepts_only_smart_money(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    assert ev.accepts(_alert(body={}, detector="smart_money")) is True
    assert ev.accepts(_alert(body={}, detector="velocity")) is False


def test_parse_extracts_wallet_condition_side(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    body = {"wallet": "0xabc", "condition_id": "0xc1", "side": "yes"}
    [parsed] = ev.parse(_alert(body=body))
    assert parsed.condition_id == ConditionId("0xc1")
    assert parsed.side == "yes"
    assert parsed.rule_variant is None
    assert parsed.metadata == {"wallet": "0xabc"}


def test_parse_returns_empty_on_missing_field(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    # Missing 'side'.
    body = {"wallet": "0xabc", "condition_id": "0xc1"}
    assert ev.parse(_alert(body=body)) == []


def test_quality_passes_uses_weighted_edge(tmp_db: sqlite3.Connection) -> None:
    ev, repo = _evaluator(tmp_db, min_weighted_edge=0.0)
    _seed_wallet(repo, address="0xabc", weighted_edge=0.4)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        metadata={"wallet": "0xabc"},
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_rejects_unknown_wallet(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        metadata={"wallet": "0xunknown"},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_rejects_below_min_edge(tmp_db: sqlite3.Connection) -> None:
    ev, repo = _evaluator(tmp_db, min_weighted_edge=0.5)
    _seed_wallet(repo, address="0xabc", weighted_edge=0.3)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        metadata={"wallet": "0xabc"},
    )
    assert ev.quality_passes(parsed) is False


def test_size_returns_bankroll_times_fraction(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db, position_fraction=0.01)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        metadata={"wallet": "0xabc"},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(10.0)
