"""Unit tests for VelocityEvaluator (twin trades)."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import VelocityEvaluatorConfig
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import CachedMarket, MarketCacheRepo
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.velocity import VelocityEvaluator


def _alert(
    *,
    body: dict[str, Any],
    detector: str = "velocity",
    severity: str = "high",
) -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key="k1",
        severity=severity,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        title="t",
        body=body,
        created_at=0,
    )


def _seed_market(
    cache: MarketCacheRepo,
    *,
    condition_id: str,
    outcomes: list[str],
    asset_ids: list[str],
) -> None:
    cache.upsert(
        CachedMarket(
            market_id="mkt-1",  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_id=None,
            title="Test market",
            liquidity_usd=100_000.0,
            volume_usd=10_000.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=0,
            condition_id=ConditionId(condition_id),
            event_slug=None,
            outcomes=outcomes,
            asset_ids=[AssetId(a) for a in asset_ids],
        ),
    )


def _evaluator(
    tmp_db: sqlite3.Connection,
    *,
    position_fraction: float = 0.0025,
    min_severity: str = "high",
    allow_consolidation: bool = False,
) -> tuple[VelocityEvaluator, MarketCacheRepo]:
    cfg = VelocityEvaluatorConfig(
        position_fraction=position_fraction,
        min_severity=min_severity,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        allow_consolidation=allow_consolidation,
    )
    cache = MarketCacheRepo(tmp_db)
    return VelocityEvaluator(config=cfg, market_cache=cache), cache


def test_accepts_only_velocity(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    assert ev.accepts(_alert(body={}, detector="velocity")) is True
    assert ev.accepts(_alert(body={}, detector="smart_money")) is False


def test_parse_returns_two_paired_signals(tmp_db: sqlite3.Connection) -> None:
    """A binary YES/NO market emits a follow + fade pair."""
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes", "no"],
        asset_ids=["asset-yes", "asset-no"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-yes",
        "change_pct": 0.12,
        "consolidation": False,
    }

    signals = ev.parse(_alert(body=body))

    assert len(signals) == 2
    follow = next(s for s in signals if s.rule_variant == "follow")
    fade = next(s for s in signals if s.rule_variant == "fade")
    assert follow.side == "yes"
    assert fade.side == "no"
    assert follow.condition_id == fade.condition_id == ConditionId("0xc1")


def test_parse_follow_is_second_outcome_when_alert_on_no(
    tmp_db: sqlite3.Connection,
) -> None:
    """Symmetric case: alert on outcomes[1] → follow=no, fade=yes.

    The parallel-list walk must correctly identify follow regardless of
    which slot the alert's asset_id occupies. A regression in the loop
    ordering would only show up here, not in the
    test_parse_returns_two_paired_signals case.
    """
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes", "no"],
        asset_ids=["asset-yes", "asset-no"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-no",  # second outcome
        "change_pct": -0.12,
        "consolidation": False,
    }

    signals = ev.parse(_alert(body=body))

    assert len(signals) == 2
    follow = next(s for s in signals if s.rule_variant == "follow")
    fade = next(s for s in signals if s.rule_variant == "fade")
    assert follow.side == "no"
    assert fade.side == "yes"


def test_parse_returns_empty_on_cache_miss(tmp_db: sqlite3.Connection) -> None:
    """If market_cache has no entry, neither side resolves; skip the alert."""
    ev, _ = _evaluator(tmp_db)
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-yes",
        "change_pct": 0.12,
        "consolidation": False,
    }
    assert ev.parse(_alert(body=body)) == []


def test_parse_returns_only_follow_if_market_has_one_outcome(
    tmp_db: sqlite3.Connection,
) -> None:
    """1-outcome market: fade has no opposing side, return follow only."""
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes"],
        asset_ids=["asset-yes"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-yes",
        "change_pct": 0.12,
        "consolidation": False,
    }
    signals = ev.parse(_alert(body=body))
    assert len(signals) == 1
    assert signals[0].rule_variant == "follow"


def test_parse_returns_empty_when_alert_asset_id_unknown(
    tmp_db: sqlite3.Connection,
) -> None:
    """Cached market exists but doesn't contain the alert's asset_id."""
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes", "no"],
        asset_ids=["asset-yes", "asset-no"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-other",  # not in the cached market
        "change_pct": 0.12,
        "consolidation": False,
    }
    assert ev.parse(_alert(body=body)) == []


def test_parse_returns_empty_on_missing_field(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    # Missing condition_id.
    assert ev.parse(_alert(body={"asset_id": "asset-yes"})) == []


def test_parse_returns_empty_on_missing_asset_id(
    tmp_db: sqlite3.Connection,
) -> None:
    """The isinstance guard rejects when asset_id is absent (symmetric to
    test_parse_returns_empty_on_missing_field which covers condition_id)."""
    ev, _ = _evaluator(tmp_db)
    assert ev.parse(_alert(body={"condition_id": "0xc1"})) == []


def test_quality_passes_high_severity_no_consolidation(
    tmp_db: sqlite3.Connection,
) -> None:
    ev, _ = _evaluator(tmp_db)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "high", "consolidation": False},
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_rejects_low_severity(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "low", "consolidation": False},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_rejects_consolidation_when_disallowed(
    tmp_db: sqlite3.Connection,
) -> None:
    ev, _ = _evaluator(tmp_db, allow_consolidation=False)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "high", "consolidation": True},
    )
    assert ev.quality_passes(parsed) is False


def test_size_per_entry_is_bankroll_times_fraction(tmp_db: sqlite3.Connection) -> None:
    """Velocity sizes per-entry; pair total = 2x per-entry."""
    ev, _ = _evaluator(tmp_db, position_fraction=0.0025)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "high", "consolidation": False},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(2.5)
