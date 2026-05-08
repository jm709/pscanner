"""Integration tests for PaperTrader.replay_unbooked (#105)."""

from __future__ import annotations

import time
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.config import (
    EvaluatorsConfig,
    GateModelEvaluatorConfig,
    PaperTradingConfig,
)
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
)
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
from pscanner.strategies.paper_trader import PaperTrader


def _make_gate_alert(*, key: str, ts: int, condition_id: str = "0xc1") -> Alert:
    return Alert(
        detector=cast(DetectorName, "gate_buy"),
        alert_key=key,
        severity=cast(Severity, "med"),
        title=f"gate_buy on {condition_id}",
        body={
            "wallet": "0xabc",
            "condition_id": condition_id,
            "side": "YES",
            "implied_prob_at_buy": 0.5,
            "pred": 0.8,
            "edge": 0.3,
            "top_category": "esports",
            "model_version": "v1",
            "trade_ts": ts,
            "bet_size_usd": 100.0,
        },
        created_at=ts,
    )


def _seed_market(cache: MarketCacheRepo, condition_id: str = "0xc1") -> None:
    cache.upsert(
        CachedMarket(
            market_id=cast(Any, "m1"),
            event_id=cast(Any, "e1"),
            title="t",
            liquidity_usd=1000.0,
            volume_usd=1000.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=cast(Any, condition_id),
            event_slug=None,
            outcomes=["YES", "NO"],
            asset_ids=[cast(Any, "0xa1"), cast(Any, "0xa2")],
        )
    )


@pytest.mark.asyncio
async def test_replay_books_unbooked_alerts_in_window(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An alert in the lookback window with no paper_trades row is booked."""
    conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        alerts_repo = AlertsRepo(conn)
        paper = PaperTradesRepo(conn)
        cache = MarketCacheRepo(conn)
        ticks = MarketTicksRepo(conn)
        _seed_market(cache)

        now = int(time.time())
        alerts_repo.insert_if_new(_make_gate_alert(key="A", ts=now - 60))

        cfg = PaperTradingConfig(
            enabled=True,
            starting_bankroll_usd=1000.0,
            replay_lookback_seconds=300,
            evaluators=EvaluatorsConfig(
                gate_model=GateModelEvaluatorConfig(
                    enabled=True,
                    position_fraction=0.005,
                    min_edge_pct=0.01,
                ),
            ),
        )
        evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
        trader = PaperTrader(
            config=cfg,
            evaluators=[evaluator],
            market_cache=cache,
            paper_trades=paper,
            market_ticks=ticks,
            data_client=cast(Any, MagicMock()),
            gamma_client=cast(Any, MagicMock()),
            alerts_repo=alerts_repo,
        )

        booked_count = await trader.replay_unbooked()
        assert booked_count == 1
        rows = conn.execute(
            "SELECT triggering_alert_key FROM paper_trades WHERE trade_kind='entry'"
        ).fetchall()
        assert [r[0] for r in rows] == ["A"]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_replay_disabled_when_lookback_is_zero(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """replay_lookback_seconds=0 means no replay query and no books."""
    conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        alerts_repo = AlertsRepo(conn)
        paper = PaperTradesRepo(conn)
        cache = MarketCacheRepo(conn)
        ticks = MarketTicksRepo(conn)
        _seed_market(cache)

        now = int(time.time())
        alerts_repo.insert_if_new(_make_gate_alert(key="A", ts=now - 60))

        cfg = PaperTradingConfig(
            enabled=True,
            replay_lookback_seconds=0,
            evaluators=EvaluatorsConfig(
                gate_model=GateModelEvaluatorConfig(enabled=True),
            ),
        )
        evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
        trader = PaperTrader(
            config=cfg,
            evaluators=[evaluator],
            market_cache=cache,
            paper_trades=paper,
            market_ticks=ticks,
            data_client=cast(Any, MagicMock()),
            gamma_client=cast(Any, MagicMock()),
            alerts_repo=alerts_repo,
        )

        booked_count = await trader.replay_unbooked()
    finally:
        conn.close()

    assert booked_count == 0


@pytest.mark.asyncio
async def test_replay_skips_already_booked_alerts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Alerts with a paper_trades entry are excluded from replay."""
    conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        alerts_repo = AlertsRepo(conn)
        paper = PaperTradesRepo(conn)
        cache = MarketCacheRepo(conn)
        ticks = MarketTicksRepo(conn)
        _seed_market(cache)

        now = int(time.time())
        alerts_repo.insert_if_new(_make_gate_alert(key="already", ts=now - 60))
        paper.insert_entry(
            triggering_alert_key="already",
            triggering_alert_detector="gate_buy",
            rule_variant=None,
            source_wallet="0xabc",
            condition_id=cast(Any, "0xc1"),
            asset_id=cast(Any, "0xa1"),
            outcome="YES",
            shares=10.0,
            fill_price=0.5,
            cost_usd=5.0,
            nav_after_usd=1000.0,
            ts=now - 50,
        )

        cfg = PaperTradingConfig(
            enabled=True,
            replay_lookback_seconds=300,
            evaluators=EvaluatorsConfig(
                gate_model=GateModelEvaluatorConfig(enabled=True),
            ),
        )
        evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
        trader = PaperTrader(
            config=cfg,
            evaluators=[evaluator],
            market_cache=cache,
            paper_trades=paper,
            market_ticks=ticks,
            data_client=cast(Any, MagicMock()),
            gamma_client=cast(Any, MagicMock()),
            alerts_repo=alerts_repo,
        )

        booked_count = await trader.replay_unbooked()
    finally:
        conn.close()

    assert booked_count == 0
