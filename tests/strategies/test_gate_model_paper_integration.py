"""End-to-end: gate_buy alert -> PaperTrader -> paper_trades row (#80)."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import (
    EvaluatorsConfig,
    GateModelEvaluatorConfig,
    PaperTradingConfig,
)
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
)
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
from pscanner.strategies.paper_trader import PaperTrader

_NOW = 1_700_000_000


def _gate_buy_alert(
    *,
    tx: str = "tx-gate-1",
    condition_id: str = "0xc1",
    side: str = "YES",
    pred: float = 0.85,
    implied: float = 0.40,
) -> Alert:
    return Alert(
        detector="gate_buy",
        alert_key=f"gate:{tx}:{side}",
        severity="med",
        title=f"gate_buy {side} on {condition_id}",
        body={
            "wallet": "0xabc",
            "condition_id": condition_id,
            "side": side,
            "implied_prob_at_buy": implied,
            "pred": pred,
            "edge": pred - implied,
            "top_category": "esports",
            "model_version": "abc123",
            "trade_ts": _NOW,
            "bet_size_usd": 42.0,
        },
        created_at=_NOW,
    )


def _cache_market(repo: MarketCacheRepo) -> None:
    repo.upsert(
        CachedMarket(
            market_id=MarketId("mkt-c1"),
            event_id=None,
            title="Esports market",
            liquidity_usd=1000.0,
            volume_usd=5_000.0,
            outcome_prices=[0.6, 0.4],
            outcomes=["YES", "NO"],
            asset_ids=[AssetId("asset-yes"), AssetId("asset-no")],
            active=True,
            cached_at=_NOW,
            condition_id=ConditionId("0xc1"),
            event_slug=None,
        ),
    )


def _seed_tick(conn: sqlite3.Connection, *, asset_id: str, best_ask: float) -> None:
    conn.execute(
        """
        INSERT INTO market_ticks (asset_id, condition_id, snapshot_at, mid_price,
          best_bid, best_ask, spread, bid_depth_top5, ask_depth_top5,
          last_trade_price)
        VALUES (?, '0xc1', ?, NULL, NULL, ?, NULL, NULL, NULL, NULL)
        """,
        (asset_id, _NOW, best_ask),
    )
    conn.commit()


def _no_op_clients() -> tuple[AsyncMock, AsyncMock]:
    data = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = None
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = None
    return data, gamma


def _build_trader(tmp_db: sqlite3.Connection) -> tuple[AlertSink, PaperTrader, PaperTradesRepo]:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.40)
    sink = AlertSink(AlertsRepo(tmp_db))
    paper = PaperTradesRepo(tmp_db)
    cfg = PaperTradingConfig(
        enabled=True,
        starting_bankroll_usd=10_000.0,
        evaluators=EvaluatorsConfig(
            gate_model=GateModelEvaluatorConfig(
                enabled=True, min_edge_pct=0.01, position_fraction=0.005
            ),
        ),
    )
    evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
    data, gamma = _no_op_clients()
    trader = PaperTrader(
        config=cfg,
        evaluators=[evaluator],
        market_cache=cache,
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=data,
        gamma_client=gamma,
    )
    sink.subscribe(trader.handle_alert_sync)
    return sink, trader, paper


async def _drain() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_gate_buy_alert_books_paper_trade(tmp_db: sqlite3.Connection) -> None:
    sink, trader, paper = _build_trader(tmp_db)
    await sink.emit(_gate_buy_alert())
    await _drain()
    await trader.aclose()
    rows = paper.list_open_positions()
    assert len(rows) == 1
    p = rows[0]
    assert p.triggering_alert_detector == "gate_buy"
    assert p.rule_variant is None
    assert p.outcome.upper() == "YES"
    assert p.fill_price == pytest.approx(0.40)
    # cost = 10_000 * 0.005 = 50; shares = cost / fill = 125
    assert p.cost_usd == pytest.approx(50.0)
    assert p.shares == pytest.approx(125.0)


@pytest.mark.asyncio
async def test_gate_buy_below_edge_floor_skipped(tmp_db: sqlite3.Connection) -> None:
    sink, trader, paper = _build_trader(tmp_db)
    # edge = pred - implied = 0.41 - 0.40 = 0.01 == min_edge_pct, passes;
    # bump implied to 0.405 to drop below 0.01.
    await sink.emit(_gate_buy_alert(pred=0.41, implied=0.405))
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_per_source_breakdown_aggregates_gate_buy(tmp_db: sqlite3.Connection) -> None:
    sink, trader, _paper = _build_trader(tmp_db)
    for i in range(3):
        await sink.emit(_gate_buy_alert(tx=f"tx-{i}"))
        await _drain()
    await trader.aclose()
    rows = tmp_db.execute(
        """
        SELECT triggering_alert_detector, rule_variant, COUNT(*) AS n
        FROM paper_trades
        GROUP BY triggering_alert_detector, rule_variant
        """
    ).fetchall()
    by_source = {(row[0], row[1]): row[2] for row in rows}
    assert by_source.get(("gate_buy", None)) == 3
