"""Unit tests for GateModelDetector (#79)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from structlog.testing import capture_logs

from pscanner.alerts.sink import AlertSink
from pscanner.config import GateModelConfig
from pscanner.corpus.features import MarketMetadata
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, WalletTrade


def _train_dummy_model(out_dir: Path) -> None:
    """Train a 1-feature stub model and persist artifacts in the layout the detector expects."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 1))
    y = (x[:, 0] > 0.5).astype(int)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "max_depth": 2,
            "tree_method": "hist",
            "verbosity": 0,
        },
        dtrain=xgb.DMatrix(x, label=y, feature_names=["x0"]),
        num_boost_round=5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    (out_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "leakage_cols": [],
                "carrier_cols": [],
                "encoder": {"levels": {}},
                "accepted_categories": ["esports"],
                "platform": "polymarket",
            }
        )
    )


def _new_db() -> sqlite3.Connection:
    return init_db(Path(":memory:"))


def test_detector_loads_model_and_accepted_categories(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.7)
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=cfg,
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
    finally:
        conn.close()
    assert detector.name == "gate_model"
    assert detector.accepted_categories == ("esports",)


def test_detector_overrides_accepted_categories_from_config(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(
        enabled=True,
        artifact_dir=artifact_dir,
        accepted_categories=("sports", "esports"),
    )
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=AlertsRepo(conn))
    finally:
        conn.close()
    assert detector.accepted_categories == ("sports", "esports")


def _make_wallet_trade(
    *,
    side: str = "BUY",
    wallet: str = "0xabc",
    condition_id: str = "0xc1",
    asset_id: str = "0xa1",
    price: float = 0.42,
    size: float = 100.0,
    usd_value: float = 42.0,
    timestamp: int = 1_700_000_000,
) -> WalletTrade:
    return WalletTrade(
        transaction_hash=f"tx{timestamp}",
        asset_id=AssetId(asset_id),
        side=side,
        wallet=wallet,
        condition_id=ConditionId(condition_id),
        size=size,
        price=price,
        usd_value=usd_value,
        status="filled",
        source="market_scoped",
        timestamp=timestamp,
        recorded_at=timestamp + 1,
    )


def test_pre_screen_skips_sell_trade(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(side="SELL")
        assert detector._should_score(trade) is False
    finally:
        conn.close()


def test_pre_screen_accepts_buy(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(side="BUY")
        assert detector._should_score(trade) is True
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_queue_full_drops_with_warning(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        cfg = GateModelConfig(
            enabled=True,
            artifact_dir=artifact_dir,
            queue_max_size=2,
        )
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=AlertsRepo(conn))
        # Don't start the worker — queue fills up.
        with capture_logs() as logs:
            for i in range(5):
                detector.handle_trade_sync(_make_wallet_trade(timestamp=1_700_000_000 + i))
        events = [le["event"] for le in logs]
    finally:
        conn.close()
    assert any(e == "gate_model.queue_full" for e in events)


@pytest.mark.asyncio
async def test_queue_drops_sell_trades_before_enqueue(tmp_path: Path) -> None:
    """SELLs should never enter the queue (pre-screen happens before enqueue)."""
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, queue_max_size=1)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=AlertsRepo(conn))
        # 10 SELLs into a queue with capacity 1 — none should drop because
        # all are filtered out before enqueue.
        with capture_logs() as logs:
            for i in range(10):
                detector.handle_trade_sync(
                    _make_wallet_trade(side="SELL", timestamp=1_700_000_000 + i)
                )
        events = [le["event"] for le in logs]
    finally:
        conn.close()
    assert "gate_model.queue_full" not in events


@pytest.mark.asyncio
async def test_evaluate_emits_alert_when_gates_pass(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="esports",
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        # Replace prediction + outcome side with deterministic stubs.
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]
        sink = AlertSink(alerts_repo=alerts_repo)
        detector._sink = sink
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert len(recent) == 1
    body = recent[0].body
    assert isinstance(body, dict)
    assert body["condition_id"] == "0xc1"
    assert body["pred"] == pytest.approx(0.85)
    assert body["implied_prob_at_buy"] == pytest.approx(0.40)
    assert body["edge"] == pytest.approx(0.85 - 0.40)


@pytest.mark.asyncio
async def test_evaluate_skips_when_pred_below_floor(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="esports",
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.7)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.30  # type: ignore[method-assign,assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.20)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert recent == []


@pytest.mark.asyncio
async def test_evaluate_skips_when_category_not_accepted(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="politics",  # NOT in accepted_categories=("esports",)
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert recent == []
