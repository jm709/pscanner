"""Unit tests for GateModelDetector (#79)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import xgboost as xgb

from pscanner.config import GateModelConfig
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
