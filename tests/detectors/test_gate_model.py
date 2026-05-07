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
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo


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
