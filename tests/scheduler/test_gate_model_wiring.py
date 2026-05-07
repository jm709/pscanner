"""Wiring tests: Scanner builds gate-model components when enabled (#79)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import xgboost as xgb

from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector
from pscanner.config import Config, GateModelConfig, GateModelMarketFilterConfig
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.scheduler import Scanner, SchedulerClients, _load_corpus_metadata


def _make_stub_clients() -> SchedulerClients:
    gamma_http = MagicMock()
    gamma_http.aclose = AsyncMock()
    data_http = MagicMock()
    data_http.aclose = AsyncMock()

    gamma_client = MagicMock()
    gamma_client.aclose = AsyncMock()
    gamma_client.list_events = AsyncMock(return_value=[])
    gamma_client.list_markets = AsyncMock(return_value=[])

    data_client = MagicMock()
    data_client.aclose = AsyncMock()
    data_client.get_leaderboard = AsyncMock(return_value=[])
    data_client.get_activity = AsyncMock(return_value=[])
    data_client.get_market_trades = AsyncMock(return_value=[])

    ticks_ws = MagicMock()
    ticks_ws.close = AsyncMock()
    ticks_ws.connect = AsyncMock()
    ticks_ws.subscribe = AsyncMock()

    async def _empty_messages() -> AsyncIterator[Any]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    ticks_ws.messages = MagicMock(return_value=_empty_messages())

    return SchedulerClients(
        gamma_http=gamma_http,
        data_http=data_http,
        gamma_client=gamma_client,
        data_client=data_client,
        ticks_ws=ticks_ws,
    )


def _train_dummy_model(out_dir: Path) -> None:
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


def _make_config(
    *,
    artifact_dir: Path,
    gate_enabled: bool,
    filter_enabled: bool,
) -> Config:
    return Config(
        gate_model=GateModelConfig(enabled=gate_enabled, artifact_dir=artifact_dir),
        gate_model_market_filter=GateModelMarketFilterConfig(enabled=filter_enabled),
    )


def _seed_wallet_state_live(scanner: Scanner) -> None:
    scanner._db.execute(
        """
        INSERT INTO wallet_state_live (wallet_address, first_seen_ts)
        VALUES ('0xseed', 1700000000)
        """
    )
    scanner._db.commit()


@pytest.mark.asyncio
async def test_scanner_omits_gate_model_when_disabled(tmp_path: Path) -> None:
    cfg = _make_config(artifact_dir=tmp_path / "model", gate_enabled=False, filter_enabled=False)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        assert "gate_model" not in scanner._detectors
        assert "market_scoped_trades" not in scanner._collectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_builds_gate_model_when_enabled(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(artifact_dir=artifact_dir, gate_enabled=True, filter_enabled=True)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        assert isinstance(scanner._detectors.get("gate_model"), GateModelDetector)
        assert isinstance(
            scanner._collectors.get("market_scoped_trades"), MarketScopedTradeCollector
        )
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_preflight_refuses_to_start_when_wallet_state_empty(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(artifact_dir=artifact_dir, gate_enabled=True, filter_enabled=True)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        with pytest.raises(RuntimeError, match="bootstrap-features"):
            scanner.preflight()
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_preflight_passes_when_wallet_state_seeded(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(artifact_dir=artifact_dir, gate_enabled=True, filter_enabled=True)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        _seed_wallet_state_live(scanner)
        scanner.preflight()  # no exception
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_preflight_noops_when_gate_model_disabled(tmp_path: Path) -> None:
    cfg = _make_config(artifact_dir=tmp_path / "model", gate_enabled=False, filter_enabled=False)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        scanner.preflight()  # no exception
    finally:
        await scanner.aclose()


def test_load_corpus_metadata_filters_by_platform(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``_load_corpus_metadata(platform=...)`` returns only that platform's markets."""
    corpus_path = tmp_path / "corpus.sqlite3"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    real_corpus_path = tmp_path / "data" / "corpus.sqlite3"
    conn = init_corpus_db(real_corpus_path)
    try:
        markets = CorpusMarketsRepo(conn)
        markets.insert_pending(
            CorpusMarket(
                condition_id="0xpoly",
                event_slug="p",
                category="esports",
                closed_at=1_700_001_000,
                total_volume_usd=1.0,
                enumerated_at=1_699_900_000,
                market_slug="p-m",
                platform="polymarket",
            )
        )
        markets.insert_pending(
            CorpusMarket(
                condition_id="manifold-cond",
                event_slug="m",
                category="politics",
                closed_at=1_700_001_500,
                total_volume_usd=2.0,
                enumerated_at=1_699_950_000,
                market_slug="m-m",
                platform="manifold",
            )
        )
    finally:
        conn.close()
    del corpus_path

    poly = _load_corpus_metadata(platform="polymarket")
    manifold = _load_corpus_metadata(platform="manifold")

    assert set(poly.keys()) == {"0xpoly"}
    assert set(manifold.keys()) == {"manifold-cond"}
    assert poly["0xpoly"].category == "esports"
    assert manifold["manifold-cond"].category == "politics"
