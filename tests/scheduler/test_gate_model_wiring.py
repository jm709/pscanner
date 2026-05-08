"""Wiring tests: Scanner builds gate-model components when enabled (#79)."""

from __future__ import annotations

import json
import sqlite3
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
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    MarketResolution,
    MarketResolutionsRepo,
)
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.scheduler import (
    Scanner,
    SchedulerClients,
    _load_corpus_metadata,
    _load_corpus_resolutions,
)
from pscanner.store.db import init_db


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


def test_load_corpus_resolutions_populates_provider(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Resolutions loaded at scheduler boot enable the lazy drain in wallet_state."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    real_corpus_path = tmp_path / "data" / "corpus.sqlite3"
    conn = init_corpus_db(real_corpus_path)
    try:
        resolutions = MarketResolutionsRepo(conn)
        resolutions.upsert(
            MarketResolution(
                condition_id="0xpoly-1",
                winning_outcome_index=0,
                outcome_yes_won=1,
                resolved_at=1_700_001_000,
                source="gamma",
                platform="polymarket",
            ),
            recorded_at=1_700_001_500,
        )
        resolutions.upsert(
            MarketResolution(
                condition_id="0xpoly-2",
                winning_outcome_index=0,
                outcome_yes_won=0,
                resolved_at=1_700_002_000,
                source="gamma",
                platform="polymarket",
            ),
            recorded_at=1_700_002_500,
        )
        resolutions.upsert(
            MarketResolution(
                condition_id="manifold-cond",
                winning_outcome_index=0,
                outcome_yes_won=1,
                resolved_at=1_700_003_000,
                source="manifold",
                platform="manifold",
            ),
            recorded_at=1_700_003_500,
        )
    finally:
        conn.close()

    daemon_conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        provider = LiveHistoryProvider(conn=daemon_conn, metadata={})
        n = _load_corpus_resolutions(provider, platform="polymarket")
        assert n == 2  # only Polymarket rows; the manifold row is filtered
        assert provider.get_resolution("0xpoly-1") == (1_700_001_000, 1)
        assert provider.get_resolution("0xpoly-2") == (1_700_002_000, 0)
        assert provider.get_resolution("manifold-cond") is None
    finally:
        daemon_conn.close()


def test_load_corpus_resolutions_returns_zero_when_corpus_missing(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Empty-dict fallback path mirrors `_load_corpus_metadata`'s contract."""
    monkeypatch.chdir(tmp_path)
    # No data/corpus.sqlite3 exists.
    daemon_conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        provider = LiveHistoryProvider(conn=daemon_conn, metadata={})
        n = _load_corpus_resolutions(provider, platform="polymarket")
    finally:
        daemon_conn.close()
    assert n == 0


def test_load_corpus_resolutions_handles_unmigrated_market_resolutions(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Pre-platform corpus DBs (no ``platform`` column on ``market_resolutions``)
    fall back to zero rather than crashing the daemon at boot.

    The laptop and any pre-2026-05-04 corpus may have ``corpus_markets``
    migrated but ``market_resolutions`` not yet (partial-migration state).
    The function should warn and return 0 instead of bubbling
    ``OperationalError`` up through ``Scanner.__init__``.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    corpus_path = tmp_path / "data" / "corpus.sqlite3"
    # Build a market_resolutions table WITHOUT the platform column —
    # mirrors a pre-migration corpus snapshot.
    conn = sqlite3.connect(str(corpus_path))
    try:
        conn.execute(
            """
            CREATE TABLE market_resolutions (
              condition_id TEXT PRIMARY KEY,
              winning_outcome_index INTEGER NOT NULL,
              outcome_yes_won INTEGER NOT NULL,
              resolved_at INTEGER NOT NULL,
              source TEXT NOT NULL,
              recorded_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO market_resolutions VALUES (?, ?, ?, ?, ?, ?)",
            ("0xtest", 0, 1, 1_700_001_000, "gamma", 1_700_001_500),
        )
        conn.commit()
    finally:
        conn.close()

    daemon_conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        provider = LiveHistoryProvider(conn=daemon_conn, metadata={})
        n = _load_corpus_resolutions(provider, platform="polymarket")
    finally:
        daemon_conn.close()
    assert n == 0
    assert provider.get_resolution("0xtest") is None


@pytest.mark.asyncio
async def test_scanner_loads_resolutions_when_gate_model_enabled(tmp_path: Path) -> None:
    """Scanner.__init__ calls _load_corpus_resolutions on the live provider.

    End-to-end: seed the corpus with resolutions, enable gate_model, build a
    Scanner, observe that provider.get_resolution returns the loaded entry.
    """
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    (tmp_path / "data").mkdir()
    real_corpus_path = tmp_path / "data" / "corpus.sqlite3"
    conn = init_corpus_db(real_corpus_path)
    try:
        resolutions = MarketResolutionsRepo(conn)
        resolutions.upsert(
            MarketResolution(
                condition_id="0xseed",
                winning_outcome_index=0,
                outcome_yes_won=1,
                resolved_at=1_700_001_000,
                source="gamma",
                platform="polymarket",
            ),
            recorded_at=1_700_001_500,
        )
    finally:
        conn.close()
    cfg = _make_config(artifact_dir=artifact_dir, gate_enabled=True, filter_enabled=True)
    clients = _make_stub_clients()
    # Scanner reads ``data/corpus.sqlite3`` relative to cwd; chdir into the
    # seeded tmp dir so it picks up the test fixture.
    monkeypatch_setter = pytest.MonkeyPatch()
    monkeypatch_setter.chdir(tmp_path)
    try:
        scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    finally:
        monkeypatch_setter.undo()
    try:
        provider = scanner._live_history_provider
        assert provider is not None
        assert provider.get_resolution("0xseed") == (1_700_001_000, 1)
    finally:
        await scanner.aclose()
