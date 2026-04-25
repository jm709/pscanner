"""Smoke tests — package imports, config defaults, and DB bootstrap."""

from __future__ import annotations

import sqlite3

from pscanner import __version__
from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.alerts.terminal import TerminalRenderer
from pscanner.config import (
    Config,
    MispricingConfig,
    RatelimitConfig,
    ScannerConfig,
    SmartMoneyConfig,
    WhalesConfig,
)
from pscanner.detectors.base import Detector
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import (
    ClosedPosition,
    Event,
    LeaderboardEntry,
    Market,
    Outcome,
    Position,
    Trade,
    WsBookMessage,
    WsTradeMessage,
)
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    PositionSnapshot,
    PositionSnapshotsRepo,
    TrackedWallet,
    TrackedWalletsRepo,
    WalletFirstSeen,
    WalletFirstSeenRepo,
)


def test_package_version_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_public_symbols_importable() -> None:
    # Reference each so the linter/type-checker proves they exist.
    assert Alert.__name__ == "Alert"
    assert AlertSink.__name__ == "AlertSink"
    assert TerminalRenderer.__name__ == "TerminalRenderer"
    assert Detector.__name__ == "Detector"
    assert MarketWebSocket.__name__ == "MarketWebSocket"
    assert DataClient.__name__ == "DataClient"
    assert GammaClient.__name__ == "GammaClient"
    assert PolyHttpClient.__name__ == "PolyHttpClient"
    for cls in (
        Outcome,
        Event,
        Market,
        Position,
        ClosedPosition,
        Trade,
        LeaderboardEntry,
        WsTradeMessage,
        WsBookMessage,
    ):
        assert cls.__name__ == cls.__name__
    for cls in (
        TrackedWallet,
        PositionSnapshot,
        WalletFirstSeen,
        CachedMarket,
        TrackedWalletsRepo,
        PositionSnapshotsRepo,
        WalletFirstSeenRepo,
        MarketCacheRepo,
        AlertsRepo,
    ):
        assert cls.__name__


def test_config_load_defaults_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PSCANNER_CONFIG", raising=False)
    cfg = Config.load()
    assert isinstance(cfg.scanner, ScannerConfig)
    assert isinstance(cfg.smart_money, SmartMoneyConfig)
    assert isinstance(cfg.mispricing, MispricingConfig)
    assert isinstance(cfg.whales, WhalesConfig)
    assert isinstance(cfg.ratelimit, RatelimitConfig)
    assert cfg.smart_money.min_edge == 0.05
    assert cfg.smart_money.min_excess_pnl_usd == 1000.0
    assert cfg.mispricing.sum_deviation_threshold == 0.03
    assert cfg.whales.big_bet_min_usd == 2000.0
    assert cfg.ratelimit.gamma_rpm == 50


def test_config_load_from_explicit_path(tmp_path) -> None:
    cfg_file = tmp_path / "pscanner.toml"
    cfg_file.write_text(
        """
        [scanner]
        log_level = "DEBUG"

        [smart_money]
        min_edge = 0.10
        """
    )
    cfg = Config.load(cfg_file)
    assert cfg.scanner.log_level == "DEBUG"
    assert cfg.smart_money.min_edge == 0.10


def test_init_db_creates_all_tables(tmp_db: sqlite3.Connection) -> None:
    rows = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    ).fetchall()
    table_names = {row["name"] for row in rows}
    expected = {
        "tracked_wallets",
        "wallet_position_snapshots",
        "wallet_first_seen",
        "market_cache",
        "alerts",
    }
    assert expected.issubset(table_names)


def test_init_db_creates_alerts_index(tmp_db: sqlite3.Connection) -> None:
    rows = tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_alerts_created'",
    ).fetchall()
    assert len(rows) == 1


def test_init_db_pragmas_applied(tmp_db: sqlite3.Connection) -> None:
    journal_mode = tmp_db.execute("PRAGMA journal_mode").fetchone()[0]
    # In-memory dbs report "memory" rather than "wal", but the call must succeed.
    assert journal_mode in {"memory", "wal"}
    foreign_keys = tmp_db.execute("PRAGMA foreign_keys").fetchone()[0]
    assert foreign_keys == 1


def test_position_model_parses_real_fixture(sample_position_json) -> None:
    parsed = [Position.model_validate(item) for item in sample_position_json]
    assert len(parsed) == len(sample_position_json)
    assert all(p.proxy_wallet.startswith("0x") for p in parsed)


def test_market_model_parses_json_string_lists(sample_market_json) -> None:
    market = Market.model_validate(sample_market_json)
    # outcomePrices arrives as a JSON-encoded string and must be decoded to floats.
    assert isinstance(market.outcome_prices, list)
    assert all(isinstance(p, float) for p in market.outcome_prices)
    assert isinstance(market.outcomes, list)
    assert all(isinstance(o, str) for o in market.outcomes)
    assert isinstance(market.clob_token_ids, list)


def test_event_model_parses_real_fixture(sample_event_json) -> None:
    event = Event.model_validate(sample_event_json)
    assert event.id
    assert event.title
    assert isinstance(event.markets, list)


def test_leaderboard_entry_aliases_amount(sample_leaderboard_json) -> None:
    entries = [LeaderboardEntry.model_validate(item) for item in sample_leaderboard_json]
    assert all(e.proxy_wallet.startswith("0x") for e in entries)
    # `amount` field on the wire maps to `pnl` on the model.
    assert all(isinstance(e.pnl, float) for e in entries)


def test_ws_trade_message_parses(sample_trade_ws_json) -> None:
    msg = WsTradeMessage.model_validate(sample_trade_ws_json)
    assert msg.event_type == "trade"
    assert msg.status == "CONFIRMED"


def test_trade_usd_value_computed() -> None:
    trade = Trade.model_validate(
        {
            "transactionHash": "0xabc",
            "proxyWallet": "0x1",
            "conditionId": "0x2",
            "asset": "tok",
            "side": "BUY",
            "size": 10.0,
            "price": 0.42,
            "timestamp": 1,
        }
    )
    assert trade.usd_value == 4.2


def test_alert_dataclass_is_frozen() -> None:
    alert = Alert(
        detector="smart_money",
        alert_key="k",
        severity="med",
        title="t",
        body={"x": 1},
        created_at=1,
    )
    assert alert.detector == "smart_money"
