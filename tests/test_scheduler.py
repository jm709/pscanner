"""Tests for ``pscanner.scheduler.Scanner``.

Uses dependency injection through the ``clients`` parameter to substitute
mocked Polymarket clients. Each test exercises the public Scanner surface
without touching real network/IO.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.collectors.activity import ActivityCollector
from pscanner.collectors.events import EventCollector
from pscanner.collectors.markets import MarketCollector
from pscanner.collectors.positions import PositionCollector
from pscanner.collectors.trades import TradeCollector
from pscanner.collectors.watchlist import WatchlistSyncer
from pscanner.config import (
    ActivityConfig,
    Config,
    ConvergenceConfig,
    EventsConfig,
    MarketsConfig,
    MispricingConfig,
    PositionsConfig,
    RatelimitConfig,
    ScannerConfig,
    SmartMoneyConfig,
    WhalesConfig,
)
from pscanner.detectors.convergence import ConvergenceDetector
from pscanner.poly.models import Event, LeaderboardEntry, Market, Position
from pscanner.scheduler import Scanner, SchedulerClients


def _make_market(*, market_id: str, yes_price: float) -> Market:
    return Market.model_validate(
        {
            "id": market_id,
            "question": f"market {market_id}",
            "slug": f"slug-{market_id}",
            "outcomes": ["Yes", "No"],
            "outcomePrices": [yes_price, 1.0 - yes_price],
            "liquidity": 50000.0,
            "volume": 100000.0,
            "enableOrderBook": True,
            "active": True,
            "closed": False,
            "clobTokenIds": [],
            "event_id": "evt-1",
        }
    )


def _make_event(*, mispriced: bool) -> Event:
    yes_a = 0.7 if mispriced else 0.5
    yes_b = 0.7 if mispriced else 0.5
    return Event.model_validate(
        {
            "id": "evt-1",
            "title": "Test event",
            "slug": "test",
            "liquidity": 50000.0,
            "volume": 100000.0,
            "active": True,
            "closed": False,
            "markets": [
                _make_market(market_id="m1", yes_price=yes_a).model_dump(by_alias=True),
                _make_market(market_id="m2", yes_price=yes_b).model_dump(by_alias=True),
            ],
        }
    )


def _make_config(
    *,
    enable_smart: bool = True,
    enable_misprice: bool = True,
    enable_whales: bool = True,
    enable_convergence: bool = True,
    enable_positions: bool = True,
    enable_activity: bool = True,
    enable_markets: bool = True,
    enable_events: bool = True,
) -> Config:
    return Config(
        scanner=ScannerConfig(),
        smart_money=SmartMoneyConfig(enabled=enable_smart),
        mispricing=MispricingConfig(enabled=enable_misprice),
        whales=WhalesConfig(enabled=enable_whales),
        convergence=ConvergenceConfig(enabled=enable_convergence),
        ratelimit=RatelimitConfig(),
        positions=PositionsConfig(enabled=enable_positions),
        activity=ActivityConfig(enabled=enable_activity),
        markets=MarketsConfig(enabled=enable_markets),
        events=EventsConfig(enabled=enable_events),
    )


def _events_iter(events: list[Event]) -> AsyncIterator[Event]:
    async def _gen() -> AsyncIterator[Event]:
        for event in events:
            yield event

    return _gen()


def _markets_iter(markets: list[Market]) -> AsyncIterator[Market]:
    async def _gen() -> AsyncIterator[Market]:
        for market in markets:
            yield market

    return _gen()


def _make_clients(
    *,
    events: list[Event] | None = None,
    markets: list[Market] | None = None,
    leaderboard: list[LeaderboardEntry] | None = None,
    positions: list[Position] | None = None,
) -> SchedulerClients:
    gamma_http = MagicMock()
    gamma_http.aclose = AsyncMock()
    data_http = MagicMock()
    data_http.aclose = AsyncMock()

    gamma_client = MagicMock()
    gamma_client.iter_events = MagicMock(return_value=_events_iter(events or []))
    gamma_client.iter_markets = MagicMock(return_value=_markets_iter(markets or []))
    gamma_client.list_events = AsyncMock(return_value=events or [])
    gamma_client.list_markets = AsyncMock(return_value=markets or [])
    gamma_client.aclose = AsyncMock()

    data_client = MagicMock()
    data_client.get_leaderboard = AsyncMock(return_value=leaderboard or [])
    data_client.get_positions = AsyncMock(return_value=positions or [])
    data_client.get_closed_positions = AsyncMock(return_value=[])
    data_client.get_activity = AsyncMock(return_value=[])
    data_client.aclose = AsyncMock()

    return SchedulerClients(
        gamma_http=gamma_http,
        data_http=data_http,
        gamma_client=gamma_client,
        data_client=data_client,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "scheduler.sqlite3"


@pytest.mark.asyncio
async def test_scanner_constructs_with_all_detectors_enabled(db_path: Path) -> None:
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert isinstance(scanner.sink, AlertSink)
        assert scanner.renderer is not None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_run_once_with_no_data_returns_zero_counts(db_path: Path) -> None:
    config = _make_config(enable_smart=True, enable_misprice=True, enable_whales=True)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result == {
        "events_scanned": 0,
        "alerts_emitted": 0,
        "tracked_wallets": 0,
        "markets_cached": 0,
        "watched_wallets": 0,
        "trades_recorded": 0,
        "position_snapshots": 0,
        "activity_events": 0,
        "market_snapshots": 0,
        "event_snapshots": 0,
    }


@pytest.mark.asyncio
async def test_run_once_emits_mispricing_alert(db_path: Path) -> None:
    config = _make_config(enable_smart=False, enable_misprice=True, enable_whales=False)
    events = [_make_event(mispriced=True)]
    clients = _make_clients(events=events)
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["events_scanned"] == 1
    assert result["alerts_emitted"] == 1


@pytest.mark.asyncio
async def test_run_once_caches_markets_via_whales(db_path: Path) -> None:
    config = _make_config(enable_smart=False, enable_misprice=False, enable_whales=True)
    markets = [
        _make_market(market_id="m1", yes_price=0.5),
        _make_market(market_id="m2", yes_price=0.5),
    ]
    clients = _make_clients(markets=markets)
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["markets_cached"] == 2


@pytest.mark.asyncio
async def test_run_supervisor_restarts_returning_detector(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Skip the real backoff so the test finishes quickly.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    config = _make_config(enable_smart=False, enable_misprice=True, enable_whales=False)
    clients = _make_clients()
    # Mispricing detector that returns immediately every iteration.
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    detector = scanner._detectors["mispricing"]
    call_count = {"n": 0}

    async def fast_run(_sink: AlertSink) -> None:
        call_count["n"] += 1

    detector.run = fast_run  # type: ignore[method-assign]
    with pytest.raises(BaseExceptionGroup) as excinfo:
        await scanner.run()
    matched, _ = excinfo.value.split(RuntimeError)
    assert matched is not None
    # Restart cap is 3, plus the initial attempt → at least 4 calls.
    assert call_count["n"] >= 4


@pytest.mark.asyncio
async def test_run_invokes_shutdown_on_taskgroup_failure(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT-equivalent: any unrecoverable exit from ``run`` must call ``aclose``."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    config = _make_config(enable_smart=False, enable_misprice=True, enable_whales=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)

    async def crash_run(_sink: AlertSink) -> None:
        raise RuntimeError("boom")

    scanner._detectors["mispricing"].run = crash_run  # type: ignore[method-assign]
    with pytest.raises(BaseExceptionGroup):
        await scanner.run()
    assert scanner._closed is True


@pytest.mark.asyncio
async def test_shutdown_is_idempotent(db_path: Path) -> None:
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    await scanner.aclose()
    await scanner.aclose()
    assert scanner._closed is True


@pytest.mark.asyncio
async def test_shutdown_closes_owned_clients(db_path: Path) -> None:
    config = _make_config()
    scanner = Scanner(config=config, db_path=db_path)
    # Replace owned clients with mocks to verify aclose calls without networking.
    mocked = _make_clients()
    scanner._clients = mocked
    await scanner.aclose()
    cast("MagicMock", mocked.gamma_http).aclose.assert_awaited()
    cast("MagicMock", mocked.data_http).aclose.assert_awaited()


@pytest.mark.asyncio
async def test_run_once_with_disabled_detectors_does_no_work(db_path: Path) -> None:
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_events=False,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["events_scanned"] == 0
    cast("MagicMock", clients.gamma_client).iter_events.assert_not_called()
    cast("MagicMock", clients.gamma_client).list_markets.assert_not_called()


@pytest.mark.asyncio
async def test_run_with_supervisor_cancellation(db_path: Path) -> None:
    config = _make_config(enable_smart=False, enable_misprice=True, enable_whales=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)

    async def long_run(_sink: AlertSink) -> None:
        await asyncio.sleep(60)

    scanner._detectors["mispricing"].run = long_run  # type: ignore[method-assign]
    task = asyncio.create_task(scanner.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scanner._closed is True


@pytest.mark.asyncio
async def test_scanner_wires_collectors_and_repos(db_path: Path) -> None:
    """Construction wires watchlist + wallet-trades repos and both collectors."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "watchlist_sync" in scanner._collectors
        assert "trade_collector" in scanner._collectors
        assert isinstance(scanner._collectors["watchlist_sync"], WatchlistSyncer)
        assert isinstance(scanner._collectors["trade_collector"], TradeCollector)
        assert scanner._watchlist_repo is not None
        assert scanner._wallet_trades_repo is not None
        assert scanner._watchlist_registry is not None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_run_once_reports_collector_metrics(db_path: Path) -> None:
    """``run_once`` includes ``watched_wallets`` and ``trades_recorded`` keys."""
    config = _make_config(enable_smart=False, enable_misprice=False, enable_whales=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["watched_wallets"] == 0
    assert result["trades_recorded"] == 0


@pytest.mark.asyncio
async def test_run_once_drives_collectors_with_active_watchlist(db_path: Path) -> None:
    """Pre-seeded watchlist drives ``poll_all_wallets`` to fetch /activity."""
    config = _make_config(enable_smart=False, enable_misprice=False, enable_whales=False)
    activity = [
        {
            "type": "TRADE",
            "transactionHash": "0xtxa",
            "asset": "asset-1",
            "side": "BUY",
            "size": 100.0,
            "price": 0.4,
            "conditionId": "cond-1",
            "timestamp": 1_700_000_000,
            "usdcSize": 40.0,
        },
    ]
    clients = _make_clients()
    cast("AsyncMock", clients.data_client.get_activity).return_value = activity
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    scanner._watchlist_repo.upsert(address="0xabc", source="manual", reason="test")
    scanner._watchlist_registry.reload()
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["watched_wallets"] == 1
    assert result["trades_recorded"] == 1
    cast("AsyncMock", clients.data_client.get_activity).assert_any_await(
        "0xabc",
        type="TRADE",
        limit=200,
    )


@pytest.mark.asyncio
async def test_run_once_mirrors_tracked_wallets_into_watchlist(db_path: Path) -> None:
    """``run_once`` syncs ``tracked_wallets`` rows into the registry as smart-money."""
    config = _make_config(enable_smart=False, enable_misprice=False, enable_whales=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    scanner._tracked_repo.upsert(
        address="0xleader",
        closed_position_count=30,
        closed_position_wins=22,
        winrate=0.73,
        leaderboard_pnl=10000.0,
    )
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["tracked_wallets"] == 1
    assert "0xleader" in scanner._watchlist_registry.addresses()


@pytest.mark.asyncio
async def test_aclose_sets_collectors_stop_event(db_path: Path) -> None:
    """``aclose`` sets ``_collectors_stop`` so collectors can drain cleanly."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    assert not scanner._collectors_stop.is_set()
    await scanner.aclose()
    assert scanner._collectors_stop.is_set()


@pytest.mark.asyncio
async def test_scanner_wires_whales_to_trade_collector_callback(db_path: Path) -> None:
    """DC-1.5: whales detector's sink + trade-collector callback wired at init."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        whales = scanner._detectors["whales"]
        trades = scanner._collectors["trade_collector"]
        assert isinstance(trades, TradeCollector)
        assert whales._sink is scanner.sink
        assert whales.handle_trade_sync in trades._new_trade_callbacks
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_whales_callback_when_disabled(db_path: Path) -> None:
    """When whales+convergence are disabled, the trade collector has no callbacks."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        trades = scanner._collectors["trade_collector"]
        assert isinstance(trades, TradeCollector)
        assert trades._new_trade_callbacks == []
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_wires_convergence_to_trade_collector_callback(db_path: Path) -> None:
    """DC-1.8.B: convergence detector's sink + trade-collector callback wired at init."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        convergence = scanner._detectors["convergence"]
        trades = scanner._collectors["trade_collector"]
        assert isinstance(convergence, ConvergenceDetector)
        assert isinstance(trades, TradeCollector)
        assert convergence._sink is scanner.sink
        assert convergence.handle_trade_sync in trades._new_trade_callbacks
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_convergence_when_disabled(db_path: Path) -> None:
    """When convergence is disabled, no detector entry is created."""
    config = _make_config(enable_convergence=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "convergence" not in scanner._detectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_constructs_dc2_collectors_when_enabled(db_path: Path) -> None:
    """DC-2 Wave 1: position + activity collectors live in ``_collectors``."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_positions=True,
        enable_activity=True,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "position_collector" in scanner._collectors
        assert "activity_collector" in scanner._collectors
        assert isinstance(scanner._collectors["position_collector"], PositionCollector)
        assert isinstance(scanner._collectors["activity_collector"], ActivityCollector)
        assert scanner._positions_repo is not None
        assert scanner._activity_repo is not None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_dc2_collectors_when_disabled(db_path: Path) -> None:
    """When ``positions``/``activity`` are disabled, neither collector is wired."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_positions=False,
        enable_activity=False,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "position_collector" not in scanner._collectors
        assert "activity_collector" not in scanner._collectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_run_once_dc2_stub_metrics_remain_zero(db_path: Path) -> None:
    """Wave 1 stubs raise; ``_run_once_collectors`` swallows so metrics stay 0."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_positions=True,
        enable_activity=True,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["position_snapshots"] == 0
    assert result["activity_events"] == 0


@pytest.mark.asyncio
async def test_scanner_constructs_dc3_collectors_when_enabled(db_path: Path) -> None:
    """DC-3 Wave 1: market + event collectors live in ``_collectors``."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=True,
        enable_events=True,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "market_collector" in scanner._collectors
        assert "event_collector" in scanner._collectors
        assert isinstance(scanner._collectors["market_collector"], MarketCollector)
        assert isinstance(scanner._collectors["event_collector"], EventCollector)
        assert scanner._market_snapshots_repo is not None
        assert scanner._event_snapshots_repo is not None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_dc3_collectors_when_disabled(db_path: Path) -> None:
    """When ``markets``/``events`` are disabled, neither collector is wired."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=False,
        enable_events=False,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "market_collector" not in scanner._collectors
        assert "event_collector" not in scanner._collectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_run_once_dc3_stub_metrics_remain_zero(db_path: Path) -> None:
    """DC-3 Wave 1 stubs raise; ``run_once`` swallows so metrics stay 0."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=True,
        enable_events=True,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["market_snapshots"] == 0
    assert result["event_snapshots"] == 0
