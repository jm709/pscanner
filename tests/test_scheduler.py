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
from pscanner.collectors.trades import TradeCollector
from pscanner.collectors.watchlist import WatchlistSyncer
from pscanner.config import (
    Config,
    MispricingConfig,
    RatelimitConfig,
    ScannerConfig,
    SmartMoneyConfig,
    WhalesConfig,
)
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
) -> Config:
    return Config(
        scanner=ScannerConfig(),
        smart_money=SmartMoneyConfig(enabled=enable_smart),
        mispricing=MispricingConfig(enabled=enable_misprice),
        whales=WhalesConfig(enabled=enable_whales),
        ratelimit=RatelimitConfig(),
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
    data_client.aclose = AsyncMock()

    market_ws = MagicMock()
    market_ws.connect = AsyncMock()
    market_ws.subscribe = AsyncMock()
    market_ws.close = AsyncMock()

    trade_ws = MagicMock()
    trade_ws.connect = AsyncMock()
    trade_ws.subscribe = AsyncMock()
    trade_ws.close = AsyncMock()

    return SchedulerClients(
        gamma_http=gamma_http,
        data_http=data_http,
        gamma_client=gamma_client,
        data_client=data_client,
        market_ws=market_ws,
        trade_ws=trade_ws,
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
    cast("MagicMock", clients.market_ws).connect.assert_not_called()


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
    cast("MagicMock", clients.market_ws).close.assert_awaited()


@pytest.mark.asyncio
async def test_shutdown_closes_owned_clients(db_path: Path) -> None:
    config = _make_config()
    scanner = Scanner(config=config, db_path=db_path)
    # Replace owned clients with mocks to verify aclose calls without networking.
    mocked = _make_clients()
    scanner._clients = mocked
    await scanner.aclose()
    cast("MagicMock", mocked.market_ws).close.assert_awaited()
    cast("MagicMock", mocked.gamma_http).aclose.assert_awaited()
    cast("MagicMock", mocked.data_http).aclose.assert_awaited()


@pytest.mark.asyncio
async def test_run_once_with_disabled_detectors_does_no_work(db_path: Path) -> None:
    config = _make_config(enable_smart=False, enable_misprice=False, enable_whales=False)
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
    """Pre-seeded watchlist drives ``refresh_subscriptions`` to fetch positions."""
    config = _make_config(enable_smart=False, enable_misprice=False, enable_whales=False)
    positions = [
        Position.model_validate(
            {
                "proxyWallet": "0xabc",
                "asset": "asset-1",
                "conditionId": "cond-1",
                "size": 100.0,
                "avgPrice": 0.4,
                "currentValue": 50.0,
                "outcome": "Yes",
                "outcomeIndex": 0,
                "title": "test market",
            }
        ),
    ]
    clients = _make_clients(positions=positions)
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    scanner._watchlist_repo.upsert(address="0xabc", source="manual", reason="test")
    scanner._watchlist_registry.reload()
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["watched_wallets"] == 1
    cast("AsyncMock", clients.data_client.get_positions).assert_awaited_with("0xabc")
    cast("AsyncMock", clients.trade_ws.subscribe).assert_awaited()


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
async def test_aclose_closes_trade_ws(db_path: Path) -> None:
    """``aclose`` must close both ``market_ws`` and ``trade_ws``."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    await scanner.aclose()
    cast("MagicMock", clients.market_ws).close.assert_awaited()
    cast("MagicMock", clients.trade_ws).close.assert_awaited()


@pytest.mark.asyncio
async def test_aclose_sets_collectors_stop_event(db_path: Path) -> None:
    """``aclose`` sets ``_collectors_stop`` so collectors can drain cleanly."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    assert not scanner._collectors_stop.is_set()
    await scanner.aclose()
    assert scanner._collectors_stop.is_set()
