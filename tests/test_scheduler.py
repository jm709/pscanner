"""Tests for ``pscanner.scheduler.Scanner``.

Uses dependency injection through the ``clients`` parameter to substitute
mocked Polymarket clients. Each test exercises the public Scanner surface
without touching real network/IO.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.collectors.activity import ActivityCollector
from pscanner.collectors.events import EventCollector
from pscanner.collectors.markets import MarketCollector
from pscanner.collectors.positions import PositionCollector
from pscanner.collectors.ticks import MarketTickCollector
from pscanner.collectors.trades import TradeCollector
from pscanner.collectors.watchlist import WatchlistSyncer
from pscanner.config import (
    ActivityConfig,
    ClusterConfig,
    Config,
    ConvergenceConfig,
    EventsConfig,
    MarketsConfig,
    MispricingConfig,
    MoveAttributionConfig,
    PositionsConfig,
    RatelimitConfig,
    ScannerConfig,
    SmartMoneyConfig,
    TicksConfig,
    VelocityConfig,
    WhalesConfig,
)
from pscanner.detectors.convergence import ConvergenceDetector
from pscanner.detectors.move_attribution import MoveAttributionDetector
from pscanner.detectors.velocity import PriceVelocityDetector
from pscanner.poly.models import Event, LeaderboardEntry, Market, Position
from pscanner.scheduler import Scanner, SchedulerClients
from pscanner.util.clock import FakeClock


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
    enable_ticks: bool = True,
    enable_velocity: bool = True,
    enable_cluster: bool = True,
    enable_move_attribution: bool = True,
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
        ticks=TicksConfig(enabled=enable_ticks),
        velocity=VelocityConfig(enabled=enable_velocity),
        cluster=ClusterConfig(enabled=enable_cluster),
        move_attribution=MoveAttributionConfig(enabled=enable_move_attribution),
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
    data_client.get_market_trades = AsyncMock(return_value=[])
    data_client.aclose = AsyncMock()

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
        "tick_snapshots": 0,
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
    fake_clock: FakeClock,
) -> None:
    """Supervisor retries a fast-returning detector up to the restart cap.

    With every detector and collector enabled, the shared ``FakeClock``
    keeps every other ``while True: await self._clock.sleep(...)`` loop
    parked on its first sleep — proving the new clock injection
    eliminates the test-deadlock class of bug #23 was filed for.
    """
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients, clock=fake_clock)
    detector = scanner._detectors["mispricing"]
    call_count = {"n": 0}

    async def fast_run(_sink: AlertSink) -> None:
        call_count["n"] += 1

    detector.run = fast_run  # type: ignore[method-assign]

    async def _drive_clock() -> None:
        # Each restart waits up to ~30s of backoff. Advance generously
        # several times so the supervisor blasts through the restart cap.
        for _ in range(10):
            await fake_clock.advance(60.0)

    async def _run_scanner() -> None:
        with pytest.raises(BaseExceptionGroup) as excinfo:
            await scanner.run()
        matched, _ = excinfo.value.split(RuntimeError)
        assert matched is not None

    await asyncio.gather(_run_scanner(), _drive_clock())
    # Restart cap is 3, plus the initial attempt → at least 4 calls.
    assert call_count["n"] >= 4


@pytest.mark.asyncio
async def test_run_invokes_shutdown_on_taskgroup_failure(
    db_path: Path,
    fake_clock: FakeClock,
) -> None:
    """Any unrecoverable exit from ``run`` must call ``aclose``.

    Every detector and collector stays enabled — the shared ``FakeClock``
    keeps siblings parked while the crashing detector exhausts its
    restart cap, eliminating the deadlock that previously forced this
    test to disable everything else.
    """
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients, clock=fake_clock)

    async def crash_run(_sink: AlertSink) -> None:
        raise RuntimeError("boom")

    scanner._detectors["mispricing"].run = crash_run  # type: ignore[method-assign]

    async def _drive_clock() -> None:
        for _ in range(10):
            await fake_clock.advance(60.0)

    async def _run_scanner() -> None:
        with pytest.raises(BaseExceptionGroup):
            await scanner.run()

    await asyncio.gather(_run_scanner(), _drive_clock())
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
    """When whales+convergence+cluster are disabled, no trade-collector callbacks."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
        enable_cluster=False,
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


@pytest.mark.asyncio
async def test_scanner_constructs_dc4_collector_and_detector_when_enabled(
    db_path: Path,
) -> None:
    """DC-4 Wave 1: tick collector + velocity detector live in their dicts."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=False,
        enable_events=False,
        enable_ticks=True,
        enable_velocity=True,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "tick_collector" in scanner._collectors
        assert isinstance(scanner._collectors["tick_collector"], MarketTickCollector)
        assert "velocity" in scanner._detectors
        assert isinstance(scanner._detectors["velocity"], PriceVelocityDetector)
        assert scanner._ticks_repo is not None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_dc4_collector_and_detector_when_disabled(
    db_path: Path,
) -> None:
    """When ``ticks``/``velocity`` are disabled, neither is wired."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=False,
        enable_events=False,
        enable_ticks=False,
        enable_velocity=False,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "tick_collector" not in scanner._collectors
        assert "velocity" not in scanner._detectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_velocity_when_ticks_disabled(db_path: Path) -> None:
    """Velocity depends on tick_collector; with ticks off it must not be built."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=False,
        enable_events=False,
        enable_ticks=False,
        enable_velocity=True,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "tick_collector" not in scanner._collectors
        assert "velocity" not in scanner._detectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_run_once_includes_tick_snapshots_key(db_path: Path) -> None:
    """``run_once`` returns the new ``tick_snapshots`` count (0 while stub raises)."""
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=False,
        enable_events=False,
        enable_ticks=True,
        enable_velocity=False,
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert "tick_snapshots" in result
    assert result["tick_snapshots"] == 0


@pytest.mark.asyncio
async def test_scanner_wires_move_attribution_to_alert_sink(db_path: Path) -> None:
    """T8: move-attribution detector's sink + AlertSink subscription wired at init."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        detector = scanner._detectors["move_attribution"]
        assert isinstance(detector, MoveAttributionDetector)
        assert detector._sink is scanner.sink
        assert detector.handle_alert_sync in scanner.sink._subscribers
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_move_attribution_when_disabled(db_path: Path) -> None:
    """When ``move_attribution`` is disabled, no detector entry is created."""
    config = _make_config(enable_move_attribution=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "move_attribution" not in scanner._detectors
        # And no MoveAttributionDetector callback is wired to the sink.
        assert all(
            not isinstance(getattr(cb, "__self__", None), MoveAttributionDetector)
            for cb in scanner.sink._subscribers
        )
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_move_attribution_smoke_emits_cluster_candidate(
    db_path: Path,
) -> None:
    """End-to-end: a velocity alert through the live AlertSink reaches the
    MoveAttributionDetector, which emits a ``cluster.candidate`` alert and
    upserts contributors into ``wallet_watchlist``."""
    alert_ts = 1_700_086_400
    burst_trades = [
        {
            "proxyWallet": f"0x{i:04d}",
            "timestamp": alert_ts - 30,
            "side": "BUY",
            "outcome": "Yes",
            "size": 500.0 + i,
            "price": 0.95,
        }
        for i in range(6)
    ]
    config = _make_config(
        enable_smart=False,
        enable_misprice=False,
        enable_whales=False,
        enable_convergence=False,
        enable_cluster=False,
        enable_positions=False,
        enable_activity=False,
        enable_markets=False,
        enable_events=False,
        enable_ticks=False,
        enable_velocity=False,
        enable_move_attribution=True,
    )
    clients = _make_clients()
    cast("AsyncMock", clients.data_client.get_market_trades).return_value = burst_trades
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    detector = scanner._detectors["move_attribution"]
    assert isinstance(detector, MoveAttributionDetector)
    try:
        await scanner.sink.emit(
            Alert(
                detector="velocity",
                alert_key="velocity:0xabc:1",
                severity="med",
                title="market moved",
                body={"condition_id": "0xabc"},
                created_at=alert_ts,
            )
        )
        await detector.aclose()
        cast("AsyncMock", clients.data_client.get_market_trades).assert_awaited_once()
        recent = scanner._alerts_repo.recent(limit=10)
        candidates = [a for a in recent if a.detector == "move_attribution"]
        assert len(candidates) >= 1
        watchlist = scanner._watchlist_repo.list_active()
        assert any(w.source == "cluster.candidate" for w in watchlist)
    finally:
        await scanner.aclose()
