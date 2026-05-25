"""Tests for ``pscanner.scheduler.Scanner``.

Uses dependency injection through the ``clients`` parameter to substitute
mocked Polymarket clients. Each test exercises the public Scanner surface
without touching real network/IO.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import pscanner.scheduler as scheduler_mod
from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.collectors.events import EventCollector
from pscanner.collectors.markets import MarketCollector
from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
from pscanner.collectors.watchlist import WatchlistSyncer
from pscanner.config import (
    Config,
    EvaluatorsConfig,
    EventsConfig,
    MarketsConfig,
    PaperTradingConfig,
    RatelimitConfig,
    ScannerConfig,
    SubgraphCopyEvaluatorConfig,
    SubgraphTradeCollectorConfig,
)
from pscanner.corpus.db import init_corpus_db
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.poly.models import Event, Market
from pscanner.scheduler import Scanner, SchedulerClients
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    PaperTradesRepo,
    TrackedWalletsRepo,
)
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator
from pscanner.strategies.paper_resolver import PaperResolver
from pscanner.strategies.paper_trader import PaperTrader
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


def _make_config(
    *,
    enable_markets: bool = True,
    enable_events: bool = True,
    enable_paper_trading: bool = False,
) -> Config:
    return Config(
        scanner=ScannerConfig(),
        ratelimit=RatelimitConfig(),
        markets=MarketsConfig(enabled=enable_markets),
        events=EventsConfig(enabled=enable_events),
        paper_trading=PaperTradingConfig(enabled=enable_paper_trading),
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
    data_client.get_leaderboard = AsyncMock(return_value=[])
    data_client.get_positions = AsyncMock(return_value=[])
    data_client.get_activity = AsyncMock(return_value=[])
    data_client.get_market_trades = AsyncMock(return_value=[])
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
async def test_scanner_constructs_with_defaults(db_path: Path) -> None:
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
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result == {
        "alerts_emitted": 0,
        "tracked_wallets": 0,
        "markets_cached": 0,
        "watched_wallets": 0,
        "market_snapshots": 0,
        "event_snapshots": 0,
    }


@pytest.mark.asyncio
async def test_run_supervisor_restarts_returning_detector(
    db_path: Path,
    fake_clock: FakeClock,
) -> None:
    """Supervisor retries a fast-returning detector up to the restart cap.

    Uses paper-trading (the only detector kind that survives this PR) plus
    the shared ``FakeClock`` so sibling collector loops stay parked on
    their first sleep — proving the clock injection eliminates the test
    deadlocks issue #23 was filed for.
    """
    config = _make_config(enable_paper_trading=True)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients, clock=fake_clock)
    detector = scanner._detectors["paper_trader"]
    call_count = {"n": 0}

    async def fast_run(_sink: AlertSink) -> None:
        call_count["n"] += 1

    detector.run = fast_run  # type: ignore[method-assign]

    async def _drive_clock() -> None:
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
    """Any unrecoverable exit from ``run`` must call ``aclose``."""
    config = _make_config(enable_paper_trading=True)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients, clock=fake_clock)

    async def crash_run(_sink: AlertSink) -> None:
        raise RuntimeError("boom")

    scanner._detectors["paper_trader"].run = crash_run  # type: ignore[method-assign]

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
async def test_run_with_supervisor_cancellation(db_path: Path) -> None:
    config = _make_config(enable_paper_trading=True)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)

    async def long_run(_sink: AlertSink) -> None:
        await asyncio.sleep(60)

    scanner._detectors["paper_trader"].run = long_run  # type: ignore[method-assign]
    task = asyncio.create_task(scanner.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scanner._closed is True


@pytest.mark.asyncio
async def test_scanner_wires_watchlist_registry_and_syncer(db_path: Path) -> None:
    """Construction wires the watchlist registry + syncer collector."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "watchlist_sync" in scanner._collectors
        assert isinstance(scanner._collectors["watchlist_sync"], WatchlistSyncer)
        assert scanner._watchlist_repo is not None
        assert scanner._watchlist_registry is not None
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_run_once_reports_watched_wallets_key(db_path: Path) -> None:
    """``run_once`` reports ``watched_wallets`` count from the registry."""
    config = _make_config()
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        result = await scanner.run_once()
    finally:
        await scanner.aclose()
    assert result["watched_wallets"] == 0


@pytest.mark.asyncio
async def test_run_once_mirrors_tracked_wallets_into_watchlist(db_path: Path) -> None:
    """``run_once`` syncs ``tracked_wallets`` rows into the registry as smart-money."""
    config = _make_config()
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
async def test_scanner_constructs_dc3_collectors_when_enabled(db_path: Path) -> None:
    """DC-3 Wave 1: market + event collectors live in ``_collectors``."""
    config = _make_config(enable_markets=True, enable_events=True)
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
    config = _make_config(enable_markets=False, enable_events=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "market_collector" not in scanner._collectors
        assert "event_collector" not in scanner._collectors
    finally:
        await scanner.aclose()


def _seed_paper_smoke_db(db_file: Path) -> None:
    """Pre-seed a tracked wallet, cached market, and one tick row."""
    seed_conn = init_db(db_file)
    try:
        TrackedWalletsRepo(seed_conn).upsert(
            address="0xwallet1",
            closed_position_count=50,
            closed_position_wins=42,
            winrate=0.84,
            leaderboard_pnl=1000.0,
            mean_edge=0.4,
            weighted_edge=0.4,
            excess_pnl_usd=1000.0,
            total_stake_usd=1000.0,
        )
        MarketCacheRepo(seed_conn).upsert(
            CachedMarket(
                market_id=MarketId("mkt-1"),
                event_id=None,
                title="t",
                liquidity_usd=1.0,
                volume_usd=1.0,
                outcome_prices=[0.6, 0.4],
                outcomes=["Yes", "No"],
                asset_ids=[AssetId("asset-yes"), AssetId("asset-no")],
                active=True,
                cached_at=1700000000,
                condition_id=ConditionId("0xcond-1"),
                event_slug=None,
            ),
        )
        seed_conn.execute(
            """
            INSERT INTO market_ticks (asset_id, condition_id, snapshot_at,
              mid_price, best_bid, best_ask, spread, bid_depth_top5,
              ask_depth_top5, last_trade_price)
            VALUES ('asset-yes', '0xcond-1', 1700000000, NULL, NULL, 0.5,
              NULL, NULL, NULL, NULL)
            """,
        )
        seed_conn.commit()
    finally:
        seed_conn.close()


def _resolve_paper_trades(db_file: Path) -> None:
    """Mark the seeded market as resolved with the YES leg winning."""
    verify_conn = sqlite3.connect(db_file)
    verify_conn.row_factory = sqlite3.Row
    try:
        verify_conn.execute(
            "UPDATE market_cache SET active = 0, "
            "outcome_prices_json = '[1.0, 0.0]' WHERE condition_id = '0xcond-1'",
        )
        verify_conn.commit()
    finally:
        verify_conn.close()


@pytest.mark.asyncio
async def test_paper_resolver_books_winning_position(tmp_path: Path) -> None:
    """End-to-end: seed paper trade, mark market resolved, resolver books PnL.

    Inserts a paper_trade entry row directly (the per-detector evaluator
    chain that used to spawn those rows is gone in this PR), then drives
    one resolver scan and verifies the exit row + NAV.
    """
    db_file = tmp_path / "pscanner.sqlite3"
    base_cfg = _make_config()
    cfg = base_cfg.model_copy(
        update={
            "paper_trading": base_cfg.paper_trading.model_copy(update={"enabled": True}),
        },
    )
    _seed_paper_smoke_db(db_file)

    clients = _make_clients()
    # Insert a fake entry row so the resolver has something to resolve.
    entry_conn = init_db(db_file)
    try:
        PaperTradesRepo(entry_conn).insert_entry(
            triggering_alert_key="alert-1",
            triggering_alert_detector="subgraph_copy",
            rule_variant=None,
            source_wallet="0xwallet1",
            condition_id=ConditionId("0xcond-1"),
            asset_id=AssetId("asset-yes"),
            outcome="Yes",
            shares=20.0,
            fill_price=0.5,
            cost_usd=10.0,
            nav_after_usd=1000.0,
            ts=1_700_000_000,
        )
    finally:
        entry_conn.close()

    _resolve_paper_trades(db_file)

    resolver_conn = init_db(db_file)
    try:
        resolver = PaperResolver(
            config=cfg.paper_trading,
            market_cache=MarketCacheRepo(resolver_conn),
            paper_trades=PaperTradesRepo(resolver_conn),
            data_client=clients.data_client,
            gamma_client=clients.gamma_client,
        )
        await resolver._scan(AlertSink(AlertsRepo(resolver_conn)))
        assert PaperTradesRepo(resolver_conn).list_open_positions() == []
        nav = PaperTradesRepo(resolver_conn).compute_cost_basis_nav(
            starting_bankroll=cfg.paper_trading.starting_bankroll_usd,
        )
        assert nav == 1010.0  # 1000 + (20 shares * $1.0 - $10 cost)
    finally:
        resolver_conn.close()


@pytest.mark.asyncio
async def test_replay_paper_trader_calls_replay_when_present(db_path: Path) -> None:
    """``_replay_paper_trader`` invokes ``replay_unbooked`` on the registered trader."""
    config = _make_config(enable_paper_trading=True)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        trader = scanner._detectors["paper_trader"]
        assert isinstance(trader, PaperTrader)
        trader.replay_unbooked = AsyncMock(return_value=3)  # type: ignore[method-assign]
        await scanner._replay_paper_trader()
        trader.replay_unbooked.assert_awaited_once_with()
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_replay_paper_trader_noop_when_disabled(db_path: Path) -> None:
    """No-op when paper_trading is disabled (no PaperTrader in detectors)."""
    config = _make_config(enable_paper_trading=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "paper_trader" not in scanner._detectors
        await scanner._replay_paper_trader()  # must not raise
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_subgraph_trades_wired_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPH_API_KEY", "test-key")
    # Redirect the corpus DB to a tmp path so we don't touch the real one.
    corpus_path = tmp_path / "corpus.sqlite3"
    monkeypatch.setattr(
        scheduler_mod,
        "init_corpus_db",
        lambda _p: init_corpus_db(corpus_path),
    )
    config = Config(
        subgraph_trades=SubgraphTradeCollectorConfig(enabled=True),
        paper_trading=PaperTradingConfig(
            enabled=True,
            evaluators=EvaluatorsConfig(
                subgraph_copy=SubgraphCopyEvaluatorConfig(enabled=True),
            ),
        ),
    )
    scanner = Scanner(config=config, db_path=tmp_path / "p.sqlite3")
    try:
        assert isinstance(scanner._collectors.get("subgraph_trades"), SubgraphTradeCollector)
        pt = scanner._detectors["paper_trader"]
        assert any(isinstance(e, SubgraphCopyEvaluator) for e in pt._evaluators)
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_subgraph_trades_preflight_requires_graph_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    corpus_path = tmp_path / "corpus.sqlite3"
    monkeypatch.setattr(
        scheduler_mod,
        "init_corpus_db",
        lambda _p: init_corpus_db(corpus_path),
    )
    config = Config(subgraph_trades=SubgraphTradeCollectorConfig(enabled=True))
    scanner = Scanner(config=config, db_path=tmp_path / "p.sqlite3")
    try:
        with pytest.raises(RuntimeError, match="GRAPH_API_KEY"):
            scanner.preflight()
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_subgraph_trades_aclose_closes_subgraph_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Scanner.aclose() must close the SubgraphClient (#152 review)."""
    monkeypatch.setenv("GRAPH_API_KEY", "test-key")
    corpus_path = tmp_path / "corpus.sqlite3"
    monkeypatch.setattr(
        scheduler_mod,
        "init_corpus_db",
        lambda _p: init_corpus_db(corpus_path),
    )
    config = Config(subgraph_trades=SubgraphTradeCollectorConfig(enabled=True))
    scanner = Scanner(config=config, db_path=tmp_path / "p.sqlite3")
    assert scanner._subgraph_client is not None
    sub_client = scanner._subgraph_client
    await scanner.aclose()
    # SubgraphClient delegates closure to its shared RateLimitedHttpClient inner.
    assert sub_client._inner._closed is True


@pytest.mark.asyncio
async def test_alert_emission_through_sink(tmp_path: Path) -> None:
    """End-to-end: a subgraph_copy alert through the live AlertSink lands in
    ``alerts`` and reaches PaperTrader's subscription callback."""
    db_file = tmp_path / "pscanner.sqlite3"
    cfg = _make_config(enable_paper_trading=True)
    _seed_paper_smoke_db(db_file)

    scanner = Scanner(config=cfg, db_path=db_file, clients=_make_clients())
    try:
        await scanner.sink.emit(
            Alert(
                detector="subgraph_copy",
                alert_key="subgraph:0xwallet1:0xcond-1:Yes:smoke",
                severity="med",
                title="copy",
                body={
                    "wallet": "0xwallet1",
                    "condition_id": "0xcond-1",
                    "side": "Yes",
                },
                created_at=1700000000,
            ),
        )
        for _ in range(10):
            await asyncio.sleep(0)
        await scanner._detectors["paper_trader"].aclose()
        recent = scanner._alerts_repo.recent(limit=10)
        assert any(a.detector == "subgraph_copy" for a in recent)
    finally:
        await scanner.aclose()


# Keep the unused-symbol import bindings used somewhere meaningful.
_ = Any
