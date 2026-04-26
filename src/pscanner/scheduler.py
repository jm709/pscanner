"""Composition root and asyncio orchestrator for the pscanner daemon.

The :class:`Scanner` wires every Polymarket client, repo, detector, and the
alert sink + terminal renderer into a single object. Two run modes are
supported:

* :meth:`Scanner.run` — the long-running daemon. Drives the renderer plus each
  enabled detector inside an :class:`asyncio.TaskGroup`. Detector failures are
  logged and the detector is restarted up to ``_MAX_RESTARTS`` times within
  ``_RESTART_WINDOW_SECONDS`` before the daemon bails out.
* :meth:`Scanner.run_once` — a single-shot snapshot, used for ``pscanner run
  --once``. Refreshes catalog state for each detector but does not open the
  websocket and does not block on long polls.

The constructor can either build its own clients/repos or accept an injected
:class:`SchedulerClients` bundle, which is what the tests rely on.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.alerts.terminal import TerminalRenderer
from pscanner.collectors.activity import ActivityCollector
from pscanner.collectors.base import Collector
from pscanner.collectors.events import EventCollector
from pscanner.collectors.markets import MarketCollector
from pscanner.collectors.positions import PositionCollector
from pscanner.collectors.ticks import MarketTickCollector
from pscanner.collectors.trades import TradeCollector
from pscanner.collectors.watchlist import WatchlistRegistry, WatchlistSyncer
from pscanner.config import Config
from pscanner.detectors.convergence import ConvergenceDetector
from pscanner.detectors.mispricing import MispricingDetector
from pscanner.detectors.smart_money import SmartMoneyDetector
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.detectors.velocity import PriceVelocityDetector
from pscanner.detectors.whales import WhalesDetector
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.http import PolyHttpClient
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    EventOutcomeSumRepo,
    EventSnapshotsRepo,
    EventTagCacheRepo,
    MarketCacheRepo,
    MarketSnapshotsRepo,
    MarketTicksRepo,
    PositionSnapshotsRepo,
    TrackedWalletCategoriesRepo,
    TrackedWalletsRepo,
    WalletActivityEventsRepo,
    WalletFirstSeenRepo,
    WalletPositionsHistoryRepo,
    WalletTradesRepo,
    WatchlistRepo,
)
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)

_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
_DATA_BASE_URL = "https://data-api.polymarket.com"

_MAX_RESTARTS = 3
_RESTART_WINDOW_SECONDS = 300.0
_RUN_ONCE_MARKETS_PAGE_SIZE = 200
_RUN_ONCE_EVENTS_PAGE_SIZE = 100
_RUN_ONCE_MAX_EVENT_PAGES = 5
_RUN_ONCE_LEADERBOARD_LIMIT = 25


@dataclass(frozen=True, slots=True)
class SchedulerClients:
    """Bundle of HTTP/WebSocket clients consumed by the scanner.

    Exposed for dependency injection in tests; production code lets
    :class:`Scanner` build the bundle itself.
    """

    gamma_http: PolyHttpClient
    data_http: PolyHttpClient
    gamma_client: GammaClient
    data_client: DataClient
    ticks_ws: MarketWebSocket


class Scanner:
    """Async composition root that drives every detector against shared state."""

    def __init__(
        self,
        *,
        config: Config,
        db_path: Path | None = None,
        clients: SchedulerClients | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Construct the daemon, opening the DB and wiring detectors.

        Args:
            config: Validated pscanner configuration.
            db_path: Override for ``config.scanner.db_path`` (testing).
            clients: Optional injected client bundle. When ``None`` the
                scanner constructs its own ``PolyHttpClient`` / WS instances.
            clock: Injectable :class:`Clock` shared with every detector and
                the supervisor backoff. Defaults to :class:`RealClock`.
        """
        self._config = config
        self._clock: Clock = clock if clock is not None else RealClock()
        resolved_db = db_path if db_path is not None else config.scanner.db_path
        self._db = init_db(resolved_db)
        self._tracked_repo = TrackedWalletsRepo(self._db)
        self._snapshots_repo = PositionSnapshotsRepo(self._db)
        self._first_seen_repo = WalletFirstSeenRepo(self._db)
        self._market_cache_repo = MarketCacheRepo(self._db)
        self._alerts_repo = AlertsRepo(self._db)
        self._positions_repo = WalletPositionsHistoryRepo(self._db)
        self._activity_repo = WalletActivityEventsRepo(self._db)
        self._market_snapshots_repo = MarketSnapshotsRepo(self._db)
        self._event_snapshots_repo = EventSnapshotsRepo(self._db)
        self._sum_history_repo = EventOutcomeSumRepo(self._db)
        self._watchlist_repo = WatchlistRepo(self._db)
        self._wallet_trades_repo = WalletTradesRepo(self._db)
        self._categories_repo = TrackedWalletCategoriesRepo(self._db)
        self._event_tag_cache_repo = EventTagCacheRepo(self._db)
        self._ticks_repo = MarketTicksRepo(self._db)
        self._owns_clients = clients is None
        self._clients = clients or self._build_default_clients()
        self._renderer = TerminalRenderer()
        self._sink = AlertSink(self._alerts_repo, renderer=self._renderer)
        self._watchlist_registry = WatchlistRegistry(self._watchlist_repo)
        self._collectors = self._build_collectors()
        self._detectors = self._build_detectors()
        self._wire_trade_callbacks()
        self._collectors_stop = asyncio.Event()
        self._closed = False

    def _wire_trade_callbacks(self) -> None:
        """Wire trade-collector callbacks for every ``TradeDrivenDetector``.

        Each trade-driven detector evaluates per-trade and needs its
        ``_sink`` populated before the run-loop starts so callbacks fire
        correctly during the first poll cycle.
        """
        trade_collector = self._collectors.get("trade_collector")
        if not isinstance(trade_collector, TradeCollector):
            return
        for detector in self._detectors.values():
            if isinstance(detector, TradeDrivenDetector):
                detector._sink = self._sink
                trade_collector.subscribe_new_trade(detector.handle_trade_sync)
                _LOG.info("scanner.trade_driven_detector_wired", detector=detector.name)

    def _build_default_clients(self) -> SchedulerClients:
        """Construct the production HTTP/WS clients from config."""
        gamma_http = PolyHttpClient(
            base_url=_GAMMA_BASE_URL,
            rpm=self._config.ratelimit.gamma_rpm,
        )
        data_http = PolyHttpClient(
            base_url=_DATA_BASE_URL,
            rpm=self._config.ratelimit.data_rpm,
        )
        return SchedulerClients(
            gamma_http=gamma_http,
            data_http=data_http,
            gamma_client=GammaClient(http=gamma_http),
            data_client=DataClient(http=data_http),
            ticks_ws=MarketWebSocket(),
        )

    def _build_collectors(self) -> dict[str, Collector]:
        """Instantiate the watchlist syncer, trade collector, and DC-2 collectors.

        The watchlist syncer and trade collector are always-on (integral to
        the DC-1 data-collection contract). The DC-2 position and activity
        collectors are gated on their respective config flags so operators
        can disable them while the Wave 2 implementation is in flight.
        """
        syncer = WatchlistSyncer(
            registry=self._watchlist_registry,
            tracked_repo=self._tracked_repo,
            sink=self._sink,
            sync_interval_seconds=60.0,
        )
        trades = TradeCollector(
            registry=self._watchlist_registry,
            data_client=self._clients.data_client,
            trades_repo=self._wallet_trades_repo,
        )
        collectors: dict[str, Collector] = {syncer.name: syncer, trades.name: trades}
        if self._config.positions.enabled:
            collectors["position_collector"] = PositionCollector(
                registry=self._watchlist_registry,
                data_client=self._clients.data_client,
                positions_repo=self._positions_repo,
                snapshot_interval_seconds=self._config.positions.snapshot_interval_seconds,
            )
        if self._config.activity.enabled:
            collectors["activity_collector"] = ActivityCollector(
                registry=self._watchlist_registry,
                data_client=self._clients.data_client,
                activity_repo=self._activity_repo,
                poll_interval_seconds=self._config.activity.poll_interval_seconds,
                activity_page_limit=self._config.activity.activity_page_limit,
                max_pages=self._config.activity.max_pages,
                dup_lookback=self._config.activity.dup_lookback,
            )
        if self._config.markets.enabled:
            collectors["market_collector"] = MarketCollector(
                gamma_client=self._clients.gamma_client,
                markets_repo=self._market_snapshots_repo,
                snapshot_interval_seconds=self._config.markets.snapshot_interval_seconds,
                snapshot_max=self._config.markets.snapshot_max,
            )
        if self._config.events.enabled:
            collectors["event_collector"] = EventCollector(
                gamma_client=self._clients.gamma_client,
                events_repo=self._event_snapshots_repo,
                event_tag_cache=self._event_tag_cache_repo,
                snapshot_interval_seconds=self._config.events.snapshot_interval_seconds,
                snapshot_max=self._config.events.snapshot_max,
            )
        if self._config.ticks.enabled:
            collectors["tick_collector"] = MarketTickCollector(
                config=self._config.ticks,
                ws=self._clients.ticks_ws,
                gamma_client=self._clients.gamma_client,
                data_client=self._clients.data_client,
                registry=self._watchlist_registry,
                ticks_repo=self._ticks_repo,
                market_cache=self._market_cache_repo,
            )
        return collectors

    def _build_detectors(self) -> dict[str, Any]:
        """Instantiate the enabled detectors from config."""
        detectors: dict[str, Any] = {}
        if self._config.smart_money.enabled:
            detectors["smart_money"] = SmartMoneyDetector(
                config=self._config.smart_money,
                data_client=self._clients.data_client,
                gamma_client=self._clients.gamma_client,
                tracked_repo=self._tracked_repo,
                snapshots_repo=self._snapshots_repo,
                categories_repo=self._categories_repo,
                event_tag_cache=self._event_tag_cache_repo,
                clock=self._clock,
            )
        if self._config.mispricing.enabled:
            detectors["mispricing"] = MispricingDetector(
                config=self._config.mispricing,
                gamma_client=self._clients.gamma_client,
                sum_history_repo=self._sum_history_repo,
                clock=self._clock,
            )
        if self._config.whales.enabled:
            detectors["whales"] = WhalesDetector(
                config=self._config.whales,
                gamma_client=self._clients.gamma_client,
                data_client=self._clients.data_client,
                market_cache=self._market_cache_repo,
                wallet_first_seen=self._first_seen_repo,
                clock=self._clock,
            )
        if self._config.convergence.enabled:
            detectors["convergence"] = ConvergenceDetector(
                config=self._config.convergence,
                trades_repo=self._wallet_trades_repo,
                category_repo=self._categories_repo,
                market_cache=self._market_cache_repo,
                event_tag_cache=self._event_tag_cache_repo,
                smart_money_config=self._config.smart_money,
            )
        self._maybe_attach_velocity_detector(detectors)
        return detectors

    def _maybe_attach_velocity_detector(self, detectors: dict[str, Any]) -> None:
        """Attach the velocity detector if both ticks and velocity are enabled.

        Velocity depends on the ``tick_collector`` being built (its frozen API
        is the data source); skip silently when ticks are disabled.
        """
        if not self._config.velocity.enabled:
            return
        tick_collector = self._collectors.get("tick_collector")
        if not isinstance(tick_collector, MarketTickCollector):
            return
        detectors["velocity"] = PriceVelocityDetector(
            config=self._config.velocity,
            ticks_collector=tick_collector,
            market_cache=self._market_cache_repo,
            clock=self._clock,
        )

    @property
    def sink(self) -> AlertSink:
        """The shared alert sink (exposed for tests)."""
        return self._sink

    @property
    def renderer(self) -> TerminalRenderer:
        """The terminal renderer (exposed for tests)."""
        return self._renderer

    async def run(self) -> None:
        """Drive the renderer plus every enabled detector and collector forever.

        Detectors and collectors are individually supervised: if one returns
        or raises, the scheduler restarts it up to :data:`_MAX_RESTARTS`
        times within a rolling :data:`_RESTART_WINDOW_SECONDS` window. Beyond
        that the loop gives up and re-raises so the operator sees the
        failure.

        Catches :class:`KeyboardInterrupt` to perform graceful shutdown.
        """
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._renderer.run(), name="renderer")
                for name, detector in self._detectors.items():
                    tg.create_task(
                        self._supervise_detector(name, detector.run),
                        name=f"detector:{name}",
                    )
                for name, collector in self._collectors.items():
                    tg.create_task(
                        self._supervise_collector(name, collector),
                        name=f"collector:{name}",
                    )
        except* KeyboardInterrupt:
            _LOG.info("scanner.shutdown.signal", source="keyboard_interrupt")
        finally:
            await self.aclose()

    async def _supervise_detector(
        self,
        name: str,
        run_fn: Callable[[AlertSink], Awaitable[None]],
    ) -> None:
        """Restart a detector on unexpected return/exception, up to a cap."""
        restarts: list[float] = []
        while True:
            try:
                await run_fn(self._sink)
                _LOG.warning("scanner.detector.returned", detector=name)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("scanner.detector.crashed", detector=name)
            now = time.monotonic()
            restarts = [t for t in restarts if now - t < _RESTART_WINDOW_SECONDS]
            restarts.append(now)
            if len(restarts) > _MAX_RESTARTS:
                msg = f"detector {name} restarted too many times; giving up"
                raise RuntimeError(msg)
            backoff = min(2.0 ** (len(restarts) - 1), 30.0)
            _LOG.info("scanner.detector.restart", detector=name, backoff=backoff)
            await self._clock.sleep(backoff)

    async def _supervise_collector(self, name: str, collector: Collector) -> None:
        """Restart a collector on unexpected return/exception, up to a cap.

        Mirrors :meth:`_supervise_detector` but passes the shared
        ``_collectors_stop`` event so the collector can drain cleanly when
        the daemon shuts down.
        """
        restarts: list[float] = []
        while True:
            if self._collectors_stop.is_set():
                return
            try:
                await collector.run(self._collectors_stop)
                _LOG.info("scanner.collector.returned", collector=name)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("scanner.collector.crashed", collector=name)
            if self._collectors_stop.is_set():
                return
            now = time.monotonic()
            restarts = [t for t in restarts if now - t < _RESTART_WINDOW_SECONDS]
            restarts.append(now)
            if len(restarts) > _MAX_RESTARTS:
                msg = f"collector {name} restarted too many times; giving up"
                raise RuntimeError(msg)
            backoff = min(2.0 ** (len(restarts) - 1), 30.0)
            _LOG.info("scanner.collector.restart", collector=name, backoff=backoff)
            await self._clock.sleep(backoff)

    async def run_once(self) -> dict[str, Any]:
        """Single-pass snapshot: refresh catalog state without opening the WS.

        Returns:
            Counts of work done in this pass — useful for the ``--once`` CLI:
            ``events_scanned``, ``alerts_emitted``, ``tracked_wallets``,
            ``markets_cached``, ``watched_wallets``, ``trades_recorded``,
            ``position_snapshots``, ``activity_events``, ``market_snapshots``,
            ``event_snapshots``.
        """
        baseline_alerts = self._alerts_repo.recent(limit=10000)
        before = len(baseline_alerts)
        trades_before = sum(self._wallet_trades_repo.count_by_wallet().values())
        events_scanned = await self._run_once_mispricing()
        await self._run_once_smart_money()
        await self._run_once_whales()
        collector_counts = await self._run_once_collectors()
        after_alerts = self._alerts_repo.recent(limit=10000)
        markets = self._market_cache_repo.list_active()
        tracked = self._tracked_repo.list_all()
        trades_after = sum(self._wallet_trades_repo.count_by_wallet().values())
        return {
            "events_scanned": events_scanned,
            "alerts_emitted": len(after_alerts) - before,
            "tracked_wallets": len(tracked),
            "markets_cached": len(markets),
            "watched_wallets": len(self._watchlist_registry.addresses()),
            "trades_recorded": trades_after - trades_before,
            "position_snapshots": collector_counts["position_snapshots"],
            "activity_events": collector_counts["activity_events"],
            "market_snapshots": collector_counts["market_snapshots"],
            "event_snapshots": collector_counts["event_snapshots"],
            "tick_snapshots": collector_counts["tick_snapshots"],
        }

    async def _run_once_collectors(self) -> dict[str, int]:
        """Drive a single iteration of each collector.

        ``WatchlistSyncer.sync_smart_money`` mirrors any tracked wallets the
        smart-money detector just upserted into the watchlist, so the trade
        collector's subsequent ``poll_all_wallets`` covers them too. Errors
        are logged and swallowed — single-shot mode should report whatever
        it can finish, not bail on the first transient failure.

        Returns:
            Mapping with ``position_snapshots``, ``activity_events``,
            ``market_snapshots``, ``event_snapshots``, ``tick_snapshots`` keys.
            Each is 0 when the corresponding collector is disabled or raises
            (Wave 1 stubs always raise).
        """
        await self._run_once_watchlist_sync()
        await self._run_once_trade_collector()
        return {
            "position_snapshots": await self._run_once_position_collector(),
            "activity_events": await self._run_once_activity_collector(),
            "market_snapshots": await self._run_once_market_collector(),
            "event_snapshots": await self._run_once_event_collector(),
            "tick_snapshots": await self._run_once_tick_collector(),
        }

    async def _run_once_watchlist_sync(self) -> None:
        """Single-shot mirror of tracked wallets into the watchlist registry."""
        syncer = self._collectors.get("watchlist_sync")
        if not isinstance(syncer, WatchlistSyncer):
            return
        try:
            await syncer.sync_smart_money()
        except Exception:
            _LOG.exception("scanner.run_once.watchlist_sync.failed")

    async def _run_once_trade_collector(self) -> None:
        """Single-shot poll of every watched wallet's ``/activity`` for trades."""
        trades = self._collectors.get("trade_collector")
        if not isinstance(trades, TradeCollector):
            return
        try:
            await trades.poll_all_wallets()
        except Exception:
            _LOG.exception("scanner.run_once.trade_collector.failed")

    async def _run_once_position_collector(self) -> int:
        """Single-shot snapshot of every watched wallet's positions."""
        collector = self._collectors.get("position_collector")
        if not isinstance(collector, PositionCollector):
            return 0
        try:
            return await collector.snapshot_all_wallets()
        except Exception:
            _LOG.exception("scanner.run_once.positions_failed")
            return 0

    async def _run_once_activity_collector(self) -> int:
        """Single-shot poll of every watched wallet's full activity stream."""
        collector = self._collectors.get("activity_collector")
        if not isinstance(collector, ActivityCollector):
            return 0
        try:
            return await collector.poll_all_wallets()
        except Exception:
            _LOG.exception("scanner.run_once.activity_failed")
            return 0

    async def _run_once_market_collector(self) -> int:
        """Single-shot snapshot of every active market."""
        collector = self._collectors.get("market_collector")
        if not isinstance(collector, MarketCollector):
            return 0
        try:
            return await collector.snapshot_all_markets()
        except Exception:
            _LOG.exception("scanner.run_once.markets_failed")
            return 0

    async def _run_once_event_collector(self) -> int:
        """Single-shot snapshot of every active event."""
        collector = self._collectors.get("event_collector")
        if not isinstance(collector, EventCollector):
            return 0
        try:
            return await collector.snapshot_all_events()
        except Exception:
            _LOG.exception("scanner.run_once.events_failed")
            return 0

    async def _run_once_tick_collector(self) -> int:
        """Single-shot snapshot of the WS-driven tick collector's in-memory state."""
        collector = self._collectors.get("tick_collector")
        if not isinstance(collector, MarketTickCollector):
            return 0
        try:
            return await collector.snapshot_once()
        except Exception:
            _LOG.exception("scanner.run_once.ticks_failed")
            return 0

    async def _run_once_mispricing(self) -> int:
        """Run the mispricing scan once over a bounded slice of the event catalogue.

        The full ``/events`` endpoint paginates over thousands of rows; in
        single-shot mode we only need a representative slice so the smoke
        test verifies wiring. The daemon (`run`) iterates everything via the
        detector's own loop.
        """
        detector = self._detectors.get("mispricing")
        if not isinstance(detector, MispricingDetector):
            return 0
        events_scanned = 0
        for page_index in range(_RUN_ONCE_MAX_EVENT_PAGES):
            try:
                page = await self._clients.gamma_client.list_events(
                    active=True,
                    closed=False,
                    limit=_RUN_ONCE_EVENTS_PAGE_SIZE,
                    offset=page_index * _RUN_ONCE_EVENTS_PAGE_SIZE,
                )
            except Exception:
                _LOG.exception("scanner.run_once.mispricing.list_failed")
                break
            if not page:
                break
            for event in page:
                events_scanned += 1
                try:
                    await detector.evaluate_event(event, self._sink)
                except Exception:
                    _LOG.exception(
                        "scanner.run_once.mispricing.event_failed",
                        event_id=event.id,
                    )
            if len(page) < _RUN_ONCE_EVENTS_PAGE_SIZE:
                break
        return events_scanned

    async def _run_once_smart_money(self) -> None:
        """Refresh a small leaderboard slice and poll positions once.

        The daemon's smart-money detector pulls the full top-N leaderboard
        on its hourly cadence; here we cap the candidate pool so a single
        shot stays under the rate limit while still exercising every code
        path (leaderboard → closed-positions → upsert → poll → diff → emit).
        """
        detector = self._detectors.get("smart_money")
        if not isinstance(detector, SmartMoneyDetector):
            return
        try:
            entries = await self._clients.data_client.get_leaderboard(
                period="all",
                limit=_RUN_ONCE_LEADERBOARD_LIMIT,
            )
        except Exception:
            _LOG.exception("scanner.run_once.smart_money.leaderboard_failed")
            entries = []
        for entry in entries:
            try:
                await detector.refresh_one_wallet(entry)
            except Exception:
                _LOG.exception(
                    "scanner.run_once.smart_money.refresh_one_failed",
                    wallet=entry.proxy_wallet,
                )
        try:
            await detector.poll_positions(self._sink)
        except Exception:
            _LOG.exception("scanner.run_once.smart_money.poll_failed")

    async def _run_once_whales(self) -> None:
        """Snapshot the markets cache (one bounded page).

        ``WhalesDetector._refresh_market_cache`` paginates the entire active
        market catalogue (50k+ rows on Polymarket today). That's appropriate
        for a long-running daemon, but in single-shot mode we only need
        enough markets to populate the cache so downstream `pscanner status`
        and the smoke test can verify the wiring. This caches the first
        ``_RUN_ONCE_MARKETS_PAGE_SIZE`` markets directly.
        """
        if "whales" not in self._detectors:
            return
        try:
            markets = await self._clients.gamma_client.list_markets(
                active=True,
                closed=False,
                limit=_RUN_ONCE_MARKETS_PAGE_SIZE,
            )
        except Exception:
            _LOG.exception("scanner.run_once.whales.list_markets_failed")
            return
        for market in markets:
            try:
                self._market_cache_repo.upsert(market)
            except Exception:
                _LOG.exception("scanner.run_once.whales.upsert_failed", market_id=market.id)

    async def aclose(self) -> None:
        """Tear down sockets, HTTP clients, renderer, and DB. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._collectors_stop.set()
        with contextlib.suppress(Exception):
            await self._renderer.stop()
        if self._owns_clients:
            await self._close_owned_clients()
        with contextlib.suppress(sqlite3.Error):
            self._db.close()
        _LOG.info("scanner.shutdown.complete")

    async def _close_owned_clients(self) -> None:
        """Close HTTP clients we own (data_client owns its lb_http internally)."""
        with contextlib.suppress(Exception):
            await self._clients.data_client.aclose()
        with contextlib.suppress(Exception):
            await self._clients.gamma_client.aclose()
        with contextlib.suppress(Exception):
            await self._clients.gamma_http.aclose()
        with contextlib.suppress(Exception):
            await self._clients.data_http.aclose()
        with contextlib.suppress(Exception):
            await self._clients.ticks_ws.close()
