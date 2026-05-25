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
import os
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.alerts.terminal import TerminalRenderer
from pscanner.collectors.activity import ActivityCollector
from pscanner.collectors.base import Collector
from pscanner.collectors.events import EventCollector
from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector
from pscanner.collectors.markets import MarketCollector
from pscanner.collectors.positions import PositionCollector
from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
from pscanner.collectors.trades import TradeCollector
from pscanner.collectors.watchlist import WatchlistRegistry, WatchlistSyncer
from pscanner.config import Config
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.daemon.corpus_loader import (
    DEFAULT_CORPUS_DB,
    load_corpus_metadata,
    load_corpus_resolutions_into,
)
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.subgraph import SubgraphClient
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    EventOutcomeSumRepo,
    EventSnapshotsRepo,
    EventTagCacheRepo,
    MarketCacheRepo,
    MarketSnapshotsRepo,
    MarketTicksRepo,
    PaperTradesRepo,
    PositionSnapshotsRepo,
    SubgraphWatchStateRepo,
    TrackedWalletCategoriesRepo,
    TrackedWalletsRepo,
    WalletActivityEventsRepo,
    WalletClusterMembersRepo,
    WalletClustersRepo,
    WalletFirstSeenRepo,
    WalletPositionsHistoryRepo,
    WalletTradesRepo,
    WatchlistRepo,
)
from pscanner.strategies.evaluators import (
    SignalEvaluator,
    SubgraphCopyEvaluator,
)
from pscanner.strategies.paper_resolver import PaperResolver
from pscanner.strategies.paper_trader import PaperTrader
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)

_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
_DATA_BASE_URL = "https://data-api.polymarket.com"

_MAX_RESTARTS = 3
_RESTART_WINDOW_SECONDS = 300.0

_CollectorT = TypeVar("_CollectorT", bound=Collector)


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
        self._corpus_conn: sqlite3.Connection | None = None
        self._subgraph_client: SubgraphClient | None = None
        if self._config.subgraph_trades.enabled:
            self._corpus_conn = init_corpus_db(DEFAULT_CORPUS_DB)
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
        self._clusters_repo = WalletClustersRepo(self._db)
        self._cluster_members_repo = WalletClusterMembersRepo(self._db)
        self._live_history_provider: LiveHistoryProvider | None = None
        if self._config.gate_model.enabled:
            self._live_history_provider = LiveHistoryProvider(
                conn=self._db,
                metadata=load_corpus_metadata(platform=self._config.gate_model.platform),
            )
            load_corpus_resolutions_into(
                self._live_history_provider,
                platform=self._config.gate_model.platform,
            )
        self._owns_clients = clients is None
        self._clients = clients or self._build_default_clients()
        self._renderer = TerminalRenderer()
        self._sink = AlertSink(self._alerts_repo, renderer=self._renderer)
        self._watchlist_registry = WatchlistRegistry(self._watchlist_repo)
        self._collectors = self._build_collectors()
        self._detectors = self._build_detectors()
        self._wire_alert_subscribers()
        self._collectors_stop = asyncio.Event()
        self._closed = False

    def _wire_alert_subscribers(self) -> None:
        """Register every alert-driven detector as a subscriber on the sink.

        :class:`PaperTrader` only consumes alerts — it never emits — so it
        just needs the subscription.
        """
        for detector in self._detectors.values():
            if isinstance(detector, PaperTrader):
                self._sink.subscribe(detector.handle_alert_sync)
                _LOG.info("scanner.alert_driven_detector_wired", detector=detector.name)

    def _build_default_clients(self) -> SchedulerClients:
        """Construct the production HTTP clients from config."""
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
            wallet_first_seen=self._first_seen_repo,
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
                market_cache=self._market_cache_repo,
            )
        if self._config.events.enabled:
            collectors["event_collector"] = EventCollector(
                gamma_client=self._clients.gamma_client,
                events_repo=self._event_snapshots_repo,
                event_tag_cache=self._event_tag_cache_repo,
                snapshot_interval_seconds=self._config.events.snapshot_interval_seconds,
                snapshot_max=self._config.events.snapshot_max,
            )
        if self._config.gate_model_market_filter.enabled:
            collectors["market_scoped_trades"] = MarketScopedTradeCollector(
                config=self._config.gate_model_market_filter,
                gamma=self._clients.gamma_client,
                data_client=self._clients.data_client,
                provider=self._live_history_provider,
                market_cache=self._market_cache_repo,
            )
        self._maybe_attach_subgraph_trade_collector(collectors)
        return collectors

    def _maybe_attach_subgraph_trade_collector(
        self,
        collectors: dict[str, Collector],
    ) -> None:
        """Wire the SubgraphTradeCollector when ``subgraph_trades`` is enabled."""
        if not self._config.subgraph_trades.enabled:
            return
        if self._corpus_conn is None:
            msg = (
                "subgraph_trades.enabled=true but corpus connection is None; "
                "Scanner.__init__ should have opened it."
            )
            raise RuntimeError(msg)
        api_key = os.environ.get("GRAPH_API_KEY", "")
        subgraph_url = (
            f"https://gateway.thegraph.com/api/{api_key}"
            f"/subgraphs/id/{self._config.subgraph_trades.subgraph_id}"
        )
        self._subgraph_client = SubgraphClient(
            url=subgraph_url,
            rpm=self._config.subgraph_trades.rpm,
        )
        collectors["subgraph_trades"] = SubgraphTradeCollector(
            config=self._config.subgraph_trades,
            subgraph_client=self._subgraph_client,
            gamma_client=self._clients.gamma_client,
            watchlist=self._watchlist_registry,
            asset_index=AssetIndexRepo(self._corpus_conn),
            market_cache=self._market_cache_repo,
            sink=self._sink,
            state_repo=SubgraphWatchStateRepo(self._db),
            clock=self._clock,
        )

    def _build_detectors(self) -> dict[str, Any]:
        """Instantiate the enabled detectors from config."""
        detectors: dict[str, Any] = {}
        self._maybe_attach_paper_trading(detectors)
        return detectors

    def _maybe_attach_paper_trading(self, detectors: dict[str, Any]) -> None:
        """Attach paper-trader and paper-resolver when paper trading is enabled."""
        if not self._config.paper_trading.enabled:
            return
        paper_trades_repo = PaperTradesRepo(self._db)
        detectors["paper_trader"] = PaperTrader(
            config=self._config.paper_trading,
            evaluators=self._build_paper_evaluators(paper_trades_repo),
            market_cache=self._market_cache_repo,
            paper_trades=paper_trades_repo,
            market_ticks=self._ticks_repo,
            data_client=self._clients.data_client,
            gamma_client=self._clients.gamma_client,
            alerts_repo=self._alerts_repo,
        )
        detectors["paper_resolver"] = PaperResolver(
            config=self._config.paper_trading,
            market_cache=self._market_cache_repo,
            paper_trades=paper_trades_repo,
            clock=self._clock,
        )

    def _build_paper_evaluators(
        self,
        paper_trades_repo: PaperTradesRepo | None = None,
    ) -> list[SignalEvaluator]:
        """Construct enabled evaluators in fixed order.

        Each evaluator is gated by its ``enabled`` flag in
        ``paper_trading.evaluators.<source>``; disabled sources are simply
        not in the list.
        """
        cfg = self._config.paper_trading.evaluators
        evaluators: list[SignalEvaluator] = []
        if cfg.subgraph_copy.enabled:
            repo = paper_trades_repo if paper_trades_repo is not None else PaperTradesRepo(self._db)
            evaluators.append(
                SubgraphCopyEvaluator(
                    config=cfg.subgraph_copy,
                    watchlist_repo=self._watchlist_repo,
                    paper_trades=repo,
                )
            )
        return evaluators

    def preflight(self) -> None:
        """Run startup checks before entering the run loop.

        When ``gate_model`` is enabled, refuses to start unless:
        - ``wallet_state_live`` has been populated via
          ``pscanner daemon bootstrap-features``.
        - ``markets.enabled`` is true (the markets collector populates the
          ``MarketCacheRepo`` that the gate-model live path depends on to
          map ``asset_id`` to YES/NO; without it, every trade silently
          drops — see issue #101).

        When ``subgraph_trades`` is enabled, refuses to start unless
        ``GRAPH_API_KEY`` is set in the environment — the SubgraphClient
        needs it to construct the gateway URL.
        """
        if self._config.gate_model.enabled:
            row = self._db.execute("SELECT 1 FROM wallet_state_live LIMIT 1").fetchone()
            if row is None:
                msg = (
                    "gate_model.enabled=true but wallet_state_live is empty. "
                    "Run `pscanner daemon bootstrap-features` first."
                )
                raise RuntimeError(msg)
            if not self._config.markets.enabled:
                msg = (
                    "gate_model.enabled=true but markets.enabled=false. "
                    "The gate-model detector requires the markets collector to "
                    "populate MarketCacheRepo (used to map asset_id -> YES/NO). "
                    "Set [markets] enabled = true in your config."
                )
                raise RuntimeError(msg)
        if self._config.subgraph_trades.enabled and not os.environ.get("GRAPH_API_KEY"):
            msg = (
                "subgraph_trades.enabled=true but GRAPH_API_KEY is not set. "
                "Export it before starting the daemon."
            )
            raise RuntimeError(msg)

    async def _replay_paper_trader(self) -> None:
        """Replay unbooked alerts when paper-trading is enabled (issue #105)."""
        trader = self._detectors.get("paper_trader")
        if not isinstance(trader, PaperTrader):
            return
        try:
            await trader.replay_unbooked()
        except Exception:
            _LOG.exception("scanner.paper_trader_replay_failed")

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
        self.preflight()
        await self._replay_paper_trader()
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
            ``tracked_wallets``, ``markets_cached``, ``watched_wallets``,
            ``trades_recorded``, ``position_snapshots``, ``activity_events``,
            ``market_snapshots``, ``event_snapshots``.
        """
        baseline_alerts = self._alerts_repo.recent(limit=10000)
        before = len(baseline_alerts)
        trades_before = sum(self._wallet_trades_repo.count_by_wallet().values())
        collector_counts = await self._run_once_collectors()
        after_alerts = self._alerts_repo.recent(limit=10000)
        markets = self._market_cache_repo.list_active()
        tracked = self._tracked_repo.list_all()
        trades_after = sum(self._wallet_trades_repo.count_by_wallet().values())
        return {
            "alerts_emitted": len(after_alerts) - before,
            "tracked_wallets": len(tracked),
            "markets_cached": len(markets),
            "watched_wallets": len(self._watchlist_registry.addresses()),
            "trades_recorded": trades_after - trades_before,
            "position_snapshots": collector_counts["position_snapshots"],
            "activity_events": collector_counts["activity_events"],
            "market_snapshots": collector_counts["market_snapshots"],
            "event_snapshots": collector_counts["event_snapshots"],
        }

    async def _run_once_collectors(self) -> dict[str, int]:
        """Drive a single iteration of each collector.

        ``WatchlistSyncer.sync_smart_money`` mirrors any tracked wallets the
        upstream catalog refresh just upserted into the watchlist, so the
        trade collector's subsequent ``poll_all_wallets`` covers them too.
        Errors are logged and swallowed — single-shot mode should report
        whatever it can finish, not bail on the first transient failure.

        Returns:
            Mapping with ``position_snapshots``, ``activity_events``,
            ``market_snapshots``, ``event_snapshots`` keys. Each is 0 when
            the corresponding collector is disabled or raises.
        """
        await self._run_once_watchlist_sync()
        await self._run_once_trade_collector()
        return {
            "position_snapshots": await self._run_once_collector(
                key="position_collector",
                type_=PositionCollector,
                call=PositionCollector.snapshot_all_wallets,
                log_event="scanner.run_once.positions_failed",
            ),
            "activity_events": await self._run_once_collector(
                key="activity_collector",
                type_=ActivityCollector,
                call=ActivityCollector.poll_all_wallets,
                log_event="scanner.run_once.activity_failed",
            ),
            "market_snapshots": await self._run_once_collector(
                key="market_collector",
                type_=MarketCollector,
                call=MarketCollector.snapshot_all_markets,
                log_event="scanner.run_once.markets_failed",
            ),
            "event_snapshots": await self._run_once_collector(
                key="event_collector",
                type_=EventCollector,
                call=EventCollector.snapshot_all_events,
                log_event="scanner.run_once.events_failed",
            ),
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

    async def _run_once_collector(
        self,
        *,
        key: str,
        type_: type[_CollectorT],
        call: Callable[[_CollectorT], Awaitable[int]],
        log_event: str,
    ) -> int:
        """Run one int-returning collector's single-pass entrypoint.

        Returns 0 when the collector is missing, the wrong type, or the
        call raises — matches the pre-collapse per-method behaviour where
        single-shot mode reports whatever it can finish.
        """
        collector = self._collectors.get(key)
        if not isinstance(collector, type_):
            return 0
        try:
            return await call(collector)
        except Exception:
            _LOG.exception(log_event)
            return 0

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
        if self._subgraph_client is not None:
            with contextlib.suppress(Exception):
                await self._subgraph_client.aclose()
        with contextlib.suppress(sqlite3.Error):
            self._db.close()
        if self._corpus_conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self._corpus_conn.close()
        _LOG.info("scanner.shutdown.complete")

    async def _close_owned_clients(self) -> None:
        """Close HTTP clients we own (data_client owns its lb_http internally)."""
        closers: tuple[Callable[[], Awaitable[None]], ...] = (
            self._clients.data_client.aclose,
            self._clients.gamma_client.aclose,
            self._clients.gamma_http.aclose,
            self._clients.data_http.aclose,
        )
        for closer in closers:
            with contextlib.suppress(Exception):
                await closer()
