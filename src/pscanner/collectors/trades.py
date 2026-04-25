"""Trade collector — DC-1 Wave B.

Records every CONFIRMED trade by a watched wallet to ``wallet_trades``. The
collector owns its own websocket connection (separate from the whales
detector) to keep its subscription set focused on the watchlist's open
positions.

Subscription strategy:

* On startup and every ``subscription_refresh_seconds`` thereafter, fetch the
  current open positions for every wallet in ``WatchlistRegistry`` and
  subscribe the websocket to the union of asset ids encountered. New asset
  ids are sent in batches of 100 (matches the whales-detector pattern).
* When a wallet is added to the watchlist mid-run, the ``_on_watchlist_add``
  callback schedules an immediate (off-cycle) subscribe for that wallet's
  positions so we never miss the very first trade after enrolment.

Persistence:

* Each ``CONFIRMED`` ``WsTradeMessage`` whose ``taker_proxy`` is in the
  registry is converted to a :class:`WalletTrade` and inserted via
  :class:`WalletTradesRepo`. The repo's composite primary key handles dedupe.
* Trades arriving without a ``transaction_hash`` are dropped — the dedupe key
  relies on the hash, so persisting them would risk silent duplicates.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator

import structlog

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.poly.models import WsTradeMessage
from pscanner.store.repo import WalletTrade, WalletTradesRepo

_LOG = structlog.get_logger(__name__)
_SUBSCRIBE_BATCH_SIZE = 100
_MAX_POSITIONS_PER_WALLET = 500


class TradeCollector:
    """Records every CONFIRMED trade by a watched wallet to ``wallet_trades``.

    Owns its own :class:`MarketWebSocket` connection. Subscribes to the union
    of asset ids across watched wallets' open positions; refreshes that
    subscription set when the watchlist changes or every
    ``subscription_refresh_seconds``.
    """

    name: str = "trade_collector"

    def __init__(
        self,
        *,
        registry: WatchlistRegistry,
        data_client: DataClient,
        ws: MarketWebSocket,
        trades_repo: WalletTradesRepo,
        subscription_refresh_seconds: float = 300.0,
    ) -> None:
        """Wire the collector to the registry, data client, websocket, and repo.

        Args:
            registry: In-memory watchlist of wallet addresses to record.
            data_client: REST client used to fetch each wallet's open positions.
            ws: Market websocket. The collector owns the connection lifecycle.
            trades_repo: Append-only repo for ``wallet_trades`` rows.
            subscription_refresh_seconds: Interval between full subscription
                refreshes.
        """
        self._registry = registry
        self._data_client = data_client
        self._ws = ws
        self._trades_repo = trades_repo
        self._subscription_refresh_seconds = subscription_refresh_seconds
        self._subscribed_assets: set[str] = set()
        self._pending_add_tasks: set[asyncio.Task[None]] = set()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the collector until ``stop_event`` is set.

        Connects the websocket, registers the watchlist-add callback, and
        runs the subscription-refresh loop and message-consume loop
        concurrently. When ``stop_event`` fires, both loops are cancelled
        and the websocket is closed.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        await self._ws.connect()
        self._registry.subscribe(self._on_watchlist_add)
        refresh_task = asyncio.create_task(self._subscription_refresh_loop(stop_event))
        consume_task = asyncio.create_task(self._consume_loop(stop_event))
        worker_tasks = (refresh_task, consume_task)
        try:
            await stop_event.wait()
        finally:
            for task in worker_tasks:
                task.cancel()
            for task in worker_tasks:
                with _suppress_cancelled():
                    await task
            await self._ws.close()

    async def _subscription_refresh_loop(self, stop_event: asyncio.Event) -> None:
        """Refresh asset subscriptions for every watched wallet on a cadence."""
        while not stop_event.is_set():
            try:
                await self._refresh_subscriptions()
            except Exception:
                _LOG.exception("trades.subscription_refresh_failed")
            await self._sleep_until_stop(self._subscription_refresh_seconds, stop_event)

    async def _consume_loop(self, stop_event: asyncio.Event) -> None:
        """Consume websocket messages and persist trades by watched wallets."""
        async for msg in self._ws.messages():
            if stop_event.is_set():
                return
            if not isinstance(msg, WsTradeMessage):
                continue
            if msg.taker_proxy not in self._registry:
                continue
            try:
                self._record_trade(msg)
            except Exception:
                _LOG.exception(
                    "trades.record_failed",
                    wallet=msg.taker_proxy,
                    tx_hash=msg.transaction_hash,
                )

    async def _refresh_subscriptions(self) -> None:
        """Pull each watched wallet's positions and subscribe to new asset ids."""
        watched = self._registry.addresses()
        new_ids: set[str] = set()
        for wallet in watched:
            asset_ids = await self._fetch_wallet_assets(wallet)
            new_ids.update(asset_ids)
        await self._subscribe_new(new_ids)
        _LOG.info(
            "trades.subscriptions.refreshed",
            watched_wallets=len(watched),
            assets=len(self._subscribed_assets),
        )

    async def _fetch_wallet_assets(self, wallet: str) -> list[str]:
        """Return the asset ids of a wallet's currently-open positions.

        Applies the ``_MAX_POSITIONS_PER_WALLET`` sanity cap so a single huge
        wallet cannot blow through the websocket subscription budget.
        """
        try:
            positions = await self._data_client.get_positions(wallet)
        except Exception:
            _LOG.exception("trades.get_positions_failed", wallet=wallet)
            return []
        if len(positions) > _MAX_POSITIONS_PER_WALLET:
            _LOG.warning(
                "trades.positions.capped",
                wallet=wallet,
                seen=len(positions),
                cap=_MAX_POSITIONS_PER_WALLET,
            )
            positions = positions[:_MAX_POSITIONS_PER_WALLET]
        return [position.asset for position in positions]

    async def _subscribe_new(self, asset_ids: set[str]) -> None:
        """Subscribe to ``asset_ids`` not yet in ``_subscribed_assets`` in batches."""
        new_ids = sorted(asset_ids - self._subscribed_assets)
        if not new_ids:
            return
        for start in range(0, len(new_ids), _SUBSCRIBE_BATCH_SIZE):
            batch = new_ids[start : start + _SUBSCRIBE_BATCH_SIZE]
            await self._ws.subscribe(batch)
        self._subscribed_assets.update(new_ids)

    def _on_watchlist_add(self, address: str) -> None:
        """Schedule an immediate subscribe for ``address``'s positions.

        Called from inside :meth:`WatchlistRegistry.add`, so this must be
        non-blocking. We spawn a task and keep a strong reference to it so
        it isn't garbage-collected before completion.
        """
        task = asyncio.create_task(self._subscribe_for_new_wallet(address))
        self._pending_add_tasks.add(task)
        task.add_done_callback(self._pending_add_tasks.discard)

    async def _subscribe_for_new_wallet(self, address: str) -> None:
        """Fetch ``address``'s open positions and subscribe to their assets."""
        try:
            asset_ids = await self._fetch_wallet_assets(address)
            await self._subscribe_new(set(asset_ids))
        except Exception:
            _LOG.exception("trades.subscribe_for_new_wallet_failed", wallet=address)

    def _record_trade(self, msg: WsTradeMessage) -> None:
        """Persist a single ``CONFIRMED`` trade message via the repo."""
        tx_hash = msg.transaction_hash or ""
        if not tx_hash:
            _LOG.warning(
                "trades.skip_missing_tx_hash",
                wallet=msg.taker_proxy,
                asset_id=msg.asset_id,
            )
            return
        trade = WalletTrade(
            transaction_hash=tx_hash,
            asset_id=msg.asset_id,
            side=msg.side,
            wallet=msg.taker_proxy,
            condition_id=msg.condition_id,
            size=msg.size,
            price=msg.price,
            usd_value=msg.size * msg.price,
            status=msg.status,
            source="ws",
            timestamp=msg.timestamp,
            recorded_at=int(time.time()),
        )
        inserted = self._trades_repo.insert(trade)
        if not inserted:
            _LOG.debug(
                "trades.duplicate",
                wallet=msg.taker_proxy,
                tx_hash=tx_hash,
                asset_id=msg.asset_id,
                side=msg.side,
            )

    @staticmethod
    async def _sleep_until_stop(seconds: float, stop_event: asyncio.Event) -> None:
        """Sleep ``seconds`` or until ``stop_event`` is set, whichever first."""
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        except TimeoutError:
            return


@contextlib.contextmanager
def _suppress_cancelled() -> Iterator[None]:
    """Swallow ``asyncio.CancelledError`` while joining a cancelled worker task."""
    try:
        yield
    except asyncio.CancelledError:
        return
