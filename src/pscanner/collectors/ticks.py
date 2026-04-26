"""Market tick collector — append-only price snapshots from the WS orderbook.

Owns its own ``MarketWebSocket`` connection. Maintains an in-memory orderbook
per subscribed asset (driven by ``book``/``price_change``/``last_trade_price``
events) and, on each tick interval, writes one row per asset to
``MarketTicksRepo`` with derived best_bid/best_ask/mid/spread/depth fields.

Subscription scope = (assets held by watched wallets) U (active markets above
``tick_volume_floor_usd``), capped at ``max_assets``. Re-evaluated on
``subscription_refresh_seconds``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import TicksConfig
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Position, WsBookMessage
from pscanner.store.repo import MarketTick, MarketTicksRepo

_LOG = structlog.get_logger(__name__)

_SUBSCRIBE_BATCH_SIZE = 100
_MID_HISTORY_CAP = 200
_DEPTH_TOP_N = 5


@dataclass(slots=True)
class _Orderbook:
    """In-memory orderbook for a single asset."""

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_trade_price: float | None = None
    condition_id: str | None = None


class MarketTickCollector:
    """Maintains an in-memory orderbook per subscribed asset and writes ticks.

    Owns its own ``MarketWebSocket`` connection (separate from any other WS
    consumer). Subscription scope = (assets held by watched wallets) U (active
    markets above ``tick_volume_floor_usd``), capped at ``max_assets``.
    """

    name: str = "tick_collector"

    def __init__(
        self,
        *,
        config: TicksConfig,
        ws: MarketWebSocket,
        gamma_client: GammaClient,
        data_client: DataClient,
        registry: WatchlistRegistry,
        ticks_repo: MarketTicksRepo,
    ) -> None:
        """Build the collector. See module docstring for behaviour."""
        self._config = config
        self._ws = ws
        self._gamma = gamma_client
        self._data = data_client
        self._registry = registry
        self._repo = ticks_repo
        self._books: dict[str, _Orderbook] = {}
        self._asset_to_condition: dict[str, str] = {}
        self._mid_history: dict[str, list[tuple[int, float]]] = {}
        self._subscribed: set[str] = set()
        self._lock = asyncio.Lock()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Connect WS, drive subscription/consume/snapshot loops until stopped.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        await self._ws.connect()
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(self._subscription_refresh_loop(stop_event)),
            asyncio.create_task(self._consume_loop(stop_event)),
            asyncio.create_task(self._snapshot_loop(stop_event)),
        ]
        try:
            await stop_event.wait()
        finally:
            with contextlib.suppress(Exception):
                await self._ws.close()
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _subscription_refresh_loop(self, stop_event: asyncio.Event) -> None:
        """Refresh the subscription set immediately, then on a cadence."""
        interval = self._config.subscription_refresh_seconds
        while not stop_event.is_set():
            try:
                await self._refresh_subscriptions()
            except Exception:
                _LOG.exception("ticks.subscription_refresh_failed")
            if await _wait_or_stop(stop_event, interval):
                return

    async def _consume_loop(self, stop_event: asyncio.Event) -> None:
        """Consume WS messages and dispatch to the handler until stopped."""
        try:
            async for msg in self._ws.messages():
                if stop_event.is_set():
                    return
                if not isinstance(msg, WsBookMessage):
                    continue
                try:
                    await self._handle_message(msg)
                except Exception:
                    _LOG.exception("ticks.handle_message_failed")
        except Exception:
            _LOG.exception("ticks.consume_loop_failed")

    async def _snapshot_loop(self, stop_event: asyncio.Event) -> None:
        """Run ``snapshot_once`` every ``tick_interval_seconds`` until stopped."""
        interval = self._config.tick_interval_seconds
        while not stop_event.is_set():
            if await _wait_or_stop(stop_event, interval):
                return
            try:
                await self.snapshot_once()
            except Exception:
                _LOG.exception("ticks.snapshot_iteration_failed")

    async def _refresh_subscriptions(self) -> None:
        """Recompute the asset universe and subscribe to any new ids."""
        wallet_assets, wallet_lookup = await self._collect_wallet_assets()
        volume_assets, volume_lookup = await self._collect_volume_floor_assets(
            existing_count=len(wallet_assets),
        )
        target = wallet_assets | volume_assets
        target = self._cap_target(target, wallet_assets)
        merged_lookup: dict[str, str] = {}
        merged_lookup.update(volume_lookup)
        merged_lookup.update(wallet_lookup)

        new_ids = target - self._subscribed
        async with self._lock:
            for asset_id, condition_id in merged_lookup.items():
                if condition_id:
                    self._asset_to_condition[asset_id] = condition_id

        if new_ids:
            await self._subscribe_in_batches(sorted(new_ids))
            self._subscribed |= new_ids

        _LOG.info(
            "ticks.subscriptions.refreshed",
            assets=len(self._subscribed),
            new=len(new_ids),
        )

    async def _collect_wallet_assets(self) -> tuple[set[str], dict[str, str]]:
        """Return assets held by watched wallets and the asset→condition map."""
        assets: set[str] = set()
        lookup: dict[str, str] = {}
        for address in sorted(self._registry.addresses()):
            try:
                positions = await self._data.get_positions(address)
            except Exception:
                _LOG.exception("ticks.get_positions_failed", wallet=address)
                continue
            for pos in positions:
                _ingest_position(pos, assets=assets, lookup=lookup)
        return assets, lookup

    async def _collect_volume_floor_assets(
        self,
        *,
        existing_count: int,
    ) -> tuple[set[str], dict[str, str]]:
        """Return assets from active markets above the volume floor."""
        assets: set[str] = set()
        lookup: dict[str, str] = {}
        floor = self._config.tick_volume_floor_usd
        cap = self._config.max_assets
        async for market in self._gamma.iter_markets(active=True, closed=False):
            if not market.enable_order_book:
                continue
            volume = market.volume or 0.0
            if volume < floor:
                continue
            for asset_id in market.clob_token_ids:
                if not asset_id:
                    continue
                assets.add(asset_id)
                if market.condition_id:
                    lookup[asset_id] = market.condition_id
            if existing_count + len(assets) >= cap:
                break
        return assets, lookup

    def _cap_target(
        self,
        target: set[str],
        wallet_assets: set[str],
    ) -> set[str]:
        """Truncate ``target`` to ``max_assets``, preferring wallet assets."""
        cap = self._config.max_assets
        if len(target) <= cap:
            return target
        capped: set[str] = set()
        for asset_id in sorted(wallet_assets & target):
            if len(capped) >= cap:
                break
            capped.add(asset_id)
        remaining = sorted(target - capped)
        for asset_id in remaining:
            if len(capped) >= cap:
                break
            capped.add(asset_id)
        return capped

    async def _subscribe_in_batches(self, asset_ids: list[str]) -> None:
        """Send subscription requests in chunks of ``_SUBSCRIBE_BATCH_SIZE``."""
        for start in range(0, len(asset_ids), _SUBSCRIBE_BATCH_SIZE):
            chunk = asset_ids[start : start + _SUBSCRIBE_BATCH_SIZE]
            try:
                await self._ws.subscribe(chunk)
            except Exception:
                _LOG.exception("ticks.subscribe_failed", batch_size=len(chunk))

    async def _handle_message(self, msg: WsBookMessage) -> None:
        """Apply one WS message to the in-memory orderbook state."""
        async with self._lock:
            asset_id = msg.asset_id or ""
            book = self._books.setdefault(asset_id, _Orderbook())
            if msg.market and not book.condition_id:
                book.condition_id = msg.market

            if msg.event_type == "book":
                self._apply_book_snapshot(book, msg)
            elif msg.event_type == "price_change":
                self._apply_price_changes(msg)
            elif msg.event_type == "last_trade_price":
                self._apply_last_trade_price(book, msg)
            # ``tick_size_change`` is intentionally ignored.

    def _apply_book_snapshot(self, book: _Orderbook, msg: WsBookMessage) -> None:
        """Replace ``book``'s state from a full ``book`` snapshot payload."""
        book.bids = self._parse_levels(msg.bids or [])
        book.asks = self._parse_levels(msg.asks or [])
        if msg.last_trade_price is not None:
            try:
                book.last_trade_price = float(msg.last_trade_price)
            except (ValueError, TypeError):
                _LOG.debug("ticks.book.bad_last_trade_price", value=msg.last_trade_price)

    def _apply_price_changes(self, msg: WsBookMessage) -> None:
        """Apply each per-asset price change inside a ``price_change`` payload."""
        changes = msg.price_changes or []
        for change in changes:
            if not isinstance(change, dict):
                continue
            self._apply_one_price_change(change, market=msg.market)

    def _apply_one_price_change(
        self,
        change: dict[str, Any],
        *,
        market: str | None,
    ) -> None:
        """Apply a single entry from a ``price_changes`` array."""
        asset_id = change.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            return
        sub = self._books.setdefault(asset_id, _Orderbook())
        if market and not sub.condition_id:
            sub.condition_id = market
        try:
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
        except (ValueError, TypeError):
            return
        side = str(change.get("side", "")).upper()
        if side == "BUY":
            book_dict: dict[float, float] | None = sub.bids
        elif side == "SELL":
            book_dict = sub.asks
        else:
            book_dict = None
        if book_dict is None:
            return
        if size == 0:
            book_dict.pop(price, None)
        else:
            book_dict[price] = size

    def _apply_last_trade_price(self, book: _Orderbook, msg: WsBookMessage) -> None:
        """Apply a ``last_trade_price`` payload to ``book``."""
        if msg.last_trade_price is None:
            return
        try:
            value = float(msg.last_trade_price)
        except (ValueError, TypeError):
            return
        if value:
            book.last_trade_price = value

    @staticmethod
    def _parse_levels(items: Any) -> dict[float, float]:
        """Convert a ``[{"price": "0.5", "size": "100"}, ...]`` list to a dict.

        Args:
            items: Raw level list from the WS payload.

        Returns:
            ``price -> size`` dict; entries with ``size <= 0`` are dropped.
        """
        out: dict[float, float] = {}
        if not isinstance(items, list):
            return out
        for entry in items:
            if not isinstance(entry, dict):
                continue
            try:
                price = float(entry.get("price", 0))
                size = float(entry.get("size", 0))
            except (ValueError, TypeError):
                continue
            if size > 0:
                out[price] = size
        return out

    async def snapshot_once(self) -> int:
        """Walk in-memory orderbook state, write one row per asset.

        Returns:
            Number of rows newly inserted.
        """
        snapshot_at = int(time.time())
        async with self._lock:
            books_copy = list(self._books.items())
            condition_lookup = dict(self._asset_to_condition)
        inserted = 0
        for asset_id, book in books_copy:
            if not asset_id:
                continue
            if self._persist_tick(
                asset_id=asset_id,
                book=book,
                condition_lookup=condition_lookup,
                snapshot_at=snapshot_at,
            ):
                inserted += 1
        _LOG.info("ticks.snapshot_complete", assets=len(books_copy), inserted=inserted)
        return inserted

    def _persist_tick(
        self,
        *,
        asset_id: str,
        book: _Orderbook,
        condition_lookup: dict[str, str],
        snapshot_at: int,
    ) -> bool:
        """Build and insert one tick row; return True on insert."""
        best_bid = max(book.bids) if book.bids else None
        best_ask = min(book.asks) if book.asks else None
        if best_bid is not None and best_ask is not None:
            mid: float | None = (best_bid + best_ask) / 2
            spread: float | None = best_ask - best_bid
        else:
            mid = None
            spread = None
        condition_id = book.condition_id or condition_lookup.get(asset_id, "")
        if not condition_id and mid is None:
            return False
        bid_depth = _depth_top_n(book.bids, top=_DEPTH_TOP_N, descending=True)
        ask_depth = _depth_top_n(book.asks, top=_DEPTH_TOP_N, descending=False)
        tick = MarketTick(
            asset_id=asset_id,
            condition_id=condition_id,
            snapshot_at=snapshot_at,
            mid_price=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            bid_depth_top5=bid_depth,
            ask_depth_top5=ask_depth,
            last_trade_price=book.last_trade_price,
        )
        try:
            inserted = self._repo.insert(tick)
        except Exception:
            _LOG.exception("ticks.insert_failed", asset_id=asset_id)
            return False
        if inserted and mid is not None:
            self._record_mid_history(asset_id, snapshot_at=snapshot_at, mid=mid)
        return inserted

    def _record_mid_history(self, asset_id: str, *, snapshot_at: int, mid: float) -> None:
        """Append ``(snapshot_at, mid)`` to the asset's history ring buffer."""
        history = self._mid_history.setdefault(asset_id, [])
        history.append((snapshot_at, mid))
        if len(history) > _MID_HISTORY_CAP:
            del history[: len(history) - _MID_HISTORY_CAP]

    def get_recent_mids(
        self,
        asset_id: str,
        *,
        window_seconds: int,
    ) -> list[tuple[int, float]]:
        """Return ``(snapshot_at, mid_price)`` pairs within the trailing window.

        FROZEN API: the velocity detector calls this. Reads from in-memory
        state appended on each successful ``snapshot_once`` write.

        Args:
            asset_id: CLOB token id.
            window_seconds: Inclusive trailing window length (seconds).

        Returns:
            Pairs ordered by ``snapshot_at`` ascending.
        """
        now = int(time.time())
        cutoff = now - window_seconds
        history = self._mid_history.get(asset_id, [])
        return [(ts, mid) for ts, mid in history if ts > cutoff]

    def subscribed_asset_ids(self) -> set[str]:
        """Return a copy of the currently-subscribed asset id set."""
        return self._subscribed.copy()


def _ingest_position(
    pos: Position,
    *,
    assets: set[str],
    lookup: dict[str, str],
) -> None:
    """Add ``pos.asset`` to ``assets`` and map it to its condition id."""
    asset_id = pos.asset
    if not asset_id:
        return
    assets.add(asset_id)
    if pos.condition_id:
        lookup[asset_id] = pos.condition_id


def _depth_top_n(
    levels: dict[float, float],
    *,
    top: int,
    descending: bool,
) -> float | None:
    """Sum the sizes of the top-``N`` price levels in ``levels``.

    Args:
        levels: ``price -> size`` dict.
        top: Maximum levels to include.
        descending: ``True`` for bids (highest first), ``False`` for asks.

    Returns:
        Sum of top-N sizes, or ``None`` when ``levels`` is empty.
    """
    if not levels:
        return None
    sorted_prices = sorted(levels, reverse=descending)[:top]
    return sum(levels[price] for price in sorted_prices)


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
    """Wait up to ``seconds`` for the stop event.

    Returns:
        ``True`` if the stop event was set during the wait, ``False`` if the
        timeout elapsed first.
    """
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True
