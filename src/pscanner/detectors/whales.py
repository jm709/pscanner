"""Whale detector — newly-active wallet places an outsized bet on a tiny market.

Subscribes to the CLOB websocket trade stream, joins each trade to the cached
gamma market metadata, and emits an alert when:

* the market's liquidity is below ``small_market_max_liquidity_usd``,
* the trade USD notional is above ``big_bet_min_usd`` *and* above
  ``big_bet_min_pct_of_liquidity`` of the market's liquidity, and
* the taker wallet was first seen less than ``new_account_max_age_days`` ago
  with fewer than ``new_account_max_trades`` total trades.

Wallet age and total-trade counts are cached in ``wallet_first_seen`` for 24h
to keep data-API request volume bounded.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import WhalesConfig
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import WsTradeMessage
from pscanner.store.repo import (
    CachedMarket,
    MarketCacheRepo,
    WalletFirstSeen,
    WalletFirstSeenRepo,
)

if TYPE_CHECKING:
    from pscanner.poly.models import Market

_LOG = structlog.get_logger(__name__)
_WALLET_CACHE_TTL_SECONDS = 86400
_HIGH_SEVERITY_USD = 10000.0
_HIGH_SEVERITY_PCT = 0.20
_ACTIVITY_PROBE_LIMIT = 200


class WhalesDetector:
    """Detector that emits alerts for big trades by new wallets on tiny markets."""

    name = "whales"

    def __init__(
        self,
        *,
        config: WhalesConfig,
        ws: MarketWebSocket,
        gamma_client: GammaClient,
        data_client: DataClient,
        market_cache: MarketCacheRepo,
        wallet_first_seen: WalletFirstSeenRepo,
    ) -> None:
        """Build the detector with its config and external dependencies.

        Args:
            config: Threshold settings from ``WhalesConfig``.
            ws: Connected (or pending) CLOB websocket client.
            gamma_client: Source of the active markets catalogue.
            data_client: Source of wallet activity for age lookups.
            market_cache: Persistent cache of market metadata.
            wallet_first_seen: Persistent cache of wallet first-activity rows.
        """
        self._config = config
        self._ws = ws
        self._gamma_client = gamma_client
        self._data_client = data_client
        self._market_cache = market_cache
        self._wallet_first_seen = wallet_first_seen
        self._asset_to_market: dict[str, str] = {}

    async def run(self, sink: AlertSink) -> None:
        """Connect the websocket and run the subscription + consume loops."""
        await self._ws.connect()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._subscription_loop())
            tg.create_task(self._consume_loop(sink))

    async def _subscription_loop(self) -> None:
        """Refresh the active markets and resubscribe forever."""
        while True:
            await self._refresh_subscriptions()
            await asyncio.sleep(self._config.ws_resubscribe_interval_seconds)

    async def _refresh_subscriptions(self) -> None:
        """Pull every active market, cache it, and (re)subscribe in batches."""
        asset_ids: list[str] = []
        async for market in self._gamma_client.iter_markets(active=True, closed=False):  # ty:ignore[not-iterable]
            self._cache_and_index(market, asset_ids)
        await self._subscribe_in_batches(asset_ids)
        _LOG.debug(
            "whales.subscriptions.refreshed",
            markets=len(self._asset_to_market),
            assets=len(asset_ids),
        )

    def _cache_and_index(self, market: Market, asset_ids: list[str]) -> None:
        """Upsert ``market`` into the cache and extend the asset→market map."""
        self._market_cache.upsert(market)
        for asset_id in market.clob_token_ids:
            self._asset_to_market[asset_id] = market.id
            asset_ids.append(asset_id)

    async def _subscribe_in_batches(self, asset_ids: list[str]) -> None:
        """Send subscribe frames in chunks of ``ws_subscribe_batch_size``."""
        batch_size = self._config.ws_subscribe_batch_size
        for start in range(0, len(asset_ids), batch_size):
            await self._ws.subscribe(asset_ids[start : start + batch_size])

    async def _consume_loop(self, sink: AlertSink) -> None:
        """Read messages from the websocket and dispatch trades to handlers."""
        async for msg in self._ws.messages():  # ty:ignore[not-iterable]
            if not isinstance(msg, WsTradeMessage):
                continue
            await self._handle_trade(msg, sink)

    async def _handle_trade(self, msg: WsTradeMessage, sink: AlertSink) -> None:
        """Apply the whale-detection rules and emit an alert when matched."""
        cached = self._lookup_market(msg)
        if cached is None or cached.liquidity_usd is None:
            return
        if cached.liquidity_usd > self._config.small_market_max_liquidity_usd:
            return
        usd = msg.size * msg.price
        if not self._is_big_bet(usd, cached.liquidity_usd):
            return
        seen = await self._get_or_refresh_wallet(msg.taker_proxy)
        if seen is None or seen.first_activity_at is None:
            return
        if not self._is_new_wallet(seen, msg.timestamp):
            return
        await sink.emit(self._build_alert(msg, cached, usd, seen))

    def _lookup_market(self, msg: WsTradeMessage) -> CachedMarket | None:
        """Return the cached market for a trade, or ``None`` if unknown."""
        market_id = self._asset_to_market.get(msg.asset_id)
        if market_id is None:
            return None
        return self._market_cache.get(market_id)

    def _is_big_bet(self, usd: float, liquidity_usd: float) -> bool:
        """Return whether ``usd`` clears both the absolute and pct thresholds."""
        if usd < self._config.big_bet_min_usd:
            return False
        return usd / liquidity_usd >= self._config.big_bet_min_pct_of_liquidity

    async def _get_or_refresh_wallet(self, address: str) -> WalletFirstSeen | None:
        """Return a fresh first-seen row, refetching when missing or stale."""
        seen = self._wallet_first_seen.get(address)
        if seen is not None and int(time.time()) - seen.cached_at <= _WALLET_CACHE_TTL_SECONDS:
            return seen
        first_at = await self._data_client.get_first_activity_timestamp(address)
        activity = await self._data_client.get_activity(address, limit=_ACTIVITY_PROBE_LIMIT)
        self._wallet_first_seen.upsert(address, first_at, len(activity))
        return self._wallet_first_seen.get(address)

    def _is_new_wallet(self, seen: WalletFirstSeen, now_ts: int) -> bool:
        """Return whether a wallet meets the new-account age and trade caps."""
        if seen.first_activity_at is None:
            return False
        age_seconds = now_ts - seen.first_activity_at
        if age_seconds > self._config.new_account_max_age_days * 86400:
            return False
        max_trades = self._config.new_account_max_trades
        return not (seen.total_trades is not None and seen.total_trades > max_trades)

    def _build_alert(
        self,
        msg: WsTradeMessage,
        cached: CachedMarket,
        usd: float,
        seen: WalletFirstSeen,
    ) -> Alert:
        """Construct the ``Alert`` payload for an emitted whale event."""
        liquidity = cached.liquidity_usd
        # liquidity is non-None — _handle_trade rejects None upstream — but assert
        # for the type-checker.
        assert liquidity is not None  # noqa: S101
        pct = usd / liquidity
        # first_activity_at is non-None — _is_new_wallet rejects None upstream.
        assert seen.first_activity_at is not None  # noqa: S101
        age_days = (msg.timestamp - seen.first_activity_at) / 86400
        severity = "high" if usd > _HIGH_SEVERITY_USD or pct > _HIGH_SEVERITY_PCT else "med"
        body = {
            "wallet": msg.taker_proxy,
            "market_title": cached.title,
            "condition_id": msg.condition_id,
            "side": msg.side,
            "size": msg.size,
            "price": msg.price,
            "usd_value": usd,
            "market_liquidity": liquidity,
            "age_days": age_days,
            "total_trades": seen.total_trades,
        }
        title = f"Whale on {cached.title or msg.condition_id}: ${usd:,.0f}"
        return Alert(
            detector="whales",
            alert_key=_alert_key(msg),
            severity=severity,
            title=title,
            body=body,
            created_at=msg.timestamp,
        )


def _alert_key(msg: WsTradeMessage) -> str:
    """Compute an idempotent dedupe key for a whale alert."""
    if msg.transaction_hash:
        return f"whale:{msg.transaction_hash}"
    return f"whale:{msg.condition_id}:{msg.taker_proxy}:{msg.timestamp}"
