"""Whale detector — newly-active wallet places an outsized bet on a tiny market.

DC-1.5 rewires the detector to consume ``wallet_trades`` rows produced by the
trade collector (REST polling against ``/activity``) instead of the public
CLOB websocket, which never carried the per-wallet trade events the detector
needs. The detector now fires when:

* the market's liquidity is below ``small_market_max_liquidity_usd``,
* the trade USD notional is above ``big_bet_min_usd`` *and* above
  ``big_bet_min_pct_of_liquidity`` of the market's liquidity, and
* the taker wallet was first seen less than ``new_account_max_age_days`` ago
  with fewer than ``new_account_max_trades`` total trades.

Wallet age and total-trade counts are cached in ``wallet_first_seen`` for 24h
to keep data-API request volume bounded. The :meth:`run` loop only refreshes
the market metadata cache used for the liquidity filter; trade evaluation is
driven by :meth:`evaluate`, dispatched via the inherited
:meth:`TradeDrivenDetector.handle_trade_sync` and registered with the trade
collector via ``TradeCollector.subscribe_new_trade``.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import WhalesConfig
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import (
    CachedMarket,
    MarketCacheRepo,
    WalletFirstSeen,
    WalletFirstSeenRepo,
    WalletTrade,
)
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)
_WALLET_CACHE_TTL_SECONDS = 86400
_HIGH_SEVERITY_USD = 10000.0
_HIGH_SEVERITY_PCT = 0.20
_ACTIVITY_PROBE_LIMIT = 200


class WhalesDetector(TradeDrivenDetector):
    """Detector that emits alerts for big trades by new wallets on tiny markets."""

    name = "whales"

    def __init__(
        self,
        *,
        config: WhalesConfig,
        gamma_client: GammaClient,
        data_client: DataClient,
        market_cache: MarketCacheRepo,
        wallet_first_seen: WalletFirstSeenRepo,
        clock: Clock | None = None,
    ) -> None:
        """Build the detector with its config and external dependencies.

        Args:
            config: Threshold settings from ``WhalesConfig``.
            gamma_client: Source of the active markets catalogue, used to
                periodically refresh ``market_cache`` for liquidity lookups.
            data_client: Source of wallet activity for age lookups.
            market_cache: Persistent cache of market metadata.
            wallet_first_seen: Persistent cache of wallet first-activity rows.
            clock: Injectable :class:`Clock`. Defaults to :class:`RealClock`
                so production wiring needs no changes.
        """
        super().__init__()
        self._config = config
        self._gamma_client = gamma_client
        self._data_client = data_client
        self._market_cache = market_cache
        self._wallet_first_seen = wallet_first_seen
        self._condition_to_market: dict[ConditionId, CachedMarket] = {}
        self._clock: Clock = clock if clock is not None else RealClock()

    async def run(self, sink: AlertSink) -> None:
        """Long-running task: periodically refresh the market cache.

        Trade alerts are NOT driven from this loop. They fire via
        :meth:`evaluate` invoked by the trade collector callback. This loop
        only keeps the ``market_cache`` fresh so liquidity lookups in
        :meth:`evaluate` are accurate.

        Args:
            sink: Alert sink used by :meth:`evaluate` for emission.
        """
        if self._sink is None:
            self._sink = sink
        while True:
            try:
                await self._refresh_market_cache()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("whales.refresh_failed")
            await self._clock.sleep(self._config.ws_resubscribe_interval_seconds)

    async def _refresh_market_cache(self) -> None:
        """Page the active markets, upsert into market_cache, rebuild condition map."""
        fresh: dict[ConditionId, CachedMarket] = {}
        count = 0
        async for market in self._gamma_client.iter_markets(active=True, closed=False):
            if not market.enable_order_book:
                continue
            if (market.volume or 0.0) < self._config.subscription_min_volume_usd:
                continue
            if count >= self._config.subscription_max_markets:
                break
            self._market_cache.upsert(market)
            cached = self._market_cache.get(market.id)
            if cached is not None and market.condition_id:
                fresh[market.condition_id] = cached
                count += 1
        self._condition_to_market = fresh
        _LOG.info("whales.market_cache.refreshed", markets=count)

    async def evaluate(self, trade: WalletTrade) -> None:
        """Apply the new+small+big filter to a freshly-inserted trade.

        Wired via :meth:`TradeCollector.subscribe_new_trade` (through
        :meth:`TradeDrivenDetector.handle_trade_sync`). Fires only when the
        trade was a new insert (callback contract). Skips silently when the
        sink has not been wired yet (defensive — Scanner sets it at
        construction).

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """
        if self._sink is None:
            _LOG.warning("whales.no_sink", tx=trade.transaction_hash)
            return
        cached = self._condition_to_market.get(trade.condition_id)
        if cached is None or cached.liquidity_usd is None:
            return
        if cached.liquidity_usd > self._config.small_market_max_liquidity_usd:
            return
        usd = trade.usd_value
        if not self._is_big_bet(usd, cached.liquidity_usd):
            return
        seen = await self._get_or_refresh_wallet(trade.wallet)
        if seen is None or seen.first_activity_at is None:
            return
        if not self._is_new_wallet(seen, trade.timestamp):
            return
        await self._sink.emit(self._build_alert(trade, cached, usd, seen))

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
        trade: WalletTrade,
        cached: CachedMarket,
        usd: float,
        seen: WalletFirstSeen,
    ) -> Alert:
        """Construct the ``Alert`` payload for an emitted whale event."""
        liquidity = cached.liquidity_usd
        # liquidity is non-None — evaluate rejects None upstream — but assert
        # for the type-checker.
        assert liquidity is not None  # noqa: S101
        pct = usd / liquidity
        # first_activity_at is non-None — _is_new_wallet rejects None upstream.
        assert seen.first_activity_at is not None  # noqa: S101
        age_days = (trade.timestamp - seen.first_activity_at) / 86400
        severity = "high" if usd > _HIGH_SEVERITY_USD or pct > _HIGH_SEVERITY_PCT else "med"
        body = {
            "wallet": trade.wallet,
            "market_title": cached.title,
            "condition_id": trade.condition_id,
            "side": trade.side,
            "size": trade.size,
            "price": trade.price,
            "usd_value": usd,
            "market_liquidity": liquidity,
            "age_days": age_days,
            "total_trades": seen.total_trades,
        }
        title = f"Whale on {cached.title or trade.condition_id}: ${usd:,.0f}"
        return Alert(
            detector="whales",
            alert_key=_alert_key(trade),
            severity=severity,
            title=title,
            body=body,
            created_at=trade.timestamp,
        )


def _alert_key(trade: WalletTrade) -> str:
    """Compute an idempotent dedupe key for a whale alert."""
    if trade.transaction_hash:
        return f"whale:{trade.transaction_hash}"
    return f"whale:{trade.condition_id}:{trade.wallet}:{trade.timestamp}"
