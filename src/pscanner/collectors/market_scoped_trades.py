"""Market-scoped trade collector for the gate-model loop (#79).

Enumerates open events matching ``accepted_categories`` AND polls each
event's markets whose ``volume_24h_usd >= min_volume_24h_usd``. Dispatches
each new trade through the subscribe-callback bus so
:class:`GateModelDetector` can score it.

Per the v1.0 scope (esports-only), ~tens of markets at <data_rpm=50, so
the polling budget comfortably covers the working set at 60s freshness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from pscanner.categories import categorize_event
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import WalletTrade
from pscanner.util.clock import Clock, RealClock

if TYPE_CHECKING:
    from pscanner.config import GateModelMarketFilterConfig
    from pscanner.poly.data import DataClient
    from pscanner.poly.gamma import GammaClient

_LOG = structlog.get_logger(__name__)
_FAR_FUTURE_TS = 2_000_000_000  # ~2033; well past any current trade timestamp


class MarketScopedTradeCollector:
    """Polls top-volume open markets in accepted categories and fans out new trades."""

    name = "market_scoped_trades"

    def __init__(
        self,
        *,
        config: GateModelMarketFilterConfig,
        gamma: GammaClient,
        data_client: DataClient,
    ) -> None:
        """Initialize the collector with configuration and API clients."""
        self._config = config
        self._gamma = gamma
        self._data_client = data_client
        self._markets: list[str] = []
        self._callbacks: list[Callable[[WalletTrade], None]] = []
        self._last_seen_ts: dict[str, int] = {}

    def subscribe_new_trade(self, callback: Callable[[WalletTrade], None]) -> None:
        """Register a callback fired on every newly observed trade."""
        self._callbacks.append(callback)

    async def refresh_market_set(self) -> list[str]:
        """Enumerate events, filter by category + volume, return top-N condition_ids."""
        accepted = set(self._config.accepted_categories)
        floor = self._config.min_volume_24h_usd
        candidates: list[tuple[float, str]] = []
        async for event in self._gamma.iter_events():
            category = categorize_event(event).value
            if category not in accepted:
                continue
            for market in event.markets:
                cond_id = market.condition_id
                if cond_id is None:
                    continue
                volume = float(market.volume or 0.0)
                if volume < floor:
                    continue
                candidates.append((volume, str(cond_id)))
        candidates.sort(reverse=True)
        selected = [cid for _, cid in candidates[: self._config.max_markets]]
        self._markets = selected
        return selected

    async def poll_once(self) -> int:
        """Poll all markets in the working set; return total trades dispatched."""
        n = 0
        for cid in self._markets:
            since_ts = self._last_seen_ts.get(cid, 0)
            try:
                rows = await self._data_client.get_market_trades(
                    cid, since_ts=since_ts, until_ts=_FAR_FUTURE_TS
                )
            except Exception:
                _LOG.warning(
                    "market_scoped.poll_failed",
                    condition_id=cid,
                    exc_info=True,
                )
                continue
            for row in rows:
                trade = self._row_to_wallet_trade(row)
                if trade is None:
                    continue
                self._dispatch(trade)
                if trade.timestamp > self._last_seen_ts.get(cid, 0):
                    self._last_seen_ts[cid] = trade.timestamp
                n += 1
        return n

    def _row_to_wallet_trade(self, row: dict[str, Any]) -> WalletTrade | None:
        try:
            tx_hash = str(row["tx_hash"])
            asset_id = str(row["asset_id"])
            bs = str(row["bs"]).upper()
            wallet = str(row["wallet_address"])
            condition_id = str(row["condition_id"])
            price = float(row["price"])
            size = float(row["size"])
            notional = float(row.get("notional_usd", price * size))
            timestamp = int(row["timestamp"])
        except (KeyError, TypeError, ValueError):
            _LOG.warning("market_scoped.bad_row", row_keys=list(row.keys()))
            return None
        return WalletTrade(
            transaction_hash=tx_hash,
            asset_id=AssetId(asset_id),
            side=bs,
            wallet=wallet,
            condition_id=ConditionId(condition_id),
            size=size,
            price=price,
            usd_value=notional,
            status="filled",
            source="market_scoped",
            timestamp=timestamp,
            recorded_at=timestamp,
        )

    def _dispatch(self, trade: WalletTrade) -> None:
        for callback in self._callbacks:
            try:
                callback(trade)
            except Exception:
                _LOG.exception(
                    "market_scoped.callback_failed",
                    tx=trade.transaction_hash,
                )

    async def run(self, *, clock: Clock | None = None) -> None:
        """Long-running loop: refresh market set + poll on cadence."""
        clk = clock or RealClock()
        while True:
            await self.refresh_market_set()
            await self.poll_once()
            await clk.sleep(self._config.poll_interval_seconds)
