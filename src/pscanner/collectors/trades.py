"""Trade collector — DC-1 Wave A contract stub.

Records every CONFIRMED trade by a watched wallet to ``wallet_trades``. The
collector owns its own websocket connection (separate from the whales
detector) to keep its subscription set focused on the watchlist's open
positions; Wave B fills in the implementation.
"""

from __future__ import annotations

import asyncio

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.store.repo import WalletTradesRepo


class TradeCollector:
    """Records every CONFIRMED trade by a watched wallet to wallet_trades.

    Owns its own MarketWebSocket connection (separate from the whales
    detector). Subscribes to the union of asset_ids across watched wallets'
    open positions; refreshes that subscription set when the watchlist
    changes or every ``subscription_refresh_seconds``.
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
        """Wire the collector to the registry, data client, websocket, and repo."""
        # wave b: store deps + subscribe to registry change events.
        raise NotImplementedError("DC-1 Wave B: trades")

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the trade-collection loop until ``stop_event`` is set."""
        raise NotImplementedError("DC-1 Wave B: trades")
