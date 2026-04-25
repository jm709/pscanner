"""Market tick collector — append-only price snapshots from the WS orderbook.

Wave 1 contract; Wave 2 fills in the implementation.
"""

from __future__ import annotations

import asyncio

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import TicksConfig
from pscanner.poly.clob_ws import MarketWebSocket
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.store.repo import MarketTicksRepo


class MarketTickCollector:
    """Maintains an in-memory orderbook per subscribed asset and writes ticks.

    Owns its own ``MarketWebSocket`` connection (separate from any other WS
    consumer). Subscription scope = (assets held by watched wallets) U (active
    markets above ``tick_volume_floor_usd``), capped at ``max_assets``.

    Wave 1 stores the dependencies; Wave 2 implements ``run`` and the public
    helpers below.
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
        """Build the collector. Wave 2 fills in the body."""
        self._config = config
        self._ws = ws
        self._gamma = gamma_client
        self._data = data_client
        self._registry = registry
        self._repo = ticks_repo

    async def run(self, stop_event: asyncio.Event) -> None:
        """Connect WS, drive subscription/consume/snapshot loops until stopped."""
        raise NotImplementedError("DC-4 Wave 2: ticks")

    async def snapshot_once(self) -> int:
        """Walk in-memory orderbook state, write one row per asset.

        Returns:
            Number of rows newly inserted. Used by ``run_once`` and by smoke
            verification.
        """
        raise NotImplementedError("DC-4 Wave 2: ticks")

    def get_recent_mids(
        self,
        asset_id: str,
        *,
        window_seconds: int,
    ) -> list[tuple[int, float]]:
        """Return ``(snapshot_at, mid_price)`` pairs within the trailing window.

        FROZEN API: the velocity detector calls this. Wave 2 reads from
        in-memory state OR delegates to ``MarketTicksRepo.recent_mids_in_window``
        — caller-agnostic. Order ascending by ``snapshot_at``.

        Args:
            asset_id: CLOB token id.
            window_seconds: Inclusive trailing window length.

        Returns:
            Pairs ordered by ``snapshot_at`` ascending.
        """
        raise NotImplementedError("DC-4 Wave 2: ticks")

    def subscribed_asset_ids(self) -> set[str]:
        """Return a copy of the currently-subscribed asset id set.

        Useful for the velocity detector to know which assets to poll.
        """
        raise NotImplementedError("DC-4 Wave 2: ticks")
