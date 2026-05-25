"""PaperResolver — periodic detector that books PnL on resolved markets.

Inherits ``PollingDetector``. Each scan: list open positions, check each
position's market in the cache for a definitive ``[1, 0]`` / ``[0, 1]``
outcome split, insert an exit row that books realized PnL.
"""

from __future__ import annotations

from typing import TypeGuard

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.detectors.polling import PollingDetector
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    CachedMarket,
    MarketCacheRepo,
    OpenPaperPosition,
    PaperTradesRepo,
)
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row
from pscanner.util.clock import Clock

_LOG = structlog.get_logger(__name__)

_DEFINITIVE = 1.0
_ZERO = 0.0


def _is_resolved(cached: CachedMarket | None) -> TypeGuard[CachedMarket]:
    """Return True iff ``cached``'s prices are a definitive ``[1, 0]`` split.

    The ``cached.active`` flag is deliberately ignored: gamma reports
    resolved markets as ``active=True closed=True`` with the definitive
    price split, so ``active`` is unreliable as a resolution signal.
    Polymarket prices clamp to the open ``(0, 1)`` interval on tradeable
    markets — only resolved markets ever hit the ``0.0`` / ``1.0``
    boundary — so the price-split check is sufficient on its own.
    """
    if cached is None:
        return False
    prices = cached.outcome_prices
    if len(prices) != len(cached.asset_ids):
        return False
    if sum(prices) != _DEFINITIVE:
        return False
    return _DEFINITIVE in prices and _ZERO in prices


def _check_resolution(
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
) -> AssetId | None:
    """Return the winning ``AssetId`` if the market has resolved, else None.

    A market is considered resolved iff its ``outcome_prices`` is a clean
    ``[1.0, 0.0]`` or ``[0.0, 1.0]`` split — see :func:`_is_resolved` for
    why the ``active`` flag is not consulted.
    """
    cached = market_cache.get_by_condition_id(condition_id)
    if not _is_resolved(cached):
        return None
    for price, asset_id in zip(cached.outcome_prices, cached.asset_ids, strict=True):
        if price == _DEFINITIVE:
            return asset_id
    return None


def _compute_payout(
    *,
    position_asset_id: AssetId,
    winning_asset_id: AssetId,
) -> float:
    """Return ``1.0`` if our outcome won, ``0.0`` otherwise."""
    return _DEFINITIVE if position_asset_id == winning_asset_id else _ZERO


class PaperResolver(PollingDetector):
    """Books exits on open paper positions whose markets have resolved."""

    name = "paper_resolver"

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        market_cache: MarketCacheRepo,
        paper_trades: PaperTradesRepo,
        data_client: DataClient,
        gamma_client: GammaClient,
        clock: Clock | None = None,
    ) -> None:
        """Wire dependencies; see :class:`PollingDetector` for the loop shape.

        Args:
            config: Paper-trading config; supplies the scan interval and the
                starting bankroll used when stamping ``nav_after_usd`` on
                exit rows.
            market_cache: Read/write access to the cached market table. The
                resolver refreshes stale-active rows for open positions
                before checking resolution (#170).
            paper_trades: Read/write repo for ``paper_trades``.
            data_client: Used to resolve a market's slug from its
                ``condition_id`` during refresh.
            gamma_client: Used to fetch a market by slug during refresh.
            clock: Optional injected :class:`Clock`; defaults to a real clock.
        """
        super().__init__(clock=clock)
        self._config = config
        self._market_cache = market_cache
        self._paper_trades = paper_trades
        self._data_client = data_client
        self._gamma_client = gamma_client

    def _interval_seconds(self) -> float:
        return self._config.resolver_scan_interval_seconds

    async def _scan(self, sink: AlertSink) -> None:
        """Refresh stale-active market_cache rows, then book exits.

        Errors on individual positions or refresh calls are logged and
        skipped — one bad row never blocks the rest.
        """
        del sink  # contract: _scan accepts a sink; we don't emit
        open_positions = list(self._paper_trades.list_open_positions())
        await self._refresh_stale_markets(open_positions)
        booked = 0
        for pos in open_positions:
            if self._maybe_book_exit(pos):
                booked += 1
        if booked:
            _LOG.info("paper_resolver.scan_completed", booked=booked)

    async def _refresh_stale_markets(
        self,
        open_positions: list[OpenPaperPosition],
    ) -> None:
        """Refresh ``market_cache`` rows for open positions that aren't yet resolved.

        Skips markets whose cache already shows the definitive
        ``[1, 0]`` / ``[0, 1]`` resolution split — those don't need a
        gamma round-trip because we already know the winner.
        Deduplicates by ``condition_id`` so twin positions on the same
        market only trigger one gamma call per scan. Sequential awaits —
        no ``gather`` — to keep gamma traffic predictable under the
        shared rate limiter.
        """
        seen: set[ConditionId] = set()
        refreshed = 0
        failed = 0
        for pos in open_positions:
            if pos.condition_id in seen:
                continue
            seen.add(pos.condition_id)
            cached = self._market_cache.get_by_condition_id(pos.condition_id)
            if _is_resolved(cached):
                continue
            ok = await refresh_market_cache_row(
                data_client=self._data_client,
                gamma_client=self._gamma_client,
                market_cache=self._market_cache,
                condition_id=pos.condition_id,
            )
            if ok:
                refreshed += 1
            else:
                failed += 1
        if refreshed or failed:
            _LOG.info(
                "paper_resolver.refresh_completed",
                refreshed=refreshed,
                failed=failed,
            )

    def _maybe_book_exit(self, pos: OpenPaperPosition) -> bool:
        """Check resolution for one position; insert exit if resolved.

        Returns True iff an exit row was written.
        """
        try:
            winning = _check_resolution(self._market_cache, pos.condition_id)
            if winning is None:
                return False
            payout_per_share = _compute_payout(
                position_asset_id=pos.asset_id,
                winning_asset_id=winning,
            )
            proceeds = pos.shares * payout_per_share
            nav_before = self._paper_trades.compute_cost_basis_nav(
                starting_bankroll=self._config.starting_bankroll_usd,
            )
            self._paper_trades.insert_exit(
                parent_trade_id=pos.trade_id,
                condition_id=pos.condition_id,
                asset_id=pos.asset_id,
                outcome=pos.outcome,
                shares=pos.shares,
                fill_price=payout_per_share,
                cost_usd=proceeds,
                nav_after_usd=nav_before + (proceeds - pos.cost_usd),
                ts=int(self._clock.now()),
            )
        except Exception:
            _LOG.warning(
                "paper_resolver.insert_failed",
                trade_id=pos.trade_id,
                exc_info=True,
            )
            return False
        return True


__all__ = ["PaperResolver", "_check_resolution", "_compute_payout"]
