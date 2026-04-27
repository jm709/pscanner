"""PaperResolver — periodic detector that books PnL on resolved markets.

Inherits ``PollingDetector``. Each scan: list open positions, check each
position's market in the cache for a definitive ``[1, 0]`` / ``[0, 1]``
outcome split, insert an exit row that books realized PnL.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.detectors.polling import PollingDetector
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    MarketCacheRepo,
    OpenPaperPosition,
    PaperTradesRepo,
)
from pscanner.util.clock import Clock

_LOG = structlog.get_logger(__name__)

_DEFINITIVE = 1.0
_ZERO = 0.0


def _check_resolution(
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
) -> AssetId | None:
    """Return the winning ``AssetId`` if the market has resolved, else None.

    A market is considered resolved when ``active=False`` AND its
    ``outcome_prices`` is a clean ``[1.0, 0.0]`` or ``[0.0, 1.0]`` split.
    """
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None or cached.active:
        return None
    prices = cached.outcome_prices
    if len(prices) != len(cached.asset_ids):
        return None
    for price, asset_id in zip(prices, cached.asset_ids, strict=True):
        if price == _DEFINITIVE and sum(prices) == _DEFINITIVE:
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
        clock: Clock | None = None,
    ) -> None:
        """Wire dependencies; see :class:`PollingDetector` for the loop shape.

        Args:
            config: Paper-trading config; supplies the scan interval and the
                starting bankroll used when stamping ``nav_after_usd`` on
                exit rows.
            market_cache: Read-only access to the cached market table.
            paper_trades: Read/write repo for ``paper_trades``.
            clock: Optional injected :class:`Clock`; defaults to a real clock.
        """
        super().__init__(clock=clock)
        self._config = config
        self._market_cache = market_cache
        self._paper_trades = paper_trades

    def _interval_seconds(self) -> float:
        return self._config.resolver_scan_interval_seconds

    async def _scan(self, sink: AlertSink) -> None:
        """Walk open positions; insert exit rows for any that resolved.

        Errors on individual positions are logged and skipped — one bad row
        never blocks the rest.
        """
        del sink  # contract: _scan accepts a sink; we don't emit
        booked = 0
        for pos in self._paper_trades.list_open_positions():
            if self._maybe_book_exit(pos):
                booked += 1
        if booked:
            _LOG.info("paper_resolver.scan_completed", booked=booked)

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
