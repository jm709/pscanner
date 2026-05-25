"""Refresh one ``market_cache`` row from gamma via the 2-hop slug lookup.

Used by ``PaperResolver`` (to keep cache rows truthful for newly-resolved
markets) and by ``PaperTrader._backfill_market_cache`` (its original
caller, kept on the same helper for a single source of truth).
"""

from __future__ import annotations

import structlog

from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import MarketCacheRepo

_LOG = structlog.get_logger(__name__)


async def refresh_market_cache_row(
    *,
    data_client: DataClient,
    gamma_client: GammaClient,
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
) -> bool:
    """Fetch one market via the slug→gamma 2-hop and upsert into market_cache.

    The 2-hop sequence is the existing pattern from
    ``paper_trader._backfill_market_cache``: data-api ``/trades`` exposes a
    market's slug per trade row, gamma ``/markets?slug=`` returns the full
    ``Market``. ``gamma.get_market_by_slug`` internally passes
    ``closed=true`` so the lookup succeeds for both active and resolved
    markets.

    Args:
        data_client: For ``get_market_slug_by_condition_id``.
        gamma_client: For ``get_market_by_slug``.
        market_cache: Where the resolved ``Market`` is upserted.
        condition_id: The on-chain market identifier to refresh.

    Returns:
        ``True`` iff a row was successfully upserted. ``False`` on slug
        miss, gamma miss, or any swallowed exception.
    """
    try:
        slug = await data_client.get_market_slug_by_condition_id(condition_id)
        if slug is None:
            _LOG.debug("market_cache.refresh.no_slug", condition_id=condition_id)
            return False
        market = await gamma_client.get_market_by_slug(slug)
        if market is None:
            _LOG.debug(
                "market_cache.refresh.no_gamma_market",
                condition_id=condition_id,
                slug=slug,
            )
            return False
    except Exception:
        _LOG.warning(
            "market_cache.refresh.failed",
            condition_id=condition_id,
            exc_info=True,
        )
        return False
    market_cache.upsert(market)
    _LOG.info(
        "market_cache.refresh.ok",
        condition_id=condition_id,
        slug=market.slug,
        active=market.active,
    )
    return True


__all__ = ["refresh_market_cache_row"]
