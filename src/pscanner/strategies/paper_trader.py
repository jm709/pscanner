"""Smart-money copy-trade paper-trading subscriber.

Subscribes to :class:`AlertSink`. Filters to ``smart_money`` alerts whose
source wallet has positive ``weighted_edge``. Resolves the alerted outcome
to an ``asset_id`` via :class:`MarketCacheRepo` and a fill price via
``market_ticks``. Sizes trades at ``cfg.evaluators.smart_money.position_fraction`` of cost-basis
NAV. Inserts an ``entry`` row into ``paper_trades``.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import TypeIs

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
    TrackedWalletsRepo,
)

_LOG = structlog.get_logger(__name__)

_FILL_PRICE_LO = 0.0
_FILL_PRICE_HI = 1.0


def _size_trade(
    *,
    nav: float,
    fill_price: float,
    cfg: PaperTradingConfig,
) -> tuple[float, float] | None:
    """Return ``(cost_usd, shares)`` or ``None`` if the trade can't be sized.

    Returns ``None`` when the computed cost falls below
    ``min_position_cost_usd`` or when ``fill_price`` is outside ``(0, 1)``.
    """
    if not (_FILL_PRICE_LO < fill_price < _FILL_PRICE_HI):
        return None
    cost = nav * cfg.evaluators.smart_money.position_fraction
    if cost < cfg.min_position_cost_usd:
        return None
    shares = cost / fill_price
    return (cost, shares)


def _is_valid_price(value: object) -> TypeIs[int | float]:
    """Return ``True`` when ``value`` is a numeric fill price in ``(0, 1)``."""
    if not isinstance(value, int | float):
        return False
    return _FILL_PRICE_LO < value < _FILL_PRICE_HI


class PaperTrader:
    """Alert-driven paper-trading subscriber."""

    name = "paper_trader"

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        market_cache: MarketCacheRepo,
        tracked_wallets: TrackedWalletsRepo,
        paper_trades: PaperTradesRepo,
        market_ticks: MarketTicksRepo,
        data_client: DataClient,
        gamma_client: GammaClient,
    ) -> None:
        """Bind dependencies. Subscribers must call :meth:`subscribe` separately.

        Args:
            config: Tuning thresholds (bankroll, fraction, min cost, edge cut).
            market_cache: Read-side cache mapping condition+outcome to asset_id.
            tracked_wallets: Lookup for the source wallet's edge metadata.
            paper_trades: Repo that owns the entry/exit ledger.
            market_ticks: Tick history repo for the entry-price lookup.
            data_client: Polymarket data-API client. Used by the cache-miss
                fallback to discover an unknown market's slug from one of its
                trades.
            gamma_client: Polymarket gamma-API client. Used by the cache-miss
                fallback to fetch the full ``Market`` once a slug is known.
        """
        self._config = config
        self._market_cache = market_cache
        self._tracked_wallets = tracked_wallets
        self._paper_trades = paper_trades
        self._market_ticks = market_ticks
        self._data_client = data_client
        self._gamma_client = gamma_client
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def run(self, sink: AlertSink) -> None:
        """Park forever — this subscriber is alert-driven, not periodic.

        The :class:`Scanner` supervision loop calls ``run`` on every
        registered detector. ``PaperTrader`` has no internal loop of its
        own; entries are spawned from :meth:`handle_alert_sync` in
        response to ``AlertSink`` callbacks. Returning would trigger the
        supervisor's restart logic, so we wait on a sentinel forever.
        """
        del sink  # contract: detectors take a sink; we don't emit
        await asyncio.Event().wait()

    def handle_alert_sync(self, alert: Alert) -> None:
        """:meth:`AlertSink.subscribe` callback. Spawns evaluate as a tracked task."""
        if alert.detector != "smart_money":
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("paper_trader.no_event_loop", alert_key=alert.alert_key)
            return
        task = loop.create_task(self.evaluate(alert))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate(self, alert: Alert) -> None:
        """Run the entry pipeline for one smart-money alert."""
        parsed = self._parse_alert(alert)
        if parsed is None:
            return
        wallet, cond, side = parsed
        if not self._wallet_passes_edge_filter(wallet):
            return
        resolved = await self._resolve_outcome(cond, side)
        if resolved is None:
            return
        asset_id, fill_price = resolved
        nav = self._paper_trades.compute_cost_basis_nav(
            starting_bankroll=self._config.starting_bankroll_usd,
        )
        if nav <= 0:
            _LOG.info(
                "paper_trade.bankroll_exhausted",
                alert_key=alert.alert_key,
                nav=nav,
            )
            return
        sized = _size_trade(nav=nav, fill_price=fill_price, cfg=self._config)
        if sized is None:
            _LOG.debug(
                "paper_trade.size_too_small_or_bad_price",
                alert_key=alert.alert_key,
                nav=nav,
                fill_price=fill_price,
            )
            return
        cost_usd, shares = sized
        self._insert_entry(
            alert=alert,
            wallet=wallet,
            cond=cond,
            asset_id=asset_id,
            side=side,
            shares=shares,
            fill_price=fill_price,
            cost_usd=cost_usd,
            nav=nav,
        )

    def _parse_alert(self, alert: Alert) -> tuple[str, ConditionId, str] | None:
        """Extract ``(wallet, condition_id, side)`` from a smart-money body."""
        body = alert.body if isinstance(alert.body, dict) else {}
        wallet = body.get("wallet")
        condition_id_str = body.get("condition_id")
        side = body.get("side")
        if not (
            isinstance(wallet, str) and isinstance(condition_id_str, str) and isinstance(side, str)
        ):
            _LOG.debug("paper_trader.bad_body", alert_key=alert.alert_key)
            return None
        return (wallet, ConditionId(condition_id_str), side)

    def _insert_entry(
        self,
        *,
        alert: Alert,
        wallet: str,
        cond: ConditionId,
        asset_id: AssetId,
        side: str,
        shares: float,
        fill_price: float,
        cost_usd: float,
        nav: float,
    ) -> None:
        """Persist the ``entry`` row, swallowing duplicate-key collisions."""
        try:
            self._paper_trades.insert_entry(
                triggering_alert_key=alert.alert_key,
                triggering_alert_detector="smart_money",
                rule_variant=None,
                source_wallet=wallet,
                condition_id=cond,
                asset_id=asset_id,
                outcome=side,
                shares=shares,
                fill_price=fill_price,
                cost_usd=cost_usd,
                nav_after_usd=nav,
                ts=int(time.time()),
            )
        except sqlite3.IntegrityError:
            _LOG.debug("paper_trader.duplicate_alert", alert_key=alert.alert_key)
        except Exception:
            _LOG.warning(
                "paper_trader.insert_failed",
                alert_key=alert.alert_key,
                exc_info=True,
            )

    def _wallet_passes_edge_filter(self, wallet: str) -> bool:
        """Skip wallets whose ``weighted_edge`` is None or ≤ ``min_weighted_edge``."""
        tracked = self._tracked_wallets.get(wallet)
        if tracked is None:
            _LOG.debug("paper_trader.no_edge", wallet=wallet)
            return False
        edge = tracked.weighted_edge
        if edge is None or edge <= self._config.evaluators.smart_money.min_weighted_edge:
            _LOG.debug("paper_trader.below_edge", wallet=wallet, edge=edge)
            return False
        return True

    async def _resolve_outcome(
        self,
        condition_id: ConditionId,
        side: str,
    ) -> tuple[AssetId, float] | None:
        """Map ``side`` (outcome name) to ``(asset_id, fill_price)``.

        Returns ``None`` when the market is not cached (and cannot be
        backfilled), the outcome name is not in the cached outcomes, no
        price is available, or the price is outside ``(0, 1)``.
        """
        asset_id = self._market_cache.outcome_to_asset(condition_id, side)
        if asset_id is None:
            if await self._backfill_market_cache(condition_id):
                asset_id = self._market_cache.outcome_to_asset(condition_id, side)
            if asset_id is None:
                _LOG.warning(
                    "paper_trade.outcome_unmappable",
                    condition_id=condition_id,
                    side=side,
                )
                return None
        fill_price = self._lookup_fill_price(condition_id, asset_id)
        if fill_price is None:
            return None
        return (asset_id, fill_price)

    async def _backfill_market_cache(self, condition_id: ConditionId) -> bool:
        """Fetch a market's metadata via gamma and write it to ``market_cache``.

        Two-step sequence: the data-api ``/trades`` endpoint exposes a
        market's slug per trade row, gamma's ``/markets?slug=`` returns the
        full ``Market``. Returns ``True`` on success, ``False`` if either
        step fails or raises.

        Args:
            condition_id: The market's on-chain condition id.

        Returns:
            ``True`` when the cache was successfully populated.
        """
        try:
            slug = await self._data_client.get_market_slug_by_condition_id(
                condition_id,
            )
            if slug is None:
                _LOG.debug("paper_trader.no_slug", condition_id=condition_id)
                return False
            market = await self._gamma_client.get_market_by_slug(slug)
            if market is None:
                _LOG.debug(
                    "paper_trader.no_gamma_market",
                    condition_id=condition_id,
                    slug=slug,
                )
                return False
        except Exception:
            _LOG.warning(
                "paper_trader.backfill_failed",
                condition_id=condition_id,
                exc_info=True,
            )
            return False
        self._market_cache.upsert(market)
        _LOG.info(
            "paper_trader.market_cache_backfilled",
            condition_id=condition_id,
            slug=market.slug,
        )
        return True

    def _lookup_fill_price(
        self,
        condition_id: ConditionId,
        asset_id: AssetId,
    ) -> float | None:
        """Resolve a fill price via the prioritised lookup chain.

        Order:

        1. ``market_ticks.best_ask`` (live orderbook ask).
        2. ``market_ticks.last_trade_price`` (last printed trade).
        3. ``market_cache.outcome_prices[outcome_index]`` (gamma's cached
           last-known quote — populated at backfill time, seconds-stale
           but always available immediately after a cache miss recovery).

        Returns ``None`` when none of the three sources yields a price in
        ``(0, 1)``. The fallback path emits ``paper_trade.fill_price_fallback``
        at INFO level so operators can grep how often live ticks were
        unavailable.
        """
        tick = self._market_ticks.latest_for_asset(asset_id)
        if tick is not None:
            if _is_valid_price(tick.best_ask):
                return float(tick.best_ask)
            if _is_valid_price(tick.last_trade_price):
                return float(tick.last_trade_price)
        fallback_price = self._cached_outcome_price(condition_id, asset_id)
        if fallback_price is not None:
            _LOG.info(
                "paper_trade.fill_price_fallback",
                asset_id=asset_id,
                condition_id=condition_id,
                fallback_price=fallback_price,
            )
            return fallback_price
        _LOG.warning(
            "paper_trade.no_price",
            asset_id=asset_id,
            condition_id=condition_id,
            best_ask=tick.best_ask if tick is not None else None,
            last_trade=tick.last_trade_price if tick is not None else None,
        )
        return None

    def _cached_outcome_price(
        self,
        condition_id: ConditionId,
        asset_id: AssetId,
    ) -> float | None:
        """Return the cached gamma outcome price for ``asset_id``, or None.

        Used as a third-tier fill-price fallback when ``market_ticks`` has
        no row for an asset (e.g. immediately after a cache backfill,
        before the tick stream picks up the new asset). Reads
        ``market_cache``'s ``asset_ids`` / ``outcome_prices`` lists, looks
        up ``asset_id``'s index, and returns ``outcome_prices[index]`` if
        it's a valid fill price (in ``(0, 1)``).

        Returns ``None`` when the market is missing, the asset_id is not
        in the cached asset_ids, the lists have mismatched lengths, or
        the price is not in the valid range.

        Args:
            condition_id: The market's on-chain condition id.
            asset_id: The CLOB token id whose price to look up.
        """
        cached = self._market_cache.get_by_condition_id(condition_id)
        if cached is None:
            return None
        if len(cached.asset_ids) != len(cached.outcome_prices):
            return None
        try:
            idx = cached.asset_ids.index(asset_id)
        except ValueError:
            return None
        price = cached.outcome_prices[idx]
        if not _is_valid_price(price):
            return None
        return float(price)

    async def aclose(self) -> None:
        """Wait for any in-flight evaluation tasks (test helper)."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)


__all__ = ["PaperTrader", "_size_trade"]
