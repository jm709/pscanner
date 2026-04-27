"""Multi-signal paper-trading subscriber.

Subscribes to :class:`AlertSink`. Walks a list of :class:`SignalEvaluator`
instances; the first one whose ``accepts(alert)`` returns ``True`` parses
the alert into one or more :class:`ParsedSignal` instances. Each signal is
independently quality-gated, resolved to an ``asset_id`` + fill price,
sized at constant ``starting_bankroll_usd * position_fraction``, and
booked as a ``paper_trades`` entry row.
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
)
from pscanner.strategies.evaluators import ParsedSignal, SignalEvaluator

_LOG = structlog.get_logger(__name__)

_FILL_PRICE_LO = 0.0
_FILL_PRICE_HI = 1.0


def _is_valid_price(value: object) -> TypeIs[int | float]:
    """Return ``True`` when ``value`` is a numeric fill price in ``(0, 1)``."""
    if not isinstance(value, int | float):
        return False
    return _FILL_PRICE_LO < value < _FILL_PRICE_HI


def _size_valid(cost: float, fill_price: float, *, min_cost: float) -> bool:
    """Reject sizes below the floor or fill prices outside ``(0, 1)``."""
    if cost < min_cost:
        return False
    return _FILL_PRICE_LO < fill_price < _FILL_PRICE_HI


class PaperTrader:
    """Alert-driven multi-signal paper-trading subscriber."""

    name = "paper_trader"

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        evaluators: list[SignalEvaluator],
        market_cache: MarketCacheRepo,
        paper_trades: PaperTradesRepo,
        market_ticks: MarketTicksRepo,
        data_client: DataClient,
        gamma_client: GammaClient,
    ) -> None:
        """Bind dependencies. Subscribers must call :meth:`subscribe` separately.

        Args:
            config: Bankroll + min-cost thresholds (per-source tunables live
                under each evaluator's own config).
            evaluators: Per-detector :class:`SignalEvaluator` instances. The
                first one whose ``accepts`` returns ``True`` for an alert
                runs the parse → quality → size pipeline.
            market_cache: Read-side cache mapping ``(condition_id, outcome)``
                to ``asset_id``.
            paper_trades: Repo that owns the entry/exit ledger.
            market_ticks: Tick history repo for the entry-price lookup.
            data_client: Polymarket data-API client. Used by the cache-miss
                fallback to discover an unknown market's slug from one of its
                trades.
            gamma_client: Polymarket gamma-API client. Used by the cache-miss
                fallback to fetch the full ``Market`` once a slug is known.
        """
        self._config = config
        self._evaluators = evaluators
        self._market_cache = market_cache
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("paper_trader.no_event_loop", alert_key=alert.alert_key)
            return
        task = loop.create_task(self.evaluate(alert))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate(self, alert: Alert) -> None:
        """Run the evaluator pipeline for one alert.

        Walks ``self._evaluators`` in order and runs the first one whose
        ``accepts(alert)`` returns True. Assumes evaluator accept-sets are
        disjoint; if they overlap, list ordering is load-bearing.
        """
        for evaluator in self._evaluators:
            if not evaluator.accepts(alert):
                continue
            try:
                await self._run_pipeline(evaluator, alert)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.warning(
                    "paper_trader.evaluator_failed",
                    detector=alert.detector,
                    evaluator=type(evaluator).__name__,
                    alert_key=alert.alert_key,
                    exc_info=True,
                )
            return
        _LOG.debug(
            "paper_trader.no_evaluator",
            detector=alert.detector,
            alert_key=alert.alert_key,
        )

    async def _run_pipeline(
        self,
        evaluator: SignalEvaluator,
        alert: Alert,
    ) -> None:
        """Parse, quality-gate, resolve, size, and insert each ParsedSignal."""
        parsed_list = evaluator.parse(alert)
        if not parsed_list:
            return
        # bankroll feeds evaluator sizing (constant per run); nav is for the
        # ledger row only and may be negative if losses outpace wins.
        bankroll = self._config.starting_bankroll_usd
        nav = self._paper_trades.compute_cost_basis_nav(
            starting_bankroll=bankroll,
        )
        for parsed in parsed_list:
            if not evaluator.quality_passes(parsed):
                continue
            resolved = await self._resolve_outcome(parsed.condition_id, parsed.side)
            if resolved is None:
                continue
            asset_id, fill_price = resolved
            cost = evaluator.size(bankroll, parsed)
            if not _size_valid(
                cost,
                fill_price,
                min_cost=self._config.min_position_cost_usd,
            ):
                _LOG.debug(
                    "paper_trade.size_too_small_or_bad_price",
                    alert_key=alert.alert_key,
                    cost=cost,
                    fill_price=fill_price,
                )
                continue
            shares = cost / fill_price
            self._insert_entry(
                alert=alert,
                parsed=parsed,
                asset_id=asset_id,
                shares=shares,
                fill_price=fill_price,
                cost_usd=cost,
                nav=nav,
            )

    def _insert_entry(
        self,
        *,
        alert: Alert,
        parsed: ParsedSignal,
        asset_id: AssetId,
        shares: float,
        fill_price: float,
        cost_usd: float,
        nav: float,
    ) -> None:
        """Persist the ``entry`` row, swallowing duplicate-key collisions."""
        wallet = parsed.metadata.get("wallet")
        source_wallet = wallet if isinstance(wallet, str) else None
        try:
            self._paper_trades.insert_entry(
                triggering_alert_key=alert.alert_key,
                triggering_alert_detector=alert.detector,
                rule_variant=parsed.rule_variant,
                source_wallet=source_wallet,
                condition_id=parsed.condition_id,
                asset_id=asset_id,
                outcome=parsed.side,
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


__all__ = ["PaperTrader"]
