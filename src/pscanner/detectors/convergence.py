"""Convergence detector — multiple smart wallets enter the same condition.

DC-1.8.B. The detector is driven by :class:`TradeCollector` callbacks: every
freshly-inserted ``wallet_trades`` row triggers
:meth:`TradeDrivenDetector.handle_trade_sync`, which spawns
:meth:`ConvergenceDetector.evaluate` on the running event loop.

Logic:

1. Look up the trade's condition in ``market_cache`` to recover the parent
   ``event_slug``.
2. Resolve the event's tag list via ``event_tag_cache``.
3. Categorise the event (``thesis`` / ``sports`` / ``esports``).
4. Pull the per-category smart-wallet roster (``tracked_wallet_categories``),
   filtered by the per-category ``mean_edge`` threshold from
   :class:`SmartMoneyConfig`.
5. Query the recent distinct wallets that traded this condition within the
   category's configured window.
6. The intersection is the "convergent" set — emit an alert when its size
   reaches ``convergence_min_wallets``.

Alerts dedupe per ``condition_id`` per UTC day via the alert key.
"""

from __future__ import annotations

import time

import structlog

from pscanner.alerts.models import Alert, Severity
from pscanner.categories import Category, categorize_tags, settings_for
from pscanner.config import ConvergenceConfig, SmartMoneyConfig
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import (
    CachedMarket,
    EventTagCacheRepo,
    MarketCacheRepo,
    TrackedWalletCategoriesRepo,
    WalletTrade,
    WalletTradesRepo,
)

_LOG = structlog.get_logger(__name__)
_HIGH_SEVERITY_WALLET_COUNT = 3
_MIN_EXCESS_PNL_FOR_ROSTER = 0.0


class ConvergenceDetector(TradeDrivenDetector):
    """Detector that emits alerts when smart wallets cluster on a condition."""

    name = "convergence"

    def __init__(
        self,
        *,
        config: ConvergenceConfig,
        trades_repo: WalletTradesRepo,
        category_repo: TrackedWalletCategoriesRepo,
        market_cache: MarketCacheRepo,
        event_tag_cache: EventTagCacheRepo,
        smart_money_config: SmartMoneyConfig,
    ) -> None:
        """Build the detector with its config and persistence dependencies.

        Args:
            config: Convergence-specific thresholds and per-category windows.
            trades_repo: Read access to ``wallet_trades`` rows.
            category_repo: Read access to per-category smart-wallet roster.
            market_cache: Used to recover ``event_slug`` from a trade's
                ``condition_id``.
            event_tag_cache: Used to categorise events by their tag list.
            smart_money_config: Source of ``category_min_edge`` thresholds
                applied when reading the per-category roster.
        """
        super().__init__()
        self._config = config
        self._trades_repo = trades_repo
        self._category_repo = category_repo
        self._market_cache = market_cache
        self._event_tag_cache = event_tag_cache
        self._smart_config = smart_money_config

    async def evaluate(self, trade: WalletTrade) -> None:
        """Apply the convergence filter to a freshly-inserted trade.

        Skips silently when:

        * the sink has not been wired yet,
        * the market cache does not know this ``condition_id``,
        * the cached row has no ``event_slug``,
        * the event-tag cache has no tags for the slug, or
        * the convergent-wallet set is below the configured minimum.

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """
        if self._sink is None:
            _LOG.warning("convergence.no_sink", tx=trade.transaction_hash)
            return
        cached = self._market_cache.get_by_condition_id(trade.condition_id)
        if cached is None or cached.event_slug is None:
            return
        tags = self._event_tag_cache.get(cached.event_slug)
        if tags is None:
            return
        category = categorize_tags(tags)
        window = self._window_seconds_for(category)
        smart_wallets = self._smart_wallets_in_category(category)
        if not smart_wallets:
            return
        since = trade.timestamp - window
        recent = self._trades_repo.distinct_wallets_for_condition(
            trade.condition_id,
            since=since,
        )
        convergent = recent & smart_wallets
        if len(convergent) < self._config.convergence_min_wallets:
            return
        alert = self._build_alert(trade, cached, category, convergent)
        await self._sink.emit(alert)

    def _window_seconds_for(self, category: Category) -> int:
        """Return the convergence window (seconds) for ``category``.

        Uses :class:`ConvergenceConfig.window_seconds_overrides` when set,
        otherwise falls back to the taxonomy default in
        :data:`pscanner.categories.DEFAULT_TAXONOMY`.
        """
        overrides = self._config.window_seconds_overrides
        if overrides is not None and category in overrides:
            return overrides[category]
        return settings_for(category).convergence_window_seconds

    def _min_edge_for(self, category: Category) -> float:
        """Return the smart-money min-edge threshold for ``category``."""
        overrides = self._smart_config.category_min_edge
        if overrides is not None and category.value in overrides:
            return overrides[category.value]
        return settings_for(category).min_edge

    def _smart_wallets_in_category(self, category: Category) -> set[str]:
        """Return the wallet addresses qualifying as smart in ``category``."""
        rows = self._category_repo.list_by_category(
            category.value,
            min_edge=self._min_edge_for(category),
            min_excess_pnl_usd=_MIN_EXCESS_PNL_FOR_ROSTER,
            min_resolved=self._smart_config.min_resolved_positions,
        )
        return {row.wallet for row in rows}

    def _build_alert(
        self,
        trade: WalletTrade,
        cached: CachedMarket,
        category: Category,
        convergent: set[str],
    ) -> Alert:
        """Construct the convergence :class:`Alert` payload."""
        now = int(time.time())
        day = time.strftime("%Y%m%d", time.gmtime(now))
        alert_key = f"convergence:{trade.condition_id}:{day}"
        wallets_sorted = sorted(convergent)
        window = self._window_seconds_for(category)
        total_usd = self._sum_recent_usd(
            trade.condition_id,
            wallets_sorted,
            since=trade.timestamp - window,
        )
        severity: Severity = "high" if len(convergent) >= _HIGH_SEVERITY_WALLET_COUNT else "med"
        title = (
            f"convergence on {cached.title or trade.condition_id}: "
            f"{len(convergent)} {category.value} wallets"
        )
        body = {
            "condition_id": trade.condition_id,
            "market_title": cached.title,
            "event_slug": cached.event_slug,
            "category": category.value,
            "convergent_wallets": wallets_sorted,
            "convergent_count": len(wallets_sorted),
            "window_seconds": window,
            "total_usd_value": total_usd,
        }
        return Alert(
            detector="convergence",
            alert_key=alert_key,
            severity=severity,
            title=title,
            body=body,
            created_at=now,
        )

    def _sum_recent_usd(
        self,
        condition_id: ConditionId,
        wallets: list[str],
        *,
        since: int,
    ) -> float:
        """Sum ``usd_value`` across recent trades by ``wallets`` on ``condition_id``.

        Reads via :meth:`WalletTradesRepo.recent_for_wallet` per-wallet (small
        N — the convergent set is by design tiny). Skips trades on other
        conditions and trades older than ``since``.
        """
        total = 0.0
        for wallet in wallets:
            rows = self._trades_repo.recent_for_wallet(wallet, limit=200)
            for row in rows:
                if row.condition_id != condition_id:
                    continue
                if row.timestamp < since:
                    continue
                total += row.usd_value
        return total


__all__ = ["ConvergenceDetector"]
