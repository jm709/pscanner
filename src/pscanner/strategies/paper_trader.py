"""Smart-money copy-trade paper-trading subscriber.

Subscribes to :class:`AlertSink`. Filters to ``smart_money`` alerts whose
source wallet has positive ``weighted_edge``. Resolves the alerted outcome
to an ``asset_id`` via :class:`MarketCacheRepo` and a fill price via
``market_ticks``. Sizes trades at ``cfg.position_fraction`` of cost-basis
NAV. Inserts an ``entry`` row into ``paper_trades``.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    MarketCacheRepo,
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
    cost = nav * cfg.position_fraction
    if cost < cfg.min_position_cost_usd:
        return None
    shares = cost / fill_price
    return (cost, shares)


def _is_valid_price(value: object) -> bool:
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
        conn: sqlite3.Connection,
    ) -> None:
        """Bind dependencies. Subscribers must call :meth:`subscribe` separately.

        Args:
            config: Tuning thresholds (bankroll, fraction, min cost, edge cut).
            market_cache: Read-side cache mapping condition+outcome to asset_id.
            tracked_wallets: Lookup for the source wallet's edge metadata.
            paper_trades: Repo that owns the entry/exit ledger.
            conn: SQLite connection (for the ``market_ticks`` price lookup).
        """
        self._config = config
        self._market_cache = market_cache
        self._tracked_wallets = tracked_wallets
        self._paper_trades = paper_trades
        self._conn = conn
        self._pending_tasks: set[asyncio.Task[None]] = set()

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
        resolved = self._resolve_outcome(cond, side)
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
                source_wallet=wallet,
                condition_id=cond,
                asset_id=asset_id,
                outcome=side,
                shares=shares,
                fill_price=fill_price,
                cost_usd=cost_usd,
                nav_after_usd=nav - cost_usd,
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
        if edge is None or edge <= self._config.min_weighted_edge:
            _LOG.debug("paper_trader.below_edge", wallet=wallet, edge=edge)
            return False
        return True

    def _resolve_outcome(
        self,
        condition_id: ConditionId,
        side: str,
    ) -> tuple[AssetId, float] | None:
        """Map ``side`` (outcome name) to ``(asset_id, fill_price)``.

        Returns ``None`` when the market is not cached, the outcome name is
        not in the cached outcomes, no price is available, or the price is
        outside ``(0, 1)``.
        """
        asset_id = self._market_cache.outcome_to_asset(condition_id, side)
        if asset_id is None:
            _LOG.warning(
                "paper_trade.outcome_unmappable",
                condition_id=condition_id,
                side=side,
            )
            return None
        fill_price = self._lookup_fill_price(asset_id)
        if fill_price is None:
            return None
        return (asset_id, fill_price)

    def _lookup_fill_price(self, asset_id: AssetId) -> float | None:
        """Read the latest ``best_ask`` (or ``last_trade_price`` fallback)."""
        row = self._conn.execute(
            """
            SELECT best_ask, last_trade_price FROM market_ticks
             WHERE asset_id = ?
             ORDER BY snapshot_at DESC
             LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        if row is None:
            _LOG.warning("paper_trade.no_price", asset_id=asset_id)
            return None
        best_ask = row["best_ask"]
        last_trade = row["last_trade_price"]
        if _is_valid_price(best_ask):
            return float(best_ask)
        if _is_valid_price(last_trade):
            return float(last_trade)
        _LOG.warning(
            "paper_trade.no_price",
            asset_id=asset_id,
            best_ask=best_ask,
            last_trade=last_trade,
        )
        return None

    async def aclose(self) -> None:
        """Wait for any in-flight evaluation tasks (test helper)."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)


__all__ = ["PaperTrader", "_size_trade"]
