"""Trade collector — DC-1 Wave B (REST-polling rewrite).

Records trades by every watched wallet to ``wallet_trades``. The public
Polymarket WS market channel does not emit per-wallet trade events (those
live on the authenticated ``/ws/user`` channel only), so this collector
polls the public ``/activity`` endpoint on a fixed cadence and inserts
``CONFIRMED`` rows by composite primary key (``transaction_hash``,
``asset_id``, ``side``).

Polling strategy:

* On startup, and every ``poll_interval_seconds`` thereafter, iterate every
  watched wallet sequentially and pull a single page of ``TRADE`` events from
  ``DataClient.get_activity``. Sequential polling respects the data API
  per-host rate limit.
* When a wallet is added to the watchlist mid-run, ``_on_watchlist_add``
  schedules an immediate (off-cycle) poll for that wallet so we capture its
  recent history without waiting for the next cycle.

Persistence:

* Each ``TRADE`` activity entry is converted to a :class:`WalletTrade` and
  inserted via :class:`WalletTradesRepo`. The repo's composite primary key
  handles dedupe across overlapping polls.
* Entries with no ``transactionHash`` are dropped — the dedupe key relies on
  the hash, so persisting them would risk silent duplicates.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.poly.data import DataClient
from pscanner.store.repo import WalletTrade, WalletTradesRepo

_LOG = structlog.get_logger(__name__)


class TradeCollector:
    """Polls the public ``/activity`` endpoint for trades by watched wallets.

    Replaces the earlier WS-driven collector: the public market WS channel
    never emits per-wallet trade events (those are authenticated on
    ``/ws/user`` only), so the collector now uses REST polling against the
    same ``wallet_trades`` schema with ``source="activity_api"``.
    """

    name: str = "trade_collector"

    def __init__(
        self,
        *,
        registry: WatchlistRegistry,
        data_client: DataClient,
        trades_repo: WalletTradesRepo,
        poll_interval_seconds: float = 60.0,
        activity_page_limit: int = 200,
    ) -> None:
        """Wire the collector to the registry, data client, and repo.

        Args:
            registry: In-memory watchlist of wallet addresses to record.
            data_client: REST client used to fetch each wallet's activity feed.
            trades_repo: Append-only repo for ``wallet_trades`` rows.
            poll_interval_seconds: Cadence for full-watchlist polling cycles.
            activity_page_limit: Per-wallet ``/activity`` page size.
        """
        self._registry = registry
        self._data_client = data_client
        self._trades_repo = trades_repo
        self._poll_interval_seconds = poll_interval_seconds
        self._activity_page_limit = activity_page_limit
        self._pending_add_tasks: set[asyncio.Task[int]] = set()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the polling loop until ``stop_event`` is set.

        On each iteration, polls every watched wallet once, then sleeps for
        ``poll_interval_seconds`` (or returns early if the stop event fires).
        Per-iteration exceptions are logged and swallowed so a transient
        upstream hiccup does not kill the loop.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        self._registry.subscribe(self._on_watchlist_add)
        while not stop_event.is_set():
            try:
                await self.poll_all_wallets()
            except Exception:
                _LOG.exception("trades.poll_iteration_failed")
            if await self._wait_or_stop(stop_event, self._poll_interval_seconds):
                return

    async def poll_all_wallets(self) -> int:
        """Poll every watched wallet once; return total new rows inserted.

        Sequential per-wallet to respect rate limits. Per-wallet exceptions
        are caught inside :meth:`_poll_wallet`, so a single failing wallet
        cannot break the broader cycle.

        Returns:
            Total number of newly-inserted ``wallet_trades`` rows across
            every watched wallet.
        """
        watched = sorted(self._registry.addresses())
        total = 0
        for wallet in watched:
            total += await self._poll_wallet(wallet)
        _LOG.info(
            "trades.poll.completed",
            watched_wallets=len(watched),
            inserted=total,
        )
        return total

    async def _poll_wallet(self, address: str) -> int:
        """Poll one wallet's ``/activity`` feed and persist new TRADE rows.

        Args:
            address: 0x-prefixed proxy wallet to poll.

        Returns:
            Number of newly-inserted rows for this wallet (0 on error).
        """
        try:
            events = await self._data_client.get_activity(
                address,
                type="TRADE",
                limit=self._activity_page_limit,
            )
        except Exception:
            _LOG.exception("trades.get_activity_failed", wallet=address)
            return 0
        inserted = 0
        for event in events:
            trade = _build_trade_from_activity(event, wallet=address)
            if trade is None:
                continue
            try:
                if self._trades_repo.insert(trade):
                    inserted += 1
            except Exception:
                _LOG.exception(
                    "trades.insert_failed",
                    wallet=address,
                    tx_hash=trade.transaction_hash,
                )
        return inserted

    def _on_watchlist_add(self, address: str) -> None:
        """Schedule an immediate poll for ``address`` when it joins the watchlist.

        Called from inside :meth:`WatchlistRegistry.add`, so this must be
        non-blocking. We spawn a task and keep a strong reference to it so
        it isn't garbage-collected before completion.
        """
        try:
            task = asyncio.create_task(self._poll_wallet(address))
        except RuntimeError:
            # No running event loop — caller is in a sync context where we
            # cannot schedule the poll. The next periodic cycle will pick
            # this wallet up.
            _LOG.debug("trades.on_watchlist_add.no_loop", wallet=address)
            return
        self._pending_add_tasks.add(task)
        task.add_done_callback(self._pending_add_tasks.discard)

    @staticmethod
    async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
        """Wait up to ``seconds`` for the stop event.

        Returns:
            ``True`` if the stop event was set during the wait, ``False`` if
            the timeout elapsed first.
        """
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True


def _build_trade_from_activity(
    event: dict[str, Any],
    *,
    wallet: str,
) -> WalletTrade | None:
    """Convert one ``/activity`` JSON event to a :class:`WalletTrade`.

    The public activity API returns heterogeneous shapes (TRADE, REWARD,
    SPLIT, MERGE, ...). This builder accepts only ``type == "TRADE"`` rows
    that carry a non-empty transaction hash; everything else is dropped.

    Args:
        event: A single decoded JSON dict from ``DataClient.get_activity``.
        wallet: The address we polled (canonical wallet for this row).

    Returns:
        A populated ``WalletTrade``, or ``None`` if the row is not a usable
        TRADE event.
    """
    if event.get("type") != "TRADE":
        return None
    tx_hash = _coerce_str(event.get("transactionHash") or event.get("tx_hash"))
    if not tx_hash:
        return None
    asset_id = _coerce_str(event.get("asset"))
    side = _coerce_str(event.get("side")).upper()
    condition_id = _coerce_str(event.get("conditionId"))
    size = _coerce_float(event.get("size"))
    price = _coerce_float(event.get("price"))
    if size is None or price is None:
        return None
    usd_value = _coerce_float(event.get("usdcSize"))
    if usd_value is None:
        usd_value = size * price
    timestamp = _coerce_int(event.get("timestamp")) or 0
    return WalletTrade(
        transaction_hash=tx_hash,
        asset_id=asset_id,
        side=side,
        wallet=wallet,
        condition_id=condition_id,
        size=size,
        price=price,
        usd_value=usd_value,
        status="CONFIRMED",
        source="activity_api",
        timestamp=timestamp,
        recorded_at=int(time.time()),
    )


def _coerce_str(value: Any) -> str:
    """Return ``value`` as a stripped string, or ``""`` if missing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` if missing/unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    """Return ``value`` as an int, or ``None`` if missing/unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None
