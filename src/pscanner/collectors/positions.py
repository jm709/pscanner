"""Position-snapshot collector — append-only history of watched-wallet positions.

Records every watched wallet's open positions to ``wallet_positions_history``
on a fixed cadence. Each cycle stamps a single ``snapshot_at`` timestamp so
all rows from one poll share a clock and can be reconstructed as the wallet's
position vector at that instant. The repo deduplicates on
``(wallet, condition_id, outcome, snapshot_at)``, so two cycles in the same
second collapse to one row — that is the intended idempotency behaviour.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.poly.data import DataClient
from pscanner.poly.models import Position
from pscanner.store.repo import WalletPositionsHistoryRepo, WalletPositionsHistoryRow

_LOG = structlog.get_logger(__name__)


class PositionCollector:
    """Periodically snapshots every watched wallet's open positions.

    Iterates :meth:`WatchlistRegistry.addresses` each cycle, calls
    :meth:`DataClient.get_positions` for every address sequentially (rate-
    limit safety), and appends rows to :class:`WalletPositionsHistoryRepo`.
    Per-wallet exceptions are caught so a single bad poll does not break the
    cycle, and per-iteration exceptions are caught so a transient hiccup does
    not kill the loop.
    """

    name: str = "position_collector"

    def __init__(
        self,
        *,
        registry: WatchlistRegistry,
        data_client: DataClient,
        positions_repo: WalletPositionsHistoryRepo,
        snapshot_interval_seconds: float = 300.0,
    ) -> None:
        """Build the collector.

        Args:
            registry: In-memory watchlist of wallets to snapshot.
            data_client: REST client for ``/positions`` queries.
            positions_repo: Append-only repo for history rows.
            snapshot_interval_seconds: Cadence for full-watchlist snapshots.
        """
        self._registry = registry
        self._data_client = data_client
        self._positions_repo = positions_repo
        self._snapshot_interval_seconds = snapshot_interval_seconds

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the snapshot loop until ``stop_event`` is set.

        On each iteration calls :meth:`snapshot_all_wallets`, then waits up to
        ``snapshot_interval_seconds`` for the stop event. Per-iteration
        exceptions are logged and swallowed so a transient upstream hiccup
        does not kill the loop.

        Args:
            stop_event: Cooperative shutdown signal set by the scheduler.
        """
        while not stop_event.is_set():
            try:
                await self.snapshot_all_wallets()
            except Exception:
                _LOG.exception("positions.snapshot_iteration_failed")
            if await self._wait_or_stop(stop_event, self._snapshot_interval_seconds):
                return

    async def snapshot_all_wallets(self) -> int:
        """Snapshot every watched wallet once.

        The watchlist is captured at the start of the cycle so a concurrent
        change does not affect this iteration. Wallets are snapshotted
        sequentially to respect the data-API per-host rate limit.

        Returns:
            Total number of newly-inserted history rows across every wallet.
        """
        watched = sorted(self._registry.addresses())
        total = 0
        for address in watched:
            total += await self._snapshot_wallet(address)
        _LOG.info(
            "positions.snapshot.completed",
            watched_wallets=len(watched),
            inserted=total,
        )
        return total

    async def _snapshot_wallet(self, address: str) -> int:
        """Snapshot one wallet's open positions and append rows.

        Args:
            address: 0x-prefixed proxy wallet address.

        Returns:
            Number of newly-inserted rows for this wallet (0 on error).
        """
        try:
            positions = await self._data_client.get_positions(address)
        except Exception:
            _LOG.exception("positions.get_positions_failed", wallet=address)
            return 0
        snapshot_at = int(time.time())
        inserted = 0
        for position in positions:
            row = _build_history_row(position, wallet=address, snapshot_at=snapshot_at)
            if row is None:
                continue
            try:
                inserted_now = self._positions_repo.insert(row)
            except Exception:
                _LOG.exception(
                    "positions.insert_failed",
                    wallet=address,
                    condition_id=row.condition_id,
                    outcome=row.outcome,
                )
                continue
            if inserted_now:
                inserted += 1
        _LOG.debug("positions.snapshot", wallet=address, rows=inserted)
        return inserted

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


def _build_history_row(
    position: Position,
    *,
    wallet: str,
    snapshot_at: int,
) -> WalletPositionsHistoryRow | None:
    """Convert a :class:`Position` into a history row.

    Defensive: returns ``None`` when ``condition_id`` or ``outcome`` is
    empty, since those columns participate in the composite primary key
    and an empty value would corrupt dedupe semantics.

    Args:
        position: Open-position model from ``DataClient.get_positions``.
        wallet: Address polled (canonical wallet for this row).
        snapshot_at: Unix-seconds timestamp shared across the cycle.

    Returns:
        A populated ``WalletPositionsHistoryRow``, or ``None`` if the
        position is missing required identifying fields.
    """
    if not position.condition_id or not position.outcome:
        return None
    return WalletPositionsHistoryRow(
        wallet=wallet,
        condition_id=position.condition_id,
        outcome=position.outcome,
        size=position.size,
        avg_price=position.avg_price,
        current_value=position.current_value,
        cash_pnl=position.cash_pnl,
        realized_pnl=position.realized_pnl,
        redeemable=position.redeemable,
        snapshot_at=snapshot_at,
    )
