"""Base class for detectors driven by ``TradeCollector`` callbacks.

``WhalesDetector`` and ``ConvergenceDetector`` both subscribe to the trade
collector and react to each newly-recorded ``WalletTrade``. The orchestration
is identical: store an injected ``AlertSink``, expose ``handle_trade_sync``
that dispatches into an async ``evaluate`` method, and track in-flight tasks
so they aren't garbage collected mid-flight.

Concrete subclasses override ``evaluate`` (the actual signal logic). The
``run`` method satisfies the ``Detector`` protocol but only stores the sink
and parks — actual work happens via the trade callback.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.store.repo import WalletTrade

_LOG = structlog.get_logger(__name__)


class TradeDrivenDetector(ABC):
    """Abstract base for detectors that react to ``wallet_trades`` inserts."""

    name: str = ""

    def __init__(self) -> None:
        """Initialise the shared sink slot and pending-tasks tracker."""
        self._sink: AlertSink | None = None
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @abstractmethod
    async def evaluate(self, trade: WalletTrade) -> None:
        """Process one freshly-recorded trade. Subclass-specific logic.

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """

    def handle_trade_sync(self, trade: WalletTrade) -> None:
        """Sync entry for ``TradeCollector.subscribe_new_trade``.

        Spawns ``evaluate(trade)`` as an async task and tracks it so it
        isn't garbage collected before completion. No-ops if there's no
        running event loop (e.g., test setup that hasn't started one).

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug(
                "trade_driven.no_event_loop",
                detector=self.name,
                tx=trade.transaction_hash,
            )
            return
        task = loop.create_task(self.evaluate(trade))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def run(self, sink: AlertSink) -> None:
        """Park forever — the detector is callback-driven, not loop-driven.

        Stores ``sink`` if not already pre-wired by the scheduler. Returns
        only on cancellation.

        Args:
            sink: Shared alert sink used by :meth:`evaluate` for emission.
        """
        if self._sink is None:
            self._sink = sink
        await asyncio.Event().wait()
