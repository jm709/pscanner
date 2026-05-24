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

from pscanner.alerts.sink import AlertSink
from pscanner.store.repo import WalletTrade
from pscanner.util.async_dispatch import AsyncDispatcher


class TradeDrivenDetector(ABC):
    """Abstract base for detectors that react to ``wallet_trades`` inserts."""

    name: str = ""

    def __init__(self) -> None:
        """Initialise the shared sink slot and async dispatcher."""
        self._sink: AlertSink | None = None
        self._dispatcher = AsyncDispatcher(log_event_no_loop="trade_driven.no_event_loop")

    @property
    def pending_tasks(self) -> set[asyncio.Task[None]]:
        """Live view of in-flight ``evaluate`` tasks (test + aclose hook)."""
        return self._dispatcher.pending

    @abstractmethod
    async def evaluate(self, trade: WalletTrade) -> None:
        """Process one freshly-recorded trade. Subclass-specific logic.

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """

    def wire_sink(self, sink: AlertSink) -> None:
        """Pre-wire the alert sink before :meth:`run` starts.

        Used by the scheduler (and tests) to seed the sink before the
        callback-driven path fires for the first time. Mirrors the
        ``if self._sink is None: self._sink = sink`` ratcheting in
        :meth:`run` — callers that drive ``run()`` directly without
        pre-wiring still get the fallback assignment.
        """
        self._sink = sink

    def handle_trade_sync(self, trade: WalletTrade) -> None:
        """Sync entry for ``TradeCollector.subscribe_new_trade``.

        Spawns ``evaluate(trade)`` as a tracked task. No-ops (with a
        ``trade_driven.no_event_loop`` debug event) when no event loop is
        running, e.g. test setup that hasn't started one.

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """
        self._dispatcher.spawn(
            self.evaluate(trade),
            detector=self.name,
            tx=trade.transaction_hash,
        )

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
