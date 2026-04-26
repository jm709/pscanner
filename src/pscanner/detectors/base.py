"""Detector protocol that the scheduler runs concurrently.

Three lifecycle patterns coexist behind this protocol:

* **Polling** — extend :class:`pscanner.detectors.polling.PollingDetector`
  for a fixed-interval ``_scan(sink)`` loop. Used by ``MispricingDetector``.
* **Trade-driven** — extend
  :class:`pscanner.detectors.trade_driven.TradeDrivenDetector` and implement
  ``evaluate(trade)``. The orchestrator wires the detector's
  ``handle_trade_sync`` into the trade collector's callback fan-out and
  ``run`` simply parks. Used by ``WhalesDetector`` and
  ``ConvergenceDetector``.
* **Stream-driven** — implement ``run(sink)`` directly with
  ``async for event in stream.subscribe(): await evaluate(event, sink)``.
  Used by ``PriceVelocityDetector``.

Hybrid detectors (``WhalesDetector`` adds a periodic market-cache refresh on
top of trade callbacks; ``ClusterDetector`` runs a periodic discovery scan
alongside trade callbacks; ``SmartMoneyDetector`` runs two concurrent
polling loops in one ``TaskGroup``) compose these patterns. There is no
shared abstraction for hybrids — by design, they're rare.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pscanner.alerts.sink import AlertSink


@runtime_checkable
class Detector(Protocol):
    """A long-running coroutine that emits :class:`Alert` instances to a sink.

    Implementers are responsible for their own scheduling cadence; the
    orchestrator simply ``await``s :meth:`run` inside an ``asyncio.TaskGroup``.

    Attributes:
        name: Stable identifier used for logging and supervised restart.
    """

    name: str

    async def run(self, sink: AlertSink) -> None:
        """Run forever (or until cancelled) and publish alerts to ``sink``."""
        ...
