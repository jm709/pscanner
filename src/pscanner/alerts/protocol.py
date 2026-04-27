"""``IAlertSink`` Protocol — the surface tick-driven detectors depend on.

Both :class:`AlertSink` (synchronous in-process delivery) and
:class:`WorkerSink` (queue-deferred delivery) implement this Protocol so
detectors can accept either without code changes.
"""

from __future__ import annotations

from typing import Protocol

from pscanner.alerts.models import Alert


class IAlertSink(Protocol):
    """The single method tick-driven detectors call to publish an alert."""

    async def emit(self, alert: Alert) -> bool:
        """Publish ``alert``.

        Return ``True`` if accepted (newly inserted or successfully enqueued),
        ``False`` if rejected (dedupe hit on the synchronous path, or worker
        closed on the deferred path).
        """
        ...
