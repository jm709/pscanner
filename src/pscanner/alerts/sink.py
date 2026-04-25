"""AlertSink — fan-in for every detector.

The sink dedupes by ``alert_key`` (delegated to ``AlertsRepo.insert_if_new``),
persists newly-seen alerts to SQLite, and forwards them to the optional
terminal renderer plus any registered subscribers. Wave 1 freezes the shape;
Wave 2's ``alert-sink`` agent implements the body.
"""

from __future__ import annotations

from collections.abc import Callable

from pscanner.alerts.models import Alert
from pscanner.alerts.terminal import TerminalRenderer
from pscanner.store.repo import AlertsRepo


class AlertSink:
    """Single fan-in point that detectors call to publish an :class:`Alert`."""

    def __init__(
        self,
        alerts_repo: AlertsRepo,
        renderer: TerminalRenderer | None = None,
    ) -> None:
        """Build a sink wired to the persistence layer and optional renderer.

        Args:
            alerts_repo: Repo used for the dedupe-then-insert step.
            renderer: Optional terminal renderer that receives every newly
                inserted alert via ``push``.
        """
        raise NotImplementedError("Wave 2: alert-sink")

    async def emit(self, alert: Alert) -> bool:
        """Publish ``alert`` once. Returns whether it was newly inserted.

        Args:
            alert: The alert to publish.

        Returns:
            ``True`` if the alert was new (and thus forwarded to renderer +
            subscribers); ``False`` if it was a dedupe hit.
        """
        raise NotImplementedError("Wave 2: alert-sink")

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        """Register a synchronous callback fired on every newly-inserted alert.

        Args:
            callback: Synchronous fn invoked from within ``emit``. Must not block.
        """
        raise NotImplementedError("Wave 2: alert-sink")
