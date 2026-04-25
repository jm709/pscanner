"""AlertSink — fan-in for every detector.

The sink dedupes by ``alert_key`` (delegated to ``AlertsRepo.insert_if_new``),
persists newly-seen alerts to SQLite, and forwards them to the optional
terminal renderer plus any registered subscribers.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.terminal import TerminalRenderer
from pscanner.store.repo import AlertsRepo

_log = structlog.get_logger(__name__)


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
        self._alerts_repo = alerts_repo
        self._renderer = renderer
        self._subscribers: list[Callable[[Alert], None]] = []

    async def emit(self, alert: Alert) -> bool:
        """Publish ``alert`` once. Returns whether it was newly inserted.

        Args:
            alert: The alert to publish.

        Returns:
            ``True`` if the alert was new (and thus forwarded to renderer +
            subscribers); ``False`` if it was a dedupe hit.
        """
        inserted = self._alerts_repo.insert_if_new(alert)
        if not inserted:
            return False
        if self._renderer is not None:
            self._renderer.push(alert)
        for callback in self._subscribers:
            callback(alert)
        _log.info(
            "alert.emitted",
            detector=alert.detector,
            alert_key=alert.alert_key,
            severity=alert.severity,
        )
        return True

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        """Register a synchronous callback fired on every newly-inserted alert.

        Args:
            callback: Synchronous fn invoked from within ``emit``. Must not block.
        """
        self._subscribers.append(callback)
