"""Terminal alert renderer using ``rich.live.Live``.

Wave 1 freezes the public surface; Wave 2's ``alert-sink`` agent implements
the body, including the three-section table layout (one section per detector).
"""

from __future__ import annotations

from pscanner.alerts.models import Alert


class TerminalRenderer:
    """Drive a live ``rich`` panel showing the most recent alerts per detector."""

    def __init__(self, *, max_per_detector: int = 20) -> None:
        """Configure the renderer's per-section ring-buffer size.

        Args:
            max_per_detector: Number of recent alerts to keep visible per
                detector section.
        """
        raise NotImplementedError("Wave 2: alert-sink")

    def push(self, alert: Alert) -> None:
        """Append ``alert`` to its detector's ring buffer (thread-safe).

        Args:
            alert: The alert to render.
        """
        raise NotImplementedError("Wave 2: alert-sink")

    async def run(self) -> None:
        """Long-running coroutine that drives the ``rich.live.Live`` panel.

        Returns when :meth:`stop` is called or the task is cancelled.
        """
        raise NotImplementedError("Wave 2: alert-sink")

    async def stop(self) -> None:
        """Signal :meth:`run` to exit and tear down the live display."""
        raise NotImplementedError("Wave 2: alert-sink")
