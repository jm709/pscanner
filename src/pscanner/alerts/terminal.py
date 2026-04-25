"""Terminal alert renderer using ``rich.live.Live``.

The renderer keeps a per-detector ring buffer of recent alerts and rebuilds a
three-section ``rich`` layout on a fixed cadence inside an asyncio task. The
``run`` coroutine drives the live display until ``stop`` is called or the task
is cancelled.
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import threading
from typing import Final, get_args

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table

from pscanner.alerts.models import Alert, DetectorName, Severity

_DETECTORS: Final[tuple[DetectorName, ...]] = get_args(DetectorName)
_SEVERITY_STYLE: Final[dict[Severity, str]] = {
    "low": "dim",
    "med": "yellow",
    "high": "red",
}
_BODY_TRUNCATE: Final[int] = 80
_RENDER_INTERVAL_S: Final[float] = 0.2


class TerminalRenderer:
    """Drive a live ``rich`` panel showing the most recent alerts per detector."""

    def __init__(self, *, max_per_detector: int = 20) -> None:
        """Configure the renderer's per-section ring-buffer size.

        Args:
            max_per_detector: Number of recent alerts to keep visible per
                detector section.
        """
        self._max_per_detector = max_per_detector
        self._buffers: dict[DetectorName, collections.deque[Alert]] = {
            name: collections.deque(maxlen=max_per_detector) for name in _DETECTORS
        }
        self._lock = threading.Lock()
        self._stop_event = asyncio.Event()
        self._render_interval_s = _RENDER_INTERVAL_S

    def push(self, alert: Alert) -> None:
        """Append ``alert`` to its detector's ring buffer (thread-safe).

        Args:
            alert: The alert to render.
        """
        with self._lock:
            self._buffers[alert.detector].append(alert)

    async def run(self) -> None:
        """Long-running coroutine that drives the ``rich.live.Live`` panel.

        Returns when :meth:`stop` is called or the task is cancelled.
        """
        self._stop_event.clear()
        console = Console()
        with Live(
            self._build_layout(),
            console=console,
            refresh_per_second=max(1.0, 1.0 / self._render_interval_s),
            screen=False,
            transient=True,
        ) as live:
            while not self._stop_event.is_set():
                live.update(self._build_layout())
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._render_interval_s,
                    )
                except TimeoutError:
                    continue

    async def stop(self) -> None:
        """Signal :meth:`run` to exit and tear down the live display."""
        self._stop_event.set()

    def _snapshot(self) -> dict[DetectorName, list[Alert]]:
        """Return a copy of the per-detector buffers under the lock."""
        with self._lock:
            return {name: list(buf) for name, buf in self._buffers.items()}

    def _build_layout(self) -> Layout:
        """Build a fresh layout with one sub-table per detector."""
        snapshot = self._snapshot()
        layout = Layout()
        layout.split_column(
            *[Layout(name=name, ratio=1) for name in _DETECTORS],
        )
        for name in _DETECTORS:
            layout[name].update(_render_table(name, snapshot[name]))
        return layout


def _render_table(detector: DetectorName, alerts: list[Alert]) -> Table:
    """Render a single detector's table from its alert buffer."""
    table = Table(
        title=detector,
        title_style="bold",
        expand=True,
        show_lines=False,
    )
    table.add_column("Time", no_wrap=True, width=19)
    table.add_column("Severity", no_wrap=True, width=8)
    table.add_column("Title", overflow="fold")
    table.add_column("Body", overflow="fold")
    for alert in alerts:
        table.add_row(*_alert_row(alert))
    return table


def _alert_row(alert: Alert) -> tuple[str, str, str, str]:
    """Format an :class:`Alert` for a single table row."""
    when = datetime.datetime.fromtimestamp(
        alert.created_at,
        tz=datetime.UTC,
    ).strftime("%Y-%m-%d %H:%M:%S")
    style = _SEVERITY_STYLE.get(alert.severity, "")
    severity_cell = f"[{style}]{alert.severity}[/{style}]" if style else alert.severity
    body_text = _truncate(_format_body(alert.body), _BODY_TRUNCATE)
    return when, severity_cell, alert.title, body_text


def _format_body(body: dict[str, object]) -> str:
    """Render a body dict as a compact ``key=value`` string."""
    return ", ".join(f"{k}={v}" for k, v in body.items())


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending an ellipsis when cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
