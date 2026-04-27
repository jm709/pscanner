"""Behavioural tests for :class:`WorkerSink`.

All tests use :class:`FakeClock` and a stub :class:`IAlertSink` that
records calls. No real DB needed. Log assertions use
``structlog.testing.capture_logs`` because pscanner pipes structlog
through ``PrintLoggerFactory`` (see ``src/pscanner/cli.py``) — stdlib
``caplog`` never sees structlog events.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — consumed by tests appended in T3

import pytest  # noqa: F401 — consumed by tests appended in T3
import structlog  # noqa: F401 — consumed by tests appended in T3
from structlog.testing import capture_logs  # noqa: F401 — consumed by tests appended in T3

from pscanner.alerts.models import Alert  # noqa: F401 — consumed by tests appended in T3
from pscanner.alerts.protocol import IAlertSink
from pscanner.alerts.sink import AlertSink


def test_alert_sink_satisfies_ialertsink_structurally() -> None:
    """``AlertSink`` already conforms to ``IAlertSink`` without inheritance.

    This guards against accidental signature drift on either side. The
    assertion runs at type-check time via ``ty``; this test is the
    runtime mirror.
    """
    sink: IAlertSink = AlertSink.__new__(AlertSink)
    assert callable(sink.emit)
