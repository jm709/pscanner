"""Tests for ``AlertSink`` fan-in behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink


def _make_alert(key: str = "k1") -> Alert:
    return Alert(
        detector="smart_money",
        alert_key=key,
        severity="med",
        title="title",
        body={"x": 1},
        created_at=1,
    )


@pytest.mark.asyncio
async def test_emit_returns_true_and_pushes_when_inserted() -> None:
    repo = MagicMock()
    repo.insert_if_new.return_value = True
    renderer = MagicMock()
    sink = AlertSink(repo, renderer=renderer)
    alert = _make_alert()

    result = await sink.emit(alert)

    assert result is True
    repo.insert_if_new.assert_called_once_with(alert)
    renderer.push.assert_called_once_with(alert)


@pytest.mark.asyncio
async def test_emit_calls_subscribers_when_inserted() -> None:
    repo = MagicMock()
    repo.insert_if_new.return_value = True
    sink = AlertSink(repo, renderer=None)
    cb_a = MagicMock()
    cb_b = MagicMock()
    sink.subscribe(cb_a)
    sink.subscribe(cb_b)
    alert = _make_alert()

    await sink.emit(alert)

    cb_a.assert_called_once_with(alert)
    cb_b.assert_called_once_with(alert)


@pytest.mark.asyncio
async def test_emit_returns_false_and_skips_when_dedupe_hit() -> None:
    repo = MagicMock()
    repo.insert_if_new.return_value = False
    renderer = MagicMock()
    sink = AlertSink(repo, renderer=renderer)
    cb = MagicMock()
    sink.subscribe(cb)
    alert = _make_alert()

    result = await sink.emit(alert)

    assert result is False
    renderer.push.assert_not_called()
    cb.assert_not_called()


@pytest.mark.asyncio
async def test_emit_works_without_renderer() -> None:
    repo = MagicMock()
    repo.insert_if_new.return_value = True
    sink = AlertSink(repo)
    cb = MagicMock()
    sink.subscribe(cb)
    alert = _make_alert()

    result = await sink.emit(alert)

    assert result is True
    cb.assert_called_once_with(alert)
