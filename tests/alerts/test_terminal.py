"""Tests for ``TerminalRenderer`` ring-buffer + live-display behaviour."""

from __future__ import annotations

import asyncio
from typing import get_args

import pytest

from pscanner.alerts.models import Alert, DetectorName
from pscanner.alerts.terminal import TerminalRenderer


def _alert(detector: DetectorName, key: str) -> Alert:
    return Alert(
        detector=detector,
        alert_key=key,
        severity="low",
        title=f"title-{key}",
        body={"i": key},
        created_at=1,
    )


def test_push_routes_to_correct_detector_buffer() -> None:
    renderer = TerminalRenderer(max_per_detector=5)
    renderer.push(_alert("smart_money", "a"))
    renderer.push(_alert("mispricing", "b"))
    renderer.push(_alert("whales", "c"))

    snapshot = renderer._snapshot()
    assert [a.alert_key for a in snapshot["smart_money"]] == ["a"]
    assert [a.alert_key for a in snapshot["mispricing"]] == ["b"]
    assert [a.alert_key for a in snapshot["whales"]] == ["c"]


def test_push_buffer_bounded_by_max_per_detector() -> None:
    renderer = TerminalRenderer(max_per_detector=20)
    for i in range(25):
        renderer.push(_alert("whales", str(i)))

    snapshot = renderer._snapshot()
    assert len(snapshot["whales"]) == 20
    # Oldest 5 dropped, retained keys are 5..24.
    assert [a.alert_key for a in snapshot["whales"]] == [str(i) for i in range(5, 25)]


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stopped() -> None:
    renderer = TerminalRenderer(max_per_detector=5)
    renderer._render_interval_s = 0.05

    task = asyncio.create_task(renderer.run())
    await asyncio.sleep(0.1)
    await renderer.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert task.done()
    assert task.exception() is None


@pytest.mark.asyncio
async def test_pushing_while_run_active_does_not_crash() -> None:
    renderer = TerminalRenderer(max_per_detector=5)
    renderer._render_interval_s = 0.02

    task = asyncio.create_task(renderer.run())
    try:
        for i in range(10):
            renderer.push(_alert("mispricing", str(i)))
            await asyncio.sleep(0.01)
    finally:
        await renderer.stop()
        await asyncio.wait_for(task, timeout=1.0)

    snapshot = renderer._snapshot()
    assert len(snapshot["mispricing"]) == 5


def test_move_attribution_in_detector_literal() -> None:
    assert "move_attribution" in get_args(DetectorName)


def test_renderer_handles_move_attribution_alert() -> None:
    renderer = TerminalRenderer(max_per_detector=5)
    alert = Alert(
        detector="move_attribution",  # type: ignore[arg-type]
        alert_key="cluster.candidate:0xabc:Yes:BUY:1700000000",
        severity="med",
        title="cluster candidate burst",
        body={},
        created_at=1700000000,
    )
    renderer.push(alert)  # must not KeyError
