"""Tests for ``PriceVelocityDetector`` (DC-4 Wave 2).

The detector is exercised against ``MagicMock`` stand-ins for the tick
collector and market-cache repo — no SQLite, no network. The async sink is
captured with an ``AsyncMock`` whose ``emit`` side-effect appends to a list.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import VelocityConfig
from pscanner.detectors.velocity import PriceVelocityDetector


def _capturing_sink() -> tuple[AsyncMock, list[Alert]]:
    """Return an AsyncMock sink that records every emitted Alert."""
    captured: list[Alert] = []

    async def _emit(alert: Alert) -> bool:
        captured.append(alert)
        return True

    sink = AsyncMock()
    sink.emit.side_effect = _emit
    return sink, captured


def _make_detector(
    *,
    threshold: float = 0.05,
    window: int = 60,
    poll_interval: float = 5.0,
    ticks: MagicMock | None = None,
    market_cache: MagicMock | None = None,
) -> tuple[PriceVelocityDetector, MagicMock, MagicMock]:
    """Build a detector wired to mocked collaborators.

    Returns ``(detector, ticks_mock, market_cache_mock)`` so tests can assert
    on call args after driving the detector.
    """
    ticks_mock = ticks if ticks is not None else MagicMock()
    cache_mock = market_cache if market_cache is not None else MagicMock()
    config = VelocityConfig(
        velocity_threshold_pct=threshold,
        velocity_window_seconds=window,
        poll_interval_seconds=poll_interval,
    )
    detector = PriceVelocityDetector(
        config=config,
        ticks_collector=ticks_mock,
        market_cache=cache_mock,
    )
    return detector, ticks_mock, cache_mock


async def test_six_percent_move_emits_med_alert() -> None:
    """6% move > 5% threshold but ≤ 2x threshold → med severity."""
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = [(100, 0.50), (130, 0.53)]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.detector == "velocity"
    assert alert.severity == "med"
    assert alert.body["asset_id"] == "A1"
    assert alert.body["start_price"] == pytest.approx(0.50)
    assert alert.body["end_price"] == pytest.approx(0.53)
    assert alert.body["change_pct"] == pytest.approx(0.06)
    assert alert.body["samples_in_window"] == 2
    assert alert.body["window_seconds"] == 30
    ticks.get_recent_mids.assert_called_once_with("A1", window_seconds=60)


async def test_fifteen_percent_move_emits_high_alert() -> None:
    """15% move > 2x 5% threshold → high severity."""
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = [(100, 0.40), (130, 0.46)]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["change_pct"] == pytest.approx(0.15)


async def test_move_within_threshold_does_not_alert() -> None:
    """2% move below 5% threshold → no alert."""
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = [(100, 0.50), (130, 0.51)]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_negative_move_triggers_high_severity() -> None:
    """A 20% drop is > 2x threshold and registers as high with negative change."""
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = [(100, 0.50), (130, 0.40)]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["change_pct"] == pytest.approx(-0.20)
    assert alert.body["start_price"] == pytest.approx(0.50)
    assert alert.body["end_price"] == pytest.approx(0.40)


async def test_fewer_than_two_mids_does_not_alert() -> None:
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = [(100, 0.40)]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_empty_mids_does_not_alert() -> None:
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = []
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert captured == []


async def test_zero_start_price_does_not_alert() -> None:
    """Degenerate start_price avoids division by zero — no alert, no error."""
    ticks = MagicMock()
    ticks.get_recent_mids.return_value = [(100, 0.0), (130, 0.42)]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_key_uses_60s_bucket() -> None:
    """Two evaluations within the same 60s bucket share the alert_key."""
    ticks = MagicMock()
    ticks.get_recent_mids.side_effect = [
        [(100, 0.40), (130, 0.46)],
        [(100, 0.40), (155, 0.46)],
    ]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)
    await detector.evaluate_asset("A1", sink)

    assert len(captured) == 2
    assert captured[0].alert_key == captured[1].alert_key
    assert captured[0].alert_key == "velocity:A1:2"  # 130 // 60 == 2 == 155 // 60


async def test_alert_keys_distinct_across_buckets() -> None:
    """Evaluations in different 60s buckets produce different keys."""
    ticks = MagicMock()
    ticks.get_recent_mids.side_effect = [
        [(100, 0.40), (130, 0.46)],
        [(100, 0.40), (200, 0.46)],
    ]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)
    await detector.evaluate_asset("A1", sink)

    assert len(captured) == 2
    assert captured[0].alert_key != captured[1].alert_key


async def test_per_asset_isolation() -> None:
    """Alerts for distinct asset ids carry distinct keys and metadata."""
    ticks = MagicMock()

    def _by_asset(asset_id: str, *, window_seconds: int) -> list[tuple[int, float]]:
        del window_seconds
        if asset_id == "A1":
            return [(100, 0.40), (130, 0.46)]
        if asset_id == "A2":
            return [(100, 0.50), (130, 0.45)]
        return []

    ticks.get_recent_mids.side_effect = _by_asset
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset("A1", sink)
    await detector.evaluate_asset("A2", sink)

    assert len(captured) == 2
    keys = {a.alert_key for a in captured}
    assert keys == {"velocity:A1:2", "velocity:A2:2"}
    bodies = {a.body["asset_id"]: a.body for a in captured}
    assert bodies["A1"]["change_pct"] == pytest.approx(0.15)
    assert bodies["A2"]["change_pct"] == pytest.approx(-0.10)


async def test_run_polls_all_subscribed_assets_then_cancels() -> None:
    """``run`` iterates every subscribed asset and exits cleanly on cancel."""
    ticks = MagicMock()
    ticks.subscribed_asset_ids.return_value = {"A1", "A2"}
    ticks.get_recent_mids.return_value = [(100, 0.40), (130, 0.41)]
    detector, _, _ = _make_detector(ticks=ticks, poll_interval=0.05)
    sink, _ = _capturing_sink()

    task = asyncio.create_task(detector.run(sink))
    # Yield enough times for at least one full sweep across both assets.
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    seen_assets = {call.args[0] for call in ticks.get_recent_mids.call_args_list}
    assert seen_assets == {"A1", "A2"}


async def test_run_swallows_evaluate_exceptions() -> None:
    """An exception inside one sweep is caught; the loop keeps running."""
    ticks = MagicMock()
    ticks.subscribed_asset_ids.return_value = {"A1"}
    detector, _, _ = _make_detector(ticks=ticks, poll_interval=0.01)
    sink, _ = _capturing_sink()

    call_count = 0

    async def _flaky(asset_id: str, sink_arg: Any) -> None:
        del asset_id, sink_arg
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "transient failure"
            raise RuntimeError(msg)

    detector.evaluate_asset = _flaky  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    task = asyncio.create_task(detector.run(sink))
    # Three poll cycles at 0.01s each = ~0.03s; give a generous budget.
    for _ in range(30):
        await asyncio.sleep(0.01)
        if call_count >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call_count >= 3
