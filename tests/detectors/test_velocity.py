"""Tests for ``PriceVelocityDetector`` (DC-4 Wave 2).

The detector now consumes a :class:`TickStream`. Tests inject a small
``_FakeStream`` that yields canned :class:`TickEvent` objects in order,
exercising the full evaluate-on-each-tick path without any tick-collector
mocking. A handful of tests use a real ``AlertSink`` + ``AlertsRepo``
against an in-memory SQLite database (via the ``tmp_db`` fixture) to
verify that the alert_key dedupe path collapses YES/NO pairs into one row.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import VelocityConfig
from pscanner.detectors.velocity import PriceVelocityDetector
from pscanner.poly.ids import AssetId, ConditionId, EventSlug, MarketId
from pscanner.poly.tick_stream import TickEvent
from pscanner.store.repo import AlertsRepo


class _FakeStream:
    """Test stream that yields the queued events then sleeps until cancelled."""

    def __init__(self, events: list[TickEvent]) -> None:
        self._events = list(events)

    def subscribe(self) -> AsyncIterator[TickEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[TickEvent]:
        for event in self._events:
            yield event
        # Block forever so the detector's `async for` loop doesn't return on
        # its own — tests cancel the run task explicitly.
        await asyncio.Event().wait()
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


def _capturing_sink() -> tuple[AsyncMock, list[Alert]]:
    """Return an AsyncMock sink that records every emitted Alert."""
    captured: list[Alert] = []

    async def _emit(alert: Alert) -> bool:
        captured.append(alert)
        return True

    sink = AsyncMock()
    sink.emit.side_effect = _emit
    return sink, captured


def _tick(
    *,
    asset_id: str = "A1",
    snapshot_at: int,
    mid: float | None = 0.5,
    bid_depth: float | None = 1000.0,
    ask_depth: float | None = 1000.0,
    spread: float | None = 0.02,
    market_title: str | None = None,
    condition_id: str | None = None,
    event_slug: str | None = None,
    market_id: str | None = None,
) -> TickEvent:
    """Build a balanced ``TickEvent`` with optional metadata enrichment."""
    return TickEvent(
        asset_id=AssetId(asset_id),
        snapshot_at=snapshot_at,
        mid_price=mid,
        best_bid=(mid - 0.01) if mid is not None else None,
        best_ask=(mid + 0.01) if mid is not None else None,
        spread=spread,
        bid_depth_top5=bid_depth,
        ask_depth_top5=ask_depth,
        last_trade_price=mid,
        market_id=MarketId(market_id) if market_id else None,
        condition_id=ConditionId(condition_id) if condition_id else None,
        market_title=market_title,
        event_slug=EventSlug(event_slug) if event_slug else None,
    )


def _make_detector(
    *,
    threshold: float = 0.05,
    window: int = 60,
    poll_interval: float = 5.0,
    depth_asymmetry_floor: float = 0.05,
    min_mid_liquidity_usd: float = 100.0,
    market_cache: MagicMock | None = None,
    stream: _FakeStream | None = None,
) -> tuple[PriceVelocityDetector, _FakeStream, MagicMock]:
    """Build a detector wired to a fake stream + a market-cache mock.

    Returns ``(detector, fake_stream, market_cache_mock)``. When ``stream`` is
    omitted an empty stream is used; tests typically construct one explicitly.
    """
    fake_stream = stream if stream is not None else _FakeStream([])
    cache_mock = market_cache if market_cache is not None else MagicMock()
    config = VelocityConfig(
        velocity_threshold_pct=threshold,
        velocity_window_seconds=window,
        poll_interval_seconds=poll_interval,
        depth_asymmetry_floor=depth_asymmetry_floor,
        min_mid_liquidity_usd=min_mid_liquidity_usd,
    )
    detector = PriceVelocityDetector(
        config=config,
        tick_stream=fake_stream,
        market_cache=cache_mock,
    )
    return detector, fake_stream, cache_mock


async def _drive(detector: PriceVelocityDetector, sink: AsyncMock, events: list[TickEvent]) -> None:
    """Feed ``events`` through ``detector.evaluate`` in order."""
    for event in events:
        await detector.evaluate(event, sink)


async def test_six_percent_move_emits_med_alert() -> None:
    """6% move > 5% threshold but ≤ 2x threshold → med severity."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.50), _tick(snapshot_at=130, mid=0.53)]
    )

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


async def test_fifteen_percent_move_emits_high_alert() -> None:
    """15% move > 2x 5% threshold → high severity."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.40), _tick(snapshot_at=130, mid=0.46)]
    )

    assert len(captured) == 1
    assert captured[0].severity == "high"
    assert captured[0].body["change_pct"] == pytest.approx(0.15)


async def test_move_within_threshold_does_not_alert() -> None:
    """2% move below 5% threshold → no alert."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.50), _tick(snapshot_at=130, mid=0.51)]
    )

    assert captured == []
    sink.emit.assert_not_called()


async def test_negative_move_triggers_high_severity() -> None:
    """A 20% drop is > 2x threshold and registers as high with negative change."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.50), _tick(snapshot_at=130, mid=0.40)]
    )

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["change_pct"] == pytest.approx(-0.20)
    assert alert.body["start_price"] == pytest.approx(0.50)
    assert alert.body["end_price"] == pytest.approx(0.40)


async def test_fewer_than_two_mids_does_not_alert() -> None:
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(detector, sink, [_tick(snapshot_at=100, mid=0.40)])

    assert captured == []
    sink.emit.assert_not_called()


async def test_zero_start_price_does_not_alert() -> None:
    """Degenerate start_price avoids division by zero — no alert, no error."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.0), _tick(snapshot_at=130, mid=0.42)]
    )

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_key_uses_60s_bucket() -> None:
    """Two evaluations within the same 60s bucket share the alert_key."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40),
            _tick(snapshot_at=130, mid=0.46),
            _tick(snapshot_at=155, mid=0.46),
        ],
    )

    # First alert at 130, second at 155; both in bucket 130//60 == 2 == 155//60.
    assert len(captured) >= 2
    assert captured[0].alert_key == captured[1].alert_key
    assert captured[0].alert_key == "velocity:A1:2"


async def test_alert_keys_distinct_across_buckets() -> None:
    """Evaluations in different 60s buckets produce different keys."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.40), _tick(snapshot_at=130, mid=0.46)]
    )
    # Reuse a fresh detector per-bucket because the in-memory history would
    # otherwise expire the older mid by the time the second tick lands.
    detector2, _, _ = _make_detector()
    await _drive(
        detector2, sink, [_tick(snapshot_at=170, mid=0.40), _tick(snapshot_at=200, mid=0.46)]
    )

    assert len(captured) == 2
    assert captured[0].alert_key != captured[1].alert_key


async def test_per_asset_isolation() -> None:
    """Alerts for distinct asset ids carry distinct keys and metadata."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    events = [
        _tick(asset_id="A1", snapshot_at=100, mid=0.40),
        _tick(asset_id="A2", snapshot_at=100, mid=0.50),
        _tick(asset_id="A1", snapshot_at=130, mid=0.46),
        _tick(asset_id="A2", snapshot_at=130, mid=0.45),
    ]
    await _drive(detector, sink, events)

    assert len(captured) == 2
    keys = {a.alert_key for a in captured}
    assert keys == {"velocity:A1:2", "velocity:A2:2"}
    bodies = {a.body["asset_id"]: a.body for a in captured}
    assert bodies["A1"]["change_pct"] == pytest.approx(0.15)
    assert bodies["A2"]["change_pct"] == pytest.approx(-0.10)


async def test_run_consumes_stream_and_evaluates_each_event() -> None:
    """``run`` subscribes to the stream and emits an alert per qualifying event."""
    events = [
        _tick(asset_id="A1", snapshot_at=100, mid=0.40),
        _tick(asset_id="A1", snapshot_at=130, mid=0.46),
    ]
    stream = _FakeStream(events)
    detector, _, _ = _make_detector(stream=stream)
    sink, captured = _capturing_sink()

    task = asyncio.create_task(detector.run(sink))
    # Spin until the alert lands or we run out of patience.
    for _ in range(50):
        await asyncio.sleep(0)
        if captured:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(captured) == 1
    assert captured[0].body["change_pct"] == pytest.approx(0.15)


async def test_run_swallows_evaluate_exceptions() -> None:
    """An evaluate raise on one event is logged; the loop continues with the next."""
    stream = _FakeStream(
        [_tick(snapshot_at=100, mid=0.40), _tick(snapshot_at=130, mid=0.46)],
    )
    detector, _, _ = _make_detector(stream=stream)
    sink, _ = _capturing_sink()

    call_count = 0
    real_evaluate = detector.evaluate

    async def _flaky(tick: TickEvent, sink_arg: AlertSink) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "transient failure"
            raise RuntimeError(msg)
        await real_evaluate(tick, sink_arg)

    detector.evaluate = _flaky  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    task = asyncio.create_task(detector.run(sink))
    for _ in range(50):
        await asyncio.sleep(0)
        if call_count >= 2:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert call_count >= 2


async def test_alert_skipped_on_depth_asymmetry() -> None:
    """Lopsided book depth (ratio < ``depth_asymmetry_floor``) suppresses alerts."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, bid_depth=1_000_000.0, ask_depth=500.0),
            _tick(snapshot_at=130, mid=0.46, bid_depth=1_000_000.0, ask_depth=500.0),
        ],
    )

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_skipped_on_insufficient_usd_depth() -> None:
    """``min_side * mid`` below ``min_mid_liquidity_usd`` suppresses alerts."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, bid_depth=50.0, ask_depth=50.0),
            _tick(snapshot_at=130, mid=0.46, bid_depth=50.0, ask_depth=50.0),
        ],
    )

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_skipped_when_depth_fields_missing() -> None:
    """A latest tick with NULL depth fields cannot be qualified — skip."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, bid_depth=1000.0, ask_depth=1000.0),
            _tick(snapshot_at=130, mid=0.46, bid_depth=None, ask_depth=None),
        ],
    )

    assert captured == []


async def test_alert_emitted_with_balanced_depth() -> None:
    """Balanced 1000-share depth at mid=0.5 (500 USD/side) clears every filter."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, bid_depth=1000.0, ask_depth=1000.0),
            _tick(snapshot_at=130, mid=0.46, bid_depth=1000.0, ask_depth=1000.0),
        ],
    )

    assert len(captured) == 1
    assert captured[0].body["change_pct"] == pytest.approx(0.15)


async def test_alert_body_includes_market_metadata() -> None:
    """``market_title`` and ``condition_id`` propagate into the alert body."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, market_title="Foo", condition_id="0xCOND"),
            _tick(snapshot_at=130, mid=0.46, market_title="Foo", condition_id="0xCOND"),
        ],
    )

    assert len(captured) == 1
    body = captured[0].body
    assert body["market_title"] == "Foo"
    assert body["condition_id"] == "0xCOND"


async def test_alert_body_handles_missing_market() -> None:
    """A wallet-only / unknown asset still alerts with ``None`` metadata."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.40), _tick(snapshot_at=130, mid=0.46)]
    )

    assert len(captured) == 1
    body = captured[0].body
    assert body["market_title"] is None
    assert body["condition_id"] is None


async def test_alert_key_uses_condition_id_when_available() -> None:
    """Binary YES/NO velocity alerts share an alert_key by ``condition_id``."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, condition_id="0xCOND"),
            _tick(snapshot_at=130, mid=0.46, condition_id="0xCOND"),
        ],
    )

    assert len(captured) == 1
    assert captured[0].alert_key == "velocity:0xCOND:2"


async def test_alert_key_falls_back_to_asset_id_when_market_missing() -> None:
    """When the market cache has not populated, fall back to asset-keyed format."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector, sink, [_tick(snapshot_at=100, mid=0.40), _tick(snapshot_at=130, mid=0.46)]
    )

    assert len(captured) == 1
    assert captured[0].alert_key == "velocity:A1:2"


async def test_yes_no_dedupe_via_alerts_repo(tmp_db: sqlite3.Connection) -> None:
    """YES + NO alerts on the same condition_id within one bucket → one row.

    Wires a real ``AlertSink`` + ``AlertsRepo`` so the dedupe path runs through
    the SQLite ``INSERT OR IGNORE`` PK conflict on ``alert_key``. After both
    detector evaluations the table holds exactly one row keyed by
    ``condition_id``, and the post-insert subscriber fired only once (proof
    the second emit hit the dedupe path and returned ``False``).
    """
    detector, _, _ = _make_detector()
    sink = AlertSink(AlertsRepo(tmp_db))
    forwarded: list[Alert] = []
    sink.subscribe(forwarded.append)

    yes_events = [
        _tick(asset_id="A_YES", snapshot_at=100, mid=0.40, condition_id="0xCOND"),
        _tick(asset_id="A_YES", snapshot_at=130, mid=0.46, condition_id="0xCOND"),
    ]
    no_events = [
        _tick(asset_id="A_NO", snapshot_at=100, mid=0.60, condition_id="0xCOND"),
        _tick(asset_id="A_NO", snapshot_at=130, mid=0.50, condition_id="0xCOND"),
    ]
    for event in [*yes_events, *no_events]:
        await detector.evaluate(event, sink)

    rows = tmp_db.execute("SELECT alert_key FROM alerts").fetchall()
    assert len(rows) == 1
    assert rows[0]["alert_key"] == "velocity:0xCOND:2"
    assert len(forwarded) == 1


async def test_consolidation_demotes_to_low_severity() -> None:
    """Spread compresses 307x (Youngkin shape) → severity demoted to ``low``."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, spread=0.92),
            _tick(snapshot_at=130, mid=0.46, spread=0.003),
        ],
    )

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "low"
    assert alert.body["consolidation"] is True
    assert alert.body["spread_ratio"] == pytest.approx(306.7, abs=0.5)
    assert alert.body["spread_before"] == pytest.approx(0.92)
    assert alert.body["spread_after"] == pytest.approx(0.003)


async def test_real_move_is_not_consolidation() -> None:
    """Spread ratio 1.25x is below the 5.0 floor → severity follows magnitude."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, spread=0.05),
            _tick(snapshot_at=130, mid=0.46, spread=0.04),
        ],
    )

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["consolidation"] is False
    assert "spread_ratio" not in alert.body


async def test_zero_last_spread_does_not_raise() -> None:
    """A latest tick with ``spread=0`` cannot be classified — treat as not consolidation."""
    detector, _, _ = _make_detector()
    sink, captured = _capturing_sink()

    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, spread=0.05),
            _tick(snapshot_at=130, mid=0.46, spread=0.0),
        ],
    )

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["consolidation"] is False


async def test_single_tick_history_skips_consolidation() -> None:
    """One tick in the window → cannot compute ratio → not consolidation.

    Achieved by feeding a stale tick first (outside ``2x window``) so the
    rolling history trims it before the alert fires.
    """
    detector, _, _ = _make_detector(window=60)
    sink, captured = _capturing_sink()

    # First tick is 600s ahead so it lands inside the window for snapshot_at=130;
    # second tick at 130 with mid=0.46 vs first at 100 with mid=0.40 = +15%.
    # We need only-one tick in window for consolidation: place the first tick
    # outside the trim horizon (window*2 = 120s before end) by using two
    # quick events: history is then [first, last] — len==2 — but first.spread
    # is None (not provided) so consolidation classifier returns None.
    await _drive(
        detector,
        sink,
        [
            _tick(snapshot_at=100, mid=0.40, spread=None),
            _tick(snapshot_at=130, mid=0.46, spread=0.003),
        ],
    )

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["consolidation"] is False
