"""Tests for ``PriceVelocityDetector`` (DC-4 Wave 2).

The detector is exercised against ``MagicMock`` stand-ins for the tick
collector and market-cache repo — no SQLite, no network. The async sink is
captured with an ``AsyncMock`` whose ``emit`` side-effect appends to a list.

A handful of tests use a real ``AlertSink`` + ``AlertsRepo`` against an
in-memory SQLite database (via the ``tmp_db`` fixture) to verify that the
alert_key dedupe path collapses YES/NO pairs into a single row.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import VelocityConfig
from pscanner.detectors.velocity import PriceVelocityDetector
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import AlertsRepo, CachedMarket, MarketTick


def _capturing_sink() -> tuple[AsyncMock, list[Alert]]:
    """Return an AsyncMock sink that records every emitted Alert."""
    captured: list[Alert] = []

    async def _emit(alert: Alert) -> bool:
        captured.append(alert)
        return True

    sink = AsyncMock()
    sink.emit.side_effect = _emit
    return sink, captured


def _balanced_tick(
    *,
    bid_depth: float = 1000.0,
    ask_depth: float = 1000.0,
    mid: float = 0.5,
    snapshot_at: int = 130,
    spread: float | None = 0.02,
) -> MarketTick:
    """Build a ``MarketTick`` with balanced, liquid depth that passes filters."""
    return MarketTick(
        asset_id=AssetId("A1"),
        condition_id=ConditionId("0xcond"),
        snapshot_at=snapshot_at,
        mid_price=mid,
        best_bid=mid - 0.01,
        best_ask=mid + 0.01,
        spread=spread,
        bid_depth_top5=bid_depth,
        ask_depth_top5=ask_depth,
        last_trade_price=mid,
    )


def _ticks_mock_with_defaults(
    *,
    mids: list[tuple[int, float]] | None = None,
    tick: MarketTick | None = None,
    ticks: list[MarketTick] | None = None,
    market: CachedMarket | None = None,
) -> MagicMock:
    """Build a tick-collector mock with balanced depth + no market by default.

    ``tick`` overrides the latest-tick history (a single-element list).
    ``ticks`` overrides the full window for consolidation classifier tests.
    Pass at most one of the two.
    """
    if tick is not None and ticks is not None:
        msg = "pass either tick or ticks, not both"
        raise ValueError(msg)
    mock = MagicMock()
    mock.get_recent_mids.return_value = mids if mids is not None else [(100, 0.40), (130, 0.46)]
    if ticks is not None:
        mock.get_recent_ticks.return_value = ticks
    else:
        mock.get_recent_ticks.return_value = [tick if tick is not None else _balanced_tick()]
    mock.get_market_for_asset.return_value = market
    return mock


def _make_detector(
    *,
    threshold: float = 0.05,
    window: int = 60,
    poll_interval: float = 5.0,
    depth_asymmetry_floor: float = 0.05,
    min_mid_liquidity_usd: float = 100.0,
    ticks: MagicMock | None = None,
    market_cache: MagicMock | None = None,
) -> tuple[PriceVelocityDetector, MagicMock, MagicMock]:
    """Build a detector wired to mocked collaborators.

    Returns ``(detector, ticks_mock, market_cache_mock)`` so tests can assert
    on call args after driving the detector. When ``ticks`` is ``None``, the
    helper installs a tick mock that returns balanced depth and a 15% move so
    the happy-path defaults emit an alert.
    """
    ticks_mock = ticks if ticks is not None else _ticks_mock_with_defaults()
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
        ticks_collector=ticks_mock,
        market_cache=cache_mock,
    )
    return detector, ticks_mock, cache_mock


async def test_six_percent_move_emits_med_alert() -> None:
    """6% move > 5% threshold but ≤ 2x threshold → med severity."""
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.50), (130, 0.53)])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

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
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.40), (130, 0.46)])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["change_pct"] == pytest.approx(0.15)


async def test_move_within_threshold_does_not_alert() -> None:
    """2% move below 5% threshold → no alert."""
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.50), (130, 0.51)])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_negative_move_triggers_high_severity() -> None:
    """A 20% drop is > 2x threshold and registers as high with negative change."""
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.50), (130, 0.40)])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["change_pct"] == pytest.approx(-0.20)
    assert alert.body["start_price"] == pytest.approx(0.50)
    assert alert.body["end_price"] == pytest.approx(0.40)


async def test_fewer_than_two_mids_does_not_alert() -> None:
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.40)])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_empty_mids_does_not_alert() -> None:
    ticks = _ticks_mock_with_defaults(mids=[])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []


async def test_zero_start_price_does_not_alert() -> None:
    """Degenerate start_price avoids division by zero — no alert, no error."""
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.0), (130, 0.42)])
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_key_uses_60s_bucket() -> None:
    """Two evaluations within the same 60s bucket share the alert_key."""
    ticks = _ticks_mock_with_defaults()
    ticks.get_recent_mids.side_effect = [
        [(100, 0.40), (130, 0.46)],
        [(100, 0.40), (155, 0.46)],
    ]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)
    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 2
    assert captured[0].alert_key == captured[1].alert_key
    assert captured[0].alert_key == "velocity:A1:2"  # 130 // 60 == 2 == 155 // 60


async def test_alert_keys_distinct_across_buckets() -> None:
    """Evaluations in different 60s buckets produce different keys."""
    ticks = _ticks_mock_with_defaults()
    ticks.get_recent_mids.side_effect = [
        [(100, 0.40), (130, 0.46)],
        [(100, 0.40), (200, 0.46)],
    ]
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)
    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 2
    assert captured[0].alert_key != captured[1].alert_key


async def test_per_asset_isolation() -> None:
    """Alerts for distinct asset ids carry distinct keys and metadata."""
    ticks = _ticks_mock_with_defaults()

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

    await detector.evaluate_asset(AssetId("A1"), sink)
    await detector.evaluate_asset(AssetId("A2"), sink)

    assert len(captured) == 2
    keys = {a.alert_key for a in captured}
    assert keys == {"velocity:A1:2", "velocity:A2:2"}
    bodies = {a.body["asset_id"]: a.body for a in captured}
    assert bodies["A1"]["change_pct"] == pytest.approx(0.15)
    assert bodies["A2"]["change_pct"] == pytest.approx(-0.10)


async def test_run_polls_all_subscribed_assets_then_cancels() -> None:
    """``run`` iterates every subscribed asset and exits cleanly on cancel."""
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.40), (130, 0.41)])
    ticks.subscribed_asset_ids.return_value = {"A1", "A2"}
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


async def test_alert_skipped_on_depth_asymmetry() -> None:
    """Lopsided book depth (ratio < ``depth_asymmetry_floor``) suppresses alerts."""
    tick = _balanced_tick(bid_depth=1_000_000.0, ask_depth=500.0)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],  # +15% — well above threshold
        tick=tick,
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_skipped_on_insufficient_usd_depth() -> None:
    """``min_side * mid`` below ``min_mid_liquidity_usd`` suppresses alerts."""
    # bid=ask=50 shares, mid=0.5 → 25 USD per side, below the 100 USD floor.
    tick = _balanced_tick(bid_depth=50.0, ask_depth=50.0, mid=0.5)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],
        tick=tick,
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_alert_skipped_when_recent_ticks_empty() -> None:
    """Empty tick history disables the filter check and suppresses the alert."""
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.40), (130, 0.46)])
    ticks.get_recent_ticks.return_value = []
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []


async def test_alert_skipped_when_depth_fields_missing() -> None:
    """A latest tick with NULL depth fields cannot be qualified — skip."""
    tick = MarketTick(
        asset_id=AssetId("A1"),
        condition_id=ConditionId("0xcond"),
        snapshot_at=130,
        mid_price=0.5,
        best_bid=0.49,
        best_ask=0.51,
        spread=0.02,
        bid_depth_top5=None,
        ask_depth_top5=None,
        last_trade_price=0.5,
    )
    ticks = _ticks_mock_with_defaults(mids=[(100, 0.40), (130, 0.46)], tick=tick)
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert captured == []


async def test_alert_emitted_with_balanced_depth() -> None:
    """Balanced 1000-share depth at mid=0.5 (500 USD/side) clears every filter."""
    tick = _balanced_tick(bid_depth=1000.0, ask_depth=1000.0, mid=0.5)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],
        tick=tick,
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    assert captured[0].body["change_pct"] == pytest.approx(0.15)


async def test_alert_body_includes_market_metadata() -> None:
    """``market_title`` and ``condition_id`` propagate into the alert body."""
    market = CachedMarket(
        market_id=MarketId("m1"),
        event_id=None,
        title="Foo",
        liquidity_usd=None,
        volume_usd=None,
        outcome_prices=[0.5, 0.5],
        active=True,
        cached_at=0,
        condition_id=ConditionId("0xCOND"),
        event_slug=None,
    )
    ticks = _ticks_mock_with_defaults(market=market)
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    body = captured[0].body
    assert body["market_title"] == "Foo"
    assert body["condition_id"] == "0xCOND"
    ticks.get_market_for_asset.assert_called_once_with("A1")


async def test_alert_body_handles_missing_market() -> None:
    """A wallet-only / unknown asset still alerts with ``None`` metadata."""
    ticks = _ticks_mock_with_defaults(market=None)
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    body = captured[0].body
    assert body["market_title"] is None
    assert body["condition_id"] is None


def _make_cached_market(condition_id: str, *, market_id: str = "m1") -> CachedMarket:
    """Build a minimal ``CachedMarket`` whose only meaningful field is ``condition_id``."""
    return CachedMarket(
        market_id=MarketId(market_id),
        event_id=None,
        title="Binary market",
        liquidity_usd=None,
        volume_usd=None,
        outcome_prices=[0.5, 0.5],
        active=True,
        cached_at=0,
        condition_id=ConditionId(condition_id),
        event_slug=None,
    )


async def test_alert_key_uses_condition_id_when_available() -> None:
    """Binary YES/NO velocity alerts share an alert_key by ``condition_id``."""
    market = _make_cached_market("0xCOND")
    ticks = _ticks_mock_with_defaults(market=market)
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    assert captured[0].alert_key == "velocity:0xCOND:2"  # 130 // 60 == 2


async def test_alert_key_falls_back_to_asset_id_when_market_missing() -> None:
    """When the market cache has not populated, fall back to asset-keyed format."""
    ticks = _ticks_mock_with_defaults(market=None)
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

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
    market = _make_cached_market("0xCOND")
    ticks = _ticks_mock_with_defaults(market=market)
    ticks.get_recent_mids.side_effect = [
        [(100, 0.40), (130, 0.46)],  # YES side, +15%
        [(100, 0.60), (130, 0.50)],  # NO side, opposite move, same end_ts bucket
    ]
    detector, _, _ = _make_detector(ticks=ticks)

    sink = AlertSink(AlertsRepo(tmp_db))
    forwarded: list[Alert] = []
    sink.subscribe(forwarded.append)

    await detector.evaluate_asset(AssetId("A_YES"), sink)
    await detector.evaluate_asset(AssetId("A_NO"), sink)

    rows = tmp_db.execute("SELECT alert_key FROM alerts").fetchall()
    assert len(rows) == 1
    assert rows[0]["alert_key"] == "velocity:0xCOND:2"
    # Subscribers only fire on newly-inserted alerts; second emit was a dedupe hit.
    assert len(forwarded) == 1


async def test_consolidation_demotes_to_low_severity() -> None:
    """Spread compresses 307x (Youngkin shape) → severity demoted to ``low``."""
    first = _balanced_tick(snapshot_at=70, spread=0.92)
    last = _balanced_tick(snapshot_at=130, spread=0.003)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],  # +15% would normally be high
        ticks=[first, last],
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "low"
    assert alert.body["consolidation"] is True
    assert alert.body["spread_ratio"] == pytest.approx(306.7, abs=0.5)
    assert alert.body["spread_before"] == pytest.approx(0.92)
    assert alert.body["spread_after"] == pytest.approx(0.003)


async def test_real_move_is_not_consolidation() -> None:
    """Spread ratio 1.25x is below the 5.0 floor → severity follows magnitude."""
    first = _balanced_tick(snapshot_at=70, spread=0.05)
    last = _balanced_tick(snapshot_at=130, spread=0.04)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],  # +15% — high severity
        ticks=[first, last],
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["consolidation"] is False
    assert "spread_ratio" not in alert.body


async def test_zero_last_spread_does_not_raise() -> None:
    """A latest tick with ``spread=0`` cannot be classified — treat as not consolidation."""
    first = _balanced_tick(snapshot_at=70, spread=0.05)
    last = _balanced_tick(snapshot_at=130, spread=0.0)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],
        ticks=[first, last],
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["consolidation"] is False


async def test_single_tick_history_skips_consolidation() -> None:
    """One tick in the window → cannot compute ratio → not consolidation."""
    only = _balanced_tick(snapshot_at=130, spread=0.003)
    ticks = _ticks_mock_with_defaults(
        mids=[(100, 0.40), (130, 0.46)],
        ticks=[only],
    )
    detector, _, _ = _make_detector(ticks=ticks)
    sink, captured = _capturing_sink()

    await detector.evaluate_asset(AssetId("A1"), sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.severity == "high"
    assert alert.body["consolidation"] is False
