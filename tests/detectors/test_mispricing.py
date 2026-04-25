"""Tests for the mispricing detector."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MispricingConfig
from pscanner.detectors.mispricing import MispricingDetector
from pscanner.poly.models import Event, Market


def _market(
    *,
    market_id: str = "m1",
    question: str = "Will X happen?",
    yes_price: float | None = 0.5,
    enable_order_book: bool = True,
) -> Market:
    """Build a synthetic Market with optional YES price.

    A ``yes_price`` of ``None`` produces an empty ``outcome_prices`` list,
    matching the malformed-payload edge case the detector must tolerate.
    """
    payload: dict[str, Any] = {
        "id": market_id,
        "question": question,
        "slug": f"slug-{market_id}",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [] if yes_price is None else [yes_price, 1.0 - yes_price],
        "enableOrderBook": enable_order_book,
    }
    return Market.model_validate(payload)


def _event(
    *,
    event_id: str = "e1",
    title: str = "Event",
    liquidity: float | None = 50000.0,
    markets: Iterable[Market] | None = None,
) -> Event:
    """Build a synthetic Event with the provided markets."""
    return Event.model_validate(
        {
            "id": event_id,
            "title": title,
            "slug": f"slug-{event_id}",
            "liquidity": liquidity,
            "markets": [m.model_dump(by_alias=True) for m in (markets or [])],
        }
    )


def _async_iter(events: Iterable[Event]) -> AsyncIterator[Event]:
    """Wrap a sync iterable as an async iterator for ``iter_events`` mocks."""

    async def _gen() -> AsyncIterator[Event]:
        for event in events:
            yield event

    return _gen()


def _make_detector(
    events: Iterable[Event],
    *,
    sum_deviation_threshold: float = 0.03,
    min_event_liquidity_usd: float = 10000.0,
) -> tuple[MispricingDetector, AsyncMock]:
    """Construct a detector wired to a mocked GammaClient."""
    gamma = AsyncMock()
    gamma.iter_events = lambda **_kwargs: _async_iter(list(events))
    config = MispricingConfig(
        sum_deviation_threshold=sum_deviation_threshold,
        min_event_liquidity_usd=min_event_liquidity_usd,
    )
    return MispricingDetector(config=config, gamma_client=gamma), gamma


def _capturing_sink() -> tuple[AsyncMock, list[Alert]]:
    """Return an AsyncMock sink that records every emitted Alert."""
    captured: list[Alert] = []

    async def _emit(alert: Alert) -> bool:
        captured.append(alert)
        return True

    sink = AsyncMock()
    sink.emit.side_effect = _emit
    return sink, captured


async def test_three_market_event_sum_above_threshold_alerts_high() -> None:
    markets = [
        _market(market_id="m1", yes_price=0.5),
        _market(market_id="m2", yes_price=0.4),
        _market(market_id="m3", yes_price=0.2),
    ]
    event = _event(markets=markets)
    detector, _gamma = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    alert = captured[0]
    assert alert.detector == "mispricing"
    assert alert.severity == "high"
    assert alert.alert_key == "mispricing:e1:1.1"
    assert alert.body["market_count"] == 3
    assert alert.body["price_sum"] == pytest.approx(1.1)
    assert alert.body["deviation"] == pytest.approx(0.1)
    assert alert.title.startswith("Event")
    assert "1.100" in alert.title


async def test_event_summing_to_one_does_not_alert() -> None:
    markets = [
        _market(market_id="m1", yes_price=0.6),
        _market(market_id="m2", yes_price=0.4),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []
    sink.emit.assert_not_called()


async def test_event_within_threshold_does_not_alert() -> None:
    # sum = 0.99, deviation = 0.01 < 0.03 threshold
    markets = [
        _market(market_id="m1", yes_price=0.5),
        _market(market_id="m2", yes_price=0.49),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_with_order_book_disabled_market_is_skipped() -> None:
    markets = [
        _market(market_id="m1", yes_price=0.5),
        _market(market_id="m2", yes_price=0.4, enable_order_book=False),
        _market(market_id="m3", yes_price=0.2),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_below_min_liquidity_is_skipped() -> None:
    markets = [
        _market(market_id="m1", yes_price=0.5),
        _market(market_id="m2", yes_price=0.4),
        _market(market_id="m3", yes_price=0.2),
    ]
    event = _event(markets=markets, liquidity=500.0)
    detector, _ = _make_detector([event], min_event_liquidity_usd=10000.0)
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_with_none_liquidity_is_skipped() -> None:
    markets = [
        _market(market_id="m1", yes_price=0.5),
        _market(market_id="m2", yes_price=0.4),
    ]
    event = _event(markets=markets, liquidity=None)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_with_single_market_is_skipped() -> None:
    markets = [_market(market_id="m1", yes_price=0.5)]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_malformed_market_kept_when_two_others_remain_valid() -> None:
    # Three markets, one with empty outcome_prices. Remaining two sum to 1.2.
    markets = [
        _market(market_id="m1", yes_price=0.6),
        _market(market_id="m2", yes_price=0.6),
        _market(market_id="m3", yes_price=None),  # empty outcome_prices
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["market_count"] == 2
    assert captured[0].body["price_sum"] == pytest.approx(1.2)
    # The malformed market still appears in the per-market summary with None price.
    yes_prices = [m["yes_price"] for m in captured[0].body["markets"]]
    assert None in yes_prices


async def test_event_with_too_few_valid_markets_is_skipped() -> None:
    # Two markets, but only one has outcome prices, so valid count == 1.
    markets = [
        _market(market_id="m1", yes_price=0.5),
        _market(market_id="m2", yes_price=None),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_severity_boundary_just_above_med_threshold() -> None:
    # deviation = 0.06 -> magnitude > 0.05 -> "med"
    markets = [
        _market(market_id="m1", yes_price=0.53),
        _market(market_id="m2", yes_price=0.53),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].severity == "med"
    assert captured[0].body["deviation"] == pytest.approx(0.06)


async def test_severity_boundary_just_above_high_threshold() -> None:
    # deviation = 0.11 -> magnitude > 0.10 -> "high"
    markets = [
        _market(market_id="m1", yes_price=0.555),
        _market(market_id="m2", yes_price=0.555),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].severity == "high"
    assert captured[0].body["deviation"] == pytest.approx(0.11)


async def test_severity_low_when_just_above_main_threshold() -> None:
    # deviation = 0.04 -> magnitude > 0.03 threshold but <= 0.05 -> "low"
    markets = [
        _market(market_id="m1", yes_price=0.52),
        _market(market_id="m2", yes_price=0.52),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].severity == "low"


async def test_negative_deviation_below_one_alerts_with_correct_severity() -> None:
    # sum = 0.85, deviation = -0.15 -> magnitude 0.15 -> "high"
    markets = [
        _market(market_id="m1", yes_price=0.45),
        _market(market_id="m2", yes_price=0.4),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].severity == "high"
    assert captured[0].body["deviation"] == pytest.approx(-0.15)


async def test_alert_key_uses_rounded_sum_for_dedupe() -> None:
    markets = [
        _market(market_id="m1", yes_price=0.55),
        _market(market_id="m2", yes_price=0.553),
    ]
    event = _event(event_id="abc", markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].alert_key == "mispricing:abc:1.1"


async def test_run_invokes_scan_and_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` calls ``_scan`` repeatedly and sleeps between iterations."""
    detector, _ = _make_detector([])
    sink, _captured = _capturing_sink()

    sleeps: list[float] = []
    scans: list[int] = []

    async def fake_scan(_self: MispricingDetector, _sink: Any) -> None:
        scans.append(len(scans))
        if len(scans) >= 2:
            raise StopAsyncIteration  # pragma: no cover - sentinel below

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _StopLoop

    monkeypatch.setattr(MispricingDetector, "_scan", fake_scan)
    monkeypatch.setattr("pscanner.detectors.mispricing.asyncio.sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        await detector.run(sink)

    assert len(scans) >= 2
    assert sleeps == [300, 300]


async def test_run_logs_and_continues_on_scan_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single scan raising must not break the loop."""
    detector, _ = _make_detector([])
    sink, _ = _capturing_sink()

    calls: list[int] = []

    async def fake_scan(_self: MispricingDetector, _sink: Any) -> None:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("transient gamma error")

    async def fake_sleep(_seconds: float) -> None:
        if len(calls) >= 2:
            raise _StopLoop

    monkeypatch.setattr(MispricingDetector, "_scan", fake_scan)
    monkeypatch.setattr("pscanner.detectors.mispricing.asyncio.sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        await detector.run(sink)

    assert len(calls) >= 2


class _StopLoop(BaseException):
    """Sentinel used to escape the otherwise-infinite ``run`` loop in tests."""
