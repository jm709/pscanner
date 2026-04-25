"""Tests for the mispricing detector."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MispricingConfig
from pscanner.detectors.mispricing import MispricingDetector
from pscanner.poly.models import Event, Market
from pscanner.store.repo import EventOutcomeSumRepo


def _market(
    *,
    market_id: str = "m1",
    question: str = "Will X happen?",
    yes_price: float | None = 0.5,
    enable_order_book: bool = True,
    group_item_title: str | None = None,
    group_item_threshold: str | None = None,
    liquidity: float | None = None,
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
        "groupItemTitle": group_item_title,
        "groupItemThreshold": group_item_threshold,
        "liquidity": liquidity,
    }
    return Market.model_validate(payload)


def _event(
    *,
    event_id: str = "e1",
    title: str = "Event",
    liquidity: float | None = 50000.0,
    markets: Iterable[Market] | None = None,
    tags: list[str] | None = None,
) -> Event:
    """Build a synthetic Event with the provided markets and optional tags."""
    payload: dict[str, Any] = {
        "id": event_id,
        "title": title,
        "slug": f"slug-{event_id}",
        "liquidity": liquidity,
        "markets": [m.model_dump(by_alias=True) for m in (markets or [])],
    }
    if tags is not None:
        payload["tags"] = tags
    return Event.model_validate(payload)


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
    alert_max_deviation: float = 0.5,
    min_event_liquidity_usd: float = 10000.0,
    min_market_liquidity_usd: float = 0.0,
    sum_history_repo: EventOutcomeSumRepo | None = None,
) -> tuple[MispricingDetector, AsyncMock]:
    """Construct a detector wired to a mocked GammaClient and a repo.

    ``sum_history_repo`` defaults to a stub repo. Tests that need to assert on
    persisted rows pass an explicit repo backed by the ``tmp_db`` fixture so
    cleanup is handled by pytest.
    """
    gamma = AsyncMock()
    gamma.iter_events = lambda **_kwargs: _async_iter(list(events))
    config = MispricingConfig(
        sum_deviation_threshold=sum_deviation_threshold,
        alert_max_deviation=alert_max_deviation,
        min_event_liquidity_usd=min_event_liquidity_usd,
        min_market_liquidity_usd=min_market_liquidity_usd,
    )
    repo = sum_history_repo if sum_history_repo is not None else _StubSumRepo()
    detector = MispricingDetector(
        config=config,
        gamma_client=gamma,
        sum_history_repo=cast(EventOutcomeSumRepo, repo),
    )
    return detector, gamma


class _StubSumRepo:
    """Minimal in-process stand-in for EventOutcomeSumRepo (no SQLite)."""

    def __init__(self) -> None:
        self.rows: list[Any] = []

    def insert(self, row: Any) -> bool:
        self.rows.append(row)
        return True


def _count_history(repo: EventOutcomeSumRepo | _StubSumRepo) -> int:
    """Count all rows in the repo (test helper)."""
    if isinstance(repo, _StubSumRepo):
        return len(repo.rows)
    return len(repo.recent(limit=10_000))


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


async def test_iso_date_bucket_event_is_skipped() -> None:
    """ISO-format date-bucket layouts are skipped."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="2025-12-31"),
        _market(market_id="m2", yes_price=0.5, group_item_title="2026-12-31"),
        _market(market_id="m3", yes_price=0.5, group_item_title="2027-12-31"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_month_name_date_bucket_event_is_skipped() -> None:
    """Month-name date-bucket layouts are skipped."""
    markets = [
        _market(market_id="m1", yes_price=0.75, group_item_title="December 31, 2025"),
        _market(market_id="m2", yes_price=0.75, group_item_title="December 31, 2026"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_numeric_threshold_bucket_event_is_skipped() -> None:
    """Numeric/dollar-threshold bucket layouts are skipped."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="$300M"),
        _market(market_id="m2", yes_price=0.5, group_item_title=">$1B"),
        _market(market_id="m3", yes_price=0.5, group_item_title="<$100M"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_candidate_mutex_event_with_arbitrary_names_alerts() -> None:
    """Candidate-style mutex events (arbitrary string titles) still alert."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Trump"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Harris"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Other"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["price_sum"] == pytest.approx(1.1)


async def test_mixed_bucket_and_non_bucket_titles_alerts() -> None:
    """One non-bucket title means the whole event is treated as eligible."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="December 31, 2025"),
        _market(market_id="m2", yes_price=0.4, group_item_title="December 31, 2026"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Other"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["price_sum"] == pytest.approx(1.1)


async def test_all_empty_group_item_title_event_alerts() -> None:
    """Regression: events with no groupItemTitle on any market still alert."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title=None),
        _market(market_id="m2", yes_price=0.4, group_item_title=None),
        _market(market_id="m3", yes_price=0.2, group_item_title=None),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["price_sum"] == pytest.approx(1.1)


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


async def test_event_tagged_sports_is_skipped() -> None:
    """Sports-tagged events skip even with arbitrary mutex-looking titles."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Team A"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Team B"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Team C"),
    ]
    event = _event(markets=markets, tags=["Sports"])
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_tagged_esports_is_skipped() -> None:
    """Esports-tagged events are also skipped."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Team Liquid"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Team Falcons"),
    ]
    event = _event(markets=markets, tags=["Esports", "Dota 2"])
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_with_non_excluded_tags_remains_eligible() -> None:
    """Events tagged with non-excluded labels still alert when mispriced."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Trump"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Harris"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Other"),
    ]
    event = _event(markets=markets, tags=["Politics", "Election"])
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["price_sum"] == pytest.approx(1.1)


async def test_event_tag_match_is_case_insensitive() -> None:
    """Lowercase ``sports`` still matches the excluded ``Sports`` tag."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Team A"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Team B"),
    ]
    event = _event(markets=markets, tags=["sports"])
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_event_with_no_tags_field_unchanged() -> None:
    """Events without a tags field behave exactly as before."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Trump"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Harris"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Other"),
    ]
    event = _event(markets=markets)
    assert event.tags == []
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["price_sum"] == pytest.approx(1.1)


async def test_above_dollar_threshold_buckets_are_skipped() -> None:
    """``Above $300M`` / ``Above $500M`` / ``Above $1B`` is a pure threshold layout."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Above $300M"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Above $500M"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Above $1B"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_below_numeric_buckets_are_skipped() -> None:
    """``Below 100`` / ``Below 200`` / ``Below 500`` is a pure threshold layout."""
    markets = [
        _market(market_id="m1", yes_price=0.5, group_item_title="Below 100"),
        _market(market_id="m2", yes_price=0.4, group_item_title="Below 200"),
        _market(market_id="m3", yes_price=0.2, group_item_title="Below 500"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_at_least_at_most_buckets_are_skipped() -> None:
    """``At least 5`` / ``At most 10`` are recognised as range keywords."""
    markets = [
        _market(market_id="m1", yes_price=0.7, group_item_title="At least 5"),
        _market(market_id="m2", yes_price=0.5, group_item_title="At most 10"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_unicode_inequality_buckets_are_skipped() -> None:
    """``≥ 10`` and ``<= 5`` are recognised as range keywords."""
    markets = [
        _market(market_id="m1", yes_price=0.7, group_item_title="≥ 10"),
        _market(market_id="m2", yes_price=0.5, group_item_title="<= 5"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert captured == []


async def test_mixed_above_dollar_and_other_alerts() -> None:
    """``Above $300M`` mixed with ``Other`` is not a pure range — eligible."""
    markets = [
        _market(market_id="m1", yes_price=0.6, group_item_title="Above $300M"),
        _market(market_id="m2", yes_price=0.6, group_item_title="Other"),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event])
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert len(captured) == 1
    assert captured[0].body["price_sum"] == pytest.approx(1.2)


async def test_history_row_written_for_eligible_event_within_band(
    tmp_db: sqlite3.Connection,
) -> None:
    """An eligible in-band event records exactly one history row + one alert."""
    repo = EventOutcomeSumRepo(tmp_db)
    markets = [
        _market(market_id="m1", yes_price=0.6),
        _market(market_id="m2", yes_price=0.6),  # Σ = 1.2, |dev| = 0.2
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector([event], sum_history_repo=repo)
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert _count_history(repo) == 1
    rows = repo.by_event_id(event.id)
    assert len(rows) == 1
    assert rows[0].market_count == 2
    assert rows[0].price_sum == pytest.approx(1.2)
    assert rows[0].deviation == pytest.approx(0.2)
    assert len(captured) == 1


async def test_alert_cap_skips_alert_but_writes_history(
    tmp_db: sqlite3.Connection,
) -> None:
    """Σ above ``alert_max_deviation`` writes history but emits no alert."""
    repo = EventOutcomeSumRepo(tmp_db)
    # Five markets at 0.5 → Σ = 2.5, |deviation| = 1.5 > alert_max_deviation=0.5
    markets = [_market(market_id=f"m{i}", yes_price=0.5) for i in range(5)]
    event = _event(markets=markets)
    detector, _ = _make_detector(
        [event],
        alert_max_deviation=0.5,
        sum_history_repo=repo,
    )
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert _count_history(repo) == 1
    history = repo.by_event_id(event.id)
    assert len(history) == 1
    assert history[0].price_sum == pytest.approx(2.5)
    assert history[0].deviation == pytest.approx(1.5)
    assert captured == []


async def test_below_threshold_writes_history_but_no_alert(
    tmp_db: sqlite3.Connection,
) -> None:
    """Σ inside the deadband writes history but emits no alert."""
    repo = EventOutcomeSumRepo(tmp_db)
    # Σ = 1.01, |deviation| = 0.01 < sum_deviation_threshold=0.03
    markets = [
        _market(market_id="m1", yes_price=0.51),
        _market(market_id="m2", yes_price=0.50),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector(
        [event],
        sum_deviation_threshold=0.03,
        sum_history_repo=repo,
    )
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert _count_history(repo) == 1
    assert captured == []


async def test_per_market_liquidity_floor_skips_event_and_history(
    tmp_db: sqlite3.Connection,
) -> None:
    """Any market below ``min_market_liquidity_usd`` skips eligibility entirely."""
    repo = EventOutcomeSumRepo(tmp_db)
    markets = [
        _market(market_id="m1", yes_price=0.6, liquidity=50.0),  # below floor
        _market(market_id="m2", yes_price=0.6, liquidity=5000.0),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector(
        [event],
        min_market_liquidity_usd=100.0,
        sum_history_repo=repo,
    )
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert _count_history(repo) == 0
    assert captured == []


async def test_per_market_liquidity_floor_disabled_when_zero(
    tmp_db: sqlite3.Connection,
) -> None:
    """Default ``min_market_liquidity_usd=0`` keeps backward-compatible behaviour."""
    repo = EventOutcomeSumRepo(tmp_db)
    markets = [
        _market(market_id="m1", yes_price=0.6, liquidity=None),
        _market(market_id="m2", yes_price=0.6, liquidity=None),
    ]
    event = _event(markets=markets)
    detector, _ = _make_detector(
        [event],
        min_market_liquidity_usd=0.0,
        sum_history_repo=repo,
    )
    sink, captured = _capturing_sink()

    await detector._scan(sink)

    assert _count_history(repo) == 1
    assert len(captured) == 1


async def test_history_uses_real_repo_persists_via_db(
    tmp_db: sqlite3.Connection,
) -> None:
    """Smoke: a real repo + in-memory db round-trips a history row."""
    repo = EventOutcomeSumRepo(tmp_db)
    markets = [
        _market(market_id="m1", yes_price=0.6),
        _market(market_id="m2", yes_price=0.6),
    ]
    event = _event(event_id="evt-real", markets=markets)
    detector, _ = _make_detector([event], sum_history_repo=repo)
    sink, _ = _capturing_sink()

    await detector._scan(sink)

    rows = tmp_db.execute(
        "SELECT event_id, market_count, price_sum FROM event_outcome_sum_history",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event_id"] == "evt-real"
    assert rows[0]["market_count"] == 2
    assert rows[0]["price_sum"] == pytest.approx(1.2)


class _StopLoop(BaseException):
    """Sentinel used to escape the otherwise-infinite ``run`` loop in tests."""
