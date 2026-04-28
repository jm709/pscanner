"""Unit tests for the monotone detector."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.config import MonotoneConfig
from pscanner.detectors.monotone import (
    AxisSelection,
    MonotoneDetector,
    MonotoneMarket,
    extract_date_axis,
    extract_threshold_axis,
    find_violations,
    select_axis,
)
from pscanner.poly.models import Event, Market
from pscanner.store.repo import AlertsRepo


def test_extract_date_axis_iso() -> None:
    assert extract_date_axis("2026-04-30") == date(2026, 4, 30)


def test_extract_date_axis_iso_with_suffix() -> None:
    """A date prefix wins; trailing text is ignored."""
    assert extract_date_axis("2026-04-30 23:59 UTC") == date(2026, 4, 30)


def test_extract_date_axis_month_day_year() -> None:
    assert extract_date_axis("April 30, 2026") == date(2026, 4, 30)


def test_extract_date_axis_short_month() -> None:
    assert extract_date_axis("Apr 30, 2026") == date(2026, 4, 30)


def test_extract_date_axis_month_day_no_year() -> None:
    """Year defaults to ``year_hint`` when missing from the label."""
    assert extract_date_axis("June 30", year_hint=2026) == date(2026, 6, 30)


def test_extract_date_axis_returns_none_when_no_date() -> None:
    assert extract_date_axis("$1.5T") is None
    assert extract_date_axis("") is None
    assert extract_date_axis(None) is None


def test_extract_date_axis_ordinal_suffix() -> None:
    """Ordinal suffixes (``st``/``nd``/``rd``/``th``) are tolerated."""
    assert extract_date_axis("April 30th", year_hint=2026) == date(2026, 4, 30)
    assert extract_date_axis("July 1st, 2026") == date(2026, 7, 1)


def test_extract_date_axis_month_day_with_trailing_text() -> None:
    """Trailing tokens after the date are ignored (matches ISO behaviour)."""
    assert extract_date_axis("April 30, 2026 resolution") == date(2026, 4, 30)
    assert extract_date_axis("June 30 (deadline)", year_hint=2026) == date(2026, 6, 30)


def test_extract_date_axis_dotted_short_month() -> None:
    """``Apr. 30`` form is accepted (Polymarket label variant)."""
    assert extract_date_axis("Apr. 30, 2026") == date(2026, 4, 30)


def test_extract_date_axis_sept_alias() -> None:
    """``Sept`` is recognised alongside ``Sep`` and ``September``."""
    assert extract_date_axis("Sept 30, 2026") == date(2026, 9, 30)


def test_extract_date_axis_returns_none_when_year_missing_and_no_hint() -> None:
    """Without a ``year_hint``, a year-less label produces ``None``."""
    assert extract_date_axis("April 30th") is None
    assert extract_date_axis("June 30") is None


def test_extract_threshold_above_dollar() -> None:
    assert extract_threshold_axis("Above $1.5T") == (1_500_000_000_000.0, "higher_is_stricter")


def test_extract_threshold_at_least_keyword() -> None:
    assert extract_threshold_axis("At least 5") == (5.0, "higher_is_stricter")


def test_extract_threshold_more_than_keyword() -> None:
    assert extract_threshold_axis("more than 9") == (9.0, "higher_is_stricter")


def test_extract_threshold_ge_symbol() -> None:
    assert extract_threshold_axis(">= 100") == (100.0, "higher_is_stricter")


def test_extract_threshold_geq_unicode() -> None:
    assert extract_threshold_axis("≥ 1000") == (1000.0, "higher_is_stricter")


def test_extract_threshold_below() -> None:
    assert extract_threshold_axis("Below $500B") == (500_000_000_000.0, "lower_is_stricter")


def test_extract_threshold_at_most() -> None:
    assert extract_threshold_axis("at most 3") == (3.0, "lower_is_stricter")


def test_extract_threshold_less_than() -> None:
    assert extract_threshold_axis("Less than 2 seconds") == (2.0, "lower_is_stricter")


def test_extract_threshold_returns_none_for_range_bucket() -> None:
    """Range buckets are mutex partitions, not nested — reject."""
    assert extract_threshold_axis("$1T-$1.25T") is None
    assert extract_threshold_axis("$500B - $750B") is None


def test_extract_threshold_returns_none_for_exactly() -> None:
    """`Exactly N` is mutex with other counts, not nested — reject."""
    assert extract_threshold_axis("Exactly 7") is None


def test_extract_threshold_returns_none_for_plain_number() -> None:
    """A bare number with no direction keyword is ambiguous — reject."""
    assert extract_threshold_axis("$1.5T") is None
    assert extract_threshold_axis("100") is None


def test_extract_threshold_returns_none_for_empty() -> None:
    assert extract_threshold_axis(None) is None
    assert extract_threshold_axis("") is None


def test_extract_threshold_returns_none_for_malformed_number_double_decimal() -> None:
    """``Above 1.2.3`` matches the regex but fails ``float()`` — return ``None``."""
    assert extract_threshold_axis("Above 1.2.3") is None


def test_extract_threshold_returns_none_for_malformed_number_only_separators() -> None:
    """``Above ,,,`` matches the regex but is not a valid float — return ``None``."""
    assert extract_threshold_axis("Above ,,,") is None


# ---------------------------------------------------------------------------
# select_axis tests
# ---------------------------------------------------------------------------


def _market(
    *,
    market_id: str,
    yes_price: float,
    group_item_title: str | None = None,
) -> Market:
    payload: dict[str, Any] = {
        "id": market_id,
        "conditionId": f"0x{market_id}",
        "question": f"Will it happen for {market_id}?",
        "slug": f"slug-{market_id}",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [yes_price, 1.0 - yes_price],
        "groupItemTitle": group_item_title,
    }
    return Market.model_validate(payload)


def test_select_axis_date_sorted_strict_first() -> None:
    markets = [
        _market(market_id="m_jun", yes_price=0.70, group_item_title="June 30, 2026"),
        _market(market_id="m_apr", yes_price=0.07, group_item_title="April 30, 2026"),
    ]
    selection = select_axis(markets, year_hint=2026)
    assert selection is not None
    assert selection.kind == "date"
    assert [m.market.id for m in selection.markets] == ["m_apr", "m_jun"]
    assert selection.markets[0].sort_key < selection.markets[1].sort_key


def test_select_axis_threshold_higher_strict_first() -> None:
    """For ``higher_is_stricter`` axis, the largest value sorts FIRST."""
    markets = [
        _market(market_id="m_low", yes_price=0.40, group_item_title="Above $500B"),
        _market(market_id="m_high", yes_price=0.20, group_item_title="Above $1.5T"),
    ]
    selection = select_axis(markets, year_hint=2026)
    assert selection is not None
    assert selection.kind == "threshold"
    assert selection.direction == "higher_is_stricter"
    assert [m.market.id for m in selection.markets] == ["m_high", "m_low"]


def test_select_axis_threshold_lower_strict_first() -> None:
    """For ``lower_is_stricter`` axis, the smallest value sorts FIRST."""
    markets = [
        _market(market_id="m_2", yes_price=0.05, group_item_title="Less than 2 seconds"),
        _market(market_id="m_10", yes_price=0.20, group_item_title="Less than 10 seconds"),
    ]
    selection = select_axis(markets, year_hint=2026)
    assert selection is not None
    assert selection.direction == "lower_is_stricter"
    assert [m.market.id for m in selection.markets] == ["m_2", "m_10"]


def test_select_axis_drops_unparseable_markets_when_others_succeed() -> None:
    """A non-axis market in a mostly-axis event is dropped, not a skip-trigger."""
    markets = [
        _market(market_id="m_jun", yes_price=0.70, group_item_title="June 30, 2026"),
        _market(market_id="m_other", yes_price=0.15, group_item_title="Doesn't IPO"),
        _market(market_id="m_apr", yes_price=0.07, group_item_title="April 30, 2026"),
    ]
    selection = select_axis(markets, year_hint=2026)
    assert selection is not None
    assert {m.market.id for m in selection.markets} == {"m_apr", "m_jun"}


def test_select_axis_returns_none_when_directions_mixed() -> None:
    """Mixed-direction threshold events have no clean axis."""
    markets = [
        _market(market_id="m_above", yes_price=0.30, group_item_title="Above 5"),
        _market(market_id="m_below", yes_price=0.40, group_item_title="Below 3"),
    ]
    assert select_axis(markets, year_hint=2026) is None


def test_select_axis_returns_none_when_fewer_than_two_match() -> None:
    """A single matchable market can't yield a comparison."""
    markets = [
        _market(market_id="m_apr", yes_price=0.07, group_item_title="April 30, 2026"),
        _market(market_id="m_other", yes_price=0.15, group_item_title="Doesn't IPO"),
    ]
    assert select_axis(markets, year_hint=2026) is None


def test_select_axis_skips_market_without_yes_price() -> None:
    """Empty ``outcome_prices`` is unusable; market is dropped."""
    markets = [
        _market(market_id="m_jun", yes_price=0.70, group_item_title="June 30, 2026"),
        _market(market_id="m_apr", yes_price=0.07, group_item_title="April 30, 2026"),
    ]
    # Force one market to have no prices.
    markets[0] = Market.model_validate(
        {**markets[0].model_dump(by_alias=True), "outcomePrices": []},
    )
    assert select_axis(markets, year_hint=2026) is None


def test_select_axis_returns_none_for_empty_market_list() -> None:
    """An event with zero markets is trivially monotone-ineligible."""
    assert select_axis([], year_hint=2026) is None


def test_axis_selection_rejects_inconsistent_kind_and_direction() -> None:
    """The dataclass rejects ``kind=date`` with a direction or ``kind=threshold`` without."""
    with pytest.raises(ValueError, match="direction"):
        AxisSelection(kind="date", direction="higher_is_stricter", markets=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="direction"):
        AxisSelection(kind="threshold", direction=None, markets=())


# ---------------------------------------------------------------------------
# find_violations tests
# ---------------------------------------------------------------------------


def _selection_for_test(*pairs: tuple[str, float, float]) -> AxisSelection:
    """Build an AxisSelection from (market_id, sort_key, yes_price) tuples.

    The pairs are expected pre-sorted strict-first.
    """
    monotone_markets = tuple(
        MonotoneMarket(
            market=_market(market_id=mid, yes_price=price),
            sort_key=key,
            yes_price=price,
        )
        for (mid, key, price) in pairs
    )
    return AxisSelection(kind="date", direction=None, markets=monotone_markets)


def test_find_violations_none_when_monotone() -> None:
    """Non-decreasing prices in loose-direction → no violations."""
    selection = _selection_for_test(("m_apr", 1.0, 0.10), ("m_jun", 2.0, 0.30))
    assert find_violations(selection, min_violation=0.02) == []


def test_find_violations_flags_strict_richer_than_loose() -> None:
    """Strict leg priced higher than loose leg → violation."""
    selection = _selection_for_test(("m_apr", 1.0, 0.40), ("m_jun", 2.0, 0.30))
    [v] = find_violations(selection, min_violation=0.02)
    assert v.strict.market.id == "m_apr"
    assert v.loose.market.id == "m_jun"
    assert v.gap == pytest.approx(0.10)


def test_find_violations_skips_when_below_tolerance() -> None:
    """Gap smaller than ``min_violation`` is ignored."""
    selection = _selection_for_test(("m_apr", 1.0, 0.31), ("m_jun", 2.0, 0.30))
    assert find_violations(selection, min_violation=0.02) == []


def test_find_violations_emits_one_per_adjacent_pair() -> None:
    """Three markets, two adjacent violations, two records."""
    selection = _selection_for_test(
        ("m1", 1.0, 0.40),
        ("m2", 2.0, 0.30),
        ("m3", 3.0, 0.10),
    )
    violations = find_violations(selection, min_violation=0.02)
    assert [(v.strict.market.id, v.loose.market.id) for v in violations] == [
        ("m1", "m2"),
        ("m2", "m3"),
    ]


def test_find_violations_handles_equal_prices() -> None:
    """Equal prices are not a violation."""
    selection = _selection_for_test(("m_apr", 1.0, 0.30), ("m_jun", 2.0, 0.30))
    assert find_violations(selection, min_violation=0.02) == []


def test_find_violations_includes_borderline_gap_after_rounding() -> None:
    """A true 2-cent gap survives float subtraction noise.

    ``0.30 - 0.28`` in IEEE-754 produces ``0.019999999999999997`` — without
    rounding, ``gap >= 0.02`` would silently exclude this real violation.
    """
    selection = _selection_for_test(("m_a", 1.0, 0.30), ("m_b", 2.0, 0.28))
    [v] = find_violations(selection, min_violation=0.02)
    assert v.gap == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# MonotoneDetector integration tests
# ---------------------------------------------------------------------------


def _event(
    *,
    event_id: str = "e1",
    title: str = "Event",
    liquidity: float | None = 50000.0,
    markets: Iterable[Market] = (),
    tags: list[str] | None = None,
) -> Event:
    payload: dict[str, Any] = {
        "id": event_id,
        "title": title,
        "slug": f"slug-{event_id}",
        "liquidity": liquidity,
        "markets": [m.model_dump(by_alias=True) for m in markets],
    }
    if tags is not None:
        payload["tags"] = tags
    return Event.model_validate(payload)


def _async_iter(events: Iterable[Event]) -> AsyncIterator[Event]:
    async def _gen() -> AsyncIterator[Event]:
        for e in events:
            yield e

    return _gen()


def _make_detector(
    events: Iterable[Event],
    *,
    min_violation: float = 0.02,
    min_event_liquidity_usd: float = 10000.0,
    min_market_liquidity_usd: float = 100.0,
    max_market_count: int = 12,
) -> tuple[MonotoneDetector, AsyncMock]:
    gamma = AsyncMock()
    gamma.iter_events = lambda **_kwargs: _async_iter(list(events))
    config = MonotoneConfig(
        min_violation=min_violation,
        min_event_liquidity_usd=min_event_liquidity_usd,
        min_market_liquidity_usd=min_market_liquidity_usd,
        max_market_count=max_market_count,
    )
    detector = MonotoneDetector(
        config=config,
        gamma_client=gamma,
    )
    return detector, gamma


async def _drain_one_scan(detector: MonotoneDetector, sink: AlertSink) -> None:
    """Run one scan iteration via the ``_scan`` entry-point."""
    await detector._scan(sink)


def _liquid_market(
    market_id: str,
    yes_price: float,
    group_item_title: str,
    *,
    liquidity: float = 5000.0,
) -> Market:
    payload: dict[str, Any] = {
        "id": market_id,
        "conditionId": f"0x{market_id}",
        "question": f"{market_id}?",
        "slug": f"slug-{market_id}",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [yes_price, 1.0 - yes_price],
        "groupItemTitle": group_item_title,
        "liquidity": liquidity,
        "enableOrderBook": True,
    }
    return Market.model_validate(payload)


async def test_scan_emits_alert_for_date_violation(tmp_db: Any) -> None:
    """Bannon-style ``by April 30 / by June 30`` with strict richer than loose."""
    markets = [
        _liquid_market("m_apr", yes_price=0.40, group_item_title="April 30, 2026"),
        _liquid_market("m_jun", yes_price=0.30, group_item_title="June 30, 2026"),
    ]
    event = _event(event_id="ev1", title="Bannon exonerated", markets=markets)
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    rows = repo.recent(limit=10)
    assert len(rows) == 1
    body = rows[0].body
    assert body["axis_kind"] == "date"
    assert body["strict_condition_id"] == "0xm_apr"
    assert body["loose_condition_id"] == "0xm_jun"
    assert body["gap"] == pytest.approx(0.10)


async def test_scan_no_alert_for_monotone_event(tmp_db: Any) -> None:
    markets = [
        _liquid_market("m_apr", yes_price=0.10, group_item_title="April 30, 2026"),
        _liquid_market("m_jun", yes_price=0.30, group_item_title="June 30, 2026"),
    ]
    event = _event(event_id="ev1", markets=markets)
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    assert repo.recent(limit=10) == []


async def test_scan_skips_event_below_liquidity_floor(tmp_db: Any) -> None:
    markets = [
        _liquid_market("m_apr", yes_price=0.40, group_item_title="April 30, 2026"),
        _liquid_market("m_jun", yes_price=0.30, group_item_title="June 30, 2026"),
    ]
    event = _event(event_id="ev1", liquidity=500.0, markets=markets)  # below 10k
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    assert repo.recent(limit=10) == []


async def test_scan_drops_low_liquidity_markets(tmp_db: Any) -> None:
    """Markets below ``min_market_liquidity_usd`` are dropped from comparison."""
    markets = [
        _liquid_market("m_apr", yes_price=0.40, group_item_title="April 30, 2026"),
        _liquid_market("m_jun", yes_price=0.30, group_item_title="June 30, 2026", liquidity=5.0),
    ]
    event = _event(event_id="ev1", markets=markets)
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    # m_jun dropped → only one market left → no comparison possible.
    assert repo.recent(limit=10) == []


async def test_scan_severity_high_above_ten_cents(tmp_db: Any) -> None:
    markets = [
        _liquid_market("m_apr", yes_price=0.50, group_item_title="April 30, 2026"),
        _liquid_market("m_jun", yes_price=0.30, group_item_title="June 30, 2026"),
    ]
    event = _event(event_id="ev1", markets=markets)
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    [row] = repo.recent(limit=10)
    assert row.severity == "high"


async def test_scan_alert_key_collapses_repeated_violations(tmp_db: Any) -> None:
    """Same pair, two scans → one persisted alert (key dedupe via AlertsRepo)."""
    markets = [
        _liquid_market("m_apr", yes_price=0.40, group_item_title="April 30, 2026"),
        _liquid_market("m_jun", yes_price=0.30, group_item_title="June 30, 2026"),
    ]
    event = _event(event_id="ev1", markets=markets)
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    await _drain_one_scan(detector, sink)
    rows = repo.recent(limit=10)
    assert len(rows) == 1


async def test_scan_no_alert_when_axis_extraction_fails(tmp_db: Any) -> None:
    """Two markets with non-axis labels yield no axis selection → no alert."""
    markets = [
        _liquid_market("m_a", yes_price=0.40, group_item_title="Doesn't IPO"),
        _liquid_market("m_b", yes_price=0.30, group_item_title="Other outcome"),
    ]
    event = _event(event_id="ev1", markets=markets)
    detector, _ = _make_detector([event])
    repo = AlertsRepo(tmp_db)
    sink = AlertSink(repo)
    await _drain_one_scan(detector, sink)
    assert repo.recent(limit=10) == []
