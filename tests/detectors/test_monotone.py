"""Unit tests for the monotone detector."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from pscanner.detectors.monotone import (
    AxisSelection,
    extract_date_axis,
    extract_threshold_axis,
    select_axis,
)
from pscanner.poly.models import Market


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
