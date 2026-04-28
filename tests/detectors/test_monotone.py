"""Unit tests for the monotone detector."""

from __future__ import annotations

from datetime import date

from pscanner.detectors.monotone import extract_date_axis, extract_threshold_axis


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
