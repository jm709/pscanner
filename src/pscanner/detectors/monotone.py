"""Monotone-arbitrage detector — flags adjacent-axis monotonicity violations.

Within an event whose markets line up on a sortable axis (date deadline or
single-direction threshold), the no-arb constraint ``P(strict) <= P(loose)``
must hold for every adjacent pair (where "strict" means earlier deadline or
higher threshold, depending on direction). A violation lets a trader buy NO
on the strict leg and YES on the loose leg for a guaranteed profit equal to
the gap, in every world.

Axis extraction operates per-market; markets where extraction fails are
dropped from the comparison (not reasons to skip the whole event).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

_ISO_DATE_PATTERN = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_MONTH_NAMES: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_DAY_YEAR_PATTERN = re.compile(
    r"^(?P<month>[A-Za-z]+)\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?\b",
)


def _parse_iso_date(text: str) -> date | None:
    """Return a date from an ISO-prefixed string, or None."""
    iso_match = _ISO_DATE_PATTERN.match(text)
    if iso_match is None:
        return None
    try:
        return date(
            int(iso_match.group("y")),
            int(iso_match.group("m")),
            int(iso_match.group("d")),
        )
    except ValueError:
        return None


def _parse_month_day_date(text: str, year_hint: int | None) -> date | None:
    """Return a date from a ``Month Day[, Year]`` string, or None."""
    month_match = _MONTH_DAY_YEAR_PATTERN.match(text)
    if month_match is None:
        return None
    month_token = month_match.group("month").lower()
    month_num = _MONTH_NAMES.get(month_token)
    if month_num is None:
        return None
    year_token = month_match.group("year")
    if year_token is not None:
        year = int(year_token)
    elif year_hint is not None:
        year = year_hint
    else:
        return None
    try:
        return date(year, month_num, int(month_match.group("day")))
    except ValueError:
        return None


def extract_date_axis(label: str | None, *, year_hint: int | None = None) -> date | None:
    """Extract a sortable :class:`date` from a market label.

    Recognised formats:

    * ISO: ``2026-04-30`` (with optional trailing text).
    * ``Month Day[, Year]``: ``April 30, 2026`` / ``Apr. 30, 2026`` /
      ``April 30th`` / ``Sept 30``. Trailing text is ignored. When the
      year is omitted, ``year_hint`` is used; without a hint, the
      function returns ``None``.

    Args:
        label: Candidate string (typically ``groupItemTitle``).
        year_hint: Year to assume when the label omits it.

    Returns:
        A :class:`date`, or ``None`` if no date is recognised.
    """
    if not label:
        return None
    text = label.strip()
    if not text:
        return None
    return _parse_iso_date(text) or _parse_month_day_date(text, year_hint)


ThresholdDirection = Literal["higher_is_stricter", "lower_is_stricter"]

_HIGHER_KEYWORDS = (
    # Two-char operators must precede single-char to ensure correct alternation match.
    r">=",
    r">",
    r"≥",
    r"above",
    r"over",
    r"at\s+least",
    r"more\s+than",
)
_LOWER_KEYWORDS = (
    # Two-char operators must precede single-char to ensure correct alternation match.
    r"<=",
    r"<",
    r"≤",
    r"below",
    r"under",
    r"at\s+most",
    r"less\s+than",
)
_HIGHER_PATTERN = re.compile(
    r"^(?:" + r"|".join(_HIGHER_KEYWORDS) + r")\s*\$?(?P<num>[\d,.]+)\s*(?P<suffix>[KMBT]?)",
    re.IGNORECASE,
)
_LOWER_PATTERN = re.compile(
    r"^(?:" + r"|".join(_LOWER_KEYWORDS) + r")\s*\$?(?P<num>[\d,.]+)\s*(?P<suffix>[KMBT]?)",
    re.IGNORECASE,
)
# Range bucket: ``$1T-$1.25T``, ``$500B - $750B``. Reject — mutex, not nested.
_RANGE_BUCKET_PATTERN = re.compile(
    r"^\$?[\d,.]+\s*[KMBT]?\s*[-–]\s*\$?[\d,.]+",  # noqa: RUF001 - en-dash is intentional
)
_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "": 1.0,
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}


def extract_threshold_axis(label: str | None) -> tuple[float, ThresholdDirection] | None:
    """Extract a numeric threshold + direction from a market label.

    Recognised forms:

    * ``Above|Over|At least|More than|>=|>|≥`` <number> → ``higher_is_stricter``
    * ``Below|Under|At most|Less than|<=|<|≤`` <number> → ``lower_is_stricter``

    Range buckets (``$1T-$1.25T``), bare numbers, and ``exactly N`` are not
    recognised — those are mutex partitions, not nested events.

    Numeric suffixes ``K``, ``M``, ``B``, ``T`` are honoured (case-insensitive).

    Trailing text after the number and optional suffix is ignored
    (e.g. ``"Less than 2 seconds"`` parses to ``(2.0, "lower_is_stricter")``).

    Args:
        label: Candidate string (typically ``groupItemTitle``).

    Returns:
        ``(value, direction)`` tuple, or ``None`` when no monotone-eligible
        threshold is recognised.
    """
    if not label:
        return None
    text = label.strip()
    if not text:
        return None
    if _RANGE_BUCKET_PATTERN.match(text):
        return None
    higher_match = _HIGHER_PATTERN.match(text)
    if higher_match is not None:
        value = _parse_number(higher_match)
        if value is not None:
            return (value, "higher_is_stricter")
    else:
        lower_match = _LOWER_PATTERN.match(text)
        if lower_match is not None:
            value = _parse_number(lower_match)
            if value is not None:
                return (value, "lower_is_stricter")
    return None


def _parse_number(match: re.Match[str]) -> float | None:
    """Combine a regex-matched number and KMBT suffix into a float.

    Returns ``None`` when the captured digit/comma/dot run isn't a valid
    Python float (e.g. ``"1.2.3"``, ``",,,"``); the regex is broad enough
    to admit those patterns but ``float()`` rejects them.
    """
    raw = match.group("num").replace(",", "")
    try:
        base = float(raw)
    except ValueError:
        return None
    suffix = match.group("suffix").lower()
    return base * _SUFFIX_MULTIPLIERS[suffix]
