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
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pscanner.poly.models import Market

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


@dataclass(frozen=True, slots=True)
class MonotoneMarket:
    """One market that survived axis extraction.

    ``sort_key`` is the raw axis value:

    * Date axis: ``date.toordinal()`` — smaller = earlier = stricter.
    * Threshold axis: the numeric threshold value. For
      ``higher_is_stricter`` selections the markets list is sorted
      *descending* (largest key first = strictest); for
      ``lower_is_stricter`` it is sorted *ascending* (smallest key
      first = strictest). Don't compare sort_keys across markets
      without knowing the axis direction.

    ``yes_price`` is the YES-leg price as published in the snapshot.
    """

    market: Market
    sort_key: float
    yes_price: float


@dataclass(frozen=True, slots=True)
class AxisSelection:
    """A monotone-eligible event view with markets sorted strict-first."""

    kind: Literal["date", "threshold"]
    direction: ThresholdDirection | None
    markets: tuple[MonotoneMarket, ...]

    def __post_init__(self) -> None:
        """Enforce the kind/direction invariant set by the constructors."""
        if self.kind == "date" and self.direction is not None:
            msg = f"date-axis AxisSelection must have direction=None, got {self.direction!r}"
            raise ValueError(msg)
        if self.kind == "threshold" and self.direction is None:
            msg = "threshold-axis AxisSelection requires a direction"
            raise ValueError(msg)


def _yes_price(market: Market) -> float | None:
    """Return market's YES price (``outcome_prices[0]``) or None if missing."""
    if not market.outcome_prices:
        return None
    return market.outcome_prices[0]


_GAP_ROUND_DIGITS = 6
"""Decimal precision for ``MonotoneViolation.gap``.

Polymarket prices have at most 4 decimal places; rounding to 6 absorbs
the IEEE-754 noise from subtraction without losing any meaningful
precision. This makes the ``gap >= min_violation`` boundary
deterministic regardless of which specific prices were subtracted.
"""

# Minimum number of markets required to form a meaningful monotone axis.
_MIN_AXIS_MARKETS = 2


def _try_date_axis(markets: list[Market], year_hint: int) -> AxisSelection | None:
    """Attempt to build a date-axis selection from the market list.

    Returns an :class:`AxisSelection` with ``kind="date"`` when at least two
    markets yield parseable dates, else ``None``.
    """
    extracted: list[MonotoneMarket] = []
    for market in markets:
        price = _yes_price(market)
        if price is None:
            continue
        parsed = extract_date_axis(market.group_item_title, year_hint=year_hint)
        if parsed is None:
            continue
        extracted.append(
            MonotoneMarket(market=market, sort_key=parsed.toordinal(), yes_price=price),
        )
    if len(extracted) < _MIN_AXIS_MARKETS:
        return None
    ordered = tuple(sorted(extracted, key=lambda m: m.sort_key))
    return AxisSelection(kind="date", direction=None, markets=ordered)


def _try_threshold_axis(markets: list[Market]) -> AxisSelection | None:
    """Attempt to build a single-direction threshold-axis selection.

    Returns an :class:`AxisSelection` with ``kind="threshold"`` when at least
    two markets share the same direction, else ``None``.
    """
    buckets: dict[ThresholdDirection, list[MonotoneMarket]] = {
        "higher_is_stricter": [],
        "lower_is_stricter": [],
    }
    for market in markets:
        price = _yes_price(market)
        if price is None:
            continue
        result = extract_threshold_axis(market.group_item_title)
        if result is None:
            continue
        value, direction = result
        buckets[direction].append(
            MonotoneMarket(market=market, sort_key=value, yes_price=price),
        )
    # higher_is_stricter is checked first; on a tie at the min-axis-markets
    # threshold it wins by virtue of dict insertion order.
    for direction, group in buckets.items():
        if len(group) >= _MIN_AXIS_MARKETS:
            reverse = direction == "higher_is_stricter"
            ordered = tuple(sorted(group, key=lambda m: m.sort_key, reverse=reverse))
            return AxisSelection(kind="threshold", direction=direction, markets=ordered)
    return None


def select_axis(markets: list[Market], *, year_hint: int) -> AxisSelection | None:
    """Pick an event-wide axis (date or single-direction threshold).

    Tries date first. Markets where extraction fails are dropped, not
    reasons to skip the whole event. Returns ``None`` when fewer than two
    markets remain on a consistent axis.

    Args:
        markets: All markets in the event.
        year_hint: Year passed to :func:`extract_date_axis` for labels
            without an explicit year.
    """
    return _try_date_axis(markets, year_hint) or _try_threshold_axis(markets)


@dataclass(frozen=True, slots=True)
class MonotoneViolation:
    """One adjacent-pair violation of the monotone constraint.

    ``strict`` is the leg that should be cheaper (earlier deadline or
    stricter threshold); ``loose`` is the leg that should be richer.
    ``gap`` is ``round(strict.yes_price - loose.yes_price, 6)``, ensuring the
    boundary is deterministic across float-subtraction noise.
    """

    strict: MonotoneMarket
    loose: MonotoneMarket
    gap: float


def find_violations(
    selection: AxisSelection,
    *,
    min_violation: float,
) -> list[MonotoneViolation]:
    """Find adjacent-pair monotonicity violations in a sorted selection.

    Args:
        selection: Axis-sorted markets (strict-first order).
        min_violation: Minimum ``gap`` value that counts as a violation.

    Returns:
        One :class:`MonotoneViolation` per adjacent pair where the strict
        leg's YES price exceeds the loose leg's by at least ``min_violation``.
    """
    violations: list[MonotoneViolation] = []
    pairs = zip(selection.markets, selection.markets[1:], strict=False)
    for strict, loose in pairs:
        gap = round(strict.yes_price - loose.yes_price, _GAP_ROUND_DIGITS)
        if gap >= min_violation:
            violations.append(MonotoneViolation(strict=strict, loose=loose, gap=gap))
    return violations
