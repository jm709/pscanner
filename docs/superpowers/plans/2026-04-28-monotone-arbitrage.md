# Monotone-Arbitrage Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `monotone` detector + paper-trading evaluator that flags within-event logical-implication arbitrage on date-axis or threshold-axis markets, where the prices on a sortable axis violate the no-arbitrage constraint `P(strict event) ≤ P(loose event)`.

**Architecture:** New polling detector iterates the gamma `/events` catalog. For each event, attempt to extract a sortable axis from each market's `groupItemTitle` (date or single-direction threshold). Markets where extraction fails are dropped. If at least 2 markets remain with consistent axis kind and direction, sort them along the axis and alert when adjacent pairs violate monotonicity (i.e. the stricter market trades richer than the looser one). The evaluator emits a 2-leg paired trade (NO on strict + YES on loose), reusing the existing twin-trade pattern from `VelocityEvaluator`.

**Tech Stack:** Python 3.13 · pydantic 2 · pytest · structlog. Reuses `PollingDetector` base, `AlertSink`, `MarketCacheRepo`, `ParsedSignal` Protocol, and `PaperTrader` machinery already in the repo.

**Out of scope (deferred to a later plan):**
- **Cross-event monotone pairing** (e.g. separate "BTC > $70k by 1 month" and "BTC > $70k by 2 months" events with different `event_id`s). Requires a market-pairing layer; build after within-event signal stabilizes.
- **Removing or demoting `MispricingDetector`.** The two detectors look at different invariants. Operator can disable mispricing via `[mispricing] enabled = false` if desired; this plan is purely additive.
- **Range-bucket events** (e.g. OpenAI cap "$X-$Y" buckets). These are mutex range partitions, not monotone-nested — they belong to a future plan that targets the Σ=1 invariant on negRisk events.
- **Persisting non-violation snapshots** (analogue of `event_outcome_sum_history`). Add only if research demand emerges.

**File structure:**
- Create: `src/pscanner/detectors/monotone.py` — detector + axis-extraction + violation-finding helpers.
- Create: `src/pscanner/strategies/evaluators/monotone.py` — 2-leg paired-trade evaluator.
- Create: `tests/detectors/test_monotone.py` — unit tests for detector + helpers.
- Create: `tests/strategies/evaluators/test_monotone.py` — unit tests for evaluator.
- Modify: `src/pscanner/alerts/models.py` — add `"monotone"` to `DetectorName` Literal.
- Modify: `src/pscanner/config.py` — add `MonotoneConfig` + `MonotoneEvaluatorConfig`; wire into `Config` + `EvaluatorsConfig`.
- Modify: `src/pscanner/strategies/evaluators/__init__.py` — export `MonotoneEvaluator`.
- Modify: `src/pscanner/scheduler.py` — instantiate detector in `_build_detectors`, append evaluator in `_build_paper_evaluators`, add `_run_once_monotone` analogous to `_run_once_mispricing`.
- Modify: `config.toml` — add `[monotone]` and `[paper_trading.evaluators.monotone]` sections.
- Modify: `tests/test_config.py` — assert defaults for new config sections.
- Modify: `tests/test_scheduler.py` — assert detector + evaluator wired when enabled.

**Side-string convention:** Use uppercase `"YES"` / `"NO"` (matching the existing `MispricingEvaluator`, which already round-trips correctly through `PaperTrader._resolve_outcome`).

---

## Task 1: Register detector name + config skeleton

**Files:**
- Modify: `src/pscanner/alerts/models.py` (line 12-20)
- Modify: `src/pscanner/config.py` (add `MonotoneConfig` near `MispricingConfig`, line 59-89; add to `Config` near line 392-407)
- Modify: `tests/test_config.py` (add a default-roundtrip test for the new section)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_config_default_monotone_section() -> None:
    """Monotone defaults match the documented values."""
    cfg = Config()
    assert cfg.monotone.enabled is True
    assert cfg.monotone.scan_interval_seconds == 300
    assert cfg.monotone.min_violation == 0.02
    assert cfg.monotone.min_event_liquidity_usd == 10000.0
    assert cfg.monotone.min_market_liquidity_usd == 100.0
    assert cfg.monotone.max_market_count == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_config_default_monotone_section -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'monotone'`

- [ ] **Step 3: Add `"monotone"` to `DetectorName`**

In `src/pscanner/alerts/models.py`, replace lines 12-20 with:

```python
DetectorName = Literal[
    "smart_money",
    "mispricing",
    "monotone",
    "whales",
    "convergence",
    "velocity",
    "cluster",
    "move_attribution",
]
```

- [ ] **Step 4: Add `MonotoneConfig` to config**

In `src/pscanner/config.py`, after the `MispricingConfig` class (after line 89), add:

```python
class MonotoneConfig(_Section):
    """Thresholds for the monotone-arbitrage detector.

    Within an event, sort markets along an extracted axis (date deadline
    or single-direction threshold) and alert when adjacent pairs violate
    ``P(strict) <= P(loose)``. Markets where axis extraction fails are
    dropped (not reasons to skip the whole event).

    ``min_violation`` is the minimum gap ``P(strict) - P(loose)`` that
    will produce an alert. ``min_market_liquidity_usd`` filters illiquid
    legs whose snapshot prices are likely stale; markets below the floor
    are dropped from the comparison just like extraction failures.

    ``max_market_count`` skips events with more than this many markets;
    high-count layouts are typically multi-checkbox where extraction is
    unreliable.
    """

    enabled: bool = True
    scan_interval_seconds: int = 300
    min_violation: float = 0.02
    min_event_liquidity_usd: float = 10000.0
    min_market_liquidity_usd: float = 100.0
    max_market_count: int = 12
```

In the same file, in the `Config` class (around line 387-407), add a `monotone` field after `mispricing`:

```python
    monotone: MonotoneConfig = Field(default_factory=MonotoneConfig)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v && uv run ty check src/pscanner/alerts/models.py src/pscanner/config.py`
Expected: PASS for the test; no errors from `ty check`.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/alerts/models.py src/pscanner/config.py tests/test_config.py
git commit -m "feat(monotone): register detector name and config skeleton"
```

---

## Task 2: Date-axis extractor (pure function)

**Files:**
- Create: `src/pscanner/detectors/monotone.py`
- Create: `tests/detectors/test_monotone.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_monotone.py`:

```python
"""Unit tests for the monotone detector."""

from __future__ import annotations

from datetime import date

from pscanner.detectors.monotone import extract_date_axis


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_monotone.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_date_axis'`

- [ ] **Step 3: Implement the extractor**

Create `src/pscanner/detectors/monotone.py` with:

```python
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

_ISO_DATE_PATTERN = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_MONTH_NAMES: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_DAY_YEAR_PATTERN = re.compile(
    r"^(?P<month>[A-Za-z]+)\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?\b",
)


def extract_date_axis(label: str | None, *, year_hint: int | None = None) -> date | None:
    """Extract a sortable :class:`date` from a market label.

    Recognised formats:

    * ISO: ``2026-04-30`` (with optional trailing text).
    * ``Month Day[, Year]``: ``April 30, 2026`` / ``Apr 30`` / ``April 30th``.
      When the year is omitted, ``year_hint`` is used (caller passes the
      current year for in-flight events).

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
    iso_match = _ISO_DATE_PATTERN.match(text)
    if iso_match is not None:
        try:
            return date(
                int(iso_match.group("y")),
                int(iso_match.group("m")),
                int(iso_match.group("d")),
            )
        except ValueError:
            return None
    month_match = _MONTH_DAY_YEAR_PATTERN.match(text)
    if month_match is None:
        return None
    month_token = month_match.group("month").lower()
    if month_token not in _MONTH_NAMES:
        return None
    year_token = month_match.group("year")
    if year_token is not None:
        year = int(year_token)
    elif year_hint is not None:
        year = year_hint
    else:
        return None
    try:
        return date(year, _MONTH_NAMES[month_token], int(month_match.group("day")))
    except ValueError:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_monotone.py -v && uv run ruff check src/pscanner/detectors/monotone.py && uv run ty check src/pscanner/detectors/monotone.py`
Expected: PASS for tests; clean lint and type check.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/monotone.py tests/detectors/test_monotone.py
git commit -m "feat(monotone): add date-axis extractor"
```

---

## Task 3: Threshold-axis extractor (pure function)

**Files:**
- Modify: `src/pscanner/detectors/monotone.py` (append helpers)
- Modify: `tests/detectors/test_monotone.py` (append tests)

The threshold extractor returns `(value, direction)` where `direction` is `"higher_is_stricter"` (e.g. ``above $X`` or ``≥ N``) or `"lower_is_stricter"` (``below $X`` or ``≤ N``). Range buckets (``$X-$Y``) and ``exactly N`` are intentionally *not* recognised — they're not nested events, just mutex partitions.

- [ ] **Step 1: Write the failing tests**

Append to `tests/detectors/test_monotone.py`:

```python
from pscanner.detectors.monotone import extract_threshold_axis


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_monotone.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_threshold_axis'`

- [ ] **Step 3: Implement the extractor**

Append to `src/pscanner/detectors/monotone.py`:

```python
from typing import Literal

ThresholdDirection = Literal["higher_is_stricter", "lower_is_stricter"]

_HIGHER_KEYWORDS = (
    r">=", r">", r"≥",
    r"above", r"over", r"at\s+least", r"more\s+than",
)
_LOWER_KEYWORDS = (
    r"<=", r"<", r"≤",
    r"below", r"under", r"at\s+most", r"less\s+than",
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
    r"^\$?[\d,.]+\s*[KMBT]?\s*[-–]\s*\$?[\d,.]+",
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
        return (_parse_number(higher_match), "higher_is_stricter")
    lower_match = _LOWER_PATTERN.match(text)
    if lower_match is not None:
        return (_parse_number(lower_match), "lower_is_stricter")
    return None


def _parse_number(match: re.Match[str]) -> float:
    """Combine a regex-matched number and KMBT suffix into a float."""
    raw = match.group("num").replace(",", "")
    base = float(raw)
    suffix = match.group("suffix").lower()
    return base * _SUFFIX_MULTIPLIERS.get(suffix, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_monotone.py -v && uv run ruff check src/pscanner/detectors/monotone.py && uv run ty check src/pscanner/detectors/monotone.py`
Expected: PASS; clean lint + type check.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/monotone.py tests/detectors/test_monotone.py
git commit -m "feat(monotone): add threshold-axis extractor"
```

---

## Task 4: Per-event axis selection (pure function)

**Files:**
- Modify: `src/pscanner/detectors/monotone.py`
- Modify: `tests/detectors/test_monotone.py`

Selection rules:
1. Try date axis on every market's `groupItemTitle`. If at least 2 markets yield a date, use the date axis (and drop markets that didn't yield).
2. Else try threshold axis on every market. If at least 2 markets yield a threshold *with the same direction*, use the threshold axis (and drop the rest).
3. Else return `None` — event is not monotone-eligible.

Output is a list of `MonotoneMarket` records, each carrying the kept market plus its sort key, sorted by sort key in *strict-first* order (i.e. earlier date, or stricter threshold).

- [ ] **Step 1: Write the failing tests**

Append to `tests/detectors/test_monotone.py`:

```python
from typing import Any

from pscanner.detectors.monotone import MonotoneMarket, select_axis
from pscanner.poly.models import Market


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_monotone.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_axis'`

- [ ] **Step 3: Implement the selector**

Append to `src/pscanner/detectors/monotone.py`:

```python
from dataclasses import dataclass

from pscanner.poly.models import Market


@dataclass(frozen=True, slots=True)
class MonotoneMarket:
    """One market that survived axis extraction.

    ``sort_key`` is in *strict-first* order: for date axes, an earlier date
    has a smaller key; for ``higher_is_stricter`` thresholds, the higher
    value has a smaller key (so the strictest leg sorts first).
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


def _yes_price(market: Market) -> float | None:
    """Return market's YES price (``outcome_prices[0]``) or None if missing."""
    if not market.outcome_prices:
        return None
    return market.outcome_prices[0]


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
    date_extracted: list[MonotoneMarket] = []
    for market in markets:
        price = _yes_price(market)
        if price is None:
            continue
        parsed = extract_date_axis(market.group_item_title, year_hint=year_hint)
        if parsed is None:
            continue
        date_extracted.append(
            MonotoneMarket(market=market, sort_key=parsed.toordinal(), yes_price=price),
        )
    if len(date_extracted) >= 2:
        ordered = tuple(sorted(date_extracted, key=lambda m: m.sort_key))
        return AxisSelection(kind="date", direction=None, markets=ordered)
    threshold_extracted: dict[ThresholdDirection, list[MonotoneMarket]] = {
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
        threshold_extracted[direction].append(
            MonotoneMarket(market=market, sort_key=value, yes_price=price),
        )
    for direction, group in threshold_extracted.items():
        if len(group) >= 2:
            reverse = direction == "higher_is_stricter"
            ordered = tuple(sorted(group, key=lambda m: m.sort_key, reverse=reverse))
            return AxisSelection(kind="threshold", direction=direction, markets=ordered)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_monotone.py -v && uv run ruff check src/pscanner/detectors/monotone.py && uv run ty check src/pscanner/detectors/monotone.py`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/monotone.py tests/detectors/test_monotone.py
git commit -m "feat(monotone): add per-event axis selection"
```

---

## Task 5: Adjacent-pair violation finder (pure function)

**Files:**
- Modify: `src/pscanner/detectors/monotone.py`
- Modify: `tests/detectors/test_monotone.py`

For an axis-sorted list `[m_0, m_1, ..., m_n]` in strict-first order, a violation exists for any adjacent pair where `yes_price[i] > yes_price[i+1] + tolerance`. Each violation produces one `MonotoneViolation` record carrying both markets and the gap.

- [ ] **Step 1: Write the failing tests**

Append to `tests/detectors/test_monotone.py`:

```python
from pscanner.detectors.monotone import find_violations


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
    assert {(v.strict.market.id, v.loose.market.id) for v in violations} == {
        ("m1", "m2"),
        ("m2", "m3"),
    }


def test_find_violations_handles_equal_prices() -> None:
    """Equal prices are not a violation."""
    selection = _selection_for_test(("m_apr", 1.0, 0.30), ("m_jun", 2.0, 0.30))
    assert find_violations(selection, min_violation=0.02) == []
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_monotone.py -v`
Expected: FAIL with `ImportError: cannot import name 'find_violations'`

- [ ] **Step 3: Implement the violation finder**

Append to `src/pscanner/detectors/monotone.py`:

```python
@dataclass(frozen=True, slots=True)
class MonotoneViolation:
    """One adjacent-pair violation of the monotone constraint.

    ``strict`` is the leg that should be cheaper (earlier deadline or
    stricter threshold); ``loose`` is the leg that should be richer.
    ``gap`` is ``strict.yes_price - loose.yes_price`` (strictly positive).
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
        gap = strict.yes_price - loose.yes_price
        if gap >= min_violation:
            violations.append(MonotoneViolation(strict=strict, loose=loose, gap=gap))
    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_monotone.py -v && uv run ruff check src/pscanner/detectors/monotone.py && uv run ty check src/pscanner/detectors/monotone.py`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/monotone.py tests/detectors/test_monotone.py
git commit -m "feat(monotone): add adjacent-pair violation finder"
```

---

## Task 6: `MonotoneDetector` class + alert emission

**Files:**
- Modify: `src/pscanner/detectors/monotone.py`
- Modify: `tests/detectors/test_monotone.py`

The detector:
- Inherits from `PollingDetector`.
- `_scan` iterates `gamma_client.iter_events(active=True, closed=False)`.
- `evaluate_event` runs eligibility filter + `select_axis` + `find_violations`, emits an alert per violation.
- Severity: `gap > 0.10` → `"high"`; `gap > 0.05` → `"med"`; else `"low"`.
- Alert key: `monotone:{event_id}:{strict_condition_id}:{loose_condition_id}` (no price-rounding — natural dedupe is per-pair).
- Alert body fields: `event_id`, `event_title`, `axis_kind` (`"date"` or `"threshold"`), `axis_direction` (or `None`), `strict_condition_id`, `loose_condition_id`, `strict_yes_price`, `loose_yes_price`, `gap`, `markets` (compact list of all selected markets).

- [ ] **Step 1: Write the failing tests**

Append to `tests/detectors/test_monotone.py`:

```python
from collections.abc import AsyncIterator, Iterable
from unittest.mock import AsyncMock

from pscanner.alerts.sink import AlertSink
from pscanner.config import MonotoneConfig
from pscanner.detectors.monotone import MonotoneDetector
from pscanner.poly.models import Event
from pscanner.store.repo import AlertsRepo


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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_monotone.py -v`
Expected: FAIL with `ImportError: cannot import name 'MonotoneDetector'`

- [ ] **Step 3: Implement the detector**

Append to `src/pscanner/detectors/monotone.py`:

```python
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from pscanner.alerts.models import Alert, Severity
from pscanner.alerts.sink import AlertSink
from pscanner.config import MonotoneConfig
from pscanner.detectors.polling import PollingDetector
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event
from pscanner.util.clock import Clock

_LOG = structlog.get_logger(__name__)

_HIGH_SEVERITY_GAP = 0.10
_MED_SEVERITY_GAP = 0.05
_MIN_VALID_MARKETS = 2


class MonotoneDetector(PollingDetector):
    """Detector that flags adjacent-axis monotonicity violations.

    Iterates the gamma ``/events`` catalogue (active, open) and, per event,
    extracts a date or threshold axis from each market's ``groupItemTitle``.
    Markets where extraction fails or whose liquidity falls below the
    configured floor are dropped — they don't disqualify the event. With
    at least two markets remaining on a consistent axis, the detector
    flags adjacent pairs whose YES prices violate ``P(strict) <= P(loose)``.
    """

    name: str = "monotone"

    def __init__(
        self,
        *,
        config: MonotoneConfig,
        gamma_client: GammaClient,
        clock: Clock | None = None,
    ) -> None:
        """Build a detector bound to a config and gamma client."""
        super().__init__(clock=clock)
        self._config = config
        self._gamma = gamma_client

    def _interval_seconds(self) -> float:
        """Return the configured scan cadence."""
        return float(self._config.scan_interval_seconds)

    async def _scan(self, sink: AlertSink) -> None:
        """Run a single pass over the active-event catalogue."""
        events = self._gamma.iter_events(active=True, closed=False)
        async for event in events:
            await self.evaluate_event(event, sink)

    async def evaluate_event(self, event: Event, sink: AlertSink) -> None:
        """Evaluate a single event and emit one alert per adjacent violation."""
        if not self._is_eligible(event):
            return
        markets = self._filter_liquid_markets(event)
        if len(markets) < _MIN_VALID_MARKETS:
            return
        year_hint = datetime.now(tz=timezone.utc).year
        selection = select_axis(markets, year_hint=year_hint)
        if selection is None:
            return
        violations = find_violations(selection, min_violation=self._config.min_violation)
        for violation in violations:
            await sink.emit(self._build_alert(event, selection, violation))

    def _is_eligible(self, event: Event) -> bool:
        """Cheap pre-filters that don't depend on axis extraction."""
        market_count = len(event.markets)
        if market_count < _MIN_VALID_MARKETS or market_count > self._config.max_market_count:
            return False
        if event.liquidity is None or event.liquidity < self._config.min_event_liquidity_usd:
            return False
        return True

    def _filter_liquid_markets(self, event: Event) -> list[Market]:
        """Drop markets below the per-market liquidity floor or with disabled books."""
        floor = self._config.min_market_liquidity_usd
        kept: list[Market] = []
        for market in event.markets:
            if not market.enable_order_book:
                continue
            if floor > 0:
                if market.liquidity is None or market.liquidity < floor:
                    continue
            kept.append(market)
        return kept

    def _build_alert(
        self,
        event: Event,
        selection: AxisSelection,
        violation: MonotoneViolation,
    ) -> Alert:
        """Construct the Alert payload for a monotone violation."""
        body: dict[str, Any] = {
            "event_id": event.id,
            "event_title": event.title,
            "axis_kind": selection.kind,
            "axis_direction": selection.direction,
            "strict_condition_id": violation.strict.market.condition_id,
            "loose_condition_id": violation.loose.market.condition_id,
            "strict_yes_price": violation.strict.yes_price,
            "loose_yes_price": violation.loose.yes_price,
            "gap": violation.gap,
            "markets": [
                {
                    "id": m.market.id,
                    "condition_id": m.market.condition_id,
                    "yes_price": m.yes_price,
                    "axis_value": m.sort_key,
                    "label": m.market.group_item_title,
                }
                for m in selection.markets
            ],
        }
        return Alert(
            detector="monotone",
            alert_key=(
                f"monotone:{event.id}:"
                f"{violation.strict.market.condition_id}:"
                f"{violation.loose.market.condition_id}"
            ),
            severity=_severity_for(violation.gap),
            title=(
                f"{event.title} — strict {violation.strict.yes_price:.3f} "
                f"> loose {violation.loose.yes_price:.3f} "
                f"(gap {violation.gap:.3f})"
            ),
            body=body,
            created_at=int(time.time()),
        )


def _severity_for(gap: float) -> Severity:
    """Map a violation gap to a triage bucket."""
    if gap > _HIGH_SEVERITY_GAP:
        return "high"
    if gap > _MED_SEVERITY_GAP:
        return "med"
    return "low"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_monotone.py -v && uv run ruff check src/pscanner/detectors/monotone.py && uv run ty check src/pscanner/detectors/monotone.py`
Expected: PASS; clean.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/monotone.py tests/detectors/test_monotone.py
git commit -m "feat(monotone): implement detector with alert emission"
```

---

## Task 7: `MonotoneEvaluator` (paired-trade evaluator)

**Files:**
- Create: `src/pscanner/strategies/evaluators/monotone.py`
- Create: `tests/strategies/evaluators/test_monotone.py`
- Modify: `src/pscanner/config.py` (add `MonotoneEvaluatorConfig`, wire into `EvaluatorsConfig`)
- Modify: `src/pscanner/strategies/evaluators/__init__.py` (re-export `MonotoneEvaluator`)

The evaluator emits **two** ParsedSignals per alert, mirroring `VelocityEvaluator`'s twin-trade pattern:
- `(strict_condition_id, side="NO", rule_variant="strict_no")`
- `(loose_condition_id, side="YES", rule_variant="loose_yes")`

Each side is sized at `position_fraction` (the *per-leg* fraction; both legs together equal `2 * position_fraction` of bankroll). Quality gate is `gap >= min_edge_dollars`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/evaluators/test_monotone.py`:

```python
"""Unit tests for MonotoneEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MonotoneEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.monotone import MonotoneEvaluator


def _alert(*, body: dict[str, Any], detector: str = "monotone") -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        alert_key="k1",
        severity="med",
        title="t",
        body=body,
        created_at=0,
    )


def _evaluator(
    *,
    position_fraction: float = 0.005,
    min_edge_dollars: float = 0.02,
) -> MonotoneEvaluator:
    return MonotoneEvaluator(
        config=MonotoneEvaluatorConfig(
            position_fraction=position_fraction,
            min_edge_dollars=min_edge_dollars,
        ),
    )


def _good_body() -> dict[str, Any]:
    return {
        "strict_condition_id": "0xstrict",
        "loose_condition_id": "0xloose",
        "strict_yes_price": 0.40,
        "loose_yes_price": 0.30,
        "gap": 0.10,
    }


def test_accepts_only_monotone() -> None:
    ev = _evaluator()
    assert ev.accepts(_alert(body={}, detector="monotone")) is True
    assert ev.accepts(_alert(body={}, detector="mispricing")) is False


def test_parse_emits_two_legs() -> None:
    ev = _evaluator()
    signals = ev.parse(_alert(body=_good_body()))
    assert len(signals) == 2
    by_variant = {s.rule_variant: s for s in signals}
    strict = by_variant["strict_no"]
    loose = by_variant["loose_yes"]
    assert strict.condition_id == ConditionId("0xstrict")
    assert strict.side == "NO"
    assert strict.metadata["gap"] == 0.10
    assert loose.condition_id == ConditionId("0xloose")
    assert loose.side == "YES"


def test_parse_returns_empty_when_required_fields_missing() -> None:
    ev = _evaluator()
    bad = {"strict_condition_id": "0xstrict"}
    assert ev.parse(_alert(body=bad)) == []


def test_quality_passes_above_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.05)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xstrict"),
        side="NO",
        rule_variant="strict_no",
        metadata={"gap": 0.10},
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_below_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.05)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xstrict"),
        side="NO",
        rule_variant="strict_no",
        metadata={"gap": 0.03},
    )
    assert ev.quality_passes(parsed) is False


def test_size_returns_per_leg_fraction() -> None:
    ev = _evaluator(position_fraction=0.005)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xstrict"),
        side="NO",
        rule_variant="strict_no",
        metadata={"gap": 0.10},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/evaluators/test_monotone.py -v`
Expected: FAIL with `ModuleNotFoundError` or `cannot import name 'MonotoneEvaluator'`.

- [ ] **Step 3: Add `MonotoneEvaluatorConfig` to `config.py`**

In `src/pscanner/config.py`, after `MispricingEvaluatorConfig` (around line 336), add:

```python
class MonotoneEvaluatorConfig(_Section):
    """Monotone-arbitrage evaluator tunables.

    Each alert spawns two ParsedSignals (strict_no + loose_yes) at the
    per-side ``position_fraction`` (default 0.5%, pair total 1%). Constant
    size off ``starting_bankroll_usd``, not running NAV.
    """

    enabled: bool = True
    position_fraction: float = 0.005
    min_edge_dollars: float = 0.02
```

In the same file, in `EvaluatorsConfig` (around line 338-352), add a `monotone` field:

```python
    monotone: MonotoneEvaluatorConfig = Field(default_factory=MonotoneEvaluatorConfig)
```

- [ ] **Step 4: Implement the evaluator**

Create `src/pscanner/strategies/evaluators/monotone.py`:

```python
"""``MonotoneEvaluator`` — paired-trade NO-strict + YES-loose per alert.

Each monotone alert names the two condition_ids of the violating adjacent
pair. The evaluator emits two ParsedSignals (``strict_no`` + ``loose_yes``)
sized at the per-leg ``position_fraction``. Quality gate is the ``gap``
field on the alert body — when below ``min_edge_dollars`` no signals fire.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import MonotoneEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class MonotoneEvaluator:
    """Two-leg paired-trade evaluator for monotone-arb alerts."""

    def __init__(self, *, config: MonotoneEvaluatorConfig) -> None:
        """Bind dependencies for the monotone evaluator."""
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert was emitted by the monotone detector."""
        return alert.detector == "monotone"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull pair fields from the alert body and emit two ParsedSignals."""
        body = alert.body if isinstance(alert.body, dict) else {}
        strict = body.get("strict_condition_id")
        loose = body.get("loose_condition_id")
        gap = body.get("gap")
        if not (
            isinstance(strict, str)
            and isinstance(loose, str)
            and isinstance(gap, int | float)
        ):
            _LOG.debug("monotone_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        meta = {"gap": float(gap)}
        return [
            ParsedSignal(
                condition_id=ConditionId(strict),
                side="NO",
                rule_variant="strict_no",
                metadata=meta,
            ),
            ParsedSignal(
                condition_id=ConditionId(loose),
                side="YES",
                rule_variant="loose_yes",
                metadata=meta,
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject signals whose gap is below the edge floor."""
        gap = parsed.metadata.get("gap")
        if not isinstance(gap, int | float):
            return False
        return float(gap) >= self._config.min_edge_dollars

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return constant ``bankroll * position_fraction`` per leg."""
        del parsed
        return bankroll * self._config.position_fraction
```

- [ ] **Step 5: Re-export from the evaluators package**

In `src/pscanner/strategies/evaluators/__init__.py`, replace the file with:

```python
"""Per-detector :class:`SignalEvaluator` implementations.

PaperTrader walks a list of evaluators on each alert; the first one whose
``accepts`` returns ``True`` runs the parse → quality → size pipeline.
"""

from pscanner.strategies.evaluators.mispricing import MispricingEvaluator
from pscanner.strategies.evaluators.monotone import MonotoneEvaluator
from pscanner.strategies.evaluators.move_attribution import MoveAttributionEvaluator
from pscanner.strategies.evaluators.protocol import (
    ParsedSignal,
    SignalEvaluator,
)
from pscanner.strategies.evaluators.smart_money import SmartMoneyEvaluator
from pscanner.strategies.evaluators.velocity import VelocityEvaluator

__all__ = [
    "MispricingEvaluator",
    "MonotoneEvaluator",
    "MoveAttributionEvaluator",
    "ParsedSignal",
    "SignalEvaluator",
    "SmartMoneyEvaluator",
    "VelocityEvaluator",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/evaluators/test_monotone.py tests/test_config.py -v && uv run ruff check src/pscanner/strategies/evaluators/monotone.py src/pscanner/strategies/evaluators/__init__.py src/pscanner/config.py && uv run ty check src/pscanner/strategies/evaluators/monotone.py`
Expected: PASS; clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/strategies/evaluators/monotone.py src/pscanner/strategies/evaluators/__init__.py src/pscanner/config.py tests/strategies/evaluators/test_monotone.py
git commit -m "feat(monotone): add paired-trade evaluator"
```

---

## Task 8: Wire detector + evaluator into scheduler + config.toml

**Files:**
- Modify: `src/pscanner/scheduler.py` (import, instantiate in `_build_detectors`, append in `_build_paper_evaluators`)
- Modify: `config.toml` (add `[monotone]` and `[paper_trading.evaluators.monotone]`)
- Modify: `tests/test_scheduler.py` (extend `_make_config`, add wiring tests)

Note on `run_once`: deliberately NOT adding a `_run_once_monotone` helper. The existing `events_scanned` count is owned by the mispricing pass; double-counting via a second iteration would change the contract of `run_once` and break `test_run_once_with_no_data_returns_zero_counts` invariants downstream. The daemon loop covers monotone — `--once` is for smoke-testing wiring, not signal output. Smoke verification of monotone happens in Task 9 against the real daemon.

- [ ] **Step 1: Extend `_make_config` and add the failing wiring tests**

In `tests/test_scheduler.py`, update the imports (around lines 28-44) to add `MonotoneConfig` and `PaperTradingConfig`:

```python
from pscanner.config import (
    ActivityConfig,
    ClusterConfig,
    Config,
    ConvergenceConfig,
    EventsConfig,
    MarketsConfig,
    MispricingConfig,
    MonotoneConfig,
    MoveAttributionConfig,
    PaperTradingConfig,
    PositionsConfig,
    RatelimitConfig,
    ScannerConfig,
    SmartMoneyConfig,
    TicksConfig,
    VelocityConfig,
    WhalesConfig,
)
```

Modify `_make_config` (around lines 103-132) to accept `enable_monotone` and pass through:

```python
def _make_config(
    *,
    enable_smart: bool = True,
    enable_misprice: bool = True,
    enable_monotone: bool = True,
    enable_whales: bool = True,
    enable_convergence: bool = True,
    enable_positions: bool = True,
    enable_activity: bool = True,
    enable_markets: bool = True,
    enable_events: bool = True,
    enable_ticks: bool = True,
    enable_velocity: bool = True,
    enable_cluster: bool = True,
    enable_move_attribution: bool = True,
) -> Config:
    return Config(
        scanner=ScannerConfig(),
        smart_money=SmartMoneyConfig(enabled=enable_smart),
        mispricing=MispricingConfig(enabled=enable_misprice),
        monotone=MonotoneConfig(enabled=enable_monotone),
        whales=WhalesConfig(enabled=enable_whales),
        convergence=ConvergenceConfig(enabled=enable_convergence),
        ratelimit=RatelimitConfig(),
        positions=PositionsConfig(enabled=enable_positions),
        activity=ActivityConfig(enabled=enable_activity),
        markets=MarketsConfig(enabled=enable_markets),
        events=EventsConfig(enabled=enable_events),
        ticks=TicksConfig(enabled=enable_ticks),
        velocity=VelocityConfig(enabled=enable_velocity),
        cluster=ClusterConfig(enabled=enable_cluster),
        move_attribution=MoveAttributionConfig(enabled=enable_move_attribution),
    )
```

Append the wiring tests at the end of `tests/test_scheduler.py`:

```python
@pytest.mark.asyncio
async def test_scanner_wires_monotone_detector_when_enabled(db_path: Path) -> None:
    """When ``monotone.enabled`` is True, the detector is in ``_detectors``."""
    config = _make_config(enable_monotone=True)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "monotone" in scanner._detectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_skips_monotone_detector_when_disabled(db_path: Path) -> None:
    """When ``monotone.enabled`` is False, the detector is not constructed."""
    config = _make_config(enable_monotone=False)
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        assert "monotone" not in scanner._detectors
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_scanner_wires_monotone_evaluator_when_paper_enabled(db_path: Path) -> None:
    """Monotone evaluator appended when paper_trading + monotone evaluator enabled."""
    from pscanner.strategies.evaluators import MonotoneEvaluator

    config = Config(
        scanner=ScannerConfig(),
        smart_money=SmartMoneyConfig(enabled=False),
        mispricing=MispricingConfig(enabled=False),
        monotone=MonotoneConfig(enabled=True),
        whales=WhalesConfig(enabled=False),
        convergence=ConvergenceConfig(enabled=False),
        ratelimit=RatelimitConfig(),
        positions=PositionsConfig(enabled=False),
        activity=ActivityConfig(enabled=False),
        markets=MarketsConfig(enabled=False),
        events=EventsConfig(enabled=False),
        ticks=TicksConfig(enabled=False),
        velocity=VelocityConfig(enabled=False),
        cluster=ClusterConfig(enabled=False),
        move_attribution=MoveAttributionConfig(enabled=False),
        paper_trading=PaperTradingConfig(enabled=True),
    )
    clients = _make_clients()
    scanner = Scanner(config=config, db_path=db_path, clients=clients)
    try:
        evaluators = scanner._build_paper_evaluators()
        assert any(isinstance(e, MonotoneEvaluator) for e in evaluators)
    finally:
        await scanner.aclose()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py -k monotone -v`
Expected: FAIL — `"monotone"` not in `_detectors`; `MonotoneEvaluator` not in evaluator list.

- [ ] **Step 3: Wire the detector and evaluator in scheduler.py**

In `src/pscanner/scheduler.py`, add to the imports (after line 47, near other detector imports):

```python
from pscanner.detectors.monotone import MonotoneDetector
```

Replace the evaluator import block (around lines 79-85) with:

```python
from pscanner.strategies.evaluators import (
    MispricingEvaluator,
    MonotoneEvaluator,
    MoveAttributionEvaluator,
    SignalEvaluator,
    SmartMoneyEvaluator,
    VelocityEvaluator,
)
```

In `_build_detectors`, after the `mispricing` block (around lines 314-320), add:

```python
        if self._config.monotone.enabled:
            detectors["monotone"] = MonotoneDetector(
                config=self._config.monotone,
                gamma_client=self._clients.gamma_client,
                clock=self._clock,
            )
```

In `_build_paper_evaluators`, after the mispricing block (around lines 398-401), add:

```python
        if cfg.monotone.enabled:
            evaluators.append(
                MonotoneEvaluator(config=cfg.monotone),
            )
```

- [ ] **Step 4: Update config.toml**

In `config.toml`, after the `[mispricing]` block (after line 33), insert:

```toml
[monotone]
enabled = true
scan_interval_seconds = 300
min_violation = 0.02
min_event_liquidity_usd = 10000
min_market_liquidity_usd = 100
max_market_count = 12
```

After the `[paper_trading.evaluators.mispricing]` block (after line 140), insert:

```toml
[paper_trading.evaluators.monotone]
enabled = true
position_fraction = 0.005
min_edge_dollars = 0.02
```

- [ ] **Step 5: Run scheduler + config tests**

Run: `uv run pytest tests/test_scheduler.py tests/test_config.py -v && uv run ruff check src/pscanner/scheduler.py && uv run ty check src/pscanner/scheduler.py`
Expected: PASS; clean.

- [ ] **Step 6: Run the full test suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: PASS overall (verifies no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/scheduler.py config.toml tests/test_scheduler.py
git commit -m "feat(monotone): wire detector + evaluator into scheduler"
```

---

## Task 9: Smoke test against live data

**Files:**
- None (manual verification step).

This is a verification-only task. Run the daemon briefly against live gamma + check that the alerts make sense before declaring victory.

- [ ] **Step 1: Drop the local DB for a clean smoke run**

Run: `rm -f data/pscanner.sqlite3`

- [ ] **Step 2: Run the daemon for 5 minutes**

Run: `timeout 300 uv run pscanner run > /tmp/monotone-smoke.log 2>&1; echo exit=$?`
Expected: exit=124 (timed out cleanly) and the log contains structured `monotone` alert events.

- [ ] **Step 3: Inspect the alerts**

Run a SQL inspection:

```bash
uv run --quiet python <<'EOF'
import sqlite3, json
conn = sqlite3.connect("./data/pscanner.sqlite3")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
n = cur.execute("SELECT COUNT(*) FROM alerts WHERE detector='monotone'").fetchone()[0]
print(f"monotone alerts: {n}")
for row in cur.execute("""
    SELECT severity, title, body_json
    FROM alerts WHERE detector='monotone'
    ORDER BY created_at DESC LIMIT 6
"""):
    body = json.loads(row["body_json"])
    print(f"\n[{row['severity']}] {row['title']}")
    print(f"  axis={body['axis_kind']} dir={body.get('axis_direction')}")
    print(f"  strict={body['strict_condition_id'][:14]}… @ {body['strict_yes_price']:.3f}")
    print(f"  loose ={body['loose_condition_id'][:14]}… @ {body['loose_yes_price']:.3f}")
    print(f"  gap={body['gap']:.3f}")
    for m in body["markets"]:
        print(f"    axis={m['axis_value']}  yes={m['yes_price']:.3f}  {m['label']}")
EOF
```

For each alert, verify by eye:
- The `axis_kind` matches the event's actual layout (date vs threshold).
- The two markets' `groupItemTitle`s are *logically* nested (by-Date1 ⇒ by-Date2; or threshold-X ⇒ threshold-Y).
- The strict leg really should be ≤ loose leg, and the price snapshot really violates that.
- The `gap` is non-trivial (≥ a few cents).

- [ ] **Step 4: Inspect the paper trades**

```bash
uv run --quiet python <<'EOF'
import sqlite3
conn = sqlite3.connect("./data/pscanner.sqlite3")
conn.row_factory = sqlite3.Row
cur = conn.cursor()
n = cur.execute("SELECT COUNT(*) FROM paper_trades WHERE triggering_alert_detector='monotone'").fetchone()[0]
print(f"monotone paper trades: {n}")
for row in cur.execute("""
    SELECT outcome, fill_price, cost_usd, rule_variant, condition_id
    FROM paper_trades WHERE triggering_alert_detector='monotone'
    ORDER BY ts DESC LIMIT 10
"""):
    print(f"  {row['outcome']:3s} @ {row['fill_price']:.3f}  ${row['cost_usd']:6.2f}  variant={row['rule_variant']}  cond={row['condition_id'][:14]}…")
EOF
```

Verify: every alert produced **two** paper trades (one `strict_no` + one `loose_yes`). The two trades reference *different* condition_ids and have outcomes `NO` and `YES` respectively.

- [ ] **Step 5: Document findings**

If alerts and trades look correct, no further action — the feature is shipped. If false positives surface (e.g. axis extraction misclassifies a non-nested layout), capture concrete examples and file them as follow-ups under `## Open follow-ups (no issues filed)` in `CLAUDE.md`. Do not silently re-tune thresholds without surfacing the failure mode first.

---

## Self-review notes

- **Spec coverage:** Tasks 2-3 cover axis extraction; 4 covers per-event selection; 5 covers violation finding; 6 covers detector + alert emission; 7 covers paired-trade evaluator; 8 covers scheduler + config wiring; 9 covers smoke validation. The within-event temporal (#1) and within-event threshold (#2) flavors from the design conversation are both supported by the same axis-selection pipeline (Task 4).
- **Cross-event monotone (#3):** Out of scope — explicitly deferred in the header.
- **Two-leg paired-trade infrastructure:** Reuses `VelocityEvaluator`'s twin-trade pattern (two `ParsedSignal`s with `rule_variant`). No new `PaperTrader` machinery required.
- **Repeat-alert problem from the previous mispricing audit:** addressed by the natural pair-based dedupe key (`monotone:{event_id}:{strict_cond}:{loose_cond}`); same pair → one row in `alerts` (PK on `alert_key`).
