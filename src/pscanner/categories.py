"""Category taxonomy — single source of truth for thesis/sports/esports.

Polymarket events fall into three coarse categories that affect detector
behavior in different ways:

- Smart-money applies different edge thresholds per category (sports
  markets are tighter, so a higher edge bar is appropriate).
- Convergence applies different time windows (sports games resolve in
  hours, thesis bets resolve in days/weeks).
- Mispricing skips events tagged Sports/Esports because they're
  tournament aggregations, not mutex outcomes.

The taxonomy lives here so adding a fourth category (e.g. "crypto") is
a single-place edit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pscanner.poly.models import Event


class Category(StrEnum):
    """Coarse event category for detector behavior dispatch."""

    THESIS = "thesis"
    SPORTS = "sports"
    ESPORTS = "esports"
    MACRO = "macro"
    ELECTIONS = "elections"
    CRYPTO = "crypto"
    GEOPOLITICS = "geopolitics"
    TECH = "tech"
    CULTURE = "culture"


@dataclass(frozen=True, slots=True)
class CategorySettings:
    """Per-category detector parameters.

    Attributes:
        category: Which category these settings apply to.
        min_edge: Smart-money minimum mean-edge threshold.
        convergence_window_seconds: How far back convergence looks for
            other smart wallets in this category trading the same condition.
        mispricing_skip: When True, mispricing skips events of this category
            (sports tournaments aren't mutex events).
        tag_labels: Polymarket gamma tag labels (case-insensitive) that map
            an event to this category. The first taxonomy entry whose labels
            match wins; an entry with an empty ``tag_labels`` tuple is the
            fallback bucket.
        tag_exclusions: Polymarket gamma tag labels (case-insensitive) that
            disqualify an event from this category match even if a label in
            ``tag_labels`` is present. Used to prevent automated recurring
            markets (e.g. ``Crypto Prices``) from sweeping into CRYPTO.
    """

    category: Category
    min_edge: float
    convergence_window_seconds: int
    mispricing_skip: bool
    tag_labels: tuple[str, ...]
    tag_exclusions: tuple[str, ...] = ()


# Default taxonomy. SPORTS / ESPORTS retain their priority spots and legacy
# detector tuning. The 6 named subdomains (MACRO, CRYPTO, ELECTIONS,
# GEOPOLITICS, TECH, CULTURE) carve the former THESIS bucket along
# polymarket gamma tag families — see #119 for motivation. Priority is
# most-specific-first so a Fed-decision-during-election event resolves to
# MACRO. THESIS keeps an empty ``tag_labels`` tuple and is the default
# fallback for events that match no labelled entry. Detector knobs
# (min_edge, convergence_window, mispricing_skip) on the new entries
# default to legacy THESIS values — tuning per-subdomain is a separate
# follow-up.
DEFAULT_TAXONOMY: tuple[CategorySettings, ...] = (
    CategorySettings(
        category=Category.SPORTS,
        min_edge=0.10,
        convergence_window_seconds=6 * 3600,
        mispricing_skip=True,
        tag_labels=("Sports",),
    ),
    CategorySettings(
        category=Category.ESPORTS,
        min_edge=0.05,
        convergence_window_seconds=24 * 3600,
        mispricing_skip=True,
        tag_labels=("Esports",),
    ),
    CategorySettings(
        category=Category.MACRO,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=("Fed Rates", "Fed", "fomc", "Jerome Powell", "Economic Policy"),
    ),
    CategorySettings(
        category=Category.CRYPTO,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=(
            "Crypto",
            "Bitcoin",
            "Ethereum",
            "Solana",
            "XRP",
            "Ripple",
            "Dogecoin",
            "BNB",
            "$TRUMP",
        ),
        tag_exclusions=("Crypto Prices", "Recurring", "Up or Down", "Hide From New"),
    ),
    CategorySettings(
        category=Category.ELECTIONS,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=(
            "Global Elections",
            "World Elections",
            "US Election",
            "Mayoral Elections",
            "Elections",
        ),
    ),
    CategorySettings(
        category=Category.GEOPOLITICS,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=("Geopolitics", "Foreign Policy", "Middle East"),
    ),
    CategorySettings(
        category=Category.TECH,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=("AI", "Big Tech", "Tech"),
    ),
    CategorySettings(
        category=Category.CULTURE,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=("Culture", "Movies", "Celebrities"),
    ),
    CategorySettings(
        category=Category.THESIS,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=(),
    ),
)


def categorize_tags(tags: Iterable[str]) -> frozenset[Category]:
    """Return every :class:`Category` whose tag labels match ``tags``.

    Match is case-insensitive. A taxonomy entry is skipped when any of its
    ``tag_exclusions`` is present in the input tag set, even if its
    ``tag_labels`` would otherwise match. Non-string entries in ``tags``
    are ignored. When no labelled entry matches the input, the returned
    set contains only :attr:`Category.THESIS` (the fallback bucket).

    Use :func:`primary_category` when a single :class:`Category` is needed
    for detector behaviour dispatch.

    Args:
        tags: Iterable of tag label strings.

    Returns:
        A non-empty :class:`frozenset` of matching categories. The empty
        case is replaced by ``frozenset({Category.THESIS})``.
    """
    lower = {tag.lower() for tag in tags if isinstance(tag, str)}
    matched: set[Category] = set()
    for settings in DEFAULT_TAXONOMY:
        if not settings.tag_labels:
            continue
        if any(label.lower() in lower for label in settings.tag_exclusions):
            continue
        if any(label.lower() in lower for label in settings.tag_labels):
            matched.add(settings.category)
    if not matched:
        return frozenset({Category.THESIS})
    return frozenset(matched)


def primary_category(tags: Iterable[str]) -> Category:
    """Return the priority-ordered first :class:`Category` match for ``tags``.

    Walks :data:`DEFAULT_TAXONOMY` in tuple order; the first entry whose
    ``tag_labels`` match (and whose ``tag_exclusions`` do not match) wins.
    Returns the highest-priority element of :func:`categorize_tags`'s
    result. Kept as a stable single-Category accessor so detector dispatch
    sites have a clear contract independent of the multi-label
    :func:`categorize_tags` return type.

    Args:
        tags: Iterable of tag label strings.

    Returns:
        The priority-ordered first :class:`Category` match, or
        :attr:`Category.THESIS` when no labelled entry matches.
    """
    lower = {tag.lower() for tag in tags if isinstance(tag, str)}
    for settings in DEFAULT_TAXONOMY:
        if not settings.tag_labels:
            continue
        if any(label.lower() in lower for label in settings.tag_exclusions):
            continue
        if any(label.lower() in lower for label in settings.tag_labels):
            return settings.category
    return Category.THESIS


def categorize_event(event: Event) -> Category:
    """Categorize an :class:`Event` by its tags via priority dispatch.

    Args:
        event: Polymarket event whose ``tags`` drive categorisation.

    Returns:
        The :class:`Category` returned by :func:`primary_category` on the
        event's tags. Kept single-valued for backward compat at the
        mispricing detector's dispatch site.
    """
    return primary_category(event.tags)


def settings_for(category: Category) -> CategorySettings:
    """Look up the :class:`CategorySettings` for a given :class:`Category`.

    Args:
        category: The category to resolve.

    Returns:
        The matching :class:`CategorySettings` from
        :data:`DEFAULT_TAXONOMY`.

    Raises:
        ValueError: If ``category`` has no entry in the taxonomy. This
            should not occur for any :class:`Category` member but guards
            future drift.
    """
    for settings in DEFAULT_TAXONOMY:
        if settings.category is category:
            return settings
    msg = f"unknown category: {category}"
    raise ValueError(msg)


__all__ = [
    "DEFAULT_TAXONOMY",
    "Category",
    "CategorySettings",
    "categorize_event",
    "categorize_tags",
    "primary_category",
    "settings_for",
]
