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
    """

    category: Category
    min_edge: float
    convergence_window_seconds: int
    mispricing_skip: bool
    tag_labels: tuple[str, ...]


# Default taxonomy. Numbers preserve current production behavior:
# - sports: 0.10 edge floor, 6h convergence window, mispricing-skipped
# - esports: 0.05 edge floor, 24h convergence window, mispricing-skipped
# - thesis: 0.05 edge floor, 48h convergence window, mispricing-eligible
#
# Sports is listed first so it wins over esports when both tags are
# present, mirroring the legacy ``_categorize`` helpers. Thesis carries
# an empty ``tag_labels`` tuple and is the default fallback.
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
        category=Category.THESIS,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=(),
    ),
)


def categorize_tags(tags: Iterable[str]) -> Category:
    """Return the Category that matches the first applicable tag.

    Falls back to :attr:`Category.THESIS` if no tag matches a labelled
    category. Match is case-insensitive. Non-string entries in ``tags``
    are ignored.

    Args:
        tags: Iterable of tag label strings.

    Returns:
        The matched :class:`Category`, or :attr:`Category.THESIS` when no
        labelled entry matches.
    """
    lower = {tag.lower() for tag in tags if isinstance(tag, str)}
    for settings in DEFAULT_TAXONOMY:
        if not settings.tag_labels:
            continue
        if any(label.lower() in lower for label in settings.tag_labels):
            return settings.category
    return Category.THESIS


def categorize_event(event: Event) -> Category:
    """Categorize an :class:`Event` by its tags.

    Args:
        event: Polymarket event whose ``tags`` drive categorisation.

    Returns:
        The matched :class:`Category`.
    """
    return categorize_tags(event.tags)


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
    "settings_for",
]
