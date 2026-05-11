"""Tests for the category taxonomy module."""

from __future__ import annotations

from typing import Any, cast

import pytest

from pscanner.categories import (
    DEFAULT_TAXONOMY,
    Category,
    CategorySettings,
    categorize_event,
    categorize_tags,
    primary_category,
    settings_for,
)
from pscanner.poly.models import Event


def _event(*, tags: list[str] | None = None) -> Event:
    """Build a synthetic Event with optional tag labels."""
    payload: dict[str, Any] = {
        "id": "evt-1",
        "title": "Test event",
        "slug": "test-event",
        "markets": [],
    }
    if tags is not None:
        payload["tags"] = tags
    return Event.model_validate(payload)


def test_sports_tag_classifies_as_sports() -> None:
    assert categorize_tags(["Sports"]) == frozenset({Category.SPORTS})


def test_esports_tag_classifies_as_esports_case_insensitive() -> None:
    assert categorize_tags(["esports"]) == frozenset({Category.ESPORTS})


def test_unrelated_tag_falls_back_to_thesis() -> None:
    assert categorize_tags(["Politics"]) == frozenset({Category.THESIS})


def test_empty_tag_list_falls_back_to_thesis() -> None:
    assert categorize_tags([]) == frozenset({Category.THESIS})


def test_categorize_tags_returns_all_matches_for_multi_tag_event() -> None:
    """``categorize_tags`` returns every matching category, not just the first."""
    result = categorize_tags(["Esports", "Sports"])
    assert result == frozenset({Category.SPORTS, Category.ESPORTS})


def test_non_string_tag_entries_are_ignored() -> None:
    """Malformed entries don't crash categorisation."""
    raw: list[Any] = ["Sports", 42, None]
    assert categorize_tags(cast("list[str]", raw)) == frozenset({Category.SPORTS})


def test_settings_for_sports_skips_mispricing() -> None:
    assert settings_for(Category.SPORTS).mispricing_skip is True


def test_settings_for_esports_skips_mispricing() -> None:
    assert settings_for(Category.ESPORTS).mispricing_skip is True


def test_settings_for_thesis_does_not_skip_mispricing() -> None:
    assert settings_for(Category.THESIS).mispricing_skip is False


def test_settings_for_thesis_min_edge_matches_legacy_default() -> None:
    assert settings_for(Category.THESIS).min_edge == pytest.approx(0.05)


def test_settings_for_sports_min_edge_matches_legacy_default() -> None:
    assert settings_for(Category.SPORTS).min_edge == pytest.approx(0.10)


def test_settings_for_esports_min_edge_matches_legacy_default() -> None:
    assert settings_for(Category.ESPORTS).min_edge == pytest.approx(0.05)


def test_thesis_window_matches_legacy_48h() -> None:
    assert settings_for(Category.THESIS).convergence_window_seconds == 48 * 3600


def test_sports_window_matches_legacy_6h() -> None:
    assert settings_for(Category.SPORTS).convergence_window_seconds == 6 * 3600


def test_esports_window_matches_legacy_24h() -> None:
    assert settings_for(Category.ESPORTS).convergence_window_seconds == 24 * 3600


def test_categorize_event_routes_through_event_tags() -> None:
    event = _event(tags=["Sports", "NBA"])
    assert categorize_event(event) is Category.SPORTS


def test_categorize_event_with_no_tags_is_thesis() -> None:
    event = _event(tags=[])
    assert categorize_event(event) is Category.THESIS


def test_default_taxonomy_covers_every_category_member() -> None:
    """Every Category enum member must have a settings row."""
    covered = {entry.category for entry in DEFAULT_TAXONOMY}
    assert covered == set(Category)


def test_category_settings_default_tag_exclusions_is_empty() -> None:
    settings = CategorySettings(
        category=Category.THESIS,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=(),
    )
    assert settings.tag_exclusions == ()


def test_category_settings_accepts_tag_exclusions() -> None:
    settings = CategorySettings(
        category=Category.THESIS,
        min_edge=0.05,
        convergence_window_seconds=48 * 3600,
        mispricing_skip=False,
        tag_labels=("Crypto",),
        tag_exclusions=("Crypto Prices",),
    )
    assert settings.tag_exclusions == ("Crypto Prices",)


def test_category_enum_contains_six_new_members() -> None:
    assert Category.MACRO.value == "macro"
    assert Category.ELECTIONS.value == "elections"
    assert Category.CRYPTO.value == "crypto"
    assert Category.GEOPOLITICS.value == "geopolitics"
    assert Category.TECH.value == "tech"
    assert Category.CULTURE.value == "culture"


def test_macro_tag_classifies_as_macro() -> None:
    assert categorize_tags(["Fed Rates"]) == frozenset({Category.MACRO})


def test_elections_us_tag_classifies_as_elections() -> None:
    assert categorize_tags(["US Election"]) == frozenset({Category.ELECTIONS})


def test_crypto_bitcoin_classifies_as_crypto() -> None:
    assert categorize_tags(["Bitcoin"]) == frozenset({Category.CRYPTO})


def test_crypto_solana_classifies_as_crypto() -> None:
    """Decision A: cover the high-volume tokens missed in the original proposal."""
    assert categorize_tags(["Solana"]) == frozenset({Category.CRYPTO})


def test_geopolitics_middle_east_classifies_as_geopolitics() -> None:
    assert categorize_tags(["Middle East"]) == frozenset({Category.GEOPOLITICS})


def test_geopolitics_drops_country_specific_labels() -> None:
    """Decision D: Iran/Israel/Ukraine are caught via umbrella labels; not in tag_labels."""
    assert categorize_tags(["Iran"]) == frozenset({Category.THESIS})
    assert categorize_tags(["Israel"]) == frozenset({Category.THESIS})
    assert categorize_tags(["Ukraine"]) == frozenset({Category.THESIS})


def test_tech_ai_classifies_as_tech() -> None:
    assert categorize_tags(["AI"]) == frozenset({Category.TECH})


def test_culture_movies_classifies_as_culture() -> None:
    assert categorize_tags(["Movies"]) == frozenset({Category.CULTURE})


def test_categorize_tags_returns_multiple_matches_for_macro_election_event() -> None:
    """A Fed-during-election event matches both MACRO and ELECTIONS."""
    assert categorize_tags(["Fed Rates", "Global Elections"]) == frozenset(
        {Category.MACRO, Category.ELECTIONS}
    )


def test_crypto_prices_event_does_not_match_crypto() -> None:
    """Decision A4: tag_exclusions prevents Crypto-Prices recurring markets from CRYPTO."""
    assert categorize_tags(["Crypto", "Crypto Prices", "Recurring"]) == frozenset(
        {Category.THESIS}
    )


def test_crypto_with_no_exclusion_tag_matches_crypto() -> None:
    """Sanity: a plain ``Crypto`` event still matches CRYPTO."""
    assert categorize_tags(["Crypto"]) == frozenset({Category.CRYPTO})


def test_crypto_excluded_returns_only_other_matching_categories() -> None:
    """If excluded from CRYPTO, sports still appears in the result set."""
    assert categorize_tags(["Sports", "Crypto", "Crypto Prices"]) == frozenset(
        {Category.SPORTS}
    )


def test_primary_category_returns_single_category() -> None:
    assert primary_category(["Sports"]) is Category.SPORTS


def test_primary_category_priority_order_macro_over_elections() -> None:
    assert primary_category(["Global Elections", "Fed Rates"]) is Category.MACRO


def test_primary_category_no_match_falls_back_to_thesis() -> None:
    assert primary_category(["random-tag-not-in-taxonomy"]) is Category.THESIS


def test_primary_category_empty_tags_is_thesis() -> None:
    assert primary_category([]) is Category.THESIS


def test_primary_category_honors_tag_exclusions() -> None:
    assert primary_category(["Crypto", "Crypto Prices"]) is Category.THESIS
