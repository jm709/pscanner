"""Tests for the pure insider-episode library."""

from __future__ import annotations

import pytest
from scripts.insider_episode_lib import (
    HoldoutLockedError,
    MarketRef,
    PublicCase,
    select_cases,
)


def _case(cid: str, split: str, *, markets: tuple[MarketRef, ...] = ()) -> PublicCase:
    return PublicCase(
        id=cid,
        name=cid,
        split=split,
        confidence="alleged",
        analyzable=True,
        news_ts=1_700_000_000,
        news_precision="minute",
        markets=markets,
    )


def test_primary_market_is_first_primary_role() -> None:
    case = _case(
        "c",
        "dev",
        markets=(
            MarketRef(slug="s-sibling", winning_side="YES", role="sibling"),
            MarketRef(slug="s-primary", winning_side="YES", role="primary (main)"),
            MarketRef(slug="s-primary-2", winning_side="YES", role="primary"),
        ),
    )
    assert case.primary is not None
    assert case.primary.slug == "s-primary"


def test_primary_market_none_when_no_primary_role() -> None:
    case = _case("c", "dev", markets=(MarketRef(slug="s", winning_side="YES", role="sibling"),))
    assert case.primary is None


def test_select_cases_dev_only_by_default() -> None:
    cases = [_case("a", "dev"), _case("b", "holdout")]
    assert [c.id for c in select_cases(cases, "dev", scorer_frozen=False)] == ["a"]


def test_select_cases_holdout_refused_without_frozen_flag() -> None:
    cases = [_case("a", "dev"), _case("b", "holdout")]
    with pytest.raises(HoldoutLockedError):
        select_cases(cases, "holdout", scorer_frozen=False)
    with pytest.raises(HoldoutLockedError):
        select_cases(cases, "all", scorer_frozen=False)


def test_select_cases_holdout_allowed_with_frozen_flag() -> None:
    cases = [_case("a", "dev"), _case("b", "holdout")]
    assert [c.id for c in select_cases(cases, "holdout", scorer_frozen=True)] == ["b"]
    assert [c.id for c in select_cases(cases, "all", scorer_frozen=True)] == ["a", "b"]
