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


from scripts.insider_episode_lib import (  # noqa: E402
    HourBucket,
    WindowTrade,
    auto_anchor,
    hourly_buckets,
)

H = 3600


def _t(ts: int, side: str, price: float, notional: float = 100.0, wallet: str = "w") -> WindowTrade:
    return WindowTrade(
        wallet=wallet, ts=ts, outcome_side=side, bs="BUY", price=price, notional=notional
    )


def test_hourly_buckets_side_normalizes_and_volume_weights() -> None:
    trades = [
        _t(10 * H + 5, "YES", 0.20, notional=100.0),
        _t(10 * H + 50, "NO", 0.70, notional=300.0),  # YES-side prob 0.30
        _t(12 * H, "YES", 0.50),
    ]
    b = hourly_buckets(trades, "YES")
    assert [x.hour_start for x in b] == [10 * H, 12 * H]
    assert b[0].n_trades == 2
    assert b[0].notional == 400.0
    assert abs(b[0].side_prob_vwap - (0.20 * 100 + 0.30 * 300) / 400) < 1e-9


def test_hourly_buckets_for_no_winner_flips_prices() -> None:
    b = hourly_buckets([_t(0, "YES", 0.80)], "NO")
    assert abs(b[0].side_prob_vwap - 0.20) < 1e-9


def _flat(n: int, p: float) -> list[HourBucket]:
    return [
        HourBucket(hour_start=i * H, side_prob_vwap=p, notional=100.0, n_trades=1) for i in range(n)
    ]


def _bucket(i: int, p: float) -> HourBucket:
    return HourBucket(hour_start=i * H, side_prob_vwap=p, notional=100.0, n_trades=1)


def test_auto_anchor_finds_start_of_largest_jump() -> None:
    buckets = _flat(40, 0.10)
    # jump begins at hour 30: 0.10 -> 0.45 over hours 30..33
    for i, p in zip(range(30, 34), (0.10, 0.25, 0.40, 0.45), strict=True):
        buckets[i] = _bucket(i, p)
    assert auto_anchor(buckets, resolved_at=100 * H) == 30 * H


def test_auto_anchor_none_when_move_below_minimum() -> None:
    buckets = _flat(40, 0.10)
    buckets[20] = _bucket(20, 0.15)
    assert auto_anchor(buckets, resolved_at=100 * H) is None


def test_auto_anchor_ignores_buckets_at_or_after_resolution() -> None:
    buckets = _flat(10, 0.10)
    buckets[9] = _bucket(9, 0.99)
    # resolution at hour 9: the 0.99 bucket is the resolution snap, not a pre-move
    assert auto_anchor(buckets, resolved_at=9 * H) is None


def test_auto_anchor_empty_input() -> None:
    assert auto_anchor([], resolved_at=1) is None
