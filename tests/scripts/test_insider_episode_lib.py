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


from scripts.insider_episode_lib import window_features  # noqa: E402


def _tr(wallet: str, ts: int, side: str, bs: str, price: float, notional: float) -> WindowTrade:
    return WindowTrade(
        wallet=wallet, ts=ts, outcome_side=side, bs=bs, price=price, notional=notional
    )


def test_window_features_sums_winning_buys_only() -> None:
    trades = [
        _tr("a", 100, "YES", "BUY", 0.10, 1000.0),
        _tr("a", 200, "NO", "BUY", 0.80, 500.0),  # wrong side, ignored for n_w
        _tr("a", 300, "YES", "SELL", 0.30, 200.0),  # sell, ignored
        _tr("a", 400, "YES", "BUY", 0.20, 2000.0),
        _tr("b", 150, "YES", "BUY", 0.10, 50.0),
    ]
    n_w, n_buys, first, last, flip = window_features(trades, "a", "YES")
    assert n_w == 3000.0
    assert n_buys == 2
    assert (first, last) == (100, 400)
    assert flip is False


def test_window_features_flip_requires_opposing_sell_before_first_buy() -> None:
    flipper = [
        _tr("f", 100, "NO", "SELL", 0.90, 500.0),
        _tr("f", 200, "YES", "BUY", 0.15, 1000.0),
    ]
    assert window_features(flipper, "f", "YES")[4] is True
    late_sell = [
        _tr("g", 100, "YES", "BUY", 0.15, 1000.0),
        _tr("g", 200, "NO", "SELL", 0.90, 500.0),  # after the buy: not a flip
    ]
    assert window_features(late_sell, "g", "YES")[4] is False


import math  # noqa: E402

from scripts.insider_episode_lib import (  # noqa: E402
    Provenance,
    SealedCase,
    WalletFeatures,
    age_days,
    compare_published,
    freshness,
    rank_wallets,
    score,
)

DAY = 86_400


def _f(
    wallet: str,
    *,
    n_w: float,
    breadth: int = 1,
    n_life: float | None = None,
    flip: bool = False,
    first_buy_ts: int = 100 * DAY,
    corpus_first_ts: int = 99 * DAY,
) -> WalletFeatures:
    return WalletFeatures(
        wallet=wallet,
        n_w=n_w,
        n_buys=1,
        first_buy_ts=first_buy_ts,
        last_buy_ts=first_buy_ts,
        flip=flip,
        breadth=breadth,
        n_life=n_w if n_life is None else n_life,
        corpus_first_ts=corpus_first_ts,
    )


def test_freshness_plateau_decay_floor() -> None:
    assert freshness(0) == 1.0
    assert freshness(10) == 1.0
    assert math.isclose(freshness(40), 0.5)
    assert math.isclose(freshness(64), 0.1)
    assert freshness(400) == 0.1


def test_age_days_prefers_api_then_corpus_and_never_negative() -> None:
    f = _f("w", n_w=100.0, first_buy_ts=100 * DAY, corpus_first_ts=98 * DAY)
    assert math.isclose(age_days(f, Provenance("w", first_activity_ts=90 * DAY, n_events=1)), 10.0)
    assert math.isclose(age_days(f, Provenance("w", first_activity_ts=None, n_events=0)), 2.0)
    assert math.isclose(age_days(f, None), 2.0)
    api_after_buy = Provenance("w", first_activity_ts=101 * DAY, n_events=1)
    assert age_days(f, api_after_buy) == 0.0


def test_score_matches_amendment_formula() -> None:
    f = _f("w", n_w=9_999.0, breadth=8, n_life=19_998.0, flip=True)  # conc = 0.5
    p = Provenance("w", first_activity_ts=100 * DAY - 40 * DAY, n_events=1)  # age 40 -> 0.5
    expected = math.log10(1 + 9_999.0) * 0.5 * (0.5 + 0.5 * 0.5) / math.log10(2 + 8) + 0.5
    assert math.isclose(score(f, p), expected)


def test_score_concentration_exactly_one_when_window_is_whole_life() -> None:
    f = _f("w", n_w=1_000.0, n_life=1_000.0)
    assert math.isclose(score(f, None), math.log10(1_001.0) * 1.0 * 1.0 / math.log10(3))


def test_rank_wallets_order_and_tiebreaks() -> None:
    feats = [
        _f("0xbb", n_w=1_000.0),
        _f("0xaa", n_w=1_000.0),  # identical score, n_w, first_buy: wallet asc wins
        _f("0xmm", n_w=50_000.0, breadth=900, n_life=5_000_000.0),  # market-maker, low score
        _f("0xff", n_w=200.0, flip=True),
    ]
    ranked = rank_wallets(feats, {})
    order = [r.features.wallet for r in ranked]
    assert order[:2] == ["0xaa", "0xbb"]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]
    assert order.index("0xmm") > order.index("0xaa")


def test_rank_wallets_fresh_single_market_beats_market_maker_at_equal_notional() -> None:
    fresh = _f("0xfresh", n_w=20_000.0, breadth=1, n_life=20_000.0)
    mm = _f("0xmm", n_w=20_000.0, breadth=800, n_life=30_000_000.0, corpus_first_ts=0)
    ranked = rank_wallets([mm, fresh], {})
    assert ranked[0].features.wallet == "0xfresh"


def test_compare_published_ranks_and_pass() -> None:
    feats = [_f(f"0x{i:02d}", n_w=float(1_000 - i)) for i in range(15)]
    ranked = rank_wallets(feats, {})
    sealed = SealedCase(id="c", wallets=("0x03", "0x12", "0xzz"))
    res = compare_published("c", "news", ranked, sealed)
    assert res.n_candidates == 15
    assert res.ranks == {"0x03": 4, "0x12": 13, "0xzz": None}
    assert res.best_rank == 4
    assert res.passed is True
    res2 = compare_published("c", "news", ranked, SealedCase(id="c", wallets=("0x12",)))
    assert res2.passed is False
    res3 = compare_published("c", "news", ranked, SealedCase(id="c", wallets=("0xzz",)))
    assert res3.best_rank is None
    assert res3.passed is None


def test_compare_published_is_case_insensitive_on_addresses() -> None:
    ranked = rank_wallets([_f("0xabc", n_w=1.0)], {})
    res = compare_published("c", "news", ranked, SealedCase(id="c", wallets=("0xABC",)))
    assert res.ranks == {"0xabc": 1}
