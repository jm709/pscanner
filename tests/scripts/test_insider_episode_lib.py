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


def test_age_days_takes_earliest_known_activity() -> None:
    f = _f("w", n_w=100.0, first_buy_ts=100 * DAY, corpus_first_ts=98 * DAY)
    assert math.isclose(age_days(f, Provenance("w", first_activity_ts=90 * DAY, n_events=1)), 10.0)
    assert math.isclose(age_days(f, Provenance("w", first_activity_ts=None, n_events=0)), 2.0)
    assert math.isclose(age_days(f, None), 2.0)
    # a truncated API walk can only see recent events: the corpus first trade is earlier and wins
    api_after_buy = Provenance("w", first_activity_ts=101 * DAY, n_events=1, truncated=True)
    assert math.isclose(age_days(f, api_after_buy), 2.0)


def test_auto_anchor_only_scans_within_horizon() -> None:
    # 20k flat buckets then a jump; must finish quickly (horizon-bounded inner loop)
    buckets = _flat(20_000, 0.10)
    buckets[19_990] = _bucket(19_990, 0.60)
    assert auto_anchor(buckets, resolved_at=30_000 * H) == 19_989 * H


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


from scripts.insider_episode_lib import MarketRow, match_controls, sibling_exclusions  # noqa: E402


def _m(
    cid: str, *, cat: str = "geo", vol: float = 10e6, res: int = 1_000 * DAY, yes: int = 1
) -> MarketRow:
    return MarketRow(
        condition_id=cid,
        slug=f"slug-{cid}",
        category=cat,
        volume=vol,
        resolved_at=res,
        yes_won=yes,
        truncated=False,
    )


def test_match_controls_filters_and_orders() -> None:
    case = _m("case")
    universe = [
        case,
        _m("c_close", vol=11e6),
        _m("c_far", vol=29e6),
        _m("c_too_big", vol=31e6),
        _m("c_too_small", vol=3e6),
        _m("c_cat", cat="sports"),
        _m("c_era", res=1_000 * DAY + 181 * DAY),
        _m("c_side", yes=0),
        _m("c_registry"),
        _m("c_exact", vol=10e6),
    ]
    out = match_controls(case, universe, excluded={"c_registry"}, k=5)
    assert [m.condition_id for m in out] == ["c_exact", "c_close", "c_far"]


def test_match_controls_respects_k_and_is_deterministic() -> None:
    case = _m("case")
    universe = [case] + [_m(f"c{i}", vol=10e6) for i in range(9)]
    out = match_controls(case, universe, excluded=set(), k=3)
    assert [m.condition_id for m in out] == ["c0", "c1", "c2"]


def test_match_controls_empty_when_nothing_matches() -> None:
    case = _m("case")
    assert match_controls(case, [case], excluded=set()) == []


from scripts.insider_episode_lib import (  # noqa: E402
    BlindResult,
    CaseReport,
    ControlResult,
    CoverageRow,
    MarketResult,
    csv_rows,
    fingerprint_table,
    render_case_report,
    render_summary,
)


def _ranked(prefix: str, n: int, *, n_w: float = 1_000.0) -> list:
    return rank_wallets([_f(f"0x{prefix}{i:02d}", n_w=n_w - i) for i in range(n)], {})


def _report(*, with_controls: bool, holdout: bool) -> CaseReport:
    case = PublicCase(
        id="case-x",
        name="Case X",
        split="holdout" if holdout else "dev",
        confidence="alleged",
        analyzable=True,
        news_ts=1_700_000_000,
        news_precision="minute",
        markets=(MarketRef("slug-x", "YES", "primary", "cidx"),),
    )
    primary = MarketResult(case.markets[0], "news", 1_700_000_000, _ranked("p", 12), 40, 12_345.0)
    controls = (
        [ControlResult(_m("ctl1"), 999, _ranked("c", 12, n_w=100.0))] if with_controls else []
    )
    blind = [BlindResult("case-x", "news", 12, {"0xp03": 4}, 4, True)] if holdout else []
    return CaseReport(case, [primary], controls, blind, auto_anchor_ts=1_700_000_000 - 7_200)


def test_fingerprint_table_empty_without_controls() -> None:
    r = _report(with_controls=False, holdout=False)
    assert fingerprint_table(r.results[0].ranked, r.controls) == []


def test_fingerprint_table_has_five_features_with_controls() -> None:
    r = _report(with_controls=True, holdout=False)
    rows = fingerprint_table(r.results[0].ranked, r.controls)
    assert [x[0] for x in rows] == ["age", "log10_n_w", "breadth", "concentration", "flip"]
    log_row = rows[1]
    assert log_row[1] > log_row[2]  # case top-10 has larger notional than control top-10


def test_render_case_report_mentions_no_controls() -> None:
    text = render_case_report(_report(with_controls=False, holdout=False))
    assert "# Case X" in text
    assert "no controls" in text.lower()
    assert "Blind test" not in text  # dev case


def test_render_case_report_holdout_includes_blind_result() -> None:
    text = render_case_report(_report(with_controls=True, holdout=True))
    assert "Blind test" in text
    assert "0xp03" in text
    assert "4 / 12" in text
    assert "PASS" in text


def test_render_summary_and_csv_rows() -> None:
    reports = [
        _report(with_controls=True, holdout=True),
        _report(with_controls=False, holdout=False),
    ]
    cov = [CoverageRow("case-x", "0xmissing", "outside_window", "first -90.0h")]
    text = render_summary(reports, cov)
    assert "Median best rank" in text
    assert "0xmissing" in text
    assert "outside_window" in text
    rows = csv_rows(reports)
    assert len(rows) == 12 + 12 + 12  # primary + control (holdout report) + primary (dev report)
    assert {r["is_control"] for r in rows} == {False, True}
    assert set(rows[0]) >= {"case_id", "market_slug", "rank", "wallet", "score", "age_days", "flip"}


def test_compare_published_carries_market_and_sample_label() -> None:
    ranked = rank_wallets([_f("0xabc", n_w=1.0)], {})
    sealed = SealedCase(id="c", wallets=("0xabc",))
    res = compare_published("c", "news", ranked, sealed, market_slug="slug-p", in_sample=True)
    assert res.market_slug == "slug-p"
    assert res.in_sample is True


def _blind(
    case_id: str, n: int, best: int | None, *, in_sample: bool, kind: str = "news"
) -> BlindResult:
    ranks: dict[str, int | None] = {} if best is None else {"0xw": best}
    passed = None if best is None else best <= 10
    return BlindResult(case_id, kind, n, ranks, best, passed, market_slug="s", in_sample=in_sample)


def _report_with_blind(case_id: str, split: str, blind: list[BlindResult]) -> CaseReport:
    case = PublicCase(
        id=case_id, name=case_id, split=split, confidence="alleged", analyzable=True,
        news_ts=1_700_000_000, news_precision="minute",
        markets=(MarketRef("s", "YES", "primary", "cid"),),
    )  # fmt: skip
    primary = MarketResult(case.markets[0], "news", 1_700_000_000, _ranked("p", 12), 40, 1.0)
    return CaseReport(case, [primary], [], blind, auto_anchor_ts=None)


def test_render_summary_loo_banner_and_small_case_exclusion() -> None:
    reports = [
        _report_with_blind("h1", "holdout", [_blind("h1", 500, 3, in_sample=False)]),
        _report_with_blind("d1", "dev", [_blind("d1", 400, 7, in_sample=True)]),
        _report_with_blind("d2", "dev", [_blind("d2", 5, 1, in_sample=True)]),  # <10 candidates
    ]
    text = render_summary(reports, [])
    assert "Median best rank (news anchor, holdout, 1 testable" in text
    assert "fewer than three testable holdout cases" in text.lower()
    assert "all testable cases" in text.lower()
    assert "d2" in text
    assert "excluded" in text.lower()


def test_render_case_report_labels_in_sample_blind() -> None:
    rep = _report_with_blind("d1", "dev", [_blind("d1", 400, 7, in_sample=True)])
    text = render_case_report(rep)
    assert "Blind test (in-sample" in text
    assert "7 / 400" in text
    assert "top 10" in text


def test_sibling_exclusions_by_event_slug() -> None:
    u = [
        MarketRow("a", "slug-a", "geo", 1.0, 1, 1, False, event_slug="ev-1"),
        MarketRow("b", "slug-b", "geo", 1.0, 1, 1, False, event_slug="ev-1"),
        MarketRow("c", "slug-c", "geo", 1.0, 1, 1, False, event_slug="ev-2"),
        MarketRow("d", "slug-d", "geo", 1.0, 1, 1, False, event_slug=""),
        MarketRow("e", "slug-e", "geo", 1.0, 1, 1, False, event_slug=""),
    ]
    assert sibling_exclusions(u, {"a", "d"}) == {"a", "b", "d"}
