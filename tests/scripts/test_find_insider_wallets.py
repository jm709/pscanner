"""Tests for insider-wallet discovery."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.find_insider_wallets import WalletAgg, split_cohorts, wallet_aggregates


def _agg(wallet: str, *, n: int, edge: float, pnl: float, first_ts: int) -> WalletAgg:
    return WalletAgg(
        wallet=wallet,
        n_resolved_buys=n,
        n_distinct_markets=n,
        first_ts=first_ts,
        last_ts=first_ts + 100,
        active_lifespan_days=1.0,
        total_notional_usd=1000.0,
        mean_bet_usd=100.0,
        max_bet_usd=500.0,
        mean_edge=edge,
        cash_pnl_usd=pnl,
        mean_entry_price=0.3,
        improbability_z=2.0,
        mean_ttr_days=5.0,
        prior_activity_count=0,
    )


def test_wallet_aggregates_basic_stats(corpus_factory) -> None:
    # Wallet A: 2 resolved buys. YES@0.10 wins, YES@0.50 loses.
    trades = [
        ("A", "m1", "YES", "BUY", 0.10, 100.0, 1_000),
        ("A", "m2", "YES", "BUY", 0.50, 200.0, 2_000),
    ]
    resolutions = [("m1", 1, 5_000), ("m2", 0, 6_000)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)

    aggs = {a.wallet: a for a in wallet_aggregates(db, max_trades=10, max_lifespan_days=30)}
    a = aggs["A"]
    assert a.n_resolved_buys == 2
    assert a.n_distinct_markets == 2
    assert a.max_bet_usd == 200.0
    # cash PnL: win -> 100*(1-0.10)/0.10 = 900 ; loss -> -200 ; net 700
    assert math.isclose(a.cash_pnl_usd, 700.0, rel_tol=1e-9)
    # mean edge: ((1-0.10) + (0-0.50)) / 2 = 0.20
    assert math.isclose(a.mean_edge, 0.20, rel_tol=1e-9)
    # improbability z: (obs_wins - exp) / sqrt(sum p(1-p))
    #   obs=1, exp=0.10+0.50=0.60, var=0.10*0.90+0.50*0.50=0.34
    assert math.isclose(a.improbability_z, (1 - 0.60) / math.sqrt(0.34), rel_tol=1e-9)


def test_shape_gate_excludes_too_many_trades(corpus_factory) -> None:
    trades = [("A", f"m{i}", "YES", "BUY", 0.5, 50.0, 1_000 + i) for i in range(11)]
    resolutions = [(f"m{i}", 1, 5_000) for i in range(11)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    aggs = wallet_aggregates(db, max_trades=10, max_lifespan_days=30)
    assert aggs == []  # 11 > max_trades=10


def test_shape_gate_excludes_long_lifespan(corpus_factory) -> None:
    # two trades 31 days apart
    trades = [
        ("A", "m1", "YES", "BUY", 0.5, 50.0, 1_000),
        ("A", "m2", "YES", "BUY", 0.5, 50.0, 1_000 + 31 * 86_400),
    ]
    resolutions = [("m1", 1, 1_000 + 40 * 86_400), ("m2", 1, 1_000 + 40 * 86_400)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    assert wallet_aggregates(db, max_trades=10, max_lifespan_days=30) == []


def test_improbability_z_single_cheap_win_is_large(corpus_factory) -> None:
    trades = [("A", "m1", "YES", "BUY", 0.05, 100.0, 1_000)]
    db: Path = corpus_factory(trades, [("m1", 1, 5_000)], with_platform=True)
    a = wallet_aggregates(db, max_trades=10, max_lifespan_days=30)[0]
    # z = (1 - 0.05) / sqrt(0.05*0.95) ~= 4.36
    assert a.improbability_z > 4.0


def test_improbability_z_zero_variance_is_zero(corpus_factory) -> None:
    # single bet at price 1.0 -> variance 0 -> NULLIF guard -> coalesced 0.0
    trades = [("A", "m1", "YES", "BUY", 1.0, 100.0, 1_000)]
    db: Path = corpus_factory(trades, [("m1", 1, 5_000)], with_platform=True)
    a = wallet_aggregates(db, max_trades=10, max_lifespan_days=30)[0]
    assert a.improbability_z == 0.0


def test_aggregates_runs_without_platform_column(corpus_factory) -> None:
    trades = [("A", "m1", "YES", "BUY", 0.2, 100.0, 1_000)]
    db: Path = corpus_factory(trades, [("m1", 1, 5_000)], with_platform=False)
    assert len(wallet_aggregates(db, max_trades=10, max_lifespan_days=30)) == 1


def test_split_cohorts_cases_and_matched_controls() -> None:
    aggs = [_agg(f"win{i}", n=2, edge=0.3, pnl=500.0, first_ts=1_000) for i in range(2)]
    aggs += [_agg(f"lose{i}", n=2, edge=-0.2, pnl=-100.0, first_ts=1_000) for i in range(10)]
    cases, controls = split_cohorts(aggs, control_ratio=3, seed=0)
    assert {c.wallet for c in cases} == {"win0", "win1"}
    assert len(controls) == 6  # 2 cases * ratio 3, same (n, era) stratum
    assert all(c.cash_pnl_usd <= 0 for c in controls)


def test_split_cohorts_degrades_when_controls_scarce() -> None:
    aggs = [_agg("win0", n=2, edge=0.3, pnl=500.0, first_ts=1_000)]
    aggs += [_agg("lose0", n=2, edge=-0.2, pnl=-100.0, first_ts=1_000)]
    cases, controls = split_cohorts(aggs, control_ratio=3, seed=0)
    assert len(cases) == 1
    assert len(controls) == 1  # only one control available; no error
