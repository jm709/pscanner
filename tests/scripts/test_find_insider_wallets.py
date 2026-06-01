"""Tests for insider-wallet discovery."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.find_insider_wallets import WalletAgg, wallet_aggregates


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
