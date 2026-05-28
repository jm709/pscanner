"""Opt-in smoke test for the backtest script against the production corpus.

Skipped by default; run with ``uv run pytest -m slow``.

PAUSE BEFORE RUNNING -- operator approval is required before this touches
production data. See the implementation plan and the spec at
``docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from scripts.backtest_copy_sizing import (
    ConcentrationCapped,
    EdgeWeightedCausal,
    EqualWeight,
    FollowSeedSize,
    Simulator,
    TradeEvent,
    load_event_stream,
    load_watchlist,
    render_report,
)

pytestmark = pytest.mark.slow

_CORPUS_DB = Path("data/corpus.sqlite3")
_DAEMON_DB = Path("data/pscanner.sqlite3")


def _skip_if_missing() -> None:
    if not _CORPUS_DB.exists():
        pytest.skip(f"{_CORPUS_DB} not present; smoke test requires production data")
    if not _DAEMON_DB.exists():
        pytest.skip(f"{_DAEMON_DB} not present; smoke test requires daemon DB")


def test_backtest_produces_finite_report_against_corpus() -> None:
    _skip_if_missing()
    watchlist = load_watchlist(_DAEMON_DB)
    assert watchlist, "watchlist is empty; nothing to backtest"
    limited = set(sorted(watchlist)[:10])
    bankroll = 10_000.0
    schemes = [
        EqualWeight(position_fraction=0.01),
        ConcentrationCapped(
            position_fraction=0.01, min_multiplier=0.10, watchlist_size=len(limited)
        ),
        FollowSeedSize(scale_factor=0.01, max_cost_per_trade=1_000.0),
        EdgeWeightedCausal(
            position_fraction=0.01,
            edge_scale=5.0,
            min_multiplier=0.25,
            max_multiplier=3.0,
            min_trades_for_edge=10,
        ),
    ]
    sim = Simulator(schemes=schemes, bankroll=bankroll)
    n_trades = 0
    n_resolutions = 0
    for event in load_event_stream(_CORPUS_DB, watchlist=limited, platform="polymarket"):
        if isinstance(event, TradeEvent):
            sim.on_trade(event.trade)
            n_trades += 1
        else:
            sim.on_resolution(event.resolution)
            n_resolutions += 1
    assert n_trades > 0, "no eligible trades found for the limited watchlist"
    report = render_report(sim, schemes=schemes, bankroll=bankroll)
    for scheme in schemes:
        assert scheme.name in report
        state = sim.state_for(scheme)
        for r in state.resolved_trades:
            assert math.isfinite(r.pnl), f"non-finite PnL in {scheme.name}"
