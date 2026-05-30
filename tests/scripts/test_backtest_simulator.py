"""Integration tests for the backtest event-walk simulator."""

from __future__ import annotations

import csv
import sqlite3
import subprocess
import sys as _sys
from pathlib import Path

import pytest
from scripts.backtest_copy_sizing import (
    EdgeWeightedCausal,
    EqualWeight,
    Resolution,
    ResolutionEvent,
    Simulator,
    Trade,
    TradeEvent,
    build_parser,
    load_event_stream,
    load_watchlist,
    main,
    render_report,
)


def _make_daemon_db(path: Path, addresses: list[str]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE wallet_watchlist ("
        " address TEXT PRIMARY KEY,"
        " source TEXT NOT NULL,"
        " reason TEXT,"
        " added_at INTEGER NOT NULL,"
        " active INTEGER NOT NULL DEFAULT 1)"
    )
    for addr in addresses:
        conn.execute(
            "INSERT INTO wallet_watchlist (address, source, reason, added_at, active)"
            " VALUES (?, 'test', 'test', 1, 1)",
            (addr,),
        )
    conn.commit()
    conn.close()


def _make_corpus_db(
    path: Path,
    trades: list[tuple[str, str, str, str, float, float, int]],
    resolutions: list[tuple[str, int, int]],
) -> None:
    """trades: (wallet, condition_id, outcome_side, bs, price, notional, ts).

    resolutions: (condition_id, outcome_yes_won, resolved_at).
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE corpus_trades (
          platform TEXT NOT NULL DEFAULT 'polymarket',
          tx_hash TEXT NOT NULL,
          asset_id TEXT NOT NULL,
          wallet_address TEXT NOT NULL,
          condition_id TEXT NOT NULL,
          outcome_side TEXT NOT NULL,
          bs TEXT NOT NULL,
          price REAL NOT NULL,
          size REAL NOT NULL,
          notional_usd REAL NOT NULL,
          ts INTEGER NOT NULL,
          PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)
        );
        CREATE TABLE market_resolutions (
          platform TEXT NOT NULL DEFAULT 'polymarket',
          condition_id TEXT NOT NULL,
          winning_outcome_index INTEGER NOT NULL,
          outcome_yes_won INTEGER NOT NULL,
          resolved_at INTEGER NOT NULL,
          source TEXT NOT NULL,
          recorded_at INTEGER NOT NULL,
          PRIMARY KEY (platform, condition_id)
        );
        """
    )
    for i, (w, cid, side, bs, price, notional, ts) in enumerate(trades):
        conn.execute(
            "INSERT INTO corpus_trades VALUES ('polymarket', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"0xtx{i}",
                f"asset{i}",
                w,
                cid,
                side,
                bs,
                price,
                notional / price,
                notional,
                ts,
            ),
        )
    for cid, yes_won, resolved_at in resolutions:
        conn.execute(
            "INSERT INTO market_resolutions VALUES ('polymarket', ?, ?, ?, ?, 'test', ?)",
            (cid, 0 if yes_won else 1, yes_won, resolved_at, resolved_at),
        )
    conn.commit()
    conn.close()


def test_load_watchlist_returns_active_addresses(tmp_path: Path) -> None:
    db = tmp_path / "daemon.sqlite3"
    _make_daemon_db(db, ["0xa", "0xb"])
    addrs = load_watchlist(db)
    assert addrs == {"0xa", "0xb"}


def test_load_watchlist_excludes_inactive(tmp_path: Path) -> None:
    db = tmp_path / "daemon.sqlite3"
    _make_daemon_db(db, ["0xa"])
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO wallet_watchlist VALUES ('0xc', 'test', 'test', 1, 0)")
    conn.commit()
    conn.close()
    addrs = load_watchlist(db)
    assert addrs == {"0xa"}


def test_load_event_stream_orders_by_ts(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 100),
            ("0xa", "0xM2", "NO", "BUY", 0.3, 500.0, 300),
        ],
        resolutions=[
            ("0xM1", 1, 200),
            ("0xM2", 0, 400),
        ],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    assert [e.ts for e in events] == [100, 200, 300, 400]
    assert isinstance(events[0], TradeEvent)
    assert isinstance(events[1], ResolutionEvent)
    assert isinstance(events[3], ResolutionEvent)
    assert events[1].resolution.winning_side == "YES"
    assert events[3].resolution.winning_side == "NO"


def test_load_event_stream_excludes_sells(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "SELL", 0.5, 1_000.0, 100),
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 110),
        ],
        resolutions=[("0xM1", 1, 200)],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    trade_events = [e for e in events if isinstance(e, TradeEvent)]
    assert len(trade_events) == 1
    assert trade_events[0].ts == 110


def test_load_event_stream_excludes_non_watchlist(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 100),
            ("0xb", "0xM1", "YES", "BUY", 0.5, 1_000.0, 110),
        ],
        resolutions=[("0xM1", 1, 200)],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    trade_events = [e for e in events if isinstance(e, TradeEvent)]
    assert len(trade_events) == 1
    assert trade_events[0].trade.wallet == "0xa"


def test_load_event_stream_skips_unresolved_markets(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 100),
            ("0xa", "0xM2", "YES", "BUY", 0.5, 1_000.0, 110),
        ],
        resolutions=[("0xM1", 1, 200)],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    cids = {
        (e.trade.condition_id if isinstance(e, TradeEvent) else e.resolution.condition_id)
        for e in events
    }
    assert cids == {"0xM1"}


def test_render_report_includes_all_scheme_rows() -> None:
    schemes = [EqualWeight(position_fraction=0.01)]
    sim = Simulator(schemes=schemes, bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xa",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200))
    report = render_report(sim, schemes=schemes, bankroll=10_000.0)
    assert "equal_weight" in report
    assert "PnL" in report
    assert "Win rate" in report
    assert "100.00%" in report or "100.0%" in report
    assert "$100" in report


def test_render_report_includes_quarterly_grid_and_unresolved_count() -> None:
    schemes = [EqualWeight(position_fraction=0.01)]
    sim = Simulator(schemes=schemes, bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xa",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=1_700_000_000,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xa",
            condition_id="0xM2",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=1_700_000_100,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=1_700_000_500)
    )
    report = render_report(sim, schemes=schemes, bankroll=10_000.0)
    assert "Unresolved" in report
    assert "Quarterly" in report or "quarter" in report.lower()


def test_build_parser_has_expected_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.starting_bankroll_usd == 10_000.0
    assert args.position_fraction == 0.01
    assert args.min_multiplier == 0.10
    assert args.scale_factor == 0.01
    assert args.max_cost_per_trade == 1_000.0
    assert args.edge_scale == 5.0
    assert args.max_multiplier == 3.0
    assert args.min_trades_for_edge == 10
    assert args.platform == "polymarket"
    assert args.start_ts is None
    assert args.end_ts is None
    assert args.enforce_capacity is False
    assert args.max_open_exposure_usd is None
    assert args.max_open_exposure_frac is None


def test_build_parser_accepts_csv_path(tmp_path: Path) -> None:
    parser = build_parser()
    target = str(tmp_path / "x.csv")
    args = parser.parse_args(["--csv", target])
    assert args.csv == target


def test_build_parser_accepts_capacity_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--enforce-capacity", "--max-open-exposure-usd", "5000"])
    assert args.enforce_capacity is True
    assert args.max_open_exposure_usd == 5_000.0
    assert args.max_open_exposure_frac is None


def test_build_parser_accepts_exposure_frac() -> None:
    parser = build_parser()
    args = parser.parse_args(["--max-open-exposure-frac", "0.5"])
    assert args.max_open_exposure_frac == 0.5
    assert args.max_open_exposure_usd is None


def test_build_parser_rejects_both_exposure_flags() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-open-exposure-usd", "5000", "--max-open-exposure-frac", "0.5"])


def test_simulator_initializes_per_scheme_state() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert state.resolved_trades == []
    assert state.cumulative_pnl == 0.0
    assert state.nav_series == []
    assert state.open_cost == 0.0
    assert state.skipped_trades == 0


def test_simulator_books_pnl_on_resolution() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200))
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert len(state.resolved_trades) == 1
    rec = state.resolved_trades[0]
    assert rec.payout == 1.0
    assert rec.proceeds == 200.0
    assert rec.pnl == 100.0
    assert state.cumulative_pnl == 100.0
    assert state.nav_series == [(200, 100.0)]


def test_simulator_books_zero_payout_on_losing_outcome() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="NO", resolved_at=200))
    rec = sim.state_for(scheme).resolved_trades[0]
    assert rec.payout == 0.0
    assert rec.pnl == -100.0


def test_simulator_temporal_correctness_with_edge_weighted() -> None:
    """Resolutions must surface to EdgeWeightedCausal before any later trade.

    Scenario: 2 prior wins for wallet A at price 0.5 (edge=+0.5/trade),
    then a 3rd trade for A. After the 2 resolutions, the 3rd trade
    must be sized at the boosted multiplier.
    """
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=2,
    )
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM2",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=200,
        )
    )
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="YES", resolved_at=300))
    sim.on_resolution(Resolution(condition_id="0xM2", winning_side="YES", resolved_at=400))
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM3",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=500,
        )
    )
    state = sim.state_for(scheme)
    third = state.open_positions[next(iter(state.open_positions))]
    assert third.cost == 300.0


def test_simulator_resolves_multiple_open_positions_on_same_market() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xB",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=150,
        )
    )
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200))
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert len(state.resolved_trades) == 2


def test_simulator_unresolved_trade_stays_open() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 1
    assert state.resolved_trades == []


def _buy(cid: str, ts: int, *, wallet: str = "0xA", price: float = 0.5) -> Trade:
    return Trade(
        wallet=wallet,
        condition_id=cid,
        outcome_side="YES",
        price=price,
        notional_usd=1_000.0,
        ts=ts,
    )


def test_simulator_tracks_open_cost_across_open_and_close() -> None:
    scheme = EqualWeight(position_fraction=0.5)  # cost = 500 per trade
    sim = Simulator(schemes=[scheme], bankroll=1_000.0)
    sim.on_trade(_buy("0xM1", 100))
    sim.on_trade(_buy("0xM2", 110))
    state = sim.state_for(scheme)
    assert state.open_cost == 1_000.0
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="NO", resolved_at=200))
    assert state.open_cost == 500.0


def test_capacity_gate_disabled_by_default_overspends_bankroll() -> None:
    scheme = EqualWeight(position_fraction=0.5)  # cost = 500 per trade
    sim = Simulator(schemes=[scheme], bankroll=1_000.0)
    for i in range(4):
        sim.on_trade(_buy(f"0xM{i}", 100 + i))
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 4
    assert state.skipped_trades == 0


def test_capacity_gate_refuses_trade_when_capital_exhausted() -> None:
    scheme = EqualWeight(position_fraction=0.5)  # cost = 500 per trade
    sim = Simulator(schemes=[scheme], bankroll=1_000.0, enforce_capacity=True)
    sim.on_trade(_buy("0xM1", 100))  # available 1000 >= 500 -> open
    sim.on_trade(_buy("0xM2", 110))  # available 500 >= 500 -> open
    sim.on_trade(_buy("0xM3", 120))  # available 0 < 500 -> skip
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 2
    assert state.skipped_trades == 1


def test_capacity_gate_frees_capital_after_winning_resolution() -> None:
    scheme = EqualWeight(position_fraction=1.0)  # cost = 1000 (whole bankroll)
    sim = Simulator(schemes=[scheme], bankroll=1_000.0, enforce_capacity=True)
    sim.on_trade(_buy("0xM1", 100))  # opens, open_cost=1000
    sim.on_trade(_buy("0xM2", 110))  # available 0 < 1000 -> skip
    # Win at price 0.5: shares=2000, proceeds=2000, pnl=+1000.
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200))
    sim.on_trade(_buy("0xM3", 300))  # available 1000+1000-0=2000 >= 1000 -> open
    state = sim.state_for(scheme)
    assert state.skipped_trades == 1
    assert len(state.open_positions) == 1
    assert next(iter(state.open_positions.values())).condition_id == "0xM3"


def test_exposure_cap_disabled_by_default() -> None:
    scheme = EqualWeight(position_fraction=0.5)  # cost = 500 per trade
    sim = Simulator(schemes=[scheme], bankroll=1_000.0)
    sim.on_trade(_buy("0xM1", 100))
    sim.on_trade(_buy("0xM2", 110))
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 2
    assert state.skipped_trades == 0


def test_exposure_cap_refuses_trade_over_cap() -> None:
    scheme = EqualWeight(position_fraction=0.5)  # cost = 500 per trade
    sim = Simulator(schemes=[scheme], bankroll=1_000.0, max_open_exposure_usd=600.0)
    sim.on_trade(_buy("0xM1", 100))  # open_cost 0+500 <= 600 -> open
    sim.on_trade(_buy("0xM2", 110))  # open_cost 500+500=1000 > 600 -> skip
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 1
    assert state.skipped_trades == 1


def test_exposure_cap_frees_after_resolution() -> None:
    scheme = EqualWeight(position_fraction=0.5)  # cost = 500 per trade
    sim = Simulator(schemes=[scheme], bankroll=1_000.0, max_open_exposure_usd=600.0)
    sim.on_trade(_buy("0xM1", 100))  # open
    sim.on_resolution(Resolution(condition_id="0xM1", winning_side="NO", resolved_at=200))
    sim.on_trade(_buy("0xM2", 300))  # open_cost back to 0 -> 500 <= 600 -> open
    state = sim.state_for(scheme)
    assert state.skipped_trades == 0
    assert len(state.open_positions) == 1


def test_render_report_includes_skipped_column_and_capacity_note() -> None:
    schemes = [EqualWeight(position_fraction=0.5)]  # cost = 500 per trade
    sim = Simulator(schemes=schemes, bankroll=1_000.0, enforce_capacity=True)
    sim.on_trade(_buy("0xM1", 100))
    sim.on_trade(_buy("0xM2", 110))
    sim.on_trade(_buy("0xM3", 120))  # skipped
    report = render_report(
        sim,
        schemes=schemes,
        bankroll=1_000.0,
        enforce_capacity=True,
    )
    assert "Skipped" in report
    assert "Capacity enforcement" in report


def test_build_parser_causal_select_defaults() -> None:
    args = build_parser().parse_args(["--causal-select"])
    assert args.causal_select is True
    assert args.min_resolved == 20
    assert args.edge_window == 0
    assert args.rebalance_days == 14
    assert args.copy_top_k is None
    assert args.copy_capital_per_wallet is None
    assert args.copy_top_frac is None


def test_build_parser_causal_off_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.causal_select is False


def test_build_parser_rejects_two_copy_policies() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--copy-top-k", "10", "--copy-top-frac", "0.1"])


def test_main_causal_select_end_to_end(corpus_factory, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A qualifies positive and is copied; B never positive -> never copied.
    # Historical trades (h*) resolve before the second rebalance boundary
    # (at ts=86401 for --rebalance-days 1); the new-period trades (n1 for A,
    # n2 for B) fall inside that boundary's copy window. Selection must filter
    # B out so only A's n1 is booked.
    day = 86_400
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("B", "h3", "YES", "BUY", 0.90, 100.0, 1),
        ("B", "h4", "YES", "BUY", 0.90, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, day + 100),
        ("B", "n2", "YES", "BUY", 0.50, 100.0, day + 110),
    ]
    resolutions = [
        ("h1", 1, 50),
        ("h2", 1, 60),
        ("h3", 0, 50),
        ("h4", 0, 60),
        ("n1", 1, day + 300),
        ("n2", 1, day + 300),
    ]
    db = corpus_factory(trades, resolutions)
    csv_path = tmp_path / "out.csv"
    rc = main(
        [
            "--db",
            str(db),
            "--causal-select",
            "--min-resolved",
            "2",
            "--rebalance-days",
            "1",
            "--copy-top-k",
            "5",
            "--csv",
            str(csv_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Causal selection" in out
    assert "equal_weight" in out
    # equal_weight booked exactly 1 resolved trade (only A's n1; B's n2 filtered).
    assert "| equal_weight | 1 |" in out
    # Every booked row across all schemes belongs to wallet A — B was excluded.
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    equal_weight_rows = [r for r in rows if r["scheme"] == "equal_weight"]
    assert len(equal_weight_rows) == 1
    assert equal_weight_rows[0]["wallet"] == "A"
    assert equal_weight_rows[0]["condition_id"] == "n1"
    assert {r["wallet"] for r in rows} == {"A"}


def test_main_causal_select_top_frac_policy(corpus_factory, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Exercise the --copy-top-frac policy through main() (the top_k path is
    # covered above). Same corpus: A is positive-edge, B never positive, so the
    # positive-edge floor leaves only A regardless of the fraction; frac=1.0
    # copies all qualified (= A). Proves _resolve_policy + wl_size wiring for frac.
    day = 86_400
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("B", "h3", "YES", "BUY", 0.90, 100.0, 1),
        ("B", "h4", "YES", "BUY", 0.90, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, day + 100),
        ("B", "n2", "YES", "BUY", 0.50, 100.0, day + 110),
    ]
    resolutions = [
        ("h1", 1, 50), ("h2", 1, 60), ("h3", 0, 50), ("h4", 0, 60),
        ("n1", 1, day + 300), ("n2", 1, day + 300),
    ]
    db = corpus_factory(trades, resolutions)
    csv_path = tmp_path / "frac.csv"
    rc = main([
        "--db", str(db), "--causal-select", "--min-resolved", "2",
        "--rebalance-days", "1", "--copy-top-frac", "1.0", "--csv", str(csv_path),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "top_frac=1.0" in out  # policy threaded through to the header
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["wallet"] for r in rows} == {"A"}


def test_script_runs_via_direct_execution() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [_sys.executable, "scripts/backtest_copy_sizing.py", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--causal-select" in result.stdout
