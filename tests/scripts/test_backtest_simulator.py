"""Integration tests for the backtest event-walk simulator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
            "INSERT INTO corpus_trades VALUES "
            "('polymarket', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            "INSERT INTO market_resolutions VALUES "
            "('polymarket', ?, ?, ?, ?, 'test', ?)",
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
    conn.execute(
        "INSERT INTO wallet_watchlist VALUES ('0xc', 'test', 'test', 1, 0)"
    )
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
        (
            e.trade.condition_id
            if isinstance(e, TradeEvent)
            else e.resolution.condition_id
        )
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
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
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
        Resolution(
            condition_id="0xM1", winning_side="YES", resolved_at=1_700_000_500
        )
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


def test_build_parser_accepts_csv_path(tmp_path: Path) -> None:
    parser = build_parser()
    target = str(tmp_path / "x.csv")
    args = parser.parse_args(["--csv", target])
    assert args.csv == target


def test_simulator_initializes_per_scheme_state() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert state.resolved_trades == []
    assert state.cumulative_pnl == 0.0
    assert state.nav_series == []


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
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
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
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="NO", resolved_at=200)
    )
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
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=300)
    )
    sim.on_resolution(
        Resolution(condition_id="0xM2", winning_side="YES", resolved_at=400)
    )
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
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
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
