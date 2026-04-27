"""Tests for the ``pscanner`` command-line entrypoint.

The ``filterwarnings`` marker silences ``PytestUnraisableExceptionWarning``
from upstream tests' httpx/respx fixtures: when an async test fixture
leaves a transport open in a reference cycle, garbage collection during
the next test boundary emits a ``ResourceWarning``. Those leaks don't
originate here, but pytest's ``--strict`` warnings policy surfaces them
on whichever test happens to come next.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.cli import main
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, PaperTradesRepo, WatchlistRepo

pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning",
)


@pytest.fixture(autouse=True)
def _flush_unraisable() -> Iterator[None]:
    """Run garbage collection so prior tests' leaks surface before this test."""
    gc.collect()
    yield
    gc.collect()


def _write_config(path: Path, db_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[scanner]",
                f'db_path = "{db_path}"',
                'log_level = "INFO"',
                "",
                "[smart_money]",
                "enabled = false",
                "",
                "[mispricing]",
                "enabled = false",
                "",
                "[whales]",
                "enabled = false",
                "",
                "[ratelimit]",
                "gamma_rpm = 50",
                "data_rpm = 50",
            ],
        )
    )


def test_main_run_once_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)

    fake_scanner = MagicMock()
    fake_scanner.run_once = AsyncMock(
        return_value={
            "events_scanned": 10,
            "alerts_emitted": 0,
            "tracked_wallets": 0,
            "markets_cached": 100,
        },
    )
    fake_scanner.aclose = AsyncMock()

    def fake_ctor(*args: object, **kwargs: object) -> MagicMock:
        return fake_scanner

    monkeypatch.setattr("pscanner.cli.Scanner", fake_ctor)
    rc = main(["--config", str(cfg), "run", "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "events_scanned" in out
    assert "100" in out
    fake_scanner.run_once.assert_awaited_once()
    fake_scanner.aclose.assert_awaited_once()


def test_main_run_once_propagates_failure_as_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)

    fake_scanner = MagicMock()
    fake_scanner.run_once = AsyncMock(side_effect=RuntimeError("boom"))
    fake_scanner.aclose = AsyncMock()
    monkeypatch.setattr("pscanner.cli.Scanner", lambda *a, **kw: fake_scanner)

    rc = main(["--config", str(cfg), "run", "--once"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "run --once failed" in err


def test_main_status_prints_alerts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    conn = init_db(db)
    repo = AlertsRepo(conn)
    repo.insert_if_new(
        Alert(
            detector="mispricing",
            alert_key="m1",
            severity="med",
            title="example mispricing",
            body={"event_id": "evt-1", "price_sum": 1.07},
            created_at=1000,
        ),
    )
    repo.insert_if_new(
        Alert(
            detector="smart_money",
            alert_key="s1",
            severity="low",
            title="example smart-money",
            body={"wallet": "0xabc"},
            created_at=2000,
        ),
    )
    conn.close()

    rc = main(["--config", str(cfg), "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "example mispricing" in out
    assert "example smart-money" in out
    assert "mispricing" in out


def test_main_status_no_db(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "missing.sqlite3"
    _write_config(cfg, db)
    rc = main(["--config", str(cfg), "status"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no database" in err


def test_main_status_empty_alerts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    conn = init_db(db)
    conn.close()
    rc = main(["--config", str(cfg), "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no alerts" in out


def test_main_missing_config_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nonexistent = tmp_path / "does-not-exist.toml"
    rc = main(["--config", str(nonexistent), "run", "--once"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "config file not found" in err


def test_main_invalid_config_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[scanner]\nlog_level = 42\n",  # invalid type
    )
    rc = main(["--config", str(cfg), "run", "--once"])
    assert rc == 2


def test_main_no_subcommand_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_main_run_daemon_clean_exit_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)

    fake_scanner = MagicMock()
    fake_scanner.run = AsyncMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr("pscanner.cli.Scanner", lambda *a, **kw: fake_scanner)

    rc = main(["--config", str(cfg), "run"])
    assert rc == 0


def test_main_run_daemon_unexpected_failure_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)

    fake_scanner = MagicMock()
    fake_scanner.run = AsyncMock(side_effect=RuntimeError("kaboom"))
    monkeypatch.setattr("pscanner.cli.Scanner", lambda *a, **kw: fake_scanner)

    rc = main(["--config", str(cfg), "run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "kaboom" in err


def test_status_table_uses_existing_schema(tmp_path: Path) -> None:
    """Sanity: ensure status path opens the SQLite with the expected schema."""
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    conn = init_db(db)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'",
    ).fetchall()
    table_names = {r[0] for r in rows}
    assert "alerts" in table_names
    conn.close()
    # Re-open via CLI flow indirectly by calling main.
    rc = main(["--config", str(cfg), "status"])
    assert rc == 0


def test_main_run_once_via_no_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --config, Config.load uses defaults — Scanner gets built normally."""
    fake_scanner = MagicMock()
    fake_scanner.run_once = AsyncMock(
        return_value={
            "events_scanned": 0,
            "alerts_emitted": 0,
            "tracked_wallets": 0,
            "markets_cached": 0,
        },
    )
    fake_scanner.aclose = AsyncMock()
    monkeypatch.setattr("pscanner.cli.Scanner", lambda *a, **kw: fake_scanner)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PSCANNER_CONFIG", raising=False)
    rc = main(["run", "--once"])
    assert rc == 0


def test_main_watch_inserts_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    rc = main(["--config", str(cfg), "watch", "0xabc", "--reason", "test"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0xabc" in out
    conn = init_db(db)
    try:
        row = WatchlistRepo(conn).get("0xabc")
    finally:
        conn.close()
    assert row is not None
    assert row.source == "manual"
    assert row.reason == "test"
    assert row.active is True


def test_main_watch_is_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    rc1 = main(["--config", str(cfg), "watch", "0xabc"])
    rc2 = main(["--config", str(cfg), "watch", "0xabc"])
    assert rc1 == 0
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "already in watchlist" in out


def test_main_unwatch_flips_active_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    main(["--config", str(cfg), "watch", "0xabc"])
    rc = main(["--config", str(cfg), "unwatch", "0xabc"])
    assert rc == 0
    conn = init_db(db)
    try:
        row = WatchlistRepo(conn).get("0xabc")
    finally:
        conn.close()
    assert row is not None
    assert row.active is False
    out = capsys.readouterr().out
    assert "unwatched" in out


def test_main_unwatch_unknown_address_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    rc = main(["--config", str(cfg), "unwatch", "0xnotpresent"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "not in watchlist" in out


def test_main_watchlist_prints_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    main(["--config", str(cfg), "watch", "0xabc", "--reason", "first"])
    main(["--config", str(cfg), "watch", "0xdef", "--reason", "second"])
    capsys.readouterr()  # discard prior output
    rc = main(["--config", str(cfg), "watchlist"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0xabc" in out
    assert "0xdef" in out
    assert "manual" in out


def test_main_watchlist_empty(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "test.sqlite3"
    _write_config(cfg, db)
    rc = main(["--config", str(cfg), "watchlist"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "empty" in out


def test_paper_status_renders_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``pscanner paper status`` prints the bankroll/NAV summary, counts,
    realized PnL, top-N best/worst, and per-wallet leaderboard.
    """
    cfg = tmp_path / "config.toml"
    db = tmp_path / "pscanner.sqlite3"
    _write_config(cfg, db)
    conn = init_db(db)
    try:
        repo = PaperTradesRepo(conn)
        p1 = repo.insert_entry(
            triggering_alert_key="smart:0xa:0xc:yes:1",
            triggering_alert_detector="smart_money",
            rule_variant=None,
            source_wallet="0xa",
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=0.5,
            cost_usd=10.0,
            nav_after_usd=990.0,
            ts=1700000000,
        )
        repo.insert_entry(
            triggering_alert_key="smart:0xb:0xc:no:2",
            triggering_alert_detector="smart_money",
            rule_variant=None,
            source_wallet="0xb",
            condition_id=ConditionId("0xc2"),
            asset_id=AssetId("a-n"),
            outcome="no",
            shares=20.0,
            fill_price=0.5,
            cost_usd=10.0,
            nav_after_usd=980.0,
            ts=1700000010,
        )
        repo.insert_exit(
            parent_trade_id=p1,
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=1.0,
            cost_usd=20.0,
            nav_after_usd=1000.0,
            ts=1700000100,
        )
    finally:
        conn.close()

    rc = main(["--config", str(cfg), "paper", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "starting bankroll" in out.lower()
    # Current NAV after one $10 win
    assert "1010" in out or "1,010" in out
    # Counts visible somehow (open=1 closed=1)
    assert "open" in out.lower()
    assert "closed" in out.lower()
    # Realized PnL line
    assert "realized" in out.lower()
    # Per-wallet leaderboard mentions both wallets
    assert "0xa" in out
    assert "0xb" in out


def test_paper_status_empty_db(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "empty.sqlite3"
    _write_config(cfg, db)
    init_db(db).close()
    rc = main(["--config", str(cfg), "paper", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    # Empty case still renders the headline numbers
    assert "open" in out.lower()
    assert "closed" in out.lower()


def test_paper_status_shows_per_source_breakdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "pscanner.sqlite3"
    _write_config(cfg, db)
    conn = init_db(db)
    try:
        repo = PaperTradesRepo(conn)
        # smart_money entry (resolved win)
        e1 = repo.insert_entry(
            triggering_alert_key="smart:0xa:1",
            triggering_alert_detector="smart_money",
            rule_variant=None,
            source_wallet="0xa",
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=0.5,
            cost_usd=10.0,
            nav_after_usd=1000.0,
            ts=1700000000,
        )
        repo.insert_exit(
            parent_trade_id=e1,
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=1.0,
            cost_usd=20.0,
            nav_after_usd=1010.0,
            ts=1700000100,
        )
        # velocity follow (open)
        repo.insert_entry(
            triggering_alert_key="vel:0xb:1",
            triggering_alert_detector="velocity",
            rule_variant="follow",
            source_wallet=None,
            condition_id=ConditionId("0xc2"),
            asset_id=AssetId("b-y"),
            outcome="yes",
            shares=10.0,
            fill_price=0.25,
            cost_usd=2.5,
            nav_after_usd=1010.0,
            ts=1700000200,
        )
        # mispricing entry, still open
        repo.insert_entry(
            triggering_alert_key="mis:ev1:1",
            triggering_alert_detector="mispricing",
            rule_variant=None,
            source_wallet=None,
            condition_id=ConditionId("0xc3"),
            asset_id=AssetId("c-no"),
            outcome="NO",
            shares=10.0,
            fill_price=0.5,
            cost_usd=5.0,
            nav_after_usd=1010.0,
            ts=1700000300,
        )
    finally:
        conn.close()

    rc = main(["--config", str(cfg), "paper", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Per-source breakdown" in out
    assert "smart_money" in out
    assert "velocity" in out
    assert "follow" in out
    assert "mispricing" in out
