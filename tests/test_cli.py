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
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo

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
