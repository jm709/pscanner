"""Command-line entrypoint for ``pscanner``.

Sub-commands:

* ``pscanner run`` — start the long-running daemon.
* ``pscanner run --once`` — run a single-shot scan and exit.
* ``pscanner status`` — print the most-recent alerts from SQLite.

The CLI returns an integer exit code so it composes cleanly with shell
pipelines and ``uv run pscanner ...``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Final

import structlog
from rich.console import Console
from rich.table import Table

from pscanner.alerts.models import Alert
from pscanner.config import Config
from pscanner.scheduler import Scanner
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo

_PROG = "pscanner"
_STATUS_LIMIT: Final[int] = 50


def main(argv: list[str] | None = None) -> int:
    """Run the pscanner CLI.

    Args:
        argv: Optional explicit argument list (excludes ``argv[0]``).

    Returns:
        Process exit code: ``0`` on success, ``2`` on argument errors,
        ``1`` on runtime errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = _resolve_config_path(args)
    if not _config_path_is_acceptable(config_path):
        sys.stderr.write(f"{_PROG}: config file not found: {config_path}\n")
        return 2
    try:
        config = Config.load(config_path)
    except ValueError as exc:
        sys.stderr.write(f"{_PROG}: invalid config: {exc}\n")
        return 2
    _configure_logging(config.scanner.log_level)
    if args.command == "run":
        return _cmd_run(config, once=bool(args.once))
    if args.command == "status":
        return _cmd_status(config)
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable; argparse exits


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with the run/status subcommands."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Polymarket scanner daemon (smart-money, mispricing, whales).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: $PSCANNER_CONFIG or ./config.toml).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scanner daemon")
    run.add_argument(
        "--once",
        action="store_true",
        help="execute a single scan pass and exit",
    )

    sub.add_parser("status", help="print recent alerts from the SQLite log")
    return parser


def _resolve_config_path(args: argparse.Namespace) -> Path | None:
    """Resolve the config-path argument honouring ``--config`` only here.

    ``Config.load`` separately checks ``PSCANNER_CONFIG``; when ``--config``
    is omitted we let ``Config.load(None)`` apply the documented precedence.
    """
    return args.config


def _config_path_is_acceptable(path: Path | None) -> bool:
    """Validate explicit ``--config`` path: only fail if user named a missing file."""
    if path is None:
        return True
    return path.exists()


def _configure_logging(level: str) -> None:
    """Configure stdlib + structlog so detector logs land on stderr.

    ``cache_logger_on_first_use`` is left at the default ``False`` so
    structlog re-resolves ``sys.stderr`` on each call (pytest's ``capsys``
    swaps the stream between tests).
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        stream=sys.stderr,
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def _cmd_run(config: Config, *, once: bool) -> int:
    """Dispatch to the daemon or the single-shot snapshot path."""
    if once:
        return _run_once(config)
    return _run_daemon(config)


def _run_once(config: Config) -> int:
    """Execute a single pass and print the resulting counters as a table."""

    async def _go() -> dict[str, object]:
        scanner = Scanner(config=config)
        try:
            return await scanner.run_once()
        finally:
            await scanner.aclose()

    try:
        result = asyncio.run(_go())
    except Exception as exc:
        sys.stderr.write(f"{_PROG}: run --once failed: {exc}\n")
        return 1
    _print_run_once_table(result)
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def _run_daemon(config: Config) -> int:
    """Run the long-lived daemon until SIGINT/KeyboardInterrupt."""
    scanner = Scanner(config=config)

    async def _go() -> None:
        await scanner.run()

    try:
        asyncio.run(_go())
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        sys.stderr.write(f"{_PROG}: scanner failed: {exc}\n")
        return 1
    return 0


def _print_run_once_table(result: dict[str, object]) -> None:
    """Render the single-shot result dict as a small ``rich`` table."""
    console = Console()
    table = Table(title="pscanner run --once", show_lines=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)


def _cmd_status(config: Config) -> int:
    """Open the DB read-only and print the most-recent alerts grouped by detector."""
    db_path = config.scanner.db_path
    if not Path(db_path).exists():
        sys.stderr.write(f"{_PROG}: no database at {db_path}; run the daemon first\n")
        return 1
    conn = init_db(Path(db_path))
    try:
        repo = AlertsRepo(conn)
        alerts = repo.recent(limit=_STATUS_LIMIT)
    finally:
        conn.close()
    _print_status_table(alerts)
    return 0


def _print_status_table(alerts: list[Alert]) -> None:
    """Render a recent-alerts table grouped by detector."""
    console = Console()
    if not alerts:
        console.print("[dim]no alerts logged yet[/dim]")
        return
    table = Table(title=f"recent alerts (last {len(alerts)})", show_lines=False)
    table.add_column("time")
    table.add_column("detector")
    table.add_column("severity")
    table.add_column("title", overflow="fold")
    grouped = sorted(alerts, key=lambda a: (a.detector, -a.created_at))
    for alert in grouped:
        when = datetime.datetime.fromtimestamp(
            alert.created_at,
            tz=datetime.UTC,
        ).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(when, alert.detector, alert.severity, alert.title)
    console.print(table)
