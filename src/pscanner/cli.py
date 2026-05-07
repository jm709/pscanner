"""Command-line entrypoint for ``pscanner``.

Sub-commands:

* ``pscanner run`` — start the long-running daemon.
* ``pscanner run --once`` — run a single-shot scan and exit.
* ``pscanner status`` — print the most-recent alerts from SQLite.
* ``pscanner watch <address>`` — add a wallet to the data-collection watchlist.
* ``pscanner unwatch <address>`` — deactivate a watchlist entry.
* ``pscanner watchlist`` — print every watchlist entry as a table.
* ``pscanner paper status`` — summarise the paper-trading bankroll and PnL.

The CLI returns an integer exit code so it composes cleanly with shell
pipelines and ``uv run pscanner ...``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import structlog
from rich.console import Console
from rich.table import Table

from pscanner.alerts.models import Alert
from pscanner.config import Config
from pscanner.corpus.cli import run_corpus_command
from pscanner.daemon.bootstrap import run_bootstrap
from pscanner.ml.cli import run_ml_command
from pscanner.scheduler import Scanner
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    PaperSummary,
    PaperTradesRepo,
    SourceSummary,
    WatchlistEntry,
    WatchlistRepo,
)

_PROG = "pscanner"
_STATUS_LIMIT: Final[int] = 50
_PAPER_TOP_N: Final[int] = 3
_PAPER_COND_PREFIX: Final[int] = 16
_PAPER_WALLET_PREFIX: Final[int] = 10


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
    if args.command == "corpus":
        return asyncio.run(run_corpus_command(args.corpus_argv))
    if args.command == "ml":
        return run_ml_command(args.ml_argv)
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
    return _dispatch_command(parser, args, config)


def _dispatch_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: Config,
) -> int:
    """Route ``args.command`` to the matching ``_cmd_*`` handler."""
    handlers: dict[str, Callable[[], int]] = {
        "run": lambda: _cmd_run(config, once=bool(args.once)),
        "status": lambda: _cmd_status(config),
        "watch": lambda: _cmd_watch(config, address=args.address, reason=args.reason),
        "unwatch": lambda: _cmd_unwatch(config, address=args.address),
        "watchlist": lambda: _cmd_watchlist(config),
        "paper": lambda: _dispatch_paper(parser, args, config),
        "daemon": lambda: _dispatch_daemon(parser, args, config),
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
        return 2  # unreachable; argparse exits
    return handler()


def _dispatch_paper(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: Config,
) -> int:
    """Route ``pscanner paper <subcmd>`` to the matching handler."""
    if args.paper_cmd == "status":
        return _cmd_paper_status(config)
    parser.error(f"unknown paper subcommand: {args.paper_cmd}")
    return 2  # unreachable; argparse exits


def _dispatch_daemon(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: Config,
) -> int:
    """Route ``pscanner daemon <subcmd>`` to the matching handler."""
    del config  # daemon ops use --corpus-db/--daemon-db flags directly
    if args.daemon_cmd == "bootstrap-features":
        return _cmd_daemon_bootstrap(args.corpus_db, args.daemon_db)
    parser.error(f"unknown daemon subcommand: {args.daemon_cmd}")
    return 2  # unreachable; argparse exits


def _cmd_daemon_bootstrap(corpus_db: Path, daemon_db: Path) -> int:
    """Cold-start the live history tables from corpus_trades."""
    n = run_bootstrap(corpus_db=corpus_db, daemon_db=daemon_db)
    print(f"bootstrap-features: folded {n} trades")  # noqa: T201
    return 0


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

    watch = sub.add_parser("watch", help="add a wallet to the watchlist")
    watch.add_argument("address", type=str, help="0x-prefixed proxy wallet address")
    watch.add_argument(
        "--reason",
        type=str,
        default=None,
        help="free-form note to record alongside the entry",
    )

    unwatch = sub.add_parser("unwatch", help="deactivate a watchlist entry")
    unwatch.add_argument("address", type=str, help="0x-prefixed proxy wallet address")

    sub.add_parser("watchlist", help="print every watchlist entry as a table")

    paper = sub.add_parser("paper", help="paper-trading commands")
    paper_sub = paper.add_subparsers(dest="paper_cmd", required=True)
    paper_sub.add_parser("status", help="summarise paper-trading bankroll and PnL")

    daemon = sub.add_parser("daemon", help="daemon-side ops")
    daemon_sub = daemon.add_subparsers(dest="daemon_cmd", required=True)
    bootstrap_parser = daemon_sub.add_parser(
        "bootstrap-features",
        help="cold-start wallet_state_live + market_state_live from corpus_trades",
    )
    bootstrap_parser.add_argument("--corpus-db", type=Path, default=Path("data/corpus.sqlite3"))
    bootstrap_parser.add_argument("--daemon-db", type=Path, default=Path("data/pscanner.sqlite3"))

    corpus = sub.add_parser(
        "corpus",
        help="historical trade corpus subcommands",
    )
    corpus.add_argument(
        "corpus_argv",
        nargs=argparse.REMAINDER,
        help="forwarded to `pscanner corpus --help`",
    )

    ml = sub.add_parser(
        "ml",
        help="machine-learning training pipeline subcommands",
    )
    ml.add_argument(
        "ml_argv",
        nargs=argparse.REMAINDER,
        help="forwarded to `pscanner ml --help`",
    )
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


def _cmd_watch(config: Config, *, address: str, reason: str | None) -> int:
    """Add a wallet to the watchlist via the manual source.

    Idempotent: re-running for the same address simply preserves the existing
    row (``WatchlistRepo.upsert`` keeps first-seen provenance).
    """
    db_path = Path(config.scanner.db_path)
    conn = init_db(db_path)
    try:
        repo = WatchlistRepo(conn)
        inserted = repo.upsert(address=address, source="manual", reason=reason)
    finally:
        conn.close()
    console = Console()
    if inserted:
        console.print(f"watching [bold]{address}[/bold]")
    else:
        console.print(f"[dim]{address} already in watchlist (no-op)[/dim]")
    return 0


def _cmd_unwatch(config: Config, *, address: str) -> int:
    """Deactivate a watchlist entry. No-op when the address is unknown."""
    db_path = Path(config.scanner.db_path)
    conn = init_db(db_path)
    try:
        repo = WatchlistRepo(conn)
        existing = repo.get(address)
        if existing is None:
            Console().print(f"[dim]{address} not in watchlist (no-op)[/dim]")
            return 0
        repo.set_active(address, False)
    finally:
        conn.close()
    Console().print(f"unwatched [bold]{address}[/bold]")
    return 0


def _cmd_watchlist(config: Config) -> int:
    """Print every watchlist entry (active + inactive) as a ``rich`` table."""
    db_path = Path(config.scanner.db_path)
    conn = init_db(db_path)
    try:
        repo = WatchlistRepo(conn)
        entries = repo.list_all()
    finally:
        conn.close()
    _print_watchlist_table(entries)
    return 0


def _print_watchlist_table(entries: list[WatchlistEntry]) -> None:
    """Render the watchlist as a ``rich`` table."""
    console = Console()
    if not entries:
        console.print("[dim]watchlist is empty[/dim]")
        return
    table = Table(title=f"watchlist ({len(entries)} entries)", show_lines=False)
    table.add_column("address", overflow="fold")
    table.add_column("source")
    table.add_column("reason", overflow="fold")
    table.add_column("added_at")
    table.add_column("active")
    for entry in entries:
        added = datetime.datetime.fromtimestamp(
            entry.added_at,
            tz=datetime.UTC,
        ).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(
            entry.address,
            entry.source,
            entry.reason or "",
            added,
            "yes" if entry.active else "no",
        )
    console.print(table)


def _cmd_paper_status(config: Config) -> int:
    """Print paper-trading status (NAV, open/closed counts, realized PnL, top trades)."""
    conn = init_db(Path(config.scanner.db_path))
    try:
        paper = PaperTradesRepo(conn)
        summary = paper.summary_stats(
            starting_bankroll=config.paper_trading.starting_bankroll_usd,
        )
        leaderboard = _paper_leaderboard_rows(conn)
        best = _paper_extreme_rows(conn, order="DESC")
        worst = _paper_extreme_rows(conn, order="ASC")
        sources = paper.summary_by_source()
    finally:
        conn.close()
    console = Console(highlight=False)
    _print_paper_summary(console, summary)
    _print_paper_leaderboard(console, leaderboard)
    _print_paper_extremes(console, "top 3 best settled trades", best)
    _print_paper_extremes(console, "top 3 worst settled trades", worst)
    _print_paper_sources(console, sources)
    return 0


def _paper_leaderboard_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Per-wallet leaderboard: realized PnL, settled count, and open count.

    Includes every wallet that has at least one entry — wallets with only
    open positions still appear (with ``settled = 0`` and ``realized = 0``).
    """
    return conn.execute(
        """
        SELECT e.source_wallet AS wallet,
               COALESCE(SUM(x.cost_usd - e.cost_usd), 0.0) AS realized,
               SUM(CASE WHEN x.trade_id IS NOT NULL THEN 1 ELSE 0 END) AS settled,
               SUM(CASE WHEN x.trade_id IS NULL     THEN 1 ELSE 0 END) AS open_n
          FROM paper_trades e
          LEFT JOIN paper_trades x
                 ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
         WHERE e.trade_kind = 'entry'
         GROUP BY e.source_wallet
         ORDER BY realized DESC, e.source_wallet ASC
        """,
    ).fetchall()


def _paper_extreme_rows(
    conn: sqlite3.Connection,
    *,
    order: str,
) -> list[sqlite3.Row]:
    """Top-N best (``DESC``) or worst (``ASC``) settled trades by realized PnL."""
    if order not in {"ASC", "DESC"}:
        msg = f"order must be ASC or DESC, got {order!r}"
        raise ValueError(msg)
    return conn.execute(
        f"""
        SELECT e.condition_id AS condition_id,
               e.outcome AS outcome,
               e.source_wallet AS wallet,
               (x.cost_usd - e.cost_usd) AS pnl
          FROM paper_trades x
          JOIN paper_trades e ON e.trade_id = x.parent_trade_id
         WHERE x.trade_kind = 'exit' AND e.trade_kind = 'entry'
         ORDER BY pnl {order}
         LIMIT ?
        """,  # noqa: S608 — `order` is whitelist-validated above
        (_PAPER_TOP_N,),
    ).fetchall()


def _print_paper_summary(console: Console, summary: PaperSummary) -> None:
    """Render the bankroll / NAV / realized PnL / counts header block."""
    console.print(f"starting bankroll: ${summary.starting_bankroll:,.2f}")
    console.print(f"current NAV:       ${summary.current_nav:,.2f}")
    console.print(
        f"realized PnL:      ${summary.realized_pnl:+,.2f} ({summary.total_return_pct:+.2f}%)",
    )
    console.print(
        f"open positions: {summary.open_positions}    closed positions: {summary.closed_positions}",
    )


def _print_paper_leaderboard(console: Console, rows: list[sqlite3.Row]) -> None:
    """Render the per-wallet leaderboard, skipping when empty."""
    if not rows:
        return
    console.print("")
    console.print("per-wallet PnL (realized + open counts):")
    for row in rows:
        wallet = str(row["wallet"] or "")
        realized = float(row["realized"] or 0.0)
        settled = int(row["settled"] or 0)
        open_n = int(row["open_n"] or 0)
        console.print(
            f"  {wallet:<46}  PnL=${realized:+,.2f}  settled={settled}  open={open_n}",
        )


def _print_paper_extremes(
    console: Console,
    title: str,
    rows: list[sqlite3.Row],
) -> None:
    """Render a best/worst settled-trades section, skipping when empty."""
    if not rows:
        return
    console.print("")
    console.print(f"{title}:")
    for row in rows:
        cond = str(row["condition_id"] or "")[:_PAPER_COND_PREFIX]
        wallet = str(row["wallet"] or "")[:_PAPER_WALLET_PREFIX]
        outcome = str(row["outcome"] or "")
        pnl = float(row["pnl"] or 0.0)
        console.print(
            f"  PnL=${pnl:+,.2f}  cond={cond}…  outcome={outcome}  wallet={wallet}…",
        )


def _print_paper_sources(console: Console, sources: list[SourceSummary]) -> None:
    """Render the per-(detector, rule_variant) breakdown, skipping when empty."""
    if not sources:
        return
    console.print("")
    console.print("Per-source breakdown:")
    header = (
        f"  {'detector':<20s} {'variant':<8s} "
        f"{'open':>5s} {'resolved':>9s} {'pnl':>9s} {'win_rate':>9s}"
    )
    console.print(header)
    for s in sources:
        det = s.detector or "(unknown)"
        variant = s.rule_variant or "-"
        console.print(
            f"  {det:<20s} {variant:<8s} "
            f"{s.open_count:>5d} {s.resolved_count:>9d} "
            f"{s.realized_pnl:>+9.2f} {s.win_rate * 100:>8.1f}%",
        )
