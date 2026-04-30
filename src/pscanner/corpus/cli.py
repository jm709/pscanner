"""argparse handlers for ``pscanner corpus {backfill, refresh, build-features}``.

Each handler opens ``corpus.sqlite3``, instantiates the gamma + data
clients with their own rate budget, runs the orchestration, and exits
with 0 on success.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from contextlib import AsyncExitStack
from pathlib import Path

import structlog

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.enumerator import enumerate_closed_markets
from pscanner.corpus.examples import build_features
from pscanner.corpus.market_walker import walk_market
from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusStateRepo,
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)
from pscanner.corpus.resolutions import record_resolutions
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)


def _add_db_arg(p: argparse.ArgumentParser) -> None:
    """Add the shared ``--db`` flag to a subparser."""
    p.add_argument(
        "--db",
        default="data/corpus.sqlite3",
        type=str,
        help="path to corpus SQLite database (default: data/corpus.sqlite3)",
    )


def build_corpus_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the corpus subcommand group."""
    parser = argparse.ArgumentParser(prog="pscanner corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    backfill = sub.add_parser(
        "backfill", help="Bulk historical pull of every closed qualifying market"
    )
    _add_db_arg(backfill)
    refresh = sub.add_parser("refresh", help="Incremental pass for newly-resolved markets")
    _add_db_arg(refresh)
    bf = sub.add_parser("build-features", help="Rebuild training_examples from raw events")
    _add_db_arg(bf)
    bf.add_argument("--rebuild", action="store_true", help="Drop and recreate the table")
    return parser


class _GammaCM:
    """Async context manager that owns a fresh GammaClient + closes it."""

    async def __aenter__(self) -> GammaClient:
        self._client = GammaClient(rpm=50)
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()


class _DataCM:
    """Async context manager that owns a fresh DataClient + closes it."""

    async def __aenter__(self) -> DataClient:
        self._client = DataClient(rpm=50)
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()


def _make_gamma_client() -> _GammaCM:
    return _GammaCM()


def _make_data_client() -> _DataCM:
    return _DataCM()


async def _drain_pending(*, conn: sqlite3.Connection, data: DataClient) -> int:
    """Drain `corpus_markets` work queue, walking each pending market once."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    total = 0
    while True:
        batch = markets_repo.next_pending(limit=10)
        if not batch:
            return total
        for m in batch:
            try:
                inserted = await walk_market(
                    condition_id=m.condition_id,
                    data=data,
                    markets_repo=markets_repo,
                    trades_repo=trades_repo,
                    now_ts=int(time.time()),
                )
                total += inserted
            except Exception as exc:  # walk_market already records failed state
                _log.warning("corpus.walk_failed", condition_id=m.condition_id, error=str(exc))


async def _cmd_backfill(args: argparse.Namespace) -> int:
    """Bulk-pull every closed qualifying market into the corpus."""
    conn = init_corpus_db(Path(args.db))
    try:
        async with AsyncExitStack() as stack:
            gamma = await stack.enter_async_context(_make_gamma_client())
            data = await stack.enter_async_context(_make_data_client())
            await enumerate_closed_markets(
                gamma=gamma,
                repo=CorpusMarketsRepo(conn),
                now_ts=int(time.time()),
                since_ts=None,
            )
            await _drain_pending(conn=conn, data=data)
        return 0
    finally:
        conn.close()


async def _cmd_refresh(args: argparse.Namespace) -> int:
    """Incremental sweep for newly-closed markets and missing resolutions."""
    conn = init_corpus_db(Path(args.db))
    try:
        state = CorpusStateRepo(conn)
        async with AsyncExitStack() as stack:
            gamma = await stack.enter_async_context(_make_gamma_client())
            data = await stack.enter_async_context(_make_data_client())
            since_ts = state.get_int("last_gamma_sweep_ts")
            await enumerate_closed_markets(
                gamma=gamma,
                repo=CorpusMarketsRepo(conn),
                now_ts=int(time.time()),
                since_ts=since_ts,
            )
            await _drain_pending(conn=conn, data=data)
            res_repo = MarketResolutionsRepo(conn)
            rows = conn.execute(
                """
                SELECT m.condition_id, m.market_slug, m.closed_at
                FROM corpus_markets m
                LEFT JOIN market_resolutions r USING (condition_id)
                WHERE r.condition_id IS NULL AND m.backfill_state = 'complete'
                  AND m.market_slug IS NOT NULL AND m.market_slug != ''
                """
            ).fetchall()
            await record_resolutions(
                gamma=gamma,
                repo=res_repo,
                targets=[(r["condition_id"], r["market_slug"], r["closed_at"]) for r in rows],
                now_ts=int(time.time()),
            )
            state.set("last_gamma_sweep_ts", str(int(time.time())), updated_at=int(time.time()))
        return 0
    finally:
        conn.close()


async def _cmd_build_features(args: argparse.Namespace) -> int:
    """Rebuild the training_examples table from raw corpus_trades + resolutions."""
    conn = init_corpus_db(Path(args.db))
    try:
        written = build_features(
            trades_repo=CorpusTradesRepo(conn),
            resolutions_repo=MarketResolutionsRepo(conn),
            examples_repo=TrainingExamplesRepo(conn),
            markets_conn=conn,
            now_ts=int(time.time()),
            rebuild=bool(getattr(args, "rebuild", False)),
        )
        _log.info("corpus.build_features_done", written=written)
        return 0
    finally:
        conn.close()


_HANDLERS = {
    "backfill": _cmd_backfill,
    "refresh": _cmd_refresh,
    "build-features": _cmd_build_features,
}


async def run_corpus_command(argv: list[str]) -> int:
    """Parse ``argv`` (excluding the leading ``corpus``) and dispatch."""
    parser = build_corpus_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    return await handler(args)
