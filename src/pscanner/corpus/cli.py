"""argparse handlers for ``pscanner corpus`` subcommands.

Covers ``backfill``, ``refresh``, ``build-features``, ``onchain-backfill``,
``onchain-backfill-targeted``, and ``subgraph-backfill``.
Each handler opens ``corpus.sqlite3``, instantiates the required clients with
their own rate budget, runs the orchestration, and exits with 0 on success.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from contextlib import AsyncExitStack
from pathlib import Path

import structlog

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.enumerator import enumerate_closed_markets
from pscanner.corpus.examples import build_features
from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets
from pscanner.corpus.manifold_walker import walk_manifold_market
from pscanner.corpus.market_walker import walk_market
from pscanner.corpus.onchain_backfill import (
    clear_truncation_flags,
    run_onchain_backfill,
)
from pscanner.corpus.onchain_targeted import run_targeted_backfill
from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusStateRepo,
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)
from pscanner.corpus.resolutions import record_manifold_resolutions, record_resolutions
from pscanner.corpus.subgraph_ingest import run_subgraph_backfill
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.ids import ManifoldMarketId
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.onchain_rpc import OnchainRpcClient
from pscanner.poly.subgraph import SubgraphClient

_log = structlog.get_logger(__name__)

_DEFAULT_RPC_URL = "https://polygon.gateway.tenderly.co"
_DEFAULT_FROM_BLOCK = 33_605_403  # CTF Exchange deployment, Polygon block, 2022-09-26
_DEFAULT_CHUNK_SIZE = 5_000
_DEFAULT_MAX_BLOCKS = 1_000_000
_DEFAULT_TARGETED_CHUNK_SIZE = 500
_DEFAULT_BLOCK_SLACK = 5_000
_DEFAULT_SUBGRAPH_RPM = 600
_DEFAULT_SUBGRAPH_PAGE_SIZE = 1000
# Polymarket Orderbook subgraph on The Graph's hosted gateway. Verified via
# Graph Explorer (https://thegraph.com/explorer) — the subgraph's title is
# "Polymarket Orderbook" and both Exchange + NegRiskExchange contracts write
# into the same OrderFilledEvent entity.
_DEFAULT_SUBGRAPH_ID = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
_GATEWAY_URL_TEMPLATE = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"


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
    backfill.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "manifold"],
        default="polymarket",
        help=(
            "Platform to ingest. Defaults to polymarket. "
            "`manifold` runs the Manifold REST enumerator + bet walker."
        ),
    )
    refresh = sub.add_parser("refresh", help="Incremental pass for newly-resolved markets")
    _add_db_arg(refresh)
    refresh.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "manifold"],
        default="polymarket",
        help=(
            "Platform to ingest. Defaults to polymarket. "
            "`manifold` re-enumerates resolved markets and records resolutions."
        ),
    )
    bf = sub.add_parser("build-features", help="Rebuild training_examples from raw events")
    _add_db_arg(bf)
    bf.add_argument("--rebuild", action="store_true", help="Drop and recreate the table")
    ob = sub.add_parser(
        "onchain-backfill",
        help="Walk CTF Exchange OrderFilled events and write to corpus_trades",
    )
    _add_db_arg(ob)
    ob.add_argument(
        "--from-block",
        type=int,
        default=None,
        help=(
            "First block (inclusive). Default: corpus_state['onchain_last_block'] + 1, "
            f"or {_DEFAULT_FROM_BLOCK} on first run."
        ),
    )
    ob.add_argument(
        "--to-block",
        type=int,
        default=None,
        help="Last block (inclusive). Default: current Polygon head.",
    )
    ob.add_argument(
        "--rpc-url",
        type=str,
        default=_DEFAULT_RPC_URL,
        help=f"Polygon RPC endpoint (default: {_DEFAULT_RPC_URL})",
    )
    ob.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help=f"Blocks per eth_getLogs call (default: {_DEFAULT_CHUNK_SIZE})",
    )
    ob.add_argument(
        "--max-blocks",
        type=int,
        default=_DEFAULT_MAX_BLOCKS,
        help=f"Safety cap per run (default: {_DEFAULT_MAX_BLOCKS})",
    )
    ob.add_argument(
        "--rpm",
        type=int,
        default=600,
        help="RPC requests per minute ceiling (default: 600)",
    )
    ot = sub.add_parser(
        "onchain-backfill-targeted",
        help=(
            "Per-market on-chain backfill of truncated markets (resumable). "
            "For each market with truncated_at_offset_cap=1 and no "
            "onchain_processed_at, walks its trade-time-window block range, "
            "filters OrderFilled events to that market's asset_ids, and inserts."
        ),
    )
    _add_db_arg(ot)
    ot.add_argument(
        "--rpc-url",
        type=str,
        default=_DEFAULT_RPC_URL,
        help=f"Polygon RPC endpoint (default: {_DEFAULT_RPC_URL})",
    )
    ot.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_TARGETED_CHUNK_SIZE,
        help=f"Blocks per eth_getLogs call (default: {_DEFAULT_TARGETED_CHUNK_SIZE})",
    )
    ot.add_argument(
        "--block-slack",
        type=int,
        default=_DEFAULT_BLOCK_SLACK,
        help=(
            "Extra blocks padded around each market's trade window (default: "
            f"{_DEFAULT_BLOCK_SLACK}). Absorbs interpolator drift."
        ),
    )
    ot.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N markets in this run (default: no limit).",
    )
    ot.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="RPC requests per minute ceiling (default: 60).",
    )
    sg = sub.add_parser(
        "subgraph-backfill",
        help=(
            "Per-market subgraph-driven backfill of truncated markets (resumable). "
            "Replaces eth_getLogs path with GraphQL queries against The Graph."
        ),
    )
    _add_db_arg(sg)
    sg.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Graph Studio API key. Falls back to $GRAPH_API_KEY.",
    )
    sg.add_argument(
        "--subgraph-id",
        type=str,
        default=_DEFAULT_SUBGRAPH_ID,
        help=(
            "Subgraph deployment id. The default is the verified Polymarket "
            "Orderbook subgraph on The Graph."
        ),
    )
    sg.add_argument(
        "--rpm",
        type=int,
        default=_DEFAULT_SUBGRAPH_RPM,
        help=f"Subgraph queries per minute (default: {_DEFAULT_SUBGRAPH_RPM}).",
    )
    sg.add_argument(
        "--page-size",
        type=int,
        default=_DEFAULT_SUBGRAPH_PAGE_SIZE,
        help=f"Rows per query, max 1000 (default: {_DEFAULT_SUBGRAPH_PAGE_SIZE}).",
    )
    sg.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N markets in this run (default: no limit).",
    )
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
    """Run the corpus backfill for the requested platform."""
    if args.platform == "manifold":
        return await _run_manifold_backfill(args)
    return await _run_polymarket_backfill(args)


async def _run_polymarket_backfill(args: argparse.Namespace) -> int:
    """Bulk-pull every closed qualifying Polymarket market into the corpus."""
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


async def _run_manifold_backfill(args: argparse.Namespace) -> int:
    """Manifold path: enumerate resolved binary markets, then walk each one."""
    conn = init_corpus_db(Path(args.db))
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    now_ts = int(time.time())
    try:
        async with ManifoldClient() as client:
            await enumerate_resolved_manifold_markets(client, markets_repo, now_ts=now_ts)
            while pending := markets_repo.next_pending(limit=10, platform="manifold"):
                for market in pending:
                    await walk_manifold_market(
                        client,
                        markets_repo,
                        trades_repo,
                        market_id=ManifoldMarketId(market.condition_id),
                        now_ts=now_ts,
                    )
        return 0
    finally:
        conn.close()


async def _cmd_refresh(args: argparse.Namespace) -> int:
    """Run the corpus refresh for the requested platform."""
    if args.platform == "manifold":
        return await _run_manifold_refresh(args)
    return await _run_polymarket_refresh(args)


async def _run_polymarket_refresh(args: argparse.Namespace) -> int:
    """Incremental sweep for newly-closed Polymarket markets and missing resolutions."""
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


async def _run_manifold_refresh(args: argparse.Namespace) -> int:
    """Manifold refresh: re-enumerate, then record resolutions for missing markets."""
    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    now_ts = int(time.time())
    try:
        async with ManifoldClient() as client:
            await enumerate_resolved_manifold_markets(client, markets_repo, now_ts=now_ts)
            rows = conn.execute(
                "SELECT condition_id, closed_at FROM corpus_markets "
                "WHERE platform = 'manifold' AND backfill_state = 'complete'"
            ).fetchall()
            condition_ids = [row["condition_id"] for row in rows]
            missing = resolutions_repo.missing_for(condition_ids, platform="manifold")
            missing_set = set(missing)
            targets = [
                (row["condition_id"], int(row["closed_at"]))
                for row in rows
                if row["condition_id"] in missing_set
            ]
            await record_manifold_resolutions(
                client=client,
                repo=resolutions_repo,
                targets=targets,
                now_ts=now_ts,
            )
    finally:
        conn.close()
    return 0


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


async def _cmd_onchain_backfill(args: argparse.Namespace) -> int:
    """Walk on-chain `OrderFilled` events into corpus_trades."""
    conn = init_corpus_db(Path(args.db))
    try:
        state = CorpusStateRepo(conn)
        cursor = state.get_int("onchain_last_block")
        from_block: int = (
            args.from_block
            if args.from_block is not None
            else (cursor + 1 if cursor is not None else _DEFAULT_FROM_BLOCK)
        )
        async with OnchainRpcClient(rpc_url=args.rpc_url, rpm=args.rpm) as rpc:
            to_block: int = (
                args.to_block if args.to_block is not None else await rpc.get_block_number()
            )
            if to_block < from_block:
                _log.info(
                    "onchain.nothing_to_do",
                    from_block=from_block,
                    to_block=to_block,
                )
                return 0
            capped_to = min(to_block, from_block + args.max_blocks - 1)
            if capped_to < to_block:
                _log.warning(
                    "onchain.capped_to_block",
                    requested_to=to_block,
                    capped_to=capped_to,
                    max_blocks=args.max_blocks,
                )
            summary = await run_onchain_backfill(
                conn=conn,
                rpc=rpc,
                from_block=from_block,
                to_block=capped_to,
                chunk_size=args.chunk_size,
            )
        cleared = clear_truncation_flags(conn=conn)
        _log.info(
            "onchain.run_summary",
            chunks=summary.chunks_processed,
            events=summary.events_decoded,
            inserted=summary.trades_inserted,
            skipped_unsupported=summary.skipped_unsupported,
            skipped_unresolvable=summary.skipped_unresolvable,
            last_block=summary.last_block,
            truncation_flags_cleared=cleared,
        )
        return 0
    finally:
        conn.close()


async def _cmd_onchain_backfill_targeted(args: argparse.Namespace) -> int:
    """Run the per-market targeted on-chain backfill (resumable)."""
    conn = init_corpus_db(Path(args.db))
    try:
        async with OnchainRpcClient(rpc_url=args.rpc_url, rpm=args.rpm) as rpc:
            summary = await run_targeted_backfill(
                conn=conn,
                rpc=rpc,
                chunk_size=args.chunk_size,
                block_slack=args.block_slack,
                limit=args.limit,
            )
        _log.info(
            "onchain_targeted.cli_summary",
            markets_processed=summary.markets_processed,
            markets_failed=summary.markets_failed,
            events_decoded=summary.events_decoded,
            trades_inserted=summary.trades_inserted,
            skipped_unsupported=summary.skipped_unsupported,
            skipped_unresolvable=summary.skipped_unresolvable,
        )
        return 0
    finally:
        conn.close()


async def _cmd_subgraph_backfill(args: argparse.Namespace) -> int:
    """Run the subgraph-driven per-market backfill."""
    api_key = args.api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        raise SystemExit("subgraph-backfill requires --api-key or $GRAPH_API_KEY")
    url = _GATEWAY_URL_TEMPLATE.format(api_key=api_key, subgraph_id=args.subgraph_id)
    conn = init_corpus_db(Path(args.db))
    try:
        async with SubgraphClient(url=url, rpm=args.rpm) as client:
            summary = await run_subgraph_backfill(
                conn=conn,
                client=client,
                page_size=args.page_size,
                limit=args.limit,
            )
        _log.info(
            "subgraph.cli_summary",
            markets_processed=summary.markets_processed,
            markets_failed=summary.markets_failed,
            events_decoded=summary.events_decoded,
            trades_inserted=summary.trades_inserted,
            skipped_unsupported=summary.skipped_unsupported,
            skipped_unresolvable=summary.skipped_unresolvable,
            truncation_flags_cleared=summary.truncation_flags_cleared,
        )
        return 0
    finally:
        conn.close()


_HANDLERS = {
    "backfill": _cmd_backfill,
    "refresh": _cmd_refresh,
    "build-features": _cmd_build_features,
    "onchain-backfill": _cmd_onchain_backfill,
    "onchain-backfill-targeted": _cmd_onchain_backfill_targeted,
    "subgraph-backfill": _cmd_subgraph_backfill,
}


async def run_corpus_command(argv: list[str]) -> int:
    """Parse ``argv`` (excluding the leading ``corpus``) and dispatch."""
    parser = build_corpus_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    return await handler(args)
