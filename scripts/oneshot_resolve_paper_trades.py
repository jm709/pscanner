"""One-shot recovery: refresh stuck market_cache rows and book exit rows.

Run on the daemon host to retroactively book exit rows for paper_trades
positions whose markets resolved before PR #182's PaperResolver refresh
fix landed. Mirrors what a single PaperResolver scan cycle does, but
runnable standalone so the daemon code doesn't need to be merged first.

Steps performed:

1. List open paper_trades positions.
2. For each unique condition_id whose market_cache row is missing or
   still ``active=True``, call ``refresh_market_cache_row`` (2-hop
   data->slug->gamma lookup) to refresh the cache.
3. For each open position, run the same ``_check_resolution`` +
   ``_compute_payout`` logic the PaperResolver uses, and insert an exit
   row when the underlying market resolved.

Usage::

    uv run python scripts/oneshot_resolve_paper_trades.py [--dry-run]
        [--db data/pscanner.sqlite3] [--starting-bankroll 1000.0]

``--dry-run`` skips the exit-row INSERT but otherwise runs the full
refresh + check pass so you can preview how many exits would land.
"""
# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import time
from pathlib import Path

from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import MarketCacheRepo, PaperTradesRepo
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row
from pscanner.strategies.paper_resolver import _check_resolution, _compute_payout

_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
_DATA_BASE_URL = "https://data-api.polymarket.com"
_PROGRESS_EVERY = 25


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=Path,
        default=Path("data/pscanner.sqlite3"),
        help="Path to the daemon SQLite DB (default: data/pscanner.sqlite3).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full refresh + check pass but skip the exit-row INSERT.",
    )
    p.add_argument(
        "--starting-bankroll",
        type=float,
        default=1000.0,
        help="Bankroll constant for nav_after_usd stamping (default: 1000.0).",
    )
    p.add_argument(
        "--gamma-rpm",
        type=int,
        default=50,
        help="Gamma API rate limit, requests/min (default: 50).",
    )
    p.add_argument(
        "--data-rpm",
        type=int,
        default=50,
        help="Data API rate limit, requests/min (default: 50).",
    )
    return p.parse_args()


async def _refresh_phase(
    *,
    cache: MarketCacheRepo,
    data: DataClient,
    gamma: GammaClient,
    cond_ids: list[ConditionId],
) -> tuple[int, int, int]:
    """Refresh stale-active or missing market_cache rows.

    Returns ``(refreshed, failed, skipped)``.
    """
    refreshed = 0
    failed = 0
    skipped = 0
    total = len(cond_ids)
    for i, cid in enumerate(cond_ids, start=1):
        cached = cache.get_by_condition_id(cid)
        if cached is not None and not cached.active:
            skipped += 1
            continue
        ok = await refresh_market_cache_row(
            data_client=data,
            gamma_client=gamma,
            market_cache=cache,
            condition_id=cid,
        )
        if ok:
            refreshed += 1
        else:
            failed += 1
        if (refreshed + failed) % _PROGRESS_EVERY == 0:
            print(
                f"  refresh progress: {i}/{total} "
                f"({refreshed} ok / {failed} fail / {skipped} skip)",
                flush=True,
            )
    return refreshed, failed, skipped


def _book_phase(
    *,
    cache: MarketCacheRepo,
    paper: PaperTradesRepo,
    starting_bankroll: float,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Book exit rows for any position whose market resolved.

    Returns ``(booked_won, booked_lost, unresolved)``.
    """
    booked_won = 0
    booked_lost = 0
    unresolved = 0
    now = int(time.time())
    open_positions = list(paper.list_open_positions())
    total = len(open_positions)
    for i, pos in enumerate(open_positions, start=1):
        winning = _check_resolution(cache, pos.condition_id)
        if winning is None:
            unresolved += 1
            continue
        payout = _compute_payout(
            position_asset_id=pos.asset_id,
            winning_asset_id=winning,
        )
        proceeds = pos.shares * payout
        nav_before = paper.compute_cost_basis_nav(starting_bankroll=starting_bankroll)
        if not dry_run:
            paper.insert_exit(
                parent_trade_id=pos.trade_id,
                condition_id=pos.condition_id,
                asset_id=pos.asset_id,
                outcome=pos.outcome,
                shares=pos.shares,
                fill_price=payout,
                cost_usd=proceeds,
                nav_after_usd=nav_before + (proceeds - pos.cost_usd),
                ts=now,
            )
        if payout == 1.0:
            booked_won += 1
        else:
            booked_lost += 1
        if (booked_won + booked_lost) % (_PROGRESS_EVERY * 4) == 0:
            print(
                f"  book progress: {i}/{total} "
                f"({booked_won} won / {booked_lost} lost / {unresolved} open)",
                flush=True,
            )
    return booked_won, booked_lost, unresolved


async def _amain() -> int:
    args = _parse_args()

    if not args.db.exists():
        print(f"error: DB not found at {args.db}", file=sys.stderr)
        return 2

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row
    cache = MarketCacheRepo(db)
    paper = PaperTradesRepo(db)

    gamma_http = PolyHttpClient(base_url=_GAMMA_BASE_URL, rpm=args.gamma_rpm)
    data_http = PolyHttpClient(base_url=_DATA_BASE_URL, rpm=args.data_rpm)
    gamma = GammaClient(http=gamma_http)
    data = DataClient(http=data_http)

    try:
        open_positions = list(paper.list_open_positions())
        cond_ids = sorted({p.condition_id for p in open_positions})
        print(
            f"Open positions: {len(open_positions)} (distinct condition_ids: {len(cond_ids)})",
            flush=True,
        )
        if args.dry_run:
            print("DRY RUN — no exit rows will be written.", flush=True)

        print("\n=== refresh phase ===", flush=True)
        refreshed, failed, skipped = await _refresh_phase(
            cache=cache,
            data=data,
            gamma=gamma,
            cond_ids=cond_ids,
        )
        print(
            f"refresh complete: {refreshed} ok / {failed} fail / {skipped} skip",
            flush=True,
        )

        print("\n=== book phase ===", flush=True)
        booked_won, booked_lost, unresolved = _book_phase(
            cache=cache,
            paper=paper,
            starting_bankroll=args.starting_bankroll,
            dry_run=args.dry_run,
        )
        print(
            f"book complete: {booked_won} won / {booked_lost} lost / {unresolved} still open",
            flush=True,
        )

        nav = paper.compute_cost_basis_nav(starting_bankroll=args.starting_bankroll)
        print(f"\nfinal NAV: {nav:.2f}", flush=True)
    finally:
        await gamma_http.aclose()
        await data_http.aclose()
        db.close()

    return 0


def main() -> None:
    """Entry point — run the async pipeline and propagate exit code."""
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
