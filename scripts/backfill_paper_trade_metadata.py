"""Backfill market_cache + event_tag_cache for paper_trades condition_ids.

Closes the analysis-side blind spot from #170 follow-up:
``MarketCollector`` and ``EventCollector`` both sweep
``active=True, closed=False`` from gamma, so markets/events that closed
before the daemon's first sweep never land in our local caches. As a
result, the copy-trading analyzer reports the bulk of subgraph_copy
positions as ``(uncategorized)``.

This script bounds the gap fix to "markets we actually paper-traded":

1. For every distinct ``condition_id`` in ``paper_trades`` that has no
   ``market_cache`` row (or has one with no ``event_slug``), call the
   2-hop ``refresh_market_cache_row`` helper.
2. For every distinct ``event_slug`` referenced by ``market_cache``
   that has no ``event_tag_cache`` row, fetch the event via
   ``gamma.get_event_by_slug`` and upsert tags.

Cap on gamma + data RPM is the standard 50/min for both. The gap is
finite (bounded by open + resolved paper-trade count) so the run is a
one-shot, not a periodic loop. Use the companion
``PaperTradesMetadataCollector`` (added in the same PR) for ongoing
prevention.

Usage::

    uv run python scripts/backfill_paper_trade_metadata.py [--dry-run]
        [--db data/pscanner.sqlite3] [--gamma-rpm 50] [--data-rpm 50]
"""
# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.ids import AssetId, ConditionId, EventSlug
from pscanner.store.repo import EventTagCacheRepo, MarketCacheRepo
from pscanner.strategies.market_cache_refresh import (
    refresh_market_cache_by_asset_id,
    refresh_market_cache_row,
)

_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
_DATA_BASE_URL = "https://data-api.polymarket.com"
_PROGRESS_EVERY = 25


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path("data/pscanner.sqlite3"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report gap sizes without making network calls.",
    )
    p.add_argument("--gamma-rpm", type=int, default=50)
    p.add_argument("--data-rpm", type=int, default=50)
    return p.parse_args()


def _condition_ids_missing_market_cache(db: sqlite3.Connection) -> list[ConditionId]:
    """Return distinct paper_trades condition_ids with no market_cache row."""
    rows = db.execute(
        """
        SELECT DISTINCT e.condition_id
          FROM paper_trades e
          LEFT JOIN market_cache mc ON mc.condition_id = e.condition_id
         WHERE e.trade_kind = 'entry'
           AND mc.condition_id IS NULL
        """,
    ).fetchall()
    return [ConditionId(r[0]) for r in rows]


def _asset_ids_needing_event_slug(db: sqlite3.Connection) -> list[AssetId]:
    """One asset_id per paper-trade condition_id whose market_cache has event_slug=NULL.

    Used by the token-id refresh path. Gamma's slug response sometimes
    omits event nesting for recurring/date-rolled markets (Iran-themed,
    MicroStrategy, etc.); the token-id response reliably includes it.
    """
    rows = db.execute(
        """
        SELECT MIN(e.asset_id) AS asset_id
          FROM paper_trades e
          JOIN market_cache mc ON mc.condition_id = e.condition_id
         WHERE e.trade_kind = 'entry'
           AND mc.event_slug IS NULL
         GROUP BY e.condition_id
        """,
    ).fetchall()
    return [AssetId(r["asset_id"]) for r in rows if r["asset_id"]]


def _event_slugs_missing_tag_cache(db: sqlite3.Connection) -> list[EventSlug]:
    """Return distinct market_cache event_slugs with no event_tag_cache row."""
    rows = db.execute(
        """
        SELECT DISTINCT mc.event_slug
          FROM market_cache mc
          JOIN paper_trades e ON e.condition_id = mc.condition_id
          LEFT JOIN event_tag_cache etc ON etc.event_slug = mc.event_slug
         WHERE e.trade_kind = 'entry'
           AND mc.event_slug IS NOT NULL
           AND etc.event_slug IS NULL
        """,
    ).fetchall()
    return [EventSlug(r[0]) for r in rows]


async def _backfill_market_cache(
    *,
    market_cache: MarketCacheRepo,
    data: DataClient,
    gamma: GammaClient,
    cond_ids: list[ConditionId],
) -> tuple[int, int]:
    """Refresh market_cache for each condition_id. Returns ``(ok, fail)``."""
    ok = 0
    fail = 0
    total = len(cond_ids)
    for i, cid in enumerate(cond_ids, start=1):
        result = await refresh_market_cache_row(
            data_client=data,
            gamma_client=gamma,
            market_cache=market_cache,
            condition_id=cid,
        )
        if result:
            ok += 1
        else:
            fail += 1
        if (ok + fail) % _PROGRESS_EVERY == 0:
            print(f"  market_cache progress: {i}/{total} ({ok} ok / {fail} fail)", flush=True)
    return ok, fail


async def _backfill_market_cache_by_asset_id(
    *,
    market_cache: MarketCacheRepo,
    gamma: GammaClient,
    asset_ids: list[AssetId],
) -> tuple[int, int]:
    """Re-fetch via token-id path for markets where slug path didn't populate event_slug.

    Returns ``(ok, fail)``.
    """
    ok = 0
    fail = 0
    total = len(asset_ids)
    for i, asset_id in enumerate(asset_ids, start=1):
        result = await refresh_market_cache_by_asset_id(
            gamma_client=gamma,
            market_cache=market_cache,
            asset_id=asset_id,
        )
        if result:
            ok += 1
        else:
            fail += 1
        if (ok + fail) % _PROGRESS_EVERY == 0:
            print(
                f"  market_cache (asset-id) progress: {i}/{total} ({ok} ok / {fail} fail)",
                flush=True,
            )
    return ok, fail


async def _backfill_event_tag_cache(
    *,
    event_tag_cache: EventTagCacheRepo,
    gamma: GammaClient,
    slugs: list[EventSlug],
) -> tuple[int, int]:
    """Fetch each event via gamma and upsert its tags. Returns ``(ok, fail)``."""
    ok = 0
    fail = 0
    total = len(slugs)
    for i, slug in enumerate(slugs, start=1):
        try:
            event = await gamma.get_event_by_slug(slug)
        except Exception as exc:
            fail += 1
            print(f"  event_tag_cache fetch failed: slug={slug} err={exc}", flush=True)
        else:
            if event is None:
                fail += 1
            else:
                event_tag_cache.upsert(slug, list(event.tags))
                ok += 1
        if (ok + fail) % _PROGRESS_EVERY == 0:
            print(f"  event_tag_cache progress: {i}/{total} ({ok} ok / {fail} fail)", flush=True)
    return ok, fail


async def _amain() -> int:
    args = _parse_args()
    if not args.db.exists():
        print(f"error: DB not found at {args.db}", file=sys.stderr)
        return 2

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row
    market_cache = MarketCacheRepo(db)
    event_tag_cache = EventTagCacheRepo(db)

    missing_cids = _condition_ids_missing_market_cache(db)
    print(f"distinct paper_trades condition_ids missing market_cache: {len(missing_cids)}")

    gamma_http = PolyHttpClient(base_url=_GAMMA_BASE_URL, rpm=args.gamma_rpm)
    data_http = PolyHttpClient(base_url=_DATA_BASE_URL, rpm=args.data_rpm)
    gamma = GammaClient(http=gamma_http)
    data = DataClient(http=data_http)

    try:
        if args.dry_run:
            print("DRY RUN — no network calls; quitting after gap report.", flush=True)
            slugs_for_estimate = _event_slugs_missing_tag_cache(db)
            print(
                f"distinct event_slugs missing event_tag_cache (current state): "
                f"{len(slugs_for_estimate)}",
            )
            return 0

        print("\n=== phase 1a: backfill market_cache via slug ===", flush=True)
        mc_ok, mc_fail = await _backfill_market_cache(
            market_cache=market_cache,
            data=data,
            gamma=gamma,
            cond_ids=missing_cids,
        )
        print(f"phase 1a complete: {mc_ok} ok / {mc_fail} fail", flush=True)

        # Phase 1b: token-id fallback for rows that have a market_cache row but
        # no event_slug (gamma's slug response sometimes omits event nesting).
        missing_asset_ids = _asset_ids_needing_event_slug(db)
        print(
            f"\n=== phase 1b: backfill via asset_id (n={len(missing_asset_ids)}) ===",
            flush=True,
        )
        mc2_ok, mc2_fail = await _backfill_market_cache_by_asset_id(
            market_cache=market_cache,
            gamma=gamma,
            asset_ids=missing_asset_ids,
        )
        print(f"phase 1b complete: {mc2_ok} ok / {mc2_fail} fail", flush=True)

        # Re-derive after market_cache is filled — fresh rows expose new event_slugs.
        missing_slugs = _event_slugs_missing_tag_cache(db)
        print(
            f"\ndistinct event_slugs missing event_tag_cache (post market_cache backfill): "
            f"{len(missing_slugs)}",
            flush=True,
        )

        print("\n=== phase 2: backfill event_tag_cache ===", flush=True)
        etc_ok, etc_fail = await _backfill_event_tag_cache(
            event_tag_cache=event_tag_cache,
            gamma=gamma,
            slugs=missing_slugs,
        )
        print(f"event_tag_cache complete: {etc_ok} ok / {etc_fail} fail", flush=True)
    finally:
        await gamma_http.aclose()
        await data_http.aclose()
        db.close()
    return 0


def main() -> None:
    """Run the async backfill pipeline and propagate exit code."""
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
