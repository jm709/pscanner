"""Expand a known wallet cluster by fingerprint-matching co-traders.

Given seed wallets (a known coordinated cluster), this script:

1. Reads each seed wallet's trades from the local ``wallet_trades`` table.
2. Selects "high-coordination" shared markets (≥3 seed wallets co-traded OR
   ≥4 total seed trades) and computes the cluster's per-market trade window.
3. Paginates ``data-api.polymarket.com/trades?market=`` for each shared market
   within ±1h of the cluster's window, collecting every counterparty.
4. Scores candidates by market-overlap and a behavioral-fingerprint distance
   (sell rate, ≥0.95 price share, $500-999 / sub-$100 size bands).
5. Looks up each candidate's first-activity timestamp from data-api and flags
   whether it falls inside the seed cluster's creation window.

Validated 2026-04-26 against the Cavill cluster: a 9-wallet seed expanded to
~190 wallets, with 99.5% of new candidates created in the same Feb 20-21
24-hour window as the seed.

Usage:
    uv run python scripts/expand_cluster.py --wallet 0xabc... --wallet 0xdef...
    uv run python scripts/expand_cluster.py --cluster-id cluster-abc123
"""

# ruff: noqa: T201  # script prints progress to stdout by design
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from pscanner.config import Config
from pscanner.poly.data import DataClient
from pscanner.store.db import init_db

_DATA_API = "https://data-api.polymarket.com"
_PAGE_SIZE = 500
_WINDOW_PAD_SEC = 3600
_PAGE_CAP = 30  # 15k trades per market max
_HIGH_COORD_MIN_WALLETS = 3
_HIGH_COORD_MIN_TRADES = 4
_CANDIDATE_MIN_MARKETS = 2
_CANDIDATE_MIN_TRADES = 4

# Cluster behavioral fingerprint (from Cavill 9, n=290):
# sell rate 33%, price ≥0.95 share 56%, sub-$100 share 52%, $500-999 share 47%.
_FP_SELL = 0.33
_FP_HIGH_PRICE = 0.56
_FP_SUB_100 = 0.52
_FP_BAND = 0.47

# Size buckets for the fingerprint computation (USD).
_HIGH_PRICE_THRESHOLD = 0.95
_DUST_USD = 100.0
_BAND_LO_USD = 500.0
_BAND_HI_USD = 1000.0
_MIN_SEED_WALLETS = 2


@dataclass(frozen=True)
class SeedMarket:
    """One market the seed cluster traded, with the cluster's window on it."""

    condition_id: str
    ts_min: int
    ts_max: int
    n_seed_trades: int
    n_seed_wallets: int


@dataclass
class Candidate:
    """A non-seed wallet scored against the cluster's behavioral fingerprint."""

    wallet: str
    n_trades: int
    n_markets: int
    markets: list[str]
    sell_rate: float
    hi_price_rate: float
    sub100_rate: float
    band_500_999_rate: float
    median_usd: float
    fingerprint_score: float
    pseudonym: str
    name: str
    ts_min: int
    ts_max: int
    first_activity_ts: int | None = None


def _seed_wallets_from_cluster(conn: sqlite3.Connection, cluster_id: str) -> list[str]:
    """Look up cluster members from the local DB."""
    rows = conn.execute(
        "SELECT wallet FROM wallet_cluster_members WHERE cluster_id = ? ORDER BY wallet",
        (cluster_id,),
    ).fetchall()
    return [r["wallet"].lower() for r in rows]


def _select_seed_markets(conn: sqlite3.Connection, seeds: list[str]) -> list[SeedMarket]:
    """Group seed-wallet trades by condition_id; keep high-coordination markets."""
    # Placeholders are a fixed comma-separated list of '?' — no user data; safe.
    placeholders = ",".join("?" * len(seeds))
    query = f"""
        SELECT condition_id, COUNT(*) AS n_trades,
               COUNT(DISTINCT wallet) AS n_wallets,
               MIN(timestamp) AS ts_min, MAX(timestamp) AS ts_max
        FROM wallet_trades
        WHERE wallet IN ({placeholders})
        GROUP BY condition_id ORDER BY n_trades DESC
    """  # noqa: S608
    rows = conn.execute(query, seeds).fetchall()
    return [
        SeedMarket(
            condition_id=r["condition_id"],
            ts_min=r["ts_min"],
            ts_max=r["ts_max"],
            n_seed_trades=r["n_trades"],
            n_seed_wallets=r["n_wallets"],
        )
        for r in rows
        if r["n_wallets"] >= _HIGH_COORD_MIN_WALLETS or r["n_trades"] >= _HIGH_COORD_MIN_TRADES
    ]


async def _fetch_market_trades(
    client: httpx.AsyncClient,
    market: SeedMarket,
) -> list[dict[str, Any]]:
    """Page newest-first; stop once trades fall below the cluster window."""
    out: list[dict[str, Any]] = []
    pad_min = market.ts_min - _WINDOW_PAD_SEC
    pad_max = market.ts_max + _WINDOW_PAD_SEC
    offset = 0
    for _ in range(_PAGE_CAP):
        r = await client.get(
            f"{_DATA_API}/trades",
            params={"market": market.condition_id, "limit": _PAGE_SIZE, "offset": offset},
        )
        r.raise_for_status()
        page = r.json()
        if not isinstance(page, list) or not page:
            break
        page_max_ts = max((t.get("timestamp", 0) for t in page), default=0)
        for t in page:
            ts = t.get("timestamp")
            if isinstance(ts, int) and pad_min <= ts <= pad_max:
                out.append(t)
        if page_max_ts < pad_min or len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return out


async def _collect_co_trader_rows(
    markets: list[SeedMarket],
    *,
    rpm: int,
) -> list[dict[str, Any]]:
    """Pull trades for every seed market; return merged rows tagged with cond."""
    rows: list[dict[str, Any]] = []
    inter_request = 60.0 / max(rpm, 1)
    async with httpx.AsyncClient(
        timeout=60.0,
        headers={"user-agent": "pscanner-cluster-expand/1.0"},
    ) as client:
        for i, market in enumerate(markets, 1):
            t0 = time.time()
            page = await _fetch_market_trades(client, market)
            elapsed = time.time() - t0
            for r in page:
                r["_seed_cond"] = market.condition_id
            rows.extend(page)
            print(
                f"[{i:>2}/{len(markets)}] cond={market.condition_id[:14]}…  "
                f"window={market.ts_min}..{market.ts_max}  fetched={len(page):>4}  "
                f"({elapsed:.1f}s)"
            )
            await asyncio.sleep(inter_request)
    return rows


def _score_candidates(rows: list[dict[str, Any]], seeds: set[str]) -> list[Candidate]:
    """Group rows by wallet, drop seeds, score each non-seed against fingerprint."""
    by_wallet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        wallet = (r.get("proxyWallet") or "").lower()
        if wallet and wallet not in seeds:
            by_wallet[wallet].append(r)

    candidates: list[Candidate] = []
    for wallet, trades in by_wallet.items():
        markets = sorted({t["_seed_cond"] for t in trades})
        if len(markets) < _CANDIDATE_MIN_MARKETS and len(trades) < _CANDIDATE_MIN_TRADES:
            continue
        candidates.append(_score_one(wallet, trades, markets))
    candidates.sort(key=lambda c: (c.n_markets, c.fingerprint_score), reverse=True)
    return candidates


def _score_one(wallet: str, trades: list[dict[str, Any]], markets: list[str]) -> Candidate:
    """Compute one candidate's behavioral metrics and fingerprint score."""
    n = len(trades)
    usd = [(t.get("size") or 0.0) * (t.get("price") or 0.0) for t in trades]
    sells = sum(1 for t in trades if t.get("side") == "SELL")
    hi = sum(1 for t in trades if (t.get("price") or 0.0) >= _HIGH_PRICE_THRESHOLD)
    sub100 = sum(1 for v in usd if v < _DUST_USD)
    band = sum(1 for v in usd if _BAND_LO_USD <= v < _BAND_HI_USD)
    median_usd = sorted(usd)[n // 2] if usd else 0.0
    sell_rate, hi_rate, sub_rate, band_rate = sells / n, hi / n, sub100 / n, band / n
    score = sum(
        max(0.0, 1.0 - abs(actual - target) / target)
        for actual, target in (
            (sell_rate, _FP_SELL),
            (hi_rate, _FP_HIGH_PRICE),
            (sub_rate, _FP_SUB_100),
            (band_rate, _FP_BAND),
        )
    )
    score += min(len(markets), 5) * 0.5
    return Candidate(
        wallet=wallet,
        n_trades=n,
        n_markets=len(markets),
        markets=markets,
        sell_rate=sell_rate,
        hi_price_rate=hi_rate,
        sub100_rate=sub_rate,
        band_500_999_rate=band_rate,
        median_usd=median_usd,
        fingerprint_score=score,
        pseudonym=trades[0].get("pseudonym") or "",
        name=trades[0].get("name") or "",
        ts_min=min(t.get("timestamp", 0) for t in trades),
        ts_max=max(t.get("timestamp", 0) for t in trades),
    )


async def _verify_first_activity(
    candidates: list[Candidate],
    *,
    rpm: int,
) -> None:
    """Populate ``first_activity_ts`` on every candidate via the data-api."""
    client = DataClient(rpm=rpm)
    inter_request = 60.0 / max(rpm, 1)
    try:
        for i, c in enumerate(candidates, 1):
            try:
                ts = await client.get_first_activity_timestamp(c.wallet)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"[{i:>3}/{len(candidates)}] {c.wallet}  ERR {exc}")
                continue
            c.first_activity_ts = ts
            ts_str = dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M") if ts else "—"
            print(
                f"[{i:>3}/{len(candidates)}] {c.wallet}  first={ts_str}  "
                f"mkts={c.n_markets} score={c.fingerprint_score:.2f}"
            )
            await asyncio.sleep(inter_request)
    finally:
        await client.aclose()


async def _seed_first_activity(seeds: list[str], *, rpm: int) -> dict[str, int]:
    """Fetch each seed's first-activity ts."""
    client = DataClient(rpm=rpm)
    inter_request = 60.0 / max(rpm, 1)
    out: dict[str, int] = {}
    try:
        for seed in seeds:
            ts = await client.get_first_activity_timestamp(seed)
            if ts is not None:
                out[seed] = ts
            await asyncio.sleep(inter_request)
    finally:
        await client.aclose()
    return out


def _summarize(
    candidates: list[Candidate],
    *,
    seed_window: tuple[int, int] | None,
) -> dict[str, Any]:
    """Bucket candidates by whether their first-activity falls in the seed window."""
    in_window = [c for c in candidates if _ts_in(c.first_activity_ts, seed_window)]
    out_window = [
        c
        for c in candidates
        if c.first_activity_ts is not None and not _ts_in(c.first_activity_ts, seed_window)
    ]
    no_ts = [c for c in candidates if c.first_activity_ts is None]
    return {
        "total": len(candidates),
        "in_seed_window": len(in_window),
        "outside_window": len(out_window),
        "no_first_activity": len(no_ts),
        "seed_window": list(seed_window) if seed_window else None,
        "in_seed_window_addresses": [c.wallet for c in in_window],
    }


def _ts_in(ts: int | None, window: tuple[int, int] | None) -> bool:
    """Return True iff ``ts`` is non-null and inside the inclusive window."""
    if ts is None or window is None:
        return False
    return window[0] <= ts <= window[1]


def _to_jsonable(c: Candidate) -> dict[str, Any]:
    """Dataclass → plain dict (json doesn't handle dataclasses natively)."""
    return {
        "wallet": c.wallet,
        "n_trades": c.n_trades,
        "n_markets": c.n_markets,
        "markets": c.markets,
        "sell_rate": c.sell_rate,
        "hi_price_rate": c.hi_price_rate,
        "sub100_rate": c.sub100_rate,
        "band_500_999_rate": c.band_500_999_rate,
        "median_usd": c.median_usd,
        "fingerprint_score": c.fingerprint_score,
        "pseudonym": c.pseudonym,
        "name": c.name,
        "ts_min": c.ts_min,
        "ts_max": c.ts_max,
        "first_activity_ts": c.first_activity_ts,
        "first_activity_iso": (
            dt.datetime.fromtimestamp(c.first_activity_ts, dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
            if c.first_activity_ts
            else None
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a known wallet cluster.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--wallet",
        action="append",
        help="Seed wallet 0x address (repeat for multiple).",
    )
    src.add_argument(
        "--cluster-id",
        help="Seed wallets via wallet_cluster_members lookup.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=50,
        help="Request rate budget for data-api (default 50).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: ./data/cluster_expansion_<seed_hash>.json).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip per-candidate first-activity lookup (faster, less precise).",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    config = Config.load()
    conn = init_db(config.scanner.db_path)
    try:
        seeds = sorted({w.lower() for w in (args.wallet or [])})
        if args.cluster_id:
            seeds = _seed_wallets_from_cluster(conn, args.cluster_id)
        if len(seeds) < _MIN_SEED_WALLETS:
            print(f"need ≥{_MIN_SEED_WALLETS} seed wallets, got {len(seeds)}", file=sys.stderr)
            return 2
        print(f"seeds: {len(seeds)} wallets")

        seed_markets = _select_seed_markets(conn, seeds)
        if not seed_markets:
            print("no high-coordination markets in local DB for these seeds", file=sys.stderr)
            return 3
        print(f"high-coordination markets: {len(seed_markets)}")

        rows = await _collect_co_trader_rows(seed_markets, rpm=args.rpm)
        print(f"co-trader rows fetched: {len(rows)}")

        candidates = _score_candidates(rows, set(seeds))
        print(f"candidates passing overlap/trade-count gate: {len(candidates)}")

        seed_window: tuple[int, int] | None = None
        if not args.skip_verify and candidates:
            await _verify_first_activity(candidates, rpm=args.rpm)
            seed_first = await _seed_first_activity(seeds, rpm=args.rpm)
            if seed_first:
                seed_window = (min(seed_first.values()), max(seed_first.values()))
                print(f"seed creation window: {seed_window[0]}..{seed_window[1]}")

        summary = _summarize(candidates, seed_window=seed_window)
        terse = {k: v for k, v in summary.items() if k != "in_seed_window_addresses"}
        print(f"\nSUMMARY: {json.dumps(terse, indent=2)}")

        out = args.out or _default_out_path(config.scanner.db_path, seeds)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "seeds": seeds,
                    "seed_markets": [m.__dict__ for m in seed_markets],
                    "summary": summary,
                    "candidates": [_to_jsonable(c) for c in candidates],
                },
                indent=2,
            )
        )
        print(f"\nwrote {len(candidates)} candidates → {out}")
        return 0
    finally:
        conn.close()


def _default_out_path(db_path: Path, seeds: list[str]) -> Path:
    """Derive a stable output filename from the seed addresses."""
    digest = hashlib.sha256("|".join(seeds).encode()).hexdigest()[:12]
    return db_path.parent / f"cluster_expansion_{digest}.json"


def main() -> int:
    """Entry point — parse args and run the async pipeline."""
    return asyncio.run(_async_main(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
