"""Show what the watchlisted wallets bet on for a given market.

Given a Polymarket market (slug, full URL, or an event slug/URL that groups
several markets), this resolves the market's CTF token ids via gamma, then
queries the V2 Polymarket subgraph for every *position-increasing* trade made
by any active ``wallet_watchlist`` wallet on those tokens. It reports the
side split (by distinct wallets, trade count, shares, and notional), the
wallet directionality breakdown, and the top wallets by notional.

The "did this wallet buy this token?" attribution mirrors the daemon's
``SubgraphTradeCollector`` (#152): a watchlist wallet's position increases
when it is the maker on a BUY order (``side == 0``) or the taker against a
SELL order (``side == 1``); the ``tokenId`` it acquires identifies the side.

Notional is ``price * size`` (USDC). It is NOT realized PnL — see
``scripts/check_polymarket_wallet_pnl.py`` for the on-chain cash reality
check before trusting any wallet as a copy seed.

Usage::

    uv run python scripts/analyze_market_watchlist_picks.py \
        https://polymarket.com/sports/nba/nba-sas-okc-2026-05-30
    uv run python scripts/analyze_market_watchlist_picks.py nba-sas-okc-2026-05-30 \
        --db data/pscanner.sqlite3 --top 20

Requires ``GRAPH_API_KEY`` (and optionally ``GRAPH_API_KEY_BACKUP`` for
quota failover) in the environment or in a ``.env`` file at the repo root.
"""
# ruff: noqa: T201  # script prints a report to stdout by design

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Market
from pscanner.poly.subgraph import SubgraphClient

_DEFAULT_DB = Path("data/pscanner.sqlite3")
_DEFAULT_SUBGRAPH_ID = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
_DEFAULT_TOP = 20
_PAGE_SIZE = 1000
_GATEWAY = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{sid}"


def _load_env_keys() -> tuple[str | None, str | None]:
    """Return ``(GRAPH_API_KEY, GRAPH_API_KEY_BACKUP)`` from env or ``.env``.

    Environment variables win; the ``.env`` file at the repo root is parsed
    only for keys not already present in ``os.environ`` (the daemon reads
    ``os.environ`` directly, so this mirrors its precedence).
    """
    keys = {
        "GRAPH_API_KEY": os.environ.get("GRAPH_API_KEY"),
        "GRAPH_API_KEY_BACKUP": os.environ.get("GRAPH_API_KEY_BACKUP"),
    }
    env_file = Path(".env")
    if env_file.exists() and not all(keys.values()):
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            name = name.strip()
            if name in keys and not keys[name]:
                keys[name] = value.strip().strip('"').strip("'")
    return keys["GRAPH_API_KEY"], keys["GRAPH_API_KEY_BACKUP"]


def _load_watchlist(db: Path) -> set[str]:
    """Return the lowercased set of active ``wallet_watchlist`` addresses."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT address FROM wallet_watchlist WHERE active = 1").fetchall()
    finally:
        conn.close()
    return {r[0].lower() for r in rows}


def _slug_from_arg(market_arg: str) -> str:
    """Extract a slug from a bare slug or a full Polymarket URL."""
    cleaned = market_arg.strip().rstrip("/")
    if "polymarket.com" in cleaned:
        return cleaned.split("/")[-1]
    return cleaned


def _render_where(where: dict[str, Any]) -> str:
    """Render a ``where:`` dict as a GraphQL object literal (keys unquoted)."""
    if isinstance(where, str):
        return json.dumps(where)
    if isinstance(where, bool):
        return "true" if where else "false"
    if isinstance(where, int):
        return str(where)
    if isinstance(where, list):
        return "[" + ",".join(_render_where(x) for x in where) + "]"
    if isinstance(where, dict):
        return "{" + ",".join(f"{k}:{_render_where(v)}" for k, v in where.items()) + "}"
    raise TypeError(f"unsupported where value: {where!r}")


def _is_buy(maker: str, taker: str, side: int, watchlist: set[str]) -> str | None:
    """Return the buying watchlist wallet, or ``None`` if not a position-add.

    Mirrors ``SubgraphTradeCollector._compute_copy_direction`` (#152):
    maker+side==0 (maker accumulates) or taker+side==1 (taker bought).
    """
    maker_l, taker_l = maker.lower(), taker.lower()
    if maker_l in watchlist and side == 0:
        return maker_l
    if taker_l in watchlist and side == 1:
        return taker_l
    return None


async def _resolve_markets(gamma: GammaClient, slug: str) -> list[Market]:
    """Resolve a slug to one or more markets (single market, else event)."""
    market = await gamma.get_market_by_slug(slug)
    if market is not None:
        return [market]
    event = await gamma.get_event_by_slug(slug)
    if event is not None and event.markets:
        return event.markets
    raise SystemExit(f"no market or event found for slug {slug!r}")


async def _fetch_buys(
    subgraph: SubgraphClient, tokens: list[str], addrs: list[str]
) -> list[dict[str, Any]]:
    """Drain the subgraph for watchlist position-increasing trades on tokens.

    Filters server-side by ``market_in`` (the CTF token ids) AND watchlist
    membership on the buying side, paginated ascending by timestamp.
    """
    fields = "transactionHash timestamp maker{id} taker{id} tokenId side price size"
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ts = 0
    while True:
        where = {
            "or": [
                {"timestamp_gte": str(ts), "market_in": tokens, "maker_in": addrs, "side": 0},
                {"timestamp_gte": str(ts), "market_in": tokens, "taker_in": addrs, "side": 1},
            ]
        }
        query = (
            f"{{orderFilledEvents(where:{_render_where(where)} first:{_PAGE_SIZE} "
            f"orderBy:timestamp orderDirection:asc){{{fields}}}}}"
        )
        page = (await subgraph.query(query, {})).get("orderFilledEvents") or []
        for ev in page:
            key = (ev["transactionHash"], ev["tokenId"])
            if key in seen:
                continue
            seen.add(key)
            events.append(ev)
        if len(page) < _PAGE_SIZE:
            return events
        ts = max(int(ev["timestamp"]) for ev in page)


def _aggregate(
    events: list[dict[str, Any]], watchlist: set[str], token_to_outcome: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]], tuple[int, int] | None]:
    """Roll events up into per-side stats and per-wallet per-side notional."""
    side_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "wallets": set(), "shares": 0.0, "notional": 0.0}
    )
    wallet_side: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    span: list[int] = []
    for ev in events:
        wallet = _is_buy(ev["maker"]["id"], ev["taker"]["id"], int(ev["side"]), watchlist)
        if wallet is None:
            continue
        outcome = token_to_outcome.get(ev["tokenId"], ev["tokenId"])
        notional = float(ev["price"]) * float(ev["size"])
        span.append(int(ev["timestamp"]))
        stat = side_stats[outcome]
        stat["trades"] += 1
        stat["wallets"].add(wallet)
        stat["shares"] += float(ev["size"])
        stat["notional"] += notional
        wallet_side[wallet][outcome] += notional
    tspan = (min(span), max(span)) if span else None
    return side_stats, wallet_side, tspan


def _fmt_ts(ts: int) -> str:
    """Format a unix timestamp as ``YYYY-MM-DD HH:MM UTC``."""
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def _render(
    market: Market,
    side_stats: dict[str, dict[str, Any]],
    wallet_side: dict[str, dict[str, float]],
    tspan: tuple[int, int] | None,
    top: int,
) -> None:
    """Print the per-market watchlist-picks report."""
    outcomes = market.outcomes or list(side_stats)
    prices = dict(zip(outcomes, market.outcome_prices, strict=False))
    total = sum(s["trades"] for s in side_stats.values())
    print(f"\n=== {market.question} ({market.slug}) ===")
    print(f"condition_id: {market.condition_id}")
    if prices:
        print(
            "market price: " + "  ".join(f"{o} {prices.get(o, float('nan')):.3f}" for o in outcomes)
        )
    span = f"{_fmt_ts(tspan[0])} -> {_fmt_ts(tspan[1])}" if tspan else "(no trades)"
    print(f"watchlist buys: {total} trades by {len(wallet_side)} wallets   span: {span}\n")
    print(
        f"{'SIDE':14} {'wallets':>8} {'trades':>7} {'shares':>13} {'notional$':>13} {'avg_px':>7}"
    )
    for outcome in outcomes:
        stat = side_stats.get(outcome)
        if not stat:
            print(f"{outcome[:14]:14} {0:>8} {0:>7} {0:>13} {0:>13} {'-':>7}")
            continue
        avg = stat["notional"] / stat["shares"] if stat["shares"] else 0.0
        print(
            f"{outcome[:14]:14} {len(stat['wallets']):>8} {stat['trades']:>7} "
            f"{stat['shares']:>13,.0f} {stat['notional']:>13,.0f} {avg:>7.3f}"
        )
    _render_directionality(wallet_side, outcomes, top)


def _render_directionality(
    wallet_side: dict[str, dict[str, float]], outcomes: list[str], top: int
) -> None:
    """Print one-sided wallet counts and the top wallets by notional."""
    if not wallet_side:
        return
    one_sided: dict[str, int] = dict.fromkeys(outcomes, 0)
    mixed = 0
    for sides in wallet_side.values():
        active = [o for o in sides if sides[o] > 0]
        if len(active) == 1:
            one_sided[active[0]] = one_sided.get(active[0], 0) + 1
        else:
            mixed += 1
    print(
        "\ndirectionality: "
        + "  ".join(f"{o}-only={one_sided.get(o, 0)}" for o in outcomes)
        + f"  mixed={mixed}"
    )
    ranked = sorted(wallet_side.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
    print(f"\ntop {min(top, len(ranked))} wallets by notional:")
    for wallet, sides in ranked[:top]:
        dom = max(sides, key=lambda o: sides[o])
        legs = "  ".join(f"{o}=${sides.get(o, 0.0):>11,.0f}" for o in outcomes)
        print(f"  {wallet}  {dom[:10]:10}  {legs}")


async def _run(args: argparse.Namespace) -> None:
    """Resolve the market(s) and render a report for each."""
    primary, backup = _load_env_keys()
    if not primary and not backup:
        raise SystemExit("GRAPH_API_KEY not set (env or .env)")
    watchlist = _load_watchlist(args.db)
    if not watchlist:
        raise SystemExit(f"no active watchlist wallets in {args.db}")
    addrs = sorted(watchlist)
    primary_key = primary or backup
    url = _GATEWAY.format(key=primary_key, sid=args.subgraph_id)
    fallback = (
        _GATEWAY.format(key=backup, sid=args.subgraph_id)
        if backup and backup != primary_key
        else None
    )
    gamma = GammaClient(rpm=args.rpm)
    subgraph = SubgraphClient(url=url, rpm=args.rpm, fallback_url=fallback)
    try:
        markets = await _resolve_markets(gamma, _slug_from_arg(args.market))
        print(f"watchlist: {len(addrs)} active wallets   markets: {len(markets)}")
        for market in markets:
            tokens = [str(t) for t in market.clob_token_ids]
            token_to_outcome = dict(zip(tokens, market.outcomes, strict=False))
            events = await _fetch_buys(subgraph, tokens, addrs)
            side_stats, wallet_side, tspan = _aggregate(events, watchlist, token_to_outcome)
            _render(market, side_stats, wallet_side, tspan, args.top)
    finally:
        await subgraph.aclose()
        await gamma.aclose()


def main() -> None:
    """Parse CLI args and run the report."""
    parser = argparse.ArgumentParser(
        description="Show what watchlisted wallets bet on for a given market."
    )
    parser.add_argument("market", help="market slug or Polymarket URL (market or event)")
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help=f"daemon DB with wallet_watchlist (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--subgraph-id", default=_DEFAULT_SUBGRAPH_ID, help="V2 subgraph id to query"
    )
    parser.add_argument("--rpm", type=int, default=50, help="requests/min budget")
    parser.add_argument(
        "--top", type=int, default=_DEFAULT_TOP, help="how many top wallets to list per market"
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
