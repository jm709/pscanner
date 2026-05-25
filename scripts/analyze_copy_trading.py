"""Analyze the subgraph_copy paper-trading portfolio.

Prints a multi-section report covering:

- Headline aggregates (positions, exposure, realized PnL, win rate).
- Top wallets by open cost basis (with HHI concentration index).
- Top markets by open cost basis (with titles from market_cache).
- Per-category exposure (open + resolved $, win rate) via event_tag_cache
  and ``pscanner.categories.categorize_tags``.
- Delay distribution + delay-vs-PnL bucket table.
- Concentration warnings (per-wallet and per-market exposure thresholds).

Usage::

    uv run python scripts/analyze_copy_trading.py [--db data/pscanner.sqlite3]
        [--top-n 20] [--wallet-concentration-threshold 0.05]
        [--market-concentration-threshold 0.02]
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pscanner.categories import primary_category

_UNCATEGORIZED = "(uncategorized)"

_DEFAULT_DB = Path("data/pscanner.sqlite3")
_DEFAULT_TOP_N = 20
_DEFAULT_WALLET_THRESHOLD = 0.05
_DEFAULT_MARKET_THRESHOLD = 0.02
_DETECTOR = "subgraph_copy"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB, help="Daemon SQLite DB path.")
    p.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N, help="Top-N rows per section.")
    p.add_argument(
        "--wallet-concentration-threshold",
        type=float,
        default=_DEFAULT_WALLET_THRESHOLD,
        help="Flag wallets whose share of total open exposure exceeds this fraction.",
    )
    p.add_argument(
        "--market-concentration-threshold",
        type=float,
        default=_DEFAULT_MARKET_THRESHOLD,
        help="Flag markets whose share of total open exposure exceeds this fraction.",
    )
    return p.parse_args()


def _load_entries(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return one row per subgraph_copy entry, joined to the matching exit (if any).

    Columns: trade_id, source_wallet, condition_id, asset_id, outcome,
    fill_price, cost_usd, paper_ts, body_json, exit_cost_usd, event_slug, title.
    """
    return db.execute(
        """
        SELECT e.trade_id, e.source_wallet, e.condition_id, e.asset_id, e.outcome,
               e.fill_price, e.cost_usd, e.ts AS paper_ts,
               a.body_json,
               x.cost_usd AS exit_cost_usd,
               mc.event_slug, mc.title
          FROM paper_trades e
          JOIN alerts a ON a.alert_key = e.triggering_alert_key
          LEFT JOIN paper_trades x ON x.parent_trade_id = e.trade_id AND x.trade_kind='exit'
          LEFT JOIN market_cache mc ON mc.condition_id = e.condition_id
         WHERE e.trade_kind = 'entry'
           AND e.triggering_alert_detector = ?
        """,
        (_DETECTOR,),
    ).fetchall()


def _load_event_tags(db: sqlite3.Connection) -> dict[str, list[str]]:
    """Return ``{event_slug: [tag, ...]}`` from event_tag_cache."""
    out: dict[str, list[str]] = {}
    for r in db.execute("SELECT event_slug, tags_json FROM event_tag_cache"):
        try:
            tags = json.loads(r["tags_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(tags, list):
            out[r["event_slug"]] = [str(t) for t in tags if isinstance(t, str)]
    return out


def _update_bucket(bucket: dict[str, float], row: sqlite3.Row) -> None:
    """Fold one paper_trades row into an accumulator bucket.

    Bucket keys: open_cost, resolved_cost, pnl, n_open, n_resolved, wins.
    Open rows (exit_cost_usd IS NULL) contribute to open_cost/n_open.
    Resolved rows contribute to resolved_cost/n_resolved/pnl/wins.
    """
    cost = float(row["cost_usd"])
    exit_cost = row["exit_cost_usd"]
    if exit_cost is None:
        bucket["open_cost"] = float(bucket["open_cost"]) + cost
        bucket["n_open"] = int(bucket["n_open"]) + 1
        return
    bucket["resolved_cost"] = float(bucket["resolved_cost"]) + cost
    bucket["n_resolved"] = int(bucket["n_resolved"]) + 1
    pnl = float(exit_cost) - cost
    bucket["pnl"] = float(bucket["pnl"]) + pnl
    if pnl > 0:
        bucket["wins"] = int(bucket["wins"]) + 1


def _hhi(shares: Iterable[float]) -> float:
    """Herfindahl-Hirschman Index on shares that sum to <= 1.0.

    Returns 0.0 for empty input. A perfectly diversified portfolio with N
    equal-sized positions has HHI = 1/N. A single position has HHI = 1.0.
    """
    s = list(shares)
    if not s:
        return 0.0
    return sum(x * x for x in s)


def _print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _print_headline(entries: list[sqlite3.Row]) -> None:
    open_n = sum(1 for r in entries if r["exit_cost_usd"] is None)
    resolved = [r for r in entries if r["exit_cost_usd"] is not None]
    open_cost = sum(r["cost_usd"] for r in entries if r["exit_cost_usd"] is None)
    res_cost = sum(r["cost_usd"] for r in resolved)
    realized_pnl = sum(r["exit_cost_usd"] - r["cost_usd"] for r in resolved)
    wins = sum(1 for r in resolved if r["exit_cost_usd"] > r["cost_usd"])
    _print_section("subgraph_copy portfolio headline")
    print(f"  total positions:    {len(entries):>8}")
    print(f"  open positions:     {open_n:>8}    open cost basis:   ${open_cost:>12.2f}")
    print(f"  resolved positions: {len(resolved):>8}    resolved entry $:  ${res_cost:>12.2f}")
    if resolved:
        print(
            f"  realized PnL:       ${realized_pnl:>+12.2f}"
            f"    return on resolved: {realized_pnl / res_cost * 100:+.1f}%"
        )
        print(f"  win rate (resolved): {wins / len(resolved) * 100:.1f}%")


def _print_wallet_exposure(
    entries: list[sqlite3.Row],
    *,
    top_n: int,
    concentration_threshold: float,
) -> None:
    per_wallet: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "open_cost": 0.0,
            "resolved_cost": 0.0,
            "pnl": 0.0,
            "n_open": 0,
            "n_resolved": 0,
            "wins": 0,
        },
    )
    for r in entries:
        w = r["source_wallet"] or "(none)"
        _update_bucket(per_wallet[w], r)

    total_open = sum(b["open_cost"] for b in per_wallet.values())
    rows = sorted(per_wallet.items(), key=lambda kv: kv[1]["open_cost"], reverse=True)

    _print_section(f"top {min(top_n, len(rows))} wallets by open cost basis")
    print(
        f"  {'wallet':42s}  {'open$':>10}  {'open#':>5}  "
        f"{'res$':>10}  {'res#':>5}  {'pnl':>10}  {'win%':>6}  {'share':>6}"
    )
    for w, b in rows[:top_n]:
        share = (b["open_cost"] / total_open) if total_open else 0.0
        win = (b["wins"] / b["n_resolved"] * 100) if b["n_resolved"] else 0.0
        print(
            f"  {w[:42]:42s}  ${b['open_cost']:>9.2f}  {b['n_open']:>5d}  "
            f"${b['resolved_cost']:>9.2f}  {b['n_resolved']:>5d}  "
            f"${b['pnl']:>+9.2f}  {win:>5.1f}%  {share * 100:>5.1f}%"
        )

    shares = [b["open_cost"] / total_open for b in per_wallet.values() if total_open]
    hhi = _hhi(shares)
    effective_n = (1.0 / hhi) if hhi else 0.0
    print()
    print(
        f"  wallet HHI: {hhi:.4f}    effective N (1/HHI): {effective_n:.1f}    "
        f"distinct wallets: {len(per_wallet)}"
    )


def _print_market_exposure(entries: list[sqlite3.Row], *, top_n: int) -> None:
    per_market: dict[str, dict[str, float | str]] = defaultdict(
        lambda: {
            "open_cost": 0.0,
            "resolved_cost": 0.0,
            "pnl": 0.0,
            "n_open": 0,
            "n_resolved": 0,
            "wins": 0,
            "title": "(unknown)",
        },
    )
    for r in entries:
        cid = r["condition_id"]
        bucket = per_market[cid]
        if r["title"]:
            bucket["title"] = r["title"]
        _update_bucket(bucket, r)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    rows = sorted(per_market.items(), key=lambda kv: float(kv[1]["open_cost"]), reverse=True)

    _print_section(f"top {min(top_n, len(rows))} markets by open cost basis")
    print(f"  {'condition_id':24s}  {'open$':>10}  {'open#':>5}  {'pnl':>10}  {'res#':>5}  title")
    for cid, b in rows[:top_n]:
        title = str(b["title"])[:60]
        print(
            f"  {cid[:24]:24s}  ${float(b['open_cost']):>9.2f}  {int(b['n_open']):>5d}  "
            f"${float(b['pnl']):>+9.2f}  {int(b['n_resolved']):>5d}  {title}"
        )

    print(f"\n  distinct markets: {len(per_market)}")


def _print_category_exposure(
    entries: list[sqlite3.Row],
    event_tags: dict[str, list[str]],
) -> None:
    per_category: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "open_cost": 0.0,
            "resolved_cost": 0.0,
            "pnl": 0.0,
            "n_open": 0,
            "n_resolved": 0,
            "wins": 0,
        },
    )
    no_tags = 0
    for r in entries:
        slug = r["event_slug"]
        tags = event_tags.get(slug, []) if slug else []
        if not tags:
            no_tags += 1
        # Use primary category as the bucket key — multi-label would over-count $.
        cat = primary_category(tags).value if tags else _UNCATEGORIZED
        _update_bucket(per_category[cat], r)

    _print_section("category exposure (by primary category)")
    print(
        f"  {'category':14s}  {'open$':>10}  {'open#':>6}  "
        f"{'res$':>10}  {'res#':>6}  {'pnl':>10}  {'win%':>6}"
    )
    for cat, b in sorted(per_category.items(), key=lambda kv: kv[1]["open_cost"], reverse=True):
        win = (b["wins"] / b["n_resolved"] * 100) if b["n_resolved"] else 0.0
        print(
            f"  {cat:14s}  ${b['open_cost']:>9.2f}  {b['n_open']:>6d}  "
            f"${b['resolved_cost']:>9.2f}  {b['n_resolved']:>6d}  "
            f"${b['pnl']:>+9.2f}  {win:>5.1f}%"
        )
    if no_tags:
        print(f"\n  (entries without tag data: {no_tags} — counted under '{_UNCATEGORIZED}')")


_DELAY_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0-1m", 0, 60),
    ("1-5m", 60, 300),
    ("5-15m", 300, 900),
    ("15-30m", 900, 1800),
    ("30-60m", 1800, 3600),
    ("1-2h", 3600, 7200),
    ("2-6h", 7200, 21600),
    ("6h+", 21600, 10**9),
)


def _collect_delays(
    entries: list[sqlite3.Row],
) -> tuple[list[int], list[tuple[int, float]]]:
    """Return ``(delays, resolved_pairs)`` for delay analysis.

    ``delays`` is every (paper_ts - seed_ts) that is >= 0. ``resolved_pairs``
    is ``(delay, pnl_per_dollar)`` for the subset that has booked an exit.
    """
    delays: list[int] = []
    resolved_pairs: list[tuple[int, float]] = []
    for r in entries:
        try:
            seed_ts = int(json.loads(r["body_json"])["ts"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        delay = r["paper_ts"] - seed_ts
        if delay < 0:
            continue
        delays.append(delay)
        if r["exit_cost_usd"] is not None and r["cost_usd"] > 0:
            pnl = (r["exit_cost_usd"] - r["cost_usd"]) / r["cost_usd"]
            resolved_pairs.append((delay, pnl))
    return delays, resolved_pairs


def _print_delay_distribution(delays: list[int]) -> None:
    _print_section(f"delay distribution (n={len(delays)})")
    if not delays:
        return
    delays.sort()
    n = len(delays)
    print(f"  min:    {delays[0]:>8d}s ({delays[0] / 60:>6.1f} min)")
    print(f"  p25:    {delays[n // 4]:>8d}s ({delays[n // 4] / 60:>6.1f} min)")
    print(f"  median: {delays[n // 2]:>8d}s ({delays[n // 2] / 60:>6.1f} min)")
    print(f"  p75:    {delays[n * 3 // 4]:>8d}s ({delays[n * 3 // 4] / 60:>6.1f} min)")
    print(f"  p95:    {delays[n * 95 // 100]:>8d}s ({delays[n * 95 // 100] / 60:>6.1f} min)")
    print(f"  max:    {delays[-1]:>8d}s ({delays[-1] / 3600:>6.1f} h)")
    print(f"  mean:   {statistics.mean(delays):>8.0f}s ({statistics.mean(delays) / 60:>6.1f} min)")


def _print_delay_vs_pnl(resolved_pairs: list[tuple[int, float]]) -> None:
    _print_section(f"delay vs PnL/$ (resolved n={len(resolved_pairs)})")
    if not resolved_pairs:
        return
    print(f"  {'bucket':10s}  {'n':>6}  {'mean':>10}  {'median':>10}  {'win%':>6}")
    for label, lo, hi in _DELAY_BUCKETS:
        in_b = [pnl for d, pnl in resolved_pairs if lo <= d < hi]
        if not in_b:
            continue
        wins = sum(1 for p in in_b if p > 0)
        print(
            f"  {label:10s}  {len(in_b):>6d}  {_fmt_pct(statistics.mean(in_b)):>10}  "
            f"{_fmt_pct(statistics.median(in_b)):>10}  {wins / len(in_b) * 100:>5.1f}%"
        )


def _print_delay_summary(entries: list[sqlite3.Row]) -> None:
    delays, resolved_pairs = _collect_delays(entries)
    _print_delay_distribution(delays)
    _print_delay_vs_pnl(resolved_pairs)


def _flagged(
    totals: dict[str, float],
    *,
    grand_total: float,
    threshold: float,
) -> list[tuple[str, float, float]]:
    """Return ``[(key, cost, share), ...]`` for keys whose share > threshold."""
    return sorted(
        ((k, c, c / grand_total) for k, c in totals.items() if c / grand_total > threshold),
        key=lambda x: x[2],
        reverse=True,
    )


def _print_flagged_wallets(rows: list[tuple[str, float, float]]) -> None:
    if not rows:
        return
    print(f"  wallets ({len(rows)} flagged):")
    for w, c, share in rows:
        print(f"    {w}  ${c:.2f}  {share * 100:.1f}% of open")


def _print_flagged_markets(
    rows: list[tuple[str, float, float]],
    titles: dict[str, str],
) -> None:
    if not rows:
        return
    print(f"\n  markets ({len(rows)} flagged):")
    for cid, c, share in rows:
        title = titles.get(cid, "(unknown)")[:60]
        print(f"    {cid[:24]}...  ${c:.2f}  {share * 100:.1f}% of open  {title}")


def _print_concentration_warnings(
    entries: list[sqlite3.Row],
    *,
    wallet_threshold: float,
    market_threshold: float,
) -> None:
    open_entries = [r for r in entries if r["exit_cost_usd"] is None]
    total_open = sum(r["cost_usd"] for r in open_entries)
    if total_open <= 0:
        _print_section("concentration warnings")
        print("  (no open exposure)")
        return

    per_wallet: dict[str, float] = defaultdict(float)
    per_market: dict[str, float] = defaultdict(float)
    titles: dict[str, str] = {}
    for r in open_entries:
        per_wallet[r["source_wallet"] or "(none)"] += r["cost_usd"]
        per_market[r["condition_id"]] += r["cost_usd"]
        if r["title"]:
            titles[r["condition_id"]] = r["title"]

    flagged_wallets = _flagged(per_wallet, grand_total=total_open, threshold=wallet_threshold)
    flagged_markets = _flagged(per_market, grand_total=total_open, threshold=market_threshold)

    _print_section(
        f"concentration warnings (wallet>{wallet_threshold:.0%}, market>{market_threshold:.0%})"
    )
    if not flagged_wallets and not flagged_markets:
        print("  (no entries exceed thresholds)")
        return
    _print_flagged_wallets(flagged_wallets)
    _print_flagged_markets(flagged_markets, titles)


def main() -> None:
    """Run the full analysis report."""
    args = _parse_args()
    if not args.db.exists():
        msg = f"DB not found at {args.db}"
        raise SystemExit(msg)

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row

    entries = _load_entries(db)
    if not entries:
        print(f"No {_DETECTOR} entries in {args.db}.")
        return
    event_tags = _load_event_tags(db)

    _print_headline(entries)
    _print_wallet_exposure(
        entries,
        top_n=args.top_n,
        concentration_threshold=args.wallet_concentration_threshold,
    )
    _print_market_exposure(entries, top_n=args.top_n)
    _print_category_exposure(entries, event_tags)
    _print_delay_summary(entries)
    _print_concentration_warnings(
        entries,
        wallet_threshold=args.wallet_concentration_threshold,
        market_threshold=args.market_concentration_threshold,
    )


if __name__ == "__main__":
    main()
