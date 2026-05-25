"""Per-wallet trade/PnL report across all paper-trading detectors.

For each ``source_wallet`` in ``paper_trades``, aggregate:

- number of entries (total / open / resolved)
- wins / losses (on the resolved subset)
- realized PnL ($)
- win rate, ROI on resolved cost basis
- first/last entry timestamps
- which detectors contributed (e.g. ``subgraph_copy``, ``gate_buy``)

Sorted by total trade count by default. Use ``--by`` to re-order:
``trades`` (default), ``pnl``, ``roi``, ``wins``, ``losses``, ``win_rate``.

Usage::

    uv run python scripts/analyze_wallets.py [--db data/pscanner.sqlite3]
        [--top-n 50] [--by trades] [--min-resolved 1]
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_DB = Path("data/pscanner.sqlite3")
_DEFAULT_TOP_N = 50
_NONE_WALLET = "(no wallet)"


@dataclass
class WalletBucket:
    """Per-wallet accumulator for the report."""

    n_entries: int = 0
    n_open: int = 0
    n_resolved: int = 0
    wins: int = 0
    losses: int = 0
    total_cost: float = 0.0
    resolved_cost: float = 0.0
    pnl: float = 0.0
    first_ts: int | None = None
    last_ts: int | None = None
    detectors: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def win_rate(self) -> float:
        """Fraction of resolved entries that booked positive PnL."""
        return (self.wins / self.n_resolved) if self.n_resolved else 0.0

    @property
    def roi(self) -> float:
        """Realized PnL divided by resolved cost basis."""
        return (self.pnl / self.resolved_cost) if self.resolved_cost else 0.0


_SORT_KEYS: dict[str, Callable[[WalletBucket], float]] = {
    "trades": lambda b: float(b.n_entries),
    "pnl": lambda b: b.pnl,
    "roi": lambda b: b.roi,
    "wins": lambda b: float(b.wins),
    "losses": lambda b: float(b.losses),
    "win_rate": lambda b: b.win_rate,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB, help="Daemon SQLite DB path.")
    p.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N, help="Top-N rows to print.")
    p.add_argument(
        "--by",
        choices=sorted(_SORT_KEYS.keys()),
        default="trades",
        help="Sort key for the top-N table (default: trades).",
    )
    p.add_argument(
        "--min-resolved",
        type=int,
        default=0,
        help="Filter wallets with fewer than N resolved trades (default: 0).",
    )
    return p.parse_args()


def _load_rows(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """One row per paper_trades entry, joined to its exit (if any).

    Columns: source_wallet, detector, cost_usd, ts, exit_cost_usd.
    """
    return db.execute(
        """
        SELECT e.source_wallet,
               e.triggering_alert_detector AS detector,
               e.cost_usd,
               e.ts,
               x.cost_usd AS exit_cost_usd
          FROM paper_trades e
          LEFT JOIN paper_trades x ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
         WHERE e.trade_kind = 'entry'
        """,
    ).fetchall()


def _aggregate(rows: list[sqlite3.Row]) -> dict[str, WalletBucket]:
    """Build the per-wallet aggregate."""
    per_wallet: dict[str, WalletBucket] = defaultdict(WalletBucket)
    for r in rows:
        wallet = r["source_wallet"] or _NONE_WALLET
        b = per_wallet[wallet]
        b.n_entries += 1
        cost = float(r["cost_usd"])
        b.total_cost += cost
        ts = int(r["ts"])
        if b.first_ts is None or ts < b.first_ts:
            b.first_ts = ts
        if b.last_ts is None or ts > b.last_ts:
            b.last_ts = ts
        det = r["detector"] or "(none)"
        b.detectors[det] += 1
        exit_cost = r["exit_cost_usd"]
        if exit_cost is None:
            b.n_open += 1
            continue
        b.n_resolved += 1
        b.resolved_cost += cost
        pnl = float(exit_cost) - cost
        b.pnl += pnl
        if pnl > 0:
            b.wins += 1
        else:
            b.losses += 1
    return per_wallet


def _fmt_detectors(detectors: dict[str, int]) -> str:
    if len(detectors) == 1:
        return next(iter(detectors.keys()))
    pairs = sorted(detectors.items(), key=lambda kv: kv[1], reverse=True)
    return ",".join(f"{d}:{n}" for d, n in pairs)


def _fmt_ts(ts: int | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _print_headline(per_wallet: dict[str, WalletBucket]) -> None:
    n_wallets = len(per_wallet)
    n_entries = sum(b.n_entries for b in per_wallet.values())
    n_open = sum(b.n_open for b in per_wallet.values())
    n_resolved = sum(b.n_resolved for b in per_wallet.values())
    wins = sum(b.wins for b in per_wallet.values())
    losses = sum(b.losses for b in per_wallet.values())
    cost_sum = sum(b.total_cost for b in per_wallet.values())
    resolved_cost_sum = sum(b.resolved_cost for b in per_wallet.values())
    pnl_sum = sum(b.pnl for b in per_wallet.values())
    print(f"distinct wallets:     {n_wallets:>8}")
    print(f"total entries:        {n_entries:>8}")
    print(f"  open:               {n_open:>8}")
    print(f"  resolved:           {n_resolved:>8}    wins: {wins}   losses: {losses}")
    print(f"total cost basis:     ${cost_sum:>13.2f}")
    print(f"resolved cost basis:  ${resolved_cost_sum:>13.2f}")
    print(f"realized PnL:         ${pnl_sum:>+13.2f}")
    if resolved_cost_sum:
        print(f"realized ROI:         {pnl_sum / resolved_cost_sum * 100:>+13.1f}%")
    if n_resolved:
        print(f"aggregate win rate:   {wins / n_resolved * 100:>13.1f}%")


def _print_table(
    per_wallet: dict[str, WalletBucket],
    *,
    top_n: int,
    sort_by: str,
    min_resolved: int,
) -> None:
    sort_key = _SORT_KEYS[sort_by]
    rows = [(w, b) for w, b in per_wallet.items() if b.n_resolved >= min_resolved]
    rows.sort(key=lambda kv: sort_key(kv[1]), reverse=True)

    print(
        f"\ntop {min(top_n, len(rows))} wallets by {sort_by} "
        f"(filter: min_resolved={min_resolved}, {len(rows)} eligible)\n"
    )
    print(
        f"  {'wallet':44s}  {'tot':>5}  {'open':>5}  {'res':>5}  "
        f"{'won':>5}  {'lost':>5}  {'win%':>6}  {'cost$':>10}  "
        f"{'pnl$':>10}  {'roi':>7}  {'detectors':12}  {'last_seen':10}"
    )
    for wallet, b in rows[:top_n]:
        dets = _fmt_detectors(b.detectors)
        last = _fmt_ts(b.last_ts)
        print(
            f"  {wallet[:44]:44s}  {b.n_entries:>5d}  {b.n_open:>5d}  {b.n_resolved:>5d}  "
            f"{b.wins:>5d}  {b.losses:>5d}  {b.win_rate * 100:>5.1f}%  "
            f"${b.total_cost:>9.2f}  ${b.pnl:>+9.2f}  {b.roi * 100:>+6.1f}%  "
            f"{dets[:12]:12s}  {last}"
        )


def main() -> None:
    """Run the report."""
    args = _parse_args()
    if not args.db.exists():
        msg = f"DB not found at {args.db}"
        raise SystemExit(msg)

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row

    rows = _load_rows(db)
    if not rows:
        print("No paper_trades entries.")
        return
    per_wallet = _aggregate(rows)

    print("=== per-wallet paper-trade summary ===")
    _print_headline(per_wallet)
    _print_table(
        per_wallet,
        top_n=args.top_n,
        sort_by=args.by,
        min_resolved=args.min_resolved,
    )


if __name__ == "__main__":
    main()
