"""Check Polymarket on-chain PnL for the paper-trade top/bottom wallet cohorts.

Selects the same cohorts as ``compare_wallet_cohorts.py`` (top-N and
bottom-N by paper-trade realized PnL) and queries Polymarket's data
API for each wallet's settled-position history. Surfaces whether the
paper-trade winners and losers are also winners and losers on-chain,
or whether the paper sim is mis-classifying them.

Per cohort:

- Aggregate on-chain cash PnL, settled-position count, win count.
- Median per-wallet PnL / position count / win rate.
- Per-wallet detail rows.

Bounded by 2 * cohort_size network calls (default 30). Uses gamma-style
RPM throttling via :class:`PolyHttpClient`.

Usage::

    uv run python scripts/check_polymarket_wallet_pnl.py [--db data/pscanner.sqlite3]
        [--cohort-size 15] [--min-resolved 10] [--detector subgraph_copy]
        [--per-wallet-limit 500] [--data-rpm 50]
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pscanner.poly.data import DataClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import ClosedPosition

_DEFAULT_DB = Path("data/pscanner.sqlite3")
_DEFAULT_COHORT_SIZE = 15
_DEFAULT_MIN_RESOLVED = 10
_DEFAULT_DETECTOR = "subgraph_copy"
_DEFAULT_PER_WALLET_LIMIT = 500


@dataclass
class WalletOnchain:
    """Aggregated on-chain stats for one wallet."""

    address: str
    n_positions: int
    wins: int
    cash_pnl: float
    total_volume: float  # sum of (avg_price * size) across positions

    @property
    def win_rate(self) -> float:
        """Fraction of settled positions that won."""
        return self.wins / self.n_positions if self.n_positions else 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--cohort-size", type=int, default=_DEFAULT_COHORT_SIZE)
    p.add_argument("--min-resolved", type=int, default=_DEFAULT_MIN_RESOLVED)
    p.add_argument("--detector", default=_DEFAULT_DETECTOR)
    p.add_argument("--per-wallet-limit", type=int, default=_DEFAULT_PER_WALLET_LIMIT)
    p.add_argument("--data-rpm", type=int, default=50)
    return p.parse_args()


def _rank_wallets_by_paper_pnl(
    db: sqlite3.Connection,
    *,
    detector: str,
    min_resolved: int,
) -> list[tuple[str, float]]:
    """Return ``[(wallet, paper_pnl), ...]`` sorted best-first, filtered by min_resolved."""
    per_wallet: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "res": 0.0})
    rows = db.execute(
        """
        SELECT e.source_wallet,
               e.cost_usd,
               x.cost_usd AS exit_cost_usd
          FROM paper_trades e
          LEFT JOIN paper_trades x ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
         WHERE e.trade_kind = 'entry'
           AND e.triggering_alert_detector = ?
        """,
        (detector,),
    ).fetchall()
    for r in rows:
        if r["exit_cost_usd"] is None or r["source_wallet"] is None:
            continue
        per_wallet[r["source_wallet"]]["res"] += 1
        per_wallet[r["source_wallet"]]["pnl"] += float(r["exit_cost_usd"]) - float(r["cost_usd"])
    eligible = [(w, b["pnl"]) for w, b in per_wallet.items() if int(b["res"]) >= min_resolved]
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    return eligible


async def _fetch_wallet(
    *,
    data_client: DataClient,
    address: str,
    limit: int,
) -> WalletOnchain:
    """Fetch one wallet's settled positions and aggregate. Returns zeroed on error.

    ``limit`` is in positions; we convert to ``max_pages`` for
    ``get_settled_positions`` (server caps at 50 rows/page).
    """
    max_pages = max(1, limit // 50)
    try:
        positions: list[ClosedPosition] = await data_client.get_settled_positions(
            address,
            max_pages=max_pages,
        )
    except Exception as exc:
        print(f"  WARN: fetch failed for {address}: {exc}", flush=True)
        return WalletOnchain(address=address, n_positions=0, wins=0, cash_pnl=0.0, total_volume=0.0)
    cash_pnl = sum(p.cash_pnl for p in positions)
    wins = sum(1 for p in positions if p.won)
    volume = sum(p.avg_price * p.size for p in positions)
    return WalletOnchain(
        address=address,
        n_positions=len(positions),
        wins=wins,
        cash_pnl=cash_pnl,
        total_volume=volume,
    )


async def _fetch_cohort(
    *,
    data_client: DataClient,
    addresses: list[str],
    limit: int,
    label: str,
) -> list[WalletOnchain]:
    out: list[WalletOnchain] = []
    for i, addr in enumerate(addresses, start=1):
        info = await _fetch_wallet(data_client=data_client, address=addr, limit=limit)
        print(
            f"  {label} {i}/{len(addresses)}  {addr}  "
            f"positions={info.n_positions}  cash_pnl=${info.cash_pnl:+.2f}  "
            f"wins={info.wins}",
            flush=True,
        )
        out.append(info)
    return out


def _summary_line(label: str, top: str, bot: str) -> str:
    return f"  {label:30s} {top:>18s} {bot:>18s}"


def _print_cohort_summary(top: list[WalletOnchain], bot: list[WalletOnchain]) -> None:
    print()
    print(_summary_line("metric", "TOP cohort", "BOTTOM cohort"))
    print(_summary_line("─" * 30, "─" * 18, "─" * 18))
    for label, top_v, bot_v in (
        ("wallets", str(len(top)), str(len(bot))),
        (
            "total settled positions",
            str(sum(w.n_positions for w in top)),
            str(sum(w.n_positions for w in bot)),
        ),
        ("total wins", str(sum(w.wins for w in top)), str(sum(w.wins for w in bot))),
        (
            "total cash PnL",
            f"${sum(w.cash_pnl for w in top):+.2f}",
            f"${sum(w.cash_pnl for w in bot):+.2f}",
        ),
        (
            "total volume",
            f"${sum(w.total_volume for w in top):,.0f}",
            f"${sum(w.total_volume for w in bot):,.0f}",
        ),
        (
            "aggregate win rate",
            (
                f"{sum(w.wins for w in top) / sum(w.n_positions for w in top) * 100:.1f}%"
                if sum(w.n_positions for w in top)
                else "n/a"
            ),
            (
                f"{sum(w.wins for w in bot) / sum(w.n_positions for w in bot) * 100:.1f}%"
                if sum(w.n_positions for w in bot)
                else "n/a"
            ),
        ),
        (
            "median wallet PnL",
            f"${statistics.median(w.cash_pnl for w in top):+.2f}",
            f"${statistics.median(w.cash_pnl for w in bot):+.2f}",
        ),
        (
            "median positions/wallet",
            f"{statistics.median(w.n_positions for w in top):.0f}",
            f"{statistics.median(w.n_positions for w in bot):.0f}",
        ),
        (
            "median win rate",
            f"{statistics.median(w.win_rate for w in top) * 100:.1f}%",
            f"{statistics.median(w.win_rate for w in bot) * 100:.1f}%",
        ),
    ):
        print(_summary_line(label, top_v, bot_v))


def _print_per_wallet(label: str, cohort: list[WalletOnchain]) -> None:
    print(f"\n  {label} cohort detail:")
    print(
        f"  {'wallet':44s}  {'positions':>10}  {'wins':>5}  "
        f"{'win%':>6}  {'cash_pnl':>14}  {'volume':>14}"
    )
    for w in sorted(cohort, key=lambda x: x.cash_pnl, reverse=True):
        print(
            f"  {w.address:44s}  {w.n_positions:>10d}  {w.wins:>5d}  "
            f"{w.win_rate * 100:>5.1f}%  ${w.cash_pnl:>+13.2f}  ${w.total_volume:>13,.0f}"
        )


async def _amain() -> int:
    args = _parse_args()
    if not args.db.exists():
        print(f"DB not found at {args.db}")
        return 2

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row
    ranked = _rank_wallets_by_paper_pnl(
        db,
        detector=args.detector,
        min_resolved=args.min_resolved,
    )
    if len(ranked) < 2 * args.cohort_size:
        print(
            f"Only {len(ranked)} wallets pass min_resolved={args.min_resolved}; "
            f"need {2 * args.cohort_size}.",
        )
        return 1

    top_addrs = [w for w, _ in ranked[: args.cohort_size]]
    bot_addrs = [w for w, _ in ranked[-args.cohort_size :]]

    print(
        f"=== fetching on-chain PnL for top {args.cohort_size} + "
        f"bottom {args.cohort_size} wallets ===",
    )
    print(
        f"detector={args.detector}  min_resolved={args.min_resolved}  "
        f"data_rpm={args.data_rpm}  per_wallet_limit={args.per_wallet_limit}",
    )
    print()

    http = PolyHttpClient(base_url="https://data-api.polymarket.com", rpm=args.data_rpm)
    data_client = DataClient(http=http)
    try:
        top_info = await _fetch_cohort(
            data_client=data_client,
            addresses=top_addrs,
            limit=args.per_wallet_limit,
            label="TOP",
        )
        bot_info = await _fetch_cohort(
            data_client=data_client,
            addresses=bot_addrs,
            limit=args.per_wallet_limit,
            label="BOT",
        )
    finally:
        await http.aclose()

    _print_cohort_summary(top_info, bot_info)
    _print_per_wallet("TOP", top_info)
    _print_per_wallet("BOTTOM", bot_info)
    return 0


def main() -> None:
    """Run the on-chain PnL comparison."""
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
