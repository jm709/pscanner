"""Side-by-side comparison of top vs bottom paper-trading wallet cohorts.

Splits wallets into top-N and bottom-N by realized PnL (default 15 each)
after filtering to wallets with ``--min-resolved`` resolved trades
(default 10). Reports per-cohort aggregates that surface structural
differences — outcome-side bias, fill-price bands, average position
size, category mix, market overlap — so the operator can answer "what
do my winners do that my losers don't?"

Default scope: ``triggering_alert_detector = 'subgraph_copy'`` (the
primary copy-trading signal). Use ``--detector`` to inspect a different
source.

Usage::

    uv run python scripts/compare_wallet_cohorts.py [--db data/pscanner.sqlite3]
        [--cohort-size 15] [--min-resolved 10] [--detector subgraph_copy]
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pscanner.categories import primary_category

_DEFAULT_DB = Path("data/pscanner.sqlite3")
_DEFAULT_COHORT_SIZE = 15
_DEFAULT_MIN_RESOLVED = 10
_DEFAULT_DETECTOR = "subgraph_copy"
_UNCATEGORIZED = "(uncategorized)"


@dataclass
class CohortStats:
    """Aggregate metrics for one cohort of wallets."""

    wallets: set[str]
    n_entries: int = 0
    n_resolved: int = 0
    wins: int = 0
    losses: int = 0
    cost_basis: float = 0.0
    resolved_cost: float = 0.0
    pnl: float = 0.0
    fill_sum: float = 0.0
    cost_sum: float = 0.0
    yes_count: int = 0
    no_count: int = 0
    markets: set[str] = field(default_factory=set)
    categories: Counter[str] = field(default_factory=Counter)

    @property
    def win_rate(self) -> float:
        """Resolved-trade win rate."""
        return self.wins / self.n_resolved if self.n_resolved else 0.0

    @property
    def roi(self) -> float:
        """Realized PnL / resolved cost basis."""
        return self.pnl / self.resolved_cost if self.resolved_cost else 0.0

    @property
    def avg_fill(self) -> float:
        """Mean fill price across all entries."""
        return self.fill_sum / self.n_entries if self.n_entries else 0.0

    @property
    def avg_cost(self) -> float:
        """Mean $ per entry across all entries."""
        return self.cost_sum / self.n_entries if self.n_entries else 0.0

    @property
    def yes_pct(self) -> float:
        """Fraction of entries on the YES outcome (excludes non-binary)."""
        binary = self.yes_count + self.no_count
        return self.yes_count / binary if binary else 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--cohort-size", type=int, default=_DEFAULT_COHORT_SIZE)
    p.add_argument("--min-resolved", type=int, default=_DEFAULT_MIN_RESOLVED)
    p.add_argument(
        "--detector",
        default=_DEFAULT_DETECTOR,
        help="Restrict to one detector (default: subgraph_copy).",
    )
    return p.parse_args()


def _load_rows(db: sqlite3.Connection, detector: str) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT e.source_wallet,
               e.condition_id,
               e.outcome,
               e.fill_price,
               e.cost_usd,
               x.cost_usd AS exit_cost_usd,
               mc.event_slug
          FROM paper_trades e
          LEFT JOIN paper_trades x ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
          LEFT JOIN market_cache mc ON mc.condition_id = e.condition_id
         WHERE e.trade_kind = 'entry'
           AND e.triggering_alert_detector = ?
        """,
        (detector,),
    ).fetchall()


def _per_wallet_pnl(rows: list[sqlite3.Row], *, min_resolved: int) -> list[tuple[str, float]]:
    per_wallet: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "res": 0.0})
    for r in rows:
        w = r["source_wallet"] or "(none)"
        if r["exit_cost_usd"] is None:
            continue
        per_wallet[w]["res"] += 1
        per_wallet[w]["pnl"] += float(r["exit_cost_usd"]) - float(r["cost_usd"])
    eligible = [(w, b["pnl"]) for w, b in per_wallet.items() if int(b["res"]) >= min_resolved]
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    return eligible


def _build_cohort(
    rows: list[sqlite3.Row],
    wallets: set[str],
    tag_cache: dict[str, str],
) -> CohortStats:
    c = CohortStats(wallets=wallets)
    for r in rows:
        if (r["source_wallet"] or "(none)") not in wallets:
            continue
        c.n_entries += 1
        c.cost_sum += float(r["cost_usd"])
        c.fill_sum += float(r["fill_price"])
        c.markets.add(r["condition_id"])
        c.categories[tag_cache.get(r["event_slug"] or "", _UNCATEGORIZED)] += 1
        outcome = (r["outcome"] or "").lower()
        if outcome.startswith("y"):
            c.yes_count += 1
        elif outcome.startswith("n"):
            c.no_count += 1
        if r["exit_cost_usd"] is None:
            continue
        c.n_resolved += 1
        c.resolved_cost += float(r["cost_usd"])
        pnl = float(r["exit_cost_usd"]) - float(r["cost_usd"])
        c.pnl += pnl
        if pnl > 0:
            c.wins += 1
        else:
            c.losses += 1
    return c


def _load_tag_categories(db: sqlite3.Connection) -> dict[str, str]:
    """Return ``{event_slug: primary_category}`` for slugs in event_tag_cache."""
    out: dict[str, str] = {}
    for r in db.execute("SELECT event_slug, tags_json FROM event_tag_cache"):
        try:
            tags = json.loads(r["tags_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(tags, list) and tags:
            tag_strs = [str(t) for t in tags if isinstance(t, str)]
            out[r["event_slug"]] = primary_category(tag_strs).value
    return out


def _row(label: str, top: str, bot: str) -> str:
    return f"  {label:28s} {top:>16s} {bot:>16s}"


def _print_comparison(top: CohortStats, bot: CohortStats) -> None:
    print(_row("metric", "TOP", "BOTTOM"))
    print(_row("─" * 28, "─" * 16, "─" * 16))
    print(_row("wallets", str(len(top.wallets)), str(len(bot.wallets))))
    print(_row("total entries", f"{top.n_entries}", f"{bot.n_entries}"))
    print(_row("  open", f"{top.n_entries - top.n_resolved}", f"{bot.n_entries - bot.n_resolved}"))
    print(_row("  resolved", f"{top.n_resolved}", f"{bot.n_resolved}"))
    print(_row("  wins", f"{top.wins}", f"{bot.wins}"))
    print(_row("  losses", f"{top.losses}", f"{bot.losses}"))
    print(_row("win rate", f"{top.win_rate * 100:.1f}%", f"{bot.win_rate * 100:.1f}%"))
    print(_row("realized PnL", f"${top.pnl:+.2f}", f"${bot.pnl:+.2f}"))
    print(_row("resolved cost basis", f"${top.resolved_cost:.2f}", f"${bot.resolved_cost:.2f}"))
    print(_row("ROI", f"{top.roi * 100:+.1f}%", f"{bot.roi * 100:+.1f}%"))
    print(_row("avg $ / trade", f"${top.avg_cost:.3f}", f"${bot.avg_cost:.3f}"))
    print(_row("avg fill price", f"{top.avg_fill:.3f}", f"{bot.avg_fill:.3f}"))
    print(_row("YES outcome share", f"{top.yes_pct * 100:.1f}%", f"{bot.yes_pct * 100:.1f}%"))
    print(_row("distinct markets", str(len(top.markets)), str(len(bot.markets))))
    overlap = len(top.markets & bot.markets)
    union = len(top.markets | bot.markets)
    print(
        _row(
            "shared markets",
            f"{overlap} ({overlap / len(top.markets) * 100:.0f}%)" if top.markets else "0",
            f"{overlap} ({overlap / len(bot.markets) * 100:.0f}%)" if bot.markets else "0",
        ),
    )
    print(_row("market jaccard", f"{overlap / union * 100:.1f}%" if union else "0%", "(same)"))


def _print_categories(top: CohortStats, bot: CohortStats) -> None:
    print("\n  category breakdown (% of cohort's entries)")
    cats = sorted(set(top.categories) | set(bot.categories))
    print(_row("category", "TOP", "BOTTOM"))
    print(_row("─" * 28, "─" * 16, "─" * 16))
    for cat in cats:
        t_n = top.categories.get(cat, 0)
        b_n = bot.categories.get(cat, 0)
        t_pct = t_n / top.n_entries * 100 if top.n_entries else 0
        b_pct = b_n / bot.n_entries * 100 if bot.n_entries else 0
        print(_row(cat, f"{t_pct:.1f}% (n={t_n})", f"{b_pct:.1f}% (n={b_n})"))


def main() -> None:
    """Run the comparison report."""
    args = _parse_args()
    if not args.db.exists():
        msg = f"DB not found at {args.db}"
        raise SystemExit(msg)
    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row

    rows = _load_rows(db, args.detector)
    if not rows:
        print(f"No entries for detector={args.detector!r}.")
        return
    tag_cache = _load_tag_categories(db)
    ranked = _per_wallet_pnl(rows, min_resolved=args.min_resolved)
    if len(ranked) < 2 * args.cohort_size:
        print(
            f"Only {len(ranked)} wallets pass min_resolved={args.min_resolved}; "
            f"need {2 * args.cohort_size} for cohort-size={args.cohort_size}.",
        )
        return

    top_wallets = {w for w, _ in ranked[: args.cohort_size]}
    bot_wallets = {w for w, _ in ranked[-args.cohort_size :]}

    top = _build_cohort(rows, top_wallets, tag_cache)
    bot = _build_cohort(rows, bot_wallets, tag_cache)

    print(
        f"=== top {args.cohort_size} vs bottom {args.cohort_size} wallets by PnL "
        f"(detector={args.detector}, min_resolved={args.min_resolved}, "
        f"{len(ranked)} wallets eligible) ==="
    )
    _print_comparison(top, bot)
    _print_categories(top, bot)


if __name__ == "__main__":
    main()
