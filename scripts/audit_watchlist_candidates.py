"""Audit wallets from watchlist_candidates.txt vs their on-chain categorical PnL.

Parses the categorically-grouped wallet list in
``watchlist_candidates.txt``, samples N wallets per category, fetches
each wallet's settled positions from Polymarket's data API, and breaks
PnL + win rate down by category using ``event_tag_cache`` +
``primary_category``. The flagged category (from the file's section
header) is highlighted; a wallet is "confirmed" when its flagged
category is also its best on-chain category by realized PnL.

The script does NOT recompute the file's exact ``edge`` metric (which
relies on implied-probability-at-buy from a different data source). It
verifies the cheaper but actionable signal: is the wallet actually
profitable in the category it was flagged for?

Usage::

    uv run python scripts/audit_watchlist_candidates.py
        [--file watchlist_candidates.txt] [--per-category 5]
        [--db data/pscanner.sqlite3] [--per-wallet-limit 500]
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pscanner.categories import primary_category
from pscanner.poly.data import DataClient
from pscanner.poly.http import PolyHttpClient

_DEFAULT_FILE = Path("watchlist_candidates.txt")
_DEFAULT_DB = Path("data/pscanner.sqlite3")
_DEFAULT_PER_CATEGORY = 5
_DEFAULT_PER_WALLET_LIMIT = 500
_UNCATEGORIZED = "(uncategorized)"

_SECTION_HEADER = re.compile(r"^#\s*======\s*([A-Z]+)\b.*======", re.IGNORECASE)
_ADDR_LINE = re.compile(r"^(0x[0-9a-fA-F]{40})\s*(?:#(.*))?$")
_FIELD_PATTERNS = {
    "n": re.compile(r"\bn=\s*(\d+)"),
    "edge": re.compile(r"\bedge=\s*([+-]?[\d.]+)"),
    "win": re.compile(r"\bwin=\s*([\d.]+)"),
    "vol": re.compile(r"\$\s*([\d,]+)"),
}


@dataclass
class Candidate:
    """One row from watchlist_candidates.txt."""

    address: str
    flagged_category: str  # canonical lowercase Category.value
    n_resolved: int
    edge: float
    win_rate: float
    volume_usd: float


@dataclass
class CategoryBucket:
    """On-chain aggregate for one category."""

    n_positions: int = 0
    wins: int = 0
    cash_pnl: float = 0.0
    volume: float = 0.0

    @property
    def win_rate(self) -> float:
        """Settled-position win rate within this category."""
        return self.wins / self.n_positions if self.n_positions else 0.0


@dataclass
class WalletAudit:
    """Per-wallet on-chain category breakdown."""

    candidate: Candidate
    fetched_positions: int = 0
    buckets: dict[str, CategoryBucket] = field(
        default_factory=lambda: defaultdict(CategoryBucket),
    )

    @property
    def best_category_by_pnl(self) -> str | None:
        """Category with the largest positive cash PnL, or None if all negative/zero."""
        positive = {c: b.cash_pnl for c, b in self.buckets.items() if b.cash_pnl > 0}
        if not positive:
            return None
        return max(positive.items(), key=lambda kv: kv[1])[0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=_DEFAULT_FILE)
    p.add_argument("--per-category", type=int, default=_DEFAULT_PER_CATEGORY)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--per-wallet-limit", type=int, default=_DEFAULT_PER_WALLET_LIMIT)
    p.add_argument("--data-rpm", type=int, default=50)
    return p.parse_args()


def _parse_candidates(path: Path) -> list[Candidate]:
    """Walk the file and return one Candidate per uncommented address line."""
    out: list[Candidate] = []
    current_category: str | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        header_match = _SECTION_HEADER.match(line)
        if header_match:
            current_category = header_match.group(1).lower()
            continue
        if line.startswith("#"):
            continue
        addr_match = _ADDR_LINE.match(line)
        if not addr_match:
            continue
        if current_category is None:
            continue
        address = addr_match.group(1).lower()
        suffix = addr_match.group(2) or ""
        n_match = _FIELD_PATTERNS["n"].search(suffix)
        edge_match = _FIELD_PATTERNS["edge"].search(suffix)
        win_match = _FIELD_PATTERNS["win"].search(suffix)
        vol_match = _FIELD_PATTERNS["vol"].search(suffix)
        out.append(
            Candidate(
                address=address,
                flagged_category=current_category,
                n_resolved=int(n_match.group(1)) if n_match else 0,
                edge=float(edge_match.group(1)) if edge_match else 0.0,
                win_rate=float(win_match.group(1)) if win_match else 0.0,
                volume_usd=float(vol_match.group(1).replace(",", "")) if vol_match else 0.0,
            ),
        )
    return out


def _load_tag_categories(db: sqlite3.Connection) -> dict[str, str]:
    """Return ``{event_slug: primary_category_value}`` from event_tag_cache."""
    out: dict[str, str] = {}
    for r in db.execute("SELECT event_slug, tags_json FROM event_tag_cache"):
        try:
            tags = json.loads(r["tags_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(tags, list) and tags:
            out[r["event_slug"]] = primary_category(
                [str(t) for t in tags if isinstance(t, str)],
            ).value
    return out


def _sample_per_category(
    candidates: list[Candidate],
    *,
    per_category: int,
) -> list[Candidate]:
    """Take up to N candidates per flagged_category, ordered as they appear in the file."""
    by_cat: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_cat[c.flagged_category].append(c)
    out: list[Candidate] = []
    for cat in sorted(by_cat.keys()):
        out.extend(by_cat[cat][:per_category])
    return out


async def _audit_wallet(
    *,
    data_client: DataClient,
    candidate: Candidate,
    tag_categories: dict[str, str],
    limit: int,
) -> WalletAudit:
    audit = WalletAudit(candidate=candidate)
    try:
        positions = await data_client.get_settled_positions(candidate.address, limit=limit)
    except Exception as exc:
        print(f"  WARN: fetch failed for {candidate.address}: {exc}", flush=True)
        return audit
    audit.fetched_positions = len(positions)
    for p in positions:
        slug = str(p.event_slug) if p.event_slug else ""
        cat = tag_categories.get(slug, _UNCATEGORIZED)
        b = audit.buckets[cat]
        b.n_positions += 1
        b.cash_pnl += p.cash_pnl
        b.volume += p.avg_price * p.size
        if p.won:
            b.wins += 1
    return audit


def _print_audit(audit: WalletAudit) -> None:
    c = audit.candidate
    print(
        f"\n=== {c.address}  (flagged: {c.flagged_category}, "
        f"file n={c.n_resolved}, edge={c.edge:+.4f}, win={c.win_rate:.3f}, "
        f"vol=${c.volume_usd:,.0f}) ==="
    )
    print(
        f"  on-chain settled positions: {audit.fetched_positions}    "
        f"best category by PnL: {audit.best_category_by_pnl or '(none positive)'}",
    )
    if audit.fetched_positions == 0:
        return
    rows = sorted(audit.buckets.items(), key=lambda kv: kv[1].cash_pnl, reverse=True)
    print(
        f"  {'category':16s}  {'pos':>5}  {'wins':>5}  "
        f"{'win%':>6}  {'cash_pnl':>13}  {'volume':>12}  {'flag':>6}"
    )
    for cat, b in rows:
        flag = "  ★" if cat == c.flagged_category else ""
        print(
            f"  {cat:16s}  {b.n_positions:>5d}  {b.wins:>5d}  "
            f"{b.win_rate * 100:>5.1f}%  ${b.cash_pnl:>+12.2f}  "
            f"${b.volume:>11,.0f}  {flag:>6}"
        )


def _print_summary(audits: list[WalletAudit]) -> None:
    print("\n=== summary: flagged-category confirmation rate ===")
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "confirmed_best": 0, "confirmed_positive": 0},
    )
    for a in audits:
        c = a.candidate
        if a.fetched_positions == 0:
            continue
        by_cat[c.flagged_category]["total"] += 1
        flagged_bucket = a.buckets.get(c.flagged_category)
        if flagged_bucket and flagged_bucket.cash_pnl > 0:
            by_cat[c.flagged_category]["confirmed_positive"] += 1
        if a.best_category_by_pnl == c.flagged_category:
            by_cat[c.flagged_category]["confirmed_best"] += 1
    print(
        f"  {'flagged_category':18s}  {'sampled':>8}  "
        f"{'positive in flagged':>20}  {'flagged is best':>16}",
    )
    for cat, stats in sorted(by_cat.items()):
        total = stats["total"]
        if total == 0:
            continue
        pos = stats["confirmed_positive"]
        best = stats["confirmed_best"]
        print(
            f"  {cat:18s}  {total:>8d}  "
            f"{f'{pos}/{total} ({pos / total * 100:.0f}%)':>20s}  "
            f"{f'{best}/{total} ({best / total * 100:.0f}%)':>16s}"
        )


async def _amain() -> int:
    args = _parse_args()
    if not args.file.exists():
        print(f"watchlist file not found: {args.file}")
        return 2
    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return 2

    candidates = _parse_candidates(args.file)
    print(f"parsed {len(candidates)} candidates from {args.file}")
    sampled = _sample_per_category(candidates, per_category=args.per_category)
    print(f"sampled {len(sampled)} (≤ {args.per_category}/category)")
    print()

    db = sqlite3.connect(str(args.db))
    db.row_factory = sqlite3.Row
    tag_categories = _load_tag_categories(db)
    print(f"event_tag_cache: {len(tag_categories)} slugs available\n")

    http = PolyHttpClient(base_url="https://data-api.polymarket.com", rpm=args.data_rpm)
    data_client = DataClient(http=http)
    try:
        audits = []
        for i, c in enumerate(sampled, start=1):
            print(f"  [{i:>2}/{len(sampled)}] fetching {c.address}", flush=True)
            audit = await _audit_wallet(
                data_client=data_client,
                candidate=c,
                tag_categories=tag_categories,
                limit=args.per_wallet_limit,
            )
            audits.append(audit)
    finally:
        await http.aclose()

    for a in audits:
        _print_audit(a)
    _print_summary(audits)
    return 0


def main() -> None:
    """Run the audit pipeline."""
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
