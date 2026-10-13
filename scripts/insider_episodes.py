"""Insider-episode catalogue: rank pre-news buyers on documented case markets.

Reads ``docs/research/insider-cases.yaml``, extracts the pre-anchor window on
each case market from the corpus, enriches buyers with data-api provenance,
scores them with the frozen scorer, compares against matched control markets,
blind-tests published wallets on holdout cases, and writes markdown reports.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""
# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from pscanner.poly.data import DataClient

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.insider_episode_lib import (  # noqa: E402  (after sys.path bootstrap)
    CoverageRow,
    MarketRef,
    MarketRow,
    Provenance,
    PublicCase,
    SealedCase,
    WalletFeatures,
    WindowTrade,
    window_features,
)

_FULL_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")


def _parse_ts(value: str) -> int:
    """Parse an ISO-8601 UTC timestamp like ``2026-01-03T06:00:00Z`` to unix seconds."""
    return int(dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _published_wallets(raw: list[dict[str, Any]]) -> tuple[str, ...]:
    """Keep only full 42-character addresses, lowercased and stripped."""
    out: list[str] = []
    for w in raw:
        addr = w.get("address")
        if not isinstance(addr, str):
            continue
        addr = addr.strip().lower()
        if _FULL_ADDRESS.match(addr):
            out.append(addr)
    return tuple(out)


def _market_refs(raw: list[dict[str, Any]]) -> tuple[MarketRef, ...]:
    return tuple(
        MarketRef(
            slug=str(m["market_slug"]),
            winning_side=str(m["winning_side"]).upper(),
            role=str(m.get("role", "")),
        )
        for m in raw
    )


def load_registry(path: Path) -> tuple[list[PublicCase], dict[str, SealedCase]]:
    """Load the case registry and split it into public and sealed records.

    Args:
        path: Path to ``insider-cases.yaml``.

    Returns:
        ``(public_cases, sealed_by_case_id)``. Only the comparator may read the
        sealed mapping.
    """
    doc = yaml.safe_load(path.read_text())
    public: list[PublicCase] = []
    sealed: dict[str, SealedCase] = {}
    for raw in doc["cases"]:
        cid = str(raw["id"])
        public.append(
            PublicCase(
                id=cid,
                name=str(raw.get("name", cid)),
                split=str(raw["split"]),
                confidence=str(raw["confidence"]),
                analyzable=bool(raw.get("analyzable", False)),
                news_ts=_parse_ts(str(raw["news_ts_utc"])),
                news_precision=str(raw.get("news_ts_precision", "day")),
                markets=_market_refs(raw.get("corpus_markets") or []),
            )
        )
        sealed[cid] = SealedCase(id=cid, wallets=_published_wallets(raw.get("wallets") or []))
    return public, sealed


def open_corpus(path: Path) -> sqlite3.Connection:
    """Open the corpus read-only. Never writes; safe against a running backfill."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_market(conn: sqlite3.Connection, ref: MarketRef) -> MarketRef:
    """Return ``ref`` with ``condition_id`` filled from ``corpus_markets``.

    Raises:
        LookupError: the slug is not in the corpus.
    """
    row = conn.execute(
        "SELECT condition_id FROM corpus_markets WHERE market_slug = ?", (ref.slug,)
    ).fetchone()
    if row is None:
        msg = f"market slug not in corpus: {ref.slug}"
        raise LookupError(msg)
    return MarketRef(
        slug=ref.slug,
        winning_side=ref.winning_side,
        role=ref.role,
        condition_id=str(row["condition_id"]),
    )


_TRADE_COLS = "wallet_address, ts, outcome_side, bs, price, notional_usd"


def _to_trades(rows: list[sqlite3.Row]) -> list[WindowTrade]:
    return [
        WindowTrade(
            wallet=str(r["wallet_address"]),
            ts=int(r["ts"]),
            outcome_side=str(r["outcome_side"]),
            bs=str(r["bs"]),
            price=float(r["price"]),
            notional=float(r["notional_usd"]),
        )
        for r in rows
    ]


def extract_window(
    conn: sqlite3.Connection, condition_id: str, anchor: int, window_hours: int
) -> list[WindowTrade]:
    """Every trade on ``condition_id`` with ``anchor - window < ts <= anchor``."""
    lo = anchor - window_hours * 3600
    rows = conn.execute(
        f"SELECT {_TRADE_COLS} FROM corpus_trades "  # noqa: S608  (constant column list)
        "WHERE condition_id = ? AND ts > ? AND ts <= ? ORDER BY ts, rowid",
        (condition_id, lo, anchor),
    ).fetchall()
    return _to_trades(rows)


def market_trades(conn: sqlite3.Connection, condition_id: str) -> list[WindowTrade]:
    """The market's whole trade history, oldest first (for the auto anchor)."""
    rows = conn.execute(
        f"SELECT {_TRADE_COLS} FROM corpus_trades WHERE condition_id = ? ORDER BY ts, rowid",  # noqa: S608
        (condition_id,),
    ).fetchall()
    return _to_trades(rows)


def candidate_wallets(trades: list[WindowTrade], winning_side: str) -> list[str]:
    """Distinct wallets with at least one BUY of ``winning_side``, by first such buy."""
    seen: dict[str, None] = {}
    for t in trades:
        if t.bs == "BUY" and t.outcome_side == winning_side and t.wallet not in seen:
            seen[t.wallet] = None
    return list(seen)


def lifetime_stats(conn: sqlite3.Connection, wallet: str, anchor: int) -> tuple[int, float, int]:
    """``(distinct markets, total notional, first trade ts)`` for ``wallet``, ``ts <= anchor``."""
    row = conn.execute(
        "SELECT count(DISTINCT condition_id), coalesce(sum(notional_usd), 0), min(ts) "
        "FROM corpus_trades WHERE wallet_address = ? AND ts <= ?",
        (wallet, anchor),
    ).fetchone()
    return int(row[0]), float(row[1]), int(row[2])


def build_features(
    conn: sqlite3.Connection,
    trades: list[WindowTrade],
    wallets: list[str],
    winning_side: str,
    anchor: int,
) -> list[WalletFeatures]:
    """Join window features with anchor-bounded lifetime stats for each candidate."""
    out: list[WalletFeatures] = []
    for w in wallets:
        n_w, n_buys, first_buy, last_buy, flip = window_features(trades, w, winning_side)
        breadth, n_life, first_ts = lifetime_stats(conn, w, anchor)
        out.append(
            WalletFeatures(
                wallet=w,
                n_w=n_w,
                n_buys=n_buys,
                first_buy_ts=first_buy,
                last_buy_ts=last_buy,
                flip=flip,
                breadth=breadth,
                n_life=n_life,
                corpus_first_ts=first_ts,
            )
        )
    return out


_ACTIVITY_PAGE = 500
_ACTIVITY_OFFSET_CAP = 10_000


class ProvenanceCache:
    """SQLite cache of data-api provenance keyed by wallet. Failed fetches are never stored."""

    def __init__(self, path: Path) -> None:
        """Open (and create) the cache at ``path``."""
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS wallet_provenance ("
            " wallet TEXT PRIMARY KEY, first_activity_ts INTEGER, n_events INTEGER,"
            " fetched_at INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get(self, wallet: str) -> Provenance | None:
        """Return the cached row for ``wallet`` or ``None``."""
        row = self._conn.execute(
            "SELECT first_activity_ts, n_events FROM wallet_provenance WHERE wallet = ?", (wallet,)
        ).fetchone()
        if row is None:
            return None
        return Provenance(wallet=wallet, first_activity_ts=row[0], n_events=row[1])

    def put(self, p: Provenance) -> None:
        """Insert or replace ``p``."""
        self._conn.execute(
            "INSERT OR REPLACE INTO wallet_provenance VALUES (?,?,?,?)",
            (p.wallet, p.first_activity_ts, p.n_events, int(time.time())),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()


async def enrich_provenance(
    cache: ProvenanceCache,
    wallets: list[str],
    fetch: Callable[[str], Awaitable[Provenance]],
) -> dict[str, Provenance]:
    """Resolve provenance for ``wallets`` through the cache, fetching misses.

    A fetch that raises ``httpx.HTTPError`` or ``ValueError`` is printed and
    skipped; the wallet is absent from the result and nothing is cached, so a
    later run retries it.
    """
    out: dict[str, Provenance] = {}
    for i, w in enumerate(wallets, 1):
        hit = cache.get(w)
        if hit is not None:
            out[w] = hit
            continue
        try:
            p = await fetch(w)
        except (httpx.HTTPError, ValueError) as exc:
            print(f"[provenance {i}/{len(wallets)}] {w} ERR {exc}")
            continue
        cache.put(p)
        out[w] = p
    return out


async def fetch_provenance_api(client: DataClient, wallet: str) -> Provenance:
    """Page ``/activity`` for ``wallet`` and return first-activity ts and event count."""
    offset = 0
    n = 0
    earliest: int | None = None
    while offset < _ACTIVITY_OFFSET_CAP:
        page = await client.get_activity(wallet, limit=_ACTIVITY_PAGE, offset=offset)
        for e in page:
            n += 1
            ts = e.get("timestamp")
            if isinstance(ts, int) and (earliest is None or ts < earliest):
                earliest = ts
        if len(page) < _ACTIVITY_PAGE:
            break
        offset += _ACTIVITY_PAGE
    return Provenance(wallet=wallet, first_activity_ts=earliest, n_events=n)


def load_market_rows(conn: sqlite3.Connection) -> list[MarketRow]:
    """Every resolved corpus market (the control universe)."""
    rows = conn.execute(
        "SELECT m.condition_id, m.market_slug, m.category, m.total_volume_usd,"
        " r.resolved_at, r.outcome_yes_won, m.truncated_at_offset_cap"
        " FROM corpus_markets m JOIN market_resolutions r ON r.condition_id = m.condition_id"
    ).fetchall()
    return [
        MarketRow(
            condition_id=str(r[0]),
            slug=str(r[1] or ""),
            category=None if r[2] is None else str(r[2]),
            volume=float(r[3]),
            resolved_at=int(r[4]),
            yes_won=int(r[5]),
            truncated=bool(r[6]),
        )
        for r in rows
    ]


def diagnose_missing(
    conn: sqlite3.Connection,
    case: PublicCase,
    wallet: str,
    anchor: int,
    window_hours: int,
    market_row: MarketRow | None,
) -> CoverageRow | None:
    """Explain why ``wallet`` is not a candidate on the primary market; ``None`` if it is."""
    primary = case.primary
    if primary is None or primary.condition_id is None or market_row is None:
        return CoverageRow(case.id, wallet, "market_below_volume_floor", "primary market absent")
    rows = conn.execute(
        "SELECT ts, outcome_side, bs FROM corpus_trades"
        " WHERE wallet_address = ? AND condition_id = ? ORDER BY ts",
        (wallet, primary.condition_id),
    ).fetchall()
    if not rows:
        reason = "backfill_truncated" if market_row.truncated else "no_trades_in_market"
        return CoverageRow(case.id, wallet, reason, "no corpus trades on the primary market")
    lo = anchor - window_hours * 3600
    in_window = [r for r in rows if lo < int(r[0]) <= anchor]
    if not in_window:
        first_h = (int(rows[0][0]) - anchor) / 3600
        last_h = (int(rows[-1][0]) - anchor) / 3600
        detail = f"first {first_h:+.1f}h / last {last_h:+.1f}h vs anchor"
        return CoverageRow(case.id, wallet, "outside_window", detail)
    if any(str(r[2]) == "BUY" and str(r[1]) == primary.winning_side for r in in_window):
        return None
    detail = f"{len(in_window)} window trades, none buying {primary.winning_side}"
    return CoverageRow(case.id, wallet, "losing_side_only", detail)
