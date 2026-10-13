"""Insider-episode catalogue: rank pre-news buyers on documented case markets.

Reads ``docs/research/insider-cases.yaml``, extracts the pre-anchor window on
each case market from the corpus, enriches buyers with data-api provenance,
scores them with the frozen scorer, compares against matched control markets,
blind-tests published wallets on holdout cases, and writes markdown reports.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""
# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import argparse
import asyncio
import csv
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
    BlindResult,
    CaseReport,
    ControlResult,
    CoverageRow,
    HoldoutLockedError,
    MarketRef,
    MarketResult,
    MarketRow,
    Provenance,
    PublicCase,
    SealedCase,
    WalletFeatures,
    WindowTrade,
    auto_anchor,
    compare_published,
    csv_rows,
    hourly_buckets,
    match_controls,
    rank_wallets,
    render_case_report,
    render_summary,
    select_cases,
    sibling_exclusions,
    window_features,
)

_FULL_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")


def _parse_ts(value: str) -> int:
    """Parse an ISO-8601 UTC timestamp like ``2026-01-03T06:00:00Z`` to unix seconds."""
    return int(dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _published_wallets(raw: list[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split registry wallets into full lowercase addresses and unpublished labels.

    A label is the truncated address text, the username, or ``unnamed``.
    """
    full: list[str] = []
    unpublished: list[str] = []
    for w in raw:
        addr = w.get("address")
        norm = addr.strip().lower() if isinstance(addr, str) else ""
        if _FULL_ADDRESS.match(norm):
            full.append(norm)
            continue
        label = addr if isinstance(addr, str) and addr.strip() else w.get("username")
        unpublished.append(str(label) if label else "unnamed")
    return tuple(full), tuple(unpublished)


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
        full, unpublished = _published_wallets(raw.get("wallets") or [])
        sealed[cid] = SealedCase(id=cid, wallets=full, unpublished=unpublished)
    return public, sealed


def open_corpus(path: Path) -> sqlite3.Connection:
    """Open the corpus read-only. Never writes; safe against a running backfill."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _has_platform(conn: sqlite3.Connection, table: str) -> bool:
    cols = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")]
    return "platform" in cols


def _plat(conn: sqlite3.Connection, table: str, alias: str = "") -> str:
    """``AND <alias.>platform = 'polymarket'`` when the table carries a platform column."""
    if not _has_platform(conn, table):
        return ""
    prefix = f"{alias}." if alias else ""
    return f" AND {prefix}platform = 'polymarket'"


def resolve_market(conn: sqlite3.Connection, ref: MarketRef) -> MarketRef:
    """Return ``ref`` with ``condition_id`` filled from ``corpus_markets``.

    Raises:
        LookupError: the slug is not in the corpus.
    """
    row = conn.execute(
        "SELECT condition_id FROM corpus_markets WHERE market_slug = ?"  # noqa: S608
        f"{_plat(conn, 'corpus_markets')}",
        (ref.slug,),
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
        f"WHERE condition_id = ? AND ts > ? AND ts <= ?{_plat(conn, 'corpus_trades')}"
        " ORDER BY ts, rowid",
        (condition_id, lo, anchor),
    ).fetchall()
    return _to_trades(rows)


def market_trades(conn: sqlite3.Connection, condition_id: str) -> list[WindowTrade]:
    """The market's whole trade history, oldest first (for the auto anchor)."""
    rows = conn.execute(
        f"SELECT {_TRADE_COLS} FROM corpus_trades WHERE condition_id = ?"  # noqa: S608
        f"{_plat(conn, 'corpus_trades')} ORDER BY ts, rowid",
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
        "SELECT count(DISTINCT condition_id), coalesce(sum(notional_usd), 0), min(ts) "  # noqa: S608
        f"FROM corpus_trades WHERE wallet_address = ? AND ts <= ?{_plat(conn, 'corpus_trades')}",
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
# Polymarket rejects /activity pagination at this offset with HTTP 400 (see poly/data.py).
_ACTIVITY_OFFSET_CAP = 3_000
_HTTP_BAD_REQUEST = 400


class ProvenanceCache:
    """SQLite cache of data-api provenance keyed by wallet. Failed fetches are never stored."""

    def __init__(self, path: Path) -> None:
        """Open (and create) the cache at ``path``."""
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS wallet_provenance ("
            " wallet TEXT PRIMARY KEY, first_activity_ts INTEGER, n_events INTEGER,"
            " truncated INTEGER NOT NULL DEFAULT 0, fetched_at INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get(self, wallet: str) -> Provenance | None:
        """Return the cached row for ``wallet`` or ``None``."""
        row = self._conn.execute(
            "SELECT first_activity_ts, n_events, truncated FROM wallet_provenance WHERE wallet = ?",
            (wallet,),
        ).fetchone()
        if row is None:
            return None
        return Provenance(
            wallet=wallet, first_activity_ts=row[0], n_events=row[1], truncated=bool(row[2])
        )

    def put(self, p: Provenance) -> None:
        """Insert or replace ``p``."""
        self._conn.execute(
            "INSERT OR REPLACE INTO wallet_provenance VALUES (?,?,?,?,?)",
            (p.wallet, p.first_activity_ts, p.n_events, int(p.truncated), int(time.time())),
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
    truncated = False
    while True:
        try:
            page = await client.get_activity(wallet, limit=_ACTIVITY_PAGE, offset=offset)
        except httpx.HTTPStatusError as exc:
            capped = exc.response.status_code == _HTTP_BAD_REQUEST
            if capped and offset >= _ACTIVITY_OFFSET_CAP:
                truncated = True
                break
            raise
        for e in page:
            n += 1
            ts = e.get("timestamp")
            if isinstance(ts, int) and (earliest is None or ts < earliest):
                earliest = ts
        if len(page) < _ACTIVITY_PAGE:
            break
        offset += _ACTIVITY_PAGE
    return Provenance(wallet=wallet, first_activity_ts=earliest, n_events=n, truncated=truncated)


def load_market_rows(conn: sqlite3.Connection) -> list[MarketRow]:
    """Every resolved corpus market (the control universe)."""
    join_plat = " AND r.platform = m.platform" if _has_platform(conn, "market_resolutions") else ""
    rows = conn.execute(
        "SELECT m.condition_id, m.market_slug, m.category, m.total_volume_usd,"  # noqa: S608
        " r.resolved_at, r.outcome_yes_won, m.truncated_at_offset_cap, m.event_slug"
        " FROM corpus_markets m JOIN market_resolutions r ON r.condition_id = m.condition_id"
        f"{join_plat} WHERE 1 = 1{_plat(conn, 'corpus_markets', 'm')}"
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
            event_slug=str(r[7] or ""),
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
        "SELECT ts, outcome_side, bs FROM corpus_trades"  # noqa: S608
        f" WHERE wallet_address = ? AND condition_id = ?{_plat(conn, 'corpus_trades')} ORDER BY ts",
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


def run_market(
    conn: sqlite3.Connection,
    ref: MarketRef,
    anchor_kind: str,
    anchor: int,
    window_hours: int,
    prov: dict[str, Provenance],
) -> MarketResult:
    """Extract, featurize, and rank one market under one anchor."""
    if ref.condition_id is None:
        msg = f"market {ref.slug} was not resolved against the corpus"
        raise ValueError(msg)
    trades = extract_window(conn, ref.condition_id, anchor, window_hours)
    wallets = candidate_wallets(trades, ref.winning_side)
    feats = build_features(conn, trades, wallets, ref.winning_side, anchor)
    return MarketResult(
        ref=ref,
        anchor_kind=anchor_kind,
        anchor_ts=anchor,
        ranked=rank_wallets(feats, prov),
        n_window_trades=len(trades),
        window_volume=sum(t.notional for t in trades),
    )


def _auto_anchor_for(
    conn: sqlite3.Connection, condition_id: str, side: str, resolved_at: int
) -> int | None:
    return auto_anchor(hourly_buckets(market_trades(conn, condition_id), side), resolved_at)


def _side(m: MarketRow) -> str:
    return "YES" if m.yes_won else "NO"


def _window_candidates(
    conn: sqlite3.Connection, condition_id: str, side: str, anchor: int, window_hours: int
) -> list[str]:
    return candidate_wallets(extract_window(conn, condition_id, anchor, window_hours), side)


ControlPlan = dict[str, list[tuple[MarketRow, int]]]


def plan_controls(
    conn: sqlite3.Connection,
    cases: list[PublicCase],
    universe: list[MarketRow],
    excluded: set[str],
    *,
    k: int,
) -> ControlPlan:
    """Match controls per case and compute each control's auto anchor exactly once."""
    rows = {m.condition_id: m for m in universe}
    anchor_memo: dict[str, int | None] = {}
    plan: ControlPlan = {}
    for case in cases:
        primary = case.primary
        prow = rows.get(primary.condition_id) if primary and primary.condition_id else None
        if prow is None:
            plan[case.id] = []
            continue
        pairs: list[tuple[MarketRow, int]] = []
        for ctl in match_controls(prow, universe, excluded, k=k):
            if ctl.condition_id not in anchor_memo:
                anchor_memo[ctl.condition_id] = _auto_anchor_for(
                    conn, ctl.condition_id, _side(ctl), ctl.resolved_at
                )
            anchor = anchor_memo[ctl.condition_id]
            if anchor is not None:
                pairs.append((ctl, anchor))
        plan[case.id] = pairs
    return plan


def collect_candidates(
    conn: sqlite3.Connection,
    cases: list[PublicCase],
    window_hours: int,
    *,
    controls: ControlPlan,
    auto_anchors: dict[str, int | None],
) -> set[str]:
    """Every wallet that scoring will touch, so provenance is fetched once up front."""
    out: set[str] = set()
    for case in cases:
        for ref in case.markets:
            if ref.condition_id is not None:
                out.update(
                    _window_candidates(
                        conn, ref.condition_id, ref.winning_side, case.news_ts, window_hours
                    )
                )
        primary = case.primary
        aa = auto_anchors.get(case.id)
        if primary is not None and primary.condition_id is not None and aa is not None:
            out.update(
                _window_candidates(
                    conn, primary.condition_id, primary.winning_side, aa, window_hours
                )
            )
        for ctl, anchor in controls.get(case.id, []):
            out.update(_window_candidates(conn, ctl.condition_id, _side(ctl), anchor, window_hours))
    return out


def _run_controls(
    conn: sqlite3.Connection,
    controls: list[tuple[MarketRow, int]],
    *,
    window_hours: int,
    prov: dict[str, Provenance],
) -> list[ControlResult]:
    out: list[ControlResult] = []
    for ctl, anchor in controls:
        ref = MarketRef(
            slug=ctl.slug, winning_side=_side(ctl), role="control", condition_id=ctl.condition_id
        )
        ranked = run_market(conn, ref, "auto", anchor, window_hours, prov).ranked
        out.append(ControlResult(ctl, anchor, ranked))
    return out


def run_case(
    conn: sqlite3.Connection,
    case: PublicCase,
    sealed: SealedCase | None,
    *,
    window_hours: int,
    controls: list[tuple[MarketRow, int]],
    prov: dict[str, Provenance],
    auto_anchor_ts: int | None,
) -> CaseReport:
    """Rank every resolved market of ``case`` and its controls; blind-test published wallets.

    Every primary-role market is compared under every anchor it was ranked
    under. Dev cases are labelled in-sample.
    """
    results = [
        run_market(conn, ref, "news", case.news_ts, window_hours, prov)
        for ref in case.markets
        if ref.condition_id is not None
    ]
    control_results: list[ControlResult] = []
    blind: list[BlindResult] = []
    primary = case.primary
    if primary is not None and primary.condition_id is not None:
        if auto_anchor_ts is not None:
            results.append(run_market(conn, primary, "auto", auto_anchor_ts, window_hours, prov))
        control_results = _run_controls(conn, controls, window_hours=window_hours, prov=prov)
    if sealed is not None and sealed.wallets:
        blind = [
            compare_published(
                case.id,
                res.anchor_kind,
                res.ranked,
                sealed,
                market_slug=res.ref.slug,
                in_sample=case.split != "holdout",
            )
            for res in results
            if res.ref.role.startswith("primary")
        ]
    return CaseReport(
        case=case,
        results=results,
        controls=control_results,
        blind=blind,
        auto_anchor_ts=auto_anchor_ts,
    )


def _registry_stakes(path: Path) -> dict[str, dict[str, float]]:
    """``{case_id: {wallet: stake_usd}}`` for wallets whose stake is recorded in the registry."""
    doc = yaml.safe_load(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for raw in doc["cases"]:
        for w in raw.get("wallets") or []:
            addr, stake = w.get("address"), w.get("stake_usd")
            if isinstance(addr, str) and isinstance(stake, (int, float)):
                out.setdefault(str(raw["id"]), {})[addr.strip().lower()] = float(stake)
    return out


def _resolve_case(conn: sqlite3.Connection, case: PublicCase) -> PublicCase:
    resolved: list[MarketRef] = []
    for ref in case.markets:
        try:
            resolved.append(resolve_market(conn, ref))
        except LookupError as exc:
            print(f"[{case.id}] {exc}; skipping market")
    return PublicCase(
        id=case.id,
        name=case.name,
        split=case.split,
        confidence=case.confidence,
        analyzable=case.analyzable,
        news_ts=case.news_ts,
        news_precision=case.news_precision,
        markets=tuple(resolved),
    )


_NOTIONAL_FLOOR_USD = 10.0


def _coverage(
    conn: sqlite3.Connection,
    cases: list[PublicCase],
    sealed: dict[str, SealedCase],
    stakes: dict[str, dict[str, float]],
    *,
    window_hours: int,
    rows: dict[str, MarketRow],
) -> list[CoverageRow]:
    out: list[CoverageRow] = []
    for case in cases:
        primary = case.primary
        prow = rows.get(primary.condition_id) if primary and primary.condition_id else None
        sealed_case = sealed.get(case.id, SealedCase(case.id, ()))
        out += [
            CoverageRow(case.id, label, "address_not_published", "no full address in sources")
            for label in sealed_case.unpublished
        ]
        for w in sealed_case.wallets:
            row = diagnose_missing(conn, case, w, case.news_ts, window_hours, prow)
            if row is None:
                continue
            stake = stakes.get(case.id, {}).get(w)
            if (
                row.reason == "no_trades_in_market"
                and stake is not None
                and stake < _NOTIONAL_FLOOR_USD
            ):
                detail = "registry stake under the $10 corpus floor (inferred)"
                row = CoverageRow(row.case_id, row.wallet, "below_notional_floor", detail)
            out.append(row)
    return out


async def _fetch_all(cache: ProvenanceCache, wallets: list[str], rpm: int) -> dict[str, Provenance]:
    client = DataClient(rpm=rpm)
    try:
        return await enrich_provenance(cache, wallets, lambda w: fetch_provenance_api(client, w))
    finally:
        await client.aclose()


def _provenance(args: argparse.Namespace, wallets: list[str]) -> dict[str, Provenance]:
    """Cache-only when ``--skip-provenance``; otherwise fetch misses through the data-api."""
    cache = ProvenanceCache(Path(args.provenance_cache))
    try:
        if args.skip_provenance:
            return {w: p for w in wallets if (p := cache.get(w)) is not None}
        return asyncio.run(_fetch_all(cache, wallets, args.provenance_rpm))
    finally:
        cache.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", default="data/corpus.sqlite3")
    p.add_argument("--registry", default="docs/research/insider-cases.yaml")
    p.add_argument("--out-dir", default="docs/research/insider-episodes")
    p.add_argument("--window-hours", type=int, default=72)
    p.add_argument("--controls-per-case", type=int, default=5)
    p.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    p.add_argument(
        "--scorer-frozen", action="store_true", help="attest spec Amendment 1 is signed off"
    )
    p.add_argument("--skip-provenance", action="store_true", help="use cached provenance only")
    p.add_argument("--provenance-cache", default="data/insider_provenance.sqlite3")
    p.add_argument("--provenance-rpm", type=int, default=60)
    return p.parse_args(argv)


def _warn_late_anchors(cases: list[PublicCase], rows: dict[str, MarketRow]) -> None:
    """Print a warning for any case whose news anchor falls after its primary resolved."""
    for c in cases:
        p = c.primary
        if p is None or p.condition_id is None or p.condition_id not in rows:
            continue
        resolved = rows[p.condition_id].resolved_at
        if c.news_ts > resolved:
            hours = (c.news_ts - resolved) / 3600
            print(
                f"[{c.id}] WARNING: news anchor is {hours:.1f}h after resolution of {p.slug};"
                " the pre-news window is empty"
            )


def _auto_anchors(
    conn: sqlite3.Connection, cases: list[PublicCase], rows: dict[str, MarketRow]
) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for c in cases:
        p = c.primary
        if p is None or p.condition_id is None or p.condition_id not in rows:
            continue
        out[c.id] = _auto_anchor_for(
            conn, p.condition_id, p.winning_side, rows[p.condition_id].resolved_at
        )
    return out


def _write_outputs(out_dir: Path, reports: list[CaseReport], coverage: list[CoverageRow]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in reports:
        (out_dir / f"{r.case.id}.md").write_text(render_case_report(r))
        prim = [
            x for x in r.results if x.ref.role.startswith("primary") and x.anchor_kind == "news"
        ]
        n = len(prim[0].ranked) if prim else 0
        best = [b.best_rank for b in r.blind]
        print(f"[{r.case.id}] {n} candidates, {len(r.controls)} controls, blind={best}")
    (out_dir / "SUMMARY.md").write_text(render_summary(reports, coverage))
    rows_csv = csv_rows(reports)
    with (out_dir / "scored_wallets.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0]) if rows_csv else ["case_id"])
        writer.writeheader()
        writer.writerows(rows_csv)


def _run(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    public: list[PublicCase],
    selected: list[PublicCase],
    sealed: dict[str, SealedCase],
) -> tuple[list[CaseReport], list[CoverageRow]]:
    cases = [_resolve_case(conn, c) for c in selected if c.analyzable]
    universe = load_market_rows(conn)
    rows = {m.condition_id: m for m in universe}
    registry_cids = {
        m.condition_id
        for c in (_resolve_case(conn, x) for x in public if x.analyzable)
        for m in c.markets
        if m.condition_id
    }
    excluded = sibling_exclusions(universe, registry_cids)
    _warn_late_anchors(cases, rows)
    auto_anchors = _auto_anchors(conn, cases, rows)
    controls = plan_controls(conn, cases, universe, excluded, k=args.controls_per_case)
    wanted = sorted(
        collect_candidates(
            conn, cases, args.window_hours, controls=controls, auto_anchors=auto_anchors
        )
    )
    print(f"{len(cases)} cases, {len(wanted)} wallets need provenance")
    prov = _provenance(args, wanted)
    reports = [
        run_case(
            conn,
            c,
            sealed.get(c.id),
            window_hours=args.window_hours,
            controls=controls.get(c.id, []),
            prov=prov,
            auto_anchor_ts=auto_anchors.get(c.id),
        )
        for c in cases
    ]
    coverage = _coverage(
        conn,
        [_resolve_case(conn, c) for c in selected],
        sealed,
        _registry_stakes(Path(args.registry)),
        window_hours=args.window_hours,
        rows=rows,
    )
    return reports, coverage


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    public, sealed = load_registry(Path(args.registry))
    try:
        selected = select_cases(public, args.split, scorer_frozen=args.scorer_frozen)
    except HoldoutLockedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    conn = open_corpus(Path(args.db))
    try:
        reports, coverage = _run(conn, args, public, selected, sealed)
    finally:
        conn.close()
    _write_outputs(Path(args.out_dir), reports, coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
