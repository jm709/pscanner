"""Regenerate watchlist_candidates.txt from the DuckDB wallet edge leaderboard.

Mirrors the structure of the 2026-05-20 file: per-category sections with
section headers, a "Dropped" line per section listing the rank numbers
that were filtered out, and one wallet per line with a trailing comment
that carries the leaderboard's per-wallet metrics.

Categories + per-category limits (matching the original file's shape):

* ESPORTS  — top 5
* SPORTS   — split into ranks 1-100 and 101-260 (single query, top 260)
* CRYPTO   — top 124  (matches original file count; was top 200, hand-curated)
* ELECTIONS — top 81
* MACRO    — top 29
* GEOPOLITICS — top 60
* TECH     — top 14
* CULTURE  — top 20

Per-row drop filter (relaxed, matching the original file's stated rules):

* ``lt_edge < -0.30`` — extreme historical loss anomaly.

The original file also dropped on ``SELL ratio >= 70%`` (market-makers)
and ``top_event >= 85%`` (single-event one-shots) via a per-wallet
``/activity`` spot-check. Those are NOT automated here — they require
network calls per wallet. Run separately as a follow-up audit if you
want full parity with the curated 2026-05-20 cut.

Usage::

    uv run python scripts/regen_watchlist_candidates.py
        [--db data/corpus.sqlite3] [--out watchlist_candidates.txt]
        [--lt-edge-floor -0.30] [--recency-days 60] [--min-resolved 20]
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

_DEFAULT_DB: Final[Path] = Path("data/corpus.sqlite3")
_DEFAULT_OUT: Final[Path] = Path("watchlist_candidates.txt")
_DEFAULT_LT_EDGE_FLOOR: Final[float] = -0.30
_DEFAULT_RECENCY_DAYS: Final[int] = 60
_DEFAULT_MIN_RESOLVED: Final[int] = 20
_SECONDS_PER_DAY: Final[int] = 86_400
_RANK_COMMENT_THRESHOLD: Final[int] = 5  # sections > N wallets render with rank= prefix


@dataclass(frozen=True)
class Section:
    """One categorical section of the candidates file."""

    label: str  # rendered header, e.g. "SPORTS ranks 1-100"
    category: str  # canonical category for the leaderboard query
    rank_start: int  # 1-indexed
    rank_end: int  # inclusive


_SECTIONS: Final[tuple[Section, ...]] = (
    Section(label="ESPORTS", category="esports", rank_start=1, rank_end=5),
    Section(label="SPORTS ranks 1-100", category="sports", rank_start=1, rank_end=100),
    Section(label="SPORTS ranks 101-260", category="sports", rank_start=101, rank_end=260),
    Section(label="CRYPTO", category="crypto", rank_start=1, rank_end=124),
    Section(label="ELECTIONS", category="elections", rank_start=1, rank_end=81),
    Section(label="MACRO", category="macro", rank_start=1, rank_end=29),
    Section(label="GEOPOLITICS", category="geopolitics", rank_start=1, rank_end=60),
    Section(label="TECH", category="tech", rank_start=1, rank_end=14),
    Section(label="CULTURE", category="culture", rank_start=1, rank_end=20),
)


@dataclass
class Row:
    """One leaderboard row used to emit a wallet line."""

    rank: int
    wallet: str
    n_resolved: int
    edge: float
    win_rate: float
    total_notional: float
    last_trade_ts: int
    lifetime_edge: float | None

    @property
    def is_dropped(self) -> bool:
        """Will be set by the filter pass."""
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--lt-edge-floor", type=float, default=_DEFAULT_LT_EDGE_FLOOR)
    p.add_argument("--recency-days", type=int, default=_DEFAULT_RECENCY_DAYS)
    p.add_argument("--min-resolved", type=int, default=_DEFAULT_MIN_RESOLVED)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def _has_platform_column(db_path: Path) -> bool:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(training_examples)")}
        return "platform" in cols
    finally:
        conn.close()


def _run_category_query(
    duck: duckdb.DuckDBPyConnection,
    *,
    category: str,
    limit: int,
    cutoff_ts: int,
    min_resolved: int,
    has_platform: bool,
) -> list[Row]:
    """Run one category's leaderboard query and return ranked rows."""
    platform_predicate = "WHERE platform = ?" if has_platform else "WHERE 1=1"
    sql = f"""
        WITH per_wallet AS (
          SELECT
            wallet_address,
            COUNT(*)                              AS n_resolved,
            AVG(label_won - implied_prob_at_buy)  AS edge_in_window,
            AVG(CAST(label_won AS DOUBLE))        AS win_rate,
            SUM(bet_size_usd)                     AS total_notional,
            MAX(trade_ts)                         AS last_trade_ts
          FROM s.training_examples
          {platform_predicate}
            AND top_category = ?
          GROUP BY wallet_address
          HAVING COUNT(*) >= ?
             AND MAX(trade_ts) >= ?
        )
        SELECT
          pw.wallet_address,
          pw.n_resolved,
          pw.edge_in_window,
          pw.win_rate,
          pw.total_notional,
          pw.last_trade_ts,
          (SELECT realized_edge_pp
             FROM s.training_examples te
            WHERE te.wallet_address = pw.wallet_address
              {"AND te.platform = ?" if has_platform else ""}
            ORDER BY te.trade_ts DESC
            LIMIT 1)                              AS latest_lifetime_edge
        FROM per_wallet pw
        ORDER BY pw.edge_in_window DESC
        LIMIT ?
    """  # noqa: S608 — only category/limit/platform interpolated; values bind via ?
    params: list[object] = []
    if has_platform:
        params.append("polymarket")
    params.append(category)
    params.append(min_resolved)
    params.append(cutoff_ts)
    if has_platform:
        params.append("polymarket")
    params.append(limit)
    rows = duck.execute(sql, params).fetchall()
    return [
        Row(
            rank=i + 1,
            wallet=str(r[0]),
            n_resolved=int(r[1]),  # type: ignore[arg-type]
            edge=float(r[2]),  # type: ignore[arg-type]
            win_rate=float(r[3]),  # type: ignore[arg-type]
            total_notional=float(r[4]),  # type: ignore[arg-type]
            last_trade_ts=int(r[5]),  # type: ignore[arg-type]
            lifetime_edge=float(r[6]) if r[6] is not None else None,  # type: ignore[arg-type]
        )
        for i, r in enumerate(rows)
    ]


def _format_lt_edge(v: float | None) -> str:
    return f"{v:+.4f}" if v is not None else "    -   "


def _format_row(r: Row, *, with_rank: bool) -> str:
    rank_str = f"rank={r.rank:>3d} " if with_rank else ""
    lt = _format_lt_edge(r.lifetime_edge)
    return (
        f"{r.wallet}  # {rank_str}"
        f"n={r.n_resolved:>3d} edge={r.edge:+.4f} win={r.win_rate:.3f} "
        f"${r.total_notional:>10,.0f} lt={lt}"
    )


def _render_section(section: Section, rows: list[Row], *, lt_edge_floor: float) -> list[str]:
    """Return the lines for one section, with drop annotations."""
    sliced = [r for r in rows if section.rank_start <= r.rank <= section.rank_end]
    kept: list[Row] = []
    dropped_ranks: list[int] = []
    for r in sliced:
        if r.lifetime_edge is not None and r.lifetime_edge <= lt_edge_floor:
            dropped_ranks.append(r.rank)
            continue
        kept.append(r)
    header = f"# ====== {section.label} ({len(kept)} wallets) ======"
    lines = [header]
    if dropped_ranks:
        ranks_str = " ".join(f"#{n}" for n in dropped_ranks)
        lines.append(f"# Dropped ({len(dropped_ranks)}): {ranks_str}")
    lines.append("#")
    # ESPORTS / non-tiered sections render WITHOUT rank=, matching the original file.
    with_rank = section.rank_end > _RANK_COMMENT_THRESHOLD
    for r in kept:
        lines.append(_format_row(r, with_rank=with_rank))
    lines.append("#")
    return lines


def main() -> int:
    """Run the regenerator. Writes to ``--out``."""
    args = _parse_args()
    if not args.db.exists():
        print(f"corpus DB not found: {args.db}")
        return 2

    has_platform = _has_platform_column(args.db)
    cutoff_ts = int(time.time()) - args.recency_days * _SECONDS_PER_DAY

    duck = duckdb.connect(":memory:")
    duck.execute(f"SET threads = {args.threads}")
    duck.execute(f"ATTACH '{args.db}' AS s (TYPE sqlite)")

    # Cache per-category queries — sports is needed for ranks 1-260, so the
    # single sports query covers both sports sections.
    category_cache: dict[str, list[Row]] = {}
    for section in _SECTIONS:
        if section.category in category_cache:
            continue
        category_cache[section.category] = _run_category_query(
            duck,
            category=section.category,
            limit=max(s.rank_end for s in _SECTIONS if s.category == section.category),
            cutoff_ts=cutoff_ts,
            min_resolved=args.min_resolved,
            has_platform=has_platform,
        )
    duck.close()

    today = time.strftime("%Y-%m-%d", time.gmtime())
    output_lines: list[str] = [
        "# Wallet edge leaderboard candidates - validated subset",
        f"# Generated: {today}",
        "# Source: scripts/regen_watchlist_candidates.py (DuckDB wallet_edge_leaderboard)",
        "#",
        f"# Drop criteria (automated subset of 2026-05-20 cut, lt_edge < {args.lt_edge_floor}):",
        f"#   * lt_edge <= {args.lt_edge_floor:.2f}      (extreme historical loss anomaly)",
        "# NOT auto-applied (requires /activity spot-check):",
        "#   * SELL ratio >= 70%      (true market-makers)",
        "#   * top_event >= 85%       (single-event one-shots)",
        "# Run a per-wallet /activity audit to apply the SELL/top_event filters.",
        "#",
        f"# Add via: pscanner watch <address> --reason wallet-edge-leaderboard-{today[:7]}",
        "#",
    ]
    for section in _SECTIONS:
        output_lines.extend(
            _render_section(
                section,
                category_cache[section.category],
                lt_edge_floor=args.lt_edge_floor,
            )
        )
    output_lines.append("")  # trailing newline

    args.out.write_text("\n".join(output_lines))
    kept_total = sum(
        1
        for s in _SECTIONS
        for r in category_cache[s.category]
        if s.rank_start <= r.rank <= s.rank_end
        and (r.lifetime_edge is None or r.lifetime_edge > args.lt_edge_floor)
    )
    print(f"wrote {kept_total} wallets across {len(_SECTIONS)} sections to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
