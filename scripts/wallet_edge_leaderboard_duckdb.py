"""DuckDB-backed rewrite of wallet_edge_leaderboard.py.

Single pass over ``training_examples`` computes leaderboards for ALL
categories in one query, exploiting DuckDB's parallel GROUP BY and
``QUALIFY`` clause. Replaces the SQLite script's N-categories x
GROUP-BY-per-category pattern. The correlated subquery for
``latest_lifetime_edge`` is replaced by a single
``ROW_NUMBER() OVER (PARTITION BY wallet_address ORDER BY trade_ts DESC)``
window pass.

The corpus is read read-only via DuckDB's ``sqlite_scanner``
(``ATTACH ... TYPE sqlite``); no scratch DB is needed.

Output format matches the original script row-for-row so existing
downstream consumers (watchlist_candidates.txt generator, audit
script) keep working.

Usage::

    # Default: every category, top-N per category in one pass.
    uv run python scripts/wallet_edge_leaderboard_duckdb.py

    # Single-category mode for backward compatibility.
    uv run python scripts/wallet_edge_leaderboard_duckdb.py --category sports
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from pathlib import Path
from typing import Final

import duckdb

_KNOWN_CATEGORIES: Final[tuple[str, ...]] = (
    "sports",
    "esports",
    "thesis",
    "macro",
    "elections",
    "crypto",
    "geopolitics",
    "tech",
    "culture",
)
_SECONDS_PER_DAY: Final[int] = 86_400


def _has_platform_column(conn: sqlite3.Connection) -> bool:
    """Detect whether the corpus has the multi-platform schema."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(training_examples)")}
    return "platform" in cols


def _build_query(*, has_platform: bool, category_filter: str | None) -> str:
    """Build the single-pass leaderboard SQL.

    ``has_platform`` toggles the ``WHERE platform = ?`` filter so the
    same query works against pre- and post-RFC-#35 corpora.
    ``category_filter`` is one of:

    - ``None`` — compute leaderboards for every category in one query.
    - A specific ``Category.value`` (e.g. ``"sports"``) — restrict to
      that single category. Mirrors the original script's
      ``--category`` flag for parity testing.
    """
    platform_predicate = "WHERE platform = ?" if has_platform else "WHERE 1=1"
    cat_predicate = (
        "AND top_category = ?" if category_filter is not None else "AND top_category IS NOT NULL"
    )
    # SQLite-parity: recency filter in HAVING (wallet-eligibility), not WHERE
    # (trade-eligibility). Per-wallet stats are LIFETIME counts; recency
    # only filters out inactive wallets. See original script.
    return f"""
        WITH per_wallet_category AS (
          SELECT
            wallet_address,
            top_category,
            COUNT(*)                              AS n_resolved,
            AVG(label_won - implied_prob_at_buy)  AS edge_in_window,
            AVG(CAST(label_won AS DOUBLE))        AS win_rate,
            SUM(bet_size_usd)                     AS total_notional,
            MAX(trade_ts)                         AS last_trade_ts
          FROM s.training_examples
          {platform_predicate}
            {cat_predicate}
          GROUP BY wallet_address, top_category
          HAVING COUNT(*) >= ?
             AND MAX(trade_ts) >= ?
        ),
        wallet_lifetime AS (
          SELECT
            wallet_address,
            realized_edge_pp AS latest_lifetime_edge
          FROM (
            SELECT
              wallet_address,
              realized_edge_pp,
              ROW_NUMBER() OVER (
                PARTITION BY wallet_address
                ORDER BY trade_ts DESC
              ) AS rn
            FROM s.training_examples
            {platform_predicate}
          )
          WHERE rn = 1
        )
        SELECT
          pwc.top_category,
          pwc.wallet_address,
          pwc.n_resolved,
          pwc.edge_in_window,
          pwc.win_rate,
          pwc.total_notional,
          pwc.last_trade_ts,
          wl.latest_lifetime_edge
        FROM per_wallet_category pwc
        LEFT JOIN wallet_lifetime wl USING (wallet_address)
        QUALIFY ROW_NUMBER() OVER (
          PARTITION BY pwc.top_category
          ORDER BY pwc.edge_in_window DESC
        ) <= ?
        ORDER BY pwc.top_category, pwc.edge_in_window DESC
    """  # noqa: S608 — interpolated fragments are module constants; values bind via ?


def _bind_params(
    *,
    has_platform: bool,
    platform: str,
    category: str | None,
    cutoff_ts: int,
    min_resolved: int,
    per_category_limit: int,
) -> list[object]:
    """Match the ? placeholders in _build_query in order."""
    params: list[object] = []
    if has_platform:
        params.append(platform)  # WHERE platform = ?  (per_wallet_category)
    if category is not None:
        params.append(category)  # AND top_category = ?  (per_wallet_category)
    params.append(min_resolved)  # HAVING COUNT(*) >= ?
    params.append(cutoff_ts)  # AND MAX(trade_ts) >= ?
    if has_platform:
        params.append(platform)  # WHERE platform = ?  (wallet_lifetime inner)
    params.append(per_category_limit)  # QUALIFY ... <= ?
    return params


def _run(
    *,
    db_path: Path,
    platform: str,
    category: str | None,
    recency_days: int,
    min_resolved: int,
    per_category_limit: int,
    threads: int,
) -> tuple[list[tuple[object, ...]], list[str], float]:
    """Execute the leaderboard via DuckDB sqlite_scanner. Returns (rows, columns, elapsed)."""
    sqlite_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        has_platform = _has_platform_column(sqlite_conn)
    finally:
        sqlite_conn.close()
    cutoff_ts = int(time.time()) - recency_days * _SECONDS_PER_DAY

    duck = duckdb.connect(":memory:")
    duck.execute(f"SET threads = {threads}")
    duck.execute(f"ATTACH '{db_path}' AS s (TYPE sqlite)")
    sql = _build_query(has_platform=has_platform, category_filter=category)
    params = _bind_params(
        has_platform=has_platform,
        platform=platform,
        category=category,
        cutoff_ts=cutoff_ts,
        min_resolved=min_resolved,
        per_category_limit=per_category_limit,
    )
    t0 = time.perf_counter()
    cursor = duck.execute(sql, params)
    rows = cursor.fetchall()
    elapsed = time.perf_counter() - t0
    columns = [d[0] for d in cursor.description]
    duck.close()
    return rows, columns, elapsed


def _format_table(rows: list[tuple[object, ...]]) -> str:
    """Render the ranked rows as a plain monospaced text table, grouped by category."""
    if not rows:
        return "(no wallets matched the filter)"
    now_ts = int(time.time())
    lines: list[str] = []
    current_cat: str | None = None
    rank_in_cat = 0
    header = (
        f"{'rank':>4s} {'wallet':<44s} "
        f"{'n':>6s} {'edge':>7s} {'win':>6s} {'$total':>11s} "
        f"{'last':>5s} {'lt_edge':>8s}"
    )
    for r in rows:
        cat = str(r[0])
        wallet = str(r[1])
        n_resolved = int(r[2])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        edge = float(r[3])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        win = float(r[4])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        total_notional = float(r[5])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        last_ts = int(r[6])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        lifetime_edge = r[7]
        if cat != current_cat:
            if current_cat is not None:
                lines.append("")
            lines.append(f"=== CATEGORY: {cat} ===")
            lines.append(header)
            lines.append("-" * len(header))
            current_cat = cat
            rank_in_cat = 0
        rank_in_cat += 1
        days_ago = (now_ts - last_ts) // _SECONDS_PER_DAY
        lt_str = (
            f"{float(lifetime_edge):>+8.4f}"  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            if lifetime_edge is not None
            else f"{'-':>8s}"
        )
        lines.append(
            f"{rank_in_cat:>4d} {wallet:<44s} "
            f"{n_resolved:>6d} "
            f"{edge:>+7.4f} "
            f"{win:>6.3f} "
            f"{total_notional:>11,.0f} "
            f"{days_ago:>4d}d "
            f"{lt_str}"
        )
    return "\n".join(lines)


def _write_csv(rows: list[tuple[object, ...]], csv_path: Path) -> None:
    """Persist the rows for downstream batch consumption."""
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "top_category",
                "rank_in_category",
                "wallet_address",
                "n_resolved",
                "edge_in_window",
                "win_rate",
                "total_notional",
                "last_trade_ts",
                "latest_lifetime_edge",
            ],
        )
        current_cat: str | None = None
        rank = 0
        for r in rows:
            cat = r[0]
            if cat != current_cat:
                current_cat = str(cat)
                rank = 0
            rank += 1
            writer.writerow([cat, rank, *r[1:]])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/corpus.sqlite3"))
    parser.add_argument(
        "--platform",
        choices=["polymarket", "kalshi", "manifold"],
        default="polymarket",
    )
    parser.add_argument(
        "--category",
        choices=_KNOWN_CATEGORIES,
        default=None,
        help="Restrict to one category. Default: every category in one pass.",
    )
    parser.add_argument(
        "--min-resolved",
        type=int,
        default=20,
        help="Per-(wallet, category) min trade count (default: 20).",
    )
    parser.add_argument(
        "--recency-days",
        type=int,
        default=60,
        help="Only include trades within N days (default: 60).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Per-category top-N (default: 200).",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Entry point: parse args, run query, print + optionally CSV."""
    args = _parse_args()
    cat_label = args.category if args.category else "ALL"
    print(
        f"Querying {args.db} (platform={args.platform}) category={cat_label}, "
        f"min_resolved={args.min_resolved}, recency_days={args.recency_days}, "
        f"limit={args.limit}/category"
    )
    rows, _columns, elapsed = _run(
        db_path=args.db,
        platform=args.platform,
        category=args.category,
        recency_days=args.recency_days,
        min_resolved=args.min_resolved,
        per_category_limit=args.limit,
        threads=args.threads,
    )
    print(f"Matched {len(rows)} (wallet, category) row(s) in {elapsed:.2f}s")
    print(_format_table(rows))
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"\nCSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
