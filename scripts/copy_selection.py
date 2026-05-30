"""Causal copy-trading wallet selection precompute (DuckDB-backed).

Qualifies wallets at >= min_resolved resolved trades, ranks by causal
(no-lookahead) edge, freezes a global top-K copy set per rebalance
boundary, and emits the selected trades + their resolutions as event
rows for scripts.backtest_copy_sizing's Simulator.

Spec: docs/superpowers/specs/2026-05-30-causal-copy-selection-design.md
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class KPolicy:
    """Top-K policy for the copy set. Exactly one field is set."""

    top_k: int | None = None
    capital_per_wallet: float | None = None
    top_frac: float | None = None


def resolve_k(policy: KPolicy, *, bankroll: float, qualified_count: int) -> int:
    """Return the top-K cut for one rebalance boundary.

    Args:
        policy: which sizing rule to apply.
        bankroll: constant starting bankroll (USD).
        qualified_count: number of qualified wallets at this boundary.
    """
    if policy.top_k is not None:
        return policy.top_k
    if policy.capital_per_wallet is not None:
        return max(0, int(bankroll // policy.capital_per_wallet))
    if policy.top_frac is not None:
        return math.ceil(policy.top_frac * qualified_count)
    raise ValueError("KPolicy has no mode set")


def has_platform_column(db_path: Path) -> bool:
    """Return True if corpus_trades carries the multi-platform `platform` column."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(corpus_trades)")]
    conn.close()
    return "platform" in cols


def _attach(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute(f"ATTACH '{db_path}' AS s (TYPE sqlite, READONLY)")
    return con


def _build_rb(con: duckdb.DuckDBPyConnection, *, platform: str, has_platform: bool) -> None:
    """Materialize the causal resolved-buy fact table `rb`."""
    tpred = "AND t.platform = ?" if has_platform else ""
    rpred = "AND r.platform = ?" if has_platform else ""
    params = ([platform, platform] if has_platform else [])
    con.execute(
        f"""
        CREATE TEMP TABLE rb AS
        SELECT t.wallet_address AS wallet, t.condition_id AS condition_id,
               t.price AS price, t.ts AS ts, t.outcome_side AS outcome_side,
               r.resolved_at AS resolved_at,
               CASE WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES')
                      OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO')
                    THEN 1 ELSE 0 END AS won
        FROM s.corpus_trades t
        JOIN s.market_resolutions r ON r.condition_id = t.condition_id {rpred}
        WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at {tpred}
        """,  # noqa: S608 -- predicates are fixed literals; values via ? params
        params,
    )


def ranked_qualifiers(
    db_path: Path, *, platform: str, min_resolved: int, edge_window_days: int,
    boundaries: Sequence[int],
) -> list[tuple[int, str, float, int, int, int]]:
    """Return (boundary_ts, wallet, edge, n_resolved, rank, n_qualified) rows.

    Only qualified (>= min_resolved within window) AND positive-edge wallets,
    ranked per boundary by (edge DESC, wallet ASC). No lookahead: each
    boundary's edge uses only resolutions with resolved_at < boundary_ts.
    """
    if not boundaries:
        return []
    has_plat = has_platform_column(db_path)
    con = _attach(db_path)
    try:
        _build_rb(con, platform=platform, has_platform=has_plat)
        con.execute("CREATE TEMP TABLE bnd(boundary_ts BIGINT)")
        con.executemany("INSERT INTO bnd VALUES (?)", [(int(b),) for b in boundaries])
        window_pred = (
            "" if edge_window_days == 0
            else "AND rb.resolved_at >= b.boundary_ts - ? * 86400"
        )
        params = [] if edge_window_days == 0 else [edge_window_days]
        rows = con.execute(
            f"""
            WITH agg AS (
              SELECT b.boundary_ts AS boundary_ts, rb.wallet AS wallet,
                     COUNT(*) AS n_resolved, AVG(rb.won - rb.price) AS edge
              FROM bnd b
              JOIN rb ON rb.resolved_at < b.boundary_ts {window_pred}
              GROUP BY b.boundary_ts, rb.wallet
              HAVING COUNT(*) >= ? AND AVG(rb.won - rb.price) > 0
            )
            SELECT boundary_ts, wallet, edge, n_resolved,
                   ROW_NUMBER() OVER (PARTITION BY boundary_ts
                                      ORDER BY edge DESC, wallet ASC) AS rank,
                   COUNT(*) OVER (PARTITION BY boundary_ts) AS n_qualified
            FROM agg
            ORDER BY boundary_ts, rank
            """,  # noqa: S608 -- window_pred is a fixed literal; values via ? params
            [*params, min_resolved],
        ).fetchall()
    finally:
        con.close()
    return [(int(b), str(w), float(e), int(n), int(rk), int(nq))
            for (b, w, e, n, rk, nq) in rows]
