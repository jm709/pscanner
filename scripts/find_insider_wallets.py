"""Insider-wallet discovery via case-control fingerprinting (DuckDB-backed).

Discovers hit-and-run winner wallets in the corpus, finds trade-time
features separating them from matched losers, and causally forward-tests
the fingerprint.

Spec: docs/superpowers/specs/2026-06-01-insider-wallet-discovery-design.md
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from scripts.copy_selection import has_platform_column

_SECONDS_PER_DAY: Final[int] = 86_400


@dataclass(frozen=True, slots=True)
class WalletAgg:
    """Per-wallet aggregate over causal resolved buys (within the shape gate)."""

    wallet: str
    n_resolved_buys: int
    n_distinct_markets: int
    first_ts: int
    last_ts: int
    active_lifespan_days: float
    total_notional_usd: float
    mean_bet_usd: float
    max_bet_usd: float
    mean_edge: float
    cash_pnl_usd: float
    mean_entry_price: float
    improbability_z: float
    mean_ttr_days: float
    prior_activity_count: int

    @property
    def conviction_frac(self) -> float:
        """Largest single bet as a share of lifetime notional (0..1)."""
        if self.total_notional_usd <= 0:
            return 0.0
        return self.max_bet_usd / self.total_notional_usd


def _attach(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA temp_directory='data/duckdb_spill'")
    con.execute(f"ATTACH '{db_path}' AS s (TYPE sqlite, READONLY)")
    return con


def _won_expr() -> str:
    return (
        "CASE WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES') "
        "OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO') THEN 1 ELSE 0 END"
    )


def wallet_aggregates(db_path: Path, *, max_trades: int, max_lifespan_days: int) -> list[WalletAgg]:
    """Return per-wallet aggregates for wallets inside the hit-and-run shape gate.

    Shape gate: ``n_resolved_buys <= max_trades`` AND
    ``active_lifespan_days <= max_lifespan_days``.
    """
    has_plat = has_platform_column(db_path)
    tpred = "AND t.platform = 'polymarket'" if has_plat else ""
    rpred = "AND r.platform = 'polymarket'" if has_plat else ""
    con = _attach(db_path)
    try:
        con.execute(
            f"""
            CREATE TEMP TABLE rb AS
            SELECT t.wallet_address AS wallet, t.condition_id AS condition_id,
                   t.price AS price, t.ts AS ts, t.notional_usd AS notional,
                   r.resolved_at AS resolved_at, {_won_expr()} AS won
            FROM s.corpus_trades t
            JOIN s.market_resolutions r
              ON r.condition_id = t.condition_id {rpred}
            WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at {tpred}
            """  # noqa: S608 -- predicates are fixed literals
        )
        con.execute(
            f"""
            CREATE TEMP TABLE prior AS
            SELECT t.wallet_address AS wallet, COUNT(*) AS c
            FROM s.corpus_trades t
            JOIN (SELECT wallet, MIN(ts) AS ft FROM rb GROUP BY wallet) f
              ON f.wallet = t.wallet_address
            WHERE t.ts < f.ft {tpred}
            GROUP BY t.wallet_address
            """  # noqa: S608 -- predicates are fixed literals
        )
        rows = con.execute(
            f"""
            SELECT rb.wallet,
                   COUNT(*) AS n_resolved_buys,
                   COUNT(DISTINCT rb.condition_id) AS n_distinct_markets,
                   MIN(rb.ts) AS first_ts, MAX(rb.ts) AS last_ts,
                   (MAX(rb.ts) - MIN(rb.ts)) / {_SECONDS_PER_DAY}.0 AS lifespan_days,
                   SUM(rb.notional) AS total_notional,
                   AVG(rb.notional) AS mean_bet, MAX(rb.notional) AS max_bet,
                   AVG(rb.won - rb.price) AS mean_edge,
                   SUM(CASE WHEN rb.won = 1
                            THEN rb.notional * (1 - rb.price) / rb.price
                            ELSE -rb.notional END) AS cash_pnl,
                   AVG(rb.price) AS mean_entry_price,
                   (SUM(rb.won) - SUM(rb.price))
                     / NULLIF(sqrt(SUM(rb.price * (1 - rb.price))), 0) AS improb_z,
                   AVG((rb.resolved_at - rb.ts) / {_SECONDS_PER_DAY}.0) AS mean_ttr_days,
                   COALESCE(MAX(p.c), 0) AS prior_count
            FROM rb LEFT JOIN prior p ON p.wallet = rb.wallet
            GROUP BY rb.wallet
            HAVING COUNT(*) <= ?
               AND (MAX(rb.ts) - MIN(rb.ts)) / {_SECONDS_PER_DAY}.0 <= ?
            """,  # noqa: S608 -- _SECONDS_PER_DAY is a fixed literal; values via ?
            [max_trades, max_lifespan_days],
        ).fetchall()
    finally:
        con.close()
    return [
        WalletAgg(
            wallet=str(r[0]),
            n_resolved_buys=int(r[1]),
            n_distinct_markets=int(r[2]),
            first_ts=int(r[3]),
            last_ts=int(r[4]),
            active_lifespan_days=float(r[5]),
            total_notional_usd=float(r[6]),
            mean_bet_usd=float(r[7]),
            max_bet_usd=float(r[8]),
            mean_edge=float(r[9]),
            cash_pnl_usd=float(r[10]),
            mean_entry_price=float(r[11]),
            improbability_z=float(r[12]) if r[12] is not None else 0.0,
            mean_ttr_days=float(r[13]),
            prior_activity_count=int(r[14]),
        )
        for r in rows
    ]


def _era(first_ts: int) -> str:
    """Calendar-quarter bucket of a wallet's first resolved buy."""
    d = dt.datetime.fromtimestamp(first_ts, tz=dt.UTC)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _stratum(a: WalletAgg) -> tuple[int, str]:
    return (a.n_resolved_buys, _era(a.first_ts))


def split_cohorts(
    aggs: list[WalletAgg], *, control_ratio: int, seed: int = 0
) -> tuple[list[WalletAgg], list[WalletAgg]]:
    """Split shape wallets into PnL-positive cases and matched negative controls.

    Controls are sampled at ``control_ratio`` per case within each
    ``(n_resolved_buys, era)`` stratum. Degrades gracefully when a stratum has
    fewer controls than requested.
    """
    cases = [a for a in aggs if a.cash_pnl_usd > 0 and a.mean_edge > 0]
    losers = [a for a in aggs if a.cash_pnl_usd <= 0]
    pool: dict[tuple[int, str], list[WalletAgg]] = {}
    for a in losers:
        pool.setdefault(_stratum(a), []).append(a)
    rng = random.Random(seed)
    for bucket in pool.values():
        bucket.sort(key=lambda a: a.wallet)
        rng.shuffle(bucket)
    need: dict[tuple[int, str], int] = {}
    for c in cases:
        need[_stratum(c)] = need.get(_stratum(c), 0) + control_ratio
    controls: list[WalletAgg] = []
    for stratum, n in need.items():
        controls.extend(pool.get(stratum, [])[:n])
    cases.sort(key=lambda a: a.cash_pnl_usd, reverse=True)
    return cases, controls


def compute_drift(db_path: Path, wallets: list[str], *, window_days: int) -> dict[str, float]:
    """Mean post-entry price drift toward each wallet's side.

    Returns only wallets with at least one entry that has an in-window later
    trade on the same market.
    """
    if not wallets:
        return {}
    has_plat = has_platform_column(db_path)
    tpred = "AND t.platform = 'polymarket'" if has_plat else ""
    lpred = "AND l.platform = 'polymarket'" if has_plat else ""
    rpred = "AND r.platform = 'polymarket'" if has_plat else ""
    con = _attach(db_path)
    try:
        con.execute("CREATE TEMP TABLE w(wallet VARCHAR)")
        con.executemany("INSERT INTO w VALUES (?)", [(x,) for x in wallets])
        rows = con.execute(
            f"""
            WITH entries AS (
              SELECT t.wallet_address AS wallet, t.condition_id AS cid,
                     t.outcome_side AS side, t.price AS p0, t.ts AS t0,
                     r.resolved_at AS resolved_at
              FROM s.corpus_trades t
              JOIN w ON w.wallet = t.wallet_address
              JOIN s.market_resolutions r
                ON r.condition_id = t.condition_id {rpred}
              WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at {tpred}
            ),
            per_entry AS (
              SELECT e.wallet,
                     AVG(CASE WHEN l.outcome_side = e.side
                              THEN l.price ELSE 1 - l.price END) - e.p0 AS drift
              FROM entries e
              JOIN s.corpus_trades l ON l.condition_id = e.cid {lpred}
              WHERE l.ts > e.t0
                AND l.ts <= LEAST(e.t0 + ? * {_SECONDS_PER_DAY},
                                  e.resolved_at - {_SECONDS_PER_DAY})
              GROUP BY e.wallet, e.cid, e.t0, e.p0
            )
            SELECT wallet, AVG(drift) FROM per_entry GROUP BY wallet
            """,  # noqa: S608 -- _SECONDS_PER_DAY is a fixed literal; window via ?
            [window_days],
        ).fetchall()
    finally:
        con.close()
    return {str(w): float(d) for w, d in rows}
