"""Insider-wallet discovery via case-control fingerprinting (DuckDB-backed).

Discovers hit-and-run winner wallets in the corpus, finds trade-time
features separating them from matched losers, and causally forward-tests
the fingerprint.

Spec: docs/superpowers/specs/2026-06-01-insider-wallet-discovery-design.md
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
from scipy import stats as _sps

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


FEATURE_NAMES: Final[tuple[str, ...]] = (
    "improbability_z",
    "max_bet_usd",
    "conviction_frac",
    "mean_entry_price",
    "mean_ttr_days",
    "prior_activity_count",
    "mean_drift",
)


@dataclass(frozen=True, slots=True)
class FeatureStat:
    """Case-vs-control separation for one feature."""

    name: str
    case_mean: float
    case_median: float
    control_mean: float
    control_median: float
    cohen_d: float
    mw_p: float


def _feature_values(rows: list[WalletAgg], name: str, drift: dict[str, float]) -> list[float]:
    if name == "mean_drift":
        return [drift[a.wallet] for a in rows if a.wallet in drift]
    return [float(getattr(a, name)) for a in rows]


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def discriminate(
    cases: list[WalletAgg],
    controls: list[WalletAgg],
    *,
    drift: dict[str, float],
    features: tuple[str, ...],
) -> list[FeatureStat]:
    """Rank features by |Cohen's d| of case-vs-control separation."""
    out: list[FeatureStat] = []
    for name in features:
        cv = np.asarray(_feature_values(cases, name, drift), dtype=float)
        kv = np.asarray(_feature_values(controls, name, drift), dtype=float)
        d = _cohen_d(cv, kv)
        if len(cv) >= 2 and len(kv) >= 2:
            mw_p = float(_sps.mannwhitneyu(cv, kv, alternative="two-sided").pvalue)
        else:
            mw_p = 1.0
        out.append(
            FeatureStat(
                name=name,
                case_mean=float(cv.mean()) if len(cv) else 0.0,
                case_median=float(np.median(cv)) if len(cv) else 0.0,
                control_mean=float(kv.mean()) if len(kv) else 0.0,
                control_median=float(np.median(kv)) if len(kv) else 0.0,
                cohen_d=d,
                mw_p=mw_p,
            )
        )
    out.sort(key=lambda s: abs(s.cohen_d), reverse=True)
    return out


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Held-out forward-test of the pre-cutoff fingerprint."""

    cutoff_ts: int
    n_flagged: int
    flagged_edge: float
    base_rate_edge: float
    fingerprint: tuple[FeatureStat, ...]


def _cutoff_ts(db_path: Path, *, cutoff_pct: int) -> int:
    has_plat = has_platform_column(db_path)
    rpred = "WHERE platform = 'polymarket'" if has_plat else ""
    con = _attach(db_path)
    try:
        row = con.execute(
            f"SELECT quantile_cont(resolved_at, ?) FROM s.market_resolutions {rpred}",  # noqa: S608
            [cutoff_pct / 100.0],
        ).fetchone()
    finally:
        con.close()
    return int(row[0]) if row and row[0] is not None else 0


def _trade_feature(name: str, *, price: float, notional: float, agg: WalletAgg) -> float | None:
    """Trade-time value of a fingerprint feature, per-trade where one exists.

    Returns None for features with no trade-time analogue (``mean_drift``).
    """
    if name == "mean_entry_price":
        return price
    if name == "max_bet_usd":
        return notional
    if name == "conviction_frac":
        return notional / agg.total_notional_usd if agg.total_notional_usd > 0 else 0.0
    if name == "mean_drift":
        return None
    return float(getattr(agg, name))


def _post_cutoff_scored_buys(
    db_path: Path, *, cutoff_ts: int, fingerprint: tuple[FeatureStat, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (score, edge) arrays for resolved buys resolving after cutoff_ts.

    Each trade is scored by a sign-weighted sum of the fingerprint features,
    using the trade's own observables (entry price, notional) where the feature
    is per-trade and the wallet's prior aggregate otherwise. No post-cutoff
    outcome enters the score.
    """
    aggs = wallet_aggregates(db_path, max_trades=10**9, max_lifespan_days=10**9)
    by_wallet = {a.wallet: a for a in aggs}
    has_plat = has_platform_column(db_path)
    tpred = "AND t.platform = 'polymarket'" if has_plat else ""
    rpred = "AND r.platform = 'polymarket'" if has_plat else ""
    con = _attach(db_path)
    try:
        rows = con.execute(
            f"""
            SELECT t.wallet_address, t.price, t.notional_usd,
                   (({_won_expr()}) - t.price) AS edge
            FROM s.corpus_trades t
            JOIN s.market_resolutions r ON r.condition_id = t.condition_id {rpred}
            WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at
              AND r.resolved_at > ? {tpred}
            """,  # noqa: S608 -- predicates are fixed literals; cutoff via ?
            [cutoff_ts],
        ).fetchall()
    finally:
        con.close()
    weights = {s.name: (1.0 if s.cohen_d >= 0 else -1.0) * abs(s.cohen_d) for s in fingerprint}
    scores: list[float] = []
    edges: list[float] = []
    for wallet, price, notional, edge in rows:
        a = by_wallet.get(str(wallet))
        if a is None:
            continue
        score = 0.0
        for name, w in weights.items():
            value = _trade_feature(name, price=float(price), notional=float(notional), agg=a)
            if value is not None:
                score += w * value
        scores.append(score)
        edges.append(float(edge))
    return np.asarray(scores, dtype=float), np.asarray(edges, dtype=float)


def forward_test(
    db_path: Path,
    *,
    cutoff_pct: int,
    max_trades: int,
    max_lifespan_days: int,
    control_ratio: int,
    drift_window_days: int,
    top_k_features: int,
    seed: int,
) -> ForwardResult:
    """Derive the fingerprint pre-cutoff, score post-cutoff trades at trade-time."""
    cutoff_ts = _cutoff_ts(db_path, cutoff_pct=cutoff_pct)
    pre_aggs = [
        a
        for a in wallet_aggregates(
            db_path, max_trades=max_trades, max_lifespan_days=max_lifespan_days
        )
        if a.last_ts <= cutoff_ts
    ]
    cases, controls = split_cohorts(pre_aggs, control_ratio=control_ratio, seed=seed)
    drift = compute_drift(
        db_path, [a.wallet for a in cases + controls], window_days=drift_window_days
    )
    fingerprint = tuple(
        discriminate(cases, controls, drift=drift, features=FEATURE_NAMES)[:top_k_features]
    )
    scores, edges = _post_cutoff_scored_buys(db_path, cutoff_ts=cutoff_ts, fingerprint=fingerprint)
    if len(scores) == 0:
        return ForwardResult(cutoff_ts, 0, 0.0, 0.0, fingerprint)
    threshold = float(np.median(scores))
    flagged = scores >= threshold
    flagged_edge = float(edges[flagged].mean()) if flagged.any() else 0.0
    return ForwardResult(
        cutoff_ts=cutoff_ts,
        n_flagged=int(flagged.sum()),
        flagged_edge=flagged_edge,
        base_rate_edge=float(edges.mean()),
        fingerprint=fingerprint,
    )


def _print_report(
    *,
    aggs: list[WalletAgg],
    cases: list[WalletAgg],
    controls: list[WalletAgg],
    stats: list[FeatureStat],
    drift: dict[str, float],
    fwd: ForwardResult,
    top_n: int,
) -> None:
    print("=== Cohort summary ===")
    print(f"shape wallets: {len(aggs)}  cases: {len(cases)}  controls: {len(controls)}")
    print("\n=== Discrimination report (ranked by |Cohen's d|) ===")
    print(f"{'feature':22} {'case_mean':>12} {'ctrl_mean':>12} {'cohen_d':>9} {'mw_p':>9}")
    for s in stats:
        print(
            f"{s.name:22} {s.case_mean:12.4f} {s.control_mean:12.4f} "
            f"{s.cohen_d:9.3f} {s.mw_p:9.4f}"
        )
    print("\n=== Top case wallets (by cash PnL) ===")
    print(
        f"{'wallet':14} {'n':>3} {'pnl_usd':>12} {'edge':>7} {'z':>6} "
        f"{'max_bet':>10} {'conv':>6} {'drift':>7}"
    )
    for a in cases[:top_n]:
        d = drift.get(a.wallet, float("nan"))
        print(
            f"{a.wallet[:14]:14} {a.n_resolved_buys:3d} {a.cash_pnl_usd:12.0f} "
            f"{a.mean_edge:7.3f} {a.improbability_z:6.2f} {a.max_bet_usd:10.0f} "
            f"{a.conviction_frac:6.2f} {d:7.3f}"
        )
    print("\n=== Forward-test ===")
    print(
        f"cutoff_ts={fwd.cutoff_ts} n_flagged={fwd.n_flagged} "
        f"flagged_edge={fwd.flagged_edge:.4f} base_rate_edge={fwd.base_rate_edge:.4f}"
    )


def _write_csv(
    path: Path, cases: list[WalletAgg], controls: list[WalletAgg], drift: dict[str, float]
) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "wallet",
                "cohort",
                "n_resolved_buys",
                "cash_pnl_usd",
                "mean_edge",
                "improbability_z",
                "max_bet_usd",
                "conviction_frac",
                "mean_drift",
            ]
        )
        for cohort, rows in (("case", cases), ("control", controls)):
            for a in rows:
                w.writerow(
                    [
                        a.wallet,
                        cohort,
                        a.n_resolved_buys,
                        a.cash_pnl_usd,
                        a.mean_edge,
                        a.improbability_z,
                        a.max_bet_usd,
                        a.conviction_frac,
                        drift.get(a.wallet, ""),
                    ]
                )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Insider-wallet discovery")
    p.add_argument("--db", type=Path, default=Path("data/corpus.sqlite3"))
    p.add_argument("--max-trades", type=int, default=10)
    p.add_argument("--max-lifespan-days", type=int, default=30)
    p.add_argument("--control-ratio", type=int, default=3)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--forward-cutoff-pct", type=int, default=70)
    p.add_argument("--drift-window-days", type=int, default=7)
    p.add_argument("--top-k-features", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    aggs = wallet_aggregates(
        args.db, max_trades=args.max_trades, max_lifespan_days=args.max_lifespan_days
    )
    cases, controls = split_cohorts(aggs, control_ratio=args.control_ratio, seed=args.seed)
    drift = compute_drift(
        args.db, [a.wallet for a in cases + controls], window_days=args.drift_window_days
    )
    stats = discriminate(cases, controls, drift=drift, features=FEATURE_NAMES)
    fwd = forward_test(
        args.db,
        cutoff_pct=args.forward_cutoff_pct,
        max_trades=args.max_trades,
        max_lifespan_days=args.max_lifespan_days,
        control_ratio=args.control_ratio,
        drift_window_days=args.drift_window_days,
        top_k_features=args.top_k_features,
        seed=args.seed,
    )
    _print_report(
        aggs=aggs, cases=cases, controls=controls, stats=stats, drift=drift, fwd=fwd, top_n=args.top_n
    )
    if args.csv is not None:
        _write_csv(args.csv, cases, controls, drift)
    return 0


if __name__ == "__main__":
    sys.exit(main())
