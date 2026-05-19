r"""Diagnostics for a paper-trading smoke run.

Answers three questions about the gate-model paper-trading loop:

1. **Latency** — how long between a trade happening on-chain and it being
   scored by the gate model? Computed as
   ``alert.created_at - body.trade_ts`` (seconds).
2. **Activity** — what trades are being taken and how concentrated are
   they on the same ``(condition_id, outcome)``?
3. **Price drift** — does the orderbook price at paper-trade time match
   the source-wallet fill price? Computed as
   ``paper_trade.fill_price - body.implied_prob_at_buy``. Positive means
   the market moved against the buyer between the source trade and the
   paper fill (we'd buy higher).

Usage:
    uv run python scripts/analyze_paper_trading.py --db data/pscanner.sqlite3
"""

# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Final

_FILL_PRICE_FALLBACK: Final[float] = 0.5


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summary(label: str, values: list[float], unit: str) -> None:
    if not values:
        print(f"  {label}: (no data)")
        return
    s = sorted(values)
    print(
        f"  {label} (n={len(s):,}): "
        f"min={s[0]:.3f}{unit}  "
        f"p50={_quantile(s, 0.5):.3f}{unit}  "
        f"p90={_quantile(s, 0.9):.3f}{unit}  "
        f"p99={_quantile(s, 0.99):.3f}{unit}  "
        f"max={s[-1]:.3f}{unit}  "
        f"mean={statistics.fmean(s):.3f}{unit}"
    )


def _bucket_histogram(values: list[float], bins: list[tuple[float, float, str]]) -> None:
    for lo, hi, label in bins:
        n = sum(1 for v in values if lo <= v < hi)
        pct = 100.0 * n / len(values) if values else 0.0
        bar = "█" * int(pct / 2)
        print(f"    {label:>20}  n={n:>6,}  {pct:5.1f}%  {bar}")


def analyze_latency(conn: sqlite3.Connection) -> None:
    """Q1: alert.created_at - body.trade_ts for gate_buy alerts."""
    print("\n=== Q1: Latency (source trade → model alert) ===")
    rows = conn.execute(
        "SELECT created_at, body_json FROM alerts WHERE detector='gate_buy'"
    ).fetchall()
    latencies = []
    for created_at, body_json in rows:
        try:
            body = json.loads(body_json)
        except json.JSONDecodeError:
            continue
        trade_ts = body.get("trade_ts")
        if isinstance(trade_ts, int | float):
            latencies.append(float(created_at) - float(trade_ts))
    _summary("delay", latencies, "s")
    if latencies:
        print("\n  distribution:")
        _bucket_histogram(
            latencies,
            [
                (-1e9, 0, "negative (clock skew)"),
                (0, 5, "0-5s"),
                (5, 15, "5-15s"),
                (15, 30, "15-30s"),
                (30, 60, "30-60s"),
                (60, 120, "60-120s"),
                (120, 300, "2-5 min"),
                (300, 900, "5-15 min"),
                (900, 1e9, "15 min+"),
            ],
        )


def analyze_activity(conn: sqlite3.Connection) -> None:
    """Q2: per-market+side concentration."""
    print("\n=== Q2: Activity (per-market same-side concentration) ===")
    rows = conn.execute(
        """
        SELECT condition_id, outcome, COUNT(*) AS n
        FROM paper_trades
        WHERE triggering_alert_detector='gate_buy'
        GROUP BY condition_id, outcome
        ORDER BY n DESC
        """
    ).fetchall()
    total = sum(r[2] for r in rows)
    print(f"  total gate_buy paper trades: {total:,}")
    print(f"  distinct (condition_id, outcome) keys: {len(rows):,}")
    if not rows:
        return
    print("\n  trades-per-key distribution:")
    counts = [r[2] for r in rows]
    _summary("trades/key", [float(c) for c in counts], "")
    print(f"\n  top 20 most-repeated (condition_id, outcome) of {len(rows):,}:")
    print(f"    {'condition_id':<68} {'side':<5} {'n':>5}")
    for cid, side, n in rows[:20]:
        print(f"    {cid:<68} {side:<5} {n:>5}")
    # category breakdown via alert body
    print("\n  by market_category (from alert body):")
    cat_counts: Counter[str] = Counter()
    rows2 = conn.execute(
        """
        SELECT a.body_json
        FROM paper_trades p
        JOIN alerts a ON a.alert_key = p.triggering_alert_key
        WHERE p.triggering_alert_detector='gate_buy'
        """
    ).fetchall()
    for (body_json,) in rows2:
        try:
            body = json.loads(body_json)
            cat = body.get("market_category", "(missing)")
            cat_counts[str(cat)] += 1
        except json.JSONDecodeError:
            pass
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total
        print(f"    {cat:<20} n={n:>5,}  {pct:5.1f}%")


def analyze_price_drift(conn: sqlite3.Connection) -> None:
    """Q3: paper fill_price vs source wallet's implied_prob_at_buy."""
    print("\n=== Q3: Price drift (paper fill_price vs source implied) ===")
    rows = conn.execute(
        """
        SELECT p.fill_price, p.outcome, a.body_json
        FROM paper_trades p
        JOIN alerts a ON a.alert_key = p.triggering_alert_key
        WHERE p.triggering_alert_detector='gate_buy'
        """
    ).fetchall()
    drifts: list[float] = []
    abs_drifts: list[float] = []
    n_fallback_05 = 0
    for fill_price, _outcome, body_json in rows:
        try:
            body = json.loads(body_json)
        except json.JSONDecodeError:
            continue
        implied = body.get("implied_prob_at_buy")
        if not isinstance(implied, int | float):
            continue
        # fill_price == 0.5 is the documented fallback when orderbook tick
        # is unavailable. Bucket separately so the real signal is clean.
        if float(fill_price) == _FILL_PRICE_FALLBACK:
            n_fallback_05 += 1
            continue
        # Source trade was a BUY at body.implied. Paper-trader BUYs the same
        # side at paper_trades.fill_price. Drift = fill - implied (positive =
        # we paid more than the source did = market moved against us).
        drift = float(fill_price) - float(implied)
        drifts.append(drift)
        abs_drifts.append(abs(drift))
    total = len(rows)
    print(f"  total gate_buy paper trades: {total:,}")
    print(f"  rows skipped (fill_price=0.5 fallback): {n_fallback_05:,}")
    print(f"  rows analyzed: {len(drifts):,}")
    if not drifts:
        return
    print()
    _summary("signed drift  ", drifts, "")
    _summary("absolute drift", abs_drifts, "")
    print("\n  signed drift distribution:")
    _bucket_histogram(
        drifts,
        [
            (-1.0, -0.05, "<-5pp (favorable)"),
            (-0.05, -0.01, "-5pp..-1pp"),
            (-0.01, -0.001, "-1pp..-0.1pp"),
            (-0.001, 0.001, "~0 (no drift)"),
            (0.001, 0.01, "+0.1pp..+1pp"),
            (0.01, 0.05, "+1pp..+5pp"),
            (0.05, 1.0, "+5pp+ (adverse)"),
        ],
    )
    favorable = sum(1 for d in drifts if d < 0)
    adverse = sum(1 for d in drifts if d > 0)
    flat = len(drifts) - favorable - adverse
    print(
        f"\n  signs: favorable={favorable:,} ({100 * favorable / len(drifts):.1f}%)  "
        f"flat={flat:,}  "
        f"adverse={adverse:,} ({100 * adverse / len(drifts):.1f}%)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=str,
        default="data/pscanner.sqlite3",
        help="Path to the daemon SQLite database",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: parse args, open DB, run three analyses."""
    args = _parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1
    conn = sqlite3.connect(str(db_path))
    try:
        analyze_latency(conn)
        analyze_activity(conn)
        analyze_price_drift(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
