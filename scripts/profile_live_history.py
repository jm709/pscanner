"""Quick profile: LiveHistoryProvider.observe() median + p99 latency.

Usage: uv run python scripts/profile_live_history.py [--n 10000]

Synthetic workload — same shape as the parity fixture but at production
scale. Reports median + p99 in microseconds. The plan's DoD requires
p99 < 5_000 us on the production wallet/market scale.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db
from pscanner.testing.synthetic_trades import build_metadata, build_synthetic_trades


def main() -> int:
    """Profile observe() median and p99 latency on synthetic workload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trades = build_synthetic_trades(seed=args.seed, n=args.n, n_wallets=8, n_markets=5)
    metadata = build_metadata(trades, with_categories=False)
    conn = init_db(Path(":memory:"))
    try:
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        observe_times: list[int] = []
        for trade in trades:
            t0 = time.perf_counter_ns()
            provider.observe(trade)
            observe_times.append(time.perf_counter_ns() - t0)
    finally:
        conn.close()
    median = statistics.median(observe_times) / 1_000
    p99 = sorted(observe_times)[int(0.99 * len(observe_times))] / 1_000
    print(f"observe median={median:.1f} us p99={p99:.1f} us  (n={args.n})")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
