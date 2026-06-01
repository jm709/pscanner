# Insider-Wallet Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/find_insider_wallets.py`, a DuckDB-backed research tool that discovers the hit-and-run winner cohort in the corpus, finds the trade-time features separating winners from matched losers, and causally forward-tests that fingerprint.

**Architecture:** A read-only DuckDB attach over `data/corpus.sqlite3` does the heavy aggregation (per-wallet stats + improbability z + cash PnL + post-entry price drift). Python does the small-frame work: case/control matching, discrimination stats (Cohen's d + Mann–Whitney), and the held-out forward-test. Cases are defined by realized PnL; improbability and conviction-size stay as separate lenses, never as the case gate.

**Tech Stack:** Python 3.13, DuckDB 1.5.2 (`sqlite_scanner`), numpy, scipy.stats, pytest. Mirrors `scripts/wallet_edge_leaderboard.py` and `scripts/copy_selection.py`.

**Spec:** `docs/superpowers/specs/2026-06-01-insider-wallet-discovery-design.md`

## File Structure

- Create: `scripts/find_insider_wallets.py` — all discovery logic + CLI.
- Create: `tests/scripts/test_find_insider_wallets.py` — unit + integration tests.
- Modify: `pyproject.toml` — add explicit `numpy` + `scipy` deps (already resolved transitively).
- Reuse (import, do not reimplement): `scripts.copy_selection.has_platform_column`.
- Reuse (test fixture): `tests/scripts/conftest.py::corpus_factory`.

**Trade tuple** (from `conftest.py`): `(wallet, condition_id, outcome_side, bs, price, notional_usd, ts)`.
**Resolution tuple**: `(condition_id, outcome_yes_won, resolved_at)`.

**Key fact:** in `corpus_trades`, `price` is already the entry price of the *side bought* (so `edge = won − price` and `price` = the market-implied probability of that side). No YES/NO normalization is needed for improbability or edge; it is only needed for the post-entry drift feature, where a later trade on the opposite side contributes `1 − later.price`.

---

### Task 1: Add numpy + scipy as explicit dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Confirm the resolved versions in the current env**

Run: `uv run python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"`
Expected: `2.4.4 1.17.1` (use whatever prints if different; pin to those exact strings).

- [ ] **Step 2: Add pinned deps**

In `pyproject.toml`, in the `dependencies` array next to `"duckdb==1.5.2",`, add:

```toml
    "numpy==2.4.4",
    "scipy==1.17.1",
```

- [ ] **Step 3: Sync and verify**

Run: `uv sync && uv run python -c "import scipy.stats, numpy; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add numpy + scipy deps for insider-wallet discovery"
```

---

### Task 2: Module skeleton + per-wallet aggregate query

**Files:**
- Create: `scripts/find_insider_wallets.py`
- Test: `tests/scripts/test_find_insider_wallets.py`

The aggregate computes everything SQL-expressible in one GROUP BY over a causal
resolved-buy CTE: counts, lifespan, bet-size moments, mean edge, cash PnL, the
Poisson-binomial improbability z, mean time-to-resolution, and prior-activity
count (history before the wallet's first resolved buy).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for insider-wallet discovery."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.find_insider_wallets import WalletAgg, wallet_aggregates


def test_wallet_aggregates_basic_stats(corpus_factory) -> None:
    # Wallet A: 2 resolved buys. YES@0.10 wins, YES@0.50 loses.
    trades = [
        ("A", "m1", "YES", "BUY", 0.10, 100.0, 1_000),
        ("A", "m2", "YES", "BUY", 0.50, 200.0, 2_000),
    ]
    resolutions = [("m1", 1, 5_000), ("m2", 0, 6_000)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)

    aggs = {a.wallet: a for a in wallet_aggregates(db, max_trades=10, max_lifespan_days=30)}
    a = aggs["A"]
    assert a.n_resolved_buys == 2
    assert a.n_distinct_markets == 2
    assert a.max_bet_usd == 200.0
    # cash PnL: win -> 100*(1-0.10)/0.10 = 900 ; loss -> -200 ; net 700
    assert math.isclose(a.cash_pnl_usd, 700.0, rel_tol=1e-9)
    # mean edge: ((1-0.10) + (0-0.50)) / 2 = 0.20
    assert math.isclose(a.mean_edge, 0.20, rel_tol=1e-9)
    # improbability z: (obs_wins - exp) / sqrt(sum p(1-p))
    #   obs=1, exp=0.10+0.50=0.60, var=0.10*0.90+0.50*0.50=0.34
    assert math.isclose(a.improbability_z, (1 - 0.60) / math.sqrt(0.34), rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py::test_wallet_aggregates_basic_stats -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.find_insider_wallets'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/find_insider_wallets.py`:

```python
"""Insider-wallet discovery via case-control fingerprinting (DuckDB-backed).

Discovers hit-and-run winner wallets in the corpus, finds trade-time
features separating them from matched losers, and causally forward-tests
the fingerprint.

Spec: docs/superpowers/specs/2026-06-01-insider-wallet-discovery-design.md
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

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


def wallet_aggregates(
    db_path: Path, *, max_trades: int, max_lifespan_days: int
) -> list[WalletAgg]:
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
            wallet=str(r[0]), n_resolved_buys=int(r[1]), n_distinct_markets=int(r[2]),
            first_ts=int(r[3]), last_ts=int(r[4]), active_lifespan_days=float(r[5]),
            total_notional_usd=float(r[6]), mean_bet_usd=float(r[7]),
            max_bet_usd=float(r[8]), mean_edge=float(r[9]), cash_pnl_usd=float(r[10]),
            mean_entry_price=float(r[11]),
            improbability_z=float(r[12]) if r[12] is not None else 0.0,
            mean_ttr_days=float(r[13]), prior_activity_count=int(r[14]),
        )
        for r in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py::test_wallet_aggregates_basic_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "feat(insider): per-wallet aggregate with improbability z + cash PnL"
```

---

### Task 3: Shape gate + improbability-z edge cases

**Files:**
- Test: `tests/scripts/test_find_insider_wallets.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_shape_gate_excludes_too_many_trades(corpus_factory) -> None:
    trades = [("A", f"m{i}", "YES", "BUY", 0.5, 50.0, 1_000 + i) for i in range(11)]
    resolutions = [(f"m{i}", 1, 5_000) for i in range(11)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    aggs = wallet_aggregates(db, max_trades=10, max_lifespan_days=30)
    assert aggs == []  # 11 > max_trades=10


def test_shape_gate_excludes_long_lifespan(corpus_factory) -> None:
    # two trades 31 days apart
    trades = [
        ("A", "m1", "YES", "BUY", 0.5, 50.0, 1_000),
        ("A", "m2", "YES", "BUY", 0.5, 50.0, 1_000 + 31 * 86_400),
    ]
    resolutions = [("m1", 1, 1_000 + 40 * 86_400), ("m2", 1, 1_000 + 40 * 86_400)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    assert wallet_aggregates(db, max_trades=10, max_lifespan_days=30) == []


def test_improbability_z_single_cheap_win_is_large(corpus_factory) -> None:
    trades = [("A", "m1", "YES", "BUY", 0.05, 100.0, 1_000)]
    db: Path = corpus_factory(trades, [("m1", 1, 5_000)], with_platform=True)
    a = wallet_aggregates(db, max_trades=10, max_lifespan_days=30)[0]
    # z = (1 - 0.05) / sqrt(0.05*0.95) ~= 4.36
    assert a.improbability_z > 4.0


def test_improbability_z_zero_variance_is_zero(corpus_factory) -> None:
    # single bet at price 1.0 -> variance 0 -> NULLIF guard -> coalesced 0.0
    trades = [("A", "m1", "YES", "BUY", 1.0, 100.0, 1_000)]
    db: Path = corpus_factory(trades, [("m1", 1, 5_000)], with_platform=True)
    a = wallet_aggregates(db, max_trades=10, max_lifespan_days=30)[0]
    assert a.improbability_z == 0.0


def test_aggregates_runs_without_platform_column(corpus_factory) -> None:
    trades = [("A", "m1", "YES", "BUY", 0.2, 100.0, 1_000)]
    db: Path = corpus_factory(trades, [("m1", 1, 5_000)], with_platform=False)
    assert len(wallet_aggregates(db, max_trades=10, max_lifespan_days=30)) == 1
```

- [ ] **Step 2: Run tests to verify they fail or pass**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -v -k "shape or improbability or without_platform"`
Expected: these should PASS with the Task 2 implementation (they exercise existing code paths). If any fail, fix `wallet_aggregates` until green. The `without_platform` test confirms the `has_platform` branch.

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_find_insider_wallets.py
git commit -m "test(insider): shape gate + improbability-z edge cases"
```

---

### Task 4: Case/control split with stratified matching

**Files:**
- Modify: `scripts/find_insider_wallets.py`
- Test: `tests/scripts/test_find_insider_wallets.py`

Cases = shape wallets with `cash_pnl_usd > 0` AND `mean_edge > 0`. Controls =
shape wallets with `cash_pnl_usd <= 0`, sampled to `control_ratio` per case,
stratified on `(n_resolved_buys, era)` where era is the calendar quarter of
`first_ts`. Deterministic via a seeded `random.Random`.

- [ ] **Step 1: Write the failing test**

```python
from scripts.find_insider_wallets import split_cohorts


def _agg(wallet, *, n, edge, pnl, first_ts):
    from scripts.find_insider_wallets import WalletAgg
    return WalletAgg(
        wallet=wallet, n_resolved_buys=n, n_distinct_markets=n,
        first_ts=first_ts, last_ts=first_ts + 100, active_lifespan_days=1.0,
        total_notional_usd=1000.0, mean_bet_usd=100.0, max_bet_usd=500.0,
        mean_edge=edge, cash_pnl_usd=pnl, mean_entry_price=0.3,
        improbability_z=2.0, mean_ttr_days=5.0, prior_activity_count=0,
    )


def test_split_cohorts_cases_and_matched_controls() -> None:
    aggs = [_agg(f"win{i}", n=2, edge=0.3, pnl=500.0, first_ts=1_000) for i in range(2)]
    aggs += [_agg(f"lose{i}", n=2, edge=-0.2, pnl=-100.0, first_ts=1_000) for i in range(10)]
    cases, controls = split_cohorts(aggs, control_ratio=3, seed=0)
    assert {c.wallet for c in cases} == {"win0", "win1"}
    assert len(controls) == 6  # 2 cases * ratio 3, same (n, era) stratum
    assert all(c.cash_pnl_usd <= 0 for c in controls)


def test_split_cohorts_degrades_when_controls_scarce() -> None:
    aggs = [_agg("win0", n=2, edge=0.3, pnl=500.0, first_ts=1_000)]
    aggs += [_agg("lose0", n=2, edge=-0.2, pnl=-100.0, first_ts=1_000)]
    cases, controls = split_cohorts(aggs, control_ratio=3, seed=0)
    assert len(cases) == 1
    assert len(controls) == 1  # only one control available; no error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k split_cohorts -v`
Expected: FAIL — `ImportError: cannot import name 'split_cohorts'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/find_insider_wallets.py` (add `import random` and `import datetime as dt` to imports):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k split_cohorts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "feat(insider): stratified case/control split"
```

---

### Task 5: Post-entry price-drift feature (market-moved-after)

**Files:**
- Modify: `scripts/find_insider_wallets.py`
- Test: `tests/scripts/test_find_insider_wallets.py`

For each cohort wallet's resolved buy on market `m`, side `s`, entry price `p0`
at `t0` resolving at `R`: average the side-normalized implied probability of `s`
across *all* corpus trades on `m` in `(t0, LEAST(t0 + window, R − 1 day)]`, then
`drift = windowed_prob − p0`. Per wallet: mean drift over its measurable trades.
The trailing-24h exclusion prevents resolution-driven price snapping from looking
like foresight.

- [ ] **Step 1: Write the failing test**

```python
from scripts.find_insider_wallets import compute_drift


def test_drift_positive_when_market_moves_toward_wallet(corpus_factory) -> None:
    day = 86_400
    trades = [
        ("A", "m1", "YES", "BUY", 0.20, 100.0, 1_000),           # entry @0.20
        ("B", "m1", "YES", "BUY", 0.70, 50.0, 1_000 + 2 * day),  # market drifts to 0.70
    ]
    resolutions = [("m1", 1, 1_000 + 30 * day)]
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    drift = compute_drift(db, ["A"], window_days=7)
    assert drift["A"] > 0.4  # ~0.70 - 0.20


def test_drift_excludes_final_24h_snap(corpus_factory) -> None:
    day = 86_400
    R = 1_000 + 30 * day
    trades = [
        ("A", "m1", "YES", "BUY", 0.20, 100.0, 1_000),
        ("B", "m1", "YES", "BUY", 0.99, 50.0, R - 3_600),  # snap within final 24h
    ]
    db: Path = corpus_factory(trades, [("m1", 1, R)], with_platform=True)
    drift = compute_drift(db, ["A"], window_days=7)
    assert "A" not in drift  # no measurable in-window later trade


def test_drift_side_normalizes_opposite_side(corpus_factory) -> None:
    day = 86_400
    trades = [
        ("A", "m1", "NO", "BUY", 0.30, 100.0, 1_000),            # NO @0.30 (YES=0.70)
        ("B", "m1", "YES", "BUY", 0.20, 50.0, 1_000 + 2 * day),  # YES=0.20 -> NO=0.80
    ]
    db: Path = corpus_factory(trades, [("m1", 0, 1_000 + 30 * day)], with_platform=True)
    drift = compute_drift(db, ["A"], window_days=7)
    assert drift["A"] > 0.4  # NO prob moved 0.30 -> 0.80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k drift -v`
Expected: FAIL — `ImportError: cannot import name 'compute_drift'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/find_insider_wallets.py`:

```python
def compute_drift(
    db_path: Path, wallets: list[str], *, window_days: int
) -> dict[str, float]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k drift -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "feat(insider): post-entry price-drift feature"
```

---

### Task 6: Discrimination stats (Cohen's d + Mann–Whitney)

**Files:**
- Modify: `scripts/find_insider_wallets.py`
- Test: `tests/scripts/test_find_insider_wallets.py`

For each named feature, build case/control value vectors (drift pulled from the
`compute_drift` map; missing-drift wallets dropped from that feature only),
compute Cohen's d (pooled-SD standardized mean difference) and a Mann–Whitney U
two-sided p-value. Return sorted by `|cohen_d|` descending.

- [ ] **Step 1: Write the failing test**

```python
from scripts.find_insider_wallets import FEATURE_NAMES, FeatureStat, discriminate


def test_discriminate_ranks_separating_feature_first() -> None:
    # cases have high improbability_z, controls low; everything else equal.
    cases = [_agg(f"c{i}", n=2, edge=0.3, pnl=500.0, first_ts=1_000) for i in range(8)]
    controls = [_agg(f"k{i}", n=2, edge=-0.1, pnl=-50.0, first_ts=1_000) for i in range(8)]
    cases = [c.__class__(**{**c.__dict__, "improbability_z": 4.0}) for c in cases]
    controls = [c.__class__(**{**c.__dict__, "improbability_z": 0.1}) for c in controls]
    stats = discriminate(cases, controls, drift={}, features=FEATURE_NAMES)
    assert stats[0].name == "improbability_z"
    assert stats[0].cohen_d > 1.0
    assert stats[0].mw_p < 0.05


def test_discriminate_handles_missing_drift() -> None:
    cases = [_agg(f"c{i}", n=2, edge=0.3, pnl=500.0, first_ts=1_000) for i in range(4)]
    controls = [_agg(f"k{i}", n=2, edge=-0.1, pnl=-50.0, first_ts=1_000) for i in range(4)]
    # only one case has a drift value -> drift feature has too few samples
    stats = {s.name: s for s in discriminate(cases, controls, drift={"c0": 0.5},
                                             features=("mean_drift",))}
    assert stats["mean_drift"].cohen_d == 0.0  # insufficient samples -> sentinel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k discriminate -v`
Expected: FAIL — `ImportError: cannot import name 'discriminate'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/find_insider_wallets.py` (add `import numpy as np` and
`from scipy import stats as _sps` to imports):

```python
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


def _feature_values(
    rows: list[WalletAgg], name: str, drift: dict[str, float]
) -> list[float]:
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
```

Note: the test calls `discriminate(cases, controls, drift=..., features=...)`.
Update the first test's call to use the `drift=` keyword (it is keyword-only).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k discriminate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "feat(insider): case-control discrimination stats"
```

---

### Task 7: Causal forward-test

**Files:**
- Modify: `scripts/find_insider_wallets.py`
- Test: `tests/scripts/test_find_insider_wallets.py`

Split markets by `resolved_at` at the cutoff percentile `T`. Derive the
fingerprint on pre-`T` data only (run `wallet_aggregates` → `split_cohorts` →
`discriminate` over markets resolving before `T`). Build a sign-weighted score
from the top-`k` discriminating features. Apply that score to post-`T` resolved
buys at *trade-time* (features known at entry; no post-`T` outcome used), flag
trades scoring above the median, and measure their realized forward edge
(`mean(won − price)`) vs the post-`T` base rate.

- [ ] **Step 1: Write the failing test**

```python
from scripts.find_insider_wallets import ForwardResult, forward_test


def test_forward_test_no_lookahead_and_reports_edge(corpus_factory) -> None:
    day = 86_400
    trades, resolutions = [], []
    # pre-T: skilled wallets buy cheap and win; noise wallets buy and lose.
    for i in range(20):
        trades.append((f"skill{i}", f"pre{i}", "YES", "BUY", 0.10, 100.0, 1_000 + i))
        resolutions.append((f"pre{i}", 1, 10 * day + i))
    for i in range(20):
        trades.append((f"noise{i}", f"prn{i}", "YES", "BUY", 0.60, 100.0, 1_000 + i))
        resolutions.append((f"prn{i}", 0, 10 * day + i))
    # post-T: cheap-buy trades win, expensive-buy trades lose.
    for i in range(20):
        trades.append((f"q{i}", f"post{i}", "YES", "BUY", 0.10, 100.0, 100 * day + i))
        resolutions.append((f"post{i}", 1, 200 * day + i))
        trades.append((f"q{i}", f"poex{i}", "YES", "BUY", 0.80, 100.0, 100 * day + i))
        resolutions.append((f"poex{i}", 0, 200 * day + i))
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    res = forward_test(
        db, cutoff_pct=50, max_trades=10, max_lifespan_days=30,
        control_ratio=3, drift_window_days=7, top_k_features=3, seed=0,
    )
    assert isinstance(res, ForwardResult)
    assert res.n_flagged > 0
    assert res.flagged_edge > res.base_rate_edge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k forward_test -v`
Expected: FAIL — `ImportError: cannot import name 'forward_test'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/find_insider_wallets.py`:

```python
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


def _post_cutoff_scored_buys(
    db_path: Path, *, cutoff_ts: int, fingerprint: tuple[FeatureStat, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (score, edge) arrays for resolved buys resolving after cutoff_ts.

    Score is a sign-weighted sum over the per-wallet aggregate features of the
    fingerprint, computed only from trade-time-observable per-wallet stats.
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
            SELECT t.wallet_address, (({_won_expr()}) - t.price) AS edge
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
    scores, edges = [], []
    for wallet, edge in rows:
        a = by_wallet.get(str(wallet))
        if a is None:
            continue
        score = sum(
            w * float(getattr(a, name)) for name, w in weights.items() if name != "mean_drift"
        )
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
    scores, edges = _post_cutoff_scored_buys(
        db_path, cutoff_ts=cutoff_ts, fingerprint=fingerprint
    )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k forward_test -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "feat(insider): causal forward-test of the fingerprint"
```

---

### Task 8: CLI + report rendering

**Files:**
- Modify: `scripts/find_insider_wallets.py`
- Test: `tests/scripts/test_find_insider_wallets.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts.find_insider_wallets import main


def test_main_runs_end_to_end(corpus_factory, capsys) -> None:
    day = 86_400
    trades, resolutions = [], []
    for i in range(15):
        trades.append((f"win{i}", f"m{i}", "YES", "BUY", 0.10, 500.0, 1_000 + i))
        resolutions.append((f"m{i}", 1, 10 * day + i))
    for i in range(15):
        trades.append((f"lose{i}", f"n{i}", "YES", "BUY", 0.60, 100.0, 1_000 + i))
        resolutions.append((f"n{i}", 0, 10 * day + i))
    db: Path = corpus_factory(trades, resolutions, with_platform=True)
    rc = main(["--db", str(db), "--max-trades", "10", "--max-lifespan-days", "30",
               "--forward-cutoff-pct", "50"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Cohort summary" in out
    assert "Discrimination report" in out
    assert "Forward-test" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k main_runs -v`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/find_insider_wallets.py` (add `import argparse`, `import csv`,
`import sys`):

```python
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
        print(f"{s.name:22} {s.case_mean:12.4f} {s.control_mean:12.4f} "
              f"{s.cohen_d:9.3f} {s.mw_p:9.4f}")
    print("\n=== Top case wallets (by cash PnL) ===")
    print(f"{'wallet':14} {'n':>3} {'pnl_usd':>12} {'edge':>7} {'z':>6} "
          f"{'max_bet':>10} {'conv':>6} {'drift':>7}")
    for a in cases[:top_n]:
        d = drift.get(a.wallet, float('nan'))
        print(f"{a.wallet[:14]:14} {a.n_resolved_buys:3d} {a.cash_pnl_usd:12.0f} "
              f"{a.mean_edge:7.3f} {a.improbability_z:6.2f} {a.max_bet_usd:10.0f} "
              f"{a.conviction_frac:6.2f} {d:7.3f}")
    print("\n=== Forward-test ===")
    print(f"cutoff_ts={fwd.cutoff_ts} n_flagged={fwd.n_flagged} "
          f"flagged_edge={fwd.flagged_edge:.4f} base_rate_edge={fwd.base_rate_edge:.4f}")


def _write_csv(path: Path, cases: list[WalletAgg], controls: list[WalletAgg],
               drift: dict[str, float]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wallet", "cohort", "n_resolved_buys", "cash_pnl_usd", "mean_edge",
                    "improbability_z", "max_bet_usd", "conviction_frac", "mean_drift"])
        for cohort, rows in (("case", cases), ("control", controls)):
            for a in rows:
                w.writerow([a.wallet, cohort, a.n_resolved_buys, a.cash_pnl_usd,
                            a.mean_edge, a.improbability_z, a.max_bet_usd,
                            a.conviction_frac, drift.get(a.wallet, "")])


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
        args.db, cutoff_pct=args.forward_cutoff_pct, max_trades=args.max_trades,
        max_lifespan_days=args.max_lifespan_days, control_ratio=args.control_ratio,
        drift_window_days=args.drift_window_days, top_k_features=args.top_k_features,
        seed=args.seed,
    )
    _print_report(aggs=aggs, cases=cases, controls=controls, stats=stats,
                  drift=drift, fwd=fwd, top_n=args.top_n)
    if args.csv is not None:
        _write_csv(args.csv, cases, controls, drift)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_find_insider_wallets.py -k main_runs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "feat(insider): CLI + report rendering"
```

---

### Task 9: Full gate + sys.path bootstrap check

**Files:**
- Modify: `scripts/find_insider_wallets.py` (only if direct execution needs the path shim)

`scripts/backtest_copy_sizing.py` bootstraps `sys.path` so the `scripts` package
imports work under direct `python scripts/...` execution. Confirm this script's
`from scripts.copy_selection import ...` works under `uv run python scripts/find_insider_wallets.py`; if it raises `ModuleNotFoundError`, add the same shim.

- [ ] **Step 1: Check direct execution**

Run: `uv run python scripts/find_insider_wallets.py --help`
Expected: argparse help text prints, exit 0. If `ModuleNotFoundError: No module named 'scripts'`, proceed to Step 2; otherwise skip to Step 3.

- [ ] **Step 2: Add the path shim (only if Step 1 failed)**

At the very top of `scripts/find_insider_wallets.py`, immediately after the
module docstring and before other imports, mirror the pattern in
`scripts/backtest_copy_sizing.py`:

```python
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Then re-run Step 1 and confirm it prints help.

- [ ] **Step 3: Run the full gate**

Run: `uv run ruff check scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py && uv run ruff format --check scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py && uv run ty check scripts/find_insider_wallets.py && uv run pytest tests/scripts/test_find_insider_wallets.py -q`
Expected: ruff clean, ty clean, all tests PASS. Fix any warnings (zero-warnings policy) before committing.

- [ ] **Step 4: Commit**

```bash
git add scripts/find_insider_wallets.py tests/scripts/test_find_insider_wallets.py
git commit -m "chore(insider): satisfy ruff/ty/pytest gate"
```

---

### Task 10: Desktop-corpus smoke run

**Files:** none (operational)

- [ ] **Step 1: Sync the script to the desktop**

Run (from `LOCAL_NOTES.md`, LAN path):

```bash
rsync -avh -e "ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  scripts/find_insider_wallets.py scripts/copy_selection.py \
  macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/scripts/
```

- [ ] **Step 2: Run against the real corpus**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner && \
   export PATH="$HOME/.local/bin:$PATH" && \
   uv run python scripts/find_insider_wallets.py \
     --db data/corpus.sqlite3 --top-n 50 --csv /tmp/insider_cohorts.csv \
     2>&1 | tee /tmp/insider_run.log'
```

Expected: cohort summary, discrimination report ranked by |Cohen's d|, top case
wallets, and a forward-test line. Wall time: low minutes (aggregate is fast; the
drift self-join is bounded by the cohort trade set).

- [ ] **Step 3: Review the fingerprint and record findings**

Read `/tmp/insider_run.log`. Note which features separate cases from controls
(|d| and `mw_p`), and whether `flagged_edge > base_rate_edge` in the forward-test
(the green/red light for a phase-2 detector). Capture the top case wallets for
the follow-up on-chain cross-check.

- [ ] **Step 4: Commit any tuning**

If the run motivates default changes (e.g. `--max-trades`, `--drift-window-days`),
make them and re-run the Task 9 gate, then commit.

---

## Self-Review

**Spec coverage:**
- Hit-and-run shape gate → Task 2/3 (`wallet_aggregates` HAVING). ✓
- Cases by PnL, not luck score → Task 4 (`split_cohorts`). ✓
- Improbability z (separate lens) → Task 2 (SQL), Task 3 (edge cases). ✓
- Conviction size (separate lens) → `max_bet_usd` + `conviction_frac` (Task 2 property), feature in Task 6. ✓
- Matched controls (count-bucket × era) → Task 4. ✓
- Market-moved-after-entry drift, trailing-24h exclusion, side normalization → Task 5. ✓
- Discrimination report (Cohen's d + Mann–Whitney, ranked) → Task 6. ✓
- Causal forward-test, no-lookahead → Task 7. ✓
- CLI flags (all from spec table) → Task 8 (`_parse_args`). ✓
- CSV output, ranked case list, cohort summary, forward-test line → Task 8. ✓
- Platform-column portability → reused `has_platform_column`, tested in Task 3. ✓
- Desktop run + on-chain cross-check is a follow-up → Task 10 + noted as out-of-scope. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `WalletAgg`, `FeatureStat`, `ForwardResult`, `FEATURE_NAMES`,
`split_cohorts`, `compute_drift`, `discriminate`, `forward_test`, `main` names and
signatures match across tasks. `discriminate` uses keyword-only `drift=` everywhere
(Task 6 note flags the first test's call). `_won_expr()` reused in Tasks 2 and 7. ✓

**Note on forward-test scoring:** the sign-weighted score (Task 7) deliberately
excludes `mean_drift` (a per-trade-set quantity not available as a flat per-wallet
aggregate in the post-cutoff scoring path). This is a simple, defensible v1 scoring
rule; richer scoring is a future refinement, not a gap in this plan.
