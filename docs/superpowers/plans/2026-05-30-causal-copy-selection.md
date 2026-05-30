# Causal Wallet Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--causal-select` mode to `scripts/backtest_copy_sizing.py` that qualifies wallets at ≥N resolved trades, ranks them by causal (no-lookahead) edge, freezes a global top-K copy set on a rebalance cadence (positive-edge hard floor), and backtests copying only that evolving set — so results resemble live performance.

**Architecture:** A new `scripts/copy_selection.py` does the heavy lifting in DuckDB (resolved-buy fact table → per-boundary edge → ranked qualifiers), returns the same event-row tuples the existing Simulator already consumes. The per-boundary top-K cut is applied in Python (`resolve_k`) for testability, then pushed back as a DuckDB temp table to emit the selected-trade stream. The existing `Simulator`, four sizing schemes, and #204 capacity gates are unchanged.

**Tech Stack:** Python 3.13, DuckDB (`ATTACH ... TYPE sqlite`), sqlite3, pytest, uv/ruff/ty.

**Spec:** `docs/superpowers/specs/2026-05-30-causal-copy-selection-design.md`

---

## File structure

- **Create** `scripts/copy_selection.py` — DuckDB causal-selection precompute + selected-trade stream. Public surface: `KPolicy`, `resolve_k`, `has_platform_column`, `iter_selected_rows`.
- **Create** `tests/scripts/test_copy_selection.py` — unit + cross-validation tests for the selector.
- **Create** `tests/scripts/conftest.py` — shared `corpus_factory` fixture (builds a tiny corpus sqlite DB).
- **Modify** `scripts/backtest_copy_sizing.py` — add a shared `has_platform_column` use, refactor row→event into `_rows_to_events`, add `--causal-select` + new flags, branch `main`, add report selection-summary.
- **Modify** `tests/scripts/test_backtest_simulator.py` — parser tests for the new flags + an end-to-end causal-mode integration test.

The event-row tuple shape (used everywhere) matches the existing `load_event_stream` SELECT:
`(kind, ts, wallet, condition_id, outcome_side, price, notional_usd, outcome_yes_won)`.

---

## Task 1: Shared `corpus_factory` test fixture

**Files:**
- Create: `tests/scripts/conftest.py`

- [ ] **Step 1: Write the fixture**

```python
"""Shared fixtures for scripts tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

# trade tuple: (wallet, condition_id, outcome_side, bs, price, notional_usd, ts)
# resolution tuple: (condition_id, outcome_yes_won, resolved_at)
Trade = tuple[str, str, str, str, float, float, int]
Resolution = tuple[str, int, int]


def _build(path: Path, trades: list[Trade], resolutions: list[Resolution],
           *, with_platform: bool) -> None:
    plat_col = "platform TEXT NOT NULL DEFAULT 'polymarket'," if with_platform else ""
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""
        CREATE TABLE corpus_trades (
          {plat_col}
          tx_hash TEXT NOT NULL, asset_id TEXT NOT NULL,
          wallet_address TEXT NOT NULL, condition_id TEXT NOT NULL,
          outcome_side TEXT NOT NULL, bs TEXT NOT NULL,
          price REAL NOT NULL, size REAL NOT NULL,
          notional_usd REAL NOT NULL, ts INTEGER NOT NULL
        );
        CREATE TABLE market_resolutions (
          {plat_col}
          condition_id TEXT NOT NULL, winning_outcome_index INTEGER NOT NULL,
          outcome_yes_won INTEGER NOT NULL, resolved_at INTEGER NOT NULL,
          source TEXT NOT NULL, recorded_at INTEGER NOT NULL
        );
        """
    )
    prefix = "'polymarket'," if with_platform else ""
    for i, (w, cid, side, bs, price, notional, ts) in enumerate(trades):
        conn.execute(
            f"INSERT INTO corpus_trades VALUES ({prefix}?,?,?,?,?,?,?,?,?,?)",
            (f"0xtx{i}", f"asset{i}", w, cid, side, bs, price,
             notional / price, notional, ts),
        )
    for cid, yes_won, resolved_at in resolutions:
        conn.execute(
            f"INSERT INTO market_resolutions VALUES ({prefix}?,?,?,?,'test',?)",
            (cid, 0 if yes_won else 1, yes_won, resolved_at, resolved_at),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def corpus_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a builder: corpus_factory(trades, resolutions, with_platform=True) -> db path."""
    counter = {"n": 0}

    def make(trades: list[Trade], resolutions: list[Resolution],
             *, with_platform: bool = True) -> Path:
        counter["n"] += 1
        db = tmp_path / f"corpus{counter['n']}.sqlite3"
        _build(db, trades, resolutions, with_platform=with_platform)
        return db

    return make
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `uv run pytest tests/scripts/ -q --co`
Expected: collection succeeds (no errors), existing tests still listed.

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/conftest.py
git commit -m "test(scripts): shared corpus_factory fixture"
```

---

## Task 2: `KPolicy` + `resolve_k`

**Files:**
- Create: `scripts/copy_selection.py`
- Test: `tests/scripts/test_copy_selection.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the causal copy-selection precompute."""

from __future__ import annotations

import pytest

from scripts.copy_selection import KPolicy, resolve_k


def test_resolve_k_fixed_count() -> None:
    assert resolve_k(KPolicy(top_k=25), bankroll=10_000.0, qualified_count=500) == 25


def test_resolve_k_capital_per_wallet_floors() -> None:
    # 10_000 / 750 = 13.33 -> 13
    assert resolve_k(KPolicy(capital_per_wallet=750.0), bankroll=10_000.0,
                     qualified_count=500) == 13


def test_resolve_k_top_frac_ceils_against_qualified() -> None:
    # ceil(0.1 * 95) = 10
    assert resolve_k(KPolicy(top_frac=0.1), bankroll=10_000.0, qualified_count=95) == 10


def test_resolve_k_top_frac_zero_qualified_is_zero() -> None:
    assert resolve_k(KPolicy(top_frac=0.1), bankroll=10_000.0, qualified_count=0) == 0


def test_resolve_k_no_mode_raises() -> None:
    with pytest.raises(ValueError, match="no mode"):
        resolve_k(KPolicy(), bankroll=10_000.0, qualified_count=10)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_copy_selection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.copy_selection'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Causal copy-trading wallet selection precompute (DuckDB-backed).

Qualifies wallets at >= min_resolved resolved trades, ranks by causal
(no-lookahead) edge, freezes a global top-K copy set per rebalance
boundary, and emits the selected trades + their resolutions as event
rows for scripts.backtest_copy_sizing's Simulator.

Spec: docs/superpowers/specs/2026-05-30-causal-copy-selection-design.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_copy_selection.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/copy_selection.py tests/scripts/test_copy_selection.py
git commit -m "feat(copy-selection): KPolicy + resolve_k"
```

---

## Task 3: `has_platform_column`

**Files:**
- Modify: `scripts/copy_selection.py`
- Test: `tests/scripts/test_copy_selection.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from scripts.copy_selection import has_platform_column


def test_has_platform_column_true(corpus_factory) -> None:
    db: Path = corpus_factory([], [], with_platform=True)
    assert has_platform_column(db) is True


def test_has_platform_column_false(corpus_factory) -> None:
    db: Path = corpus_factory([], [], with_platform=False)
    assert has_platform_column(db) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k has_platform -q`
Expected: FAIL — `ImportError: cannot import name 'has_platform_column'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/copy_selection.py` (and `import sqlite3` / `from pathlib import Path` at top):

```python
def has_platform_column(db_path: Path) -> bool:
    """Return True if corpus_trades carries the multi-platform `platform` column."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(corpus_trades)")]
    conn.close()
    return "platform" in cols
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k has_platform -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/copy_selection.py tests/scripts/test_copy_selection.py
git commit -m "feat(copy-selection): has_platform_column detection"
```

---

## Task 4: `_ranked_qualifiers` — per-boundary edge, lifetime (the causal core)

This builds the DuckDB pipeline up to the ranked, qualified, positive-edge wallet list per boundary. It returns rows `(boundary_ts, wallet, edge, n_resolved, rank, n_qualified)` ordered by `(boundary_ts, rank)`.

**Files:**
- Modify: `scripts/copy_selection.py`
- Test: `tests/scripts/test_copy_selection.py`

- [ ] **Step 1: Write the failing test (cross-validation vs brute-force Python)**

```python
from scripts.copy_selection import ranked_qualifiers


def _brute_edge(trades, resolutions, boundary_ts, *, min_resolved, window_days):
    """Reference: causal mean(won - price) per wallet at one boundary."""
    res = {cid: (yw, rt) for cid, yw, rt in resolutions}
    per_wallet: dict[str, list[tuple[int, float]]] = {}
    for w, cid, side, bs, price, _notional, ts in trades:
        if bs != "BUY" or cid not in res:
            continue
        yw, rt = res[cid]
        if not (ts <= rt and rt < boundary_ts):
            continue
        if window_days and rt < boundary_ts - window_days * 86400:
            continue
        won = 1.0 if ((yw == 1 and side == "YES") or (yw == 0 and side == "NO")) else 0.0
        per_wallet.setdefault(w, []).append((rt, won - price))
    out = {}
    for w, recs in per_wallet.items():
        if len(recs) < min_resolved:
            continue
        edge = sum(d for _, d in recs) / len(recs)
        if edge > 0:
            out[w] = edge
    return out


def test_ranked_qualifiers_matches_bruteforce_lifetime(corpus_factory) -> None:
    # Two wallets: A strongly positive, B negative. min_resolved=2 for a tiny case.
    trades = [
        ("A", "m1", "YES", "BUY", 0.40, 100.0, 10),
        ("A", "m2", "YES", "BUY", 0.30, 100.0, 20),
        ("B", "m3", "YES", "BUY", 0.80, 100.0, 15),
        ("B", "m4", "YES", "BUY", 0.70, 100.0, 25),
    ]
    resolutions = [("m1", 1, 100), ("m2", 1, 110), ("m3", 0, 120), ("m4", 0, 130)]
    boundary = 200
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=0,
        boundaries=[boundary],
    )
    got = {w: edge for (b, w, edge, n, rk, nq) in rows if b == boundary}
    expected = _brute_edge(trades, resolutions, boundary, min_resolved=2, window_days=0)
    assert set(got) == set(expected)  # only A qualifies (positive edge); B excluded
    for w in expected:
        assert got[w] == pytest.approx(expected[w])
    # rank is deterministic and 1-based
    a_row = next(r for r in rows if r[1] == "A")
    assert a_row[4] == 1


def test_ranked_qualifiers_excludes_below_min_resolved(corpus_factory) -> None:
    trades = [("A", "m1", "YES", "BUY", 0.40, 100.0, 10)]  # only 1 resolved
    resolutions = [("m1", 1, 100)]
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=0, boundaries=[200],
    )
    assert rows == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k ranked_qualifiers -q`
Expected: FAIL — `ImportError: cannot import name 'ranked_qualifiers'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/copy_selection.py` (and `import duckdb`, `from collections.abc import Sequence` at top):

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k ranked_qualifiers -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/copy_selection.py tests/scripts/test_copy_selection.py
git commit -m "feat(copy-selection): ranked per-boundary qualifiers (causal, positive-edge floor)"
```

---

## Task 5: No-lookahead + rolling-window coverage for `ranked_qualifiers`

**Files:**
- Modify: `tests/scripts/test_copy_selection.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_ranked_qualifiers_no_lookahead(corpus_factory) -> None:
    # A's trades resolve at 250/260, AFTER boundary 200 -> A not qualified at 200.
    trades = [
        ("A", "m1", "YES", "BUY", 0.40, 100.0, 10),
        ("A", "m2", "YES", "BUY", 0.30, 100.0, 20),
    ]
    resolutions = [("m1", 1, 250), ("m2", 1, 260)]
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=0, boundaries=[200],
    )
    assert rows == []  # no resolutions before the boundary


def test_ranked_qualifiers_rolling_window_drops_old_trades(corpus_factory) -> None:
    # window=1 day: at boundary 200 (ts), only trades resolved within [200-86400, 200).
    # Put 2 old resolutions far in the past and 1 recent -> under min_resolved=2 in window.
    day = 86400
    trades = [
        ("A", "m1", "YES", "BUY", 0.40, 100.0, 1),
        ("A", "m2", "YES", "BUY", 0.40, 100.0, 2),
        ("A", "m3", "YES", "BUY", 0.40, 100.0, 3),
    ]
    boundary = 10 * day
    resolutions = [("m1", 1, 1 * day), ("m2", 1, 2 * day), ("m3", 1, boundary - 100)]
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=1, boundaries=[boundary],
    )
    assert rows == []  # only 1 trade inside the 1-day window -> below min_resolved
```

- [ ] **Step 2: Run to verify failure first, then pass**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k "no_lookahead or rolling_window" -q`
Expected: PASS (these exercise already-implemented logic; if either fails, the bug is in Task 4's SQL — fix there, do not weaken the test).

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_copy_selection.py
git commit -m "test(copy-selection): no-lookahead + rolling-window qualifier coverage"
```

---

## Task 6: `iter_selected_rows` — copy-set freeze → selected event stream

Applies `resolve_k` per boundary to `ranked_qualifiers`, builds the frozen copy set, and yields event rows (trades + their resolutions) ordered by ts.

**Files:**
- Modify: `scripts/copy_selection.py`
- Test: `tests/scripts/test_copy_selection.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts.copy_selection import iter_selected_rows


def test_iter_selected_rows_only_copies_top_k_in_frozen_period(corpus_factory) -> None:
    # Boundaries every 100s from ts=0. A & B both qualify positive by boundary 100;
    # A has higher edge. top_k=1 -> only A copied. Each makes a NEW trade in [100,200).
    trades = [
        # qualifying history (resolves before boundary 100)
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("B", "h3", "YES", "BUY", 0.45, 100.0, 1),
        ("B", "h4", "YES", "BUY", 0.45, 100.0, 2),
        # new trades inside period [100,200)
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
        ("B", "n2", "YES", "BUY", 0.50, 100.0, 160),
    ]
    resolutions = [
        ("h1", 1, 50), ("h2", 1, 60), ("h3", 1, 50), ("h4", 1, 60),
        ("n1", 1, 300), ("n2", 1, 300),
    ]
    rows = list(iter_selected_rows(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=0,
        rebalance_days=None, rebalance_seconds=100, policy=KPolicy(top_k=1),
        bankroll=10_000.0, start_ts=None, end_ts=None,
    ))
    trade_rows = [r for r in rows if r[0] == "trade"]
    copied_new = {r[2] for r in trade_rows if r[3] in ("n1", "n2")}
    assert copied_new == {"A"}  # only top-1 wallet A copied; B excluded
    # resolutions for copied markets are present and stream is ts-ordered
    assert any(r[0] == "resolution" and r[3] == "n1" for r in rows)
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k iter_selected -q`
Expected: FAIL — `ImportError: cannot import name 'iter_selected_rows'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/copy_selection.py` (add `from collections.abc import Iterator` at top):

```python
def _make_boundaries(con: duckdb.DuckDBPyConnection, *, period: int,
                     start_ts: int | None, end_ts: int | None) -> list[int]:
    """Boundary grid spanning the trade ts range (or the provided window), step=period."""
    lo, hi = con.execute("SELECT MIN(ts), MAX(ts) FROM rb").fetchone()
    if lo is None:
        return []
    lo = start_ts if start_ts is not None else int(lo)
    hi = end_ts if end_ts is not None else int(hi)
    return list(range(lo, hi + 1, period))


def iter_selected_rows(
    db_path: Path, *, platform: str, min_resolved: int, edge_window_days: int,
    rebalance_days: int | None, policy: KPolicy, bankroll: float,
    start_ts: int | None, end_ts: int | None, rebalance_seconds: int | None = None,
) -> Iterator[tuple]:
    """Yield (kind, ts, wallet, condition_id, outcome_side, price, notional_usd,
    outcome_yes_won) rows for the causally-selected copy stream, ts-ordered.

    rebalance_seconds overrides rebalance_days (tests use small windows).
    """
    period = rebalance_seconds if rebalance_seconds is not None else rebalance_days * 86400
    has_plat = has_platform_column(db_path)
    con = _attach(db_path)
    try:
        _build_rb(con, platform=platform, has_platform=has_plat)
        boundaries = _make_boundaries(con, period=period, start_ts=start_ts, end_ts=end_ts)
        if not boundaries:
            return
        con.close()  # ranked_qualifiers reopens its own connection
        con = None
        ranked = ranked_qualifiers(
            db_path, platform=platform, min_resolved=min_resolved,
            edge_window_days=edge_window_days, boundaries=boundaries,
        )
        # apply per-boundary K cut
        n_qual_by_b: dict[int, int] = {}
        for b, _w, _e, _n, _rk, nq in ranked:
            n_qual_by_b[b] = nq
        copyset: list[tuple[int, str]] = []
        for b, w, _e, _n, rk, _nq in ranked:
            k = resolve_k(policy, bankroll=bankroll, qualified_count=n_qual_by_b[b])
            if rk <= k:
                copyset.append((b, w))
        if not copyset:
            return
        yield from _stream_selected(
            db_path, platform=platform, has_platform=has_plat, period=period,
            copyset=copyset,
        )
    finally:
        if con is not None:
            con.close()


def _stream_selected(
    db_path: Path, *, platform: str, has_platform: bool, period: int,
    copyset: list[tuple[int, str]],
) -> Iterator[tuple]:
    tpred = "AND t.platform = ?" if has_platform else ""
    rpred = "AND r.platform = ?" if has_platform else ""
    tparams = [platform] if has_platform else []
    rparams = [platform] if has_platform else []
    con = _attach(db_path)
    try:
        con.execute("CREATE TEMP TABLE copyset(boundary_ts BIGINT, wallet VARCHAR)")
        con.executemany("INSERT INTO copyset VALUES (?, ?)", copyset)
        query = f"""
            WITH selected AS (
              SELECT t.wallet_address AS wallet, t.condition_id AS condition_id,
                     t.outcome_side AS outcome_side, t.price AS price,
                     t.notional_usd AS notional_usd, t.ts AS ts
              FROM s.corpus_trades t
              JOIN copyset cs ON cs.wallet = t.wallet_address
                AND t.ts >= cs.boundary_ts AND t.ts < cs.boundary_ts + {period}
              WHERE t.bs = 'BUY' {tpred}
            ),
            sel_res AS (
              SELECT r.condition_id AS condition_id, r.outcome_yes_won AS outcome_yes_won,
                     r.resolved_at AS ts
              FROM s.market_resolutions r
              WHERE r.condition_id IN (SELECT DISTINCT condition_id FROM selected) {rpred}
            )
            SELECT 'trade' AS kind, ts, wallet, condition_id, outcome_side,
                   price, notional_usd, NULL AS outcome_yes_won FROM selected
            UNION ALL
            SELECT 'resolution' AS kind, ts, NULL, condition_id, NULL,
                   NULL, NULL, outcome_yes_won FROM sel_res
            ORDER BY ts ASC
        """  # noqa: S608 -- period/predicates are fixed literals; values via ? params
        cur = con.execute(query, [*tparams, *rparams])
        while True:
            batch = cur.fetchmany(100_000)
            if not batch:
                break
            yield from batch
    finally:
        con.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_copy_selection.py -k iter_selected -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/copy_selection.py tests/scripts/test_copy_selection.py
git commit -m "feat(copy-selection): iter_selected_rows freeze + selected stream"
```

---

## Task 7: Edge-case coverage for `iter_selected_rows`

**Files:**
- Modify: `tests/scripts/test_copy_selection.py`

- [ ] **Step 1: Write the tests**

```python
def test_iter_selected_rows_empty_universe(corpus_factory) -> None:
    rows = list(iter_selected_rows(
        corpus_factory([], []), platform="polymarket", min_resolved=20,
        edge_window_days=0, rebalance_days=None, rebalance_seconds=100,
        policy=KPolicy(top_k=5), bankroll=10_000.0, start_ts=None, end_ts=None,
    ))
    assert rows == []


def test_iter_selected_rows_k_larger_than_qualified(corpus_factory) -> None:
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
    ]
    resolutions = [("h1", 1, 50), ("h2", 1, 60), ("n1", 1, 300)]
    rows = list(iter_selected_rows(
        corpus_factory(trades, resolutions), platform="polymarket", min_resolved=2,
        edge_window_days=0, rebalance_days=None, rebalance_seconds=100,
        policy=KPolicy(top_k=50), bankroll=10_000.0, start_ts=None, end_ts=None,
    ))
    copied = {r[2] for r in rows if r[0] == "trade" and r[3] == "n1"}
    assert copied == {"A"}  # only 1 qualifies; K=50 just takes all qualified


def test_iter_selected_rows_no_platform_corpus(corpus_factory) -> None:
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
    ]
    resolutions = [("h1", 1, 50), ("h2", 1, 60), ("n1", 1, 300)]
    rows = list(iter_selected_rows(
        corpus_factory(trades, resolutions, with_platform=False),
        platform="polymarket", min_resolved=2, edge_window_days=0,
        rebalance_days=None, rebalance_seconds=100, policy=KPolicy(top_k=5),
        bankroll=10_000.0, start_ts=None, end_ts=None,
    ))
    assert any(r[0] == "trade" and r[3] == "n1" for r in rows)
```

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/scripts/test_copy_selection.py -q`
Expected: PASS (all copy_selection tests green).

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_copy_selection.py
git commit -m "test(copy-selection): empty universe, K>qualified, no-platform corpus"
```

---

## Task 8: Refactor `backtest_copy_sizing` row→event into a shared helper

Avoids duplicating row→event construction between the watchlist and causal paths.

**Files:**
- Modify: `scripts/backtest_copy_sizing.py` (the `load_event_stream` body, ~lines 398-421)

- [ ] **Step 1: Add the shared helper and call it from `load_event_stream`**

Add this module-level function (place it just above `load_event_stream`):

```python
def _rows_to_events(rows: Iterable[tuple]) -> Iterator[TradeEvent | ResolutionEvent]:
    """Convert event-row tuples into TradeEvent / ResolutionEvent objects.

    Row shape: (kind, ts, wallet, condition_id, outcome_side, price,
    notional_usd, outcome_yes_won).
    """
    for kind, ts, wallet, cid, side, price, notional, yes_won in rows:
        if kind == "trade":
            yield TradeEvent(
                kind="trade", ts=int(ts),
                trade=Trade(wallet=str(wallet), condition_id=str(cid),
                            outcome_side=str(side), price=float(price),
                            notional_usd=float(notional), ts=int(ts)),
            )
        else:
            yield ResolutionEvent(
                kind="resolution", ts=int(ts),
                resolution=Resolution(condition_id=str(cid),
                                      winning_side="YES" if int(yes_won) == 1 else "NO",
                                      resolved_at=int(ts)),
            )
```

Then replace the trailing `for kind, ts, ... yield ...` loop at the end of `load_event_stream` with:

```python
    rows = con.execute(query, [platform, *params, platform]).fetchall()
    con.close()
    yield from _rows_to_events(rows)
```

Add `Iterable` to the existing `from collections.abc import Iterator, Sequence` import → `from collections.abc import Iterable, Iterator, Sequence`.

- [ ] **Step 2: Run existing stream tests to verify no behavior change**

Run: `uv run pytest tests/scripts/test_backtest_simulator.py -k load_event_stream -q`
Expected: PASS (all existing `load_event_stream` tests still green).

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_copy_sizing.py
git commit -m "refactor(backtest): extract _rows_to_events shared helper"
```

---

## Task 9: Add `--causal-select` + selection flags to the parser

**Files:**
- Modify: `scripts/backtest_copy_sizing.py` (`build_parser`)
- Test: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_parser_causal_select_defaults() -> None:
    args = build_parser().parse_args(["--causal-select"])
    assert args.causal_select is True
    assert args.min_resolved == 20
    assert args.edge_window == 0
    assert args.rebalance_days == 14
    assert args.copy_top_k is None
    assert args.copy_capital_per_wallet is None
    assert args.copy_top_frac is None


def test_build_parser_causal_off_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.causal_select is False


def test_build_parser_rejects_two_copy_policies() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--copy-top-k", "10", "--copy-top-frac", "0.1"])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_backtest_simulator.py -k "causal or copy_polic" -q`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'causal_select'`.

- [ ] **Step 3: Add the flags in `build_parser`** (just before the `--csv` argument)

```python
    p.add_argument("--causal-select", action="store_true",
                   help="Causally qualify+rank+select wallets from the corpus"
                        " (ignores --watchlist-db).")
    p.add_argument("--min-resolved", type=int, default=20,
                   help="Qualification: min resolved trades within the edge window.")
    p.add_argument("--edge-window", type=int, default=0,
                   help="Rolling edge window in days; 0 = lifetime.")
    p.add_argument("--rebalance-days", type=int, default=14,
                   help="Days between top-K copy-set recomputes.")
    copy_policy = p.add_mutually_exclusive_group()
    copy_policy.add_argument("--copy-top-k", type=int, default=None,
                             help="Copy the top N wallets by causal edge.")
    copy_policy.add_argument("--copy-capital-per-wallet", type=float, default=None,
                             help="K = floor(bankroll / C).")
    copy_policy.add_argument("--copy-top-frac", type=float, default=None,
                             help="Copy the top X fraction of qualified wallets.")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_backtest_simulator.py -k "causal or copy_polic" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): --causal-select + selection flags"
```

---

## Task 10: Wire the causal stream into `main` + report selection summary

**Files:**
- Modify: `scripts/backtest_copy_sizing.py` (`main`, `render_report`/header, add `_resolve_policy`)
- Test: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing integration test**

```python
def test_main_causal_select_end_to_end(corpus_factory, capsys) -> None:
    # A qualifies positive and is copied; B never positive -> never copied.
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("B", "h3", "YES", "BUY", 0.90, 100.0, 1),
        ("B", "h4", "YES", "BUY", 0.90, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
        ("B", "n2", "YES", "BUY", 0.50, 100.0, 160),
    ]
    resolutions = [
        ("h1", 1, 50), ("h2", 1, 60), ("h3", 0, 50), ("h4", 0, 60),
        ("n1", 1, 300), ("n2", 1, 300),
    ]
    db = corpus_factory(trades, resolutions)
    rc = main([
        "--db", str(db), "--causal-select", "--min-resolved", "2",
        "--rebalance-days", "1", "--copy-top-k", "5",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Causal selection" in out
    assert "equal_weight" in out
```

(Note: `--rebalance-days 1` → 86400s period; the test's ts values fall in the first
period since they are < 86400.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/scripts/test_backtest_simulator.py -k causal_select_end_to_end -q`
Expected: FAIL — `--causal-select` is parsed but `main` ignores it / no "Causal selection" header.

- [ ] **Step 3: Implement the branch + summary**

Add the import at the top of `backtest_copy_sizing.py`: `from scripts import copy_selection`.

Add a policy resolver just above `main` (defaults to top-k 25 so causal mode always has a
policy; `ConcentrationCapped`'s anti-concentration target uses the copy-set size as the
watchlist-size proxy, computed as `wl_size` below):

```python
def _resolve_policy(args: argparse.Namespace) -> copy_selection.KPolicy:
    """Build the KPolicy from CLI args, defaulting to top-k 25."""
    if args.copy_capital_per_wallet is not None:
        return copy_selection.KPolicy(capital_per_wallet=args.copy_capital_per_wallet)
    if args.copy_top_frac is not None:
        return copy_selection.KPolicy(top_frac=args.copy_top_frac)
    return copy_selection.KPolicy(top_k=args.copy_top_k if args.copy_top_k is not None else 25)
```

Then replace the body of `main` (everything after `args = build_parser().parse_args(argv)`)
with this single concrete block, which decides the mode, builds the schemes, runs the walk,
and prints the report:

```python
    if args.causal_select:
        wl_size = args.copy_top_k or 25
    else:
        watchlist = load_watchlist(Path(args.watchlist_db))
        if not watchlist:
            print("Watchlist is empty; nothing to backtest.", file=sys.stderr)
            return 1
        wl_size = len(watchlist)
    schemes = _build_schemes(args, wl_size)
    max_exposure = args.max_open_exposure_usd
    if args.max_open_exposure_frac is not None:
        max_exposure = args.max_open_exposure_frac * args.starting_bankroll_usd
    sim = Simulator(schemes=schemes, bankroll=args.starting_bankroll_usd,
                    enforce_capacity=args.enforce_capacity,
                    max_open_exposure_usd=max_exposure)
    if args.causal_select:
        policy = _resolve_policy(args)
        events = _rows_to_events(copy_selection.iter_selected_rows(
            Path(args.db), platform=args.platform, min_resolved=args.min_resolved,
            edge_window_days=args.edge_window, rebalance_days=args.rebalance_days,
            policy=policy, bankroll=args.starting_bankroll_usd,
            start_ts=args.start_ts, end_ts=args.end_ts))
        selection_header = (
            f"Causal selection: min_resolved={args.min_resolved}, "
            f"edge_window={args.edge_window}d, rebalance_days={args.rebalance_days}, "
            f"policy={policy}\n")
    else:
        events = load_event_stream(Path(args.db), watchlist=watchlist,
                                   platform=args.platform, start_ts=args.start_ts,
                                   end_ts=args.end_ts)
        selection_header = ""
    for event in events:
        if isinstance(event, TradeEvent):
            sim.on_trade(event.trade)
        else:
            sim.on_resolution(event.resolution)
    print(selection_header + render_report(
        sim, schemes=schemes, bankroll=args.starting_bankroll_usd,
        enforce_capacity=args.enforce_capacity, max_open_exposure_usd=max_exposure))
    if args.csv:
        _write_csv(sim, schemes, args.csv)
    return 0
```

Remove the now-duplicated original `watchlist = load_watchlist(...)` / loop / print block.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/scripts/test_backtest_simulator.py -k causal_select_end_to_end -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): wire causal-select stream into main + selection header"
```

---

## Task 11: Full gate — lint, format, type-check, whole suite

**Files:** none (verification only)

- [ ] **Step 1: Run the project gate on the changed files**

Run:
```bash
uv run ruff check scripts/copy_selection.py scripts/backtest_copy_sizing.py tests/scripts/
uv run ruff format --check scripts/copy_selection.py scripts/backtest_copy_sizing.py tests/scripts/
uv run ty check scripts/copy_selection.py scripts/backtest_copy_sizing.py
uv run pytest tests/scripts/ -q
```
Expected: ruff clean, format clean, ty "All checks passed!", all tests pass.

- [ ] **Step 2: Fix any findings**

Fix lint/format/type issues inline (e.g. add `# noqa: S608` justifications already present; ensure `Iterable` import added in Task 8). Re-run until clean. Do not weaken tests to pass.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore(copy-selection): satisfy ruff/ty gate"
```

---

## Task 12: Smoke against the real desktop corpus (manual, operator step)

**Files:** none (operator validation; not a unit test)

- [ ] **Step 1: Run a small causal backtest on the desktop corpus**

From the laptop (the desktop has the 69 GB corpus + the merged branch — pull this branch there first):
```bash
ssh -p 2222 macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner && export PATH="$HOME/.local/bin:$PATH" && \
   uv run python scripts/backtest_copy_sizing.py --db data/corpus.sqlite3 \
     --causal-select --min-resolved 20 --edge-window 0 --rebalance-days 14 \
     --copy-top-k 25 --enforce-capacity --max-open-exposure-frac 1 \
     > /tmp/causal_backtest.txt 2>&1; echo exit=$?'
```
Expected: completes in ~3–5 min, exit=0, report prints with the "Causal selection" header and a much lower "Skipped" count than the 1,790-wallet runs.

- [ ] **Step 2: Sanity-check the output**

Confirm: trade counts are non-zero, win rate is plausible (not 100%), and the headline ROI is far more conservative than the in-sample 1,790-wallet run. Record numbers for comparison.

---

## Self-review notes

- **Spec coverage:** universe pre-filter (Task 4 `_build_rb` + min_resolved) ✓; causal edge lifetime (Task 4) ✓ / rolling (Task 5) ✓; qualification-within-window (Tasks 4-5) ✓; positive-edge floor (Task 4 `HAVING ... > 0`) ✓; top-K freeze + three policies (Tasks 2, 6) ✓; deterministic tiebreak (Task 4 `ORDER BY edge DESC, wallet ASC`) ✓; freeze-on-trade-ts (Task 6 `_stream_selected` join) ✓; no-lookahead two guards (Task 4 `t.ts <= r.resolved_at` + `rb.resolved_at < boundary_ts`) ✓; platform detection both paths (Task 3 + Task 8 reused) ✓; CLI surface (Task 9) ✓; report header (Task 10) ✓; perf tactics — read-only attach, memory_limit, fetchmany (Tasks 4, 6) ✓; testing incl. cross-validation (Task 4) ✓.
- **Deviation noted:** the per-boundary K cut is applied in Python (`resolve_k`) rather than SQL, for testability — the heavy edge/qualification stays in DuckDB, faithful to the design's intent. The "Not selected" per-scheme counter from the spec is realized instead as the selection header + the (already-present) capacity `Skipped` column, because non-selected trades never enter Python in the DuckDB-first design — so a per-scheme not-selected count isn't meaningful; the header conveys the selector's filtering.
- **Type consistency:** row tuple shape `(kind, ts, wallet, condition_id, outcome_side, price, notional_usd, outcome_yes_won)` is identical in `copy_selection._stream_selected`, `backtest_copy_sizing._rows_to_events`, and `load_event_stream`. `KPolicy` fields (`top_k`, `capital_per_wallet`, `top_frac`) match across `resolve_k`, `_resolve_policy`, and the parser flags.
