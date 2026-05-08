# Issue #106 — Edge-Only Gate Defaults + Per-Pred-Bucket Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dominant `min_pred = 0.7` floor with a less-restrictive `min_pred = 0.5` sanity floor + a meaningful `min_edge_pct = 0.05` edge floor, and add per-pred-bucket telemetry to `pscanner paper status` so the variance question becomes empirical.

**Architecture:** Two config-default changes + one telemetry addition. The config edits flip three numbers (detector's `min_pred`, detector's `min_edge_pct`, evaluator's `min_edge_pct` — kept in sync because the evaluator double-checks). The telemetry queries `paper_trades` JOIN-ed to `alerts` and uses SQLite's `json_extract(body_json, '$.pred')` to bucket by the pre-computed prediction; this works without any schema change because `alerts.body_json` already carries `pred` for every `gate_buy` alert.

**Tech Stack:** Python 3.13, sqlite3 (with JSON1 extension — already in stdlib's bundled SQLite), rich (already used by `_cmd_paper_status`). No new deps.

---

## File Structure

- Modify: `src/pscanner/config.py:308-329` — flip `GateModelConfig.min_pred` and `min_edge_pct` defaults.
- Modify: `src/pscanner/config.py:418-427` — flip `GateModelEvaluatorConfig.min_edge_pct` default to match.
- Modify: `src/pscanner/store/repo.py` (after `summary_by_source` around line 2421) — add `summary_by_pred_bucket() -> list[PredBucketSummary]`.
- Modify: `src/pscanner/cli.py:434-454` and `:552-570` — wire the new aggregate into `_cmd_paper_status` and print it.
- Test: `tests/store/test_paper_trades_pred_bucket.py` (create).
- Test: `tests/test_cli_paper.py` (existing, extend) — confirm the new section renders when there are bucketed entries; otherwise the existing tests' silent skip behavior matches the no-data case.
- Modify: `CLAUDE.md` — gate-model bullet update.

Tasks 1, 2, 3 are independent. Task 4 (CLI wiring) depends on Task 3. Task 5 (CLAUDE.md) depends on the smoke run; the empirical numbers go in once an operator captures them.

---

## Task 1: Lower `min_pred` default to 0.5

**Files:**
- Modify: `src/pscanner/config.py:316-321`
- Test: none — config-default flip is data, not behavior. Existing tests that pin `min_pred=0.7` explicitly continue to work.

The detector reads `self._config.min_pred` at `gate_model.py:197`. Existing detector tests construct the config explicitly (e.g. `GateModelConfig(min_pred=0.7)`), so they're insulated from the default change. Search for any test that relies on the old default and update it inline.

- [ ] **Step 1: Find any test that relies on `min_pred = 0.7` as a default**

Run: `grep -rn "GateModelConfig\b" tests/ | grep -v "min_pred"`

Expected: most matches construct `GateModelConfig` with explicit `min_pred=` (insulated). Any match without `min_pred=` is at risk — read it to confirm whether the test depends on the old default. If it does, pin the value explicitly.

- [ ] **Step 2: Flip the default**

Edit `src/pscanner/config.py:316-321`. Replace:

```python
    enabled: bool = False
    artifact_dir: Path = Field(default=Path("models/current"))
    min_pred: float = 0.7
    min_edge_pct: float = 0.01
    accepted_categories: tuple[str, ...] | None = None
    queue_max_size: int = 1024
```

with:

```python
    enabled: bool = False
    artifact_dir: Path = Field(default=Path("models/current"))
    min_pred: float = 0.5
    """Sanity floor — never bet on outcomes the model thinks are <50% likely
    to win. The previous default ``0.7`` dominated the edge gate and
    systematically excluded the long-shot mispricing signal (high edge,
    low pred). See issue #106 for the realized-edge analysis behind this.
    """
    min_edge_pct: float = 0.05
    """Meaningful edge floor (5pp). Pairs with the lowered ``min_pred=0.5``
    so the gate does economic work instead of being a 1pp formality. See
    issue #106.
    """
    accepted_categories: tuple[str, ...] | None = None
    queue_max_size: int = 1024
```

- [ ] **Step 3: Run the gate-model detector test suite**

Run: `uv run pytest tests/detectors/test_gate_model.py -v`

Expected: all pass. If any test fails because it implicitly relied on the old default, pin `min_pred=0.7` and `min_edge_pct=0.01` explicitly in that test's config construction.

- [ ] **Step 4: Run lint + types**

Run: `uv run ruff check src/pscanner/config.py && uv run ruff format --check src/pscanner/config.py && uv run ty check src/pscanner/config.py`

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/config.py tests/detectors/test_gate_model.py
git commit -m "fix(gate_model): lower min_pred default to 0.5, raise min_edge_pct to 0.05 (#106)"
```

---

## Task 2: Sync `GateModelEvaluatorConfig.min_edge_pct` to 0.05

**Files:**
- Modify: `src/pscanner/config.py:418-427`
- Test: existing evaluator tests (review for default-dependent assertions).

The evaluator's `min_edge_pct` is a defensive double-check — it must match the detector's floor or the operator drift this guards against re-emerges in the opposite direction. Plan #106 explicitly calls these "must stay in sync."

- [ ] **Step 1: Flip the evaluator default**

Edit `src/pscanner/config.py:425-427`. Replace:

```python
    enabled: bool = False
    min_edge_pct: float = 0.01
    position_fraction: float = 0.005
```

with:

```python
    enabled: bool = False
    min_edge_pct: float = 0.05
    """Defensive double-check — must match ``GateModelConfig.min_edge_pct``.
    The evaluator gate guards against operator config drift between detector
    and evaluator (see CLAUDE.md gate-model evaluator bullet). Keeping these
    in sync is the operator contract; #106 raised both to 0.05.
    """
    position_fraction: float = 0.005
```

- [ ] **Step 2: Find existing tests that depend on the old default**

Run: `grep -rn "GateModelEvaluatorConfig\b" tests/ | grep -v "min_edge_pct"`

Expected: most matches construct the config explicitly. Pin the old value in any test that depended on `0.01` for behavior.

- [ ] **Step 3: Run the evaluator + paper-trader test suite**

Run: `uv run pytest tests/strategies/ -v`

Expected: all pass.

- [ ] **Step 4: Lint + types + commit**

```bash
uv run ruff check src/pscanner/config.py && uv run ruff format --check src/pscanner/config.py && uv run ty check src/pscanner/config.py
git add src/pscanner/config.py tests/strategies/
git commit -m "fix(gate_model_evaluator): raise min_edge_pct default to 0.05 (#106)"
```

---

## Task 3: Add `PaperTradesRepo.summary_by_pred_bucket`

**Files:**
- Modify: `src/pscanner/store/repo.py` (after `summary_by_source` around line 2421)
- Test: `tests/store/test_paper_trades_pred_bucket.py` (create)

A new aggregate method that bins gate-model paper-trade entries by their alert-body's `pred` field (extracted via SQLite JSON1) and reports per-bucket open / resolved / realized / win_rate. The bucket boundaries are `[0.5, 0.6) / [0.6, 0.7) / [0.7, 0.8) / [0.8, 0.9) / [0.9, 1.0]` per the issue's spec — note the closed upper bound on the last bucket so a `pred = 1.0` row lands in `0.9-1.0` (no NULL bucket).

The query is scoped to `triggering_alert_detector = 'gate_buy'` because non-gate alerts don't carry `pred` in their body.

- [ ] **Step 1: Write the failing tests**

Create `tests/store/test_paper_trades_pred_bucket.py`:

```python
"""Unit tests for PaperTradesRepo.summary_by_pred_bucket (#106)."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, cast

import pytest

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, PaperTradesRepo


@pytest.fixture
def conn(tmp_path):  # type: ignore[no-untyped-def]
    db_path = tmp_path / "daemon.sqlite3"
    c = init_db(db_path)
    yield c
    c.close()


def _book_gate_alert(
    conn: sqlite3.Connection,
    *,
    key: str,
    pred: float,
    resolved_pnl: float | None = None,
    ts: int | None = None,
) -> None:
    """Insert an alert and an entry; if ``resolved_pnl`` given, also an exit."""
    ts = ts or int(time.time())
    alert = Alert(
        detector=cast(DetectorName, "gate_buy"),
        alert_key=key,
        severity=cast(Severity, "med"),
        title=f"t-{key}",
        body={"pred": pred, "side": "YES", "implied_prob_at_buy": pred - 0.1},
        created_at=ts,
    )
    AlertsRepo(conn).insert_if_new(alert)
    paper = PaperTradesRepo(conn)
    paper.insert_entry(
        triggering_alert_key=key,
        triggering_alert_detector="gate_buy",
        rule_variant=None,
        source_wallet=None,
        condition_id=cast(Any, f"0xc-{key}"),
        asset_id=cast(Any, f"0xa-{key}"),
        outcome="YES",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=1000.0,
        ts=ts,
    )
    if resolved_pnl is not None:
        # Exit row: cost_usd carries the proceeds, so realized_pnl = exit.cost - entry.cost
        proceeds = 5.0 + resolved_pnl
        # Need the entry's trade_id to set parent_trade_id
        entry_id = conn.execute(
            "SELECT trade_id FROM paper_trades WHERE triggering_alert_key=?", (key,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO paper_trades (
                triggering_alert_key, triggering_alert_detector, rule_variant,
                source_wallet, condition_id, asset_id, outcome, shares,
                fill_price, cost_usd, nav_after_usd, ts, trade_kind, parent_trade_id
            ) VALUES (?, 'gate_buy', NULL, NULL, ?, ?, 'YES', 10.0, 1.0, ?, 1000.0, ?, 'exit', ?)
            """,
            (key, f"0xc-{key}", f"0xa-{key}", proceeds, ts + 100, entry_id),
        )
        conn.commit()


def test_pred_bucket_summary_returns_one_row_per_active_bucket(
    conn: sqlite3.Connection,
) -> None:
    """Buckets with no entries are omitted."""
    _book_gate_alert(conn, key="A", pred=0.55)  # 0.5-0.6
    _book_gate_alert(conn, key="B", pred=0.85)  # 0.8-0.9

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    labels = [r.bucket_label for r in rows]
    assert labels == ["0.5-0.6", "0.8-0.9"]


def test_pred_bucket_open_and_resolved_counts(conn: sqlite3.Connection) -> None:
    """Open vs resolved counts split correctly within a bucket."""
    _book_gate_alert(conn, key="open", pred=0.75)
    _book_gate_alert(conn, key="won", pred=0.75, resolved_pnl=+5.0)  # full payout
    _book_gate_alert(conn, key="lost", pred=0.75, resolved_pnl=-5.0)  # zero payout

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()
    assert len(rows) == 1
    bucket = rows[0]
    assert bucket.bucket_label == "0.7-0.8"
    assert bucket.open_count == 1
    assert bucket.resolved_count == 2
    assert bucket.win_rate == pytest.approx(0.5)
    assert bucket.realized_pnl == pytest.approx(0.0)


def test_pred_bucket_excludes_non_gate_buy(conn: sqlite3.Connection) -> None:
    """Only ``triggering_alert_detector = 'gate_buy'`` is bucketed."""
    # Insert a non-gate paper trade with no pred in body
    AlertsRepo(conn).insert_if_new(
        Alert(
            detector=cast(DetectorName, "smart_money"),
            alert_key="sm-1",
            severity=cast(Severity, "med"),
            title="t",
            body={"foo": "bar"},
            created_at=int(time.time()),
        ),
    )
    PaperTradesRepo(conn).insert_entry(
        triggering_alert_key="sm-1",
        triggering_alert_detector="smart_money",
        rule_variant=None,
        source_wallet=None,
        condition_id=cast(Any, "0xc"),
        asset_id=cast(Any, "0xa"),
        outcome="YES",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=1000.0,
        ts=int(time.time()),
    )

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    assert rows == []


def test_pred_bucket_pred_one_lands_in_top_bucket(conn: sqlite3.Connection) -> None:
    """``pred = 1.0`` exactly is bucketed into ``0.9-1.0`` (closed upper bound)."""
    _book_gate_alert(conn, key="exact-one", pred=1.0)

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    assert len(rows) == 1
    assert rows[0].bucket_label == "0.9-1.0"
    assert rows[0].open_count == 1


def test_pred_bucket_below_floor_omitted(conn: sqlite3.Connection) -> None:
    """``pred < 0.5`` does not appear (matches the new ``min_pred`` floor).

    Defensive: the detector should never emit such alerts post-#106, but if
    a stale model artifact does, the bucket query still tolerates them by
    filtering them out.
    """
    _book_gate_alert(conn, key="too-low", pred=0.3)

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/store/test_paper_trades_pred_bucket.py -v`

Expected: 5 FAILs — `AttributeError: 'PaperTradesRepo' object has no attribute 'summary_by_pred_bucket'`.

- [ ] **Step 3: Add the dataclass**

Edit `src/pscanner/store/repo.py`. Insert near `SourceSummary` (around line 2199):

```python
@dataclass(frozen=True, slots=True)
class PredBucketSummary:
    """Per-pred-bucket PnL aggregate for the gate-model paper status (#106)."""

    bucket_label: str  # e.g. "0.7-0.8"
    open_count: int
    resolved_count: int
    realized_pnl: float
    win_rate: float
```

- [ ] **Step 4: Add the repo method**

Edit `src/pscanner/store/repo.py`. Insert after `summary_by_source` (around line 2422, before the closing of the class):

```python
    def summary_by_pred_bucket(self) -> list[PredBucketSummary]:
        """Per-pred-bucket aggregate of open/resolved/realized PnL/win rate.

        Buckets gate-model paper-trade entries by ``alerts.body_json.$.pred``
        into 0.1-wide bins from ``[0.5, 0.6)`` through ``[0.9, 1.0]``. The
        upper bound on the top bucket is closed so ``pred = 1.0`` exactly
        is included. Buckets with zero entries are omitted (caller renders
        only what's present). Non-gate-model paper trades (no ``pred`` in
        body) are excluded by the ``triggering_alert_detector`` filter.

        See issue #106 for the variance/calibration analysis this enables.
        """
        rows = self._conn.execute(
            """
            SELECT
              CASE
                WHEN p < 0.6 THEN '0.5-0.6'
                WHEN p < 0.7 THEN '0.6-0.7'
                WHEN p < 0.8 THEN '0.7-0.8'
                WHEN p < 0.9 THEN '0.8-0.9'
                ELSE              '0.9-1.0'
              END AS bucket_label,
              SUM(CASE WHEN x.trade_id IS NULL THEN 1 ELSE 0 END) AS open_count,
              SUM(CASE WHEN x.trade_id IS NOT NULL THEN 1 ELSE 0 END) AS resolved_count,
              COALESCE(SUM(x.cost_usd - e.cost_usd), 0.0) AS realized_pnl,
              AVG(CASE WHEN x.trade_id IS NOT NULL
                       THEN CASE WHEN x.cost_usd > e.cost_usd THEN 1.0 ELSE 0.0 END
                       ELSE NULL END) AS win_rate
            FROM (
              SELECT e.*, CAST(json_extract(a.body_json, '$.pred') AS REAL) AS p
                FROM paper_trades e
                JOIN alerts a ON a.alert_key = e.triggering_alert_key
               WHERE e.trade_kind = 'entry'
                 AND e.triggering_alert_detector = 'gate_buy'
            ) e
            LEFT JOIN paper_trades x
              ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
            WHERE e.p >= 0.5
            GROUP BY bucket_label
            ORDER BY bucket_label
            """,
        ).fetchall()
        return [
            PredBucketSummary(
                bucket_label=str(r["bucket_label"]),
                open_count=int(r["open_count"] or 0),
                resolved_count=int(r["resolved_count"] or 0),
                realized_pnl=float(r["realized_pnl"] or 0.0),
                win_rate=float(r["win_rate"] or 0.0),
            )
            for r in rows
        ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/store/test_paper_trades_pred_bucket.py -v`

Expected: 5 PASS.

- [ ] **Step 6: Lint + types**

Run: `uv run ruff check src/pscanner/store/repo.py tests/store/test_paper_trades_pred_bucket.py && uv run ruff format --check src/pscanner/store/repo.py tests/store/test_paper_trades_pred_bucket.py && uv run ty check src/pscanner/store/repo.py tests/store/test_paper_trades_pred_bucket.py`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/store/repo.py tests/store/test_paper_trades_pred_bucket.py
git commit -m "feat(store): summary_by_pred_bucket for gate-model telemetry (#106)"
```

---

## Task 4: Wire bucket telemetry into `pscanner paper status`

**Files:**
- Modify: `src/pscanner/cli.py:434-454` (caller) + `:552-570` (printer)

Add a printer function and call it from `_cmd_paper_status` after `_print_paper_sources`. Match the existing `_print_paper_sources` shape so the output is visually consistent.

- [ ] **Step 1: Add the printer**

Edit `src/pscanner/cli.py`. Append to the printer block (after `_print_paper_sources` around line 571):

```python
def _print_paper_pred_buckets(
    console: Console,
    buckets: list[PredBucketSummary],
) -> None:
    """Render the per-pred-bucket gate-model breakdown, skipping when empty.

    Issue #106: makes the variance/calibration question empirical — operators
    can see whether the booked edge in the 0.5-0.6 bucket actually realizes
    over enough resolutions, vs. raising the floor on assumption.
    """
    if not buckets:
        return
    console.print("")
    console.print("Per-pred-bucket breakdown (gate_buy):")
    header = (
        f"  {'pred':<10s} "
        f"{'open':>5s} {'resolved':>9s} {'pnl':>9s} {'win_rate':>9s}"
    )
    console.print(header)
    for b in buckets:
        console.print(
            f"  {b.bucket_label:<10s} "
            f"{b.open_count:>5d} {b.resolved_count:>9d} "
            f"{b.realized_pnl:>+9.2f} {b.win_rate * 100:>8.1f}%",
        )
```

- [ ] **Step 2: Update the import**

Edit `src/pscanner/cli.py`. Find the import line for `SourceSummary` (search: `grep -n "SourceSummary" src/pscanner/cli.py`). Add `PredBucketSummary` alongside it:

```python
from pscanner.store.repo import (
    ...,
    PredBucketSummary,
    SourceSummary,
    ...,
)
```

(The `...` parts are placeholder for whatever else is in that import. Keep the alphabetical order if the file uses it.)

- [ ] **Step 3: Call the new printer from `_cmd_paper_status`**

Edit `src/pscanner/cli.py:434-454`. Replace the `_cmd_paper_status` body's tail:

```python
def _cmd_paper_status(config: Config) -> int:
    """Print paper-trading status (NAV, open/closed counts, realized PnL, top trades)."""
    conn = init_db(Path(config.scanner.db_path))
    try:
        paper = PaperTradesRepo(conn)
        summary = paper.summary_stats(
            starting_bankroll=config.paper_trading.starting_bankroll_usd,
        )
        leaderboard = _paper_leaderboard_rows(conn)
        best = _paper_extreme_rows(conn, order="DESC")
        worst = _paper_extreme_rows(conn, order="ASC")
        sources = paper.summary_by_source()
        pred_buckets = paper.summary_by_pred_bucket()
    finally:
        conn.close()
    console = Console(highlight=False)
    _print_paper_summary(console, summary)
    _print_paper_leaderboard(console, leaderboard)
    _print_paper_extremes(console, "top 3 best settled trades", best)
    _print_paper_extremes(console, "top 3 worst settled trades", worst)
    _print_paper_sources(console, sources)
    _print_paper_pred_buckets(console, pred_buckets)
    return 0
```

- [ ] **Step 4: Run any existing CLI test**

Run: `grep -l "_cmd_paper_status\|paper status\|paper_status" tests/`

If a test exists, run it: `uv run pytest tests/test_cli_paper.py -v` (or whatever path matched).

If no test exists for `_cmd_paper_status`, that's pre-existing — don't add one for this task. Telemetry rendering is hand-verified.

- [ ] **Step 5: Hand-verify with an in-memory DB**

Run a quick smoke from the project root:

```bash
uv run python -c "
import sqlite3, time, json
from pathlib import Path
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, PaperTradesRepo
from pscanner.alerts.models import Alert
conn = init_db(Path(':memory:'))
ar = AlertsRepo(conn)
pr = PaperTradesRepo(conn)
ts = int(time.time())
ar.insert_if_new(Alert(detector='gate_buy', alert_key='k1', severity='med', title='t', body={'pred': 0.55, 'side': 'YES', 'implied_prob_at_buy': 0.1}, created_at=ts))
pr.insert_entry(triggering_alert_key='k1', triggering_alert_detector='gate_buy', rule_variant=None, source_wallet=None, condition_id='0xc', asset_id='0xa', outcome='YES', shares=10.0, fill_price=0.5, cost_usd=5.0, nav_after_usd=1000.0, ts=ts)
print(pr.summary_by_pred_bucket())
"
```

Expected: a single `PredBucketSummary(bucket_label='0.5-0.6', open_count=1, resolved_count=0, realized_pnl=0.0, win_rate=0.0)`.

- [ ] **Step 6: Lint + types + commit**

```bash
uv run ruff check src/pscanner/cli.py && uv run ruff format --check src/pscanner/cli.py && uv run ty check src/pscanner/cli.py
git add src/pscanner/cli.py
git commit -m "feat(cli): per-pred-bucket telemetry in paper status (#106)"
```

---

## Task 5: Document the rationale in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (gate-model bullet)

The issue's acceptance criterion 4 asks for "the rationale and links to per-bucket numbers from a smoke run." The numbers come from a smoke run AFTER this lands; this task captures the rationale and leaves a placeholder pointing to the future smoke note.

- [ ] **Step 1: Update the gate-model bullet**

Edit `CLAUDE.md`. Locate the line `- **Gate-model loop (#77/#78/#79).** ...` (the same bullet edited by issues #101 and #102). Append at the end of that bullet:

```
Defaults shifted in #106: `min_pred` lowered from `0.7` to `0.5` (sanity floor only — never bet on outcomes the model thinks are <50% likely), `min_edge_pct` raised from `0.01` to `0.05` (meaningful edge floor, not a 1pp formality). The previous combination had `min_pred` dominating and excluding the long-shot mispricing signal. Per-pred-bucket telemetry in `pscanner paper status` lets operators decide empirically whether to raise the floor — drop a smoke-run snapshot here once the 0.5-0.6 bucket has 200+ resolutions.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): note #106 gate defaults + per-bucket telemetry"
```

---

## Verification

After all tasks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: zero failures, zero new warnings.

Hand-verification with the project's daemon DB (if any paper trades exist):

```bash
uv run pscanner paper status
```

Expected output includes a new "Per-pred-bucket breakdown (gate_buy):" table when there are gate_buy paper trades, OR omits the section silently when there are none. The header order on the existing sections is unchanged.

---

## Self-Review

**Spec coverage:**
- Acceptance criterion 1 (`min_pred` default lowered to 0.5): Task 1.
- Acceptance criterion 2 (`min_edge_pct` default raised to 0.05 — both detector AND evaluator): Tasks 1 + 2.
- Acceptance criterion 3 (per-pred-bucket telemetry in paper status): Tasks 3 + 4.
- Acceptance criterion 4 (CLAUDE.md updated with rationale): Task 5.

**Out-of-scope checks (per the issue):**
- Per-category `min_pred`: NOT addressed (v2).
- Fractional Kelly sizing: NOT addressed (separate Phase 2 issue).
- Per-decile gate policies: NOT addressed (data-driven decision deferred to after telemetry shows whether it's needed).

**Placeholder scan:** The Task 4 import update uses `...` placeholders to mean "keep the rest of the import line as-is." That's a structural pattern — the implementer is told to find the actual import via `grep`. Not a content placeholder.

The CLAUDE.md note in Task 5 says "drop a smoke-run snapshot here once the 0.5-0.6 bucket has 200+ resolutions" — that's a deliberately-deferred follow-up, not a missing acceptance criterion (the criterion only asks for the rationale text now).

**Type consistency:**
- `min_pred: float` and `min_edge_pct: float` are both already `float`; defaults change but types don't.
- `PredBucketSummary` follows the same dataclass shape as `SourceSummary` (frozen, slots, ints + floats + str).
- The CLI's `_print_paper_pred_buckets(console: Console, buckets: list[PredBucketSummary])` matches the existing `_print_paper_sources(console, sources: list[SourceSummary])` signature shape.
- The new SQL query uses SQLite's JSON1 `json_extract(body_json, '$.pred')` which is part of bundled SQLite in Python 3.13 — no driver change needed.
