# Paper-Trading Expansion (multi-signal `SignalEvaluator`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `PaperTrader` from a smart_money-only orchestrator to a thin pipeline that fans every alert through per-detector `SignalEvaluator` classes — shipping four evaluators (smart_money, move_attribution, velocity-with-twin-trades, mispricing-with-detector-enrichment), per-source PnL tracking columns, constant-size betting decoupled from realized PnL, and a per-source CLI breakdown.

**Architecture:** A `SignalEvaluator` Protocol (`accepts`, `parse`, `quality_passes`, `size`) lives in `src/pscanner/strategies/evaluators/protocol.py`. Four concrete classes (one per source) live in sibling files. `PaperTrader.evaluate` walks a list of evaluators, picks the first acceptor, runs the pipeline, and inserts one paper_trade per `ParsedSignal` returned. Sizing reads `starting_bankroll_usd` (constant) instead of running NAV; the `bankroll_exhausted` gate is removed. The mispricing detector is enriched with `target_*` body fields so its alerts become tradeable. Two new columns (`triggering_alert_detector`, `rule_variant`) on `paper_trades` enable per-source PnL queries.

**Tech Stack:** Python 3.13, sqlite3, pydantic, structlog, pytest + pytest-asyncio. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-04-27-paper-trading-expansion-design.md`

---

## File Structure

**Create:**
- `src/pscanner/strategies/evaluators/__init__.py` — re-export public names.
- `src/pscanner/strategies/evaluators/protocol.py` — `SignalEvaluator` Protocol + `ParsedSignal` dataclass.
- `src/pscanner/strategies/evaluators/smart_money.py` — `SmartMoneyEvaluator`.
- `src/pscanner/strategies/evaluators/move_attribution.py` — `MoveAttributionEvaluator`.
- `src/pscanner/strategies/evaluators/velocity.py` — `VelocityEvaluator` (twin trades).
- `src/pscanner/strategies/evaluators/mispricing.py` — `MispricingEvaluator`.
- `tests/strategies/evaluators/__init__.py` — empty package marker.
- `tests/strategies/evaluators/test_smart_money.py`
- `tests/strategies/evaluators/test_move_attribution.py`
- `tests/strategies/evaluators/test_velocity.py`
- `tests/strategies/evaluators/test_mispricing.py`

**Modify:**
- `src/pscanner/store/db.py` — schema migration (2 new columns + new unique-on-entry index + backfill).
- `src/pscanner/store/repo.py` — `insert_entry` signature, `OpenPaperPosition` dataclass, new `summary_by_source` method.
- `src/pscanner/config.py` — add `EvaluatorsConfig` section + sub-blocks; remove `position_fraction` and `min_weighted_edge` from `PaperTradingConfig`.
- `src/pscanner/detectors/mispricing.py` — enrich alert body with `target_*` fields via proportional rebalancing.
- `src/pscanner/strategies/paper_trader.py` — refactor into evaluator orchestrator; drop `bankroll_exhausted` gate.
- `src/pscanner/scheduler.py` — construct evaluators, pass list to `PaperTrader`.
- `src/pscanner/cli.py` — `paper status` per-source breakdown table.
- `tests/test_config.py` — defaults for new config sections.
- `tests/store/test_paper_trades_repo.py` — new columns roundtrip + index update.
- `tests/detectors/test_mispricing.py` — assert `target_*` fields in emitted bodies.
- `tests/strategies/test_paper_trader.py` — refactor for evaluator-list ctor.
- `tests/test_cli.py` — assert per-source breakdown.

## Task ordering

8 sequential tasks. Each is small enough for a single subagent and produces a working partial state.

| # | Task | Touches | Depends on |
|---|------|---------|------------|
| 1 | Protocol + ParsedSignal | new `protocol.py` + `__init__.py` + import-only test | — |
| 2 | DB schema + repo | `db.py`, `repo.py`, repo tests | — |
| 3 | EvaluatorsConfig | `config.py`, `tests/test_config.py` | — |
| 4 | Mispricing detector enrichment | `mispricing.py`, `tests/detectors/test_mispricing.py` | — |
| 5 | SmartMoneyEvaluator + MoveAttributionEvaluator + MispricingEvaluator | 3 evaluator files + 3 tests | T1, T3, T4 |
| 6 | VelocityEvaluator (twin trades) | `velocity.py` + tests | T1, T3 |
| 7 | PaperTrader refactor | `paper_trader.py`, repo `summary_by_source`, paper_trader tests | T1, T2, T3, T5, T6 |
| 8 | Scheduler wiring + CLI | `scheduler.py`, `cli.py`, scheduler/CLI tests | T7 |

---

## Task 1: SignalEvaluator Protocol + ParsedSignal

Smallest task. Pure type addition — Protocol + dataclass + package init. No behavior; future tasks pull in.

**Files:**
- Create: `src/pscanner/strategies/evaluators/__init__.py`
- Create: `src/pscanner/strategies/evaluators/protocol.py`
- Create: `tests/strategies/evaluators/__init__.py`
- Create: `tests/strategies/evaluators/test_protocol.py`

- [ ] **Step 1.1: Create `src/pscanner/strategies/evaluators/__init__.py`**

```python
"""Per-detector :class:`SignalEvaluator` implementations.

PaperTrader walks a list of evaluators on each alert; the first one whose
``accepts`` returns ``True`` runs the parse → quality → size pipeline.
"""

from pscanner.strategies.evaluators.protocol import (
    ParsedSignal,
    SignalEvaluator,
)

__all__ = ["ParsedSignal", "SignalEvaluator"]
```

- [ ] **Step 1.2: Create `src/pscanner/strategies/evaluators/protocol.py`**

```python
"""``SignalEvaluator`` Protocol — contract every per-detector evaluator implements.

PaperTrader fans every alert through a list of evaluators. The first one
whose ``accepts`` returns ``True`` parses the alert into one or more
``ParsedSignal`` instances; each signal is independently quality-gated,
sized, and booked as a paper_trade row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pscanner.alerts.models import Alert
from pscanner.poly.ids import ConditionId


@dataclass(frozen=True, slots=True)
class ParsedSignal:
    """One tradeable direction extracted from a single alert.

    Attributes:
        condition_id: Market identifier the entry will be booked against.
        side: Outcome name (e.g. ``"yes"``, ``"Trump"``) used by
            ``PaperTrader._resolve_outcome`` to look up the asset_id and
            fill price via :class:`MarketCacheRepo`.
        rule_variant: ``"follow"``/``"fade"`` for velocity twin-trades;
            ``None`` for single-entry sources.
        metadata: Pass-through bag of fields each evaluator may stash for
            its own ``quality_passes`` (e.g. SmartMoney stores ``wallet``).
    """

    condition_id: ConditionId
    side: str
    rule_variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalEvaluator(Protocol):
    """Contract for one per-detector trade-signal extractor.

    Each method is called by ``PaperTrader.evaluate`` in order:

    1. ``accepts(alert)`` — does this evaluator handle this alert's detector?
    2. ``parse(alert)`` — extract zero-or-more :class:`ParsedSignal` instances
       (zero on body-shape mismatch; one for single-entry sources; two for
       velocity twin-trades).
    3. ``quality_passes(parsed)`` — per-signal quality gate.
    4. ``size(bankroll, parsed)`` — return cost in USD. Bankroll is
       ``starting_bankroll_usd`` (constant), not running NAV — sizing is
       independent of cumulative PnL by design.
    """

    def accepts(self, alert: Alert) -> bool:
        """Return ``True`` iff this evaluator handles ``alert.detector``."""
        ...

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Extract one or more :class:`ParsedSignal` from ``alert``.

        Returns ``[]`` on body-shape mismatch (treated as a soft failure).
        """
        ...

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Per-signal quality gate; ``False`` skips this signal."""
        ...

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return USD cost for this signal. ``bankroll`` is constant per run."""
        ...
```

- [ ] **Step 1.3: Create `tests/strategies/evaluators/__init__.py`**

```python
"""Tests for the per-detector :class:`SignalEvaluator` implementations."""
```

- [ ] **Step 1.4: Create `tests/strategies/evaluators/test_protocol.py`**

```python
"""Smoke tests for the Protocol surface (import + dataclass roundtrip)."""

from __future__ import annotations

from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal, SignalEvaluator


def test_parsed_signal_roundtrip() -> None:
    """ParsedSignal stores its fields and exposes them by attribute."""
    sig = ParsedSignal(
        condition_id=ConditionId("0xabc"),
        side="yes",
        rule_variant="follow",
        metadata={"wallet": "0x123"},
    )
    assert sig.condition_id == "0xabc"
    assert sig.side == "yes"
    assert sig.rule_variant == "follow"
    assert sig.metadata == {"wallet": "0x123"}


def test_parsed_signal_default_rule_variant_is_none() -> None:
    """Single-entry sources omit rule_variant; defaults to None."""
    sig = ParsedSignal(condition_id=ConditionId("0xabc"), side="no")
    assert sig.rule_variant is None
    assert sig.metadata == {}


def test_signal_evaluator_protocol_importable() -> None:
    """Protocol type is importable; structural-only — no runtime check."""
    # Ensures the import path is stable for downstream evaluators.
    assert SignalEvaluator is not None
```

- [ ] **Step 1.5: Run, verify they pass**

```bash
uv run pytest tests/strategies/evaluators/test_protocol.py -v
```

Expected: 3 passed.

- [ ] **Step 1.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/strategies/evaluators/ tests/strategies/evaluators/
uv run ruff format --check src/pscanner/strategies/evaluators/ tests/strategies/evaluators/
uv run ty check src/pscanner/strategies/evaluators/
```

Expected: clean.

- [ ] **Step 1.7: Commit**

```bash
git add src/pscanner/strategies/evaluators/ tests/strategies/evaluators/
git commit -m "feat(evaluators): add SignalEvaluator Protocol + ParsedSignal

Foundation for the paper-trading expansion. Concrete evaluators land in
following commits. Zero behaviour change today.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: DB schema + repo

Adds two columns (`triggering_alert_detector`, `rule_variant`) to `paper_trades`, replaces the unique-on-entry index with one keyed on `(triggering_alert_key, COALESCE(rule_variant, ''))`, backfills the existing rows. Updates the repo's `insert_entry` signature + `OpenPaperPosition` dataclass to surface the new fields. Adds `summary_by_source` for per-detector PnL queries.

**Files:**
- Modify: `src/pscanner/store/db.py`
- Modify: `src/pscanner/store/repo.py`
- Modify: `tests/store/test_paper_trades_repo.py`

- [ ] **Step 2.1: Locate the schema constants in `src/pscanner/store/db.py`**

```bash
grep -n "_SCHEMA_STATEMENTS\|_MIGRATIONS\|paper_trades" src/pscanner/store/db.py
```

Identify:
- The `CREATE TABLE paper_trades (...)` line in `_SCHEMA_STATEMENTS` (the original schema for fresh DBs).
- The `_MIGRATIONS` tuple of idempotent `ALTER TABLE` strings (applied to existing DBs).
- The existing UNIQUE index on `triggering_alert_key WHERE trade_kind='entry'`.

- [ ] **Step 2.2: Write a failing test for the new columns**

Append to `tests/store/test_paper_trades_repo.py`:

```python
def test_insert_entry_records_detector_and_variant(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    trade_id = repo.insert_entry(
        triggering_alert_key="vel:0xa:1",
        triggering_alert_detector="velocity",
        rule_variant="follow",
        source_wallet=None,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0,
        fill_price=0.5,
        cost_usd=2.5,
        nav_after_usd=1000.0,
        ts=1700000000,
    )
    assert trade_id > 0

    rows = list(tmp_db.execute(
        "SELECT triggering_alert_detector, rule_variant FROM paper_trades WHERE trade_id = ?",
        (trade_id,),
    ))
    assert rows == [("velocity", "follow")]
```

- [ ] **Step 2.3: Write a failing test for the index update**

```python
def test_unique_entry_index_allows_paired_velocity(tmp_db: sqlite3.Connection) -> None:
    """Two velocity entries with the same alert_key but different rule_variants
    both succeed; same key + same variant raises IntegrityError."""
    repo = PaperTradesRepo(tmp_db)
    repo.insert_entry(
        triggering_alert_key="vel:0xa:1",
        triggering_alert_detector="velocity",
        rule_variant="follow",
        source_wallet=None,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0, fill_price=0.5, cost_usd=2.5, nav_after_usd=1000.0,
        ts=1700000000,
    )
    repo.insert_entry(
        triggering_alert_key="vel:0xa:1",
        triggering_alert_detector="velocity",
        rule_variant="fade",
        source_wallet=None,
        condition_id=ConditionId("0xc"),
        asset_id=AssetId("a-n"),
        outcome="no",
        shares=20.0, fill_price=0.5, cost_usd=2.5, nav_after_usd=1000.0,
        ts=1700000000,
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_entry(
            triggering_alert_key="vel:0xa:1",
            triggering_alert_detector="velocity",
            rule_variant="follow",  # duplicate of the first
            source_wallet=None,
            condition_id=ConditionId("0xc"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0, fill_price=0.5, cost_usd=2.5, nav_after_usd=1000.0,
            ts=1700000000,
        )
```

- [ ] **Step 2.4: Write a failing test for the backfill of pre-existing rows**

```python
def test_existing_rows_backfilled_to_smart_money(tmp_db: sqlite3.Connection) -> None:
    """A row inserted with NULL triggering_alert_detector gets backfilled to
    'smart_money' on next migration apply (simulates upgrade from old schema)."""
    # Insert a raw row bypassing the new repo signature by directly executing.
    # NOTE: this test relies on the migration step being idempotent — it
    # assumes _apply_migrations runs again post-fixture-init and backfills.
    tmp_db.execute(
        """
        INSERT INTO paper_trades (
          trade_kind, triggering_alert_key, source_wallet, condition_id,
          asset_id, outcome, shares, fill_price, cost_usd, nav_after_usd, ts,
          triggering_alert_detector
        ) VALUES ('entry', 'smart:0xa:1', '0xa', '0xc', 'a-y', 'yes',
                  20.0, 0.5, 10.0, 990.0, 1700000000, NULL)
        """,
    )
    tmp_db.commit()
    # Re-run the backfill explicitly. The production path runs on init.
    from pscanner.store.db import _apply_migrations
    _apply_migrations(tmp_db)

    detector = tmp_db.execute(
        "SELECT triggering_alert_detector FROM paper_trades WHERE triggering_alert_key = 'smart:0xa:1'"
    ).fetchone()[0]
    assert detector == "smart_money"
```

- [ ] **Step 2.5: Write a failing test for `summary_by_source`**

```python
def test_summary_by_source_groups_correctly(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    # Three entries: smart_money (no variant), velocity follow, velocity fade.
    e1 = repo.insert_entry(
        triggering_alert_key="smart:0xa:1",
        triggering_alert_detector="smart_money",
        rule_variant=None,
        source_wallet="0xa",
        condition_id=ConditionId("0xc1"),
        asset_id=AssetId("a-y"),
        outcome="yes",
        shares=20.0, fill_price=0.5, cost_usd=10.0, nav_after_usd=1000.0,
        ts=1700000000,
    )
    e2 = repo.insert_entry(
        triggering_alert_key="vel:0xb:1",
        triggering_alert_detector="velocity",
        rule_variant="follow",
        source_wallet=None,
        condition_id=ConditionId("0xc2"),
        asset_id=AssetId("b-y"),
        outcome="yes",
        shares=10.0, fill_price=0.25, cost_usd=2.5, nav_after_usd=1000.0,
        ts=1700000010,
    )
    e3 = repo.insert_entry(
        triggering_alert_key="vel:0xb:1",
        triggering_alert_detector="velocity",
        rule_variant="fade",
        source_wallet=None,
        condition_id=ConditionId("0xc2"),
        asset_id=AssetId("b-n"),
        outcome="no",
        shares=10.0, fill_price=0.25, cost_usd=2.5, nav_after_usd=1000.0,
        ts=1700000010,
    )
    # Resolve e1 as a win, e2 as a win, e3 as a loss.
    repo.insert_exit(
        parent_trade_id=e1, condition_id=ConditionId("0xc1"), asset_id=AssetId("a-y"),
        outcome="yes", shares=20.0, fill_price=1.0, cost_usd=20.0,
        nav_after_usd=1010.0, ts=1700000100,
    )
    repo.insert_exit(
        parent_trade_id=e2, condition_id=ConditionId("0xc2"), asset_id=AssetId("b-y"),
        outcome="yes", shares=10.0, fill_price=1.0, cost_usd=10.0,
        nav_after_usd=1017.5, ts=1700000200,
    )
    repo.insert_exit(
        parent_trade_id=e3, condition_id=ConditionId("0xc2"), asset_id=AssetId("b-n"),
        outcome="no", shares=10.0, fill_price=0.0, cost_usd=0.0,
        nav_after_usd=1015.0, ts=1700000200,
    )

    rows = repo.summary_by_source(starting_bankroll=1000.0)
    by_key = {(r.detector, r.rule_variant): r for r in rows}

    assert by_key[("smart_money", None)].resolved_count == 1
    assert by_key[("smart_money", None)].realized_pnl == pytest.approx(10.0)
    assert by_key[("smart_money", None)].win_rate == pytest.approx(1.0)

    assert by_key[("velocity", "follow")].resolved_count == 1
    assert by_key[("velocity", "follow")].realized_pnl == pytest.approx(7.5)
    assert by_key[("velocity", "follow")].win_rate == pytest.approx(1.0)

    assert by_key[("velocity", "fade")].resolved_count == 1
    assert by_key[("velocity", "fade")].realized_pnl == pytest.approx(-2.5)
    assert by_key[("velocity", "fade")].win_rate == pytest.approx(0.0)
```

- [ ] **Step 2.6: Run, verify all four tests fail**

```bash
uv run pytest tests/store/test_paper_trades_repo.py -v -k "records_detector or unique_entry_index_allows_paired or backfilled or summary_by_source"
```

Expected: failures (missing kwargs / methods / columns).

- [ ] **Step 2.7: Update `_SCHEMA_STATEMENTS` in `src/pscanner/store/db.py`**

Find the `CREATE TABLE paper_trades (...)` block. Add two new column declarations (the column position in the body is open — append after `ts` for clarity):

```
triggering_alert_detector TEXT,
rule_variant              TEXT,
```

Find the existing `CREATE UNIQUE INDEX idx_paper_trades_alert_key ON paper_trades(triggering_alert_key) WHERE trade_kind='entry'` (or whatever the current name is — locate via `grep "idx_paper_trades_alert_key" src/pscanner/store/db.py`). Replace with:

```
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trades_alert_key
ON paper_trades(triggering_alert_key, COALESCE(rule_variant, ''))
WHERE trade_kind='entry' AND triggering_alert_key IS NOT NULL
```

The `COALESCE(rule_variant, '')` is critical: SQLite UNIQUE indexes treat
NULLs as distinct, so without COALESCE two `(key, NULL)` rows would both
be permitted — defeating the per-key uniqueness for non-twin sources.

- [ ] **Step 2.8: Update `_MIGRATIONS` in `src/pscanner/store/db.py`**

Append (in order — the index DROP must come before the new CREATE):

```python
"ALTER TABLE paper_trades ADD COLUMN triggering_alert_detector TEXT",
"ALTER TABLE paper_trades ADD COLUMN rule_variant TEXT",
"DROP INDEX IF EXISTS idx_paper_trades_alert_key",
(
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trades_alert_key "
    "ON paper_trades(triggering_alert_key, COALESCE(rule_variant, '')) "
    "WHERE trade_kind='entry' AND triggering_alert_key IS NOT NULL"
),
"UPDATE paper_trades SET triggering_alert_detector = 'smart_money' WHERE triggering_alert_detector IS NULL AND trade_kind = 'entry'",
```

The existing `_apply_migrations` swallows `OperationalError` for "duplicate column name" and "no such column" so each line is idempotent on re-runs.

- [ ] **Step 2.9: Update `OpenPaperPosition` and `insert_entry` in `src/pscanner/store/repo.py`**

In `OpenPaperPosition` (around line 2168), append two fields:

```python
@dataclass(frozen=True, slots=True)
class OpenPaperPosition:
    """An entry row in ``paper_trades`` with no matching exit."""

    trade_id: int
    triggering_alert_key: str | None
    source_wallet: str | None
    condition_id: ConditionId
    asset_id: AssetId
    outcome: str
    shares: float
    fill_price: float
    cost_usd: float
    nav_after_usd: float
    ts: int
    triggering_alert_detector: str | None = None
    rule_variant: str | None = None
```

In `PaperTradesRepo.insert_entry` (around line 2208), add two kwargs and write them:

```python
def insert_entry(
    self,
    *,
    triggering_alert_key: str | None,
    triggering_alert_detector: str | None,
    rule_variant: str | None,
    source_wallet: str | None,
    condition_id: ConditionId,
    asset_id: AssetId,
    outcome: str,
    shares: float,
    fill_price: float,
    cost_usd: float,
    nav_after_usd: float,
    ts: int,
) -> int:
    """..."""
    cur = self._conn.execute(
        """
        INSERT INTO paper_trades (
          trade_kind, triggering_alert_key, parent_trade_id, source_wallet,
          condition_id, asset_id, outcome, shares, fill_price, cost_usd,
          nav_after_usd, ts, triggering_alert_detector, rule_variant
        ) VALUES ('entry', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            triggering_alert_key,
            source_wallet,
            condition_id,
            asset_id,
            outcome,
            shares,
            fill_price,
            cost_usd,
            nav_after_usd,
            ts,
            triggering_alert_detector,
            rule_variant,
        ),
    )
    self._conn.commit()
    return int(cur.lastrowid or 0)
```

Update any read paths (`list_open_positions`) to populate the two new fields. Locate via `grep "OpenPaperPosition(" src/pscanner/store/repo.py`. The SELECT must include `triggering_alert_detector, rule_variant`; the dataclass construction passes them through.

- [ ] **Step 2.10: Add `summary_by_source` to `PaperTradesRepo`**

Add a new dataclass `SourceSummary` near the top of `repo.py` (with the other dataclasses):

```python
@dataclass(frozen=True, slots=True)
class SourceSummary:
    """Per-(detector, rule_variant) PnL aggregate for the paper status CLI."""

    detector: str | None
    rule_variant: str | None
    open_count: int
    resolved_count: int
    realized_pnl: float
    win_rate: float
```

Add the method to `PaperTradesRepo`:

```python
def summary_by_source(self, *, starting_bankroll: float) -> list[SourceSummary]:
    """Per-source aggregate of open count, resolved count, realized PnL, win rate."""
    del starting_bankroll  # reserved for future allocation analysis
    rows = self._conn.execute(
        """
        SELECT
          e.triggering_alert_detector AS detector,
          e.rule_variant AS rule_variant,
          SUM(CASE WHEN x.trade_id IS NULL THEN 1 ELSE 0 END) AS open_count,
          SUM(CASE WHEN x.trade_id IS NOT NULL THEN 1 ELSE 0 END) AS resolved_count,
          COALESCE(SUM(x.cost_usd - e.cost_usd), 0.0) AS realized_pnl,
          AVG(CASE WHEN x.trade_id IS NOT NULL
                   THEN CASE WHEN x.cost_usd > e.cost_usd THEN 1.0 ELSE 0.0 END
                   ELSE NULL END) AS win_rate
        FROM paper_trades e
        LEFT JOIN paper_trades x
          ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
        WHERE e.trade_kind = 'entry'
        GROUP BY e.triggering_alert_detector, e.rule_variant
        ORDER BY e.triggering_alert_detector, e.rule_variant
        """,
    ).fetchall()
    return [
        SourceSummary(
            detector=r["detector"],
            rule_variant=r["rule_variant"],
            open_count=int(r["open_count"] or 0),
            resolved_count=int(r["resolved_count"] or 0),
            realized_pnl=float(r["realized_pnl"] or 0.0),
            win_rate=float(r["win_rate"] or 0.0),
        )
        for r in rows
    ]
```

- [ ] **Step 2.11: Update existing callers of `insert_entry`**

`grep -rn "paper_trades.insert_entry\|self._paper_trades.insert_entry" src/ tests/ | head -20`. Existing callers (paper_trader.py, paper_resolver.py won't call insert_entry — only insert_exit; tests/test_cli.py; tests/store/test_paper_trades_repo.py) need the two new kwargs added.

For tests that are setting up fixtures (CLI test, repo test), pass `triggering_alert_detector="smart_money"` and `rule_variant=None` — preserves existing behavior.

For `paper_trader.py`, leave alone — Task 7 will refactor that file, and adding the kwargs there now would conflict.

Actually: `paper_trader.py` does call `insert_entry` today. The cleanest path:
1. Update the signature to require the new kwargs (no defaults — caller must pass).
2. Pass `triggering_alert_detector="smart_money"`, `rule_variant=None` from paper_trader.py's existing single call site at `paper_trader.py:200-212`.
3. Task 7 will reuse those kwargs but populate from the alert/parsed signal.

Edit `src/pscanner/strategies/paper_trader.py` `_insert_entry` body to pass the two new kwargs — for now just hardcoded:

```python
self._paper_trades.insert_entry(
    triggering_alert_key=alert.alert_key,
    triggering_alert_detector="smart_money",  # T7 will replace with alert.detector
    rule_variant=None,  # T7 will replace with parsed.rule_variant
    source_wallet=wallet,
    # ... existing kwargs ...
)
```

- [ ] **Step 2.12: Run tests, verify they pass**

```bash
uv run pytest tests/store/test_paper_trades_repo.py -v
uv run pytest tests/strategies/test_paper_trader.py -q
```

Expected: previously-failing tests now pass; existing tests still pass.

- [ ] **Step 2.13: Run the full suite for regressions**

```bash
uv run pytest -q
```

Expected: all green. Anything failing in `tests/test_cli.py` due to missing kwargs gets the same `triggering_alert_detector="smart_money"` + `rule_variant=None` injection.

- [ ] **Step 2.14: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/store/ src/pscanner/strategies/paper_trader.py tests/store/test_paper_trades_repo.py tests/strategies/test_paper_trader.py
uv run ruff format --check src/pscanner/store/ src/pscanner/strategies/paper_trader.py tests/store/test_paper_trades_repo.py
uv run ty check src/pscanner/store/ src/pscanner/strategies/paper_trader.py
```

- [ ] **Step 2.15: Commit**

```bash
git add src/pscanner/store/ src/pscanner/strategies/paper_trader.py tests/store/test_paper_trades_repo.py tests/strategies/test_paper_trader.py tests/test_cli.py
git commit -m "feat(store): paper_trades gains detector + rule_variant columns

New nullable columns triggering_alert_detector and rule_variant enable
per-source PnL attribution. The unique-on-entry index moves to
(triggering_alert_key, COALESCE(rule_variant, '')) so velocity twin-trades
(same key, distinct variant) coexist while non-twin sources keep
per-key uniqueness.

OpenPaperPosition exposes the new fields. PaperTradesRepo.summary_by_source
joins entries to exits and aggregates open count, resolved count,
realized PnL, and win rate per (detector, rule_variant). Existing rows
backfilled to detector='smart_money' on migration apply.

PaperTrader call site updated to pass the new kwargs as constants for
now (T7 refactor wires them to the parsed signal).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: EvaluatorsConfig

Adds the `EvaluatorsConfig` section (four sub-blocks) to `Config`, removes `position_fraction` and `min_weighted_edge` from `PaperTradingConfig`. The PaperTrader config-read sites get updated to pull from the new path; values are unchanged at default.

**Files:**
- Modify: `src/pscanner/config.py`
- Modify: `src/pscanner/strategies/paper_trader.py` (read path only — sizing logic refactor lands in T7)
- Modify: `tests/test_config.py`

- [ ] **Step 3.1: Write failing tests for the new defaults**

Append to `tests/test_config.py`:

```python
def test_evaluators_config_defaults() -> None:
    from pscanner.config import (
        Config,
        EvaluatorsConfig,
        MispricingEvaluatorConfig,
        MoveAttributionEvaluatorConfig,
        SmartMoneyEvaluatorConfig,
        VelocityEvaluatorConfig,
    )

    cfg = EvaluatorsConfig()
    assert cfg.smart_money == SmartMoneyEvaluatorConfig()
    assert cfg.move_attribution == MoveAttributionEvaluatorConfig()
    assert cfg.velocity == VelocityEvaluatorConfig()
    assert cfg.mispricing == MispricingEvaluatorConfig()

    sm = SmartMoneyEvaluatorConfig()
    assert sm.enabled is True
    assert sm.position_fraction == 0.01
    assert sm.min_weighted_edge == 0.0

    ma = MoveAttributionEvaluatorConfig()
    assert ma.enabled is True
    assert ma.position_fraction == 0.01
    assert ma.min_severity == "med"
    assert ma.min_wallets == 3

    v = VelocityEvaluatorConfig()
    assert v.enabled is True
    assert v.position_fraction == 0.0025
    assert v.min_severity == "high"
    assert v.allow_consolidation is False

    m = MispricingEvaluatorConfig()
    assert m.enabled is True
    assert m.position_fraction == 0.01
    assert m.min_edge_dollars == 0.05

    root = Config()
    assert root.paper_trading.evaluators == cfg


def test_paper_trading_config_no_longer_has_position_fraction() -> None:
    """The old `position_fraction` and `min_weighted_edge` fields are removed
    from PaperTradingConfig — they live under evaluators.smart_money now."""
    from pscanner.config import PaperTradingConfig
    cfg = PaperTradingConfig()
    assert not hasattr(cfg, "position_fraction"), (
        "position_fraction must move to evaluators.smart_money.position_fraction"
    )
    assert not hasattr(cfg, "min_weighted_edge"), (
        "min_weighted_edge must move to evaluators.smart_money.min_weighted_edge"
    )
```

- [ ] **Step 3.2: Run, verify they fail**

```bash
uv run pytest tests/test_config.py -v -k "evaluators_config_defaults or paper_trading_config_no_longer"
```

Expected: ImportError / AttributeError / equality failures.

- [ ] **Step 3.3: Add `EvaluatorsConfig` and sub-blocks to `src/pscanner/config.py`**

Place after `PaperTradingConfig` (around line 282-296) and before `class Config(BaseModel)` (around line 299). Use `from typing import Literal` (verify it's already imported; add if not).

```python
class SmartMoneyEvaluatorConfig(_Section):
    """Smart-money copy-trade evaluator tunables.

    Today's PaperTrader config — moved here so each source has its own
    config block with the same shape (enabled, position_fraction, +
    source-specific gates).
    """

    enabled: bool = True
    position_fraction: float = 0.01
    min_weighted_edge: float = 0.0


class MoveAttributionEvaluatorConfig(_Section):
    """Move-attribution evaluator tunables.

    Trades the (outcome, side) pair surfaced by MoveAttributionDetector
    when ≥ ``min_wallets`` distinct wallets converged on a market in
    the burst window.
    """

    enabled: bool = True
    position_fraction: float = 0.01
    min_severity: Literal["low", "med", "high"] = "med"
    min_wallets: int = 3


class VelocityEvaluatorConfig(_Section):
    """Velocity twin-trade evaluator tunables.

    Each velocity alert spawns two ParsedSignals (follow + fade) at the
    per-side ``position_fraction`` (default 0.25%, pair total 0.5%).
    Constant size off ``starting_bankroll_usd``, not running NAV.
    """

    enabled: bool = True
    position_fraction: float = 0.0025
    min_severity: Literal["low", "med", "high"] = "high"
    allow_consolidation: bool = False


class MispricingEvaluatorConfig(_Section):
    """Mispricing arbitrage evaluator tunables.

    Trades the most-overpriced or most-underpriced YES leg of a mispriced
    event, using the detector-emitted ``target_*`` body fields. The
    edge gate is the gap between fair (proportional rebalance) and
    current price.
    """

    enabled: bool = True
    position_fraction: float = 0.01
    min_edge_dollars: float = 0.05


class EvaluatorsConfig(_Section):
    """Container for the four per-source evaluator configs.

    Disabling a source via its ``enabled`` flag prevents that Evaluator
    from being constructed at scheduler boot — no detector code path
    changes; the alert is simply not handled by anyone.
    """

    smart_money: SmartMoneyEvaluatorConfig = Field(default_factory=SmartMoneyEvaluatorConfig)
    move_attribution: MoveAttributionEvaluatorConfig = Field(default_factory=MoveAttributionEvaluatorConfig)
    velocity: VelocityEvaluatorConfig = Field(default_factory=VelocityEvaluatorConfig)
    mispricing: MispricingEvaluatorConfig = Field(default_factory=MispricingEvaluatorConfig)
```

- [ ] **Step 3.4: Modify `PaperTradingConfig`**

Find `PaperTradingConfig` (around line 282) and remove `position_fraction` and `min_weighted_edge`; add `evaluators`:

```python
class PaperTradingConfig(_Section):
    """Thresholds + cadence for the paper-trading subsystem.

    Off by default. When enabled, PaperTrader subscribes to AlertSink and
    fans every alert through the evaluators list to mirror trades onto a
    virtual bankroll. PaperResolver runs as a periodic detector that books
    PnL when the underlying market resolves. State lives in ``paper_trades``.

    Per-source tunables (enabled, position_fraction, quality gates) live
    under ``evaluators.<source>``.
    """

    enabled: bool = False
    starting_bankroll_usd: float = 1000.0
    min_position_cost_usd: float = 0.50
    resolver_scan_interval_seconds: float = 300.0
    evaluators: EvaluatorsConfig = Field(default_factory=EvaluatorsConfig)
```

Verify `Config` does NOT need a separate `evaluators` field at the top level — `evaluators` lives under `paper_trading`.

- [ ] **Step 3.5: Update `paper_trader.py` config-read sites**

The current PaperTrader reads `self._config.position_fraction` and `self._config.min_weighted_edge`. These move to `self._config.evaluators.smart_money.*`. Locate the read sites:

```bash
grep -n "position_fraction\|min_weighted_edge" src/pscanner/strategies/paper_trader.py
```

For each, update the path. Example:

```python
# before:
cost = nav * cfg.position_fraction
# after:
cost = nav * cfg.evaluators.smart_money.position_fraction
```

```python
# before:
if edge is None or edge <= self._config.min_weighted_edge:
# after:
if edge is None or edge <= self._config.evaluators.smart_money.min_weighted_edge:
```

This is a *temporary* shape — Task 7 will move these reads into `SmartMoneyEvaluator`. For now we keep PaperTrader behavior unchanged with the new config path.

- [ ] **Step 3.6: Run tests, verify they pass**

```bash
uv run pytest tests/test_config.py -v -k "evaluators_config_defaults or paper_trading_config_no_longer"
uv run pytest tests/strategies/test_paper_trader.py -q
uv run pytest -q
```

Expected: all green. If `tests/strategies/test_paper_trader.py` breaks because tests poke `PaperTradingConfig(position_fraction=...)`, those tests now need to use `PaperTradingConfig(evaluators=EvaluatorsConfig(smart_money=SmartMoneyEvaluatorConfig(position_fraction=...)))`. Update the failing fixtures.

- [ ] **Step 3.7: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/config.py src/pscanner/strategies/paper_trader.py tests/test_config.py tests/strategies/test_paper_trader.py
uv run ruff format --check src/pscanner/config.py src/pscanner/strategies/paper_trader.py tests/test_config.py tests/strategies/test_paper_trader.py
uv run ty check src/pscanner/config.py src/pscanner/strategies/paper_trader.py
```

- [ ] **Step 3.8: Commit**

```bash
git add src/pscanner/config.py src/pscanner/strategies/paper_trader.py tests/test_config.py tests/strategies/test_paper_trader.py
git commit -m "feat(config): EvaluatorsConfig with per-source tunables

Four per-source config blocks (smart_money, move_attribution, velocity,
mispricing) under paper_trading.evaluators. Each carries an enabled flag,
position_fraction, and source-specific quality gates. The old
position_fraction and min_weighted_edge fields on PaperTradingConfig are
removed (per replace-don't-deprecate); their semantics move to
evaluators.smart_money.

PaperTrader config-read sites updated to the new path. Behaviour
unchanged at defaults. Future evaluators (T5-T6) will read their own
config blocks; the orchestrator refactor (T7) finishes the migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Mispricing detector enrichment

Add `target_condition_id`, `target_side`, `target_current_price`, `target_fair_price` to the mispricing alert body using proportional rebalancing as the fair-price model. The MispricingEvaluator (T5) reads these directly.

**Files:**
- Modify: `src/pscanner/detectors/mispricing.py`
- Modify: `tests/detectors/test_mispricing.py`

- [ ] **Step 4.1: Write failing tests for the body enrichment**

Append to `tests/detectors/test_mispricing.py`:

```python
def test_alert_body_includes_target_fields_for_overpriced() -> None:
    """deviation > 0 → most-overpriced YES leg → target_side='NO' with flipped prices."""
    # 3 markets with YES prices [0.4, 0.5, 0.5], sum 1.4, deviation +0.4.
    # Proportional fair = [0.286, 0.357, 0.357].
    # Most-extreme deviation: market 1 or 2 (both 0.143). First-by-tiebreak: market 1.
    # target_side="NO" because YES (0.5) > fair (0.357).
    # target_current_price = 1 - 0.5 = 0.5.
    # target_fair_price = 1 - 0.357 ≈ 0.643.
    event = _make_event_with_yes_prices(
        event_id="ev1",
        title="Three-way race",
        markets=[("m0", "Q0?", 0.4), ("m1", "Q1?", 0.5), ("m2", "Q2?", 0.5)],
    )
    alert = _build_alert_for_event(event)

    body = alert.body
    assert body["target_condition_id"] == "m1-cond"  # the first 0.5 market wins tiebreak
    assert body["target_side"] == "NO"
    assert body["target_current_price"] == pytest.approx(0.5)
    assert body["target_fair_price"] == pytest.approx(1 - 0.5 / 1.4, abs=1e-3)


def test_alert_body_includes_target_fields_for_underpriced() -> None:
    """deviation < 0 → most-underpriced YES leg → target_side='YES' with raw prices."""
    # 3 markets with YES prices [0.2, 0.2, 0.2], sum 0.6, deviation -0.4.
    # Proportional fair = [0.333, 0.333, 0.333].
    # All equally under-priced (deviation -0.133). First-by-tiebreak: market 0.
    # target_side="YES" because YES (0.2) < fair (0.333).
    event = _make_event_with_yes_prices(
        event_id="ev2",
        title="Three-way underpriced",
        markets=[("m0", "Q0?", 0.2), ("m1", "Q1?", 0.2), ("m2", "Q2?", 0.2)],
    )
    alert = _build_alert_for_event(event)

    body = alert.body
    assert body["target_condition_id"] == "m0-cond"
    assert body["target_side"] == "YES"
    assert body["target_current_price"] == pytest.approx(0.2)
    assert body["target_fair_price"] == pytest.approx(0.2 / 0.6, abs=1e-3)
```

The `_make_event_with_yes_prices` and `_build_alert_for_event` helpers may already exist in the test file; if not, add them (or reuse the existing fixture pattern — locate via `grep "def _make_event\|def _build_alert" tests/detectors/test_mispricing.py`). The third tuple element is the YES price; the helper builds an `Event` with the corresponding `Market` objects whose `outcome_prices[0] = yes_price` and `condition_id = f"{market_id}-cond"`.

- [ ] **Step 4.2: Run, verify they fail**

```bash
uv run pytest tests/detectors/test_mispricing.py -v -k "target_fields"
```

Expected: KeyError / fixture missing.

- [ ] **Step 4.3: Update `_build_body` and `_market_summary` in `mispricing.py`**

Replace `_build_body` (around line 267) with the enriched version:

```python
def _build_body(
    event: Event,
    price_sum: float,
    deviation: float,
    count: int,
) -> dict[str, Any]:
    """Build the JSON-serialisable body the alert sink will persist.

    Includes target_* fields naming the most-extreme leg for tradeable
    arbitrage entry. Fair price uses proportional rebalancing:
    fair[i] = current[i] / sum(current). The most-extreme leg is the one
    whose absolute deviation from fair is largest (first-by-iteration on
    ties). Direction follows the sign: current > fair → YES is over-priced
    → trade NO with flipped current/fair prices; otherwise trade YES.
    """
    target = _pick_target_market(event, price_sum)
    body: dict[str, Any] = {
        "event_id": event.id,
        "event_title": event.title,
        "price_sum": price_sum,
        "deviation": deviation,
        "market_count": count,
        "markets": [_market_summary(market) for market in event.markets],
    }
    if target is not None:
        target_market, target_yes_price, target_fair_yes_price = target
        if target_yes_price > target_fair_yes_price:
            target_side = "NO"
            target_current = 1.0 - target_yes_price
            target_fair = 1.0 - target_fair_yes_price
        else:
            target_side = "YES"
            target_current = target_yes_price
            target_fair = target_fair_yes_price
        body["target_condition_id"] = target_market.condition_id
        body["target_side"] = target_side
        body["target_current_price"] = target_current
        body["target_fair_price"] = target_fair
    return body


def _pick_target_market(
    event: Event, price_sum: float,
) -> tuple[Market, float, float] | None:
    """Return (market, yes_price, fair_yes_price) for the leg with the
    largest absolute deviation from proportional-rebalance fair, or None
    if no market in the event has a populated YES price."""
    if price_sum <= 0.0:
        return None
    best: tuple[Market, float, float] | None = None
    best_dev = -1.0
    for market in event.markets:
        if not market.outcome_prices:
            continue
        yes_price = market.outcome_prices[0]
        fair_yes_price = yes_price / price_sum
        dev = abs(yes_price - fair_yes_price)
        if dev > best_dev:
            best_dev = dev
            best = (market, yes_price, fair_yes_price)
    return best
```

(`Market` should be importable in this file already — verify via the existing imports near the top.)

- [ ] **Step 4.4: Run tests, verify they pass**

```bash
uv run pytest tests/detectors/test_mispricing.py -v
```

Expected: all pass.

- [ ] **Step 4.5: Run the full suite**

```bash
uv run pytest -q
```

Expected: green.

- [ ] **Step 4.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/detectors/mispricing.py tests/detectors/test_mispricing.py
uv run ruff format --check src/pscanner/detectors/mispricing.py tests/detectors/test_mispricing.py
uv run ty check src/pscanner/detectors/mispricing.py
```

- [ ] **Step 4.7: Commit**

```bash
git add src/pscanner/detectors/mispricing.py tests/detectors/test_mispricing.py
git commit -m "feat(mispricing): emit target_* fields for tradeable arbitrage entry

Mispricing alerts now carry target_condition_id, target_side,
target_current_price, and target_fair_price — the leg with the largest
absolute deviation from a proportional-rebalance fair price. Direction
follows the sign of the deviation: over-priced YES (current > fair) is
traded as NO with flipped prices; under-priced YES is traded as YES with
raw prices.

The MispricingEvaluator (next task wave) reads these fields directly.
Existing fields (event_id, event_title, deviation, markets[]) preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: SmartMoneyEvaluator + MoveAttributionEvaluator + MispricingEvaluator

Three single-entry evaluators in one task — they share enough structure to plan together but each gets its own file + test file.

**Files:**
- Create: `src/pscanner/strategies/evaluators/smart_money.py`
- Create: `src/pscanner/strategies/evaluators/move_attribution.py`
- Create: `src/pscanner/strategies/evaluators/mispricing.py`
- Create: `tests/strategies/evaluators/test_smart_money.py`
- Create: `tests/strategies/evaluators/test_move_attribution.py`
- Create: `tests/strategies/evaluators/test_mispricing.py`
- Modify: `src/pscanner/strategies/evaluators/__init__.py` (re-exports)

### Phase A — SmartMoneyEvaluator

- [ ] **Step 5.1: Write failing tests in `tests/strategies/evaluators/test_smart_money.py`**

```python
"""Unit tests for SmartMoneyEvaluator."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import SmartMoneyEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import TrackedWalletsRepo
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.smart_money import SmartMoneyEvaluator


_NOW = 1_700_000_000


def _alert(*, body: dict[str, Any], detector: str = "smart_money") -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]
        alert_key="k1",
        severity="med",
        title="t",
        body=body,
        created_at=_NOW,
    )


def _evaluator(
    tmp_db: sqlite3.Connection,
    *,
    position_fraction: float = 0.01,
    min_weighted_edge: float = 0.0,
) -> tuple[SmartMoneyEvaluator, TrackedWalletsRepo]:
    cfg = SmartMoneyEvaluatorConfig(
        position_fraction=position_fraction,
        min_weighted_edge=min_weighted_edge,
    )
    repo = TrackedWalletsRepo(tmp_db)
    return SmartMoneyEvaluator(config=cfg, tracked_wallets=repo), repo


def test_accepts_only_smart_money(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    assert ev.accepts(_alert(body={}, detector="smart_money")) is True
    assert ev.accepts(_alert(body={}, detector="velocity")) is False


def test_parse_extracts_wallet_condition_side(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    body = {"wallet": "0xabc", "condition_id": "0xc1", "side": "yes"}
    [parsed] = ev.parse(_alert(body=body))
    assert parsed.condition_id == ConditionId("0xc1")
    assert parsed.side == "yes"
    assert parsed.rule_variant is None
    assert parsed.metadata == {"wallet": "0xabc"}


def test_parse_returns_empty_on_missing_field(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    # Missing 'side'.
    body = {"wallet": "0xabc", "condition_id": "0xc1"}
    assert ev.parse(_alert(body=body)) == []


def test_quality_passes_uses_weighted_edge(tmp_db: sqlite3.Connection) -> None:
    ev, repo = _evaluator(tmp_db, min_weighted_edge=0.0)
    repo.upsert(wallet="0xabc", weighted_edge=0.4, mean_edge=0.4, n_trades=10, last_seen_at=_NOW, source="test")
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"), side="yes", metadata={"wallet": "0xabc"},
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_rejects_unknown_wallet(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"), side="yes", metadata={"wallet": "0xunknown"},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_rejects_below_min_edge(tmp_db: sqlite3.Connection) -> None:
    ev, repo = _evaluator(tmp_db, min_weighted_edge=0.5)
    repo.upsert(wallet="0xabc", weighted_edge=0.3, mean_edge=0.3, n_trades=10, last_seen_at=_NOW, source="test")
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"), side="yes", metadata={"wallet": "0xabc"},
    )
    assert ev.quality_passes(parsed) is False


def test_size_returns_bankroll_times_fraction(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db, position_fraction=0.01)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"), side="yes", metadata={"wallet": "0xabc"},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(10.0)
```

(`TrackedWalletsRepo.upsert` argument names may differ slightly — check
`src/pscanner/store/repo.py` for the exact signature, adapt the test
helper accordingly. Same for the `Severity` import path if needed.)

- [ ] **Step 5.2: Run, verify they fail**

```bash
uv run pytest tests/strategies/evaluators/test_smart_money.py -v
```

Expected: ImportError on `SmartMoneyEvaluator`.

- [ ] **Step 5.3: Implement `src/pscanner/strategies/evaluators/smart_money.py`**

```python
"""``SmartMoneyEvaluator`` — copy-trade smart_money alerts.

Today's PaperTrader behaviour, lifted into the per-detector evaluator
shape. Quality gate is wallet ``weighted_edge``; sizing is constant
``bankroll * position_fraction``.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import SmartMoneyEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import TrackedWalletsRepo
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class SmartMoneyEvaluator:
    """Smart-money copy-trade evaluator."""

    def __init__(
        self,
        *,
        config: SmartMoneyEvaluatorConfig,
        tracked_wallets: TrackedWalletsRepo,
    ) -> None:
        self._config = config
        self._tracked_wallets = tracked_wallets

    def accepts(self, alert: Alert) -> bool:
        return alert.detector == "smart_money"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        body = alert.body if isinstance(alert.body, dict) else {}
        wallet = body.get("wallet")
        condition_id_str = body.get("condition_id")
        side = body.get("side")
        if not (
            isinstance(wallet, str)
            and isinstance(condition_id_str, str)
            and isinstance(side, str)
        ):
            _LOG.debug("smart_money_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(condition_id_str),
                side=side,
                rule_variant=None,
                metadata={"wallet": wallet},
            )
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        wallet = parsed.metadata.get("wallet")
        if not isinstance(wallet, str):
            return False
        tracked = self._tracked_wallets.get(wallet)
        if tracked is None:
            _LOG.debug("smart_money_evaluator.no_edge", wallet=wallet)
            return False
        edge = tracked.weighted_edge
        if edge is None or edge <= self._config.min_weighted_edge:
            _LOG.debug("smart_money_evaluator.below_edge", wallet=wallet, edge=edge)
            return False
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        del parsed  # SmartMoney sizes uniformly across alerts
        return bankroll * self._config.position_fraction
```

- [ ] **Step 5.4: Run, verify they pass**

```bash
uv run pytest tests/strategies/evaluators/test_smart_money.py -v
```

Expected: all 7 tests pass.

### Phase B — MoveAttributionEvaluator

- [ ] **Step 5.5: Write failing tests in `tests/strategies/evaluators/test_move_attribution.py`**

```python
"""Unit tests for MoveAttributionEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MoveAttributionEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.move_attribution import MoveAttributionEvaluator


def _alert(
    *,
    body: dict[str, Any],
    detector: str = "move_attribution",
    severity: str = "med",
) -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]
        alert_key="k1",
        severity=severity,  # type: ignore[arg-type]
        title="t",
        body=body,
        created_at=0,
    )


def _evaluator(
    *,
    position_fraction: float = 0.01,
    min_severity: str = "med",
    min_wallets: int = 3,
) -> MoveAttributionEvaluator:
    cfg = MoveAttributionEvaluatorConfig(
        position_fraction=position_fraction,
        min_severity=min_severity,  # type: ignore[arg-type]
        min_wallets=min_wallets,
    )
    return MoveAttributionEvaluator(config=cfg)


def test_accepts_only_move_attribution() -> None:
    ev = _evaluator()
    assert ev.accepts(_alert(body={}, detector="move_attribution")) is True
    assert ev.accepts(_alert(body={}, detector="smart_money")) is False


def test_parse_extracts_outcome_as_side() -> None:
    """The outcome name (e.g. 'Anastasia Potapova') becomes the ParsedSignal side
    for cache lookup; the body's 'side' field (BUY/SELL taker) is ignored."""
    ev = _evaluator()
    body = {
        "condition_id": "0xc1",
        "outcome": "Anastasia Potapova",
        "side": "BUY",
        "n_wallets": 4,
    }
    [parsed] = ev.parse(_alert(body=body))
    assert parsed.condition_id == ConditionId("0xc1")
    assert parsed.side == "Anastasia Potapova"
    assert parsed.rule_variant is None


def test_parse_returns_empty_on_missing_outcome() -> None:
    ev = _evaluator()
    body = {"condition_id": "0xc1", "side": "BUY", "n_wallets": 4}
    assert ev.parse(_alert(body=body)) == []


def test_quality_passes_below_severity_rejected() -> None:
    """Default min_severity='med' rejects 'low'."""
    ev = _evaluator()
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "low", "n_wallets": 10},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_below_min_wallets_rejected() -> None:
    ev = _evaluator(min_wallets=5)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "high", "n_wallets": 3},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_high_severity_and_n_wallets() -> None:
    ev = _evaluator()
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "med", "n_wallets": 4},
    )
    assert ev.quality_passes(parsed) is True


def test_size_returns_bankroll_times_fraction() -> None:
    ev = _evaluator(position_fraction=0.01)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="yes",
        metadata={"severity": "med", "n_wallets": 4},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(10.0)
```

- [ ] **Step 5.6: Run, verify they fail**

```bash
uv run pytest tests/strategies/evaluators/test_move_attribution.py -v
```

Expected: ImportError.

- [ ] **Step 5.7: Implement `src/pscanner/strategies/evaluators/move_attribution.py`**

```python
"""``MoveAttributionEvaluator`` — trade the (outcome, side) pair surfaced
by MoveAttributionDetector when ≥ ``min_wallets`` distinct wallets
converged on a market in the burst window.

Note: the alert body's ``side`` field is a taker action ("BUY"/"SELL"),
NOT an outcome name. We use ``outcome`` as the cache lookup key (the
actual outcome name like "Anastasia Potapova" or "yes").
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import MoveAttributionEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)

_SEVERITY_RANK = {"low": 0, "med": 1, "high": 2}


class MoveAttributionEvaluator:
    """Trade move-attribution bursts named by upstream detectors."""

    def __init__(self, *, config: MoveAttributionEvaluatorConfig) -> None:
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        return alert.detector == "move_attribution"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        body = alert.body if isinstance(alert.body, dict) else {}
        condition_id_str = body.get("condition_id")
        outcome = body.get("outcome")
        n_wallets = body.get("n_wallets")
        if not (
            isinstance(condition_id_str, str)
            and isinstance(outcome, str)
        ):
            _LOG.debug("move_attribution_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(condition_id_str),
                side=outcome,
                rule_variant=None,
                metadata={
                    "severity": alert.severity,
                    "n_wallets": int(n_wallets) if isinstance(n_wallets, int) else 0,
                },
            )
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        severity = parsed.metadata.get("severity")
        n_wallets = parsed.metadata.get("n_wallets")
        if not isinstance(severity, str) or not isinstance(n_wallets, int):
            return False
        if _SEVERITY_RANK.get(severity, -1) < _SEVERITY_RANK.get(self._config.min_severity, 1):
            return False
        if n_wallets < self._config.min_wallets:
            return False
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        del parsed
        return bankroll * self._config.position_fraction
```

- [ ] **Step 5.8: Run, verify they pass**

```bash
uv run pytest tests/strategies/evaluators/test_move_attribution.py -v
```

Expected: all 7 tests pass.

### Phase C — MispricingEvaluator

- [ ] **Step 5.9: Write failing tests in `tests/strategies/evaluators/test_mispricing.py`**

```python
"""Unit tests for MispricingEvaluator."""

from __future__ import annotations

from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import MispricingEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.mispricing import MispricingEvaluator


def _alert(*, body: dict[str, Any], detector: str = "mispricing") -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]
        alert_key="k1",
        severity="med",
        title="t",
        body=body,
        created_at=0,
    )


def _evaluator(
    *, position_fraction: float = 0.01, min_edge_dollars: float = 0.05,
) -> MispricingEvaluator:
    return MispricingEvaluator(
        config=MispricingEvaluatorConfig(
            position_fraction=position_fraction,
            min_edge_dollars=min_edge_dollars,
        ),
    )


def test_accepts_only_mispricing() -> None:
    ev = _evaluator()
    assert ev.accepts(_alert(body={}, detector="mispricing")) is True
    assert ev.accepts(_alert(body={}, detector="smart_money")) is False


def test_parse_extracts_target_fields() -> None:
    ev = _evaluator()
    body = {
        "target_condition_id": "0xc1",
        "target_side": "NO",
        "target_current_price": 0.5,
        "target_fair_price": 0.643,
    }
    [parsed] = ev.parse(_alert(body=body))
    assert parsed.condition_id == ConditionId("0xc1")
    assert parsed.side == "NO"
    assert parsed.metadata == {"current": 0.5, "fair": 0.643}


def test_parse_returns_empty_when_target_fields_missing() -> None:
    """Legacy alerts without target_* (pre-T4) are silently skipped."""
    ev = _evaluator()
    body = {"event_id": "ev1", "deviation": 0.4}  # no target_*
    assert ev.parse(_alert(body=body)) == []


def test_quality_passes_above_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.05)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="NO",
        metadata={"current": 0.5, "fair": 0.643},  # edge = 0.143
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_below_min_edge() -> None:
    ev = _evaluator(min_edge_dollars=0.10)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="NO",
        metadata={"current": 0.5, "fair": 0.55},  # edge = 0.05 < 0.10
    )
    assert ev.quality_passes(parsed) is False


def test_size_returns_bankroll_times_fraction() -> None:
    ev = _evaluator(position_fraction=0.01)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc"),
        side="NO",
        metadata={"current": 0.5, "fair": 0.643},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(10.0)
```

- [ ] **Step 5.10: Run, verify they fail**

```bash
uv run pytest tests/strategies/evaluators/test_mispricing.py -v
```

Expected: ImportError.

- [ ] **Step 5.11: Implement `src/pscanner/strategies/evaluators/mispricing.py`**

```python
"""``MispricingEvaluator`` — trade the most-extreme leg of a mispriced event.

Reads the detector-emitted target_* fields from the alert body. Quality
gate is the magnitude of the gap between fair and current price.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import MispricingEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class MispricingEvaluator:
    """Trade the most-overpriced/most-underpriced YES leg of a mispriced event."""

    def __init__(self, *, config: MispricingEvaluatorConfig) -> None:
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        return alert.detector == "mispricing"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        body = alert.body if isinstance(alert.body, dict) else {}
        cond = body.get("target_condition_id")
        side = body.get("target_side")
        current = body.get("target_current_price")
        fair = body.get("target_fair_price")
        if not (
            isinstance(cond, str)
            and isinstance(side, str)
            and isinstance(current, int | float)
            and isinstance(fair, int | float)
        ):
            _LOG.debug("mispricing_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(cond),
                side=side,
                rule_variant=None,
                metadata={"current": float(current), "fair": float(fair)},
            )
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        current = parsed.metadata.get("current")
        fair = parsed.metadata.get("fair")
        if not (isinstance(current, int | float) and isinstance(fair, int | float)):
            return False
        edge = abs(float(fair) - float(current))
        return edge >= self._config.min_edge_dollars

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        del parsed
        return bankroll * self._config.position_fraction
```

- [ ] **Step 5.12: Run, verify they pass**

```bash
uv run pytest tests/strategies/evaluators/test_mispricing.py -v
```

Expected: all 6 tests pass.

### Phase D — wire up `__init__.py` and run full suite

- [ ] **Step 5.13: Update `src/pscanner/strategies/evaluators/__init__.py`**

```python
"""Per-detector :class:`SignalEvaluator` implementations."""

from pscanner.strategies.evaluators.mispricing import MispricingEvaluator
from pscanner.strategies.evaluators.move_attribution import MoveAttributionEvaluator
from pscanner.strategies.evaluators.protocol import (
    ParsedSignal,
    SignalEvaluator,
)
from pscanner.strategies.evaluators.smart_money import SmartMoneyEvaluator

__all__ = [
    "MispricingEvaluator",
    "MoveAttributionEvaluator",
    "ParsedSignal",
    "SignalEvaluator",
    "SmartMoneyEvaluator",
]
```

- [ ] **Step 5.14: Run the full evaluator test suite + repo regression**

```bash
uv run pytest tests/strategies/evaluators/ tests/strategies/test_paper_trader.py tests/store/test_paper_trades_repo.py -q
uv run pytest -q
```

Expected: all green.

- [ ] **Step 5.15: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/strategies/evaluators/ tests/strategies/evaluators/
uv run ruff format --check src/pscanner/strategies/evaluators/ tests/strategies/evaluators/
uv run ty check src/pscanner/strategies/evaluators/
```

- [ ] **Step 5.16: Commit**

```bash
git add src/pscanner/strategies/evaluators/ tests/strategies/evaluators/
git commit -m "feat(evaluators): add SmartMoney, MoveAttribution, and Mispricing evaluators

Three single-entry evaluators implementing the SignalEvaluator Protocol.
Each is self-contained: own config block, own quality gate, constant
bankroll-fraction sizing.

- SmartMoneyEvaluator: ports today's PaperTrader copy-trade behaviour.
  Quality gate: tracked_wallets[wallet].weighted_edge > min_weighted_edge.
- MoveAttributionEvaluator: trades the (condition_id, outcome) pair from
  burst alerts; ignores the body's BUY/SELL taker side modifier.
  Quality gate: severity floor + min_wallets.
- MispricingEvaluator: reads the target_* fields emitted by T4's mispricing
  detector enrichment. Quality gate: |fair - current| >= min_edge_dollars.

VelocityEvaluator (twin trades) lands separately in T6. PaperTrader
orchestrator refactor in T7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: VelocityEvaluator (twin trades)

The most complex evaluator. `parse` returns 2 ParsedSignals (follow + fade) by resolving the alert's `asset_id` against `MarketCacheRepo` to find both outcome names. If the cache misses, returns `[]` (can't trade either side without knowing the outcome names).

**Files:**
- Create: `src/pscanner/strategies/evaluators/velocity.py`
- Create: `tests/strategies/evaluators/test_velocity.py`
- Modify: `src/pscanner/strategies/evaluators/__init__.py` (add export)

- [ ] **Step 6.1: Write failing tests in `tests/strategies/evaluators/test_velocity.py`**

```python
"""Unit tests for VelocityEvaluator (twin trades)."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import VelocityEvaluatorConfig
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import CachedMarket, MarketCacheRepo
from pscanner.strategies.evaluators import ParsedSignal
from pscanner.strategies.evaluators.velocity import VelocityEvaluator


def _alert(
    *,
    body: dict[str, Any],
    detector: str = "velocity",
    severity: str = "high",
) -> Alert:
    return Alert(
        detector=detector,  # type: ignore[arg-type]
        alert_key="k1",
        severity=severity,  # type: ignore[arg-type]
        title="t",
        body=body,
        created_at=0,
    )


def _seed_market(
    cache: MarketCacheRepo,
    *,
    condition_id: str,
    outcomes: list[str],
    asset_ids: list[str],
) -> None:
    cache.upsert(
        CachedMarket(
            market_id="mkt-1",  # type: ignore[arg-type]
            event_id=None,
            title="Test market",
            liquidity_usd=100_000.0,
            volume_usd=10_000.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=0,
            condition_id=ConditionId(condition_id),
            event_slug=None,
            outcomes=outcomes,
            asset_ids=[AssetId(a) for a in asset_ids],
        ),
    )


def _evaluator(
    tmp_db: sqlite3.Connection,
    *,
    position_fraction: float = 0.0025,
    min_severity: str = "high",
    allow_consolidation: bool = False,
) -> tuple[VelocityEvaluator, MarketCacheRepo]:
    cfg = VelocityEvaluatorConfig(
        position_fraction=position_fraction,
        min_severity=min_severity,  # type: ignore[arg-type]
        allow_consolidation=allow_consolidation,
    )
    cache = MarketCacheRepo(tmp_db)
    return VelocityEvaluator(config=cfg, market_cache=cache), cache


def test_accepts_only_velocity(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    assert ev.accepts(_alert(body={}, detector="velocity")) is True
    assert ev.accepts(_alert(body={}, detector="smart_money")) is False


def test_parse_returns_two_paired_signals(tmp_db: sqlite3.Connection) -> None:
    """A binary YES/NO market emits a follow + fade pair."""
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes", "no"],
        asset_ids=["asset-yes", "asset-no"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-yes",
        "change_pct": 0.12,
        "consolidation": False,
    }

    signals = ev.parse(_alert(body=body))

    assert len(signals) == 2
    follow = next(s for s in signals if s.rule_variant == "follow")
    fade = next(s for s in signals if s.rule_variant == "fade")
    assert follow.side == "yes"
    assert fade.side == "no"
    assert follow.condition_id == fade.condition_id == ConditionId("0xc1")


def test_parse_returns_empty_on_cache_miss(tmp_db: sqlite3.Connection) -> None:
    """If market_cache has no entry, neither side resolves; skip the alert."""
    ev, _ = _evaluator(tmp_db)
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-yes",
        "change_pct": 0.12,
        "consolidation": False,
    }
    assert ev.parse(_alert(body=body)) == []


def test_parse_returns_only_follow_if_market_has_one_outcome(
    tmp_db: sqlite3.Connection,
) -> None:
    """1-outcome market: fade has no opposing side, return follow only."""
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes"],
        asset_ids=["asset-yes"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-yes",
        "change_pct": 0.12,
        "consolidation": False,
    }
    signals = ev.parse(_alert(body=body))
    assert len(signals) == 1
    assert signals[0].rule_variant == "follow"


def test_parse_returns_empty_when_alert_asset_id_unknown(
    tmp_db: sqlite3.Connection,
) -> None:
    """Cached market exists but doesn't contain the alert's asset_id."""
    ev, cache = _evaluator(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes", "no"],
        asset_ids=["asset-yes", "asset-no"],
    )
    body = {
        "condition_id": "0xc1",
        "asset_id": "asset-other",  # not in the cached market
        "change_pct": 0.12,
        "consolidation": False,
    }
    assert ev.parse(_alert(body=body)) == []


def test_parse_returns_empty_on_missing_field(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    # Missing condition_id.
    assert ev.parse(_alert(body={"asset_id": "asset-yes"})) == []


def test_quality_passes_high_severity_no_consolidation(
    tmp_db: sqlite3.Connection,
) -> None:
    ev, _ = _evaluator(tmp_db)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "high", "consolidation": False},
    )
    assert ev.quality_passes(parsed) is True


def test_quality_passes_rejects_low_severity(tmp_db: sqlite3.Connection) -> None:
    ev, _ = _evaluator(tmp_db)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "low", "consolidation": False},
    )
    assert ev.quality_passes(parsed) is False


def test_quality_passes_rejects_consolidation_when_disallowed(
    tmp_db: sqlite3.Connection,
) -> None:
    ev, _ = _evaluator(tmp_db, allow_consolidation=False)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "high", "consolidation": True},
    )
    assert ev.quality_passes(parsed) is False


def test_size_per_entry_is_bankroll_times_fraction(tmp_db: sqlite3.Connection) -> None:
    """Velocity sizes per-entry; pair total = 2 × per-entry."""
    ev, _ = _evaluator(tmp_db, position_fraction=0.0025)
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="yes",
        rule_variant="follow",
        metadata={"severity": "high", "consolidation": False},
    )
    assert ev.size(bankroll=1000.0, parsed=parsed) == pytest.approx(2.5)
```

(`MarketCacheRepo.upsert` argument shape may be `cache.upsert(market)`
or `cache.upsert(**fields)` — check `src/pscanner/store/repo.py`.
Adapt the helper accordingly.)

- [ ] **Step 6.2: Run, verify they fail**

```bash
uv run pytest tests/strategies/evaluators/test_velocity.py -v
```

Expected: ImportError.

- [ ] **Step 6.3: Implement `src/pscanner/strategies/evaluators/velocity.py`**

```python
"""``VelocityEvaluator`` — twin-trade follow + fade per velocity alert.

Each velocity alert spawns two ParsedSignals at half-size: one with
rule_variant='follow' (buying the moving side, i.e. the asset_id named
in the alert) and one with rule_variant='fade' (buying the opposing
outcome on the same condition_id). Resolving the opposing-side outcome
requires a MarketCacheRepo lookup. Cache misses yield no signals — we
can't trade either side without knowing the outcome names.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import VelocityEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import MarketCacheRepo
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)

_SEVERITY_RANK = {"low": 0, "med": 1, "high": 2}


class VelocityEvaluator:
    """Twin-trade evaluator for velocity alerts."""

    def __init__(
        self,
        *,
        config: VelocityEvaluatorConfig,
        market_cache: MarketCacheRepo,
    ) -> None:
        self._config = config
        self._market_cache = market_cache

    def accepts(self, alert: Alert) -> bool:
        return alert.detector == "velocity"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        body = alert.body if isinstance(alert.body, dict) else {}
        condition_id_str = body.get("condition_id")
        asset_id_str = body.get("asset_id")
        consolidation = bool(body.get("consolidation", False))
        if not (
            isinstance(condition_id_str, str)
            and isinstance(asset_id_str, str)
        ):
            _LOG.debug("velocity_evaluator.bad_body", alert_key=alert.alert_key)
            return []

        condition_id = ConditionId(condition_id_str)
        cached = self._market_cache.get_by_condition_id(condition_id)
        if cached is None or not cached.outcomes or not cached.asset_ids:
            _LOG.debug(
                "velocity_evaluator.cache_miss",
                alert_key=alert.alert_key,
                condition_id=condition_id_str,
            )
            return []

        # Identify follow + fade outcomes.
        follow_outcome: str | None = None
        fade_outcome: str | None = None
        for outcome_name, oid in zip(cached.outcomes, cached.asset_ids, strict=False):
            if oid == asset_id_str:
                follow_outcome = outcome_name
            elif fade_outcome is None:  # first non-matching outcome wins
                fade_outcome = outcome_name
        if follow_outcome is None:
            _LOG.debug(
                "velocity_evaluator.unknown_asset",
                alert_key=alert.alert_key,
                asset_id=asset_id_str,
            )
            return []

        meta = {"severity": alert.severity, "consolidation": consolidation}
        signals = [
            ParsedSignal(
                condition_id=condition_id,
                side=follow_outcome,
                rule_variant="follow",
                metadata=meta,
            ),
        ]
        if fade_outcome is not None:
            signals.append(
                ParsedSignal(
                    condition_id=condition_id,
                    side=fade_outcome,
                    rule_variant="fade",
                    metadata=meta,
                ),
            )
        return signals

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        severity = parsed.metadata.get("severity")
        consolidation = parsed.metadata.get("consolidation")
        if not isinstance(severity, str):
            return False
        if _SEVERITY_RANK.get(severity, -1) < _SEVERITY_RANK.get(self._config.min_severity, 2):
            return False
        if bool(consolidation) and not self._config.allow_consolidation:
            return False
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        del parsed
        return bankroll * self._config.position_fraction
```

- [ ] **Step 6.4: Update `__init__.py` to re-export**

```python
"""Per-detector :class:`SignalEvaluator` implementations."""

from pscanner.strategies.evaluators.mispricing import MispricingEvaluator
from pscanner.strategies.evaluators.move_attribution import MoveAttributionEvaluator
from pscanner.strategies.evaluators.protocol import (
    ParsedSignal,
    SignalEvaluator,
)
from pscanner.strategies.evaluators.smart_money import SmartMoneyEvaluator
from pscanner.strategies.evaluators.velocity import VelocityEvaluator

__all__ = [
    "MispricingEvaluator",
    "MoveAttributionEvaluator",
    "ParsedSignal",
    "SignalEvaluator",
    "SmartMoneyEvaluator",
    "VelocityEvaluator",
]
```

- [ ] **Step 6.5: Run, verify they pass**

```bash
uv run pytest tests/strategies/evaluators/test_velocity.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 6.6: Run the full evaluator + repo suite**

```bash
uv run pytest tests/strategies/evaluators/ tests/strategies/test_paper_trader.py tests/store/test_paper_trades_repo.py -q
uv run pytest -q
```

Expected: green.

- [ ] **Step 6.7: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/strategies/evaluators/velocity.py tests/strategies/evaluators/test_velocity.py src/pscanner/strategies/evaluators/__init__.py
uv run ruff format --check src/pscanner/strategies/evaluators/velocity.py tests/strategies/evaluators/test_velocity.py src/pscanner/strategies/evaluators/__init__.py
uv run ty check src/pscanner/strategies/evaluators/velocity.py
```

- [ ] **Step 6.8: Commit**

```bash
git add src/pscanner/strategies/evaluators/velocity.py src/pscanner/strategies/evaluators/__init__.py tests/strategies/evaluators/test_velocity.py
git commit -m "feat(evaluators): add VelocityEvaluator with twin trades

Each velocity alert spawns two ParsedSignals: one with rule_variant='follow'
(buying the moving side from the alert's asset_id) and one with
rule_variant='fade' (buying the opposing outcome on the same condition).
Resolving the opposing outcome requires a MarketCacheRepo lookup; cache
misses yield no signals (can't trade either side without outcome names).

Constant per-side size off starting bankroll: default 0.25% per entry,
0.5% pair total — smaller than other signals because the directional
rule is unproven and we want labeled-outcome data per (severity,
change_pct) bucket without burning bankroll allocation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: PaperTrader refactor

Strip source-specific logic out of `PaperTrader.evaluate`; replace with a generic loop over an injected list of `SignalEvaluator`s. Drop the `bankroll_exhausted` gate (constant sizing per spec). Wire `triggering_alert_detector` and `rule_variant` into the entry insert.

**Files:**
- Modify: `src/pscanner/strategies/paper_trader.py`
- Modify: `tests/strategies/test_paper_trader.py`

- [ ] **Step 7.1: Write a failing test for the new ctor signature**

Append to `tests/strategies/test_paper_trader.py`:

```python
def test_paper_trader_accepts_evaluator_list(tmp_db: sqlite3.Connection) -> None:
    """The new ctor takes evaluators=[...]; old position_fraction kwarg is gone."""
    cfg = PaperTradingConfig(enabled=True)
    market_cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    market_ticks = MarketTicksRepo(tmp_db)
    data = _stub_data_client()
    gamma = _stub_gamma_client()

    trader = PaperTrader(
        config=cfg,
        evaluators=[],  # empty list — no detector handled
        market_cache=market_cache,
        paper_trades=paper,
        market_ticks=market_ticks,
        data_client=data,
        gamma_client=gamma,
    )
    assert trader is not None
```

- [ ] **Step 7.2: Write a failing test for evaluator dispatch**

```python
async def test_evaluate_dispatches_to_first_acceptor(
    tmp_db: sqlite3.Connection,
) -> None:
    """PaperTrader walks the list, picks the first ev whose accepts() is True."""
    seen: list[str] = []

    class _StubEvaluator:
        def __init__(self, name: str, accepts_detector: str) -> None:
            self._name = name
            self._accepts_detector = accepts_detector

        def accepts(self, alert: Alert) -> bool:
            return alert.detector == self._accepts_detector

        def parse(self, alert: Alert) -> list[ParsedSignal]:
            seen.append(self._name)
            return []  # no signals — we just verify dispatch

        def quality_passes(self, parsed: ParsedSignal) -> bool:
            return True

        def size(self, bankroll: float, parsed: ParsedSignal) -> float:
            return 0.0

    cfg = PaperTradingConfig(enabled=True)
    paper = PaperTradesRepo(tmp_db)
    trader = PaperTrader(
        config=cfg,
        evaluators=[
            _StubEvaluator("smart_money_ev", "smart_money"),
            _StubEvaluator("velocity_ev", "velocity"),
        ],
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=_stub_data_client(),
        gamma_client=_stub_gamma_client(),
    )

    await trader.evaluate(_velocity_alert())
    assert seen == ["velocity_ev"]
```

- [ ] **Step 7.3: Write a failing test that an evaluator's exception is contained**

```python
async def test_evaluator_exception_logs_and_continues(
    tmp_db: sqlite3.Connection,
) -> None:
    """A raising evaluator does not kill PaperTrader; warning is logged."""
    from structlog.testing import capture_logs

    class _RaisingEvaluator:
        def accepts(self, alert: Alert) -> bool:
            return True

        def parse(self, alert: Alert) -> list[ParsedSignal]:
            raise RuntimeError("boom")

        def quality_passes(self, parsed: ParsedSignal) -> bool:
            return True

        def size(self, bankroll: float, parsed: ParsedSignal) -> float:
            return 0.0

    cfg = PaperTradingConfig(enabled=True)
    paper = PaperTradesRepo(tmp_db)
    trader = PaperTrader(
        config=cfg,
        evaluators=[_RaisingEvaluator()],
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=_stub_data_client(),
        gamma_client=_stub_gamma_client(),
    )

    with capture_logs() as logs:
        await trader.evaluate(_smart_money_alert())  # detector is irrelevant; stub accepts all
    assert any(log["event"] == "paper_trader.evaluator_failed" for log in logs)
```

- [ ] **Step 7.4: Write a failing test that triggering_alert_detector + rule_variant are passed through to insert_entry**

```python
async def test_evaluate_writes_detector_and_variant_to_entry(
    tmp_db: sqlite3.Connection,
) -> None:
    """Each ParsedSignal becomes an insert_entry with the alert detector +
    parsed rule_variant stamped onto the row."""

    class _DummyEvaluator:
        def accepts(self, alert: Alert) -> bool:
            return alert.detector == "velocity"

        def parse(self, alert: Alert) -> list[ParsedSignal]:
            return [
                ParsedSignal(
                    condition_id=ConditionId("0xc1"),
                    side="yes",
                    rule_variant="follow",
                ),
                ParsedSignal(
                    condition_id=ConditionId("0xc1"),
                    side="no",
                    rule_variant="fade",
                ),
            ]

        def quality_passes(self, parsed: ParsedSignal) -> bool:
            return True

        def size(self, bankroll: float, parsed: ParsedSignal) -> float:
            return 2.5

    cfg = PaperTradingConfig(enabled=True)
    paper = PaperTradesRepo(tmp_db)
    cache = MarketCacheRepo(tmp_db)
    _seed_market(cache, condition_id="0xc1", outcomes=["yes", "no"],
                 asset_ids=["a-y", "a-n"])
    _seed_tick(tmp_db, asset_id="a-y", best_ask=0.5)
    _seed_tick(tmp_db, asset_id="a-n", best_ask=0.5)

    trader = PaperTrader(
        config=cfg,
        evaluators=[_DummyEvaluator()],
        market_cache=cache,
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=_stub_data_client(),
        gamma_client=_stub_gamma_client(),
    )

    await trader.evaluate(_velocity_alert())

    rows = list(tmp_db.execute(
        "SELECT triggering_alert_detector, rule_variant FROM paper_trades "
        "WHERE trade_kind = 'entry' ORDER BY trade_id"
    ))
    assert rows == [("velocity", "follow"), ("velocity", "fade")]
```

(`_seed_tick` is the existing test helper; preserve from current test file.
`_velocity_alert` and `_smart_money_alert` are simple Alert factories —
add them if missing.)

- [ ] **Step 7.5: Run, verify they fail**

```bash
uv run pytest tests/strategies/test_paper_trader.py -v -k "evaluator_list or dispatches_to_first or evaluator_exception or detector_and_variant"
```

Expected: TypeErrors / failures.

- [ ] **Step 7.6: Refactor `src/pscanner/strategies/paper_trader.py`**

Replace the file with the orchestrator shape. Key changes:
- Remove `tracked_wallets` ctor arg (smart_money's evaluator owns it now).
- Add `evaluators: list[SignalEvaluator]` ctor arg.
- Remove `_parse_alert`, `_wallet_passes_edge_filter`.
- Remove the `bankroll_exhausted` gate.
- Read `bankroll = self._config.starting_bankroll_usd` (constant).
- For each ParsedSignal, run resolve → size → insert.
- Wrap per-evaluator pipeline in try/except.

```python
"""Multi-signal paper-trading subscriber.

Subscribes to :class:`AlertSink`. Walks a list of :class:`SignalEvaluator`
instances; the first one whose ``accepts(alert)`` returns True parses the
alert into one or more :class:`ParsedSignal`s. Each signal is independently
quality-gated, resolved to an asset_id + fill price, sized at constant
``starting_bankroll_usd * position_fraction``, and booked as a paper_trades
entry row.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import TypeIs

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
)
from pscanner.strategies.evaluators import ParsedSignal, SignalEvaluator

_LOG = structlog.get_logger(__name__)

_FILL_PRICE_LO = 0.0
_FILL_PRICE_HI = 1.0


def _is_valid_price(value: object) -> TypeIs[int | float]:
    if not isinstance(value, int | float):
        return False
    return _FILL_PRICE_LO < value < _FILL_PRICE_HI


def _size_valid(cost: float, fill_price: float, *, min_cost: float) -> bool:
    if cost < min_cost:
        return False
    if not (_FILL_PRICE_LO < fill_price < _FILL_PRICE_HI):
        return False
    return True


class PaperTrader:
    """Alert-driven multi-signal paper-trader."""

    name = "paper_trader"

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        evaluators: list[SignalEvaluator],
        market_cache: MarketCacheRepo,
        paper_trades: PaperTradesRepo,
        market_ticks: MarketTicksRepo,
        data_client: DataClient,
        gamma_client: GammaClient,
    ) -> None:
        self._config = config
        self._evaluators = evaluators
        self._market_cache = market_cache
        self._paper_trades = paper_trades
        self._market_ticks = market_ticks
        self._data_client = data_client
        self._gamma_client = gamma_client
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def run(self, sink: AlertSink) -> None:
        del sink
        await asyncio.Event().wait()

    def handle_alert_sync(self, alert: Alert) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("paper_trader.no_event_loop", alert_key=alert.alert_key)
            return
        task = loop.create_task(self.evaluate(alert))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate(self, alert: Alert) -> None:
        """Run the evaluator pipeline for one alert."""
        for evaluator in self._evaluators:
            if not evaluator.accepts(alert):
                continue
            try:
                await self._run_pipeline(evaluator, alert)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.warning(
                    "paper_trader.evaluator_failed",
                    detector=alert.detector,
                    evaluator=type(evaluator).__name__,
                    alert_key=alert.alert_key,
                    exc_info=True,
                )
            return

    async def _run_pipeline(
        self, evaluator: SignalEvaluator, alert: Alert,
    ) -> None:
        parsed_list = evaluator.parse(alert)
        if not parsed_list:
            return
        bankroll = self._config.starting_bankroll_usd
        nav = self._paper_trades.compute_cost_basis_nav(
            starting_bankroll=bankroll,
        )
        for parsed in parsed_list:
            if not evaluator.quality_passes(parsed):
                continue
            resolved = await self._resolve_outcome(parsed.condition_id, parsed.side)
            if resolved is None:
                continue
            asset_id, fill_price = resolved
            cost = evaluator.size(bankroll, parsed)
            if not _size_valid(cost, fill_price, min_cost=self._config.min_position_cost_usd):
                _LOG.debug(
                    "paper_trader.size_too_small_or_bad_price",
                    alert_key=alert.alert_key,
                    cost=cost,
                    fill_price=fill_price,
                )
                continue
            shares = cost / fill_price
            self._insert_entry(
                alert=alert,
                parsed=parsed,
                asset_id=asset_id,
                shares=shares,
                fill_price=fill_price,
                cost_usd=cost,
                nav=nav,
            )

    def _insert_entry(
        self,
        *,
        alert: Alert,
        parsed: ParsedSignal,
        asset_id: AssetId,
        shares: float,
        fill_price: float,
        cost_usd: float,
        nav: float,
    ) -> None:
        wallet = parsed.metadata.get("wallet")
        try:
            self._paper_trades.insert_entry(
                triggering_alert_key=alert.alert_key,
                triggering_alert_detector=alert.detector,
                rule_variant=parsed.rule_variant,
                source_wallet=wallet if isinstance(wallet, str) else None,
                condition_id=parsed.condition_id,
                asset_id=asset_id,
                outcome=parsed.side,
                shares=shares,
                fill_price=fill_price,
                cost_usd=cost_usd,
                nav_after_usd=nav,
                ts=int(time.time()),
            )
        except sqlite3.IntegrityError:
            _LOG.debug("paper_trader.duplicate_alert", alert_key=alert.alert_key)
        except Exception:
            _LOG.warning(
                "paper_trader.insert_failed",
                alert_key=alert.alert_key,
                exc_info=True,
            )

    async def _resolve_outcome(
        self,
        condition_id: ConditionId,
        side: str,
    ) -> tuple[AssetId, float] | None:
        # ... existing _resolve_outcome body ...
```

For `_resolve_outcome`, **keep the existing implementation verbatim** —
that logic (cache lookup → backfill via gamma+data on miss → tick lookup
or fall back to `outcome_prices`) is unchanged. Just paste it from the
old file into the new file.

For `aclose` (or whatever cleanup method exists today), preserve it — the
`_pending_tasks` lifecycle is unchanged.

- [ ] **Step 7.7: Run, verify they pass**

```bash
uv run pytest tests/strategies/test_paper_trader.py -v
```

Expected: all green. Existing smart_money-flavored tests may need refactor to use a `SmartMoneyEvaluator` instead of the removed `tracked_wallets` ctor arg — update fixtures accordingly. Tests should construct PaperTrader with `evaluators=[SmartMoneyEvaluator(...)]` for those scenarios.

- [ ] **Step 7.8: Run the full suite**

```bash
uv run pytest -q
```

Expected: green. Anything that broke in `tests/test_cli.py` or `tests/scheduler*` because of ctor-arg changes gets adjusted to match — those tests likely thin shims.

- [ ] **Step 7.9: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader.py
uv run ruff format --check src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader.py
uv run ty check src/pscanner/strategies/paper_trader.py
```

- [ ] **Step 7.10: Commit**

```bash
git add src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader.py
git commit -m "refactor(paper_trader): orchestrate per-detector evaluators

PaperTrader.evaluate now walks an injected list of SignalEvaluators,
picks the first acceptor, runs parse → quality → resolve → size → insert
once per ParsedSignal. Source-specific logic (smart_money body parsing,
tracked-wallets edge filter) lifted out into the evaluators landed in
T5-T6.

Sizing reads starting_bankroll_usd (constant), not running NAV. The
bankroll_exhausted gate is removed — trades keep booking even with
negative cost-basis NAV (research configuration; data over realism).

Each entry row stamps the alert.detector and parsed.rule_variant into the
new paper_trades columns for per-source PnL queries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Scheduler wiring + CLI per-source breakdown

Final integration step. Scheduler builds the four evaluators (gated by their `enabled` flags) and passes the list to PaperTrader. CLI `paper status` grows a per-source breakdown table.

**Files:**
- Modify: `src/pscanner/scheduler.py`
- Modify: `src/pscanner/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 8.1: Locate the PaperTrader construction site in scheduler.py**

```bash
grep -n "PaperTrader\(\|_attach_paper" src/pscanner/scheduler.py
```

Identify the `_attach_paper_trader` (or equivalent) method that constructs `PaperTrader` today.

- [ ] **Step 8.2: Update scheduler.py to build evaluators**

Add imports near the existing strategies imports:

```python
from pscanner.strategies.evaluators import (
    MispricingEvaluator,
    MoveAttributionEvaluator,
    SignalEvaluator,
    SmartMoneyEvaluator,
    VelocityEvaluator,
)
```

In `_attach_paper_trader` (or equivalent), replace the PaperTrader construction with:

```python
def _build_paper_evaluators(self) -> list[SignalEvaluator]:
    """Build the enabled evaluators in fixed order: smart_money,
    move_attribution, mispricing, velocity. Disabled evaluators are not
    constructed at all."""
    cfg = self._config.paper_trading.evaluators
    evaluators: list[SignalEvaluator] = []
    if cfg.smart_money.enabled:
        evaluators.append(
            SmartMoneyEvaluator(
                config=cfg.smart_money,
                tracked_wallets=self._tracked_wallets_repo,
            )
        )
    if cfg.move_attribution.enabled:
        evaluators.append(
            MoveAttributionEvaluator(config=cfg.move_attribution)
        )
    if cfg.mispricing.enabled:
        evaluators.append(
            MispricingEvaluator(config=cfg.mispricing)
        )
    if cfg.velocity.enabled:
        evaluators.append(
            VelocityEvaluator(
                config=cfg.velocity,
                market_cache=self._market_cache_repo,
            )
        )
    return evaluators

# In _attach_paper_trader, replace the old construction with:
detectors["paper_trader"] = PaperTrader(
    config=self._config.paper_trading,
    evaluators=self._build_paper_evaluators(),
    market_cache=self._market_cache_repo,
    paper_trades=self._paper_trades_repo,
    market_ticks=self._market_ticks_repo,
    data_client=self._clients.data_client,
    gamma_client=self._clients.gamma_client,
)
```

(Adapt attribute names to match what's already on `Scanner` — names like
`self._tracked_wallets_repo` vs `self._tracked_wallets` should be checked
via `grep "tracked_wallets" src/pscanner/scheduler.py` and matched.)

- [ ] **Step 8.3: Update CLI's `paper status` for per-source breakdown**

In `src/pscanner/cli.py`, locate the `paper status` command body. Add a per-source table after the existing aggregate stats. Sketch:

```python
# After the existing aggregate output ...
sources = paper_trades.summary_by_source(
    starting_bankroll=cfg.paper_trading.starting_bankroll_usd,
)
if sources:
    typer.echo("")
    typer.echo("Per-source breakdown:")
    typer.echo(f"{'detector':<20s} {'variant':<8s} {'open':>5s} {'resolved':>9s} {'pnl':>9s} {'win_rate':>9s}")
    for s in sources:
        det = s.detector or "(unknown)"
        variant = s.rule_variant or "-"
        typer.echo(
            f"{det:<20s} {variant:<8s} {s.open_count:>5d} {s.resolved_count:>9d} "
            f"{s.realized_pnl:>9.2f} {s.win_rate*100:>8.1f}%"
        )
```

(The exact CLI library may be `typer` or `click` — match what's already used.)

- [ ] **Step 8.4: Write a failing test for the per-source breakdown**

In `tests/test_cli.py`, append:

```python
def test_paper_status_shows_per_source_breakdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "config.toml"
    db = tmp_path / "pscanner.sqlite3"
    _write_config(cfg, db)
    conn = init_db(db)
    try:
        repo = PaperTradesRepo(conn)
        # smart_money entry (resolved win)
        e1 = repo.insert_entry(
            triggering_alert_key="smart:0xa:1",
            triggering_alert_detector="smart_money",
            rule_variant=None,
            source_wallet="0xa",
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0, fill_price=0.5, cost_usd=10.0,
            nav_after_usd=1000.0, ts=1700000000,
        )
        repo.insert_exit(
            parent_trade_id=e1, condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"), outcome="yes",
            shares=20.0, fill_price=1.0, cost_usd=20.0,
            nav_after_usd=1010.0, ts=1700000100,
        )
        # velocity follow (open)
        repo.insert_entry(
            triggering_alert_key="vel:0xb:1",
            triggering_alert_detector="velocity",
            rule_variant="follow",
            source_wallet=None,
            condition_id=ConditionId("0xc2"),
            asset_id=AssetId("b-y"),
            outcome="yes",
            shares=10.0, fill_price=0.25, cost_usd=2.5,
            nav_after_usd=1010.0, ts=1700000200,
        )
        # mispricing (open)
        repo.insert_entry(
            triggering_alert_key="mis:ev1:1",
            triggering_alert_detector="mispricing",
            rule_variant=None,
            source_wallet=None,
            condition_id=ConditionId("0xc3"),
            asset_id=AssetId("c-no"),
            outcome="NO",
            shares=10.0, fill_price=0.5, cost_usd=5.0,
            nav_after_usd=1010.0, ts=1700000300,
        )
    finally:
        conn.close()

    rc = main(["--config", str(cfg), "paper", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Per-source breakdown" in out
    assert "smart_money" in out
    assert "velocity" in out
    assert "follow" in out
    assert "mispricing" in out
```

- [ ] **Step 8.5: Run, verify it fails**

```bash
uv run pytest tests/test_cli.py -v -k paper_status_shows_per_source
```

Expected: assertion failure on "Per-source breakdown" missing from output.

- [ ] **Step 8.6: Run after CLI update, verify it passes**

```bash
uv run pytest tests/test_cli.py -v -k paper_status_shows_per_source
```

Expected: pass.

- [ ] **Step 8.7: Smoke `pscanner run --once` to verify wiring**

```bash
rm -f data/pscanner.sqlite3
timeout 30 uv run pscanner run --once > /tmp/wiring-smoke.log 2>&1
echo "exit=$?"
grep -E "paper_trader|evaluator" /tmp/wiring-smoke.log | head -10
```

Expected: clean run; no exceptions about evaluator construction or
`PaperTrader.__init__` signature.

- [ ] **Step 8.8: Run the full suite**

```bash
uv run pytest -q
```

Expected: green.

- [ ] **Step 8.9: Lint / format / type-check the whole repo**

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

- [ ] **Step 8.10: Commit**

```bash
git add src/pscanner/scheduler.py src/pscanner/cli.py tests/test_cli.py
git commit -m "feat(scheduler+cli): wire 4 evaluators + paper status per-source breakdown

Scanner._build_paper_evaluators constructs each Evaluator gated by its
enabled config flag; disabled sources are simply not in the list. Fixed
order: smart_money, move_attribution, mispricing, velocity.

paper status CLI gains a per-(detector, rule_variant) breakdown table
showing open count, resolved count, realized PnL, and win rate. Reads
PaperTradesRepo.summary_by_source.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (notes to the executor)

After all 8 tasks complete, verify the spec is fully covered:

### Spec coverage

- **`SignalEvaluator` Protocol** — Task 1.
- **`ParsedSignal` dataclass** — Task 1.
- **`SmartMoneyEvaluator`** — Task 5 (Phase A).
- **`MoveAttributionEvaluator`** — Task 5 (Phase B).
- **`MispricingEvaluator`** — Task 5 (Phase C).
- **`VelocityEvaluator` with twin trades** — Task 6.
- **Mispricing detector enrichment (proportional rebalancing)** — Task 4.
- **`EvaluatorsConfig` per-source blocks with `enabled`** — Task 3.
- **Removal of `position_fraction` and `min_weighted_edge` from `PaperTradingConfig`** — Task 3.
- **DB columns `triggering_alert_detector` + `rule_variant`** — Task 2.
- **Backfill existing rows** — Task 2.
- **Index update with `COALESCE(rule_variant, '')`** — Task 2.
- **`PaperTradesRepo.summary_by_source`** — Task 2.
- **PaperTrader refactor (orchestrator)** — Task 7.
- **Constant sizing off `starting_bankroll_usd`** — Task 7.
- **Removal of `bankroll_exhausted` gate** — Task 7.
- **Per-evaluator failure isolation (`paper_trader.evaluator_failed` log)** — Task 7.
- **Scheduler builds evaluators in fixed order** — Task 8.
- **`paper status` per-source breakdown** — Task 8.

### Type / signature consistency

- `SignalEvaluator.accepts(alert: Alert) -> bool` — Task 1, used T5/T6/T7.
- `SignalEvaluator.parse(alert: Alert) -> list[ParsedSignal]` — same.
- `SignalEvaluator.quality_passes(parsed: ParsedSignal) -> bool` — same.
- `SignalEvaluator.size(bankroll: float, parsed: ParsedSignal) -> float` — same. **Note**: argument is `bankroll`, not `nav`. Task 7 passes `self._config.starting_bankroll_usd`.
- `ParsedSignal(condition_id, side, rule_variant=None, metadata={})` — defined T1, used everywhere.
- `PaperTradesRepo.insert_entry(..., triggering_alert_detector, rule_variant, ...)` — signature pinned in T2; used T7.
- `OpenPaperPosition.triggering_alert_detector` and `.rule_variant` — T2.
- `EvaluatorsConfig.<source>.enabled` — T3, used T8.
- `SmartMoneyEvaluatorConfig.position_fraction = 0.01`, `VelocityEvaluatorConfig.position_fraction = 0.0025` — T3.

### Placeholder scan

No `TBD` / `TODO` / "fill in details" in the plan. Every step has runnable code or commands with expected output.

### Commit cadence

8 commits, one per task.

---

## Out-of-plan follow-ups (not blocking)

- **Per-bucket sizing inside Evaluators.** Once we have a few weeks of resolution data per `(detector, rule_variant, severity, change_pct_bucket, market_category)`, the `size` method on the relevant Evaluator can grow data-driven rules. Architecture supports it; data does not yet.
- **Aggregate open-exposure cap.** Deferred per user direction. Worth revisiting once we observe resolution cadence in production.
- **Velocity rule winners.** After accumulating resolved pairs, query
  `summary_by_source` segmented by category/severity to refine velocity's `parse` (e.g., only fade in sports, only follow in thesis markets) — a 1-class change inside `VelocityEvaluator`.
- **Convergence as a future signal.** Not in scope. Detector enrichment would be needed (alert body lacks direction). New `ConvergenceEvaluator` once that's done.
