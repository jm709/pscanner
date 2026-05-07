# Gate-model evaluator (Issue #80) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `GateModelEvaluator` so `gate_buy` alerts emitted by `GateModelDetector` (#79) drive paper trades through the existing `PaperTrader` infrastructure. The evaluator's edge will be observable in `pscanner paper status` and the per-source breakdown table.

**Architecture:** New `pscanner.strategies.evaluators.gate_model.GateModelEvaluator` implementing the `SignalEvaluator` Protocol (`accepts`/`parse`/`quality_passes`/`size`). Single-leg signal — one `ParsedSignal` per accepted alert, no twin/paired logic. Quality gate reads `pred - implied` from the alert body and rejects below `min_edge_pct`. Sizing is constant `bankroll * position_fraction` (matches the project's "constant sizing, infinite paper bankroll" research config per CLAUDE.md). Wired into `_build_paper_evaluators` after the existing 5 evaluators.

**Tech Stack:** Python 3.13, pytest. Quick verify: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Spec:** GitHub issue #80 (under RFC #77). Depends on #79 (`GateModelDetector` + `gate_buy` in `DetectorName`) being merged.

---

## File map

- **Create** `src/pscanner/strategies/evaluators/gate_model.py` — `GateModelEvaluator` class. ~80 lines.
- **Modify** `src/pscanner/config.py` — add `GateModelEvaluatorConfig` `_Section` subclass; wire into `EvaluatorsConfig` (or wherever the per-evaluator configs aggregate; mirrors `SmartMoneyEvaluatorConfig` placement).
- **Modify** `src/pscanner/scheduler.py:388-426` (`_build_paper_evaluators`) — append `GateModelEvaluator` when enabled.
- **Create** `tests/strategies/test_gate_model_evaluator.py` — unit tests for the evaluator.
- **Create** `tests/strategies/test_gate_model_paper_integration.py` — end-to-end: synthetic `gate_buy` alert → `paper_trades` row.
- **Modify** `CLAUDE.md` — add a paragraph under "Codebase conventions" describing the gate evaluator and the per-source breakdown row.

Out of scope: the detector itself (#79) and the live history provider (#78) — both must be merged before this plan runs.

---

### Task 1: `GateModelEvaluatorConfig` in `config.py`

**Files:**
- Modify: `src/pscanner/config.py`
- Test: `tests/test_config.py`

The config tracks two fields per the issue spec: `min_edge_pct` (quality gate floor) and `position_fraction` (constant sizing factor). Defaults: 0.01 and 0.005 (1¢ edge floor, 0.5% bankroll per bet).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
from pscanner.config import Config, GateModelEvaluatorConfig


def test_gate_model_evaluator_config_defaults() -> None:
    cfg = GateModelEvaluatorConfig()
    assert cfg.enabled is False
    assert cfg.min_edge_pct == 0.01
    assert cfg.position_fraction == 0.005


def test_root_config_aggregates_gate_model_evaluator() -> None:
    cfg = Config()
    assert isinstance(cfg.evaluators.gate_model, GateModelEvaluatorConfig)
```

(Note: the test assumes evaluator configs hang off `cfg.evaluators.<name>`. Verify this matches the existing layout for `cfg.evaluators.smart_money` etc. before writing the test — if the actual layout differs, mirror it.)

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_config.py -k gate_model_evaluator -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add `GateModelEvaluatorConfig`**

In `src/pscanner/config.py`, place near the existing `SmartMoneyEvaluatorConfig`/`VelocityEvaluatorConfig` definitions:

```python
class GateModelEvaluatorConfig(_Section):
    """Tunables for the gate-model paper-trading evaluator (#80)."""

    enabled: bool = False
    min_edge_pct: float = 0.01
    position_fraction: float = 0.005
```

Then in the `EvaluatorsConfig` aggregator (where the existing `smart_money: SmartMoneyEvaluatorConfig` etc. live), add:

```python
    gate_model: GateModelEvaluatorConfig = Field(
        default_factory=GateModelEvaluatorConfig
    )
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/test_config.py -k gate_model_evaluator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/config.py tests/test_config.py
git commit -m "feat(config): GateModelEvaluatorConfig for #80"
```

---

### Task 2: Skeleton `GateModelEvaluator` — `accepts` + `parse`

**Files:**
- Create: `src/pscanner/strategies/evaluators/gate_model.py`
- Create: `tests/strategies/test_gate_model_evaluator.py`

Mirror `SmartMoneyEvaluator` structure (`pscanner.strategies.evaluators.smart_money`). `accepts` returns True for `detector == "gate_buy"`. `parse` reads `condition_id`, `side`, `pred`, `implied_prob_at_buy` from the alert body and emits one `ParsedSignal`.

- [ ] **Step 1: Write the failing test**

Create `tests/strategies/test_gate_model_evaluator.py`:

```python
"""Unit tests for GateModelEvaluator (#80)."""

from __future__ import annotations

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import GateModelEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
from pscanner.strategies.evaluators.protocol import ParsedSignal


def _make_alert(
    *,
    detector: str = "gate_buy",
    condition_id: str = "0xc1",
    side: str = "YES",
    pred: float = 0.85,
    implied: float = 0.40,
) -> Alert:
    return Alert(
        alert_key=f"gate:{condition_id}:{side}",
        detector=detector,
        severity="med",
        created_at=1_700_000_000,
        body={
            "wallet": "0xabc",
            "condition_id": condition_id,
            "side": side,
            "pred": pred,
            "implied_prob_at_buy": implied,
            "edge": pred - implied,
            "top_category": "esports",
            "model_version": "abc123",
            "trade_ts": 1_700_000_000,
            "bet_size_usd": 42.0,
        },
    )


def test_accepts_only_gate_buy() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    assert evaluator.accepts(_make_alert(detector="gate_buy")) is True
    assert evaluator.accepts(_make_alert(detector="smart_money")) is False
    assert evaluator.accepts(_make_alert(detector="velocity")) is False


def test_parse_returns_one_signal_with_metadata() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    alert = _make_alert(condition_id="0xc1", side="YES", pred=0.85, implied=0.40)
    signals = evaluator.parse(alert)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.condition_id == ConditionId("0xc1")
    assert sig.side == "YES"
    assert sig.rule_variant is None
    assert sig.metadata["pred"] == pytest.approx(0.85)
    assert sig.metadata["implied"] == pytest.approx(0.40)
    assert sig.metadata["edge"] == pytest.approx(0.45)


def test_parse_returns_empty_on_bad_body_shape() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    alert = Alert(
        alert_key="gate:bad",
        detector="gate_buy",
        severity="med",
        created_at=1_700_000_000,
        body={"missing": "fields"},
    )
    assert evaluator.parse(alert) == []


def test_parse_rejects_non_dict_body() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    alert = Alert(
        alert_key="gate:bad",
        detector="gate_buy",
        severity="med",
        created_at=1_700_000_000,
        body="not-a-dict",  # type: ignore[arg-type]
    )
    assert evaluator.parse(alert) == []
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/strategies/test_gate_model_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the evaluator**

Create `src/pscanner/strategies/evaluators/gate_model.py`:

```python
"""``GateModelEvaluator`` — paper-trade ``gate_buy`` alerts (#80).

Single-leg evaluator: one ParsedSignal per alert. No twin/paired logic
(that's velocity's domain). Sizing is constant ``bankroll * position_fraction``
matching the project's "constant sizing, infinite paper bankroll"
research config per CLAUDE.md. The evaluator does NOT call the booster —
the prediction comes pre-computed in the alert body, written by
:class:`GateModelDetector`.
"""

from __future__ import annotations

import structlog

from pscanner.alerts.models import Alert
from pscanner.config import GateModelEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.strategies.evaluators.protocol import ParsedSignal

_LOG = structlog.get_logger(__name__)


class GateModelEvaluator:
    """Paper-trades alerts emitted by the gate-model detector."""

    def __init__(self, *, config: GateModelEvaluatorConfig) -> None:
        self._config = config

    def accepts(self, alert: Alert) -> bool:
        """Return True iff the alert came from the gate-model detector."""
        return alert.detector == "gate_buy"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Pull condition_id/side/pred/implied from the alert body."""
        body = alert.body if isinstance(alert.body, dict) else {}
        condition_id = body.get("condition_id")
        side = body.get("side")
        pred = body.get("pred")
        implied = body.get("implied_prob_at_buy")
        if not (
            isinstance(condition_id, str)
            and side in ("YES", "NO")
            and isinstance(pred, (int, float))
            and isinstance(implied, (int, float))
        ):
            _LOG.debug("gate_evaluator.bad_body", alert_key=alert.alert_key)
            return []
        return [
            ParsedSignal(
                condition_id=ConditionId(condition_id),
                side=side,
                rule_variant=None,
                metadata={
                    "pred": float(pred),
                    "implied": float(implied),
                    "edge": float(pred) - float(implied),
                },
            )
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Stub — quality gate lands in Task 3."""
        del parsed
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Stub — sizing lands in Task 4."""
        del bankroll, parsed
        return 0.0
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/strategies/test_gate_model_evaluator.py -v`
Expected: PASS for the four tests written so far (`accepts`, `parse_returns_one_signal_with_metadata`, `parse_returns_empty_on_bad_body_shape`, `parse_rejects_non_dict_body`).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/strategies/evaluators/gate_model.py tests/strategies/test_gate_model_evaluator.py
git commit -m "feat(strategies): GateModelEvaluator skeleton (accepts + parse)"
```

---

### Task 3: `quality_passes` — edge-floor gate

**Files:**
- Modify: `src/pscanner/strategies/evaluators/gate_model.py`
- Test: `tests/strategies/test_gate_model_evaluator.py`

Reject signals where `metadata["edge"] < min_edge_pct`. The detector already gates on the same threshold, but defensive double-check protects against any drift between detector config and evaluator config (e.g., operator changes one but forgets the other).

- [ ] **Step 1: Write the failing tests**

Add to `tests/strategies/test_gate_model_evaluator.py`:

```python
def test_quality_passes_accepts_above_floor() -> None:
    evaluator = GateModelEvaluator(
        config=GateModelEvaluatorConfig(min_edge_pct=0.01)
    )
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.85, "implied": 0.40, "edge": 0.45},
    )
    assert evaluator.quality_passes(parsed) is True


def test_quality_passes_rejects_below_floor() -> None:
    evaluator = GateModelEvaluator(
        config=GateModelEvaluatorConfig(min_edge_pct=0.10)
    )
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.45, "implied": 0.40, "edge": 0.05},
    )
    assert evaluator.quality_passes(parsed) is False


def test_quality_passes_rejects_missing_edge_metadata() -> None:
    evaluator = GateModelEvaluator(config=GateModelEvaluatorConfig())
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.85},  # no edge key
    )
    assert evaluator.quality_passes(parsed) is False
```

- [ ] **Step 2: Run, expect 2/3 fail (the stub returns True for everything)**

Run: `uv run pytest tests/strategies/test_gate_model_evaluator.py -k quality_passes -v`
Expected: FAIL — `test_quality_passes_rejects_below_floor` and `test_quality_passes_rejects_missing_edge_metadata`. The accept-above-floor case passes incidentally.

- [ ] **Step 3: Implement the gate**

Replace the `quality_passes` stub in `gate_model.py`:

```python
    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Reject signals whose edge falls below ``min_edge_pct``."""
        edge = parsed.metadata.get("edge")
        if not isinstance(edge, (int, float)):
            return False
        return float(edge) >= self._config.min_edge_pct
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/strategies/test_gate_model_evaluator.py -k quality_passes -v`
Expected: PASS for all three.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/strategies/evaluators/gate_model.py tests/strategies/test_gate_model_evaluator.py
git commit -m "feat(strategies): edge-floor quality gate on GateModelEvaluator"
```

---

### Task 4: `size` — constant `bankroll * position_fraction`

**Files:**
- Modify: `src/pscanner/strategies/evaluators/gate_model.py`
- Test: `tests/strategies/test_gate_model_evaluator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/strategies/test_gate_model_evaluator.py`:

```python
def test_size_is_constant_bankroll_fraction() -> None:
    evaluator = GateModelEvaluator(
        config=GateModelEvaluatorConfig(position_fraction=0.005)
    )
    parsed = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        rule_variant=None,
        metadata={"pred": 0.85, "implied": 0.40, "edge": 0.45},
    )
    assert evaluator.size(bankroll=10_000.0, parsed=parsed) == pytest.approx(50.0)
    assert evaluator.size(bankroll=1_000.0, parsed=parsed) == pytest.approx(5.0)


def test_size_independent_of_metadata() -> None:
    """Sizing is uniform across alerts — no Kelly, no edge-scaled sizing."""
    evaluator = GateModelEvaluator(
        config=GateModelEvaluatorConfig(position_fraction=0.01)
    )
    a = ParsedSignal(
        condition_id=ConditionId("0xc1"),
        side="YES",
        metadata={"pred": 0.55, "implied": 0.50, "edge": 0.05},
    )
    b = ParsedSignal(
        condition_id=ConditionId("0xc2"),
        side="NO",
        metadata={"pred": 0.95, "implied": 0.10, "edge": 0.85},
    )
    assert evaluator.size(bankroll=1_000.0, parsed=a) == evaluator.size(
        bankroll=1_000.0, parsed=b
    )
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/strategies/test_gate_model_evaluator.py -k test_size -v`
Expected: FAIL — stub returns 0.0.

- [ ] **Step 3: Implement `size`**

Replace the `size` stub:

```python
    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return constant ``bankroll * position_fraction`` per signal."""
        del parsed  # uniform sizing across alerts by design
        return bankroll * self._config.position_fraction
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/strategies/test_gate_model_evaluator.py -v`
Expected: ALL PASS (the seven tests from Tasks 2-4).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/strategies/evaluators/gate_model.py tests/strategies/test_gate_model_evaluator.py
git commit -m "feat(strategies): constant sizing on GateModelEvaluator"
```

---

### Task 5: Wire into `_build_paper_evaluators`

**Files:**
- Modify: `src/pscanner/scheduler.py:388-426`
- Test: `tests/scheduler/test_gate_model_evaluator_wiring.py`

The scheduler's `_build_paper_evaluators` returns the ordered list of evaluators that `PaperTrader` walks per alert. Add the gate-model evaluator at the end (after velocity) so existing evaluators take precedence on detectors they share — though in practice no other evaluator accepts `gate_buy`, so order is informational.

- [ ] **Step 1: Write the failing test**

Create `tests/scheduler/test_gate_model_evaluator_wiring.py`:

```python
"""Wiring test: scheduler builds GateModelEvaluator when enabled (#80)."""

from __future__ import annotations

from pathlib import Path

from pscanner.config import Config, GateModelEvaluatorConfig
from pscanner.scheduler import Scheduler
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator


def test_scheduler_builds_gate_model_evaluator_when_enabled(tmp_path: Path) -> None:
    cfg = Config()
    cfg.evaluators.gate_model = GateModelEvaluatorConfig(enabled=True)
    sched = Scheduler.build_for_test(config=cfg, tmp_path=tmp_path)
    evaluators = sched.build_paper_evaluators()
    assert any(isinstance(e, GateModelEvaluator) for e in evaluators)


def test_scheduler_omits_gate_model_evaluator_when_disabled(tmp_path: Path) -> None:
    cfg = Config()
    cfg.evaluators.gate_model = GateModelEvaluatorConfig(enabled=False)
    sched = Scheduler.build_for_test(config=cfg, tmp_path=tmp_path)
    evaluators = sched.build_paper_evaluators()
    assert not any(isinstance(e, GateModelEvaluator) for e in evaluators)
```

(`Scheduler.build_for_test` is the test helper introduced in #79's plan. If #79 is still in flight, write a minimal `_build_evaluators_for_test` helper in this plan's task instead.)

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/scheduler/test_gate_model_evaluator_wiring.py -v`
Expected: FAIL — evaluator not yet wired.

- [ ] **Step 3: Add the wiring**

In `src/pscanner/scheduler.py`, locate `_build_paper_evaluators`. Add the import:

```python
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
```

After the existing `if cfg.evaluators.velocity.enabled: ...` block (and before the function returns), add:

```python
        if cfg.evaluators.gate_model.enabled:
            evaluators.append(
                GateModelEvaluator(config=cfg.evaluators.gate_model)
            )
```

- [ ] **Step 4: Re-run the wiring tests, expect pass**

Run: `uv run pytest tests/scheduler/test_gate_model_evaluator_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/scheduler.py tests/scheduler/test_gate_model_evaluator_wiring.py
git commit -m "feat(scheduler): wire GateModelEvaluator into _build_paper_evaluators"
```

---

### Task 6: End-to-end paper-trade integration test

**Files:**
- Create: `tests/strategies/test_gate_model_paper_integration.py`

Synthesize a `gate_buy` alert, push it through `PaperTrader.evaluate`, assert a row lands in `paper_trades` with `triggering_alert_detector="gate_buy"` and `rule_variant IS NULL`.

- [ ] **Step 1: Write the integration test**

Create `tests/strategies/test_gate_model_paper_integration.py`:

```python
"""End-to-end: gate_buy alert -> PaperTrader -> paper_trades row (#80)."""

from __future__ import annotations

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import GateModelEvaluatorConfig
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    PaperTradesRepo,
)
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
from pscanner.strategies.paper_trader import PaperTrader


@pytest.mark.asyncio
async def test_gate_buy_alert_books_paper_trade(tmp_db) -> None:  # type: ignore[no-untyped-def]
    # Seed market_cache so PaperTrader can resolve YES asset_id + fill price.
    market_cache = MarketCacheRepo(tmp_db)
    market_cache.upsert(
        market_id="m1",
        event_id="e1",
        condition_id="0xc1",
        title="Esports market",
        outcomes=[("0xa1", "YES"), ("0xa2", "NO")],
        liquidity_usd=1000.0,
        volume_usd=5_000.0,
        end_time_iso="2027-01-01T00:00:00Z",
    )
    paper_trades = PaperTradesRepo(tmp_db)
    alerts_repo = AlertsRepo(tmp_db)
    evaluator = GateModelEvaluator(
        config=GateModelEvaluatorConfig(
            enabled=True, min_edge_pct=0.01, position_fraction=0.005
        )
    )
    trader = PaperTrader(
        evaluators=[evaluator],
        paper_trades=paper_trades,
        market_cache=market_cache,
        alerts_repo=alerts_repo,
        starting_bankroll_usd=10_000.0,
    )
    alert = Alert(
        alert_key="gate:tx-1:YES",
        detector="gate_buy",
        severity="med",
        created_at=1_700_000_000,
        body={
            "wallet": "0xabc",
            "condition_id": "0xc1",
            "side": "YES",
            "pred": 0.85,
            "implied_prob_at_buy": 0.40,
            "edge": 0.45,
            "top_category": "esports",
            "model_version": "abc123",
            "trade_ts": 1_700_000_000,
            "bet_size_usd": 50.0,
        },
    )
    await trader.evaluate(alert)
    rows = list(paper_trades.iter_all())
    assert len(rows) == 1
    row = rows[0]
    assert row.triggering_alert_detector == "gate_buy"
    assert row.rule_variant is None
    assert row.cost_usd == pytest.approx(50.0)  # 10_000 * 0.005
```

(`PaperTrader.__init__` signature is approximate — match the actual constructor in `src/pscanner/strategies/paper_trader.py` and adjust args/kwarg names. Same for `PaperTradesRepo.iter_all`. The test's value is the assertion shape, not the constructor minutiae.)

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/strategies/test_gate_model_paper_integration.py -v`
Expected: PASS — one row in `paper_trades`.

If it fails because `PaperTrader._resolve_outcome` can't find the YES asset, double-check that `MarketCacheRepo.upsert` writes `outcomes` as the test expects. If `paper_trades` ends up empty without a clear failure, add `caplog`/`structlog.testing.capture_logs` to the test to surface why `PaperTrader._run_pipeline` skipped the signal (e.g., `quality_passes=False` because of a metadata-shape mismatch).

- [ ] **Step 3: Commit**

```bash
git add tests/strategies/test_gate_model_paper_integration.py
git commit -m "test(strategies): end-to-end gate_buy -> paper_trades row"
```

---

### Task 7: `pscanner paper status` per-source row check

**Files:**
- Modify: `tests/strategies/test_gate_model_paper_integration.py` (extend)

Per the issue spec: "`pscanner paper status` per-source breakdown picks up the new `(triggering_alert_detector="gate_buy", rule_variant=NULL)` row automatically because the table groups by those two columns." We don't need to add rendering code — just verify the aggregation query returns the expected row.

- [ ] **Step 1: Add the breakdown test**

Append to `tests/strategies/test_gate_model_paper_integration.py`:

```python
@pytest.mark.asyncio
async def test_per_source_breakdown_includes_gate_buy(tmp_db) -> None:  # type: ignore[no-untyped-def]
    # Build the same setup as test_gate_buy_alert_books_paper_trade and
    # then run the per-source aggregation query directly.
    market_cache = MarketCacheRepo(tmp_db)
    market_cache.upsert(
        market_id="m1",
        event_id="e1",
        condition_id="0xc1",
        title="Esports",
        outcomes=[("0xa1", "YES"), ("0xa2", "NO")],
        liquidity_usd=1000.0,
        volume_usd=5_000.0,
        end_time_iso="2027-01-01T00:00:00Z",
    )
    paper_trades = PaperTradesRepo(tmp_db)
    alerts_repo = AlertsRepo(tmp_db)
    evaluator = GateModelEvaluator(
        config=GateModelEvaluatorConfig(
            enabled=True, min_edge_pct=0.01, position_fraction=0.005
        )
    )
    trader = PaperTrader(
        evaluators=[evaluator],
        paper_trades=paper_trades,
        market_cache=market_cache,
        alerts_repo=alerts_repo,
        starting_bankroll_usd=10_000.0,
    )
    for i in range(3):
        await trader.evaluate(
            Alert(
                alert_key=f"gate:tx-{i}:YES",
                detector="gate_buy",
                severity="med",
                created_at=1_700_000_000 + i,
                body={
                    "wallet": "0xabc",
                    "condition_id": "0xc1",
                    "side": "YES",
                    "pred": 0.85,
                    "implied_prob_at_buy": 0.40,
                    "edge": 0.45,
                    "top_category": "esports",
                    "model_version": "abc123",
                    "trade_ts": 1_700_000_000 + i,
                    "bet_size_usd": 50.0,
                },
            )
        )
    # The per-source breakdown query (mirror what `pscanner paper status` runs).
    rows = tmp_db.execute(
        """
        SELECT triggering_alert_detector, rule_variant, COUNT(*) AS n
        FROM paper_trades
        GROUP BY triggering_alert_detector, rule_variant
        """
    ).fetchall()
    by_source = {(row[0], row[1]): row[2] for row in rows}
    assert by_source.get(("gate_buy", None)) == 3
```

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/strategies/test_gate_model_paper_integration.py::test_per_source_breakdown_includes_gate_buy -v`
Expected: PASS — three rows aggregated under `("gate_buy", None)`.

- [ ] **Step 3: Commit**

```bash
git add tests/strategies/test_gate_model_paper_integration.py
git commit -m "test(strategies): per-source breakdown aggregates gate_buy correctly"
```

---

### Task 8: CLAUDE.md note + final verify

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the documentation paragraph**

Find "Codebase conventions" in `CLAUDE.md`. Either add a new bullet near the existing `**Paper-trading evaluators**: ...` bullet or extend that bullet. Suggested addition (new bullet):

```markdown
- **Gate-model evaluator (#80).** `pscanner.strategies.evaluators.gate_model.GateModelEvaluator` differs from the other 5 evaluators in that it consumes a model prediction from the alert body rather than re-deriving anything from market state. The detector (#79) does the booster.predict; the evaluator's `quality_passes` only re-checks the edge floor as a defensive double-check (catches operator config drift between detector `min_edge_pct` and evaluator `min_edge_pct`). Sizing is constant `bankroll * position_fraction` (default 0.5%, no Kelly, no edge-scaled sizing — uniform across alerts). The per-source breakdown row in `pscanner paper status` aggregates under `(triggering_alert_detector="gate_buy", rule_variant=NULL)` automatically.
```

- [ ] **Step 2: Run the full verify gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: ALL PASS, no warnings (`pyproject.toml` has `filterwarnings = ["error"]`).

- [ ] **Step 3: Live smoke (manual)**

With #78, #79, and this PR landed, and `data/pscanner.sqlite3` populated via `pscanner daemon bootstrap-features`:

```bash
# In your project config, set:
#   gate_model.enabled = true
#   gate_model.artifact_dir = "models/current"
#   gate_model_market_filter.enabled = true
#   evaluators.gate_model.enabled = true

uv run pscanner run --once
uv run pscanner paper status
```

Expected: at least one `gate_buy` row in `pscanner status` output, and the `paper status` per-source breakdown shows a `gate_buy` row with `n_trades >= 1`. If `n_trades == 0` after a 5-minute run, check for `gate_model.queue_full` warnings in the daemon log — the test scope is small enough that the queue should never fill.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note gate-model evaluator semantics in CLAUDE.md"
```

---

## Self-review checklist

- **Spec coverage:** every DoD item in #80 has a task. Config (Task 1), `accepts`/`parse` (Task 2), `quality_passes` (Task 3), `size` (Task 4), scheduler wiring (Task 5), end-to-end paper-trade test (Task 6), per-source breakdown test (Task 7), docs + verify (Task 8).
- **No placeholders:** the integration test in Task 6 calls real `PaperTrader.__init__` — if the actual signature differs, adjust at run time. The skill's instructions allow this kind of "match the actual API" fix-up during execution; the test's *intent* (one alert in, one row out, with the right detector/rule_variant) is concrete.
- **Type consistency:** `GateModelEvaluator(config: GateModelEvaluatorConfig)` matches across all tasks. `ParsedSignal(condition_id=ConditionId(...), side, rule_variant=None, metadata={"pred":..., "implied":..., "edge":...})` shape is identical between Task 2 and Task 6.
- **No paper-trade feedback loop:** the evaluator does not call `provider.observe(...)`. Per RFC #77 Q4, paper trades must NOT update the streaming history provider's state — only real on-chain trades observed via the trade collector get folded in. The detector (#79) calls `provider.observe(feature_trade)` for every scored trade; the evaluator does nothing to provider state. This plan preserves that invariant by simply not touching the provider.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-06-gate-evaluator.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans`.

This plan depends on `2026-05-06-gate-detector.md` (Issue #79) being merged first, which itself depends on `2026-05-06-gate-live-history.md` (Issue #78). Run order: #78 → #79 → #80.

After this plan lands, the gate-model loop is complete. Follow-ups deferred to v2 per RFC #77: hot-reload (Q3), drift detection (Q5), v1.1 sports-expansion (config flag flip — no code change).
