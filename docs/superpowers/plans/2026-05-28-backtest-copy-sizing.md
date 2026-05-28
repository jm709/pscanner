# Backtest Copy-Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python script that backtests four sizing schemes (equal-weight, concentration-capped, follow-seed-size, edge-weighted causal) against the historical trade stream of the daemon's current watchlist, holding fill price and resolution constant so sizing is the only variable.

**Architecture:** A single script at `scripts/backtest_copy_sizing.py` loads watchlist + trades + resolutions via DuckDB, materializes them as a chronologically-ordered event stream, walks the stream once through a per-scheme state machine (each `on_trade` sizes the position, each `on_resolution` books the payout), and renders a markdown report. No imports from `pscanner.*` — the script re-expresses sizing logic standalone so it can be evolved without touching the daemon.

**Tech Stack:** Python 3.13, `duckdb`, `argparse`, stdlib `dataclasses`. Tests use `pytest` + structlog log capture. Spec: `docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md`.

---

## File structure

| File | Purpose |
|---|---|
| Create: `scripts/backtest_copy_sizing.py` | The script — CLI, DuckDB query, simulator, report rendering. Single file (~450 LOC) per the operator-script convention (see `scripts/wallet_edge_leaderboard.py`). |
| Create: `tests/scripts/test_backtest_sizing_schemes.py` | Unit tests for the four sizing scheme classes. |
| Create: `tests/scripts/test_backtest_simulator.py` | Integration test for the event walk + temporal-correctness guarantee. |
| Create: `tests/slow/test_backtest_corpus_smoke.py` | Opt-in (`pytest -m slow`) smoke test against the production corpus. Implementation pauses for operator approval before running. |

Repo conventions to follow:
- Script header: `# ruff: noqa: T201  # script prints diagnostics to stdout by design`
- `from __future__ import annotations`
- Constants as `_NAME: Final[...] = ...`
- Slow test marker: `pytestmark = pytest.mark.slow`
- New test file? Update `tests/scripts/__init__.py` is not needed — it already exists.

---

## Task 1: Scaffold module + EqualWeight scheme (TDD)

**Files:**
- Create: `scripts/backtest_copy_sizing.py`
- Test: `tests/scripts/test_backtest_sizing_schemes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_backtest_sizing_schemes.py
"""Unit tests for backtest sizing schemes."""

from __future__ import annotations

from scripts.backtest_copy_sizing import EqualWeight, Trade


def _trade(
    *,
    wallet: str = "0xwallet",
    condition_id: str = "0xcond",
    outcome_side: str = "YES",
    price: float = 0.5,
    notional_usd: float = 1000.0,
    ts: int = 1_700_000_000,
) -> Trade:
    return Trade(
        wallet=wallet,
        condition_id=condition_id,
        outcome_side=outcome_side,
        price=price,
        notional_usd=notional_usd,
        ts=ts,
    )


def test_equal_weight_returns_constant_cost() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    cost = scheme.compute(_trade(), bankroll=10_000.0)
    assert cost == 100.0


def test_equal_weight_ignores_trade_details() -> None:
    scheme = EqualWeight(position_fraction=0.02)
    cost_a = scheme.compute(_trade(price=0.1, notional_usd=50.0), bankroll=5_000.0)
    cost_b = scheme.compute(_trade(price=0.9, notional_usd=500.0), bankroll=5_000.0)
    assert cost_a == cost_b == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: `ImportError: cannot import name 'EqualWeight' from 'scripts.backtest_copy_sizing'` (or similar — the module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/backtest_copy_sizing.py
"""Backtest harness for copy-trading sizing schemes.

Reads the daemon's current watchlist, pulls every position-increasing
(BUY) trade by those wallets from `corpus_trades`, joins with
`market_resolutions`, and walks the chronological event stream
through four sizing schemes. Outputs a markdown report comparing
their realized PnL, ROI, win rate, drawdown, and quarterly P&L.

Spec: docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    """One BUY trade by a watchlist wallet."""

    wallet: str
    condition_id: str
    outcome_side: str  # "YES" or "NO"
    price: float
    notional_usd: float
    ts: int


class EqualWeight:
    """cost = bankroll * position_fraction. No state, no scaling."""

    name = "equal_weight"

    def __init__(self, *, position_fraction: float) -> None:
        self._position_fraction = position_fraction

    def compute(self, trade: Trade, bankroll: float) -> float:
        del trade
        return bankroll * self._position_fraction

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        del trade, payout
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_sizing_schemes.py
git commit -m "feat(backtest): scaffold module with EqualWeight sizing scheme"
```

---

## Task 2: ConcentrationCapped scheme (production reference)

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_sizing_schemes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_backtest_sizing_schemes.py`:

```python
from scripts.backtest_copy_sizing import ConcentrationCapped


def test_concentration_capped_first_trade_per_wallet_is_unit_multiplier() -> None:
    scheme = ConcentrationCapped(
        position_fraction=0.01, min_multiplier=0.10, watchlist_size=10
    )
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    # First trade ever: counts[A]==0 BEFORE compute, share=0, target=0.1,
    # raw=min(1, 0.1/max(0, 0.1))=1.0, mult=1.0.
    assert cost == 100.0


def test_concentration_capped_decays_with_repeat_offender() -> None:
    scheme = ConcentrationCapped(
        position_fraction=0.01, min_multiplier=0.10, watchlist_size=10
    )
    # Two trades by A, one by B. Counts BEFORE A's 3rd would be {A: 2, B: 1}.
    scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    scheme.compute(_trade(wallet="0xB"), bankroll=10_000.0)
    # Now A's 3rd: counts={A:2, B:1}, total=3, share=2/3, target=1/10,
    # raw=min(1, 0.1/(2/3))=0.15, mult=max(0.15, 0.10)=0.15.
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    assert cost == 10_000.0 * 0.01 * 0.15


def test_concentration_capped_floors_at_min_multiplier() -> None:
    scheme = ConcentrationCapped(
        position_fraction=0.01, min_multiplier=0.25, watchlist_size=100
    )
    # Hammer the same wallet 50 times.
    for _ in range(50):
        scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    # By trade 51 the raw multiplier is far below 0.25; the floor kicks in.
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    assert cost == 10_000.0 * 0.01 * 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 3 new ImportError failures (class doesn't exist).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
class ConcentrationCapped:
    """Production sizing: cost = bankroll * position_fraction * mult.

    Mirrors `pscanner.strategies.evaluators.subgraph_copy
    .SubgraphCopyEvaluator._concentration_multiplier`. The running
    count is incremented AFTER the size is computed so the first trade
    per wallet always gets a clean 1.0 multiplier.
    """

    name = "concentration_capped"

    def __init__(
        self,
        *,
        position_fraction: float,
        min_multiplier: float,
        watchlist_size: int,
    ) -> None:
        self._position_fraction = position_fraction
        self._min_multiplier = min_multiplier
        self._watchlist_size = max(1, watchlist_size)
        self._counts: dict[str, int] = {}

    def compute(self, trade: Trade, bankroll: float) -> float:
        total = sum(self._counts.values())
        share = self._counts.get(trade.wallet, 0) / total if total else 0.0
        target_share = 1.0 / self._watchlist_size
        raw = min(1.0, target_share / max(share, target_share))
        mult = max(raw, self._min_multiplier)
        self._counts[trade.wallet] = self._counts.get(trade.wallet, 0) + 1
        return bankroll * self._position_fraction * mult

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        del trade, payout
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_sizing_schemes.py
git commit -m "feat(backtest): add ConcentrationCapped sizing scheme"
```

---

## Task 3: FollowSeedSize scheme

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_sizing_schemes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_backtest_sizing_schemes.py`:

```python
from scripts.backtest_copy_sizing import FollowSeedSize


def test_follow_seed_size_scales_with_notional() -> None:
    scheme = FollowSeedSize(scale_factor=0.01, max_cost_per_trade=1_000.0)
    cost = scheme.compute(_trade(notional_usd=5_000.0), bankroll=10_000.0)
    assert cost == 50.0  # 5_000 * 0.01


def test_follow_seed_size_caps_at_max_cost_per_trade() -> None:
    scheme = FollowSeedSize(scale_factor=0.01, max_cost_per_trade=1_000.0)
    # 500_000 * 0.01 = 5_000, capped to 1_000.
    cost = scheme.compute(_trade(notional_usd=500_000.0), bankroll=10_000.0)
    assert cost == 1_000.0


def test_follow_seed_size_ignores_bankroll() -> None:
    scheme = FollowSeedSize(scale_factor=0.01, max_cost_per_trade=1_000.0)
    cost_a = scheme.compute(_trade(notional_usd=2_000.0), bankroll=100.0)
    cost_b = scheme.compute(_trade(notional_usd=2_000.0), bankroll=1_000_000.0)
    assert cost_a == cost_b == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 3 new ImportErrors.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
class FollowSeedSize:
    """cost = min(trade.notional_usd * scale_factor, max_cost_per_trade)."""

    name = "follow_seed_size"

    def __init__(self, *, scale_factor: float, max_cost_per_trade: float) -> None:
        self._scale_factor = scale_factor
        self._max_cost = max_cost_per_trade

    def compute(self, trade: Trade, bankroll: float) -> float:
        del bankroll
        return min(trade.notional_usd * self._scale_factor, self._max_cost)

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        del trade, payout
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_sizing_schemes.py
git commit -m "feat(backtest): add FollowSeedSize sizing scheme"
```

---

## Task 4: EdgeWeightedCausal scheme

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_sizing_schemes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_backtest_sizing_schemes.py`:

```python
from scripts.backtest_copy_sizing import EdgeWeightedCausal


def test_edge_weighted_returns_unit_multiplier_when_below_min_trades() -> None:
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=10,
    )
    # 5 prior resolved trades — under the floor.
    for _ in range(5):
        scheme.observe_resolution(_trade(wallet="0xA", price=0.5), payout=1.0)
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    # mult=1.0 → 10_000 * 0.01 * 1.0 = 100.
    assert cost == 100.0


def test_edge_weighted_scales_up_with_positive_edge() -> None:
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=10,
    )
    # 10 wins at price 0.5: edge = 1 - 0.5 = 0.5 per trade.
    for _ in range(10):
        scheme.observe_resolution(_trade(wallet="0xA", price=0.5), payout=1.0)
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    # mult = clip(1 + 5*0.5, 0.25, 3.0) = clip(3.5, ...) = 3.0
    # → 10_000 * 0.01 * 3.0 = 300.
    assert cost == 300.0


def test_edge_weighted_scales_down_with_negative_edge() -> None:
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=10,
    )
    # 10 losses at price 0.5: edge = 0 - 0.5 = -0.5 per trade.
    for _ in range(10):
        scheme.observe_resolution(_trade(wallet="0xA", price=0.5), payout=0.0)
    cost = scheme.compute(_trade(wallet="0xA"), bankroll=10_000.0)
    # mult = clip(1 + 5*-0.5, 0.25, 3.0) = clip(-1.5, ...) = 0.25
    # → 10_000 * 0.01 * 0.25 = 25.
    assert cost == 25.0


def test_edge_weighted_state_is_per_wallet() -> None:
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=2,
    )
    for _ in range(2):
        scheme.observe_resolution(_trade(wallet="0xA", price=0.5), payout=1.0)
    # B has no resolved trades — should still be 1.0x.
    cost_b = scheme.compute(_trade(wallet="0xB"), bankroll=10_000.0)
    assert cost_b == 100.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 4 new ImportErrors.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
class EdgeWeightedCausal:
    """Edge-weighted sizing with strict no-look-ahead.

    `compute` uses ONLY trades passed via `observe_resolution` BEFORE
    this call. The event-walk caller is responsible for processing
    resolutions in `ts` order before any subsequent trade with `ts >=
    resolution.ts`.
    """

    name = "edge_weighted_causal"

    def __init__(
        self,
        *,
        position_fraction: float,
        edge_scale: float,
        min_multiplier: float,
        max_multiplier: float,
        min_trades_for_edge: int,
    ) -> None:
        self._position_fraction = position_fraction
        self._edge_scale = edge_scale
        self._min_multiplier = min_multiplier
        self._max_multiplier = max_multiplier
        self._min_trades_for_edge = min_trades_for_edge
        # Per-wallet: list of (payout, price) pairs for resolved trades.
        self._resolved: dict[str, list[tuple[float, float]]] = {}

    def compute(self, trade: Trade, bankroll: float) -> float:
        prior = self._resolved.get(trade.wallet, [])
        if len(prior) < self._min_trades_for_edge:
            mult = 1.0
        else:
            edge = sum(p - imp for p, imp in prior) / len(prior)
            raw = 1.0 + self._edge_scale * edge
            mult = max(self._min_multiplier, min(self._max_multiplier, raw))
        return bankroll * self._position_fraction * mult

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        self._resolved.setdefault(trade.wallet, []).append((payout, trade.price))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_sizing_schemes.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_sizing_schemes.py
git commit -m "feat(backtest): add EdgeWeightedCausal sizing scheme"
```

---

## Task 5: Event types + Simulator scaffold

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Create: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_backtest_simulator.py
"""Integration tests for the backtest event-walk simulator."""

from __future__ import annotations

from scripts.backtest_copy_sizing import (
    EqualWeight,
    Resolution,
    Simulator,
    Trade,
    TradeEvent,
    ResolutionEvent,
)


def test_simulator_initializes_per_scheme_state() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert state.resolved_trades == []
    assert state.cumulative_pnl == 0.0
    assert state.nav_series == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: ImportError (Simulator, Resolution, TradeEvent, ResolutionEvent don't exist).

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
from typing import Literal, Protocol


@dataclass(frozen=True)
class Resolution:
    """A market resolution event from `market_resolutions`."""

    condition_id: str
    winning_side: str  # "YES" or "NO"
    resolved_at: int


@dataclass(frozen=True)
class TradeEvent:
    kind: Literal["trade"]
    ts: int
    trade: Trade


@dataclass(frozen=True)
class ResolutionEvent:
    kind: Literal["resolution"]
    ts: int
    resolution: Resolution


@dataclass
class OpenPos:
    trade_id: int
    wallet: str
    condition_id: str
    outcome_side: str
    shares: float
    cost: float
    ts: int
    price: float


@dataclass
class ResolvedTradeRecord:
    open_pos: OpenPos
    payout: float
    proceeds: float
    pnl: float
    resolved_at: int


@dataclass
class BacktestState:
    open_positions: dict[int, OpenPos]
    resolved_trades: list[ResolvedTradeRecord]
    cumulative_pnl: float
    nav_series: list[tuple[int, float]]


class SizingScheme(Protocol):
    name: str

    def compute(self, trade: Trade, bankroll: float) -> float: ...

    def observe_resolution(self, trade: Trade, payout: float) -> None: ...


class Simulator:
    """Walks an event stream once, dispatching to per-scheme state."""

    def __init__(self, *, schemes: list[SizingScheme], bankroll: float) -> None:
        self._schemes = schemes
        self._bankroll = bankroll
        self._states: dict[str, BacktestState] = {
            s.name: BacktestState(
                open_positions={},
                resolved_trades=[],
                cumulative_pnl=0.0,
                nav_series=[],
            )
            for s in schemes
        }

    def state_for(self, scheme: SizingScheme) -> BacktestState:
        return self._states[scheme.name]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): add event types and Simulator scaffold"
```

---

## Task 6: Simulator walk — on_trade + on_resolution + temporal correctness

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_backtest_simulator.py`:

```python
def test_simulator_books_pnl_on_resolution() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert len(state.resolved_trades) == 1
    rec = state.resolved_trades[0]
    # cost=100, shares=200, payout=1.0, proceeds=200, pnl=100.
    assert rec.payout == 1.0
    assert rec.proceeds == 200.0
    assert rec.pnl == 100.0
    assert state.cumulative_pnl == 100.0
    assert state.nav_series == [(200, 100.0)]


def test_simulator_books_zero_payout_on_losing_outcome() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="NO", resolved_at=200)
    )
    state = sim.state_for(scheme)
    rec = state.resolved_trades[0]
    assert rec.payout == 0.0
    assert rec.pnl == -100.0


def test_simulator_temporal_correctness_with_edge_weighted() -> None:
    """The simulator must surface resolved trades to EdgeWeightedCausal
    before any subsequent on_trade with ts >= resolution.ts.

    Scenario: 2 prior wins for wallet A at price 0.5 (edge=+0.5/trade),
    then a 3rd trade for A. After the 2 resolutions, the 3rd trade
    must be sized at the boosted multiplier.
    """
    scheme = EdgeWeightedCausal(
        position_fraction=0.01,
        edge_scale=5.0,
        min_multiplier=0.25,
        max_multiplier=3.0,
        min_trades_for_edge=2,
    )
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM2",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=200,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=300)
    )
    sim.on_resolution(
        Resolution(condition_id="0xM2", winning_side="YES", resolved_at=400)
    )
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM3",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=500,
        )
    )
    state = sim.state_for(scheme)
    # Third trade should be sized at the boosted multiplier (3.0x).
    third = state.open_positions[next(iter(state.open_positions))]
    # bankroll * position_fraction * 3.0 = 10000 * 0.01 * 3.0 = 300
    assert third.cost == 300.0


def test_simulator_resolves_multiple_open_positions_on_same_market() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xB",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=150,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
    state = sim.state_for(scheme)
    assert state.open_positions == {}
    assert len(state.resolved_trades) == 2


def test_simulator_unresolved_trade_stays_open() -> None:
    scheme = EqualWeight(position_fraction=0.01)
    sim = Simulator(schemes=[scheme], bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xA",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=500.0,
            ts=100,
        )
    )
    # No resolution.
    state = sim.state_for(scheme)
    assert len(state.open_positions) == 1
    assert state.resolved_trades == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 5 new failures — `Simulator` has no `on_trade` / `on_resolution` methods.

- [ ] **Step 3: Write minimal implementation**

Replace the `Simulator` class in `scripts/backtest_copy_sizing.py`:

```python
class Simulator:
    """Walks an event stream once, dispatching to per-scheme state."""

    def __init__(self, *, schemes: list[SizingScheme], bankroll: float) -> None:
        self._schemes = schemes
        self._bankroll = bankroll
        self._states: dict[str, BacktestState] = {
            s.name: BacktestState(
                open_positions={},
                resolved_trades=[],
                cumulative_pnl=0.0,
                nav_series=[],
            )
            for s in schemes
        }
        self._next_trade_id = 0

    def state_for(self, scheme: SizingScheme) -> BacktestState:
        return self._states[scheme.name]

    def on_trade(self, trade: Trade) -> None:
        """Size and open a position for every scheme."""
        for scheme in self._schemes:
            state = self._states[scheme.name]
            cost = scheme.compute(trade, self._bankroll)
            shares = cost / trade.price if trade.price > 0 else 0.0
            trade_id = self._next_trade_id
            state.open_positions[trade_id] = OpenPos(
                trade_id=trade_id,
                wallet=trade.wallet,
                condition_id=trade.condition_id,
                outcome_side=trade.outcome_side,
                shares=shares,
                cost=cost,
                ts=trade.ts,
                price=trade.price,
            )
        self._next_trade_id += 1

    def on_resolution(self, resolution: Resolution) -> None:
        """Close every open position whose market matches, per scheme."""
        for scheme in self._schemes:
            state = self._states[scheme.name]
            closing = [
                tid
                for tid, pos in state.open_positions.items()
                if pos.condition_id == resolution.condition_id
            ]
            for tid in closing:
                pos = state.open_positions.pop(tid)
                payout = 1.0 if resolution.winning_side == pos.outcome_side else 0.0
                proceeds = pos.shares * payout
                pnl = proceeds - pos.cost
                state.resolved_trades.append(
                    ResolvedTradeRecord(
                        open_pos=pos,
                        payout=payout,
                        proceeds=proceeds,
                        pnl=pnl,
                        resolved_at=resolution.resolved_at,
                    )
                )
                state.cumulative_pnl += pnl
                # Feed the scheme so future trades can see this outcome.
                scheme.observe_resolution(
                    Trade(
                        wallet=pos.wallet,
                        condition_id=pos.condition_id,
                        outcome_side=pos.outcome_side,
                        price=pos.price,
                        notional_usd=pos.cost,
                        ts=pos.ts,
                    ),
                    payout=payout,
                )
                state.nav_series.append(
                    (resolution.resolved_at, state.cumulative_pnl)
                )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 6 passed (1 from Task 5 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): implement Simulator on_trade and on_resolution"
```

---

## Task 7: DuckDB loaders — watchlist + chronological event stream

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_backtest_simulator.py`:

```python
import sqlite3
from pathlib import Path

import pytest

from scripts.backtest_copy_sizing import (
    load_event_stream,
    load_watchlist,
)


def _make_daemon_db(path: Path, addresses: list[str]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE wallet_watchlist ("
        " address TEXT PRIMARY KEY,"
        " source TEXT NOT NULL,"
        " reason TEXT,"
        " added_at INTEGER NOT NULL,"
        " active INTEGER NOT NULL DEFAULT 1)"
    )
    for addr in addresses:
        conn.execute(
            "INSERT INTO wallet_watchlist (address, source, reason, added_at, active)"
            " VALUES (?, 'test', 'test', 1, 1)",
            (addr,),
        )
    conn.commit()
    conn.close()


def _make_corpus_db(
    path: Path,
    trades: list[tuple[str, str, str, str, float, float, int]],
    resolutions: list[tuple[str, int, int]],
) -> None:
    """trades: (wallet, condition_id, outcome_side, bs, price, notional, ts)
    resolutions: (condition_id, outcome_yes_won, resolved_at)
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE corpus_trades (
          platform TEXT NOT NULL DEFAULT 'polymarket',
          tx_hash TEXT NOT NULL,
          asset_id TEXT NOT NULL,
          wallet_address TEXT NOT NULL,
          condition_id TEXT NOT NULL,
          outcome_side TEXT NOT NULL,
          bs TEXT NOT NULL,
          price REAL NOT NULL,
          size REAL NOT NULL,
          notional_usd REAL NOT NULL,
          ts INTEGER NOT NULL,
          PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)
        );
        CREATE TABLE market_resolutions (
          platform TEXT NOT NULL DEFAULT 'polymarket',
          condition_id TEXT NOT NULL,
          winning_outcome_index INTEGER NOT NULL,
          outcome_yes_won INTEGER NOT NULL,
          resolved_at INTEGER NOT NULL,
          source TEXT NOT NULL,
          recorded_at INTEGER NOT NULL,
          PRIMARY KEY (platform, condition_id)
        );
        """
    )
    for i, (w, cid, side, bs, price, notional, ts) in enumerate(trades):
        conn.execute(
            "INSERT INTO corpus_trades VALUES "
            "('polymarket', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"0xtx{i}", f"asset{i}", w, cid, side, bs, price, notional / price, notional, ts),
        )
    for cid, yes_won, resolved_at in resolutions:
        conn.execute(
            "INSERT INTO market_resolutions VALUES "
            "('polymarket', ?, ?, ?, ?, 'test', ?)",
            (cid, 0 if yes_won else 1, yes_won, resolved_at, resolved_at),
        )
    conn.commit()
    conn.close()


def test_load_watchlist_returns_active_addresses(tmp_path: Path) -> None:
    db = tmp_path / "daemon.sqlite3"
    _make_daemon_db(db, ["0xa", "0xb"])
    addrs = load_watchlist(db)
    assert addrs == {"0xa", "0xb"}


def test_load_watchlist_excludes_inactive(tmp_path: Path) -> None:
    db = tmp_path / "daemon.sqlite3"
    _make_daemon_db(db, ["0xa"])
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO wallet_watchlist VALUES ('0xc', 'test', 'test', 1, 0)"
    )
    conn.commit()
    conn.close()
    addrs = load_watchlist(db)
    assert addrs == {"0xa"}


def test_load_event_stream_orders_by_ts(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 100),
            ("0xa", "0xM2", "NO", "BUY", 0.3, 500.0, 300),
        ],
        resolutions=[
            ("0xM1", 1, 200),  # M1 YES wins at ts=200
            ("0xM2", 0, 400),  # M2 NO wins at ts=400
        ],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    assert [e.ts for e in events] == [100, 200, 300, 400]
    assert isinstance(events[0], TradeEvent)
    assert isinstance(events[1], ResolutionEvent)
    assert events[1].resolution.winning_side == "YES"
    assert events[3].resolution.winning_side == "NO"


def test_load_event_stream_excludes_sells(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "SELL", 0.5, 1_000.0, 100),
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 110),
        ],
        resolutions=[("0xM1", 1, 200)],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    trade_events = [e for e in events if isinstance(e, TradeEvent)]
    assert len(trade_events) == 1
    assert trade_events[0].ts == 110


def test_load_event_stream_excludes_non_watchlist(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 100),
            ("0xb", "0xM1", "YES", "BUY", 0.5, 1_000.0, 110),
        ],
        resolutions=[("0xM1", 1, 200)],
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    trade_events = [e for e in events if isinstance(e, TradeEvent)]
    assert len(trade_events) == 1
    assert trade_events[0].trade.wallet == "0xa"


def test_load_event_stream_skips_unresolved_markets(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    _make_corpus_db(
        db,
        trades=[
            ("0xa", "0xM1", "YES", "BUY", 0.5, 1_000.0, 100),
            ("0xa", "0xM2", "YES", "BUY", 0.5, 1_000.0, 110),
        ],
        resolutions=[("0xM1", 1, 200)],  # only M1 has a resolution
    )
    events = list(load_event_stream(db, watchlist={"0xa"}, platform="polymarket"))
    cids = {
        (e.trade.condition_id if isinstance(e, TradeEvent) else e.resolution.condition_id)
        for e in events
    }
    assert cids == {"0xM1"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 6 new ImportErrors.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
from collections.abc import Iterator
from pathlib import Path

import duckdb


def load_watchlist(daemon_db: Path) -> set[str]:
    """Return the set of active watchlist addresses (lowercase)."""
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{daemon_db}' AS daemon (TYPE sqlite)")
    rows = con.execute(
        "SELECT LOWER(address) FROM daemon.wallet_watchlist WHERE active = 1"
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


def load_event_stream(
    corpus_db: Path,
    *,
    watchlist: set[str],
    platform: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Iterator[TradeEvent | ResolutionEvent]:
    """Yield TradeEvent + ResolutionEvent in `ts` ascending order.

    A trade is included only when:
      - direction is BUY,
      - wallet_address is in `watchlist`,
      - the trade's market has a `market_resolutions` row.

    A resolution is included only when at least one of its market's
    BUY trades made it past the filter above (LEFT SEMI JOIN).
    """
    if not watchlist:
        return
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{corpus_db}' AS c (TYPE sqlite)")
    # Materialize watchlist into a temp table for the IN-subquery.
    con.execute("CREATE TEMP TABLE wl(addr TEXT)")
    con.executemany("INSERT INTO wl VALUES (?)", [(a,) for a in watchlist])
    ts_filter = ""
    params: list[int] = []
    if start_ts is not None:
        ts_filter += " AND t.ts >= ?"
        params.append(start_ts)
    if end_ts is not None:
        ts_filter += " AND t.ts < ?"
        params.append(end_ts)
    query = f"""
        WITH eligible_trades AS (
          SELECT
            t.wallet_address AS wallet,
            t.condition_id AS condition_id,
            t.outcome_side AS outcome_side,
            t.price AS price,
            t.notional_usd AS notional_usd,
            t.ts AS ts
          FROM c.corpus_trades t
          JOIN c.market_resolutions r
            ON r.platform = t.platform AND r.condition_id = t.condition_id
          WHERE t.platform = ?
            AND t.bs = 'BUY'
            AND LOWER(t.wallet_address) IN (SELECT addr FROM wl)
            {ts_filter}
        ),
        eligible_resolutions AS (
          SELECT
            r.condition_id AS condition_id,
            r.outcome_yes_won AS outcome_yes_won,
            r.resolved_at AS ts
          FROM c.market_resolutions r
          WHERE r.platform = ?
            AND r.condition_id IN (SELECT DISTINCT condition_id FROM eligible_trades)
        )
        SELECT
          'trade' AS kind, ts, wallet, condition_id, outcome_side,
          price, notional_usd, NULL AS outcome_yes_won
        FROM eligible_trades
        UNION ALL
        SELECT
          'resolution' AS kind, ts, NULL, condition_id, NULL,
          NULL, NULL, outcome_yes_won
        FROM eligible_resolutions
        ORDER BY ts ASC
    """
    rows = con.execute(query, [platform, *params, platform]).fetchall()
    con.close()
    for kind, ts, wallet, cid, side, price, notional, yes_won in rows:
        if kind == "trade":
            yield TradeEvent(
                kind="trade",
                ts=int(ts),
                trade=Trade(
                    wallet=str(wallet),
                    condition_id=str(cid),
                    outcome_side=str(side),
                    price=float(price),
                    notional_usd=float(notional),
                    ts=int(ts),
                ),
            )
        else:
            yield ResolutionEvent(
                kind="resolution",
                ts=int(ts),
                resolution=Resolution(
                    condition_id=str(cid),
                    winning_side="YES" if int(yes_won) == 1 else "NO",
                    resolved_at=int(ts),
                ),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 12 passed (6 from prior tasks + 6 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): DuckDB loaders for watchlist + event stream"
```

---

## Task 8: Report rendering

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_backtest_simulator.py`:

```python
from scripts.backtest_copy_sizing import render_report


def test_render_report_includes_all_scheme_rows() -> None:
    schemes = [
        EqualWeight(position_fraction=0.01),
    ]
    sim = Simulator(schemes=schemes, bankroll=10_000.0)
    sim.on_trade(
        Trade(
            wallet="0xa",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=100,
        )
    )
    sim.on_resolution(
        Resolution(condition_id="0xM1", winning_side="YES", resolved_at=200)
    )
    report = render_report(sim, schemes=schemes, bankroll=10_000.0)
    assert "equal_weight" in report
    assert "PnL" in report
    assert "Win rate" in report
    assert "100.00%" in report or "100.0%" in report  # one trade, one win
    # cost=100, proceeds=200, pnl=100. ROI = 100 / 100 = 100%.
    assert "+$100" in report or "$100.00" in report


def test_render_report_includes_quarterly_grid_and_unresolved_count() -> None:
    schemes = [EqualWeight(position_fraction=0.01)]
    sim = Simulator(schemes=schemes, bankroll=10_000.0)
    # One resolved, one unresolved.
    sim.on_trade(
        Trade(
            wallet="0xa",
            condition_id="0xM1",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=1_700_000_000,
        )
    )
    sim.on_trade(
        Trade(
            wallet="0xa",
            condition_id="0xM2",
            outcome_side="YES",
            price=0.5,
            notional_usd=1_000.0,
            ts=1_700_000_100,
        )
    )
    sim.on_resolution(
        Resolution(
            condition_id="0xM1", winning_side="YES", resolved_at=1_700_000_500
        )
    )
    report = render_report(sim, schemes=schemes, bankroll=10_000.0)
    assert "Unresolved" in report
    assert "Quarterly" in report or "quarter" in report.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 2 ImportErrors for `render_report`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
import datetime as dt
import math


def _running_max_drawdown(nav_series: list[tuple[int, float]]) -> tuple[float, int]:
    """Return (max_drawdown, drawdown_duration_seconds)."""
    if not nav_series:
        return 0.0, 0
    peak = nav_series[0][1]
    peak_ts = nav_series[0][0]
    max_dd = 0.0
    max_dd_duration = 0
    for ts, nav in nav_series:
        if nav > peak:
            peak = nav
            peak_ts = ts
        dd = nav - peak  # ≤ 0
        if dd < max_dd:
            max_dd = dd
            max_dd_duration = ts - peak_ts
    return max_dd, max_dd_duration


def _sharpe_like(nav_series: list[tuple[int, float]]) -> float:
    """Mean(daily PnL) / stdev(daily PnL) * sqrt(365). 0.0 on < 2 days."""
    if len(nav_series) < 2:
        return 0.0
    by_day: dict[int, float] = {}
    prev_nav = 0.0
    for ts, nav in nav_series:
        day = ts // 86_400
        by_day[day] = nav - prev_nav + by_day.get(day, 0.0)
        prev_nav = nav
    daily = list(by_day.values())
    if len(daily) < 2:
        return 0.0
    mean = sum(daily) / len(daily)
    var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var) * math.sqrt(365)


def _quarter_label(ts: int) -> str:
    d = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def render_report(
    sim: Simulator,
    *,
    schemes: list[SizingScheme],
    bankroll: float,
) -> str:
    """Render the backtest as a multi-section markdown report."""
    out: list[str] = []
    out.append("# Backtest: copy-trading sizing comparison\n")
    out.append(f"Bankroll: ${bankroll:,.2f}\n")

    # Headline table.
    out.append("\n## Headline\n")
    out.append(
        "| Scheme | Trades | Cost | Proceeds | PnL | ROI | Win rate"
        " | Avg cost/trade | Unresolved |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for scheme in schemes:
        state = sim.state_for(scheme)
        n = len(state.resolved_trades)
        cost = sum(r.open_pos.cost for r in state.resolved_trades)
        proceeds = sum(r.proceeds for r in state.resolved_trades)
        pnl = proceeds - cost
        roi = pnl / cost if cost else 0.0
        wins = sum(1 for r in state.resolved_trades if r.payout > 0)
        win_rate = wins / n if n else 0.0
        avg_cost = cost / n if n else 0.0
        unresolved = len(state.open_positions)
        out.append(
            f"| {scheme.name} | {n:,} | ${cost:,.2f} | ${proceeds:,.2f}"
            f" | {'+' if pnl >= 0 else ''}${pnl:,.2f}"
            f" | {roi * 100:+.2f}% | {win_rate * 100:.2f}%"
            f" | ${avg_cost:,.2f} | {unresolved} |\n"
        )

    # Risk metrics.
    out.append("\n## Risk\n")
    out.append(
        "| Scheme | Max DD | DD duration (days) | Sharpe-like"
        " | Worst trade | Best trade |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for scheme in schemes:
        state = sim.state_for(scheme)
        max_dd, dd_dur_sec = _running_max_drawdown(state.nav_series)
        sharpe = _sharpe_like(state.nav_series)
        pnls = [r.pnl for r in state.resolved_trades]
        worst = min(pnls) if pnls else 0.0
        best = max(pnls) if pnls else 0.0
        out.append(
            f"| {scheme.name} | ${max_dd:,.2f}"
            f" | {dd_dur_sec / 86_400:.1f}"
            f" | {sharpe:.2f} | ${worst:,.2f} | ${best:,.2f} |\n"
        )

    # Quarterly grid.
    out.append("\n## Quarterly PnL\n")
    all_quarters: set[str] = set()
    quarter_pnl: dict[str, dict[str, float]] = {s.name: {} for s in schemes}
    for scheme in schemes:
        state = sim.state_for(scheme)
        for r in state.resolved_trades:
            q = _quarter_label(r.resolved_at)
            all_quarters.add(q)
            quarter_pnl[scheme.name][q] = (
                quarter_pnl[scheme.name].get(q, 0.0) + r.pnl
            )
    sorted_q = sorted(all_quarters)
    if sorted_q:
        out.append("| Scheme | " + " | ".join(sorted_q) + " |\n")
        out.append("|---|" + "|".join("---:" for _ in sorted_q) + "|\n")
        for scheme in schemes:
            row = "| " + scheme.name + " | "
            row += " | ".join(
                f"${quarter_pnl[scheme.name].get(q, 0.0):+,.0f}"
                for q in sorted_q
            )
            row += " |\n"
            out.append(row)
    else:
        out.append("(no resolved trades)\n")

    # Top contributors (best PnL scheme only).
    out.append("\n## Top contributors (best-PnL scheme)\n")
    best_scheme = max(
        schemes,
        key=lambda s: sum(r.pnl for r in sim.state_for(s).resolved_trades),
        default=None,
    )
    if best_scheme is not None:
        state = sim.state_for(best_scheme)
        by_wallet: dict[str, tuple[int, float, float]] = {}
        for r in state.resolved_trades:
            n, c, p = by_wallet.get(r.open_pos.wallet, (0, 0.0, 0.0))
            by_wallet[r.open_pos.wallet] = (n + 1, c + r.open_pos.cost, p + r.pnl)
        ranked = sorted(by_wallet.items(), key=lambda kv: kv[1][2], reverse=True)[:10]
        out.append(f"Scheme: **{best_scheme.name}**\n\n")
        out.append("| Wallet | Copies | Total cost | PnL |\n")
        out.append("|---|---:|---:|---:|\n")
        for wallet, (n, c, p) in ranked:
            out.append(f"| `{wallet}` | {n} | ${c:,.2f} | ${p:+,.2f} |\n")
    return "".join(out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): render markdown report (headline, risk, quarterly, contributors)"
```

---

## Task 9: CLI parser + main()

**Files:**
- Modify: `scripts/backtest_copy_sizing.py`
- Modify: `tests/scripts/test_backtest_simulator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/scripts/test_backtest_simulator.py`:

```python
from scripts.backtest_copy_sizing import build_parser


def test_build_parser_has_expected_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.starting_bankroll_usd == 10_000.0
    assert args.position_fraction == 0.01
    assert args.min_multiplier == 0.10
    assert args.scale_factor == 0.01
    assert args.max_cost_per_trade == 1_000.0
    assert args.edge_scale == 5.0
    assert args.max_multiplier == 3.0
    assert args.min_trades_for_edge == 10
    assert args.platform == "polymarket"
    assert args.start_ts is None
    assert args.end_ts is None


def test_build_parser_accepts_csv_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["--csv", "/tmp/x.csv"])
    assert args.csv == "/tmp/x.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 2 new ImportErrors for `build_parser`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/backtest_copy_sizing.py`:

```python
import argparse
import csv as _csv
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Backtest copy-trading sizing schemes against the historical"
            " trade stream of the daemon's current watchlist."
        )
    )
    p.add_argument("--db", default="data/corpus.sqlite3")
    p.add_argument("--watchlist-db", default="data/pscanner.sqlite3")
    p.add_argument("--starting-bankroll-usd", type=float, default=10_000.0)
    p.add_argument("--position-fraction", type=float, default=0.01)
    p.add_argument("--min-multiplier", type=float, default=0.10)
    p.add_argument("--scale-factor", type=float, default=0.01)
    p.add_argument("--max-cost-per-trade", type=float, default=1_000.0)
    p.add_argument("--edge-scale", type=float, default=5.0)
    p.add_argument("--max-multiplier", type=float, default=3.0)
    p.add_argument("--min-trades-for-edge", type=int, default=10)
    p.add_argument("--platform", default="polymarket")
    p.add_argument("--start-ts", type=int, default=None)
    p.add_argument("--end-ts", type=int, default=None)
    p.add_argument(
        "--csv", default=None, help="Optional path to dump per-trade per-scheme rows."
    )
    return p


def _write_csv(sim: Simulator, schemes: list[SizingScheme], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(
            ["scheme", "wallet", "condition_id", "outcome_side", "price",
             "shares", "cost", "payout", "proceeds", "pnl",
             "trade_ts", "resolved_at"]
        )
        for scheme in schemes:
            state = sim.state_for(scheme)
            for r in state.resolved_trades:
                pos = r.open_pos
                w.writerow([
                    scheme.name, pos.wallet, pos.condition_id, pos.outcome_side,
                    f"{pos.price}", f"{pos.shares}", f"{pos.cost}",
                    f"{r.payout}", f"{r.proceeds}", f"{r.pnl}",
                    pos.ts, r.resolved_at,
                ])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    watchlist = load_watchlist(Path(args.watchlist_db))
    if not watchlist:
        print("Watchlist is empty; nothing to backtest.", file=sys.stderr)
        return 1
    schemes: list[SizingScheme] = [
        EqualWeight(position_fraction=args.position_fraction),
        ConcentrationCapped(
            position_fraction=args.position_fraction,
            min_multiplier=args.min_multiplier,
            watchlist_size=len(watchlist),
        ),
        FollowSeedSize(
            scale_factor=args.scale_factor,
            max_cost_per_trade=args.max_cost_per_trade,
        ),
        EdgeWeightedCausal(
            position_fraction=args.position_fraction,
            edge_scale=args.edge_scale,
            min_multiplier=max(args.min_multiplier, 0.25),
            max_multiplier=args.max_multiplier,
            min_trades_for_edge=args.min_trades_for_edge,
        ),
    ]
    sim = Simulator(schemes=schemes, bankroll=args.starting_bankroll_usd)
    events = load_event_stream(
        Path(args.db),
        watchlist=watchlist,
        platform=args.platform,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )
    for event in events:
        if isinstance(event, TradeEvent):
            sim.on_trade(event.trade)
        else:
            sim.on_resolution(event.resolution)
    print(render_report(sim, schemes=schemes, bankroll=args.starting_bankroll_usd))
    if args.csv:
        _write_csv(sim, schemes, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/scripts/test_backtest_simulator.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Verify lint + format + types**

```bash
uv run ruff check scripts/backtest_copy_sizing.py tests/scripts/
uv run ruff format --check scripts/backtest_copy_sizing.py tests/scripts/
uv run ty check
```

Expected: ruff clean. ty diagnostic count must equal the pre-change baseline — if it grew, fix the new diagnostics before committing.

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest_copy_sizing.py tests/scripts/test_backtest_simulator.py
git commit -m "feat(backtest): wire CLI parser, main(), optional CSV dump"
```

---

## Task 10: Slow corpus smoke test (PAUSE BEFORE RUNNING)

**Files:**
- Create: `tests/slow/test_backtest_corpus_smoke.py`

- [ ] **Step 1: Write the test**

```python
# tests/slow/test_backtest_corpus_smoke.py
"""Opt-in smoke test for the backtest script against the production corpus.

Skipped by default; run with `uv run pytest -m slow`.

PAUSE BEFORE RUNNING — the implementing agent must get operator approval
before invoking this test or the script against the real corpus DB.
See the implementation plan and the spec at
`docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.backtest_copy_sizing import (
    ConcentrationCapped,
    EdgeWeightedCausal,
    EqualWeight,
    FollowSeedSize,
    Simulator,
    TradeEvent,
    load_event_stream,
    load_watchlist,
    render_report,
)

pytestmark = pytest.mark.slow

_CORPUS_DB = Path("data/corpus.sqlite3")
_DAEMON_DB = Path("data/pscanner.sqlite3")


def _skip_if_missing() -> None:
    if not _CORPUS_DB.exists():
        pytest.skip(f"{_CORPUS_DB} not present; smoke test requires production data")
    if not _DAEMON_DB.exists():
        pytest.skip(f"{_DAEMON_DB} not present; smoke test requires daemon DB")


def test_backtest_produces_finite_report_against_corpus() -> None:
    _skip_if_missing()
    watchlist = load_watchlist(_DAEMON_DB)
    assert watchlist, "watchlist is empty; nothing to backtest"
    # Cap watchlist for smoke test wall-time. Take the first 10 deterministic.
    limited = set(sorted(watchlist)[:10])
    bankroll = 10_000.0
    schemes = [
        EqualWeight(position_fraction=0.01),
        ConcentrationCapped(
            position_fraction=0.01, min_multiplier=0.10, watchlist_size=len(limited)
        ),
        FollowSeedSize(scale_factor=0.01, max_cost_per_trade=1_000.0),
        EdgeWeightedCausal(
            position_fraction=0.01,
            edge_scale=5.0,
            min_multiplier=0.25,
            max_multiplier=3.0,
            min_trades_for_edge=10,
        ),
    ]
    sim = Simulator(schemes=schemes, bankroll=bankroll)
    n_trades = 0
    n_resolutions = 0
    for event in load_event_stream(
        _CORPUS_DB, watchlist=limited, platform="polymarket"
    ):
        if isinstance(event, TradeEvent):
            sim.on_trade(event.trade)
            n_trades += 1
        else:
            sim.on_resolution(event.resolution)
            n_resolutions += 1
    assert n_trades > 0, "no eligible trades found for the limited watchlist"
    report = render_report(sim, schemes=schemes, bankroll=bankroll)
    for scheme in schemes:
        assert scheme.name in report
        state = sim.state_for(scheme)
        for r in state.resolved_trades:
            assert math.isfinite(r.pnl), f"non-finite PnL in {scheme.name}"
```

- [ ] **Step 2: DO NOT RUN THIS TEST YET**

This is the operator-approval checkpoint. The implementing agent MUST:

1. Stop after writing the test file.
2. Tell the operator: "Slow corpus smoke test written but not yet executed. The spec/plan requires operator approval before this touches production data. Inspect `tests/slow/test_backtest_corpus_smoke.py` and the script's behaviour, then approve or request changes."
3. Wait for explicit operator approval.
4. Only run `uv run pytest -m slow tests/slow/test_backtest_corpus_smoke.py -v` after approval.

- [ ] **Step 3: Commit (test added, not yet run)**

```bash
git add tests/slow/test_backtest_corpus_smoke.py
git commit -m "test(backtest): add opt-in corpus smoke test (gated by operator approval)"
```

- [ ] **Step 4 (after approval): Run the smoke test**

```bash
uv run pytest -m slow tests/slow/test_backtest_corpus_smoke.py -v
```

Expected: 1 passed (or `pytest.skip` if `data/corpus.sqlite3` / `data/pscanner.sqlite3` are absent in this checkout).

---

## Task 11: Final verify gate

**Files:** none modified — verification only.

- [ ] **Step 1: Run lint, format, types, full pytest**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected:
- `ruff check`: All checks passed.
- `ruff format --check`: 0 files would be reformatted.
- `ty check`: diagnostic count equals the pre-change baseline. If it grew, fix before declaring complete.
- `pytest -q`: all tests pass, no warnings (the project has `filterwarnings = ["error"]`).

- [ ] **Step 2: Confirm completion**

Report to the operator: PR-ready (or, if running off `main` without a branch, suggest opening a PR).
