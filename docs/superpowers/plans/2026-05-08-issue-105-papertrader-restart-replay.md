# Issue #105 — PaperTrader Restart Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PaperTrader` book alerts that fired before it was wired (mid-day restart, "enable detector first then evaluator" workflow). Bounded by a configurable lookback window.

**Architecture:** Option B from the issue. On scheduler boot, `PaperTrader` runs a one-shot replay pass: fetch alerts from the last `replay_lookback_seconds` whose `alert_key` doesn't appear in `paper_trades.triggering_alert_key`, then drive each through the existing `evaluate()` pipeline. Idempotency is already guaranteed by the existing unique constraint on `paper_trades` and the `IntegrityError` swallow in `_insert_entry`. Default `replay_lookback_seconds = 0` (disabled — opt-in).

**Tech Stack:** Python 3.13, sqlite3, structlog, pytest-asyncio. No new deps.

---

## File Structure

- Modify: `src/pscanner/config.py:448-465` — add `replay_lookback_seconds: int = 0` to `PaperTradingConfig`.
- Modify: `src/pscanner/store/repo.py:634-700` (AlertsRepo class) — add `fetch_unbooked_since(min_created_at: int) -> list[Alert]`.
- Modify: `src/pscanner/strategies/paper_trader.py:53-118` — add `async def replay_unbooked(self) -> int` method.
- Modify: `src/pscanner/scheduler.py:540-553` (Scanner.run / preflight area) — call replay before entering the supervisor loop.
- Test: `tests/store/test_alerts_repo_unbooked.py` (new) — unit test for the query.
- Test: `tests/strategies/test_paper_trader_replay.py` (new) — integration test for the replay pass.
- Modify: `CLAUDE.md` — append a one-line note to the paper-trading bullet area.

Tasks 1, 2, 3, 4 are sequential because each later task uses APIs introduced earlier.

---

## Task 1: Add `replay_lookback_seconds` to PaperTradingConfig

**Files:**
- Modify: `src/pscanner/config.py:448-465`
- Test: `tests/test_config.py` (likely exists; if not, skip — pydantic catches the typing)

A two-line config addition. No TDD needed — config is data. Lands in its own commit so later tasks can rebase against it.

- [ ] **Step 1: Add the field**

Edit `src/pscanner/config.py`. Replace the `PaperTradingConfig` body (lines 460-464):

```python
    enabled: bool = False
    starting_bankroll_usd: float = 1000.0
    min_position_cost_usd: float = 0.50
    resolver_scan_interval_seconds: float = 300.0
    replay_lookback_seconds: int = 0
    """On boot, replay alerts emitted in the last N seconds that don't yet have
    a paper_trades entry through the evaluator pipeline. ``0`` disables the
    replay (default). Set to e.g. ``900`` (15 minutes) to recover from a
    daemon restart without losing in-flight alerts. See issue #105.
    """
    evaluators: EvaluatorsConfig = Field(default_factory=EvaluatorsConfig)
```

- [ ] **Step 2: Verify config loads**

Run: `uv run python -c "from pscanner.config import PaperTradingConfig; c = PaperTradingConfig(); print(c.replay_lookback_seconds)"`

Expected: `0`

- [ ] **Step 3: Run existing config tests**

Run: `uv run pytest tests/test_config.py -v` (skip if file does not exist)

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/config.py
git commit -m "feat(config): add paper_trading.replay_lookback_seconds (#105)"
```

---

## Task 2: Add `AlertsRepo.fetch_unbooked_since`

**Files:**
- Modify: `src/pscanner/store/repo.py` (inside `AlertsRepo`, after `fetch_recent` around line 700)
- Test: `tests/store/test_alerts_repo_unbooked.py` (create)

The query LEFT JOINs `alerts` to `paper_trades` on `alert_key = triggering_alert_key` (entry rows only) and returns alerts where the JOIN produced NULL. That naturally excludes both fully-booked alerts AND alerts that were rejected by the evaluator's quality gates last time (which is desired — replay re-runs the pipeline so an alert previously rejected with `bankroll_exhausted=False` only re-books if quality still passes).

- [ ] **Step 1: Write the failing unit test**

Create `tests/store/test_alerts_repo_unbooked.py`:

```python
"""Unit tests for AlertsRepo.fetch_unbooked_since (#105)."""

from __future__ import annotations

import sqlite3
import time
from typing import cast

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


def _make_alert(*, key: str, ts: int, detector: str = "gate_buy") -> Alert:
    return Alert(
        detector=cast(DetectorName, detector),
        alert_key=key,
        severity=cast(Severity, "med"),
        title=f"t-{key}",
        body={"foo": "bar"},
        created_at=ts,
    )


def test_fetch_unbooked_since_returns_alerts_without_entry(conn: sqlite3.Connection) -> None:
    """An alert with no paper_trades entry inside the window is returned."""
    alerts = AlertsRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="recent-1", ts=now - 60))
    alerts.insert_if_new(_make_alert(key="recent-2", ts=now - 120))

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert sorted(keys) == ["recent-1", "recent-2"]


def test_fetch_unbooked_since_excludes_alerts_with_entry(conn: sqlite3.Connection) -> None:
    """An alert that already has a paper_trades entry row is excluded."""
    alerts = AlertsRepo(conn)
    paper = PaperTradesRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="booked", ts=now - 60))
    alerts.insert_if_new(_make_alert(key="unbooked", ts=now - 60))
    paper.insert_entry(
        triggering_alert_key="booked",
        triggering_alert_detector="gate_buy",
        rule_variant=None,
        source_wallet=None,
        condition_id="0xc",
        asset_id="0xa",
        outcome="YES",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=1000.0,
        ts=now - 50,
    )

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert keys == ["unbooked"]


def test_fetch_unbooked_since_excludes_alerts_outside_window(
    conn: sqlite3.Connection,
) -> None:
    """An alert older than the cutoff is excluded even if unbooked."""
    alerts = AlertsRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="too-old", ts=now - 3600))
    alerts.insert_if_new(_make_alert(key="in-window", ts=now - 60))

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert keys == ["in-window"]


def test_fetch_unbooked_since_returns_oldest_first(conn: sqlite3.Connection) -> None:
    """Replay processes alerts in chronological order, not newest-first."""
    alerts = AlertsRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="newer", ts=now - 60))
    alerts.insert_if_new(_make_alert(key="older", ts=now - 200))

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert keys == ["older", "newer"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/store/test_alerts_repo_unbooked.py -v`

Expected: 4 FAILs — `AttributeError: 'AlertsRepo' object has no attribute 'fetch_unbooked_since'`.

- [ ] **Step 3: Implement the method**

Edit `src/pscanner/store/repo.py`. Insert after `AlertsRepo.fetch_recent` (around line 700, before `_row_to_alert`):

```python
    def fetch_unbooked_since(self, *, min_created_at: int) -> list[Alert]:
        """Return alerts with no ``paper_trades`` entry, ascending by created_at.

        LEFT JOIN against ``paper_trades`` on ``alert_key = triggering_alert_key``
        (entry rows only); rows where the JOIN produced NULL are unbooked.
        Used by :meth:`PaperTrader.replay_unbooked` (issue #105) to recover
        alerts that fired before the evaluator subscribed.

        Args:
            min_created_at: Lower bound on ``alerts.created_at`` (Unix seconds).
                Alerts older than this are excluded; the caller derives this
                from ``replay_lookback_seconds``.

        Returns:
            Alerts in ascending ``created_at`` order so replay processes
            oldest first, matching live emission ordering.
        """
        rows = self._conn.execute(
            """
            SELECT a.alert_key, a.detector, a.severity, a.title, a.body_json, a.created_at
              FROM alerts a
              LEFT JOIN paper_trades p
                     ON p.triggering_alert_key = a.alert_key
                    AND p.trade_kind = 'entry'
             WHERE a.created_at >= ?
               AND p.trade_id IS NULL
             ORDER BY a.created_at ASC
            """,
            (min_created_at,),
        ).fetchall()
        return [_row_to_alert(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/store/test_alerts_repo_unbooked.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/pscanner/store/repo.py tests/store/test_alerts_repo_unbooked.py && uv run ruff format --check src/pscanner/store/repo.py tests/store/test_alerts_repo_unbooked.py && uv run ty check src/pscanner/store/repo.py tests/store/test_alerts_repo_unbooked.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/store/repo.py tests/store/test_alerts_repo_unbooked.py
git commit -m "feat(store): AlertsRepo.fetch_unbooked_since for restart replay (#105)"
```

---

## Task 3: Add `PaperTrader.replay_unbooked`

**Files:**
- Modify: `src/pscanner/strategies/paper_trader.py:53-118`
- Test: `tests/strategies/test_paper_trader_replay.py` (create)

Adds an async one-shot method that pulls unbooked alerts from the lookback window and drives each through `evaluate()`. Returns the count for log/metrics. The method is a public coroutine so the scheduler can `await` it before entering the supervisor loop.

`PaperTrader.__init__` doesn't currently hold an `AlertsRepo`. Pass one in at construction.

- [ ] **Step 1: Write the failing integration test**

Create `tests/strategies/test_paper_trader_replay.py`:

```python
"""Integration tests for PaperTrader.replay_unbooked (#105)."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.config import (
    EvaluatorsConfig,
    GateModelEvaluatorConfig,
    PaperTradingConfig,
)
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
)
from pscanner.strategies.evaluators.gate_model import GateModelEvaluator
from pscanner.strategies.paper_trader import PaperTrader


def _make_gate_alert(*, key: str, ts: int, condition_id: str = "0xc1") -> Alert:
    return Alert(
        detector=cast(DetectorName, "gate_buy"),
        alert_key=key,
        severity=cast(Severity, "med"),
        title=f"gate_buy on {condition_id}",
        body={
            "wallet": "0xabc",
            "condition_id": condition_id,
            "side": "YES",
            "implied_prob_at_buy": 0.5,
            "pred": 0.8,
            "edge": 0.3,
            "top_category": "esports",
            "model_version": "v1",
            "trade_ts": ts,
            "bet_size_usd": 100.0,
        },
        created_at=ts,
    )


def _seed_market(cache: MarketCacheRepo, condition_id: str = "0xc1") -> None:
    cache.upsert(
        CachedMarket(
            market_id=cast(Any, "m1"),
            event_id=cast(Any, "e1"),
            title="t",
            liquidity_usd=1000.0,
            volume_usd=1000.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=cast(Any, condition_id),
            event_slug=None,
            outcomes=["YES", "NO"],
            asset_ids=[cast(Any, "0xa1"), cast(Any, "0xa2")],
        )
    )


@pytest.mark.asyncio
async def test_replay_books_unbooked_alerts_in_window(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An alert in the lookback window with no paper_trades row is booked."""
    conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        alerts_repo = AlertsRepo(conn)
        paper = PaperTradesRepo(conn)
        cache = MarketCacheRepo(conn)
        ticks = MarketTicksRepo(conn)
        _seed_market(cache)

        now = int(time.time())
        alerts_repo.insert_if_new(_make_gate_alert(key="A", ts=now - 60))

        cfg = PaperTradingConfig(
            enabled=True,
            starting_bankroll_usd=1000.0,
            replay_lookback_seconds=300,
            evaluators=EvaluatorsConfig(
                gate_model=GateModelEvaluatorConfig(
                    enabled=True,
                    position_fraction=0.005,
                    min_pred=0.7,
                    min_edge_pct=0.01,
                ),
            ),
        )
        evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
        trader = PaperTrader(
            config=cfg,
            evaluators=[evaluator],
            market_cache=cache,
            paper_trades=paper,
            market_ticks=ticks,
            data_client=cast(Any, MagicMock()),
            gamma_client=cast(Any, MagicMock()),
            alerts_repo=alerts_repo,
        )

        booked_count = await trader.replay_unbooked()
    finally:
        conn.close()

    assert booked_count == 1
    rows = conn.execute(
        "SELECT triggering_alert_key FROM paper_trades WHERE trade_kind='entry'"
    ).fetchall()
    assert [r[0] for r in rows] == ["A"]


@pytest.mark.asyncio
async def test_replay_disabled_when_lookback_is_zero(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """replay_lookback_seconds=0 means no replay query and no books."""
    conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        alerts_repo = AlertsRepo(conn)
        paper = PaperTradesRepo(conn)
        cache = MarketCacheRepo(conn)
        ticks = MarketTicksRepo(conn)
        _seed_market(cache)

        now = int(time.time())
        alerts_repo.insert_if_new(_make_gate_alert(key="A", ts=now - 60))

        cfg = PaperTradingConfig(
            enabled=True,
            replay_lookback_seconds=0,
            evaluators=EvaluatorsConfig(
                gate_model=GateModelEvaluatorConfig(enabled=True),
            ),
        )
        evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
        trader = PaperTrader(
            config=cfg,
            evaluators=[evaluator],
            market_cache=cache,
            paper_trades=paper,
            market_ticks=ticks,
            data_client=cast(Any, MagicMock()),
            gamma_client=cast(Any, MagicMock()),
            alerts_repo=alerts_repo,
        )

        booked_count = await trader.replay_unbooked()
    finally:
        conn.close()

    assert booked_count == 0


@pytest.mark.asyncio
async def test_replay_skips_already_booked_alerts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Alerts with a paper_trades entry are excluded from replay."""
    conn = init_db(tmp_path / "daemon.sqlite3")
    try:
        alerts_repo = AlertsRepo(conn)
        paper = PaperTradesRepo(conn)
        cache = MarketCacheRepo(conn)
        ticks = MarketTicksRepo(conn)
        _seed_market(cache)

        now = int(time.time())
        alerts_repo.insert_if_new(_make_gate_alert(key="already", ts=now - 60))
        paper.insert_entry(
            triggering_alert_key="already",
            triggering_alert_detector="gate_buy",
            rule_variant=None,
            source_wallet="0xabc",
            condition_id=cast(Any, "0xc1"),
            asset_id=cast(Any, "0xa1"),
            outcome="YES",
            shares=10.0,
            fill_price=0.5,
            cost_usd=5.0,
            nav_after_usd=1000.0,
            ts=now - 50,
        )

        cfg = PaperTradingConfig(
            enabled=True,
            replay_lookback_seconds=300,
            evaluators=EvaluatorsConfig(
                gate_model=GateModelEvaluatorConfig(enabled=True),
            ),
        )
        evaluator = GateModelEvaluator(config=cfg.evaluators.gate_model)
        trader = PaperTrader(
            config=cfg,
            evaluators=[evaluator],
            market_cache=cache,
            paper_trades=paper,
            market_ticks=ticks,
            data_client=cast(Any, MagicMock()),
            gamma_client=cast(Any, MagicMock()),
            alerts_repo=alerts_repo,
        )

        booked_count = await trader.replay_unbooked()
    finally:
        conn.close()

    assert booked_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_paper_trader_replay.py -v`

Expected: 3 FAILs — `TypeError: PaperTrader.__init__() got an unexpected keyword argument 'alerts_repo'`.

- [ ] **Step 3: Implement `replay_unbooked` and ctor parameter**

Edit `src/pscanner/strategies/paper_trader.py`. Add `AlertsRepo` to the imports near line 26:

```python
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
)
```

Replace `__init__` (lines 58-94):

```python
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
        alerts_repo: AlertsRepo,
    ) -> None:
        """Bind dependencies. Subscribers must call :meth:`subscribe` separately.

        Args:
            config: Bankroll + min-cost thresholds (per-source tunables live
                under each evaluator's own config).
            evaluators: Per-detector :class:`SignalEvaluator` instances. The
                first one whose ``accepts`` returns ``True`` for an alert
                runs the parse → quality → size pipeline.
            market_cache: Read-side cache mapping ``(condition_id, outcome)``
                to ``asset_id``.
            paper_trades: Repo that owns the entry/exit ledger.
            market_ticks: Tick history repo for the entry-price lookup.
            data_client: Polymarket data-API client. Used by the cache-miss
                fallback to discover an unknown market's slug from one of its
                trades.
            gamma_client: Polymarket gamma-API client. Used by the cache-miss
                fallback to fetch the full ``Market`` once a slug is known.
            alerts_repo: Read-side access for restart replay
                (:meth:`replay_unbooked`).
        """
        self._config = config
        self._evaluators = evaluators
        self._market_cache = market_cache
        self._paper_trades = paper_trades
        self._market_ticks = market_ticks
        self._data_client = data_client
        self._gamma_client = gamma_client
        self._alerts_repo = alerts_repo
        self._pending_tasks: set[asyncio.Task[None]] = set()
```

Insert a new method after `evaluate` (before `_run_pipeline`, around line 147):

```python
    async def replay_unbooked(self) -> int:
        """Drive any unbooked alert from the lookback window through the pipeline.

        Reads ``self._config.replay_lookback_seconds``; ``0`` (default) is a
        no-op. Returns the number of alerts pushed through ``evaluate()``.
        Each alert's actual book/skip outcome is decided by the existing
        evaluator chain — replay does not bypass quality gates.

        Idempotent: alerts that already have a ``paper_trades`` entry row are
        excluded by the SQL JOIN, and the existing ``IntegrityError`` swallow
        in :meth:`_insert_entry` covers race conditions where the same alert
        gets re-emitted by the live path mid-replay.
        """
        lookback = self._config.replay_lookback_seconds
        if lookback <= 0:
            return 0
        cutoff = int(time.time()) - lookback
        unbooked = self._alerts_repo.fetch_unbooked_since(min_created_at=cutoff)
        for alert in unbooked:
            await self.evaluate(alert)
        if unbooked:
            _LOG.info(
                "paper_trader.replay_complete",
                count=len(unbooked),
                lookback_seconds=lookback,
            )
        return len(unbooked)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/test_paper_trader_replay.py -v`

Expected: 3 PASS.

- [ ] **Step 5: Run the existing PaperTrader tests for regression**

Run: `uv run pytest tests/strategies/ -v`

Expected: all pass. The new `alerts_repo` ctor kwarg may have broken existing constructions — fix by passing `alerts_repo=AlertsRepo(conn)` in those tests' fixtures.

- [ ] **Step 6: Lint + types**

Run: `uv run ruff check src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader_replay.py && uv run ruff format --check src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader_replay.py && uv run ty check src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader_replay.py`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/strategies/paper_trader.py tests/strategies/test_paper_trader_replay.py
git commit -m "feat(paper_trader): replay_unbooked() for restart recovery (#105)"
```

---

## Task 4: Wire replay into Scanner.run

**Files:**
- Modify: `src/pscanner/scheduler.py:540-553` (Scanner.run start) + the `PaperTrader` construction site at `_maybe_attach_paper_trading` (line 416-435)
- Test: extend an existing scheduler test

The scheduler now passes `alerts_repo=self._alerts_repo` when constructing `PaperTrader`, and calls `await trader.replay_unbooked()` between `preflight()` and the supervisor loop.

- [ ] **Step 1: Update `_maybe_attach_paper_trading` to thread `alerts_repo`**

Edit `src/pscanner/scheduler.py`. Replace the `PaperTrader(...)` construction (lines 421-429):

```python
        detectors["paper_trader"] = PaperTrader(
            config=self._config.paper_trading,
            evaluators=self._build_paper_evaluators(),
            market_cache=self._market_cache_repo,
            paper_trades=paper_trades_repo,
            market_ticks=self._ticks_repo,
            data_client=self._clients.data_client,
            gamma_client=self._clients.gamma_client,
            alerts_repo=self._alerts_repo,
        )
```

- [ ] **Step 2: Call replay before the supervisor loop**

Edit `src/pscanner/scheduler.py`. Replace the early portion of `Scanner.run` (around line 551, right after `self.preflight()`):

```python
    async def run(self) -> None:
        """Drive the renderer plus every enabled detector and collector forever.

        ... (existing docstring)
        """
        self.preflight()
        await self._replay_paper_trader()
        for worker in self._workers:
            await worker.start()
        try:
            ...  # rest unchanged
```

Add a new helper method on `Scanner` (place it right after `preflight`):

```python
    async def _replay_paper_trader(self) -> None:
        """Replay unbooked alerts when paper-trading is enabled (issue #105)."""
        trader = self._detectors.get("paper_trader")
        if not isinstance(trader, PaperTrader):
            return
        try:
            await trader.replay_unbooked()
        except Exception:
            _LOG.exception("scanner.paper_trader_replay_failed")
```

The `PaperTrader` import already exists at line 95.

- [ ] **Step 3: Add a wiring test**

Append to `tests/test_scheduler.py` (or wherever scheduler-paper-trader wiring tests live — find via `grep -n "paper_trader\|paper_trading" tests/test_scheduler.py`):

```python
@pytest.mark.asyncio
async def test_scanner_replays_paper_trader_on_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Scanner.run calls PaperTrader.replay_unbooked between preflight and supervisor.

    Issue #105: bridges the alert→trade gap on daemon restart.
    """
    # Construct a minimal Config with paper_trading.enabled and a non-zero
    # replay_lookback_seconds. The test asserts replay_unbooked was invoked
    # by intercepting it on the constructed PaperTrader.
    ...
```

(The exact test code depends on the scheduler test file's fixtures — e.g. how it stubs clients, whether `Scanner.run` is called with a stop sentinel. If the existing scheduler test file does not already exercise `Scanner.run` end-to-end, instead add a unit test that invokes `_replay_paper_trader` directly with a mock `PaperTrader`. Keep this test in scope to under 40 lines.)

If the scheduler tests don't already start the supervisor loop, take the simpler path: assert that `_replay_paper_trader` calls `trader.replay_unbooked` exactly once when a `PaperTrader` is registered, and is a no-op otherwise:

```python
@pytest.mark.asyncio
async def test_replay_paper_trader_calls_replay_when_present(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``_replay_paper_trader`` invokes ``replay_unbooked`` on the registered trader."""
    from unittest.mock import AsyncMock

    cfg = _minimal_config_with_paper_trading(tmp_path)  # use existing helper
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        trader = scanner._detectors["paper_trader"]
        trader.replay_unbooked = AsyncMock(return_value=3)  # type: ignore[method-assign]
        await scanner._replay_paper_trader()
        trader.replay_unbooked.assert_awaited_once_with()
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_replay_paper_trader_noop_when_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No-op when paper_trading is disabled (no PaperTrader in detectors)."""
    cfg = _minimal_config()  # paper_trading.enabled=False (default)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        await scanner._replay_paper_trader()  # raises nothing, returns nothing
    finally:
        await scanner.aclose()
```

(`_minimal_config_with_paper_trading` and `_make_stub_clients` are placeholders — find the actual helper names by `grep -n "def _make_stub_clients\|def _minimal_config" tests/test_scheduler.py tests/scheduler/`.)

- [ ] **Step 4: Run scheduler tests**

Run: `uv run pytest tests/test_scheduler.py tests/scheduler/ -v`

Expected: all pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src/pscanner/scheduler.py tests/test_scheduler.py && uv run ruff format --check src/pscanner/scheduler.py tests/test_scheduler.py && uv run ty check src/pscanner/scheduler.py`

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): replay paper_trader on boot (#105)"
```

---

## Task 5: Document in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — paper-trading area or the gate-model bullet (whichever is closest in context)

- [ ] **Step 1: Locate the right bullet**

Find a paper-trading-related bullet in CLAUDE.md. The "Paper-trading evaluators" bullet under `## Codebase conventions` is the natural anchor. Append to the end of that bullet:

```
On daemon boot, `PaperTrader.replay_unbooked()` re-runs any alert from the last `paper_trading.replay_lookback_seconds` (default `0` = disabled) that has no `paper_trades` entry row, so a mid-day restart or "enable detector first then evaluator" workflow doesn't strand in-flight alerts (#105).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): note paper_trader restart replay (#105)"
```

---

## Verification

After all 5 tasks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: zero failures, zero new warnings.

Smoke verification (per the issue's acceptance criterion):

```bash
# 1. Start daemon with paper_trading.enabled=true, replay_lookback_seconds=900,
#    gate_model.enabled=true, gate_model_market_filter.enabled=true.
# 2. Wait for at least one gate_buy alert to land in the alerts table without
#    a paper_trades row (e.g. by stopping the daemon mid-cycle, or by config-
#    flipping the evaluator off then on across a restart).
# 3. Restart the daemon.
# 4. Confirm in the logs: structured `paper_trader.replay_complete count=N
#    lookback_seconds=900` event.
# 5. Confirm via `pscanner paper status` that the booked count went up.
```

---

## Self-Review

**Spec coverage:**
- Acceptance criterion 1 (decide A/B/C — committed to B): Tasks 1-4.
- Acceptance criterion 2 (document lookback knob): Task 5.
- Acceptance criterion 3 (test: restart books pre-restart alerts): Task 3's `test_replay_books_unbooked_alerts_in_window`.

**Out-of-scope checks:**
- Live-vs-historical price replay accuracy: NOT addressed (per issue's "Out of scope" — fill price comes from alert body, same as live path).
- Backfilling beyond the lookback window: NOT addressed (separate `pscanner paper replay` story).

**Placeholder scan:** Two callsites in Task 4's tests reference `_minimal_config` / `_make_stub_clients` helper names that need verifying against the actual test file. The implementer is told to grep for the real names — that's a structural pattern, not a placeholder.

**Type consistency:**
- `replay_lookback_seconds: int` (Task 1) → `min_created_at: int` (Task 2 query) → `int(time.time()) - lookback` (Task 3). All ints, all Unix seconds.
- `AlertsRepo` is imported from `pscanner.store.repo` consistently.
- `replay_unbooked() -> int` returns the count; the scheduler ignores it. Logging the count happens inside `replay_unbooked` itself.
