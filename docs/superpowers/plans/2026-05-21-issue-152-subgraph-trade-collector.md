# SubgraphTradeCollector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `scripts/watch_subgraph_copy.py` into a daemon `SubgraphTradeCollector` that emits `subgraph_copy` alerts, paired with a `SubgraphCopyEvaluator` that books to `paper_trades` with anti-concentration sizing.

**Architecture:** Collector polls the Polymarket V2 subgraph for trades by watchlisted wallets, resolves each event to an outcome name via the existing `pscanner.poly.token_resolver`, and emits one `subgraph_copy` `Alert` per BUY-direction event into the shared `AlertSink`. `SubgraphCopyEvaluator` subscribes via the existing `PaperTrader`, sizes each copy by `bankroll * position_fraction * concentration_multiplier(source_wallet)` where the multiplier decays as that wallet's share of total `subgraph_copy` trades exceeds `1.0 / active_watchlist_size`, floored at `min_multiplier`. Coexists with `/activity`-based `TradeCollector` — different tables, zero shared state.

**Tech Stack:** Python 3.13, asyncio, SQLite via `sqlite3`, pydantic for config, structlog for logs, pytest. The Graph (subgraph) via the existing `pscanner.poly.subgraph.SubgraphClient`.

**Spec:** `docs/superpowers/specs/2026-05-21-issue-152-subgraph-trade-collector-design.md`

---

## File Structure

**New files:**
- `src/pscanner/collectors/subgraph_trades.py` — collector implementation
- `src/pscanner/strategies/evaluators/subgraph_copy.py` — evaluator implementation
- `tests/collectors/test_subgraph_trades.py` — collector unit tests
- `tests/strategies/evaluators/test_subgraph_copy.py` — evaluator unit tests
- `tests/store/test_subgraph_watch_state_repo.py` — new state repo tests
- `tests/collectors/test_subgraph_trades_integration.py` — end-to-end integration test

**Modified files:**
- `src/pscanner/store/db.py` — add `subgraph_watch_state` to `_SCHEMA_STATEMENTS`
- `src/pscanner/store/repo.py` — add `SubgraphWatchStateRepo` + `PaperTradesRepo.count_by_source_wallet`
- `src/pscanner/alerts/models.py` — add `"subgraph_copy"` to `DetectorName`
- `src/pscanner/config.py` — add `SubgraphTradeCollectorConfig` + `SubgraphCopyEvaluatorConfig` + wire into `Config` + `EvaluatorsConfig`
- `src/pscanner/scheduler.py` — instantiate the collector + evaluator + open corpus DB conn
- `src/pscanner/strategies/evaluators/__init__.py` — export `SubgraphCopyEvaluator`
- `tests/store/test_paper_trades_repo.py` — extend with `count_by_source_wallet` tests
- `tests/test_scheduler.py` — wiring test

---

### Task 1: Schema for `subgraph_watch_state`

**Files:**
- Modify: `src/pscanner/store/db.py`

The collector needs a tiny single-row key/value table to persist `last_seen_ts` across daemon restarts. Additive `CREATE IF NOT EXISTS` — no migration step needed.

- [ ] **Step 1: Locate `_SCHEMA_STATEMENTS` and add a new entry**

Read `src/pscanner/store/db.py` to find the `_SCHEMA_STATEMENTS` tuple (currently includes `wallet_state_live`, `market_state_live`, `paper_trades`, etc.). Append a new statement immediately before the `*KALSHI_SCHEMA_STATEMENTS,` spread.

Add:

```python
    """
    CREATE TABLE IF NOT EXISTS subgraph_watch_state (
      key TEXT PRIMARY KEY,
      last_seen_ts INTEGER NOT NULL
    )
    """,
```

- [ ] **Step 2: Run the existing schema-init tests to confirm the new table creates**

Run: `uv run pytest tests/store/test_db.py -q`
Expected: PASS (no test should regress; the new statement is just an additional `CREATE IF NOT EXISTS`)

- [ ] **Step 3: Commit**

```bash
git add src/pscanner/store/db.py
git commit -m "feat(store): add subgraph_watch_state schema (#152)"
```

---

### Task 2: `SubgraphWatchStateRepo` with tests

**Files:**
- Create: `tests/store/test_subgraph_watch_state_repo.py`
- Modify: `src/pscanner/store/repo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/store/test_subgraph_watch_state_repo.py`:

```python
"""Tests for the SubgraphWatchStateRepo."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.store.db import init_db
from pscanner.store.repo import SubgraphWatchStateRepo


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    db = init_db(tmp_path / "pscanner.sqlite3")
    yield db
    db.close()


def test_get_returns_none_when_no_row(conn: sqlite3.Connection) -> None:
    repo = SubgraphWatchStateRepo(conn)
    assert repo.get_last_seen_ts() is None


def test_set_then_get_roundtrip(conn: sqlite3.Connection) -> None:
    repo = SubgraphWatchStateRepo(conn)
    repo.set_last_seen_ts(1_700_000_000)
    assert repo.get_last_seen_ts() == 1_700_000_000


def test_set_overwrites_existing_row(conn: sqlite3.Connection) -> None:
    repo = SubgraphWatchStateRepo(conn)
    repo.set_last_seen_ts(1_700_000_000)
    repo.set_last_seen_ts(1_700_000_500)
    assert repo.get_last_seen_ts() == 1_700_000_500
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/store/test_subgraph_watch_state_repo.py -q`
Expected: FAIL with `ImportError: cannot import name 'SubgraphWatchStateRepo' from 'pscanner.store.repo'`

- [ ] **Step 3: Implement the repo**

In `src/pscanner/store/repo.py`, add this class (place it near the other small key/value-style repos; e.g. just after `PaperTradesRepo`):

```python
class SubgraphWatchStateRepo:
    """Key/value persistence for the SubgraphTradeCollector watermark.

    One row keyed by ``"default"`` holds the most recently observed
    ``timestamp`` from the V2 subgraph's ``orderFilledEvents`` feed.
    Survives daemon restarts so we resume rather than re-scan.
    """

    _KEY = "default"

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def get_last_seen_ts(self) -> int | None:
        """Return the persisted watermark, or ``None`` if no row exists."""
        row = self._conn.execute(
            "SELECT last_seen_ts FROM subgraph_watch_state WHERE key = ?",
            (self._KEY,),
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def set_last_seen_ts(self, ts: int) -> None:
        """Atomically upsert the watermark to ``ts``."""
        self._conn.execute(
            """
            INSERT INTO subgraph_watch_state (key, last_seen_ts)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET last_seen_ts = excluded.last_seen_ts
            """,
            (self._KEY, int(ts)),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/store/test_subgraph_watch_state_repo.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/store/repo.py tests/store/test_subgraph_watch_state_repo.py
git commit -m "feat(store): SubgraphWatchStateRepo (#152)"
```

---

### Task 3: `PaperTradesRepo.count_by_source_wallet`

**Files:**
- Modify: `src/pscanner/store/repo.py` (extend `PaperTradesRepo`)
- Modify: `tests/store/test_paper_trades_repo.py`

- [ ] **Step 1: Write failing tests**

Append these tests to `tests/store/test_paper_trades_repo.py` (if the file doesn't exist, create it following the project's existing `tests/store/test_*_repo.py` shape — fixture `tmp_db` from `tests/conftest.py` gives an in-memory connection):

```python
def test_count_by_source_wallet_empty(tmp_db) -> None:
    from pscanner.store.repo import PaperTradesRepo
    repo = PaperTradesRepo(tmp_db)
    assert repo.count_by_source_wallet(detector="subgraph_copy") == {}


def test_count_by_source_wallet_groups(tmp_db) -> None:
    from pscanner.poly.ids import AssetId, ConditionId
    from pscanner.store.repo import PaperTradesRepo
    repo = PaperTradesRepo(tmp_db)
    cond = ConditionId("0xcond")
    ass = AssetId("123")

    def _insert(wallet: str, key: str, detector: str) -> None:
        repo.insert_entry(
            triggering_alert_key=key,
            triggering_alert_detector=detector,
            rule_variant=None,
            source_wallet=wallet,
            condition_id=cond,
            asset_id=ass,
            outcome="Yes",
            shares=1.0,
            fill_price=0.5,
            cost_usd=0.5,
            nav_after_usd=1000.0,
            ts=1_700_000_000,
        )

    _insert("0xaa", "k1", "subgraph_copy")
    _insert("0xaa", "k2", "subgraph_copy")
    _insert("0xbb", "k3", "subgraph_copy")
    _insert("0xaa", "k4", "gate_buy")  # different detector, should NOT count

    counts = repo.count_by_source_wallet(detector="subgraph_copy")
    assert counts == {"0xaa": 2, "0xbb": 1}


def test_count_by_source_wallet_excludes_null_wallet(tmp_db) -> None:
    from pscanner.poly.ids import AssetId, ConditionId
    from pscanner.store.repo import PaperTradesRepo
    repo = PaperTradesRepo(tmp_db)
    repo.insert_entry(
        triggering_alert_key="k1",
        triggering_alert_detector="subgraph_copy",
        rule_variant=None,
        source_wallet=None,
        condition_id=ConditionId("0xcond"),
        asset_id=AssetId("123"),
        outcome="Yes",
        shares=1.0,
        fill_price=0.5,
        cost_usd=0.5,
        nav_after_usd=1000.0,
        ts=1_700_000_000,
    )
    counts = repo.count_by_source_wallet(detector="subgraph_copy")
    assert counts == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/store/test_paper_trades_repo.py::test_count_by_source_wallet_empty -q`
Expected: FAIL with `AttributeError: 'PaperTradesRepo' object has no attribute 'count_by_source_wallet'`

- [ ] **Step 3: Implement the method**

Inside `PaperTradesRepo` (in `src/pscanner/store/repo.py`), add immediately after `summary_by_pred_bucket`:

```python
    def count_by_source_wallet(self, *, detector: str) -> dict[str, int]:
        """Return ``{source_wallet: count}`` for entries with the given detector.

        Rows whose ``source_wallet`` is NULL are excluded — the consumer
        (`SubgraphCopyEvaluator`) only meaningfully counts trades attributed
        to a known wallet. Entries of any ``rule_variant`` are aggregated
        together.

        Args:
            detector: Value of ``triggering_alert_detector`` to filter on
                (e.g. ``"subgraph_copy"``).

        Returns:
            Mapping from lower-cased wallet address to total entry count.
        """
        rows = self._conn.execute(
            """
            SELECT source_wallet, COUNT(*)
              FROM paper_trades
             WHERE triggering_alert_detector = ?
               AND trade_kind = 'entry'
               AND source_wallet IS NOT NULL
             GROUP BY source_wallet
            """,
            (detector,),
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/store/test_paper_trades_repo.py -k count_by_source_wallet -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/store/repo.py tests/store/test_paper_trades_repo.py
git commit -m "feat(store): PaperTradesRepo.count_by_source_wallet (#152)"
```

---

### Task 4: Add `"subgraph_copy"` to `DetectorName`

**Files:**
- Modify: `src/pscanner/alerts/models.py`

- [ ] **Step 1: Write a failing assertion in a temporary test (sanity check)**

Run this one-liner — it confirms the literal isn't there yet:

```bash
uv run python -c "from pscanner.alerts.models import DetectorName; from typing import get_args; assert 'subgraph_copy' in get_args(DetectorName), 'literal missing'"
```
Expected: `AssertionError: literal missing`

- [ ] **Step 2: Add the literal**

In `src/pscanner/alerts/models.py`, modify the `DetectorName` Literal:

```python
DetectorName = Literal[
    "smart_money",
    "mispricing",
    "monotone",
    "whales",
    "convergence",
    "velocity",
    "cluster",
    "move_attribution",
    "gate_buy",
    "subgraph_copy",
]
```

- [ ] **Step 3: Verify**

Run: `uv run python -c "from pscanner.alerts.models import DetectorName; from typing import get_args; assert 'subgraph_copy' in get_args(DetectorName)"`
Expected: exits 0, no output.

- [ ] **Step 4: Run linters + ty to make sure no consumer broke**

Run: `uv run ruff check . && uv run ty check`
Expected: clean (no new diagnostics)

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/alerts/models.py
git commit -m "feat(alerts): add subgraph_copy to DetectorName literal (#152)"
```

---

### Task 5: Config sections for collector + evaluator

**Files:**
- Modify: `src/pscanner/config.py`

- [ ] **Step 1: Write a failing test**

Create `tests/test_config_subgraph.py`:

```python
"""Defaults + validation for SubgraphTradeCollectorConfig and SubgraphCopyEvaluatorConfig."""

from __future__ import annotations

from pscanner.config import Config, SubgraphCopyEvaluatorConfig, SubgraphTradeCollectorConfig


def test_subgraph_collector_defaults() -> None:
    cfg = SubgraphTradeCollectorConfig()
    assert cfg.enabled is False
    assert cfg.subgraph_id == "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
    assert cfg.poll_interval_seconds == 10.0
    assert cfg.rpm == 60
    assert cfg.page_size == 1000
    assert cfg.cold_start_lookback_seconds == 0
    assert cfg.indexer_lag_warn_seconds == 60
    assert cfg.indexer_lag_error_seconds == 600


def test_subgraph_copy_evaluator_defaults() -> None:
    cfg = SubgraphCopyEvaluatorConfig()
    assert cfg.enabled is False
    assert cfg.position_fraction == 0.005
    assert cfg.min_multiplier == 0.10


def test_root_config_wires_subgraph_sections() -> None:
    root = Config()
    assert isinstance(root.subgraph_trades, SubgraphTradeCollectorConfig)
    assert isinstance(
        root.paper_trading.evaluators.subgraph_copy, SubgraphCopyEvaluatorConfig
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_subgraph.py -q`
Expected: FAIL with `ImportError: cannot import name 'SubgraphTradeCollectorConfig' from 'pscanner.config'`

- [ ] **Step 3: Add the config classes**

In `src/pscanner/config.py`, add these two new classes immediately after `GateModelMarketFilterConfig`:

```python
class SubgraphTradeCollectorConfig(_Section):
    """Tunables for the live SubgraphTradeCollector (#152).

    Polls the Polymarket V2 subgraph for trades by watchlisted wallets and
    emits ``subgraph_copy`` alerts. Coexists with the ``/activity``-based
    ``TradeCollector`` — both run independently.
    """

    enabled: bool = False
    subgraph_id: str = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
    poll_interval_seconds: float = 10.0
    rpm: int = 60
    page_size: int = 1000
    cold_start_lookback_seconds: int = 0
    """Seconds before ``now()`` to start from on first daemon boot when
    ``subgraph_watch_state`` is empty. ``0`` ignores history."""
    indexer_lag_warn_seconds: int = 60
    indexer_lag_error_seconds: int = 600
```

Add this near the other evaluator config classes (immediately after `GateModelEvaluatorConfig`):

```python
class SubgraphCopyEvaluatorConfig(_Section):
    """Tunables for the subgraph-copy paper-trading evaluator (#152).

    Sizes each copy at ``bankroll * position_fraction * multiplier`` where
    ``multiplier`` decays as a wallet's share of total subgraph_copy trades
    exceeds ``1.0 / active_watchlist_size``, floored at ``min_multiplier``.
    """

    enabled: bool = False
    position_fraction: float = 0.005
    min_multiplier: float = 0.10
```

Wire `SubgraphCopyEvaluatorConfig` into `EvaluatorsConfig`:

```python
class EvaluatorsConfig(_Section):
    smart_money: SmartMoneyEvaluatorConfig = Field(default_factory=SmartMoneyEvaluatorConfig)
    move_attribution: MoveAttributionEvaluatorConfig = Field(
        default_factory=MoveAttributionEvaluatorConfig,
    )
    velocity: VelocityEvaluatorConfig = Field(default_factory=VelocityEvaluatorConfig)
    mispricing: MispricingEvaluatorConfig = Field(default_factory=MispricingEvaluatorConfig)
    monotone: MonotoneEvaluatorConfig = Field(default_factory=MonotoneEvaluatorConfig)
    gate_model: GateModelEvaluatorConfig = Field(default_factory=GateModelEvaluatorConfig)
    subgraph_copy: SubgraphCopyEvaluatorConfig = Field(
        default_factory=SubgraphCopyEvaluatorConfig,
    )
```

Wire `SubgraphTradeCollectorConfig` into `Config`. Add this line near the other top-level fields (e.g. right after `gate_model_market_filter`):

```python
    subgraph_trades: SubgraphTradeCollectorConfig = Field(
        default_factory=SubgraphTradeCollectorConfig,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_subgraph.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Run linters**

Run: `uv run ruff check . && uv run ty check`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/config.py tests/test_config_subgraph.py
git commit -m "feat(config): SubgraphTradeCollectorConfig + SubgraphCopyEvaluatorConfig (#152)"
```

---

### Task 6: `SubgraphCopyEvaluator` — sizing + parse

**Files:**
- Create: `src/pscanner/strategies/evaluators/subgraph_copy.py`
- Create: `tests/strategies/evaluators/test_subgraph_copy.py`
- Modify: `src/pscanner/strategies/evaluators/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/evaluators/test_subgraph_copy.py`:

```python
"""Tests for SubgraphCopyEvaluator (#152)."""

from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import SubgraphCopyEvaluatorConfig
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.db import init_db
from pscanner.store.repo import PaperTradesRepo, WatchlistRepo
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    db = init_db(tmp_path / "pscanner.sqlite3")
    yield db
    db.close()


def _build_alert(*, wallet: str = "0xAA", outcome: str = "Yes", tx: str = "0xt1") -> Alert:
    return Alert(
        detector="subgraph_copy",
        alert_key=f"subgraph:{tx}:{outcome}",
        severity="med",
        title="copy",
        body={
            "source_wallet": wallet,
            "tx_hash": tx,
            "condition_id": "0xcond",
            "outcome": outcome,
            "ts": 1_700_000_000,
        },
        created_at=1_700_000_000,
    )


def _seed_watchlist(conn: sqlite3.Connection, *addrs: str) -> None:
    repo = WatchlistRepo(conn)
    for a in addrs:
        repo.upsert(address=a, source="manual", reason="test")


def _insert_paper_trade(conn: sqlite3.Connection, wallet: str, key: str) -> None:
    repo = PaperTradesRepo(conn)
    repo.insert_entry(
        triggering_alert_key=key,
        triggering_alert_detector="subgraph_copy",
        rule_variant=None,
        source_wallet=wallet,
        condition_id=ConditionId("0xcond"),
        asset_id=AssetId("123"),
        outcome="Yes",
        shares=1.0,
        fill_price=0.5,
        cost_usd=0.5,
        nav_after_usd=1000.0,
        ts=int(time.time()),
    )


def test_accepts_only_subgraph_copy(conn: sqlite3.Connection) -> None:
    cfg = SubgraphCopyEvaluatorConfig(enabled=True)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    assert ev.accepts(_build_alert()) is True

    other = Alert(
        detector="smart_money",
        alert_key="k",
        severity="med",
        title="",
        body={},
        created_at=0,
    )
    assert ev.accepts(other) is False


def test_parse_returns_one_signal_with_metadata(conn: sqlite3.Connection) -> None:
    cfg = SubgraphCopyEvaluatorConfig(enabled=True)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    alert = _build_alert(wallet="0xAA", outcome="Cavaliers", tx="0xtx")
    signals = ev.parse(alert)
    assert len(signals) == 1
    sig = signals[0]
    assert str(sig.condition_id) == "0xcond"
    assert sig.side == "Cavaliers"
    assert sig.rule_variant is None
    assert sig.metadata["wallet"] == "0xAA"
    assert sig.metadata["tx_hash"] == "0xtx"
    assert sig.metadata["ts"] == 1_700_000_000


def test_quality_passes_always_true(conn: sqlite3.Connection) -> None:
    cfg = SubgraphCopyEvaluatorConfig(enabled=True)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert())
    assert ev.quality_passes(sig) is True


def test_size_full_base_when_no_prior_trades(conn: sqlite3.Connection) -> None:
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    assert ev.size(1000.0, sig) == pytest.approx(5.0)


def test_size_full_base_at_target_share(conn: sqlite3.Connection) -> None:
    # 3 wallets watched; target_share = 1/3 = 0.333.
    # 0xAA has 1/3 of trades -> share exactly at target -> multiplier 1.0.
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    _insert_paper_trade(conn, "0xAA", "k1")
    _insert_paper_trade(conn, "0xBB", "k2")
    _insert_paper_trade(conn, "0xCC", "k3")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    assert ev.size(1000.0, sig) == pytest.approx(5.0)


def test_size_decays_above_target_share(conn: sqlite3.Connection) -> None:
    # 3 wallets watched; target_share = 0.333.
    # 0xAA has 3 of 4 (75%); multiplier = min(1, 0.333/0.75) = 0.444.
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    _insert_paper_trade(conn, "0xAA", "k1")
    _insert_paper_trade(conn, "0xAA", "k2")
    _insert_paper_trade(conn, "0xAA", "k3")
    _insert_paper_trade(conn, "0xBB", "k4")
    cfg = SubgraphCopyEvaluatorConfig(
        enabled=True, position_fraction=0.005, min_multiplier=0.10
    )
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    expected = 1000.0 * 0.005 * (1.0 / 3.0) / 0.75
    assert ev.size(1000.0, sig) == pytest.approx(expected)


def test_size_floored_at_min_multiplier(conn: sqlite3.Connection) -> None:
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC", "0xDD", "0xEE")
    # Only 0xAA has trades; share = 1.0.
    _insert_paper_trade(conn, "0xAA", "k1")
    cfg = SubgraphCopyEvaluatorConfig(
        enabled=True, position_fraction=0.005, min_multiplier=0.10
    )
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    # raw = min(1, 0.2/1.0) = 0.2; floor = 0.1 -> raw wins.
    assert ev.size(1000.0, sig) == pytest.approx(1000.0 * 0.005 * 0.2)


def test_size_wallet_lookup_is_case_insensitive(conn: sqlite3.Connection) -> None:
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    _insert_paper_trade(conn, "0xaa", "k1")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    # 0xAA share = 1/1 = 1.0, target = 1/3, raw = 0.333, floored at 0.10.
    assert ev.size(1000.0, sig) == pytest.approx(1000.0 * 0.005 * (1.0 / 3.0))


def test_size_empty_watchlist_treats_as_one(conn: sqlite3.Connection) -> None:
    # Defensive: no active watchlist rows; target_share = 1/1 = 1.0.
    # No trades yet => total=0 => multiplier 1.0 => base size.
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    assert ev.size(1000.0, sig) == pytest.approx(5.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/evaluators/test_subgraph_copy.py -q`
Expected: FAIL with `ImportError: cannot import name 'SubgraphCopyEvaluator' from 'pscanner.strategies.evaluators.subgraph_copy'`

- [ ] **Step 3: Implement the evaluator**

Create `src/pscanner/strategies/evaluators/subgraph_copy.py`:

```python
"""SubgraphCopyEvaluator — books paper copies of watchlisted wallets' trades.

The evaluator is paired with :class:`pscanner.collectors.subgraph_trades.SubgraphTradeCollector`
(spec: ``docs/superpowers/specs/2026-05-21-issue-152-subgraph-trade-collector-design.md``).

Sizing is constant ``bankroll * position_fraction`` times a per-wallet
concentration multiplier that decays as one wallet's share of total
``subgraph_copy`` trades exceeds ``1.0 / active_watchlist_size``, floored at
``min_multiplier``. The floor guarantees the noisiest wallet still trades at
``≥ min_multiplier × base`` rather than being silenced entirely.
"""

from __future__ import annotations

from pscanner.alerts.models import Alert
from pscanner.config import SubgraphCopyEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import PaperTradesRepo, WatchlistRepo
from pscanner.strategies.evaluators.protocol import ParsedSignal

_DETECTOR_NAME = "subgraph_copy"


class SubgraphCopyEvaluator:
    """Single-leg evaluator for the SubgraphTradeCollector alert stream."""

    def __init__(
        self,
        *,
        config: SubgraphCopyEvaluatorConfig,
        watchlist_repo: WatchlistRepo,
        paper_trades: PaperTradesRepo,
    ) -> None:
        """Bind config + read-only repos used at sizing time.

        Args:
            config: Tunables (position_fraction, min_multiplier).
            watchlist_repo: Source of ``active_watchlist_size`` for target_share.
            paper_trades: Source of ``count_by_source_wallet`` for share.
        """
        self._config = config
        self._watchlist = watchlist_repo
        self._paper_trades = paper_trades

    def accepts(self, alert: Alert) -> bool:
        """Only handle ``subgraph_copy`` alerts."""
        return alert.detector == _DETECTOR_NAME

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Extract a single :class:`ParsedSignal` from the alert body.

        Returns an empty list on body-shape mismatch so PaperTrader's
        soft-failure path applies (skip without crashing).
        """
        body = alert.body
        try:
            condition_id = ConditionId(str(body["condition_id"]))
            outcome = str(body["outcome"])
            wallet = str(body["source_wallet"])
            tx_hash = str(body["tx_hash"])
            ts = int(body["ts"])
        except (KeyError, TypeError, ValueError):
            return []
        return [
            ParsedSignal(
                condition_id=condition_id,
                side=outcome,
                rule_variant=None,
                metadata={"wallet": wallet, "tx_hash": tx_hash, "ts": ts},
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:  # noqa: ARG002
        """No quality gate — watchlist admission is the gate."""
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return ``bankroll * position_fraction * concentration_multiplier``."""
        base = bankroll * self._config.position_fraction
        wallet = str(parsed.metadata.get("wallet", ""))
        return base * self._concentration_multiplier(wallet)

    def _concentration_multiplier(self, wallet: str) -> float:
        """Compute the per-wallet sizing multiplier in ``[min_multiplier, 1.0]``.

        ``share = trades_copied[wallet] / total_subgraph_copy_trades``.
        ``target_share = 1.0 / max(1, active_watchlist_size)``.
        ``raw = min(1.0, target_share / max(share, target_share))``.
        Final = ``max(raw, min_multiplier)``.
        """
        counts = self._paper_trades.count_by_source_wallet(detector=_DETECTOR_NAME)
        total = sum(counts.values())
        if total == 0:
            return 1.0
        wallet_lower = wallet.lower()
        # Counts are keyed by the stored (case-sensitive) source_wallet.
        # Match both raw and lower-cased variants defensively.
        wallet_count = counts.get(wallet, 0) + counts.get(wallet_lower, 0)
        if wallet in counts and wallet_lower in counts and wallet != wallet_lower:
            wallet_count = counts[wallet] + counts[wallet_lower]
        share = wallet_count / total
        active_n = max(1, len(self._watchlist.list_active()))
        target_share = 1.0 / active_n
        raw = min(1.0, target_share / max(share, target_share))
        return max(raw, self._config.min_multiplier)
```

Then re-export it from `src/pscanner/strategies/evaluators/__init__.py` by adding to the import block AND the `__all__` (if present). Open that file and append:

```python
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator
```

(If `__all__` lists each evaluator name, add `"SubgraphCopyEvaluator"` to it as well.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/evaluators/test_subgraph_copy.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Run linters**

Run: `uv run ruff check . && uv run ty check`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/strategies/evaluators/subgraph_copy.py \
        src/pscanner/strategies/evaluators/__init__.py \
        tests/strategies/evaluators/test_subgraph_copy.py
git commit -m "feat(strategies): SubgraphCopyEvaluator with concentration-aware sizing (#152)"
```

---

### Task 7: `SubgraphTradeCollector` — copy-direction + alert emission (no I/O)

**Files:**
- Create: `src/pscanner/collectors/subgraph_trades.py`
- Create: `tests/collectors/test_subgraph_trades.py`

Build the collector in two halves to keep tests targeted. This task lands the pure logic — direction computation, alert body shape, alert key format — without wiring the subgraph client or the run loop. Task 8 adds the poll loop and integration with `SubgraphClient`.

- [ ] **Step 1: Write the failing tests**

Create `tests/collectors/test_subgraph_trades.py`:

```python
"""Unit tests for SubgraphTradeCollector internals (#152)."""

from __future__ import annotations

from pscanner.collectors.subgraph_trades import (
    DETECTOR_TAG,
    SUBGRAPH_ID,
    _build_where_clause,
    _compute_copy_direction,
    _serialize_where_inline,
)


def test_compute_copy_direction_maker_buy() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=0, watchlist={"0xaa"}
    )
    assert direction == "BUY"


def test_compute_copy_direction_taker_sell_is_buy() -> None:
    # watchlist == taker AND side == 1 -> taker bought (hit a sell order).
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=1, watchlist={"0xbb"}
    )
    assert direction == "BUY"


def test_compute_copy_direction_maker_sell_is_skip() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=1, watchlist={"0xaa"}
    )
    assert direction == "SKIP"


def test_compute_copy_direction_taker_buy_is_skip() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=0, watchlist={"0xbb"}
    )
    assert direction == "SKIP"


def test_compute_copy_direction_neither_side_watched() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=0, watchlist={"0xcc"}
    )
    assert direction == "SKIP"


def test_serialize_where_inline_renders_object_literal() -> None:
    rendered = _serialize_where_inline(
        {"timestamp_gte": "100", "maker_in": ["0xaa", "0xbb"]}
    )
    # GraphQL object literals do NOT quote keys.
    assert rendered == '{timestamp_gte:"100",maker_in:["0xaa","0xbb"]}'


def test_build_where_clause_repeats_timestamp_inside_or() -> None:
    where = _build_where_clause(["0xaa", "0xbb"], 1_700_000_000)
    assert where == {
        "or": [
            {"timestamp_gte": "1700000000", "maker_in": ["0xaa", "0xbb"]},
            {"timestamp_gte": "1700000000", "taker_in": ["0xaa", "0xbb"]},
        ],
    }


def test_constants_exposed_for_callers() -> None:
    assert DETECTOR_TAG == "subgraph_copy"
    assert SUBGRAPH_ID == "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/collectors/test_subgraph_trades.py -q`
Expected: FAIL with `ImportError: cannot import name '_compute_copy_direction' from 'pscanner.collectors.subgraph_trades'`

- [ ] **Step 3: Implement the helpers**

Create `src/pscanner/collectors/subgraph_trades.py`:

```python
"""Subgraph-driven copy-trade collector (#152).

Polls the Polymarket V2 subgraph for ``orderFilledEvents`` involving any
watchlisted wallet and emits a ``subgraph_copy`` :class:`Alert` for every
position-increasing trade. Booking happens downstream via
:class:`pscanner.strategies.evaluators.subgraph_copy.SubgraphCopyEvaluator`.

Lifecycle is owned by the daemon scheduler — restart-on-crash, shared
``stop_event`` for clean shutdown.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from typing import Any, Final

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.protocol import IAlertSink
from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import SubgraphTradeCollectorConfig
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId
from pscanner.poly.subgraph import SubgraphClient
from pscanner.poly.token_resolver import resolve_token
from pscanner.store.repo import MarketCacheRepo, SubgraphWatchStateRepo
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)

SUBGRAPH_ID: Final[str] = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
DETECTOR_TAG: Final[str] = "subgraph_copy"
ALERT_KEY_PREFIX: Final[str] = "subgraph"
PAGE_SIZE: Final[int] = 1000

_GRAPHQL_QUERY: Final[str] = f"""
{{
  orderFilledEvents(
    where: $where
    first: {PAGE_SIZE}
    orderBy: timestamp
    orderDirection: asc
  ) {{
    transactionHash
    timestamp
    maker {{ id }}
    taker {{ id }}
    market {{ id }}
    tokenId
    side
    price
    size
  }}
  _meta {{ block {{ number timestamp }} }}
}}
"""


def _compute_copy_direction(
    maker: str, taker: str, side: int, watchlist: set[str]
) -> str:
    """Return ``"BUY"`` iff the watchlist wallet's position increases.

    Subgraph ``side``: 0=BUY, 1=SELL on the order's direction.
    Maker placed the resting order; taker hit it from the opposite side.

    - watchlist == maker AND side == 0 -> maker accumulates -> BUY
    - watchlist == maker AND side == 1 -> maker reduces      -> SKIP
    - watchlist == taker AND side == 0 -> taker sold         -> SKIP
    - watchlist == taker AND side == 1 -> taker bought       -> BUY
    """
    maker_l = maker.lower()
    taker_l = taker.lower()
    if maker_l in watchlist and side == 0:
        return "BUY"
    if taker_l in watchlist and side == 1:
        return "BUY"
    return "SKIP"


def _serialize_where_inline(where: dict[str, Any]) -> str:
    """Render ``where:`` as a GraphQL object literal (NOT JSON).

    GraphQL object literals do not quote keys. We hand-emit a minimal
    serializer to avoid pulling in a full GraphQL client.
    """

    def render(v: Any) -> str:
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ",".join(render(x) for x in v) + "]"
        if isinstance(v, dict):
            inner = ",".join(f"{k}:{render(val)}" for k, val in v.items())
            return "{" + inner + "}"
        raise TypeError(f"unsupported where value: {v!r}")

    return render(where)


def _build_where_clause(addrs: list[str], last_seen_ts: int) -> dict[str, Any]:
    """Build the ``where:`` argument for ``orderFilledEvents``.

    TheGraph rejects ``or`` mixed with same-level column filters, so the
    timestamp predicate must be repeated inside each ``or`` branch.
    ``timestamp_gte`` (not ``_gt``) plus a within-cycle ``tx_hash`` dedupe
    gives strict no-loss boundary behaviour.
    """
    ts_str = str(last_seen_ts)
    return {
        "or": [
            {"timestamp_gte": ts_str, "maker_in": addrs},
            {"timestamp_gte": ts_str, "taker_in": addrs},
        ],
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/collectors/test_subgraph_trades.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/collectors/subgraph_trades.py tests/collectors/test_subgraph_trades.py
git commit -m "feat(collectors): subgraph_trades helpers + module scaffold (#152)"
```

---

### Task 8: `SubgraphTradeCollector` class — poll cycle + alert emission

**Files:**
- Modify: `src/pscanner/collectors/subgraph_trades.py` (add the class)
- Modify: `tests/collectors/test_subgraph_trades.py` (add lifecycle + cycle tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/collectors/test_subgraph_trades.py`:

```python
import asyncio
import sqlite3
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from structlog.testing import capture_logs

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import SubgraphTradeCollectorConfig
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    SubgraphWatchStateRepo,
    WatchlistRepo,
)
from pscanner.util.clock import FakeClock


@pytest.fixture
def daemon_conn(tmp_path) -> sqlite3.Connection:
    db = init_db(tmp_path / "pscanner.sqlite3")
    yield db
    db.close()


@pytest.fixture
def corpus_conn(tmp_path) -> sqlite3.Connection:
    db = init_corpus_db(tmp_path / "corpus.sqlite3")
    yield db
    db.close()


def _make_sink(daemon_conn) -> AlertSink:
    return AlertSink(AlertsRepo(daemon_conn))


def _make_event(*, tx: str, ts: int, maker: str, taker: str, side: int, token: str) -> dict:
    return {
        "transactionHash": tx,
        "timestamp": str(ts),
        "maker": {"id": maker},
        "taker": {"id": taker},
        "market": {"id": token},
        "tokenId": token,
        "side": side,
        "price": "0.5",
        "size": "1.0",
    }


@pytest.mark.asyncio
async def test_poll_once_empty_watchlist_short_circuits(daemon_conn, corpus_conn) -> None:
    registry = WatchlistRegistry(WatchlistRepo(daemon_conn))  # empty
    cfg = SubgraphTradeCollectorConfig(enabled=True)
    sub_client = MagicMock()
    sub_client.query = AsyncMock()
    collector = SubgraphTradeCollector(
        config=cfg,
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=_make_sink(daemon_conn),
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    with capture_logs() as logs:
        await collector._poll_once()
    sub_client.query.assert_not_called()
    assert any(log["event"] == "subgraph_trades.empty_watchlist" for log in logs)


@pytest.mark.asyncio
async def test_poll_once_emits_alert_for_watchlist_buy(
    daemon_conn, corpus_conn, monkeypatch,
) -> None:
    repo = WatchlistRepo(daemon_conn)
    repo.upsert(address="0xaa", source="manual", reason="test")
    registry = WatchlistRegistry(repo)

    cfg = SubgraphTradeCollectorConfig(enabled=True)
    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                _make_event(
                    tx="0xtx1",
                    ts=1_700_000_100,
                    maker="0xAA",
                    taker="0xBB",
                    side=0,  # maker BUY -> watchlist wallet accumulates
                    token="9999",
                ),
            ],
            "_meta": {"block": {"number": "1", "timestamp": str(int(time.time()))}},
        }
    )

    # Patch resolve_token to bypass gamma/asset_index.
    from pscanner.collectors import subgraph_trades as mod
    from pscanner.poly.ids import AssetId, ConditionId
    from pscanner.poly.token_resolver import ResolvedToken

    async def fake_resolve(*, token_id, asset_index, market_cache, gamma):
        return ResolvedToken(
            condition_id=ConditionId("0xcond"),
            asset_id=AssetId(str(token_id)),
            outcome_name="Yes",
            outcome_index=0,
        )

    monkeypatch.setattr(mod, "resolve_token", fake_resolve)

    sink = _make_sink(daemon_conn)
    emitted: list[Alert] = []
    sink.subscribe(emitted.append)
    collector = SubgraphTradeCollector(
        config=cfg,
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=sink,
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    await collector._poll_once()

    assert len(emitted) == 1
    alert = emitted[0]
    assert alert.detector == "subgraph_copy"
    assert alert.alert_key == "subgraph:0xtx1:Yes"
    assert alert.body["source_wallet"].lower() == "0xaa"
    assert alert.body["condition_id"] == "0xcond"
    assert alert.body["outcome"] == "Yes"
    assert alert.body["ts"] == 1_700_000_100


@pytest.mark.asyncio
async def test_poll_once_persists_new_last_seen_ts(
    daemon_conn, corpus_conn, monkeypatch,
) -> None:
    WatchlistRepo(daemon_conn).upsert(address="0xaa", source="manual", reason="t")
    registry = WatchlistRegistry(WatchlistRepo(daemon_conn))
    state_repo = SubgraphWatchStateRepo(daemon_conn)
    state_repo.set_last_seen_ts(1_700_000_000)

    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                _make_event(
                    tx="0xtx",
                    ts=1_700_000_200,
                    maker="0xAA",
                    taker="0xBB",
                    side=0,
                    token="t1",
                ),
            ],
            "_meta": {"block": {"number": "1", "timestamp": str(int(time.time()))}},
        }
    )

    from pscanner.collectors import subgraph_trades as mod
    from pscanner.poly.ids import AssetId, ConditionId
    from pscanner.poly.token_resolver import ResolvedToken

    async def fake_resolve(*, token_id, asset_index, market_cache, gamma):
        return ResolvedToken(
            condition_id=ConditionId("0xcond"),
            asset_id=AssetId(str(token_id)),
            outcome_name="Yes",
            outcome_index=0,
        )

    monkeypatch.setattr(mod, "resolve_token", fake_resolve)

    collector = SubgraphTradeCollector(
        config=SubgraphTradeCollectorConfig(enabled=True),
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=_make_sink(daemon_conn),
        state_repo=state_repo,
        clock=FakeClock(),
    )
    await collector._poll_once()
    assert state_repo.get_last_seen_ts() == 1_700_000_200


@pytest.mark.asyncio
async def test_poll_once_skips_sells_silently(daemon_conn, corpus_conn) -> None:
    WatchlistRepo(daemon_conn).upsert(address="0xaa", source="manual", reason="t")
    registry = WatchlistRegistry(WatchlistRepo(daemon_conn))
    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                _make_event(
                    tx="0xtx",
                    ts=1_700_000_300,
                    maker="0xAA",
                    taker="0xBB",
                    side=1,  # maker SELL -> watchlist reduces -> SKIP
                    token="t1",
                ),
            ],
            "_meta": {"block": {"number": "1", "timestamp": str(int(time.time()))}},
        }
    )
    sink = _make_sink(daemon_conn)
    emitted: list[Alert] = []
    sink.subscribe(emitted.append)
    collector = SubgraphTradeCollector(
        config=SubgraphTradeCollectorConfig(enabled=True),
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=sink,
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    await collector._poll_once()
    assert emitted == []


@pytest.mark.asyncio
async def test_run_stops_on_stop_event(daemon_conn, corpus_conn) -> None:
    cfg = SubgraphTradeCollectorConfig(enabled=True, poll_interval_seconds=0.01)
    sub_client = MagicMock()
    sub_client.query = AsyncMock(return_value={"orderFilledEvents": [], "_meta": {"block": {}}})
    collector = SubgraphTradeCollector(
        config=cfg,
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=WatchlistRegistry(WatchlistRepo(daemon_conn)),
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=_make_sink(daemon_conn),
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    stop = asyncio.Event()

    async def _stopper() -> None:
        await asyncio.sleep(0.05)
        stop.set()

    await asyncio.gather(collector.run(stop), _stopper())
```

(`FakeClock` and `pytest-asyncio` are already used elsewhere — see `tests/conftest.py` and other collector tests for the canonical import.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/collectors/test_subgraph_trades.py -q`
Expected: FAIL with `ImportError: cannot import name 'SubgraphTradeCollector' from 'pscanner.collectors.subgraph_trades'`

- [ ] **Step 3: Implement the class**

Append to `src/pscanner/collectors/subgraph_trades.py`:

```python
async def _fetch_events_since(
    client: SubgraphClient,
    *,
    addrs: list[str],
    last_seen_ts: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Drain the subgraph for all events newer than ``last_seen_ts``.

    Watermark pagination: each page advances ``ts`` to the most recent event
    seen. Loop terminates when a page returns fewer than ``PAGE_SIZE`` events.
    Within-cycle ``tx_hash`` dedupe catches boundary events re-fetched by
    ``timestamp_gte``.

    Returns the unique events (asc ``ts`` order) and the indexer's
    ``_meta.block.timestamp`` from the last page (used by the caller for lag
    detection).
    """
    events: list[dict[str, Any]] = []
    seen_tx: set[str] = set()
    ts = last_seen_ts
    indexer_ts: int | None = None
    while True:
        where = _build_where_clause(addrs, ts)
        graphql = _GRAPHQL_QUERY.replace("$where", _serialize_where_inline(where))
        data = await client.query(graphql, {})
        page = data.get("orderFilledEvents") or []
        for e in page:
            tx = e["transactionHash"]
            if tx in seen_tx:
                continue
            seen_tx.add(tx)
            events.append(e)
        meta_block = (data.get("_meta") or {}).get("block") or {}
        meta_ts_raw = meta_block.get("timestamp")
        if meta_ts_raw is not None:
            indexer_ts = int(meta_ts_raw)
        if len(page) < PAGE_SIZE:
            break
        ts = max(int(e["timestamp"]) for e in page)
    return events, indexer_ts


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
    """Wait up to ``seconds`` for ``stop_event``. Return True if it was set."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


class SubgraphTradeCollector:
    """Polls the V2 subgraph for watchlist trades and emits ``subgraph_copy`` alerts."""

    name: str = "subgraph_trades"

    def __init__(
        self,
        *,
        config: SubgraphTradeCollectorConfig,
        subgraph_client: SubgraphClient,
        gamma_client: GammaClient,
        watchlist: WatchlistRegistry,
        asset_index: AssetIndexRepo,
        market_cache: MarketCacheRepo,
        sink: IAlertSink,
        state_repo: SubgraphWatchStateRepo,
        clock: Clock | None = None,
    ) -> None:
        """Wire the collector to its inputs and outputs.

        Args:
            config: Polling cadence + cold-start + lag thresholds.
            subgraph_client: GraphQL client (already RPM-budgeted).
            gamma_client: Used by ``resolve_token`` on local-cache miss.
            watchlist: Active address set (thread-safe registry).
            asset_index: Corpus-DB asset_id -> condition_id mapping (read/write).
            market_cache: Daemon-DB market metadata (read/write).
            sink: Where to emit ``subgraph_copy`` alerts.
            state_repo: Watermark persistence.
            clock: Injectable clock for tests; defaults to :class:`RealClock`.
        """
        self._config = config
        self._subgraph = subgraph_client
        self._gamma = gamma_client
        self._watchlist = watchlist
        self._asset_index = asset_index
        self._market_cache = market_cache
        self._sink = sink
        self._state = state_repo
        self._clock = clock if clock is not None else RealClock()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the polling loop until ``stop_event`` is set.

        Per-cycle exceptions are logged and swallowed so a transient upstream
        hiccup does not kill the loop.
        """
        while not stop_event.is_set():
            try:
                await self._poll_once()
            except Exception:
                _LOG.exception("subgraph_trades.cycle_failed")
            if await _wait_or_stop(stop_event, self._config.poll_interval_seconds):
                return

    async def _poll_once(self) -> None:
        """Run a single poll cycle: fetch, emit alerts, persist watermark."""
        addrs = sorted({a.lower() for a in self._watchlist.addresses()})
        if not addrs:
            _LOG.warning("subgraph_trades.empty_watchlist")
            return
        last_seen = self._state.get_last_seen_ts()
        if last_seen is None:
            last_seen = int(time.time()) - int(self._config.cold_start_lookback_seconds)
        _LOG.info(
            "subgraph_trades.poll_start", addrs=len(addrs), last_seen_ts=last_seen,
        )
        events, indexer_ts = await _fetch_events_since(
            self._subgraph, addrs=addrs, last_seen_ts=last_seen,
        )
        self._warn_on_lag(indexer_ts)
        new_last_seen = last_seen
        watchlist_set = set(addrs)
        emitted = 0
        for ev in events:
            new_last_seen = max(new_last_seen, int(ev["timestamp"]))
            body = await self._build_alert_body(ev, watchlist_set)
            if body is None:
                continue
            alert = Alert(
                detector="subgraph_copy",
                alert_key=f"{ALERT_KEY_PREFIX}:{ev['transactionHash']}:{body['outcome']}",
                severity="med",
                title=(
                    f"copy {body['source_wallet'][:14]}.. {body['outcome']}"
                    f" @ {float(ev.get('price', 0.0)):.3f}"
                ),
                body=body,
                created_at=int(ev["timestamp"]),
            )
            await self._sink.emit(alert)
            emitted += 1
        self._state.set_last_seen_ts(new_last_seen)
        _LOG.info(
            "subgraph_trades.poll_done",
            events_seen=len(events),
            emitted=emitted,
            new_last_seen_ts=new_last_seen,
        )

    async def _build_alert_body(
        self, ev: dict[str, Any], watchlist: set[str]
    ) -> dict[str, Any] | None:
        """Resolve the event into an alert body, or ``None`` to skip."""
        maker = ev["maker"]["id"]
        taker = ev["taker"]["id"]
        side = int(ev["side"])
        if _compute_copy_direction(maker, taker, side, watchlist) != "BUY":
            return None
        source_wallet = maker if maker.lower() in watchlist else taker
        try:
            resolved = await resolve_token(
                token_id=AssetId(ev["tokenId"]),
                asset_index=self._asset_index,
                market_cache=self._market_cache,
                gamma=self._gamma,
            )
        except Exception:
            _LOG.exception(
                "subgraph_trades.resolve_failed", token_id=ev.get("tokenId")
            )
            return None
        if resolved is None:
            _LOG.warning(
                "subgraph_trades.token_unresolved", token_id=ev.get("tokenId")
            )
            return None
        return {
            "source_wallet": source_wallet,
            "tx_hash": ev["transactionHash"],
            "condition_id": str(resolved.condition_id),
            "outcome": resolved.outcome_name,
            "ts": int(ev["timestamp"]),
        }

    def _warn_on_lag(self, indexer_ts: int | None) -> None:
        """Emit warn/error logs when the subgraph indexer lags the head."""
        if indexer_ts is None:
            return
        lag = int(time.time()) - indexer_ts
        if lag >= self._config.indexer_lag_error_seconds:
            _LOG.error("subgraph_trades.indexer_lag", lag_seconds=lag)
        elif lag >= self._config.indexer_lag_warn_seconds:
            _LOG.warning("subgraph_trades.indexer_lag", lag_seconds=lag)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/collectors/test_subgraph_trades.py -q`
Expected: PASS (13 passed — 8 from Task 7 + 5 new)

- [ ] **Step 5: Run linters + ty**

Run: `uv run ruff check . && uv run ty check`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/collectors/subgraph_trades.py tests/collectors/test_subgraph_trades.py
git commit -m "feat(collectors): SubgraphTradeCollector poll loop + alert emission (#152)"
```

---

### Task 9: Scheduler wiring + preflight + corpus DB connection

**Files:**
- Modify: `src/pscanner/scheduler.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scheduler.py` (extend the existing file; if it doesn't import a Scanner-construction helper, use the bare `Scanner(...)` ctor that other tests already use):

```python
def test_subgraph_trades_wired_when_enabled(tmp_path, monkeypatch) -> None:
    from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
    from pscanner.config import (
        Config,
        PaperTradingConfig,
        SubgraphCopyEvaluatorConfig,
        SubgraphTradeCollectorConfig,
    )
    from pscanner.scheduler import Scanner
    from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator

    monkeypatch.setenv("GRAPH_API_KEY", "test-key")
    config = Config(
        subgraph_trades=SubgraphTradeCollectorConfig(enabled=True),
        paper_trading=PaperTradingConfig(
            enabled=True,
            evaluators={"subgraph_copy": SubgraphCopyEvaluatorConfig(enabled=True)},
        ),
    )
    scanner = Scanner(config=config, db_path=tmp_path / "p.sqlite3")
    assert isinstance(
        scanner._collectors.get("subgraph_trades"), SubgraphTradeCollector
    )
    pt = scanner._detectors["paper_trader"]
    assert any(isinstance(e, SubgraphCopyEvaluator) for e in pt._evaluators)


def test_subgraph_trades_preflight_requires_graph_api_key(tmp_path, monkeypatch) -> None:
    from pscanner.config import Config, SubgraphTradeCollectorConfig
    from pscanner.scheduler import Scanner

    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    config = Config(subgraph_trades=SubgraphTradeCollectorConfig(enabled=True))
    scanner = Scanner(config=config, db_path=tmp_path / "p.sqlite3")
    with pytest.raises(RuntimeError, match="GRAPH_API_KEY"):
        scanner.preflight()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_scheduler.py::test_subgraph_trades_wired_when_enabled -q`
Expected: FAIL — `KeyError: 'subgraph_trades'` (collector not yet wired) or similar.

- [ ] **Step 3: Wire the collector + evaluator into the scheduler**

In `src/pscanner/scheduler.py`:

1. Add these imports (at the top, grouped with the existing collector / strategy imports):

```python
from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.subgraph import SubgraphClient
from pscanner.store.repo import SubgraphWatchStateRepo
from pscanner.strategies.evaluators import SubgraphCopyEvaluator
```

2. In `Scanner.__init__`, after `self._db = init_db(resolved_db)`, lazily open the corpus DB only when needed:

```python
self._corpus_conn: sqlite3.Connection | None = None
if self._config.subgraph_trades.enabled:
    corpus_path = Path("data/corpus.sqlite3")
    self._corpus_conn = init_corpus_db(corpus_path)
```

3. Inside `_build_collectors`, after the `market_scoped_trades` block, add:

```python
        if self._config.subgraph_trades.enabled:
            assert self._corpus_conn is not None  # guaranteed by __init__
            api_key = os.environ.get("GRAPH_API_KEY", "")
            subgraph_url = (
                f"https://gateway.thegraph.com/api/{api_key}"
                f"/subgraphs/id/{self._config.subgraph_trades.subgraph_id}"
            )
            collectors["subgraph_trades"] = SubgraphTradeCollector(
                config=self._config.subgraph_trades,
                subgraph_client=SubgraphClient(
                    url=subgraph_url, rpm=self._config.subgraph_trades.rpm,
                ),
                gamma_client=self._clients.gamma_client,
                watchlist=self._watchlist_registry,
                asset_index=AssetIndexRepo(self._corpus_conn),
                market_cache=self._market_cache_repo,
                sink=self._sink,
                state_repo=SubgraphWatchStateRepo(self._db),
                clock=self._clock,
            )
```

(Add `import os` at the top if not already imported.)

4. In `_build_paper_evaluators`, after the `gate_model` block, add:

```python
        if cfg.subgraph_copy.enabled:
            evaluators.append(
                SubgraphCopyEvaluator(
                    config=cfg.subgraph_copy,
                    watchlist_repo=self._watchlist_repo,
                    paper_trades=paper_trades_repo,
                )
            )
```

5. In `Scanner.preflight`, add a new check at the end:

```python
        if self._config.subgraph_trades.enabled:
            if not os.environ.get("GRAPH_API_KEY"):
                msg = (
                    "subgraph_trades.enabled=true but GRAPH_API_KEY is not set. "
                    "Export it before starting the daemon."
                )
                raise RuntimeError(msg)
```

6. In `Scanner.aclose`, close the corpus connection. Find the existing `with contextlib.suppress(sqlite3.Error): self._db.close()` block and add immediately after it:

```python
        if self._corpus_conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self._corpus_conn.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -q`
Expected: the two new tests pass, all existing scheduler tests still pass.

- [ ] **Step 5: Run linters**

Run: `uv run ruff check . && uv run ty check`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): wire SubgraphTradeCollector + SubgraphCopyEvaluator (#152)"
```

---

### Task 10: End-to-end integration test (also satisfies #163)

**Files:**
- Create: `tests/collectors/test_subgraph_trades_integration.py`

This is the integration test issue #163 requested — exercises the full collector→sink→evaluator→`paper_trades` path against real SQLite + the production `init_corpus_db` / `init_db` helpers, with only the subgraph + gamma clients mocked. Catches `mode=ro` / connection-permission regressions like the one fixed in commit `0f54e56`.

- [ ] **Step 1: Write the failing integration test**

Create `tests/collectors/test_subgraph_trades_integration.py`:

```python
"""End-to-end smoke test: poll-cycle -> alert -> paper_trades booking."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import (
    PaperTradingConfig,
    SubgraphCopyEvaluatorConfig,
    SubgraphTradeCollectorConfig,
)
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.token_resolver import ResolvedToken
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    PaperTradesRepo,
    SubgraphWatchStateRepo,
    WatchlistRepo,
)
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator
from pscanner.strategies.paper_trader import PaperTrader
from pscanner.util.clock import FakeClock


@pytest.mark.asyncio
async def test_full_cycle_books_paper_trade(tmp_path, monkeypatch) -> None:
    daemon = init_db(tmp_path / "pscanner.sqlite3")
    corpus = init_corpus_db(tmp_path / "corpus.sqlite3")

    # Seed watchlist.
    WatchlistRepo(daemon).upsert(address="0xaa", source="manual", reason="t")

    # Pre-seed market_cache so PaperTrader._resolve_outcome can find the asset.
    # MarketCacheRepo.upsert takes a Market or CachedMarket; the minimum we
    # need is condition_id, outcomes and clob_token_ids parallel arrays, and
    # an outcome_prices array for the price fallback.
    from pscanner.poly.models import Market

    market = Market(
        id="m1",
        question="ignored",
        slug="seed-slug",
        condition_id=ConditionId("0xcond"),
        outcomes=["Yes", "No"],
        outcome_prices=[0.5, 0.5],
        clob_token_ids=[AssetId("token-yes"), AssetId("token-no")],
        active=True,
        closed=False,
    )
    MarketCacheRepo(daemon).upsert(market)

    # Build sink + evaluator pipeline.
    sink = AlertSink(AlertsRepo(daemon))
    paper_trades = PaperTradesRepo(daemon)
    paper_trader = PaperTrader(
        config=PaperTradingConfig(
            enabled=True,
            starting_bankroll_usd=1000.0,
            evaluators={"subgraph_copy": SubgraphCopyEvaluatorConfig(enabled=True)},
        ),
        evaluators=[
            SubgraphCopyEvaluator(
                config=SubgraphCopyEvaluatorConfig(enabled=True),
                watchlist_repo=WatchlistRepo(daemon),
                paper_trades=paper_trades,
            ),
        ],
        market_cache=MarketCacheRepo(daemon),
        paper_trades=paper_trades,
        market_ticks=MagicMock(latest_for_asset=MagicMock(return_value=None)),
        data_client=MagicMock(),
        gamma_client=MagicMock(),
        alerts_repo=AlertsRepo(daemon),
    )
    sink.subscribe(paper_trader.handle_alert_sync)

    # Fake subgraph: returns one BUY event by the watchlisted wallet.
    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                {
                    "transactionHash": "0xtxabc",
                    "timestamp": "1700000100",
                    "maker": {"id": "0xAA"},
                    "taker": {"id": "0xBB"},
                    "market": {"id": "token-yes"},
                    "tokenId": "token-yes",
                    "side": 0,
                    "price": "0.5",
                    "size": "1.0",
                },
            ],
            "_meta": {"block": {"number": "1", "timestamp": "1700000100"}},
        }
    )

    # Stub resolve_token so we don't need a real gamma response.
    from pscanner.collectors import subgraph_trades as mod

    async def fake_resolve(*, token_id, asset_index, market_cache, gamma):
        return ResolvedToken(
            condition_id=ConditionId("0xcond"),
            asset_id=AssetId(str(token_id)),
            outcome_name="Yes",
            outcome_index=0,
        )

    monkeypatch.setattr(mod, "resolve_token", fake_resolve)

    collector = SubgraphTradeCollector(
        config=SubgraphTradeCollectorConfig(enabled=True),
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=WatchlistRegistry(WatchlistRepo(daemon)),
        asset_index=AssetIndexRepo(corpus),
        market_cache=MarketCacheRepo(daemon),
        sink=sink,
        state_repo=SubgraphWatchStateRepo(daemon),
        clock=FakeClock(),
    )

    await collector._poll_once()
    # PaperTrader's handle_alert_sync schedules a task; let it run.
    await asyncio.sleep(0)
    # Drain any pending tasks the handler scheduled.
    for _ in range(5):
        await asyncio.sleep(0)

    # Assert: one paper_trades row exists with the right detector tag.
    rows = list(daemon.execute(
        "SELECT triggering_alert_detector, source_wallet, outcome "
        "FROM paper_trades WHERE trade_kind='entry'"
    ))
    assert len(rows) == 1
    detector, wallet, outcome = rows[0]
    assert detector == "subgraph_copy"
    assert wallet.lower() == "0xaa"
    assert outcome == "Yes"

    daemon.close()
    corpus.close()
```

- [ ] **Step 2: Run the test to verify it fails (or surfaces issues)**

Run: `uv run pytest tests/collectors/test_subgraph_trades_integration.py -q`
Expected: PASS if all earlier tasks landed correctly. If it fails, the failure surfaces a real integration issue (e.g. evaluator not in PaperTrader's evaluator list, market_cache schema not seeded right). Fix the underlying issue rather than trimming the test.

- [ ] **Step 3: Run full project gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/collectors/test_subgraph_trades_integration.py
git commit -m "test(collectors): integration smoke for subgraph_copy pipeline (closes #163)"
```

---

### Task 11: Desktop deploy + live smoke

All live runs happen on the desktop (`10.0.0.143:2222`) per `LOCAL_NOTES.md`. The laptop's watchlist is empty and its corpus is stale.

- [ ] **Step 1: Push branch and rsync to desktop**

```bash
git push origin HEAD
rsync -avh -e "ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  /home/macph/projects/polymarketScanner/src/pscanner/ \
  macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/src/pscanner/
rsync -avh -e "ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  /home/macph/projects/polymarketScanner/tests/ \
  macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/tests/
```

- [ ] **Step 2: Run the test suite on the desktop**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run ruff check . && uv run ty check && uv run pytest -q'
```
Expected: green.

- [ ] **Step 3: Enable the collector + evaluator in `~/projects/polymarketscanner/pscanner/config.toml` on the desktop**

Add:

```toml
[subgraph_trades]
enabled = true
poll_interval_seconds = 10.0

[paper_trading.evaluators.subgraph_copy]
enabled = true
position_fraction = 0.005
min_multiplier = 0.10
```

(`paper_trading.enabled` must already be true.)

- [ ] **Step 4: Verify GRAPH_API_KEY is exported**

```bash
ssh -p 2222 macph@10.0.0.143 'grep -E "^export GRAPH_API_KEY" ~/.bashrc || echo MISSING'
```

If MISSING, set it before continuing.

- [ ] **Step 5: Start the daemon and watch the logs**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   nohup uv run pscanner run > /tmp/subgraph_smoke.log 2>&1 < /dev/null & \
   disown'
```

Wait ~2 minutes, then:

```bash
ssh -p 2222 macph@10.0.0.143 'grep -E "subgraph_trades|subgraph_copy" /tmp/subgraph_smoke.log | head -40'
```

Expected:
- One `subgraph_trades.poll_start` log per cycle.
- `subgraph_trades.poll_done` lines with `emitted=N`.
- Zero `subgraph_trades.cycle_failed` entries.

- [ ] **Step 6: Confirm DB state**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run python <<PY
import sqlite3
c = sqlite3.connect("data/pscanner.sqlite3")
c.row_factory = sqlite3.Row
def q(s, *a): return [dict(r) for r in c.execute(s, a).fetchall()]
print("subgraph_watch_state:", q("SELECT * FROM subgraph_watch_state"))
print("subgraph_copy alerts:", q("SELECT COUNT(*) n FROM alerts WHERE detector=?", "subgraph_copy"))
print("paper_trades by source:", q("SELECT triggering_alert_detector det, COUNT(*) n FROM paper_trades GROUP BY 1"))
print("top wallets:", q("SELECT source_wallet, COUNT(*) n FROM paper_trades WHERE triggering_alert_detector=? GROUP BY 1 ORDER BY n DESC LIMIT 5", "subgraph_copy"))
PY'
```

Expected:
- `subgraph_watch_state` shows one row with a recent `last_seen_ts`.
- `subgraph_copy` row count in `alerts` is non-zero.
- `paper_trades` continues growing under `triggering_alert_detector='subgraph_copy'`.
- The single chatty wallet (`0x6a67...`) — once it crosses `share > target_share = 1/5 = 0.20` — books at `1000 * 0.005 * min_multiplier = $0.50` instead of `$5.00`.

- [ ] **Step 7: Sample a sized row to confirm the multiplier is active**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run python <<PY
import sqlite3
c = sqlite3.connect("data/pscanner.sqlite3")
c.row_factory = sqlite3.Row
row = c.execute(
    "SELECT source_wallet, cost_usd, ts FROM paper_trades "
    "WHERE triggering_alert_detector=? AND trade_kind=\"entry\" "
    "ORDER BY ts DESC LIMIT 5",
    ("subgraph_copy",),
).fetchall()
for r in row:
    print(dict(r))
PY'
```

Expected: at least one row with `cost_usd <= 1.0` for the chatty wallet (multiplier active). New wallets entering with share < target should book at `cost_usd ~= 5.00`.

- [ ] **Step 8: Let the daemon run for 1 hour, then re-check**

After ~1h, repeat Step 6's query. Verify `subgraph_watch_state.last_seen_ts` has advanced, alert counts grew, and no `cycle_failed` entries appeared.

- [ ] **Step 9: Stop the daemon, commit the config change**

```bash
ssh -p 2222 macph@10.0.0.143 'pkill -f "pscanner run"'
```

If config lives in-repo on the desktop, commit + push. If config is local-only, note the live-config change in a follow-up CLAUDE.md operator-notes update.

- [ ] **Step 10: Run `pscanner paper status` to confirm per-source breakdown shows subgraph_copy**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run pscanner paper status'
```

Expected: a `subgraph_copy` row in the per-source breakdown.

---

### Task 12: Update CLAUDE.md operator notes

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a bullet under "Codebase conventions" or "Tracked work in flight"**

Append (or update an existing related bullet):

```markdown
- **SubgraphTradeCollector (#152, live 2026-05-21).** Daemon collector at
  `pscanner.collectors.subgraph_trades.SubgraphTradeCollector` polls the V2
  Polymarket subgraph for trades by every wallet in `wallet_watchlist`
  (active=1) and emits `subgraph_copy` alerts. Booking goes through
  `SubgraphCopyEvaluator` with anti-concentration sizing: per-wallet size is
  `bankroll * position_fraction * multiplier` where multiplier decays as a
  wallet's share of total subgraph_copy trades exceeds `1.0 / active_watchlist_size`,
  floored at `min_multiplier`. Watermark persists in
  `subgraph_watch_state.last_seen_ts`. Refuses to start when `GRAPH_API_KEY`
  is not set. Coexists with the `/activity` REST `TradeCollector` — different
  tables, zero shared state. The earlier research script
  `scripts/watch_subgraph_copy.py` is retired (delete in follow-up).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): SubgraphTradeCollector daemon promotion (#152)"
```

- [ ] **Step 3: Decommission the script (separate follow-up PR)**

Once the daemon path is live-validated for at least one 24-hour window with non-zero alert + paper_trade activity:

```bash
git rm scripts/watch_subgraph_copy.py
git rm docs/superpowers/specs/2026-05-20-subgraph-watcher-copy-design.md
git rm docs/superpowers/plans/2026-05-20-subgraph-watcher-copy.md
git rm tests/scripts/test_watch_subgraph_copy.py
git commit -m "chore(scripts): remove subgraph_copy research script (#152 follow-up)"
```

(Reusable helpers — `lookup_fill_price`, `resolve_token` — already live outside the script.)

---

## Self-Review

**Spec coverage:**
- Goal 1 (run in daemon, supervised): Task 8 (run loop) + Task 9 (scheduler wiring).
- Goal 2 (emit subgraph_copy alerts, book via evaluator): Task 6 (evaluator), Task 8 (collector), Task 9 (PaperTrader registration).
- Goal 3 (anti-concentration sizing): Task 6 — multiple test cases cover the curve.
- Goal 4 (coexists with TradeCollector): No change to TradeCollector; Task 11 step 6 confirms via DB state.
- Goal 5 (preserve script semantics): Task 7 ports helpers verbatim; Task 8's query string is byte-identical to the script's.
- Spec section "Schema additions": Task 1.
- Spec section "Config additions": Task 5.
- Spec section "Scheduler wiring + corpus DB connection + preflight": Task 9.
- Spec section "Renderer": automatic via `DetectorName` Literal (Task 4) — no per-detector format string is needed; spec acknowledges this.
- Spec section "Smoke plan" (all on desktop): Task 11.
- Spec section "Test surface": Tasks 6, 7, 8, 9, 10.
- Spec section "Decommissioning the script": Task 12 Step 3 (deferred follow-up).
- Issue #163 (integration smoke test): Task 10.

**Placeholder scan:** No "TBD" or "implement later". Every code step has the actual code. Every test has the actual assertions.

**Type / name consistency:**
- Collector constructor parameter is `watchlist: WatchlistRegistry` everywhere (Tasks 8, 9, 10).
- Evaluator constructor parameter is `watchlist_repo: WatchlistRepo` everywhere (Tasks 6, 9, 10).
- `DETECTOR_TAG = "subgraph_copy"` used consistently in Tasks 6, 7, 8, plus matches the `DetectorName` literal added in Task 4.
- Alert body keys (`source_wallet`, `tx_hash`, `condition_id`, `outcome`, `ts`) match between Task 6 (evaluator's `parse`) and Task 8 (collector's `_build_alert_body`).
- `count_by_source_wallet(detector=...)` signature matches between Task 3 (impl) and Task 6 (evaluator's use).
- `SubgraphWatchStateRepo.get_last_seen_ts` / `set_last_seen_ts` consistent between Tasks 2, 8, 9, 10.
