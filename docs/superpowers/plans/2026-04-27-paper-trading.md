# Paper-Trading Copy-Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `PaperTrader` subscriber + `PaperResolver` polling detector that mirror smart-money entries on a virtual bankroll, hold to resolution, and book PnL — exposed via `pscanner paper status`.

**Architecture:** Two new in-daemon strategies under `src/pscanner/strategies/`. `PaperTrader` subscribes to `AlertSink` and inserts `entry` rows into a new `paper_trades` table on filtered smart-money alerts. `PaperResolver` is a `PollingDetector` that scans open positions and inserts `exit` rows when the underlying market resolves. State is a single new table; sizing is cost-basis NAV × `position_fraction` (default 1%).

**Tech Stack:** Python 3.13, asyncio, sqlite3, structlog, pytest + respx.

**Spec:** `docs/superpowers/specs/2026-04-27-paper-trading-design.md`

**Spec data-gap discovered during planning:** `market_cache` only stores prices, not outcomes or asset_ids. The spec's `outcome_to_asset(condition_id, "oilers")` lookup has no data source. T1 extends the cache schema to persist `outcomes_json` and `asset_ids_json` from the gamma `/markets` response (already parsed into the `Market` pydantic model — just dropped on the way to SQLite today). This is independent infrastructure that several future strategies will benefit from.

**Subagent dispatch friendliness:** Tasks within each wave touch disjoint files, so they can be dispatched in parallel worktrees. Wave-cross dependencies are explicit.

---

## File structure

**Create:**
- `src/pscanner/strategies/__init__.py` — namespace marker (T3 or T4, whichever runs first)
- `src/pscanner/strategies/paper_trader.py` — `PaperTrader` class + helpers (T3)
- `src/pscanner/strategies/paper_resolver.py` — `PaperResolver` class + helpers (T4)
- `tests/strategies/__init__.py` — namespace marker
- `tests/strategies/test_paper_trader.py` (T3)
- `tests/strategies/test_paper_resolver.py` (T4)

**Modify:**
- `src/pscanner/store/db.py` — add `paper_trades` schema + indexes; add `ALTER market_cache ADD …` migrations (T1)
- `src/pscanner/store/repo.py` — extend `CachedMarket` + `MarketCacheRepo`; add `PaperTradesRepo` + `OpenPaperPosition`/`PaperSummary` rows (T1)
- `src/pscanner/collectors/markets.py` — pass `outcomes` + `clob_token_ids` through to `CachedMarket` (T1)
- `src/pscanner/poly/models.py` — no changes (parsing already exists)
- `src/pscanner/config.py` — add `PaperTradingConfig` + wire into `Config` (T2)
- `tests/test_config.py` — defaults test for the new section (T2)
- `tests/store/test_repo.py` (or wherever existing repo tests live) — coverage for new repo methods (T1)
- `tests/collectors/test_markets.py` — coverage for the new fields persisted (T1)
- `src/pscanner/scheduler.py` — instantiate `PaperTrader`+`PaperResolver`; subscribe to `AlertSink`; poll cadence wired (T5)
- `tests/test_scheduler.py` — integration smoke test (T5)
- `src/pscanner/cli.py` — add `paper status` subcommand (T6)
- `tests/test_cli.py` — coverage for the new subcommand (T6)

## Wave structure (subagent dispatch)

| Wave | Parallel tasks | Files touched |
|------|---|---|
| 1 | **T1**, **T2** | T1: db.py + repo.py + collectors/markets.py · T2: config.py |
| 2 | **T3**, **T4** | T3: strategies/paper_trader.py · T4: strategies/paper_resolver.py |
| 3 | **T5**, **T6** | T5: scheduler.py · T6: cli.py |

Within each wave, the listed tasks are file-disjoint and can run in parallel git worktrees.

---

## Task 1: Data layer (market_cache extension + paper_trades schema/repo)

Extends `market_cache` to persist the outcomes and asset_ids gamma already returns, adds the `paper_trades` table and `PaperTradesRepo`, and updates `MarketsCollector` to populate the new fields. Two logical pieces, both in the same task because they both touch `db.py` + `repo.py` and would conflict in parallel.

**Files:**
- Modify: `src/pscanner/store/db.py`
- Modify: `src/pscanner/store/repo.py`
- Modify: `src/pscanner/collectors/markets.py`
- Test: `tests/store/test_repo.py` (or new file `tests/store/test_paper_trades_repo.py` — check for existing repo tests first)
- Test: `tests/collectors/test_markets.py`

**Wave:** 1 (parallel with T2). Worktree: `pscanner-worktrees/paper-data-layer`. Branch: `feat/paper-data-layer`.

### Step 1.1: market_cache schema migration

- [ ] **Step 1.1.1: Write failing tests for the new fields**

In `tests/store/test_repo.py` (or wherever `MarketCacheRepo` tests live; if none, create `tests/store/test_market_cache_outcomes.py`):

```python
import sqlite3
import pytest

from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import CachedMarket, MarketCacheRepo


def _build_market(
    *,
    market_id: str = "mkt-1",
    condition_id: str = "0xcond-1",
    outcomes: list[str] | None = None,
    asset_ids: list[str] | None = None,
    outcome_prices: list[float] | None = None,
) -> CachedMarket:
    return CachedMarket(
        market_id=MarketId(market_id),
        event_id=None,
        title="t",
        liquidity_usd=1.0,
        volume_usd=1.0,
        outcome_prices=outcome_prices or [0.6, 0.4],
        outcomes=outcomes or ["Yes", "No"],
        asset_ids=[AssetId(a) for a in (asset_ids or ["asset-yes", "asset-no"])],
        active=True,
        cached_at=1700000000,
        condition_id=ConditionId(condition_id),
        event_slug=None,
    )


def test_market_cache_persists_outcomes_and_asset_ids(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market())
    got = repo.get_by_condition_id(ConditionId("0xcond-1"))
    assert got is not None
    assert got.outcomes == ["Yes", "No"]
    assert got.asset_ids == [AssetId("asset-yes"), AssetId("asset-no")]


def test_outcome_to_asset_exact_match(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market(outcomes=["Yes", "No"], asset_ids=["asset-yes", "asset-no"]))
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "Yes") == AssetId("asset-yes")
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "No") == AssetId("asset-no")


def test_outcome_to_asset_case_and_whitespace_tolerant(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market(outcomes=["Oilers", "Ducks"], asset_ids=["a-oil", "a-duck"]))
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "oilers") == AssetId("a-oil")
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), " DUCKS ") == AssetId("a-duck")


def test_outcome_to_asset_returns_none_when_market_missing(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    assert repo.outcome_to_asset(ConditionId("0xnope"), "Yes") is None


def test_outcome_to_asset_returns_none_when_outcome_missing(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market(outcomes=["Yes", "No"], asset_ids=["a-y", "a-n"]))
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "Maybe") is None


def test_outcome_to_asset_returns_none_when_lengths_mismatch(tmp_db: sqlite3.Connection) -> None:
    # Defensive: if outcomes and asset_ids have different lengths, treat as missing
    repo = MarketCacheRepo(tmp_db)
    m = _build_market(outcomes=["Yes", "No"], asset_ids=["only-one"])
    repo.upsert(m)
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "Yes") is None
```

- [ ] **Step 1.1.2: Run, verify they fail**

Run: `uv run pytest tests/store/ -v -k "outcome_to_asset or persists_outcomes"`
Expected: FAIL — `CachedMarket` does not yet have `outcomes` or `asset_ids`; `outcome_to_asset` does not exist.

- [ ] **Step 1.1.3: Add the schema migrations**

In `src/pscanner/store/db.py`, append to the `_MIGRATIONS` tuple:

```python
    "ALTER TABLE market_cache ADD COLUMN outcomes_json TEXT",
    "ALTER TABLE market_cache ADD COLUMN asset_ids_json TEXT",
```

(Existing `_apply_migrations` already swallows the "duplicate column name" `OperationalError`, so this is idempotent.)

- [ ] **Step 1.1.4: Extend `CachedMarket` and `MarketCacheRepo`**

In `src/pscanner/store/repo.py`, find the existing `CachedMarket` dataclass and `MarketCacheRepo` class. Add the two list fields to `CachedMarket`:

```python
@dataclass(frozen=True, slots=True)
class CachedMarket:
    """A cached gamma market."""

    market_id: MarketId
    event_id: str | None
    title: str | None
    liquidity_usd: float | None
    volume_usd: float | None
    outcome_prices: list[float]
    outcomes: list[str]
    asset_ids: list[AssetId]
    active: bool
    cached_at: int
    condition_id: ConditionId | None
    event_slug: EventSlug | None
```

Update the upsert SQL in `MarketCacheRepo.upsert` to write the new columns, and update every SELECT (`get`, `get_by_condition_id`, `list_active`) to read them. Add a helper `_decode_outcomes_and_assets(row)` that JSON-decodes both columns and returns `(list[str], list[AssetId])` (empty lists when null/missing).

Add a new method on `MarketCacheRepo`:

```python
    def outcome_to_asset(
        self,
        condition_id: ConditionId,
        outcome_name: str,
    ) -> AssetId | None:
        """Resolve ``outcome_name`` to an ``AssetId`` on the given market.

        Case- and whitespace-insensitive. Returns ``None`` if the market is
        missing, the outcome is not in the cached list, or the outcomes /
        asset_ids lists have mismatched lengths (defensive).
        """
        cached = self.get_by_condition_id(condition_id)
        if cached is None:
            return None
        if len(cached.outcomes) != len(cached.asset_ids):
            return None
        target = outcome_name.strip().casefold()
        for name, asset_id in zip(cached.outcomes, cached.asset_ids, strict=True):
            if name.strip().casefold() == target:
                return asset_id
        return None
```

Update the row-decoder helper that `get`/`get_by_condition_id`/`list_active` all use, so each `CachedMarket` constructed from a row gets `outcomes` and `asset_ids` populated.

- [ ] **Step 1.1.5: Update MarketsCollector to persist the new fields**

In `src/pscanner/collectors/markets.py`, find where `MarketsCollector` constructs the upsert payload (~line 121 today writes only `outcome_prices_json`). Pass `market.outcomes` and `market.clob_token_ids` through into the `CachedMarket` constructed for upsert. Pydantic `Market` already exposes both fields parsed.

- [ ] **Step 1.1.6: Update tests/collectors/test_markets.py**

Find the existing test that asserts a row was upserted and add assertions for the new fields:

```python
    cached = market_cache.get_by_condition_id(ConditionId("0xcond-1"))
    assert cached is not None
    assert cached.outcomes == ["Yes", "No"]
    assert len(cached.asset_ids) == 2
```

(The exact existing test name will dictate the patch shape; locate via `grep -n market_cache tests/collectors/test_markets.py`.)

- [ ] **Step 1.1.7: Run all tests**

```bash
uv run pytest tests/store/ tests/collectors/test_markets.py -v
```

Expected: all pass.

- [ ] **Step 1.1.8: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/store/ src/pscanner/collectors/markets.py tests/store/ tests/collectors/test_markets.py
uv run ruff format src/pscanner/store/ src/pscanner/collectors/markets.py tests/store/ tests/collectors/test_markets.py
uv run ty check src/pscanner
```

Expected: all clean.

- [ ] **Step 1.1.9: Commit**

```bash
git add src/pscanner/store/ src/pscanner/collectors/markets.py tests/store/ tests/collectors/test_markets.py
git commit -m "feat(market_cache): persist outcomes and asset_ids; add outcome_to_asset helper

Gamma /markets already returns outcomes + clobTokenIds (parsed into the
Market pydantic model); we just dropped them on the way to SQLite. Add
two columns to market_cache, plumb them through MarketCacheRepo.upsert/
get/list_active, and add outcome_to_asset(condition_id, outcome_name) for
strategies that need the name → asset_id mapping (e.g. paper-trading).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Step 1.2: paper_trades schema + PaperTradesRepo

- [ ] **Step 1.2.1: Write failing tests**

Create `tests/store/test_paper_trades_repo.py`:

```python
"""Tests for PaperTradesRepo."""
from __future__ import annotations

import sqlite3

import pytest

from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    OpenPaperPosition,
    PaperSummary,
    PaperTradesRepo,
)


_NOW = 1700000000


def _entry(repo: PaperTradesRepo, **overrides) -> int:
    args = {
        "triggering_alert_key": "smart:0xw1:0xc1:yes:20260427",
        "source_wallet": "0xwallet1",
        "condition_id": ConditionId("0xcond-1"),
        "asset_id": AssetId("asset-yes"),
        "outcome": "yes",
        "shares": 20.0,
        "fill_price": 0.5,
        "cost_usd": 10.0,
        "nav_after_usd": 990.0,
        "ts": _NOW,
    }
    args.update(overrides)
    return repo.insert_entry(**args)


def test_insert_entry_returns_trade_id(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    tid = _entry(repo)
    assert tid >= 1


def test_insert_entry_unique_alert_key(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    _entry(repo, triggering_alert_key="smart:0xa:0xc:yes:1")
    with pytest.raises(sqlite3.IntegrityError):
        _entry(repo, triggering_alert_key="smart:0xa:0xc:yes:1")


def test_insert_entry_null_alert_key_allowed(tmp_db: sqlite3.Connection) -> None:
    # Defensive: not strictly used in v1, but the partial UNIQUE index allows it
    repo = PaperTradesRepo(tmp_db)
    _entry(repo, triggering_alert_key=None)
    _entry(repo, triggering_alert_key=None)  # second null is fine


def test_insert_exit_links_parent(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    parent = _entry(repo)
    repo.insert_exit(
        parent_trade_id=parent,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,
        nav_after_usd=1010.0,
        ts=_NOW + 100,
    )
    assert repo.list_open_positions() == []  # closed by the exit


def test_list_open_positions_returns_unmatched_entries(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    p1 = _entry(repo, triggering_alert_key="a1")
    p2 = _entry(repo, triggering_alert_key="a2")
    repo.insert_exit(
        parent_trade_id=p1,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,
        nav_after_usd=1010.0,
        ts=_NOW + 100,
    )
    open_positions = repo.list_open_positions()
    assert [p.trade_id for p in open_positions] == [p2]
    assert isinstance(open_positions[0], OpenPaperPosition)


def test_compute_cost_basis_nav_empty(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 1000.0


def test_compute_cost_basis_nav_open_only(tmp_db: sqlite3.Connection) -> None:
    # Open positions don't move NAV — they stay on cost basis
    repo = PaperTradesRepo(tmp_db)
    _entry(repo)
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 1000.0


def test_compute_cost_basis_nav_one_winning_exit(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    parent = _entry(repo, cost_usd=10.0, shares=20.0, fill_price=0.5)
    repo.insert_exit(
        parent_trade_id=parent,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=1.0,
        cost_usd=20.0,  # proceeds
        nav_after_usd=1010.0,
        ts=_NOW + 100,
    )
    # NAV = 1000 + (20 proceeds - 10 cost) = 1010
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 1010.0


def test_compute_cost_basis_nav_one_losing_exit(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    parent = _entry(repo, cost_usd=10.0)
    repo.insert_exit(
        parent_trade_id=parent,
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-yes"),
        outcome="yes",
        shares=20.0,
        fill_price=0.0,
        cost_usd=0.0,
        nav_after_usd=990.0,
        ts=_NOW + 100,
    )
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 990.0


def test_compute_cost_basis_nav_mixed(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    p1 = _entry(repo, cost_usd=10.0, triggering_alert_key="a1")
    p2 = _entry(repo, cost_usd=20.0, triggering_alert_key="a2")
    p3 = _entry(repo, cost_usd=15.0, triggering_alert_key="a3")  # remains open
    repo.insert_exit(
        parent_trade_id=p1, condition_id=ConditionId("0xc"), asset_id=AssetId("a"),
        outcome="yes", shares=1, fill_price=1, cost_usd=15.0,
        nav_after_usd=0, ts=_NOW + 1,
    )  # win: +5
    repo.insert_exit(
        parent_trade_id=p2, condition_id=ConditionId("0xc"), asset_id=AssetId("a"),
        outcome="yes", shares=1, fill_price=0, cost_usd=0.0,
        nav_after_usd=0, ts=_NOW + 2,
    )  # loss: -20
    # Net realized: 15 - 10 + 0 - 20 = -15. NAV = 1000 - 15 = 985.
    # p3 still open — doesn't affect NAV.
    assert repo.compute_cost_basis_nav(starting_bankroll=1000.0) == 985.0


def test_summary_stats(tmp_db: sqlite3.Connection) -> None:
    repo = PaperTradesRepo(tmp_db)
    p1 = _entry(repo, cost_usd=10.0, triggering_alert_key="a1", source_wallet="0xw1")
    _entry(repo, cost_usd=20.0, triggering_alert_key="a2", source_wallet="0xw2")  # open
    repo.insert_exit(
        parent_trade_id=p1, condition_id=ConditionId("0xc"), asset_id=AssetId("a"),
        outcome="yes", shares=1, fill_price=1, cost_usd=15.0,
        nav_after_usd=0, ts=_NOW + 1,
    )
    summary: PaperSummary = repo.summary_stats(starting_bankroll=1000.0)
    assert summary.starting_bankroll == 1000.0
    assert summary.current_nav == 1005.0  # 1000 + (15 - 10)
    assert summary.realized_pnl == 5.0
    assert summary.open_positions == 1
    assert summary.closed_positions == 1
```

- [ ] **Step 1.2.2: Run, verify they fail**

```bash
uv run pytest tests/store/test_paper_trades_repo.py -v
```

Expected: FAIL — `PaperTradesRepo`, `OpenPaperPosition`, `PaperSummary` don't exist.

- [ ] **Step 1.2.3: Add the table to schema**

In `src/pscanner/store/db.py`, append to `_SCHEMA_STATEMENTS`:

```python
    """
    CREATE TABLE IF NOT EXISTS paper_trades (
      trade_id             INTEGER PRIMARY KEY AUTOINCREMENT,
      trade_kind           TEXT    NOT NULL,
      triggering_alert_key TEXT,
      parent_trade_id      INTEGER,
      source_wallet        TEXT,
      condition_id         TEXT    NOT NULL,
      asset_id             TEXT    NOT NULL,
      outcome              TEXT    NOT NULL,
      shares               REAL    NOT NULL,
      fill_price           REAL    NOT NULL,
      cost_usd             REAL    NOT NULL,
      nav_after_usd        REAL    NOT NULL,
      ts                   INTEGER NOT NULL,
      FOREIGN KEY (parent_trade_id) REFERENCES paper_trades(trade_id)
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trades_alert_key "
    "ON paper_trades(triggering_alert_key) "
    "WHERE trade_kind = 'entry' AND triggering_alert_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_open "
    "ON paper_trades(condition_id, asset_id) WHERE trade_kind = 'entry'",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_parent "
    "ON paper_trades(parent_trade_id)",
```

- [ ] **Step 1.2.4: Add `PaperTradesRepo` and supporting types**

Append to `src/pscanner/store/repo.py`:

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


@dataclass(frozen=True, slots=True)
class PaperSummary:
    """Aggregate stats for the ``paper status`` CLI."""

    starting_bankroll: float
    current_nav: float
    total_return_pct: float
    realized_pnl: float
    open_positions: int
    closed_positions: int


class PaperTradesRepo:
    """CRUD + aggregates for the ``paper_trades`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_entry(
        self,
        *,
        triggering_alert_key: str | None,
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
        """Insert an entry row and return its ``trade_id``."""
        cur = self._conn.execute(
            """
            INSERT INTO paper_trades (
              trade_kind, triggering_alert_key, parent_trade_id, source_wallet,
              condition_id, asset_id, outcome, shares, fill_price, cost_usd,
              nav_after_usd, ts
            ) VALUES ('entry', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                triggering_alert_key, source_wallet, condition_id, asset_id,
                outcome, shares, fill_price, cost_usd, nav_after_usd, ts,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def insert_exit(
        self,
        *,
        parent_trade_id: int,
        condition_id: ConditionId,
        asset_id: AssetId,
        outcome: str,
        shares: float,
        fill_price: float,
        cost_usd: float,
        nav_after_usd: float,
        ts: int,
    ) -> int:
        """Insert an exit row linked to ``parent_trade_id``."""
        cur = self._conn.execute(
            """
            INSERT INTO paper_trades (
              trade_kind, triggering_alert_key, parent_trade_id, source_wallet,
              condition_id, asset_id, outcome, shares, fill_price, cost_usd,
              nav_after_usd, ts
            ) VALUES ('exit', NULL, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent_trade_id, condition_id, asset_id, outcome, shares,
                fill_price, cost_usd, nav_after_usd, ts,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def list_open_positions(self) -> list[OpenPaperPosition]:
        """Return entries with no matching exit, oldest first."""
        rows = self._conn.execute(
            """
            SELECT e.trade_id, e.triggering_alert_key, e.source_wallet,
                   e.condition_id, e.asset_id, e.outcome, e.shares,
                   e.fill_price, e.cost_usd, e.nav_after_usd, e.ts
              FROM paper_trades e
             WHERE e.trade_kind = 'entry'
               AND NOT EXISTS (
                 SELECT 1 FROM paper_trades x
                  WHERE x.parent_trade_id = e.trade_id
               )
             ORDER BY e.ts ASC
            """,
        ).fetchall()
        return [
            OpenPaperPosition(
                trade_id=int(r["trade_id"]),
                triggering_alert_key=r["triggering_alert_key"],
                source_wallet=r["source_wallet"],
                condition_id=ConditionId(r["condition_id"]),
                asset_id=AssetId(r["asset_id"]),
                outcome=r["outcome"],
                shares=float(r["shares"]),
                fill_price=float(r["fill_price"]),
                cost_usd=float(r["cost_usd"]),
                nav_after_usd=float(r["nav_after_usd"]),
                ts=int(r["ts"]),
            )
            for r in rows
        ]

    def compute_cost_basis_nav(self, *, starting_bankroll: float) -> float:
        """Return ``starting_bankroll + realized_pnl`` (open positions excluded).

        Realized PnL = Σ(exit.cost_usd − parent_entry.cost_usd) over all
        resolved trades. Open positions sit at cost basis and don't move NAV.
        """
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(x.cost_usd - e.cost_usd), 0.0) AS realized_pnl
              FROM paper_trades x
              JOIN paper_trades e ON e.trade_id = x.parent_trade_id
             WHERE x.trade_kind = 'exit' AND e.trade_kind = 'entry'
            """,
        ).fetchone()
        realized = float(row["realized_pnl"] or 0.0)
        return starting_bankroll + realized

    def summary_stats(self, *, starting_bankroll: float) -> PaperSummary:
        """Aggregate stats for the ``paper status`` CLI."""
        nav = self.compute_cost_basis_nav(starting_bankroll=starting_bankroll)
        realized = nav - starting_bankroll
        open_n = self._conn.execute(
            """
            SELECT COUNT(*) AS n FROM paper_trades e
             WHERE e.trade_kind = 'entry'
               AND NOT EXISTS (
                 SELECT 1 FROM paper_trades x
                  WHERE x.parent_trade_id = e.trade_id
               )
            """,
        ).fetchone()
        closed_n = self._conn.execute(
            "SELECT COUNT(*) AS n FROM paper_trades WHERE trade_kind = 'exit'",
        ).fetchone()
        return PaperSummary(
            starting_bankroll=starting_bankroll,
            current_nav=nav,
            total_return_pct=(realized / starting_bankroll * 100.0) if starting_bankroll else 0.0,
            realized_pnl=realized,
            open_positions=int(open_n["n"]),
            closed_positions=int(closed_n["n"]),
        )
```

- [ ] **Step 1.2.5: Run, verify all paper_trades_repo tests pass**

```bash
uv run pytest tests/store/test_paper_trades_repo.py -v
```

Expected: PASS for all (10 tests).

- [ ] **Step 1.2.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/store/ tests/store/test_paper_trades_repo.py
uv run ruff format src/pscanner/store/ tests/store/test_paper_trades_repo.py
uv run ty check src/pscanner
```

Expected: all clean.

- [ ] **Step 1.2.7: Run full test suite as a regression check**

```bash
uv run pytest -q
```

Expected: all green.

- [ ] **Step 1.2.8: Commit**

```bash
git add src/pscanner/store/ tests/store/
git commit -m "feat(store): paper_trades schema + PaperTradesRepo

Single new table holding both entries and exits, linked via parent_trade_id.
Open positions = entries with no matching exit. NAV = starting + realized
PnL (cost-basis valuation). Adds OpenPaperPosition + PaperSummary value
types and corresponding repo methods.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: PaperTradingConfig

**Files:**
- Modify: `src/pscanner/config.py`
- Modify or create: `tests/test_config.py`

**Wave:** 1 (parallel with T1 — different files). Worktree: `pscanner-worktrees/paper-config`. Branch: `feat/paper-config`.

- [ ] **Step 2.1: Find existing config test file**

```bash
ls tests/test_config.py 2>/dev/null && echo "use existing" || echo "create new"
```

- [ ] **Step 2.2: Write failing tests**

Append to `tests/test_config.py` (or create with module docstring + imports if absent):

```python
from pscanner.config import Config, PaperTradingConfig


def test_paper_trading_defaults() -> None:
    cfg = PaperTradingConfig()
    assert cfg.enabled is False  # opt-in
    assert cfg.starting_bankroll_usd == 1000.0
    assert cfg.position_fraction == 0.01
    assert cfg.min_weighted_edge == 0.0
    assert cfg.min_position_cost_usd == 0.50
    assert cfg.resolver_scan_interval_seconds == 300.0


def test_paper_trading_attached_to_root_config() -> None:
    cfg = Config()
    assert isinstance(cfg.paper_trading, PaperTradingConfig)
    assert cfg.paper_trading.enabled is False
```

- [ ] **Step 2.3: Run, verify they fail**

```bash
uv run pytest -k paper_trading -v
```

Expected: FAIL — `PaperTradingConfig` not importable.

- [ ] **Step 2.4: Implement the config**

In `src/pscanner/config.py`, immediately after the existing `MoveAttributionConfig` class, add:

```python
class PaperTradingConfig(_Section):
    """Thresholds + cadence for the smart-money copy-trade paper strategy.

    Off by default. When enabled, the in-daemon ``PaperTrader`` subscribes
    to ``AlertSink`` and mirrors ``smart_money`` alerts onto a virtual
    bankroll. ``PaperResolver`` runs as a periodic detector that books PnL
    when the underlying market resolves. State lives in ``paper_trades``.
    """

    enabled: bool = False
    starting_bankroll_usd: float = 1000.0
    position_fraction: float = 0.01
    min_weighted_edge: float = 0.0
    min_position_cost_usd: float = 0.50
    resolver_scan_interval_seconds: float = 300.0
```

In the same file on the `Config` class, immediately after the
`move_attribution` field, add:

```python
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
```

- [ ] **Step 2.5: Run, verify they pass**

```bash
uv run pytest -k paper_trading -v
```

Expected: PASS for both new tests.

- [ ] **Step 2.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/config.py tests/test_config.py
uv run ruff format src/pscanner/config.py tests/test_config.py
uv run ty check src/pscanner/config.py
```

- [ ] **Step 2.7: Commit**

```bash
git add src/pscanner/config.py tests/test_config.py
git commit -m "feat(config): PaperTradingConfig with spec defaults

Off by default. Wired into root Config alongside the existing detector
configs. Defaults match docs/superpowers/specs/2026-04-27-paper-trading-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: PaperTrader (alert-driven subscriber)

The class itself, including `_resolve_outcome` and `_size_trade` helpers as private methods.

**Files:**
- Create: `src/pscanner/strategies/__init__.py` (empty package marker)
- Create: `src/pscanner/strategies/paper_trader.py`
- Create: `tests/strategies/__init__.py`
- Create: `tests/strategies/test_paper_trader.py`

**Wave:** 2 (parallel with T4 — different files). **Depends on T1+T2 being merged to main.** Worktree: `pscanner-worktrees/paper-trader`. Branch: `feat/paper-trader`.

- [ ] **Step 3.1: Create the namespace markers**

```bash
mkdir -p src/pscanner/strategies tests/strategies
echo '"""Strategies (paper-trade and beyond)."""' > src/pscanner/strategies/__init__.py
echo '' > tests/strategies/__init__.py
```

- [ ] **Step 3.2: Write failing tests**

Create `tests/strategies/test_paper_trader.py`:

```python
"""Tests for PaperTrader."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    PaperTradesRepo,
    TrackedWallet,
    TrackedWalletsRepo,
)
from pscanner.strategies.paper_trader import (
    PaperTrader,
    _size_trade,
)

_NOW = 1700000000


def _smart_money_alert(
    *,
    wallet: str = "0xwallet1",
    condition_id: str = "0xcond-1",
    side: str = "yes",
    delta_usd: float = 100.0,
    alert_key: str = "smart:0xwallet1:0xcond-1:yes:20260427",
) -> Alert:
    return Alert(
        detector="smart_money",
        alert_key=alert_key,
        severity="med",
        title=f"smart-money {wallet[:8]} +{side}",
        body={
            "wallet": wallet,
            "market_title": "Test market",
            "condition_id": condition_id,
            "side": side,
            "new_size": 200.0,
            "prev_size": 100.0,
            "delta_usd": delta_usd,
            "winrate": 0.85,
            "mean_edge": 0.4,
            "excess_pnl_usd": 1000.0,
            "closed_position_count": 50,
        },
        created_at=_NOW,
    )


def _track_wallet(
    repo: TrackedWalletsRepo,
    *,
    address: str = "0xwallet1",
    weighted_edge: float | None = 0.4,
) -> None:
    """Upsert a tracked wallet with a given edge."""
    repo.upsert(
        TrackedWallet(
            address=address,
            closed_position_count=50,
            closed_position_wins=42,
            winrate=0.84,
            leaderboard_pnl=1000.0,
            mean_edge=0.4,
            weighted_edge=weighted_edge,
            excess_pnl_usd=1000.0,
            total_stake_usd=1000.0,
            last_refreshed_at=_NOW,
        ),
    )


def _cache_market(
    repo: MarketCacheRepo,
    *,
    condition_id: str = "0xcond-1",
    outcomes: list[str] | None = None,
    asset_ids: list[str] | None = None,
) -> None:
    repo.upsert(
        CachedMarket(
            market_id=MarketId(f"mkt-{condition_id}"),
            event_id=None,
            title="Test market",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.6, 0.4],
            outcomes=outcomes or ["Yes", "No"],
            asset_ids=[AssetId(a) for a in (asset_ids or ["asset-yes", "asset-no"])],
            active=True,
            cached_at=_NOW,
            condition_id=ConditionId(condition_id),
            event_slug=None,
        ),
    )


def _seed_tick(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    best_ask: float | None,
    last_trade_price: float | None = None,
    ts: int = _NOW,
) -> None:
    """Insert one row into market_ticks."""
    conn.execute(
        """
        INSERT INTO market_ticks (asset_id, condition_id, snapshot_at, mid_price,
          best_bid, best_ask, spread, bid_depth_top5, ask_depth_top5,
          last_trade_price)
        VALUES (?, '0xcond-1', ?, NULL, NULL, ?, NULL, NULL, NULL, ?)
        """,
        (asset_id, ts, best_ask, last_trade_price),
    )
    conn.commit()


def test_size_trade_happy_path() -> None:
    cfg = PaperTradingConfig(
        enabled=True, starting_bankroll_usd=1000.0,
        position_fraction=0.01, min_position_cost_usd=0.5,
    )
    result = _size_trade(nav=1000.0, fill_price=0.5, cfg=cfg)
    assert result is not None
    cost, shares = result
    assert cost == 10.0
    assert shares == 20.0


def test_size_trade_below_minimum_returns_none() -> None:
    cfg = PaperTradingConfig(min_position_cost_usd=0.50, position_fraction=0.01)
    # NAV $40 × 1% = $0.40 < $0.50 minimum
    assert _size_trade(nav=40.0, fill_price=0.5, cfg=cfg) is None


def test_size_trade_bad_fill_price_returns_none() -> None:
    cfg = PaperTradingConfig()
    assert _size_trade(nav=1000.0, fill_price=0.0, cfg=cfg) is None
    assert _size_trade(nav=1000.0, fill_price=1.0, cfg=cfg) is None
    assert _size_trade(nav=1000.0, fill_price=-0.1, cfg=cfg) is None
    assert _size_trade(nav=1000.0, fill_price=1.5, cfg=cfg) is None


@pytest.mark.asyncio
async def test_paper_trader_inserts_entry_on_smart_money_alert(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)

    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    trader = PaperTrader(
        config=cfg,
        market_cache=cache,
        tracked_wallets=wallets,
        paper_trades=paper,
        conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()

    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    p = open_positions[0]
    assert p.source_wallet == "0xwallet1"
    assert p.outcome == "yes"
    assert p.asset_id == AssetId("asset-yes")
    assert p.fill_price == 0.5
    assert p.cost_usd == 10.0  # 1000 × 0.01
    assert p.shares == 20.0   # 10 / 0.5


@pytest.mark.asyncio
async def test_paper_trader_skips_non_smart_money(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    paper = PaperTradesRepo(tmp_db)
    trader = PaperTrader(
        config=cfg,
        market_cache=MarketCacheRepo(tmp_db),
        tracked_wallets=TrackedWalletsRepo(tmp_db),
        paper_trades=paper,
        conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(Alert(
        detector="velocity", alert_key="v:1", severity="med", title="t",
        body={"condition_id": "0xc"}, created_at=_NOW,
    ))
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_paper_trader_skips_wallet_below_edge(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True, min_weighted_edge=0.0)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=-0.1)  # below threshold
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_paper_trader_skips_null_edge(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=None)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_paper_trader_skips_when_no_market_cache(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)  # empty
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=0.4)
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_paper_trader_skips_when_outcome_unmappable(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache, outcomes=["Yes", "No"], asset_ids=["a-y", "a-n"])
    _seed_tick(tmp_db, asset_id="a-y", best_ask=0.5)
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    # Body says "side": "Maybe" — not in cached outcomes
    bad = _smart_money_alert(side="Maybe")
    await sink.emit(bad)
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_paper_trader_skips_when_no_price(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    # No tick row at all
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_paper_trader_falls_back_to_last_trade_price(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    # best_ask null but last_trade_price present
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=None, last_trade_price=0.55)
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].fill_price == 0.55


@pytest.mark.asyncio
async def test_paper_trader_idempotent_on_duplicate_alert_key(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = PaperTrader(
        config=cfg, market_cache=cache, tracked_wallets=wallets,
        paper_trades=paper, conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)

    alert = _smart_money_alert(alert_key="dup-key")
    await sink.emit(alert)
    # AlertSink dedups by alert_key, so the second emit is a no-op at sink level.
    # Manually call handle_alert_sync a second time to exercise the
    # paper_trader-side UNIQUE-violation path.
    trader.handle_alert_sync(alert)
    for _ in range(10):
        await asyncio.sleep(0)
    await trader.aclose()
    assert len(paper.list_open_positions()) == 1
```

- [ ] **Step 3.3: Run, verify they fail**

```bash
uv run pytest tests/strategies/test_paper_trader.py -v
```

Expected: FAIL — `pscanner.strategies.paper_trader` does not exist.

- [ ] **Step 3.4: Implement `PaperTrader`**

Create `src/pscanner/strategies/paper_trader.py`:

```python
"""Smart-money copy-trade paper-trading subscriber.

Subscribes to ``AlertSink``. Filters to ``smart_money`` alerts whose source
wallet has positive ``weighted_edge``. Resolves the alerted outcome to an
``asset_id`` via ``MarketCacheRepo`` and a fill price via ``market_ticks``.
Sizes trades at ``cfg.position_fraction`` of cost-basis NAV. Inserts an
``entry`` row into ``paper_trades``.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import (
    MarketCacheRepo,
    PaperTradesRepo,
    TrackedWalletsRepo,
)

_LOG = structlog.get_logger(__name__)

_FILL_PRICE_LO = 0.0
_FILL_PRICE_HI = 1.0


def _size_trade(
    *,
    nav: float,
    fill_price: float,
    cfg: PaperTradingConfig,
) -> tuple[float, float] | None:
    """Return ``(cost_usd, shares)`` or ``None`` if the trade can't be sized.

    Returns ``None`` when the computed cost falls below ``min_position_cost_usd``,
    or when ``fill_price`` is outside ``(0, 1)``.
    """
    if not (_FILL_PRICE_LO < fill_price < _FILL_PRICE_HI):
        return None
    cost = nav * cfg.position_fraction
    if cost < cfg.min_position_cost_usd:
        return None
    shares = cost / fill_price
    return (cost, shares)


class PaperTrader:
    """Alert-driven paper-trading subscriber."""

    name = "paper_trader"

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        market_cache: MarketCacheRepo,
        tracked_wallets: TrackedWalletsRepo,
        paper_trades: PaperTradesRepo,
        conn: sqlite3.Connection,
    ) -> None:
        self._config = config
        self._market_cache = market_cache
        self._tracked_wallets = tracked_wallets
        self._paper_trades = paper_trades
        self._conn = conn
        self._pending_tasks: set[asyncio.Task[None]] = set()

    def handle_alert_sync(self, alert: Alert) -> None:
        """``AlertSink.subscribe`` callback. Spawns evaluate as a tracked task."""
        if alert.detector != "smart_money":
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("paper_trader.no_event_loop", alert_key=alert.alert_key)
            return
        task = loop.create_task(self.evaluate(alert))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate(self, alert: Alert) -> None:
        """Run the entry pipeline for one smart-money alert."""
        body = alert.body if isinstance(alert.body, dict) else {}
        wallet = body.get("wallet")
        condition_id_str = body.get("condition_id")
        side = body.get("side")
        if not (isinstance(wallet, str) and isinstance(condition_id_str, str) and isinstance(side, str)):
            _LOG.debug("paper_trader.bad_body", alert_key=alert.alert_key)
            return
        if not self._wallet_passes_edge_filter(wallet):
            return
        cond = ConditionId(condition_id_str)
        resolved = self._resolve_outcome(cond, side)
        if resolved is None:
            return
        asset_id, fill_price = resolved
        nav = self._paper_trades.compute_cost_basis_nav(
            starting_bankroll=self._config.starting_bankroll_usd,
        )
        if nav <= 0:
            _LOG.info("paper_trade.bankroll_exhausted", alert_key=alert.alert_key, nav=nav)
            return
        sized = _size_trade(nav=nav, fill_price=fill_price, cfg=self._config)
        if sized is None:
            _LOG.debug(
                "paper_trade.size_too_small_or_bad_price",
                alert_key=alert.alert_key, nav=nav, fill_price=fill_price,
            )
            return
        cost_usd, shares = sized
        try:
            self._paper_trades.insert_entry(
                triggering_alert_key=alert.alert_key,
                source_wallet=wallet,
                condition_id=cond,
                asset_id=asset_id,
                outcome=side,
                shares=shares,
                fill_price=fill_price,
                cost_usd=cost_usd,
                nav_after_usd=nav - cost_usd,
                ts=int(time.time()),
            )
        except sqlite3.IntegrityError:
            _LOG.debug("paper_trader.duplicate_alert", alert_key=alert.alert_key)
        except Exception:
            _LOG.warning("paper_trader.insert_failed", alert_key=alert.alert_key, exc_info=True)

    def _wallet_passes_edge_filter(self, wallet: str) -> bool:
        """Skip wallets whose ``weighted_edge`` is None or ≤ ``min_weighted_edge``."""
        tracked = self._tracked_wallets.get(wallet)
        if tracked is None:
            _LOG.debug("paper_trader.no_edge", wallet=wallet)
            return False
        edge = tracked.weighted_edge
        if edge is None or edge <= self._config.min_weighted_edge:
            _LOG.debug("paper_trader.below_edge", wallet=wallet, edge=edge)
            return False
        return True

    def _resolve_outcome(
        self,
        condition_id: ConditionId,
        side: str,
    ) -> tuple[AssetId, float] | None:
        """Map ``side`` (outcome name) to ``(asset_id, fill_price)``.

        Returns ``None`` when the market is not cached, the outcome name is
        not in the cached outcomes, no price is available, or the price is
        outside ``(0, 1)``.
        """
        asset_id = self._market_cache.outcome_to_asset(condition_id, side)
        if asset_id is None:
            _LOG.warning(
                "paper_trade.outcome_unmappable",
                condition_id=condition_id, side=side,
            )
            return None
        row = self._conn.execute(
            """
            SELECT best_ask, last_trade_price FROM market_ticks
             WHERE asset_id = ?
             ORDER BY snapshot_at DESC
             LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        if row is None:
            _LOG.warning("paper_trade.no_price", asset_id=asset_id)
            return None
        best_ask = row["best_ask"]
        last_trade = row["last_trade_price"]
        fill_price: float | None = None
        if isinstance(best_ask, (int, float)) and 0 < best_ask < 1:
            fill_price = float(best_ask)
        elif isinstance(last_trade, (int, float)) and 0 < last_trade < 1:
            fill_price = float(last_trade)
        if fill_price is None:
            _LOG.warning(
                "paper_trade.no_price",
                asset_id=asset_id, best_ask=best_ask, last_trade=last_trade,
            )
            return None
        return (asset_id, fill_price)

    async def aclose(self) -> None:
        """Wait for any in-flight evaluation tasks (test helper)."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)


__all__ = ["PaperTrader", "_size_trade"]
```

- [ ] **Step 3.5: Run, verify all tests pass**

```bash
uv run pytest tests/strategies/test_paper_trader.py -v
```

Expected: PASS for all 11 tests.

- [ ] **Step 3.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/strategies/ tests/strategies/test_paper_trader.py
uv run ruff format src/pscanner/strategies/ tests/strategies/test_paper_trader.py
uv run ty check src/pscanner
```

Note: if `ruff check` flags magic numbers (PLR2004) for `0`, `1`, etc., extract module constants (`_BANKROLL_FLOOR = 0.0` etc.) and re-run.

- [ ] **Step 3.7: Commit**

```bash
git add src/pscanner/strategies/__init__.py src/pscanner/strategies/paper_trader.py tests/strategies/__init__.py tests/strategies/test_paper_trader.py
git commit -m "feat(strategies): PaperTrader subscriber for smart-money copy-trade

Subscribes to AlertSink. Filters to smart_money alerts whose source wallet
has positive weighted_edge. Resolves outcome name → asset_id via
MarketCacheRepo, looks up fill price (best_ask, fall back to last_trade),
sizes at cost-basis NAV × position_fraction, inserts entry row into
paper_trades. Fail-soft on every external lookup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: PaperResolver (periodic detector)

A `PollingDetector` subclass that scans open positions and books exits when the underlying market is resolved.

**Files:**
- Create: `src/pscanner/strategies/paper_resolver.py`
- Create: `tests/strategies/test_paper_resolver.py`

**Wave:** 2 (parallel with T3 — different files). **Depends on T1+T2 being merged to main.** Worktree: `pscanner-worktrees/paper-resolver`. Branch: `feat/paper-resolver`.

- [ ] **Step 4.1: Write failing tests**

Create `tests/strategies/test_paper_resolver.py`:

```python
"""Tests for PaperResolver."""
from __future__ import annotations

import sqlite3

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    PaperTradesRepo,
)
from pscanner.strategies.paper_resolver import (
    PaperResolver,
    _check_resolution,
    _compute_payout,
)
from pscanner.util.clock import FakeClock

_NOW = 1700000000


def _cache_market(
    repo: MarketCacheRepo,
    *,
    condition_id: str = "0xcond-1",
    active: bool = True,
    outcome_prices: list[float] | None = None,
    outcomes: list[str] | None = None,
    asset_ids: list[str] | None = None,
) -> None:
    repo.upsert(
        CachedMarket(
            market_id=MarketId(f"mkt-{condition_id}"),
            event_id=None,
            title="Test market",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=outcome_prices or [0.5, 0.5],
            outcomes=outcomes or ["Yes", "No"],
            asset_ids=[AssetId(a) for a in (asset_ids or ["asset-yes", "asset-no"])],
            active=active,
            cached_at=_NOW,
            condition_id=ConditionId(condition_id),
            event_slug=None,
        ),
    )


def _open_position(
    repo: PaperTradesRepo,
    *,
    condition_id: str = "0xcond-1",
    asset_id: str = "asset-yes",
    outcome: str = "yes",
    cost_usd: float = 10.0,
    shares: float = 20.0,
    fill_price: float = 0.5,
) -> int:
    return repo.insert_entry(
        triggering_alert_key=f"k-{condition_id}-{outcome}",
        source_wallet="0xw1",
        condition_id=ConditionId(condition_id),
        asset_id=AssetId(asset_id),
        outcome=outcome,
        shares=shares,
        fill_price=fill_price,
        cost_usd=cost_usd,
        nav_after_usd=1000.0 - cost_usd,
        ts=_NOW,
    )


def test_check_resolution_active_market_returns_none(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    assert _check_resolution(cache, ConditionId("0xcond-1")) is None


def test_check_resolution_yes_won(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    res = _check_resolution(cache, ConditionId("0xcond-1"))
    assert res == AssetId("asset-yes")


def test_check_resolution_no_won(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[0.0, 1.0])
    res = _check_resolution(cache, ConditionId("0xcond-1"))
    assert res == AssetId("asset-no")


def test_check_resolution_ambiguous_outcomes_returns_none(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[0.5, 0.5])
    assert _check_resolution(cache, ConditionId("0xcond-1")) is None


def test_check_resolution_market_missing_returns_none(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    assert _check_resolution(cache, ConditionId("0xnope")) is None


def test_compute_payout_winner() -> None:
    assert _compute_payout(
        position_asset_id=AssetId("asset-yes"),
        winning_asset_id=AssetId("asset-yes"),
    ) == 1.0


def test_compute_payout_loser() -> None:
    assert _compute_payout(
        position_asset_id=AssetId("asset-yes"),
        winning_asset_id=AssetId("asset-no"),
    ) == 0.0


@pytest.mark.asyncio
async def test_resolver_books_winning_exit(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    sink = AlertSink(AlertsRepo(tmp_db))
    clock = FakeClock(start_time=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg, market_cache=cache, paper_trades=paper, clock=clock,
    )
    await resolver._scan(sink)  # noqa: SLF001 — driving one iteration directly
    assert paper.list_open_positions() == []
    nav = paper.compute_cost_basis_nav(starting_bankroll=1000.0)
    assert nav == 1010.0  # 1000 + (20 - 10)


@pytest.mark.asyncio
async def test_resolver_books_losing_exit(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[0.0, 1.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    sink = AlertSink(AlertsRepo(tmp_db))
    clock = FakeClock(start_time=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg, market_cache=cache, paper_trades=paper, clock=clock,
    )
    await resolver._scan(sink)  # noqa: SLF001
    assert paper.list_open_positions() == []
    nav = paper.compute_cost_basis_nav(starting_bankroll=1000.0)
    assert nav == 990.0


@pytest.mark.asyncio
async def test_resolver_skips_unresolved(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    _open_position(paper)
    clock = FakeClock(start_time=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg, market_cache=cache, paper_trades=paper, clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))  # noqa: SLF001
    assert len(paper.list_open_positions()) == 1


@pytest.mark.asyncio
async def test_resolver_books_multiple_in_one_scan(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(
        cache, condition_id="0xcond-1",
        active=False, outcome_prices=[1.0, 0.0],
        asset_ids=["a-y1", "a-n1"],
    )
    _cache_market(
        cache, condition_id="0xcond-2",
        active=False, outcome_prices=[0.0, 1.0],
        asset_ids=["a-y2", "a-n2"],
    )
    _open_position(paper, condition_id="0xcond-1", asset_id="a-y1", outcome="yes")
    _open_position(paper, condition_id="0xcond-2", asset_id="a-y2", outcome="yes")
    clock = FakeClock(start_time=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg, market_cache=cache, paper_trades=paper, clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))  # noqa: SLF001
    assert paper.list_open_positions() == []


def test_resolver_interval_from_config(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True, resolver_scan_interval_seconds=120.0)
    resolver = PaperResolver(
        config=cfg,
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=PaperTradesRepo(tmp_db),
    )
    assert resolver._interval_seconds() == 120.0  # noqa: SLF001
```

- [ ] **Step 4.2: Run, verify they fail**

```bash
uv run pytest tests/strategies/test_paper_resolver.py -v
```

Expected: FAIL — `pscanner.strategies.paper_resolver` does not exist.

- [ ] **Step 4.3: Implement `PaperResolver`**

Create `src/pscanner/strategies/paper_resolver.py`:

```python
"""PaperResolver — periodic detector that books PnL on resolved markets.

Inherits ``PollingDetector``. Each scan: list open positions, check each
position's market in the cache for a definitive ``[1, 0]`` / ``[0, 1]``
outcome split, insert an exit row that books realized PnL.
"""
from __future__ import annotations

import structlog

from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.detectors.polling import PollingDetector
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import MarketCacheRepo, OpenPaperPosition, PaperTradesRepo
from pscanner.util.clock import Clock

_LOG = structlog.get_logger(__name__)

_DEFINITIVE = 1.0
_ZERO = 0.0


def _check_resolution(
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
) -> AssetId | None:
    """Return the winning ``AssetId`` if the market has resolved, else None.

    A market is considered "resolved" when ``active=False`` AND its
    ``outcome_prices`` is a clean ``[1.0, 0.0]`` or ``[0.0, 1.0]`` split.
    """
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None or cached.active:
        return None
    prices = cached.outcome_prices
    if len(prices) != len(cached.asset_ids):
        return None
    for price, asset_id in zip(prices, cached.asset_ids, strict=True):
        if price == _DEFINITIVE:
            other = sum(p for p in prices if p is not None)
            if other == _DEFINITIVE:  # exactly one outcome at 1.0 — definitive
                return asset_id
    return None


def _compute_payout(
    *,
    position_asset_id: AssetId,
    winning_asset_id: AssetId,
) -> float:
    """Return ``1.0`` if our outcome won, ``0.0`` otherwise."""
    return _DEFINITIVE if position_asset_id == winning_asset_id else _ZERO


class PaperResolver(PollingDetector):
    """Books exits on open paper positions whose markets have resolved."""

    name = "paper_resolver"

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        market_cache: MarketCacheRepo,
        paper_trades: PaperTradesRepo,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(clock=clock)
        self._config = config
        self._market_cache = market_cache
        self._paper_trades = paper_trades

    def _interval_seconds(self) -> float:
        return self._config.resolver_scan_interval_seconds

    async def _scan(self, sink: AlertSink) -> None:
        """Walk open positions; insert exit rows for any that resolved.

        Errors on individual positions are logged and skipped — one bad row
        never blocks the rest. The polling-loop superclass calls this on
        every interval; ``sink`` is unused (this detector doesn't emit alerts).
        """
        del sink  # contract: _scan accepts a sink; we don't emit
        booked = 0
        for pos in self._paper_trades.list_open_positions():
            if self._maybe_book_exit(pos):
                booked += 1
        if booked:
            _LOG.info("paper_resolver.scan_completed", booked=booked)

    def _maybe_book_exit(self, pos: OpenPaperPosition) -> bool:
        """Check resolution for one position; insert exit if resolved.

        Returns True iff an exit row was written. Per-position try/except
        isolates a bad row from blocking the rest of the scan.
        """
        try:
            winning = _check_resolution(self._market_cache, pos.condition_id)
            if winning is None:
                return False
            payout_per_share = _compute_payout(
                position_asset_id=pos.asset_id,
                winning_asset_id=winning,
            )
            proceeds = pos.shares * payout_per_share
            nav_before = self._paper_trades.compute_cost_basis_nav(
                starting_bankroll=self._config.starting_bankroll_usd,
            )
            self._paper_trades.insert_exit(
                parent_trade_id=pos.trade_id,
                condition_id=pos.condition_id,
                asset_id=pos.asset_id,
                outcome=pos.outcome,
                shares=pos.shares,
                fill_price=payout_per_share,
                cost_usd=proceeds,
                nav_after_usd=nav_before + (proceeds - pos.cost_usd),
                ts=int(self._clock.now()),
            )
        except Exception:
            _LOG.warning(
                "paper_resolver.insert_failed",
                trade_id=pos.trade_id, exc_info=True,
            )
            return False
        return True


__all__ = ["PaperResolver", "_check_resolution", "_compute_payout"]
```

`PollingDetector` lives at `src/pscanner/detectors/polling.py`. Its actual
contract is: subclass overrides `async def _scan(self, sink)` and
`def _interval_seconds(self) -> float`; the base owns the polling loop and
exception handling.

- [ ] **Step 4.4: Run tests, verify pass**

```bash
uv run pytest tests/strategies/test_paper_resolver.py -v
```

Expected: PASS for all 11 tests.

- [ ] **Step 4.5: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/strategies/paper_resolver.py tests/strategies/test_paper_resolver.py
uv run ruff format src/pscanner/strategies/paper_resolver.py tests/strategies/test_paper_resolver.py
uv run ty check src/pscanner
```

- [ ] **Step 4.6: Commit**

```bash
git add src/pscanner/strategies/paper_resolver.py tests/strategies/test_paper_resolver.py
git commit -m "feat(strategies): PaperResolver — books exits on resolved markets

Periodic detector. Each scan: list open paper positions, check each
position's market_cache row for a definitive [1,0]/[0,1] outcome split,
insert exit row with payout 1.0 (winner) or 0.0 (loser). Fail-soft per
position — a bad row never blocks the rest.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Scheduler wiring + integration smoke test

Wire `PaperTrader` (alert subscriber) and `PaperResolver` (polling detector) into `Scheduler`, gated on `config.paper_trading.enabled`. Add an end-to-end smoke test that exercises the full chain.

**Files:**
- Modify: `src/pscanner/scheduler.py`
- Modify: `tests/test_scheduler.py`

**Wave:** 3 (parallel with T6 — different files). **Depends on T1+T2+T3+T4 being merged to main.** Worktree: `pscanner-worktrees/paper-scheduler`. Branch: `feat/paper-scheduler`.

- [ ] **Step 5.1: Write failing integration smoke test**

Append to `tests/test_scheduler.py`:

```python
import pytest

from pscanner.alerts.models import Alert
from pscanner.config import Config


@pytest.mark.asyncio
async def test_scanner_wires_paper_trader_to_alert_sink_when_enabled(tmp_path) -> None:
    """End-to-end: smart_money alert through the live AlertSink reaches
    PaperTrader, which inserts an entry in paper_trades. Then a manual
    flip of market_cache.active=False with a [1,0] outcome split + a
    resolver scan books the exit and updates NAV."""
    import asyncio
    import sqlite3

    from pscanner.poly.ids import AssetId, ConditionId, MarketId
    from pscanner.scheduler import Scanner
    from pscanner.store.db import init_db
    from pscanner.store.repo import (
        AlertsRepo, CachedMarket, MarketCacheRepo, PaperTradesRepo,
        TrackedWallet, TrackedWalletsRepo,
    )

    db_path = tmp_path / "pscanner.sqlite3"
    cfg = Config().model_copy(update={
        "scanner": Config().scanner.model_copy(update={"db_path": db_path}),
        "paper_trading": Config().paper_trading.model_copy(update={"enabled": True}),
    })

    # Pre-seed the DB with a tracked wallet, cached market, and a tick.
    conn = init_db(db_path)
    try:
        TrackedWalletsRepo(conn).upsert(TrackedWallet(
            address="0xwallet1", closed_position_count=50,
            closed_position_wins=42, winrate=0.84, leaderboard_pnl=1000.0,
            mean_edge=0.4, weighted_edge=0.4, excess_pnl_usd=1000.0,
            total_stake_usd=1000.0, last_refreshed_at=1700000000,
        ))
        MarketCacheRepo(conn).upsert(CachedMarket(
            market_id=MarketId("mkt-1"), event_id=None, title="t",
            liquidity_usd=1.0, volume_usd=1.0, outcome_prices=[0.6, 0.4],
            outcomes=["Yes", "No"],
            asset_ids=[AssetId("asset-yes"), AssetId("asset-no")],
            active=True, cached_at=1700000000,
            condition_id=ConditionId("0xcond-1"), event_slug=None,
        ))
        conn.execute(
            """
            INSERT INTO market_ticks (asset_id, condition_id, snapshot_at,
              mid_price, best_bid, best_ask, spread, bid_depth_top5,
              ask_depth_top5, last_trade_price)
            VALUES ('asset-yes', '0xcond-1', 1700000000, NULL, NULL, 0.5,
              NULL, NULL, NULL, NULL)
            """,
        )
        conn.commit()
    finally:
        conn.close()

    scanner = Scanner(config=cfg)
    try:
        sink = scanner.sink
        await sink.emit(Alert(
            detector="smart_money",
            alert_key="smart:0xwallet1:0xcond-1:Yes:smoke",
            severity="med",
            title="t",
            body={
                "wallet": "0xwallet1",
                "condition_id": "0xcond-1",
                "side": "Yes",
                "delta_usd": 100.0,
            },
            created_at=1700000000,
        ))
        for _ in range(10):
            await asyncio.sleep(0)
    finally:
        await scanner.aclose()

    # Verify the entry row was written.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE trade_kind='entry'"
        ).fetchall()
        assert len(rows) == 1
        entry = rows[0]
        assert entry["source_wallet"] == "0xwallet1"
        assert entry["outcome"] == "Yes"
        assert entry["fill_price"] == 0.5
        assert entry["cost_usd"] == 10.0  # 1000 × 1%

        # Now resolve the market and run the resolver once.
        conn.execute(
            "UPDATE market_cache SET active = 0, "
            "outcome_prices_json = '[1.0, 0.0]' WHERE condition_id = '0xcond-1'"
        )
        conn.commit()
    finally:
        conn.close()

    # Run the resolver path manually (the scheduler's poll loop already wired it).
    from pscanner.store.repo import PaperTradesRepo, MarketCacheRepo
    from pscanner.strategies.paper_resolver import PaperResolver
    from pscanner.alerts.sink import AlertSink
    from pscanner.store.repo import AlertsRepo as _AlertsRepo

    conn = init_db(db_path)
    try:
        resolver = PaperResolver(
            config=cfg.paper_trading,
            market_cache=MarketCacheRepo(conn),
            paper_trades=PaperTradesRepo(conn),
        )
        await resolver._scan(AlertSink(_AlertsRepo(conn)))  # noqa: SLF001
        assert PaperTradesRepo(conn).list_open_positions() == []
        nav = PaperTradesRepo(conn).compute_cost_basis_nav(
            starting_bankroll=cfg.paper_trading.starting_bankroll_usd,
        )
        assert nav == 1010.0  # 1000 + (20 shares × $1.0 - $10 cost)
    finally:
        conn.close()
```

- [ ] **Step 5.2: Run, verify it fails**

```bash
uv run pytest tests/test_scheduler.py -v -k paper_trader
```

Expected: FAIL — Scanner does not yet construct PaperTrader/PaperResolver.

- [ ] **Step 5.3: Wire PaperTrader and PaperResolver into Scanner**

In `src/pscanner/scheduler.py`, add the imports near the existing detector imports:

```python
from pscanner.strategies.paper_trader import PaperTrader
from pscanner.strategies.paper_resolver import PaperResolver
```

In `Scanner._build_detectors`, after the existing `move_attribution` block, add:

```python
        if self._config.paper_trading.enabled:
            detectors["paper_trader"] = PaperTrader(
                config=self._config.paper_trading,
                market_cache=self._market_cache_repo,
                tracked_wallets=self._tracked_repo,
                paper_trades=PaperTradesRepo(self._conn),
                conn=self._conn,
            )
            detectors["paper_resolver"] = PaperResolver(
                config=self._config.paper_trading,
                market_cache=self._market_cache_repo,
                paper_trades=PaperTradesRepo(self._conn),
                clock=self._clock,
            )
```

(Add `from pscanner.store.repo import PaperTradesRepo` to the import block.)

In `Scanner._wire_alert_subscribers`, extend the `isinstance` check to include `PaperTrader`:

```python
    def _wire_alert_subscribers(self) -> None:
        for detector in self._detectors.values():
            if isinstance(detector, MoveAttributionDetector | PaperTrader):
                detector._sink = self._sink
                self._sink.subscribe(detector.handle_alert_sync)
                _LOG.info("scanner.alert_driven_detector_wired", detector=detector.name)
```

(`PaperTrader.handle_alert_sync` matches the same shape as `MoveAttributionDetector`'s; if `PaperTrader` doesn't track `_sink` because it doesn't emit alerts, the `_sink` assignment is a no-op — keep it for symmetry.)

- [ ] **Step 5.4: Run, verify it passes**

```bash
uv run pytest tests/test_scheduler.py -v -k paper_trader
```

Expected: PASS.

- [ ] **Step 5.5: Run full suite**

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

Expected: all green.

- [ ] **Step 5.6: Commit**

```bash
git add src/pscanner/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): wire PaperTrader subscriber + PaperResolver poll loop

Gated on config.paper_trading.enabled. Mirrors the move_attribution wiring
pattern for the alert subscriber. End-to-end smoke test verifies the chain
from a smart_money alert through a paper_trades entry, then through
resolution to a booked exit and updated NAV.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `pscanner paper status` CLI

Add a new subcommand under the existing `pscanner` CLI that reads `paper_trades` and prints a summary plus best/worst settled trades and a per-wallet leaderboard.

**Files:**
- Modify: `src/pscanner/cli.py`
- Modify: `tests/test_cli.py`

**Wave:** 3 (parallel with T5 — different files). **Depends on T1 and T2.** Worktree: `pscanner-worktrees/paper-cli`. Branch: `feat/paper-cli`.

- [ ] **Step 6.1: Locate the existing CLI dispatch pattern**

Read `src/pscanner/cli.py` to find:

- `_build_parser()` — argparse setup
- `_dispatch_command()` — switch on subcommand name
- existing `_cmd_status` / `_cmd_watch` etc. for the canonical command shape

Use whichever shape matches.

- [ ] **Step 6.2: Write failing test**

Append to `tests/test_cli.py`:

```python
import sqlite3
from pathlib import Path

import pytest

from pscanner.cli import main as cli_main


def _seed_paper_trades(db_path: Path) -> None:
    """Seed two entries (one resolved win, one open) so the CLI has data."""
    from pscanner.store.db import init_db
    from pscanner.store.repo import PaperTradesRepo
    from pscanner.poly.ids import AssetId, ConditionId

    conn = init_db(db_path)
    try:
        repo = PaperTradesRepo(conn)
        p1 = repo.insert_entry(
            triggering_alert_key="smart:0xa:0xc:yes:1",
            source_wallet="0xa",
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=0.5,
            cost_usd=10.0,
            nav_after_usd=990.0,
            ts=1700000000,
        )
        repo.insert_entry(
            triggering_alert_key="smart:0xb:0xc:no:2",
            source_wallet="0xb",
            condition_id=ConditionId("0xc2"),
            asset_id=AssetId("a-n"),
            outcome="no",
            shares=20.0,
            fill_price=0.5,
            cost_usd=10.0,
            nav_after_usd=980.0,
            ts=1700000010,
        )
        repo.insert_exit(
            parent_trade_id=p1,
            condition_id=ConditionId("0xc1"),
            asset_id=AssetId("a-y"),
            outcome="yes",
            shares=20.0,
            fill_price=1.0,
            cost_usd=20.0,
            nav_after_usd=1000.0,
            ts=1700000100,
        )
    finally:
        conn.close()


def test_paper_status_renders_summary(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "pscanner.sqlite3"
    _seed_paper_trades(db_path)
    rc = cli_main([
        "--db-path", str(db_path),
        "paper", "status",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Headline numbers
    assert "starting bankroll" in out.lower()
    assert "1010" in out or "1,010" in out  # current NAV after a $10 win
    # Counts
    assert "1 open" in out or "open: 1" in out.lower() or "open positions" in out.lower()
    # Realized PnL line
    assert "realized" in out.lower()
    assert "10" in out  # a $10 realized PnL
    # Per-wallet leaderboard mentions both wallets
    assert "0xa" in out
    assert "0xb" in out


def test_paper_status_empty_db(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "empty.sqlite3"
    from pscanner.store.db import init_db
    init_db(db_path).close()
    rc = cli_main([
        "--db-path", str(db_path),
        "paper", "status",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0 open" in out or "open: 0" in out.lower() or "open positions" in out.lower()
```

(If `cli_main` doesn't accept `argv` directly, locate the main entry function in `cli.py` and adjust. The point is: invoking the CLI with `paper status` against a tmp DB succeeds.)

- [ ] **Step 6.3: Run, verify the test fails**

```bash
uv run pytest tests/test_cli.py -v -k paper_status
```

Expected: FAIL — `paper` subcommand does not exist.

- [ ] **Step 6.4: Add the `paper status` subcommand**

In `src/pscanner/cli.py`:

1. In `_build_parser`, after the existing subparsers, add:

```python
    paper = sub.add_parser("paper", help="Paper-trading commands.")
    paper_sub = paper.add_subparsers(dest="paper_cmd", required=True)
    paper_sub.add_parser("status", help="Print paper-trading status.")
```

2. In `_dispatch_command`, add a branch that catches `args.command == "paper"`:

```python
    if args.command == "paper":
        if args.paper_cmd == "status":
            return _cmd_paper_status(config)
        return 2
```

3. Add the new `_cmd_paper_status` function:

```python
def _cmd_paper_status(config: Config) -> int:
    """Print paper-trading status (NAV, open/closed, realized PnL, top trades)."""
    conn = init_db(config.scanner.db_path)
    try:
        paper = PaperTradesRepo(conn)
        summary = paper.summary_stats(
            starting_bankroll=config.paper_trading.starting_bankroll_usd,
        )
        print(f"starting bankroll: ${summary.starting_bankroll:,.2f}")
        print(f"current NAV:       ${summary.current_nav:,.2f}")
        print(f"realized PnL:      ${summary.realized_pnl:+,.2f} ({summary.total_return_pct:+.2f}%)")
        print(f"open positions: {summary.open_positions}    closed positions: {summary.closed_positions}")
        # Per-wallet leaderboard (realized only)
        rows = conn.execute(
            """
            SELECT e.source_wallet,
                   SUM(x.cost_usd - e.cost_usd) AS realized,
                   COUNT(*) AS n
              FROM paper_trades x
              JOIN paper_trades e ON e.trade_id = x.parent_trade_id
             WHERE x.trade_kind='exit' AND e.trade_kind='entry'
             GROUP BY e.source_wallet
             ORDER BY realized DESC
            """,
        ).fetchall()
        if rows:
            print()
            print("per-wallet realized PnL (settled trades):")
            for r in rows:
                print(f"  {r['source_wallet']:<46}  PnL=${r['realized']:+,.2f}  n={r['n']}")
        # Top-3 best/worst settled trades by PnL
        best = conn.execute(
            """
            SELECT e.condition_id, e.outcome, e.source_wallet,
                   (x.cost_usd - e.cost_usd) AS pnl
              FROM paper_trades x
              JOIN paper_trades e ON e.trade_id = x.parent_trade_id
             WHERE x.trade_kind='exit' AND e.trade_kind='entry'
             ORDER BY pnl DESC LIMIT 3
            """,
        ).fetchall()
        if best:
            print()
            print("top 3 best settled trades:")
            for r in best:
                print(f"  PnL=${r['pnl']:+,.2f}  cond={r['condition_id'][:16]}…  outcome={r['outcome']}  wallet={r['source_wallet'][:10]}…")
        worst = conn.execute(
            """
            SELECT e.condition_id, e.outcome, e.source_wallet,
                   (x.cost_usd - e.cost_usd) AS pnl
              FROM paper_trades x
              JOIN paper_trades e ON e.trade_id = x.parent_trade_id
             WHERE x.trade_kind='exit' AND e.trade_kind='entry'
             ORDER BY pnl ASC LIMIT 3
            """,
        ).fetchall()
        if worst:
            print()
            print("top 3 worst settled trades:")
            for r in worst:
                print(f"  PnL=${r['pnl']:+,.2f}  cond={r['condition_id'][:16]}…  outcome={r['outcome']}  wallet={r['source_wallet'][:10]}…")
        return 0
    finally:
        conn.close()
```

4. Add the import: `from pscanner.store.repo import PaperTradesRepo`.

- [ ] **Step 6.5: Run, verify the test passes**

```bash
uv run pytest tests/test_cli.py -v -k paper_status
```

Expected: PASS for both new tests.

- [ ] **Step 6.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/cli.py tests/test_cli.py
uv run ruff format src/pscanner/cli.py tests/test_cli.py
uv run ty check src/pscanner/cli.py
```

- [ ] **Step 6.7: Commit**

```bash
git add src/pscanner/cli.py tests/test_cli.py
git commit -m "feat(cli): pscanner paper status — bankroll, PnL, leaderboard

Reads paper_trades and prints starting bankroll, current NAV, realized
PnL, open/closed counts, per-wallet realized PnL leaderboard, and top-3
best/worst settled trades.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (notes to executor)

After all 6 tasks complete, verify:

1. **Spec coverage:**
   - market_cache extension (outcomes + asset_ids) — T1
   - paper_trades schema + repo — T1
   - PaperTradingConfig — T2
   - PaperTrader (entry path) — T3
   - PaperResolver (exit path) — T4
   - Scheduler wiring + integration smoke — T5
   - `pscanner paper status` CLI — T6
   - Spec error-handling matrix — covered piecemeal across T3 and T4 tests
   - Spec testing breakdown — ~16 unit (T1) + ~11 detector (T3) + ~11 detector (T4) + 1 wiring (T5) + 2 CLI (T6) = ~41 tests. Estimate in spec was ~27 — extra reflects the schema-extension tests this plan adds.

2. **Type consistency:**
   - `OpenPaperPosition.asset_id: AssetId`, `condition_id: ConditionId` consistent across T1, T3, T4.
   - `_size_trade` returns `tuple[float, float] | None` — used as `(cost, shares)`.
   - `_check_resolution` returns `AssetId | None` — used by PaperResolver as a winning marker.
   - `PaperTradesRepo.compute_cost_basis_nav(*, starting_bankroll: float) -> float` consistent across T1 and T3-T6.

3. **No placeholders:** every code block contains real code.

4. **Commit cadence:** 7 commits across 6 tasks (T1 has two logical commits; the rest are one each).

---

## Out-of-plan follow-ups (not blocking)

- The `_check_resolution` heuristic (`exactly one of [outcomes] == 1.0`) does not validate that the *other* outcome is exactly 0.0. Fine for binary markets where the prices sum to 1.0, but if Polymarket ever introduces 3-way markets the heuristic would mis-fire. Defer until ternary markets are observed.
- Mark-to-market PnL display in `paper status` is out of scope (spec). When desired, query `market_ticks` for each open position's current best_bid and add an "unrealized" column.
- `pscanner paper reset` command. Defer until workflow demands it.
- `--strategy` filter on `paper_trades` for multi-strategy v2.
