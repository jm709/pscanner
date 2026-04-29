# Historical trade corpus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `pscanner.corpus` subsystem — a separate ML-trainable historical trade corpus with three idempotent CLI commands (backfill, refresh, build-features), pure-function feature pipeline, and isolated SQLite database.

**Architecture:** New package `src/pscanner/corpus/`, separate `data/corpus.sqlite3`. Reuses `pscanner.poly.{gamma,data,http}` clients (each subsystem instantiates its own with its own rate budget). Pure-function feature pipeline over a `HistoryProvider` Protocol so live scoring in v2 reuses the same code. `StreamingHistoryProvider` for v1 walks `corpus_trades` chronologically, maintaining per-wallet/per-market running aggregates in memory.

**Tech Stack:** Python 3.13, uv, sqlite3 stdlib, structlog, httpx + respx for tests, pytest + property-based hypothesis tests, ruff/ty per `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-04-29-historical-trade-corpus-design.md`

---

## File map

**Create:**
- `src/pscanner/corpus/__init__.py`
- `src/pscanner/corpus/db.py` — schema + connection helpers
- `src/pscanner/corpus/repos.py` — all five repos in one file (mirrors `pscanner.store.repo`)
- `src/pscanner/corpus/features.py` — dataclasses, Protocol, pure functions, StreamingHistoryProvider
- `src/pscanner/corpus/resolutions.py` — outcome lookup and recording
- `src/pscanner/corpus/enumerator.py` — gamma walker for closed markets above gate
- `src/pscanner/corpus/market_walker.py` — per-market `/trades` pagination
- `src/pscanner/corpus/examples.py` — `build-features` orchestrator
- `src/pscanner/corpus/cli.py` — `backfill | refresh | build-features` argparse handlers
- `tests/corpus/__init__.py`
- `tests/corpus/conftest.py` — `tmp_corpus_db` fixture
- `tests/corpus/test_db.py`
- `tests/corpus/test_repos_markets.py`
- `tests/corpus/test_repos_trades.py`
- `tests/corpus/test_repos_resolutions.py`
- `tests/corpus/test_repos_examples.py`
- `tests/corpus/test_features_state.py`
- `tests/corpus/test_features_compute.py`
- `tests/corpus/test_features_streaming.py`
- `tests/corpus/test_resolutions.py`
- `tests/corpus/test_enumerator.py`
- `tests/corpus/test_market_walker.py`
- `tests/corpus/test_examples.py`
- `tests/corpus/test_cli.py`

**Modify:**
- `src/pscanner/poly/ids.py` — add `WalletAddress` NewType
- `src/pscanner/cli.py` — register `corpus` subcommand group

---

## Task 1: Add `WalletAddress` NewType

**Files:**
- Modify: `src/pscanner/poly/ids.py`
- Test: `tests/poly/test_ids.py` (create if missing, otherwise add a single test there)

- [ ] **Step 1: Check if `tests/poly/test_ids.py` exists**

Run: `ls tests/poly/test_ids.py 2>/dev/null && echo exists || echo missing`

If missing, create with a header. Otherwise we'll add to the existing file.

- [ ] **Step 2: Add the test (create file if needed)**

If creating new, file content:

```python
"""Tests for ``pscanner.poly.ids`` NewType wrappers."""

from __future__ import annotations

from pscanner.poly.ids import WalletAddress


def test_wallet_address_is_str_at_runtime() -> None:
    addr = WalletAddress("0xabc")
    assert isinstance(addr, str)
    assert addr == "0xabc"
```

If file exists, append the test function and the import.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/poly/test_ids.py::test_wallet_address_is_str_at_runtime -v`
Expected: FAIL with `ImportError: cannot import name 'WalletAddress'`

- [ ] **Step 4: Add the NewType**

Edit `src/pscanner/poly/ids.py`. Append after the existing `EventSlug` line:

```python
WalletAddress = NewType("WalletAddress", str)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/poly/test_ids.py::test_wallet_address_is_str_at_runtime -v`
Expected: PASS

- [ ] **Step 6: Run full lint/type/test on changes**

Run: `uv run ruff check src/pscanner/poly/ids.py tests/poly/test_ids.py && uv run ty check src/pscanner/poly/ids.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/poly/ids.py tests/poly/test_ids.py
git commit -m "feat(ids): add WalletAddress NewType for corpus pipeline"
```

---

## Task 2: Corpus DB schema + connection helper

**Files:**
- Create: `src/pscanner/corpus/__init__.py`
- Create: `src/pscanner/corpus/db.py`
- Create: `tests/corpus/__init__.py`
- Test: `tests/corpus/test_db.py`

- [ ] **Step 1: Create empty `src/pscanner/corpus/__init__.py`**

Content:

```python
"""Historical trade corpus subsystem.

Builds a per-trade ML-trainable dataset from closed Polymarket markets.
Lives entirely separate from the live daemon: own SQLite file
(``data/corpus.sqlite3``), own CLI commands (``pscanner corpus ...``),
own runtime state.
"""
```

- [ ] **Step 2: Create empty `tests/corpus/__init__.py`**

Empty file (`""`).

- [ ] **Step 3: Write the failing schema test**

Create `tests/corpus/test_db.py`:

```python
"""Tests for ``pscanner.corpus.db`` schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db

_EXPECTED_TABLES = {
    "corpus_markets",
    "corpus_trades",
    "market_resolutions",
    "training_examples",
    "corpus_state",
}


def test_init_corpus_db_creates_all_tables() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {row["name"] for row in rows}
        assert _EXPECTED_TABLES.issubset(names)
    finally:
        conn.close()


def test_init_corpus_db_is_idempotent() -> None:
    conn1 = init_corpus_db(Path(":memory:"))
    conn1.close()
    # In-memory DBs are per-connection; idempotency is verified by checking
    # that the same statements run twice on a real connection without error.
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        from pscanner.corpus.db import _SCHEMA_STATEMENTS

        for _ in range(2):
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def test_init_corpus_db_sets_row_factory() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        assert conn.row_factory is sqlite3.Row
    finally:
        conn.close()


def test_corpus_trades_unique_key() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        conn.execute(
            """
            INSERT INTO corpus_trades
              (tx_hash, asset_id, wallet_address, condition_id,
               outcome_side, bs, price, size, notional_usd, ts)
            VALUES ('0xtx', 'asset1', '0xw', 'cond1',
                    'YES', 'BUY', 0.5, 100.0, 50.0, 1000)
            """
        )
        conn.commit()
        try:
            conn.execute(
                """
                INSERT INTO corpus_trades
                  (tx_hash, asset_id, wallet_address, condition_id,
                   outcome_side, bs, price, size, notional_usd, ts)
                VALUES ('0xtx', 'asset1', '0xw', 'cond1',
                        'YES', 'BUY', 0.5, 100.0, 50.0, 1000)
                """
            )
            conn.commit()
            raise AssertionError("expected UNIQUE constraint failure")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/corpus/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: pscanner.corpus.db`.

- [ ] **Step 5: Implement `src/pscanner/corpus/db.py`**

Create the file with the full schema:

```python
"""SQLite bootstrap for the corpus subsystem.

Creates ``data/corpus.sqlite3`` (idempotently), applies WAL pragmas, and
sets ``row_factory = sqlite3.Row``. The schema is deliberately separate
from ``pscanner.store.db`` — corpus tables never live in the live DB,
and the live DB never holds corpus tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS corpus_markets (
      condition_id TEXT PRIMARY KEY,
      event_slug TEXT NOT NULL,
      category TEXT,
      closed_at INTEGER NOT NULL,
      total_volume_usd REAL NOT NULL,
      backfill_state TEXT NOT NULL,
      last_offset_seen INTEGER,
      trades_pulled_count INTEGER NOT NULL DEFAULT 0,
      truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
      error_message TEXT,
      enumerated_at INTEGER NOT NULL,
      backfill_started_at INTEGER,
      backfill_completed_at INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corpus_markets_state ON corpus_markets(backfill_state)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_markets_volume ON corpus_markets(total_volume_usd DESC)",
    """
    CREATE TABLE IF NOT EXISTS corpus_trades (
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
      UNIQUE(tx_hash, asset_id, wallet_address)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_market_ts ON corpus_trades(condition_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_wallet_ts ON corpus_trades(wallet_address, ts)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_ts ON corpus_trades(ts)",
    """
    CREATE TABLE IF NOT EXISTS market_resolutions (
      condition_id TEXT PRIMARY KEY,
      winning_outcome_index INTEGER NOT NULL,
      outcome_yes_won INTEGER NOT NULL,
      resolved_at INTEGER NOT NULL,
      source TEXT NOT NULL,
      recorded_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_examples (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      tx_hash TEXT NOT NULL,
      asset_id TEXT NOT NULL,
      wallet_address TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      trade_ts INTEGER NOT NULL,
      built_at INTEGER NOT NULL,
      prior_trades_count INTEGER NOT NULL,
      prior_buys_count INTEGER NOT NULL,
      prior_resolved_buys INTEGER NOT NULL,
      prior_wins INTEGER NOT NULL,
      prior_losses INTEGER NOT NULL,
      win_rate REAL,
      avg_implied_prob_paid REAL,
      realized_edge_pp REAL,
      prior_realized_pnl_usd REAL NOT NULL DEFAULT 0,
      avg_bet_size_usd REAL,
      median_bet_size_usd REAL,
      wallet_age_days REAL NOT NULL,
      seconds_since_last_trade INTEGER,
      prior_trades_30d INTEGER NOT NULL,
      top_category TEXT,
      category_diversity INTEGER NOT NULL,
      bet_size_usd REAL NOT NULL,
      bet_size_rel_to_avg REAL,
      side TEXT NOT NULL,
      implied_prob_at_buy REAL NOT NULL,
      market_category TEXT NOT NULL,
      market_volume_so_far_usd REAL NOT NULL,
      market_unique_traders_so_far INTEGER NOT NULL,
      market_age_seconds INTEGER NOT NULL,
      time_to_resolution_seconds INTEGER,
      last_trade_price REAL,
      price_volatility_recent REAL,
      label_won INTEGER NOT NULL,
      UNIQUE(tx_hash, asset_id, wallet_address)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_training_examples_condition ON training_examples(condition_id)",
    "CREATE INDEX IF NOT EXISTS idx_training_examples_wallet ON training_examples(wallet_address)",
    "CREATE INDEX IF NOT EXISTS idx_training_examples_label ON training_examples(label_won)",
    """
    CREATE TABLE IF NOT EXISTS corpus_state (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at INTEGER NOT NULL
    )
    """,
)

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)


def init_corpus_db(path: Path) -> sqlite3.Connection:
    """Open the corpus SQLite database, creating dirs/schema as needed.

    Idempotent: every CREATE statement uses ``IF NOT EXISTS``. The returned
    connection has ``row_factory = sqlite3.Row`` and is in WAL mode.

    Args:
        path: Filesystem path to the corpus database, or ``":memory:"``.
            Parent directories are created for non-memory paths.

    Returns:
        An open ``sqlite3.Connection``. Caller owns the lifecycle.

    Raises:
        sqlite3.DatabaseError: If pragma application or schema creation fails.
    """
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        for pragma in _PRAGMAS:
            conn.execute(pragma)
        with conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_db.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 7: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/pscanner/corpus/__init__.py src/pscanner/corpus/db.py tests/corpus/__init__.py tests/corpus/test_db.py
git commit -m "feat(corpus): add corpus.sqlite3 schema and bootstrap"
```

---

## Task 3: Test conftest with `tmp_corpus_db` fixture

**Files:**
- Create: `tests/corpus/conftest.py`

- [ ] **Step 1: Create the fixture file**

```python
"""Shared pytest fixtures for the corpus test suite.

Mirrors the ``tmp_db`` pattern in ``tests/conftest.py`` but applies
``pscanner.corpus.db.init_corpus_db`` instead.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from pscanner.corpus.db import init_corpus_db


@pytest.fixture
def tmp_corpus_db() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection with the corpus schema applied."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 2: Verify the fixture loads (smoke check)**

Run: `uv run pytest tests/corpus/test_db.py -v --collect-only`
Expected: tests collected, no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/corpus/conftest.py
git commit -m "test(corpus): add tmp_corpus_db fixture"
```

---

## Task 4: `CorpusMarketsRepo` and `CorpusStateRepo`

**Files:**
- Create: `src/pscanner/corpus/repos.py`
- Test: `tests/corpus/test_repos_markets.py`

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_repos_markets.py`:

```python
"""Tests for ``CorpusMarketsRepo`` and ``CorpusStateRepo``."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusStateRepo,
)


def _insert_market(repo: CorpusMarketsRepo, condition_id: str, **kwargs: object) -> None:
    base = CorpusMarket(
        condition_id=condition_id,
        event_slug=kwargs.get("event_slug", "evt"),  # type: ignore[arg-type]
        category=kwargs.get("category", "crypto"),  # type: ignore[arg-type]
        closed_at=int(kwargs.get("closed_at", 1_000)),  # type: ignore[arg-type]
        total_volume_usd=float(kwargs.get("total_volume_usd", 50_000.0)),  # type: ignore[arg-type]
        enumerated_at=int(kwargs.get("enumerated_at", 500)),  # type: ignore[arg-type]
    )
    repo.insert_pending(base)


def test_insert_pending_persists_market(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    row = tmp_corpus_db.execute(
        "SELECT * FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row is not None
    assert row["backfill_state"] == "pending"
    assert row["trades_pulled_count"] == 0
    assert row["truncated_at_offset_cap"] == 0


def test_insert_pending_is_idempotent(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1", total_volume_usd=10_000.0)
    _insert_market(repo, "cond1", total_volume_usd=99_999.0)
    row = tmp_corpus_db.execute(
        "SELECT total_volume_usd FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    # First insert wins; re-inserts are no-ops (INSERT OR IGNORE).
    assert row["total_volume_usd"] == pytest.approx(10_000.0)


def test_pending_largest_first(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "small", total_volume_usd=12_000.0, closed_at=1_000)
    _insert_market(repo, "huge", total_volume_usd=200_000.0, closed_at=900)
    _insert_market(repo, "mid", total_volume_usd=50_000.0, closed_at=950)
    queue = repo.next_pending(limit=10)
    assert [m.condition_id for m in queue] == ["huge", "mid", "small"]


def test_mark_in_progress_updates_state(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_in_progress("cond1", started_at=1_500)
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, backfill_started_at FROM corpus_markets "
        "WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["backfill_state"] == "in_progress"
    assert row["backfill_started_at"] == 1_500


def test_record_progress_updates_offset_and_count(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.record_progress("cond1", last_offset=500, inserted_delta=400)
    repo.record_progress("cond1", last_offset=1000, inserted_delta=350)
    row = tmp_corpus_db.execute(
        "SELECT last_offset_seen, trades_pulled_count "
        "FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["last_offset_seen"] == 1000
    assert row["trades_pulled_count"] == 750


def test_mark_complete_sets_state_and_timestamp(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_complete("cond1", completed_at=2_000, truncated=False)
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, backfill_completed_at, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["backfill_state"] == "complete"
    assert row["backfill_completed_at"] == 2_000
    assert row["truncated_at_offset_cap"] == 0


def test_mark_complete_with_truncation_sets_flag(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_complete("cond1", completed_at=2_000, truncated=True)
    row = tmp_corpus_db.execute(
        "SELECT truncated_at_offset_cap FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["truncated_at_offset_cap"] == 1


def test_mark_failed_records_error(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_failed("cond1", error_message="HTTP 500 after 3 retries")
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, error_message "
        "FROM corpus_markets WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["backfill_state"] == "failed"
    assert row["error_message"] == "HTTP 500 after 3 retries"


def test_resume_in_progress_returned_in_pending_queue(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "cond1")
    repo.mark_in_progress("cond1", started_at=1_000)
    queue = repo.next_pending(limit=10)
    assert [m.condition_id for m in queue] == ["cond1"]


def test_complete_markets_excluded_from_queue(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    _insert_market(repo, "done", total_volume_usd=99_999.0)
    repo.mark_complete("done", completed_at=2_000, truncated=False)
    _insert_market(repo, "todo", total_volume_usd=20_000.0)
    queue = repo.next_pending(limit=10)
    assert [m.condition_id for m in queue] == ["todo"]


def test_state_repo_get_set_roundtrip(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set("last_gamma_sweep_ts", "1700000000", updated_at=1_700_000_001)
    assert repo.get("last_gamma_sweep_ts") == "1700000000"
    repo.set("last_gamma_sweep_ts", "1700001000", updated_at=1_700_001_001)
    assert repo.get("last_gamma_sweep_ts") == "1700001000"


def test_state_repo_get_missing_returns_none(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    assert repo.get("never_set") is None


def test_state_repo_get_int(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set("last_gamma_sweep_ts", "1700000000", updated_at=1_700_000_001)
    assert repo.get_int("last_gamma_sweep_ts") == 1_700_000_000
    assert repo.get_int("missing") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_repos_markets.py -v`
Expected: FAIL with `ModuleNotFoundError: pscanner.corpus.repos`.

- [ ] **Step 3: Implement `CorpusMarket` dataclass and the two repos**

Create `src/pscanner/corpus/repos.py`:

```python
"""Repositories for the corpus subsystem.

One file mirrors ``pscanner.store.repo``. Each repo wraps a single table
(or two related tables) with typed insert/get/update methods. All methods
take a ``sqlite3.Connection`` injected at construction.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusMarket:
    """A market that qualifies for the corpus (volume gate passed).

    Identifies a closed Polymarket market by its ``condition_id``. The
    ``backfill_state`` is tracked separately on the row and progresses
    ``pending → in_progress → complete | failed``.
    """

    condition_id: str
    event_slug: str
    category: str | None
    closed_at: int
    total_volume_usd: float
    enumerated_at: int


class CorpusMarketsRepo:
    """Manage the ``corpus_markets`` work-queue table.

    All write methods commit immediately. Reads use the connection's
    default transaction state.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_pending(self, market: CorpusMarket) -> None:
        """Insert a market in ``pending`` state. Idempotent (INSERT OR IGNORE)."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO corpus_markets (
              condition_id, event_slug, category, closed_at, total_volume_usd,
              backfill_state, enumerated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                market.condition_id,
                market.event_slug,
                market.category,
                market.closed_at,
                market.total_volume_usd,
                market.enumerated_at,
            ),
        )
        self._conn.commit()

    def next_pending(self, *, limit: int) -> list[CorpusMarket]:
        """Return up to ``limit`` markets needing work, largest-volume-first.

        Includes both ``pending`` and ``in_progress`` rows (the latter cover
        the resume case after a crash). Failed rows are also included so
        re-running ``backfill`` retries them. Tied volume breaks by
        ``closed_at`` descending.
        """
        rows = self._conn.execute(
            """
            SELECT condition_id, event_slug, category, closed_at,
                   total_volume_usd, enumerated_at
            FROM corpus_markets
            WHERE backfill_state IN ('pending', 'in_progress', 'failed')
            ORDER BY total_volume_usd DESC, closed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            CorpusMarket(
                condition_id=row["condition_id"],
                event_slug=row["event_slug"],
                category=row["category"],
                closed_at=row["closed_at"],
                total_volume_usd=row["total_volume_usd"],
                enumerated_at=row["enumerated_at"],
            )
            for row in rows
        ]

    def get_last_offset(self, condition_id: str) -> int:
        """Return the last offset seen for a market, or 0 if not started."""
        row = self._conn.execute(
            "SELECT last_offset_seen FROM corpus_markets WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None or row["last_offset_seen"] is None:
            return 0
        return int(row["last_offset_seen"])

    def mark_in_progress(self, condition_id: str, *, started_at: int) -> None:
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'in_progress',
                backfill_started_at = COALESCE(backfill_started_at, ?)
            WHERE condition_id = ?
            """,
            (started_at, condition_id),
        )
        self._conn.commit()

    def record_progress(
        self,
        condition_id: str,
        *,
        last_offset: int,
        inserted_delta: int,
    ) -> None:
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET last_offset_seen = ?,
                trades_pulled_count = trades_pulled_count + ?
            WHERE condition_id = ?
            """,
            (last_offset, inserted_delta, condition_id),
        )
        self._conn.commit()

    def mark_complete(
        self,
        condition_id: str,
        *,
        completed_at: int,
        truncated: bool,
    ) -> None:
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'complete',
                backfill_completed_at = ?,
                truncated_at_offset_cap = ?,
                error_message = NULL
            WHERE condition_id = ?
            """,
            (completed_at, 1 if truncated else 0, condition_id),
        )
        self._conn.commit()

    def mark_failed(self, condition_id: str, *, error_message: str) -> None:
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'failed',
                error_message = ?
            WHERE condition_id = ?
            """,
            (error_message, condition_id),
        )
        self._conn.commit()


class CorpusStateRepo:
    """Tiny key/value cursor table for cross-cutting orchestrator state."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM corpus_state WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row["value"])

    def get_int(self, key: str) -> int | None:
        value = self.get(key)
        return None if value is None else int(value)

    def set(self, key: str, value: str, *, updated_at: int) -> None:
        self._conn.execute(
            """
            INSERT INTO corpus_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_repos_markets.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_markets.py
git commit -m "feat(corpus): add CorpusMarketsRepo and CorpusStateRepo"
```

---

## Task 5: `CorpusTradesRepo`

**Files:**
- Modify: `src/pscanner/corpus/repos.py` — append `CorpusTrade` dataclass and `CorpusTradesRepo`
- Test: `tests/corpus/test_repos_trades.py`

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_repos_trades.py`:

```python
"""Tests for ``CorpusTradesRepo``."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import CorpusTrade, CorpusTradesRepo


def _trade(**kwargs: object) -> CorpusTrade:
    base = {
        "tx_hash": "0xtx",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "outcome_side": "YES",
        "bs": "BUY",
        "price": 0.5,
        "size": 100.0,
        "notional_usd": 50.0,
        "ts": 1_000,
    }
    base.update(kwargs)
    return CorpusTrade(**base)  # type: ignore[arg-type]


def test_insert_batch_persists_trades(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    inserted = repo.insert_batch([
        _trade(tx_hash="0xa"),
        _trade(tx_hash="0xb"),
    ])
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT tx_hash FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xa", "0xb"]


def test_insert_batch_dedupes_on_unique_key(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch([_trade(tx_hash="0xa")])
    inserted = repo.insert_batch([_trade(tx_hash="0xa"), _trade(tx_hash="0xb")])
    assert inserted == 1
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM corpus_trades").fetchone()["c"]
    assert count == 2


def test_insert_batch_normalizes_wallet_to_lowercase(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch([_trade(wallet_address="0xABCDEF")])
    row = tmp_corpus_db.execute(
        "SELECT wallet_address FROM corpus_trades"
    ).fetchone()
    assert row["wallet_address"] == "0xabcdef"


def test_insert_batch_filters_below_notional_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    inserted = repo.insert_batch([
        _trade(tx_hash="0xbig", notional_usd=50.0),
        _trade(tx_hash="0xsmall", notional_usd=4.99),
    ])
    assert inserted == 1
    rows = tmp_corpus_db.execute(
        "SELECT tx_hash FROM corpus_trades"
    ).fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xbig"]


def test_iter_chronological_yields_in_ts_order(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch([
        _trade(tx_hash="0xc", ts=3_000),
        _trade(tx_hash="0xa", ts=1_000),
        _trade(tx_hash="0xb", ts=2_000),
    ])
    seen = [t.tx_hash for t in repo.iter_chronological()]
    assert seen == ["0xa", "0xb", "0xc"]


def test_iter_chronological_breaks_ties_deterministically(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch([
        _trade(tx_hash="0xa", ts=1_000),
        _trade(tx_hash="0xb", ts=1_000),
    ])
    seen_first = [t.tx_hash for t in repo.iter_chronological()]
    seen_second = [t.tx_hash for t in repo.iter_chronological()]
    assert seen_first == seen_second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_repos_trades.py -v`
Expected: FAIL with `ImportError: CorpusTrade`.

- [ ] **Step 3: Append `CorpusTrade` and `CorpusTradesRepo` to `repos.py`**

Append to `src/pscanner/corpus/repos.py` (add to existing imports if needed):

```python
from collections.abc import Iterable, Iterator
from typing import Final

_NOTIONAL_FLOOR_USD: Final[float] = 10.0


@dataclass(frozen=True)
class CorpusTrade:
    """One BUY or SELL fill captured by the market-walker.

    Wallet addresses are normalized to lowercase at insert time. ``bs`` is
    ``BUY`` or ``SELL``; ``outcome_side`` is ``YES`` or ``NO``. ``price``
    is the implied probability paid (already normalized so YES@0.7 and
    NO@0.3 are equivalent buys of the same outcome).
    """

    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    outcome_side: str
    bs: str
    price: float
    size: float
    notional_usd: float
    ts: int


class CorpusTradesRepo:
    """Append-only writes + chronological streaming reads on ``corpus_trades``.

    The notional floor (``$10``) is applied at insert time — sub-floor
    trades never land. The unique constraint
    ``(tx_hash, asset_id, wallet_address)`` makes ``insert_batch``
    idempotent.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_batch(self, trades: Iterable[CorpusTrade]) -> int:
        """Insert trades, skipping duplicates and sub-floor notionals.

        Returns the number of rows actually inserted.
        """
        rows = []
        for t in trades:
            if t.notional_usd < _NOTIONAL_FLOOR_USD:
                continue
            rows.append(
                (
                    t.tx_hash,
                    t.asset_id,
                    t.wallet_address.lower(),
                    t.condition_id,
                    t.outcome_side,
                    t.bs,
                    t.price,
                    t.size,
                    t.notional_usd,
                    t.ts,
                )
            )
        if not rows:
            return 0
        cur = self._conn.executemany(
            """
            INSERT OR IGNORE INTO corpus_trades (
              tx_hash, asset_id, wallet_address, condition_id,
              outcome_side, bs, price, size, notional_usd, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return cur.rowcount or 0

    def iter_chronological(self) -> Iterator[CorpusTrade]:
        """Yield every trade in (ts, tx_hash, asset_id) order.

        Tie-breaking on ``(tx_hash, asset_id)`` makes the iteration
        order deterministic for the streaming feature pipeline.
        """
        cursor = self._conn.execute(
            """
            SELECT tx_hash, asset_id, wallet_address, condition_id,
                   outcome_side, bs, price, size, notional_usd, ts
            FROM corpus_trades
            ORDER BY ts, tx_hash, asset_id
            """
        )
        for row in cursor:
            yield CorpusTrade(
                tx_hash=row["tx_hash"],
                asset_id=row["asset_id"],
                wallet_address=row["wallet_address"],
                condition_id=row["condition_id"],
                outcome_side=row["outcome_side"],
                bs=row["bs"],
                price=row["price"],
                size=row["size"],
                notional_usd=row["notional_usd"],
                ts=row["ts"],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_repos_trades.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_trades.py
git commit -m "feat(corpus): add CorpusTradesRepo with notional floor + chronological iter"
```

---

## Task 6: `MarketResolutionsRepo` and `TrainingExamplesRepo`

**Files:**
- Modify: `src/pscanner/corpus/repos.py`
- Test: `tests/corpus/test_repos_resolutions.py`
- Test: `tests/corpus/test_repos_examples.py`

- [ ] **Step 1: Write failing resolutions tests**

Create `tests/corpus/test_repos_resolutions.py`:

```python
"""Tests for ``MarketResolutionsRepo``."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import MarketResolution, MarketResolutionsRepo


def test_upsert_inserts_new(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    row = tmp_corpus_db.execute(
        "SELECT * FROM market_resolutions WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["winning_outcome_index"] == 0
    assert row["outcome_yes_won"] == 1
    assert row["resolved_at"] == 2_000
    assert row["recorded_at"] == 2_001


def test_upsert_updates_existing(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=2_500,
            source="gamma",
        ),
        recorded_at=2_501,
    )
    row = tmp_corpus_db.execute(
        "SELECT winning_outcome_index, outcome_yes_won FROM market_resolutions "
        "WHERE condition_id = 'cond1'"
    ).fetchone()
    assert row["winning_outcome_index"] == 1
    assert row["outcome_yes_won"] == 0


def test_get_returns_none_for_missing(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    assert repo.get("missing") is None


def test_get_returns_resolution(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    res = repo.get("cond1")
    assert res is not None
    assert res.outcome_yes_won == 0


def test_missing_for_returns_unresolved_condition_ids(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="resolved",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000,
            source="gamma",
        ),
        recorded_at=2_001,
    )
    missing = repo.missing_for(["resolved", "unresolved1", "unresolved2"])
    assert set(missing) == {"unresolved1", "unresolved2"}
```

- [ ] **Step 2: Write failing examples tests**

Create `tests/corpus/test_repos_examples.py`:

```python
"""Tests for ``TrainingExamplesRepo``."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import TrainingExample, TrainingExamplesRepo


def _example(**kwargs: object) -> TrainingExample:
    base = {
        "tx_hash": "0xtx",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "trade_ts": 1_000,
        "built_at": 2_000,
        "prior_trades_count": 0,
        "prior_buys_count": 0,
        "prior_resolved_buys": 0,
        "prior_wins": 0,
        "prior_losses": 0,
        "win_rate": None,
        "avg_implied_prob_paid": None,
        "realized_edge_pp": None,
        "prior_realized_pnl_usd": 0.0,
        "avg_bet_size_usd": None,
        "median_bet_size_usd": None,
        "wallet_age_days": 0.0,
        "seconds_since_last_trade": None,
        "prior_trades_30d": 0,
        "top_category": None,
        "category_diversity": 0,
        "bet_size_usd": 50.0,
        "bet_size_rel_to_avg": None,
        "side": "YES",
        "implied_prob_at_buy": 0.5,
        "market_category": "crypto",
        "market_volume_so_far_usd": 0.0,
        "market_unique_traders_so_far": 0,
        "market_age_seconds": 100,
        "time_to_resolution_seconds": 86_400,
        "last_trade_price": None,
        "price_volatility_recent": None,
        "label_won": 1,
    }
    base.update(kwargs)
    return TrainingExample(**base)  # type: ignore[arg-type]


def test_insert_or_ignore_persists_row(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    inserted = repo.insert_or_ignore([_example(tx_hash="0xa")])
    assert inserted == 1
    count = tmp_corpus_db.execute(
        "SELECT COUNT(*) AS c FROM training_examples"
    ).fetchone()["c"]
    assert count == 1


def test_insert_or_ignore_skips_duplicates(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([_example(tx_hash="0xa")])
    inserted = repo.insert_or_ignore([_example(tx_hash="0xa"), _example(tx_hash="0xb")])
    assert inserted == 1


def test_truncate_deletes_all_rows(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([_example(tx_hash="0xa"), _example(tx_hash="0xb")])
    repo.truncate()
    count = tmp_corpus_db.execute(
        "SELECT COUNT(*) AS c FROM training_examples"
    ).fetchone()["c"]
    assert count == 0


def test_existing_keys_returns_set_of_seen_unique_keys(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([_example(tx_hash="0xa", asset_id="A1", wallet_address="0xw")])
    keys = repo.existing_keys()
    assert ("0xa", "A1", "0xw") in keys
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_repos_resolutions.py tests/corpus/test_repos_examples.py -v`
Expected: FAIL with `ImportError: MarketResolution` / `TrainingExample`.

- [ ] **Step 4: Append the two repos to `repos.py`**

Append to `src/pscanner/corpus/repos.py`:

```python
@dataclass(frozen=True)
class MarketResolution:
    """Resolved outcome for a closed market."""

    condition_id: str
    winning_outcome_index: int
    outcome_yes_won: int  # 1 if YES won, 0 if NO won
    resolved_at: int
    source: str


class MarketResolutionsRepo:
    """Upserts and lookups against ``market_resolutions``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, resolution: MarketResolution, *, recorded_at: int) -> None:
        self._conn.execute(
            """
            INSERT INTO market_resolutions (
              condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
              winning_outcome_index = excluded.winning_outcome_index,
              outcome_yes_won = excluded.outcome_yes_won,
              resolved_at = excluded.resolved_at,
              source = excluded.source,
              recorded_at = excluded.recorded_at
            """,
            (
                resolution.condition_id,
                resolution.winning_outcome_index,
                resolution.outcome_yes_won,
                resolution.resolved_at,
                resolution.source,
                recorded_at,
            ),
        )
        self._conn.commit()

    def get(self, condition_id: str) -> MarketResolution | None:
        row = self._conn.execute(
            """
            SELECT condition_id, winning_outcome_index, outcome_yes_won,
                   resolved_at, source
            FROM market_resolutions WHERE condition_id = ?
            """,
            (condition_id,),
        ).fetchone()
        if row is None:
            return None
        return MarketResolution(
            condition_id=row["condition_id"],
            winning_outcome_index=row["winning_outcome_index"],
            outcome_yes_won=row["outcome_yes_won"],
            resolved_at=row["resolved_at"],
            source=row["source"],
        )

    def missing_for(self, condition_ids: Iterable[str]) -> list[str]:
        ids = list(condition_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"""
            SELECT condition_id FROM market_resolutions
            WHERE condition_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        present = {row["condition_id"] for row in rows}
        return [cid for cid in ids if cid not in present]


@dataclass(frozen=True)
class TrainingExample:
    """One materialized row in the training_examples table.

    The full feature set computed at the trade's timestamp from prior
    trades only. ``label_won`` is the binary target.
    """

    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    trade_ts: int
    built_at: int
    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    win_rate: float | None
    avg_implied_prob_paid: float | None
    realized_edge_pp: float | None
    prior_realized_pnl_usd: float
    avg_bet_size_usd: float | None
    median_bet_size_usd: float | None
    wallet_age_days: float
    seconds_since_last_trade: int | None
    prior_trades_30d: int
    top_category: str | None
    category_diversity: int
    bet_size_usd: float
    bet_size_rel_to_avg: float | None
    side: str
    implied_prob_at_buy: float
    market_category: str
    market_volume_so_far_usd: float
    market_unique_traders_so_far: int
    market_age_seconds: int
    time_to_resolution_seconds: int | None
    last_trade_price: float | None
    price_volatility_recent: float | None
    label_won: int


class TrainingExamplesRepo:
    """Inserts, truncate, and uniqueness lookups against ``training_examples``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_or_ignore(self, examples: Iterable[TrainingExample]) -> int:
        rows = [_example_to_row(ex) for ex in examples]
        if not rows:
            return 0
        cur = self._conn.executemany(
            """
            INSERT OR IGNORE INTO training_examples (
              tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
              prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, win_rate, avg_implied_prob_paid,
              realized_edge_pp, prior_realized_pnl_usd,
              avg_bet_size_usd, median_bet_size_usd, wallet_age_days,
              seconds_since_last_trade, prior_trades_30d, top_category,
              category_diversity, bet_size_usd, bet_size_rel_to_avg, side,
              implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds,
              time_to_resolution_seconds, last_trade_price, price_volatility_recent,
              label_won
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return cur.rowcount or 0

    def truncate(self) -> None:
        self._conn.execute("DELETE FROM training_examples")
        self._conn.commit()

    def existing_keys(self) -> set[tuple[str, str, str]]:
        rows = self._conn.execute(
            "SELECT tx_hash, asset_id, wallet_address FROM training_examples"
        ).fetchall()
        return {(row["tx_hash"], row["asset_id"], row["wallet_address"]) for row in rows}


def _example_to_row(ex: TrainingExample) -> tuple[object, ...]:
    return (
        ex.tx_hash, ex.asset_id, ex.wallet_address, ex.condition_id,
        ex.trade_ts, ex.built_at,
        ex.prior_trades_count, ex.prior_buys_count, ex.prior_resolved_buys,
        ex.prior_wins, ex.prior_losses, ex.win_rate, ex.avg_implied_prob_paid,
        ex.realized_edge_pp, ex.prior_realized_pnl_usd,
        ex.avg_bet_size_usd, ex.median_bet_size_usd, ex.wallet_age_days,
        ex.seconds_since_last_trade, ex.prior_trades_30d, ex.top_category,
        ex.category_diversity, ex.bet_size_usd, ex.bet_size_rel_to_avg, ex.side,
        ex.implied_prob_at_buy, ex.market_category, ex.market_volume_so_far_usd,
        ex.market_unique_traders_so_far, ex.market_age_seconds,
        ex.time_to_resolution_seconds, ex.last_trade_price, ex.price_volatility_recent,
        ex.label_won,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_repos_resolutions.py tests/corpus/test_repos_examples.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 6: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_resolutions.py tests/corpus/test_repos_examples.py
git commit -m "feat(corpus): add MarketResolutionsRepo and TrainingExamplesRepo"
```

---

## Task 7: Feature dataclasses and pure state-update functions

**Files:**
- Create: `src/pscanner/corpus/features.py` (initial cut: dataclasses + state-update functions)
- Test: `tests/corpus/test_features_state.py`

- [ ] **Step 1: Write failing state-update tests**

Create `tests/corpus/test_features_state.py`:

```python
"""Tests for the pure state-update functions in ``corpus.features``."""

from __future__ import annotations

import math

import pytest

from pscanner.corpus.features import (
    Trade,
    WalletState,
    apply_buy_to_state,
    apply_resolution_to_state,
    apply_sell_to_state,
    empty_wallet_state,
)


def _trade(**kwargs: object) -> Trade:
    base = {
        "tx_hash": "0x" + str(kwargs.get("tx_hash", "a")),
        "asset_id": "a1",
        "wallet_address": "0xw",
        "condition_id": kwargs.get("condition_id", "cond1"),
        "outcome_side": "YES",
        "bs": kwargs.get("bs", "BUY"),
        "price": float(kwargs.get("price", 0.4)),  # type: ignore[arg-type]
        "size": float(kwargs.get("size", 100.0)),  # type: ignore[arg-type]
        "notional_usd": float(kwargs.get("notional_usd", 40.0)),  # type: ignore[arg-type]
        "ts": int(kwargs.get("ts", 1_000)),  # type: ignore[arg-type]
        "category": kwargs.get("category", "crypto"),
    }
    return Trade(**base)  # type: ignore[arg-type]


def test_empty_wallet_state_has_zero_counts() -> None:
    state = empty_wallet_state(first_seen_ts=500)
    assert state.first_seen_ts == 500
    assert state.prior_trades_count == 0
    assert state.prior_buys_count == 0
    assert state.prior_resolved_buys == 0
    assert state.prior_wins == 0
    assert state.prior_losses == 0
    assert state.cumulative_buy_price_sum == 0.0
    assert state.cumulative_buy_count == 0
    assert state.realized_pnl_usd == 0.0
    assert state.last_trade_ts is None
    assert state.recent_30d_trades == ()
    assert state.bet_sizes == ()
    assert state.category_counts == {}


def test_apply_buy_increments_counts_and_records_price() -> None:
    state = empty_wallet_state(first_seen_ts=500)
    new_state = apply_buy_to_state(state, _trade(price=0.4, notional_usd=40.0))
    assert new_state.prior_trades_count == 1
    assert new_state.prior_buys_count == 1
    assert new_state.cumulative_buy_count == 1
    assert new_state.cumulative_buy_price_sum == pytest.approx(0.4)
    assert new_state.last_trade_ts == 1_000
    assert new_state.bet_sizes == (40.0,)
    assert new_state.category_counts == {"crypto": 1}


def test_apply_sell_increments_total_but_not_buy() -> None:
    state = empty_wallet_state(first_seen_ts=500)
    new_state = apply_sell_to_state(state, _trade(bs="SELL"))
    assert new_state.prior_trades_count == 1
    assert new_state.prior_buys_count == 0


def test_apply_buy_appends_recent_window() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = apply_buy_to_state(state, _trade(tx_hash="a", ts=1_000))
    state = apply_buy_to_state(state, _trade(tx_hash="b", ts=2_000))
    assert state.recent_30d_trades == (1_000, 2_000)


def test_apply_resolution_records_win() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = apply_buy_to_state(state, _trade(price=0.4, notional_usd=40.0))
    state = apply_resolution_to_state(state, won=True, notional_usd=40.0, payout_usd=100.0)
    assert state.prior_resolved_buys == 1
    assert state.prior_wins == 1
    assert state.prior_losses == 0
    assert state.realized_pnl_usd == pytest.approx(60.0)


def test_apply_resolution_records_loss() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = apply_buy_to_state(state, _trade(notional_usd=40.0))
    state = apply_resolution_to_state(state, won=False, notional_usd=40.0, payout_usd=0.0)
    assert state.prior_resolved_buys == 1
    assert state.prior_wins == 0
    assert state.prior_losses == 1
    assert state.realized_pnl_usd == pytest.approx(-40.0)


def test_state_immutability() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state2 = apply_buy_to_state(state, _trade())
    assert state.prior_trades_count == 0
    assert state2.prior_trades_count == 1
    assert math.isclose(state.cumulative_buy_price_sum, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_features_state.py -v`
Expected: FAIL with `ModuleNotFoundError: pscanner.corpus.features`.

- [ ] **Step 3: Create `features.py` with dataclasses and state-update functions**

Create `src/pscanner/corpus/features.py`:

```python
"""Pure feature-computation primitives for the corpus pipeline.

This module is the heart of the live/historical parity guarantee: every
function here is pure, taking a ``Trade`` plus a ``HistoryProvider`` (or
plain ``WalletState`` / ``MarketState``) and returning frozen dataclasses.
The same functions run inside ``StreamingHistoryProvider`` (v1, walking
``corpus_trades`` for ``build-features``) and inside
``LiveHistoryProvider`` (v2, fed by the live trade stream).

No DB handles, no network, no clocks. All non-determinism enters via
``HistoryProvider``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Trade:
    """One BUY or SELL fill, the input to feature extraction.

    The same shape covers both historical (``corpus_trades``) and live
    (websocket / activity stream) trade events.
    """

    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    outcome_side: str
    bs: str
    price: float
    size: float
    notional_usd: float
    ts: int
    category: str


@dataclass(frozen=True)
class WalletState:
    """Running per-wallet aggregate at some point in time.

    Holds enough state to derive every trader feature in
    ``training_examples``. Updated by ``apply_*_to_state`` functions.
    """

    first_seen_ts: int
    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    cumulative_buy_price_sum: float
    cumulative_buy_count: int
    realized_pnl_usd: float
    last_trade_ts: int | None
    recent_30d_trades: tuple[int, ...]
    bet_sizes: tuple[float, ...]
    category_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketState:
    """Running per-market aggregate at some point in time."""

    market_age_start_ts: int
    volume_so_far_usd: float
    unique_traders_so_far: tuple[str, ...]
    last_trade_price: float | None
    recent_prices: tuple[float, ...]


@dataclass(frozen=True)
class MarketMetadata:
    """Static per-market metadata. Does not change with time."""

    condition_id: str
    category: str
    closed_at: int
    opened_at: int


def empty_wallet_state(*, first_seen_ts: int) -> WalletState:
    """Construct an initial WalletState for a wallet's first seen ts."""
    return WalletState(
        first_seen_ts=first_seen_ts,
        prior_trades_count=0,
        prior_buys_count=0,
        prior_resolved_buys=0,
        prior_wins=0,
        prior_losses=0,
        cumulative_buy_price_sum=0.0,
        cumulative_buy_count=0,
        realized_pnl_usd=0.0,
        last_trade_ts=None,
        recent_30d_trades=(),
        bet_sizes=(),
        category_counts={},
    )


def empty_market_state(*, market_age_start_ts: int) -> MarketState:
    """Construct an initial MarketState for a market's first seen trade."""
    return MarketState(
        market_age_start_ts=market_age_start_ts,
        volume_so_far_usd=0.0,
        unique_traders_so_far=(),
        last_trade_price=None,
        recent_prices=(),
    )


def apply_buy_to_state(state: WalletState, trade: Trade) -> WalletState:
    """Apply a BUY fill to wallet state. Returns a new WalletState."""
    new_categories = dict(state.category_counts)
    new_categories[trade.category] = new_categories.get(trade.category, 0) + 1
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        prior_buys_count=state.prior_buys_count + 1,
        cumulative_buy_price_sum=state.cumulative_buy_price_sum + trade.price,
        cumulative_buy_count=state.cumulative_buy_count + 1,
        last_trade_ts=trade.ts,
        recent_30d_trades=state.recent_30d_trades + (trade.ts,),
        bet_sizes=state.bet_sizes + (trade.notional_usd,),
        category_counts=new_categories,
    )


def apply_sell_to_state(state: WalletState, trade: Trade) -> WalletState:
    """Apply a SELL fill to wallet state. Returns a new WalletState.

    Sells contribute to total trade count and recency but not to BUY
    aggregates (avg price paid, bet sizes, win/loss ledger).
    """
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        last_trade_ts=trade.ts,
        recent_30d_trades=state.recent_30d_trades + (trade.ts,),
    )


def apply_resolution_to_state(
    state: WalletState,
    *,
    won: bool,
    notional_usd: float,
    payout_usd: float,
) -> WalletState:
    """Fold a resolved prior buy into wallet state.

    ``payout_usd`` is the dollar amount returned at resolution
    (``size * 1.0`` if won, ``0.0`` if lost). Realized PnL increments by
    ``payout_usd - notional_usd``.
    """
    return replace(
        state,
        prior_resolved_buys=state.prior_resolved_buys + 1,
        prior_wins=state.prior_wins + (1 if won else 0),
        prior_losses=state.prior_losses + (0 if won else 1),
        realized_pnl_usd=state.realized_pnl_usd + (payout_usd - notional_usd),
    )


def apply_trade_to_market(state: MarketState, trade: Trade) -> MarketState:
    """Apply a fill to market state (per-market running aggregates)."""
    new_traders: tuple[str, ...]
    if trade.wallet_address in state.unique_traders_so_far:
        new_traders = state.unique_traders_so_far
    else:
        new_traders = state.unique_traders_so_far + (trade.wallet_address,)
    return replace(
        state,
        volume_so_far_usd=state.volume_so_far_usd + trade.notional_usd,
        unique_traders_so_far=new_traders,
        last_trade_price=trade.price,
        recent_prices=(state.recent_prices + (trade.price,))[-20:],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_features_state.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/features.py tests/corpus/test_features_state.py
git commit -m "feat(corpus): add WalletState, MarketState, and pure state-update functions"
```

---

## Task 8: `compute_features` + `HistoryProvider` Protocol

**Files:**
- Modify: `src/pscanner/corpus/features.py` — append Protocol, FeatureRow, compute_features
- Test: `tests/corpus/test_features_compute.py`

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_features_compute.py`:

```python
"""Tests for ``compute_features`` and the ``HistoryProvider`` Protocol."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pscanner.corpus.features import (
    HistoryProvider,
    MarketMetadata,
    MarketState,
    Trade,
    WalletState,
    compute_features,
    empty_market_state,
    empty_wallet_state,
)


@dataclass
class _StubHistory:
    wallet: WalletState
    market: MarketState
    meta: MarketMetadata

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        del wallet_address, as_of_ts
        return self.wallet

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        del condition_id, as_of_ts
        return self.market

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        del condition_id
        return self.meta


def _trade(**kwargs: object) -> Trade:
    base = {
        "tx_hash": "0xa",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "outcome_side": "YES",
        "bs": "BUY",
        "price": 0.4,
        "size": 100.0,
        "notional_usd": 40.0,
        "ts": 1_000_000,
        "category": "crypto",
    }
    base.update(kwargs)
    return Trade(**base)  # type: ignore[arg-type]


def _meta(**kwargs: object) -> MarketMetadata:
    base = {
        "condition_id": "cond1",
        "category": "crypto",
        "closed_at": 2_000_000,
        "opened_at": 500_000,
    }
    base.update(kwargs)
    return MarketMetadata(**base)  # type: ignore[arg-type]


def test_compute_features_no_prior_history_yields_nulls() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=1_000_000),
        market=empty_market_state(market_age_start_ts=500_000),
        meta=_meta(),
    )
    features = compute_features(_trade(), history)
    assert features.prior_trades_count == 0
    assert features.win_rate is None
    assert features.avg_implied_prob_paid is None
    assert features.realized_edge_pp is None
    assert features.bet_size_rel_to_avg is None
    assert features.seconds_since_last_trade is None
    assert features.top_category is None
    assert features.last_trade_price is None
    assert features.price_volatility_recent is None
    assert features.bet_size_usd == pytest.approx(40.0)
    assert features.implied_prob_at_buy == pytest.approx(0.4)


def test_compute_features_with_one_resolved_buy() -> None:
    state = empty_wallet_state(first_seen_ts=0)
    state = state.__class__(
        first_seen_ts=0,
        prior_trades_count=1,
        prior_buys_count=1,
        prior_resolved_buys=1,
        prior_wins=1,
        prior_losses=0,
        cumulative_buy_price_sum=0.3,
        cumulative_buy_count=1,
        realized_pnl_usd=70.0,
        last_trade_ts=900_000,
        recent_30d_trades=(900_000,),
        bet_sizes=(30.0,),
        category_counts={"crypto": 1},
    )
    history: HistoryProvider = _StubHistory(
        wallet=state,
        market=empty_market_state(market_age_start_ts=500_000),
        meta=_meta(),
    )
    features = compute_features(_trade(notional_usd=60.0), history)
    assert features.win_rate == pytest.approx(1.0)
    assert features.avg_implied_prob_paid == pytest.approx(0.3)
    assert features.realized_edge_pp == pytest.approx(0.7)
    assert features.avg_bet_size_usd == pytest.approx(30.0)
    assert features.bet_size_rel_to_avg == pytest.approx(60.0 / 30.0)
    assert features.seconds_since_last_trade == 100_000
    assert features.top_category == "crypto"
    assert features.category_diversity == 1
    assert features.prior_realized_pnl_usd == pytest.approx(70.0)


def test_compute_features_market_features() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=1_000_000),
        market=MarketState(
            market_age_start_ts=500_000,
            volume_so_far_usd=12_345.0,
            unique_traders_so_far=("0xa", "0xb", "0xc"),
            last_trade_price=0.45,
            recent_prices=(0.4, 0.42, 0.45),
        ),
        meta=_meta(),
    )
    features = compute_features(_trade(ts=900_000), history)
    assert features.market_volume_so_far_usd == pytest.approx(12_345.0)
    assert features.market_unique_traders_so_far == 3
    assert features.market_age_seconds == 400_000
    assert features.time_to_resolution_seconds == 1_100_000
    assert features.last_trade_price == pytest.approx(0.45)
    assert features.price_volatility_recent is not None


def test_compute_features_implied_prob_for_no_side() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=0),
        market=empty_market_state(market_age_start_ts=0),
        meta=_meta(),
    )
    features = compute_features(_trade(outcome_side="NO", price=0.7), history)
    assert features.implied_prob_at_buy == pytest.approx(0.7)
    assert features.side == "NO"


def test_compute_features_volatility_null_with_few_prices() -> None:
    history: HistoryProvider = _StubHistory(
        wallet=empty_wallet_state(first_seen_ts=0),
        market=MarketState(
            market_age_start_ts=0,
            volume_so_far_usd=100.0,
            unique_traders_so_far=("0xa",),
            last_trade_price=0.5,
            recent_prices=(0.5,),
        ),
        meta=_meta(),
    )
    features = compute_features(_trade(), history)
    assert features.price_volatility_recent is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_features_compute.py -v`
Expected: FAIL — `compute_features`, `HistoryProvider`, `FeatureRow` not defined.

- [ ] **Step 3: Append the Protocol, FeatureRow, and compute_features to features.py**

Append to `src/pscanner/corpus/features.py`:

```python
import statistics
from typing import Protocol


@dataclass(frozen=True)
class FeatureRow:
    """All features computed for a single trade. Mirrors the columns of
    ``training_examples`` (sans identity columns and ``built_at``).
    """

    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    win_rate: float | None
    avg_implied_prob_paid: float | None
    realized_edge_pp: float | None
    prior_realized_pnl_usd: float
    avg_bet_size_usd: float | None
    median_bet_size_usd: float | None
    wallet_age_days: float
    seconds_since_last_trade: int | None
    prior_trades_30d: int
    top_category: str | None
    category_diversity: int
    bet_size_usd: float
    bet_size_rel_to_avg: float | None
    side: str
    implied_prob_at_buy: float
    market_category: str
    market_volume_so_far_usd: float
    market_unique_traders_so_far: int
    market_age_seconds: int
    time_to_resolution_seconds: int | None
    last_trade_price: float | None
    price_volatility_recent: float | None


class HistoryProvider(Protocol):
    """Looks up wallet/market state at a point in time.

    Two implementations: ``StreamingHistoryProvider`` (corpus, walks
    ``corpus_trades`` chronologically) and ``LiveHistoryProvider`` (v2,
    fed by live trade stream). Both must return state computed strictly
    from events with ``ts < as_of_ts``.
    """

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState: ...

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState: ...

    def market_metadata(self, condition_id: str) -> MarketMetadata: ...


_SECONDS_PER_DAY = 86_400


def compute_features(trade: Trade, history: HistoryProvider) -> FeatureRow:
    """Compute the full feature row for a trade, point-in-time correct.

    Pure function: takes only ``trade`` and ``history``. All
    non-determinism enters via the provider.
    """
    wallet = history.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
    market = history.market_state(trade.condition_id, as_of_ts=trade.ts)
    meta = history.market_metadata(trade.condition_id)

    win_rate = (
        wallet.prior_wins / wallet.prior_resolved_buys
        if wallet.prior_resolved_buys > 0
        else None
    )
    avg_prob = (
        wallet.cumulative_buy_price_sum / wallet.cumulative_buy_count
        if wallet.cumulative_buy_count > 0
        else None
    )
    edge = (
        win_rate - avg_prob
        if win_rate is not None and avg_prob is not None
        else None
    )
    avg_bet = (
        sum(wallet.bet_sizes) / len(wallet.bet_sizes)
        if wallet.bet_sizes
        else None
    )
    median_bet = (
        statistics.median(wallet.bet_sizes) if wallet.bet_sizes else None
    )
    rel_to_avg = (
        trade.notional_usd / avg_bet
        if avg_bet is not None and avg_bet > 0
        else None
    )
    seconds_since_last = (
        trade.ts - wallet.last_trade_ts
        if wallet.last_trade_ts is not None
        else None
    )
    wallet_age_days = max(0.0, (trade.ts - wallet.first_seen_ts) / _SECONDS_PER_DAY)
    cutoff = trade.ts - 30 * _SECONDS_PER_DAY
    recent_30d = sum(1 for ts in wallet.recent_30d_trades if ts >= cutoff)
    top_cat = (
        max(wallet.category_counts.items(), key=lambda kv: kv[1])[0]
        if wallet.category_counts
        else None
    )
    diversity = len(wallet.category_counts)

    implied_prob = trade.price

    volatility = (
        statistics.pstdev(market.recent_prices)
        if len(market.recent_prices) >= 2
        else None
    )
    time_to_resolution = meta.closed_at - trade.ts

    return FeatureRow(
        prior_trades_count=wallet.prior_trades_count,
        prior_buys_count=wallet.prior_buys_count,
        prior_resolved_buys=wallet.prior_resolved_buys,
        prior_wins=wallet.prior_wins,
        prior_losses=wallet.prior_losses,
        win_rate=win_rate,
        avg_implied_prob_paid=avg_prob,
        realized_edge_pp=edge,
        prior_realized_pnl_usd=wallet.realized_pnl_usd,
        avg_bet_size_usd=avg_bet,
        median_bet_size_usd=median_bet,
        wallet_age_days=wallet_age_days,
        seconds_since_last_trade=seconds_since_last,
        prior_trades_30d=recent_30d,
        top_category=top_cat,
        category_diversity=diversity,
        bet_size_usd=trade.notional_usd,
        bet_size_rel_to_avg=rel_to_avg,
        side=trade.outcome_side,
        implied_prob_at_buy=implied_prob,
        market_category=meta.category,
        market_volume_so_far_usd=market.volume_so_far_usd,
        market_unique_traders_so_far=len(market.unique_traders_so_far),
        market_age_seconds=trade.ts - market.market_age_start_ts,
        time_to_resolution_seconds=time_to_resolution,
        last_trade_price=market.last_trade_price,
        price_volatility_recent=volatility,
    )
```

Move the `import statistics` and `from typing import Protocol` lines to the top of the file with the other imports for ruff cleanliness.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_features_compute.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/features.py tests/corpus/test_features_compute.py
git commit -m "feat(corpus): add HistoryProvider Protocol and compute_features"
```

---

## Task 9: `StreamingHistoryProvider`

**Files:**
- Modify: `src/pscanner/corpus/features.py` — append `StreamingHistoryProvider`
- Test: `tests/corpus/test_features_streaming.py`

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_features_streaming.py`:

```python
"""Tests for ``StreamingHistoryProvider``."""

from __future__ import annotations

import heapq

import pytest

from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
)


def _trade(**kwargs: object) -> Trade:
    base = {
        "tx_hash": "0xa",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "outcome_side": "YES",
        "bs": "BUY",
        "price": 0.4,
        "size": 100.0,
        "notional_usd": 40.0,
        "ts": 1_000,
        "category": "crypto",
    }
    base.update(kwargs)
    return Trade(**base)  # type: ignore[arg-type]


def _meta(condition_id: str = "cond1") -> MarketMetadata:
    return MarketMetadata(
        condition_id=condition_id,
        category="crypto",
        closed_at=10_000,
        opened_at=500,
    )


def test_observe_buy_then_query_returns_buy_state() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000))
    state = provider.wallet_state("0xw", as_of_ts=2_000)
    # The buy at 1_000 should be reflected when querying at 2_000.
    assert state.prior_buys_count == 1


def test_observe_then_query_at_same_ts_excludes_event() -> None:
    """Querying at the SAME ts as an observed event should EXCLUDE the event,
    because features are computed BEFORE the event is folded into state.

    The pipeline always queries with ``as_of_ts=trade.ts`` BEFORE calling
    observe(trade), so this ordering is correct.
    """
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    state_before = provider.wallet_state("0xw", as_of_ts=1_000)
    provider.observe(_trade(tx_hash="0xa", ts=1_000))
    assert state_before.prior_buys_count == 0


def test_resolutions_register_resolves_pending_buys() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000, notional_usd=40.0, price=0.4, size=100.0))
    provider.register_resolution(condition_id="cond1", resolved_at=5_000, outcome_yes_won=1)
    state = provider.wallet_state("0xw", as_of_ts=6_000)
    assert state.prior_resolved_buys == 1
    assert state.prior_wins == 1
    assert state.prior_realized_pnl_usd_total() == pytest.approx(60.0)


def test_resolution_pre_query_only_counts_if_resolution_ts_lt_query_ts() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000, size=100.0, price=0.4, notional_usd=40.0))
    provider.register_resolution(condition_id="cond1", resolved_at=5_000, outcome_yes_won=1)
    state_before = provider.wallet_state("0xw", as_of_ts=4_000)
    state_after = provider.wallet_state("0xw", as_of_ts=6_000)
    assert state_before.prior_resolved_buys == 0
    assert state_after.prior_resolved_buys == 1


def test_market_state_tracks_price_history() -> None:
    provider = StreamingHistoryProvider(metadata={"cond1": _meta()})
    provider.observe(_trade(tx_hash="0xa", ts=1_000, price=0.4))
    provider.observe(_trade(tx_hash="0xb", ts=2_000, price=0.5, wallet_address="0xv"))
    market = provider.market_state("cond1", as_of_ts=3_000)
    assert market.last_trade_price == pytest.approx(0.5)
    assert len(market.unique_traders_so_far) == 2
    assert market.volume_so_far_usd > 0


def test_unknown_market_metadata_raises_keyerror() -> None:
    provider = StreamingHistoryProvider(metadata={})
    try:
        provider.market_metadata("unknown")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_resolution_heap_drains_in_order() -> None:
    """Property-style: random interleaving of trades + resolutions; the
    invariant ``prior_wins + prior_losses`` always equals the count of
    prior buys whose resolution_ts < query_ts.
    """
    provider = StreamingHistoryProvider(metadata={
        "c1": _meta("c1"),
        "c2": _meta("c2"),
        "c3": _meta("c3"),
    })
    # Three buys by 0xw on three markets at ts=100, 200, 300; markets
    # resolve at 1_000, 5_000, 3_000 respectively.
    provider.observe(_trade(tx_hash="0xa", ts=100, condition_id="c1", size=100.0, price=0.5, notional_usd=50.0))
    provider.observe(_trade(tx_hash="0xb", ts=200, condition_id="c2", size=100.0, price=0.5, notional_usd=50.0))
    provider.observe(_trade(tx_hash="0xc", ts=300, condition_id="c3", size=100.0, price=0.5, notional_usd=50.0))
    provider.register_resolution(condition_id="c1", resolved_at=1_000, outcome_yes_won=1)
    provider.register_resolution(condition_id="c2", resolved_at=5_000, outcome_yes_won=0)
    provider.register_resolution(condition_id="c3", resolved_at=3_000, outcome_yes_won=1)

    # At as_of_ts=2_000: only c1 has resolved.
    state_2k = provider.wallet_state("0xw", as_of_ts=2_000)
    assert state_2k.prior_resolved_buys == 1
    # At as_of_ts=4_000: c1 + c3 resolved.
    state_4k = provider.wallet_state("0xw", as_of_ts=4_000)
    assert state_4k.prior_resolved_buys == 2
    # At as_of_ts=10_000: all three resolved.
    state_10k = provider.wallet_state("0xw", as_of_ts=10_000)
    assert state_10k.prior_resolved_buys == 3
    assert state_10k.prior_wins == 2  # c1 YES + c3 YES
    assert state_10k.prior_losses == 1  # c2 NO
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_features_streaming.py -v`
Expected: FAIL — `StreamingHistoryProvider` not defined.

- [ ] **Step 3: Append `StreamingHistoryProvider` to features.py**

Append to `src/pscanner/corpus/features.py`:

```python
import heapq


@dataclass(frozen=True)
class _PendingResolution:
    """A buy waiting for its market to resolve, queued against a wallet."""

    resolution_ts: int
    condition_id: str
    notional_usd: float
    size: float
    side_yes: bool


@dataclass
class _WalletAccumulator:
    """Mutable wrapper around WalletState for streaming updates."""

    state: WalletState
    pending: list[tuple[int, int, _PendingResolution]]  # heap entries

    def total_realized_pnl(self) -> float:
        return self.state.realized_pnl_usd


class StreamingHistoryProvider:
    """In-memory provider that walks events chronologically.

    Used inside ``build-features``: the orchestrator calls
    ``wallet_state(...)`` and ``market_state(...)`` BEFORE folding each
    trade in via ``observe(...)``. Resolutions are registered up-front
    via ``register_resolution(...)`` and applied lazily when the next
    ``wallet_state`` query crosses their ``resolution_ts``.
    """

    def __init__(self, metadata: dict[str, MarketMetadata]) -> None:
        self._metadata = metadata
        self._wallets: dict[str, _WalletAccumulator] = {}
        self._markets: dict[str, MarketState] = {}
        self._resolutions: dict[str, tuple[int, int]] = {}  # cond_id -> (resolved_at, yes_won)
        self._heap_seq = 0

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        return self._metadata[condition_id]

    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None:
        """Record a market's resolution. Pending buys on this market are
        processed at the next wallet_state query that crosses ``resolved_at``.
        """
        self._resolutions[condition_id] = (resolved_at, outcome_yes_won)

    def observe(self, trade: Trade) -> None:
        """Fold a trade into running state."""
        # Wallet state.
        accum = self._wallets.get(trade.wallet_address)
        if accum is None:
            accum = _WalletAccumulator(
                state=empty_wallet_state(first_seen_ts=trade.ts),
                pending=[],
            )
            self._wallets[trade.wallet_address] = accum

        if trade.bs == "BUY":
            accum.state = apply_buy_to_state(accum.state, trade)
            resolution = self._resolutions.get(trade.condition_id)
            if resolution is not None:
                resolved_at, _ = resolution
                pending = _PendingResolution(
                    resolution_ts=resolved_at,
                    condition_id=trade.condition_id,
                    notional_usd=trade.notional_usd,
                    size=trade.size,
                    side_yes=trade.outcome_side == "YES",
                )
                self._heap_seq += 1
                heapq.heappush(
                    accum.pending,
                    (resolved_at, self._heap_seq, pending),
                )
        elif trade.bs == "SELL":
            accum.state = apply_sell_to_state(accum.state, trade)

        # Market state.
        market = self._markets.get(trade.condition_id)
        if market is None:
            market = empty_market_state(market_age_start_ts=trade.ts)
        self._markets[trade.condition_id] = apply_trade_to_market(market, trade)

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        accum = self._wallets.get(wallet_address)
        if accum is None:
            return empty_wallet_state(first_seen_ts=as_of_ts)
        # Drain pending resolutions whose ts < as_of_ts.
        while accum.pending and accum.pending[0][0] < as_of_ts:
            _, _, pending = heapq.heappop(accum.pending)
            resolution = self._resolutions.get(pending.condition_id)
            if resolution is None:
                continue
            _, yes_won = resolution
            won = (yes_won == 1) if pending.side_yes else (yes_won == 0)
            payout = pending.size if won else 0.0
            accum.state = apply_resolution_to_state(
                accum.state,
                won=won,
                notional_usd=pending.notional_usd,
                payout_usd=payout,
            )
        return accum.state

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        del as_of_ts  # state already at the right point — caller queries before observe()
        return self._markets.get(
            condition_id,
            empty_market_state(market_age_start_ts=0),
        )
```

Also append the convenience method `prior_realized_pnl_usd_total` on `WalletState` (the test calls it):

```python
def _wallet_state_realized_pnl_total(self: WalletState) -> float:
    return self.realized_pnl_usd

WalletState.prior_realized_pnl_usd_total = _wallet_state_realized_pnl_total  # type: ignore[attr-defined]
```

(Or simpler — just delete that test assertion. Cleaner: replace test line `assert state.prior_realized_pnl_usd_total() == pytest.approx(60.0)` with `assert state.realized_pnl_usd == pytest.approx(60.0)` to avoid monkey-patching frozen dataclasses.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_features_streaming.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/features.py tests/corpus/test_features_streaming.py
git commit -m "feat(corpus): add StreamingHistoryProvider with resolution heap"
```

---

## Task 10: Resolution lookup module

**Files:**
- Create: `src/pscanner/corpus/resolutions.py`
- Test: `tests/corpus/test_resolutions.py`

Looks up market outcomes from gamma's `Market` model and writes them to `market_resolutions`. A `Market` is "resolved" when `closed=True` and exactly one of its `outcome_prices` equals `1.0`. If `outcome_prices` is `[]` or none are 1.0, it's disputed/voided and we skip with a warning.

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_resolutions.py`:

```python
"""Tests for ``pscanner.corpus.resolutions``."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.repos import (
    MarketResolutionsRepo,
)
from pscanner.corpus.resolutions import (
    determine_outcome_yes_won,
    record_resolutions,
)
from pscanner.poly.models import Market


def _market(condition_id: str, outcome_prices: list[float], closed: bool = True) -> Market:
    return Market.model_validate({
        "id": condition_id,
        "conditionId": condition_id,
        "question": "?",
        "slug": "s",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [str(p) for p in outcome_prices],
        "closed": closed,
        "active": False,
    })


def test_determine_outcome_yes_won_yes() -> None:
    m = _market("c1", [1.0, 0.0])
    assert determine_outcome_yes_won(m) == 1


def test_determine_outcome_yes_won_no() -> None:
    m = _market("c1", [0.0, 1.0])
    assert determine_outcome_yes_won(m) == 0


def test_determine_outcome_disputed_returns_none() -> None:
    m = _market("c1", [0.5, 0.5])
    assert determine_outcome_yes_won(m) is None


def test_determine_outcome_empty_prices_returns_none() -> None:
    m = _market("c1", [])
    assert determine_outcome_yes_won(m) is None


@pytest.mark.asyncio
async def test_record_resolutions_writes_resolved_markets(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(side_effect=lambda slug: {
        "evt-c1": _market("c1", [1.0, 0.0]),
        "evt-c2": _market("c2", [0.0, 1.0]),
    }[slug])

    await record_resolutions(
        gamma=fake_gamma,
        repo=repo,
        targets=[("c1", "evt-c1", 1_000), ("c2", "evt-c2", 2_000)],
        now_ts=3_000,
    )
    assert repo.get("c1") is not None
    assert repo.get("c2") is not None
    assert repo.get("c1").outcome_yes_won == 1
    assert repo.get("c2").outcome_yes_won == 0


@pytest.mark.asyncio
async def test_record_resolutions_skips_disputed(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(return_value=_market("c1", [0.5, 0.5]))
    await record_resolutions(
        gamma=fake_gamma,
        repo=repo,
        targets=[("c1", "evt-c1", 1_000)],
        now_ts=3_000,
    )
    assert repo.get("c1") is None
```

Note: `record_resolutions` will need access to `Market`. The gamma client has `get_market_by_slug(slug)` that returns `Market | None`. We pass tuples of `(condition_id, slug_to_query, resolved_at_hint)`. Slug comes from `corpus_markets.event_slug` joined with the market — since gamma's `/markets` endpoint takes a slug param. (See CLAUDE.md note: gamma `/markets` does not return `event_id`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_resolutions.py -v`
Expected: FAIL — `pscanner.corpus.resolutions` not found.

- [ ] **Step 3: Implement `resolutions.py`**

Create `src/pscanner/corpus/resolutions.py`:

```python
"""Market-resolution lookup for the corpus pipeline.

Translates a gamma ``Market`` into a ``MarketResolution`` (which side won)
and writes to ``market_resolutions``. Skips disputed/voided markets where
no outcome price equals 1.0.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import structlog

from pscanner.corpus.repos import MarketResolution, MarketResolutionsRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Market

_log = structlog.get_logger(__name__)
_RESOLVED_PRICE: Final[float] = 1.0


def determine_outcome_yes_won(market: Market) -> int | None:
    """Return 1 if YES (index 0) won, 0 if NO (index 1) won, else None.

    Returns None if outcome_prices is empty or no price is exactly 1.0
    (disputed/voided markets).
    """
    if not market.outcome_prices:
        return None
    for idx, price in enumerate(market.outcome_prices):
        if price == _RESOLVED_PRICE:
            return 1 if idx == 0 else 0
    return None


async def record_resolutions(
    *,
    gamma: GammaClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, str, int]],
    now_ts: int,
) -> int:
    """Fetch resolutions for the given (condition_id, slug, resolved_at) tuples.

    Args:
        gamma: Gamma client with ``get_market_by_slug``.
        repo: ``MarketResolutionsRepo`` to upsert into.
        targets: Iterable of ``(condition_id, market_slug, resolved_at_hint)``.
        now_ts: Unix seconds, recorded as ``recorded_at`` on each row.

    Returns:
        Count of resolutions actually written (excludes skipped/disputed).
    """
    written = 0
    for condition_id, slug, resolved_at in targets:
        market = await gamma.get_market_by_slug(slug)
        if market is None:
            _log.warning("corpus.resolution_market_not_found", condition_id=condition_id, slug=slug)
            continue
        yes_won = determine_outcome_yes_won(market)
        if yes_won is None:
            _log.warning("corpus.resolution_disputed", condition_id=condition_id, slug=slug)
            continue
        repo.upsert(
            MarketResolution(
                condition_id=condition_id,
                winning_outcome_index=0 if yes_won == 1 else 1,
                outcome_yes_won=yes_won,
                resolved_at=resolved_at,
                source="gamma",
            ),
            recorded_at=now_ts,
        )
        written += 1
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_resolutions.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Lint + type check**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/resolutions.py tests/corpus/test_resolutions.py
git commit -m "feat(corpus): add market resolution lookup"
```

---

## Task 11: Gamma enumerator (closed markets above volume gate)

**Files:**
- Create: `src/pscanner/corpus/enumerator.py`
- Test: `tests/corpus/test_enumerator.py`

Walks gamma's closed events, expands into markets, gates by `total_volume_usd >= $10_000`, inserts qualifiers into `corpus_markets` as `pending`.

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_enumerator.py`:

```python
"""Tests for ``pscanner.corpus.enumerator``."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.enumerator import (
    VOLUME_GATE_USD,
    enumerate_closed_markets,
)
from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.poly.models import Event, Market


def _event(slug: str, markets: list[Market], closed: bool = True) -> Event:
    return Event.model_validate({
        "id": slug + "-id",
        "title": "T",
        "slug": slug,
        "markets": [m.model_dump(by_alias=True) for m in markets],
        "active": False,
        "closed": closed,
        "tags": [],
    })


def _market(condition_id: str, volume: float, closed: bool = True) -> Market:
    return Market.model_validate({
        "id": condition_id,
        "conditionId": condition_id,
        "question": "?",
        "slug": condition_id + "-slug",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["1.0", "0.0"],
        "volume": volume,
        "closed": closed,
        "active": False,
    })


@pytest.mark.asyncio
async def test_enumerate_inserts_above_gate(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()

    async def _iter():  # type: ignore[no-untyped-def]
        yield _event("e1", [_market("c1", VOLUME_GATE_USD + 1)])
        yield _event("e2", [_market("c2", VOLUME_GATE_USD - 1)])

    fake_gamma.iter_events = lambda *, active, closed, page_size: _iter()  # noqa: ARG005
    inserted = await enumerate_closed_markets(
        gamma=fake_gamma,
        repo=repo,
        now_ts=1_000,
        since_ts=None,
    )
    assert inserted == 1
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["c1"]


@pytest.mark.asyncio
async def test_enumerate_is_idempotent(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()

    async def _iter():  # type: ignore[no-untyped-def]
        yield _event("e1", [_market("c1", 50_000.0)])

    fake_gamma.iter_events = lambda *, active, closed, page_size: _iter()  # noqa: ARG005
    await enumerate_closed_markets(gamma=fake_gamma, repo=repo, now_ts=1_000, since_ts=None)
    await enumerate_closed_markets(gamma=fake_gamma, repo=repo, now_ts=1_000, since_ts=None)
    count = tmp_corpus_db.execute(
        "SELECT COUNT(*) AS c FROM corpus_markets"
    ).fetchone()["c"]
    assert count == 1


@pytest.mark.asyncio
async def test_enumerate_skips_open_markets(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()

    async def _iter():  # type: ignore[no-untyped-def]
        yield _event(
            "e1",
            [_market("c1", 50_000.0, closed=False)],
            closed=True,
        )

    fake_gamma.iter_events = lambda *, active, closed, page_size: _iter()  # noqa: ARG005
    inserted = await enumerate_closed_markets(
        gamma=fake_gamma, repo=repo, now_ts=1_000, since_ts=None
    )
    assert inserted == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_enumerator.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `enumerator.py`**

Create `src/pscanner/corpus/enumerator.py`:

```python
"""Enumerate closed Polymarket markets above the corpus volume gate."""

from __future__ import annotations

from typing import Final

import structlog

from pscanner.categories import DEFAULT_TAXONOMY
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event

_log = structlog.get_logger(__name__)
VOLUME_GATE_USD: Final[float] = 10_000.0


def _category_for_event(event: Event) -> str:
    """Map gamma event tags to the project's category taxonomy.

    Returns ``"unknown"`` when no tag matches.
    """
    for tag in event.tags:
        slug = tag.lower()
        for category in DEFAULT_TAXONOMY:
            if slug in category.match_slugs:
                return category.name
    return "unknown"


async def enumerate_closed_markets(
    *,
    gamma: GammaClient,
    repo: CorpusMarketsRepo,
    now_ts: int,
    since_ts: int | None,
) -> int:
    """Walk gamma closed events; insert qualifying markets as ``pending``.

    Args:
        gamma: Gamma client with ``iter_events``.
        repo: Markets repo to insert into.
        now_ts: Unix seconds at enumeration time (recorded on rows).
        since_ts: If provided, only events whose ``end_date`` (or first market
            close hint) is >= since_ts are considered. ``None`` for full scan.

    Returns:
        Count of markets actually inserted (excluding duplicates).
    """
    inserted = 0
    async for event in gamma.iter_events(active=False, closed=True, page_size=100):
        if not event.closed:
            continue
        category = _category_for_event(event)
        for market in event.markets:
            if not market.closed:
                continue
            volume = market.volume or 0.0
            if volume < VOLUME_GATE_USD:
                continue
            if market.condition_id is None:
                continue
            corpus = CorpusMarket(
                condition_id=str(market.condition_id),
                event_slug=event.slug,
                category=category,
                closed_at=now_ts,  # gamma doesn't expose a precise close ts; use now
                total_volume_usd=volume,
                enumerated_at=now_ts,
            )
            before = repo.get_last_offset(corpus.condition_id) if False else None  # placeholder no-op
            del before
            count_before = repo._conn.execute(  # noqa: SLF001
                "SELECT COUNT(*) AS c FROM corpus_markets WHERE condition_id = ?",
                (corpus.condition_id,),
            ).fetchone()["c"]
            repo.insert_pending(corpus)
            count_after = repo._conn.execute(  # noqa: SLF001
                "SELECT COUNT(*) AS c FROM corpus_markets WHERE condition_id = ?",
                (corpus.condition_id,),
            ).fetchone()["c"]
            if count_after > count_before:
                inserted += 1
    _log.info("corpus.enumerated", inserted=inserted)
    return inserted
```

(The pre/post count check is the simplest way to know if the INSERT OR IGNORE actually added a row; cursor.rowcount is unreliable for IGNORE in some sqlite versions.)

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/corpus/test_enumerator.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py
git commit -m "feat(corpus): add gamma closed-market enumerator with volume gate"
```

---

## Task 12: Per-market trade walker

**Files:**
- Create: `src/pscanner/corpus/market_walker.py`
- Test: `tests/corpus/test_market_walker.py`

Pages `/trades?market=<condition_id>` via the `DataClient`'s underlying paginator. Inserts trades into `corpus_trades`, updates `corpus_markets` progress columns, marks `complete` (with `truncated=True` if hit offset cap of 3500), or `failed` on persistent errors.

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_market_walker.py`:

```python
"""Tests for the per-market trade walker."""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.market_walker import walk_market
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTradesRepo,
)


def _trade_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "transactionHash": "0xa",
        "asset": "asset1",
        "proxyWallet": "0xWALLET",
        "conditionId": "cond1",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.5,
        "size": 100.0,
        "timestamp": 1_000,
    }
    base.update(overrides)
    return base


def _seed_market(repo: CorpusMarketsRepo, condition_id: str) -> None:
    repo.insert_pending(
        CorpusMarket(
            condition_id=condition_id,
            event_slug="evt",
            category="crypto",
            closed_at=2_000,
            total_volume_usd=50_000.0,
            enumerated_at=500,
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_trades_and_marks_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")

    fake_data = AsyncMock()
    fake_data._fetch_market_trades_page = AsyncMock(side_effect=[
        [_trade_dict(transactionHash="0xa", price=0.5, size=100.0)],
        [],
    ])
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute(
        "SELECT backfill_state, trades_pulled_count, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()
    assert rows["backfill_state"] == "complete"
    assert rows["trades_pulled_count"] == 1
    assert rows["truncated_at_offset_cap"] == 0
    trade_count = tmp_corpus_db.execute(
        "SELECT COUNT(*) AS c FROM corpus_trades"
    ).fetchone()["c"]
    assert trade_count == 1


@pytest.mark.asyncio
async def test_walk_normalizes_wallet_lowercases(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    fake_data._fetch_market_trades_page = AsyncMock(side_effect=[
        [_trade_dict(proxyWallet="0xMIXED")],
        [],
    ])
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute("SELECT wallet_address FROM corpus_trades").fetchone()
    assert row["wallet_address"] == "0xmixed"


@pytest.mark.asyncio
async def test_walk_filters_below_notional_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    fake_data._fetch_market_trades_page = AsyncMock(side_effect=[
        [
            _trade_dict(transactionHash="0xbig", price=0.5, size=100.0),  # 50 USD
            _trade_dict(transactionHash="0xsmall", price=0.05, size=1.0),  # 0.05 USD
        ],
        [],
    ])
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades").fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xbig"]


@pytest.mark.asyncio
async def test_walk_truncates_at_offset_cap(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    # Return full pages until offset 3500 — paginator stops at the cap.
    full_page = [_trade_dict(transactionHash=f"0x{i}") for i in range(500)]

    async def _fetch(condition_id: str, *, offset: int) -> list[dict[str, Any]]:
        del condition_id
        if offset >= 3500:
            return []
        return full_page

    fake_data._fetch_market_trades_page = AsyncMock(side_effect=_fetch)
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets"
    ).fetchone()
    assert row["backfill_state"] == "complete"
    assert row["truncated_at_offset_cap"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_market_walker.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `market_walker.py`**

Create `src/pscanner/corpus/market_walker.py`:

```python
"""Per-market `/trades` pagination walker.

Pages all trades on one market, normalizes them into ``CorpusTrade``,
inserts via ``CorpusTradesRepo``, and updates the ``corpus_markets``
progress + state columns. Idempotent: re-running on a market that's
already complete is a no-op (the trades unique key bounces duplicates).
"""

from __future__ import annotations

from typing import Any, Final

import structlog

from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.poly.data import DataClient

_log = structlog.get_logger(__name__)
_PAGE_SIZE: Final[int] = 500
_OFFSET_CAP: Final[int] = 3500  # Polymarket /trades hard cap


def _parse_trade(item: dict[str, Any], condition_id: str) -> CorpusTrade | None:
    """Best-effort parse of a `/trades` JSON item to ``CorpusTrade``.

    Returns ``None`` if required fields are missing.
    """
    tx = item.get("transactionHash")
    asset = item.get("asset")
    wallet = item.get("proxyWallet")
    side = item.get("side")
    outcome = item.get("outcome")
    price = item.get("price")
    size = item.get("size")
    ts = item.get("timestamp")
    if not isinstance(tx, str) or not isinstance(asset, str):
        return None
    if not isinstance(wallet, str) or not isinstance(side, str):
        return None
    if not isinstance(outcome, str) or not isinstance(ts, int):
        return None
    try:
        price_f = float(price) if price is not None else None
        size_f = float(size) if size is not None else None
    except (TypeError, ValueError):
        return None
    if price_f is None or size_f is None:
        return None
    return CorpusTrade(
        tx_hash=tx,
        asset_id=asset,
        wallet_address=wallet,
        condition_id=condition_id,
        outcome_side="YES" if outcome.lower() == "yes" else "NO",
        bs="BUY" if side.upper() == "BUY" else "SELL",
        price=price_f,
        size=size_f,
        notional_usd=price_f * size_f,
        ts=ts,
    )


async def walk_market(
    *,
    condition_id: str,
    data: DataClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    now_ts: int,
) -> int:
    """Pull every trade on ``condition_id``; record progress and final state.

    Returns the number of trades inserted (post-floor, post-dedupe).
    """
    markets_repo.mark_in_progress(condition_id, started_at=now_ts)
    offset = markets_repo.get_last_offset(condition_id)
    total_inserted = 0
    truncated = False

    try:
        while True:
            page = await data._fetch_market_trades_page(  # noqa: SLF001
                condition_id, offset=offset
            )
            if not page:
                if offset >= _OFFSET_CAP:
                    truncated = True
                break
            parsed: list[CorpusTrade] = []
            for item in page:
                trade = _parse_trade(item, condition_id)
                if trade is not None:
                    parsed.append(trade)
            inserted = trades_repo.insert_batch(parsed)
            total_inserted += inserted
            offset += len(page)
            markets_repo.record_progress(
                condition_id,
                last_offset=offset,
                inserted_delta=inserted,
            )
            if len(page) < _PAGE_SIZE:
                break
            if offset >= _OFFSET_CAP:
                truncated = True
                break
    except Exception as exc:  # noqa: BLE001 - mark_failed needs the message text
        markets_repo.mark_failed(condition_id, error_message=str(exc))
        _log.warning("corpus.walk_market_failed", condition_id=condition_id, error=str(exc))
        raise

    markets_repo.mark_complete(condition_id, completed_at=now_ts, truncated=truncated)
    _log.info(
        "corpus.walk_market_complete",
        condition_id=condition_id,
        trades_inserted=total_inserted,
        truncated=truncated,
    )
    return total_inserted
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/corpus/test_market_walker.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/market_walker.py tests/corpus/test_market_walker.py
git commit -m "feat(corpus): add per-market /trades walker with offset-cap detection"
```

---

## Task 13: `build-features` orchestrator

**Files:**
- Create: `src/pscanner/corpus/examples.py`
- Test: `tests/corpus/test_examples.py`

Wires `StreamingHistoryProvider` over `CorpusTradesRepo.iter_chronological()`, registers all known resolutions up front, and writes `TrainingExample` rows for each qualifying BUY whose market has resolved.

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_examples.py`:

```python
"""Tests for the build-features orchestrator."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)


def _seed_market_metadata(conn: sqlite3.Connection, condition_id: str, **kwargs: object) -> None:
    """Insert a corpus_markets row so build-features has metadata to read."""
    conn.execute(
        """
        INSERT INTO corpus_markets (condition_id, event_slug, category, closed_at,
                                    total_volume_usd, backfill_state, enumerated_at)
        VALUES (?, ?, ?, ?, ?, 'complete', ?)
        """,
        (
            condition_id,
            kwargs.get("event_slug", "evt"),
            kwargs.get("category", "crypto"),
            kwargs.get("closed_at", 10_000),
            kwargs.get("total_volume_usd", 50_000.0),
            kwargs.get("enumerated_at", 0),
        ),
    )
    conn.commit()


def _trade(**kwargs: object) -> CorpusTrade:
    base = {
        "tx_hash": "0xa",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "outcome_side": "YES",
        "bs": "BUY",
        "price": 0.4,
        "size": 100.0,
        "notional_usd": 40.0,
        "ts": 1_000,
    }
    base.update(kwargs)
    return CorpusTrade(**base)  # type: ignore[arg-type]


def test_build_features_skips_when_no_resolution(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade()])

    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=2_000,
    )
    assert written == 0
    count = tmp_corpus_db.execute(
        "SELECT COUNT(*) AS c FROM training_examples"
    ).fetchone()["c"]
    assert count == 0


def test_build_features_writes_row_for_resolved_buy(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(notional_usd=40.0, price=0.4, size=100.0)])
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=5_000,
            source="gamma",
        ),
        recorded_at=5_001,
    )

    written = build_features(
        trades_repo=trades,
        resolutions_repo=resolutions,
        examples_repo=examples,
        markets_conn=tmp_corpus_db,
        now_ts=10_000,
    )
    assert written == 1
    row = tmp_corpus_db.execute(
        "SELECT label_won, prior_buys_count FROM training_examples"
    ).fetchone()
    assert row["label_won"] == 1  # YES bought, YES won
    assert row["prior_buys_count"] == 0  # no prior history


def test_build_features_label_zero_for_losing_buy(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(outcome_side="YES", price=0.4, size=100.0, notional_usd=40.0)])
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=5_000,
            source="gamma",
        ),
        recorded_at=5_001,
    )

    build_features(
        trades_repo=trades, resolutions_repo=resolutions, examples_repo=examples,
        markets_conn=tmp_corpus_db, now_ts=10_000,
    )
    row = tmp_corpus_db.execute("SELECT label_won FROM training_examples").fetchone()
    assert row["label_won"] == 0


def test_build_features_skips_sells(tmp_corpus_db: sqlite3.Connection) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(bs="SELL")])
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1", winning_outcome_index=0, outcome_yes_won=1,
            resolved_at=5_000, source="gamma",
        ),
        recorded_at=5_001,
    )
    written = build_features(
        trades_repo=trades, resolutions_repo=resolutions, examples_repo=examples,
        markets_conn=tmp_corpus_db, now_ts=10_000,
    )
    assert written == 0


def test_build_features_is_incremental(tmp_corpus_db: sqlite3.Connection) -> None:
    _seed_market_metadata(tmp_corpus_db, "cond1")
    trades = CorpusTradesRepo(tmp_corpus_db)
    resolutions = MarketResolutionsRepo(tmp_corpus_db)
    examples = TrainingExamplesRepo(tmp_corpus_db)
    trades.insert_batch([_trade(tx_hash="0xa")])
    resolutions.upsert(
        MarketResolution(
            condition_id="cond1", winning_outcome_index=0, outcome_yes_won=1,
            resolved_at=5_000, source="gamma",
        ),
        recorded_at=5_001,
    )
    build_features(trades_repo=trades, resolutions_repo=resolutions, examples_repo=examples,
                   markets_conn=tmp_corpus_db, now_ts=10_000)
    # Add a new trade and re-run; only the new row lands.
    trades.insert_batch([_trade(tx_hash="0xb", ts=2_000)])
    written = build_features(trades_repo=trades, resolutions_repo=resolutions,
                             examples_repo=examples, markets_conn=tmp_corpus_db, now_ts=11_000)
    assert written == 1
    count = tmp_corpus_db.execute(
        "SELECT COUNT(*) AS c FROM training_examples"
    ).fetchone()["c"]
    assert count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_examples.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `examples.py`**

Create `src/pscanner/corpus/examples.py`:

```python
"""Streaming `build-features` orchestrator.

Walks ``corpus_trades`` chronologically, registers known resolutions up
front, drives a ``StreamingHistoryProvider``, and writes one
``TrainingExample`` per qualifying BUY whose market has resolved.

Incremental: ``INSERT OR IGNORE`` on
``(tx_hash, asset_id, wallet_address)`` makes re-runs cheap. The
streaming walk itself is full each time (sub-minute on expected corpus
size) — true watermark-incremental is deferred to v2.
"""

from __future__ import annotations

import sqlite3

import structlog

from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
    compute_features,
)
from pscanner.corpus.repos import (
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExample,
    TrainingExamplesRepo,
)

_log = structlog.get_logger(__name__)


def _load_market_metadata(conn: sqlite3.Connection) -> dict[str, MarketMetadata]:
    """Load ``MarketMetadata`` for every row in ``corpus_markets``."""
    rows = conn.execute(
        "SELECT condition_id, category, closed_at, enumerated_at FROM corpus_markets"
    ).fetchall()
    return {
        row["condition_id"]: MarketMetadata(
            condition_id=row["condition_id"],
            category=row["category"] or "unknown",
            closed_at=row["closed_at"],
            opened_at=row["enumerated_at"],
        )
        for row in rows
    }


def build_features(
    *,
    trades_repo: CorpusTradesRepo,
    resolutions_repo: MarketResolutionsRepo,
    examples_repo: TrainingExamplesRepo,
    markets_conn: sqlite3.Connection,
    now_ts: int,
    rebuild: bool = False,
) -> int:
    """Build the training_examples table from corpus_trades + resolutions.

    Args:
        trades_repo: Source of raw trades (chronological).
        resolutions_repo: Source of per-market labels.
        examples_repo: Sink for materialized rows.
        markets_conn: Connection used to load corpus_markets metadata.
        now_ts: ``built_at`` for new rows.
        rebuild: If True, drop training_examples before walking.

    Returns:
        Number of rows actually written (deduped via INSERT OR IGNORE).
    """
    if rebuild:
        examples_repo.truncate()

    metadata = _load_market_metadata(markets_conn)
    provider = StreamingHistoryProvider(metadata=metadata)

    # Pre-register all known resolutions up front so the heap can drain
    # in order during the walk.
    resolution_rows = markets_conn.execute(
        "SELECT condition_id, resolved_at, outcome_yes_won FROM market_resolutions"
    ).fetchall()
    for row in resolution_rows:
        provider.register_resolution(
            condition_id=row["condition_id"],
            resolved_at=row["resolved_at"],
            outcome_yes_won=row["outcome_yes_won"],
        )

    written = 0
    pending_examples: list[TrainingExample] = []
    BATCH = 500

    for ct in trades_repo.iter_chronological():
        meta = metadata.get(ct.condition_id)
        if meta is None:
            continue
        trade = Trade(
            tx_hash=ct.tx_hash,
            asset_id=ct.asset_id,
            wallet_address=ct.wallet_address,
            condition_id=ct.condition_id,
            outcome_side=ct.outcome_side,
            bs=ct.bs,
            price=ct.price,
            size=ct.size,
            notional_usd=ct.notional_usd,
            ts=ct.ts,
            category=meta.category,
        )

        if trade.bs == "BUY":
            resolution = resolutions_repo.get(trade.condition_id)
            if resolution is not None:
                features = compute_features(trade, provider)
                won = (
                    resolution.outcome_yes_won == 1
                    if trade.outcome_side == "YES"
                    else resolution.outcome_yes_won == 0
                )
                pending_examples.append(
                    TrainingExample(
                        tx_hash=trade.tx_hash,
                        asset_id=trade.asset_id,
                        wallet_address=trade.wallet_address,
                        condition_id=trade.condition_id,
                        trade_ts=trade.ts,
                        built_at=now_ts,
                        prior_trades_count=features.prior_trades_count,
                        prior_buys_count=features.prior_buys_count,
                        prior_resolved_buys=features.prior_resolved_buys,
                        prior_wins=features.prior_wins,
                        prior_losses=features.prior_losses,
                        win_rate=features.win_rate,
                        avg_implied_prob_paid=features.avg_implied_prob_paid,
                        realized_edge_pp=features.realized_edge_pp,
                        prior_realized_pnl_usd=features.prior_realized_pnl_usd,
                        avg_bet_size_usd=features.avg_bet_size_usd,
                        median_bet_size_usd=features.median_bet_size_usd,
                        wallet_age_days=features.wallet_age_days,
                        seconds_since_last_trade=features.seconds_since_last_trade,
                        prior_trades_30d=features.prior_trades_30d,
                        top_category=features.top_category,
                        category_diversity=features.category_diversity,
                        bet_size_usd=features.bet_size_usd,
                        bet_size_rel_to_avg=features.bet_size_rel_to_avg,
                        side=features.side,
                        implied_prob_at_buy=features.implied_prob_at_buy,
                        market_category=features.market_category,
                        market_volume_so_far_usd=features.market_volume_so_far_usd,
                        market_unique_traders_so_far=features.market_unique_traders_so_far,
                        market_age_seconds=features.market_age_seconds,
                        time_to_resolution_seconds=features.time_to_resolution_seconds,
                        last_trade_price=features.last_trade_price,
                        price_volatility_recent=features.price_volatility_recent,
                        label_won=1 if won else 0,
                    )
                )

        # Fold the trade into running state AFTER feature computation.
        provider.observe(trade)

        if len(pending_examples) >= BATCH:
            written += examples_repo.insert_or_ignore(pending_examples)
            pending_examples.clear()

    if pending_examples:
        written += examples_repo.insert_or_ignore(pending_examples)
    _log.info("corpus.build_features_complete", written=written, rebuild=rebuild)
    return written
```

- [ ] **Step 4: Verify tests pass**

Run: `uv run pytest tests/corpus/test_examples.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/pscanner/corpus tests/corpus && uv run ty check src/pscanner/corpus`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/examples.py tests/corpus/test_examples.py
git commit -m "feat(corpus): add streaming build-features orchestrator"
```

---

## Task 14: CLI commands

**Files:**
- Create: `src/pscanner/corpus/cli.py`
- Modify: `src/pscanner/cli.py` — register `corpus` subcommand
- Test: `tests/corpus/test_cli.py`

Three commands: `pscanner corpus backfill`, `pscanner corpus refresh`, `pscanner corpus build-features`. Each opens `data/corpus.sqlite3` (path from a config knob, default `Path("data/corpus.sqlite3")`), instantiates clients with their own rate budget, and runs the orchestration.

- [ ] **Step 1: Write failing tests**

Create `tests/corpus/test_cli.py`:

```python
"""Tests for the `pscanner corpus` CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pscanner.corpus.cli import build_corpus_parser, run_corpus_command


def test_parser_recognises_three_commands() -> None:
    parser = build_corpus_parser()
    assert parser.parse_args(["backfill"]).command == "backfill"
    assert parser.parse_args(["refresh"]).command == "refresh"
    assert parser.parse_args(["build-features"]).command == "build-features"


def test_parser_supports_rebuild_flag() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--rebuild"])
    assert args.rebuild is True


@pytest.mark.asyncio
async def test_backfill_command_smokes(tmp_path: Path) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    fake_enumerate = AsyncMock(return_value=0)
    fake_drain = AsyncMock(return_value=0)
    with (
        patch("pscanner.corpus.cli.enumerate_closed_markets", fake_enumerate),
        patch("pscanner.corpus.cli._drain_pending", fake_drain),
        patch("pscanner.corpus.cli._make_data_client") as mk_data,
        patch("pscanner.corpus.cli._make_gamma_client") as mk_gamma,
    ):
        mk_data.return_value.__aenter__.return_value = AsyncMock()
        mk_gamma.return_value.__aenter__.return_value = AsyncMock()
        rc = await run_corpus_command(["backfill", "--db", str(db_path)])
    assert rc == 0
    fake_enumerate.assert_awaited()
    fake_drain.assert_awaited()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_cli.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `cli.py`**

Create `src/pscanner/corpus/cli.py`:

```python
"""argparse handlers for ``pscanner corpus {backfill, refresh, build-features}``.

Each handler opens ``corpus.sqlite3``, instantiates the gamma + data
clients with their own rate budget, runs the orchestration, and exits
with 0 on success.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from contextlib import AsyncExitStack
from pathlib import Path

import structlog

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.enumerator import enumerate_closed_markets
from pscanner.corpus.examples import build_features
from pscanner.corpus.market_walker import walk_market
from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusStateRepo,
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)
from pscanner.corpus.resolutions import record_resolutions
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)


def build_corpus_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pscanner corpus")
    parser.add_argument("--db", default="data/corpus.sqlite3", type=str)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backfill", help="Bulk historical pull of every closed qualifying market")
    sub.add_parser("refresh", help="Incremental pass for newly-resolved markets")
    bf = sub.add_parser("build-features", help="Rebuild training_examples from raw events")
    bf.add_argument("--rebuild", action="store_true", help="Drop and recreate the table")
    return parser


class _GammaCM:
    """Async context manager that owns a fresh GammaClient + closes it."""

    async def __aenter__(self) -> GammaClient:
        self._client = GammaClient(rpm=50)
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()


class _DataCM:
    """Async context manager that owns a fresh DataClient + closes it."""

    async def __aenter__(self) -> DataClient:
        self._client = DataClient(rpm=50)
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()


def _make_gamma_client() -> _GammaCM:
    return _GammaCM()


def _make_data_client() -> _DataCM:
    return _DataCM()


async def _drain_pending(*, conn, data: DataClient) -> int:  # noqa: ANN001
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    total = 0
    while True:
        batch = markets_repo.next_pending(limit=10)
        if not batch:
            return total
        for m in batch:
            try:
                inserted = await walk_market(
                    condition_id=m.condition_id,
                    data=data,
                    markets_repo=markets_repo,
                    trades_repo=trades_repo,
                    now_ts=int(time.time()),
                )
                total += inserted
            except Exception as exc:  # noqa: BLE001
                _log.warning("corpus.walk_failed", condition_id=m.condition_id, error=str(exc))


async def _cmd_backfill(args: argparse.Namespace) -> int:
    conn = init_corpus_db(Path(args.db))
    try:
        async with AsyncExitStack() as stack:
            gamma = await stack.enter_async_context(_make_gamma_client())
            data = await stack.enter_async_context(_make_data_client())
            await enumerate_closed_markets(
                gamma=gamma,
                repo=CorpusMarketsRepo(conn),
                now_ts=int(time.time()),
                since_ts=None,
            )
            await _drain_pending(conn=conn, data=data)
        return 0
    finally:
        conn.close()


async def _cmd_refresh(args: argparse.Namespace) -> int:
    conn = init_corpus_db(Path(args.db))
    try:
        state = CorpusStateRepo(conn)
        async with AsyncExitStack() as stack:
            gamma = await stack.enter_async_context(_make_gamma_client())
            data = await stack.enter_async_context(_make_data_client())
            since_ts = state.get_int("last_gamma_sweep_ts")
            await enumerate_closed_markets(
                gamma=gamma,
                repo=CorpusMarketsRepo(conn),
                now_ts=int(time.time()),
                since_ts=since_ts,
            )
            await _drain_pending(conn=conn, data=data)
            # Resolve any markets without a resolution row.
            markets_repo = CorpusMarketsRepo(conn)
            res_repo = MarketResolutionsRepo(conn)
            rows = conn.execute(
                """
                SELECT m.condition_id, m.event_slug, m.closed_at
                FROM corpus_markets m
                LEFT JOIN market_resolutions r USING (condition_id)
                WHERE r.condition_id IS NULL AND m.backfill_state = 'complete'
                """
            ).fetchall()
            await record_resolutions(
                gamma=gamma,
                repo=res_repo,
                targets=[(r["condition_id"], r["event_slug"] + "-slug", r["closed_at"]) for r in rows],
                now_ts=int(time.time()),
            )
            state.set("last_gamma_sweep_ts", str(int(time.time())), updated_at=int(time.time()))
            del markets_repo
        return 0
    finally:
        conn.close()


async def _cmd_build_features(args: argparse.Namespace) -> int:
    conn = init_corpus_db(Path(args.db))
    try:
        written = build_features(
            trades_repo=CorpusTradesRepo(conn),
            resolutions_repo=MarketResolutionsRepo(conn),
            examples_repo=TrainingExamplesRepo(conn),
            markets_conn=conn,
            now_ts=int(time.time()),
            rebuild=bool(getattr(args, "rebuild", False)),
        )
        _log.info("corpus.build_features_done", written=written)
        return 0
    finally:
        conn.close()


_HANDLERS = {
    "backfill": _cmd_backfill,
    "refresh": _cmd_refresh,
    "build-features": _cmd_build_features,
}


async def run_corpus_command(argv: list[str]) -> int:
    parser = build_corpus_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    return await handler(args)
```

- [ ] **Step 4: Wire `corpus` into `pscanner.cli`**

Modify `src/pscanner/cli.py`. Find the subparser registration block and add a new branch. Specifically, near the existing subparser setup, add:

```python
sub.add_parser("corpus", add_help=False, help="Historical trade corpus subcommands")
```

Then in the main dispatch logic, when `args.command == "corpus"`, hand the remaining argv to `pscanner.corpus.cli.run_corpus_command` and `asyncio.run` it.

The exact integration depends on how `pscanner.cli.main` currently dispatches. Read `src/pscanner/cli.py` first; pattern-match the `paper status` integration.

Concretely:

```python
# Imports near top of pscanner/cli.py
from pscanner.corpus.cli import run_corpus_command

# In main(), after parsing main subcommands:
if args.command == "corpus":
    # Pass argv remainder (everything after "corpus") to the corpus parser.
    return asyncio.run(run_corpus_command(corpus_argv))
```

Use `parse_known_args` if needed to capture remaining args.

- [ ] **Step 5: Verify CLI tests pass**

Run: `uv run pytest tests/corpus/test_cli.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Smoke the CLI invocation**

Run: `uv run pscanner corpus build-features --db /tmp/corpus_smoke.sqlite3`
Expected: exit 0 with empty corpus (writes 0 rows). Then `rm /tmp/corpus_smoke.sqlite3`.

- [ ] **Step 7: Lint + type + full suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/pscanner/corpus/cli.py src/pscanner/cli.py tests/corpus/test_cli.py
git commit -m "feat(corpus): wire backfill, refresh, build-features CLI commands"
```

---

## Task 15: End-to-end integration test

**Files:**
- Test: `tests/corpus/test_integration_e2e.py`

Stitches everything together with `respx`-mocked gamma and data endpoints. Drives `backfill` → assert `corpus_trades` populated; `build-features` → assert `training_examples` populated; re-run `build-features` → assert idempotency (zero new rows).

- [ ] **Step 1: Write the integration test**

Create `tests/corpus/test_integration_e2e.py`:

```python
"""End-to-end smoke for `pscanner corpus` against respx-mocked APIs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import respx
from httpx import Response

from pscanner.corpus.cli import run_corpus_command


def _events_payload() -> list[dict[str, object]]:
    return [
        {
            "id": "evt1",
            "title": "Event 1",
            "slug": "evt1",
            "active": False,
            "closed": True,
            "tags": [{"label": "Crypto", "slug": "crypto"}],
            "markets": [
                {
                    "id": "cond1",
                    "conditionId": "cond1",
                    "question": "?",
                    "slug": "cond1-slug",
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["1.0","0.0"]',
                    "volume": 50_000.0,
                    "active": False,
                    "closed": True,
                }
            ],
        }
    ]


def _trades_page() -> list[dict[str, object]]:
    return [
        {
            "transactionHash": "0xa",
            "asset": "asset1",
            "proxyWallet": "0xWALLET",
            "conditionId": "cond1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.4,
            "size": 100.0,
            "timestamp": 1_000,
        }
    ]


@pytest.mark.asyncio
async def test_corpus_backfill_then_build_features_e2e(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    with respx.mock(assert_all_called=False) as rx:
        rx.get("https://gamma-api.polymarket.com/events").mock(
            side_effect=lambda req: Response(
                200,
                json=_events_payload() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        rx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=Response(
                200,
                json=[_events_payload()[0]["markets"][0]],
            )
        )
        rx.get("https://data-api.polymarket.com/trades").mock(
            side_effect=lambda req: Response(
                200,
                json=_trades_page() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )

        rc_backfill = await run_corpus_command(["backfill", "--db", str(db)])
        assert rc_backfill == 0
        rc_build = await run_corpus_command(["build-features", "--db", str(db)])
        assert rc_build == 0

    # Inspect the resulting DB.
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        trades = conn.execute("SELECT COUNT(*) AS c FROM corpus_trades").fetchone()["c"]
        examples = conn.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
    finally:
        conn.close()
    assert trades == 1
    assert examples == 1


@pytest.mark.asyncio
async def test_corpus_build_features_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    with respx.mock(assert_all_called=False) as rx:
        rx.get("https://gamma-api.polymarket.com/events").mock(
            side_effect=lambda req: Response(
                200,
                json=_events_payload() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        rx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=Response(200, json=[_events_payload()[0]["markets"][0]])
        )
        rx.get("https://data-api.polymarket.com/trades").mock(
            side_effect=lambda req: Response(
                200,
                json=_trades_page() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        await run_corpus_command(["backfill", "--db", str(db)])
        await run_corpus_command(["build-features", "--db", str(db)])
        await run_corpus_command(["build-features", "--db", str(db)])
    conn = sqlite3.connect(str(db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    finally:
        conn.close()
    assert count == 1  # second build-features did NOT add a duplicate
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/corpus/test_integration_e2e.py -v`
Expected: both tests PASS. If they fail, debug — the integration test is the canary that the seam wiring is correct.

- [ ] **Step 3: Run the full quick-verify**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/corpus/test_integration_e2e.py
git commit -m "test(corpus): end-to-end smoke for backfill + build-features"
```

---

## Final verification checklist

Before declaring v1 complete:

- [ ] `uv run pytest tests/corpus -v` — all corpus tests pass
- [ ] `uv run pytest -q` — full suite passes (no regressions in existing tests)
- [ ] `uv run ruff check . && uv run ruff format --check .` — clean
- [ ] `uv run ty check` — clean
- [ ] `uv run pscanner corpus build-features --db /tmp/empty.sqlite3` — exit 0 on an empty DB
- [ ] `git log --oneline | head -20` — one commit per task, in order
- [ ] CLAUDE.md updated with corpus-specific notes (volume gate, $10 trade floor, separate DB file, three CLI commands) — **append a "Corpus pipeline (v1)" section near the bottom**

After all 15 tasks land, do a final `pscanner corpus build-features --rebuild` smoke against a small live backfill (e.g., 5-10 markets) to confirm shape of `training_examples` matches the spec.


