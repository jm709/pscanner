# Kalshi ingestion (corpus L1+L2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Kalshi REST through a new corpus enumerator + trades walker + resolutions function into the platform-aware corpus, exposing `pscanner corpus backfill / refresh --platform kalshi`. Anonymous L1+L2 only — no `training_examples` because Kalshi public REST trades carry no taker identity. The L3-enabling social-API path is tracked separately as #95.

**Architecture:** Two new files (`kalshi_enumerator.py`, `kalshi_walker.py`) call the existing `KalshiClient` and write into the platform-aware `CorpusMarketsRepo` / `CorpusTradesRepo`. Resolutions extend `pscanner.corpus.resolutions` with a Kalshi-flavored function reading the `result` field on `/markets/{ticker}` (verified via OpenAPI spec). One module gap closes: `KalshiMarket` gains a `result: str | None` field plus a matching `kalshi_markets.result` column with idempotent migration. Mirrors PR #84 (Manifold ingestion) closely.

**Tech Stack:** Python 3.13, async httpx + tenacity (already used by `KalshiClient`), pydantic, pytest. Quick verify: `cd /home/macph/projects/pscanner-worktrees/kalshi-clean && uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Worktree:** `/home/macph/projects/pscanner-worktrees/kalshi-clean` (branch `feat/kalshi-ingestion`, off `origin/main` plus the spec commit). All commits in this plan land on that branch in that worktree. Do NOT touch the main checkout at `/home/macph/projects/polymarketScanner`.

**Spec:** `docs/superpowers/specs/2026-05-07-kalshi-ingestion-design.md`

---

## File map

**New:**
- `src/pscanner/corpus/kalshi_enumerator.py`
- `src/pscanner/corpus/kalshi_walker.py`
- `tests/corpus/test_kalshi_enumerator.py`
- `tests/corpus/test_kalshi_walker.py`
- `tests/corpus/test_kalshi_e2e.py`

**Modify:**
- `src/pscanner/kalshi/models.py` (add `result` field to `KalshiMarket`)
- `src/pscanner/kalshi/db.py` (add `result TEXT` column + idempotent migration helper)
- `src/pscanner/kalshi/repos.py` (round-trip `result`)
- `src/pscanner/corpus/resolutions.py` (`record_kalshi_resolutions`)
- `src/pscanner/corpus/cli.py` (`--platform kalshi` flag + dispatch on `backfill`/`refresh`)
- `tests/kalshi/test_models.py`, `tests/kalshi/test_db.py`, `tests/kalshi/test_repos.py` (extend)
- `tests/corpus/test_resolutions.py`, `tests/corpus/test_cli.py` (extend)
- `CLAUDE.md` (Kalshi ingestion bullet)

Roughly 5 new files, 10 modified. ~500 lines source, ~500 lines tests.

---

### Task 1: Add `result` field to `KalshiMarket` model

**Files:**
- Modify: `src/pscanner/kalshi/models.py:31-79`
- Modify: `tests/kalshi/test_models.py`

The Kalshi REST `/markets/{ticker}` returns `result: 'yes' | 'no' | 'scalar' | ''` on every market response (per OpenAPI spec). Stage 1 didn't capture it.

- [ ] **Step 1: Verify branch state**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git status
git branch --show-current   # must be feat/kalshi-ingestion
git log --oneline -1        # must end at the spec commit (11f8896 or whatever was cherry-picked there)
```

If anything else, STOP.

- [ ] **Step 2: Write the failing test**

Add to `tests/kalshi/test_models.py`:

```python
import pytest

from pscanner.kalshi.models import KalshiMarket


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("yes", "yes"),
        ("no", "no"),
        ("scalar", "scalar"),
        ("", ""),
    ],
)
def test_kalshi_market_parses_result_field(
    raw_value: str, expected: str
) -> None:
    """`result` round-trips through validation for all four documented values."""
    market = KalshiMarket.model_validate(
        {
            "ticker": "KX-1",
            "event_ticker": "KX",
            "title": "Q?",
            "status": "finalized",
            "result": raw_value,
        }
    )
    assert market.result == expected


def test_kalshi_market_result_defaults_to_none_when_absent() -> None:
    """Active markets that haven't settled omit the field; the model defaults to None."""
    market = KalshiMarket.model_validate(
        {
            "ticker": "KX-1",
            "event_ticker": "KX",
            "title": "Q?",
            "status": "active",
        }
    )
    assert market.result is None
```

If `pytest` isn't already imported at module top, hoist it (avoid PLC0415).

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/kalshi/test_models.py::test_kalshi_market_parses_result_field -v
```
Expected: fail — `KalshiMarket` has no `result` attribute.

- [ ] **Step 4: Add the field**

In `src/pscanner/kalshi/models.py`, add `result: str | None = None` to `KalshiMarket`. Place it after `no_sub_title` and before the price fields:

```python
class KalshiMarket(BaseModel):
    """A single Kalshi binary market.

    Maps to one row in the ``kalshi_markets`` table.
    """

    model_config = _BASE_CONFIG

    ticker: str
    event_ticker: str
    title: str
    status: str
    market_type: str = ""
    open_time: str = ""
    close_time: str = ""
    expected_expiration_time: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    result: str | None = None

    # Prices are returned as dollar strings like "0.0900"; coerce to float.
    last_price_dollars: Annotated[float, Field(default=0.0)]
    # ... rest unchanged ...
```

The field is unaliased — Kalshi's wire format uses snake_case `result` natively.

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/kalshi/test_models.py -v
```
Expected: all pass (existing + 5 new).

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/kalshi/models.py tests/kalshi/test_models.py
git commit -m "feat(kalshi): parse result field on KalshiMarket model"
git branch --show-current   # verify still feat/kalshi-ingestion
```

---

### Task 2: Add `result` column to `kalshi_markets` schema

**Files:**
- Modify: `src/pscanner/kalshi/db.py`
- Modify: `tests/kalshi/test_db.py`

- [ ] **Step 1: Verify branch state**

Same as Task 1 step 1. HEAD should be the Task 1 commit.

- [ ] **Step 2: Write the failing test**

Add to `tests/kalshi/test_db.py`:

```python
import sqlite3


def test_kalshi_markets_has_result_column() -> None:
    """`kalshi_markets` exposes a nullable TEXT `result` column."""
    from pscanner.kalshi.db import KALSHI_SCHEMA_STATEMENTS, init_kalshi_tables

    conn = sqlite3.connect(":memory:")
    try:
        init_kalshi_tables(conn)
        info = conn.execute("PRAGMA table_info(kalshi_markets)").fetchall()
        cols = {row[1]: row for row in info}
        assert "result" in cols
        assert cols["result"][2].upper() == "TEXT"
        assert cols["result"][3] == 0, "result must be nullable"
    finally:
        conn.close()


def test_init_kalshi_tables_idempotent_on_result_column() -> None:
    """Calling init_kalshi_tables twice leaves the result column intact."""
    from pscanner.kalshi.db import init_kalshi_tables

    conn = sqlite3.connect(":memory:")
    try:
        init_kalshi_tables(conn)
        init_kalshi_tables(conn)
        info = conn.execute("PRAGMA table_info(kalshi_markets)").fetchall()
        cols = {row[1] for row in info}
        assert "result" in cols
    finally:
        conn.close()
```

If `tests/kalshi/test_db.py` doesn't exist yet, create it. The `init_kalshi_tables` function may not exist yet either — check `src/pscanner/kalshi/db.py`. The existing module exports `KALSHI_SCHEMA_STATEMENTS` (used by `pscanner.store.db.init_db`) but doesn't define a standalone `init_kalshi_tables(conn)`. We add one as part of step 4 below — Manifold has the analogous pattern.

If `import sqlite3` is already at module top, don't duplicate. Hoist all imports to the top.

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/kalshi/test_db.py -v
```
Expected: fail — `init_kalshi_tables` doesn't exist (or the column is missing if it does exist).

- [ ] **Step 4: Add the column to fresh-DB schema, add migration, add init function**

In `src/pscanner/kalshi/db.py`, modify the `kalshi_markets` `CREATE TABLE` statement to add `result` between `no_sub_title` and `last_price_cents`:

```python
KALSHI_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS kalshi_markets (
      ticker                    TEXT PRIMARY KEY,
      event_ticker              TEXT NOT NULL,
      title                     TEXT NOT NULL,
      status                    TEXT NOT NULL,
      market_type               TEXT NOT NULL DEFAULT '',
      open_time                 TEXT NOT NULL DEFAULT '',
      close_time                TEXT NOT NULL DEFAULT '',
      expected_expiration_time  TEXT NOT NULL DEFAULT '',
      yes_sub_title             TEXT NOT NULL DEFAULT '',
      no_sub_title              TEXT NOT NULL DEFAULT '',
      result                    TEXT,
      last_price_cents          INTEGER NOT NULL DEFAULT 0,
      yes_bid_cents             INTEGER NOT NULL DEFAULT 0,
      yes_ask_cents             INTEGER NOT NULL DEFAULT 0,
      no_bid_cents              INTEGER NOT NULL DEFAULT 0,
      no_ask_cents              INTEGER NOT NULL DEFAULT 0,
      volume_fp                 REAL NOT NULL DEFAULT 0.0,
      volume_24h_fp             REAL NOT NULL DEFAULT 0.0,
      open_interest_fp          REAL NOT NULL DEFAULT 0.0,
      cached_at                 INTEGER NOT NULL
    )
    """,
    # ... unchanged kalshi_trades, kalshi_orderbook_snapshots, indexes ...
)
```

Add the migration tuple, helper, and `init_kalshi_tables` function below `KALSHI_SCHEMA_STATEMENTS`:

```python
import sqlite3

_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE kalshi_markets ADD COLUMN result TEXT",
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations. Idempotent.

    Each migration is wrapped to swallow ``duplicate column name`` errors
    so repeated calls on already-migrated DBs are no-ops. Mirrors
    ``pscanner.manifold.db._apply_migrations``.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                continue
            raise
    conn.commit()


def init_kalshi_tables(conn: sqlite3.Connection) -> None:
    """Apply all Kalshi schema statements + migrations to ``conn``.

    Idempotent — safe to call on an already-initialised database. Mirrors
    ``pscanner.manifold.db.init_kalshi_tables``.

    Note: the daemon's ``pscanner.store.db.init_db`` already concatenates
    ``KALSHI_SCHEMA_STATEMENTS`` into its own schema. This standalone helper
    is for tests and any code path that wants to apply Kalshi schema in
    isolation.
    """
    for statement in KALSHI_SCHEMA_STATEMENTS:
        conn.execute(statement)
    _apply_migrations(conn)
    conn.commit()
```

Also, in the daemon's `pscanner/store/db.py`, find the loop that runs `KALSHI_SCHEMA_STATEMENTS` and add a call to `_apply_migrations` from `pscanner.kalshi.db` so the daemon's `init_db` also picks up the migration. The simplest approach: just call `init_kalshi_tables(conn)` from `init_db` instead of inlining the statements (mirrors how Manifold was wired in #84). Read the existing `init_db` shape first; the change is one line — replace the explicit `KALSHI_SCHEMA_STATEMENTS` loop with `init_kalshi_tables(conn)`.

If the existing pattern is `for stmt in (*KALSHI_SCHEMA_STATEMENTS, *MANIFOLD_SCHEMA_STATEMENTS, ...)`, leave that alone and instead call `_apply_migrations` explicitly from `init_db` after the schema loop. Read the file to see the actual shape.

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/kalshi/test_db.py tests/store/ -v
```
Expected: all pass — both the Kalshi tests and the daemon `init_db` tests.

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/kalshi/db.py src/pscanner/store/db.py tests/kalshi/test_db.py
git commit -m "feat(kalshi): add result column to kalshi_markets + idempotent migration"
git branch --show-current
```

---

### Task 3: Round-trip `result` through `KalshiMarketsRepo.upsert`

**Files:**
- Modify: `src/pscanner/kalshi/repos.py:80-122`
- Modify: `tests/kalshi/test_repos.py`

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 2 commit.

- [ ] **Step 2: Write the failing test**

Add to `tests/kalshi/test_repos.py`:

```python
def test_kalshi_markets_repo_roundtrips_result_field(
    tmp_kalshi_conn,  # type: ignore[no-untyped-def]
) -> None:
    """`upsert` writes the result column; the value survives round-trip."""
    from pscanner.kalshi.models import KalshiMarket
    from pscanner.kalshi.repos import KalshiMarketsRepo

    repo = KalshiMarketsRepo(tmp_kalshi_conn)
    market = KalshiMarket.model_validate(
        {
            "ticker": "KX-1",
            "event_ticker": "KX",
            "title": "Q?",
            "status": "finalized",
            "result": "yes",
        }
    )
    repo.upsert(market)
    row = tmp_kalshi_conn.execute(
        "SELECT result FROM kalshi_markets WHERE ticker = ?", (market.ticker,)
    ).fetchone()
    assert row[0] == "yes"
```

If `tmp_kalshi_conn` doesn't exist as a fixture, add it inline at the top of `test_repos.py` or in `tests/kalshi/conftest.py`:

```python
from collections.abc import Iterator

import pytest

from pscanner.kalshi.db import init_kalshi_tables


@pytest.fixture
def tmp_kalshi_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        init_kalshi_tables(conn)
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/kalshi/test_repos.py::test_kalshi_markets_repo_roundtrips_result_field -v
```
Expected: fail — `upsert` doesn't include `result` in the INSERT, so the column is NULL.

- [ ] **Step 4: Update `upsert`**

In `src/pscanner/kalshi/repos.py`, modify `KalshiMarketsRepo.upsert` to include `result` in the INSERT column list and values tuple. Place the new column between `no_sub_title` and `last_price_cents` to match the schema order:

```python
    def upsert(self, market: KalshiMarket) -> None:
        """Insert or replace a market, refreshing ``cached_at``.

        Args:
            market: Validated :class:`~pscanner.kalshi.models.KalshiMarket`.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO kalshi_markets (
              ticker, event_ticker, title, status, market_type,
              open_time, close_time, expected_expiration_time,
              yes_sub_title, no_sub_title, result,
              last_price_cents, yes_bid_cents, yes_ask_cents,
              no_bid_cents, no_ask_cents,
              volume_fp, volume_24h_fp, open_interest_fp, cached_at
            ) VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                market.ticker,
                market.event_ticker,
                market.title,
                market.status,
                market.market_type,
                market.open_time,
                market.close_time,
                market.expected_expiration_time,
                market.yes_sub_title,
                market.no_sub_title,
                market.result,
                market.last_price_cents,
                market.yes_bid_cents,
                market.yes_ask_cents,
                market.no_bid_cents,
                market.no_ask_cents,
                market.volume_fp,
                market.volume_24h_fp,
                market.open_interest_fp,
                _now_seconds(),
            ),
        )
        self._conn.commit()
```

Placeholder count: 20 (was 19). Verify by counting `?` marks.

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/kalshi/test_repos.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/kalshi/repos.py tests/kalshi/test_repos.py
git commit -m "feat(kalshi): write result column from KalshiMarketsRepo.upsert"
git branch --show-current
```

---

### Task 4: Kalshi corpus enumerator

**Files:**
- Create: `src/pscanner/corpus/kalshi_enumerator.py`
- Create: `tests/corpus/test_kalshi_enumerator.py`

Walks `/markets?status=...` paginated for each terminal status, filters resolved+binary+above-volume, inserts into `corpus_markets` with `platform='kalshi'`. Mirrors `manifold_enumerator.py`.

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 3 commit.

- [ ] **Step 2: Write the failing test**

Create `tests/corpus/test_kalshi_enumerator.py`:

```python
"""Tests for the Kalshi corpus enumerator."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.models import KalshiMarket, KalshiMarketsPage


class _FakeKalshiClient:
    """Stub returning fixed pages keyed by status, ignoring cursor."""

    def __init__(self, pages_by_status: dict[str, list[list[KalshiMarket]]]) -> None:
        self._pages_by_status = pages_by_status
        self._call_counts: dict[str, int] = {s: 0 for s in pages_by_status}

    async def get_markets(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiMarketsPage:
        if status is None or status not in self._pages_by_status:
            return KalshiMarketsPage(markets=[], cursor="")
        idx = self._call_counts[status]
        pages = self._pages_by_status[status]
        if idx >= len(pages):
            return KalshiMarketsPage(markets=[], cursor="")
        self._call_counts[status] = idx + 1
        page = pages[idx]
        next_cursor = "next" if idx + 1 < len(pages) else ""
        return KalshiMarketsPage(markets=page, cursor=next_cursor)


def _market(
    *,
    ticker: str,
    status: str = "finalized",
    result: str = "yes",
    market_type: str = "binary",
    volume_fp: float = 50_000.0,
    close_time: str = "2026-05-04T12:00:00Z",
    event_ticker: str = "KX",
) -> KalshiMarket:
    return KalshiMarket.model_validate(
        {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "title": f"Q for {ticker}",
            "status": status,
            "market_type": market_type,
            "result": result,
            "volume_fp": volume_fp,
            "close_time": close_time,
        }
    )


@pytest.mark.asyncio
async def test_enumerate_inserts_only_qualifying_kalshi_markets(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """A page with mixed conditions yields only resolved+binary+above-volume rows."""
    from pscanner.corpus.kalshi_enumerator import enumerate_resolved_kalshi_markets

    pages_by_status = {
        "determined": [[
            _market(ticker="keep1", status="determined", result="yes"),
            _market(ticker="scalar-typed", market_type="scalar", result="yes"),
            _market(ticker="lowvol", volume_fp=1000.0, result="no"),
            _market(ticker="empty-result", result=""),
        ]],
        "amended": [[
            _market(ticker="keep2", status="amended", result="no"),
        ]],
        "finalized": [[
            _market(ticker="keep3", status="finalized", result="yes"),
        ]],
    }
    client = _FakeKalshiClient(pages_by_status)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_kalshi_markets(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
        min_volume_contracts=10_000.0,
    )
    assert inserted == 3
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets WHERE platform = 'kalshi' "
        "ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["keep1", "keep2", "keep3"]


@pytest.mark.asyncio
async def test_enumerate_paginates_until_cursor_empty(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Multiple pages within a status are walked until cursor is empty."""
    from pscanner.corpus.kalshi_enumerator import enumerate_resolved_kalshi_markets

    pages_by_status = {
        "determined": [
            [_market(ticker="d_a"), _market(ticker="d_b")],
            [_market(ticker="d_c")],
        ],
        "amended": [],
        "finalized": [],
    }
    client = _FakeKalshiClient(pages_by_status)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_kalshi_markets(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
    )
    assert inserted == 3


@pytest.mark.asyncio
async def test_enumerate_idempotent_on_rerun(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Re-running the enumerator over the same markets inserts zero new rows."""
    from pscanner.corpus.kalshi_enumerator import enumerate_resolved_kalshi_markets

    repo = CorpusMarketsRepo(tmp_corpus_db)
    pages_by_status = {
        "determined": [[_market(ticker="m1"), _market(ticker="m2")]],
        "amended": [],
        "finalized": [],
    }
    client1 = _FakeKalshiClient(pages_by_status)
    first = await enumerate_resolved_kalshi_markets(client1, repo, now_ts=2_000_000_000)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    assert first == 2

    pages_by_status_2 = {
        "determined": [[_market(ticker="m1"), _market(ticker="m2")]],
        "amended": [],
        "finalized": [],
    }
    client2 = _FakeKalshiClient(pages_by_status_2)
    second = await enumerate_resolved_kalshi_markets(client2, repo, now_ts=2_000_000_001)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    assert second == 0
```

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_kalshi_enumerator.py -v
```
Expected: 3 fails — module doesn't exist.

- [ ] **Step 4: Create the enumerator**

Create `src/pscanner/corpus/kalshi_enumerator.py`:

```python
"""Corpus enumerator for settled Kalshi markets.

Walks ``/markets?status=<status>&cursor=...`` for each terminal status
(``determined``, ``amended``, ``finalized``), filters to binary markets with
a clean ``yes``/``no`` result above the volume gate, and inserts
``(platform='kalshi')`` rows into ``corpus_markets``. Idempotent — repeated
runs are no-ops on already-known markets thanks to
``CorpusMarketsRepo.insert_pending``'s ``INSERT OR IGNORE`` semantics.

The three-pass status walk is necessary because Kalshi's ``/markets`` filter
takes a single ``status`` value at a time. ``disputed`` is intentionally
skipped — contested resolutions land on a future refresh once Kalshi moves
them to a clean terminal state.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.models import KalshiMarket

_log = structlog.get_logger(__name__)

_TERMINAL_STATUSES: tuple[str, ...] = ("determined", "amended", "finalized")


async def enumerate_resolved_kalshi_markets(
    client: KalshiClient,
    repo: CorpusMarketsRepo,
    *,
    now_ts: int,
    min_volume_contracts: float = 10_000.0,
    page_size: int = 100,
) -> int:
    """Walk Kalshi markets and insert qualifying rows into corpus_markets.

    Iterates each terminal status (`determined`, `amended`, `finalized`)
    via cursor pagination on ``/markets?status=<value>``. Skips ``disputed``
    (contested resolution) and ``closed`` (trading halted, no determination yet).

    Args:
        client: Open ``KalshiClient`` with rate-limit budget available.
        repo: Corpus markets repo bound to a platform-aware corpus DB.
        now_ts: Unix seconds, recorded as ``enumerated_at`` on each row.
        min_volume_contracts: Minimum ``KalshiMarket.volume_fp`` to qualify.
            Contract count, not USD. Defaults to 10_000.
        page_size: ``limit`` parameter on ``client.get_markets``.

    Returns:
        Count of newly-inserted ``corpus_markets`` rows. Does not include
        rows that already existed (idempotent re-enumeration).
    """
    inserted_total = 0
    examined_total = 0
    for status in _TERMINAL_STATUSES:
        cursor: str | None = None
        while True:
            page = await client.get_markets(
                status=status, limit=page_size, cursor=cursor
            )
            if not page.markets:
                break
            examined_total += len(page.markets)
            for market in page.markets:
                if not _qualifies(market, min_volume_contracts=min_volume_contracts):
                    continue
                corpus_market = _to_corpus_market(market, now_ts=now_ts)
                inserted_total += repo.insert_pending(corpus_market)
            if not page.cursor:
                break
            cursor = page.cursor
    _log.info(
        "kalshi.enumerate_complete",
        examined=examined_total,
        inserted=inserted_total,
        min_volume_contracts=min_volume_contracts,
    )
    return inserted_total


def _qualifies(
    market: KalshiMarket, *, min_volume_contracts: float
) -> bool:
    """True iff the market should land in the corpus."""
    return (
        market.market_type == "binary"
        and market.result in ("yes", "no")
        and market.volume_fp >= min_volume_contracts
    )


def _to_corpus_market(market: KalshiMarket, *, now_ts: int) -> CorpusMarket:
    """Project a ``KalshiMarket`` into the corpus dataclass.

    ``closed_at`` is parsed from ``market.close_time`` (ISO datetime) into
    epoch seconds. Falls back to ``now_ts`` if parsing fails. The corpus
    invariant is that ``mark_complete`` rewrites ``closed_at`` to
    ``MAX(corpus_trades.ts)`` after the walker runs, so this initial value is
    a placeholder anyway.
    """
    return CorpusMarket(
        condition_id=market.ticker,
        event_slug=market.event_ticker,
        category=market.market_type,
        closed_at=_iso_to_epoch(market.close_time, fallback=now_ts),
        total_volume_usd=market.volume_fp,
        enumerated_at=now_ts,
        market_slug=market.ticker,
        platform="kalshi",
    )


def _iso_to_epoch(iso: str, *, fallback: int) -> int:
    """Parse an ISO 8601 datetime string to epoch seconds.

    Returns ``fallback`` if the input is empty or unparseable. Kalshi wire
    format is ``"2026-05-04T12:00:00Z"``; ``datetime.fromisoformat`` handles
    the trailing ``Z`` since Python 3.11.
    """
    if not iso:
        return fallback
    try:
        # Python 3.11+ handles "Z" via fromisoformat directly.
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())
```

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_kalshi_enumerator.py -v
```
Expected: 3/3 pass.

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/corpus/kalshi_enumerator.py tests/corpus/test_kalshi_enumerator.py
git commit -m "feat(corpus): Kalshi corpus enumerator for settled binary markets"
git branch --show-current
```

---

### Task 5: Kalshi trades walker

**Files:**
- Create: `src/pscanner/corpus/kalshi_walker.py`
- Create: `tests/corpus/test_kalshi_walker.py`

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 4 commit.

- [ ] **Step 2: Write the failing tests**

Create `tests/corpus/test_kalshi_walker.py`:

```python
"""Tests for the Kalshi per-market trades walker."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTradesRepo,
)
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import KalshiTrade, KalshiTradesPage


class _FakeKalshiClient:
    def __init__(self, pages: list[list[KalshiTrade]]) -> None:
        self._pages = pages
        self._call_count = 0

    async def get_market_trades(
        self,
        *,
        ticker: KalshiMarketTicker,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiTradesPage:
        if self._call_count >= len(self._pages):
            return KalshiTradesPage(trades=[], cursor="")
        idx = self._call_count
        self._call_count += 1
        page = self._pages[idx]
        next_cursor = "next" if idx + 1 < len(self._pages) else ""
        return KalshiTradesPage(trades=page, cursor=next_cursor)


def _trade(
    *,
    trade_id: str,
    ticker: str = "KX-1",
    taker_side: str = "yes",
    yes_price: float = 0.50,
    no_price: float = 0.50,
    count: float = 100.0,
    created_time: str = "2026-05-04T12:00:00Z",
) -> KalshiTrade:
    return KalshiTrade.model_validate(
        {
            "trade_id": trade_id,
            "ticker": ticker,
            "taker_side": taker_side,
            "yes_price_dollars": yes_price,
            "no_price_dollars": no_price,
            "count_fp": count,
            "created_time": created_time,
        }
    )


def _seed_market(
    repo: CorpusMarketsRepo, *, ticker: str = "KX-1"
) -> None:
    repo.insert_pending(
        CorpusMarket(
            condition_id=ticker,
            event_slug="KX",
            category="binary",
            closed_at=1_700_000_000,
            total_volume_usd=50_000.0,
            enumerated_at=1_700_000_000 - 1,
            market_slug=ticker,
            platform="kalshi",
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_trades_with_kalshi_platform(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Trades land in corpus_trades with platform='kalshi' and empty wallet_address."""
    from pscanner.corpus.kalshi_walker import walk_kalshi_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, ticker="KX-1")

    pages = [
        [
            _trade(trade_id="t1", taker_side="yes", yes_price=0.40, count=100.0),
            _trade(trade_id="t2", taker_side="no", no_price=0.60, count=200.0),
        ],
        [],
    ]
    client = _FakeKalshiClient(pages)

    inserted = await walk_kalshi_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_ticker=KalshiMarketTicker("KX-1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT platform, tx_hash, asset_id, wallet_address, outcome_side, "
        "price, size, notional_usd, ts FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["platform"] == "kalshi" for r in rows)
    assert all(r["wallet_address"] == "" for r in rows)
    assert {r["tx_hash"] for r in rows} == {"t1", "t2"}
    by_id = {r["tx_hash"]: r for r in rows}
    # t1: yes-side at $0.40 × 100 contracts = $40 notional
    assert by_id["t1"]["asset_id"] == "KX-1:yes"
    assert by_id["t1"]["outcome_side"] == "YES"
    assert by_id["t1"]["price"] == pytest.approx(0.40)
    assert by_id["t1"]["size"] == pytest.approx(100.0)
    assert by_id["t1"]["notional_usd"] == pytest.approx(40.0)
    # t2: no-side at $0.60 × 200 = $120 notional
    assert by_id["t2"]["asset_id"] == "KX-1:no"
    assert by_id["t2"]["outcome_side"] == "NO"
    assert by_id["t2"]["price"] == pytest.approx(0.60)
    assert by_id["t2"]["notional_usd"] == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_walk_drops_below_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Trades below the $10 Kalshi floor are dropped by CorpusTradesRepo.insert_batch."""
    from pscanner.corpus.kalshi_walker import walk_kalshi_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, ticker="KX-1")

    pages = [
        [
            _trade(trade_id="dust", yes_price=0.50, count=10.0),  # $5 notional
            _trade(trade_id="real", yes_price=0.50, count=100.0),  # $50 notional
        ],
        [],
    ]
    client = _FakeKalshiClient(pages)

    inserted = await walk_kalshi_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_ticker=KalshiMarketTicker("KX-1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 1


@pytest.mark.asyncio
async def test_walk_marks_market_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """After walk completes, the corpus_markets row transitions to backfill_state='complete'."""
    from pscanner.corpus.kalshi_walker import walk_kalshi_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, ticker="KX-1")

    pages = [[_trade(trade_id="t1", count=100.0)], []]
    client = _FakeKalshiClient(pages)

    await walk_kalshi_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_ticker=KalshiMarketTicker("KX-1"),
        now_ts=2_000_000_000,
    )
    state = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets "
        "WHERE platform = 'kalshi' AND condition_id = 'KX-1'"
    ).fetchone()
    assert state["backfill_state"] == "complete"
    assert state["truncated_at_offset_cap"] == 0
```

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_kalshi_walker.py -v
```
Expected: 3 fails — module doesn't exist.

- [ ] **Step 4: Create the walker**

Create `src/pscanner/corpus/kalshi_walker.py`:

```python
"""Per-market trades backfill from Kalshi REST into corpus_trades.

Cursor-paginates ``/markets/trades?ticker=<ticker>``, projects each fill
into ``CorpusTrade``, and upserts via ``CorpusTradesRepo``. Marks the
corpus_markets row in_progress at start and complete on successful
exhaustion.

Kalshi's cursor pagination has no offset cap (unlike Polymarket's 3000-offset
limit), so ``truncated`` is always ``False`` when ``mark_complete`` runs.

Anonymous identity convention: ``corpus_trades.wallet_address = ""`` for
every Kalshi row. Kalshi public REST trades carry no taker identity. The
L3-enabling social-API path is tracked separately as #95.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import KalshiTrade

_log = structlog.get_logger(__name__)


async def walk_kalshi_market(
    client: KalshiClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    *,
    market_ticker: KalshiMarketTicker,
    now_ts: int,
    page_size: int = 100,
) -> int:
    """Backfill all fills for one Kalshi market into corpus_trades.

    Args:
        client: Open ``KalshiClient``.
        markets_repo: Corpus markets repo (for state transitions).
        trades_repo: Corpus trades repo (for trade upserts).
        market_ticker: Kalshi market ticker.
        now_ts: Unix seconds, recorded as ``backfill_started_at`` /
            ``backfill_completed_at`` on the ``corpus_markets`` row.
        page_size: ``limit`` parameter on ``client.get_market_trades``.

    Returns:
        Count of inserted ``CorpusTrade`` rows (after the platform-aware
        notional floor in ``CorpusTradesRepo.insert_batch``).
    """
    markets_repo.mark_in_progress(
        market_ticker, started_at=now_ts, platform="kalshi"
    )
    inserted_total = 0
    examined_total = 0
    cursor: str | None = None
    while True:
        page = await client.get_market_trades(
            ticker=market_ticker, limit=page_size, cursor=cursor
        )
        if not page.trades:
            break
        examined_total += len(page.trades)
        trades = [_to_corpus_trade(t, market_ticker=market_ticker) for t in page.trades]
        if trades:
            inserted_total += trades_repo.insert_batch(trades)
        if not page.cursor:
            break
        cursor = page.cursor
    markets_repo.mark_complete(
        market_ticker,
        completed_at=now_ts,
        truncated=False,
        platform="kalshi",
    )
    _log.info(
        "kalshi.walk_complete",
        market_ticker=market_ticker,
        examined=examined_total,
        inserted=inserted_total,
    )
    return inserted_total


def _to_corpus_trade(
    trade: KalshiTrade, *, market_ticker: KalshiMarketTicker
) -> CorpusTrade:
    """Project a Kalshi trade into the corpus dataclass.

    Synthetic ``asset_id = f"{ticker}:{taker_side}"`` names the position;
    ``corpus_trades.asset_id`` is NOT NULL and Kalshi has no separate asset
    identifier. The taker price is the dollar price for the taker's side
    (yes_price for yes-takers, no_price for no-takers). ``notional_usd`` is
    real USD: contracts × price/contract.

    ``wallet_address = ""`` is the documented anonymous-trade convention for
    the L1+L2 path; #95 will surface real attribution via the social API.
    """
    price = (
        trade.yes_price_dollars
        if trade.taker_side == "yes"
        else trade.no_price_dollars
    )
    return CorpusTrade(
        tx_hash=trade.trade_id,
        asset_id=f"{market_ticker}:{trade.taker_side}",
        wallet_address="",
        condition_id=market_ticker,
        outcome_side=trade.taker_side.upper(),
        bs="BUY",
        price=price,
        size=trade.count_fp,
        notional_usd=trade.count_fp * price,
        ts=_iso_to_epoch(trade.created_time, fallback=0),
        platform="kalshi",
    )


def _iso_to_epoch(iso: str, *, fallback: int) -> int:
    """Parse an ISO 8601 datetime string to epoch seconds.

    Returns ``fallback`` if the input is empty or unparseable. Kalshi wire
    format is ``"2026-05-04T12:00:00Z"``; ``datetime.fromisoformat`` handles
    the trailing ``Z`` since Python 3.11.
    """
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())
```

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_kalshi_walker.py -v
```
Expected: 3/3 pass.

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/corpus/kalshi_walker.py tests/corpus/test_kalshi_walker.py
git commit -m "feat(corpus): Kalshi per-market trades walker"
git branch --show-current
```

---

### Task 6: `record_kalshi_resolutions`

**Files:**
- Modify: `src/pscanner/corpus/resolutions.py`
- Modify: `tests/corpus/test_resolutions.py`

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 5 commit.

- [ ] **Step 2: Write the failing tests**

Add to `tests/corpus/test_resolutions.py`:

```python
class _FakeKalshiClient:
    """Stub returning a fixed KalshiMarket by ticker."""

    def __init__(self, markets: dict[str, "KalshiMarket"]) -> None:
        self._markets = markets

    async def get_market(self, ticker: str) -> "KalshiMarket":
        return self._markets[ticker]


def _kalshi_market(
    *, ticker: str, status: str, result: str | None
) -> "KalshiMarket":
    from pscanner.kalshi.models import KalshiMarket

    payload: dict[str, object] = {
        "ticker": ticker,
        "event_ticker": "KX",
        "title": f"Q for {ticker}",
        "status": status,
    }
    if result is not None:
        payload["result"] = result
    return KalshiMarket.model_validate(payload)


@pytest.mark.asyncio
async def test_record_kalshi_resolutions_writes_yes_no(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """YES and NO resolutions land in market_resolutions with platform='kalshi'."""
    from pscanner.corpus.resolutions import record_kalshi_resolutions
    from pscanner.corpus.repos import MarketResolutionsRepo

    repo = MarketResolutionsRepo(tmp_corpus_db)
    client = _FakeKalshiClient(
        {
            "KX-YES": _kalshi_market(ticker="KX-YES", status="finalized", result="yes"),
            "KX-NO": _kalshi_market(ticker="KX-NO", status="determined", result="no"),
        }
    )
    written = await record_kalshi_resolutions(
        client=client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo=repo,
        targets=[("KX-YES", 1_700_000_000), ("KX-NO", 1_700_000_001)],
        now_ts=2_000_000_000,
    )
    assert written == 2
    yes_row = repo.get("KX-YES", platform="kalshi")
    no_row = repo.get("KX-NO", platform="kalshi")
    assert yes_row is not None
    assert yes_row.outcome_yes_won == 1
    assert yes_row.platform == "kalshi"
    assert yes_row.source == "kalshi-rest"
    assert no_row is not None
    assert no_row.outcome_yes_won == 0


@pytest.mark.asyncio
async def test_record_kalshi_resolutions_skips_disputed_undetermined_scalar(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """disputed/undetermined/scalar resolutions are logged + skipped."""
    from pscanner.corpus.resolutions import record_kalshi_resolutions
    from pscanner.corpus.repos import MarketResolutionsRepo

    repo = MarketResolutionsRepo(tmp_corpus_db)
    client = _FakeKalshiClient(
        {
            "disputed": _kalshi_market(ticker="disputed", status="disputed", result="yes"),
            "undetermined": _kalshi_market(ticker="undetermined", status="finalized", result=""),
            "scalar": _kalshi_market(ticker="scalar", status="finalized", result="scalar"),
        }
    )
    written = await record_kalshi_resolutions(
        client=client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo=repo,
        targets=[
            ("disputed", 1_700_000_000),
            ("undetermined", 1_700_000_001),
            ("scalar", 1_700_000_002),
        ],
        now_ts=2_000_000_000,
    )
    assert written == 0
    assert repo.get("disputed", platform="kalshi") is None
    assert repo.get("undetermined", platform="kalshi") is None
    assert repo.get("scalar", platform="kalshi") is None
```

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_resolutions.py -k kalshi -v
```
Expected: 2 fails — `record_kalshi_resolutions` doesn't exist.

- [ ] **Step 4: Add the function**

In `src/pscanner/corpus/resolutions.py`, add the new import at the module top alongside the existing Manifold import:

```python
from pscanner.kalshi.client import KalshiClient
```

Then append the new function below `record_manifold_resolutions`:

```python
async def record_kalshi_resolutions(
    *,
    client: KalshiClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, int]],
    now_ts: int,
) -> int:
    """Fetch resolution outcomes for settled Kalshi markets.

    For each target, calls ``KalshiClient.get_market(ticker)`` and reads
    ``market.result``. Writes a ``market_resolutions`` row for ``"yes"``
    or ``"no"``. Logs and skips for ``"scalar"`` (defensive — should be
    filtered at enumeration), ``""`` with terminal status (voided / odd
    state), and any market with ``status == "disputed"`` (contested
    resolution; the next refresh will pick it up once Kalshi moves it to
    a clean terminal state).

    Args:
        client: Open ``KalshiClient``.
        repo: ``MarketResolutionsRepo`` to upsert into.
        targets: Iterable of ``(market_ticker, resolved_at_hint)``.
        now_ts: Unix seconds, recorded as ``recorded_at`` on each row.

    Returns:
        Count of resolutions actually written (excludes skipped markets).
    """
    written = 0
    for ticker, resolved_at in targets:
        market = await client.get_market(ticker)  # type: ignore[arg-type]
        if market.status == "disputed":
            _log.warning(
                "corpus.kalshi_resolution_disputed",
                market_ticker=ticker,
                result=market.result,
            )
            continue
        if market.result == "yes":
            outcome_yes_won = 1
            winning_outcome_index = 0
        elif market.result == "no":
            outcome_yes_won = 0
            winning_outcome_index = 1
        elif market.result == "scalar":
            _log.warning(
                "corpus.kalshi_resolution_scalar",
                market_ticker=ticker,
            )
            continue
        else:
            _log.warning(
                "corpus.kalshi_resolution_undetermined",
                market_ticker=ticker,
                status=market.status,
                result=market.result,
            )
            continue
        repo.upsert(
            MarketResolution(
                condition_id=ticker,
                winning_outcome_index=winning_outcome_index,
                outcome_yes_won=outcome_yes_won,
                resolved_at=resolved_at,
                source="kalshi-rest",
                platform="kalshi",
            ),
            recorded_at=now_ts,
        )
        written += 1
    return written
```

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_resolutions.py -v
```
Expected: all pass (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/corpus/resolutions.py tests/corpus/test_resolutions.py
git commit -m "feat(corpus): record_kalshi_resolutions skips disputed/scalar/undetermined"
git branch --show-current
```

---

### Task 7: `pscanner corpus backfill --platform kalshi`

**Files:**
- Modify: `src/pscanner/corpus/cli.py`
- Modify: `tests/corpus/test_cli.py`

Extend the existing `--platform` flag's `choices` to `["polymarket", "manifold", "kalshi"]` on the `backfill` subparser, and add a `_run_kalshi_backfill(args)` branch alongside the Polymarket and Manifold dispatchers.

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 6 commit.

- [ ] **Step 2: Write the failing tests**

Add to `tests/corpus/test_cli.py`:

```python
def test_backfill_parser_accepts_platform_kalshi() -> None:
    """`pscanner corpus backfill --platform kalshi` parses correctly."""
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["backfill", "--platform", "kalshi"])
    assert args.platform == "kalshi"


def test_backfill_parser_default_platform_is_still_polymarket() -> None:
    """Adding kalshi to choices doesn't change the default."""
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["backfill"])
    assert args.platform == "polymarket"


def test_backfill_parser_rejects_unknown_after_kalshi_added() -> None:
    """An unknown platform name still fails argparse."""
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backfill", "--platform", "ftx"])
```

If the existing `test_cli.py` has tests for manifold (e.g., `test_backfill_parser_accepts_platform_manifold`), don't duplicate the rejection test — the existing one already covers the SystemExit case for unknown platforms.

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_cli.py -k backfill_parser_accepts_platform_kalshi -v
```
Expected: fail — `kalshi` is not in the `choices` list yet.

- [ ] **Step 4: Add `kalshi` to choices and dispatcher**

In `src/pscanner/corpus/cli.py`:

**(a)** Find the `backfill` subparser. Modify its `--platform` argument's `choices` from `["polymarket", "manifold"]` to `["polymarket", "manifold", "kalshi"]`. Update the help text to mention Kalshi.

**(b)** Update `_cmd_backfill` to dispatch on `kalshi`:

```python
async def _cmd_backfill(args: argparse.Namespace) -> int:
    """Run the corpus backfill for the requested platform."""
    if args.platform == "manifold":
        return await _run_manifold_backfill(args)
    if args.platform == "kalshi":
        return await _run_kalshi_backfill(args)
    return await _run_polymarket_backfill(args)
```

**(c)** Add `_run_kalshi_backfill` near `_run_manifold_backfill`. Add the imports for the new symbols at module top (mirror the manifold imports):

```python
from pscanner.corpus.kalshi_enumerator import enumerate_resolved_kalshi_markets
from pscanner.corpus.kalshi_walker import walk_kalshi_market
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.ids import KalshiMarketTicker
```

```python
async def _run_kalshi_backfill(args: argparse.Namespace) -> int:
    """Kalshi path: enumerate settled markets, walk each one's trades."""
    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    now_ts = int(time.time())
    try:
        async with KalshiClient() as client:
            await enumerate_resolved_kalshi_markets(
                client, markets_repo, now_ts=now_ts
            )
            while pending := markets_repo.next_pending(
                limit=10, platform="kalshi"
            ):
                for market in pending:
                    await walk_kalshi_market(
                        client,
                        markets_repo,
                        trades_repo,
                        market_ticker=KalshiMarketTicker(market.condition_id),
                        now_ts=now_ts,
                    )
    finally:
        conn.close()
    return 0
```

If `KalshiClient` doesn't expose `__aenter__` / `__aexit__` (look at `src/pscanner/kalshi/client.py`), use plain construction + manual close in a `try/finally`. The Manifold equivalent uses `async with`; mirror whichever idiom Kalshi's client supports.

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_cli.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(corpus): \`pscanner corpus backfill --platform kalshi\`"
git branch --show-current
```

---

### Task 8: `pscanner corpus refresh --platform kalshi`

**Files:**
- Modify: `src/pscanner/corpus/cli.py`
- Modify: `tests/corpus/test_cli.py`

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 7 commit.

- [ ] **Step 2: Write the failing tests**

Add to `tests/corpus/test_cli.py`:

```python
def test_refresh_parser_accepts_platform_kalshi() -> None:
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["refresh", "--platform", "kalshi"])
    assert args.platform == "kalshi"
```

- [ ] **Step 3: Run, expect failure**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_cli.py -k refresh_parser_accepts_platform_kalshi -v
```

- [ ] **Step 4: Add `kalshi` to refresh choices and dispatcher**

In `src/pscanner/corpus/cli.py`:

**(a)** Add `kalshi` to the `refresh` subparser's `--platform` choices.

**(b)** Update `_cmd_refresh` to dispatch:

```python
async def _cmd_refresh(args: argparse.Namespace) -> int:
    """Run the corpus refresh for the requested platform."""
    if args.platform == "manifold":
        return await _run_manifold_refresh(args)
    if args.platform == "kalshi":
        return await _run_kalshi_refresh(args)
    return await _run_polymarket_refresh(args)
```

**(c)** Add `_run_kalshi_refresh` near `_run_manifold_refresh`. Add the new import at the top alongside the existing resolutions imports:

```python
from pscanner.corpus.resolutions import (
    record_manifold_resolutions,
    record_kalshi_resolutions,
    record_resolutions,
)
```

```python
async def _run_kalshi_refresh(args: argparse.Namespace) -> int:
    """Kalshi refresh: re-enumerate, then record resolutions for missing markets."""
    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    now_ts = int(time.time())
    try:
        async with KalshiClient() as client:
            await enumerate_resolved_kalshi_markets(
                client, markets_repo, now_ts=now_ts
            )
            rows = conn.execute(
                "SELECT condition_id, closed_at FROM corpus_markets "
                "WHERE platform = 'kalshi' AND backfill_state = 'complete'"
            ).fetchall()
            condition_ids = [row["condition_id"] for row in rows]
            missing = resolutions_repo.missing_for(condition_ids, platform="kalshi")
            missing_set = set(missing)
            targets = [
                (row["condition_id"], int(row["closed_at"]))
                for row in rows
                if row["condition_id"] in missing_set
            ]
            await record_kalshi_resolutions(
                client=client,
                repo=resolutions_repo,
                targets=targets,
                now_ts=now_ts,
            )
    finally:
        conn.close()
    return 0
```

- [ ] **Step 5: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_cli.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(corpus): \`pscanner corpus refresh --platform kalshi\`"
git branch --show-current
```

---

### Task 9: Three-platform end-to-end test

**Files:**
- Create: `tests/corpus/test_kalshi_e2e.py`

Verify that Kalshi data lands cleanly in the platform-aware corpus alongside Polymarket and Manifold rows, and that `build_features(platform="manifold")` doesn't pick up any Kalshi rows. Don't run `build_features(platform="kalshi")` — anonymous trades collapse history under the existing feature schema; the absence of CLI surface (Task 7-8 don't add `--platform kalshi` on `build-features`) is the documented behavior.

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 8 commit.

- [ ] **Step 2: Write the test**

Create `tests/corpus/test_kalshi_e2e.py`:

```python
"""End-to-end test: Kalshi corpus ingestion alongside Polymarket and Manifold.

Seeds a synthetic three-platform corpus DB by hand, then verifies:
- Kalshi trades land in corpus_trades with platform='kalshi' and wallet_address=''.
- Kalshi resolutions land in market_resolutions with platform='kalshi'.
- A market with `result==''` (voided) lands its trades in corpus_trades but
  has no market_resolutions row.
- `build_features(platform='manifold')` produces only manifold rows; no
  Kalshi or Polymarket rows leak.
"""

from __future__ import annotations

import sqlite3

from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)


def _seed_polymarket_row(conn: sqlite3.Connection) -> None:
    """One polymarket market with a YES resolution and one BUY."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="0xpoly",
            event_slug="poly-event",
            category="sports",
            closed_at=1_700_000_500,
            total_volume_usd=1_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="poly-slug",
            platform="polymarket",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xpoly-tx",
                asset_id="poly-asset",
                wallet_address="0xwallet",
                condition_id="0xpoly",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=200.0,
                notional_usd=100.0,
                ts=1_700_000_100,
                platform="polymarket",
            ),
        ]
    )
    resolutions_repo.upsert(
        MarketResolution(
            condition_id="0xpoly",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_000_500,
            source="gamma",
            platform="polymarket",
        ),
        recorded_at=1_700_000_500,
    )


def _seed_manifold_yes(conn: sqlite3.Connection) -> None:
    """One manifold YES market with two bets."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="m-yes",
            event_slug="m-yes-slug",
            category="BINARY",
            closed_at=1_700_000_600,
            total_volume_usd=5_000.0,
            enumerated_at=1_700_000_000,
            market_slug="m-yes-slug",
            platform="manifold",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="b-yes-1",
                asset_id="m-yes:YES",
                wallet_address="user-yes",
                condition_id="m-yes",
                outcome_side="YES",
                bs="BUY",
                price=0.4,
                size=200.0,
                notional_usd=200.0,
                ts=1_700_000_200,
                platform="manifold",
            ),
            CorpusTrade(
                tx_hash="b-yes-2",
                asset_id="m-yes:NO",
                wallet_address="user-no",
                condition_id="m-yes",
                outcome_side="NO",
                bs="BUY",
                price=0.6,
                size=300.0,
                notional_usd=300.0,
                ts=1_700_000_300,
                platform="manifold",
            ),
        ]
    )
    resolutions_repo.upsert(
        MarketResolution(
            condition_id="m-yes",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_000_600,
            source="manifold-rest",
            platform="manifold",
        ),
        recorded_at=1_700_000_600,
    )


def _seed_kalshi_yes(conn: sqlite3.Connection) -> None:
    """One Kalshi YES market with two trades + a YES resolution."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="KX-YES",
            event_slug="KX",
            category="binary",
            closed_at=1_700_000_700,
            total_volume_usd=50_000.0,
            enumerated_at=1_700_000_000,
            market_slug="KX-YES",
            platform="kalshi",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="kx-yes-tx1",
                asset_id="KX-YES:yes",
                wallet_address="",
                condition_id="KX-YES",
                outcome_side="YES",
                bs="BUY",
                price=0.4,
                size=100.0,
                notional_usd=40.0,
                ts=1_700_000_400,
                platform="kalshi",
            ),
            CorpusTrade(
                tx_hash="kx-yes-tx2",
                asset_id="KX-YES:no",
                wallet_address="",
                condition_id="KX-YES",
                outcome_side="NO",
                bs="BUY",
                price=0.6,
                size=200.0,
                notional_usd=120.0,
                ts=1_700_000_500,
                platform="kalshi",
            ),
        ]
    )
    resolutions_repo.upsert(
        MarketResolution(
            condition_id="KX-YES",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_000_700,
            source="kalshi-rest",
            platform="kalshi",
        ),
        recorded_at=1_700_000_700,
    )


def _seed_kalshi_voided(conn: sqlite3.Connection) -> None:
    """One Kalshi market that landed in corpus but has no resolution row (result='')."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="KX-VOID",
            event_slug="KX",
            category="binary",
            closed_at=1_700_000_800,
            total_volume_usd=50_000.0,
            enumerated_at=1_700_000_000,
            market_slug="KX-VOID",
            platform="kalshi",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="kx-void-tx1",
                asset_id="KX-VOID:yes",
                wallet_address="",
                condition_id="KX-VOID",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=100.0,
                notional_usd=50.0,
                ts=1_700_000_600,
                platform="kalshi",
            ),
        ]
    )


def test_kalshi_data_isolated_from_other_platforms(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Kalshi rows land cleanly with platform='kalshi' and don't leak into other-platform queries."""
    _seed_polymarket_row(tmp_corpus_db)
    _seed_manifold_yes(tmp_corpus_db)
    _seed_kalshi_yes(tmp_corpus_db)
    _seed_kalshi_voided(tmp_corpus_db)

    # Both Kalshi markets have trades in corpus_trades.
    kalshi_trade_rows = tmp_corpus_db.execute(
        "SELECT condition_id, tx_hash FROM corpus_trades "
        "WHERE platform = 'kalshi' ORDER BY tx_hash"
    ).fetchall()
    kalshi_tx_hashes = {r["tx_hash"] for r in kalshi_trade_rows}
    assert kalshi_tx_hashes == {"kx-yes-tx1", "kx-yes-tx2", "kx-void-tx1"}

    # Wallet addresses are all empty strings on Kalshi rows.
    for row in kalshi_trade_rows:
        wallet = tmp_corpus_db.execute(
            "SELECT wallet_address FROM corpus_trades WHERE tx_hash = ?",
            (row["tx_hash"],),
        ).fetchone()
        assert wallet["wallet_address"] == ""

    # Only the YES-resolved Kalshi market has a market_resolutions row.
    kalshi_res_rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM market_resolutions WHERE platform = 'kalshi'"
    ).fetchall()
    assert {r["condition_id"] for r in kalshi_res_rows} == {"KX-YES"}


def test_build_features_manifold_does_not_leak_kalshi_or_polymarket(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """`build_features(platform='manifold')` produces only manifold-tagged rows."""
    _seed_polymarket_row(tmp_corpus_db)
    _seed_manifold_yes(tmp_corpus_db)
    _seed_kalshi_yes(tmp_corpus_db)
    _seed_kalshi_voided(tmp_corpus_db)

    examples_repo = TrainingExamplesRepo(tmp_corpus_db)
    # Read the actual build_features signature to mirror Manifold's E2E test in PR #84:
    # build_features takes (trades_repo, resolutions_repo, examples_repo, markets_conn,
    #                       now_ts, rebuild=False, platform="polymarket")
    written = build_features(
        trades_repo=CorpusTradesRepo(tmp_corpus_db),
        resolutions_repo=MarketResolutionsRepo(tmp_corpus_db),
        examples_repo=examples_repo,
        markets_conn=tmp_corpus_db,
        now_ts=2_000_000_000,
        rebuild=True,
        platform="manifold",
    )
    assert written >= 1

    rows = tmp_corpus_db.execute(
        "SELECT platform, condition_id FROM training_examples"
    ).fetchall()
    platforms = {r["platform"] for r in rows}
    condition_ids = {r["condition_id"] for r in rows}
    assert platforms == {"manifold"}, f"unexpected platforms: {platforms}"
    assert "KX-YES" not in condition_ids
    assert "KX-VOID" not in condition_ids
    assert "0xpoly" not in condition_ids
```

If the actual `build_features` signature differs (read `src/pscanner/corpus/examples.py` first), adapt. Don't change the seeding logic.

- [ ] **Step 3: Run, expect pass**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run pytest tests/corpus/test_kalshi_e2e.py -v
```
Expected: 2/2 pass on first run because Tasks 1-8 wired up the platform-aware corpus path end-to-end.

- [ ] **Step 4: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add tests/corpus/test_kalshi_e2e.py
git commit -m "test(corpus): Kalshi ingestion E2E on three-platform corpus"
git branch --show-current
```

---

### Task 10: CLAUDE.md note + final verify

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Verify branch state**

HEAD should be the Task 9 commit.

- [ ] **Step 2: Add the CLAUDE.md bullet**

In `CLAUDE.md`, find the existing "Codebase conventions" section. After the "Manifold ingestion shape" bullet (added in PR #84), append:

```markdown
- **Kalshi ingestion shape (per the integration spec).** `pscanner corpus backfill --platform kalshi` enumerates markets via `/markets?status=...` for each terminal status (`determined`, `amended`, `finalized`); skips `disputed` and `closed`. Walks `/markets/trades?ticker=<ticker>` per market into `corpus_trades`. Resolution detection uses the `result` field on the market response (`"yes"`/`"no"` → write; `"scalar"`/`""`/`disputed` → skip). Anonymous taker identity: `corpus_trades.wallet_address=""` for every Kalshi row (sentinel; no per-trade attribution available on the public REST surface). `notional_usd` is real USD (`count_fp * price`). The `_NOTIONAL_FLOORS["kalshi"] = 10.0` gate already shipped in #84. **`pscanner ml train --platform kalshi` is not supported under the L1+L2 path** — anonymous trades collapse all wallet history to the `""` key, breaking per-wallet features. The L3-enabling social-API path is tracked separately in #95.
```

If the placement is unclear, find any line that mentions Manifold ingestion conventions and place this Kalshi bullet immediately after it.

- [ ] **Step 3: Run the full quick-verify**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected:
- ruff check: clean
- ruff format --check: clean (or run `uv run ruff format .` and stage if dirty)
- ty check: diagnostic count matches the post-PR-#84 baseline (~13 in tracked test files; report the count)
- pytest: all pass; report the count

If ty surfaces new diagnostics in tracked source files, investigate. Compare to the post-Task-9 baseline by stashing and re-running.

- [ ] **Step 4: Commit**

```bash
cd /home/macph/projects/pscanner-worktrees/kalshi-clean
git add CLAUDE.md
git commit -m "docs: document Kalshi ingestion conventions"
git branch --show-current
```

---

## Self-review

**Spec coverage:**
- ✅ Module gap: `result` field on model → Task 1
- ✅ Module gap: `result` column on `kalshi_markets` + idempotent migration → Task 2
- ✅ Module gap: round-trip in repo → Task 3
- ✅ Enumerator (3-pass terminal-status walk, binary + result + volume filter) → Task 4
- ✅ Trades walker (cursor pagination, wallet_address="" sentinel, notional in real USD) → Task 5
- ✅ `record_kalshi_resolutions` (skip disputed/scalar/undetermined) → Task 6
- ✅ `--platform kalshi` on `backfill` → Task 7
- ✅ `--platform kalshi` on `refresh` → Task 8
- ✅ Three-platform E2E test → Task 9
- ✅ CLAUDE.md note + final verify → Task 10
- ✅ NO `--platform kalshi` on `build-features` (explicitly out of scope, anonymous trades) → confirmed by absence of any task adding it

**Placeholder scan:** Tasks 2, 7, 8 have inspect-then-adapt instructions for the daemon's `init_db` shape, the existing CLI dispatcher style, and the actual `KalshiClient` async-context-manager support. These are not TBDs — the implementer has explicit guidance ("read the file, mirror Manifold's pattern"). Acceptable.

**Type consistency:** `platform="kalshi"` literal used uniformly. `_NOTIONAL_FLOORS["kalshi"]=10.0` referenced consistently. `record_kalshi_resolutions` upserts with `source="kalshi-rest"`; tests match the literal. `_iso_to_epoch` helper duplicated across `kalshi_enumerator.py` and `kalshi_walker.py` deliberately (two-call rule; refactor into a shared helper if a third caller appears).

---

## Out of scope (explicit non-goals)

- L3 / `training_examples` for Kalshi — anonymous trades; tracked in #95.
- `pscanner ml train --platform kalshi` and `pscanner corpus build-features --platform kalshi`.
- Social-API attribution path (`/v1/social/*`) — tracked separately as #95.
- Authenticated WebSocket streaming (Stage 2 of #36).
- Kalshi daemon-side detector instances or paper-trading evaluators.
- Multi-platform aggregation in ML training.
- CFMM / scalar markets — filtered at enumerator.
- Voided/cancelled markets with explicit detection — covered implicitly via the `result == ""` skip path.

---

## Risks

- **Kalshi API drift on the `result` field.** OpenAPI spec lists it as a required field (verified 2026-05-07). If renamed or removed, `KalshiMarket.result` becomes silently `None`, `record_kalshi_resolutions` correctly skips with a clear log event, and tests catch the parse-path regression.
- **The three-pass status walk is slower than necessary.** At 60 RPM and ~1000 settled markets per status, the budget is well within reach (≈30s per pass, ≈90s total). Tunable via the `page_size` parameter; reorderable if Kalshi adds multi-value `status` filtering.
- **Volume gate (10K contracts ≈ $5K) might be too aggressive or too loose.** Tunable via `min_volume_contracts`. The first live run logs `kalshi.enumerate_complete` with `examined`/`inserted` counts to inform tuning.
- **Existing `kalshi_markets` rows on a real on-disk DB may pre-date the `result` column.** The idempotent migration in Task 2 handles this — `ALTER TABLE ADD COLUMN` runs once, swallows the duplicate-column error on subsequent runs. Backed by `test_init_kalshi_tables_idempotent_on_result_column`.
- **`KalshiClient` async-context-manager support unclear from the spec.** Task 7 step 4 explicitly reads `src/pscanner/kalshi/client.py` to verify, and falls back to construction + `try/finally` if `async with` isn't supported. Mirror whichever idiom Kalshi's client supports (Manifold's client uses `async with`).
- **`build_features(platform="manifold")` E2E test parameter shape.** PR A's Task 10 made the function platform-aware; the exact kwargs (e.g., `markets_conn` vs `db_path`) might require reading the actual implementation. Task 9 step 2 covers this — re-read `src/pscanner/corpus/examples.py:build_features` and adapt the call.
