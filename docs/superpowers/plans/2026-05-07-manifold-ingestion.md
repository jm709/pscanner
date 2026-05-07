# Manifold ingestion (corpus L1+L2+L3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Manifold REST through a new enumerator + bet walker + resolutions function into the platform-aware corpus, then expose `pscanner corpus backfill / refresh / build-features --platform manifold`. Single-platform per training run; mana stored as platform-native units in `notional_usd`; Manifold `user_id` reuses `wallet_address`; MKT/CANCEL markets land in corpus but skip `market_resolutions`.

**Architecture:** Two new files (`manifold_enumerator.py`, `manifold_walker.py`) call `ManifoldClient` and write into the existing platform-aware `CorpusMarketsRepo` and `CorpusTradesRepo` (post-PR-#82). Resolution recording extends `pscanner.corpus.resolutions` with a Manifold-flavored function. Build-features is unchanged — PR A's polymorphic `build_features(platform="manifold")` already works once the corpus tables hold Manifold rows. CLI dispatch is added behind `--platform manifold` flags on the three existing `pscanner corpus` subcommands.

**Tech Stack:** Python 3.13, async httpx + tenacity (already used by `ManifoldClient`), Polars (downstream in `build_features`), pydantic, pytest + respx for HTTP mocking. Quick verify: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Spec:** `docs/superpowers/specs/2026-05-07-manifold-ingestion-design.md`

---

## File map

**New:**
- `src/pscanner/corpus/manifold_enumerator.py`
- `src/pscanner/corpus/manifold_walker.py`
- `tests/corpus/test_manifold_enumerator.py`
- `tests/corpus/test_manifold_walker.py`
- `tests/corpus/test_manifold_e2e.py`

**Modify:**
- `src/pscanner/manifold/models.py` (add `resolution` field)
- `src/pscanner/manifold/db.py` (add `resolution TEXT` column + idempotent migration)
- `src/pscanner/manifold/repos.py` (round-trip `resolution`)
- `src/pscanner/store/db.py` (concatenate `MANIFOLD_SCHEMA_STATEMENTS`)
- `src/pscanner/corpus/repos.py` (`_NOTIONAL_FLOORS` dict, platform-aware floor)
- `src/pscanner/corpus/resolutions.py` (`record_manifold_resolutions`)
- `src/pscanner/corpus/cli.py` (`--platform` flag + dispatch on 3 subcommands)
- `tests/manifold/test_models.py`, `tests/manifold/test_db.py`, `tests/manifold/test_repos.py` (extend)
- `tests/corpus/test_repos_trades.py`, `tests/corpus/test_resolutions.py`, `tests/corpus/test_cli.py` (extend)
- `tests/store/test_db.py` (extend — assert manifold tables exist after `init_db()`)
- `CLAUDE.md` (mana convention, `--platform manifold` CLI surface, MKT/CANCEL skip-pattern)

Roughly 5 new files, 12 modified. ~600 lines source, ~600 lines tests.

---

### Task 1: Add `resolution` field to `ManifoldMarket` model

**Files:**
- Modify: `src/pscanner/manifold/models.py:25-52`
- Modify: `tests/manifold/test_models.py`

The Manifold REST API returns `resolution: "YES" | "NO" | "MKT" | "CANCEL" | null` on resolved markets via `/v0/market/{id}`. Stage 1 didn't model it. The integration spec needs this field to drive resolution recording.

- [ ] **Step 1: Write the failing test**

Add to `tests/manifold/test_models.py`:

```python
import pytest

from pscanner.manifold.models import ManifoldMarket


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("YES", "YES"),
        ("NO", "NO"),
        ("MKT", "MKT"),
        ("CANCEL", "CANCEL"),
        (None, None),
    ],
)
def test_manifold_market_parses_resolution_field(raw_value: str | None, expected: str | None) -> None:
    """`resolution` round-trips through validation for all four documented values + null."""
    market = ManifoldMarket.model_validate(
        {
            "id": "abc123",
            "creatorId": "user1",
            "question": "Will X happen?",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "isResolved": True,
            "resolutionTime": 1_700_000_000,
            "resolution": raw_value,
        }
    )
    assert market.resolution == expected


def test_manifold_market_resolution_defaults_to_none_when_absent() -> None:
    """Markets that haven't resolved yet omit the field; the model defaults to None."""
    market = ManifoldMarket.model_validate(
        {
            "id": "abc123",
            "creatorId": "user1",
            "question": "Will X happen?",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "isResolved": False,
        }
    )
    assert market.resolution is None
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/manifold/test_models.py::test_manifold_market_parses_resolution_field -v`
Expected: fail — `ManifoldMarket` has no `resolution` attribute.

- [ ] **Step 3: Add the field**

In `src/pscanner/manifold/models.py`, modify the `ManifoldMarket` class (after `close_time`, line 45):

```python
class ManifoldMarket(BaseModel):
    """A Manifold Markets contract (market).

    Supports binary YES/NO markets (``outcome_type == "BINARY"``). Multi-outcome
    CFMM markets are represented by the same model but the ``prob`` field may be
    ``None`` for non-binary types.
    """

    model_config = _BASE_CONFIG

    id: ManifoldMarketId
    creator_id: ManifoldUserId = Field(alias="creatorId")
    question: str
    outcome_type: str = Field(alias="outcomeType")
    mechanism: str
    prob: float | None = None
    volume: float = 0.0
    total_liquidity: float = Field(alias="totalLiquidity", default=0.0)
    is_resolved: bool = Field(alias="isResolved")
    resolution_time: int | None = Field(alias="resolutionTime", default=None)
    resolution: str | None = None
    close_time: int | None = Field(alias="closeTime", default=None)
    url: str | None = None
    slug: ManifoldSlug | None = None

    @property
    def is_binary(self) -> bool:
        """True iff this is a binary YES/NO market."""
        return self.outcome_type == "BINARY"
```

The field is unaliased because Manifold's JSON key is already snake_case (`resolution`, no camelCase variant).

- [ ] **Step 4: Run all model tests**

Run: `uv run pytest tests/manifold/test_models.py -v`
Expected: all pass (existing + 6 new parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/manifold/models.py tests/manifold/test_models.py
git commit -m "feat(manifold): parse resolution field on ManifoldMarket model"
```

---

### Task 2: Add `resolution` column to `manifold_markets` schema

**Files:**
- Modify: `src/pscanner/manifold/db.py:17-62`
- Modify: `tests/manifold/test_db.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/manifold/test_db.py`:

```python
import sqlite3

from pscanner.manifold.db import init_manifold_tables


def test_manifold_markets_has_resolution_column() -> None:
    """`manifold_markets` exposes a nullable TEXT `resolution` column."""
    conn = sqlite3.connect(":memory:")
    try:
        init_manifold_tables(conn)
        info = conn.execute("PRAGMA table_info(manifold_markets)").fetchall()
        cols = {row[1]: row for row in info}
        assert "resolution" in cols
        assert cols["resolution"][2].upper() == "TEXT"
        assert cols["resolution"][3] == 0, "resolution must be nullable"
    finally:
        conn.close()


def test_init_manifold_tables_idempotent_on_resolution_column() -> None:
    """Calling init_manifold_tables twice leaves the resolution column intact."""
    conn = sqlite3.connect(":memory:")
    try:
        init_manifold_tables(conn)
        init_manifold_tables(conn)
        info = conn.execute("PRAGMA table_info(manifold_markets)").fetchall()
        cols = {row[1] for row in info}
        assert "resolution" in cols
    finally:
        conn.close()
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/manifold/test_db.py::test_manifold_markets_has_resolution_column -v`
Expected: fail — column doesn't exist.

- [ ] **Step 3: Add the column to fresh-DB schema and a migration**

In `src/pscanner/manifold/db.py`, modify the `manifold_markets` `CREATE TABLE` statement (lines 18-33) to add the new column:

```python
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS manifold_markets (
      id TEXT PRIMARY KEY,
      creator_id TEXT NOT NULL,
      question TEXT NOT NULL,
      outcome_type TEXT NOT NULL,
      mechanism TEXT NOT NULL,
      prob_at_last_seen REAL,
      volume REAL NOT NULL DEFAULT 0.0,
      total_liquidity REAL NOT NULL DEFAULT 0.0,
      is_resolved INTEGER NOT NULL DEFAULT 0,
      resolution_time INTEGER,
      resolution TEXT,
      close_time INTEGER,
      raw_json TEXT NOT NULL
    )
    """,
    # ... unchanged manifold_bets, manifold_users, indexes ...
)
```

Add a `_MIGRATIONS` tuple and call it in `init_manifold_tables` to handle existing DBs that pre-date this column. After `_SCHEMA_STATEMENTS`:

```python
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE manifold_markets ADD COLUMN resolution TEXT",
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations. Idempotent.

    Each migration is wrapped to swallow ``duplicate column name`` errors
    so repeated calls on already-migrated DBs are no-ops.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                continue
            raise
    conn.commit()


def init_manifold_tables(conn: sqlite3.Connection) -> None:
    """Apply all Manifold schema statements to ``conn``.

    Idempotent — safe to call on an already-initialised database.
    Applies additive migrations so existing on-disk databases pick up
    new columns added in later versions.

    Args:
        conn: Open ``sqlite3.Connection`` with WAL mode already set.
    """
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)
    _apply_migrations(conn)
    conn.commit()
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/manifold/test_db.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/manifold/db.py tests/manifold/test_db.py
git commit -m "feat(manifold): add resolution column to manifold_markets + idempotent migration"
```

---

### Task 3: Round-trip `resolution` through `ManifoldMarketsRepo`

**Files:**
- Modify: `src/pscanner/manifold/repos.py:34-92`
- Modify: `tests/manifold/test_repos.py`

`insert_or_replace` writes specific columns in addition to `raw_json`. The new column needs explicit handling so reads via `iter_chronological` (which uses raw_json) AND direct column queries both see the value.

- [ ] **Step 1: Write the failing test**

Add to `tests/manifold/test_repos.py`:

```python
def test_manifold_markets_repo_roundtrips_resolution(tmp_manifold_conn: sqlite3.Connection) -> None:  # type: ignore[no-untyped-def]
    """`insert_or_replace` writes the resolution column; the value survives round-trip."""
    from pscanner.manifold.models import ManifoldMarket
    from pscanner.manifold.repos import ManifoldMarketsRepo

    repo = ManifoldMarketsRepo(tmp_manifold_conn)
    market = ManifoldMarket.model_validate(
        {
            "id": "abc123",
            "creatorId": "user1",
            "question": "Will X happen?",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "isResolved": True,
            "resolutionTime": 1_700_000_000,
            "resolution": "YES",
        }
    )
    repo.insert_or_replace(market)
    # Direct column read
    row = tmp_manifold_conn.execute(
        "SELECT resolution FROM manifold_markets WHERE id = ?", (market.id,)
    ).fetchone()
    assert row[0] == "YES"
    # Round-trip via raw_json (covers iter_chronological)
    fetched = repo.get_by_id(market.id)
    assert fetched is not None
    assert fetched.resolution == "YES"
```

If `tmp_manifold_conn` doesn't already exist as a fixture in `tests/manifold/conftest.py` (or wherever Manifold fixtures live), inspect existing tests in `test_repos.py` to find the fixture they use. Mirror its name. Or define one inline at the top of the test file:

```python
import pytest
import sqlite3
from collections.abc import Iterator
from pscanner.manifold.db import init_manifold_tables


@pytest.fixture
def tmp_manifold_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        init_manifold_tables(conn)
        yield conn
    finally:
        conn.close()
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/manifold/test_repos.py::test_manifold_markets_repo_roundtrips_resolution -v`
Expected: fail — direct column read returns NULL because `insert_or_replace` doesn't include `resolution` in the INSERT.

- [ ] **Step 3: Update `insert_or_replace`**

In `src/pscanner/manifold/repos.py`, modify `ManifoldMarketsRepo.insert_or_replace`:

```python
    def insert_or_replace(self, market: ManifoldMarket) -> None:
        """Upsert a market row. Replaces all columns on conflict.

        Args:
            market: ``ManifoldMarket`` model to persist.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO manifold_markets (
              id, creator_id, question, outcome_type, mechanism,
              prob_at_last_seen, volume, total_liquidity, is_resolved,
              resolution_time, resolution, close_time, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market.id,
                market.creator_id,
                market.question,
                market.outcome_type,
                market.mechanism,
                market.prob,
                market.volume,
                market.total_liquidity,
                int(market.is_resolved),
                market.resolution_time,
                market.resolution,
                market.close_time,
                market.model_dump_json(by_alias=True),
            ),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/manifold/test_repos.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/manifold/repos.py tests/manifold/test_repos.py
git commit -m "feat(manifold): write resolution column from insert_or_replace"
```

---

### Task 4: Wire `init_manifold_tables` into daemon `init_db`

**Files:**
- Modify: `src/pscanner/store/db.py:1-15` (imports + `_SCHEMA_STATEMENTS`)
- Modify: `src/pscanner/manifold/db.py` (export `MANIFOLD_SCHEMA_STATEMENTS` constant)
- Modify: `tests/store/test_db.py`

The daemon's `init_db()` already concatenates `KALSHI_SCHEMA_STATEMENTS`. Mirror that pattern for Manifold so a fresh daemon DB has Manifold tables out of the box.

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_db.py`:

```python
def test_init_db_creates_manifold_tables(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """`init_db` creates the Manifold daemon tables alongside Polymarket and Kalshi."""
    from pscanner.store.db import init_db

    db_path = tmp_path / "test.sqlite3"
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {row["name"] for row in rows}
        assert {"manifold_markets", "manifold_bets", "manifold_users"}.issubset(names)
    finally:
        conn.close()
```

If `Path` isn't already imported at the top of `test_db.py`, add `from pathlib import Path`.

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/store/test_db.py::test_init_db_creates_manifold_tables -v`
Expected: fail — `manifold_markets` not in `sqlite_master`.

- [ ] **Step 3: Export `MANIFOLD_SCHEMA_STATEMENTS` and concatenate into `_SCHEMA_STATEMENTS`**

In `src/pscanner/manifold/db.py`, expose the schema-statements tuple as a public re-export by changing `_SCHEMA_STATEMENTS` to `MANIFOLD_SCHEMA_STATEMENTS` (rename the module-private name, then update internal references):

```python
MANIFOLD_SCHEMA_STATEMENTS: tuple[str, ...] = (
    # ... existing CREATE TABLE statements + indexes (unchanged content) ...
)


def init_manifold_tables(conn: sqlite3.Connection) -> None:
    """..."""
    for statement in MANIFOLD_SCHEMA_STATEMENTS:
        conn.execute(statement)
    _apply_migrations(conn)
    conn.commit()
```

In `src/pscanner/store/db.py`, add the import and concatenate (mirror line 13):

```python
from pscanner.kalshi.db import KALSHI_SCHEMA_STATEMENTS
from pscanner.manifold.db import MANIFOLD_SCHEMA_STATEMENTS

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    # ... existing Polymarket daemon table definitions ...
    *KALSHI_SCHEMA_STATEMENTS,
    *MANIFOLD_SCHEMA_STATEMENTS,
)
```

(Find the existing `*KALSHI_SCHEMA_STATEMENTS` reference in `_SCHEMA_STATEMENTS` and add `*MANIFOLD_SCHEMA_STATEMENTS` next to it.)

- [ ] **Step 4: Run, expect pass — verify across the broader suite**

Run: `uv run pytest tests/store/test_db.py tests/manifold/ -q`
Expected: all pass.

Run: `uv run pytest -q`
Expected: full suite green. The wiring change shouldn't affect any test that doesn't depend on `manifold_*` tables, but verify no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/store/db.py src/pscanner/manifold/db.py tests/store/test_db.py
git commit -m "feat(store): apply Manifold schema in init_db (mirrors Kalshi)"
```

---

### Task 5: Platform-aware notional floor in `CorpusTradesRepo`

**Files:**
- Modify: `src/pscanner/corpus/repos.py:243, 281-316` (the `_NOTIONAL_FLOOR_USD` constant + `insert_batch` body)
- Modify: `tests/corpus/test_repos_trades.py`

`insert_batch` currently filters out trades with `notional_usd < 10.0`. For Manifold (mana-denominated), we want a higher floor so dust bets don't pollute. Change to a per-platform dict.

- [ ] **Step 1: Write the failing tests**

Add to `tests/corpus/test_repos_trades.py`:

```python
def test_insert_batch_uses_polymarket_floor_for_polymarket_rows(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Polymarket rows below $10 USD are dropped (existing behavior)."""
    from pscanner.corpus.repos import CorpusTrade, CorpusTradesRepo

    repo = CorpusTradesRepo(tmp_corpus_db)
    poly = CorpusTrade(
        tx_hash="0xtx", asset_id="a1", wallet_address="0xw",
        condition_id="c1", outcome_side="YES", bs="BUY",
        price=0.5, size=10.0, notional_usd=5.0, ts=1000,
        platform="polymarket",
    )
    assert repo.insert_batch([poly]) == 0


def test_insert_batch_uses_manifold_floor_for_manifold_rows(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Manifold rows below 100 mana are dropped; rows >= 100 mana are kept."""
    from pscanner.corpus.repos import CorpusTrade, CorpusTradesRepo

    repo = CorpusTradesRepo(tmp_corpus_db)
    below = CorpusTrade(
        tx_hash="m-tx-low", asset_id="m1:YES", wallet_address="userA",
        condition_id="m1", outcome_side="YES", bs="BUY",
        price=0.5, size=50.0, notional_usd=50.0, ts=1000,
        platform="manifold",
    )
    above = CorpusTrade(
        tx_hash="m-tx-high", asset_id="m1:YES", wallet_address="userA",
        condition_id="m1", outcome_side="YES", bs="BUY",
        price=0.5, size=200.0, notional_usd=200.0, ts=1001,
        platform="manifold",
    )
    assert repo.insert_batch([below, above]) == 1
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_repos_trades.py::test_insert_batch_uses_manifold_floor_for_manifold_rows -v`
Expected: fail — both rows currently pass the $10 floor (the manifold below-floor row at 50 mana > $10 USD passes the existing single-floor check).

Actually it might pass the test for the wrong reason — verify by reading the failure output. The polymarket-floor test should be unchanged from current behavior.

- [ ] **Step 3: Add `_NOTIONAL_FLOORS` dict and update `insert_batch`**

In `src/pscanner/corpus/repos.py`, replace the `_NOTIONAL_FLOOR_USD` constant (around line 243) with:

```python
_NOTIONAL_FLOORS: Final[dict[str, float]] = {
    "polymarket": 10.0,
    "manifold": 100.0,
    "kalshi": 10.0,  # placeholder; revisit when Kalshi ingestion ships
}
# Backward-compat alias for the existing module-private constant.
_NOTIONAL_FLOOR_USD: Final[float] = _NOTIONAL_FLOORS["polymarket"]
```

Update `CorpusTradesRepo.insert_batch` — find the line:

```python
            if t.notional_usd < _NOTIONAL_FLOOR_USD:
                continue
```

Replace with:

```python
            floor = _NOTIONAL_FLOORS.get(t.platform, _NOTIONAL_FLOORS["polymarket"])
            if t.notional_usd < floor:
                continue
```

The fallback to polymarket's floor handles future platform tags that aren't in the dict yet — they get the conservative $10 cut by default.

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/corpus/test_repos_trades.py -q`
Expected: all pass (existing tests + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_trades.py
git commit -m "feat(corpus): platform-aware notional floor in CorpusTradesRepo.insert_batch"
```

---

### Task 6: Manifold enumerator (`enumerate_resolved_manifold_markets`)

**Files:**
- Create: `src/pscanner/corpus/manifold_enumerator.py`
- Create: `tests/corpus/test_manifold_enumerator.py`

The enumerator walks `/v0/markets` paginated, filters to resolved+binary+above-volume, and inserts into `corpus_markets` as `(platform='manifold')` rows.

- [ ] **Step 1: Write the failing test**

Create `tests/corpus/test_manifold_enumerator.py`:

```python
"""Tests for the Manifold corpus enumerator."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.models import ManifoldMarket


class _FakeManifoldClient:
    """Tiny stub that yields fixed pages of markets per call.

    Indexes through `pages` per get_markets call; cursor parameter is ignored
    because pages are pre-built with the desired filter mix.
    """

    def __init__(self, pages: list[list[ManifoldMarket]]) -> None:
        self._pages = pages
        self._call_count = 0

    async def get_markets(
        self,
        *,
        limit: int = 1000,
        before: str | None = None,
    ) -> list[ManifoldMarket]:
        if self._call_count >= len(self._pages):
            return []
        page = self._pages[self._call_count]
        self._call_count += 1
        return page


def _market(
    *,
    market_id: str,
    is_resolved: bool = True,
    outcome_type: str = "BINARY",
    volume: float = 5_000.0,
    resolution_time: int = 1_700_000_000,
    slug: str | None = None,
) -> ManifoldMarket:
    return ManifoldMarket.model_validate(
        {
            "id": market_id,
            "creatorId": "creator1",
            "question": f"Question for {market_id}?",
            "outcomeType": outcome_type,
            "mechanism": "cpmm-1",
            "volume": volume,
            "isResolved": is_resolved,
            "resolutionTime": resolution_time,
            "slug": slug or market_id,
        }
    )


@pytest.mark.asyncio
async def test_enumerate_inserts_only_resolved_binary_above_volume(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """A page with mixed conditions yields only resolved+binary+above-volume rows."""
    from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets

    pages = [
        [
            _market(market_id="keep1"),
            _market(market_id="unresolved", is_resolved=False),
            _market(market_id="cfmm", outcome_type="MULTIPLE_CHOICE"),
            _market(market_id="lowvol", volume=100.0),
            _market(market_id="keep2"),
        ],
    ]
    client = _FakeManifoldClient(pages)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_manifold_markets(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
        min_volume_mana=1000.0,
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets WHERE platform = 'manifold' "
        "ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["keep1", "keep2"]


@pytest.mark.asyncio
async def test_enumerate_paginates_until_empty(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Multiple pages are walked until the client returns an empty list."""
    from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets

    pages = [
        [_market(market_id="p0_a"), _market(market_id="p0_b")],
        [_market(market_id="p1_a")],
        [],
    ]
    client = _FakeManifoldClient(pages)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_manifold_markets(
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
    from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets

    pages = [[_market(market_id="m1"), _market(market_id="m2")], []]
    repo = CorpusMarketsRepo(tmp_corpus_db)

    client1 = _FakeManifoldClient(pages)
    first = await enumerate_resolved_manifold_markets(client1, repo, now_ts=2_000_000_000)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    assert first == 2

    pages2 = [[_market(market_id="m1"), _market(market_id="m2")], []]
    client2 = _FakeManifoldClient(pages2)
    second = await enumerate_resolved_manifold_markets(client2, repo, now_ts=2_000_000_001)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    assert second == 0
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_manifold_enumerator.py -v`
Expected: 3 tests fail — module doesn't exist.

- [ ] **Step 3: Create the enumerator**

Create `src/pscanner/corpus/manifold_enumerator.py`:

```python
"""Corpus enumerator for closed Manifold Markets.

Walks ``/v0/markets`` paginated via ``before=<id>`` cursor, filters to resolved
binary markets above the volume gate, and inserts ``(platform='manifold')`` rows
into ``corpus_markets``. Idempotent — repeated runs are no-ops on already-known
markets thanks to ``CorpusMarketsRepo.insert_pending``'s ``INSERT OR IGNORE``
semantics.
"""

from __future__ import annotations

import structlog

from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.models import ManifoldMarket

_log = structlog.get_logger(__name__)


async def enumerate_resolved_manifold_markets(
    client: ManifoldClient,
    repo: CorpusMarketsRepo,
    *,
    now_ts: int,
    min_volume_mana: float = 1000.0,
    page_size: int = 1000,
) -> int:
    """Walk Manifold markets and insert resolved+binary+above-volume rows.

    Args:
        client: Open ``ManifoldClient`` with rate-limit budget available.
        repo: Corpus markets repo bound to a platform-aware corpus DB.
        now_ts: Unix seconds, recorded as ``enumerated_at`` on each row.
        min_volume_mana: Minimum ``ManifoldMarket.volume`` to qualify
            (mana, not USD). Defaults to 1000.
        page_size: ``limit`` parameter on ``client.get_markets``.

    Returns:
        Count of newly-inserted ``corpus_markets`` rows. Does not include
        rows that already existed (idempotent re-enumeration).
    """
    inserted_total = 0
    examined_total = 0
    cursor: str | None = None
    while True:
        page = await client.get_markets(limit=page_size, before=cursor)
        if not page:
            break
        examined_total += len(page)
        for market in page:
            if not _qualifies(market, min_volume_mana=min_volume_mana):
                continue
            corpus_market = _to_corpus_market(market, now_ts=now_ts)
            inserted_total += repo.insert_pending(corpus_market)
        cursor = page[-1].id
    _log.info(
        "manifold.enumerate_complete",
        examined=examined_total,
        inserted=inserted_total,
        min_volume_mana=min_volume_mana,
    )
    return inserted_total


def _qualifies(market: ManifoldMarket, *, min_volume_mana: float) -> bool:
    """True iff the market should land in the corpus."""
    return market.is_resolved and market.is_binary and market.volume >= min_volume_mana


def _to_corpus_market(market: ManifoldMarket, *, now_ts: int) -> CorpusMarket:
    """Project a ``ManifoldMarket`` into the corpus dataclass."""
    return CorpusMarket(
        condition_id=market.id,
        event_slug=market.slug or market.id,
        category=market.outcome_type,
        closed_at=market.resolution_time or now_ts,
        total_volume_usd=market.volume,
        enumerated_at=now_ts,
        market_slug=market.slug or market.id,
        platform="manifold",
    )
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/corpus/test_manifold_enumerator.py -v`
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/manifold_enumerator.py tests/corpus/test_manifold_enumerator.py
git commit -m "feat(corpus): Manifold corpus enumerator for resolved binary markets"
```

---

### Task 7: Manifold bet walker (`walk_manifold_market`)

**Files:**
- Create: `src/pscanner/corpus/manifold_walker.py`
- Create: `tests/corpus/test_manifold_walker.py`

Backfills every fillable bet for one Manifold market into `corpus_trades`.

- [ ] **Step 1: Write the failing test**

Create `tests/corpus/test_manifold_walker.py`:

```python
"""Tests for the Manifold per-market bet walker."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo, CorpusTradesRepo
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.ids import ManifoldMarketId
from pscanner.manifold.models import ManifoldBet


class _FakeManifoldClient:
    def __init__(self, pages: list[list[ManifoldBet]]) -> None:
        self._pages = pages
        self._call_count = 0

    async def get_bets(
        self,
        *,
        market_id: ManifoldMarketId | None = None,
        user_id: object = None,
        limit: int = 1000,
        before: str | None = None,
    ) -> list[ManifoldBet]:
        if self._call_count >= len(self._pages):
            return []
        page = self._pages[self._call_count]
        self._call_count += 1
        return page


def _bet(
    *,
    bet_id: str,
    market_id: str = "m1",
    outcome: str = "YES",
    amount: float = 200.0,
    prob_before: float = 0.5,
    is_filled: bool | None = None,
    is_cancelled: bool | None = None,
    limit_prob: float | None = None,
    user_id: str = "user1",
) -> ManifoldBet:
    return ManifoldBet.model_validate(
        {
            "id": bet_id,
            "userId": user_id,
            "contractId": market_id,
            "outcome": outcome,
            "amount": amount,
            "probBefore": prob_before,
            "probAfter": prob_before + 0.01,
            "createdTime": 1_700_000_000,
            "isFilled": is_filled,
            "isCancelled": is_cancelled,
            "limitProb": limit_prob,
        }
    )


def _seed_market(
    repo: CorpusMarketsRepo, *, market_id: str = "m1"
) -> None:
    """Insert a pending market so the walker can mark it in_progress + complete."""
    repo.insert_pending(
        CorpusMarket(
            condition_id=market_id,
            event_slug=market_id,
            category="BINARY",
            closed_at=1_700_000_000,
            total_volume_usd=5_000.0,
            enumerated_at=1_700_000_000 - 1,
            market_slug=market_id,
            platform="manifold",
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_filled_bets_with_manifold_platform(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Filled bets land in corpus_trades with platform='manifold'."""
    from pscanner.corpus.manifold_walker import walk_manifold_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [
        [_bet(bet_id="b1", amount=200.0), _bet(bet_id="b2", amount=300.0)],
        [],
    ]
    client = _FakeManifoldClient(pages)

    inserted = await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT platform, tx_hash, asset_id, wallet_address, price, size, notional_usd "
        "FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["platform"] == "manifold" for r in rows)
    assert {r["tx_hash"] for r in rows} == {"b1", "b2"}
    assert all(r["asset_id"] == "m1:YES" for r in rows)
    assert all(r["wallet_address"] == "user1" for r in rows)


@pytest.mark.asyncio
async def test_walk_skips_cancelled_and_unfilled_limit_orders(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Cancelled bets and unfilled limit orders never land in corpus_trades."""
    from pscanner.corpus.manifold_walker import walk_manifold_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [
        [
            _bet(bet_id="ok", amount=200.0),
            _bet(bet_id="cancelled", amount=200.0, is_cancelled=True),
            _bet(bet_id="unfilled-limit", amount=200.0, limit_prob=0.6, is_filled=False),
            _bet(bet_id="filled-limit", amount=200.0, limit_prob=0.6, is_filled=True),
        ],
        [],
    ]
    client = _FakeManifoldClient(pages)

    inserted = await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 2  # ok + filled-limit
    rows = tmp_corpus_db.execute(
        "SELECT tx_hash FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    assert {r["tx_hash"] for r in rows} == {"filled-limit", "ok"}


@pytest.mark.asyncio
async def test_walk_drops_below_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Bets below the 100-mana floor are dropped by CorpusTradesRepo.insert_batch."""
    from pscanner.corpus.manifold_walker import walk_manifold_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [
        [
            _bet(bet_id="dust", amount=50.0),
            _bet(bet_id="real", amount=200.0),
        ],
        [],
    ]
    client = _FakeManifoldClient(pages)

    inserted = await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 1
    rows = tmp_corpus_db.execute(
        "SELECT tx_hash FROM corpus_trades"
    ).fetchall()
    assert [r["tx_hash"] for r in rows] == ["real"]


@pytest.mark.asyncio
async def test_walk_marks_market_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """After walk completes, the corpus_markets row transitions to backfill_state='complete'."""
    from pscanner.corpus.manifold_walker import walk_manifold_market

    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [[_bet(bet_id="b1", amount=200.0)], []]
    client = _FakeManifoldClient(pages)

    await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    state = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets "
        "WHERE platform = 'manifold' AND condition_id = 'm1'"
    ).fetchone()
    assert state["backfill_state"] == "complete"
    assert state["truncated_at_offset_cap"] == 0
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_manifold_walker.py -v`
Expected: 4 tests fail — module doesn't exist.

- [ ] **Step 3: Create the walker**

Create `src/pscanner/corpus/manifold_walker.py`:

```python
"""Per-market bet backfill from Manifold REST into corpus_trades.

Cursor-paginates ``/v0/bets?contractId=<market_id>``, filters out cancelled bets
and unfilled limit orders, projects each fillable bet into ``CorpusTrade``, and
upserts via ``CorpusTradesRepo``. Marks the corpus_markets row in_progress at
start and complete on successful exhaustion.

Manifold's cursor pagination has no offset cap (unlike Polymarket's 3000-offset
limit), so ``truncated`` is always ``False`` when ``mark_complete`` runs.
"""

from __future__ import annotations

import structlog

from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.ids import ManifoldMarketId
from pscanner.manifold.models import ManifoldBet

_log = structlog.get_logger(__name__)


async def walk_manifold_market(
    client: ManifoldClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    *,
    market_id: ManifoldMarketId,
    now_ts: int,
    page_size: int = 1000,
) -> int:
    """Backfill all fillable bets for one Manifold market into corpus_trades.

    Args:
        client: Open ``ManifoldClient``.
        markets_repo: Corpus markets repo (for state transitions).
        trades_repo: Corpus trades repo (for bet upserts).
        market_id: Manifold market hash ID.
        now_ts: Unix seconds, recorded as ``backfill_started_at`` /
            ``backfill_completed_at`` on the ``corpus_markets`` row.
        page_size: ``limit`` parameter on ``client.get_bets``.

    Returns:
        Count of inserted ``CorpusTrade`` rows (after the manifold-floor filter
        in ``CorpusTradesRepo.insert_batch``).
    """
    markets_repo.mark_in_progress(
        market_id, started_at=now_ts, platform="manifold"
    )
    inserted_total = 0
    examined_total = 0
    cursor: str | None = None
    while True:
        page = await client.get_bets(
            market_id=market_id, limit=page_size, before=cursor
        )
        if not page:
            break
        examined_total += len(page)
        trades = [
            _to_corpus_trade(bet, market_id=market_id)
            for bet in page
            if _is_fillable(bet)
        ]
        if trades:
            inserted_total += trades_repo.insert_batch(trades)
        cursor = page[-1].id
    markets_repo.mark_complete(
        market_id,
        completed_at=now_ts,
        truncated=False,
        platform="manifold",
    )
    _log.info(
        "manifold.walk_complete",
        market_id=market_id,
        examined=examined_total,
        inserted=inserted_total,
    )
    return inserted_total


def _is_fillable(bet: ManifoldBet) -> bool:
    """True iff the bet represents a real fill (not cancelled, not unfilled-limit)."""
    if bet.is_cancelled is True:
        return False
    if bet.limit_prob is not None and bet.is_filled is not True:
        return False
    return True


def _to_corpus_trade(bet: ManifoldBet, *, market_id: ManifoldMarketId) -> CorpusTrade:
    """Project a Manifold bet into the corpus dataclass.

    The synthetic ``asset_id = f"{market_id}:{outcome}"`` names the position;
    Manifold has no separate asset id but ``corpus_trades.asset_id`` is NOT NULL.

    Mana goes into ``notional_usd`` as platform-native units (per the spec's
    convention; downstream readers must group by ``platform`` before any
    USD-aggregating math).
    """
    return CorpusTrade(
        tx_hash=bet.id,
        asset_id=f"{market_id}:{bet.outcome}",
        wallet_address=bet.user_id,
        condition_id=market_id,
        outcome_side=bet.outcome,
        bs="BUY",
        price=bet.prob_before,
        size=bet.amount,
        notional_usd=bet.amount,
        ts=bet.created_time,
        platform="manifold",
    )
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/corpus/test_manifold_walker.py -v`
Expected: 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/manifold_walker.py tests/corpus/test_manifold_walker.py
git commit -m "feat(corpus): Manifold per-market bet walker"
```

---

### Task 8: `record_manifold_resolutions`

**Files:**
- Modify: `src/pscanner/corpus/resolutions.py`
- Modify: `tests/corpus/test_resolutions.py`

Mirrors `record_resolutions` but reads from `ManifoldClient.get_market(market_id)` and uses the `resolution` field directly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/corpus/test_resolutions.py`:

```python
class _FakeManifoldClient:
    """Tiny stub that returns a fixed market by id."""

    def __init__(self, markets: dict[str, ManifoldMarket]) -> None:
        self._markets = markets

    async def get_market(self, market_id: str) -> ManifoldMarket:
        return self._markets[market_id]


def _resolved_manifold_market(
    *, market_id: str, resolution: str | None
) -> ManifoldMarket:
    return ManifoldMarket.model_validate(
        {
            "id": market_id,
            "creatorId": "creator",
            "question": f"Question for {market_id}?",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "isResolved": True,
            "resolutionTime": 1_700_000_000,
            "resolution": resolution,
        }
    )


@pytest.mark.asyncio
async def test_record_manifold_resolutions_writes_yes_no(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """YES and NO resolutions land in market_resolutions with platform='manifold'."""
    from pscanner.corpus.resolutions import record_manifold_resolutions
    from pscanner.corpus.repos import MarketResolutionsRepo

    repo = MarketResolutionsRepo(tmp_corpus_db)
    client = _FakeManifoldClient(
        {
            "yes-market": _resolved_manifold_market(
                market_id="yes-market", resolution="YES"
            ),
            "no-market": _resolved_manifold_market(
                market_id="no-market", resolution="NO"
            ),
        }
    )
    written = await record_manifold_resolutions(
        client=client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo=repo,
        targets=[("yes-market", 1_700_000_000), ("no-market", 1_700_000_001)],
        now_ts=2_000_000_000,
    )
    assert written == 2
    yes_row = repo.get("yes-market", platform="manifold")
    no_row = repo.get("no-market", platform="manifold")
    assert yes_row is not None and yes_row.outcome_yes_won == 1
    assert no_row is not None and no_row.outcome_yes_won == 0
    assert yes_row.platform == "manifold"
    assert yes_row.source == "manifold-rest"


@pytest.mark.asyncio
async def test_record_manifold_resolutions_skips_mkt_and_cancel(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """MKT and CANCEL resolutions are logged + skipped — no market_resolutions row."""
    from pscanner.corpus.resolutions import record_manifold_resolutions
    from pscanner.corpus.repos import MarketResolutionsRepo

    repo = MarketResolutionsRepo(tmp_corpus_db)
    client = _FakeManifoldClient(
        {
            "mkt-market": _resolved_manifold_market(
                market_id="mkt-market", resolution="MKT"
            ),
            "cancel-market": _resolved_manifold_market(
                market_id="cancel-market", resolution="CANCEL"
            ),
            "null-market": _resolved_manifold_market(
                market_id="null-market", resolution=None
            ),
        }
    )
    written = await record_manifold_resolutions(
        client=client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo=repo,
        targets=[
            ("mkt-market", 1_700_000_000),
            ("cancel-market", 1_700_000_001),
            ("null-market", 1_700_000_002),
        ],
        now_ts=2_000_000_000,
    )
    assert written == 0
    assert repo.get("mkt-market", platform="manifold") is None
    assert repo.get("cancel-market", platform="manifold") is None
    assert repo.get("null-market", platform="manifold") is None
```

If `from pscanner.manifold.models import ManifoldMarket` and `import pytest` aren't already at the top of `tests/corpus/test_resolutions.py`, add them.

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_resolutions.py -k manifold -v`
Expected: 2 fails — `record_manifold_resolutions` doesn't exist.

- [ ] **Step 3: Add the function**

In `src/pscanner/corpus/resolutions.py`, append after the existing `record_resolutions`:

```python
from pscanner.manifold.client import ManifoldClient


async def record_manifold_resolutions(
    *,
    client: ManifoldClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, int]],
    now_ts: int,
) -> int:
    """Fetch resolution outcomes for resolved Manifold markets.

    For each target, calls ``ManifoldClient.get_market(market_id)`` and reads
    the ``resolution`` field. YES/NO produce a ``market_resolutions`` row;
    MKT, CANCEL, and ``None`` are logged and skipped (no row written, so the
    inner JOIN in ``build_features`` excludes them from ``training_examples``).

    Args:
        client: Open ``ManifoldClient``.
        repo: ``MarketResolutionsRepo`` to upsert into.
        targets: Iterable of ``(market_id, resolved_at_hint)``.
        now_ts: Unix seconds, recorded as ``recorded_at`` on each row.

    Returns:
        Count of resolutions actually written (excludes skipped MKT/CANCEL/null).
    """
    written = 0
    for market_id, resolved_at in targets:
        market = await client.get_market(market_id)  # type: ignore[arg-type]
        if market.resolution == "YES":
            outcome_yes_won = 1
            winning_outcome_index = 0
        elif market.resolution == "NO":
            outcome_yes_won = 0
            winning_outcome_index = 1
        else:
            _log.warning(
                "corpus.manifold_resolution_skipped",
                market_id=market_id,
                resolution=market.resolution,
            )
            continue
        repo.upsert(
            MarketResolution(
                condition_id=market_id,
                winning_outcome_index=winning_outcome_index,
                outcome_yes_won=outcome_yes_won,
                resolved_at=resolved_at,
                source="manifold-rest",
                platform="manifold",
            ),
            recorded_at=now_ts,
        )
        written += 1
    return written
```

The `# type: ignore[arg-type]` on `client.get_market(market_id)` is because `market_id` is a plain `str` from the targets iterable but `ManifoldClient.get_market` types its parameter as `ManifoldMarketId`. The runtime types are identical (NewType wrapper around `str`); the cast is only at the type-checker level.

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/corpus/test_resolutions.py -v`
Expected: all pass (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/resolutions.py tests/corpus/test_resolutions.py
git commit -m "feat(corpus): record_manifold_resolutions skips MKT/CANCEL"
```

---

### Task 9: `pscanner corpus backfill --platform manifold`

**Files:**
- Modify: `src/pscanner/corpus/cli.py:71-90, 268-285` (parser builder + `_cmd_backfill`)
- Modify: `tests/corpus/test_cli.py`

Add the `--platform` flag to the `backfill` subparser and dispatch to a new `_run_manifold_backfill` for `--platform manifold`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/corpus/test_cli.py`:

```python
def test_backfill_parser_accepts_platform_manifold() -> None:
    """`pscanner corpus backfill --platform manifold` parses correctly."""
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["backfill", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_backfill_parser_default_platform_is_polymarket() -> None:
    """Backfill's default platform is polymarket (preserves existing behavior)."""
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["backfill"])
    assert args.platform == "polymarket"


def test_backfill_parser_rejects_unknown_platform() -> None:
    """An unknown platform name fails argparse."""
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backfill", "--platform", "ftx"])
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_cli.py -k backfill_parser -v`
Expected: fail — `--platform` flag isn't on the `backfill` subparser yet.

- [ ] **Step 3: Add the flag and dispatcher**

In `src/pscanner/corpus/cli.py`, locate `build_corpus_parser` (around line 71) and find the `backfill` sub-parser. Add the flag immediately before the parser returns:

```python
    backfill = sub.add_parser("backfill", help="Bulk pull closed markets into the corpus")
    # ... existing flags ...
    backfill.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "manifold"],
        default="polymarket",
        help=(
            "Platform to ingest. Defaults to polymarket. "
            "`manifold` runs the Manifold REST enumerator + bet walker."
        ),
    )
```

Update `_cmd_backfill` (around line 268) to dispatch:

```python
async def _cmd_backfill(args: argparse.Namespace) -> int:
    """Run the corpus backfill for the requested platform."""
    if args.platform == "manifold":
        return await _run_manifold_backfill(args)
    return await _run_polymarket_backfill(args)
```

Rename the existing `_cmd_backfill` body to `_run_polymarket_backfill` (a new helper containing the existing gamma + data-client logic). Then add a new `_run_manifold_backfill` helper:

```python
async def _run_manifold_backfill(args: argparse.Namespace) -> int:
    """Manifold path: enumerate resolved binary markets, then walk each one."""
    from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets
    from pscanner.corpus.manifold_walker import walk_manifold_market
    from pscanner.manifold.client import ManifoldClient
    from pscanner.manifold.ids import ManifoldMarketId

    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    now_ts = int(datetime.datetime.now(tz=datetime.UTC).timestamp())
    try:
        async with ManifoldClient() as client:
            await enumerate_resolved_manifold_markets(
                client, markets_repo, now_ts=now_ts
            )
            while pending := markets_repo.next_pending(
                limit=10, platform="manifold"
            ):
                for market in pending:
                    await walk_manifold_market(
                        client,
                        markets_repo,
                        trades_repo,
                        market_id=ManifoldMarketId(market.condition_id),
                        now_ts=now_ts,
                    )
    finally:
        conn.close()
    return 0
```

(Adapt to the existing `_cmd_backfill` import pattern — the module-level `from datetime` and `from pathlib` imports are likely already in `cli.py`. Don't duplicate. Use module-level imports for `Path`, `datetime`, `init_corpus_db`, `CorpusMarketsRepo`, `CorpusTradesRepo` — only the manifold-specific imports go inside the function to keep the polymarket-only path's imports small.)

- [ ] **Step 4: Run all CLI tests**

Run: `uv run pytest tests/corpus/test_cli.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(corpus): \`pscanner corpus backfill --platform manifold\`"
```

---

### Task 10: `pscanner corpus refresh --platform manifold`

**Files:**
- Modify: `src/pscanner/corpus/cli.py` (refresh sub-parser + `_cmd_refresh`)
- Modify: `tests/corpus/test_cli.py`

Same pattern as Task 9 but for the `refresh` subcommand.

- [ ] **Step 1: Write the failing tests**

Add to `tests/corpus/test_cli.py`:

```python
def test_refresh_parser_accepts_platform_manifold() -> None:
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["refresh", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_refresh_parser_default_platform_is_polymarket() -> None:
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["refresh"])
    assert args.platform == "polymarket"


def test_refresh_parser_rejects_unknown_platform() -> None:
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["refresh", "--platform", "ftx"])
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_cli.py -k refresh_parser -v`
Expected: 3 fails.

- [ ] **Step 3: Add the flag and dispatcher**

In `src/pscanner/corpus/cli.py`, find the `refresh` sub-parser and add the same `--platform` flag with `choices=["polymarket", "manifold"]`, default `polymarket`. Then update `_cmd_refresh`:

```python
async def _cmd_refresh(args: argparse.Namespace) -> int:
    """Run the corpus refresh for the requested platform."""
    if args.platform == "manifold":
        return await _run_manifold_refresh(args)
    return await _run_polymarket_refresh(args)
```

Rename the existing `_cmd_refresh` body to `_run_polymarket_refresh`. Add the manifold helper:

```python
async def _run_manifold_refresh(args: argparse.Namespace) -> int:
    """Manifold refresh: re-enumerate, then record resolutions for missing markets."""
    from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets
    from pscanner.corpus.resolutions import record_manifold_resolutions
    from pscanner.manifold.client import ManifoldClient

    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    now_ts = int(datetime.datetime.now(tz=datetime.UTC).timestamp())
    try:
        async with ManifoldClient() as client:
            await enumerate_resolved_manifold_markets(
                client, markets_repo, now_ts=now_ts
            )
            # Find manifold markets in corpus_markets that lack a resolution.
            rows = conn.execute(
                "SELECT condition_id, closed_at FROM corpus_markets "
                "WHERE platform = 'manifold' AND backfill_state = 'complete'"
            ).fetchall()
            condition_ids = [row["condition_id"] for row in rows]
            missing = resolutions_repo.missing_for(condition_ids, platform="manifold")
            missing_set = set(missing)
            targets = [
                (row["condition_id"], int(row["closed_at"]))
                for row in rows
                if row["condition_id"] in missing_set
            ]
            await record_manifold_resolutions(
                client=client,
                repo=resolutions_repo,
                targets=targets,
                now_ts=now_ts,
            )
    finally:
        conn.close()
    return 0
```

(Add the `MarketResolutionsRepo` import at module top if it isn't already imported.)

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/corpus/test_cli.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(corpus): \`pscanner corpus refresh --platform manifold\`"
```

---

### Task 11: `pscanner corpus build-features --platform manifold`

**Files:**
- Modify: `src/pscanner/corpus/cli.py` (build-features sub-parser + `_cmd_build_features`)
- Modify: `tests/corpus/test_cli.py`

Smallest CLI change: build-features just forwards `args.platform` to the polymorphic `build_features` from PR A.

- [ ] **Step 1: Write the failing tests**

Add to `tests/corpus/test_cli.py`:

```python
def test_build_features_parser_accepts_platform_manifold() -> None:
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_build_features_parser_default_platform_is_polymarket() -> None:
    from pscanner.corpus.cli import build_corpus_parser

    parser = build_corpus_parser()
    args = parser.parse_args(["build-features"])
    assert args.platform == "polymarket"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_cli.py -k build_features_parser -v`
Expected: 2 fails.

- [ ] **Step 3: Add the flag and forward**

Find the `build-features` sub-parser in `build_corpus_parser` and add:

```python
    build_features.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "manifold"],
        default="polymarket",
        help="Platform whose corpus rows feed the training_examples build.",
    )
```

In `_cmd_build_features`, find the call to `build_features(...)` and add `platform=args.platform`:

```python
    written = build_features(
        # ... existing kwargs ...
        platform=args.platform,
    )
```

(The exact call shape depends on what's already in `_cmd_build_features` post-PR-A. Read the function first; only add the `platform=args.platform` argument to the existing `build_features(...)` invocation.)

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/corpus/test_cli.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(corpus): \`pscanner corpus build-features --platform manifold\`"
```

---

### Task 12: End-to-end test on a mixed-platform corpus

**Files:**
- Create: `tests/corpus/test_manifold_e2e.py`

Verify the end-to-end pipeline produces correct `training_examples` for the Manifold subset of a mixed-platform corpus.

- [ ] **Step 1: Write the failing test**

Create `tests/corpus/test_manifold_e2e.py`:

```python
"""End-to-end Manifold ingestion → build_features test on a mixed-platform corpus.

Seeds a synthetic corpus with both Polymarket and Manifold rows (one Manifold
market YES-resolved, one CANCEL-only). Runs ``build_features(platform="manifold")``
and asserts that only the YES-resolved Manifold market's bets produce
training_examples rows, and that those rows carry ``platform='manifold'``.
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
    """Drop in one polymarket market with one resolved YES outcome."""
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
    """Drop in one manifold market that resolves YES with two bets."""
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


def _seed_manifold_cancel(conn: sqlite3.Connection) -> None:
    """Drop in one manifold market that gets CANCELed (no market_resolutions row)."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="m-cancel",
            event_slug="m-cancel-slug",
            category="BINARY",
            closed_at=1_700_000_700,
            total_volume_usd=5_000.0,
            enumerated_at=1_700_000_000,
            market_slug="m-cancel-slug",
            platform="manifold",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="b-cancel-1",
                asset_id="m-cancel:YES",
                wallet_address="user-cancel",
                condition_id="m-cancel",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=200.0,
                notional_usd=200.0,
                ts=1_700_000_400,
                platform="manifold",
            ),
        ]
    )


def test_build_features_manifold_only_emits_manifold_training_examples(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """`build_features(platform="manifold")` produces only Manifold-tagged rows
    for the YES-resolved Manifold market; CANCEL market drops out via the JOIN."""
    _seed_polymarket_row(tmp_corpus_db)
    _seed_manifold_yes(tmp_corpus_db)
    _seed_manifold_cancel(tmp_corpus_db)

    examples_repo = TrainingExamplesRepo(tmp_corpus_db)
    written = build_features(
        conn=tmp_corpus_db,
        examples_repo=examples_repo,
        rebuild=True,
        platform="manifold",
    )
    assert written >= 1  # at least one BUY emits a training example

    rows = tmp_corpus_db.execute(
        "SELECT platform, condition_id, tx_hash FROM training_examples ORDER BY tx_hash"
    ).fetchall()
    assert all(r["platform"] == "manifold" for r in rows)
    # Only the YES-resolved manifold market's bets should appear; CANCEL's bet
    # has no market_resolutions row so it drops via the inner JOIN.
    assert {r["condition_id"] for r in rows} == {"m-yes"}
    assert "b-cancel-1" not in {r["tx_hash"] for r in rows}
```

The exact `build_features` call signature was set in PR A Task 10. Read `src/pscanner/corpus/examples.py:build_features` to confirm the parameters match. Adapt the call accordingly — typical shape post-PR-A is `build_features(conn, examples_repo, rebuild=True, platform=...)`.

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/corpus/test_manifold_e2e.py -v`
Expected: pass on first run because the prior tasks already wired the platform filter through `build_features`. The mixed-platform seed is what makes this E2E meaningful.

If it fails: the most likely cause is that `build_features`'s signature doesn't match what the test calls. Inspect `src/pscanner/corpus/examples.py:build_features` and adjust the call site.

- [ ] **Step 3: Commit**

```bash
git add tests/corpus/test_manifold_e2e.py
git commit -m "test(corpus): manifold ingestion → build_features E2E on mixed-platform corpus"
```

---

### Task 13: CLAUDE.md note + final verify

**Files:**
- Modify: `CLAUDE.md`

Add a single bullet under "Codebase conventions" documenting the Manifold ingestion shape and conventions. Then run the full quick-verify suite.

- [ ] **Step 1: Add the CLAUDE.md bullet**

Find the existing "Codebase conventions" section. After the "platform column on shared corpus tables" bullet (added in PR #82), append:

```markdown
- **Manifold ingestion shape (per the integration spec).** `pscanner corpus backfill --platform manifold` enumerates resolved binary markets via `/v0/markets`, then walks `/v0/bets?contractId=<market_id>` per market into `corpus_trades`. Mana lands in `corpus_trades.notional_usd` as platform-native units (NOT USD — never aggregate Manifold volumes into real-money totals without grouping by `platform` first). `bet.user_id` is stored in `corpus_trades.wallet_address` (column-reuse convention, same as `condition_id`). The notional floor is per-platform via `pscanner.corpus.repos._NOTIONAL_FLOORS` (Polymarket: $10, Manifold: 100 mana). MKT/CANCEL resolutions land in `corpus_markets` and `corpus_trades` but are skipped by `record_manifold_resolutions` so they have no `market_resolutions` row — they drop out of `training_examples` automatically via the inner JOIN. Build features with `pscanner corpus build-features --platform manifold` and train with `pscanner ml train --platform manifold` (no new ML code; PR A's polymorphic pipeline does the work).
```

- [ ] **Step 2: Run the full quick-verify**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: clean. The ty diagnostic count should not increase from the post-PR-#83 baseline (~13 in tracked files). The new tests use the same `# type: ignore[arg-type]  # ty:ignore[invalid-argument-type]` doubled annotation pattern when stubbing `ManifoldClient` with a `_FakeManifoldClient` — that's the project convention.

If `ty check` shows new diagnostics, the most likely cause is a stub class that doesn't quite match `ManifoldClient`'s interface. Inspect the test file and either tighten the stub (add the missing attributes) or extend the doubled-ignore on the call site.

If `ruff format --check` reports differences, run `uv run ruff format .` and stage the result.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document Manifold ingestion conventions"
```

---

## Self-review

**Spec coverage:**
- ✅ Module gap: `resolution` field on model → Task 1
- ✅ Module gap: `resolution` column on `manifold_markets` → Task 2
- ✅ Module gap: `resolution` round-trip in repo → Task 3
- ✅ `init_db` wires Manifold tables → Task 4
- ✅ Platform-aware notional floor → Task 5
- ✅ Manifold enumerator → Task 6
- ✅ Manifold bet walker → Task 7
- ✅ `record_manifold_resolutions` → Task 8
- ✅ CLI `--platform manifold` on `backfill` → Task 9
- ✅ CLI `--platform manifold` on `refresh` → Task 10
- ✅ CLI `--platform manifold` on `build-features` → Task 11
- ✅ E2E test on mixed-platform corpus → Task 12
- ✅ CLAUDE.md note + final verify → Task 13

**Placeholder scan:** Task 11 step 3 says "The exact call shape depends on what's already in `_cmd_build_features` post-PR-A. Read the function first" — that's an instruction to inspect-then-adapt, not a TBD. The implementer has clear guidance: add `platform=args.platform` to whatever the existing `build_features(...)` invocation looks like. Acceptable.

Task 9 step 3 has a similar inspect-then-adapt instruction for the existing import pattern in `cli.py`. Same shape, acceptable.

**Type consistency:** `platform: str = "polymarket"` is uniform across all signatures (matches PR A's pattern). `_NOTIONAL_FLOORS` keys match the schema CHECK constraint exactly (`'polymarket'`, `'kalshi'`, `'manifold'`). `record_manifold_resolutions` upserts with `platform="manifold"` and `source="manifold-rest"`; the test matches both literals.

---

## Out of scope (explicit non-goals)

- Kalshi ingestion — separate spec.
- Manifold WebSocket bet firehose (`src/pscanner/manifold/ws.py`) — useful for future live-signal work, but solves a different problem than corpus backfill.
- Manifold daemon-side detector instances or paper-trading evaluators — Stage 2.
- Multi-platform aggregation in ML training — already deferred per the platform-filter spec.
- CFMM (multi-outcome) markets — the binary-only filter stays.

---

## Risks

- **`ManifoldMarket.resolution` API field name.** The spec assumes the API returns `resolution` as a top-level field. If Manifold's actual response shape differs (e.g., it nests under a different key, or uses camelCase like `resolutionOutcome`), the model parsing in Task 1 will accept null but never see real values. Mitigation: the integration test in Task 12 seeds the field directly via the repo, not via the network — so the field name doesn't affect test correctness. First live run against `/v0/market/{id}` will confirm. If the field name is different, change the alias in Task 1 and re-test.
- **Volume gate `1000` mana might be too aggressive or too loose.** Tunable via the `min_volume_mana` parameter; the first live run logs `manifold.enumerate_complete` with `examined`/`inserted` counts.
- **`build_features` E2E test may need parameter shape adjustments.** PR A Task 10 made the function platform-aware but the exact kwargs (e.g., whether it takes a `conn` or builds one from a path) might require reading the actual implementation. Task 12 step 2 covers this: re-read `src/pscanner/corpus/examples.py:build_features` and adapt the call.
- **`ManifoldClient.get_market` returns the new `resolution` field after Task 1.** The runtime parsing in Task 1 just adds the field to the pydantic model. The client's `_get_raw` returns the raw dict and `model_validate` consumes it — the new field flows through automatically. But if Stage-1 client code did anything to strip unknown fields, that path needs to be re-checked. (Spot check: the model uses `extra="ignore"`, which doesn't strip fields the model knows about; it only ignores genuinely unknown fields. Adding `resolution` to the model means it's no longer "unknown".)
