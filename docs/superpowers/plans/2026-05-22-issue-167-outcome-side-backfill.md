# Outcome-side Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pscanner corpus backfill-outcome-side`, a one-shot CLI subcommand that rewrites the 1,697 binary markets whose `asset_index` + `corpus_trades` rows currently store both legs as `outcome_side=NO` (the pre-#166 bug).

**Architecture:** Discover buggy markets via a `GROUP BY ... HAVING COUNT(DISTINCT outcome_side)=1` query on `asset_index`. For each market, lookup the correct `{token_id: YES/NO}` mapping via `data.get_market_slug_by_condition_id` → `gamma.get_market_by_slug` → `Market.clob_token_ids` position. Rewrite both `asset_index` (2 rows) and `corpus_trades` (~2-3K rows per market via index-seek on `idx_corpus_trades_market_ts`) in one per-market transaction. Resumable via a new `corpus_markets.outcome_side_backfilled_at` sentinel column.

**Tech Stack:** Python 3.13, asyncio, SQLite, pytest, structlog. Reuses the existing `pscanner.poly.data.DataClient` + `pscanner.poly.gamma.GammaClient` infrastructure.

**Spec:** `docs/superpowers/specs/2026-05-22-issue-167-outcome-side-backfill-design.md`

---

## File Structure

**New files:**

- `src/pscanner/corpus/outcome_side_backfill.py` — module with:
  - `find_buggy_markets(conn)` — discovery SQL
  - `resolve_correct_mapping(condition_id, data, gamma)` — gamma lookup helper
  - `apply_market_backfill(conn, condition_id, mapping)` — per-market UPDATEs in one transaction
  - `validate_backfill_state(conn)` — post-run health check
  - `run_backfill(conn, data, gamma, *, dry_run, limit)` — orchestrator
- `tests/corpus/test_outcome_side_backfill.py` — unit tests
- `tests/corpus/test_outcome_side_backfill_integration.py` — end-to-end test

**Modified files:**

- `src/pscanner/corpus/db.py` — add `ALTER TABLE corpus_markets ADD COLUMN outcome_side_backfilled_at INTEGER` to `_MIGRATIONS`
- `src/pscanner/corpus/cli.py` — add `_cmd_backfill_outcome_side` function + argparse subparser + entry in `_SUBCOMMANDS` dict
- `CLAUDE.md` — add a one-liner to the `## CLI surface` section documenting the new command

---

### Task 1: Schema migration for the sentinel column

**Files:**
- Modify: `src/pscanner/corpus/db.py`
- Test: existing `tests/corpus/test_db.py` (or its equivalent — verify by running the existing schema tests)

- [ ] **Step 1: Write a failing assertion for the new column**

Create `tests/corpus/test_outcome_side_backfilled_column.py`:

```python
"""Confirms the corpus_markets sentinel column lands via _MIGRATIONS."""

from __future__ import annotations

from pscanner.corpus.db import init_corpus_db


def test_corpus_markets_has_outcome_side_backfilled_at_column(tmp_path) -> None:
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cols = [row[1] for row in conn.execute("PRAGMA table_info(corpus_markets)")]
    assert "outcome_side_backfilled_at" in cols
    conn.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_outcome_side_backfilled_column.py -q`
Expected: FAIL with `AssertionError: assert 'outcome_side_backfilled_at' in [...]` (column missing).

- [ ] **Step 3: Add the migration**

Open `src/pscanner/corpus/db.py`, find the `_MIGRATIONS` tuple (around line 281), and add an entry to the tuple:

```python
    "ALTER TABLE corpus_markets ADD COLUMN outcome_side_backfilled_at INTEGER",
```

Place it adjacent to other recent `ALTER TABLE corpus_markets ADD COLUMN ...` lines. `_apply_migrations` already swallows the `"duplicate column name"` `OperationalError` so re-running `init_corpus_db` after the migration is a no-op.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/corpus/test_outcome_side_backfilled_column.py tests/corpus/test_db.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/db.py tests/corpus/test_outcome_side_backfilled_column.py
git commit -m "feat(corpus): add outcome_side_backfilled_at to corpus_markets (#167)"
```

---

### Task 2: Buggy-market discovery query

**Files:**
- Create: `src/pscanner/corpus/outcome_side_backfill.py`
- Create: `tests/corpus/test_outcome_side_backfill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/corpus/test_outcome_side_backfill.py`:

```python
"""Unit tests for the outcome_side backfill (#167)."""

from __future__ import annotations

import sqlite3
import time

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.outcome_side_backfill import find_buggy_markets


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    db = init_corpus_db(tmp_path / "corpus.sqlite3")
    yield db
    db.close()


def _seed_asset(
    conn: sqlite3.Connection,
    *,
    condition_id: str,
    asset_id: str,
    outcome_side: str,
    outcome_index: int,
) -> None:
    conn.execute(
        "INSERT INTO asset_index (platform, asset_id, condition_id, outcome_side, outcome_index) "
        "VALUES ('polymarket', ?, ?, ?, ?)",
        (asset_id, condition_id, outcome_side, outcome_index),
    )
    conn.commit()


def _seed_corpus_market(conn: sqlite3.Connection, condition_id: str) -> None:
    conn.execute(
        "INSERT INTO corpus_markets (platform, condition_id, event_slug, market_slug, "
        " category, enumerated_at, total_volume_usd, backfill_state) "
        "VALUES ('polymarket', ?, 'evt', ?, 'sports', ?, 0, 'complete')",
        (condition_id, f"slug-{condition_id}", int(time.time())),
    )
    conn.commit()


def test_find_buggy_markets_returns_only_no_no_pairs(conn: sqlite3.Connection) -> None:
    # Buggy: both legs NO
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t2", outcome_side="NO", outcome_index=1)

    # Correct: YES + NO
    _seed_corpus_market(conn, "correct1")
    _seed_asset(conn, condition_id="correct1", asset_id="t3", outcome_side="YES", outcome_index=0)
    _seed_asset(conn, condition_id="correct1", asset_id="t4", outcome_side="NO", outcome_index=1)

    # Single-leg (multi-outcome or partial data)
    _seed_corpus_market(conn, "single1")
    _seed_asset(conn, condition_id="single1", asset_id="t5", outcome_side="NO", outcome_index=1)

    buggy = find_buggy_markets(conn)
    assert buggy == ["buggy1"]


def test_find_buggy_markets_excludes_already_backfilled(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t2", outcome_side="NO", outcome_index=1)
    conn.execute(
        "UPDATE corpus_markets SET outcome_side_backfilled_at = ? WHERE condition_id = ?",
        (1_700_000_000, "buggy1"),
    )
    conn.commit()

    assert find_buggy_markets(conn) == []


def test_find_buggy_markets_includes_markets_with_no_corpus_markets_row(
    conn: sqlite3.Connection,
) -> None:
    # Some asset_index entries exist without a matching corpus_markets row
    # (e.g. populated via the live token_resolver). They should still surface
    # if they're NO+NO so the operator gets full coverage.
    _seed_asset(conn, condition_id="orphan1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="orphan1", asset_id="t2", outcome_side="NO", outcome_index=1)
    assert find_buggy_markets(conn) == ["orphan1"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: FAIL with `ImportError: cannot import name 'find_buggy_markets' from 'pscanner.corpus.outcome_side_backfill'`.

- [ ] **Step 3: Implement the discovery query**

Create `src/pscanner/corpus/outcome_side_backfill.py`:

```python
"""Backfill incorrect ``outcome_side`` values introduced by the pre-#166 bug.

Issue #167. Spec: ``docs/superpowers/specs/2026-05-22-issue-167-outcome-side-backfill-design.md``.

``market_walker._parse_trade`` used to collapse every non-``yes`` outcome
label to ``NO`` (#159), writing both legs of binary sports/esports markets
as ``outcome_side=NO`` in ``corpus_trades`` and downstream ``asset_index``.
PR #166 forward-fixed the parser; this module rewrites the historical rows.
"""

from __future__ import annotations

import sqlite3


def find_buggy_markets(conn: sqlite3.Connection) -> list[str]:
    """Return the ``condition_id``s of binary markets stored as NO+NO.

    A market is buggy when ``asset_index`` has exactly 2 rows for it,
    both with ``outcome_side='NO'``. Excludes markets already marked
    backfilled via ``corpus_markets.outcome_side_backfilled_at``.

    Markets with no matching ``corpus_markets`` row still surface — the
    backfill should reach them too (they get a sentinel row created later).
    """
    rows = conn.execute(
        """
        SELECT ai.condition_id
          FROM asset_index ai
          LEFT JOIN corpus_markets cm
                 ON cm.condition_id = ai.condition_id AND cm.platform = ai.platform
         WHERE ai.platform = 'polymarket'
           AND (cm.outcome_side_backfilled_at IS NULL OR cm.condition_id IS NULL)
         GROUP BY ai.condition_id
         HAVING COUNT(*) = 2
            AND COUNT(DISTINCT ai.outcome_side) = 1
            AND MIN(ai.outcome_side) = 'NO'
         ORDER BY ai.condition_id
        """,
    ).fetchall()
    return [row[0] for row in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/outcome_side_backfill.py tests/corpus/test_outcome_side_backfill.py
git commit -m "feat(corpus): outcome_side backfill — buggy-market discovery (#167)"
```

---

### Task 3: Per-market mapping resolver

**Files:**
- Modify: `src/pscanner/corpus/outcome_side_backfill.py`
- Modify: `tests/corpus/test_outcome_side_backfill.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/corpus/test_outcome_side_backfill.py`:

```python
from unittest.mock import AsyncMock

from pscanner.corpus.outcome_side_backfill import resolve_correct_mapping
from pscanner.poly.models import Market


def _make_market(*, condition_id: str, outcomes: tuple[str, str], tokens: tuple[str, str]) -> Market:
    return Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": f"slug-{condition_id}",
            "conditionId": condition_id,
            "outcomes": list(outcomes),
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": list(tokens),
            "active": True,
            "closed": False,
        }
    )


def _fake_data(slug: str | None) -> AsyncMock:
    data = AsyncMock()
    data.get_market_slug_by_condition_id = AsyncMock(return_value=slug)
    return data


def _fake_gamma(market: Market | None) -> AsyncMock:
    gamma = AsyncMock()
    gamma.get_market_by_slug = AsyncMock(return_value=market)
    return gamma


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_yes_no_dict() -> None:
    data = _fake_data("slug-cond1")
    market = _make_market(
        condition_id="cond1",
        outcomes=("Cavaliers", "Knicks"),
        tokens=("token-cavs", "token-knicks"),
    )
    gamma = _fake_gamma(market)
    mapping = await resolve_correct_mapping("cond1", data=data, gamma=gamma)
    assert mapping == {"token-cavs": ("YES", 0), "token-knicks": ("NO", 1)}


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_missing_slug() -> None:
    data = _fake_data(None)
    gamma = _fake_gamma(None)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None
    gamma.get_market_by_slug.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_missing_market() -> None:
    data = _fake_data("slug-cond1")
    gamma = _fake_gamma(None)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_non_binary() -> None:
    data = _fake_data("slug-cond1")
    three_outcome = Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": "slug-cond1",
            "outcomes": ["A", "B", "C"],
            "outcomePrices": ["0.33", "0.33", "0.34"],
            "clobTokenIds": ["t-a", "t-b", "t-c"],
            "active": True,
            "closed": False,
        }
    )
    gamma = _fake_gamma(three_outcome)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_data_exception() -> None:
    data = AsyncMock()
    data.get_market_slug_by_condition_id = AsyncMock(side_effect=RuntimeError("boom"))
    gamma = _fake_gamma(None)
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_resolve_correct_mapping_returns_none_on_gamma_exception() -> None:
    data = _fake_data("slug-cond1")
    gamma = AsyncMock()
    gamma.get_market_by_slug = AsyncMock(side_effect=RuntimeError("boom"))
    assert await resolve_correct_mapping("cond1", data=data, gamma=gamma) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 6 new tests fail with `ImportError: cannot import name 'resolve_correct_mapping'`.

- [ ] **Step 3: Implement the resolver**

Append to `src/pscanner/corpus/outcome_side_backfill.py`:

```python
import structlog

from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)

_BINARY_MARKET_OUTCOME_COUNT = 2


async def resolve_correct_mapping(
    condition_id: str,
    *,
    data: DataClient,
    gamma: GammaClient,
) -> dict[str, tuple[str, int]] | None:
    """Return ``{token_id: (outcome_side, outcome_index)}`` for ``condition_id``.

    Uses the established ``data.get_market_slug_by_condition_id`` →
    ``gamma.get_market_by_slug`` chain (the same one ``PaperTrader._backfill_market_cache``
    and ``market_walker.walk_market`` use post-#166).

    Returns ``None`` when:
    - either client raises
    - the slug lookup returns ``None``
    - the gamma market lookup returns ``None``
    - the market has ``len(clob_token_ids) != 2`` (non-binary)

    The caller treats ``None`` as "skip this market, no sentinel written".
    """
    try:
        slug = await data.get_market_slug_by_condition_id(condition_id)
    except Exception:
        _log.warning("corpus.backfill_outcome_side.slug_lookup_failed", condition_id=condition_id)
        return None
    if slug is None:
        return None
    try:
        market = await gamma.get_market_by_slug(slug)
    except Exception:
        _log.warning(
            "corpus.backfill_outcome_side.gamma_lookup_failed",
            condition_id=condition_id,
            slug=slug,
        )
        return None
    if market is None:
        return None
    if len(market.clob_token_ids) != _BINARY_MARKET_OUTCOME_COUNT:
        _log.info(
            "corpus.backfill_outcome_side.not_binary",
            condition_id=condition_id,
            n_outcomes=len(market.clob_token_ids),
        )
        return None
    return {
        str(market.clob_token_ids[0]): ("YES", 0),
        str(market.clob_token_ids[1]): ("NO", 1),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/outcome_side_backfill.py tests/corpus/test_outcome_side_backfill.py
git commit -m "feat(corpus): outcome_side backfill — per-market mapping resolver (#167)"
```

---

### Task 4: Per-market UPDATE applier

**Files:**
- Modify: `src/pscanner/corpus/outcome_side_backfill.py`
- Modify: `tests/corpus/test_outcome_side_backfill.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/corpus/test_outcome_side_backfill.py`:

```python
from pscanner.corpus.outcome_side_backfill import apply_market_backfill


def _seed_trade(
    conn: sqlite3.Connection,
    *,
    condition_id: str,
    asset_id: str,
    outcome_side: str,
    tx_hash: str,
) -> None:
    conn.execute(
        "INSERT INTO corpus_trades (platform, tx_hash, asset_id, wallet_address, "
        " condition_id, outcome_side, bs, price, size, notional_usd, ts) "
        "VALUES ('polymarket', ?, ?, '0xWALLET', ?, ?, 'BUY', 0.5, 100.0, 50.0, 1)",
        (tx_hash, asset_id, condition_id, outcome_side),
    )
    conn.commit()


def test_apply_market_backfill_rewrites_asset_index_and_corpus_trades(
    conn: sqlite3.Connection,
) -> None:
    _seed_corpus_market(conn, "cond1")
    _seed_asset(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="cond1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    _seed_trade(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", tx_hash="0xA")
    _seed_trade(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", tx_hash="0xB")
    _seed_trade(conn, condition_id="cond1", asset_id="t-no", outcome_side="NO", tx_hash="0xC")

    mapping = {"t-yes": ("YES", 0), "t-no": ("NO", 1)}
    apply_market_backfill(conn, "cond1", mapping, now_ts=1_700_000_500)

    # asset_index updated to YES + NO
    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='cond1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "YES", "t-no": "NO"}

    # corpus_trades updated: YES on t-yes rows, NO on t-no rows
    yes_rows = [
        r[0]
        for r in conn.execute(
            "SELECT tx_hash FROM corpus_trades WHERE condition_id='cond1' AND outcome_side='YES' ORDER BY tx_hash"
        )
    ]
    assert yes_rows == ["0xA", "0xB"]
    no_rows = [
        r[0]
        for r in conn.execute(
            "SELECT tx_hash FROM corpus_trades WHERE condition_id='cond1' AND outcome_side='NO' ORDER BY tx_hash"
        )
    ]
    assert no_rows == ["0xC"]

    # Sentinel set
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()[0]
    assert sentinel == 1_700_000_500


def test_apply_market_backfill_idempotent(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "cond1")
    _seed_asset(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="cond1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    _seed_trade(conn, condition_id="cond1", asset_id="t-yes", outcome_side="NO", tx_hash="0xA")
    mapping = {"t-yes": ("YES", 0), "t-no": ("NO", 1)}

    apply_market_backfill(conn, "cond1", mapping, now_ts=1_700_000_500)
    apply_market_backfill(conn, "cond1", mapping, now_ts=1_700_000_999)

    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='cond1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "YES", "t-no": "NO"}
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()[0]
    assert sentinel == 1_700_000_999  # last write wins


def test_apply_market_backfill_creates_sentinel_row_when_missing(
    conn: sqlite3.Connection,
) -> None:
    # No corpus_markets row at all (orphan asset_index entries from token_resolver)
    _seed_asset(conn, condition_id="orphan1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="orphan1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    mapping = {"t-yes": ("YES", 0), "t-no": ("NO", 1)}

    apply_market_backfill(conn, "orphan1", mapping, now_ts=1_700_000_500)

    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='orphan1'"
    ).fetchone()
    # We DON'T fabricate a corpus_markets row — the sentinel write is a no-op
    # on no-row. The point is that find_buggy_markets gates on
    # asset_index leg-sides, so once the asset_index is corrected, the
    # market drops out of the work queue regardless of sentinel state.
    assert sentinel is None

    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='orphan1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "YES", "t-no": "NO"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 3 new tests fail with `ImportError: cannot import name 'apply_market_backfill'`.

- [ ] **Step 3: Implement the applier**

Append to `src/pscanner/corpus/outcome_side_backfill.py`:

```python
def apply_market_backfill(
    conn: sqlite3.Connection,
    condition_id: str,
    mapping: dict[str, tuple[str, int]],
    *,
    now_ts: int,
) -> None:
    """Rewrite ``asset_index`` + ``corpus_trades`` for one market in one transaction.

    Args:
        conn: Corpus DB connection (caller owns lifecycle).
        condition_id: Market identifier to backfill.
        mapping: ``{token_id: (outcome_side, outcome_index)}`` from
            :func:`resolve_correct_mapping`. Always 2 entries.
        now_ts: Sentinel timestamp written to ``corpus_markets``.

    Transaction shape (auto-commit on success, rollback on raise):
      1. 2 × UPDATE asset_index (one per leg, defensive idempotent on correct legs).
      2. 2 × UPDATE corpus_trades (index-seek on ``idx_corpus_trades_market_ts``).
      3. 1 × UPDATE corpus_markets (sentinel). No-op when no row exists.

    The ``corpus_trades`` UPDATE rewrites the ``outcome_side`` portion of
    the composite PK. SQLite handles this as delete-then-insert at the
    storage layer; uniqueness is preserved because ``tx_hash`` is in the
    PK and globally unique per trade.
    """
    with conn:
        for token_id, (side, idx) in mapping.items():
            conn.execute(
                """
                UPDATE asset_index
                   SET outcome_side = ?, outcome_index = ?
                 WHERE platform = 'polymarket'
                   AND asset_id = ?
                """,
                (side, idx, token_id),
            )
            conn.execute(
                """
                UPDATE corpus_trades
                   SET outcome_side = ?
                 WHERE platform = 'polymarket'
                   AND condition_id = ?
                   AND asset_id = ?
                """,
                (side, condition_id, token_id),
            )
        conn.execute(
            """
            UPDATE corpus_markets
               SET outcome_side_backfilled_at = ?
             WHERE platform = 'polymarket'
               AND condition_id = ?
            """,
            (now_ts, condition_id),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/outcome_side_backfill.py tests/corpus/test_outcome_side_backfill.py
git commit -m "feat(corpus): outcome_side backfill — per-market UPDATE applier (#167)"
```

---

### Task 5: Validation helper

**Files:**
- Modify: `src/pscanner/corpus/outcome_side_backfill.py`
- Modify: `tests/corpus/test_outcome_side_backfill.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/corpus/test_outcome_side_backfill.py`:

```python
from pscanner.corpus.outcome_side_backfill import validate_backfill_state


def test_validate_backfill_state_returns_zero_when_clean(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "cond1")
    _seed_asset(conn, condition_id="cond1", asset_id="t1", outcome_side="YES", outcome_index=0)
    _seed_asset(conn, condition_id="cond1", asset_id="t2", outcome_side="NO", outcome_index=1)
    assert validate_backfill_state(conn) == 0


def test_validate_backfill_state_counts_remaining_buggy(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t2", outcome_side="NO", outcome_index=1)

    _seed_corpus_market(conn, "buggy2")
    _seed_asset(conn, condition_id="buggy2", asset_id="t3", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy2", asset_id="t4", outcome_side="NO", outcome_index=1)

    assert validate_backfill_state(conn) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 2 new tests fail with `ImportError: cannot import name 'validate_backfill_state'`.

- [ ] **Step 3: Implement the validator**

Append to `src/pscanner/corpus/outcome_side_backfill.py`:

```python
def validate_backfill_state(conn: sqlite3.Connection) -> int:
    """Return the count of binary markets still stored as NO+NO in asset_index.

    A clean post-backfill state returns ``0``. Non-zero indicates markets
    that failed resolution (gamma missing, non-binary, etc.); operator
    should re-run.

    Unlike :func:`find_buggy_markets`, this ignores the sentinel column
    so it reports the true asset_index health, not the work-queue size.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT condition_id
            FROM asset_index
           WHERE platform = 'polymarket'
           GROUP BY condition_id
          HAVING COUNT(*) = 2
             AND COUNT(DISTINCT outcome_side) = 1
             AND MIN(outcome_side) = 'NO'
        )
        """,
    ).fetchone()
    return int(row[0])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/outcome_side_backfill.py tests/corpus/test_outcome_side_backfill.py
git commit -m "feat(corpus): outcome_side backfill — validation helper (#167)"
```

---

### Task 6: Orchestrator

**Files:**
- Modify: `src/pscanner/corpus/outcome_side_backfill.py`
- Modify: `tests/corpus/test_outcome_side_backfill.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/corpus/test_outcome_side_backfill.py`:

```python
from pscanner.corpus.outcome_side_backfill import run_backfill


@pytest.mark.asyncio
async def test_run_backfill_processes_buggy_markets(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t-no", outcome_side="NO", outcome_index=1)
    _seed_trade(conn, condition_id="buggy1", asset_id="t-yes", outcome_side="NO", tx_hash="0xA")

    data = _fake_data("slug-buggy1")
    market = _make_market(
        condition_id="buggy1",
        outcomes=("Cavaliers", "Knicks"),
        tokens=("t-yes", "t-no"),
    )
    gamma = _fake_gamma(market)

    stats = await run_backfill(conn, data=data, gamma=gamma, dry_run=False, limit=None)

    assert stats == {"processed": 1, "resolved": 1, "skipped": 0, "remaining": 0}
    assert validate_backfill_state(conn) == 0


@pytest.mark.asyncio
async def test_run_backfill_dry_run_skips_writes(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t-yes", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t-no", outcome_side="NO", outcome_index=1)

    data = _fake_data("slug-buggy1")
    market = _make_market(
        condition_id="buggy1",
        outcomes=("Cavaliers", "Knicks"),
        tokens=("t-yes", "t-no"),
    )
    gamma = _fake_gamma(market)

    stats = await run_backfill(conn, data=data, gamma=gamma, dry_run=True, limit=None)

    assert stats == {"processed": 1, "resolved": 1, "skipped": 0, "remaining": 1}
    # asset_index unchanged
    ai = dict(
        conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='buggy1'"
        ).fetchall()
    )
    assert ai == {"t-yes": "NO", "t-no": "NO"}


@pytest.mark.asyncio
async def test_run_backfill_skips_unresolvable_markets(conn: sqlite3.Connection) -> None:
    _seed_corpus_market(conn, "buggy1")
    _seed_asset(conn, condition_id="buggy1", asset_id="t1", outcome_side="NO", outcome_index=1)
    _seed_asset(conn, condition_id="buggy1", asset_id="t2", outcome_side="NO", outcome_index=1)

    data = _fake_data(None)  # slug not found
    gamma = _fake_gamma(None)
    stats = await run_backfill(conn, data=data, gamma=gamma, dry_run=False, limit=None)

    assert stats == {"processed": 1, "resolved": 0, "skipped": 1, "remaining": 1}
    # Sentinel NOT written so the market re-queues on the next run
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='buggy1'"
    ).fetchone()[0]
    assert sentinel is None


@pytest.mark.asyncio
async def test_run_backfill_limit_caps_work(conn: sqlite3.Connection) -> None:
    for cid in ("a1", "a2", "a3"):
        _seed_corpus_market(conn, cid)
        _seed_asset(conn, condition_id=cid, asset_id=f"y-{cid}", outcome_side="NO", outcome_index=1)
        _seed_asset(conn, condition_id=cid, asset_id=f"n-{cid}", outcome_side="NO", outcome_index=1)

    data = AsyncMock()
    data.get_market_slug_by_condition_id = AsyncMock(side_effect=lambda c: f"slug-{c}")
    gamma = AsyncMock()

    def _market_for(slug: str) -> Market:
        cid = slug.replace("slug-", "")
        return _make_market(
            condition_id=cid,
            outcomes=("A", "B"),
            tokens=(f"y-{cid}", f"n-{cid}"),
        )

    gamma.get_market_by_slug = AsyncMock(side_effect=lambda slug: _market_for(slug))

    stats = await run_backfill(conn, data=data, gamma=gamma, dry_run=False, limit=2)
    assert stats["processed"] == 2
    assert stats["resolved"] == 2
    assert stats["remaining"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 4 new tests fail with `ImportError: cannot import name 'run_backfill'`.

- [ ] **Step 3: Implement the orchestrator**

Append to `src/pscanner/corpus/outcome_side_backfill.py`:

```python
import time


async def run_backfill(
    conn: sqlite3.Connection,
    *,
    data: DataClient,
    gamma: GammaClient,
    dry_run: bool,
    limit: int | None,
) -> dict[str, int]:
    """Walk every buggy market, resolve the correct mapping, apply UPDATEs.

    Returns a stats dict: ``processed``, ``resolved``, ``skipped``,
    ``remaining`` (post-run count of NO+NO markets still in asset_index).

    Per-market failures (gamma missing, non-binary) increment ``skipped``
    and do NOT touch the DB; the market stays in the work queue for a
    future re-run.
    """
    buggy = find_buggy_markets(conn)
    if limit is not None:
        buggy = buggy[:limit]

    processed = 0
    resolved = 0
    skipped = 0
    for condition_id in buggy:
        mapping = await resolve_correct_mapping(condition_id, data=data, gamma=gamma)
        processed += 1
        if mapping is None:
            skipped += 1
            continue
        resolved += 1
        if dry_run:
            _log.info(
                "corpus.backfill_outcome_side.dry_run_resolve",
                condition_id=condition_id,
                mapping={k: v[0] for k, v in mapping.items()},
            )
            continue
        apply_market_backfill(conn, condition_id, mapping, now_ts=int(time.time()))
        if processed % 100 == 0:
            _log.info(
                "corpus.backfill_outcome_side.progress",
                processed=processed,
                resolved=resolved,
                skipped=skipped,
            )

    remaining = validate_backfill_state(conn)
    _log.info(
        "corpus.backfill_outcome_side.done",
        processed=processed,
        resolved=resolved,
        skipped=skipped,
        remaining=remaining,
        dry_run=dry_run,
    )
    return {
        "processed": processed,
        "resolved": resolved,
        "skipped": skipped,
        "remaining": remaining,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill.py -q`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/outcome_side_backfill.py tests/corpus/test_outcome_side_backfill.py
git commit -m "feat(corpus): outcome_side backfill — orchestrator (#167)"
```

---

### Task 7: CLI subcommand wiring

**Files:**
- Modify: `src/pscanner/corpus/cli.py`
- Create: `tests/corpus/test_outcome_side_backfill_cli.py`

- [ ] **Step 1: Write the failing CLI test**

Create `tests/corpus/test_outcome_side_backfill_cli.py`:

```python
"""End-to-end CLI integration test for `pscanner corpus backfill-outcome-side`."""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus import cli as cli_mod
from pscanner.corpus.db import init_corpus_db
from pscanner.poly.models import Market


def _make_market(*, slug: str, tokens: tuple[str, str]) -> Market:
    return Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": slug,
            "outcomes": ["Cavaliers", "Knicks"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": list(tokens),
            "active": True,
            "closed": False,
        }
    )


def _seed_buggy_market(conn: sqlite3.Connection, condition_id: str) -> None:
    conn.execute(
        "INSERT INTO corpus_markets (platform, condition_id, event_slug, market_slug, "
        " category, enumerated_at, total_volume_usd, backfill_state) "
        "VALUES ('polymarket', ?, 'evt', ?, 'sports', 1, 0, 'complete')",
        (condition_id, f"slug-{condition_id}"),
    )
    for asset_id in (f"y-{condition_id}", f"n-{condition_id}"):
        conn.execute(
            "INSERT INTO asset_index (platform, asset_id, condition_id, outcome_side, outcome_index) "
            "VALUES ('polymarket', ?, ?, 'NO', 1)",
            (asset_id, condition_id),
        )
    conn.execute(
        "INSERT INTO corpus_trades (platform, tx_hash, asset_id, wallet_address, "
        " condition_id, outcome_side, bs, price, size, notional_usd, ts) "
        "VALUES ('polymarket', ?, ?, '0xW', ?, 'NO', 'BUY', 0.5, 100.0, 50.0, 1)",
        (f"0xtx-{condition_id}", f"y-{condition_id}", condition_id),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_cli_backfill_outcome_side_end_to_end(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    _seed_buggy_market(conn, "cond1")
    conn.close()

    # Stub the gamma + data context managers used by cli.py.
    fake_data = AsyncMock()
    fake_data.get_market_slug_by_condition_id = AsyncMock(return_value="slug-cond1")
    fake_data.aclose = AsyncMock()

    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(
        return_value=_make_market(slug="slug-cond1", tokens=("y-cond1", "n-cond1"))
    )
    fake_gamma.aclose = AsyncMock()

    @asynccontextmanager
    async def _gamma_cm():
        yield fake_gamma

    @asynccontextmanager
    async def _data_cm():
        yield fake_data

    monkeypatch.setattr(cli_mod, "_make_gamma_client", _gamma_cm)
    monkeypatch.setattr(cli_mod, "_make_data_client", _data_cm)

    import argparse

    args = argparse.Namespace(
        db=str(db_path),
        rpm=50,
        limit=None,
        dry_run=False,
    )
    rc = await cli_mod._cmd_backfill_outcome_side(args)
    assert rc == 0

    conn = sqlite3.connect(db_path)
    sides = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='cond1'"
        )
    }
    assert sides == {"y-cond1": "YES", "n-cond1": "NO"}
    trade_side = conn.execute(
        "SELECT outcome_side FROM corpus_trades WHERE tx_hash='0xtx-cond1'"
    ).fetchone()[0]
    assert trade_side == "YES"
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()[0]
    assert sentinel is not None
    conn.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill_cli.py -q`
Expected: FAIL with `AttributeError: module 'pscanner.corpus.cli' has no attribute '_cmd_backfill_outcome_side'`.

- [ ] **Step 3: Add the CLI subcommand**

Open `src/pscanner/corpus/cli.py`. Add this import near the other `pscanner.corpus.*` imports at the top:

```python
from pscanner.corpus.outcome_side_backfill import run_backfill as _run_outcome_side_backfill
```

Add this command function (place it adjacent to the other `_cmd_*` functions, e.g. just before `_cmd_subgraph_backfill`):

```python
async def _cmd_backfill_outcome_side(args: argparse.Namespace) -> int:
    """``corpus backfill-outcome-side`` — repair NO+NO binary markets (#167)."""
    conn = init_corpus_db(Path(args.db))
    try:
        async with AsyncExitStack() as stack:
            gamma = await stack.enter_async_context(_make_gamma_client())
            data = await stack.enter_async_context(_make_data_client())
            stats = await _run_outcome_side_backfill(
                conn,
                data=data,
                gamma=gamma,
                dry_run=bool(args.dry_run),
                limit=args.limit,
            )
        _log.info("corpus.backfill_outcome_side.cli_done", **stats)
        return 0
    finally:
        conn.close()
```

Find the argparse subparsers section (where `subgraph-backfill`, `backfill-gamma-tags`, etc. are defined). Add a new subparser:

```python
    sp_backfill_outcome = subparsers.add_parser(
        "backfill-outcome-side",
        help="Rewrite NO+NO binary markets in asset_index + corpus_trades (#167).",
    )
    sp_backfill_outcome.add_argument("--db", default="data/corpus.sqlite3")
    sp_backfill_outcome.add_argument("--rpm", type=int, default=50)
    sp_backfill_outcome.add_argument("--limit", type=int, default=None)
    sp_backfill_outcome.add_argument("--dry-run", action="store_true")
```

Note: `--rpm` is reserved for future use but does not currently propagate (the `_make_gamma_client()` factory uses a fixed rpm; this matches the existing pattern in the file). Document this in a follow-up if it becomes load-bearing.

Find the `_SUBCOMMANDS` mapping near the bottom of `cli.py`. Add an entry:

```python
    "backfill-outcome-side": _cmd_backfill_outcome_side,
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/corpus/test_outcome_side_backfill_cli.py -q`
Expected: 1 passed.

- [ ] **Step 5: Run the broader corpus test suite + lint**

Run: `uv run pytest tests/corpus/ -q && uv run ruff check src/pscanner/corpus/cli.py src/pscanner/corpus/outcome_side_backfill.py tests/corpus/test_outcome_side_backfill.py tests/corpus/test_outcome_side_backfill_cli.py && uv run ty check src/pscanner/corpus/cli.py src/pscanner/corpus/outcome_side_backfill.py`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_outcome_side_backfill_cli.py
git commit -m "feat(corpus): wire backfill-outcome-side CLI subcommand (#167)"
```

---

### Task 8: CLAUDE.md operator notes

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a bullet to the `## CLI surface` section**

Open `CLAUDE.md`. Find the `## CLI surface` section and add a bullet near the other `pscanner corpus *` subcommands (e.g. after `pscanner corpus subgraph-backfill ...`):

```markdown
- `pscanner corpus backfill-outcome-side [--db PATH] [--rpm N] [--limit N] [--dry-run]` — one-shot rewrite of the 1,697 binary markets whose `asset_index` + `corpus_trades` rows were stored as NO+NO due to the pre-#166 `_parse_trade` bug (#167). Resumable via the new `corpus_markets.outcome_side_backfilled_at` sentinel column. Per-market transaction; one gamma call per market at default `--rpm 50` (~34 min wall on the production 1,697-market queue). Run `pscanner corpus build-features --rebuild --engine duckdb` after a successful backfill to regenerate `training_examples.label_won` from the corrected `corpus_trades.outcome_side`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): document backfill-outcome-side CLI (#167)"
```

---

### Task 9: Desktop dry-run + live execution

This is an operator step — the user runs the commands on the desktop, not the implementer.

- [ ] **Step 1: Rsync code to the desktop**

```bash
rsync -avh -e "ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  /home/macph/projects/polymarketScanner/src/pscanner/corpus/ \
  macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/src/pscanner/corpus/
rsync -avh -e "ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  /home/macph/projects/polymarketScanner/tests/corpus/ \
  macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/tests/corpus/
```

- [ ] **Step 2: Run the test suite on the desktop**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run pytest tests/corpus/test_outcome_side_backfill.py tests/corpus/test_outcome_side_backfill_cli.py tests/corpus/test_outcome_side_backfilled_column.py -q'
```
Expected: all green (the schema migration ALTER TABLE triggers the first time the desktop's existing corpus DB opens with this code).

- [ ] **Step 3: Run a dry-run on the full production corpus**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   set -a && source .env && set +a && \
   uv run pscanner corpus backfill-outcome-side --dry-run --limit 5'
```
Expected: 5 markets resolved, 5 dry_run_resolve log events, no writes.

- [ ] **Step 4: Run the real backfill in background**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   set -a && source .env && set +a && \
   nohup uv run pscanner corpus backfill-outcome-side > /tmp/backfill_outcome_side.log 2>&1 < /dev/null & \
   echo "started pid=$!" && disown'
```

- [ ] **Step 5: Monitor progress**

```bash
ssh -p 2222 macph@10.0.0.143 'tail -f /tmp/backfill_outcome_side.log | grep "corpus.backfill_outcome_side"'
```

Wait for `corpus.backfill_outcome_side.done` with `remaining=0`.

- [ ] **Step 6: Verify final state**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run python -c "
import sqlite3
c = sqlite3.connect(\"data/corpus.sqlite3\")
remaining = c.execute(\"SELECT COUNT(*) FROM (SELECT condition_id FROM asset_index WHERE platform=\\\"polymarket\\\" GROUP BY condition_id HAVING COUNT(*)=2 AND COUNT(DISTINCT outcome_side)=1 AND MIN(outcome_side)=\\\"NO\\\")\").fetchone()[0]
print(f\"remaining NO+NO markets: {remaining}\")
"'
```
Expected: `remaining NO+NO markets: 0`.

- [ ] **Step 7: Run build-features rebuild**

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   nohup uv run pscanner corpus build-features --rebuild --engine duckdb \
     > /tmp/build_features_post_167.log 2>&1 < /dev/null & \
   echo "started pid=$!" && disown'
```

Wait ~1 hour (per the desktop's DuckDB engine timing in CLAUDE.md). Monitor via `tail -f /tmp/build_features_post_167.log`.

- [ ] **Step 8: Spot-check a fixed market**

Pick a known-bad market from the issue body (e.g. `nba-mil-ind-2025-04-22`):

```bash
ssh -p 2222 macph@10.0.0.143 \
  'export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" && \
   cd ~/projects/polymarketscanner/pscanner && \
   uv run python <<PY
import sqlite3
c = sqlite3.connect("data/corpus.sqlite3")
c.row_factory = sqlite3.Row
cid = c.execute("SELECT condition_id FROM corpus_markets WHERE market_slug=?", ("nba-mil-ind-2025-04-22",)).fetchone()[0]
print("condition_id:", cid)
for r in c.execute("SELECT asset_id, outcome_side, outcome_index FROM asset_index WHERE condition_id=?", (cid,)):
    print(" leg:", dict(r))
PY'
```
Expected: two rows with `outcome_side=YES idx=0` and `outcome_side=NO idx=1` respectively (not both NO).

---

### Task 10: File the retrain follow-up issue

- [ ] **Step 1: Open the follow-up issue**

```bash
gh issue create --title "Retrain gate model after #167 outcome_side backfill" --body "$(cat <<'EOF'
## Context

PR for #167 landed the backfill of `corpus_trades.outcome_side` + `asset_index` for the 1,697 NO+NO binary markets (mostly sports + esports). After that PR, `pscanner corpus build-features --rebuild --engine duckdb` regenerated `training_examples` with the corrected supervised-learning labels.

## Goal

Retrain the gate model on the corrected dataset and compare per-category metrics against the 2026-05-15 baseline (`models/2026-05-15-copy_trade_gate/metrics.json`).

## Steps

1. ``pscanner ml train --device cuda --n-jobs 1 --n-trials 100`` (~1h 18m on the desktop's RTX 3070 per CLAUDE.md baseline).
2. Compare ``metrics.json``:
   - Overall ``test_edge``
   - ``test_edge_filtered`` (sports + esports cohort)
   - Per-category breakdown (``per_category_any`` if present)
3. If the new edge meaningfully outperforms baseline, ship a new model artifact under ``models/YYYY-MM-DD-copy_trade_gate/`` and bump ``[gate_model] artifact_dir`` in config.

## Expected outcome

The 2026-05-15 baseline reports sports +6.9% / esports +10.7% edge, but those numbers were trained against partially-inverted labels (per #167's empirical scope analysis). The corrected retrain could shift those numbers in either direction:

- If the original edge was inflated by label inversion: lower numbers in the retrain.
- If the inversion masked real signal: higher numbers in the retrain.

Either result is informative.

## Related

- #167 — the backfill
- #159 / PR #166 — the forward fix
- 2026-05-15 baseline in `CLAUDE.md`'s ML training pipeline section
EOF
)"
```

---

## Self-Review

**1. Spec coverage:**

- Spec ``Architecture overview`` → Tasks 2-6 (discovery, resolver, applier, validator, orchestrator) ✓
- Spec ``CLI surface`` → Task 7 ✓
- Spec ``Schema additions`` → Task 1 ✓
- Spec ``Per-market transaction shape`` → Task 4 ✓
- Spec ``Error handling per market`` → Task 6 (skip-and-continue paths in the orchestrator, covered by `test_run_backfill_skips_unresolvable_markets`) ✓
- Spec ``Validation`` → Task 5 ✓
- Spec ``Phase 4 — build-features rebuild`` → Task 9, step 7 ✓
- Spec ``Phase 5 — separate issue for retrain`` → Task 10 ✓
- Spec ``Test surface`` items 1-8 → Task 2-6 unit tests ✓; item 9 → Task 7 CLI integration test ✓
- Spec ``Risks + mitigations`` — addressed where relevant in task code

**2. Placeholder scan:** No TBDs, no TODOs, no "implement later", no "similar to Task N". Every step has the actual code or command.

**3. Type / name consistency:**
- `find_buggy_markets(conn)` → tested + used in Tasks 2, 5, 6 ✓
- `resolve_correct_mapping(condition_id, *, data, gamma)` → tested + used in Task 6 ✓
- `apply_market_backfill(conn, condition_id, mapping, *, now_ts)` → tested + used in Task 6 ✓
- `validate_backfill_state(conn)` → tested + used in Task 6 ✓
- `run_backfill(conn, *, data, gamma, dry_run, limit)` → tested + used in Task 7 ✓
- `_cmd_backfill_outcome_side(args)` → defined in Task 7, registered in `_SUBCOMMANDS` ✓
- Mapping type `dict[str, tuple[str, int]]` consistent across Tasks 3, 4, 6
- Stats dict keys `processed/resolved/skipped/remaining` consistent in Task 6 tests + orchestrator

**4. Scope check:** Single PR scope. Tasks 9-10 are operator steps that don't ship code; the implementer subagent stops at Task 8.
