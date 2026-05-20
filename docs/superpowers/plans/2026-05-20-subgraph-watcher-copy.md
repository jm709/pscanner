# Subgraph watch + copy script — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `scripts/watch_subgraph_copy.py` — a standalone research script that watches Polymarket's current real-time subgraph for trades by any wallet in the daemon's `WatchlistRegistry` and books each qualifying trade into `paper_trades` under `triggering_alert_detector='subgraph_copy'`.

**Architecture:** Single Python script under `scripts/`. Reuses existing `SubgraphClient`, `WatchlistRepo`, `MarketCacheRepo`, `AssetIndexRepo`, `MarketTicksRepo`, `PaperTradesRepo`, `DataClient`, `GammaClient`. Poll loop with watermark pagination (timestamp_gte + within-cycle tx_hash dedupe). Opens two SQLite connections (daemon DB + corpus DB read-only).

**Tech Stack:** Python 3.13, `asyncio`, `httpx` (via existing `SubgraphClient`), `sqlite3`, `structlog`, `argparse`. Tested with `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-20-subgraph-watcher-copy-design.md`

---

## File structure

**New files:**
- `scripts/watch_subgraph_copy.py` — the script (~280 lines)
- `tests/scripts/__init__.py` — empty package marker
- `tests/scripts/test_watch_subgraph_copy.py` — pure-logic unit tests

**Modified files:**
- `src/pscanner/strategies/paper_trader.py` — extract `_lookup_fill_price` to a module-level helper so the script can reuse it without instantiating `PaperTrader`. Keep the existing `PaperTrader` method as a thin wrapper around the helper for back-compat with current tests.

**No schema migrations.** All writes land in the existing `paper_trades` table; the `triggering_alert_detector='subgraph_copy'` discriminator is enough to separate them.

---

## Task 1: Extract `_lookup_fill_price` from `PaperTrader` to a module-level helper

**Why first:** This refactor unblocks Task 7 (fill-price lookup in the script) without code duplication. Doing the refactor first while we have a passing test suite is safer than at the end.

**Files:**
- Modify: `src/pscanner/strategies/paper_trader.py:336-378` (the `_lookup_fill_price` method body)
- Test: existing `tests/strategies/test_paper_trader.py` — must continue to pass

- [ ] **Step 1: Confirm the existing test suite is green**

Run: `uv run pytest tests/strategies/test_paper_trader.py -q`
Expected: PASS (current baseline). If anything fails, stop and investigate before touching the code.

- [ ] **Step 2: Add a module-level helper above the `PaperTrader` class**

Open `src/pscanner/strategies/paper_trader.py`. Above the `class PaperTrader:` line, add:

```python
def lookup_fill_price(
    market_cache: MarketCacheRepo,
    market_ticks: MarketTicksRepo,
    condition_id: ConditionId,
    asset_id: AssetId,
) -> float | None:
    """Resolve a fill price via the prioritised lookup chain.

    Order:

    1. ``market_ticks.best_ask`` (live orderbook ask).
    2. ``market_ticks.last_trade_price`` (last printed trade).
    3. ``market_cache.outcome_prices[outcome_index]`` (gamma's cached
       last-known quote — populated at backfill time, seconds-stale
       but always available immediately after a cache miss recovery).

    Returns ``None`` when none of the three sources yields a price in
    ``(0, 1)``. The fallback path emits ``paper_trade.fill_price_fallback``
    at INFO level so operators can grep how often live ticks were
    unavailable.
    """
    tick = market_ticks.latest_for_asset(asset_id)
    if tick is not None:
        if _is_valid_price(tick.best_ask):
            return float(tick.best_ask)
        if _is_valid_price(tick.last_trade_price):
            return float(tick.last_trade_price)
    fallback_price = _cached_outcome_price(market_cache, condition_id, asset_id)
    if fallback_price is not None:
        _LOG.info(
            "paper_trade.fill_price_fallback",
            asset_id=asset_id,
            condition_id=condition_id,
            fallback_price=fallback_price,
        )
        return fallback_price
    _LOG.warning(
        "paper_trade.no_price",
        asset_id=asset_id,
        condition_id=condition_id,
        best_ask=tick.best_ask if tick is not None else None,
        last_trade=tick.last_trade_price if tick is not None else None,
    )
    return None


def _cached_outcome_price(
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
    asset_id: AssetId,
) -> float | None:
    """Return the cached gamma outcome price for ``asset_id``, or None."""
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None:
        return None
    if len(cached.asset_ids) != len(cached.outcome_prices):
        return None
    try:
        idx = cached.asset_ids.index(asset_id)
    except ValueError:
        return None
    price = cached.outcome_prices[idx]
    if not _is_valid_price(price):
        return None
    return float(price)
```

- [ ] **Step 3: Replace `PaperTrader._lookup_fill_price` body to delegate to the helper**

In the `PaperTrader` class, replace the `_lookup_fill_price` method (lines ~336-378) and the `_cached_outcome_price` method (lines ~380-414) with thin wrappers:

```python
    def _lookup_fill_price(
        self,
        condition_id: ConditionId,
        asset_id: AssetId,
    ) -> float | None:
        """Resolve a fill price via the prioritised lookup chain."""
        return lookup_fill_price(
            self._market_cache,
            self._market_ticks,
            condition_id,
            asset_id,
        )

    def _cached_outcome_price(
        self,
        condition_id: ConditionId,
        asset_id: AssetId,
    ) -> float | None:
        """Delegate to the module-level helper."""
        return _cached_outcome_price(self._market_cache, condition_id, asset_id)
```

- [ ] **Step 4: Re-run the paper-trader test suite to confirm no regression**

Run: `uv run pytest tests/strategies/test_paper_trader.py -q`
Expected: PASS — same set of tests pass with the refactored implementation.

- [ ] **Step 5: Run full lint + type check**

Run: `uv run ruff check src/pscanner/strategies/paper_trader.py && uv run ty check src/pscanner/strategies/paper_trader.py`
Expected: clean output.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/strategies/paper_trader.py
git commit -m "refactor(paper-trader): extract lookup_fill_price to module-level helper

Unblocks reuse from scripts/watch_subgraph_copy.py without instantiating
PaperTrader. PaperTrader._lookup_fill_price becomes a thin wrapper; all
existing tests pass unchanged.

Refs #152, prep for the subgraph-watcher script."
```

---

## Task 2: Scaffold the script with argparse + structlog setup

**Files:**
- Create: `scripts/watch_subgraph_copy.py`

- [ ] **Step 1: Create the script file with header, imports, and constants**

Write the file:

```python
r"""Watch the Polymarket subgraph for trades by watchlisted wallets and book paper copies.

Standalone research script — see
``docs/superpowers/specs/2026-05-20-subgraph-watcher-copy-design.md`` for the
design and ``docs/superpowers/plans/2026-05-20-subgraph-watcher-copy.md``
for the implementation plan.

Reads the watchlist from the daemon DB's ``WatchlistRepo``. Queries the
current Polymarket Orderbook subgraph (id ``B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR``)
for ``orderFilledEvents`` since the last checkpoint with server-side
``maker_in`` / ``taker_in`` filter. For each watchlist hit whose direction
is a position-increase (BUY-on-maker or SELL-on-taker), looks up an
outcome name + fill price and books an entry row into ``paper_trades``
under ``triggering_alert_detector='subgraph_copy'``.

Coexists with the daemon's smart_money paper trader — distinct detector
tag plus distinct ``triggering_alert_key`` prefix keep both sets parallel
in the ledger.

Usage::

    uv run python scripts/watch_subgraph_copy.py --once --since-hours 1
"""

# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Final

import structlog

from pscanner.config import load_config
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.subgraph import SubgraphClient

_LOG = structlog.get_logger(__name__)

SUBGRAPH_ID: Final[str] = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
DETECTOR_TAG: Final[str] = "subgraph_copy"
ALERT_KEY_PREFIX: Final[str] = "subgraph"
DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 10.0
DEFAULT_RPM: Final[int] = 60
PAGE_SIZE: Final[int] = 1000
DEFAULT_CHECKPOINT_PATH: Final[Path] = Path("data/subgraph_watch_state.json")
INDEXER_LAG_WARN_SECONDS: Final[int] = 60
INDEXER_LAG_ERROR_SECONDS: Final[int] = 600


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=str, default="data/pscanner.sqlite3",
                        help="Daemon SQLite path (default: data/pscanner.sqlite3)")
    parser.add_argument("--corpus-db", type=str, default="data/corpus.sqlite3",
                        help="Corpus SQLite path for AssetIndexRepo (default: data/corpus.sqlite3)")
    parser.add_argument("--subgraph-id", type=str, default=SUBGRAPH_ID,
                        help=f"Subgraph ID (default: {SUBGRAPH_ID})")
    parser.add_argument("--poll-interval-seconds", type=float,
                        default=DEFAULT_POLL_INTERVAL_SECONDS,
                        help=f"Seconds between poll cycles (default: {DEFAULT_POLL_INTERVAL_SECONDS})")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                        help=f"Subgraph queries per minute (default: {DEFAULT_RPM})")
    parser.add_argument("--since-hours", type=float, default=None,
                        help="Optional cold-start backfill window in hours; "
                             "overrides the checkpoint if set.")
    parser.add_argument("--once", action="store_true",
                        help="Single poll pass then exit (for testing).")
    parser.add_argument("--position-fraction-override", type=float, default=None,
                        help="Override paper-trader position_fraction (default: from config).")
    parser.add_argument("--bankroll-override", type=float, default=None,
                        help="Override paper-trader starting_bankroll_usd (default: from config).")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH),
                        help=f"Checkpoint JSON path (default: {DEFAULT_CHECKPOINT_PATH})")
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()
    _LOG.info(
        "subgraph_watch.startup",
        db=args.db,
        corpus_db=args.corpus_db,
        subgraph_id=args.subgraph_id,
        poll_interval_seconds=args.poll_interval_seconds,
        rpm=args.rpm,
        since_hours=args.since_hours,
        once=args.once,
    )
    # Full implementation lands in Tasks 4-8.
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 2: Verify it parses and the help works**

Run: `uv run python scripts/watch_subgraph_copy.py --help`
Expected: argparse help output shows all flags with defaults; exit code 0.

- [ ] **Step 3: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/watch_subgraph_copy.py
git commit -m "feat(scripts): scaffold watch_subgraph_copy.py

CLI argparse + structlog logger + constants pinned for the new subgraph
ID (B9mm...). Main loop body lands in subsequent commits.

Refs #152."
```

---

## Task 3: TDD — `_compute_copy_direction` helper

**Files:**
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/test_watch_subgraph_copy.py`
- Modify: `scripts/watch_subgraph_copy.py` (add `_compute_copy_direction`)

- [ ] **Step 1: Create the test package**

```bash
mkdir -p tests/scripts
touch tests/scripts/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/scripts/test_watch_subgraph_copy.py`:

```python
"""Unit tests for scripts/watch_subgraph_copy.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "watch_subgraph_copy.py"
_spec = importlib.util.spec_from_file_location("watch_subgraph_copy", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
watch_subgraph_copy = importlib.util.module_from_spec(_spec)
sys.modules["watch_subgraph_copy"] = watch_subgraph_copy
_spec.loader.exec_module(watch_subgraph_copy)


# A wallet on our watchlist, two non-watchlisted counterparties for clarity.
_WATCH = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_OTHER1 = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_OTHER2 = "0xcccccccccccccccccccccccccccccccccccccccc"


@pytest.mark.parametrize(
    ("maker", "taker", "side", "expected"),
    [
        # maker=watchlist, side=0 (BUY)  -> maker accumulates -> COPY BUY
        (_WATCH, _OTHER1, 0, "BUY"),
        # maker=watchlist, side=1 (SELL) -> maker reduces -> SKIP
        (_WATCH, _OTHER1, 1, "SKIP"),
        # taker=watchlist, side=0 (taker hit a buy order -> sold) -> taker reduces -> SKIP
        (_OTHER1, _WATCH, 0, "SKIP"),
        # taker=watchlist, side=1 (taker hit a sell order -> bought) -> taker accumulates -> BUY
        (_OTHER1, _WATCH, 1, "BUY"),
        # Neither on the watchlist (shouldn't happen at the call site, but defensive)
        (_OTHER1, _OTHER2, 0, "SKIP"),
        (_OTHER1, _OTHER2, 1, "SKIP"),
    ],
)
def test_compute_copy_direction(maker: str, taker: str, side: int, expected: str) -> None:
    watchlist = {_WATCH}
    result = watch_subgraph_copy._compute_copy_direction(maker, taker, side, watchlist)
    assert result == expected
```

- [ ] **Step 3: Run test, expect FAIL**

Run: `uv run pytest tests/scripts/test_watch_subgraph_copy.py -q`
Expected: FAIL with `AttributeError: module 'watch_subgraph_copy' has no attribute '_compute_copy_direction'`.

- [ ] **Step 4: Implement `_compute_copy_direction` in the script**

Add to `scripts/watch_subgraph_copy.py` (above `_parse_args`):

```python
def _compute_copy_direction(
    maker: str,
    taker: str,
    side: int,
    watchlist: set[str],
) -> str:
    """Return ``"BUY"`` iff the watchlist wallet's position in ``tokenId`` increases.

    The subgraph's ``side`` field is the order's direction (0=BUY, 1=SELL).
    Maker placed the resting order; taker hit it from the opposite side.
    So:

    - watchlist == maker AND side == 0 -> maker accumulates -> BUY
    - watchlist == maker AND side == 1 -> maker reduces -> SKIP
    - watchlist == taker AND side == 0 -> taker sold (hit a buy order) -> SKIP
    - watchlist == taker AND side == 1 -> taker bought (hit a sell order) -> BUY

    See the copy-direction table in the design spec for the full derivation.
    """
    maker_lower = maker.lower()
    taker_lower = taker.lower()
    if maker_lower in watchlist and side == 0:
        return "BUY"
    if taker_lower in watchlist and side == 1:
        return "BUY"
    return "SKIP"
```

Make sure the address comparisons are case-insensitive — the subgraph emits lower-case hex but watchlist entries may be checksum-cased. Normalize via `.lower()` at compare time.

- [ ] **Step 5: Run tests again, expect PASS**

Run: `uv run pytest tests/scripts/test_watch_subgraph_copy.py -q`
Expected: PASS (6 cases).

- [ ] **Step 6: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py tests/scripts/test_watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/watch_subgraph_copy.py tests/scripts/__init__.py tests/scripts/test_watch_subgraph_copy.py
git commit -m "feat(scripts): _compute_copy_direction with table-driven tests

Returns BUY iff the watchlist wallet's resulting position increases —
maker+BUY or taker+SELL. Mirrors the gate-model loop's BUY-only score
semantics for parity in paper-trade comparisons.

Refs #152."
```

---

## Task 4: TDD — `_build_where_clause` for the GraphQL filter

**Files:**
- Modify: `scripts/watch_subgraph_copy.py`
- Modify: `tests/scripts/test_watch_subgraph_copy.py`

- [ ] **Step 1: Append the test**

Add to `tests/scripts/test_watch_subgraph_copy.py`:

```python
def test_build_where_clause_emits_or_with_per_branch_timestamp() -> None:
    addrs = ["0xaaa", "0xbbb"]
    last_seen_ts = 1779225600
    where = watch_subgraph_copy._build_where_clause(addrs, last_seen_ts)

    # Top-level must NOT have timestamp_gte alongside `or` — TheGraph rejects that.
    assert "timestamp_gte" not in where
    assert "or" in where
    branches = where["or"]
    assert len(branches) == 2
    # Each branch carries the timestamp filter and one of maker/taker filters.
    maker_branches = [b for b in branches if "maker_in" in b]
    taker_branches = [b for b in branches if "taker_in" in b]
    assert len(maker_branches) == 1
    assert len(taker_branches) == 1
    assert maker_branches[0]["maker_in"] == addrs
    assert maker_branches[0]["timestamp_gte"] == str(last_seen_ts)
    assert taker_branches[0]["taker_in"] == addrs
    assert taker_branches[0]["timestamp_gte"] == str(last_seen_ts)
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/scripts/test_watch_subgraph_copy.py::test_build_where_clause_emits_or_with_per_branch_timestamp -q`
Expected: FAIL with attribute-error.

- [ ] **Step 3: Implement `_build_where_clause`**

Add to `scripts/watch_subgraph_copy.py`:

```python
def _build_where_clause(addrs: list[str], last_seen_ts: int) -> dict[str, Any]:
    """Build the ``where:`` argument for ``orderFilledEvents``.

    TheGraph rejects ``or`` mixed with same-level column filters, so the
    timestamp predicate must be repeated inside each ``or`` branch.
    ``timestamp_gte`` (not ``_gt``) plus a within-cycle tx_hash dedupe
    in the pagination loop gives strict no-loss boundary behaviour.

    Returns a dict ready to pass to :class:`SubgraphClient.query`.
    """
    ts_str = str(last_seen_ts)
    return {
        "or": [
            {"timestamp_gte": ts_str, "maker_in": addrs},
            {"timestamp_gte": ts_str, "taker_in": addrs},
        ],
    }
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/scripts/test_watch_subgraph_copy.py -q`
Expected: PASS (7 cases total — 6 from Task 3 + this one).

- [ ] **Step 5: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py tests/scripts/test_watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/watch_subgraph_copy.py tests/scripts/test_watch_subgraph_copy.py
git commit -m "feat(scripts): _build_where_clause for subgraph filter

Per-branch timestamp_gte avoids TheGraph's 'or mixed with column filter'
error. timestamp_gte + tx_hash dedupe gives strict no-loss page boundaries.

Refs #152."
```

---

## Task 5: TDD — `_fetch_events_since` pagination loop

**Files:**
- Modify: `scripts/watch_subgraph_copy.py`
- Modify: `tests/scripts/test_watch_subgraph_copy.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_watch_subgraph_copy.py`:

```python
class _FakeSubgraphClient:
    """Fake SubgraphClient that returns scripted pages of events."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = list(pages)  # consume FIFO
        self.calls: list[dict[str, Any]] = []  # recorded variables per call

    async def query(self, graphql: str, variables: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(variables)
        if not self.pages:
            return {"orderFilledEvents": [], "_meta": {"block": {"number": 1, "timestamp": "0"}}}
        page = self.pages.pop(0)
        return {
            "orderFilledEvents": page,
            "_meta": {"block": {"number": 87000000, "timestamp": "1779230000"}},
        }


def _ev(ts: int, tx: str) -> dict[str, Any]:
    return {
        "transactionHash": tx,
        "timestamp": str(ts),
        "maker": {"id": "0xmaker"},
        "taker": {"id": "0xtaker"},
        "market": {"id": "1"},
        "tokenId": "1",
        "side": 0,
        "price": "0.5",
        "size": "10",
    }


@pytest.mark.asyncio
async def test_fetch_events_since_paginates_until_partial_page() -> None:
    from typing import Any as _Any  # noqa: F401, used in _FakeSubgraphClient sig

    PAGE_SIZE = watch_subgraph_copy.PAGE_SIZE  # 1000
    # Page 1: full PAGE_SIZE events with ts 100..1099 (last tx "tx0999")
    page1 = [_ev(100 + i, f"tx{i:04d}") for i in range(PAGE_SIZE)]
    # Page 2: full PAGE_SIZE events with ts 1100..2099. Include one tx that
    # also appeared in page1's last row (boundary re-fetch) to exercise dedupe.
    page2 = [_ev(1099, "tx0999")] + [_ev(1100 + i, f"tx{1000 + i:04d}") for i in range(PAGE_SIZE - 1)]
    # Page 3: partial (< PAGE_SIZE) — terminator
    page3 = [_ev(2200, "tx2200"), _ev(2300, "tx2300")]
    fake = _FakeSubgraphClient([page1, page2, page3])

    events, indexer_ts = await watch_subgraph_copy._fetch_events_since(
        fake, addrs=["0xaaa"], last_seen_ts=99,
    )

    # 3 pages queried.
    assert len(fake.calls) == 3
    # Boundary tx not double-counted — total unique events = 1000 + 999 + 2 = 2001.
    assert len(events) == 2001
    # Indexer timestamp came through from _meta.
    assert indexer_ts == 1779230000
```

- [ ] **Step 2: Ensure pytest-asyncio is available**

Run: `uv run pytest --collect-only tests/scripts/test_watch_subgraph_copy.py -q 2>&1 | tail -3`

If pytest reports an "asyncio" fixture missing error, check `pyproject.toml`:

```bash
grep -E "pytest-asyncio|asyncio_mode" pyproject.toml
```

If `asyncio_mode = "auto"` is set, the `@pytest.mark.asyncio` decorator is redundant but harmless. If not, the decorator is required.

- [ ] **Step 3: Run test, expect FAIL**

Run: `uv run pytest tests/scripts/test_watch_subgraph_copy.py::test_fetch_events_since_paginates_until_partial_page -q`
Expected: FAIL with `AttributeError: module 'watch_subgraph_copy' has no attribute '_fetch_events_since'`.

- [ ] **Step 4: Implement `_fetch_events_since`**

Add to `scripts/watch_subgraph_copy.py`:

```python
_GRAPHQL_QUERY: Final[str] = """
{
  orderFilledEvents(
    where: $where
    first: %d
    orderBy: timestamp
    orderDirection: asc
  ) {
    transactionHash
    timestamp
    maker { id }
    taker { id }
    market { id }
    tokenId
    side
    price
    size
  }
  _meta { block { number timestamp } }
}
""" % PAGE_SIZE


async def _fetch_events_since(
    client: Any,
    *,
    addrs: list[str],
    last_seen_ts: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Drain the subgraph for all events newer than ``last_seen_ts``.

    Watermark pagination: each page advances ``ts`` to the most recent
    event seen. Loop terminates when a page returns fewer than
    ``PAGE_SIZE`` events. Within-cycle tx_hash dedupe catches boundary
    events re-fetched by ``timestamp_gte``.

    Returns the list of unique events (asc ts ordering) and the
    indexer's ``_meta.block.timestamp`` from the last page (used by
    the caller for indexing-lag detection).
    """
    events: list[dict[str, Any]] = []
    seen_tx: set[str] = set()
    ts = last_seen_ts
    indexer_ts: int | None = None
    while True:
        where = _build_where_clause(addrs, ts)
        # SubgraphClient.query takes (graphql, variables) — we inline
        # `where` into the variables payload here.
        # NOTE: GraphQL spec actually requires variable declarations on the
        # operation. The Graph tolerates inline-substituted bodies; we use
        # that simpler form because we don't need variable-binding here.
        # If The Graph tightens this, switch to a parameterised query.
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


def _serialize_where_inline(where: dict[str, Any]) -> str:
    """Render ``where:`` as a GraphQL object literal (not JSON).

    GraphQL object literals don't quote keys. We hand-emit a minimal
    serializer instead of pulling in a full GraphQL client.
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
```

- [ ] **Step 5: Run test, expect PASS**

Run: `uv run pytest tests/scripts/test_watch_subgraph_copy.py -q`
Expected: PASS (8 cases total — 6 from Task 3, 1 from Task 4, 1 here).

- [ ] **Step 6: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py tests/scripts/test_watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/watch_subgraph_copy.py tests/scripts/test_watch_subgraph_copy.py
git commit -m "feat(scripts): _fetch_events_since with watermark pagination

Single GraphQL query with the full watchlist (verified 564 addrs in 51KB
body works fine). Within-cycle tx_hash dedupe + timestamp_gte boundary
gives strict no-loss guarantee. Steady-state: single page per cycle.

Refs #152."
```

---

## Task 6: Checkpoint persistence

**Files:**
- Modify: `scripts/watch_subgraph_copy.py`

No new tests — file I/O wrapper is too thin to bother. The end-to-end smoke (Task 10) covers it.

- [ ] **Step 1: Add checkpoint helpers**

Add to `scripts/watch_subgraph_copy.py`:

```python
def _load_checkpoint(path: Path, since_hours_override: float | None) -> int:
    """Return the timestamp to resume from.

    ``--since-hours`` always wins. Otherwise read the checkpoint file;
    if it's missing or corrupt, default to ``now()``.
    """
    if since_hours_override is not None:
        return int(time.time() - 3600.0 * since_hours_override)
    if not path.exists():
        _LOG.info("subgraph_watch.checkpoint_missing", path=str(path))
        return int(time.time())
    try:
        payload = json.loads(path.read_text())
        return int(payload["last_seen_ts"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _LOG.warning("subgraph_watch.checkpoint_corrupt", path=str(path), exc=str(exc))
        return int(time.time())


def _save_checkpoint(path: Path, last_seen_ts: int) -> None:
    """Atomically write the checkpoint via tmp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"last_seen_ts": int(last_seen_ts)}))
    tmp.replace(path)
```

- [ ] **Step 2: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add scripts/watch_subgraph_copy.py
git commit -m "feat(scripts): checkpoint load + atomic save for subgraph watcher

Resumes from data/subgraph_watch_state.json across restarts. Atomic
tmp+rename write so a crash mid-write can't corrupt the file. Missing
or corrupt checkpoint resets to now() — research-mode safe.

Refs #152."
```

---

## Task 7: Token resolution + outcome lookup

**Files:**
- Modify: `scripts/watch_subgraph_copy.py`

The spec puts this resolution behind `AssetIndexRepo.get(tokenId)` (corpus DB) followed by `MarketCacheRepo.get_by_condition_id(condition_id)` (daemon DB). Skip-on-miss; if skip rate is high, operator runs `scripts/backfill_asset_index.py` separately.

- [ ] **Step 1: Add resolution helper**

Add to `scripts/watch_subgraph_copy.py`:

```python
from dataclasses import dataclass

from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.repo import CachedMarket, MarketCacheRepo


@dataclass(frozen=True, slots=True)
class _ResolvedToken:
    condition_id: ConditionId
    asset_id: AssetId
    outcome_name: str
    outcome_index: int


def _resolve_token(
    token_id: str,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
) -> _ResolvedToken | None:
    """Resolve a subgraph ``tokenId`` to ``(condition_id, outcome_name, ...)``.

    Returns ``None`` when neither the corpus ``asset_index`` table nor
    the daemon's ``market_cache`` has the asset registered. Caller logs
    ``subgraph_watch.tokenid_unresolved`` and skips the event.
    """
    entry = asset_index.get(token_id)
    if entry is None:
        _LOG.warning("subgraph_watch.tokenid_unresolved_asset_index", token_id=token_id)
        return None
    condition_id = ConditionId(entry.condition_id)
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None:
        _LOG.warning(
            "subgraph_watch.tokenid_unresolved_market_cache",
            token_id=token_id,
            condition_id=condition_id,
        )
        return None
    # Find the asset_id's position in the cached market's parallel
    # outcomes / asset_ids lists.
    asset_id = AssetId(token_id)
    try:
        idx = cached.asset_ids.index(asset_id)
    except ValueError:
        _LOG.warning(
            "subgraph_watch.tokenid_not_in_cache",
            token_id=token_id,
            condition_id=condition_id,
        )
        return None
    return _ResolvedToken(
        condition_id=condition_id,
        asset_id=asset_id,
        outcome_name=cached.outcomes[idx],
        outcome_index=idx,
    )
```

- [ ] **Step 2: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add scripts/watch_subgraph_copy.py
git commit -m "feat(scripts): _resolve_token via asset_index + market_cache

Two-step lookup: corpus asset_index for tokenId -> condition_id, daemon
market_cache for condition_id -> outcome_name. Skip-on-miss with distinct
log keys so operators can grep which step failed.

Refs #152."
```

---

## Task 8: Wire the main poll loop

**Files:**
- Modify: `scripts/watch_subgraph_copy.py`

- [ ] **Step 1: Replace the stub `main()` body**

Replace the current `async def main()` body in `scripts/watch_subgraph_copy.py` with:

```python
async def main() -> int:
    args = _parse_args()
    config = load_config()

    bankroll = args.bankroll_override or config.paper_trading.starting_bankroll_usd
    position_fraction = (
        args.position_fraction_override
        or config.paper_trading.evaluators.gate_model.position_fraction
    )
    min_position_cost = config.paper_trading.min_position_cost_usd

    _LOG.info(
        "subgraph_watch.startup",
        db=args.db,
        corpus_db=args.corpus_db,
        subgraph_id=args.subgraph_id,
        poll_interval_seconds=args.poll_interval_seconds,
        rpm=args.rpm,
        since_hours=args.since_hours,
        once=args.once,
        bankroll=bankroll,
        position_fraction=position_fraction,
        min_position_cost=min_position_cost,
    )

    api_key = os.environ.get("GRAPH_API_KEY")
    if not api_key:
        _LOG.error("subgraph_watch.missing_graph_api_key")
        return 2
    subgraph_url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{args.subgraph_id}"

    # Daemon DB — read/write for paper_trades, read for watchlist/market_cache/ticks.
    daemon_conn = sqlite3.connect(args.db)
    daemon_conn.row_factory = sqlite3.Row
    daemon_conn.execute("PRAGMA busy_timeout=5000")

    # Corpus DB — read-only for asset_index.
    corpus_uri = f"file:{args.corpus_db}?mode=ro"
    corpus_conn = sqlite3.connect(corpus_uri, uri=True)
    corpus_conn.row_factory = sqlite3.Row
    corpus_conn.execute("PRAGMA busy_timeout=5000")

    from pscanner.store.repo import (  # imports local to keep import-time light
        MarketTicksRepo,
        PaperTradesRepo,
        WatchlistRepo,
    )

    watchlist_repo = WatchlistRepo(daemon_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    market_ticks = MarketTicksRepo(daemon_conn)
    paper_trades = PaperTradesRepo(daemon_conn)
    asset_index = AssetIndexRepo(corpus_conn)

    subgraph_client = SubgraphClient(url=subgraph_url, rpm=args.rpm)
    data_client = DataClient(rpm=50)
    gamma_client = GammaClient(rpm=50)

    checkpoint_path = Path(args.checkpoint)
    last_seen_ts = _load_checkpoint(checkpoint_path, args.since_hours)
    _LOG.info("subgraph_watch.checkpoint_loaded", last_seen_ts=last_seen_ts)

    try:
        while True:
            cycle_start = time.time()
            stats = await _run_one_cycle(
                subgraph_client=subgraph_client,
                watchlist_repo=watchlist_repo,
                asset_index=asset_index,
                market_cache=market_cache,
                market_ticks=market_ticks,
                paper_trades=paper_trades,
                last_seen_ts=last_seen_ts,
                bankroll=bankroll,
                position_fraction=position_fraction,
                min_position_cost=min_position_cost,
            )
            last_seen_ts = stats["new_last_seen_ts"]
            _save_checkpoint(checkpoint_path, last_seen_ts)
            _LOG.info(
                "subgraph_watch.poll_done",
                events_seen=stats["events_seen"],
                events_copied=stats["events_copied"],
                events_skipped=stats["events_skipped"],
                wall_seconds=round(time.time() - cycle_start, 2),
                new_last_seen_ts=last_seen_ts,
            )
            if args.once:
                return 0
            await asyncio.sleep(args.poll_interval_seconds)
    except (KeyboardInterrupt, asyncio.CancelledError):
        _LOG.info("subgraph_watch.shutdown")
        _save_checkpoint(checkpoint_path, last_seen_ts)
        return 0
    finally:
        await subgraph_client.aclose()
        await data_client.aclose()
        await gamma_client.aclose()
        daemon_conn.close()
        corpus_conn.close()
```

- [ ] **Step 2: Implement `_run_one_cycle`**

Add above `main` in `scripts/watch_subgraph_copy.py`:

```python
async def _run_one_cycle(
    *,
    subgraph_client: Any,
    watchlist_repo: Any,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    market_ticks: Any,
    paper_trades: Any,
    last_seen_ts: int,
    bankroll: float,
    position_fraction: float,
    min_position_cost: float,
) -> dict[str, Any]:
    """Run a single poll cycle. Returns counts + new last_seen_ts."""
    from pscanner.strategies.paper_trader import lookup_fill_price  # local: avoid import cycle risk

    addrs_raw = sorted(watchlist_repo.active_addresses())
    if not addrs_raw:
        _LOG.warning("subgraph_watch.empty_watchlist")
        return {
            "events_seen": 0, "events_copied": 0, "events_skipped": 0,
            "new_last_seen_ts": last_seen_ts,
        }
    addrs = [a.lower() for a in addrs_raw]
    watchlist_set = set(addrs)

    _LOG.info("subgraph_watch.poll_start", addrs=len(addrs), last_seen_ts=last_seen_ts)
    events, indexer_ts = await _fetch_events_since(
        subgraph_client, addrs=addrs, last_seen_ts=last_seen_ts,
    )

    if indexer_ts is not None:
        lag = int(time.time()) - indexer_ts
        if lag >= INDEXER_LAG_ERROR_SECONDS:
            _LOG.error("subgraph_watch.indexer_lag", lag_seconds=lag)
        elif lag >= INDEXER_LAG_WARN_SECONDS:
            _LOG.warning("subgraph_watch.indexer_lag", lag_seconds=lag)

    copied = 0
    skipped = 0
    new_last_seen_ts = last_seen_ts
    for ev in events:
        ev_ts = int(ev["timestamp"])
        new_last_seen_ts = max(new_last_seen_ts, ev_ts)
        try:
            booked = _try_copy_event(
                ev=ev,
                watchlist_set=watchlist_set,
                asset_index=asset_index,
                market_cache=market_cache,
                market_ticks=market_ticks,
                paper_trades=paper_trades,
                bankroll=bankroll,
                position_fraction=position_fraction,
                min_position_cost=min_position_cost,
            )
        except Exception:
            _LOG.exception("subgraph_watch.copy_event_failed", tx=ev.get("transactionHash"))
            skipped += 1
            continue
        if booked:
            copied += 1
        else:
            skipped += 1

    return {
        "events_seen": len(events),
        "events_copied": copied,
        "events_skipped": skipped,
        "new_last_seen_ts": new_last_seen_ts,
    }


def _try_copy_event(
    *,
    ev: dict[str, Any],
    watchlist_set: set[str],
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    market_ticks: Any,
    paper_trades: Any,
    bankroll: float,
    position_fraction: float,
    min_position_cost: float,
) -> bool:
    """Attempt to book a paper copy for one event. Returns True iff inserted."""
    from pscanner.strategies.paper_trader import lookup_fill_price

    maker = ev["maker"]["id"]
    taker = ev["taker"]["id"]
    side = int(ev["side"])
    direction = _compute_copy_direction(maker, taker, side, watchlist_set)
    if direction != "BUY":
        return False

    # Whose side is on the watchlist? Use the watchlist address as source_wallet.
    if maker.lower() in watchlist_set:
        source_wallet = maker
    elif taker.lower() in watchlist_set:
        source_wallet = taker
    else:  # should be unreachable because _compute_copy_direction filtered, but defensive.
        return False

    token_id = ev["tokenId"]
    resolved = _resolve_token(token_id, asset_index, market_cache)
    if resolved is None:
        return False

    fill_price = lookup_fill_price(
        market_cache, market_ticks, resolved.condition_id, resolved.asset_id,
    )
    if fill_price is None:
        _LOG.warning("subgraph_watch.no_fill_price",
                     condition_id=resolved.condition_id, asset_id=resolved.asset_id)
        return False

    cost = bankroll * position_fraction
    if cost < min_position_cost:
        _LOG.debug("subgraph_watch.size_too_small", cost=cost, min=min_position_cost)
        return False
    if not (0.0 < fill_price < 1.0):
        _LOG.debug("subgraph_watch.bad_fill_price", fill_price=fill_price)
        return False

    shares = cost / fill_price
    nav = paper_trades.compute_cost_basis_nav(starting_bankroll=bankroll)
    alert_key = f"{ALERT_KEY_PREFIX}:{ev['transactionHash']}:{resolved.outcome_name}"
    try:
        paper_trades.insert_entry(
            triggering_alert_key=alert_key,
            triggering_alert_detector=DETECTOR_TAG,
            rule_variant=None,
            source_wallet=source_wallet,
            condition_id=resolved.condition_id,
            asset_id=resolved.asset_id,
            outcome=resolved.outcome_name,
            shares=shares,
            fill_price=fill_price,
            cost_usd=cost,
            nav_after_usd=nav,
            ts=int(ev["timestamp"]),
        )
    except sqlite3.IntegrityError:
        _LOG.debug("subgraph_watch.duplicate_alert", alert_key=alert_key)
        return False

    _LOG.info(
        "subgraph_watch.copy_inserted",
        wallet=source_wallet,
        condition_id=resolved.condition_id,
        outcome=resolved.outcome_name,
        fill_price=fill_price,
        shares=round(shares, 4),
        cost_usd=round(cost, 2),
    )
    print(
        f"COPY {source_wallet[:14]}.. {resolved.outcome_name} @ {fill_price:.3f} "
        f"shares={shares:.2f} cost=${cost:.2f} cid={resolved.condition_id[:10]}.."
    )
    return True
```

- [ ] **Step 3: Verify lint + type check**

Run: `uv run ruff check scripts/watch_subgraph_copy.py && uv run ty check scripts/watch_subgraph_copy.py`
Expected: clean.

- [ ] **Step 4: Run the full test suite to ensure nothing else broke**

Run: `uv run pytest -q`
Expected: PASS (all tests, including the 8 in `tests/scripts/test_watch_subgraph_copy.py`).

- [ ] **Step 5: Commit**

```bash
git add scripts/watch_subgraph_copy.py
git commit -m "feat(scripts): wire the main poll loop for subgraph watcher

Single-cycle orchestrator: fetch events, resolve tokenId, compute copy
direction, lookup fill price, insert into paper_trades under
triggering_alert_detector='subgraph_copy'. Indexer-lag log threshold
at 60s WARN / 600s ERROR.

Refs #152."
```

---

## Task 9: Manual smoke test against the live subgraph

**Files:**
- No code changes. This is a one-time validation step before declaring the script done.

- [ ] **Step 1: Confirm GRAPH_API_KEY is sourced**

Run: `set -a; source .env; set +a && test -n "$GRAPH_API_KEY" && echo "OK"`
Expected: `OK`.

- [ ] **Step 2: Confirm the daemon's watchlist has entries**

Run:
```bash
uv run python -c "import sqlite3; c = sqlite3.connect('data/pscanner.sqlite3'); print('watchlist:', c.execute('SELECT COUNT(*) FROM watchlist WHERE active=1').fetchone()[0])"
```
Expected: a non-zero number. If zero, populate via `pscanner watch <addr>` for at least a few wallets before continuing.

- [ ] **Step 3: Run `--once --since-hours 1` against the live subgraph**

Run:
```bash
set -a; source .env; set +a
uv run python scripts/watch_subgraph_copy.py --once --since-hours 1 2>&1 | tee /tmp/subgraph_smoke.log
```
Expected: log shows `subgraph_watch.poll_start`, then one or more `subgraph_watch.copy_inserted` lines, then `subgraph_watch.poll_done` with `events_seen > 0`. Exit code 0.

- [ ] **Step 4: Verify `paper_trades` rows landed under the correct detector tag**

Run:
```bash
uv run python -c "
import sqlite3
c = sqlite3.connect('data/pscanner.sqlite3')
c.row_factory = sqlite3.Row
rows = c.execute(\"SELECT trade_id, source_wallet, outcome, fill_price, cost_usd FROM paper_trades WHERE triggering_alert_detector='subgraph_copy' ORDER BY trade_id DESC LIMIT 5\").fetchall()
for r in rows: print(dict(r))
print(f'total subgraph_copy entries: {c.execute(\"SELECT COUNT(*) FROM paper_trades WHERE triggering_alert_detector=?\", (\"subgraph_copy\",)).fetchone()[0]}')
"
```
Expected: at least one row printed, total count > 0.

- [ ] **Step 5: Verify `pscanner paper status` shows the new source**

Run: `uv run pscanner paper status 2>&1 | grep -A1 subgraph_copy`
Expected: a row in the per-source breakdown table with `detector=subgraph_copy`.

- [ ] **Step 6: If steps 3-5 all pass, document in commit message and commit a marker (optional)**

No code commit needed if smoke passes. Make a note in the issue thread (#152) that the script is validated end-to-end.

---

## Self-review summary

**Spec coverage:**
- Architecture (Architecture section): Task 2 scaffolds, Task 8 wires.
- Data flow startup (Startup section): Task 8 step 1.
- Per-poll cycle (Data flow): Task 8 (`_run_one_cycle`).
- Copy-direction table: Task 3 covers the 4 rows + 2 defensive cases.
- Error handling matrix: Task 8 distributes the handlers; subgraph errors are at `SubgraphClient` layer (existing); per-trade errors are caught in `_run_one_cycle`; schema errors crash via `SubgraphClient.query`'s existing `RuntimeError("GraphQL errors: ...")`.
- Structlog events: Task 8 emits all the events listed in the spec.
- Pagination behavior: Task 5.
- Coexistence: distinct DETECTOR_TAG, no key collision (validated in design).
- Sizing: Task 8 reads from config.
- Testing: Tasks 3-5 cover the 3 listed unit tests; Task 9 covers the manual smoke.
- CLI: Task 2 step 1 has all flags.
- File layout: scripts + data dir match spec.

**Placeholders:** No "TBD"/"TODO". One inline note in Task 5 about graphql variable binding falling back to inline substitution — that's a documented design choice, not a placeholder.

**Type consistency:** `_compute_copy_direction(maker: str, taker: str, side: int, watchlist: set[str]) -> str` consistent across the implementation, test (Task 3), and call site (Task 8). `_build_where_clause(addrs: list[str], last_seen_ts: int) -> dict` consistent. `_resolve_token` returns `_ResolvedToken | None` and is consumed correctly in Task 8.

**Out-of-scope (correctly deferred):** Daemon collector promotion (#152 follow-on, separate issue), automated wallet pruning, cross-platform support. These are intentionally not in this plan.
