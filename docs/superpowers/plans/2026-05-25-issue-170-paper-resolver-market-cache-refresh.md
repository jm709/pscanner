# Paper-resolver market-cache refresh (#170) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PaperResolver` refresh `market_cache` for its open positions before checking resolution, so newly-resolved markets actually get exit rows booked.

**Architecture:** New async helper `refresh_market_cache_row` performs the existing 2-hop `data → slug → gamma → market` lookup and upserts. `PaperResolver._scan` calls it (deduped by `condition_id`) for any open position whose cache row is stale-active or missing, then runs the unchanged resolution check. `paper_trader._backfill_market_cache` is rewritten to delegate to the helper so there's one source of truth.

**Tech Stack:** Python 3.13, `pscanner.poly.data.DataClient`, `pscanner.poly.gamma.GammaClient`, `pscanner.store.repo.MarketCacheRepo`, structlog, pytest + pytest-asyncio + `unittest.mock.AsyncMock`.

**Spec:** `docs/superpowers/specs/2026-05-25-paper-resolver-market-cache-refresh-design.md`

---

## File structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `src/pscanner/strategies/market_cache_refresh.py` | Create | Single `async def refresh_market_cache_row(...) -> bool` helper. Owns the 2-hop slug→gamma lookup and `MarketCacheRepo.upsert`. Log-and-swallow on all failure paths. |
| `tests/strategies/test_market_cache_refresh.py` | Create | Unit tests for the helper: happy path, slug miss, gamma miss, exception swallowed. |
| `src/pscanner/strategies/paper_trader.py` | Modify | `_backfill_market_cache` delegates to the helper while preserving its existing `paper_trader.*` log event names. |
| `src/pscanner/strategies/paper_resolver.py` | Modify | `PaperResolver.__init__` gains `data_client` + `gamma_client` kwargs. `_scan` gains a refresh pass before the resolution pass. |
| `tests/strategies/test_paper_resolver.py` | Modify | Existing tests get `data_client=AsyncMock(), gamma_client=AsyncMock()` kwargs threaded through. Four new tests for refresh behavior. |
| `src/pscanner/scheduler.py` | Modify | `PaperResolver` construction (line 489) passes the two clients. |

---

## Task 1: Helper module + unit tests

**Files:**
- Create: `src/pscanner/strategies/market_cache_refresh.py`
- Create: `tests/strategies/test_market_cache_refresh.py`

- [ ] **Step 1.1: Write failing tests**

Write the test file first.

```python
# tests/strategies/test_market_cache_refresh.py
"""Tests for refresh_market_cache_row."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from pscanner.poly.ids import ConditionId
from pscanner.poly.models import Market
from pscanner.store.repo import MarketCacheRepo
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row

_RESOLVED_MARKET = Market.model_validate(
    {
        "id": "mkt-1",
        "conditionId": "0xcond-1",
        "question": "Will X happen?",
        "slug": "will-x-happen",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["1.0", "0.0"],
        "clobTokenIds": ["asset-yes", "asset-no"],
        "active": False,
        "closed": True,
    },
)


@pytest.mark.asyncio
async def test_happy_path_upserts(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = "will-x-happen"
    gamma_client = AsyncMock()
    gamma_client.get_market_by_slug.return_value = _RESOLVED_MARKET

    ok = await refresh_market_cache_row(
        data_client=data_client,
        gamma_client=gamma_client,
        market_cache=cache,
        condition_id=ConditionId("0xcond-1"),
    )

    assert ok is True
    cached = cache.get_by_condition_id(ConditionId("0xcond-1"))
    assert cached is not None
    assert cached.active is False
    assert cached.outcome_prices == [1.0, 0.0]


@pytest.mark.asyncio
async def test_slug_miss_returns_false(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = None
    gamma_client = AsyncMock()

    ok = await refresh_market_cache_row(
        data_client=data_client,
        gamma_client=gamma_client,
        market_cache=cache,
        condition_id=ConditionId("0xcond-1"),
    )

    assert ok is False
    gamma_client.get_market_by_slug.assert_not_awaited()
    assert cache.get_by_condition_id(ConditionId("0xcond-1")) is None


@pytest.mark.asyncio
async def test_gamma_miss_returns_false(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = "will-x-happen"
    gamma_client = AsyncMock()
    gamma_client.get_market_by_slug.return_value = None

    ok = await refresh_market_cache_row(
        data_client=data_client,
        gamma_client=gamma_client,
        market_cache=cache,
        condition_id=ConditionId("0xcond-1"),
    )

    assert ok is False
    assert cache.get_by_condition_id(ConditionId("0xcond-1")) is None


@pytest.mark.asyncio
async def test_exception_logged_and_swallowed(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.side_effect = RuntimeError("boom")
    gamma_client = AsyncMock()

    with capture_logs() as logs:
        ok = await refresh_market_cache_row(
            data_client=data_client,
            gamma_client=gamma_client,
            market_cache=cache,
            condition_id=ConditionId("0xcond-1"),
        )

    assert ok is False
    assert any(entry["event"] == "market_cache.refresh.failed" for entry in logs)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_market_cache_refresh.py -v`
Expected: All 4 tests fail with `ModuleNotFoundError: No module named 'pscanner.strategies.market_cache_refresh'`.

- [ ] **Step 1.3: Implement the helper**

```python
# src/pscanner/strategies/market_cache_refresh.py
"""Refresh one ``market_cache`` row from gamma via the 2-hop slug lookup.

Used by ``PaperResolver`` (to keep cache rows truthful for newly-resolved
markets) and by ``PaperTrader._backfill_market_cache`` (its original
caller, kept on the same helper for a single source of truth).
"""

from __future__ import annotations

import structlog

from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import MarketCacheRepo

_LOG = structlog.get_logger(__name__)


async def refresh_market_cache_row(
    *,
    data_client: DataClient,
    gamma_client: GammaClient,
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
) -> bool:
    """Fetch one market via the slug→gamma 2-hop and upsert into market_cache.

    The 2-hop sequence is the existing pattern from
    ``paper_trader._backfill_market_cache``: data-api ``/trades`` exposes a
    market's slug per trade row, gamma ``/markets?slug=`` returns the full
    ``Market``. ``gamma.get_market_by_slug`` internally passes
    ``closed=true`` so the lookup succeeds for both active and resolved
    markets.

    Args:
        data_client: For ``get_market_slug_by_condition_id``.
        gamma_client: For ``get_market_by_slug``.
        market_cache: Where the resolved ``Market`` is upserted.
        condition_id: The on-chain market identifier to refresh.

    Returns:
        ``True`` iff a row was successfully upserted. ``False`` on slug
        miss, gamma miss, or any swallowed exception.
    """
    try:
        slug = await data_client.get_market_slug_by_condition_id(condition_id)
        if slug is None:
            _LOG.debug("market_cache.refresh.no_slug", condition_id=condition_id)
            return False
        market = await gamma_client.get_market_by_slug(slug)
        if market is None:
            _LOG.debug(
                "market_cache.refresh.no_gamma_market",
                condition_id=condition_id,
                slug=slug,
            )
            return False
    except Exception:
        _LOG.warning(
            "market_cache.refresh.failed",
            condition_id=condition_id,
            exc_info=True,
        )
        return False
    market_cache.upsert(market)
    _LOG.info(
        "market_cache.refresh.ok",
        condition_id=condition_id,
        slug=market.slug,
        active=market.active,
    )
    return True


__all__ = ["refresh_market_cache_row"]
```

- [ ] **Step 1.4: Run tests, verify pass**

Run: `uv run pytest tests/strategies/test_market_cache_refresh.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 1.5: Lint + type check the new files**

Run: `uv run ruff check src/pscanner/strategies/market_cache_refresh.py tests/strategies/test_market_cache_refresh.py && uv run ty check src/pscanner/strategies/market_cache_refresh.py`
Expected: No errors.

- [ ] **Step 1.6: Commit**

```bash
git add src/pscanner/strategies/market_cache_refresh.py tests/strategies/test_market_cache_refresh.py
git commit -m "feat(strategies): add refresh_market_cache_row helper (#170)

2-hop data→slug→gamma→upsert lifted from paper_trader so PaperResolver
can reuse it. Log-and-swallow on all failure paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: PaperTrader delegates to the helper

**Files:**
- Modify: `src/pscanner/strategies/paper_trader.py:359-401`

- [ ] **Step 2.1: Verify existing paper_trader backfill tests pass before refactor**

Run: `uv run pytest tests/strategies/test_paper_trader.py -v -k backfill`
Expected: Existing backfill tests PASS. Note which tests run — those are the regression contract.

- [ ] **Step 2.2: Refactor `_backfill_market_cache` to delegate**

Open `src/pscanner/strategies/paper_trader.py`. Find the existing method (lines 359-401, starting `async def _backfill_market_cache`). Replace the body with a delegation that preserves the existing `paper_trader.*` log surface so any operator dashboards keep working.

Replace lines 359-401 with:

```python
    async def _backfill_market_cache(self, condition_id: ConditionId) -> bool:
        """Fetch a market's metadata via gamma and write it to ``market_cache``.

        Thin wrapper around
        :func:`pscanner.strategies.market_cache_refresh.refresh_market_cache_row`.
        Preserves the legacy ``paper_trader.market_cache_backfilled`` /
        ``paper_trader.backfill_failed`` events for operator dashboards;
        the helper emits granular ``market_cache.refresh.*`` events with
        the same payload alongside.

        Args:
            condition_id: The market's on-chain condition id.

        Returns:
            ``True`` when the cache was successfully populated.
        """
        ok = await refresh_market_cache_row(
            data_client=self._data_client,
            gamma_client=self._gamma_client,
            market_cache=self._market_cache,
            condition_id=condition_id,
        )
        if not ok:
            _LOG.debug("paper_trader.backfill_failed", condition_id=condition_id)
            return False
        cached = self._market_cache.get_by_condition_id(condition_id)
        _LOG.info(
            "paper_trader.market_cache_backfilled",
            condition_id=condition_id,
            slug=cached.event_slug if cached is not None else None,
        )
        return True
```

Add this import near the other strategies imports at the top of the file (find the line `from pscanner.store.repo import (` and add a sibling import right after the closing `)`):

```python
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row
```

- [ ] **Step 2.3: Run paper_trader tests, verify pass**

Run: `uv run pytest tests/strategies/test_paper_trader.py -v`
Expected: All tests pass. If any test asserts on the legacy `paper_trader.no_slug` or `paper_trader.no_gamma_market` event names, capture the failure — we may need to keep emitting those too. (Inspect with `rg -n "paper_trader\.no_slug|paper_trader\.no_gamma_market" tests/` if a failure surfaces.)

- [ ] **Step 2.4: Lint + type check**

Run: `uv run ruff check src/pscanner/strategies/paper_trader.py && uv run ty check src/pscanner/strategies/paper_trader.py`
Expected: No errors.

- [ ] **Step 2.5: Commit**

```bash
git add src/pscanner/strategies/paper_trader.py
git commit -m "refactor(paper_trader): delegate _backfill_market_cache to helper (#170)

Single source of truth for the 2-hop market_cache refresh. Preserves the
paper_trader.market_cache_backfilled / paper_trader.backfill_failed log
events for operator dashboards.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: PaperResolver — new kwargs + refresh pass + tests

**Files:**
- Modify: `src/pscanner/strategies/paper_resolver.py`
- Modify: `tests/strategies/test_paper_resolver.py`

This task is split across two phases:
- Phase A: thread the new kwargs through without changing behavior (so existing tests pass).
- Phase B: add the refresh pass + new tests.

### Phase A — additive ctor signature change

- [ ] **Step 3.1: Update PaperResolver ctor to accept new clients (no behavior change)**

Open `src/pscanner/strategies/paper_resolver.py`. Add imports near the top (after the existing `from pscanner.poly.ids import ...` line):

```python
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row
```

Modify `__init__` (currently lines 64-85). Replace with:

```python
    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        market_cache: MarketCacheRepo,
        paper_trades: PaperTradesRepo,
        data_client: DataClient,
        gamma_client: GammaClient,
        clock: Clock | None = None,
    ) -> None:
        """Wire dependencies; see :class:`PollingDetector` for the loop shape.

        Args:
            config: Paper-trading config; supplies the scan interval and the
                starting bankroll used when stamping ``nav_after_usd`` on
                exit rows.
            market_cache: Read/write access to the cached market table. The
                resolver refreshes stale-active rows for open positions
                before checking resolution (#170).
            paper_trades: Read/write repo for ``paper_trades``.
            data_client: Used to resolve a market's slug from its
                ``condition_id`` during refresh.
            gamma_client: Used to fetch a market by slug during refresh.
            clock: Optional injected :class:`Clock`; defaults to a real clock.
        """
        super().__init__(clock=clock)
        self._config = config
        self._market_cache = market_cache
        self._paper_trades = paper_trades
        self._data_client = data_client
        self._gamma_client = gamma_client
```

- [ ] **Step 3.2: Update existing PaperResolver tests to pass the new kwargs**

Open `tests/strategies/test_paper_resolver.py`. At the top of the file (after the existing `from pscanner.util.clock import FakeClock` import) add:

```python
from unittest.mock import AsyncMock
```

Add a small fixture-style helper just above `_cache_market` (near line 28):

```python
def _async_mocks() -> tuple[AsyncMock, AsyncMock]:
    """Return (data_client, gamma_client) AsyncMocks suitable for tests
    whose cache rows are pre-seeded ``active=False`` (refresh never fires).
    """
    return AsyncMock(), AsyncMock()
```

For every existing `PaperResolver(...)` construction in the file (search for `PaperResolver(`), add `data_client=...` and `gamma_client=...` kwargs. There are 5 sites currently:

1. `test_resolver_books_winning_exit` (line ~141): insert before `clock=clock`:
   ```python
       data, gamma = _async_mocks()
       resolver = PaperResolver(
           config=cfg,
           market_cache=cache,
           paper_trades=paper,
           data_client=data,
           gamma_client=gamma,
           clock=clock,
       )
   ```
2. `test_resolver_books_losing_exit` (line ~162): same pattern.
3. `test_resolver_skips_unresolved` (line ~182): same pattern.
4. `test_resolver_books_multiple_in_one_scan` (line ~214): same pattern.
5. `test_resolver_interval_from_config` (line ~226): same pattern (this one has no `clock`, just insert the two kwargs).
6. `test_resolver_keeps_position_open_when_insert_exit_raises` (line ~248): same pattern.

- [ ] **Step 3.3: Run existing PaperResolver tests, verify they still pass**

Run: `uv run pytest tests/strategies/test_paper_resolver.py -v`
Expected: All existing tests PASS (the new ctor kwargs are now provided; behavior is unchanged because the refresh pass isn't implemented yet — the AsyncMocks are never called when cache rows are pre-seeded `active=False`).

- [ ] **Step 3.4: Commit (additive ctor change is a safe checkpoint)**

```bash
git add src/pscanner/strategies/paper_resolver.py tests/strategies/test_paper_resolver.py
git commit -m "refactor(paper_resolver): thread data_client + gamma_client kwargs (#170)

Additive ctor change ahead of the resolution-refresh logic. Existing
tests pass AsyncMock instances; refresh path is implemented in the
follow-up commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Phase B — refresh pass + new tests (TDD)

- [ ] **Step 3.5: Write the four new failing tests**

Append to `tests/strategies/test_paper_resolver.py`:

```python
@pytest.mark.asyncio
async def test_resolver_refreshes_stale_active_market_then_books_exit(tmp_db) -> None:
    """A market whose cache row is still active=1 gets refreshed via gamma
    and, if gamma reports it resolved, an exit row is booked in the same scan.
    """
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    # Pre-seed cache as still-active (the bug condition).
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)

    from pscanner.poly.models import Market

    resolved_market = Market.model_validate(
        {
            "id": "mkt-0xcond-1",
            "conditionId": "0xcond-1",
            "question": "Test market",
            "slug": "test-market",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["asset-yes", "asset-no"],
            "active": False,
            "closed": True,
        },
    )
    data = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "test-market"
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = resolved_market

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    data.get_market_slug_by_condition_id.assert_awaited_once_with(ConditionId("0xcond-1"))
    gamma.get_market_by_slug.assert_awaited_once_with("test-market")
    assert paper.list_open_positions() == []
    assert paper.compute_cost_basis_nav(starting_bankroll=1000.0) == 1010.0


@pytest.mark.asyncio
async def test_resolver_dedups_refresh_per_scan(tmp_db) -> None:
    """Two open positions on the same condition_id trigger exactly one refresh."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    # Second open position on same condition_id, different fill (e.g. twin trade).
    paper.insert_entry(
        triggering_alert_key="k-0xcond-1-no",
        triggering_alert_detector="velocity",
        rule_variant="fade",
        source_wallet="0xw2",
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-no"),
        outcome="no",
        shares=10.0,
        fill_price=0.4,
        cost_usd=4.0,
        nav_after_usd=996.0,
        ts=_NOW,
    )

    from pscanner.poly.models import Market

    resolved_market = Market.model_validate(
        {
            "id": "mkt-0xcond-1",
            "conditionId": "0xcond-1",
            "question": "Test market",
            "slug": "test-market",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["asset-yes", "asset-no"],
            "active": False,
            "closed": True,
        },
    )
    data = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "test-market"
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = resolved_market

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    assert data.get_market_slug_by_condition_id.await_count == 1
    assert gamma.get_market_by_slug.await_count == 1
    # Both positions resolved in the same scan.
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_resolver_skips_refresh_when_cache_already_inactive(tmp_db) -> None:
    """Cache rows that already say active=False are not re-fetched."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    data = AsyncMock()
    gamma = AsyncMock()

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    data.get_market_slug_by_condition_id.assert_not_awaited()
    gamma.get_market_by_slug.assert_not_awaited()
    # Exit still booked via the unchanged resolution path.
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_resolver_refresh_failure_does_not_block_other_positions(tmp_db) -> None:
    """One market's refresh raising must not prevent another from being booked."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(
        cache,
        condition_id="0xcond-1",
        active=True,
        outcome_prices=[0.5, 0.5],
        asset_ids=["a-y1", "a-n1"],
    )
    _cache_market(
        cache,
        condition_id="0xcond-2",
        active=True,
        outcome_prices=[0.5, 0.5],
        asset_ids=["a-y2", "a-n2"],
    )
    _open_position(paper, condition_id="0xcond-1", asset_id="a-y1", outcome="yes")
    _open_position(paper, condition_id="0xcond-2", asset_id="a-y2", outcome="yes")

    from pscanner.poly.models import Market

    resolved_cond2 = Market.model_validate(
        {
            "id": "mkt-0xcond-2",
            "conditionId": "0xcond-2",
            "question": "Second market",
            "slug": "second-market",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["a-y2", "a-n2"],
            "active": False,
            "closed": True,
        },
    )

    async def slug_side_effect(cid: ConditionId) -> str | None:
        if cid == ConditionId("0xcond-1"):
            raise RuntimeError("transient gamma upstream failure")
        return "second-market"

    data = AsyncMock()
    data.get_market_slug_by_condition_id.side_effect = slug_side_effect
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = resolved_cond2

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    open_after = paper.list_open_positions()
    assert len(open_after) == 1
    assert open_after[0].condition_id == ConditionId("0xcond-1")
```

- [ ] **Step 3.6: Run the new tests to verify they fail**

Run: `uv run pytest tests/strategies/test_paper_resolver.py -v -k "refresh or dedup"`
Expected: All four new tests FAIL (gamma/data mocks are never awaited; positions remain open).

- [ ] **Step 3.7: Implement the refresh pass in `_scan`**

Open `src/pscanner/strategies/paper_resolver.py`. Replace the `_scan` method (currently lines 90-102) with:

```python
    async def _scan(self, sink: AlertSink) -> None:
        """Refresh stale-active market_cache rows, then book exits.

        Errors on individual positions or refresh calls are logged and
        skipped — one bad row never blocks the rest.
        """
        del sink  # contract: _scan accepts a sink; we don't emit
        open_positions = list(self._paper_trades.list_open_positions())
        await self._refresh_stale_markets(open_positions)
        booked = 0
        for pos in open_positions:
            if self._maybe_book_exit(pos):
                booked += 1
        if booked:
            _LOG.info("paper_resolver.scan_completed", booked=booked)

    async def _refresh_stale_markets(
        self,
        open_positions: list[OpenPaperPosition],
    ) -> None:
        """Refresh ``market_cache`` for any open-position market that's
        still cached as ``active=True`` (or missing entirely).

        Deduplicates by ``condition_id`` so twin positions on the same
        market only trigger one gamma call per scan. Sequential awaits —
        no ``gather`` — to keep gamma traffic predictable under the
        shared rate limiter.
        """
        seen: set[ConditionId] = set()
        for pos in open_positions:
            if pos.condition_id in seen:
                continue
            seen.add(pos.condition_id)
            cached = self._market_cache.get_by_condition_id(pos.condition_id)
            if cached is not None and not cached.active:
                continue
            await refresh_market_cache_row(
                data_client=self._data_client,
                gamma_client=self._gamma_client,
                market_cache=self._market_cache,
                condition_id=pos.condition_id,
            )
```

- [ ] **Step 3.8: Run the resolver tests, verify all pass**

Run: `uv run pytest tests/strategies/test_paper_resolver.py -v`
Expected: All tests PASS (existing + 4 new).

- [ ] **Step 3.9: Lint + type check**

Run: `uv run ruff check src/pscanner/strategies/paper_resolver.py tests/strategies/test_paper_resolver.py && uv run ty check src/pscanner/strategies/paper_resolver.py`
Expected: No errors.

- [ ] **Step 3.10: Commit**

```bash
git add src/pscanner/strategies/paper_resolver.py tests/strategies/test_paper_resolver.py
git commit -m "fix(paper_resolver): refresh stale-active market_cache rows in _scan (#170)

Before checking resolution, refresh each open-position market via the
shared 2-hop helper if its cache row is still active=1 (or missing).
Dedup per scan by condition_id. Refresh failures are logged-and-swallowed
so one bad market never blocks the rest of the scan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire new kwargs in scheduler

**Files:**
- Modify: `src/pscanner/scheduler.py:489-494`

- [ ] **Step 4.1: Pass the two clients into PaperResolver construction**

Open `src/pscanner/scheduler.py`. Find the `PaperResolver` construction (line 489). Replace lines 489-494 with:

```python
        detectors["paper_resolver"] = PaperResolver(
            config=self._config.paper_trading,
            market_cache=self._market_cache_repo,
            paper_trades=paper_trades_repo,
            data_client=self._clients.data_client,
            gamma_client=self._clients.gamma_client,
            clock=self._clock,
        )
```

- [ ] **Step 4.2: Run scheduler tests**

Run: `uv run pytest tests/test_scheduler.py -v -k "paper or resolver"`
Expected: PASS. (If no matching tests, run `uv run pytest tests/test_scheduler.py -v` to confirm no regressions.)

- [ ] **Step 4.3: Lint + type check**

Run: `uv run ruff check src/pscanner/scheduler.py && uv run ty check src/pscanner/scheduler.py`
Expected: No errors.

- [ ] **Step 4.4: Commit**

```bash
git add src/pscanner/scheduler.py
git commit -m "fix(scheduler): wire data_client + gamma_client into PaperResolver (#170)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Full verify gate

**Files:** none modified — verification only.

- [ ] **Step 5.1: Run full lint + format + types**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: No errors. (If `ruff format --check` flags files we touched, run `uv run ruff format <file>` and amend the relevant commit.)

- [ ] **Step 5.2: Run the touched test surface**

Run: `uv run pytest tests/strategies/ tests/test_scheduler.py -q`
Expected: All tests pass. Total runtime should be a couple minutes; this is the smallest surface that covers every modified file.

- [ ] **Step 5.3: Run full suite if Step 5.2 passes**

Run: `uv run pytest -q`
Expected: Same pass count as `main`. If pre-existing failures appear that are not caused by these changes, note them in the PR body but do NOT fix them in this PR (per the "no new diagnostics" memory).

- [ ] **Step 5.4: Push branch + open PR**

Run: `git log --oneline main..HEAD` and confirm the 4 commits map to Tasks 1, 2, 3, 4 (plus the existing spec commit). Then push and open a PR linked to #170:

```bash
git push -u origin worktree-fix-paper-trade-resolution
gh pr create --title "fix(paper_resolver): refresh stale market_cache before booking exits (#170)" --body "$(cat <<'EOF'
## Summary
- New `refresh_market_cache_row` helper performs the existing data→slug→gamma→upsert refresh.
- `PaperResolver._scan` calls it (deduped by condition_id) for any open-position market whose cache row is still `active=1` or missing, then runs the unchanged resolution check.
- `PaperTrader._backfill_market_cache` delegates to the new helper so there's one source of truth for the refresh.

Closes #170.

## Test plan
- [ ] `uv run pytest tests/strategies/test_market_cache_refresh.py -v`
- [ ] `uv run pytest tests/strategies/test_paper_resolver.py -v`
- [ ] `uv run pytest tests/strategies/test_paper_trader.py -v`
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run ty check`
- [ ] Manual smoke on desktop: run daemon for one full resolver-scan cycle and confirm exit rows land for resolved markets (per #170 acceptance criteria).

## Design doc
[`docs/superpowers/specs/2026-05-25-paper-resolver-market-cache-refresh-design.md`](docs/superpowers/specs/2026-05-25-paper-resolver-market-cache-refresh-design.md)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
