# Issue #102 — LiveHistoryProvider Live-Market Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the gate-model detector score trades on currently-open markets that aren't yet in `corpus_markets`. Today, `LiveHistoryProvider.market_metadata(condition_id)` raises `KeyError` for any market not loaded from the corpus DB at boot, and the corpus only contains backfilled (closed) markets — so the live polling target is empty by definition.

**Architecture:** Adopt option B from the issue. `MarketScopedTradeCollector` already enumerates open events via `gamma.iter_events` to filter by category and volume. Pass the `LiveHistoryProvider` into that collector so it can call a new `provider.set_market_metadata(condition_id, metadata)` for every market it considers — not only the top-N selected, since trades on borderline markets could be enqueued before the working set converges. The metadata fill is zero extra API calls because the data is already in flight.

**Tech Stack:** Python 3.13, no new deps. Edits live in three modules plus their tests; CLAUDE.md gets a one-line note.

---

## File Structure

- Modify: `src/pscanner/daemon/live_history.py:71-100` — add `set_market_metadata` method.
- Modify: `src/pscanner/collectors/market_scoped_trades.py:39-78` — accept optional `provider`, call `set_market_metadata` from `refresh_market_set` for every category-matching market.
- Modify: `src/pscanner/scheduler.py:329-334` — pass `provider=self._live_history_provider` when wiring `MarketScopedTradeCollector`.
- Modify: `tests/daemon/test_live_history.py` — add a unit test for `set_market_metadata`.
- Modify: `tests/collectors/test_market_scoped_trades.py` — add a test that `refresh_market_set` populates the provider for live markets.
- Modify: `tests/scheduler/test_gate_model_wiring.py` — assert the wired collector references the provider.
- Modify: `CLAUDE.md` — append a note to the gate-model bullet.

Tasks 1, 2, 3, 4 below are sequential — task 2 needs `set_market_metadata` from task 1, task 3 needs the collector accepting `provider` from task 2.

---

## Task 1: Add `LiveHistoryProvider.set_market_metadata`

**Files:**
- Modify: `src/pscanner/daemon/live_history.py:98-100`
- Test: `tests/daemon/test_live_history.py` (new test)

A one-line setter on the in-memory metadata dict. Idempotent overwrite — re-calling with a fresher metadata for the same `condition_id` replaces it (e.g. once the market closes, `closed_at` becomes meaningful).

- [ ] **Step 1: Write a failing test**

Append to `tests/daemon/test_live_history.py` (find a similar existing test for a model — most live_history tests build a provider with `metadata={}` and exercise the wallet/market state methods, so this fits naturally there):

```python
def test_set_market_metadata_inserts_then_overwrites(tmp_db: sqlite3.Connection) -> None:
    """`set_market_metadata` lets callers seed metadata at runtime (issue #102)."""
    provider = LiveHistoryProvider(conn=tmp_db, metadata={})
    with pytest.raises(KeyError):
        provider.market_metadata("0xc1")

    initial = MarketMetadata(
        condition_id="0xc1", category="esports", closed_at=0, opened_at=0
    )
    provider.set_market_metadata("0xc1", initial)
    assert provider.market_metadata("0xc1") == initial

    updated = MarketMetadata(
        condition_id="0xc1",
        category="esports",
        closed_at=1_700_000_000,
        opened_at=1_690_000_000,
    )
    provider.set_market_metadata("0xc1", updated)
    assert provider.market_metadata("0xc1") == updated
```

Imports the test will need at the top of the file (likely already present — confirm before editing):

```python
import sqlite3
import pytest
from pscanner.corpus.features import MarketMetadata
from pscanner.daemon.live_history import LiveHistoryProvider
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/daemon/test_live_history.py::test_set_market_metadata_inserts_then_overwrites -v`

Expected: FAIL — `AttributeError: 'LiveHistoryProvider' object has no attribute 'set_market_metadata'`.

- [ ] **Step 3: Add the method**

Insert after `market_metadata` (currently at lines 98-100) in `src/pscanner/daemon/live_history.py`:

```python
    def set_market_metadata(self, condition_id: str, metadata: MarketMetadata) -> None:
        """Insert or overwrite metadata for ``condition_id``.

        Used by :class:`MarketScopedTradeCollector` to push metadata for
        currently-open markets that aren't yet in ``corpus_markets`` — the
        boot-time corpus load only covers resolved markets, so live trading
        targets need this runtime injection (issue #102).
        """
        self._metadata[condition_id] = metadata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/daemon/test_live_history.py::test_set_market_metadata_inserts_then_overwrites -v`

Expected: PASS.

- [ ] **Step 5: Run full live_history test suite**

Run: `uv run pytest tests/daemon/test_live_history.py -v`

Expected: all PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/daemon/live_history.py tests/daemon/test_live_history.py
git commit -m "feat(live_history): add set_market_metadata for runtime injection (#102)"
```

---

## Task 2: Plumb `LiveHistoryProvider` into `MarketScopedTradeCollector`

**Files:**
- Modify: `src/pscanner/collectors/market_scoped_trades.py:39-78`
- Test: `tests/collectors/test_market_scoped_trades.py` (new test)

Make `provider` an optional ctor kwarg (so existing tests still compile without one). Inside `refresh_market_set`, call `provider.set_market_metadata` for every market that passes the category + volume filter — not only the top-N. Trades for any of those markets could be enqueued before the working-set sort converges; if a market drops out of the top-N this cycle but back in next cycle, its metadata is already cached.

`MarketMetadata` requires `category: str`, `closed_at: int`, `opened_at: int`. `category` is already computed via `categorize_event(event).value`. For live markets we don't have a meaningful `opened_at` or `closed_at`; both default to `0`. **Why this is safe:** `compute_features` reads `meta.closed_at` only for the `time_to_resolution_seconds` feature, which is in `LEAKAGE_COLS` and dropped at inference (`pscanner.ml.preprocessing.LEAKAGE_COLS`). The detector also references `metadata.category` directly (gate_model.py:189) — that's the only field that's load-bearing at inference. So zeroes are correct.

- [ ] **Step 1: Write a failing test**

Append to `tests/collectors/test_market_scoped_trades.py`:

```python
@pytest.mark.asyncio
async def test_refresh_populates_provider_metadata_for_every_candidate(
    tmp_path: Path,
) -> None:
    """`refresh_market_set` writes MarketMetadata for every category-matching market.

    Issue #102: live open markets aren't in `corpus_markets`, so the
    collector is the only thing that can teach the provider about them
    before a trade arrives.
    """
    from pscanner.daemon.live_history import LiveHistoryProvider
    from pscanner.store.db import init_db

    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=10.0,
        max_markets=2,
    )
    esports_a = _make_event(
        slug="ev-a",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xMA", volume=500.0)],
    )
    esports_b = _make_event(
        slug="ev-b",
        tags=["Esports"],
        markets=[
            _make_market(condition_id="0xMB1", volume=400.0),
            _make_market(condition_id="0xMB2", volume=15.0),  # passes floor; below top-N
        ],
    )
    politics = _make_event(
        slug="ev-p",
        tags=["Politics"],
        markets=[_make_market(condition_id="0xMP", volume=99999.0)],
    )
    gamma = _FakeGammaClient([esports_a, esports_b, politics])
    data_client = _FakeDataClient(by_market={})

    db_path = tmp_path / "daemon.sqlite3"
    conn = init_db(db_path)
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        collector = MarketScopedTradeCollector(
            config=cfg, gamma=gamma, data_client=data_client, provider=provider
        )
        selected = await collector.refresh_market_set()
    finally:
        conn.close()

    assert selected == ["0xMA", "0xMB1"]  # top-2 by volume
    # All three esports markets seeded — even the one below top-N.
    assert provider.market_metadata("0xMA").category == "esports"
    assert provider.market_metadata("0xMB1").category == "esports"
    assert provider.market_metadata("0xMB2").category == "esports"
    # Politics market is filtered out at the category gate.
    with pytest.raises(KeyError):
        provider.market_metadata("0xMP")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py::test_refresh_populates_provider_metadata_for_every_candidate -v`

Expected: FAIL — `MarketScopedTradeCollector.__init__()` rejects the `provider=` kwarg.

- [ ] **Step 3: Extend the collector**

Update `__init__` (lines 39-52) in `src/pscanner/collectors/market_scoped_trades.py`:

```python
    def __init__(
        self,
        *,
        config: GateModelMarketFilterConfig,
        gamma: GammaClient,
        data_client: DataClient,
        provider: LiveHistoryProvider | None = None,
    ) -> None:
        """Initialize the collector with configuration and API clients.

        ``provider``, when supplied, receives :class:`MarketMetadata` for every
        candidate market enumerated by :meth:`refresh_market_set`. This is how
        live open markets (not yet in ``corpus_markets``) become visible to
        :class:`GateModelDetector` (issue #102).
        """
        self._config = config
        self._gamma = gamma
        self._data_client = data_client
        self._provider = provider
        self._markets: list[str] = []
        self._callbacks: list[Callable[[WalletTrade], None]] = []
        self._last_seen_ts: dict[str, int] = {}
```

Add the `LiveHistoryProvider` import to the `TYPE_CHECKING` block (lines 25-28):

```python
if TYPE_CHECKING:
    from pscanner.config import GateModelMarketFilterConfig
    from pscanner.daemon.live_history import LiveHistoryProvider
    from pscanner.poly.data import DataClient
    from pscanner.poly.gamma import GammaClient
```

Add a runtime import for `MarketMetadata` (since `refresh_market_set` instantiates it):

```python
from pscanner.corpus.features import MarketMetadata
```

(Place it alongside the other runtime imports near the top.)

Update `refresh_market_set` (lines 58-78):

```python
    async def refresh_market_set(self) -> list[str]:
        """Enumerate events, filter by category + volume, return top-N condition_ids.

        Side-effect: when ``provider`` was supplied at construction, every
        market that passes the category + volume gate gets a
        :class:`MarketMetadata` entry pushed into the provider. We seed every
        candidate (not just the top-N selected) because a market that drops
        out of the top-N this cycle could re-enter on the next refresh, and
        callers may have already enqueued trades for it.
        """
        accepted = set(self._config.accepted_categories)
        floor = self._config.min_volume_24h_usd
        candidates: list[tuple[float, str]] = []
        async for event in self._gamma.iter_events():
            category = categorize_event(event).value
            if category not in accepted:
                continue
            for market in event.markets:
                cond_id = market.condition_id
                if cond_id is None:
                    continue
                volume = float(market.volume or 0.0)
                if volume < floor:
                    continue
                cond_id_str = str(cond_id)
                candidates.append((volume, cond_id_str))
                if self._provider is not None:
                    self._provider.set_market_metadata(
                        cond_id_str,
                        MarketMetadata(
                            condition_id=cond_id_str,
                            category=category,
                            closed_at=0,
                            opened_at=0,
                        ),
                    )
        candidates.sort(reverse=True)
        selected = [cid for _, cid in candidates[: self._config.max_markets]]
        self._markets = selected
        return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py::test_refresh_populates_provider_metadata_for_every_candidate -v`

Expected: PASS.

- [ ] **Step 5: Run full collector test suite**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py -v`

Expected: all PASS — existing tests don't pass `provider`, defaults to `None`, no behavior change.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/collectors/market_scoped_trades.py tests/collectors/test_market_scoped_trades.py
git commit -m "feat(market_scoped_trades): seed LiveHistoryProvider for live markets (#102)"
```

---

## Task 3: Wire the provider into the collector at scheduler boot

**Files:**
- Modify: `src/pscanner/scheduler.py:329-334`
- Test: `tests/scheduler/test_gate_model_wiring.py` (extend existing test)

The collector now accepts the provider; the scheduler must pass it. Both objects already exist in `Scanner` (`self._live_history_provider` is constructed at line 168-177; the collector is built at line 330).

- [ ] **Step 1: Extend a wiring test**

Update `test_scanner_builds_gate_model_when_enabled` in `tests/scheduler/test_gate_model_wiring.py:136-149` to assert the wiring:

```python
@pytest.mark.asyncio
async def test_scanner_builds_gate_model_when_enabled(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(artifact_dir=artifact_dir, gate_enabled=True, filter_enabled=True)
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        assert isinstance(scanner._detectors.get("gate_model"), GateModelDetector)
        collector = scanner._collectors.get("market_scoped_trades")
        assert isinstance(collector, MarketScopedTradeCollector)
        # Issue #102: collector must reference the live provider so it can
        # seed metadata for currently-open markets.
        assert collector._provider is scanner._live_history_provider
        assert scanner._live_history_provider is not None
    finally:
        await scanner.aclose()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py::test_scanner_builds_gate_model_when_enabled -v`

Expected: FAIL — `collector._provider` is `None`.

- [ ] **Step 3: Wire the provider in `_build_collectors`**

Replace lines 329-334 in `src/pscanner/scheduler.py`:

```python
        if self._config.gate_model_market_filter.enabled:
            collectors["market_scoped_trades"] = MarketScopedTradeCollector(
                config=self._config.gate_model_market_filter,
                gamma=self._clients.gamma_client,
                data_client=self._clients.data_client,
                provider=self._live_history_provider,
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py::test_scanner_builds_gate_model_when_enabled -v`

Expected: PASS.

- [ ] **Step 5: Run full scheduler test suite**

Run: `uv run pytest tests/scheduler/ -v`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/scheduler.py tests/scheduler/test_gate_model_wiring.py
git commit -m "feat(scheduler): wire LiveHistoryProvider into market-scoped collector (#102)"
```

---

## Task 4: End-to-end gate-model emit test for a live (non-corpus) market

**Files:**
- Test: `tests/scheduler/test_gate_model_wiring.py` (new test)

Acceptance criterion: "a live market not in the corpus, after `refresh_market_set`, has `provider.market_metadata(cond_id).category` populated." Task 2 already covers that at the unit level. This task verifies the end-to-end path: the metadata appears in the *same provider instance* the gate-model detector is reading.

- [ ] **Step 1: Write the integration-style test**

Append to `tests/scheduler/test_gate_model_wiring.py`:

```python
@pytest.mark.asyncio
async def test_collector_refresh_makes_live_market_visible_to_detector(
    tmp_path: Path,
) -> None:
    """End-to-end: a live (non-corpus) market becomes available to the detector.

    Issue #102: corpus_markets only holds backfilled markets, so live trading
    targets are absent from `provider.market_metadata` at boot. The collector's
    refresh must populate the provider so the detector's metadata lookup
    succeeds.
    """
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(artifact_dir=artifact_dir, gate_enabled=True, filter_enabled=True)
    clients = _make_stub_clients()

    # Stub gamma to return one open esports market — exactly the shape of a
    # production live market.
    from pscanner.poly.models import Event, Market

    market = Market.model_validate(
        {
            "id": "m-live",
            "conditionId": "0xLIVE",
            "question": "q",
            "slug": "slug-live",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.4","0.6"]',
            "liquidity": "1000",
            "volume": "200000",
            "active": True,
            "closed": False,
            "clobTokenIds": '["a1","a2"]',
        }
    )
    event = Event.model_validate(
        {
            "id": "e-live",
            "slug": "slug-event",
            "title": "live-event",
            "tags": [{"label": "Esports"}],
            "markets": [market.model_dump(by_alias=True)],
            "active": True,
            "closed": False,
        }
    )

    async def _fake_iter_events(**_: object):
        yield event

    clients.gamma_client.iter_events = _fake_iter_events  # type: ignore[method-assign]

    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        # Pre-condition: live market is NOT in the corpus-loaded metadata.
        assert scanner._live_history_provider is not None
        with pytest.raises(KeyError):
            scanner._live_history_provider.market_metadata("0xLIVE")

        collector = scanner._collectors["market_scoped_trades"]
        assert isinstance(collector, MarketScopedTradeCollector)
        await collector.refresh_market_set()

        # Post-condition: live market IS now visible to the detector.
        meta = scanner._live_history_provider.market_metadata("0xLIVE")
        assert meta.category == "esports"
    finally:
        await scanner.aclose()
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py::test_collector_refresh_makes_live_market_visible_to_detector -v`

Expected: PASS (Tasks 1-3 are sufficient).

- [ ] **Step 3: Commit**

```bash
git add tests/scheduler/test_gate_model_wiring.py
git commit -m "test(scheduler): live market becomes visible to gate detector after refresh (#102)"
```

---

## Task 5: Document the runtime metadata seeding in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — gate-model bullet under `## Codebase conventions`

- [ ] **Step 1: Append a sentence to the gate-model bullet**

Locate the same gate-model bullet edited in plan #101. Append:

```
`MarketScopedTradeCollector.refresh_market_set` seeds `LiveHistoryProvider` with `MarketMetadata` for every category-matching open market it enumerates (issue #102), so live markets that aren't in `corpus_markets` still resolve at metadata lookup. `closed_at`/`opened_at` default to `0` for live entries — only `category` is load-bearing at inference (`time_to_resolution_seconds` is a leakage column).
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): note runtime metadata seeding by market_scoped_trades (#102)"
```

---

## Verification

After all five tasks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: zero failures, zero warnings.

Smoke verification (per acceptance criterion 5 of issue #102 — requires the desktop training box with a populated corpus + bootstrapped daemon DB):

```bash
# 1. Start daemon with gate_model.enabled=true, gate_model_market_filter.enabled=true,
#    markets.enabled=true, on a fresh daemon DB that has been through bootstrap-features.
# 2. Wait one refresh_market_set + poll_once cycle (~60s default).
# 3. Confirm at least one `alert.emitted detector=gate_buy` row appears in
#    the alerts table for a trade on a market that is NOT in corpus_markets:
#
#      sqlite3 data/pscanner.sqlite3 \
#        "SELECT COUNT(*) FROM alerts WHERE detector='gate_buy';"
#
#    Expected: > 0 within a few minutes (depends on flow; verify against
#    `gate_model.queue_full` warning rate).
```

---

## Self-Review

**Spec coverage:**
- Acceptance criterion 1 (`MarketScopedTradeCollector` accepts optional `provider`) → Task 2.
- Acceptance criterion 2 (`refresh_market_set` populates metadata for every category+volume-passing market, not just the top-N) → Task 2's test asserts this with `0xMB2` (below top-2 cutoff but still seeded).
- Acceptance criterion 3 (`Scanner._build_collectors` wires `provider=self._live_history_provider`) → Task 3.
- Acceptance criterion 4 (unit test: live market post-refresh has `provider.market_metadata(cond_id).category` populated) → Task 2 + Task 4.
- Acceptance criterion 5 (smoke run produces `alert.emitted detector=gate_buy`) → Verification section.

All five acceptance criteria covered. The companion observability gap is correctly deferred to issue #101.

**Placeholder scan:** No TBDs. Every code block is concrete.

**Type consistency:**
- `set_market_metadata(condition_id: str, metadata: MarketMetadata) -> None` — same `str` type used by the existing `market_metadata(condition_id: str)` getter.
- `MarketScopedTradeCollector.__init__(..., provider: LiveHistoryProvider | None = None)` — matches the `LiveHistoryProvider | None = None` shape used by `Scanner._live_history_provider` (line 168).
- `MarketMetadata(condition_id=str, category=str, closed_at=int, opened_at=int)` — matches the dataclass at `pscanner/corpus/features.py:113-120`.
- The `_provider` attribute is private (single-underscore) and only read in tests via attribute access; consistent with how `_callbacks`, `_markets`, `_last_seen_ts` are accessed in the existing collector.
