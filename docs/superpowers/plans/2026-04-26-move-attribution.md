# Move-Attribution Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `MoveAttributionDetector` that hooks off existing alerts (velocity / mispricing / convergence), identifies coordinated trade bursts on the alerted market via adaptive backwalk, and auto-watchlists contributors so the existing `ClusterDetector` can verify them.

**Architecture:** Alert-driven detector subscribed to `AlertSink`. Per triggering alert: one paginated `/trades?market=` fetch → adaptive backwalk to find the burst window → bucket by `(outcome, side, 60s)` and test for ≥4 wallets with size CV ≤ 0.4 → emit `cluster.candidate` alert + upsert contributors into `wallet_watchlist`.

**Tech Stack:** Python 3.13, asyncio, httpx, respx (test mock), pytest, structlog, sqlite3.

**Spec:** `docs/superpowers/specs/2026-04-26-move-attribution-design.md`

**Spec correction discovered during planning:** The spec lists a tweak to add `keep_existing_reason: bool = True` on `WatchlistRepo.upsert`. On reading the existing code, `upsert` already uses `INSERT OR IGNORE` and preserves the original source/reason on conflict — the docstring even states "we keep the first-recorded provenance". So that change is unnecessary; the existing behavior is already what the design wants. The plan drops that task and the related test.

---

## File Structure

**Create:**
- `src/pscanner/detectors/move_attribution.py` — `MoveAttributionDetector` + helpers (`_detect_burst`, `_backwalk`, `BurstHit`)
- `tests/detectors/test_move_attribution.py` — unit + detector tests

**Modify:**
- `src/pscanner/poly/data.py` — add `DataClient.get_market_trades`
- `tests/poly/test_data.py` — test for new method (or new file `tests/poly/test_data_market_trades.py` if test_data.py doesn't exist; check first)
- `src/pscanner/alerts/sink.py` — wrap subscriber fan-out in try/except
- `tests/alerts/test_sink.py` — add isolation test (or new file `tests/alerts/test_sink_isolation.py` if test_sink.py doesn't exist)
- `src/pscanner/alerts/models.py` — add `"move_attribution"` to `DetectorName` Literal
- `src/pscanner/config.py` — new `MoveAttributionConfig` class + wire into `Config`
- `tests/test_config.py` — defaults test (or add to wherever existing config tests live; check first)
- `src/pscanner/scheduler.py` — instantiate detector, register sink subscriber

---

## Task 1: `DataClient.get_market_trades`

Lifts the inline `/trades?market=` paginator from `scripts/expand_cluster.py` into the typed client. Returns raw `list[dict]` (mirrors `get_activity`'s heterogeneous-shape contract).

**Files:**
- Modify: `src/pscanner/poly/data.py` (add method; new constant `_TRADES_PAGE_SIZE`, new `_TRADES_PAGE_CAP`)
- Test: `tests/poly/test_data.py` if it exists; otherwise create `tests/poly/test_data_market_trades.py`

- [ ] **Step 1.1: Find or create the test file**

```bash
ls tests/poly/test_data.py 2>/dev/null && echo "use existing" || echo "create new file"
```

- [ ] **Step 1.2: Write failing tests**

Path: `tests/poly/test_data.py` (or new file). Add:

```python
import respx
import httpx
import pytest

from pscanner.poly.data import DataClient
from pscanner.poly.http import PolyHttpClient

_DATA = "https://data-api.polymarket.com"


@pytest.fixture
def data_client() -> DataClient:
    return DataClient(http=PolyHttpClient(base_url=_DATA, rpm=600, timeout_seconds=5.0))


@respx.mock
async def test_get_market_trades_filters_by_window(data_client: DataClient) -> None:
    page = [
        {"proxyWallet": "0xa", "timestamp": 1500, "size": 10.0, "price": 0.5,
         "side": "BUY", "outcome": "Yes"},
        {"proxyWallet": "0xb", "timestamp": 1200, "size": 20.0, "price": 0.5,
         "side": "BUY", "outcome": "Yes"},
        {"proxyWallet": "0xc", "timestamp":  900, "size": 30.0, "price": 0.5,
         "side": "BUY", "outcome": "Yes"},
    ]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    out = await data_client.get_market_trades(
        condition_id="0xabc", since_ts=1000, until_ts=1600,
    )
    assert {t["proxyWallet"] for t in out} == {"0xa", "0xb"}


@respx.mock
async def test_get_market_trades_paginates_until_below_window(data_client: DataClient) -> None:
    # First page: all in-window; second page: drops below since_ts -> stop
    page1 = [{"proxyWallet": f"0x{i}", "timestamp": 2000 - i, "size": 1.0,
              "price": 0.5, "side": "BUY", "outcome": "Yes"} for i in range(500)]
    page2 = [{"proxyWallet": "0x_old", "timestamp": 100, "size": 1.0,
              "price": 0.5, "side": "BUY", "outcome": "Yes"}]
    route = respx.get(f"{_DATA}/trades").mock(side_effect=[
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ])
    out = await data_client.get_market_trades(
        condition_id="0xabc", since_ts=1000, until_ts=3000,
    )
    assert route.call_count == 2
    assert len(out) == 500
    assert "0x_old" not in {t["proxyWallet"] for t in out}


@respx.mock
async def test_get_market_trades_stops_at_page_cap(data_client: DataClient) -> None:
    # Every page returns 500 and is in-window — must stop at _TRADES_PAGE_CAP
    full_page = [{"proxyWallet": f"0x{i}", "timestamp": 2_000_000_000, "size": 1.0,
                  "price": 0.5, "side": "BUY", "outcome": "Yes"} for i in range(500)]
    route = respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=full_page))
    out = await data_client.get_market_trades(
        condition_id="0xabc", since_ts=0, until_ts=3_000_000_000,
    )
    assert route.call_count == 30  # _TRADES_PAGE_CAP
    assert len(out) == 30 * 500
```

Then close the client: add `await data_client.aclose()` in a `pytest_asyncio.fixture` style (check the pattern in existing `test_data*.py` first if it exists).

- [ ] **Step 1.3: Run tests, verify they fail**

```bash
uv run pytest tests/poly/test_data.py::test_get_market_trades_filters_by_window -v
```

Expected: FAIL with `AttributeError: 'DataClient' object has no attribute 'get_market_trades'`.

- [ ] **Step 1.4: Implement `get_market_trades`**

In `src/pscanner/poly/data.py`, add module constants near `_ACTIVITY_PAGE_SIZE`:

```python
_TRADES_PAGE_SIZE: Final[int] = 500
_TRADES_PAGE_CAP: Final[int] = 30  # 15k trades per condition_id maximum
```

Add the method on `DataClient`:

```python
async def get_market_trades(
    self,
    condition_id: str,
    *,
    since_ts: int,
    until_ts: int,
) -> list[dict[str, Any]]:
    """Return all CONFIRMED trades on a market within ``[since_ts, until_ts]``.

    Paginates ``/trades?market=`` newest-first. Stops as soon as the newest
    timestamp on a page is older than ``since_ts`` or a short page is
    returned. Hard-capped at ``_TRADES_PAGE_CAP`` pages (15k trades) so a
    runaway market cannot exhaust the rate budget on a single call.

    Args:
        condition_id: 0x-prefixed market condition_id.
        since_ts: Inclusive lower bound on trade ``timestamp`` (unix seconds).
        until_ts: Inclusive upper bound on trade ``timestamp`` (unix seconds).

    Returns:
        A list of raw JSON trade dicts whose timestamps fall inside the window.
        Heterogeneous shape — callers re-parse downstream.
    """
    out: list[dict[str, Any]] = []
    offset = 0
    for _ in range(_TRADES_PAGE_CAP):
        params: dict[str, Any] = {
            "market": condition_id,
            "limit": _TRADES_PAGE_SIZE,
            "offset": offset,
        }
        payload = await self._data_http.get("/trades", params=params)
        page = _ensure_list(payload, endpoint="/trades")
        if not page:
            break
        page_max_ts = max((t.get("timestamp", 0) for t in page if isinstance(t, dict)), default=0)
        for item in page:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp")
            if isinstance(ts, int) and since_ts <= ts <= until_ts:
                out.append(item)
        if page_max_ts < since_ts or len(page) < _TRADES_PAGE_SIZE:
            break
        offset += _TRADES_PAGE_SIZE
    return out
```

- [ ] **Step 1.5: Run tests, verify they pass**

```bash
uv run pytest tests/poly/test_data.py -v -k get_market_trades
```

Expected: PASS for all three tests.

- [ ] **Step 1.6: Lint, format, type-check**

```bash
uv run ruff check src/pscanner/poly/data.py tests/poly/test_data.py \
  && uv run ruff format src/pscanner/poly/data.py tests/poly/test_data.py \
  && uv run ty check src/pscanner/poly/data.py
```

Expected: all checks pass.

- [ ] **Step 1.7: Commit**

```bash
git add src/pscanner/poly/data.py tests/poly/test_data.py
git commit -m "feat(poly): DataClient.get_market_trades for /trades?market= pagination

Lifts the inline paginator from scripts/expand_cluster.py into the typed
client. Used by the upcoming MoveAttributionDetector and (next iteration)
by expand_cluster.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `AlertSink.emit` per-handler isolation

A subscriber callback that raises must not break the alert hot path or other subscribers. Wrap each callback in try/except.

**Files:**
- Modify: `src/pscanner/alerts/sink.py:55-56` (the `for callback in self._subscribers` loop)
- Test: `tests/alerts/test_sink.py` (or new file if absent)

- [ ] **Step 2.1: Find or create the test file**

```bash
ls tests/alerts/test_sink.py 2>/dev/null && echo "use existing" || echo "create new"
```

- [ ] **Step 2.2: Write failing test**

```python
# In tests/alerts/test_sink.py
import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.store.repo import AlertsRepo


def _alert(key: str = "k1") -> Alert:
    return Alert(
        detector="velocity",
        alert_key=key,
        severity="med",
        title="t",
        body={},
        created_at=1700000000,
    )


@pytest.mark.asyncio
async def test_subscriber_exception_does_not_break_emit(tmp_db) -> None:
    sink = AlertSink(AlertsRepo(tmp_db))
    received: list[Alert] = []

    def boom(_a: Alert) -> None:
        raise RuntimeError("subscriber failed")

    def good(a: Alert) -> None:
        received.append(a)

    sink.subscribe(boom)
    sink.subscribe(good)
    inserted = await sink.emit(_alert())
    assert inserted is True
    assert len(received) == 1, "the second subscriber must still fire after the first raises"
```

- [ ] **Step 2.3: Run, verify it fails**

```bash
uv run pytest tests/alerts/test_sink.py::test_subscriber_exception_does_not_break_emit -v
```

Expected: FAIL — the `RuntimeError` propagates out of `emit`.

- [ ] **Step 2.4: Implement isolation**

Replace the block at `src/pscanner/alerts/sink.py:55-56`:

```python
        for callback in self._subscribers:
            callback(alert)
```

with:

```python
        for callback in self._subscribers:
            try:
                callback(alert)
            except Exception:
                _log.warning(
                    "alert.subscriber_failed",
                    alert_key=alert.alert_key,
                    subscriber=getattr(callback, "__qualname__", repr(callback)),
                    exc_info=True,
                )
```

- [ ] **Step 2.5: Run, verify it passes**

```bash
uv run pytest tests/alerts/test_sink.py -v
```

Expected: PASS.

- [ ] **Step 2.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/alerts/sink.py tests/alerts/test_sink.py \
  && uv run ruff format src/pscanner/alerts/sink.py tests/alerts/test_sink.py \
  && uv run ty check src/pscanner/alerts/sink.py
```

- [ ] **Step 2.7: Commit**

```bash
git add src/pscanner/alerts/sink.py tests/alerts/test_sink.py
git commit -m "feat(alerts): isolate AlertSink subscribers — one bad callback can't poison others

A subscriber that raises now logs alert.subscriber_failed at WARN; other
subscribers and the alerts row write are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `"move_attribution"` to `DetectorName` Literal

Without this, the renderer's `_buffers[alert.detector]` lookup will KeyError when the new detector emits.

**Files:**
- Modify: `src/pscanner/alerts/models.py:9`
- Test: `tests/alerts/test_terminal.py` (or new file if absent)

- [ ] **Step 3.1: Write failing test**

```python
# In tests/alerts/test_terminal.py
from pscanner.alerts.models import Alert, DetectorName
from pscanner.alerts.terminal import TerminalRenderer
from typing import get_args


def test_move_attribution_in_detector_literal() -> None:
    assert "move_attribution" in get_args(DetectorName)


def test_renderer_handles_move_attribution_alert() -> None:
    renderer = TerminalRenderer(max_per_detector=5)
    alert = Alert(
        detector="move_attribution",  # type: ignore[arg-type]
        alert_key="cluster.candidate:0xabc:Yes:BUY:1700000000",
        severity="med",
        title="cluster candidate burst",
        body={},
        created_at=1700000000,
    )
    renderer.push(alert)  # must not KeyError
```

- [ ] **Step 3.2: Run, verify it fails**

```bash
uv run pytest tests/alerts/test_terminal.py -v
```

Expected: FAIL — `"move_attribution" not in get_args(...)`.

- [ ] **Step 3.3: Implement the Literal change**

In `src/pscanner/alerts/models.py:9`, change:

```python
DetectorName = Literal["smart_money", "mispricing", "whales", "convergence", "velocity", "cluster"]
```

to:

```python
DetectorName = Literal[
    "smart_money", "mispricing", "whales", "convergence",
    "velocity", "cluster", "move_attribution",
]
```

- [ ] **Step 3.4: Run, verify it passes**

```bash
uv run pytest tests/alerts/test_terminal.py -v
```

Expected: PASS.

- [ ] **Step 3.5: Run full test suite — ty-check the Literal change has no fallout**

```bash
uv run ruff check src/pscanner/alerts/models.py tests/alerts/test_terminal.py \
  && uv run ruff format src/pscanner/alerts/models.py tests/alerts/test_terminal.py \
  && uv run ty check src/pscanner
```

Expected: PASS.

- [ ] **Step 3.6: Commit**

```bash
git add src/pscanner/alerts/models.py tests/alerts/test_terminal.py
git commit -m "feat(alerts): add 'move_attribution' to DetectorName Literal

Required so the new detector's alerts don't KeyError in the renderer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `MoveAttributionConfig`

New config section + wired into root `Config`. All defaults from the spec.

**Files:**
- Modify: `src/pscanner/config.py` (add `MoveAttributionConfig` class near `ClusterConfig`; add `move_attribution` field on `Config`)
- Test: existing config test file — locate first

- [ ] **Step 4.1: Find existing config tests**

```bash
ls tests/test_config.py 2>/dev/null || ls tests/ | grep -i config
```

- [ ] **Step 4.2: Write failing test**

In the existing config test file (or a new `tests/test_config_move_attribution.py`):

```python
from pscanner.config import Config, MoveAttributionConfig


def test_move_attribution_defaults() -> None:
    cfg = MoveAttributionConfig()
    assert cfg.enabled is True
    assert cfg.trigger_detectors == ("velocity", "mispricing", "convergence")
    assert cfg.lookback_seconds_baseline == 86400
    assert cfg.backwalk_multiplier == 3.0
    assert cfg.backwalk_check_window_seconds == 300
    assert cfg.max_backwalk_seconds == 7200
    assert cfg.burst_bucket_seconds == 60
    assert cfg.min_burst_wallets == 4
    assert cfg.max_burst_size_cv == 0.4
    assert cfg.max_burst_hits_per_alert == 5
    assert cfg.max_contributors_per_burst == 50


def test_move_attribution_attached_to_root_config() -> None:
    cfg = Config()
    assert isinstance(cfg.move_attribution, MoveAttributionConfig)
    assert cfg.move_attribution.enabled is True
```

- [ ] **Step 4.3: Run, verify it fails**

```bash
uv run pytest -k move_attribution_defaults -v
```

Expected: FAIL — `ImportError: cannot import name 'MoveAttributionConfig'`.

- [ ] **Step 4.4: Implement config**

In `src/pscanner/config.py`, immediately after the `ClusterConfig` class, add:

```python
class MoveAttributionConfig(_Section):
    """Thresholds + cadence for the alert-driven move-attribution detector.

    Hooks off existing alerts (``trigger_detectors``) and, when one fires,
    fetches recent trades on the alerted market, walks back to find the
    burst window, tests for a coordinated burst (≥``min_burst_wallets``
    distinct wallets in one ``(outcome, side, burst_bucket_seconds)`` bucket
    with size CV ≤ ``max_burst_size_cv``), and emits ``cluster.candidate``
    plus auto-watchlists the contributors.
    """

    enabled: bool = True
    trigger_detectors: tuple[str, ...] = ("velocity", "mispricing", "convergence")
    # Backwalk window
    lookback_seconds_baseline: int = 86400
    backwalk_multiplier: float = 3.0
    backwalk_check_window_seconds: int = 300
    max_backwalk_seconds: int = 7200
    # Burst test
    burst_bucket_seconds: int = 60
    min_burst_wallets: int = 4
    max_burst_size_cv: float = 0.4
    # Safety caps
    max_burst_hits_per_alert: int = 5
    max_contributors_per_burst: int = 50
```

In the same file, on the `Config` class (around line 266 where `cluster: ClusterConfig` is declared), add:

```python
    move_attribution: MoveAttributionConfig = Field(default_factory=MoveAttributionConfig)
```

- [ ] **Step 4.5: Run, verify it passes**

```bash
uv run pytest -k move_attribution -v
```

Expected: PASS.

- [ ] **Step 4.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/config.py \
  && uv run ruff format src/pscanner/config.py \
  && uv run ty check src/pscanner/config.py
```

- [ ] **Step 4.7: Commit**

```bash
git add src/pscanner/config.py tests/
git commit -m "feat(config): MoveAttributionConfig with spec defaults

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `_detect_burst` pure helper

Pure function over `list[dict]` (raw `/trades` rows). Buckets `(outcome, side, ts // bucket_seconds)`. Returns `list[BurstHit]` for buckets meeting the threshold.

**Files:**
- Create: `src/pscanner/detectors/move_attribution.py` (initial structure: imports, `BurstHit`, `_detect_burst`)
- Create: `tests/detectors/test_move_attribution.py`

- [ ] **Step 5.1: Write failing tests for `_detect_burst`**

Create `tests/detectors/test_move_attribution.py`:

```python
"""Tests for MoveAttributionDetector and its pure helpers."""
from __future__ import annotations

from pscanner.config import MoveAttributionConfig
from pscanner.detectors.move_attribution import BurstHit, _detect_burst


def _trade(
    *,
    wallet: str,
    ts: int,
    side: str = "BUY",
    outcome: str = "Yes",
    size: float = 100.0,
    price: float = 0.5,
) -> dict:
    return {
        "proxyWallet": wallet, "timestamp": ts, "side": side,
        "outcome": outcome, "size": size, "price": price,
    }


def _cfg(**overrides) -> MoveAttributionConfig:
    return MoveAttributionConfig(**overrides)


def test_detect_burst_happy_path_fires() -> None:
    trades = [
        _trade(wallet=f"0x{i}", ts=1000, size=500.0 + i)  # CV ~0
        for i in range(4)
    ]
    hits = _detect_burst(trades, cfg=_cfg())
    assert len(hits) == 1
    h = hits[0]
    assert h.outcome == "Yes"
    assert h.side == "BUY"
    assert set(h.wallets) == {"0x0", "0x1", "0x2", "0x3"}


def test_detect_burst_below_wallet_threshold_no_hit() -> None:
    trades = [_trade(wallet=f"0x{i}", ts=1000) for i in range(3)]
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_high_cv_no_hit() -> None:
    sizes = [10.0, 100.0, 1000.0, 5000.0]  # CV >> 0.4
    trades = [_trade(wallet=f"0x{i}", ts=1000, size=s) for i, s in enumerate(sizes)]
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_mixed_sides_split_buckets_no_hit() -> None:
    trades = [
        _trade(wallet="0x1", ts=1000, side="BUY"),
        _trade(wallet="0x2", ts=1000, side="BUY"),
        _trade(wallet="0x3", ts=1000, side="SELL"),
        _trade(wallet="0x4", ts=1000, side="SELL"),
    ]
    # 2 BUY + 2 SELL — neither side hits min_burst_wallets=4
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_cross_bucket_no_hit() -> None:
    # 4 wallets but split across two 60s buckets (2 + 2)
    trades = [
        _trade(wallet="0x1", ts=1000),
        _trade(wallet="0x2", ts=1010),
        _trade(wallet="0x3", ts=1080),  # next bucket
        _trade(wallet="0x4", ts=1090),
    ]
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_truncates_at_max_burst_hits() -> None:
    # 8 distinct buckets each with 4 wallets at uniform size
    trades: list[dict] = []
    for bucket in range(8):
        ts = 1000 + bucket * 60
        trades.extend(
            _trade(wallet=f"0x{bucket}-{w}", ts=ts) for w in range(4)
        )
    hits = _detect_burst(trades, cfg=_cfg(max_burst_hits_per_alert=5))
    assert len(hits) == 5  # truncated


def test_detect_burst_truncates_contributors_to_top_50() -> None:
    # 80 wallets in one bucket, sizes far from median except 50 closest
    sizes = list(range(1, 81))  # 1..80
    trades = [
        _trade(wallet=f"0x{i}", ts=1000, size=float(s))
        for i, s in enumerate(sizes)
    ]
    hits = _detect_burst(
        trades,
        cfg=_cfg(min_burst_wallets=4, max_burst_size_cv=2.0,  # allow it to fire
                 max_contributors_per_burst=50),
    )
    assert len(hits) == 1
    assert len(hits[0].wallets) == 50
```

- [ ] **Step 5.2: Run, verify ImportError fails it**

```bash
uv run pytest tests/detectors/test_move_attribution.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 5.3: Implement `_detect_burst` and `BurstHit`**

Create `src/pscanner/detectors/move_attribution.py`:

```python
"""Move-attribution detector — bootstraps cluster candidates from market moves.

Subscribes to ``AlertSink``. When an upstream alert (velocity / mispricing /
convergence) names a market, fetches recent trades on that market, walks
back to the start of the burst, tests for a coordinated burst, and emits
``cluster.candidate`` plus upserts the contributors into ``wallet_watchlist``
so the existing :class:`ClusterDetector` can verify them on its next sweep.
"""
from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

from pscanner.config import MoveAttributionConfig

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BurstHit:
    """One coordinated-burst hit on a single ``(outcome, side, bucket)``."""

    outcome: str
    side: str
    bucket_ts: int
    wallets: tuple[str, ...]
    n_trades: int
    median_size: float
    cv: float


def _detect_burst(
    trades: Iterable[dict[str, Any]],
    *,
    cfg: MoveAttributionConfig,
) -> list[BurstHit]:
    """Return one ``BurstHit`` per bucket meeting the threshold.

    Buckets ``(outcome, side, ts // burst_bucket_seconds)``. A bucket fires
    when distinct-wallet count ≥ ``min_burst_wallets`` and trade-size CV
    (``pstdev / mean``) ≤ ``max_burst_size_cv``. Up to
    ``max_burst_hits_per_alert`` hits returned (sorted by wallet count desc).
    Each hit's contributor list is truncated to
    ``max_contributors_per_burst`` wallets closest to the bucket's median size.
    """
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for t in trades:
        outcome = t.get("outcome")
        side = t.get("side")
        ts = t.get("timestamp")
        if not isinstance(outcome, str) or not isinstance(side, str):
            continue
        if not isinstance(ts, int):
            continue
        bucket_ts = (ts // cfg.burst_bucket_seconds) * cfg.burst_bucket_seconds
        buckets.setdefault((outcome, side, bucket_ts), []).append(t)

    hits: list[BurstHit] = []
    for (outcome, side, bucket_ts), bucket_trades in buckets.items():
        wallet_to_trade: dict[str, dict[str, Any]] = {}
        for t in bucket_trades:
            wallet = t.get("proxyWallet")
            if isinstance(wallet, str) and wallet not in wallet_to_trade:
                wallet_to_trade[wallet] = t
        if len(wallet_to_trade) < cfg.min_burst_wallets:
            continue
        sizes = [
            float(t.get("size") or 0.0) for t in wallet_to_trade.values()
        ]
        positive_sizes = [s for s in sizes if s > 0]
        if len(positive_sizes) < cfg.min_burst_wallets:
            continue
        mean = statistics.fmean(positive_sizes)
        if mean <= 0:
            continue
        stdev = statistics.pstdev(positive_sizes)
        cv = stdev / mean
        if cv > cfg.max_burst_size_cv:
            continue
        median_size = statistics.median(positive_sizes)
        # Contributor truncation: keep wallets closest to median size
        ranked = sorted(
            wallet_to_trade.items(),
            key=lambda kv: abs(float(kv[1].get("size") or 0.0) - median_size),
        )
        kept = [w for w, _t in ranked[: cfg.max_contributors_per_burst]]
        if len(wallet_to_trade) > cfg.max_contributors_per_burst:
            _LOG.warning(
                "move_attribution.contributors_truncated",
                bucket_ts=bucket_ts, outcome=outcome, side=side,
                n_total=len(wallet_to_trade),
                n_kept=cfg.max_contributors_per_burst,
            )
        hits.append(
            BurstHit(
                outcome=outcome,
                side=side,
                bucket_ts=bucket_ts,
                wallets=tuple(sorted(kept)),
                n_trades=len(bucket_trades),
                median_size=median_size,
                cv=cv,
            )
        )
    hits.sort(key=lambda h: len(h.wallets), reverse=True)
    if len(hits) > cfg.max_burst_hits_per_alert:
        _LOG.warning(
            "move_attribution.hits_truncated",
            n_total=len(hits),
            n_kept=cfg.max_burst_hits_per_alert,
        )
        hits = hits[: cfg.max_burst_hits_per_alert]
    return hits


__all__ = ["BurstHit", "_detect_burst"]
```

- [ ] **Step 5.4: Run, verify all 7 unit tests pass**

```bash
uv run pytest tests/detectors/test_move_attribution.py -v
```

Expected: PASS for all `_detect_burst` tests.

- [ ] **Step 5.5: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py \
  && uv run ruff format src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py \
  && uv run ty check src/pscanner/detectors/move_attribution.py
```

- [ ] **Step 5.6: Commit**

```bash
git add src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py
git commit -m "feat(detectors): _detect_burst pure helper for move-attribution

Buckets trades by (outcome, side, ts//bucket_seconds), fires when wallet
count >= min_burst_wallets and trade-size CV <= max_burst_size_cv.
Truncates per-burst contributor list to top-N closest to bucket median.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `_backwalk` helper

Reads 24h of trades, computes baseline rate, walks back from `alert_ts` until the trailing-window rate drops below baseline × multiplier for two consecutive windows. Returns `(since_ts, until_ts, burst_trades)`.

**Files:**
- Modify: `src/pscanner/detectors/move_attribution.py` (add `_backwalk`)
- Modify: `tests/detectors/test_move_attribution.py` (append tests)

- [ ] **Step 6.1: Write failing tests**

Append to `tests/detectors/test_move_attribution.py`:

```python
import pytest
import respx
import httpx

from pscanner.detectors.move_attribution import _backwalk
from pscanner.poly.data import DataClient
from pscanner.poly.http import PolyHttpClient

_DATA = "https://data-api.polymarket.com"


def _backwalk_client() -> DataClient:
    return DataClient(http=PolyHttpClient(base_url=_DATA, rpm=600, timeout_seconds=5.0))


def _gen_trades(*, ts_start: int, ts_end: int, gap: int) -> list[dict]:
    """Generate trades with uniform gap, newest-first."""
    out: list[dict] = []
    ts = ts_end
    i = 0
    while ts >= ts_start:
        out.append({
            "proxyWallet": f"0x{i:04d}", "timestamp": ts,
            "side": "BUY", "outcome": "Yes", "size": 50.0, "price": 0.5,
        })
        ts -= gap
        i += 1
    return out


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_stops_on_quiescence() -> None:
    # Baseline: 1 trade per 60s for 24h prior
    # Burst: 1 trade per 5s for 30 min ending at alert_ts
    alert_ts = 1_700_086_400
    baseline_start = alert_ts - 86_400
    baseline = _gen_trades(ts_start=baseline_start, ts_end=alert_ts - 1800, gap=60)
    burst = _gen_trades(ts_start=alert_ts - 1800, ts_end=alert_ts, gap=5)
    page = sorted(burst + baseline, key=lambda t: -t["timestamp"])[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    client = _backwalk_client()
    try:
        since_ts, until_ts, burst_trades = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg(),
        )
    finally:
        await client.aclose()
    assert until_ts == alert_ts
    # Backwalk should stop within ~30 min of the alert (the burst window)
    assert alert_ts - 7200 < since_ts < alert_ts - 1500
    assert all(since_ts <= t["timestamp"] <= until_ts for t in burst_trades)


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_caps_at_max_backwalk_seconds() -> None:
    # Constant-rate market: burst rate never drops, must hit the 7200s cap
    alert_ts = 1_700_000_000
    page = _gen_trades(ts_start=alert_ts - 7300, ts_end=alert_ts, gap=5)[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    client = _backwalk_client()
    try:
        since_ts, until_ts, _ = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg(),
        )
    finally:
        await client.aclose()
    assert until_ts - since_ts == 7200


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_requires_two_consecutive_quiescent_windows() -> None:
    # Pattern: burst, single 5-min lull, burst again, then quiescence.
    # Single dip must NOT stop the backwalk.
    alert_ts = 1_700_086_400
    parts: list[dict] = []
    parts += _gen_trades(ts_start=alert_ts - 600, ts_end=alert_ts, gap=5)         # burst 0..600s back
    parts += _gen_trades(ts_start=alert_ts - 900, ts_end=alert_ts - 600, gap=180) # 5 min low rate
    parts += _gen_trades(ts_start=alert_ts - 1800, ts_end=alert_ts - 900, gap=5)  # burst again
    parts += _gen_trades(ts_start=alert_ts - 86400, ts_end=alert_ts - 1800, gap=60)
    page = sorted(parts, key=lambda t: -t["timestamp"])[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    client = _backwalk_client()
    try:
        since_ts, _, _ = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg(),
        )
    finally:
        await client.aclose()
    # Window must extend back past the single dip (>= 1800s)
    assert alert_ts - since_ts >= 1800


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_empty_trades_returns_full_cap() -> None:
    alert_ts = 1_700_000_000
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=[]))
    client = _backwalk_client()
    try:
        since_ts, until_ts, burst = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg(),
        )
    finally:
        await client.aclose()
    assert (since_ts, until_ts) == (alert_ts - 7200, alert_ts)
    assert burst == []
```

- [ ] **Step 6.2: Run, verify ImportError**

```bash
uv run pytest tests/detectors/test_move_attribution.py -v -k backwalk
```

Expected: FAIL — `ImportError`.

- [ ] **Step 6.3: Implement `_backwalk`**

Append to `src/pscanner/detectors/move_attribution.py`:

```python
from pscanner.poly.data import DataClient


async def _backwalk(
    client: DataClient,
    *,
    condition_id: str,
    alert_ts: int,
    cfg: MoveAttributionConfig,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Walk back from ``alert_ts`` to the start of the coordinated burst.

    1. Fetch ``cfg.lookback_seconds_baseline`` (default 24h) of trades.
    2. Compute ``baseline_rate`` (trades per minute) over the full window.
    3. Walk back in ``cfg.backwalk_check_window_seconds`` (default 300s)
       steps; at each step compute the trailing-window trade rate. Stop
       when rate < ``baseline_rate × cfg.backwalk_multiplier`` for two
       consecutive windows.
    4. Hard cap at ``cfg.max_backwalk_seconds``.

    Returns ``(since_ts, until_ts, burst_trades)`` where ``burst_trades`` is
    the slice of the fetched 24h list with ``since_ts ≤ ts ≤ until_ts``.
    """
    until_ts = alert_ts
    floor_ts = alert_ts - cfg.max_backwalk_seconds
    baseline_start = alert_ts - cfg.lookback_seconds_baseline
    all_trades = await client.get_market_trades(
        condition_id, since_ts=baseline_start, until_ts=alert_ts,
    )
    if not all_trades:
        return floor_ts, until_ts, []
    # baseline rate (trades per minute over the full lookback)
    baseline_minutes = max(cfg.lookback_seconds_baseline / 60.0, 1.0)
    baseline_rate = len(all_trades) / baseline_minutes
    threshold_per_window = (
        (baseline_rate * cfg.backwalk_multiplier)
        * (cfg.backwalk_check_window_seconds / 60.0)
    )
    # Sort newest-first into a list of timestamps for quick window counting
    timestamps = sorted(
        (int(t["timestamp"]) for t in all_trades if isinstance(t.get("timestamp"), int)),
        reverse=True,
    )
    since_ts = floor_ts
    consecutive_quiescent = 0
    cursor = alert_ts
    while cursor > floor_ts:
        window_lo = max(cursor - cfg.backwalk_check_window_seconds, floor_ts)
        in_window = sum(1 for t in timestamps if window_lo <= t < cursor)
        if in_window < threshold_per_window:
            consecutive_quiescent += 1
            if consecutive_quiescent >= 2:
                since_ts = cursor  # use the start of the second quiescent window
                break
        else:
            consecutive_quiescent = 0
        cursor = window_lo
        if cursor <= floor_ts:
            since_ts = floor_ts
            break
    burst_trades = [
        t for t in all_trades
        if isinstance(t.get("timestamp"), int) and since_ts <= t["timestamp"] <= until_ts
    ]
    return since_ts, until_ts, burst_trades
```

Add `_backwalk` to `__all__` at the bottom:

```python
__all__ = ["BurstHit", "_backwalk", "_detect_burst"]
```

- [ ] **Step 6.4: Run, verify the 4 backwalk tests pass**

```bash
uv run pytest tests/detectors/test_move_attribution.py -v -k backwalk
```

Expected: PASS for all four backwalk tests.

- [ ] **Step 6.5: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py \
  && uv run ruff format src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py \
  && uv run ty check src/pscanner/detectors/move_attribution.py
```

- [ ] **Step 6.6: Commit**

```bash
git add src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py
git commit -m "feat(detectors): _backwalk helper for adaptive burst-window selection

Reads 24h of trades, computes baseline trade rate, walks back from alert_ts
in 5-min steps until the trailing rate drops below baseline*multiplier for
two consecutive windows. Hard capped at max_backwalk_seconds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `MoveAttributionDetector` class

Wire the helpers and the alert-callback plumbing. Detector subscribes to `AlertSink`, ignores alerts whose `detector` field is not in the trigger set, and on a match runs the full pipeline asynchronously.

**Files:**
- Modify: `src/pscanner/detectors/move_attribution.py` (add the class)
- Modify: `tests/detectors/test_move_attribution.py` (append detector-level tests)

- [ ] **Step 7.1: Write failing detector-level tests**

Append to `tests/detectors/test_move_attribution.py`:

```python
import asyncio

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.detectors.move_attribution import MoveAttributionDetector
from pscanner.store.repo import AlertsRepo, WatchlistRepo


def _build_velocity_alert(
    *,
    condition_id: str = "0xabc",
    alert_key: str = "velocity:0xabc:1",
    created_at: int = 1_700_086_400,
) -> Alert:
    return Alert(
        detector="velocity",
        alert_key=alert_key,
        severity="med",
        title="market moved",
        body={"condition_id": condition_id, "delta_price": 0.07},
        created_at=created_at,
    )


@respx.mock
@pytest.mark.asyncio
async def test_detector_emits_candidate_and_watchlists_contributors(tmp_db) -> None:
    alert_ts = 1_700_086_400
    burst = [
        {"proxyWallet": f"0x{i:04d}", "timestamp": alert_ts - 30,
         "side": "BUY", "outcome": "Yes", "size": 500.0 + i, "price": 0.95}
        for i in range(6)
    ]
    baseline = _gen_trades(ts_start=alert_ts - 86400, ts_end=alert_ts - 1800, gap=60)
    page = sorted(burst + baseline, key=lambda t: -t["timestamp"])[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))

    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        await sink.emit(_build_velocity_alert(created_at=alert_ts))
        # Allow the spawned task to complete
        for _ in range(5):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    rendered = AlertsRepo(tmp_db).list_recent(limit=10)
    candidate_alerts = [a for a in rendered if a.detector == "move_attribution"]
    assert len(candidate_alerts) == 1
    assert candidate_alerts[0].alert_key.startswith("cluster.candidate:")
    watchlisted = [w.address for w in watchlist.list_active()]
    assert sum(1 for a in watchlisted if a.startswith("0x")) >= 6


@pytest.mark.asyncio
async def test_detector_ignores_non_trigger_detectors(tmp_db) -> None:
    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        # whales is NOT in default trigger_detectors
        a = Alert(
            detector="whales", alert_key="whales:1", severity="med",
            title="whale", body={"condition_id": "0xabc"}, created_at=1700000000,
        )
        await sink.emit(a)
        for _ in range(5):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    assert watchlist.list_active() == []


@pytest.mark.asyncio
async def test_detector_skips_alert_without_condition_id(tmp_db) -> None:
    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        a = Alert(
            detector="velocity", alert_key="velocity:nocond", severity="med",
            title="event-level", body={"event_id": "evt-1"}, created_at=1700000000,
        )
        await sink.emit(a)
        for _ in range(5):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    assert watchlist.list_active() == []


@respx.mock
@pytest.mark.asyncio
async def test_detector_swallows_trades_http_error(tmp_db) -> None:
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(500))
    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        # Must not raise; alert path must complete cleanly
        await sink.emit(_build_velocity_alert())
        for _ in range(5):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    assert watchlist.list_active() == []
```

- [ ] **Step 7.2: Run, verify ImportError fails**

```bash
uv run pytest tests/detectors/test_move_attribution.py -v -k detector
```

Expected: FAIL — `MoveAttributionDetector` not defined.

- [ ] **Step 7.3: Implement the class**

Append to `src/pscanner/detectors/move_attribution.py`:

```python
import asyncio

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.store.repo import WatchlistRepo


class MoveAttributionDetector:
    """Alert-driven detector that bootstraps cluster candidates."""

    name = "move_attribution"

    def __init__(
        self,
        *,
        config: MoveAttributionConfig,
        data_client: DataClient,
        watchlist_repo: WatchlistRepo,
    ) -> None:
        """Bind helpers and the watchlist write target.

        Args:
            config: Section of the root config controlling thresholds.
            data_client: Used for the single ``/trades?market=`` paginated
                fetch on the hot path.
            watchlist_repo: Used to upsert each contributor wallet.
        """
        self._config = config
        self._data_client = data_client
        self._watchlist_repo = watchlist_repo
        self._sink: AlertSink | None = None
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def run(self, sink: AlertSink) -> None:
        """Park forever — this detector is alert-driven, not periodic."""
        self._sink = sink
        await asyncio.Event().wait()

    def handle_alert_sync(self, alert: Alert) -> None:
        """Subscriber callback fanned out by ``AlertSink.emit``.

        Spawns ``evaluate(alert)`` as a tracked task so it isn't garbage
        collected mid-flight. No-ops if there is no running event loop.
        """
        if alert.detector not in self._config.trigger_detectors:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("move_attribution.no_event_loop", alert_key=alert.alert_key)
            return
        task = loop.create_task(self.evaluate(alert))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate(self, alert: Alert) -> None:
        """Run the full pipeline for one triggering alert."""
        condition_id = alert.body.get("condition_id") if isinstance(alert.body, dict) else None
        if not isinstance(condition_id, str):
            _LOG.debug("move_attribution.no_market", alert_key=alert.alert_key)
            return
        try:
            since_ts, until_ts, burst_trades = await _backwalk(
                self._data_client,
                condition_id=condition_id,
                alert_ts=alert.created_at,
                cfg=self._config,
            )
        except Exception:
            _LOG.warning(
                "move_attribution.fetch_failed",
                alert_key=alert.alert_key,
                condition_id=condition_id,
                exc_info=True,
            )
            return
        hits = _detect_burst(burst_trades, cfg=self._config)
        if not hits:
            return
        await self._emit_and_watchlist(alert, condition_id, hits)

    async def _emit_and_watchlist(
        self,
        triggering: Alert,
        condition_id: str,
        hits: list[BurstHit],
    ) -> None:
        """Emit one ``cluster.candidate`` per hit, watchlist contributors."""
        sink = self._sink
        if sink is None:
            _LOG.warning("move_attribution.no_sink", alert_key=triggering.alert_key)
            return
        reason = f"cluster.candidate-{triggering.alert_key}"
        for hit in hits:
            alert_key = (
                f"cluster.candidate:{condition_id}:{hit.outcome}:{hit.side}:{hit.bucket_ts}"
            )
            await sink.emit(
                Alert(
                    detector="move_attribution",
                    alert_key=alert_key,
                    severity="med",
                    title=(
                        f"cluster candidate: {len(hit.wallets)} wallets, "
                        f"{hit.outcome} {hit.side} burst"
                    ),
                    body={
                        "condition_id": condition_id,
                        "outcome": hit.outcome,
                        "side": hit.side,
                        "bucket_ts": hit.bucket_ts,
                        "n_wallets": len(hit.wallets),
                        "n_trades": hit.n_trades,
                        "median_size": hit.median_size,
                        "cv": hit.cv,
                        "triggering_alert_key": triggering.alert_key,
                    },
                    created_at=triggering.created_at,
                )
            )
            for wallet in hit.wallets:
                try:
                    self._watchlist_repo.upsert(
                        address=wallet,
                        source="cluster.candidate",
                        reason=reason,
                    )
                except Exception:
                    _LOG.warning(
                        "move_attribution.watchlist_upsert_failed",
                        alert_key=triggering.alert_key,
                        address=wallet,
                        exc_info=True,
                    )

    async def aclose(self) -> None:
        """Wait for any in-flight evaluation tasks to finish (test helper)."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)


__all__ = ["BurstHit", "MoveAttributionDetector", "_backwalk", "_detect_burst"]
```

The `run(sink)` parks via `asyncio.Event().wait()` — but it must still register subscription before parking. Since the scheduler wires subscriptions explicitly (Task 8), we don't subscribe inside `run`. The detector's only role inside `run` is to record `self._sink`.

- [ ] **Step 7.4: Run, verify the four detector tests pass**

```bash
uv run pytest tests/detectors/test_move_attribution.py -v
```

Expected: PASS for all detector-level tests.

- [ ] **Step 7.5: Confirm `aclose` waits cleanly even when no tasks are pending**

```bash
uv run pytest tests/detectors/test_move_attribution.py::test_detector_ignores_non_trigger_detectors -v
```

Expected: PASS.

- [ ] **Step 7.6: Lint / format / type-check the whole module**

```bash
uv run ruff check src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py \
  && uv run ruff format src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py \
  && uv run ty check src/pscanner
```

Expected: all checks pass.

- [ ] **Step 7.7: Commit**

```bash
git add src/pscanner/detectors/move_attribution.py tests/detectors/test_move_attribution.py
git commit -m "feat(detectors): MoveAttributionDetector wires _backwalk + _detect_burst

Subscribes to AlertSink. On a triggering alert: backwalks the alerted
market, detects coordinated bursts, emits cluster.candidate alerts, and
upserts contributors into wallet_watchlist with source=cluster.candidate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Scheduler registration + integration smoke test

Wire the new detector into `Scheduler._build_detectors`, register the alert subscriber, add an integration test that exercises the full subscribe/fan-out path with a real `AlertSink`.

**Files:**
- Modify: `src/pscanner/scheduler.py:302-313` (instantiation block) and the alert-wiring path
- Modify: `tests/test_scheduler.py` (append integration test) OR new `tests/test_scheduler_move_attribution.py`

- [ ] **Step 8.1: Locate where alert subscribers should be wired**

Read `src/pscanner/scheduler.py` to find where detectors are wired post-construction (look for `_wire_trade_callbacks` or similar). The cleanest place to add a `_wire_alert_subscribers` method is right after `_wire_trade_callbacks`. Alternatively, do the subscription inside the same `_build_detectors` function right after instantiation — verify whichever pattern existing detectors use.

- [ ] **Step 8.2: Write failing integration test**

Append to `tests/test_scheduler.py`:

```python
import asyncio

import httpx
import pytest
import respx

from pscanner.alerts.models import Alert
from pscanner.config import Config


@respx.mock
@pytest.mark.asyncio
async def test_scheduler_wires_move_attribution_subscriber(tmp_path) -> None:
    """Velocity alert through the live AlertSink reaches MoveAttributionDetector."""
    alert_ts = 1_700_086_400
    burst = [
        {"proxyWallet": f"0x{i:04d}", "timestamp": alert_ts - 30,
         "side": "BUY", "outcome": "Yes", "size": 500.0 + i, "price": 0.95}
        for i in range(6)
    ]
    baseline = [
        {"proxyWallet": f"0xbg{i:04d}", "timestamp": alert_ts - 1800 - i * 60,
         "side": "BUY", "outcome": "Yes", "size": 50.0, "price": 0.5}
        for i in range(60)
    ]
    page = sorted(burst + baseline, key=lambda t: -t["timestamp"])[:500]
    respx.get("https://data-api.polymarket.com/trades").mock(
        return_value=httpx.Response(200, json=page)
    )

    cfg = Config()
    db_path = tmp_path / "pscanner.sqlite3"
    cfg = cfg.model_copy(update={
        "scanner": cfg.scanner.model_copy(update={"db_path": db_path}),
    })
    # Build the scheduler — implementation detail: see Scheduler.__init__ usage
    # in tests/test_scheduler.py for the existing canonical fixture.
    from pscanner.scheduler import Scheduler
    scheduler = Scheduler(config=cfg)
    try:
        sink = scheduler.sink
        await sink.emit(Alert(
            detector="velocity",
            alert_key="velocity:0xabc:1",
            severity="med",
            title="market moved",
            body={"condition_id": "0xabc"},
            created_at=alert_ts,
        ))
        # Allow spawned tasks to drain
        for _ in range(10):
            await asyncio.sleep(0)
    finally:
        await scheduler.aclose()
    # Verify a cluster.candidate alert was emitted
    from pscanner.store.db import init_db
    from pscanner.store.repo import AlertsRepo, WatchlistRepo
    conn = init_db(db_path)
    try:
        recent = AlertsRepo(conn).list_recent(limit=10)
        candidates = [a for a in recent if a.detector == "move_attribution"]
        assert len(candidates) >= 1
        watchlist = WatchlistRepo(conn).list_active()
        assert any(w.source == "cluster.candidate" for w in watchlist)
    finally:
        conn.close()
```

If `Scheduler` doesn't have an `aclose` method, see how the existing scheduler test cleans up — reuse that pattern.

- [ ] **Step 8.3: Run, verify it fails**

```bash
uv run pytest tests/test_scheduler.py::test_scheduler_wires_move_attribution_subscriber -v
```

Expected: FAIL — detector is not constructed by the scheduler yet.

- [ ] **Step 8.4: Wire the detector into the scheduler**

In `src/pscanner/scheduler.py`, add the import:

```python
from pscanner.detectors.move_attribution import MoveAttributionDetector
```

After the existing cluster-detector instantiation (around line 311) and before `_maybe_attach_velocity_detector`, add:

```python
        if self._config.move_attribution.enabled:
            detectors["move_attribution"] = MoveAttributionDetector(
                config=self._config.move_attribution,
                data_client=self._clients.data_client,
                watchlist_repo=self._watchlist_repo,
            )
```

(Verify `self._watchlist_repo` exists in the scheduler — it should, given `WatchlistSyncer` already takes a registry hydrated from it. If not, instantiate `WatchlistRepo(self._conn)` inline.)

Then add a method modelled on `_wire_trade_callbacks`:

```python
    def _wire_alert_subscribers(self) -> None:
        """Register every detector that exposes ``handle_alert_sync`` with the sink.

        Mirrors :meth:`_wire_trade_callbacks` but for alert-driven detectors.
        """
        for detector in self._detectors.values():
            if isinstance(detector, MoveAttributionDetector):
                detector._sink = self._sink
                self._sink.subscribe(detector.handle_alert_sync)
                _LOG.info("scanner.alert_driven_detector_wired", detector=detector.name)
```

Find where `_wire_trade_callbacks` is invoked during scheduler startup and call `_wire_alert_subscribers` immediately after.

- [ ] **Step 8.5: Run, verify it passes**

```bash
uv run pytest tests/test_scheduler.py::test_scheduler_wires_move_attribution_subscriber -v
```

Expected: PASS.

- [ ] **Step 8.6: Run the full test suite to catch regressions**

```bash
uv run pytest -q
```

Expected: all tests pass (or, if some pre-existing tests are flaky, no regression caused by this work).

- [ ] **Step 8.7: Final lint / format / type-check**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

Expected: all checks pass.

- [ ] **Step 8.8: Commit**

```bash
git add src/pscanner/scheduler.py tests/test_scheduler.py
git commit -m "feat(scheduler): wire MoveAttributionDetector to AlertSink subscriber path

Adds _wire_alert_subscribers mirroring _wire_trade_callbacks. Gated on
config.move_attribution.enabled. End-to-end smoke test verifies a velocity
alert produces a cluster.candidate alert and watchlist entries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (notes to engineer)

After Task 8 completes, verify:

1. **Spec coverage.** Walk through the spec section-by-section:
   - Architecture (subscribe-based detector) — ✓ Tasks 2 + 7 + 8
   - `_backwalk` returns `(since_ts, until_ts, burst_trades)` — ✓ Task 6
   - `_detect_burst` with truncation knobs — ✓ Task 5
   - `DataClient.get_market_trades` — ✓ Task 1
   - `AlertSink.emit` per-handler isolation — ✓ Task 2
   - `MoveAttributionConfig` defaults — ✓ Task 4
   - `DetectorName` includes `"move_attribution"` — ✓ Task 3
   - Scheduler registration — ✓ Task 8
   - `WatchlistRepo.upsert` `keep_existing_reason` — INTENTIONALLY DROPPED (existing `INSERT OR IGNORE` already preserves first-recorded provenance; spec note explained at top of plan).

2. **Type consistency.** `BurstHit.wallets` is `tuple[str, ...]` everywhere it's referenced. `_backwalk` returns the 3-tuple shape used by `evaluate`. `MoveAttributionDetector` ctor signature matches Task 8's instantiation site.

3. **No placeholders.** Every code block contains real code, not pseudocode.

4. **Commit cadence.** Eight commits, one per task. None batched.

---

## Out-of-plan follow-ups

- The smoke test asserts an end-to-end emission but doesn't exercise the
  `cluster.discovered` confirmation handoff — that depends on the
  `TradeCollector` polling watchlisted wallets and the `ClusterDetector`
  scoring them. Verifiable manually with a local run; not worth wiring as a
  test.
- If `MoveAttributionConfig` defaults turn out to be too noisy in production,
  start with `min_burst_wallets=6` instead of 4 and iterate. No code change
  needed — config-only.
- Potential follow-up: have the detector also auto-run `expand_cluster.py`'s
  expansion pass when a hit fires (Option C from brainstorming). Deferred.
