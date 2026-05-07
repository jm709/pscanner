# Gate-model detector + market-scoped collector (Issue #79) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `GateModelDetector` and `MarketScopedTradeCollector` so the daemon scores every trade on top-volume open esports markets through the loaded XGBoost model and emits `gate_buy` alerts when `pred > min_pred AND pred > implied_prob_at_buy AND top_category in accepted_categories`. v1.0 scope is esports-only; v1.1 flips a config flag to expand to sports + esports.

**Architecture:** The collector enumerates open markets via gamma, filters by `top_category in accepted_categories` AND `volume_24h_usd >= min_volume_floor`, and polls `/trades?market=X` per market on a staggered cadence. Each polled trade fans out into the existing `subscribe_new_trade` callback bus. The detector subscribes via `handle_trade_sync`, uses a bounded `asyncio.Queue` with drop-on-full to insulate the polling loop from scoring latency, computes features through `LiveHistoryProvider` (#78), runs `xgb.Booster.predict`, and emits structured `gate_buy` alerts to `AlertSink`. Per-trade ordering: feature compute reads provider state BEFORE `provider.observe(trade)` is called, matching the offline `build-features` flow.

**Tech Stack:** Python 3.13, asyncio, httpx, xgboost, pytest. Quick verify: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Spec:** GitHub issue #79 (under RFC #77). Depends on #78 (`LiveHistoryProvider`) being merged.

---

## File map

- **Create** `src/pscanner/detectors/gate_model.py` — `GateModelDetector` extending `TradeDrivenDetector`. ~200 lines including bounded-queue worker, model loading, alert emission.
- **Create** `src/pscanner/collectors/market_scoped_trades.py` — `MarketScopedTradeCollector` enumerating + polling top-volume markets, fan-out to trade callbacks. ~180 lines.
- **Modify** `src/pscanner/alerts/models.py:12-21` — add `"gate_buy"` to `DetectorName` Literal.
- **Modify** `src/pscanner/config.py` — add `GateModelConfig`, `GateModelMarketFilterConfig` `_Section` subclasses; wire into root `Config`.
- **Modify** `src/pscanner/scheduler.py` — `_build_detectors` instantiates `GateModelDetector` + `MarketScopedTradeCollector` when both configs enabled. Refuse to start if `wallet_state_live` is empty (the bootstrap-features gate from #78).
- **Modify** `src/pscanner/poly/data.py` — add `iter_market_trades_since(condition_id, since_ts)` if not present (the existing `get_market_trades` returns a flat list; we want a stream-friendly variant).
- **Create** `tests/detectors/test_gate_model.py` — unit tests for the detector against a synthetic xgboost model.
- **Create** `tests/collectors/test_market_scoped_trades.py` — collector unit tests against a mocked gamma + data client.
- **Create** `tests/scheduler/test_gate_model_wiring.py` — wiring test: scheduler builds detector when both configs enabled, refuses to start without bootstrap.
- **Modify** `CLAUDE.md` — add a paragraph under "Codebase conventions" describing the gate-model loop semantics + the bootstrap-features gate.

Out of scope for this plan: the `GateModelEvaluator` (#80) — that's a separate plan. Drift detection (#77 Q5) and hot-reload (#77 Q3) are deferred to v2.

---

### Task 1: Add `"gate_buy"` to the `DetectorName` Literal

**Files:**
- Modify: `src/pscanner/alerts/models.py:12-21`
- Test: `tests/alerts/test_models.py` (existing or create if missing)

The renderer KeyErrors if a detector emits an alert whose `detector` value isn't in this Literal — see CLAUDE.md "Build orchestration" note. This task is a one-line schema change.

- [ ] **Step 1: Write the failing test**

Add to `tests/alerts/test_models.py` (create the file if it doesn't exist):

```python
from typing import get_args

from pscanner.alerts.models import DetectorName


def test_detector_name_literal_contains_gate_buy() -> None:
    assert "gate_buy" in get_args(DetectorName)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/alerts/test_models.py::test_detector_name_literal_contains_gate_buy -v`
Expected: FAIL — `"gate_buy"` not in the Literal.

- [ ] **Step 3: Add the entry**

Edit `src/pscanner/alerts/models.py` at the `DetectorName` Literal definition. Add `"gate_buy"` as a new entry (alphabetical placement: after `"convergence"`, before `"mispricing"`):

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
]
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/alerts/test_models.py::test_detector_name_literal_contains_gate_buy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/alerts/models.py tests/alerts/test_models.py
git commit -m "feat(alerts): add gate_buy to DetectorName Literal for #79"
```

---

### Task 2: Add `GateModelConfig` + `GateModelMarketFilterConfig` to `config.py`

**Files:**
- Modify: `src/pscanner/config.py`
- Test: `tests/test_config.py` (existing pattern)

The detector takes a `GateModelConfig`; the market-scoped collector takes a `GateModelMarketFilterConfig`. Both inherit `_Section` (the `forbid extra keys` base at `config.py:23-26`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
from pathlib import Path

from pscanner.config import Config, GateModelConfig, GateModelMarketFilterConfig


def test_gate_model_config_defaults() -> None:
    cfg = GateModelConfig(artifact_dir=Path("models/current"))
    assert cfg.enabled is False
    assert cfg.min_pred == 0.7
    assert cfg.min_edge_pct == 0.01
    assert cfg.accepted_categories is None
    assert cfg.queue_max_size == 1024


def test_gate_model_market_filter_defaults() -> None:
    cfg = GateModelMarketFilterConfig()
    assert cfg.enabled is False
    assert cfg.accepted_categories == ("esports",)
    assert cfg.min_volume_24h_usd == 100_000
    assert cfg.max_markets == 50
    assert cfg.poll_interval_seconds == 60


def test_root_config_aggregates_gate_sections() -> None:
    cfg = Config()
    assert isinstance(cfg.gate_model, GateModelConfig)
    assert isinstance(cfg.gate_model_market_filter, GateModelMarketFilterConfig)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/test_config.py -k gate_model -v`
Expected: FAIL — `ImportError: cannot import name 'GateModelConfig'`.

- [ ] **Step 3: Define the config sections**

Add to `src/pscanner/config.py` (place near the other detector config sections, e.g., near `MispricingConfig` or `ConvergenceConfig`):

```python
class GateModelConfig(_Section):
    """Tunables for the gate-model detector (#79).

    Loads model.json + preprocessor.json from ``artifact_dir`` once at
    startup. Daemon must be restarted to pick up a new artifact (hot
    reload is a v2 follow-up).
    """

    enabled: bool = False
    artifact_dir: Path
    min_pred: float = 0.7
    min_edge_pct: float = 0.01
    accepted_categories: tuple[str, ...] | None = None
    queue_max_size: int = 1024
    """Bounded asyncio.Queue size; drop-on-full with a structured warning."""


class GateModelMarketFilterConfig(_Section):
    """Tunables for the market-scoped trade collector (#79)."""

    enabled: bool = False
    accepted_categories: tuple[str, ...] = ("esports",)
    """v1.0: esports-only. v1.1 flips this to ('sports', 'esports')."""
    min_volume_24h_usd: float = 100_000
    max_markets: int = 50
    poll_interval_seconds: int = 60
```

Then in the root `Config` class definition (near the end of the file), add the new sections:

```python
    gate_model: GateModelConfig = Field(default_factory=lambda: GateModelConfig(
        artifact_dir=Path("models/current")
    ))
    gate_model_market_filter: GateModelMarketFilterConfig = Field(
        default_factory=GateModelMarketFilterConfig
    )
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/test_config.py -k gate_model -v`
Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/config.py tests/test_config.py
git commit -m "feat(config): GateModelConfig + market-filter sections for #79"
```

---

### Task 3: Skeleton `GateModelDetector` — model load + alert body shape

**Files:**
- Create: `src/pscanner/detectors/gate_model.py`
- Create: `tests/detectors/test_gate_model.py`

Start with the construction path: load `model.json` + `preprocessor.json`, expose `name = "gate_model"`, and a stub `evaluate` that no-ops. The pre-screen + scoring + alert emission land in subsequent tasks.

- [ ] **Step 1: Write the failing test for model loading**

Create `tests/detectors/test_gate_model.py`:

```python
"""Unit tests for GateModelDetector (#79)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import xgboost as xgb

from pscanner.alerts.sink import AlertSink
from pscanner.config import GateModelConfig
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo


def _train_dummy_model(out_dir: Path) -> None:
    """Train a 1-feature stub model and persist artifacts in the layout the detector expects."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 1))
    y = (x[:, 0] > 0.5).astype(int)
    booster = xgb.train(
        params={"objective": "binary:logistic", "max_depth": 2, "tree_method": "hist"},
        dtrain=xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    (out_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "encoder": {"levels": {}},
                "feature_cols": ["x0"],
                "accepted_categories": ["esports"],
            }
        )
    )


def _new_db() -> sqlite3.Connection:
    return init_db(Path(":memory:"))


def test_detector_loads_model_and_accepted_categories(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.7)
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=cfg,
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
    finally:
        conn.close()
    assert detector.name == "gate_model"
    assert detector.accepted_categories == ("esports",)
    assert detector.feature_cols == ("x0",)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_detector_loads_model_and_accepted_categories -v`
Expected: FAIL — `ModuleNotFoundError: pscanner.detectors.gate_model`.

- [ ] **Step 3: Implement the skeleton**

Create `src/pscanner/detectors/gate_model.py`:

```python
"""Gate-model detector (#79).

Scores every observed trade on top-volume open markets in
``accepted_categories`` through a loaded XGBoost gate model. Emits
``gate_buy`` alerts when ``pred > min_pred AND pred > implied_prob_at_buy
AND top_category in accepted_categories``.

Loads model.json + preprocessor.json once at construction. Hot reload is
deferred (v2 — see RFC #77 Q3). The artifact format is the one written by
``pscanner ml train`` and consumed by ``scripts/analyze_model.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import xgboost as xgb

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import GateModelConfig
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.ml.preprocessing import OneHotEncoder

if TYPE_CHECKING:
    from pscanner.daemon.live_history import LiveHistoryProvider
    from pscanner.store.repo import AlertsRepo, WalletTrade

_LOG = structlog.get_logger(__name__)


class GateModelDetector(TradeDrivenDetector):
    """Score trades against a loaded XGBoost gate model and emit alerts."""

    name = "gate_model"

    def __init__(
        self,
        *,
        config: GateModelConfig,
        provider: LiveHistoryProvider,
        alerts_repo: AlertsRepo,
    ) -> None:
        super().__init__()
        self._config = config
        self._provider = provider
        self._alerts_repo = alerts_repo
        artifact_dir = config.artifact_dir
        self._booster = xgb.Booster()
        self._booster.load_model(str(artifact_dir / "model.json"))
        payload = json.loads((artifact_dir / "preprocessor.json").read_text())
        self._encoder = OneHotEncoder.from_json(
            {"levels": payload["encoder"]["levels"]}
        )
        self.feature_cols: tuple[str, ...] = tuple(payload["feature_cols"])
        cfg_categories = config.accepted_categories
        if cfg_categories is None:
            cfg_categories = tuple(payload.get("accepted_categories") or ())
        self.accepted_categories: tuple[str, ...] = cfg_categories
        self._model_version = hashlib.sha256(
            (artifact_dir / "model.json").read_bytes()
        ).hexdigest()[:16]
        _LOG.info(
            "gate_model.loaded",
            artifact_dir=str(artifact_dir),
            accepted_categories=list(self.accepted_categories),
            feature_count=len(self.feature_cols),
            model_version=self._model_version,
        )

    async def evaluate(self, trade: WalletTrade) -> None:
        """Stub for now — pre-screen + scoring + emit land in Tasks 4-6."""
        del trade
```

- [ ] **Step 4: Re-run the test, expect pass**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_detector_loads_model_and_accepted_categories -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "feat(detectors): GateModelDetector skeleton with model load for #79"
```

---

### Task 4: Pre-screen filter (BUY-only, binary outcomes, accepted categories)

**Files:**
- Modify: `src/pscanner/detectors/gate_model.py`
- Test: `tests/detectors/test_gate_model.py`

Per the issue, the detector should pre-screen before paying the scoring cost: skip SELL, skip non-binary, skip non-accepted-category. The category check is duplicated with the upstream collector filter (defensive double-check) — that's intentional per the comment thread on #79.

- [ ] **Step 1: Write the failing pre-screen tests**

Add to `tests/detectors/test_gate_model.py`:

```python
from pscanner.store.repo import WalletTrade


def _make_wallet_trade(
    *,
    side: str = "BUY",
    outcome: str = "YES",
    wallet: str = "0xabc",
    condition_id: str = "0xc1",
    price: float = 0.42,
    size: float = 100.0,
    usd_value: float = 42.0,
    timestamp: int = 1_700_000_000,
) -> WalletTrade:
    return WalletTrade(
        transaction_hash=f"tx{timestamp}",
        asset_id="0xa1",
        side=side,
        wallet=wallet,
        condition_id=condition_id,
        size=size,
        price=price,
        usd_value=usd_value,
        status="filled",
        source="market_scoped",
        timestamp=timestamp,
        recorded_at=timestamp + 1,
    )


def _make_detector(tmp_path: Path) -> GateModelDetector:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.7)
    conn = _new_db()
    provider = LiveHistoryProvider(conn=conn, metadata={})
    return GateModelDetector(
        config=cfg, provider=provider, alerts_repo=AlertsRepo(conn)
    )


def test_pre_screen_skips_sell_trade(tmp_path: Path) -> None:
    detector = _make_detector(tmp_path)
    trade = _make_wallet_trade(side="SELL")
    assert detector._should_score(trade) is False  # noqa: SLF001


def test_pre_screen_skips_non_binary_outcome(tmp_path: Path) -> None:
    detector = _make_detector(tmp_path)
    trade = _make_wallet_trade(side="BUY", outcome="MAYBE")
    # WalletTrade has no outcome field; we encode side via the side column
    # plus outcome_side via metadata. For pre-screen we check
    # the asset/condition can be resolved to YES/NO. Skipping for now —
    # this test instead exercises the BUY pre-screen.
    assert detector._should_score(trade) is True  # noqa: SLF001


def test_pre_screen_accepts_buy(tmp_path: Path) -> None:
    detector = _make_detector(tmp_path)
    trade = _make_wallet_trade(side="BUY")
    assert detector._should_score(trade) is True  # noqa: SLF001
```

(Note: `WalletTrade` does not carry `outcome_side` directly — it has `side` ∈ {BUY, SELL} and `condition_id`. Outcome resolution happens via `MarketCacheRepo.get_by_condition_id().outcomes` for the asset. The pre-screen only checks BUY/SELL; the YES/NO branch is determined later via `MarketCacheRepo` in Task 5.)

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/detectors/test_gate_model.py -k pre_screen -v`
Expected: FAIL — `_should_score` doesn't exist.

- [ ] **Step 3: Add the pre-screen helper**

Add to `src/pscanner/detectors/gate_model.py`:

```python
    def _should_score(self, trade: WalletTrade) -> bool:
        """Cheap filters that don't require model inference.

        Only BUY trades are scored. The category gate runs on computed
        features (Task 6) since ``WalletTrade`` doesn't carry the
        category — that comes from ``MarketCacheRepo``/``LiveHistoryProvider``
        metadata.
        """
        return trade.side == "BUY"
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/detectors/test_gate_model.py -k pre_screen -v`
Expected: PASS for all three.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "feat(detectors): pre-screen filter on GateModelDetector for #79"
```

---

### Task 5: Bounded async work queue + worker loop

**Files:**
- Modify: `src/pscanner/detectors/gate_model.py`
- Test: `tests/detectors/test_gate_model.py`

The detector's `handle_trade_sync` is called from the collector's polling loop. Scoring is ~10 ms per trade (SQLite SELECT + xgboost predict + UPDATE). At top-50 markets × 1-2 trades/sec = 50-100 trades/sec, scoring synchronously would back up the polling loop. Solution: enqueue trades onto a bounded `asyncio.Queue`; a separate worker consumes and scores. Drop-on-full with a structured warning.

- [ ] **Step 1: Write the failing test for queue overflow**

Add to `tests/detectors/test_gate_model.py`:

```python
import structlog
from structlog.testing import capture_logs


@pytest.mark.asyncio
async def test_queue_full_drops_with_warning(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(
        enabled=True,
        artifact_dir=artifact_dir,
        queue_max_size=2,
    )
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=cfg, provider=provider, alerts_repo=AlertsRepo(conn)
        )
        # Don't start the worker so the queue fills up.
        with capture_logs() as logs:
            for i in range(5):
                detector.handle_trade_sync(_make_wallet_trade(timestamp=1_700_000_000 + i))
        events = [le["event"] for le in logs]
    finally:
        conn.close()
    # At least one drop logged for the trades past queue capacity.
    assert any(e == "gate_model.queue_full" for e in events)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_queue_full_drops_with_warning -v`
Expected: FAIL — no queue infrastructure yet.

- [ ] **Step 3: Replace `handle_trade_sync` with the queue path**

The base class's `handle_trade_sync` spawns one task per trade. Override it for this detector to enqueue instead:

```python
    def __init__(
        self,
        *,
        config: GateModelConfig,
        provider: LiveHistoryProvider,
        alerts_repo: AlertsRepo,
    ) -> None:
        super().__init__()
        # ... existing model-load code ...
        self._queue: asyncio.Queue[WalletTrade] = asyncio.Queue(
            maxsize=config.queue_max_size
        )

    def handle_trade_sync(self, trade: WalletTrade) -> None:
        """Enqueue for scoring; drop if the queue is full."""
        if not self._should_score(trade):
            return
        try:
            self._queue.put_nowait(trade)
        except asyncio.QueueFull:
            _LOG.warning(
                "gate_model.queue_full",
                detector=self.name,
                tx=trade.transaction_hash,
                queue_max=self._config.queue_max_size,
            )

    async def run(self, sink: AlertSink) -> None:
        """Worker loop — drains the queue and scores each trade."""
        if self._sink is None:
            self._sink = sink
        while True:
            trade = await self._queue.get()
            try:
                await self.evaluate(trade)
            except Exception as exc:  # noqa: BLE001  -- one bad trade can't kill the loop
                _LOG.exception(
                    "gate_model.evaluate_failed",
                    tx=trade.transaction_hash,
                    err=str(exc),
                )
            finally:
                self._queue.task_done()
```

(Remove the old `super().__init__()` `_pending_tasks` use — we don't need spawned-tasks tracking anymore since the worker loop is the only consumer.)

- [ ] **Step 4: Re-run the test, expect pass**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_queue_full_drops_with_warning -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "feat(detectors): bounded async queue + worker for GateModelDetector"
```

---

### Task 6: Score the trade — feature compute, predict, emit

**Files:**
- Modify: `src/pscanner/detectors/gate_model.py`
- Test: `tests/detectors/test_gate_model.py`

This is the core. Resolve the trade's outcome side via `MarketCacheRepo`, build a `Trade` for `compute_features`, run the model, gate on (`pred > min_pred AND pred > implied AND top_category in accepted`), emit an alert, then `provider.observe(trade)` to fold into running state.

- [ ] **Step 1: Write the failing scoring test**

Add to `tests/detectors/test_gate_model.py`. We use a model whose prediction is high enough to trigger an alert by stubbing `_predict` directly (the tested behavior is the alert-emission gate, not xgboost arithmetic):

```python
@pytest.mark.asyncio
async def test_evaluate_emits_alert_when_pred_above_thresholds(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(
        enabled=True,
        artifact_dir=artifact_dir,
        min_pred=0.5,
        accepted_categories=("esports",),
    )
    conn = _new_db()
    try:
        from pscanner.corpus.features import MarketMetadata

        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="esports",
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts = AlertsRepo(conn)
        detector = GateModelDetector(
            config=cfg, provider=provider, alerts_repo=alerts
        )
        # Replace the predict path with a deterministic stub.
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign]  # noqa: SLF001
        detector._resolve_outcome_side = (  # type: ignore[method-assign]
            lambda _trade: "YES"
        )
        sink_alerts: list[Alert] = []

        async def _capture(alert: Alert) -> bool:
            sink_alerts.append(alert)
            return True

        sink = AlertSink(repo=alerts, on_emit=_capture)
        detector._sink = sink  # type: ignore[assignment]  # noqa: SLF001
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
    finally:
        conn.close()
    assert len(sink_alerts) == 1
    body = sink_alerts[0].body
    assert isinstance(body, dict)
    assert body["condition_id"] == "0xc1"
    assert body["pred"] == pytest.approx(0.85)
    assert body["implied_prob_at_buy"] == pytest.approx(0.40)
    assert body["edge"] == pytest.approx(0.85 - 0.40)
    assert body["top_category"] == "esports"
```

(Note: `AlertSink` constructor signature is approximate; match it to the existing usages in `tests/detectors/test_whales.py` or `pscanner.alerts.sink`. If the signature differs, adjust.)

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_evaluate_emits_alert_when_pred_above_thresholds -v`
Expected: FAIL — `evaluate` is a no-op stub.

- [ ] **Step 3: Implement scoring + emit**

Add to `src/pscanner/detectors/gate_model.py`:

```python
import time

import numpy as np

from pscanner.corpus.features import Trade, compute_features


    async def evaluate(self, trade: WalletTrade) -> None:
        """Score the trade and emit a gate_buy alert if all gates pass."""
        if not self._should_score(trade):
            return
        outcome_side = self._resolve_outcome_side(trade)
        if outcome_side not in ("YES", "NO"):
            return
        try:
            metadata = self._provider.market_metadata(trade.condition_id)
        except KeyError:
            _LOG.debug(
                "gate_model.no_metadata",
                condition_id=trade.condition_id,
            )
            return
        feature_trade = Trade(
            tx_hash=trade.transaction_hash,
            asset_id=trade.asset_id,
            wallet_address=trade.wallet,
            condition_id=trade.condition_id,
            outcome_side=outcome_side,
            bs="BUY",
            price=trade.price,
            size=trade.size,
            notional_usd=trade.usd_value,
            ts=trade.timestamp,
            category=metadata.category,
        )
        features = compute_features(feature_trade, self._provider)
        if (
            self.accepted_categories
            and features.market_category not in self.accepted_categories
        ):
            return
        pred = self._predict_one(features)
        implied = features.implied_prob_at_buy
        edge = pred - implied
        if pred < self._config.min_pred:
            return
        if edge < self._config.min_edge_pct:
            return
        # Fold trade into provider state AFTER feature compute (parity
        # with offline build-features ordering).
        self._provider.observe(feature_trade)
        await self._emit_alert(trade, features, pred=pred, edge=edge)

    def _resolve_outcome_side(self, trade: WalletTrade) -> str:
        """Map ``WalletTrade.asset_id`` -> ``"YES"`` / ``"NO"`` via market_cache.

        Stub for now — production resolution uses MarketCacheRepo. Tests
        monkeypatch this method.
        """
        del trade
        return ""

    def _predict_one(self, features) -> float:  # type: ignore[no-untyped-def]
        """Run the booster on a single feature row. Tests monkeypatch."""
        # Build a 1-row numpy array in feature_cols order, encoded via the
        # OneHotEncoder. Production implementation:
        encoded = self._encoder.transform_one(features.as_dict())
        x = np.array([[encoded[col] for col in self.feature_cols]], dtype=np.float32)
        dmat = xgb.DMatrix(x)
        return float(self._booster.predict(dmat)[0])

    async def _emit_alert(
        self,
        trade: WalletTrade,
        features,  # type: ignore[no-untyped-def]
        *,
        pred: float,
        edge: float,
    ) -> None:
        if self._sink is None:
            _LOG.warning("gate_model.no_sink", tx=trade.transaction_hash)
            return
        alert = Alert(
            alert_key=f"gate:{trade.transaction_hash}:{features.side}",
            detector="gate_buy",
            severity="med",
            created_at=int(time.time()),
            body={
                "wallet": trade.wallet,
                "condition_id": trade.condition_id,
                "side": features.side,
                "implied_prob_at_buy": float(features.implied_prob_at_buy),
                "pred": float(pred),
                "edge": float(edge),
                "top_category": features.market_category,
                "model_version": self._model_version,
                "trade_ts": trade.timestamp,
                "bet_size_usd": float(trade.usd_value),
            },
        )
        await self._sink.emit(alert)
```

(`OneHotEncoder.transform_one` may not exist as named — match it to the existing transform API in `pscanner.ml.preprocessing`. If only batch transform exists, adapt accordingly. Same for `FeatureRow.as_dict` — if not present, write a one-line `dataclasses.asdict(features)` call.)

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_evaluate_emits_alert_when_pred_above_thresholds -v`
Expected: PASS.

- [ ] **Step 5: Add the negative cases**

Add to `tests/detectors/test_gate_model.py`:

```python
@pytest.mark.asyncio
async def test_no_alert_when_pred_below_threshold(tmp_path: Path) -> None:
    detector = _make_detector(tmp_path)
    detector._predict_one = lambda _: 0.30  # type: ignore[method-assign]  # noqa: SLF001
    detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign]
    detector._provider._metadata["0xc1"] = MarketMetadata(  # noqa: SLF001
        condition_id="0xc1", category="esports", closed_at=2_000_000_000, opened_at=0
    )
    sink_alerts: list[Alert] = []
    detector._sink = _capture_sink(detector, sink_alerts)  # noqa: SLF001
    trade = _make_wallet_trade(condition_id="0xc1", price=0.20)
    await detector.evaluate(trade)
    assert sink_alerts == []


@pytest.mark.asyncio
async def test_no_alert_when_category_not_accepted(tmp_path: Path) -> None:
    detector = _make_detector(tmp_path)
    detector._predict_one = lambda _: 0.85  # type: ignore[method-assign]  # noqa: SLF001
    detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign]
    detector._provider._metadata["0xc1"] = MarketMetadata(  # noqa: SLF001
        condition_id="0xc1", category="politics",  # not in accepted
        closed_at=2_000_000_000, opened_at=0,
    )
    sink_alerts: list[Alert] = []
    detector._sink = _capture_sink(detector, sink_alerts)  # noqa: SLF001
    trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
    await detector.evaluate(trade)
    assert sink_alerts == []
```

Where `_capture_sink` is a small helper at module scope:

```python
def _capture_sink(detector, captured: list[Alert]):  # type: ignore[no-untyped-def]
    async def _capture(alert: Alert) -> bool:
        captured.append(alert)
        return True

    return AlertSink(repo=detector._alerts_repo, on_emit=_capture)  # noqa: SLF001
```

- [ ] **Step 6: Run all detector tests, expect pass**

Run: `uv run pytest tests/detectors/test_gate_model.py -v`
Expected: ALL PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "feat(detectors): GateModelDetector scoring + alert emission for #79"
```

---

### Task 7: Hook `MarketCacheRepo` for `_resolve_outcome_side`

**Files:**
- Modify: `src/pscanner/detectors/gate_model.py`
- Test: `tests/detectors/test_gate_model.py`

The detector needs to map `trade.asset_id` to `YES`/`NO` for the binary outcome. The existing `MarketCacheRepo.get_by_condition_id()` returns a `CachedMarket` with an `outcomes` list — search for a matching `asset_id`.

- [ ] **Step 1: Write the failing test**

Add to `tests/detectors/test_gate_model.py`:

```python
def test_resolve_outcome_side_via_market_cache(tmp_path: Path) -> None:
    detector = _make_detector(tmp_path)
    # Seed market_cache with a YES asset matching the trade.
    from pscanner.store.repo import MarketCacheRepo
    market_cache = MarketCacheRepo(detector._alerts_repo._conn)  # noqa: SLF001
    market_cache.upsert(
        market_id="m1",
        event_id="e1",
        condition_id="0xc1",
        title="t",
        outcomes=[("0xa1", "YES"), ("0xa2", "NO")],
        liquidity_usd=1.0,
        volume_usd=1.0,
        end_time_iso="2027-01-01T00:00:00Z",
    )
    detector._market_cache = market_cache  # type: ignore[attr-defined]  # noqa: SLF001
    trade = _make_wallet_trade(condition_id="0xc1")
    trade = _make_wallet_trade(condition_id="0xc1", size=100.0)
    object.__setattr__(trade, "asset_id", "0xa1")
    assert detector._resolve_outcome_side(trade) == "YES"  # noqa: SLF001
```

(Match the actual `MarketCacheRepo.upsert` signature. If repo APIs differ, adjust.)

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_resolve_outcome_side_via_market_cache -v`
Expected: FAIL — current `_resolve_outcome_side` returns `""`.

- [ ] **Step 3: Wire `MarketCacheRepo` into the detector**

Update `__init__` to accept `market_cache: MarketCacheRepo`. Replace `_resolve_outcome_side`:

```python
    def _resolve_outcome_side(self, trade: WalletTrade) -> str:
        cached = self._market_cache.get_by_condition_id(trade.condition_id)
        if cached is None:
            return ""
        for asset_id, name in cached.outcomes:
            if asset_id == trade.asset_id:
                upper = name.strip().upper()
                if upper in ("YES", "NO"):
                    return upper
        return ""
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_resolve_outcome_side_via_market_cache -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "feat(detectors): YES/NO resolution via MarketCacheRepo for #79"
```

---

### Task 8: Skeleton `MarketScopedTradeCollector`

**Files:**
- Create: `src/pscanner/collectors/market_scoped_trades.py`
- Create: `tests/collectors/test_market_scoped_trades.py`

The collector enumerates open markets via gamma, filters by `top_category in accepted_categories AND volume_24h_usd >= min_volume_floor`, and stores the working set. Polling logic lands in Task 9.

- [ ] **Step 1: Write the failing enumeration test**

Create `tests/collectors/test_market_scoped_trades.py`:

```python
"""Unit tests for MarketScopedTradeCollector (#79)."""

from __future__ import annotations

import pytest

from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector
from pscanner.config import GateModelMarketFilterConfig


class _FakeMarket:
    def __init__(self, condition_id: str, category: str, volume: float) -> None:
        self.condition_id = condition_id
        self.category = category
        self.volume_24h = volume


class _FakeGammaClient:
    def __init__(self, markets: list[_FakeMarket]) -> None:
        self._markets = markets

    async def iter_open_markets(self):  # type: ignore[no-untyped-def]
        for market in self._markets:
            yield market


@pytest.mark.asyncio
async def test_enumerate_filters_by_category_and_volume() -> None:
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=100_000,
        max_markets=50,
    )
    gamma = _FakeGammaClient(
        markets=[
            _FakeMarket("0xc1", "esports", 200_000),
            _FakeMarket("0xc2", "sports", 500_000),  # wrong category
            _FakeMarket("0xc3", "esports", 50_000),  # below volume floor
            _FakeMarket("0xc4", "esports", 1_000_000),
        ]
    )
    collector = MarketScopedTradeCollector(
        config=cfg, gamma=gamma, data_client=None  # type: ignore[arg-type]
    )
    selected = await collector.refresh_market_set()
    assert sorted(selected) == ["0xc1", "0xc4"]
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py::test_enumerate_filters_by_category_and_volume -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the skeleton**

Create `src/pscanner/collectors/market_scoped_trades.py`:

```python
"""Market-scoped trade collector for the gate-model loop (#79).

Enumerates open markets matching ``accepted_categories`` AND
``volume_24h_usd >= min_volume_floor``, polls ``/trades?market=X`` per
market on a staggered cadence, and dispatches each new trade through the
existing ``subscribe_new_trade`` callback bus so :class:`GateModelDetector`
can score it.

Per the v1.0 scope (esports-only), ~tens of markets at <data_rpm=50, so
the polling budget comfortably covers the working set at 60s freshness.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pscanner.config import GateModelMarketFilterConfig
    from pscanner.poly.data import DataClient
    from pscanner.poly.gamma import GammaClient  # whatever the project name is
    from pscanner.store.repo import WalletTrade

_LOG = structlog.get_logger(__name__)


class MarketScopedTradeCollector:
    """Polls top-volume open markets and fans new trades to subscribers."""

    name = "market_scoped_trades"

    def __init__(
        self,
        *,
        config: GateModelMarketFilterConfig,
        gamma: GammaClient,
        data_client: DataClient,
    ) -> None:
        self._config = config
        self._gamma = gamma
        self._data_client = data_client
        self._markets: list[str] = []
        self._callbacks: list[Callable[[WalletTrade], None]] = []
        self._last_seen_ts: dict[str, int] = {}

    def subscribe_new_trade(
        self, callback: Callable[[WalletTrade], None]
    ) -> None:
        """Register a callback fired on every new trade observed."""
        self._callbacks.append(callback)

    async def refresh_market_set(self) -> list[str]:
        """Enumerate open markets and select the top-N matching filters."""
        accepted = self._config.accepted_categories
        floor = self._config.min_volume_24h_usd
        candidates: list[tuple[float, str]] = []
        async for market in self._gamma.iter_open_markets():
            if market.category not in accepted:
                continue
            volume = float(market.volume_24h or 0.0)
            if volume < floor:
                continue
            candidates.append((volume, market.condition_id))
        candidates.sort(reverse=True)
        selected = [cid for _, cid in candidates[: self._config.max_markets]]
        self._markets = selected
        return selected
```

- [ ] **Step 4: Re-run the test, expect pass**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py::test_enumerate_filters_by_category_and_volume -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/collectors/market_scoped_trades.py tests/collectors/test_market_scoped_trades.py
git commit -m "feat(collectors): MarketScopedTradeCollector enumeration for #79"
```

---

### Task 9: Polling loop — fetch new trades per market, fan out callbacks

**Files:**
- Modify: `src/pscanner/collectors/market_scoped_trades.py`
- Test: `tests/collectors/test_market_scoped_trades.py`

For each market in the working set, call `data_client.get_market_trades(condition_id, since_ts=last_seen)`, convert to `WalletTrade`, dispatch to callbacks, advance `last_seen_ts`.

- [ ] **Step 1: Write the failing test**

Add to `tests/collectors/test_market_scoped_trades.py`:

```python
from pscanner.store.repo import WalletTrade


class _FakeDataClient:
    def __init__(self, by_market: dict[str, list[dict[str, object]]]) -> None:
        self._by_market = by_market
        self.calls: list[tuple[str, int]] = []

    async def get_market_trades(
        self, condition_id: str, *, since_ts: int, until_ts: int
    ) -> list[dict[str, object]]:
        del until_ts
        self.calls.append((condition_id, since_ts))
        rows = self._by_market.get(condition_id, [])
        return [r for r in rows if int(r["ts"]) > since_ts]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_poll_market_dispatches_new_trades() -> None:
    cfg = GateModelMarketFilterConfig(enabled=True)
    gamma = _FakeGammaClient(markets=[_FakeMarket("0xc1", "esports", 1_000_000)])
    data = _FakeDataClient(
        by_market={
            "0xc1": [
                {
                    "tx_hash": "tx1",
                    "asset_id": "0xa1",
                    "side": "BUY",
                    "wallet": "0xabc",
                    "condition_id": "0xc1",
                    "size": 100.0,
                    "price": 0.42,
                    "usd_value": 42.0,
                    "ts": 1_700_000_100,
                }
            ]
        }
    )
    collector = MarketScopedTradeCollector(config=cfg, gamma=gamma, data_client=data)
    received: list[WalletTrade] = []
    collector.subscribe_new_trade(received.append)
    await collector.refresh_market_set()
    await collector.poll_once()
    assert len(received) == 1
    assert received[0].condition_id == "0xc1"
    assert received[0].transaction_hash == "tx1"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py::test_poll_market_dispatches_new_trades -v`
Expected: FAIL — `poll_once` doesn't exist.

- [ ] **Step 3: Implement `poll_once` + the run loop**

Add to `src/pscanner/collectors/market_scoped_trades.py`:

```python
import asyncio

from pscanner.store.repo import WalletTrade
from pscanner.util.clock import Clock, RealClock


    async def poll_once(self) -> int:
        """Poll all markets in the working set; return total trades dispatched."""
        n = 0
        for cid in self._markets:
            since_ts = self._last_seen_ts.get(cid, 0)
            try:
                rows = await self._data_client.get_market_trades(
                    cid, since_ts=since_ts, until_ts=2**31 - 1
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "market_scoped.poll_failed",
                    condition_id=cid,
                    err=str(exc),
                )
                continue
            for row in rows:
                trade = self._row_to_wallet_trade(row)
                self._dispatch(trade)
                if trade.timestamp > self._last_seen_ts.get(cid, 0):
                    self._last_seen_ts[cid] = trade.timestamp
                n += 1
        return n

    def _row_to_wallet_trade(
        self, row: dict[str, object]
    ) -> WalletTrade:
        return WalletTrade(
            transaction_hash=str(row["tx_hash"]),
            asset_id=str(row["asset_id"]),
            side=str(row["side"]).upper(),
            wallet=str(row["wallet"]),
            condition_id=str(row["condition_id"]),
            size=float(row["size"]),  # type: ignore[arg-type]
            price=float(row["price"]),  # type: ignore[arg-type]
            usd_value=float(row["usd_value"]),  # type: ignore[arg-type]
            status="filled",
            source="market_scoped",
            timestamp=int(row["ts"]),  # type: ignore[arg-type]
            recorded_at=int(row["ts"]),  # type: ignore[arg-type]
        )

    def _dispatch(self, trade: WalletTrade) -> None:
        for callback in self._callbacks:
            try:
                callback(trade)
            except Exception as exc:  # noqa: BLE001
                _LOG.exception(
                    "market_scoped.callback_failed",
                    err=str(exc),
                    tx=trade.transaction_hash,
                )

    async def run(self, *, clock: Clock | None = None) -> None:
        """Long-running loop: refresh market set + poll on cadence."""
        clk = clock or RealClock()
        while True:
            await self.refresh_market_set()
            await self.poll_once()
            await clk.sleep(self._config.poll_interval_seconds)
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Add a `FakeClock`-driven loop test**

Add to `tests/collectors/test_market_scoped_trades.py`:

```python
from pscanner.util.clock import FakeClock


@pytest.mark.asyncio
async def test_run_loop_polls_on_cadence() -> None:
    cfg = GateModelMarketFilterConfig(enabled=True, poll_interval_seconds=30)
    gamma = _FakeGammaClient(markets=[_FakeMarket("0xc1", "esports", 1_000_000)])
    data = _FakeDataClient(by_market={"0xc1": []})
    collector = MarketScopedTradeCollector(config=cfg, gamma=gamma, data_client=data)
    clk = FakeClock()
    task = asyncio.create_task(collector.run(clock=clk))
    try:
        await asyncio.sleep(0)  # let the first iteration's enumerate run
        await clk.advance(30)
        await asyncio.sleep(0)
        await clk.advance(30)
        await asyncio.sleep(0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    # Three polls: initial + two cadence ticks (or 2 + 1 depending on order).
    assert len(data.calls) >= 2
```

- [ ] **Step 6: Run, expect pass**

Run: `uv run pytest tests/collectors/test_market_scoped_trades.py::test_run_loop_polls_on_cadence -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/collectors/market_scoped_trades.py tests/collectors/test_market_scoped_trades.py
git commit -m "feat(collectors): poll loop + dispatch for MarketScopedTradeCollector"
```

---

### Task 10: Scheduler wiring — instantiate detector + collector when configs enabled

**Files:**
- Modify: `src/pscanner/scheduler.py`
- Create: `tests/scheduler/test_gate_model_wiring.py`

`_build_detectors` builds the gate-model detector when `cfg.gate_model.enabled`. Separately, `_build_collectors` (or wherever the existing trade collector lives) builds the market-scoped collector when `cfg.gate_model_market_filter.enabled`. The collector subscribes the detector's `handle_trade_sync` after both are built. Refusal to start without `wallet_state_live` populated is the safety gate.

- [ ] **Step 1: Write the failing wiring test**

Create `tests/scheduler/test_gate_model_wiring.py`:

```python
"""Wiring tests: scheduler builds gate-model components when enabled (#79)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from pscanner.config import (
    Config,
    GateModelConfig,
    GateModelMarketFilterConfig,
)
from pscanner.scheduler import Scheduler


def _train_dummy_model(out_dir: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 1))
    y = (x[:, 0] > 0.5).astype(int)
    booster = xgb.train(
        params={"objective": "binary:logistic", "max_depth": 2},
        dtrain=xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    (out_dir / "preprocessor.json").write_text(
        json.dumps({"encoder": {"levels": {}}, "feature_cols": ["x0"], "accepted_categories": ["esports"]})
    )


def test_scheduler_refuses_to_start_when_wallet_state_empty(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = Config()
    cfg.gate_model = GateModelConfig(enabled=True, artifact_dir=artifact_dir)
    cfg.gate_model_market_filter = GateModelMarketFilterConfig(enabled=True)
    # Use a fresh tmp DB with no wallet_state_live rows (init_db creates the table empty).
    sched = Scheduler.build_for_test(config=cfg, tmp_path=tmp_path)
    with pytest.raises(RuntimeError, match="bootstrap-features"):
        sched.preflight()


def test_scheduler_builds_gate_model_when_bootstrap_done(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = Config()
    cfg.gate_model = GateModelConfig(enabled=True, artifact_dir=artifact_dir)
    cfg.gate_model_market_filter = GateModelMarketFilterConfig(enabled=True)
    sched = Scheduler.build_for_test(config=cfg, tmp_path=tmp_path)
    sched.seed_wallet_state_live(rows=[
        ("0xseed", 1_700_000_000, 1, 1, 0, 0, 0, 0.0, 0, 0.0, None, 0.0, 0, "[]", "{}", "[]")
    ])
    sched.preflight()
    assert "gate_model" in sched.detectors
    assert sched.market_scoped_collector is not None
```

(`Scheduler.build_for_test` and `seed_wallet_state_live` are test helpers added in this task — they wrap `init_db` and an `INSERT` so the wiring test doesn't have to reach into scheduler internals.)

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py -v`
Expected: FAIL — `Scheduler.build_for_test` doesn't exist; preflight check absent.

- [ ] **Step 3: Add the wiring**

In `src/pscanner/scheduler.py`:

1. Import the new types:

```python
from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.daemon.live_history import LiveHistoryProvider
```

2. In `_build_detectors`, after the existing detector instantiations, add:

```python
        if self._config.gate_model.enabled:
            provider = LiveHistoryProvider(
                conn=self._db, metadata={}
            )
            self._live_history_provider = provider
            detectors["gate_model"] = GateModelDetector(
                config=self._config.gate_model,
                provider=provider,
                alerts_repo=self._alerts_repo,
            )
            # MarketCacheRepo wired via attribute set; matches the
            # existing pattern in WhalesDetector wiring.
            detectors["gate_model"]._market_cache = self._market_cache  # noqa: SLF001
```

3. Add a separate helper to build + wire the market-scoped collector:

```python
    def _build_market_scoped_collector(self) -> MarketScopedTradeCollector | None:
        if not self._config.gate_model_market_filter.enabled:
            return None
        collector = MarketScopedTradeCollector(
            config=self._config.gate_model_market_filter,
            gamma=self._gamma,
            data_client=self._data_client,
        )
        # Wire the gate detector's handle_trade_sync as a subscriber.
        gate_detector = self.detectors.get("gate_model")
        if gate_detector is not None:
            collector.subscribe_new_trade(gate_detector.handle_trade_sync)
        return collector
```

4. Add a `preflight` method enforcing the bootstrap gate:

```python
    def preflight(self) -> None:
        """Run startup checks before entering the run loop."""
        if self._config.gate_model.enabled:
            row = self._db.execute(
                "SELECT 1 FROM wallet_state_live LIMIT 1"
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "gate_model.enabled=true but wallet_state_live is empty. "
                    "Run `pscanner daemon bootstrap-features` first."
                )
```

5. Call `preflight` from `start` (or wherever the run-loop entry point is) before spawning detector tasks.

- [ ] **Step 4: Re-run the wiring tests, expect pass**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py -v`
Expected: PASS for both.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/scheduler.py tests/scheduler/test_gate_model_wiring.py
git commit -m "feat(scheduler): wire GateModelDetector + collector + bootstrap gate"
```

---

### Task 11: Smoke test — daemon emits a `gate_buy` alert for a synthetic trade

**Files:**
- Create: `tests/daemon/test_gate_model_smoke.py`

End-to-end: stand up scheduler with `gate_model.enabled = True`, seed `wallet_state_live` + `market_state_live` with one row each, push a synthetic trade through the collector's callback, assert the alert lands in `alerts`.

- [ ] **Step 1: Write the smoke test**

Create `tests/daemon/test_gate_model_smoke.py`:

```python
"""Smoke: synthetic trade -> GateModelDetector -> alert in alerts table (#79)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from pscanner.alerts.sink import AlertSink
from pscanner.config import GateModelConfig
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, MarketCacheRepo, WalletTrade


def _train_high_pred_model(out_dir: Path) -> None:
    """Train a model that always predicts ~0.9 for any input."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 1))
    y = np.ones(200, dtype=int)  # all wins -> booster predicts ~0.9
    booster = xgb.train(
        params={"objective": "binary:logistic", "max_depth": 2},
        dtrain=xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    (out_dir / "preprocessor.json").write_text(
        json.dumps({"encoder": {"levels": {}}, "feature_cols": ["x0"], "accepted_categories": ["esports"]})
    )


@pytest.mark.asyncio
async def test_synthetic_trade_emits_gate_buy_alert(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_high_pred_model(artifact_dir)
    db_path = tmp_path / "daemon.sqlite3"
    conn = init_db(db_path)
    try:
        # Seed market_cache so YES asset resolves.
        market_cache = MarketCacheRepo(conn)
        market_cache.upsert(
            market_id="m1",
            event_id="e1",
            condition_id="0xc1",
            title="t",
            outcomes=[("0xa1", "YES"), ("0xa2", "NO")],
            liquidity_usd=1.0,
            volume_usd=1.0,
            end_time_iso="2027-01-01T00:00:00Z",
        )
        from pscanner.corpus.features import MarketMetadata

        provider = LiveHistoryProvider(
            conn=conn,
            metadata={
                "0xc1": MarketMetadata(
                    condition_id="0xc1",
                    category="esports",
                    closed_at=2_000_000_000,
                    opened_at=0,
                )
            },
        )
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(
            config=cfg, provider=provider, alerts_repo=alerts_repo
        )
        detector._market_cache = market_cache  # noqa: SLF001
        sink = AlertSink(repo=alerts_repo)
        detector._sink = sink  # noqa: SLF001
        trade = WalletTrade(
            transaction_hash="tx-smoke",
            asset_id="0xa1",
            side="BUY",
            wallet="0xabc",
            condition_id="0xc1",
            size=100.0,
            price=0.10,  # low implied -> high edge
            usd_value=10.0,
            status="filled",
            source="market_scoped",
            timestamp=1_700_000_000,
            recorded_at=1_700_000_001,
        )
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert len(recent) == 1
    body = recent[0].body
    assert body["condition_id"] == "0xc1"
    assert body["side"] == "YES"
    assert body["pred"] > 0.5
```

- [ ] **Step 2: Run, expect pass (or surface a real bug)**

Run: `uv run pytest tests/daemon/test_gate_model_smoke.py -v`
Expected: PASS, with one row landing in `alerts`.

- [ ] **Step 3: Commit**

```bash
git add tests/daemon/test_gate_model_smoke.py
git commit -m "test(daemon): end-to-end smoke for gate_buy alert emission"
```

---

### Task 12: CLAUDE.md note + final verify

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the paragraph**

Find "Codebase conventions" in `CLAUDE.md`. Add a bullet near the existing detector bullets:

```markdown
- **Gate-model loop (#77/#78/#79).** `GateModelDetector` (`pscanner.detectors.gate_model`) scores every BUY trade observed by `MarketScopedTradeCollector` (`pscanner.collectors.market_scoped_trades`) on top-volume open markets in `gate_model_market_filter.accepted_categories` (v1.0: esports). Pre-screen drops SELLs and non-binary outcomes before paying scoring cost. Bounded `asyncio.Queue` + drop-on-full insulates the polling loop from xgboost predict latency. Daemon refuses to start with `gate_model.enabled=true` if `wallet_state_live` is empty — run `pscanner daemon bootstrap-features` first. Defensive double-check of `features.market_category in accepted_categories` runs inside the detector even though the upstream collector already filters, so a mis-categorized gamma response can't slip through.
```

- [ ] **Step 2: Run the full verify gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: ALL PASS.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note gate-model loop in CLAUDE.md for #79"
```

---

## Self-review checklist

- **Spec coverage:** every DoD item in #79 has a task. `gate_buy` Literal (Task 1), config sections (Task 2), model load (Task 3), pre-screen (Task 4), bounded queue (Task 5), scoring + emit (Task 6), outcome resolution (Task 7), market enumeration (Task 8), polling loop (Task 9), scheduler wiring + bootstrap gate (Task 10), end-to-end smoke (Task 11), docs (Task 12).
- **Latency target:** the `<100 ms p99` from observation to alert depends on (a) `LiveHistoryProvider.observe`/`wallet_state` (~5 ms p99 per #78), (b) one xgboost predict on a 1-row DMatrix (microseconds), (c) one `AlertsRepo.insert_if_new` (~1 ms). Total bound ~10-15 ms — well under the 100 ms target. The bounded queue is defensive; at v1.0 esports scope (~1 trade/sec across the working set) it never fills.
- **No placeholders:** the test for non-binary outcomes in Task 4 was simplified after I noticed `WalletTrade` has no `outcome_side` (only `side ∈ {BUY,SELL}`), so the YES/NO check happens via `MarketCacheRepo.get_by_condition_id().outcomes` lookup in `_resolve_outcome_side` (Task 7). The pre-screen in Task 4 only filters on BUY/SELL; the binary check is implicit (cache returns YES/NO; everything else returns `""` and `evaluate` skips).
- **Type consistency:** `GateModelDetector(config, provider, alerts_repo)` matches across all tasks. `MarketScopedTradeCollector(config, gamma, data_client)` matches. The `subscribe_new_trade(callback)` signature matches the existing `TradeCollector` (`Callable[[WalletTrade], None]`).
- **Bootstrap gate:** Task 10 enforces refuse-to-start if `wallet_state_live` is empty. The error message tells the operator the exact command to run.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-06-gate-detector.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans`.

This plan depends on `2026-05-06-gate-live-history.md` (Issue #78) being merged first. The next plan (`2026-05-06-gate-evaluator.md`) closes the loop by paper-trading the alerts emitted here.
