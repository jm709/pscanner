# Issue #101 — GateModelDetector Silent No-Op Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GateModelDetector` emit a structured debug log on every silent-skip path so an operator can tell *why* the detector is producing zero alerts when `markets.enabled = false` (or any other state-clearing condition leaves `MarketCacheRepo` empty), document the dependency, and enforce it at preflight.

**Architecture:** Three small, independent changes. (1) `_resolve_outcome_side` gets a debug log at each early-return so the operator can grep for `gate_model.no_market_cache` / `gate_model.market_not_cached` / `gate_model.outcome_not_binary` / `gate_model.asset_id_not_found`. (2) `evaluate()` gains a debug log at the outcome-skip early-return so the worker's drain progress is visible end-to-end. (3) `Scanner.preflight()` adds a hard check that `markets.enabled = true` whenever `gate_model.enabled = true`, mirroring the existing `wallet_state_live` check. CLAUDE.md's gate-model bullet records the dependency.

**Tech Stack:** Python 3.13, structlog, pytest, structlog.testing.capture_logs. No new deps.

---

## File Structure

- Modify: `src/pscanner/detectors/gate_model.py:163-225` — add four debug log lines in `_resolve_outcome_side` and one in `evaluate()`.
- Modify: `src/pscanner/scheduler.py:513-528` — extend `preflight()` with a `markets.enabled` check.
- Modify: `tests/detectors/test_gate_model.py` — add four log-capture tests for the new debug paths plus one for the `evaluate()` skip log.
- Modify: `tests/scheduler/test_gate_model_wiring.py` — add a preflight test for the `markets.enabled` requirement.
- Modify: `CLAUDE.md` — append a markets-collector dependency note to the gate-model bullet.

The five files are independent; tasks 1, 2, 3, 4 below can land as four separate commits.

---

## Task 1: Add debug logs to `_resolve_outcome_side`

**Files:**
- Modify: `src/pscanner/detectors/gate_model.py:206-225`
- Test: `tests/detectors/test_gate_model.py` (new tests appended)

The current method silently returns `""` on four distinct paths. Each gets a distinct event name plus the relevant identifying fields so an operator can grep `journalctl -u pscanner | grep gate_model` and see a steady stream of `gate_model.market_not_cached condition_id=0x...` lines instead of a quiet daemon.

- [ ] **Step 1: Write four failing log-capture tests**

Append to `tests/detectors/test_gate_model.py` (after `test_resolve_outcome_side_via_market_cache` at line 354):

```python
def test_resolve_outcome_logs_when_no_market_cache(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    events = [log["event"] for log in logs]
    assert "gate_model.no_market_cache" in events


def test_resolve_outcome_logs_when_market_not_cached(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)  # empty
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [
        log for log in logs if log["event"] == "gate_model.market_not_cached"
    ]
    assert len(matches) == 1
    assert matches[0]["condition_id"] == "0xc1"


def test_resolve_outcome_logs_when_outcome_not_binary(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)
        cached = CachedMarket(
            market_id=MarketId("m1"),
            event_id=EventId("e1"),
            title="t",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=ConditionId("0xc1"),
            event_slug=None,
            outcomes=["Trump", "Biden"],  # neither YES nor NO
            asset_ids=[AssetId("0xa1"), AssetId("0xa2")],
        )
        market_cache.upsert(cached)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [
        log for log in logs if log["event"] == "gate_model.outcome_not_binary"
    ]
    assert len(matches) == 1
    assert matches[0]["outcome"] == "Trump"


def test_resolve_outcome_logs_when_asset_id_not_found(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)
        cached = CachedMarket(
            market_id=MarketId("m1"),
            event_id=EventId("e1"),
            title="t",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=ConditionId("0xc1"),
            event_slug=None,
            outcomes=["Yes", "No"],
            asset_ids=[AssetId("0xa1"), AssetId("0xa2")],
        )
        market_cache.upsert(cached)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xother")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [
        log for log in logs if log["event"] == "gate_model.asset_id_not_found"
    ]
    assert len(matches) == 1
    assert matches[0]["asset_id"] == "0xother"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_resolve_outcome_logs_when_no_market_cache tests/detectors/test_gate_model.py::test_resolve_outcome_logs_when_market_not_cached tests/detectors/test_gate_model.py::test_resolve_outcome_logs_when_outcome_not_binary tests/detectors/test_gate_model.py::test_resolve_outcome_logs_when_asset_id_not_found -v`

Expected: All four FAIL with `AssertionError` — no log lines emitted yet.

- [ ] **Step 3: Add debug logs to `_resolve_outcome_side`**

Replace `_resolve_outcome_side` in `src/pscanner/detectors/gate_model.py:206-225`:

```python
def _resolve_outcome_side(self, trade: WalletTrade) -> str:
    """Map ``WalletTrade.asset_id`` -> ``"YES"`` / ``"NO"``.

    Returns ``""`` when no :class:`MarketCacheRepo` is wired, the market
    is not cached, the asset_id isn't found, or the matched outcome name
    is neither YES nor NO. Each silent-skip path emits a distinct debug
    log so an operator can diagnose why ``evaluate()`` is dropping every
    trade — see issue #101 for the failure mode this guards against.
    """
    if self._market_cache is None:
        _LOG.debug("gate_model.no_market_cache", tx=trade.transaction_hash)
        return ""
    cached = self._market_cache.get_by_condition_id(trade.condition_id)
    if cached is None:
        _LOG.debug(
            "gate_model.market_not_cached",
            tx=trade.transaction_hash,
            condition_id=trade.condition_id,
        )
        return ""
    for asset_id, name in zip(cached.asset_ids, cached.outcomes, strict=False):
        if asset_id == trade.asset_id:
            upper = name.strip().upper()
            if upper in ("YES", "NO"):
                return upper
            _LOG.debug(
                "gate_model.outcome_not_binary",
                tx=trade.transaction_hash,
                outcome=name,
            )
            return ""
    _LOG.debug(
        "gate_model.asset_id_not_found",
        tx=trade.transaction_hash,
        asset_id=trade.asset_id,
    )
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_gate_model.py -v -k "test_resolve_outcome_logs"`

Expected: All four PASS.

- [ ] **Step 5: Run full gate_model test suite to confirm no regression**

Run: `uv run pytest tests/detectors/test_gate_model.py -v`

Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "fix(gate_model): emit debug logs on silent outcome-resolution skips (#101)"
```

---

## Task 2: Add debug log to `evaluate()` outcome-skip early-return

**Files:**
- Modify: `src/pscanner/detectors/gate_model.py:163-169`
- Test: `tests/detectors/test_gate_model.py` (one new test)

`_resolve_outcome_side` now logs the *cause*. The caller (`evaluate()`) should also log the *effect* — that the trade was dropped before scoring — so the worker's drain progress is visible end-to-end. This complements the existing `gate_model.no_metadata` log inside `evaluate()`.

- [ ] **Step 1: Write a failing log-capture test**

Append to `tests/detectors/test_gate_model.py`:

```python
@pytest.mark.asyncio
async def test_evaluate_logs_when_outcome_unresolved(tmp_path: Path) -> None:
    """`evaluate()` emits gate_model.skip_unresolved_outcome on the early-return."""
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )  # no market_cache => _resolve_outcome_side returns ""
        trade = _make_wallet_trade(condition_id="0xc1")
        with capture_logs() as logs:
            await detector.evaluate(trade)
    finally:
        conn.close()
    events = [log["event"] for log in logs]
    assert "gate_model.skip_unresolved_outcome" in events
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_evaluate_logs_when_outcome_unresolved -v`

Expected: FAIL — `gate_model.skip_unresolved_outcome` not in events.

- [ ] **Step 3: Add the log line in `evaluate()`**

Replace lines 167-169 in `src/pscanner/detectors/gate_model.py`:

```python
        outcome_side = self._resolve_outcome_side(trade)
        if outcome_side not in ("YES", "NO"):
            _LOG.debug(
                "gate_model.skip_unresolved_outcome",
                tx=trade.transaction_hash,
            )
            return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/detectors/test_gate_model.py::test_evaluate_logs_when_outcome_unresolved -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/detectors/gate_model.py tests/detectors/test_gate_model.py
git commit -m "fix(gate_model): log evaluate() skip when outcome unresolved (#101)"
```

---

## Task 3: Enforce `markets.enabled` dependency in `Scanner.preflight()`

**Files:**
- Modify: `src/pscanner/scheduler.py:513-528`
- Test: `tests/scheduler/test_gate_model_wiring.py` (one new test plus an update to `_make_config`)

The detector silently produces zero alerts whenever the markets collector is disabled, because the markets collector is what populates `MarketCacheRepo`. Add a hard preflight check so the daemon refuses to start in this configuration with a clear message — same shape as the existing `wallet_state_live` check.

- [ ] **Step 1: Write a failing preflight test**

Append to `tests/scheduler/test_gate_model_wiring.py` after `test_preflight_passes_when_wallet_state_seeded` (around line 178):

```python
@pytest.mark.asyncio
async def test_preflight_refuses_to_start_when_markets_disabled(tmp_path: Path) -> None:
    """gate_model.enabled requires markets.enabled (issue #101)."""
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(
        artifact_dir=artifact_dir,
        gate_enabled=True,
        filter_enabled=True,
        markets_enabled=False,
    )
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        _seed_wallet_state_live(scanner)  # bypass the wallet_state_live gate
        with pytest.raises(RuntimeError, match="markets.enabled"):
            scanner.preflight()
    finally:
        await scanner.aclose()


@pytest.mark.asyncio
async def test_preflight_passes_when_markets_enabled(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = _make_config(
        artifact_dir=artifact_dir,
        gate_enabled=True,
        filter_enabled=True,
        markets_enabled=True,
    )
    clients = _make_stub_clients()
    scanner = Scanner(config=cfg, db_path=tmp_path / "daemon.sqlite3", clients=clients)
    try:
        _seed_wallet_state_live(scanner)
        scanner.preflight()  # no exception
    finally:
        await scanner.aclose()
```

Also extend `_make_config` (currently at lines 102-111) to accept the new flag — the existing tests should keep working because the default matches today's behavior:

```python
def _make_config(
    *,
    artifact_dir: Path,
    gate_enabled: bool,
    filter_enabled: bool,
    markets_enabled: bool = True,
) -> Config:
    return Config(
        gate_model=GateModelConfig(enabled=gate_enabled, artifact_dir=artifact_dir),
        gate_model_market_filter=GateModelMarketFilterConfig(enabled=filter_enabled),
        markets=MarketsConfig(enabled=markets_enabled),
    )
```

Add the import at the top of the file (alongside the existing `Config, GateModelConfig, GateModelMarketFilterConfig` import on line 17):

```python
from pscanner.config import Config, GateModelConfig, GateModelMarketFilterConfig, MarketsConfig
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py::test_preflight_refuses_to_start_when_markets_disabled tests/scheduler/test_gate_model_wiring.py::test_preflight_passes_when_markets_enabled -v`

Expected: First test FAILS (no exception raised). Second PASSES already (preflight is currently lenient on markets).

- [ ] **Step 3: Extend `preflight()` with the markets check**

Replace `preflight()` in `src/pscanner/scheduler.py:513-528`:

```python
def preflight(self) -> None:
    """Run startup checks before entering the run loop.

    When ``gate_model`` is enabled, refuses to start unless:
    - ``wallet_state_live`` has been populated via
      ``pscanner daemon bootstrap-features``.
    - ``markets.enabled`` is true (the markets collector populates the
      ``MarketCacheRepo`` that ``GateModelDetector._resolve_outcome_side``
      depends on; without it, every trade silently drops to ``""`` —
      see issue #101).
    """
    if not self._config.gate_model.enabled:
        return
    row = self._db.execute("SELECT 1 FROM wallet_state_live LIMIT 1").fetchone()
    if row is None:
        msg = (
            "gate_model.enabled=true but wallet_state_live is empty. "
            "Run `pscanner daemon bootstrap-features` first."
        )
        raise RuntimeError(msg)
    if not self._config.markets.enabled:
        msg = (
            "gate_model.enabled=true but markets.enabled=false. "
            "The gate-model detector requires the markets collector to "
            "populate MarketCacheRepo (used to map asset_id -> YES/NO). "
            "Set [markets] enabled = true in your config."
        )
        raise RuntimeError(msg)
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `uv run pytest tests/scheduler/test_gate_model_wiring.py -v -k "preflight"`

Expected: All preflight tests PASS (including the existing 3).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/scheduler.py tests/scheduler/test_gate_model_wiring.py
git commit -m "fix(scheduler): preflight enforces markets.enabled when gate_model.enabled (#101)"
```

---

## Task 4: Document the markets-collector dependency in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — gate-model bullet under `## Codebase conventions`

The CLAUDE.md gate-model bullet at the end of `## Codebase conventions` already covers wallet_state_live as a dependency. Add the markets-collector dependency to the same bullet so future agents reading the project notes don't repeat the smoke-config mistake.

- [ ] **Step 1: Locate the gate-model bullet**

Find the line starting `- **Gate-model loop (#77/#78/#79).** GateModelDetector ...` in `CLAUDE.md`. The sentence "Daemon refuses to start with `gate_model.enabled=true` if `wallet_state_live` is empty — run `pscanner daemon bootstrap-features` first." is the natural anchor.

- [ ] **Step 2: Add the markets dependency**

Append to that sentence (right after the bootstrap-features sentence):

```
Daemon also refuses to start with `gate_model.enabled=true` if `markets.enabled=false` — `MarketCacheRepo` (populated by the markets collector) is what maps `trade.asset_id` to YES/NO inside `_resolve_outcome_side`; without it every trade drops silently (see #101).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): note markets.enabled dependency for gate_model (#101)"
```

---

## Verification

Run the full project verify to confirm no regressions across the four commits:

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: zero failures, zero warnings.

Optional smoke verify (requires the daemon DB and a populated corpus):

```bash
# 1. Start daemon with gate_model.enabled=true and markets.enabled=false in config.toml.
# 2. Confirm the daemon refuses to start with the new RuntimeError ("markets.enabled=false ...").
# 3. Flip markets.enabled=true and re-run; with corpus DB freshly populated and bootstrap-features
#    run, expect debug logs `gate_model.market_not_cached` for any trade on a market the
#    markets collector hasn't yet snapshotted (one cycle after start) plus
#    `gate_model.skip_unresolved_outcome` on the same trade. After ~60s once MarketCacheRepo fills,
#    expect those logs to taper and `gate_buy` alerts to start landing.
```

---

## Self-Review

**Spec coverage:**
- Acceptance criterion 1 (each early-return path emits a distinct debug log) → Task 1.
- Acceptance criterion 2 (`evaluate()`'s outcome-side early-return emits a debug log) → Task 2.
- Acceptance criterion 3 (CLAUDE.md gate-model bullet notes markets-collector dependency) → Task 4.
- Acceptance criterion 4 (optional preflight enforces dependency) → Task 3.

All four acceptance criteria covered. The "out of scope" architectural fix for live open markets not in `corpus_markets` is correctly deferred to issue #102.

**Placeholder scan:** No TBDs or "appropriate error handling" placeholders. Every code block contains the actual content.

**Type consistency:** `_resolve_outcome_side` keeps its `str` return type. `preflight()` keeps `None` return. `_make_config` adds an optional `markets_enabled` keyword that defaults to today's behavior, so existing tests don't change. All log event names use the `gate_model.*` namespace consistent with the existing `gate_model.no_metadata` / `gate_model.queue_full` lines.
