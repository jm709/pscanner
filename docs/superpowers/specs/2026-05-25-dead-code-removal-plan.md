# Dead-code removal plan — 2026-05-25

Recovery branch: `origin/archive/pre-cleanup-2026-05-25` (points at main SHA `76a87bf`).
Any deleted file is recoverable via `git checkout archive/pre-cleanup-2026-05-25 -- <path>`.

## What we keep

- Resolved-market collection (`corpus backfill`, `corpus refresh`, `corpus subgraph-backfill`, `corpus backfill-gamma-tags`, `corpus backfill-outcome-side`)
- Feature build (DuckDB engine only) and ML training (`corpus build-features`, `ml train`)
- Copy-trading path: `SubgraphTradeCollector` → `subgraph_copy` alert → `SubgraphCopyEvaluator` → paper trades
- Manifold + Kalshi clients/walkers/db (future multi-platform expansion)
- `scripts/expand_cluster.py` (refactored in PR 4 to drop the `wallet_trades` dependency)
- `MarketCacheRepo`, `MarketCollector`, `EventCollector`, `token_resolver`, watchlist sync, paper-trading framework, alert/AlertsRepo plumbing, categories taxonomy
- `pscanner.detectors.polling.PollingDetector` (only — every other detector class is deleted; `PaperResolver` is the sole surviving subclass and lives in `pscanner.strategies.paper_resolver`)
- `pscanner.corpus._build_features_sentinel` — engine-agnostic concurrency lock for `build-features`; the DuckDB engine still uses it. Only the *python-engine* sentinel test gets deleted in PR 1.

## What we delete (overview)

| PR | Theme | Net LOC removed (est) | Depends on |
|----|------|-----------------------|------------|
| 1  | Non-DuckDB build-features engine | ~600 | — |
| 2  | 8 detectors + 6 evaluators + tick/WS stack | ~6000 | — |
| 3  | Gate-model live stack (daemon/* + MarketScopedTradeCollector) | ~1500 | PR 2 (some scheduler.py overlap) |
| 4  | Refactor `expand_cluster.py` to use DataClient | +50 / −20 | — |
| 5  | TradeCollector + wallet_trades + wallet_first_seen + Activity/Position collectors | ~1500 | PR 2, PR 4 |

Verify gate (run before opening each PR):
```
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

## Coordination files (touched by multiple PRs)

Sequence edits to these to avoid merge conflicts:

- `src/pscanner/scheduler.py` — PR 2 (rip detector wiring + tick collector), PR 3 (rip gate-model wiring + MarketScopedTradeCollector + LiveHistoryProvider), PR 5 (rip TradeCollector + Activity/Position registration). **Concurrent change from other agent:** `cool-raven-98`'s paper-resolver bug fix adds `data_client=` + `gamma_client=` kwargs to the `PaperResolver(...)` block at ~line 489. Trivial 2-line additive change; whichever PR merges second rebases. Don't touch the `PaperResolver(...)` construction in PR 2/3/5.
- `src/pscanner/config.py` — PR 2 (drop detector config blocks), PR 3 (drop gate_model + gate_model_market_filter), PR 5 (drop activity/positions).
- `src/pscanner/store/db.py` + `src/pscanner/store/repo.py` — PR 3 (drop wallet_state_live / market_state_live), PR 5 (drop wallet_trades / wallet_first_seen).
- `src/pscanner/alerts/models.py` (DetectorName Literal) — PR 2 (remove dead detector names).
- `src/pscanner/cli.py` — PR 3 (remove `daemon bootstrap-features` subcommand).
- `CLAUDE.md` — every PR trims relevant sections.

Recommended landing order: **1 → 4 → 2 → 3 → 5**. PR 1 and PR 4 are fully independent of everything and validate the workflow. PR 2 is the largest; landing it before PR 3 leaves a smaller diff for PR 3's scheduler.py edits. PR 5 lands last (depends on PR 2 removing the detector consumers and PR 4 removing the `wallet_trades` reader).

PR 1 and PR 4 can run in parallel (independent worktrees, no file overlap).

---

## PR 1 — Drop non-DuckDB build-features engine

Branch: `chore/remove-python-build-features-engine`
Worktree: `.claude/worktrees/pr1-build-features`

**Delete:**
- `src/pscanner/corpus/features.py` (the streaming Python fold path used by `--engine python`)
- `tests/corpus/test_features_compute.py`
- `tests/corpus/test_features_interaction.py`
- `tests/corpus/test_features_state.py`
- `tests/corpus/test_features_streaming.py`

**Keep:**
- `src/pscanner/corpus/_build_features_sentinel.py` — verified `corpus/cli.py:25` imports it for the DuckDB engine too. Engine-agnostic concurrency lock.
- `tests/corpus/test_build_features_sentinel.py` — keep; covers the still-live sentinel module.

**Modify:**
- `src/pscanner/corpus/cli.py` — remove `--engine python` branch, make duckdb unconditional, drop `--engine` flag entirely (or keep as no-op alias for one release if you want softer migration; default is hard removal).
- `src/pscanner/corpus/feature_projection.py` — confirm the duckdb path is the only consumer of `FEATURES` after `features.py` is gone; the registry stays since it's still the source of truth for the duckdb engine's SQL.
- `tests/corpus/test_cli_build_features.py` — drop `--engine python` cases.
- `tests/corpus/test_feature_projection_parity.py` — currently engine-vs-engine; replace with engine-vs-known-good-fixture or delete if duckdb is now the single source of truth.
- `CLAUDE.md` — trim references to "python engine" / "row-by-row streaming fold" in the build-features section.

**Verify:**
- Full pytest passes.
- `uv run pscanner corpus build-features --help` no longer mentions `--engine python`.
- Smoke: `rm -f data/corpus.sqlite3 && uv run pscanner corpus backfill --limit 5 && uv run pscanner corpus build-features` should complete without invoking the python path.

---

## PR 2 — Detectors + evaluators + tick/WS stack

Branch: `chore/remove-dead-detectors`
Worktree: `.claude/worktrees/pr2-detectors`

This is the largest PR. Split into commits by sub-area for review-friendliness.

**Delete (src):**
- `src/pscanner/detectors/velocity.py`
- `src/pscanner/detectors/cluster.py`
- `src/pscanner/detectors/whales.py`
- `src/pscanner/detectors/smart_money.py`
- `src/pscanner/detectors/mispricing.py`
- `src/pscanner/detectors/monotone.py`
- `src/pscanner/detectors/move_attribution.py`
- `src/pscanner/detectors/convergence.py`
- `src/pscanner/detectors/gate_model.py`
- `src/pscanner/detectors/trade_driven.py` (no longer used; subgraph_copy is collector-driven, not detector-driven)
- **KEEP** `src/pscanner/detectors/polling.py` — `pscanner.strategies.paper_resolver.PaperResolver` inherits `PollingDetector`. (Long-term, consider moving `PollingDetector` to `pscanner.util.loops` since it's no longer detector-specific — out of scope here.)
- `src/pscanner/detectors/base.py` — only live importer outside the deletions is `tests/test_smoke.py` (`from pscanner.detectors.base import Detector`); drop that import in PR 2.
- `src/pscanner/detectors/__init__.py` — trim to re-export only `PollingDetector` (or empty it if nothing imports from the package root).
- `src/pscanner/strategies/evaluators/velocity.py`
- `src/pscanner/strategies/evaluators/mispricing.py`
- `src/pscanner/strategies/evaluators/monotone.py`
- `src/pscanner/strategies/evaluators/smart_money.py`
- `src/pscanner/strategies/evaluators/move_attribution.py`
- `src/pscanner/strategies/evaluators/gate_model.py`
- `src/pscanner/collectors/ticks.py`
- `src/pscanner/poly/tick_stream.py`
- `src/pscanner/poly/clob_ws.py`
- `src/pscanner/alerts/worker_sink.py` — verified velocity is its only consumer. Full module deletion.
- `tests/alerts/test_worker_sink.py` — covers the deleted module.

**Delete (tests):**
- `tests/detectors/test_velocity.py`, `test_cluster.py`, `test_whales.py`, `test_smart_money.py`, `test_mispricing.py`, `test_monotone.py`, `test_move_attribution.py`, `test_convergence.py`, `test_gate_model.py`, `test_trade_driven.py`, `test_polling.py`
- `tests/strategies/evaluators/test_velocity.py`, `test_mispricing.py`, `test_monotone.py`, `test_smart_money.py`, `test_move_attribution.py`, `test_gate_model.py`
- `tests/collectors/test_ticks.py`, `test_ticks_concurrent.py`
- `tests/poly/test_tick_stream.py`, `test_clob_ws.py` (if exists)

**Modify — `src/pscanner/scheduler.py` (all explicit):**
- Drop these 11 detector imports (lines 56-65 + 36):
  - `from pscanner.alerts.worker_sink import WorkerSink`
  - `from pscanner.detectors.cluster import ClusterDetector`
  - `from pscanner.detectors.convergence import ConvergenceDetector`
  - `from pscanner.detectors.gate_model import GateModelDetector`
  - `from pscanner.detectors.mispricing import MispricingDetector`
  - `from pscanner.detectors.monotone import MonotoneDetector`
  - `from pscanner.detectors.move_attribution import MoveAttributionDetector`
  - `from pscanner.detectors.smart_money import SmartMoneyDetector`
  - `from pscanner.detectors.trade_driven import TradeDrivenDetector`
  - `from pscanner.detectors.velocity import PriceVelocityDetector`
  - `from pscanner.detectors.whales import WhalesDetector`
- Drop these collector/poly imports:
  - `from pscanner.collectors.ticks import MarketTickCollector`
  - `from pscanner.poly.clob_ws import MarketWebSocket`
  - `from pscanner.poly.tick_stream import BroadcastTickStream`
- Drop these evaluator imports (line 94-103) and keep only `SignalEvaluator`, `SmartMoneyEvaluator` (delete in same PR — see EvaluatorsConfig change below — actually delete `SmartMoneyEvaluator` too), `SubgraphCopyEvaluator`:
  - `GateModelEvaluator`, `MispricingEvaluator`, `MonotoneEvaluator`, `MoveAttributionEvaluator`, `VelocityEvaluator`, `SmartMoneyEvaluator`
- Drop `self._tick_stream` field (line 201), `self._workers: list[WorkerSink]` field (line 199), `self._detector_sinks` references for dead detector keys.
- Drop `SchedulerClients.ticks_ws` field + `MarketWebSocket()` instantiation in `_build_default_clients` (line 282).
- Drop `_build_collectors` `tick_collector` branch (lines 339-349).
- Drop `_build_detectors` branches for all 9 detectors (lines 397-470, 567-571). Keep `_build_paper_evaluators` branch for `subgraph_copy` only.
- In `_build_paper_evaluators`: drop all branches except `subgraph_copy` (lines 511-545). The gate_model branch at 537-540 dies here — `EvaluatorsConfig.gate_model` field also dies in this PR.
- Delete `_maybe_attach_velocity_detector` method entirely (lines 552-582) + its caller call site.
- Delete `_wire_trade_callbacks` method entirely (lines 210-232) + the `__init__` call to it. TradeCollector survives until PR 5 but no longer has detector subscribers.
- Delete `_wire_alert_subscribers` `MoveAttributionDetector` branch (lines 258-262). Keep `PaperTrader` branch.
- Delete whales `run_once` `MarketCacheRepo.upsert` loop (~lines 940-955).
- Drop dead-detector `isinstance` helper checks scattered through the file (e.g. lines 215-231, 228, 240, 868, 906, 932).
- **DO NOT TOUCH** the `PaperResolver(...)` construction at line 489 — `cool-raven-98` is adding `data_client=` + `gamma_client=` kwargs there.

**Modify — `src/pscanner/config.py` (explicit list):**
- Delete classes: `SmartMoneyConfig`, `MispricingConfig`, `MonotoneConfig`, `ConvergenceConfig`, `WhalesConfig`, `TicksConfig`, `VelocityConfig`, `ClusterConfig`, `MoveAttributionConfig`, `GateModelEvaluatorConfig`, `WorkerSinkConfig`, `SmartMoneyEvaluatorConfig`, `MispricingEvaluatorConfig`, `MonotoneEvaluatorConfig`, `VelocityEvaluatorConfig`, `MoveAttributionEvaluatorConfig`.
- Delete `EvaluatorsConfig` fields for `smart_money`, `move_attribution`, `velocity`, `mispricing`, `monotone`, `gate_model`. Keep `subgraph_copy`.
- Delete `Config.worker_sink` field + `Config.ticks` field + `Config.velocity` / `cluster` / `whales` / `convergence` / `smart_money` / `mispricing` / `monotone` / `move_attribution` fields.
- **DO NOT** delete `Config.gate_model` (`GateModelConfig`) or `Config.gate_model_market_filter` (`GateModelMarketFilterConfig`) — PR 3 owns those (they gate the collector + preflight that PR 3 deletes).
- `tests/test_config.py` — drop `WorkerSinkConfig` import + tests (lines 20, 64-70) and any deleted-class tests.

**Modify — `src/pscanner/alerts/models.py`:**
- Trim `DetectorName` Literal (lines 12-23) to exactly: `Literal["subgraph_copy"]`. (Verify before deleting — if the alerts schema/repo uses a wider Literal anywhere, narrow it here too.)

**Modify — `src/pscanner/strategies/evaluators/__init__.py`:**
- Drop exports of all 6 deleted evaluator classes. Keep `SignalEvaluator`, `SubgraphCopyEvaluator`.

**Modify — `src/pscanner/strategies/paper_trader.py`:**
- Remove any registration code that imports the 6 dead evaluator classes by name (if any — most wiring lives in scheduler).

**Modify — `src/pscanner/poly/models.py`:**
- Delete `WsBookMessage` (line 316) + `WsTradeMessage` + related parse helpers.

**Modify — `src/pscanner/poly/__init__.py`:**
- Drop re-exports of deleted Ws* models + `MarketWebSocket` + `BroadcastTickStream`.

**Modify — `src/pscanner/categories.py`:**
- Keep; only trim docstring references to dead detectors.

**Modify — `tests/test_scheduler.py`:**
- Drop these imports (lines 27, 28, 53, 54, 55): `MarketTickCollector`, `TradeCollector` (wait — TradeCollector survives until PR 5; only drop the test cases that exercise wiring that dies in PR 2, the import itself stays), `ConvergenceDetector`, `MoveAttributionDetector`, `PriceVelocityDetector`.
- Drop `tick_collector` assertions (lines 730, 731, 759, 767, 783).
- Drop test cases that assert dead-detector wiring.

**Modify — `tests/test_smoke.py`:**
- Drop these imports (explicit):
  - Line 19: `from pscanner.detectors.base import Detector` (Detector ABC also deleted)
  - Line 20: `from pscanner.poly.clob_ws import MarketWebSocket`
  - Lines 24-34: `WsTradeMessage`, `WsBookMessage` from the `poly.models` import (keep `Event`, `Market`, `Position`, etc.)
- Drop assertions: line 59 (`MarketWebSocket.__name__`), lines 63-74 (Ws* model assertions).
- Drop `wallet_first_seen` table assertion at line 129 — **leave for PR 5** (PR 2 only handles detector-removal smoke; PR 5 owns the table deletion).

**Modify — `tests/alerts/test_protocol.py`:**
- Drop any `WorkerSink` Protocol-compliance assertion (line 4 docstring suggests this exists).

**Modify — `CLAUDE.md`:**
- Trim sections: "Detectors", "Detector sink wiring", "Shared util modules" (WorkerSink ref), "Alert sink layering", "Velocity twin trades", "Mispricing alerts carry", "Monotone alerts carry", "Cluster detection has two paths", "Gate-model loop (#77/#78/#79)" (partial — keep #80 reference if it survives in PR 3 doc trim), "Gate-model evaluator (#80)".

**Adjacent cleanup (folded into this PR):**
- `pscanner.corpus.db.apply_read_pragmas` — flagged orphan from PR 1 (only its own self-test exercises it after `cli.py` stopped calling it). Delete the function + the self-test in `tests/corpus/test_cli.py`. ~10 lines.

**Verify:**
- Full pytest passes.
- `uv run pscanner run --once` starts and shuts down cleanly with only the subgraph copy path wired.
- Grep for orphaned config keys: `grep -rn "smart_money\|mispricing\|monotone\|velocity\|whales\|cluster\|convergence\|move_attribution\|gate_model" src/ tests/` — only legitimate references (e.g., `gate_model` doesn't appear) remain.

---

## PR 3 — Gate-model live stack

Branch: `chore/remove-gate-model-live-stack`
Worktree: `.claude/worktrees/pr3-gate-model-live`

**Delete (src):**
- `src/pscanner/daemon/live_history.py`
- `src/pscanner/daemon/bootstrap.py`
- `src/pscanner/daemon/_state_persistence.py`
- `src/pscanner/daemon/corpus_loader.py`
- `src/pscanner/daemon/__init__.py` re-exports (if no module survives in `daemon/`, delete the directory)
- `src/pscanner/collectors/market_scoped_trades.py`
- `src/pscanner/corpus/features.py` — **carried over from PR 1**. The python-fold orchestrator (`StreamingHistoryProvider`, `_UnresolvedBuy`, `_WalletAccumulator`) was kept in PR 1 because `daemon/bootstrap.py` reaches into `accum.unscheduled` / `accum.heap` for the bulk wallet_state_live write. With the daemon stack deleted in this PR, that orchestrator finally goes too. **Caveat:** `features.py` also exports shared dataclasses (`Trade`, `WalletState`, `MarketState`, `MarketMetadata`, `FeatureRow`, `_TradeFields`) + pure helpers (`apply_*_to_state`, `compute_features`, `empty_*_state`) + `_RECENT_PRICES_MAX` that `feature_projection.py` (DuckDB engine input) still needs. **Migration choice for this PR:**
  - (recommended) Move the dataclasses + pure helpers into `feature_projection.py` (single source of truth — the projection registry already imports them; collocation removes the indirection).
  - Or: keep `features.py` as a stripped-down "shared types" module containing only the dataclasses + helpers; delete only the orchestrator + Provider classes.
  Pick one in PR 3 and update CLAUDE.md's "Feature projection registry (#145)" bullet accordingly.
- `src/pscanner/testing/synthetic_trades.py` — verify post-PR-2 consumers; if only deleted daemon/* uses it, delete this too.

**Delete (tests):**
- `tests/daemon/test_bootstrap.py`
- `tests/daemon/test_live_history.py`
- `tests/daemon/test_live_history_parity.py`
- `tests/daemon/test_state_persistence_parity.py`
- `tests/collectors/test_market_scoped_trades.py`
- `tests/corpus/test_features_compute.py` — **carried over from PR 1** (covers `compute_features`; kept while gate_model still lived).
- `tests/corpus/test_features_state.py` — **carried over from PR 1** (covers `apply_*_to_state`).
- `tests/corpus/test_features_streaming.py` — **carried over from PR 1** (covers `StreamingHistoryProvider`).
  - If you take the "keep stripped features.py" migration option above, the `compute_features` + `apply_*_to_state` tests are still valuable — keep them (rename if `compute_features` moves to `feature_projection.py`, in which case move the tests too).

**Modify:**
- `src/pscanner/store/db.py` — drop `wallet_state_live` + `market_state_live` `CREATE TABLE` statements and any related migrations in `_MIGRATIONS`.
- `src/pscanner/store/repo.py` — drop `WalletStateLiveRepo` + `MarketStateLiveRepo` (or whatever they're named).
- **`src/pscanner/cli.py`** — drop **line 38** import: `from pscanner.daemon.bootstrap import run_bootstrap`. Remove `_cmd_daemon_bootstrap()` function (~lines 129-130). Remove `daemon bootstrap-features` subcommand registration (~lines 183-192).
- **`src/pscanner/scheduler.py`** (explicit):
  - Drop `from pscanner.daemon.live_history import LiveHistoryProvider`, `from pscanner.daemon.corpus_loader import load_corpus_metadata, load_corpus_resolutions_into`.
  - Drop `from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector`.
  - Drop `self._live_history_provider` field instantiation block (~lines 184-194 in `Scanner.__init__`).
  - Drop `_build_collectors` `market_scoped_trades` branch (~lines 350-357).
  - Drop `_build_detectors` `gate_model` branch (~lines 463-470).
  - Drop preflight `gate_model` block at lines **599-614** (raises if `wallet_state_live` empty or `markets.enabled=false`). Keep the `subgraph_trades`/`GRAPH_API_KEY` preflight (line ~615+).
- `src/pscanner/config.py` — drop `GateModelConfig` (line 308) + `GateModelMarketFilterConfig` (line 341) + their `Config` fields at lines 566 (`gate_model`) + 567-569 (`gate_model_market_filter`).
- `tests/test_scheduler.py` — drop bootstrap-features + market_scoped_trades + LiveHistoryProvider coverage.
- `tests/store/test_repo.py` — drop the live-state repo tests.
- `CLAUDE.md` — trim the LiveHistoryProvider / wallet_state_live / market_state_live / bootstrap-features sections; remove the "cold-start corpus loaders" and "Producer API on StreamingHistoryProvider" bullets.

**Verify:**
- Full pytest passes.
- `uv run pscanner --help` no longer surfaces `daemon bootstrap-features`.
- `uv run pscanner run --once` starts without complaining about a missing `wallet_state_live` table.

---

## PR 4 — Refactor `expand_cluster.py` to use DataClient

Branch: `chore/expand-cluster-dataclient`
Worktree: `.claude/worktrees/pr4-expand-cluster`

**Modify:**
- `scripts/expand_cluster.py` — replace the `SELECT FROM wallet_trades` query (~line 124) with an async fan-out of `DataClient.get_activity(wallet)` calls for each seed, returning the same `WalletTrade`-shaped rows the rest of the script expects. The script already imports `DataClient`, so this is mainly swapping the data source.
- Drop the `--db` / `init_db` integration if the script no longer needs the local sqlite (verify nothing else in the script reads from `wallet_trades`).
- Add a `--throttle` or rely on `DataClient`'s built-in rate limiting (token bucket lives in `RateLimitedHttpClient`).
- Update the script's docstring to reflect the new data source.

**Verify:**
- `uv run python scripts/expand_cluster.py --wallet 0x5cbd326a7f9dfac9855b9a23caee48fc097eabb0 --wallet 0x53daff4663382b86808feb77e4fcaffd94e57cc8` (Cavill seeds) produces an expansion roughly matching the historical ~190-wallet result. Exact match isn't required (live data shifts) but order of magnitude should hold.

No test changes (script has no tests).

---

## PR 5 — TradeCollector + wallet_trades + wallet_first_seen

Branch: `chore/remove-trade-collector`
Worktree: `.claude/worktrees/pr5-trade-collector`

**Delete (src):**
- `src/pscanner/collectors/trades.py`
- `src/pscanner/collectors/activity.py` (only if nothing in the surviving paths reads `wallet_activity`)
- `src/pscanner/collectors/positions.py` (only if nothing in the surviving paths reads `wallet_positions`)

**Delete (tests):**
- `tests/collectors/test_trades.py`
- `tests/collectors/test_activity.py` (if collector deleted)
- `tests/collectors/test_positions.py` (if collector deleted)

**Modify:**
- `src/pscanner/store/db.py` — drop `wallet_trades` table + indexes; drop `wallet_first_seen` table; drop `wallet_activity` + `wallet_positions` if their collectors are gone.
- `src/pscanner/store/repo.py` — drop `WalletTradesRepo`, `WalletFirstSeenRepo`, and `ActivityRepo`/`PositionsRepo` if their collectors are gone.
- `src/pscanner/scheduler.py` — drop `TradeCollector` instantiation + `ActivityCollector`/`PositionCollector` wiring + their config gates. `WatchlistSyncer` stays (subgraph copy uses it).
- `src/pscanner/config.py` — drop `ActivityConfig`, `PositionsConfig`, their `Field` declarations.
- `src/pscanner/collectors/__init__.py` — drop re-exports.
- `src/pscanner/collectors/watchlist.py` — update docstring (no longer references `pscanner.collectors.trades`).
- `tests/test_scheduler.py` — drop TradeCollector / ActivityCollector / PositionCollector coverage.
- `tests/test_smoke.py` — drop `wallet_first_seen` + `wallet_trades` table assertions.
- `tests/store/test_repo.py` — drop the deleted repo tests.
- `CLAUDE.md` — drop the `wallet_first_seen` and Cavill-cluster sections if they reference the removed table. The Cavill watchlist re-add snippet should stay; just note the watchlist itself survives (it's in `wallet_watchlist`, not `wallet_trades`).

**Verify:**
- Full pytest passes.
- `uv run pscanner run --once` starts and shuts down cleanly.
- `uv run pscanner watch 0xabc... && uv run pscanner watchlist` still works (watchlist is independent of TradeCollector).
- `uv run python scripts/expand_cluster.py --wallet 0x...` still works (PR 4 made it independent of `wallet_trades`).

---

## Subagent dispatch notes

Each subagent gets its own worktree via `EnterWorktree` (or `wt switch`). Never share a worktree between agents.

PR-1 and PR-4 are safe to dispatch in parallel as the first wave — fully independent file sets.
PR-2 should land before PR-3 to minimize scheduler.py merge conflicts.
PR-3 and PR-5 each depend on a prior PR, so dispatch them as the third wave.

When dispatching, include:
- The PR section above as the prompt context
- Reminder to run the verify gate before opening the PR
- Reminder NOT to touch coordination files outside their PR's scope
- Reference to `archive/pre-cleanup-2026-05-25` for recovery

A subagent should treat any test failure outside its PR's scope as a pre-existing issue to be flagged in the PR body, not silently fixed.
