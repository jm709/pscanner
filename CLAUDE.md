# pscanner — Claude Code notes

Polymarket data-collection daemon. Python 3.13 + uv + ruff + ty + pytest.
~5K LOC, 17 SQLite tables, 6 detectors, 7 collectors. 503 tests.

## Quick verify
`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`

## Polymarket API quirks (will bite you)
- `/v1/closed-positions` returns ONLY winning positions, hard-capped at 50. Use `/positions?user=X&closed=true&limit=500` for the real settled-positions list (winners + losers).
- Leaderboard lives at `https://lb-api.polymarket.com/profit` (NOT data-api). Window values are `1d`/`7d`/`all` — not `day`/`week`.
- Closed-position payloads expose `eventSlug`, NOT numeric `eventId`. The `event_tag_cache.event_id` column actually stores slugs (legacy name).
- `gamma /markets` does NOT return `event_id`. Backfill via `MarketCacheRepo.get(market_id)` which has it from the `/events` path.
- `gamma /events/{id}` returns 422 for slugs — use `GammaClient.get_event_by_slug` when only slug is known.
- Public WS market channel emits `book` + `price_change` ONLY — never `trade` events with wallet info. Per-wallet trades require authenticated `/ws/user` (we don't use it; trade collection is REST-polled `/activity`).
- `WsBookMessage`: `bids`/`asks`/`last_trade_price`/`price_changes` are **top-level fields**, not under `msg.data` (which is a freeform fallback that's empty in practice).
- Default `gamma_rpm = 50`, `data_rpm = 50`. Cold-start bottlenecks: smart-money refresh + events catalog sweep both compete for gamma budget; cluster + smart-money depend on watched-wallet `wallet_first_seen` populated by TradeCollector.

## Test gotchas
- `pyproject.toml` has `filterwarnings = ["error"]` — every warning fails tests. Clean up resources (httpx/respx fixtures especially).
- NEVER `monkeypatch.setattr(asyncio, "sleep", AsyncMock())` — deadlocks the suite (sibling detector loops become CPU spinners). Use `FakeClock` from `pscanner.util.clock` instead; inject via `clock=` ctor kwarg, drive with `await fake_clock.advance(seconds)`.
- Shared fixtures in `tests/conftest.py`: `tmp_db` (in-memory SQLite with schema applied), `fake_clock`.
- Detector mocks: prefer real `AlertsRepo` against `tmp_db` over MagicMock when testing dedupe / persistence behavior.

## Codebase conventions
- **Detectors**: 3 lifecycle patterns. Polling → inherit `PollingDetector`. Trade-callback → inherit `TradeDrivenDetector` (callback pattern, `run()` parks). Stream-driven → take `tick_stream: TickStream` and `async for` it. Hybrid (whales, cluster) → compose patterns by hand.
- Every detector with a sleeping run-loop accepts `clock: Clock | None = None`.
- **Identifiers**: 5 distinct types in `pscanner.poly.ids` — `MarketId`, `ConditionId`, `AssetId`, `EventId`, `EventSlug`. They're `NewType[str]`; `ty check` catches mis-uses.
- **Schema migrations**: idempotent `ALTER TABLE` in `_MIGRATIONS` tuple, wrapped by `_apply_migrations` swallowing `"duplicate column name"` and `"no such column"` `OperationalError`s. CREATE statements are `IF NOT EXISTS` in `_SCHEMA_STATEMENTS`.
- **Categories** (smart-money / convergence / mispricing): single source of truth in `pscanner.categories.DEFAULT_TAXONOMY`. Don't hardcode `"sports"`/`"esports"` strings in detectors.
- **Velocity alerts** are deduped by `condition_id` (not asset_id) so YES/NO twin alerts on a binary market collapse. Falls back to asset_id when market_cache lookup fails.
- **`wallet_first_seen`** is populated unconditionally by `TradeCollector._ensure_first_seen` on every poll cycle (TTL-gated 24h). Cluster detector + whales-detector age filter both depend on this.

## Build orchestration (when shipping multi-issue waves)
- Use `git worktree add /home/macph/projects/pscanner-worktrees/<name> -b <branch>` for parallel sub-agents (per global standards).
- Avoid `Closes #N` in commit bodies if the issue number is uncertain — direct push to main auto-closes the wrong issue without a PR review step.
- Each detector that emits alerts must be in the `DetectorName` Literal in `src/pscanner/alerts/models.py`, otherwise the renderer KeyErrors.

## CLI surface
- `pscanner run` (daemon) / `pscanner run --once` (single pass) / `pscanner status` (recent alerts)
- `pscanner watch <addr> [--reason TEXT]` / `pscanner unwatch <addr>` / `pscanner watchlist`
- DB: `./data/pscanner.sqlite3`. Drop the file for a clean smoke run.
- Smoke verification idiom: `rm -f data/pscanner.sqlite3 && timeout NNNN uv run pscanner run > /tmp/smoke.log 2>&1; echo exit=$?`

## Open follow-ups (no issues filed)
- Cluster detector default `discovery_lookback_days = 30` is too short to catch older clusters — bump to 90+ if testing against historical data.
- Cluster detector silently returns when its candidate set is empty — no INFO-level "scan ran, found 0 candidates" log. Causes confusing "no alerts, no errors, no anything" symptom.
