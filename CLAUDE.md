# pscanner — Claude Code notes

Multi-platform prediction-market data daemon. Polymarket is the primary
source (live detectors, on-chain trade backfill, paper trading); Kalshi
and Manifold ship Stage 1 REST/WS clients without detector wiring yet.
Python 3.13 + uv + ruff + ty + pytest. ~19K LOC, 30 SQLite tables,
8 detectors, 5 paper-trading evaluators, 3 platform clients. ~1072 tests.

Multi-platform architecture decisions live in
`docs/superpowers/specs/2026-05-04-multi-platform-rfc.md` (#35).

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
- `/trades` and `/activity` REST cap at `offset=3000` (server: `"max historical activity offset of 3000 exceeded"`, newest-first sort). On-chain ingest fills the gap: `pscanner corpus onchain-backfill` walks Polygon's CTF Exchange (`0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`) via `eth_getLogs` and decodes `OrderFilled` events into `corpus_trades`. Resumable via `corpus_state['onchain_last_block']`. Default RPC is `https://polygon-rpc.com/`; pass `--rpc-url` for Alchemy or other providers. CTF Exchange deployment is pinned in `cli.py` as `_DEFAULT_FROM_BLOCK = 31_478_500` (verify on PolygonScan before the first historical sync).
- **Phase 3 (subgraph) supersedes the eth_getLogs corpus path for backfill.** `pscanner corpus subgraph-backfill` queries `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY` (Polymarket Orderbook subgraph). Both the `Exchange` and `NegRiskExchange` contracts write into the same `OrderFilledEvent` entity, so neg-risk markets are auto-covered. The `eth_getLogs` paths (`onchain-backfill`, `onchain-backfill-targeted`) stay during the transition but will be deleted in a follow-up commit. The decoder (`pscanner.poly.onchain.decode_order_filled`) and `AssetIndexRepo` survive the deletion — they're still useful for low-level event inspection. **Schema note:** `OrderFilledEvent` has no `marketHash`/`condition` field; filter by `makerAssetId_in`/`takerAssetId_in` using `asset_index`. The subgraph stores `timestamp` directly so no `eth_getBlockByNumber` lookup is needed. Subgraph indexer's earliest event is `1744013119` (2025-04-07) — pre-cutoff markets still need the eth_getLogs path. Live validation: the 2026-05-05 run inserted 15.87M trade rows across 2,528 markets in 10h 30m.

## Manifold API quirks (will bite you)
- **Mana-denominated**: bets are in mana (play money), NOT USD. Never aggregate Manifold bet amounts into real-money totals or mix with Polymarket/Kalshi volumes.
- **No traditional orderbook**: limit orders are encoded as bets with a `limitProb` parameter. Fetch open limit orders via `GET /v0/bets?kinds=open-limit`.
- **CFMM markets**: many Manifold markets are multi-outcome (CFMM mechanism, `outcomeType != "BINARY"`). Stage 1 only supports binary YES/NO. Filter non-binary at the collector layer (`ManifoldMarket.is_binary` property). Non-binary markets still parse — `prob` will be `None`.
- **Rate limit**: 500 req/min per IP, applied globally across all endpoints. Multi-IP rotation is prohibited per Manifold ToS. The `ManifoldClient._TokenBucket` enforces this with capacity=500, rate=500/60.
- **WS auth-free**: subscribe to `global/new-bet` for the bet firehose; 30-60s pings required. `ManifoldStream` uses `websockets`' built-in `ping_interval=45, ping_timeout=20`. Unknown-topic frames (e.g. `global/new-contract`) that arrive on the same connection are silently skipped.
- **Identifiers**: hash strings, not numeric or 0x-prefixed hex. `ManifoldMarketId` and `ManifoldUserId` are `NewType[str]` wrappers — don't alias to Polymarket's `MarketId` or `ConditionId`.
- **Pagination cursor**: both `/v0/markets` and `/v0/bets` use a `before=<id>` cursor (the `id` of the last item from the previous page), not an offset. Pass `before=None` to start from the most recent.
- **Tables**: `manifold_markets`, `manifold_bets`, `manifold_users` live in `pscanner.manifold.db`. Apply via `init_manifold_tables(conn)` — separate from `init_db()` (daemon) and `init_corpus_db()` (corpus).

## Kalshi API quirks (will bite you)
- **Pricing**: cents expressed as dollar strings on the wire (`"0.0900"` = 9 cents). Convert to probability via `price_dollars` (already a float 0.0-1.0) or use the `.last_price_cents` property (integer 0-100). Contracts settle to $0 or $1.
- **Identifiers**: ticker strings (`"KXELONMARS-99"`), not hex. `KalshiMarketTicker`, `KalshiEventTicker`, `KalshiSeriesTicker` are `NewType[str]` in `pscanner.kalshi.ids` — distinct from `pscanner.poly.ids` per the multi-platform RFC. Pass `KalshiMarketTicker(...)` at call sites so `ty` catches cross-platform confusion.
- **Series fan-out**: a series (e.g. `"KXELONMARS"`) groups multiple events; an event groups multiple markets. On simple binary contracts the event ticker and market ticker are equal (e.g. both `"KXELONMARS-99"`).
- **Settlement**: $0 or $1 per share (0 or 100 cents). No mid-resolution prices.
- **Trades endpoint**: market trades live at `GET /markets/trades?ticker=TICKER`, NOT at `GET /markets/{ticker}/trades` (that path returns 404). `KalshiClient.get_market_trades` uses the correct URL.
- **Public REST is unauth**; WS streaming requires a Kalshi account + RSA-signed handshake (Stage 2, not yet implemented).
- **Base URL**: `https://api.elections.kalshi.com/trade-api/v2` (verified 2026-05-04).
- **Volume/size fields**: returned as fixed-point strings (`"1.00"`), coerced to `float` by pydantic. The `count_fp` on trades is a contract count, not a dollar amount.
- **Kalshi schema tables** (`kalshi_markets`, `kalshi_trades`, `kalshi_orderbook_snapshots`) are registered into `store/db.py:_SCHEMA_STATEMENTS` via `KALSHI_SCHEMA_STATEMENTS` from `pscanner.kalshi.db`. They are created by `init_db` alongside the Polymarket daemon tables — `tmp_db` in tests includes them automatically.

## Test gotchas
- `pyproject.toml` has `filterwarnings = ["error"]` — every warning fails tests. Clean up resources (httpx/respx fixtures especially).
- NEVER `monkeypatch.setattr(asyncio, "sleep", AsyncMock())` — deadlocks the suite (sibling detector loops become CPU spinners). Use `FakeClock` from `pscanner.util.clock` instead; inject via `clock=` ctor kwarg, drive with `await fake_clock.advance(seconds)`.
- Shared fixtures in `tests/conftest.py`: `tmp_db` (in-memory SQLite with schema applied), `fake_clock`.
- Detector mocks: prefer real `AlertsRepo` against `tmp_db` over MagicMock when testing dedupe / persistence behavior.
- **Structlog log assertions need `capture_logs`, NOT `caplog`.** `cli.py` configures structlog via `PrintLoggerFactory`, so stdlib `caplog` never sees structured events. Use `from structlog.testing import capture_logs; with capture_logs() as logs: ...; assert any(l["event"] == "..." for l in logs)`.
- `tmp_db` has `row_factory = sqlite3.Row`; rows compare via `tuple(r)`, not against raw tuples.
- **SQLite `UNIQUE` indexes treat NULLs as distinct.** The `paper_trades` unique-on-entry index uses `COALESCE(rule_variant, '')` so non-twin sources (`rule_variant=NULL`) keep per-key uniqueness while velocity twin trades (follow/fade) coexist. Don't strip the COALESCE.
- Many test files use the `# type: ignore[arg-type]  # ty:ignore[invalid-argument-type]` doubled annotation when passing string literals where `Literal` types are expected — `ty` doesn't honor mypy ignores so both are needed.
- `CorpusTradesRepo.insert_batch` silently filters out trades with `notional_usd < _NOTIONAL_FLOOR_USD` (default $10). Test fixtures inserting trades for downstream behavior must use ≥$10 notional or the rows won't land and the test will see an empty `corpus_trades`.
- For functions that read from a dict but don't mutate it, prefer `Mapping[str, object]` over `dict[str, object]` in the parameter type — `dict` is invariant in its value type so `dict[str, str]` doesn't satisfy `dict[str, object]`, but `Mapping` is covariant and accepts narrower-valued caller dicts.

## Codebase conventions
- **Detectors**: 3 lifecycle patterns. Polling → inherit `PollingDetector`. Trade-callback → inherit `TradeDrivenDetector` (callback pattern, `run()` parks). Stream-driven → take `tick_stream: TickStream` and `async for` it. Hybrid (whales, cluster) → compose patterns by hand.
- Every detector with a sleeping run-loop accepts `clock: Clock | None = None`.
- **Identifiers**: 5 distinct types in `pscanner.poly.ids` — `MarketId`, `ConditionId`, `AssetId`, `EventId`, `EventSlug`. They're `NewType[str]`; `ty check` catches mis-uses.
- **Schema migrations**: idempotent `ALTER TABLE` in `_MIGRATIONS` tuple, wrapped by `_apply_migrations` swallowing `"duplicate column name"` and `"no such column"` `OperationalError`s. CREATE statements are `IF NOT EXISTS` in `_SCHEMA_STATEMENTS`. **Always open the corpus DB via `init_corpus_db()`**, never raw `sqlite3.connect()` — the latter skips the migration step and pre-existing on-disk corpora won't auto-pick-up new tables.
- **`closed_at` / `resolved_at` are observational close times.** `CorpusMarketsRepo.mark_complete` rewrites `corpus_markets.closed_at` to `MAX(corpus_trades.ts)` when backfill finishes; `record_resolutions` then propagates that into `market_resolutions.resolved_at`. The previous behavior (placeholder `now_ts` from the enumerator) collapsed `temporal_split` into a hash split — see #40. If you ever bypass `mark_complete`, both columns become meaningless again.
- **`asset_index` table** maps `asset_id PRIMARY KEY → (condition_id, outcome_side, outcome_index)`. Built by Phase 1 of #42 and used by future on-chain ingest to resolve `OrderFilled` events to corpus trades. Backfilled from `corpus_trades` via `scripts/backfill_asset_index.py` (idempotent).
- **Categories** (smart-money / convergence / mispricing): single source of truth in `pscanner.categories.DEFAULT_TAXONOMY`. Don't hardcode `"sports"`/`"esports"` strings in detectors.
- **Velocity alerts** are deduped by `condition_id` (not asset_id) so YES/NO twin alerts on a binary market collapse. Falls back to asset_id when market_cache lookup fails.
- **`wallet_first_seen`** is populated unconditionally by `TradeCollector._ensure_first_seen` on every poll cycle (TTL-gated 24h). Cluster detector + whales-detector age filter both depend on this.
- **Cluster detection has two paths.** `ClusterDetector.discovery_scan` runs both `_iter_candidate_groups` (creation-window partition, ≤24h spread) and `_iter_co_trade_groups` (shared-obscure-markets graph, ≥3 markets in common). Both routed through `_consider_group` with a SHA256-of-addresses dedupe. Signal A (`_compute_creation_cohesion_score`) reads actual timestamps so co-occurrence-found clusters that span weeks score 0 on cohesion but can still pass via B+C+D.
- **Alert sink layering.** Three types implement `IAlertSink` (single method `async emit(alert) -> bool`): `AlertSink` (synchronous, with `subscribe()` for sync callbacks like `PaperTrader.handle_alert_sync`), and `WorkerSink` (per-detector queue-deferred, drains alerts to a wrapped `AlertSink` via a background task). Tick-driven detectors (currently velocity) get a `WorkerSink` from the scheduler; polling detectors get the raw `AlertSink`. `WorkerSink.emit` always returns `True` (enqueue-acknowledged) — losing the dedupe bool is fine because the only callers using it (`cluster.py:164,549`) are polling-driven and stay on `AlertSink`.
- **Mispricing alerts carry `target_*` body fields** (`target_condition_id`, `target_side`, `target_current_price`, `target_fair_price`) computed via proportional rebalancing (`fair[i] = current[i] / sum(current)`). Picks the leg with largest deviation; flips current/fair via `1 - x` when over-priced YES → trade NO. `MispricingEvaluator.parse` reads these directly.
- **Monotone alerts carry `strict_*` / `loose_*` body fields** (`strict_condition_id`, `loose_condition_id`, `strict_yes_price`, `loose_yes_price`, `gap`, plus `axis_kind` ∈ {"date","threshold"} and optional `axis_direction`). The evaluator emits a paired trade — `rule_variant="strict_no"` (NO on the leg that should be cheaper) + `rule_variant="loose_yes"` (YES on the leg that should be richer) — both sized at `position_fraction` (default 0.5% per leg, 1% pair total). Axis selection sorts strict-first: date axis ascending (earlier deadline first), `higher_is_stricter` thresholds descending (largest value first), `lower_is_stricter` ascending. Don't compare `MonotoneMarket.sort_key`s across markets without knowing the axis direction.
- **Paper-trading evaluators**: `pscanner.strategies.evaluators` has 5 `SignalEvaluator` Protocol implementations (smart_money, move_attribution, velocity, mispricing, monotone). Each owns its parse/quality/sizing. `PaperTrader` is a thin orchestrator: `evaluate(alert)` walks the list and runs the first acceptor's pipeline. Adding a new signal = one new class + one config block + one scheduler entry.
- **Constant sizing**: paper trades size off `starting_bankroll_usd` (constant), NOT running NAV. The `bankroll_exhausted` gate is removed by design — research config trades infinite paper bankroll for max data collection. NAV is still recorded on `paper_trades.nav_after_usd` for analysis (can be negative).
- **Velocity twin trades**: every alert spawns 2 ParsedSignals (`rule_variant="follow"` + `"fade"`) by walking `MarketCacheRepo.get_by_condition_id().outcomes` to find the opposing-side outcome name. Cache miss → returns `[]` (skip both). Each side sized at half the normal fraction (default 0.0025 per side, 0.5% pair total).
- **Severity ranking**: `SEVERITY_RANK = {"low": 0, "med": 1, "high": 2}` lives in `pscanner.alerts.models`. Evaluator gates use strict `[]` lookup on the configured `min_severity` (config-validated Literal) and lenient `.get(severity, -1)` on alert-provided values.
- **Multi-platform module shape (per RFC #35)**: each platform owns its own subpackage (`pscanner.poly`, `pscanner.kalshi`, `pscanner.manifold`) with parallel `ids.py` (per-platform `NewType[str]` modules; no shared `MarketId` supertype), `client.py`, `models.py`, `db.py`, `repos.py`. Daemon-side tables are namespaced (`kalshi_*`, `manifold_*`) — no `platform` column on existing Polymarket tables. The corpus-side `platform` column on shared tables is RFC PR A, not yet implemented. Stage 1 PRs (#36, #37) only ship the data-layer; trade collectors and detector instances per platform are deferred.

## ML training pipeline (`pscanner.ml`)
- **Optimization target**: `realized_edge_metric` from `pscanner.ml.metrics` — mean of `(label_won - implied_prob)` over taken bets (where `pred > implied`). Returns `-1.0` sentinel when fewer than `n_min` bets pass the gate (default 20) so trial configs that overfit a tiny lucky subset get penalized.
- **Inner-loop early-stopping uses the edge metric directly** (#43). `_make_edge_eval_metric` builds an xgboost `custom_metric` returning `("realized_edge", -edge)` (negated so xgboost's default minimize acts as edge-maximize). The `n_min` flat region becomes `+1.0` after negation; xgboost reads it as the worst possible value and continues training rather than stopping in the plateau. `eval_metric=logloss` stays in params as a calibration sanity baseline in train logs.
- **Outer Optuna loop** maximizes `realized_edge_metric` directly via `direction="maximize"`. The two loops are now aligned — previously the inner loop optimized logloss and Optuna optimized edge, picking trials' `best_iteration` from the wrong metric.
- **Category-aware filter metadata** (#41): `accepted_categories` field on `preprocessor.json` (default `(Category.SPORTS, Category.ESPORTS)`) plus `test_edge_filtered` in `metrics.json`. The filter is purely stored metadata — no inference path consumes it yet. When the gate model gets deployed, the inference path reads `accepted_categories` and gates copy signals on `top_category` membership. Out-of-time analysis showed sports + esports carry ~11% gross edge while thesis bleeds ~1% despite high classification accuracy — winners are already priced in.
- **Wallet-quality interaction features** (#44): four new columns on `training_examples` give the depth-3 booster "free depth" on the wallet-quality dimension:
  - `edge_confidence_weighted = realized_edge_pp * min(1, prior_resolved_buys/20)`
  - `win_rate_confidence_weighted = (win_rate - 0.5) * min(1, prior_resolved_buys/20)`
  - `is_high_quality_wallet = (prior_resolved_buys >= 20) AND (win_rate > 0.55)`
  - `bet_size_relative_to_history = bet_size_usd / median_bet_size_usd` — always 1.0 in v1 (the streaming feature provider doesn't yet maintain a running median; future v2 will)
  - 77% of wallets have ≤5 trades, so per-wallet edge estimates are noise. Pre-multiplying with sample-size confidence gives the tree a clean signal it currently has to derive through implicit splits.
- **Build features**: `pscanner corpus build-features --rebuild` (re)builds `training_examples` from `corpus_trades` + `market_resolutions`. The training pipeline reads `training_examples` via `pscanner.ml.preprocessing.load_dataset` which discovers columns dynamically via `PRAGMA table_info`, so adding new columns to the build pipeline doesn't require updating ML code.

## Build orchestration (when shipping multi-issue waves)
- Use `git worktree add /home/macph/projects/pscanner-worktrees/<name> -b <branch>` for parallel sub-agents (per global standards).
- Avoid `Closes #N` in commit bodies if the issue number is uncertain — direct push to main auto-closes the wrong issue without a PR review step.
- Each detector that emits alerts must be in the `DetectorName` Literal in `src/pscanner/alerts/models.py`, otherwise the renderer KeyErrors.

## CLI surface
- `pscanner run` (daemon) / `pscanner run --once` (single pass) / `pscanner status` (recent alerts)
- `pscanner watch <addr> [--reason TEXT]` / `pscanner unwatch <addr>` / `pscanner watchlist`
- `pscanner paper status` — aggregate NAV + per-wallet PnL + per-source breakdown table (one row per `(triggering_alert_detector, rule_variant)`).
- `pscanner corpus backfill` / `pscanner corpus refresh` — pull closed-market trade history into the corpus DB via gamma + data REST.
- `pscanner corpus onchain-backfill` / `pscanner corpus onchain-backfill-targeted` — Phase 2 on-chain trade backfill via `eth_getLogs`. The targeted variant walks per-market trade-time windows; the naive variant walks the full deployment-to-head range.
- `pscanner corpus build-features [--rebuild]` — (re)build `training_examples` from `corpus_trades` + `market_resolutions`.
- `pscanner ml train --device cuda --n-jobs 1` runs xgboost training on a CUDA GPU (~6.5 min for 100 trials). `--n-jobs 1` is required on GPU — parallel Optuna trials would compete for the 8 GB VRAM. CPU path stays default; the laptop dev host OOMs at ~7.4 GB on the full corpus (no GPU + 7.6 GB host). See `LOCAL_NOTES.md` (gitignored) for the desktop training-box workflow.
- `pscanner corpus onchain-backfill [--from-block N] [--to-block N] [--rpc-url URL] [--chunk-size N] [--max-blocks N] [--rpm N]` — walk Polygon CTF Exchange `OrderFilled` events into `corpus_trades`. Resumes from `corpus_state['onchain_last_block']` when `--from-block` is omitted; defaults to `_DEFAULT_FROM_BLOCK` on first run. After ingest, runs `clear_truncation_flags` to clear `corpus_markets.truncated_at_offset_cap` for markets whose on-chain trade count ≥ 3000.
- `pscanner corpus onchain-backfill-targeted [--rpc-url URL] [--chunk-size N] [--block-slack N] [--limit N] [--rpm N]` — per-market on-chain backfill for markets flagged `truncated_at_offset_cap=1`. Walks each market's trade-time-window block range, filters `OrderFilled` events to that market's `asset_ids` via `asset_index`, and inserts into `corpus_trades`. Resumable via `corpus_markets.onchain_processed_at`.
- `pscanner corpus subgraph-backfill [--api-key KEY] [--subgraph-id ID] [--rpm N] [--page-size N] [--limit N]` — preferred replacement for `onchain-backfill-targeted`. Queries Polymarket's Orderbook subgraph on The Graph (`7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`), filtered server-side by `makerAssetId_in` / `takerAssetId_in` from local `asset_index`. Resumable via `corpus_markets.onchain_processed_at`. Free-tier API key (~100K queries/month) covers full corpus runs comfortably. Requires `$GRAPH_API_KEY` (or `--api-key`).
- DBs: `./data/pscanner.sqlite3` (daemon) and `./data/corpus.sqlite3` (corpus). Drop either for a clean smoke run of its respective domain.
- Smoke verification idiom: `rm -f data/pscanner.sqlite3 && timeout NNNN uv run pscanner run > /tmp/smoke.log 2>&1; echo exit=$?`
- **`scripts/expand_cluster.py`**: cluster expansion via fingerprint matching. Reads seeds' trades from local `wallet_trades`; if seeds are freshly watchlisted (no local data yet) pre-populate via `DataClient.get_activity` first. The `hi_price_rate` field in the output is sampled from the seed-overlap window only — for true behavioral fingerprint, fetch `/positions?user=X&closed=true` and inspect `avg_price` distribution.

## Known wallet clusters

Primary investigation log: `volume-farming-cluster-investigation.md` covers the
two non-Cavill operations with full methodology + reproducible queries.

### Cavill cluster (manually discovered 2026-04-25; auto-confirmed by detector with `discovery_lookback_days=365`, score=5 via co-occurrence path, tag=`mixed`)

9-wallet coordinated operation, all created **Feb 20-21 2026** (7 of 9 within a 38-minute window on Feb 20). Bimodal trade sizing ($500-999 chunks just below $1K + sub-$100 dust), 35-40% SELL rate, 57% of trades at price ≥0.95 (BUY-NO spread harvest), $0 net exposure across all 9 wallets. Behavior: market-making / Polymarket maker-rebate farming on niche long-tail markets (Henry Cavill James Bond, Cabello as Venezuelan leader, Ferran Torres top La Liga scorer, Houston Dynamo MLS Cup, Mohammad Khatami, Manchester United 2nd place EPL). Useful as a "fastest-reactor-to-mispricings" signal, NOT as an "informed insider" signal.

```
0x5cbd326a7f9dfac9855b9a23caee48fc097eabb0
0x53daff4663382b86808feb77e4fcaffd94e57cc8
0x13b775f8a46762d031cbf9a6a478fe90a81e0aaf
0x7bfbc1e83ffb9203b29f653e5367acd3a580f6f8
0xd5983aab43ef59620fda70599e30e693fd93c659
0x43d621fc31491eec23d9f696dcfb7e8923cd8ac9
0xcbd11366479deef70576a4c7c0f6eda1bc6aed42
0xf04e089482c1349d3556a36951b033094731b79b
0x5266edffc8f4737c2b9d0fa959ecae2c7b55c8cb
```

Cavill is rediscoverable post DB reset because the seed wallets' `first_activity_at` spread (86,496s = 24h 1m 36s) is just past the 24h creation-window gate, so the new co-occurrence path (T6 of `feat/cluster-organic`) finds it via shared obscure markets, scores 5 (Signal A=0, B+C+D=5), tags as `mixed`. Detection happens within 1-2 minutes of cluster scan after the daemon comes up.

Re-add to watchlist after a DB reset:
```bash
for a in 0x5cbd326a7f9dfac9855b9a23caee48fc097eabb0 0x53daff4663382b86808feb77e4fcaffd94e57cc8 \
         0x13b775f8a46762d031cbf9a6a478fe90a81e0aaf 0x7bfbc1e83ffb9203b29f653e5367acd3a580f6f8 \
         0xd5983aab43ef59620fda70599e30e693fd93c659 0x43d621fc31491eec23d9f696dcfb7e8923cd8ac9 \
         0xcbd11366479deef70576a4c7c0f6eda1bc6aed42 0xf04e089482c1349d3556a36951b033094731b79b \
         0x5266edffc8f4737c2b9d0fa959ecae2c7b55c8cb; do
  uv run pscanner watch "$a" --reason cavill-cluster-feb2026
done
```

### Volume-farming cluster (722+ wallets, Feb-Apr 2026)
Discovered 2026-04-27 during paper-trading data exploration. Sub-$10 dust trades, 9% sell rate, gradually accumulated Feb-Apr 2026 (96% in April), 7.1% WR, **−23.8% ROI** on $6.35M. Almost certainly Polymarket points/airdrop farming. Cluster detector misses them via Signal A (creation spread > 24h gate) but the new co-occurrence path catches their shape if seeded. Full investigation in `volume-farming-cluster-investigation.md`.

### Magic / long-shot cluster (17 strict + ~700 fresh, Mar-Apr 2026)
Surfaced 2026-04-28 during expanded-paper-trading smoke. 50-day creation span, 47% Mondays, 6 wallets created in the burst day. **Buy YES at sub-$0.05 on tail outcomes**, hold to resolution. 14.6% WR, **−34.6% ROI** on $1.1M. Different fingerprint from volume-farming (mid-range $25 trade size vs sub-$10 dust). Same airdrop-farming end state. Full investigation in `volume-farming-cluster-investigation.md` (Update 2026-04-28 section).

## Open follow-ups (no issues filed)
- Cluster detector default `discovery_lookback_days = 30` is too short to catch older clusters — bump to 90+ if testing against historical data.
- Cluster detector silently returns when its candidate set is empty — no INFO-level "scan ran, found 0 candidates" log. Causes confusing "no alerts, no errors, no anything" symptom.
- **`tick_stream.subscriber_queue_full` regression in long runs.** A 6h smoke surfaced 2,708 drops over a 1h 42m mid-run window (~28/min sustained). `WorkerSink` decoupled the alert-emit hot path but velocity's per-tick consume work (record + window math + cache lookup) is still synchronous. Mitigations sketched in `docs/superpowers/specs/2026-04-27-paper-trading-expansion-design.md` post-deployment-observations section. Not blocking.
- **`paper status` rendering polish.** When `resolved_count=0`, `win_rate` shows `0.0%` (indistinguishable from "all losses"). Render as `-` instead. Also: the per-wallet PnL panel groups all non-smart_money entries under an unlabeled `(no wallet)` row; should be labeled or merged with the per-source breakdown.
- **Monotone evaluator can produce orphaned single legs** when one of the two pair `condition_id`s isn't yet in `market_cache`. The `_backfill_market_cache` recovery in `PaperTrader._resolve_outcome` runs per-leg, so a missing slug for one market won't block the other. ~25% of the live-smoke alerts hit this (1 of 4). Acceptable for research mode; if it becomes a measurement problem, add a "both-or-neither" gate in `_run_pipeline` for paired-variant evaluators.
- **Delete eth_getLogs corpus path** now that #46 is live-validated. Remove `onchain_backfill.py`, `onchain_targeted.py`, `onchain_rpc.py`, `onchain_ingest.py`, and their CLI dispatch entries (`onchain-backfill`, `onchain-backfill-targeted`). Keep `pscanner.poly.onchain.decode_order_filled` and `AssetIndexRepo` — they serve low-level inspection use-cases independent of the backfill path.
- **`_TokenBucket` duplicated across 5 modules** (`pscanner.poly.http`, `pscanner.poly.onchain_rpc`, `pscanner.poly.subgraph`, `pscanner.kalshi.client`, `pscanner.manifold.client`). Extract to a shared `pscanner.util.rate_limit` module before the copies diverge.
- **`manifold_bets` schema omits `shares` and `fees`.** `ManifoldBet.shares` / `.fees` are `None` after a DB round-trip. Add columns to the schema before relying on those fields for CFMM position-sizing analysis.
- **`build-features` deque + read-connection follow-ups (#58 fixes 3-5).** The first fix (in-memory resolutions cache + skip Trade rebuild for SELLs) shipped in PR #59 and removed 16M per-trade SQLite roundtrips. The remaining fixes — deque-backed `recent_30d_trades` (cures heavy-wallet O(window-size)), separate read connection for `iter_chronological`, drop the gratuitous `dict(state.category_counts)` copy — are open. The 15.46M-row rebuild on the desktop took ~9.7 hours; expect 1-2 hours after #58 fix #3 lands.
- **`pscanner ml train` OOMs on the post-Phase-3 corpus** (12 GB WSL2, 15.46M training rows). Hits 11.82 GB RSS during data load + DMatrix construction. Either bump WSL2 memory ceiling (`.wslconfig`), or land #39 (`xgboost.DataIter` streaming DMatrix) which caps peak memory at chunk_size × n_features instead of full-corpus.
- **RFC PR A** (`platform` column on shared corpus tables) is pending — would let the ML pipeline aggregate Kalshi + Manifold + Polymarket training rows together. Stage 1 platform PRs (#36, #37) deferred this.
