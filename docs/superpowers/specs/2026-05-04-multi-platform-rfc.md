# Multi-platform architecture RFC — Kalshi (#36) and Manifold (#37)

Date: 2026-05-04
Status: approved, pre-implementation

---

## Decision summary

| # | Decision | Chosen option | Rationale | Blast radius |
|---|----------|---------------|-----------|--------------|
| 1 | **Schema** | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` column to shared tables; namespace Polymarket-specific tables that have no cross-platform generalisation | Leaner than full per-platform table explosion; existing detector queries are unaffected until a second platform row lands in the DB (SQLite doesn't enforce DEFAULT at query time) | ~6 table DDL migrations + `init_corpus_db` / `init_db`; 0 detector query changes until Kalshi data is inserted |
| 2 | **Identifier types** | Keep `pscanner.poly.ids` intact; add parallel `pscanner.kalshi.ids` and `pscanner.manifold.ids` modules with their own `NewType` wrappers; no shared `MarketId` supertype | Preserves all existing `ty` checking power; the cost (a few `cast` calls at collector→detector boundaries) is lower than losing cross-platform type confusion detection | ~0 changes to existing code; new modules only |
| 3 | **Detectors** | Classify into "platform-portable" (mispricing, monotone, smart-money, move-attribution) and "platform-bound" (velocity, whales, cluster, convergence); portable detectors stay source-agnostic at the logic layer and receive normalized data via a thin per-platform adapter | Detector logic is already data-shape agnostic; the coupling is in the collector that feeds it, not the detector itself | Portable detectors: 0 logic changes; per-platform adapters added in collector layer |
| 4 | **Corpus** | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` to `corpus_markets`, `corpus_trades`, `market_resolutions`, and `training_examples`; backfill existing rows | Single corpus DB remains; ML training adds a `WHERE platform = ?` gate or trains per-platform models | 4 `ALTER TABLE` migrations + `backfill_platform.py` script |

---

## Decision 1 — Schema strategy: `platform` column vs table namespacing

### Options considered

**A. Per-platform table namespacing** — `poly_markets`, `poly_trades`, `kalshi_markets`, `kalshi_trades`, etc. Every table that could hold multi-platform data gets a per-platform copy. Existing queries are zero-touch.

**B. `platform` column on shared tables** — add a `TEXT NOT NULL DEFAULT 'polymarket'` column to the tables that will hold cross-platform rows. Existing queries see only Polymarket rows until a Kalshi row lands, at which point any query that omits the filter is a latent bug — but that bug surfaces on first insert, not on deploy.

**C. Hybrid** — `platform` column on the four corpus tables (which the ML pipeline aggregates across), full namespacing on the eight daemon tables (where detector queries are tight and platform-specific).

### Choice: Option C (hybrid)

The corpus tables (`corpus_markets`, `corpus_trades`, `market_resolutions`, `training_examples`) are explicitly cross-platform aggregation targets: the ML training pipeline will eventually want to train on Kalshi and Polymarket data jointly, or compare models across platforms. A `platform` column here is the right primitive for that and costs nothing while only one platform exists.

The daemon tables that are already platform-bound need no change because they will never hold cross-platform data in the same row set. `market_cache`, `wallet_position_snapshots`, `wallet_trades`, and `event_outcome_sum_history` all encode Polymarket-specific concepts (CLOB asset IDs, Polygon wallet addresses, gamma event IDs). For Kalshi and Manifold, parallel equivalents in the same tables with a `platform` column would be structurally confusing — the columns would be semantically different. The cleaner move is to leave those tables Polymarket-only and introduce `kalshi_market_cache`, etc., when #36 lands.

The `alerts` table is platform-neutral today — the detector that emits an alert doesn't care which platform the data came from as long as the normalized data shape is right. Add `platform TEXT NOT NULL DEFAULT 'polymarket'` there too; the renderer and paper-trader already key off `detector` + `alert_key`, not any platform field.

Option A was rejected because it creates 14+ new tables before a single Kalshi request has been made. Option B (pure `platform` column everywhere) was rejected because shoehorning Polygon wallet addresses and Kalshi ticker symbols into the same `wallet_trades` table makes the schema misleading and future queries error-prone.

### Affected tables summary

| Table | Action |
|-------|--------|
| `corpus_markets` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` |
| `corpus_trades` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` |
| `market_resolutions` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` |
| `training_examples` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` |
| `alerts` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` |
| `asset_index` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` (on-chain concept; Kalshi will have its own resolver) |
| `corpus_state` | Add `platform TEXT NOT NULL DEFAULT 'polymarket'` to the PK or keep key as `polymarket:onchain_last_block`-style namespaced strings (see Migration Plan) |
| All daemon wallet/market tables | Leave as-is; Kalshi adds parallel tables in `init_db` |

---

## Decision 2 — Identifier types: per-platform `NewType` modules vs shared supertypes

### Options considered

**A. Per-platform modules** — `pscanner.poly.ids.MarketId`, `pscanner.kalshi.ids.MarketId`, `pscanner.manifold.ids.MarketId`. Each is a distinct `NewType(str)`. You can't accidentally pass a Kalshi ticker (`"PRES-2026-DEM"`) where a Polymarket hex condition ID is expected.

**B. Shared supertypes** — a single `pscanner.ids.MarketId = NewType("MarketId", str)` used across all platforms. Platform-specific variants are either removed or become thin aliases. Cross-platform confusion (a Kalshi market ID in a Polymarket-specific DB query) becomes invisible to `ty`.

### Choice: Option A (per-platform modules)

The codebase's raison d'être for `pscanner.poly.ids` is stated in the module docstring: "Mixing them produces silent bugs." That rationale applies across platforms, not just across Polymarket identifier flavors. A Kalshi ticker (`"PRES-2026-DEM"`) passed to a function expecting a Polymarket `ConditionId` (0x-prefixed 66-char hex) would silently reach the DB and corrupt a query without Option A.

Option B was rejected because it trades a concrete type-checking guarantee for superficial simplicity. The cost of per-platform modules is low: each is six lines of `NewType` definitions. At the collector→detector boundary where normalized data flows, the collector owns the translation from `kalshi.ids.MarketId` → a normalized dataclass field (e.g., a `CanonicalTrade` that holds a `str` market key valid only within its platform). The detector never sees the raw platform ID.

The `pscanner.poly.ids.WalletAddress` type is Polymarket/Ethereum-specific. Kalshi uses email-based account identifiers; Manifold uses username strings. These must not be aliased to `WalletAddress` — each platform's ids module defines whatever identifier types that platform actually needs.

Existing code: zero changes. `pscanner.poly.ids` is untouched. Kalshi work starts by creating `pscanner.kalshi.ids`.

---

## Decision 3 — Detector portability

### Classification

**Platform-portable** — detectors whose detection logic operates on normalized price/trade data and does not depend on Polymarket-specific identifier shapes or on-chain semantics:

- `MispricingDetector` — consumes `Event` + `Market` pydantic models and a `GammaClient`. Porting means providing a `KalshiEventClient` that returns the same `Event`/`Market` shapes (or a parallel normalized type). The sum-to-1 invariant applies to any mutex event set.
- `MonotoneDetector` — operates on per-market axis labels (date strings, numeric thresholds) and YES prices. Fully agnostic to identifier type.
- `SmartMoneyDetector` — consumes `ClosedPosition`, `LeaderboardEntry`, `Position` from `pscanner.poly.data`. These models are Polymarket-specific, but the detection logic (edge metrics, copy-trade threshold) is platform-agnostic. Porting means providing a Kalshi client that returns normalized position objects. The detector logic itself does not change.
- `MoveAttributionDetector` — operates on `wallet_trades` rows (already normalized to a `WalletTrade` dataclass). Platform-agnostic if the collector normalizes before inserting.

**Platform-bound** — detectors that are either deeply coupled to Polymarket infrastructure or whose signal is Polymarket-exclusive:

- `PriceVelocityDetector` — subscribes to `MarketWebSocket` (CLOB WS, Polymarket-specific). A Kalshi tick stream would require a new `KalshiTickStream` and a new velocity detector or a shared abstract tick stream interface. Deferred to post-#37.
- `WhalesDetector` — reads `wallet_first_seen` (Polygon wallet addresses), queries `/activity` (Polymarket data API). The signal concept (new wallet, big bet) is portable, but the data plumbing is not. Deferred.
- `ClusterDetector` — reads `wallet_first_seen` and `wallet_trades`, both Polymarket-specific. Deferred.
- `ConvergenceDetector` — reads `event_snapshots` and `event_outcome_sum_history`, both populated by Polymarket's gamma event structure. Deferred.

### What "portable" means in practice

Portable detectors do not change their detection logic. They gain a `platform: str` parameter to tag alerts they emit. If a future Kalshi collector populates `market_cache` with Kalshi entries tagged `platform='kalshi'`, the existing `MispricingDetector` may need a platform-scoped view or a per-platform instance — not a logic rewrite. The preferred approach for #36 is **separate detector instances per platform**, each wired to a platform-specific client, rather than one detector with multi-platform fan-out. This keeps each instance's logic simple and avoids any implicit cross-platform data bleed.

---

## Decision 4 — Corpus and `training_examples`

### Options considered

**A. Add `platform` column to all four corpus tables** — simple, backward-compatible via `DEFAULT 'polymarket'`, existing ML training runs `WHERE platform = 'polymarket'` or is updated to accept a platform argument.

**B. Per-platform corpus DBs** — `corpus_polymarket.sqlite3`, `corpus_kalshi.sqlite3`. Zero migration risk for existing data; clean isolation.

**C. Per-platform tables inside one corpus DB** — `poly_corpus_trades`, `kalshi_corpus_trades`, etc.

### Choice: Option A

Option B was rejected because the ML pipeline would need to join across DB files to train a cross-platform model, and `init_corpus_db` would bifurcate into per-platform entry points. That's premature — no cross-platform training is planned yet, and you can always split later.

Option C was rejected for the same reason as Decision 1's Option A: premature table explosion.

Option A costs four migrations and one backfill script. The ML training pipeline (`pscanner ml train`) receives a new `--platform` argument defaulting to `polymarket`; internally it adds `WHERE platform = ?` to every corpus query. The `training_examples` table gains `platform` as part of the feature row so future cross-platform models can use it as a categorical feature.

The `training_examples` UNIQUE constraint on `(tx_hash, asset_id, wallet_address)` remains valid because `tx_hash` is Polymarket's on-chain transaction hash (globally unique within Polygon). Kalshi trades will have a different `tx_hash` format (or a synthetic one); uniqueness holds as long as the namespacing is consistent. Note this in the Kalshi collector implementation.

---

## Migration plan

### Corpus DB (`data/corpus.sqlite3`)

All migrations go into `corpus/db.py:_MIGRATIONS`. All are additive `ALTER TABLE` statements using the existing idempotent pattern.

| Table | Migration | Backfill |
|-------|-----------|---------|
| `corpus_markets` | `ALTER TABLE corpus_markets ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'` | Rows inserted before migration get `'polymarket'` via DEFAULT |
| `corpus_trades` | `ALTER TABLE corpus_trades ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'` | Same |
| `market_resolutions` | `ALTER TABLE market_resolutions ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'` | Same |
| `training_examples` | `ALTER TABLE training_examples ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'` | Same |
| `asset_index` | `ALTER TABLE asset_index ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'` | Same; on-chain asset IDs are Polygon-specific — a Kalshi resolver would populate its own rows |
| `corpus_state` | No DDL change. Namespace keys: existing key `onchain_last_block` is renamed to `polymarket:onchain_last_block` via a one-time data migration in `_apply_migrations`. New platforms use `kalshi:...` etc. | One `UPDATE corpus_state SET key = 'polymarket:' || key WHERE key NOT LIKE '%:%'` migration |

Index additions after `platform` column is added:

- `CREATE INDEX IF NOT EXISTS idx_corpus_markets_platform ON corpus_markets(platform)`
- `CREATE INDEX IF NOT EXISTS idx_corpus_trades_platform ON corpus_trades(platform, condition_id, ts)`
- `CREATE INDEX IF NOT EXISTS idx_training_examples_platform ON training_examples(platform)`

### Daemon DB (`data/pscanner.sqlite3`)

All migrations go into `store/db.py:_MIGRATIONS`.

| Table | Migration |
|-------|-----------|
| `alerts` | `ALTER TABLE alerts ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'` |
| `market_cache` | No change. Remains Polymarket-only. Kalshi adds `kalshi_market_cache` as a new CREATE TABLE in `_SCHEMA_STATEMENTS`. |
| `wallet_position_snapshots` | No change. Polymarket-only. |
| `wallet_first_seen` | No change. Polygon wallet addresses only. |
| `wallet_trades` | No change. Polymarket-only. |
| `wallet_positions_history` | No change. Polymarket-only. |
| `wallet_activity_events` | No change. Polymarket-only. |
| `market_snapshots` | No change. Polymarket-only. |
| `event_snapshots` | No change. Polymarket-only. |
| `event_outcome_sum_history` | No change. Polymarket-only. |
| `tracked_wallets` | No change. Polymarket-only; Kalshi account tracking adds a parallel `kalshi_tracked_accounts` table. |
| `tracked_wallet_categories` | No change. Polymarket-only. |
| `event_tag_cache` | No change. The `event_id → event_slug` rename wart (CLAUDE.md: "legacy name") is adjacent cleanup, not in scope for this RFC. Flag: a follow-up migration in `store/db.py` to rename the column to `event_slug` was already applied per CLAUDE.md; verify it's idempotent before #36 lands. |
| `market_ticks` | No change. CLOB asset IDs, Polymarket-only. |
| `wallet_clusters` | No change. Polygon wallet-based. |
| `wallet_cluster_members` | No change. |
| `paper_trades` | `ALTER TABLE paper_trades ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'`. Paper-trading evaluators will attach the alert's platform; the `paper status` renderer should group by platform eventually but doesn't need to for #36 v1. |

### New tables added in #36 (Kalshi)

These are new `CREATE TABLE IF NOT EXISTS` blocks added to `init_db`:

- `kalshi_market_cache` — maps Kalshi ticker → title, yes_price, volume, etc.
- `kalshi_account_cache` — equivalent of `tracked_wallets` for Kalshi user accounts.

### One-time backfill script

`scripts/backfill_platform.py` — idempotent, reads all rows in affected tables and confirms `platform = 'polymarket'` (the DEFAULT handles this automatically for SQLite; the script is a verification tool, not a data writer).

---

## Order of operations for #36 (Kalshi) and #37 (Manifold)

Dependencies flow strictly top-to-bottom. No PR depends on anything not yet merged.

### PR A — Schema migrations (blocker for everything)

- Add `platform` column to `corpus_markets`, `corpus_trades`, `market_resolutions`, `training_examples`, `asset_index` in `corpus/db.py:_MIGRATIONS`.
- Add `platform` column to `alerts` and `paper_trades` in `store/db.py:_MIGRATIONS`.
- Namespace `corpus_state` keys with `polymarket:` prefix.
- Add corpus `platform` indexes.
- Update `pscanner ml train` to accept `--platform` (default `polymarket`); add `WHERE platform = ?` to training queries.
- Tests: verify migration idempotency on in-memory DB; verify `corpus_state` key rename doesn't break `onchain-backfill` resume logic.

No detector or collector code changes.

### PR B — `pscanner.kalshi.ids` module

- `src/pscanner/kalshi/__init__.py` + `src/pscanner/kalshi/ids.py`: define `MarketTicker`, `SeriesId`, `AccountId` as `NewType[str]`.
- No clients, no collectors, no tests beyond a trivial smoke that the module is importable and `ty` passes.

Depends on: nothing (can be merged independently of PR A, but ship after A to keep the branch clean).

### PR C — Kalshi HTTP client

- `src/pscanner/kalshi/http.py`: thin `httpx.AsyncClient` wrapper with auth headers and base URL.
- `src/pscanner/kalshi/models.py`: Pydantic models for Kalshi REST payloads (market, event, trade, position).
- Tests: respx-mocked unit tests for each model.

Depends on: PR B (needs `kalshi.ids`).

### PR D — Kalshi market cache + collector

- `kalshi_market_cache` CREATE TABLE added to `store/db.py:_SCHEMA_STATEMENTS`.
- `src/pscanner/kalshi/collectors/markets.py`: polls Kalshi `/markets`, populates `kalshi_market_cache`.
- Tests: respx mocks, tmp_db fixture.

Depends on: PR A (for `init_db` migration path), PR C.

### PR E — Kalshi corpus enumerator + backfill

- `src/pscanner/kalshi/corpus/enumerator.py`: walks closed Kalshi markets → `corpus_markets` rows with `platform='kalshi'`.
- `src/pscanner/kalshi/corpus/backfill.py`: fetches Kalshi trade history → `corpus_trades` rows with `platform='kalshi'`.
- CLI: `pscanner corpus kalshi-backfill [--from-date DATE]`.
- Tests: mocked Kalshi API responses, verify `platform='kalshi'` on inserted rows.

Depends on: PR A, PR D.

### PR F — Kalshi-portable detector instances (#36 done)

- Wire `MispricingDetector` and `MonotoneDetector` to a `KalshiEventClient` adapter that returns `Event`/`Market` shaped data from Kalshi markets. The adapters live in `src/pscanner/kalshi/adapters.py`.
- Scheduler creates per-platform detector instances; alert body includes `platform` field.
- Paper-trading evaluators are unmodified — they parse alert body fields, not platform.
- Tests: smoke that detector emits `platform='kalshi'` alerts when fed Kalshi-shaped events.

Depends on: PR D (needs Kalshi market data in cache).

### PR G — `pscanner.manifold.ids` + client (#37 start)

Same structure as PRs B and C but for Manifold. Manifold uses `slug`-based market IDs and UUIDs for users; define `ManifoldMarketId`, `ManifoldUserId`, `ManifoldSlug` in `manifold/ids.py`.

Depends on: nothing (parallel to Kalshi PRs after PR A lands).

### PR H — Manifold corpus enumerator + detector instances (#37 done)

Same structure as PRs E and F for Manifold.

Depends on: PR A, PR G.

---

## Non-goals

The following are explicitly out of scope for #36 and #37:

**Cross-platform arbitrage detection.** Detecting the same event mispriced differently on Polymarket vs Kalshi requires joining live order books across platforms in real time. This codebase has no cross-platform event-identity mapping and no plan to build one.

**Unified market identity.** There is no `canonical_market_id` that maps a Polymarket condition ID to a Kalshi ticker for the "same" underlying event. No such mapping will be built as part of #36 or #37.

**Multi-outcome market support beyond YES/NO.** Manifold supports many-option markets (free-response, multiple-choice). The corpus schema (`outcome_side TEXT`) and the ML features assume binary outcomes. Multi-outcome support is deferred; the Manifold enumerator in PR H will skip non-binary markets on first pass.

**`pscanner status` / `paper status` cross-platform UI.** The terminal renderer will show Kalshi and Manifold alerts in the same feed with a `platform` badge. No layout redesign, no per-platform tabs.

**Unified wallet/account tracking across platforms.** A Polymarket wallet (Polygon address) and a Kalshi account (email/UUID) are not the same entity. No attempt to link them is made in #36 or #37.

**Migrating platform-bound detectors (velocity, whales, cluster, convergence) to Kalshi/Manifold.** These require platform-specific infrastructure (Kalshi WebSocket, Manifold activity feed) and are deferred past #37.

**Backfilling `corpus_state` keys with `polymarket:` prefix for historical on-disk corpora.** The migration handles it idempotently; any corpus opened before the migration will have its keys renamed on next `init_corpus_db()` call. No manual intervention needed.
