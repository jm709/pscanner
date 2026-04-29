# Historical trade corpus — design

Date: 2026-04-29
Status: design, awaiting implementation plan

## Problem

The existing `pscanner` daemon collects live data and routes it through
detectors and paper-trading evaluators. We want a separate capability: a
historical trade corpus suitable for training a binary classifier
("did this BUY trade resolve in the trader's favor?"), then applying that
model to live trades in a future v2.

The current live system is not the right substrate for this work — it is
detector-and-alert shaped, not corpus-and-features shaped. Trying to
extend it for ML data collection mixes two concerns and accumulates
complexity without serving either well.

This spec defines a parallel `pscanner.corpus` subsystem that shares only
API clients and category taxonomy with the live system. No runtime
shared state, no shared database, no cross-imports of detector logic.

## Goals (v1)

1. Bulk-pull every BUY and SELL fill (notional ≥ $10) on every closed
   Polymarket market with total volume ≥ $10k.
2. Capture per-market resolution outcomes for those markets.
3. Materialize a `training_examples` table — one row per qualifying BUY,
   carrying ~20 features computed strictly point-in-time, plus a binary
   `label_won`.
4. Provide three idempotent, resumable CLI commands for orchestrating
   the pipeline.
5. Structure feature computation as pure functions over a
   `HistoryProvider` interface, so the same code can score live trades
   in v2 without duplication.

## Non-goals (deferred to v2 or later)

- Live trade scoring (`LiveHistoryProvider`, live ingestion → feature
  row → model output).
- Model training itself.
- Order-book features (depth, mid price). These are only available from
  the live WebSocket and cannot be reconstructed historically; including
  them would break live/historical parity. v1 uses only features
  derivable from the trade stream + market metadata.
- Watermark-based incremental rebuild of `training_examples`. v1 does a
  full streaming walk on each `build-features` run; sub-minute on
  expected corpus size.
- Per-market parallel backfill workers. A single worker is bound by the
  Polymarket data API rate limit; concurrency adds complexity for no
  throughput gain.
- Cross-references from the corpus DB into the live `pscanner.sqlite3`.
  The only data flow between systems will be a trained model artifact in
  v2, not DB joins.

## Architecture

### Subsystem layout

```
src/pscanner/corpus/
    __init__.py
    db.py              # corpus.sqlite3 schema + connection helpers
    repos.py           # CorpusMarketsRepo, CorpusTradesRepo,
                       # ResolutionsRepo, ExamplesRepo
    enumerator.py      # gamma walker: list closed markets above the
                       # volume gate
    market_walker.py   # /trades pagination for one market, idempotent
    features.py        # pure feature functions + HistoryProvider
    resolutions.py     # determine winning outcome per condition_id
    examples.py        # build training_examples by streaming
                       # corpus_trades + market_resolutions
    cli.py             # `pscanner corpus backfill | refresh |
                       # build-features`
```

### Data isolation

- `data/pscanner.sqlite3` — unchanged. Live daemon, detectors,
  paper-trading, alerts. No corpus tables.
- `data/corpus.sqlite3` — new. All corpus tables. Live system never
  opens this file.

The boundary is enforced by file separation: dropping the live DB for a
smoke run does not touch the corpus, and a long-running backfill cannot
contend for write locks against the live daemon.

### Reused dependencies

- `pscanner.poly.{gamma, data, http}` — same Polymarket clients, but
  the corpus subsystem instantiates its own `DataClient` and
  `GammaClient` so each system gets the full 50 RPM independently.
- `pscanner.categories.DEFAULT_TAXONOMY` — single source of truth for
  market categories; corpus does not introduce a parallel category
  scheme.
- `pscanner.poly.ids` — `MarketId`, `ConditionId`, `AssetId`, `EventId`,
  `EventSlug` reused as-is. **v1 adds a new `WalletAddress = NewType("WalletAddress", str)`**
  to this module so corpus code (and the existing live system, if it
  wants) can type wallet identifiers distinctly. Adding the NewType is a
  one-line change with no migration impact.

## Database schema (`data/corpus.sqlite3`)

Four primary tables plus a small key/value cursor table.

### `corpus_markets`

Qualifying markets and their backfill state.

- `condition_id TEXT PRIMARY KEY`
- `event_slug TEXT NOT NULL`
- `category TEXT` — from the existing taxonomy; `unknown` if missing
- `closed_at INTEGER NOT NULL` — unix seconds
- `total_volume_usd REAL NOT NULL` — used for the $10k gate at
  enumeration time; not updated thereafter
- `backfill_state TEXT NOT NULL` — one of
  `pending | in_progress | complete | failed`
- `last_offset_seen INTEGER` — for resumable `/trades` pagination
- `trades_pulled_count INTEGER NOT NULL DEFAULT 0`
- `error_message TEXT` — last failure if any
- `enumerated_at INTEGER NOT NULL`
- `backfill_started_at INTEGER`
- `backfill_completed_at INTEGER`

Index: `(backfill_state)` for the orchestrator's work-queue query.

### `corpus_trades`

Raw trades from market-walk, append-only, post-filtered to notional ≥ $10.

- `tx_hash TEXT NOT NULL`
- `asset_id TEXT NOT NULL`
- `wallet_address TEXT NOT NULL` — normalized to lowercase at insert
- `condition_id TEXT NOT NULL`
- `outcome_side TEXT NOT NULL` — `YES` or `NO`
- `bs TEXT NOT NULL` — `BUY` or `SELL`
- `price REAL NOT NULL` — implied probability at trade time
- `size REAL NOT NULL` — token amount
- `notional_usd REAL NOT NULL` — `price * size` at insert
- `ts INTEGER NOT NULL`
- `UNIQUE(tx_hash, asset_id, wallet_address)`

Indices:
- `(condition_id, ts)` — market-walk reads.
- `(wallet_address, ts)` — streaming feature aggregation. Load-bearing;
  without it `build-features` is O(N²).

### `market_resolutions`

One row per resolved market.

- `condition_id TEXT PRIMARY KEY`
- `winning_outcome_index INTEGER NOT NULL`
- `outcome_yes_won INTEGER NOT NULL` — 1 if YES, 0 if NO; convenience
  for the labeler
- `resolved_at INTEGER NOT NULL`
- `source TEXT NOT NULL` — `gamma` or other
- `recorded_at INTEGER NOT NULL`

### `training_examples`

Materialized feature matrix. One row per qualifying BUY in
`corpus_trades` whose market has a `market_resolutions` row.

Identity columns:
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `tx_hash TEXT NOT NULL`
- `asset_id TEXT NOT NULL`
- `wallet_address TEXT NOT NULL`
- `condition_id TEXT NOT NULL`
- `trade_ts INTEGER NOT NULL`
- `built_at INTEGER NOT NULL`
- `UNIQUE(tx_hash, asset_id, wallet_address)` — the
  `INSERT OR IGNORE` target for incremental rebuild

Trader features (point-in-time, from prior trades only):
- `prior_trades_count INTEGER NOT NULL`
- `prior_buys_count INTEGER NOT NULL`
- `prior_resolved_buys INTEGER NOT NULL`
- `prior_wins INTEGER NOT NULL`
- `prior_losses INTEGER NOT NULL`
- `win_rate REAL` — null when `prior_resolved_buys = 0`
- `avg_implied_prob_paid REAL` — null when `prior_buys_count = 0`
- `realized_edge_pp REAL` — null when either input is null
- `prior_realized_pnl_usd REAL NOT NULL DEFAULT 0`
- `avg_bet_size_usd REAL` — null when no prior buys
- `median_bet_size_usd REAL` — null when no prior buys
- `wallet_age_days REAL NOT NULL`
- `seconds_since_last_trade INTEGER` — null when this is wallet's first
  trade in the corpus
- `prior_trades_30d INTEGER NOT NULL`
- `top_category TEXT` — null when no prior trades
- `category_diversity INTEGER NOT NULL`

Trade-context features (from this fill):
- `bet_size_usd REAL NOT NULL`
- `bet_size_rel_to_avg REAL` — null when no prior buys
- `side TEXT NOT NULL` — `YES` or `NO`
- `implied_prob_at_buy REAL NOT NULL`

Market features at trade time:
- `market_category TEXT NOT NULL`
- `market_volume_so_far_usd REAL NOT NULL`
- `market_unique_traders_so_far INTEGER NOT NULL`
- `market_age_seconds INTEGER NOT NULL`
- `time_to_resolution_seconds INTEGER` — negative if traded post-close
- `last_trade_price REAL` — null when this is the market's first trade
- `price_volatility_recent REAL` — null when fewer than 2 prior trades

Label:
- `label_won INTEGER NOT NULL` — 0 or 1

Indices: `(condition_id)`, `(wallet_address)`, `(label_won)`.

### `corpus_state`

Small key/value table for cross-cutting cursors.

- `key TEXT PRIMARY KEY`
- `value TEXT NOT NULL`
- `updated_at INTEGER NOT NULL`

Initial keys: `last_gamma_sweep_ts`, `last_resolution_refresh_ts`.

### Schema notes

- No declared foreign keys. SQLite supports them but the corpus
  pipeline rebuilds `training_examples` wholesale and may legitimately
  reference markets without resolutions yet. Indices over keys are
  enough.
- No `wallet_feature_state` cache table. The streaming feature pipeline
  recomputes per-wallet running aggregates on every `build-features`
  run. Materializing them would create a drift source for negligible
  speedup at v1 corpus size.

## Backfill orchestration

Three CLI commands. All idempotent. All resumable from DB state.

### `pscanner corpus backfill`

Bulk historical pull, runs to completion (1-7 hours expected).

1. **Enumerate.** Page `gamma /events?closed=true`; expand each event
   into its markets. For each market, read `total_volume_usd` from gamma
   metadata. Markets ≥ $10k get inserted into `corpus_markets` with
   `backfill_state='pending'`. Markets below the gate are skipped
   entirely — not recorded, not retained.
2. **Order the work queue.** Process pending markets
   largest-volume-first. Tied markets break by `closed_at` descending.
   If interrupted at any percent complete, the high-signal subset is
   captured rather than a uniform sample of small markets.
3. **Pull trades.** For each pending market, mark `in_progress` and
   page `/trades?market=<condition_id>` via `DataClient.get_market_trades`.
   Filter to `notional_usd ≥ $10` at insert time. Update
   `last_offset_seen` after each page so a Ctrl-C resumes correctly.
   On full pagination, mark `complete`.
4. **Errors.** Transient HTTP errors retry with backoff via the
   existing `pscanner.poly.http` retry policy. Persistent failures mark
   `failed` with `error_message`. Re-running `backfill` retries failed
   markets.
5. **Concurrency.** Single worker. At 50 RPM data and ~500 trades per
   page, single-worker throughput is ~1.5M trades/hour — backfill is
   bound by API rate limit, not by parallelism.

### `pscanner corpus refresh`

Incremental pass, run as needed.

1. Sweep `gamma /events?closed=true` for events that closed since
   `corpus_state.last_gamma_sweep_ts`. Insert new qualifiers as
   `pending`.
2. Drain the queue using the same per-market puller as `backfill`.
3. For markets in `corpus_markets` without a `market_resolutions` row
   whose `closed_at` is in the past, fetch and record the resolution.
4. Update `corpus_state.last_gamma_sweep_ts`.

### `pscanner corpus build-features`

Rebuild the materialized training table.

1. Verify `market_resolutions` covers all markets with
   `backfill_state='complete'` and `closed_at` in the past. Refresh any
   missing resolutions via the same path as `refresh`.
2. Stream-walk `corpus_trades ORDER BY ts ASC`, maintaining in memory:
   - Per-wallet `WalletState` aggregates (counts, averages, recency,
     category exposures).
   - Per-wallet resolution heap: a min-heap of
     `(resolution_ts, condition_id, side_won)` for the wallet's
     prior buys whose markets have resolved. Before computing features
     for each trade, pop entries with `resolution_ts < trade.ts` and
     fold them into `prior_wins` / `prior_losses` / `prior_realized_pnl_usd`.
   - Per-market `MarketState` aggregates (volume so far, unique traders
     so far, recent price window).
3. For each BUY with `notional_usd ≥ $10` whose market has a resolution:
   compute features, derive `label_won`, run
   `INSERT OR IGNORE INTO training_examples`. The unique constraint
   makes incremental rebuilds correct: existing rows bounce off, new
   rows land.
4. BUYs whose markets have not yet resolved are skipped silently —
   they become labelable later when `refresh` records their
   resolution and `build-features` runs again.
5. SELLs are never written to `training_examples`. They remain in
   `corpus_trades` for potential future analysis.

`--rebuild` flag drops and recreates `training_examples` before the
walk. Use when feature definitions change.

### Rate limit budget

- Live daemon and corpus each instantiate their own `DataClient` /
  `GammaClient`, so each gets the full 50 RPM independently.
- If Polymarket's server-side limit becomes a problem when running both
  simultaneously, the corpus's `data_rpm` is a config knob; lowering it
  stretches the backfill but does not require a redesign.

## Feature pipeline

### Pure functions over `HistoryProvider`

```python
class HistoryProvider(Protocol):
    def wallet_state(self, wallet: WalletAddress, as_of_ts: int) -> WalletState: ...
    def market_state(self, condition_id: ConditionId, as_of_ts: int) -> MarketState: ...
    def market_metadata(self, condition_id: ConditionId) -> MarketMetadata: ...


def compute_features(trade: Trade, history: HistoryProvider) -> FeatureRow:
    wallet = history.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
    market = history.market_state(trade.condition_id, as_of_ts=trade.ts)
    meta = history.market_metadata(trade.condition_id)
    # ... compute and return FeatureRow
```

Critical invariant: `compute_features` takes only the trade and the
history provider. No DB handle, no network, no clock. All
non-determinism enters via `HistoryProvider`. This is what makes the
function trivially testable and guarantees live/historical equivalence.

### Two `HistoryProvider` implementations

1. **`StreamingHistoryProvider`** (v1) — used by `build-features`.
   Maintains running state in memory while walking `corpus_trades`
   chronologically. `wallet_state(w, as_of_ts)` returns the in-memory
   aggregate for `w` after consuming all trades with `ts < as_of_ts`.
   O(1) per call because the walk drives state forward; `as_of_ts` is
   always exactly the next trade's ts.
2. **`LiveHistoryProvider`** (v2) — built later. Maintains the same
   in-memory `WalletState` / `MarketState` shape, fed by the live trade
   stream. The same `compute_features` function runs unmodified.

### Pure state-update functions

```python
def apply_buy_to_state(state: WalletState, trade: Trade) -> WalletState: ...
def apply_resolution_to_state(state: WalletState, resolution: ResolvedBuy) -> WalletState: ...
def apply_trade_to_market(state: MarketState, trade: Trade) -> MarketState: ...
```

Both `StreamingHistoryProvider` (v1) and `LiveHistoryProvider` (v2) call
these same update functions. Single source of truth for "what does an
event do to running state."

### Module layout

```
src/pscanner/corpus/features.py
    WalletState, MarketState, MarketMetadata, FeatureRow   # frozen dataclasses
    HistoryProvider                                         # Protocol
    compute_features(trade, history) -> FeatureRow          # pure
    apply_buy_to_state(state, trade) -> WalletState         # pure
    apply_sell_to_state(state, trade) -> WalletState        # pure
    apply_resolution_to_state(state, ...) -> WalletState    # pure
    apply_trade_to_market(state, trade) -> MarketState      # pure
    StreamingHistoryProvider                                # used by build-features
```

## Testing

Per `pyproject.toml` `filterwarnings = ["error"]` and the existing
`FakeClock` discipline:

- **Unit tests for pure functions.** Hand-crafted `WalletState` /
  `MarketState` / `Trade` triples; assert exact feature values. Cover
  null handling (no prior history), edge cases (zero resolved buys
  → null `win_rate`), and the math identity
  `realized_edge_pp = win_rate − avg_implied_prob_paid`.
- **Unit tests for state-update functions.** Apply a sequence of events
  to `WalletState`, assert the final state. Property of associativity
  not required (events are ordered).
- **Property test on the resolution heap.** Random
  `(trade, resolution)` event sequences; invariant: after applying
  events up to time `T`, `prior_wins + prior_losses` equals the count
  of resolved prior buys with `resolution_ts < T`.
- **Integration test for `StreamingHistoryProvider`.** Small synthetic
  `corpus_trades` table; run `build-features` end-to-end; assert exact
  feature values on each emitted row. Catches state-machine bugs.
- **End-to-end CLI tests.** `respx`-mocked `/trades` and `/events`;
  assert table shape after `backfill`; assert idempotency by running
  it twice and observing zero new rows on the second run.
- **Repos tested with in-memory SQLite.** Same approach as the existing
  `tmp_db` fixture, with a corpus-specific schema-applier helper.

## Operational characteristics

### Expected v1 corpus size

- ~6-7k closed markets in gamma's catalog. After the $10k volume gate,
  ~1-2k qualifying markets.
- ~2-10M raw trades on those markets; ~1-3M after the $10 ingest filter.
- BUYs are ~60-70% of fills → ~600k-2M rows in `training_examples`.
- Disk: ~200MB-1GB SQLite.
- Backfill duration: 1-7 hours bulk pull at 50 RPM data.
- `build-features`: sub-minute to a few minutes streaming walk.

### Known small risks

- **Wallet address case-sensitivity.** Polymarket sometimes returns
  mixed-case addresses in `/activity`. Normalize to lowercase at
  insert. Consistent with the existing `wallet_trades` convention.
- **Disputed/voided resolutions.** Gamma occasionally reports
  `resolution=null` for closed-but-disputed markets. Skip those at
  `market_resolutions` build time and log a warning; their trades
  remain in `corpus_trades` unlabeled until re-fetched.
- **API throttling during backfill.** Existing per-`HttpClient` rate
  limiter handles transient throttling. If sustained, lower
  `data_rpm` config knob.
- **Markets that cross the volume gate after backfill.** In practice
  this does not occur: closed markets do not accept further trades, so
  `total_volume_usd` is fixed at close. No re-evaluation needed.

## Open questions

None at design time. All major decisions resolved in brainstorming:
binary classification target, scope (closed > $10k volume, trades ≥
$10, all wallets), architecture C (raw + materialized), separate DB
file, three-CLI orchestration, pure-function feature pipeline, full
streaming walk with `INSERT OR IGNORE` for incremental rebuilds, v1
ends with the training table (no live scoring or model training).
