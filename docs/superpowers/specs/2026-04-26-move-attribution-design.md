# Move-attribution detector design

Date: 2026-04-26
Status: design — pending implementation plan

## Problem

The existing `ClusterDetector` is structurally a *verifier*, not a *discoverer*.
Its candidate population is `wallet_first_seen`, which is populated only for
already-watched wallets. It can confirm coordination among wallets we already
know about, but cannot reach unwatched siblings. The Cavill cluster (≥190
wallets) was found because a market move tripped some other signal, the trade
list behind that move was inspected by hand, and a coordinated burst was
spotted visually. Only then could the cluster be brought into the watchlist
and verified.

The repeatable expansion step (seed → siblings via `/trades?market=`
fingerprint match) now exists as `scripts/expand_cluster.py`. The remaining
gap is the *first* step: turning a market move into a candidate seed.

## Goal

A new detector that, when an upstream alert names a market, identifies the
trades that drove the move, tests whether those trades look like a coordinated
burst, and — if so — auto-watchlists the contributors. The contributors then
flow naturally into `wallet_first_seen`, where the existing `ClusterDetector`
will score and verify them on its next sweep.

The new detector's only job is **bootstrapping candidate population for the
cluster detector**. It does not duplicate cluster scoring, persistence, or
emit `cluster.discovered`.

## Decisions (locked during brainstorming)

1. **Trigger** — hooks off existing alerts (`velocity`, `convergence`).
   Mispricing was originally in scope but later moved to "Out of scope"
   because its alerts don't carry a single `condition_id`. No independent
   scan.
2. **Emission** — emits a `cluster.candidate` alert *and* upserts contributors
   into `wallet_watchlist`. Does not auto-create `wallet_clusters` rows or
   auto-run cluster expansion.
3. **Similarity test** — generic coordinated-burst test (≥N distinct wallets
   on same outcome+side within a 60s bucket, low size CV). No
   pattern-specific tuning (Cavill maker-rebate fingerprint not baked in).
4. **Attribution window** — adaptive backwalk: walk back from alert moment in
   5-min steps until trade rate falls below baseline × multiplier for two
   consecutive windows, hard-capped at 2 hours.

## Architecture

A new detector, `MoveAttributionDetector`, lives at
`src/pscanner/detectors/move_attribution.py`. It subscribes to `AlertSink` and
acts on each alert whose `detector` field is in its trigger set. For each
triggering alert it:

1. **Backwalks** trades on the alerted market until the trade rate returns to
   the market's baseline.
2. **Buckets** the recovered trades by `(outcome, side, ts // 60)` and tests
   each bucket for a coordinated burst (≥`min_burst_wallets` distinct wallets,
   trade-size CV ≤ `max_burst_size_cv`).
3. For each burst hit, emits one `cluster.candidate` alert and upserts every
   contributor into `wallet_watchlist`.

Trigger plumbing reuses the existing `AlertSink.subscribe` interface; the
detector registers a sync handler (`handle_alert_sync`) that spawns the async
work as a tracked task, mirroring the pattern used by `ClusterDetector` for
trade callbacks.

The handoff to `ClusterDetector` is **intentionally asynchronous and lossy**:
contributors land on the watchlist, get polled by `TradeCollector` on its next
cycle (≤30s), populate `wallet_first_seen`, and get evaluated by the existing
discovery sweep on its next pass (≤1h). If they don't actually cluster they
sit on the watchlist consuming a small API budget; the user can `pscanner
unwatch` them.

## Components

```
MoveAttributionDetector
├── run(sink) — parks (alert-driven, not periodic)
├── handle_alert_sync(alert) — registered via AlertSink.subscribe
└── evaluate(alert) — async work
        ├── _filter_trigger(alert)         skip unless detector ∈ trigger set
        ├── _resolve_market(alert)         pull condition_id from alert.body
        ├── _backwalk(cond, alert_ts)      → (since_ts, until_ts, burst_trades)
        ├── _detect_burst(burst_trades)    → list[BurstHit]
        └── _emit_and_watchlist(hits, alert)
              ├── AlertSink.emit("cluster.candidate" alert)
              └── WatchlistRepo.upsert(addr, source="cluster.candidate", …)
```

### Key helpers

**`_backwalk`** — owns the rate-baseline + rolling-rate logic and the only
fetch on the hot path. Calls `DataClient.get_market_trades` to pull
`lookback_seconds_baseline` (default 24h) of trades on this market into a
list, newest-first. Computes `baseline_rate = median trades/minute` over the
full 24h. Then walks back from `alert_ts` in `backwalk_check_window_seconds`
(default 300) steps; at each step computes the trailing-window trade rate
from the same list. Stops when rate <
`baseline_rate × backwalk_multiplier` (default 3.0) for two consecutive
windows. Hard cap at `max_backwalk_seconds` (default 7200 = 2h). Returns
`(since_ts, until_ts, burst_trades)` where `burst_trades` is the slice of the
24h list with `since_ts ≤ ts ≤ until_ts`. No second fetch needed.

**`DataClient.get_market_trades`** — new typed method, paginates
`/trades?market=` newest-first, filters to `since_ts ≤ ts ≤ until_ts`, hard
cap at 30 pages (15k trades). Lifts the inline paginator from
`scripts/expand_cluster.py` into the typed client so the detector and the
script share the implementation.

**`_detect_burst`** — pure function. Buckets `(outcome, side, ts // 60)`. For
each bucket: count distinct wallets, compute `pstdev/mean` size CV. If
`count ≥ min_burst_wallets` AND `CV ≤ max_burst_size_cv`, append a
`BurstHit(condition_id, outcome, side, bucket_ts, wallets, n_trades,
median_size, cv)`.

**`_emit_and_watchlist`** — builds one `cluster.candidate` alert per
`BurstHit` and upserts each contributor into `wallet_watchlist` with
`source="cluster.candidate"` and
`reason="cluster.candidate-<triggering_alert_key>"`. `AlertSink.emit` handles
dedup via the existing `alerts` table primary key.

### Required changes outside the detector

1. **`DataClient.get_market_trades`** (new method): paginates
   `data-api.polymarket.com/trades?market=<conditionId>` newest-first,
   filtered to `since_ts ≤ ts ≤ until_ts`, hard cap at 30 pages (15k trades).
2. **`AlertSink.emit`** (small change): wrap each subscriber callback in
   try/except so a raising subscriber does not affect other subscribers or
   the alerts table write. Log `alert.subscriber_failed` at WARN.
3. **`WatchlistRepo.upsert`** (small tweak): add `keep_existing_reason: bool
   = True` so a re-trigger doesn't overwrite the original reason.
4. **`MoveAttributionConfig`** in `pscanner.config`:
   ```python
   class MoveAttributionConfig(_Section):
       enabled: bool = True
       trigger_detectors: tuple[str, ...] = ("velocity", "mispricing", "convergence")
       lookback_seconds_baseline: int = 86400
       backwalk_multiplier: float = 3.0
       backwalk_check_window_seconds: int = 300
       max_backwalk_seconds: int = 7200
       burst_bucket_seconds: int = 60
       min_burst_wallets: int = 4
       max_burst_size_cv: float = 0.4
       max_burst_hits_per_alert: int = 5
       max_contributors_per_burst: int = 50
   ```
   Wired into `Config` alongside the existing detector configs.
5. **Detector registration** in `scheduler.py`: the new detector instantiated,
   `clock=` injected, `sink.subscribe(detector.handle_alert_sync)` called
   during startup. `name = "move_attribution"`.
6. **Alert renderer registration**: add `"cluster.candidate"` to the
   `DetectorName` Literal in `src/pscanner/alerts/models.py` and the alert
   renderer dispatch table — otherwise the renderer will `KeyError` (per
   CLAUDE.md note).

### No new DB tables

The detector writes only to `alerts` and `wallet_watchlist`.

## Data flow

```
   velocity / mispricing / convergence detector
                │
                │  emits Alert(detector=..., body={condition_id, ...})
                ▼
            AlertSink.emit
                │  ├── writes row to alerts (dedup by alert_key)
                │  └── fan-out to subscribers (per-handler try/except)
                ▼
   MoveAttributionDetector.handle_alert_sync(alert)
                │  spawns tracked async task
                ▼
   evaluate(alert)
        ├── _filter_trigger
        ├── _resolve_market
        ├── _backwalk            (1 paginated /trades call; returns
        │                         since_ts, until_ts, burst_trades)
        ├── _detect_burst        (pure, on burst_trades)
        └── _emit_and_watchlist
                ├── AlertSink.emit("cluster.candidate")
                └── WatchlistRepo.upsert × N contributors

                                    [next TradeCollector cycle, ≤30s]
   TradeCollector polls new watchlist entries
        ├── /activity?user=<addr>
        ├── inserts wallet_trades rows
        └── _ensure_first_seen → wallet_first_seen populated

                                    [next ClusterDetector sweep, ≤1h]
   ClusterDetector.discovery_scan
        ├── reads wallet_first_seen, finds the new wallets clumped in time
        ├── runs signals A-D, scores
        └── if score ≥ threshold → emits cluster.discovered
```

API budget per upstream alert: 1 paginated `/trades?market=` fetch
(typically 1-3 pages of 500 trades each — well within the default
`data_rpm=50`).

## Error handling

Every external call is isolated and fails soft. The detector is hooked into
the alert hot path; it must not gate or delay the alerts table write.

| Failure | Response |
|---------|----------|
| Subscriber callback raises | `AlertSink.emit` per-handler try/except. Log `alert.subscriber_failed` WARN. Other subscribers and the alerts row are unaffected. |
| `alert.body` missing `condition_id` | `_resolve_market` returns None. Log `move_attribution.no_market` DEBUG. Return. |
| `/trades` HTTP error | Caught at fetch boundary. Log `move_attribution.fetch_failed` WARN with status + condition_id. Return empty list. No emit. |
| Burst window has zero trades | `_detect_burst` returns `[]`. Exit cleanly. |
| Too many burst hits per alert | Hard cap at `max_burst_hits_per_alert=5`. Log `move_attribution.hits_truncated` WARN. |
| Too many contributors per bucket | Hard cap at `max_contributors_per_burst=50`. Sort by size proximity to bucket median, keep top N. Log `move_attribution.contributors_truncated` WARN. |
| Backwalk hits 2h cap without quiescence | Use the 2h window. Log `move_attribution.backwalk_capped` INFO. |
| `WatchlistRepo.upsert` raises | Per-row try/except. Log `move_attribution.watchlist_upsert_failed` WARN. Continue with next contributor. |
| Triggering alert replayed | `cluster.candidate` alert_key is deterministic (`cluster.candidate:{cond}:{outcome}:{side}:{bucket_ts}`). `AlertSink.emit` dedupes via `alerts` table. Watchlist upserts are idempotent. |

## Testing

Three layers, using existing fixtures (`tmp_db`, `fake_clock`, `respx`).

**Unit tests for pure functions** — no fixtures needed, fastest layer.

`_detect_burst`:
- happy path: 4 wallets, same outcome+side, same 60s bucket, low size CV → 1 hit
- below threshold: 3 wallets → 0 hits
- high CV: sizes [10, 100, 1000, 5000] → 0 hits
- mixed sides: 2 BUY + 2 SELL → 0 hits (different buckets)
- cross-bucket: 4 wallets spread across 2 buckets of 2 → 0 hits
- max-hits truncation: 8 qualifying buckets → 5 hits, WARN logged
- max-contributors truncation: 1 bucket with 80 wallets → 1 hit, top 50 kept

`_backwalk` (rate logic mocked at the `DataClient.get_market_trades`
boundary via `respx`; the rolling-rate computation itself is the unit under
test):
- quiescence within 30 min → window stops there, returns trades in window
- flat-active market → window hits `max_backwalk_seconds` cap
- two consecutive sub-baseline windows required (single dip doesn't stop)
- empty 24h trade list → baseline=0 fallback returns 2h window

**Detector-level tests** — mocked `DataClient` via `respx`, real `WatchlistRepo`
+ `AlertsRepo` against `tmp_db`, `FakeClock` for deterministic timestamps.

- end-to-end: simulated velocity alert + canned `/trades` pages with a
  Cavill-shaped burst → assert one `cluster.candidate` alert row + N
  `wallet_watchlist` rows with the right reason/source
- non-trigger detector (e.g. `whales`) → no fetch, no emit
- alert.body missing condition_id → no fetch, no emit (logs only)
- `/trades` returns 500 → no emit, alert path unaffected
- duplicate trigger alert (same alert_key) → second pass is no-op
- existing watchlist row → `keep_existing_reason=True` preserves original reason
- 80 contributors in one bucket → exactly 50 watchlist upserts

**Integration smoke test** — wires `AlertSink` + `MoveAttributionDetector` + a
fake upstream detector; fake calls `sink.emit(velocity_alert)`; asserts the
flow runs end-to-end via the real subscribe / fan-out plumbing. Catches
subscription registration / async-task tracking bugs.

**Test gotcha (per CLAUDE.md):** never
`monkeypatch.setattr(asyncio, "sleep", AsyncMock())` — use `FakeClock`
injected via `clock=` ctor kwarg.

Estimated count: ~15 unit + ~7 detector + 1 wiring = ~23 tests.

**Not tested** (out of scope here):
- `ClusterDetector` discovery sweep — separate module, already covered.
  Contract: "wallets in `wallet_first_seen` get scored". If broken, fix there.
- API rate-limit interactions — `DataClient` owns its RPM budget.

## Out of scope

- **Mispricing-triggered attribution.** The mispricing detector emits
  alerts whose body has `event_id` + `markets: [...]` rather than a single
  `condition_id`. Per-market backwalk for mispricing would require either
  extending `_resolve_market` to walk the `markets` list and run one
  backwalk per market (more API budget) or changing mispricing's emission
  shape. Both options are deferrable; v1 ships with `trigger_detectors =
  ("velocity", "convergence")`.
- Independent periodic scan over markets that moved but didn't trip an
  upstream detector. Could be added later as Option B from brainstorming if
  recall on Option A feels too low.
- Pattern-specific scorers (Cavill maker-rebate fingerprint, price-pump,
  wash-trading). Generic burst is the v1; pluggable patterns layered later
  if needed.
- Auto-running `expand_cluster.py` after a `cluster.candidate` fires (the
  full Option C from brainstorming). Would consume significant API budget on
  every burst; deferred until Option B's recall is well understood.
- Bumping the `ClusterDetector` defaults (`discovery_lookback_days = 30` is
  noted as too short in CLAUDE.md). Tracked separately.
