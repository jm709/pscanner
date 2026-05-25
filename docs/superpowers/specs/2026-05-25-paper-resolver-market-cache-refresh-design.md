# Design — Paper-resolver market-cache refresh (#170)

Date: 2026-05-25
Issue: [#170](https://github.com/jm709/pscanner/issues/170)

## Problem

`PaperResolver` polls every 5 min for open `paper_trades` positions whose
underlying markets resolved, then writes an `exit` row to book PnL. After
~14h on the desktop with 6,153 open positions, zero exits had been booked.

Root cause is two-layer:

1. `MarketCollector` and every other `market_cache` writer fetches gamma
   with `active=True, closed=False`. Once a Polymarket market resolves it
   drops out of the `active=true` page, so `market_cache.active` stays
   pinned at `1` for that condition_id forever.
2. `paper_resolver._check_resolution` returns `None` whenever
   `cached.active` is `True`. Combined with (1), the predicate is never
   satisfied for newly-resolved markets and no exit ever lands.

Issue #170 carries the desktop empirical evidence (728 distinct stuck
condition_ids, every sampled position still on `active=1`).

## Constraint discovered during design

Gamma's `/markets?closed=true` has no `endDate` / `closeTime` sort or
filter knob. A `closed=true` sweep would scan the entire Polymarket
closed-market backlog every cycle — not a workable basis for a periodic
sweep collector. The 2-hop slug lookup
(`data.get_market_slug_by_condition_id` → `gamma.get_market_by_slug`,
which already passes `closed=true` internally) gives a per-market
lookup that handles both active and closed markets.

## Goals

1. PaperResolver books exit rows within ~1 scan cycle of a Polymarket
   market actually resolving.
2. Fix is scoped to markets we already have open paper positions on —
   gamma cost is bounded by the open-position set, not by Polymarket's
   historical closed-market count.
3. Zero new collectors, scheduler tasks, or config flags. Bug fix, not
   feature.
4. Existing `PaperResolver` tests continue to pass without modification
   (they pre-seed `market_cache.active=0` and bypass the new path).

## Non-goals

- One-shot backfill for already-stuck positions. The first natural scan
  post-deploy refreshes every open-position market and books exits for
  the resolved ones; no separate CLI step is required. At
  `gamma_rpm=50` and ~728 distinct stuck condition_ids the burst takes
  ~15 min to drain — acceptable.
- Periodic refresh of cached markets that have no open paper position.
  Only the resolver reads `market_cache.active=False` rows today; no
  other consumer needs them.
- `pscanner paper status` rendering polish (separate follow-up in
  CLAUDE.md).

## Architecture overview

```
PaperResolver._scan
  ├── list_open_positions
  ├── for each distinct (condition_id) whose cached row is None OR active=1:
  │      await refresh_market_cache_row(...)   # 2-hop lookup, upsert
  └── for each open position:
         _maybe_book_exit(pos)                  # unchanged
```

The refresh step lives entirely inside the resolver's existing 5-min
scan cycle. No new task in the scheduler `TaskGroup`, no new config
section.

## Components

### `pscanner.strategies.market_cache_refresh` (new module)

```python
async def refresh_market_cache_row(
    *,
    data_client: DataClient,
    gamma_client: GammaClient,
    market_cache: MarketCacheRepo,
    condition_id: ConditionId,
) -> bool:
    """Fetch the current state of one market and upsert into market_cache.

    Returns True iff an upsert happened. Returns False (and logs at debug)
    on slug miss / gamma miss. Returns False (and logs at warning) on
    transient exceptions — never raises into the caller.
    """
```

Log event names:

- `market_cache.refresh.no_slug` (debug) — data API returned no slug for the condition_id.
- `market_cache.refresh.no_gamma_market` (debug) — gamma returned no market for the slug.
- `market_cache.refresh.failed` (warning, with `exc_info=True`) — transient exception.
- `market_cache.refresh.ok` (info) — successful upsert; payload includes `condition_id`, `slug`, `active`.

### `pscanner.strategies.paper_trader._backfill_market_cache`

Becomes a thin wrapper around `refresh_market_cache_row` to remove the
duplicate 2-hop logic. Its existing logging contract is preserved by
calling the helper and converting the boolean into the original
`paper_trader.market_cache_backfilled` / `paper_trader.no_slug` /
`paper_trader.no_gamma_market` / `paper_trader.backfill_failed` events.
That keeps the paper-trader-side log surface stable for any operator
dashboards.

### `PaperResolver`

Ctor gains two kwargs: `data_client: DataClient`, `gamma_client: GammaClient`.

`_scan` becomes:

```python
async def _scan(self, sink: AlertSink) -> None:
    del sink
    open_positions = list(self._paper_trades.list_open_positions())
    await self._refresh_stale_markets(open_positions)
    booked = sum(self._maybe_book_exit(pos) for pos in open_positions)
    if booked:
        _LOG.info("paper_resolver.scan_completed", booked=booked)
```

`_refresh_stale_markets` deduplicates by condition_id, skips any
condition_id whose cached row is already `active=False`, and awaits
`refresh_market_cache_row` sequentially for the remainder. Sequential
(not `gather`) keeps gamma traffic predictable under the existing
shared rate limiter and avoids burst contention with other detectors.

`_maybe_book_exit` is unchanged.

### `Scheduler._build_paper_trading_detectors` (scheduler.py:489)

```python
detectors["paper_resolver"] = PaperResolver(
    config=self._config.paper_trading,
    market_cache=self._market_cache_repo,
    paper_trades=paper_trades_repo,
    data_client=self._clients.data_client,    # NEW
    gamma_client=self._clients.gamma_client,  # NEW
    clock=self._clock,
)
```

Both clients are already constructed in `_clients` and threaded into
~12 other detectors/collectors — this is a purely additive wiring
change.

## Data flow

1. Scan tick fires every `paper_trading.resolver_scan_interval_seconds` (default 300s).
2. Read open positions from SQLite.
3. Compute the set `S = {pos.condition_id for pos in open if cache(pos.cid) is None or cache(pos.cid).active}`.
4. For each cid in S (sequentially): call `refresh_market_cache_row`. On success, the `market_cache` row is upserted with gamma's current `active` + `outcome_prices`. On failure, log and move on.
5. Re-iterate open positions and run the existing `_check_resolution` / `_compute_payout` / `insert_exit` path. Positions whose markets resolved since the previous scan now match the `active=False, prices=[1,0]|[0,1]` predicate and get exit rows.

## Error handling

- All exceptions inside `refresh_market_cache_row` are caught, logged
  with `exc_info=True`, and converted to a `False` return. The caller
  treats refresh failures as "skip and retry next scan."
- `_maybe_book_exit` keeps its existing per-position try/except.
- One bad market never blocks the rest of the scan.

## Testing

New unit tests in `tests/strategies/`:

| Test | Coverage |
|------|----------|
| `test_market_cache_refresh.py::test_happy_path` | data→slug→gamma→upsert; returns True; market_cache row reflects gamma's state |
| `test_market_cache_refresh.py::test_slug_miss` | `data.get_market_slug_by_condition_id` returns None; helper returns False; no upsert; no exception |
| `test_market_cache_refresh.py::test_gamma_miss` | `gamma.get_market_by_slug` returns None; helper returns False; no upsert; no exception |
| `test_market_cache_refresh.py::test_exception_swallowed` | data client raises; helper returns False; warning logged via `capture_logs` |
| `test_paper_resolver.py::test_refreshes_stale_active_markets` | open pos on `active=1` cache row; mocks return resolved market; assert exit row written |
| `test_paper_resolver.py::test_dedup_refresh_per_scan` | 2 open positions on same condition_id; assert exactly one `data.get_market_slug_by_condition_id` call |
| `test_paper_resolver.py::test_skip_refresh_when_cache_inactive` | open pos on `active=0` cache row; assert no refresh call; existing resolution path still runs |
| `test_paper_resolver.py::test_refresh_failure_does_not_block_other_positions` | gamma raises for one cid; second cid resolves cleanly; assert exit booked for the second |

Existing `PaperResolver` tests pre-seed `market_cache.active=0` so they
already bypass the new refresh path — they should continue to pass
unchanged. The new ctor kwargs need test-side defaults; will inject
`AsyncMock()` for `data_client` and `gamma_client` in the existing test
fixtures (refresh never fires when cache is already inactive, so the
mocks never get called in those tests).

`PaperTrader` tests already cover `_backfill_market_cache`; after the
wrapper change they continue to cover the same surface (slug miss,
gamma miss, exception, happy path). No behavioural change for the
PaperTrader caller.

## Migration / rollout

None required. On first daemon start with the new code:

- First scan iterates all open positions, fires one gamma refresh per
  distinct stuck condition_id (~728 on the desktop today). At
  `gamma_rpm=50` the burst takes ~15 min, then quiesces.
- Newly-resolved positions get exit rows on the same scan they were
  refreshed in (refresh upserts, then `_maybe_book_exit` runs in the
  same `_scan` invocation).
- Subsequent scans only refresh markets whose cache row is still
  `active=True` — once a market is known-resolved, it's skipped
  forever (the cache row never flips back).

## Risks

- **Gamma RPM contention.** First scan after deploy is the worst case.
  At 50 RPM and ~728 stuck condition_ids it takes ~15 min and competes
  with smart-money refresh and the events catalog sweep (per CLAUDE.md
  cold-start notes). Should not be a sustained problem after the
  initial drain. Mitigation if it bites: stagger the resolver's first
  scan with a `clock.sleep` of a few minutes at startup, or chunk the
  refresh set per scan. Will not pre-implement; will revisit if
  observed.
- **Slow scan blocks resolver loop.** Sequential awaits on ~700 gamma
  calls is ~15 min at 50 RPM. The 5-min scan interval is shorter than
  the worst-case drain. `run_periodic` doesn't overlap iterations
  (each `_scan` completes before the next starts), so this just means
  the resolver runs less often during the drain. Acceptable for a
  one-time bootstrap.
- **Data-API rate budget.** `data.get_market_slug_by_condition_id`
  pulls from a separate RPM budget (`data_rpm = 50`) shared with
  TradeCollector. Same analysis — first-scan burst, then quiesces.
