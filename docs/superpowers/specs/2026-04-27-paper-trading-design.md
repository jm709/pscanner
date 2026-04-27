# Paper-trading copy-trader design

Date: 2026-04-27
Status: design — pending implementation plan

## Problem

The pscanner detectors now produce a steady stream of high-quality signals,
but there is no closed loop from "signal fires" back to "did the signal pay
off?" Without that, threshold tuning and detector-quality decisions are blind.

The first signal worth running this loop on is `smart_money`. Each fresh
`smart_money` alert is "this pre-vetted high-winrate wallet just opened a
position." A natural copy-trade strategy mirrors those entries on a virtual
bankroll and reports realized PnL once the underlying markets resolve.

## Goal

A new opt-in `PaperTrader` subscriber that mirrors `smart_money` alerts onto
a virtual bankroll, plus a periodic `PaperResolver` that books PnL when the
mirrored markets resolve. State lives in a single new `paper_trades` table.
Visibility via `pscanner paper status`.

The strategy's only job is **measuring whether following smart-money pays
off, with realistic fills and bankroll mechanics, on Polymarket-resolved
outcomes**. It does not place real orders, expose an order API, or trade on
non-`smart_money` signals — those are deferred follow-ups.

## Decisions (locked during brainstorming)

1. **Scope**: smart-money copy-trade only. Mispricing arbitrage and other
   signal sources are deferrable v2 strategies.
2. **Sizing**: cost-basis NAV × `position_fraction` (default 1%). NAV =
   `cash + Σ(price_paid on open positions)` and only changes on resolution
   (wins/losses), not on mid-flight price swings.
3. **Entry filter**: only paper-trade alerts whose source wallet has
   `tracked_wallets.weighted_edge > 0.0`. Skip if the edge is NULL.
4. **Fill price**: `market_ticks.best_ask` for the alerted outcome's asset_id;
   fall back to `last_trade_price`; skip the trade if both are null.
5. **Exit**: hold to resolution. Detect via `market_cache.active = 0` plus a
   definitive `outcome_prices_json` split (`[1.0, 0.0]` or `[0.0, 1.0]`).
6. **Lifecycle**: in-daemon `PaperTrader` subscriber + in-daemon
   `PaperResolver` polling detector. Off by default; opt-in via
   `paper_trading.enabled`.
7. **Storage**: single new `paper_trades` table holding both entries and
   exits, linked via `parent_trade_id`. Open positions = entries without a
   matching exit.
8. **Visibility**: new `pscanner paper status` CLI.

## Architecture

Two new in-daemon components:

- **`PaperTrader`** (`src/pscanner/strategies/paper_trader.py`) — subscriber
  registered via `AlertSink.subscribe(detector.handle_alert_sync)`, mirroring
  the pattern set by `MoveAttributionDetector`. Filters to `smart_money`
  alerts, runs the entry filter / sizing / fill chain, inserts an `entry`
  row into `paper_trades`.
- **`PaperResolver`** (`src/pscanner/strategies/paper_resolver.py`) —
  polling detector inheriting `PollingDetector`. On each scan
  (`resolver_scan_interval_seconds`, default 300s) iterates open positions,
  checks `market_cache` for definitive resolution, and inserts `exit` rows.

Reuses:

- `AlertSink.subscribe` — added in T2 of the move-attribution work.
- `PollingDetector` — base class for periodic-scan detectors.
- `WatchlistRegistry` + `TrackedWalletsRepo` — already keep `weighted_edge`
  fresh for tracked wallets.
- `MarketCacheRepo` — already polled by `MarketsCollector`; the resolver
  piggybacks on its updates rather than running its own gamma poll.

## Components

```
PaperTrader (alert-driven)
├── name = "paper_trader"
├── handle_alert_sync(alert)        — registered via AlertSink.subscribe
└── evaluate(alert)
        ├── _filter_smart_money(alert)        skip non-smart-money / wrong shape
        ├── _check_wallet_edge(wallet)        skip if edge ≤ min_weighted_edge or NULL
        ├── _resolve_outcome(cond, name)      → (asset_id, fill_price) | None
        ├── _compute_nav()                    → current cost-basis NAV
        ├── _size_trade(nav, fill_price)      → (shares, cost_usd) | None
        └── _record_entry(...)                INSERT INTO paper_trades

PaperResolver (periodic, inherits PollingDetector)
├── name = "paper_resolver"
└── scan(now)
        ├── _open_positions()                 entries with no matching exit
        ├── for each open position:
        │     _check_resolution(pos)          → ResolvedOutcome | None
        │     _compute_payout(pos, resolved)  payout_per_share ∈ {0.0, 1.0}
        └── _record_exit(...)                 INSERT INTO paper_trades
```

### Key helpers (testable in isolation)

- **`compute_cost_basis_nav(conn, starting_bankroll) -> float`** — used by
  both sizing and the status CLI. NAV = `starting_bankroll + Σ(exit.cost_usd
  − parent_entry.cost_usd) over resolved positions`. Equivalently:
  `starting_bankroll + realized_pnl`. Open positions don't move NAV.
- **`resolve_outcome_to_asset(market_cache_row, outcome_name) -> (asset_id,
  normalized_name) | None`** — maps the smart-money body's `"side": "oilers"`
  to a concrete `asset_id` plus normalized outcome label, using the cached
  outcomes on the market. Case- and whitespace-tolerant.

### `PaperTradingConfig` (in `pscanner.config`)

```python
class PaperTradingConfig(_Section):
    enabled: bool = False                          # opt-in
    starting_bankroll_usd: float = 1000.0
    position_fraction: float = 0.01                # 1% of cost-basis NAV
    min_weighted_edge: float = 0.0
    min_position_cost_usd: float = 0.50            # below this, skip with size_too_small
    resolver_scan_interval_seconds: float = 300.0  # 5 min
```

Wired into root `Config` alongside the existing detector configs.

### Schema additions (one new table)

```sql
CREATE TABLE IF NOT EXISTS paper_trades (
  trade_id             INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_kind           TEXT    NOT NULL,        -- 'entry' | 'exit'
  triggering_alert_key TEXT,                    -- entries: smart-money alert key
  parent_trade_id      INTEGER,                 -- exits: trade_id of the entry
  source_wallet        TEXT,                    -- the smart-money wallet we mirrored
  condition_id         TEXT    NOT NULL,
  asset_id             TEXT    NOT NULL,
  outcome              TEXT    NOT NULL,        -- e.g. 'oilers', 'yes', 'no'
  shares               REAL    NOT NULL,
  fill_price           REAL    NOT NULL,        -- per share
  cost_usd             REAL    NOT NULL,        -- entries: paid; exits: received
  nav_after_usd        REAL    NOT NULL,        -- cost-basis NAV snapshot post-trade
  ts                   INTEGER NOT NULL,
  FOREIGN KEY (parent_trade_id) REFERENCES paper_trades(trade_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_trades_alert_key
  ON paper_trades(triggering_alert_key)
  WHERE trade_kind = 'entry' AND triggering_alert_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_paper_trades_open
  ON paper_trades(condition_id, asset_id) WHERE trade_kind = 'entry';

CREATE INDEX IF NOT EXISTS idx_paper_trades_parent
  ON paper_trades(parent_trade_id);
```

Open-position predicate:

```sql
SELECT * FROM paper_trades e
 WHERE e.trade_kind = 'entry'
   AND NOT EXISTS (
     SELECT 1 FROM paper_trades x WHERE x.parent_trade_id = e.trade_id
   );
```

### `pscanner paper status` CLI

Reads `paper_trades`. Prints:

- starting bankroll, current NAV, total return %
- open positions count, closed positions count
- realized PnL (Σ exit.cost_usd − parent entry.cost_usd over resolved trades)
- top-N best / worst settled trades by PnL
- per-wallet realized PnL leaderboard

### Required changes outside the strategy

1. New schema CREATE statements + indexes added to `src/pscanner/store/db.py`.
2. New `PaperTradesRepo` in `src/pscanner/store/repo.py` — typed wrappers for
   the queries above (open-position list, insert entry, insert exit, summary
   stats for the CLI).
3. `MarketCacheRepo`: verify it exposes (or add) a method returning
   outcome-name → asset_id pairs from the cached market row. The cache
   already stores `outcome_prices_json` and outcomes; the typed accessor
   may need a new helper.
4. `Scheduler._build_detectors` — instantiate `PaperTrader` and
   `PaperResolver` when `config.paper_trading.enabled`, gated config-side.
5. `Scheduler._wire_alert_subscribers` — subscribe `PaperTrader` to
   `AlertSink` (mirrors the move-attribution wiring).
6. `pscanner` CLI — new `paper status` subcommand wired in `src/pscanner/cli.py`.

## Data flow

```
   smart-money detector
        emits Alert(detector="smart_money",
                    body={wallet, condition_id, side, ...})
                │
                ▼
            AlertSink.emit
                │  ├── writes alerts row (dedup)
                │  └── fan-out to subscribers
                ▼
   PaperTrader.handle_alert_sync(alert)             [t = 0]
        ├── _filter_smart_money            detector ∈ {"smart_money"};
        │                                  body has wallet+condition_id+side
        ├── _check_wallet_edge             tracked_wallets.weighted_edge > 0
        ├── _resolve_outcome               market_cache.outcomes → asset_id
        │                                  market_ticks → best_ask | last_trade
        │                                  null both → skip with no_price WARN
        ├── _compute_nav                   starting_bankroll + realized_pnl
        ├── _size_trade                    cost_usd = nav × fraction
        │                                  shares   = cost_usd / fill_price
        └── _record_entry                  INSERT entry row

                                  [t = next PaperResolver scan, ≤ 5 min]
   PaperResolver.scan
        ├── _open_positions                SELECT entries with no exit
        ├── for each open position:
        │     _check_resolution            market_cache.active = 0 AND
        │                                  outcome_prices_json definitive
        │     ├── unresolved → skip cycle
        │     └── resolved → payout_per_share ∈ {0.0, 1.0}
        └── _record_exit                   INSERT exit row
                                              parent_trade_id = entry.trade_id
                                              shares = entry.shares
                                              fill_price = payout_per_share
                                              cost_usd = shares × payout_per_share
                                              nav_after_usd = nav + (cost − entry.cost)

   pscanner paper status (CLI)
        Reads paper_trades, computes:
          - NAV now, return %
          - open / closed counts
          - realized PnL = Σ(exits.cost_usd − parent_entry.cost_usd)
          - best/worst settled, per-wallet leaderboard
```

API budget per smart-money alert: zero new fetches — `market_cache` and
`market_ticks` are already populated by existing collectors. The resolver
also reads only from local cache; market resolution is detected piggyback on
`MarketsCollector`'s normal poll cycle.

### Consistency notes

- **Same-cycle duplicate alerts:** `AlertSink` dedupes via the `alerts` table
  PK. `paper_trades.idx_paper_trades_alert_key` UNIQUE on entries is a
  belt-and-suspenders.
- **Resolution race:** `_check_resolution` only fires on a definitive
  outcome split, never partial / NULL. Position stays open until the next
  resolver scan after `market_cache` catches up.
- **NAV under concurrency:** `PaperTrader` may receive several alerts in
  quick succession. Each `_compute_nav` reads the latest committed state;
  concurrent evaluates take independent NAV snapshots. Acceptable for v1
  given SQLite's per-connection locking. A mutex can be added later if
  strict ordering is needed.
- **Multiple positions on the same market:** by design. Same wallet entering
  twice or different smart-money wallets entering the same market each
  produce a separate paper position, mirroring their independent decisions.
- **Cold start:** `PaperResolver` doesn't track which markets it has seen —
  on each scan it walks every open position. So a daemon-down period where
  several positions resolved is booked in one batch on the next sweep.

## Error handling

Same fail-soft posture as `MoveAttributionDetector`: every external call
isolated, every error log-and-continue, never block the alert hot path.

| Failure | Response |
|---|---|
| Subscriber callback raises | Already isolated by `AlertSink.emit` per-callback try/except. |
| Alert wrong shape (missing wallet/cond/side) | `_filter_smart_money` returns False. Log `paper_trader.bad_body` DEBUG. |
| `tracked_wallets` row missing | Treat as edge=NULL → skip. Log `paper_trader.no_edge` DEBUG. |
| `weighted_edge ≤ min_weighted_edge` (or NULL) | Configured filter, not an error. Log `paper_trader.below_edge` DEBUG. |
| `market_cache` row missing | Skip with `paper_trade.no_market` WARN. |
| Outcome-name → asset_id mapping fails | Skip with `paper_trade.outcome_unmappable` WARN. |
| No `best_ask` and no `last_trade_price` | Skip with `paper_trade.no_price` WARN. |
| `fill_price ≤ 0` or `≥ 1.0` | Skip with `paper_trade.bad_price` WARN. |
| `compute_cost_basis_nav` ≤ 0 (bankroll wiped) | Skip with `paper_trade.bankroll_exhausted` INFO. |
| `cost_usd < min_position_cost_usd` | Skip with `paper_trade.size_too_small` DEBUG. |
| `paper_trades` insert raises | Log `paper_trader.insert_failed` WARN. No partial state. |
| UNIQUE violation on `triggering_alert_key` | Log `paper_trader.duplicate_alert` DEBUG and skip. |
| **Resolver:** market_cache missing | Skip this position this cycle. |
| **Resolver:** `outcome_prices_json` parse error / non-binary shape | Skip with `paper_resolver.bad_outcomes` WARN. |
| **Resolver:** `active=False` but ambiguous outcomes (`[0.5, 0.5]`) | Skip — market closed but not finalized. Position stays open. |
| **Resolver:** exit insert raises | Log `paper_resolver.insert_failed` WARN. Position stays open and is retried next cycle. |
| **Resolver:** outcome name mismatch (cache renormalized) | Match on `asset_id` instead — the canonical key. |

The entry path is best-effort. The resolver is patient — anything ambiguous
keeps the position open and re-checks next cycle. No path can leave a
half-closed position.

## Testing

Three layers, reusing existing fixtures (`tmp_db`, `fake_clock`, `respx`).

### Pure-function unit tests

`compute_cost_basis_nav`:
- empty trades → starting bankroll
- one entry, no exit → starting bankroll (open positions still on cost basis)
- one entry + one exit (win) → starting + (proceeds − cost)
- multiple entries / partial exits → only resolved positions move NAV
- multiple resolved positions in opposite directions → both reflected

`resolve_outcome_to_asset`:
- exact-name match → `(asset_id, normalized_name)`
- case-insensitive
- whitespace tolerance
- mismatch → None
- malformed `outcome_prices_json` → None

`_size_trade`:
- happy path: nav=1000, fill=0.5, fraction=0.01 → cost=10, shares=20
- below `min_position_cost_usd` → None
- `fill_price` outside (0, 1) → None

### Detector-level tests (real repos against `tmp_db`)

`PaperTrader`:
- end-to-end happy path: smart-money alert → entry row with correct fields
- non-`smart_money` alert → no insert
- wallet with `weighted_edge ≤ 0` → no insert (filter)
- missing `market_cache` row → no insert + WARN
- outcome name not in cached outcomes → no insert + WARN
- no `best_ask` and no `last_trade_price` → no insert + WARN
- `fill_price = 1.0` boundary → no insert + WARN
- bankroll exhausted → no insert + INFO
- duplicate `triggering_alert_key` → second insert no-ops cleanly
- multiple alerts on same `condition_id` from different wallets → both insert
- multiple alerts from the same wallet on the same market → both insert

`PaperResolver`:
- open position, market still active → no exit
- open position, our outcome won → exit with `cost_usd = shares × 1.0`
- open position, our outcome lost → exit with `cost_usd = 0`
- market closed but ambiguous outcomes → no exit, position stays open
- no `market_cache` row → no exit + DEBUG
- two positions on the same outcome_id → both exit on the same scan
- cold start with several already-resolved markets → all booked in one scan
- exit insert raises (mocked) → other positions still process; broken one stays open

### Integration smoke test

Wires `AlertSink` + `PaperTrader` + `PaperResolver` + a fake upstream emitting
a synthesized `smart_money` alert. Verifies the chain produces an entry row;
driving `fake_clock` to the resolver cadence + flipping
`market_cache.active=False` produces the exit row and updates NAV.

### CLI test

`pscanner paper status` against a `tmp_db` populated with a known mix of
trades. Verifies output contains starting bankroll, current NAV, total
return %, counts, realized PnL, top-N best/worst, per-wallet leaderboard.

### Test gotchas (per CLAUDE.md)

- Never `monkeypatch.setattr(asyncio, "sleep", AsyncMock())` — use
  `FakeClock` injected via `clock=` ctor kwarg on `PaperResolver`.
- `pyproject.toml`'s `filterwarnings = ["error"]` means any
  `RuntimeWarning` from un-awaited coros fails the suite.

Estimated test count: ~16 unit + ~9 detector + 1 wiring + 1 CLI ≈ 27 tests.

## Out of scope (deferrable)

- **Mispricing arb strategy.** Different shape (event-level arb, not
  position-mirror). Tracked as a v2 strategy; the framework here doesn't
  preclude it.
- **Other signal sources** — convergence, cluster, velocity, move-attribution.
  All deferrable v2 strategies once smart-money copy-trade results are read.
- **Per-strategy tagging on `paper_trades`.** v1 stores all trades in one
  table without a `strategy` column; we'll add that when there's a second
  strategy.
- **Mirror-exit (close paper position when the wallet closes theirs).**
  Worth measuring against hold-to-resolution once we have data on both.
- **Live order placement.** Out of scope. Paper only.
- **Mark-to-market PnL.** v1 reports realized PnL only; open positions sit
  at cost basis. Mark-to-market display is a CLI-only enhancement that
  could be added later without schema changes.
- **Reset / wipe.** v1 expects you to reset by clearing the `paper_trades`
  table manually. A `pscanner paper reset` command can be added once the
  workflow demands it.
