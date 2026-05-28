# Backtest harness for copy-trading sizing schemes

## Goal

Given the current production watchlist, compare four candidate sizing
schemes side-by-side on the historical trade stream of those wallets.
Hold everything else constant (fill price, resolution, bankroll) so the
only variable is the sizing function. Output a markdown report ranked by
realized PnL so the operator can pick the best scheme.

## Held constant

- **Watchlist**: the daemon's `wallet_watchlist` table at script
  runtime. The script does not bring its own list.
- **Bankroll**: single fixed `--starting-bankroll-usd` (default
  $10,000, matching `paper_trading.starting_bankroll_usd`).
- **Trade universe**: rows of `corpus_trades` where `wallet_address` is
  in the watchlist, `direction = 'BUY'`, `platform = 'polymarket'`, and
  the trade's `condition_id` has a `market_resolutions` row.
- **Fill price**: `corpus_trades.price` directly — the seed wallet's
  own fill, zero slippage. This is the upper bound on copy PnL.
- **Resolution**: `market_resolutions.outcome` →
  `payout = 1.0 if outcome == trade.outcome_side else 0.0`.

## Architecture

### Location

Standalone Python script at `scripts/backtest_copy_sizing.py`,
following the existing operator-script pattern
(`analyze_copy_trading.py`, `wallet_edge_leaderboard.py`,
`compare_wallet_cohorts.py`). DuckDB-backed query against the corpus
via `ATTACH ... (TYPE sqlite)`.

### Event stream

The simulator walks an interleaved stream of two event types in strict
absolute-time order:

```
TradeEvent:       (ts, "trade",      trade_row)
ResolutionEvent:  (ts, "resolution", market_resolutions_row)
```

Built from one DuckDB query that UNIONs `corpus_trades` (filtered as
above) and `market_resolutions` (for the same `condition_id` set) and
`ORDER BY ts ASC`. Materializes to a Python iterable; the simulator
loops once.

### Causal guarantee

Because resolutions are processed at their actual `resolved_at`,
scheme D's rolling-edge calculation at trade-time `t` automatically
only sees trades whose `resolved_at < t`. No look-ahead is possible
by construction. An explicit integration test (see Testing) encodes
this guarantee.

### Per-event handlers

`on_trade(trade)` — for each scheme:

1. `cost = scheme.compute(trade, bankroll, state)`
2. Append `OpenPos(trade_id, wallet, condition_id, outcome_side,
   shares = cost / price, cost, ts)` to `state.open_positions`.
3. Update scheme-specific *trade-observation* state
   (e.g., scheme B's `wallet_counts[wallet] += 1`).

`on_resolution(resolution)` — for each scheme:

1. For every `open_positions` row matching this `condition_id`:
   - `payout = 1.0 if resolution.outcome == position.outcome_side
     else 0.0`
   - `proceeds = shares * payout`
   - `pnl = proceeds - cost`
   - Move from `open_positions` → `resolved_trades`.
2. Update scheme-specific *resolution-observation* state
   (e.g., scheme D's `wallet_resolved_by_wallet[wallet].append(...)`
   so future edge calcs include it).
3. Stamp `(resolution.ts, cumulative_pnl)` into `nav_series`.

### State

Per scheme:

```python
@dataclass
class BacktestState:
    open_positions: dict[trade_id, OpenPos]
    resolved_trades: list[ResolvedTrade]
    wallet_counts: dict[wallet, int]                       # scheme B
    wallet_resolved_by_wallet: dict[wallet, list[ResolvedTrade]]  # scheme D
    cumulative_pnl: float
    nav_series: list[tuple[ts, float]]
```

### Sizing protocol

```python
class SizingScheme(Protocol):
    name: str
    def compute(self, trade: Trade, bankroll: float) -> float: ...
    def observe_resolution(
        self, trade: Trade, payout: float
    ) -> None: ...
```

Schemes are constructed once from CLI flags, then the walk loop calls
them uniformly.

### Edge cases

- **Trade-after-resolution ordering inversion**: a trade with `ts >=
  resolution.ts` on the same market. Shouldn't happen (the market was
  resolved), but if `corpus_trades` has any indexer-imprecise
  timestamps, the script logs `backtest.ordering_inversion` and skips
  the trade (it would have nothing to size against).
- **Market never resolves**: trade stays in `open_positions`
  permanently. Reported as a final `unresolved_trades` count per scheme
  in the report.
- **Empty watchlist**: script exits with a clear error before touching
  DuckDB.

## Sizing schemes

### A. equal_weight

```
cost = bankroll * position_fraction
```

No state. One config: `position_fraction` (default `0.01`). Pure
baseline so every other scheme's lift is visible.

### B. concentration_capped (production reference)

```
counts[wallet] += 1   # after cost is computed for this trade
total  = sum(counts.values())
share  = counts[wallet] / total if total else 0
target = 1.0 / len(watchlist)
raw    = min(1.0, target / max(share, target))
mult   = max(raw, min_multiplier)
cost   = bankroll * position_fraction * mult
```

State: `counts: dict[wallet, int]`. Configs: `position_fraction`,
`min_multiplier` (default `0.10`), `watchlist_size` (snapshotted at
script start). Mirrors `SubgraphCopyEvaluator._concentration_multiplier`
exactly. The running count is incremented *after* the size is computed
for this trade so the first trade per wallet always gets a clean `1.0`
multiplier (production behaviour).

### C. follow_seed_size (proportional)

```
cost = min(trade.notional_usd * scale_factor, max_cost_per_trade)
```

No state. Configs: `scale_factor` (default `0.01` — copy 1% of seed
notional), `max_cost_per_trade` (default `1000.0`) to cap whale-trade
exposure.

### D. edge_weighted_causal

```
prior = wallet_resolved_by_wallet[wallet]   # ONLY trades resolved < trade.ts
if len(prior) < min_trades_for_edge:
    mult = 1.0
else:
    edge = mean(p.payout - p.implied_prob_at_buy for p in prior)
    mult = clip(1.0 + edge_scale * edge,
                min_multiplier, max_multiplier)
cost = bankroll * position_fraction * mult
```

State: `wallet_resolved_by_wallet: dict[wallet, list[ResolvedTrade]]`.
Configs: `position_fraction`, `edge_scale` (default `5.0` — a +10%
edge maps to `1.5x` sizing), `min_multiplier` (default `0.25`),
`max_multiplier` (default `3.0`), `min_trades_for_edge` (default
`10`).

`implied_prob_at_buy` is the trade's fill price (binary outcomes; the
implied probability of the YES side equals the YES price).

## Output

Markdown to stdout. One section per logical view, all four schemes
side-by-side. Optional `--csv` flag dumps per-trade per-scheme rows
for downstream analysis.

### Headline table

```
| Scheme                    | Trades | Cost   | Proceeds | PnL  | ROI | Win rate | Avg cost/trade |
| equal_weight              | ...    | ...    | ...      | ...  | ... | ...      | ...            |
| concentration_capped      | ...    | ...    | ...      | ...  | ... | ...      | ...            |
| follow_seed_size          | ...    | ...    | ...      | ...  | ... | ...      | ...            |
| edge_weighted_causal      | ...    | ...    | ...      | ...  | ... | ...      | ...            |
```

Trades is identical across schemes (same event stream); cost is the
schemes' only knob.

### Risk metrics

```
| Scheme | Max DD | DD duration | Sharpe-like | Worst trade | Best trade |
```

`Sharpe-like = mean(daily_pnl) / stdev(daily_pnl) * sqrt(365)` from
the `nav_series` binned daily. Max DD = `min(nav - running_max(nav))`.

### Quarterly PnL grid

One row per scheme, one column per quarter from min to max event ts.
Checks whether a scheme's lift is consistent or front/back-loaded.

### Top contributors (best-PnL scheme only)

Ten wallets ranked by realized PnL with `n_copies` and `total_cost`.
Surfaces concentration risk and suggests watchlist-prune candidates.

## CLI surface

Invoked as `uv run python scripts/backtest_copy_sizing.py [flags]`.

```
--db data/corpus.sqlite3
--watchlist-db data/pscanner.sqlite3      # source of wallet_watchlist
--starting-bankroll-usd 10000
--position-fraction 0.01
--min-multiplier 0.10                      # schemes B, D
--scale-factor 0.01                        # scheme C
--max-cost-per-trade 1000.0                # scheme C
--edge-scale 5.0                           # scheme D
--max-multiplier 3.0                       # scheme D
--min-trades-for-edge 10                   # scheme D
--platform polymarket                      # forward-compat for Manifold/Kalshi
--start-ts <unix>                          # optional date filter
--end-ts <unix>                            # optional date filter
--csv path/to/dump.csv                     # optional per-trade dump
```

## Testing

### 1. Unit tests per sizing scheme

`tests/scripts/test_backtest_sizing_schemes.py`. One test per scheme
verifying `compute` against a hand-calculated cost on a small input.

- **Scheme B**: three sequential calls with two wallets to verify the
  running-counts logic mirrors `SubgraphCopyEvaluator
  ._concentration_multiplier`.
- **Scheme D**: state with N hand-picked resolved trades; assert
  multiplier matches the formula at `n < min_trades_for_edge`,
  `n = min_trades_for_edge`, `edge = 0`, `edge > 0`, `edge < 0`.

### 2. Integration test for the simulator harness

`tests/scripts/test_backtest_simulator.py`. Fixture: 5 markets,
3 wallets, ~30 trades and 5 resolutions with carefully interleaved
timestamps so:

- A wallet's first trade resolves between their 2nd and 3rd — scheme D's
  edge calc must include the first by the time the 3rd is sized.
- Two trades on the same market by different wallets resolve at one
  shared `resolved_at` — both close in a single `on_resolution`.
- One trade whose market never resolves — must not appear in
  `resolved_trades` and must not contribute to PnL.

Assertions on per-scheme `total_pnl`, `n_resolved_trades`,
`len(state.open_positions)`. Explicit assertion that the event walk
processed every `ResolutionEvent` before any `TradeEvent` with
`ts >= resolution.ts` — the temporal-correctness guarantee encoded as
a test, not just docs.

### 3. Corpus smoke test — PAUSE HERE BEFORE RUNNING

`tests/scripts/test_backtest_corpus_smoke.py`, opt-in via
`pytest -m slow`. Runs the full script against the production corpus
with `--limit-watchlist 10` to keep it under a minute.

**Implementation checkpoint**: the implementing agent must pause and
get operator approval before executing this test or running the script
against the real corpus. Reason: this is the first time the script
will touch production data and the operator wants to inspect the
output (and the size of the result set) before letting it run
end-to-end.

Once approved, asserts the report contains all four scheme rows,
non-zero trade counts, and finite (no NaN/inf) PnL/Sharpe values.
Guards against regressions where a schema change quietly breaks the
SQL.

### Deliberate non-tests

- **No mocking of DuckDB or `corpus_trades`**. The simulator runs on
  plain `dict[str, list[Trade]]` so DuckDB only owns the SELECT;
  tests construct trade lists directly.
- **No backtest-vs-production-paper-trades comparison**. The live
  `subgraph_copy` paper trades have different cost basis (live
  orderbook ask + concentration), so they won't match the backtest
  even when everything is correct. That comparison is a separate,
  more elaborate validation.

## Out of scope

- Modelling subgraph indexer lag or poll-cycle delay between the seed's
  fill and our hypothetical copy. The chosen fill-price model (seed's
  own price) deliberately abstracts this away.
- Multi-platform backtesting (Kalshi, Manifold). The `--platform`
  flag is forward-compat; only `polymarket` is exercised in v1.
- Live A/B comparison against the running paper-trader. Off-line
  analysis only.
- Watchlist tuning. The script reads the current watchlist; pruning is
  a separate decision driven by the script's "top contributors" output.
