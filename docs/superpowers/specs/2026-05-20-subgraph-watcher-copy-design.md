# Subgraph watch + copy script — design

**Status:** Approved 2026-05-20. Implementation pending.
**Related issues:** #152 (daemon-collector follow-on), #151 (subgraph migration).

## Goal

A standalone research script that watches Polymarket's current real-time subgraph for trades by any wallet in our daemon's `WatchlistRegistry`, and books each qualifying trade as a paper-trade entry under a distinct `triggering_alert_detector='subgraph_copy'` tag. The script is a validation step for the broader `SubgraphTradeCollector` daemon collector tracked in #152 — once the script proves the subgraph data is usable end-to-end, the same logic gets promoted into a long-running collector.

## Non-goals

- **Not a long-running daemon component.** This is research code under `scripts/`. Promotion to `src/pscanner/collectors/` happens in #152, not here.
- **No automated wallet pruning.** Performance-based watchlist rotation ("drop and pickup new wallets based on performance") is a separate periodic job operating on `paper_trades` PnL after the fact.
- **No CLI subcommand wiring.** Invoked as `uv run python scripts/watch_subgraph_copy.py ...`, matching the pattern of `analyze_model.py` and `wallet_edge_leaderboard.py`.
- **No live trading.** All entries land in `paper_trades`. No order submission to the CLOB.
- **No new database schema.** Reuses the existing `paper_trades` table.

## Architecture

A single new file: `scripts/watch_subgraph_copy.py` (~250-300 lines).

### Dependencies (all existing)

| component | role |
|---|---|
| `pscanner.poly.subgraph.SubgraphClient` | Generic GraphQL client with rate-limit + retry. We pass the new subgraph URL at construction. |
| `pscanner.store.repo.WatchlistRepo` | Source of truth for the wallet set; re-read every poll cycle so adds/removes via `pscanner watch`/`unwatch` propagate without restart. |
| `pscanner.store.repo.PaperTradesRepo` | Booking the entry rows. Uses the existing `insert_entry` interface; the `IntegrityError`-on-duplicate behaviour is the cross-cycle dedupe layer. |
| `pscanner.store.repo.MarketCacheRepo` | `get_by_condition_id(condition_id) -> CachedMarket` for outcome_name lookup + fallback `outcome_prices` price. |
| `pscanner.corpus.repos.AssetIndexRepo` | Primary `tokenId → (condition_id, outcome_side, outcome_index)` mapping. **Note**: lives in the corpus DB (`data/corpus.sqlite3`), not the daemon DB. Script opens both. |
| `pscanner.store.repo.MarketTicksRepo` | Live orderbook lookup (best_ask, last_trade_price) for fill price. |
| `pscanner.poly.data.DataClient` + `pscanner.poly.gamma.GammaClient` | Cache-miss backfill chain: data-API gives the slug-by-condition_id, gamma gives the full `Market`. Same flow `PaperTrader._backfill_market_cache` uses. |
| `pscanner.config.Config` | Loads `paper_trading.starting_bankroll_usd`, `paper_trading.min_position_cost_usd`, and the sizing fraction. We deliberately reuse the daemon's tunables so paper-trade comparisons across paths are apples-to-apples. |

### Constants pinned in the script

```python
SUBGRAPH_ID = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
DETECTOR_TAG = "subgraph_copy"
ALERT_KEY_PREFIX = "subgraph"
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_RPM = 60
PAGE_SIZE = 1000  # subgraph's max
CHECKPOINT_PATH = Path("data/subgraph_watch_state.json")
```

The `DETECTOR_TAG` is what isolates these paper-trades from the daemon's smart_money/gate_buy entries so `pscanner paper status` per-source breakdown will show them as a distinct row.

## Data flow

### Startup

1. Load `.env` (for `GRAPH_API_KEY`).
2. Open two SQLite connections, both with `PRAGMA busy_timeout=5000`:
   - `data/pscanner.sqlite3` (read-write) — for `WatchlistRepo`, `MarketCacheRepo`, `MarketTicksRepo`, `PaperTradesRepo`. Daemon may hold writers concurrently; WAL mode (already enabled) permits concurrent reads + brief writers.
   - `data/corpus.sqlite3` (read-only via `mode=ro` URI) — for `AssetIndexRepo`. The script never writes to the corpus DB.
3. Read CLI flags: `--poll-interval-seconds`, `--rpm`, `--since-hours N`, `--once` (single pass for testing), `--position-fraction-override`, `--bankroll-override`.
4. Read checkpoint: `last_seen_ts` from `data/subgraph_watch_state.json`. If absent or `--since-hours N` was supplied, default to `int(time.time()) - 3600 * since_hours` (or `int(time.time())` for zero).
5. Construct `SubgraphClient(url=SUBGRAPH_URL, rpm=DEFAULT_RPM)`.
6. Construct `DataClient` + `GammaClient` for cache-miss backfill.
7. Log startup parameters at INFO via structlog.

### Per-poll cycle

1. `addrs = WatchlistRepo.active_addresses()` (fresh read).
2. If empty: log WARN, sleep `poll_interval_seconds`, continue.
3. Query the subgraph with watermark pagination:

```python
events = []
seen_tx_hashes: set[str] = set()
ts = last_seen_ts

while True:
    page = subgraph_client.query(
        where={
            "or": [
                {"timestamp_gte": ts, "maker_in": addrs},
                {"timestamp_gte": ts, "taker_in": addrs},
            ],
        },
        first=PAGE_SIZE,
        orderBy="timestamp",
        orderDirection="asc",
    )
    new = [e for e in page if e["transactionHash"] not in seen_tx_hashes]
    events.extend(new)
    seen_tx_hashes.update(e["transactionHash"] for e in page)
    if len(page) < PAGE_SIZE:
        break
    ts = max(int(e["timestamp"]) for e in page)
```

4. For each event in `events`:
   - Identify the watchlist side: `maker.id` ∈ addrs → "maker", `taker.id` ∈ addrs → "taker" (both possible).
   - Compute copy direction:
     ```
     BUY iff (watchlist == maker AND side == 0)
          OR (watchlist == taker AND side == 1)
     SKIP otherwise (it's an exit / position-decrease, not a copy candidate)
     ```
   - If both maker AND taker are watchlisted: book under whichever side matches BUY direction; if both directions match (shouldn't happen at the protocol level), prefer maker side and log INFO with both.
   - Resolve `tokenId → (condition_id, outcome_side)`:
     1. `AssetIndexRepo.get(tokenId)` (corpus DB) → `AssetEntry(condition_id, outcome_side, outcome_index)`.
     2. If miss: log `subgraph_watch.tokenid_unresolved` at WARN, skip event. (Implementation note: if skip rate exceeds ~5%, run `scripts/backfill_asset_index.py` to repopulate. A live cache-miss backfill via the data API is possible but requires going `tokenId → /trades → slug → gamma /markets`, which is two RPM-budget calls per miss and not worth the complexity for the first iteration.)
     3. With `condition_id` in hand: `MarketCacheRepo.get_by_condition_id(condition_id)` for the cached market + outcome names. If that misses, use the existing `data_client.get_market_slug_by_condition_id` → `gamma_client.get_market_by_slug` → `MarketCacheRepo.upsert` chain (same as `PaperTrader._backfill_market_cache`).
   - Resolve fill price via the priority chain in `PaperTrader._lookup_fill_price`:
     1. `market_ticks.latest_for_asset(asset_id).best_ask`
     2. fallback: `.last_trade_price`
     3. fallback: `MarketCacheRepo.outcome_prices[outcome_index]`
     4. If none of the three is in `(0, 1)`: log `subgraph_watch.no_fill_price`, skip.
   - Compute size: `cost = bankroll * position_fraction` (defaults `$1000 * 0.005 = $5`).
   - If `cost < min_position_cost_usd`: log `subgraph_watch.size_too_small`, skip.
   - `shares = cost / fill_price`.
   - Call `PaperTradesRepo.insert_entry(
        triggering_alert_key = f"{ALERT_KEY_PREFIX}:{tx_hash}:{outcome}",
        triggering_alert_detector = DETECTOR_TAG,
        rule_variant = None,
        source_wallet = <watchlist addr>,
        condition_id, asset_id, outcome, shares, fill_price,
        cost_usd = cost,
        nav_after_usd = compute via `compute_cost_basis_nav` for parity,
        ts = trade.timestamp,
      )`. `IntegrityError` on duplicate is swallowed (already in repo).
   - Log `subgraph_watch.copy_inserted` at INFO with `(wallet, side, condition_id, outcome, price, size, cost)`.
   - Print a one-line stdout summary.

5. After processing all events: `last_seen_ts = max(int(e["timestamp"]) for e in events)` if any, else unchanged. Write checkpoint to `data/subgraph_watch_state.json`.

6. Check subgraph indexing lag from the `_meta { block { timestamp } }` field returned alongside the events. If `now - block.timestamp > 60s`: WARN. If `> 600s`: ERROR.

7. `await asyncio.sleep(poll_interval_seconds)` and loop.

### Shutdown

- Catch `KeyboardInterrupt` / `asyncio.CancelledError`. Finish in-flight inserts, write final checkpoint, close `httpx.AsyncClient`s, exit zero.

## Copy-direction table

The `side` field on `OrderFilledEvent` is the order's BUY/SELL intent (0 = BUY, 1 = SELL). The maker placed the order; the taker hit it from the opposite side. Mirroring a wallet means matching their position-increase direction:

| watchlist on side | order side | watchlist effective action | our copy |
|---|---|---|---|
| maker | 0 (BUY) | accumulating tokenId | **BUY tokenId** |
| maker | 1 (SELL) | reducing tokenId | SKIP |
| taker | 0 (BUY) | (hit a buy order → they sold) reducing tokenId | SKIP |
| taker | 1 (SELL) | (hit a sell order → they bought) accumulating tokenId | **BUY tokenId** |

We only copy BUYs by design. SELLs are typically exits and the gate-model loop already chose not to score them — keeping parity here so per-source comparisons remain meaningful.

## Error handling

Defaults to **skip-on-error for per-trade failures** (research code; we'd rather miss 1% of copies than die mid-run) and **crash-loud for config/schema errors** (these are operator problems worth surfacing).

| failure | handling |
|---|---|
| Subgraph 429 / 5xx | `SubgraphClient` already retries with exponential backoff. If exhausted, log + skip cycle. |
| Subgraph GraphQL errors (schema drift) | Log full error payload at ERROR + exit non-zero. |
| Indexing lag | WARN at 60s lag, ERROR at 600s, don't crash. |
| Market cache miss | Cache-miss backfill chain; if all fail, skip event with WARN. |
| `tokenId` unresolvable | Skip with WARN. |
| No fill price | Skip with WARN. |
| `PaperTradesRepo` IntegrityError | Swallow at DEBUG (expected dedupe). |
| Empty watchlist | WARN + sleep + retry. |
| Checkpoint file missing/corrupt | Treat as fresh start. |
| SIGINT mid-cycle | Finish in-flight inserts, write checkpoint, exit clean. |
| DB lock contention | `PRAGMA busy_timeout=5000` + SQLite WAL mode (already enabled). |

## Structlog events emitted

Greppable for monitoring:

- `subgraph_watch.poll_start` — every cycle, INFO with `addrs_count`, `last_seen_ts`.
- `subgraph_watch.poll_done` — INFO with `events_seen`, `events_copied`, `events_skipped`, `wall_seconds`.
- `subgraph_watch.indexer_lag` — WARN/ERROR by threshold.
- `subgraph_watch.copy_inserted` — INFO per booked trade.
- `subgraph_watch.copy_skipped` — DEBUG with reason.
- `subgraph_watch.tokenid_unresolved` / `subgraph_watch.no_fill_price` / `subgraph_watch.size_too_small` — per-skip-reason logs.
- `subgraph_watch.backfill_failed` — WARN for cache-miss backfill failure.

## Pagination behavior

- **Steady state (10s poll, normal volume)**: every cycle returns < 1000 events, single query, no pagination.
- **First run with `--since-hours 24`**: 1000+ events backfilled in one cycle, pagination loops 1-3 times.
- **After downtime**: similar — pagination drains the backlog.
- **Cross-cycle boundary safety**: `timestamp_gte` re-queries events at the boundary timestamp; cross-cycle dedupe relies on the existing UNIQUE INDEX on `paper_trades(triggering_alert_key)`. We consume one extra subgraph query token per cycle in exchange for the strict guarantee.

## Coexistence with the daemon

The script can run alongside a daemon with `paper_trading.enabled=true` and `smart_money.enabled=true`. Both paths book into `paper_trades`:
- Daemon path: `triggering_alert_detector ∈ {"smart_money", "gate_buy", ...}`.
- Script path: `triggering_alert_detector = "subgraph_copy"`.

The `idx_paper_trades_alert_key` UNIQUE INDEX is keyed on `(triggering_alert_key, COALESCE(rule_variant, ''))`. Since the script's `alert_key` prefix is `"subgraph:"` and the daemon's prefixes are different (smart_money uses wallet+market keys), there's no key collision — both paths can co-write the same underlying trade as separate entries. `pscanner paper status` per-source breakdown will show them as parallel rows, enabling direct comparison of the latency advantage.

## Sizing

Constant per copy. Default `bankroll * position_fraction` matches the gate-model default (`$1000 * 0.005 = $5`). The `paper_trading` config's `starting_bankroll_usd` is the bankroll. The `position_fraction` is read from `paper_trading.evaluators.gate_model.position_fraction` so the script's bets are sized identically to gate_buy paper-trades for fair comparison.

CLI overrides (`--position-fraction-override`, `--bankroll-override`) exist for explicit experiments.

## Testing

Following the repo convention for `scripts/` (no dedicated test files in CI), but with two cheap unit tests added for high-risk logic:

| test | scope |
|---|---|
| **GraphQL query construction** | Pure string assertion — `_build_where_clause(addrs, last_seen_ts)` emits the expected `where:` shape including the `or:` array with `timestamp_gte` repeated per branch. |
| **Copy direction logic** | Table-driven test on `_compute_copy_direction(maker, taker, side, watchlist) -> "BUY" \| "SKIP"`. 4-row table per the copy-direction table above. |
| **Pagination loop** | Uses a fake subgraph client returning 2 pages of 1000 then a partial page; asserts watermark advances, within-cycle dedupe by tx_hash works. |
| **End-to-end smoke** (manual, not CI) | One ad-hoc invocation with `--once --since-hours 1` against the real subgraph + empty `paper_trades`; verify rows land with `triggering_alert_detector='subgraph_copy'` and `pscanner paper status` reflects them. |

Skipped: integration tests over `MarketCacheRepo` / `PaperTradesRepo` / `SubgraphClient` — those modules have existing coverage. No mock-the-world test on the full loop.

## CLI surface

```
uv run python scripts/watch_subgraph_copy.py \
  [--poll-interval-seconds 10] \
  [--rpm 60] \
  [--since-hours N]     # optional cold-start backfill window
  [--once]              # single pass for testing
  [--position-fraction-override FLOAT] \
  [--bankroll-override FLOAT] \
  [--db PATH]           # daemon DB; default data/pscanner.sqlite3
  [--corpus-db PATH]    # corpus DB (asset_index); default data/corpus.sqlite3
  [--subgraph-id ID]    # override pinned default
```

## File layout

- `scripts/watch_subgraph_copy.py` — the script.
- `data/subgraph_watch_state.json` — checkpoint file (`{"last_seen_ts": int}`), git-ignored along with the rest of `data/`.

## Out-of-scope follow-ups

- **#152 daemon collector** — once the script validates the subgraph path end-to-end, promote the polling loop into `src/pscanner/collectors/subgraph_trades.py` with a `SubgraphTradeCollector` that subscribes to the existing detector bus.
- **Wallet pruning loop** — periodic job that scores per-`source_wallet` realized PnL in `paper_trades` and drops underperformers via `WatchlistRepo.deactivate`. Independent of this script.
- **Cross-platform support** — Manifold and Kalshi don't have equivalent subgraphs today; the script is Polymarket-only.
- **`--from-registry` vs `--from-file` flag** — current design reads only from `WatchlistRepo`. If we ever want to test against an arbitrary file (e.g. a candidate list before promoting), add the flag later.
