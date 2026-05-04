# pscanner

Multi-platform prediction-market data daemon. Polymarket is the primary
target — live detectors, on-chain trade backfill, paper trading, and an
xgboost gate model for filtering copy-trade signals. Kalshi and Manifold
ship Stage 1 data-layer modules (REST/WS clients, schema, repos) without
detector wiring yet.

The Polymarket detector lineup surfaces eight independent signals:

| Detector | Signal |
|---|---|
| **Smart money** | High-conviction positions from wallets with proven `mean_edge` and `excess_pnl_usd` above thresholds |
| **Mispricing** | Mutex outcomes within an event whose YES prices don't sum to 1.0 |
| **Whales** | New accounts placing outsized bets on illiquid markets |
| **Velocity** | Sudden price spikes outside normal volatility on liquid markets |
| **Move attribution** | Decomposes large price moves into the wallets that caused them |
| **Monotone arbitrage** | Inconsistent prices across events that should monotonically order (e.g. earlier deadline strictly cheaper than later) |
| **Convergence** | Markets whose outcome is largely settled but price hasn't converged |
| **Cluster** | Coordinated wallet groups (creation-window cohort + shared-obscure-market graph) |

Each Polymarket signal is also evaluated by `pscanner.strategies.evaluators`
through five paper-trading evaluators (smart-money, move-attribution,
velocity, mispricing, monotone) so PnL and edge are tracked end-to-end.

Output goes to two SQLite databases:

- `./data/pscanner.sqlite3` — daemon state: detectors, alerts, watched
  wallets, market cache, paper trades, Kalshi + Manifold market caches.
- `./data/corpus.sqlite3` — closed-market trade corpus + ML training
  examples for the xgboost copy-trade gate.

Plus a live `rich` terminal panel.

## Install

Requires Python 3.13.

```bash
uv sync
cp config.toml.example config.toml   # edit thresholds as needed
```

The defaults baked into the pydantic config exactly match
`config.toml.example`, so the daemon also runs with no config file at all.

## Configure

Edit `config.toml` to dial detector thresholds. The fields are:

| Section | Key | Meaning |
|---|---|---|
| `scanner` | `db_path` | SQLite file location. |
| `scanner` | `log_level` | `DEBUG`/`INFO`/`WARNING`/`ERROR`. |
| `smart_money` | `min_edge` | Minimum mean edge (`outcome − implied_prob`) to track a wallet. |
| `smart_money` | `min_excess_pnl_usd` | Minimum realized PnL in USD to track a wallet. |
| `smart_money` | `new_position_min_usd` | Min USD delta to alert on. |
| `mispricing` | `sum_deviation_threshold` | `\|Σ − 1\|` band before alerting. |
| `mispricing` | `min_event_liquidity_usd` | Skip events thinner than this. |
| `whales` | `small_market_max_liquidity_usd` | Cap on "tiny market" liquidity. |
| `whales` | `big_bet_min_usd` | Min USD trade to consider. |
| `whales` | `big_bet_min_pct_of_liquidity` | Trade size as fraction of liquidity. |
| `ratelimit` | `gamma_rpm` / `data_rpm` | Per-host requests-per-minute. |

You can override the config path via `--config` or the `PSCANNER_CONFIG`
env var.

## Querying the data

The schema spans 30 tables across the daemon DB (`./data/pscanner.sqlite3`)
and the corpus DB (`./data/corpus.sqlite3`). The headline tables for
direct querying:

| DB | Table | Purpose |
|---|---|---|
| daemon | `alerts` | Detector output across all 8 detectors |
| daemon | `tracked_wallets` | Smart-money wallets with computed edge/PnL metrics |
| daemon | `wallet_trades` | Append-only confirmed trade fills per watched wallet |
| daemon | `wallet_positions_history` | Append-only position snapshots per watched wallet |
| daemon | `wallet_first_seen` | Cached first-activity metadata for whale-detector age checks |
| daemon | `wallet_watchlist` | Wallets enrolled in the data-collection pipeline |
| daemon | `wallet_clusters` / `wallet_cluster_members` | Coordinated-wallet detection output |
| daemon | `market_cache` | Most-recent gamma snapshot of every active market |
| daemon | `market_snapshots` / `market_ticks` | Append-only point-in-time market state |
| daemon | `event_snapshots` | Append-only point-in-time event metadata |
| daemon | `event_outcome_sum_history` | Append-only Σ-of-outcomes per eligible event per scan |
| daemon | `paper_trades` | Paper-trading evaluator output with NAV |
| daemon | `kalshi_markets` / `kalshi_trades` / `kalshi_orderbook_snapshots` | Kalshi Stage 1 ingestion |
| daemon | `manifold_markets` / `manifold_bets` / `manifold_users` | Manifold Stage 1 ingestion |
| corpus | `corpus_markets` | Closed-market work queue + backfill state |
| corpus | `corpus_trades` | Append-only trade rows (REST + on-chain backfill) |
| corpus | `market_resolutions` | Resolved outcomes per closed market |
| corpus | `training_examples` | Per-trade rows with features + label for the gate model |
| corpus | `asset_index` | `asset_id → (condition_id, outcome_side)` lookup for on-chain ingest |

Recommended liquidity floor for analysis: when joining `wallet_trades` or
`market_snapshots` for downstream studies, filter `liquidity_usd >= 100` to
drop noise-floor markets where individual fills can swing prices by >10%.

```sql
SELECT * FROM market_snapshots
WHERE liquidity_usd >= 100
ORDER BY snapshot_at DESC LIMIT 100;
```

The `event_outcome_sum_history` table captures the Σ-of-YES-leg-prices for
every mispricing-eligible event on every scan, regardless of whether an
alert fires. This enables retroactive analysis of multi-outcome layouts
(checkbox events) that the alert path silently filters past
`alert_max_deviation`:

```sql
-- Events with extreme Σ — likely checkbox/multi-outcome layouts
SELECT event_id, market_count, price_sum, deviation, snapshot_at
FROM event_outcome_sum_history
WHERE ABS(deviation) > 5
ORDER BY ABS(deviation) DESC LIMIT 20;
```

Alerts only fire when `sum_deviation_threshold < |Σ − 1| <= alert_max_deviation`.
Events with deviation above the cap are silently captured for future analysis,
not surfaced as alerts.

## Run

```bash
uv run pscanner run            # long-running daemon
uv run pscanner run --once     # single pass, prints counts, exits 0
uv run pscanner status         # last 50 alerts from SQLite
```

`run` keeps the websocket open, drives the rich live panel, and writes new
alerts into SQLite as they arrive. `Ctrl+C` triggers graceful shutdown
(close WS, flush DB, stop renderer).

`run --once` refreshes catalog state (markets, leaderboard, tracked
wallets) and runs each enabled detector through one pass without opening
the websocket. Useful for cron runs and integration smoke tests.

`status` opens the SQLite database read-only and prints the last 50
alerts as a Rich table grouped by detector.

### Watchlist + paper trading

```bash
uv run pscanner watch <addr> [--reason TEXT]   # enroll a wallet
uv run pscanner unwatch <addr>                 # remove from the watchlist
uv run pscanner watchlist                      # list enrolled wallets

uv run pscanner paper status                   # NAV + per-wallet PnL +
                                               # per-source breakdown
```

### Corpus + ML training

The corpus pipeline builds a dataset of closed-market trades for offline
analysis and ML training:

```bash
uv run pscanner corpus backfill                # bulk-pull every closed
                                               # qualifying market
uv run pscanner corpus refresh                 # incremental sweep of
                                               # newly-closed markets

uv run pscanner corpus onchain-backfill        # walk Polygon CTF Exchange
                                               # OrderFilled events
uv run pscanner corpus onchain-backfill-targeted  # per-market backfill of
                                                  # truncated markets

uv run pscanner corpus build-features --rebuild  # (re)build training_examples
                                                 # from corpus_trades +
                                                 # market_resolutions

uv run pscanner ml train --device cuda --n-jobs 1  # xgboost gate model
                                                   # via Optuna
```

The training pipeline (`pscanner.ml`) optimizes the realized-edge metric
both at the outer Optuna level and at the inner xgboost early-stopping
level (#43). The model artifact carries `accepted_categories` metadata so
inference can gate copy signals on `top_category` membership (#41) — sports
+ esports filter lifts gross edge from ~4% to ~11% on out-of-time data.

## Develop

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```

Tests use `pytest-asyncio` (auto mode), `respx` for HTTP mocking, and
`websockets.serve` for a local CLOB websocket.

## Troubleshooting

* **HTTP 429 from gamma/data**: lower `[ratelimit] gamma_rpm` /
  `data_rpm`. The token bucket already honours `Retry-After`, but a
  burstier ceiling makes the daemon politer.
* **Whale detector stuck on "no trades"**: Polymarket's WS streams from
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`. If your network
  blocks outbound 443/wss, the detector logs `clob_ws.disconnected` and
  retries with exponential backoff up to 60s. Verify connectivity with
  `wscat -c wss://ws-subscriptions-clob.polymarket.com/ws/market`.
* **`pscanner status` reports "no database"**: the daemon hasn't run yet
  in this directory. SQLite lives at `./data/pscanner.sqlite3` by
  default; check `[scanner] db_path` if you've changed it.
* **Schema migrations**: there are none — the schema is created on
  startup with `IF NOT EXISTS`. Drop the SQLite file if you need a clean
  slate.

## License

MIT. See `pyproject.toml`.
