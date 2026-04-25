# pscanner

Polymarket scanner — a persistent asyncio daemon that watches Polymarket and
surfaces three actionable trading signals:

1. **Smart money** — high-conviction positions from wallets with proven
   *edge over the market*. For each candidate's closed positions we compute
   `edge = outcome - implied_prob_at_entry`; wallets with `mean_edge` and
   `excess_pnl_usd` above thresholds are tracked, and we then poll their
   open positions for new entries clearing `new_position_min_usd`.
2. **Mispricing** — mutex outcomes within an event whose YES prices don't
   sum to 1.0. A persistent deviation hints at either an arbitrage
   opportunity or a stale book.
3. **Whales** — newly-active accounts placing outsized bets on illiquid
   markets. Fires on each newly-recorded `wallet_trades` row when the
   trader is a new wallet, the market is small, and the bet is big.

Output goes to a SQLite log at `./data/pscanner.sqlite3` and to a live
`rich` terminal panel.

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

The SQLite database at `./data/pscanner.sqlite3` exposes nine tables:

| Table | Purpose |
|---|---|
| `tracked_wallets` | Smart-money wallets with computed edge/PnL metrics. |
| `wallet_position_snapshots` | Latest position per `(wallet, market, side)` for diff-based new-entry alerts. |
| `wallet_first_seen` | Cached first-activity metadata for whale-detector age checks. |
| `market_cache` | Most-recent gamma snapshot of every active market. |
| `wallet_watchlist` | Wallets enrolled in the data-collection pipeline. |
| `wallet_trades` | Append-only confirmed trade fills per watched wallet. |
| `wallet_positions_history` | Append-only position snapshots per watched wallet. |
| `wallet_activity_events` | Append-only `/activity` stream per watched wallet. |
| `market_snapshots` | Append-only point-in-time market state (price, liquidity, volume). |
| `event_snapshots` | Append-only point-in-time event metadata. |
| `alerts` | Detector output (smart-money / mispricing / whales). |
| `event_outcome_sum_history` | Append-only Σ-of-outcomes per eligible event per scan. |

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
