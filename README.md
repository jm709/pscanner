# pscanner

Polymarket scanner — a persistent asyncio daemon that watches Polymarket and
surfaces three actionable trading signals:

1. **Smart money** — high-conviction positions from wallets with proven
   winrate. The detector pulls the leaderboard, recomputes each candidate's
   winrate from their closed-position history, retains those above
   `min_winrate`, then polls their open positions for new entries clearing
   `new_position_min_usd`.
2. **Mispricing** — mutex outcomes within an event whose YES prices don't
   sum to 1.0. A persistent deviation hints at either an arbitrage
   opportunity or a stale book.
3. **Whales** — newly-active accounts placing outsized bets on illiquid
   markets, joined live from the CLOB websocket trade feed.

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
| `smart_money` | `min_winrate` | Minimum win-rate to track a wallet. |
| `smart_money` | `new_position_min_usd` | Min USD delta to alert on. |
| `mispricing` | `sum_deviation_threshold` | `\|Σ − 1\|` band before alerting. |
| `mispricing` | `min_event_liquidity_usd` | Skip events thinner than this. |
| `whales` | `small_market_max_liquidity_usd` | Cap on "tiny market" liquidity. |
| `whales` | `big_bet_min_usd` | Min USD trade to consider. |
| `whales` | `big_bet_min_pct_of_liquidity` | Trade size as fraction of liquidity. |
| `ratelimit` | `gamma_rpm` / `data_rpm` | Per-host requests-per-minute. |

You can override the config path via `--config` or the `PSCANNER_CONFIG`
env var.

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
