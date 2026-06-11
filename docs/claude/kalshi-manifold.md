# Kalshi & Manifold (multi-platform notes)

Referenced from `CLAUDE.md`. Read this before touching `pscanner.kalshi`, `pscanner.manifold`,
or any cross-platform corpus / ingestion code.

## Multi-platform architecture
- **Multi-platform module shape (per RFC #35)**: each platform owns its own subpackage (`pscanner.poly`, `pscanner.kalshi`, `pscanner.manifold`) with parallel `ids.py` (per-platform `NewType[str]` modules; no shared `MarketId` supertype), `client.py`, `models.py`, `db.py`, `repos.py`. Daemon-side tables are namespaced (`kalshi_*`, `manifold_*`) — no `platform` column on existing Polymarket tables. The corpus-side `platform` column on shared tables landed in RFC PR A (see the next bullet). Stage 1 PRs (#36, #37) only ship the data-layer; trade collectors and detector instances per platform are deferred.
- **`platform` column on shared corpus tables.** `corpus_markets`, `corpus_trades`, `market_resolutions`, `training_examples`, and `asset_index` carry a `platform TEXT NOT NULL DEFAULT 'polymarket' CHECK (platform IN ('polymarket','kalshi','manifold'))` column that is part of the composite primary key on each table. Repo methods and dataclass row types take a `platform: str = "polymarket"` parameter/field so existing Polymarket call sites are unchanged. The legacy column `condition_id` holds platform-native market identifiers for non-Polymarket platforms (Kalshi tickers, Manifold market hashes) — it was not renamed at PR A time. Filtering ML training to a single platform via a `--platform` flag on `pscanner ml train` is a follow-up against the streaming pipeline (#39).

## Manifold API quirks (will bite you)
- **Mana-denominated**: bets are in mana (play money), NOT USD. Never aggregate Manifold bet amounts into real-money totals or mix with Polymarket/Kalshi volumes.
- **No traditional orderbook**: limit orders are encoded as bets with a `limitProb` parameter. Fetch open limit orders via `GET /v0/bets?kinds=open-limit`.
- **CFMM markets**: many Manifold markets are multi-outcome (CFMM mechanism, `outcomeType != "BINARY"`). Stage 1 only supports binary YES/NO. Filter non-binary at the collector layer (`ManifoldMarket.is_binary` property). Non-binary markets still parse — `prob` will be `None`.
- **Rate limit**: 500 req/min per IP, applied globally across all endpoints. Multi-IP rotation is prohibited per Manifold ToS. The `ManifoldClient._TokenBucket` enforces this with capacity=500, rate=500/60.
- **WS auth-free**: subscribe to `global/new-bet` for the bet firehose; 30-60s pings required. `ManifoldStream` uses `websockets`' built-in `ping_interval=45, ping_timeout=20`. Unknown-topic frames (e.g. `global/new-contract`) that arrive on the same connection are silently skipped.
- **Identifiers**: hash strings, not numeric or 0x-prefixed hex. `ManifoldMarketId` and `ManifoldUserId` are `NewType[str]` wrappers — don't alias to Polymarket's `MarketId` or `ConditionId`.
- **Pagination cursor**: both `/v0/markets` and `/v0/bets` use a `before=<id>` cursor (the `id` of the last item from the previous page), not an offset. Pass `before=None` to start from the most recent.
- **Tables**: `manifold_markets`, `manifold_bets`, `manifold_users` live in `pscanner.manifold.db`. Apply via `init_manifold_tables(conn)` — separate from `init_db()` (daemon) and `init_corpus_db()` (corpus).

## Manifold ingestion shape (per the integration spec)
`pscanner corpus backfill --platform manifold` enumerates resolved binary markets via `/v0/markets`, then walks `/v0/bets?contractId=<market_id>` per market into `corpus_trades`. Mana lands in `corpus_trades.notional_usd` as platform-native units (NOT USD — never aggregate Manifold volumes into real-money totals without grouping by `platform` first). `bet.user_id` is stored in `corpus_trades.wallet_address` (column-reuse convention, same as `condition_id`). The notional floor is per-platform via `pscanner.corpus.repos._NOTIONAL_FLOORS` (Polymarket: $10, Manifold: 100 mana). MKT/CANCEL resolutions land in `corpus_markets` and `corpus_trades` but are skipped by `record_manifold_resolutions` so they have no `market_resolutions` row — they drop out of `training_examples` automatically via the inner JOIN. Build features with `pscanner corpus build-features --platform manifold` and train with `pscanner ml train --platform manifold` (no new ML code; PR A's polymorphic pipeline does the work).

## Kalshi API quirks (will bite you)
- **Pricing**: cents expressed as dollar strings on the wire (`"0.0900"` = 9 cents). Convert to probability via `price_dollars` (already a float 0.0-1.0) or use the `.last_price_cents` property (integer 0-100). Contracts settle to $0 or $1.
- **Identifiers**: ticker strings (`"KXELONMARS-99"`), not hex. `KalshiMarketTicker`, `KalshiEventTicker`, `KalshiSeriesTicker` are `NewType[str]` in `pscanner.kalshi.ids` — distinct from `pscanner.poly.ids` per the multi-platform RFC. Pass `KalshiMarketTicker(...)` at call sites so `ty` catches cross-platform confusion.
- **Series fan-out**: a series (e.g. `"KXELONMARS"`) groups multiple events; an event groups multiple markets. On simple binary contracts the event ticker and market ticker are equal (e.g. both `"KXELONMARS-99"`).
- **Settlement**: $0 or $1 per share (0 or 100 cents). No mid-resolution prices.
- **Trades endpoint**: market trades live at `GET /markets/trades?ticker=TICKER`, NOT at `GET /markets/{ticker}/trades` (that path returns 404). `KalshiClient.get_market_trades` uses the correct URL.
- **Public REST is unauth**; WS streaming requires a Kalshi account + RSA-signed handshake (Stage 2, not yet implemented).
- **Base URL**: `https://api.elections.kalshi.com/trade-api/v2` (verified 2026-05-04).
- **Volume/size fields**: returned as fixed-point strings (`"1.00"`), coerced to `float` by pydantic. The `count_fp` on trades is a contract count, not a dollar amount.
- **Kalshi schema tables** (`kalshi_markets`, `kalshi_trades`, `kalshi_orderbook_snapshots`) are registered into `store/db.py:_SCHEMA_STATEMENTS` via `KALSHI_SCHEMA_STATEMENTS` from `pscanner.kalshi.db`. They are created by `init_db` alongside the Polymarket daemon tables — `tmp_db` in tests includes them automatically.

## Kalshi ingestion shape (per the integration spec)
`pscanner corpus backfill --platform kalshi` enumerates markets via `/markets?status=...` for each terminal status (`determined`, `amended`, `finalized`); skips `disputed` and `closed`. Walks `/markets/trades?ticker=<ticker>` per market into `corpus_trades`. Resolution detection uses the `result` field on the market response (`"yes"`/`"no"` → write; `"scalar"`/`""`/`disputed` → skip). Anonymous taker identity: `corpus_trades.wallet_address=""` for every Kalshi row (sentinel; no per-trade attribution available on the public REST surface). `notional_usd` is real USD (`count_fp * price`). The `_NOTIONAL_FLOORS["kalshi"] = 10.0` gate already shipped in #84. **`pscanner ml train --platform kalshi` is not supported under the L1+L2 path** — anonymous trades collapse all wallet history to the `""` key, breaking per-wallet features. The L3-enabling social-API path is tracked separately in #95.

## Open follow-ups
- **`manifold_bets` schema omits `shares` and `fees`.** `ManifoldBet.shares` / `.fees` are `None` after a DB round-trip. Add columns to the schema before relying on those fields for CFMM position-sizing analysis.
