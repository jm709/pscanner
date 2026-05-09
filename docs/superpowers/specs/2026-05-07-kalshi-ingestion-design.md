# Kalshi ingestion (corpus L1+L2) design

Date: 2026-05-07
Status: pending review
Builds on: `2026-05-06-corpus-platform-column-design.md` (#82), `2026-05-06-ml-streaming-platform-filter-design.md` (#83), `2026-05-07-manifold-ingestion-design.md` (#84).

Tracks issue #85.
Companion follow-up tracked separately as #95 (Kalshi social-API attribution path, L3-enabling).

---

## Goal

Ingest Kalshi's settled binary markets into the platform-aware corpus end-to-end:

1. **L1** — discover settled binary markets via Kalshi REST and enumerate them into `corpus_markets` with `platform='kalshi'`.
2. **L2** — backfill every fill for each market into `corpus_trades`.
3. **Resolution recording** — read the authoritative `result` field on each market response and write `market_resolutions` rows for `"yes"`/`"no"` outcomes; skip `"scalar"`, `""`, and `disputed` markets.

L3 (`training_examples`) is **explicitly out of scope** under this path — Kalshi's public REST `/markets/trades` returns no taker identity, so `corpus_trades.wallet_address=""` and the per-wallet history features that drive the existing ML pipeline can't be computed. The L3-enabling social-API path lives in #95.

## Non-goals

- Per-trader attribution. Kalshi's public REST trades are anonymous; this PR commits to the empty-string sentinel and lets #95 handle the attribution work via the undocumented `/v1/social/*` API.
- Kalshi authenticated WS streaming. Stage 2 of #36; requires RSA-signed handshake + an account.
- Kalshi daemon-side detector instances or paper-trading evaluators. Stage 2.
- Multi-platform aggregation in ML training. Already deferred per the platform-filter spec.
- Scalar (non-binary) markets. Filtered out at enumeration; deferred indefinitely.
- `pscanner corpus build-features --platform kalshi`. Anonymous trades collapse every wallet to `""`; no useful training rows. Revisited if/when #95 lands.

## Convention summary (decisions locked from brainstorming)

| Concern | Choice |
|---|---|
| Kalshi market ticker in `corpus_markets.condition_id` | Reuse existing column-overload precedent (Polymarket: hex; Manifold: hash; Kalshi: ticker like `KXELONMARS-99`). |
| Anonymous trades, `wallet_address` | Empty-string sentinel `""`. Documented convention; the composite PK stays unique because `tx_hash = trade_id` is unique per fill. |
| Synthetic `asset_id = f"{ticker}:{taker_side}"` | Mirrors Manifold's `f"{market_id}:{outcome}"`. Names the position; corpus_trades.asset_id is `NOT NULL`. |
| `notional_usd = count_fp × price_dollars` | Real USD (Kalshi is real-money). `_NOTIONAL_FLOORS["kalshi"] = 10.0` (already shipped in #84) is the right gate. |
| Resolution detection | Read `result` field on the market response (verified via OpenAPI spec + live samples). `"yes"` → write; `"no"` → write; `""`/`"scalar"`/disputed → log + skip. Last-price inference is unreliable on Kalshi — three of eight sampled `result="yes"` markets had `last_price_dollars="0.0000"` because the last trade was on the losing side. |
| Status filter at enumeration | `status in ("determined", "amended", "finalized")` AND `market_type == "binary"` AND `result in ("yes", "no")` AND `volume_fp >= min_volume_contracts`. Skip `disputed` (contested resolution — comes back when status moves to a clean terminal state). Skip `closed` (trading halted but no determination yet). |
| Volume gate | `min_volume_contracts: float = 10_000.0` parameter (≈$5K USD-equivalent at average pricing). Tunable per the same shape as Manifold's `min_volume_mana`. |

## Module gaps to close

| Gap | What's needed |
|---|---|
| `KalshiMarket` doesn't model the `result` field | Add `result: str \| None = None`. The Kalshi REST `/markets/{ticker}` returns this on every settled market response (verified via OpenAPI spec); Stage 1 (#36) just didn't capture it. Mirror PR #84 Task 1 for the equivalent Manifold `resolution` field. **Skip `settlement_value_dollars` and `settlement_ts`** — for binary markets `settlement_value_dollars` is deterministic from `result` (1.0 for yes, 0.0 for no, redundant) and `corpus_markets.closed_at` already serves the timestamp role via the `MAX(corpus_trades.ts)` rewrite invariant. |
| `kalshi_markets` schema lacks the `result` column | Add one nullable `result TEXT` column + an idempotent `_apply_migrations` helper, mirroring how Manifold's `db.py` handled the `resolution` column in #84. |
| `KalshiMarketsRepo.upsert` doesn't write the new column | Add `result` to the INSERT column list + values tuple. |
| `KalshiClient.get_market(ticker)` round-trip | Pydantic auto-handles once the model has the field; verify with a parse test. |

---

## Architecture

```
              ┌──────────────────────────┐
              │ KalshiClient             │  (existing, src/pscanner/kalshi/client.py)
              │ /markets?status=...      │
              │ /markets/{ticker}        │
              │ /markets/trades?ticker=X │
              └──────────┬───────────────┘
                         │
            ┌────────────┼──────────────────────────────────────┐
            │            │                                      │
            ▼            ▼                                      ▼
   enumerate_resolved   walk_kalshi_market         record_kalshi_resolutions
   _kalshi_markets      (per-market trade backfill) (per-market resolution write)
       │                       │                              │
       │                       ▼                              ▼
       │              CorpusTradesRepo                 MarketResolutionsRepo
       │                       │                              │
       ▼                       ▼                              ▼
  CorpusMarketsRepo  ────►  corpus_trades  ◄────────  market_resolutions
       │                       │                              │
       └─── corpus_markets ────┘                              │
                                                              │
                                                              ▼
                                                  (no build-features pass —
                                                   anonymous trades, no L3)
```

The `KalshiClient` module already exists. New code is the three corpus-side modules and the CLI dispatch. Daemon-side `kalshi_*` tables are unchanged except for three additive columns on `kalshi_markets`.

---

## Enumerator: `src/pscanner/corpus/kalshi_enumerator.py` (new)

Mirrors `src/pscanner/corpus/manifold_enumerator.py` shape and idempotency contract.

```python
async def enumerate_resolved_kalshi_markets(
    client: KalshiClient,
    repo: CorpusMarketsRepo,
    *,
    now_ts: int,
    min_volume_contracts: float = 10_000.0,
    page_size: int = 100,
) -> int:
    """Walk Kalshi markets and insert qualifying rows into corpus_markets.

    Iterates each terminal status (`determined`, `amended`, `finalized`)
    via cursor pagination on `/markets?status=<value>`. Skips `disputed`
    (contested resolution; let it return on a future refresh once Kalshi
    moves it to a clean terminal state). Skips `closed` (trading halted
    but no determination yet).

    Returns count of newly-inserted corpus_markets rows.
    """
```

**Algorithm:**

1. For `status` in `("determined", "amended", "finalized")` (three passes — Kalshi's API takes one value at a time):
   1. Cursor-paginate `client.get_markets(status=status, limit=page_size, cursor=cursor)` until cursor is empty.
   2. For each market in each page, filter:
      - `market.market_type == "binary"` (skip scalar)
      - `market.result in ("yes", "no")` (skip voided / unset / scalar-result)
      - `market.volume_fp >= min_volume_contracts` (volume gate)
   3. For each survivor, build a `CorpusMarket(platform="kalshi", ...)`:
      - `condition_id = market.ticker`
      - `event_slug = market.event_ticker`
      - `category = market.market_type` (just `"binary"` after the filter — placeholder; analogous to Polymarket's gamma category strings, which Kalshi has no direct analog for)
      - `closed_at = market.settlement_ts or market.close_time or now_ts` (will be rewritten to `MAX(corpus_trades.ts)` by `mark_complete` after the walker runs)
      - `total_volume_usd = market.volume_fp` (contract count, per the platform-native column-overload precedent; Kalshi readers must convert via `× price` if they want USD)
      - `enumerated_at = now_ts`
      - `market_slug = market.ticker` (Kalshi has no separate slug)
   4. `repo.insert_pending(corpus_market)` — `INSERT OR IGNORE`, idempotent.
2. Return the cumulative inserted count.

**Why three passes vs one query for "all terminal":** Kalshi's `/markets` `status` param accepts a single value. Three passes is the simplest correct option; a future Kalshi API change to support multi-value status filtering is the natural improvement vector.

**Volume gate rationale:** `volume_fp` is contract count (each contract pays $0 or $1 at settlement). At average price ≈ $0.50, 10,000 contracts ≈ $5K USD economic activity. Tunable via the parameter; defaults are conservative for a corpus with quality bias.

---

## Walker: `src/pscanner/corpus/kalshi_walker.py` (new)

Mirrors `src/pscanner/corpus/manifold_walker.py`.

```python
async def walk_kalshi_market(
    client: KalshiClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    *,
    market_ticker: KalshiMarketTicker,
    now_ts: int,
    page_size: int = 100,
) -> int:
    """Backfill all fills for one Kalshi market into corpus_trades.

    Returns count of inserted CorpusTrade rows (after the platform-aware
    notional floor in CorpusTradesRepo.insert_batch).
    """
```

**Algorithm:**

1. `markets_repo.mark_in_progress(condition_id=market_ticker, started_at=now_ts, platform="kalshi")`.
2. Cursor-paginate `client.get_market_trades(ticker=market_ticker, limit=page_size, cursor=cursor)` until cursor returns empty. (Kalshi's trades endpoint has no offset cap, unlike Polymarket's 3000-offset limit.)
3. For each `KalshiTrade`, project to `CorpusTrade(platform="kalshi", ...)`:
   - `tx_hash = trade.trade_id`
   - `asset_id = f"{market_ticker}:{trade.taker_side}"` (synthetic; `taker_side` is `"yes"`/`"no"`)
   - `wallet_address = ""` (anonymous-path sentinel; `_canonicalize_wallet_address` already preserves this verbatim per #84's fix)
   - `condition_id = market_ticker`
   - `outcome_side = trade.taker_side.upper()` (Kalshi wire format is lowercase; corpus convention is uppercase YES/NO)
   - `bs = "BUY"` (Kalshi REST trades are taker fills; same Manifold convention)
   - `price = trade.yes_price_dollars if taker_side == "yes" else trade.no_price_dollars` (the price the taker paid, already a 0-1 float per the existing pydantic validator)
   - `size = trade.count_fp`
   - `notional_usd = trade.count_fp * price` (real USD)
   - `ts = trade.created_time` (already epoch seconds via the existing model)
4. `trades_repo.insert_batch(trades)`. The `_NOTIONAL_FLOORS["kalshi"] = 10.0` filter (shipped in #84) drops sub-$10 fills automatically.
5. `markets_repo.mark_complete(market_ticker, completed_at=now_ts, truncated=False, platform="kalshi")`.

---

## Resolution recording: extend `src/pscanner/corpus/resolutions.py`

```python
async def record_kalshi_resolutions(
    *,
    client: KalshiClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, int]],   # (market_ticker, resolved_at_hint)
    now_ts: int,
) -> int:
    """Fetch resolution outcomes for settled Kalshi markets.

    For each target, calls KalshiClient.get_market(market_ticker) and
    reads `market.result`. Writes a market_resolutions row for "yes"/"no";
    logs and skips for "scalar", "" (with terminal status — voided or
    anomaly), and "disputed" (contested — the next refresh will pick it
    up if Kalshi moves it to a clean terminal status).
    """
```

Skip events (each named to ease triage):
- `corpus.kalshi_resolution_undetermined` — `result == ""` while status is terminal.
- `corpus.kalshi_resolution_scalar` — `result == "scalar"` (defensive; should already be filtered at enumeration).
- `corpus.kalshi_resolution_disputed` — `status == "disputed"`. Note status separately because `result` may be populated.

YES/NO writes use `MarketResolution(... source="kalshi-rest", platform="kalshi", outcome_yes_won=1 or 0, winning_outcome_index=0 or 1)`. Same shape as `record_manifold_resolutions`.

---

## CLI

Two changes in `src/pscanner/corpus/cli.py`.

### `pscanner corpus backfill --platform kalshi`

Extend the existing `--platform` flag's `choices` from `["polymarket", "manifold"]` to `["polymarket", "manifold", "kalshi"]`. Add a `_run_kalshi_backfill(args)` branch alongside the Polymarket and Manifold branches.

```python
async def _run_kalshi_backfill(args: argparse.Namespace) -> int:
    """Kalshi path: enumerate settled markets, walk each one's trades."""
    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    now_ts = int(time.time())
    try:
        async with KalshiClient() as client:
            await enumerate_resolved_kalshi_markets(
                client, markets_repo, now_ts=now_ts
            )
            while pending := markets_repo.next_pending(
                limit=10, platform="kalshi"
            ):
                for market in pending:
                    await walk_kalshi_market(
                        client,
                        markets_repo,
                        trades_repo,
                        market_ticker=KalshiMarketTicker(market.condition_id),
                        now_ts=now_ts,
                    )
    finally:
        conn.close()
    return 0
```

### `pscanner corpus refresh --platform kalshi`

Mirrors `_run_manifold_refresh`: re-enumerate (catches newly-settled markets), find `corpus_markets` rows missing from `market_resolutions`, call `record_kalshi_resolutions` over them.

### **No `pscanner corpus build-features --platform kalshi`**

The `build-features` subparser's `choices` stays `["polymarket", "manifold"]`. Anonymous trades produce no useful training rows; the L3-enabling social-API path (#95) handles that.

---

## Tests

### Unit-level

- `tests/kalshi/test_models.py` — `test_kalshi_market_parses_result_field` parametrized over `"yes"`, `"no"`, `"scalar"`, `""`, and absent.
- `tests/kalshi/test_db.py` — `test_kalshi_markets_has_result_column` (nullable TEXT). Idempotent-migration test mirroring Manifold's pattern.
- `tests/kalshi/test_repos.py` — `test_kalshi_markets_repo_roundtrips_result_field` (direct column read + raw-payload round-trip via `get`).
- `tests/corpus/test_kalshi_enumerator.py` (new) — fake `KalshiClient.get_markets` returning mixed pages (yes-resolved binary, no-resolved binary, scalar-type, voided result, below-volume, active-status, disputed); assert only `(determined|amended|finalized) AND binary AND result∈{yes,no} AND volume≥gate` rows land. Also test multi-status pagination loops correctly.
- `tests/corpus/test_kalshi_walker.py` (new) — fake `KalshiClient.get_market_trades` returning pages with both YES-side and NO-side fills; assert correct projection (synthetic `asset_id`, empty `wallet_address`, uppercase `outcome_side`, BUY, correct price/size/notional, `platform='kalshi'`), notional floor drops sub-$10 fills, `mark_complete(truncated=False)`.
- `tests/corpus/test_resolutions.py` — extend with `test_record_kalshi_resolutions_writes_yes_no` and `test_record_kalshi_resolutions_skips_disputed_undetermined_and_scalar`.

### End-to-end

- `tests/corpus/test_kalshi_e2e.py` (new) — seed a synthetic three-platform corpus DB by hand: 1 Polymarket market+trade+resolution, 1 Manifold YES market+bet+resolution, 1 Kalshi YES market+trade+resolution, 1 Kalshi voided market+trade with NO resolution row. Then:
  - Verify `corpus_trades` filtered by `platform='kalshi'` returns both Kalshi markets' trades (the voided market still has its trades captured).
  - Verify `market_resolutions` filtered by `platform='kalshi'` returns only the YES row (voided market silently absent).
  - Call `build_features(platform='manifold')` and assert it returns only manifold rows; do NOT call `build_features(platform='kalshi')` (out of scope — anonymous trades don't produce useful training rows; documented absence of CLI surface).

### CLI parser

- `tests/corpus/test_cli.py` — extend with parser tests for `backfill --platform kalshi` and `refresh --platform kalshi` (accept, default polymarket, reject unknown). Standard 3-test pattern from PR #84.

### Daemon `init_db`

No new test needed — existing tests cover the daemon-side `kalshi_*` tables. New `kalshi_markets` columns are covered by `tests/kalshi/test_db.py`.

---

## Documentation

CLAUDE.md gains one bullet under "Codebase conventions" matching the structure of PR #84's Manifold bullet:

> **Kalshi ingestion shape (per the integration spec).** `pscanner corpus backfill --platform kalshi` enumerates markets via `/markets?status=...` for each terminal status (`determined`, `amended`, `finalized`), walks `/markets/trades?ticker=<ticker>` per market into `corpus_trades`. Resolution detection uses the `result` field on the market response (`"yes"`/`"no"` → write; `"scalar"`/`""`/`disputed` → skip). Anonymous taker identity: `corpus_trades.wallet_address=""` for every Kalshi row (sentinel; no per-trade attribution available on the public REST surface). `notional_usd` is real USD (`count_fp * price`). The `_NOTIONAL_FLOORS["kalshi"] = 10.0` gate already shipped in #84. **`pscanner ml train --platform kalshi` is not supported under the L1+L2 path** — anonymous trades collapse all wallet history to the `""` key, breaking per-wallet features. The L3-enabling social-API path is tracked separately in #95.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kalshi API surface drift (e.g., `result` field renamed or removed) | The OpenAPI spec lists `result` in the required fields list as of 2026-05-07. If it ever drops, the model becomes silently `None` and `record_kalshi_resolutions` correctly skips with a clear log event. Tests cover the parse path. |
| Three-pass status enumeration is slower than necessary | At today's volumes (cursor-paginated, page_size=100) and Kalshi's 60 RPM rate limit, three passes is well within budget. Concrete: ~1000 settled markets per status × 100/page × 60 RPM = ≈30s per pass, ≈90s total — order of magnitude smaller than a Polymarket backfill. |
| `volume_fp` quality gate ≈$5K is too aggressive (drops too many real markets) or too loose (admits noise) | Tunable via the `min_volume_contracts` parameter. First live run logs `kalshi.enumerate_complete` counts per filter step to inform tuning. |
| New `result` column on `kalshi_markets` breaks existing readers | The column is nullable and additive. Existing readers that do `SELECT *` see a new field they ignore; existing readers that name columns explicitly are unaffected. Test coverage on the round-trip. |
| The synthetic `asset_id = f"{ticker}:{taker_side}"` collides with another platform's asset_id format | Cannot collide structurally — Polymarket asset_ids are decimal numerics, Manifold composites use `:` but with different-shape market_ids (hash strings vs ticker), and the composite PK on `corpus_trades` includes `platform` so even structurally identical strings would still differ on platform. |
| `wallet_address=""` confuses readers who don't filter by platform | The CLAUDE.md bullet documents the convention; the platform column is the disambiguator. Existing reads that need wallet identity already filter by platform (the post-PR-A norm). |

---

## Affected files (estimate)

**New:**
- `src/pscanner/corpus/kalshi_enumerator.py`
- `src/pscanner/corpus/kalshi_walker.py`
- `tests/corpus/test_kalshi_enumerator.py`
- `tests/corpus/test_kalshi_walker.py`
- `tests/corpus/test_kalshi_e2e.py`

**Modify:**
- `src/pscanner/kalshi/models.py` (add 3 fields to `KalshiMarket`)
- `src/pscanner/kalshi/db.py` (add 3 columns + idempotent migration helper)
- `src/pscanner/kalshi/repos.py` (round-trip the new fields)
- `src/pscanner/corpus/resolutions.py` (`record_kalshi_resolutions`)
- `src/pscanner/corpus/cli.py` (`--platform kalshi` on `backfill` / `refresh`)
- `tests/kalshi/test_models.py`, `tests/kalshi/test_db.py`, `tests/kalshi/test_repos.py` (extend)
- `tests/corpus/test_resolutions.py`, `tests/corpus/test_cli.py` (extend)
- `CLAUDE.md` (Kalshi ingestion bullet)

Roughly 5 new files and 10 modified. ~500 lines of source change and ~500 lines of test additions.
