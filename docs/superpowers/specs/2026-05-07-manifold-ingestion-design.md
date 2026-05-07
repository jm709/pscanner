# Manifold ingestion (corpus L1+L2+L3) design

Date: 2026-05-07
Status: pending review
Builds on: `2026-05-06-corpus-platform-column-design.md` (RFC #35 PR A, merged as #82) and `2026-05-06-ml-streaming-platform-filter-design.md` (merged as #83).

---

## Goal

Ingest Manifold's resolved binary markets into the platform-aware corpus, end to end:

1. **L1** — discover closed binary markets via Manifold REST and enumerate them into `corpus_markets` with `platform='manifold'`.
2. **L2** — backfill every fillable bet for each market into `corpus_trades`.
3. **L3** — record YES/NO resolutions in `market_resolutions`, then run `pscanner corpus build-features --platform manifold` to produce `training_examples` rows that `pscanner ml train --platform manifold` can train on.

Single-platform per training run; cross-platform aggregation remains a deferred follow-up.

## Non-goals

- Kalshi ingestion. Separate spec (`pscanner.kalshi.*` modules already exist; the L1+L2 design is structurally similar but uses different REST endpoints and skips L3 because Kalshi REST trades carry no taker identity).
- WebSocket bet-firehose ingestion (`src/pscanner/manifold/ws.py`). Useful for future live-signal work, but solves a different problem than corpus backfill. Separate follow-up.
- Manifold daemon-side detector instances and paper-trading evaluators. Stage 2.
- Multi-platform aggregation in ML training (`--platform all`). Deferred per the platform-filter spec.
- CFMM (multi-outcome) markets. Stage 1's binary-only filter (`ManifoldMarket.is_binary`) stays.
- Cross-platform feature-schema reconciliation. Manifold's mana-as-`notional_usd` and `user_id`-as-`wallet_address` are platform-native; mixing platforms in one model is out of scope.

## Convention summary (decisions locked from brainstorming)

| Convention | Choice | Why |
|---|---|---|
| Mana in `corpus_trades.notional_usd` | Store as platform-native units (raw mana, no conversion) | No published mana↔USD rate; converting would invent a fictitious number. Within-platform models train on relative scale, which works regardless of unit. Mirrors the `condition_id` precedent — column name is Polymarket-flavored but holds platform-native semantics. |
| Manifold `user_id` in `corpus_trades.wallet_address` | Reuse the existing column | Same column-overload precedent as `condition_id` (66-char hex on Polymarket, hash string on Manifold). Zero schema migration; `idx_corpus_trades_wallet_ts` already serves both platforms. |
| MKT / CANCEL resolutions | Land markets + bets in corpus; skip the `market_resolutions` row | Mirrors the existing "disputed" Polymarket pattern in `resolutions.py`. The inner JOIN in `build_features` automatically excludes them from `training_examples` while the bets are preserved for analytical queries. |
| Build-features pipeline | Reuse `build_features(platform="manifold")` (the polymorphic post-PR-A entry point) | Wallet-history features are computed from chronological prior `corpus_trades` of the same wallet — same code, just platform-scoped via the existing WHERE filter. No new pipeline. |
| Notional floor | Hardcoded per-platform inside `CorpusTradesRepo.insert_batch` (`$10` for Polymarket, `100` mana for Manifold) | Smaller call-site changes than threading the floor as a parameter. Future tunable via constants. |

## Module gaps to close

| Gap | What's needed |
|---|---|
| `ManifoldMarket.resolution: str \| None` field | Add to `pscanner.manifold.models.ManifoldMarket`. The Manifold REST `/v0/market/{id}` returns `"YES"`/`"NO"`/`"MKT"`/`"CANCEL"`/null in this field; Stage 1 just didn't model it because nothing was reading it yet. |
| `manifold_markets.resolution TEXT` column | Add via additive migration in `pscanner.manifold.db`. Idempotent `ALTER TABLE ... ADD COLUMN` swallowing `"duplicate column name"` (mirrors the corpus-side migration pattern). |
| `init_db()` doesn't apply Manifold tables | Import `MANIFOLD_SCHEMA_STATEMENTS` from `pscanner.manifold.db` and concatenate into `_SCHEMA_STATEMENTS` in `src/pscanner/store/db.py`. Mirrors the Kalshi pattern at `store/db.py:13`. |
| `ManifoldClient.get_market` may not parse the new `resolution` field | Verify the GET path for `/v0/market/{id}` returns the field through pydantic validation. If the Stage 1 client uses a different shape than `get_markets`, ensure both produce models with `resolution` populated. |

---

## Architecture

```
              ┌───────────────────────┐
              │  ManifoldClient       │  (existing, src/pscanner/manifold/client.py)
              │  /v0/markets          │
              │  /v0/market/{id}      │
              │  /v0/bets             │
              └────────┬──────────────┘
                       │
            ┌──────────┼──────────────────────────────────────┐
            │          │                                      │
            ▼          ▼                                      ▼
  enumerate_resolved   walk_manifold_market         record_manifold_resolutions
  _manifold_markets    (per-market bet backfill)    (per-market resolution write)
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
                                                  pscanner corpus build-features
                                                       --platform manifold
                                                              │
                                                              ▼
                                                       training_examples
                                                              │
                                                              ▼
                                                  pscanner ml train --platform manifold
```

All boxes on the right of `ManifoldClient` are new code in this PR. The `Repo` and `corpus_*` / `market_resolutions` / `training_examples` surfaces are the existing platform-aware ones from PR A.

---

## Enumerator: `src/pscanner/corpus/manifold_enumerator.py` (new file)

Discovers closed binary Manifold markets and inserts them into `corpus_markets` as `(platform='manifold')` rows in `pending` state. Mirrors the shape of the existing Polymarket enumerator at `src/pscanner/corpus/enumerator.py`.

**Public API:**

```python
async def enumerate_resolved_manifold_markets(
    client: ManifoldClient,
    repo: CorpusMarketsRepo,
    *,
    now_ts: int,
    min_volume_mana: float = 1000.0,
    page_size: int = 1000,
) -> int:
    """Walk /v0/markets paginated, filter to resolved+binary+above-volume,
    insert into corpus_markets. Returns count of newly-inserted rows."""
```

**Algorithm:**

1. Cursor-paginate `client.get_markets(limit=page_size, before=cursor)` until empty. The cursor is the last item's `id` from the previous page (Manifold pagination convention).
2. Filter each page client-side:
   - `market.is_resolved is True`
   - `market.is_binary is True` (drops CFMM)
   - `market.volume >= min_volume_mana`
3. For each survivor, build a `CorpusMarket(platform="manifold", ...)` and call `repo.insert_pending(market)`. Mapping:
   - `condition_id = market.id` (Manifold market hash)
   - `event_slug = market.slug`
   - `category = market.outcome_type or "manifold-binary"` (placeholder until we have richer category mapping; analogous to Polymarket's gamma-derived `category`)
   - `closed_at = market.resolution_time` (will be rewritten to `MAX(corpus_trades.ts)` by `mark_complete` after the bet collector runs — same `closed_at` invariant as Polymarket)
   - `total_volume_usd = market.volume` (mana, per the platform-native convention)
   - `enumerated_at = now_ts`
   - `market_slug = market.slug`

**Idempotency:** `CorpusMarketsRepo.insert_pending` is `INSERT OR IGNORE`-on-PK, so re-enumeration is safe. The `repo` parameter must already point at a corpus DB that has been migrated to PR A's composite-PK shape.

**Volume gate:** `1000` mana is a reasonable starting cut for "this market saw real activity" — Manifold's economy is much smaller than Polymarket's $1M USDC gate. The exact threshold is tunable via the `min_volume_mana` parameter; the enumerator doesn't enforce a minimum lower bound.

---

## Collector: `src/pscanner/corpus/manifold_walker.py` (new file)

Backfills every fillable bet for one Manifold market into `corpus_trades`. Mirrors `src/pscanner/corpus/market_walker.py` (the Polymarket `/trades` walker).

**Public API:**

```python
async def walk_manifold_market(
    client: ManifoldClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    *,
    market_id: ManifoldMarketId,
    now_ts: int,
    page_size: int = 1000,
) -> int:
    """Backfill all bets for one Manifold market into corpus_trades.
    Returns count of inserted CorpusTrade rows."""
```

**Algorithm:**

1. `markets_repo.mark_in_progress(condition_id=market_id, started_at=now_ts, platform="manifold")`.
2. Cursor-paginate `client.get_bets(market_id=market_id, before=cursor, limit=page_size)` until empty.
3. For each bet, skip when:
   - `bet.is_cancelled is True`, OR
   - `bet.limit_prob is not None` AND `bet.is_filled is False` (unfilled limit order — book-only, never executed)
4. Construct `CorpusTrade(platform="manifold", ...)`:
   - `tx_hash = bet.id` (Manifold bet id; column-name convention)
   - `asset_id = f"{market_id}:{bet.outcome}"` (synthetic; Manifold has no separate asset id, but `(market_id, outcome)` together name the position. The `corpus_trades.asset_id` column is `NOT NULL`, so we need a value.)
   - `wallet_address = bet.user_id` (column-reuse precedent)
   - `condition_id = market_id`
   - `outcome_side = bet.outcome` (already YES/NO)
   - `bs = "BUY"` (Manifold has no SELL events; opposite-side bets express closes. Treating all bets as BUYs of their named outcome matches both the Manifold mechanism and the existing `build_features` pipeline, which only emits training rows for BUYs.)
   - `price = bet.prob_before` (the implied price the bet took at — direct semantic equivalent of Polymarket's `implied_prob_at_buy`)
   - `size = bet.amount` (mana)
   - `notional_usd = bet.amount` (mana, per platform-native convention)
   - `ts = bet.created_time`
5. Pass the list to `trades_repo.insert_batch(trades)`. The platform-aware notional floor (Polymarket: `$10`, Manifold: `100` mana) is enforced inside `insert_batch` based on each row's `platform` field.
6. On exhaustion: `markets_repo.mark_complete(market_id, completed_at=now_ts, truncated=False, platform="manifold")`. Manifold's cursor pagination has no offset cap, so `truncated` is always `False`.

**Filter rationale:** Cancelled bets and unfilled limit orders aren't trades — they're book events. The existing Polymarket walker has no analogous filter because Polymarket's `/trades` only returns fills; the filter belongs in the Manifold walker.

---

## Notional floor: `CorpusTradesRepo.insert_batch` change

`src/pscanner/corpus/repos.py` currently has:

```python
_NOTIONAL_FLOOR_USD: Final[float] = 10.0
```

and the `insert_batch` filter is `if t.notional_usd < _NOTIONAL_FLOOR_USD: continue`. This becomes platform-aware:

```python
_NOTIONAL_FLOORS: Final[dict[str, float]] = {
    "polymarket": 10.0,
    "manifold": 100.0,
    "kalshi": 10.0,  # placeholder; revisit when Kalshi ingestion lands
}
```

and the filter becomes:

```python
floor = _NOTIONAL_FLOORS.get(t.platform, 10.0)
if t.notional_usd < floor:
    continue
```

Falling back to `10.0` for unknown platforms keeps the schema CHECK constraint (`'polymarket'|'kalshi'|'manifold'`) as the source of truth on valid values. Adding a new platform tag without updating `_NOTIONAL_FLOORS` won't crash; rows just get the polymarket default. `_NOTIONAL_FLOOR_USD` constant stays as an alias to `_NOTIONAL_FLOORS["polymarket"]` so existing references in tests don't need rewriting.

---

## Resolutions: extending `src/pscanner/corpus/resolutions.py`

Add a new function next to the existing `record_resolutions`. The Polymarket function takes `(condition_id, market_slug, resolved_at)` triples and queries `gamma.get_market_by_slug`. The Manifold function takes `(market_id, resolved_at)` pairs and queries `client.get_market(market_id)`.

```python
async def record_manifold_resolutions(
    *,
    client: ManifoldClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, int]],  # (market_id, resolved_at_hint)
    now_ts: int,
) -> int:
    """Fetch resolution outcomes for resolved Manifold markets.

    For each target, calls /v0/market/{id}, reads `resolution`:
      "YES"           → outcome_yes_won=1, winning_outcome_index=0
      "NO"            → outcome_yes_won=0, winning_outcome_index=1
      "MKT" / "CANCEL" / None → log "corpus.manifold_resolution_skipped" and skip

    Upserts via repo.upsert(MarketResolution(..., platform="manifold"),
    recorded_at=now_ts) using source="manifold-rest".

    Returns count of resolutions actually written.
    """
```

The skip behavior (no `market_resolutions` row for MKT/CANCEL) propagates through the inner JOIN in `build_features` and naturally excludes those markets from `training_examples`. Bets remain in `corpus_trades` for analytical queries.

---

## Build-features (zero new code)

PR A's polymorphic `build_features` already accepts `platform="manifold"`. Once the enumerator + collector + resolutions pipeline above runs:

- `corpus_markets` has Manifold rows tagged `platform='manifold'`
- `corpus_trades` has Manifold bets tagged `platform='manifold'`
- `market_resolutions` has Manifold YES/NO rows tagged `platform='manifold'` (MKT/CANCEL excluded)

Then `pscanner corpus build-features --platform manifold` walks those rows and emits `training_examples` rows tagged `platform='manifold'`. `pscanner ml train --platform manifold` trains on them.

The wallet-history features (`prior_trades_count`, `win_rate`, `realized_edge_pp`, etc.) are computed from chronological prior `corpus_trades` entries for the same `wallet_address` — Manifold's `user_id` becomes the wallet identity. The features are Manifold-scoped because the SQL filter is platform-scoped.

The semantic overload of `bet_size_usd` (mana for Manifold rows) is the same convention as `notional_usd`; it's documented in CLAUDE.md.

---

## CLI: `pscanner corpus *` `--platform` flag

Three changes in `src/pscanner/corpus/cli.py`.

### `pscanner corpus backfill --platform polymarket|manifold`

Default `polymarket` so existing call sites are unchanged. Internal dispatch:

```python
if args.platform == "polymarket":
    return await _run_polymarket_backfill(args)  # the existing path
if args.platform == "manifold":
    return await _run_manifold_backfill(args)    # new
```

`_run_manifold_backfill` constructs a `ManifoldClient`, an `init_corpus_db()` connection, the corpus repos, then calls `enumerate_resolved_manifold_markets(...)` followed by a loop:

```python
while pending := repo.next_pending(limit=batch_size, platform="manifold"):
    for market in pending:
        await walk_manifold_market(
            client, repo, trades_repo,
            market_id=ManifoldMarketId(market.condition_id),
            now_ts=now_ts,
        )
```

### `pscanner corpus refresh --platform polymarket|manifold`

Same `--platform` flag, default `polymarket`. The Manifold path:

1. Re-enumerate (catches newly-resolved markets in `corpus_markets`).
2. Compute the set of `condition_id`s that exist in `corpus_markets` with `platform='manifold'` but are missing from `market_resolutions` (use `MarketResolutionsRepo.missing_for(condition_ids, platform="manifold")`).
3. Call `record_manifold_resolutions(client=..., repo=..., targets=[(cid, hint) for cid, hint in pairs])` with the resolved-at hint coming from `corpus_markets.closed_at`.

### `pscanner corpus build-features --platform polymarket|manifold`

Single-line change inside `_cmd_build_features`: forward `args.platform` to `build_features(platform=args.platform)`.

### Argparse boilerplate

All three subparsers gain:

```python
sub.add_argument(
    "--platform",
    type=str,
    choices=["polymarket", "manifold"],
    default="polymarket",
    help=(
        "Platform to ingest. Defaults to polymarket. Manifold support "
        "lands the markets, bets, and resolutions in the platform-aware "
        "corpus tables."
    ),
)
```

(Kalshi will be added to `choices` when the Kalshi spec lands. The CHECK constraint already allows it; the CLI flag just doesn't dispatch to it yet.)

---

## Tests

### Unit-level

- `tests/corpus/test_manifold_enumerator.py` (new) — mock `ManifoldClient.get_markets` to return paginated pages mixing resolved/unresolved/binary/CFMM/below-volume; assert that only resolved+binary+above-volume markets land in `corpus_markets` with `platform='manifold'`. Cover the cursor-pagination loop (multiple pages until empty) and idempotent re-enumeration.
- `tests/corpus/test_manifold_walker.py` (new) — mock `ManifoldClient.get_bets`; assert filtering of `is_cancelled` bets and unfilled limit orders, correct `CorpusTrade` field mapping (especially `price=prob_before`, synthetic `asset_id`, `bs="BUY"`), correct platform tag, sub-floor mana filter (`< 100` mana drops the row), `mark_complete` is called with `truncated=False`.
- Extend `tests/corpus/test_resolutions.py` — add `test_record_manifold_resolutions_writes_yes_no` (one YES + one NO market, both produce `market_resolutions` rows with `platform='manifold'` and `source='manifold-rest'`) and `test_record_manifold_resolutions_skips_mkt_and_cancel` (one MKT + one CANCEL market, neither produces a `market_resolutions` row, both log the skip event).

### End-to-end

- `tests/corpus/test_manifold_e2e.py` (new) — seed a synthetic mixed-platform corpus DB by hand (no real network): two polymarket markets + bets + resolutions, two manifold markets (one YES, one CANCEL) + bets + (one) resolution. Invoke `build_features(platform="manifold")`. Assert that only the YES-resolved manifold market produces `training_examples` rows; assert those rows have `platform='manifold'`; assert no polymarket rows appear; assert no rows for the CANCEL market.

### CLI

- Extend `tests/corpus/test_cli.py` — argparse parser tests for the new `--platform` flag on each of the three commands. Same shape as PR A Task 12 / the analogous parser tests for `pscanner ml train --platform`: accept manifold, default polymarket, reject unknown.

### Model + DB extension

- Extend `tests/manifold/test_models.py` — `test_manifold_market_parses_resolution_field` covering YES, NO, MKT, CANCEL, and null inputs.
- Extend `tests/manifold/test_db.py` — `test_manifold_markets_has_resolution_column` (assert `PRAGMA table_info(manifold_markets)` includes `resolution TEXT`).
- Extend `tests/manifold/test_repos.py` — assert `ManifoldMarketsRepo.insert_or_replace` round-trips the new field.

### Daemon `init_db` wiring

- Extend `tests/store/test_db.py` (or wherever `init_db` is tested) — `test_init_db_creates_manifold_tables` asserting `manifold_markets`, `manifold_bets`, `manifold_users` exist after `init_db()`. Mirrors the existing assertion for kalshi tables, if any.

### Notional floor

- Extend `tests/corpus/test_repos_trades.py` — `test_insert_batch_uses_polymarket_floor` (existing behavior stays the same) and `test_insert_batch_uses_manifold_floor_for_manifold_rows` (a `platform='manifold'` row with `notional_usd=50` is dropped; one with `notional_usd=150` is kept).

### No live-network tests

Manifold has no sandbox endpoint. Every test that touches the client uses respx-style mocking. The fact that manifest-level fixtures already exist in `tests/manifold/test_client.py` means we follow the same pattern.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Mana → `notional_usd` semantic overload silently misleads a future reader who aggregates volumes | CLAUDE.md gets a one-bullet note (next to the existing `condition_id`-overload note). The platform column is the disambiguator; aggregations should always group by platform. |
| `resolution` field arrives in the API in a different casing than expected | Manifold returns it as a top-level `resolution` field on the market object; the pydantic alias map in `models.py` should handle camelCase if needed. The model extension test covers the four documented values plus null. |
| Volume gate `1000` mana is too aggressive (drops too many real markets) or too loose (admits noise) | The threshold is a parameter, not a constant. The first live run logs counts per filter step (`info` events: `manifold.enumerate_resolved`, `manifold.enumerate_above_volume`); we tune based on the ratio. |
| The synthetic `asset_id = f"{market_id}:{bet.outcome}"` collides with real Polymarket asset_ids (which are uint256 strings from the CTF) | Cannot collide structurally — Polymarket asset_ids are decimal numerics, Manifold composites contain `:`, never overlapping. The composite PK on `corpus_trades` is `(platform, tx_hash, asset_id, wallet_address)` — even if a string collision somehow happened, the platform component would distinguish. |
| `walk_manifold_market` can't easily detect partial backfill state (Manifold has no offset cursor we can resume from) | Treat each `walk_manifold_market` call as atomic: it either completes fully (all bets backfilled, `mark_complete`) or fails partway (`mark_failed`, retry on next refresh from scratch). Cursor-restart-from-beginning is cheap because Manifold pagination is fast. |
| Manifold's `is_filled` semantics for limit orders are unclear from the docs | Spec assumes: a fillable bet has either no `limit_prob` (market order, always filled) or `limit_prob is not None` AND `is_filled is True` (filled limit order). Verify against a sample of real responses during implementation; if the field semantics differ, narrow the filter (this is a tactical decision, not an architectural one). |

## Affected files (estimate)

**New:**
- `src/pscanner/corpus/manifold_enumerator.py`
- `src/pscanner/corpus/manifold_walker.py`
- `tests/corpus/test_manifold_enumerator.py`
- `tests/corpus/test_manifold_walker.py`
- `tests/corpus/test_manifold_e2e.py`

**Modify:**
- `src/pscanner/manifold/models.py` (add `resolution` field)
- `src/pscanner/manifold/db.py` (add `resolution TEXT` column + idempotent migration)
- `src/pscanner/manifold/client.py` (verify `get_market` parses the new field)
- `src/pscanner/manifold/repos.py` (handle `resolution` in `insert_or_replace`)
- `src/pscanner/store/db.py` (concatenate `MANIFOLD_SCHEMA_STATEMENTS` into `_SCHEMA_STATEMENTS`)
- `src/pscanner/corpus/resolutions.py` (add `record_manifold_resolutions`)
- `src/pscanner/corpus/repos.py` (platform-aware `_NOTIONAL_FLOORS` dict on `insert_batch`)
- `src/pscanner/corpus/cli.py` (`--platform` flag and dispatch on `backfill`/`refresh`/`build-features`)
- `tests/manifold/test_models.py`, `tests/manifold/test_db.py`, `tests/manifold/test_repos.py` (extend)
- `tests/corpus/test_resolutions.py`, `tests/corpus/test_repos_trades.py`, `tests/corpus/test_cli.py` (extend)
- `tests/store/test_db.py` or equivalent (assert manifold tables exist after `init_db()`)
- `CLAUDE.md` (add bullets: mana convention, ingestion CLI surface, MKT/CANCEL skip-pattern)

Roughly 5 new files and 12 modified files. ~600 lines of source change and ~600 lines of test additions.
