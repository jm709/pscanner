# V1 subgraph adapter for pre-April-2026 historical markets

Issue: [#193](https://github.com/jm709/polymarketScanner/issues/193)
Date: 2026-05-26
Status: design approved; pending implementation plan

## Purpose

Fill in `corpus_trades` for the 2,769 markets currently flagged
`corpus_markets.v1_history_pending = 1` (2,689 pure-V1 + 80 hybrid). The
current V2-only `pscanner corpus subgraph-backfill` cannot service them:
the V2 subgraph (`B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`) only
indexes events from `1775220779` (2026-04-03) onward. Pre-V2 fills live
on the V1 subgraph (`7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`),
which was re-pushed with a different schema than pscanner ever used.

## Scope

In:

- A V1 adapter module that emits the existing `OrderFilledEvent`
  dataclass so the downstream `event_to_corpus_trade` insert path is
  shared with V2 unchanged.
- A dispatcher that drives the V1 and V2 queues independently per
  `corpus_markets.v1_history_pending` and `truncated_at_offset_cap`.
- A new `corpus_markets.onchain_v1_processed_at INTEGER` sentinel column,
  separate from V2's `onchain_processed_at`, so hybrid markets carry both
  sentinels independently.
- CLI extension: `--subgraph-version v1|v2|both` (default `both`) and
  per-version subgraph-id flags.
- Pre-implementation investigation (`scripts/investigate_v1_schema.py`)
  and overlap-window verification (`scripts/verify_v1_units.py`) staged
  so the riskiest unknowns (V1 schema semantics and economic units) are
  resolved before any production code lands.

Out:

- Recovery of V1 rows with `marketId="0"` (~65% of a 20-row sample).
  Stage 0's investigation reports the recovery rate; if recoverable,
  this gets a separate follow-up issue rather than scope-creeping the
  adapter.
- Multi-platform extension. V1 is Polymarket-specific.
- Backwards-compatibility shims beyond the deprecated `--subgraph-id`
  alias (preserves the desktop's existing scripts; emits a deprecation
  log event).

## Background: V1 vs V2

The pre-#151 pscanner code in git history (`a809378^`) queried the
`makerAssetId_in` / `takerAssetId_in` shape under `orderFilledEvents`.
That deployment was replaced. The current V1 subgraph exposes
`orderFills` instead, with a different filter shape and different field
names.

| | V1 (`7fu2DWYK…`, frozen) | V2 (`B9mm21DK…`, live) |
|---|---|---|
| Coverage | 2025-07-15 → 2026-04-28 | 2026-04-03 → live |
| Entity | `orderFills` (`OrderFill`) | `orderFilledEvents` (`OrderFilledEvent`) |
| Filter field | `marketId` (String) + `outcomeIndex` (0/1) | `tokenId` (BigInt) + `side` (Int 0/1) |
| Maker/Taker | Flat `Bytes` hex string | Nested `Account { id }` |
| Amounts | `price` (BigInt) × `size` (BigInt) | `collateralAmount` + `tokenAmount` (BigInt) |
| Order ref | `order { id, marketId, outcomeIndex, side }` | denormalized into the event |

There is a ~25-day overlap window (Apr 3 – Apr 28) where both subgraphs
have data. Stage 1 (below) exploits this to prove the unit conversion
and BUY/SELL mapping before any bulk insert.

The pre-#151 V1 module in git history (`a809378^:src/pscanner/corpus/
subgraph_ingest.py`) is the recoverable starting point. **The
scaffolding (paginator, orchestrator skeleton, batch insert, sentinel
write loop) lifts cleanly. The adapter and query string must be
rewritten** because they targeted the prior (now-gone) V1 schema.

## Implementation stages

The riskiest unknown is the V1 schema semantics — specifically whether
`marketId="0"` rows are recoverable, and whether `outcomeIndex` ↔
`Order.side` actually maps to maker-POV BUY/SELL the way V2 does. Both
get resolved before adapter code is written.

### Stage 0 — Investigation script (~1h budget)

`scripts/investigate_v1_schema.py` (committed alongside the adapter PR).

1. Pull ~1000 V1 rows from `7fu2DWYK…` via the existing
   `SubgraphClient`.
2. Group by `marketId` format: bare-`<decimal>`, `"0-<decimal>"`, `"0"`.
   Report counts.
3. For each bare- and `"0-"`-prefix row, count hits against the local
   `asset_index`.
4. For a sample of `marketId="0"` rows, fetch the `Order { id,
   marketId, outcomeIndex }` parent and check whether the parent's
   `marketId` is a recognized asset_id.
5. Emit a short markdown report (`scripts/v1_investigation_report.md`,
   committed). The report's recovery-rate finding for `"0"` rows
   determines whether to file a follow-up issue, **not** whether Stage 2
   queries them. Stage 2 explicitly does not query the `"0"` cohort.

### Stage 1 — Overlap-window verification (~30m)

`scripts/verify_v1_units.py`.

1. Pick a market with confirmed trades in the Apr 3 – Apr 28 overlap
   window (use a query against `corpus_trades` to find one).
2. Pull that market's fills from both V1 and V2.
3. Reconcile row-by-row on `(transactionHash, maker)`. For each matched
   pair:
   - Assert `V1.price * V1.size / 1_000_000 == V2.collateralAmount /
     1_000_000` (within $0.01 rounding tolerance).
   - Assert the V1-derived `(makerAssetId, takerAssetId, side)` matches
     V2's.
4. If a mismatch is found, the spec is wrong and the adapter design
   needs revision. If all matches pass, commit a small fixture
   (`tests/corpus/fixtures/v1_v2_overlap.json`, ~5 paired rows) so the
   adapter unit test can re-assert the mapping forever.

### Stage 2 — Adapter, orchestrator, dispatcher

New files; the existing 519-line `subgraph_ingest.py` is not touched.

- `src/pscanner/corpus/subgraph_ingest_v1.py`:
  - V1 query string (see below).
  - `_paginate_v1` cursor-paginator (one query path per market — V1 has
    no maker/taker split).
  - `subgraph_v1_row_to_event(row) -> OrderFilledEvent`.
  - `iter_v1_market_trades(client, condition_id, asset_ids, ...)`
    generator.
  - `run_v1_subgraph_backfill(conn, client, ...)` orchestrator.
- `src/pscanner/corpus/subgraph_dispatch.py`:
  - `run_subgraph_backfill_dispatched(conn, v1_client, v2_client, ...,
    versions: Sequence[Literal["v1", "v2"]])` — drives each enabled
    version through its own orchestrator, then calls
    `_clear_truncation_flags` once at the end.

The V1 query (one pass per market, two `marketId` formats merged in one
`marketId_in`):

```graphql
query MarketFills($market_ids: [String!]!, $cursor_ts: BigInt!, $page_size: Int!) {
  orderFills(
    first: $page_size
    where: { marketId_in: $market_ids, timestamp_gt: $cursor_ts }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id transactionHash timestamp
    marketId outcomeIndex
    maker taker
    price size fee
    order { id marketId outcomeIndex side }
  }
}
```

`market_ids` is built per-market: for each `condition_id` we resolve the
two asset_ids via `AssetIndexRepo` (one YES, one NO) and submit both
their bare-decimal form **and** their `"0-<decimal>"` form (V1 stores
either, per the issue's format distribution).

The adapter normalizes each row into the existing `OrderFilledEvent`:

- `transactionHash` → `tx_hash` (lowercase, validated 0x66 length).
- `maker`, `taker` → flat lowercase hex; V1 already stores them as bare
  `Bytes` strings, no nested unwrap.
- `price` × `size` and `outcomeIndex` + `order.side` → derive
  `(makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled)`
  per the Stage-1-verified maker-POV BUY/SELL convention. The `1_000_000`
  divisor is shared with V2.
- `fee` → BigInt passes through.
- `block_number = 0`, `log_index = 0` (same convention as V2 — not read
  downstream).

### Stage 3 — Schema migration

Add `onchain_v1_processed_at INTEGER` (nullable) to `corpus_markets`:

- Declared on the `CREATE TABLE` in `db.py:_SCHEMA_STATEMENTS`.
- Idempotent `ALTER TABLE corpus_markets ADD COLUMN
  onchain_v1_processed_at INTEGER` entry in `_MIGRATIONS`, swallowed by
  the existing `OperationalError("duplicate column name")` handler.

V1 adapter sentinel-write rules (the hardened version applied to the new
column):

- On successful market drain with ≥1 row inserted: same transaction
  sets `onchain_v1_processed_at = now` AND `v1_history_pending = 0`.
- On 0-row drain: leave both columns alone, log `subgraph.v1.zero_events`.
- On adapter exception (per row): log per-row, continue; do not stamp.
- On transport/GraphQL failure: log per-market, continue to next; do
  not stamp.

`v1_history_pending` is **only** cleared by a successful V1 pass. It
remains 1 forever for unrecoverable markets, which is the operator's
re-run signal.

The V1 subgraph is frozen (last event 2026-04-28). Once a market's V1
pages are drained, they never grow. `onchain_v1_processed_at` is
effectively terminal once set; a stamped market drops out of the queue
permanently.

### Stage 4 — CLI surface

Extend `pscanner corpus subgraph-backfill`:

- Add `--subgraph-version v1|v2|both` (default `both`).
- Add `--v1-subgraph-id` (default `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`).
- Rename `--subgraph-id` internally to `--v2-subgraph-id`; the original
  `--subgraph-id` flag stays as a deprecated alias that sets
  `--v2-subgraph-id` and emits a `subgraph.cli.deprecated_flag` event.

CLI calls `run_subgraph_backfill_dispatched` with the selected versions
and the appropriate `SubgraphClient` instances (one per subgraph id;
both share the same `--api-key` and `--rpm` settings).

## Data flow

```
v1_history_pending=1 markets
        |
        v
_load_pending_v1_markets (ORDER BY total_trade_count DESC, LIMIT)
        |
        v
AssetIndexRepo bulk lookup: {condition_id -> (asset_id_yes, asset_id_no)}
        |
        v
per market:
    market_ids = [bare(yes), bare(no), "0-"+yes, "0-"+no]
    iter_v1_market_trades(client, condition_id, market_ids)
        |
        v
    paginate orderFills WHERE marketId_in: market_ids, timestamp_gt: cursor
        |
        v
    subgraph_v1_row_to_event(row) -> OrderFilledEvent
        |
        v
    event_to_corpus_trade(event, asset_index_lookup) -> CorpusTrade
        |
        v
    CorpusTradesRepo.insert_batch  (INSERT OR IGNORE; dups counted)
        |
        v
    on success: UPDATE corpus_markets SET onchain_v1_processed_at=?,
                v1_history_pending=0 WHERE condition_id=?
        |
        v
    next market

after all markets:
    _clear_truncation_flags(conn)  (shared helper; re-evaluates the V2 flag too)
```

The dispatcher runs the V2 path first (existing code), then the V1 path
(new code), then `_clear_truncation_flags` once. V1 only modifies its
own sentinel column and `v1_history_pending`, so V2 state is undisturbed.

## Error handling

| Failure | Behavior |
|---|---|
| HTTP/transport error on V1 query | Per-market: log `subgraph.v1.market_failed` with `condition_id`, status, attempt. Retry up to N=3 with backoff. Skip on persistent failure; sentinel stays NULL. |
| GraphQL error in 200 response body | Treat like HTTP error. Log the GraphQL error message. |
| Adapter `KeyError` / `ValueError` on one row | Log `subgraph.v1.row_skipped` with row id + reason. Continue with the page. |
| `UnresolvableAsset` from `event_to_corpus_trade` | Shared with V2's path. Log `subgraph.v1.unresolvable_asset` so we can quantify `asset_index` coverage gaps. |
| `UnsupportedFill` | Same as V2: log and skip. |
| Empty `AssetIndexRepo` lookup for the market | Early-exit the market with `subgraph.v1.no_asset_index`. Sentinel stays NULL. Operator runs `AssetIndexRepo.backfill_from_corpus_trades` to repair. |
| Subgraph returns 0 rows for a market | Log `subgraph.v1.zero_events`. Sentinel **not stamped**. |
| Duplicate-key inserts (rows already present from the May 5 OLD-schema run) | `INSERT OR IGNORE` absorbs. Per-batch dup count logged as `subgraph.v1.dups_dropped`. |
| Schema migration re-run | Idempotent ALTER swallowed by existing `_apply_migrations` handler. |

## Observability

Per-market and per-stage events the operator can grep:

```
subgraph.v1.market_started      cid=...   total_pending=N
subgraph.v1.market_complete     cid=...   inserted=K  dups=D  duration_s=...
subgraph.v1.zero_events         cid=...
subgraph.v1.market_failed       cid=...   reason=...
subgraph.v1.no_asset_index      cid=...
subgraph.v1.row_skipped         row_id=... reason=...
subgraph.v1.unresolvable_asset  asset_id=...
subgraph.v1.dups_dropped        cid=...   count=...
subgraph.cli.deprecated_flag    flag=--subgraph-id
subgraph.cli_summary            v1_markets_processed=...  v2_markets_processed=...  v1_inserted=...  v2_inserted=...
```

WARN on per-market failure; INFO on per-market success and final
summary.

## Tests

- `tests/corpus/test_subgraph_ingest_v1.py`:
  - Adapter unit tests: bare `marketId`, `"0-"`-prefixed `marketId`,
    both outcome indices, both `order.side` enum values. Use the
    `v1_v2_overlap.json` fixture from Stage 1 to assert maker-POV
    parity with V2.
  - Sentinel hygiene: 0-row drain leaves columns unchanged; successful
    drain stamps both `onchain_v1_processed_at` and `v1_history_pending`.
  - Hybrid-market integration: V2 ran (sentinel set), then V1 runs;
    both sentinels are set, `v1_history_pending=0`, trade-count delta
    matches.
  - `marketId="0"` server-side filter: a fixture row with
    `marketId="0"` does NOT land in the response (assert the GraphQL
    query carries the explicit `marketId_in` allowlist, not a wildcard).
  - `asset_index` missing: orchestrator early-exits the market; re-run
    after backfilling the index completes.
- `tests/corpus/test_subgraph_dispatch.py`:
  - `--subgraph-version v1` skips the V2 queue; `v2` skips V1; `both`
    runs both.
  - `--subgraph-id` alias maps to `--v2-subgraph-id` and emits the
    deprecation event.

Tests use synthetic in-memory fixtures throughout; no `pytest -m slow`
marker required.

Out of scope for the test suite (covered elsewhere):

- Live network tests against the V1 subgraph — handled once by Stage 1.
- The `marketId="0"` recovery cohort — a separate follow-up issue if
  Stage 0 finds recovery feasible.

## Acceptance criteria

- [ ] Stage 0 investigation report committed; `marketId` format
      distribution documented.
- [ ] Stage 1 overlap-window verification passes; fixture committed.
- [ ] V1 adapter parses real V1 rows into `OrderFilledEvent` matching
      V2's maker-POV BUY/SELL convention (proven by the fixture test).
- [ ] `onchain_v1_processed_at` column migration applied idempotently.
- [ ] Dispatcher routes V1-pending markets to V1 and V2-eligible markets
      to V2.
- [ ] A backfill run on production V1-pending markets inserts trades
      for the recoverable subset, leaves sentinels unchanged on 0-row
      drains, and reports a duplicate count against the May 5 prior run.
- [ ] CLI `--subgraph-version`, `--v1-subgraph-id`, `--v2-subgraph-id`
      flags work; deprecated `--subgraph-id` alias still functions and
      emits the deprecation event.
- [ ] Tests cover adapter, dispatcher, sentinel hygiene, hybrid
      markets, and the dup-handling path.

## Open follow-ups (not blocking)

- `marketId="0"` recovery — file separately if Stage 0's recovery rate
  warrants it. Likely requires fetching the `Order` parent's
  `marketId`/`outcomeIndex` rather than relying on the event's
  `marketId`.
- Drop+recreate secondary indexes around `_copy_to_sqlite` (#136) — not
  V1-specific but the next big perf lever for the corpus pipeline.

## References

- Issue: [#193](https://github.com/jm709/polymarketScanner/issues/193)
- Pre-#151 V1 module (recyclable scaffolding): `a809378^:src/pscanner/
  corpus/subgraph_ingest.py`
- V1 subgraph: `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/
  7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`
- V2 subgraph: `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/
  B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`
- Related: `_V2_SUBGRAPH_START_TS = 1775220779` in
  `pscanner.corpus.repos`.
