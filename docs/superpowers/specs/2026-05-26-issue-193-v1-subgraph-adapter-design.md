# V1 subgraph adapter for pre-April-2026 historical markets

Issue: [#193](https://github.com/jm709/polymarketScanner/issues/193)
Date: 2026-05-26
Status: design approved; Stage 0 investigation complete; revised post-investigation

## Purpose

Fill in `corpus_trades` for the 2,769 markets currently flagged
`corpus_markets.v1_history_pending = 1` (2,689 pure-V1 + 80 hybrid). The
current V2-only `pscanner corpus subgraph-backfill` cannot service them:
the V2 subgraph (`B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`) only
indexes events from `1775220779` (2026-04-03) onward. Pre-V2 fills live
on the V1 subgraph (`7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`),
which still emits the *pre-#151* schema (verified by Stage 0
investigation, 2026-05-26).

## Scope

In:

- A V1 adapter module that emits the existing `OrderFilledEvent`
  dataclass so the downstream `event_to_corpus_trade` insert path is
  shared with V2 unchanged.
- A dispatcher that drives the V1 and V2 queues independently per
  `corpus_markets.v1_history_pending` and `truncated_at_offset_cap`.
- A new `corpus_markets.onchain_v1_processed_at INTEGER` sentinel
  column, separate from V2's `onchain_processed_at`, so hybrid markets
  carry both sentinels independently.
- CLI extension: `--subgraph-version v1|v2|both` (default `both`) and
  per-version subgraph-id flags.
- Stage 1 invariant verification — for one V1-pending market that
  traded across the April overlap, confirm the V1 row invariants the
  adapter depends on (BUY rows have `makerAssetId="0"` and
  `side="buy"`; SELL rows have `takerAssetId="0"` and `side="sell"`;
  both shapes present in the sample). Cross-subgraph amount
  reconciliation was originally planned but proven impossible — V1
  and V2 index different Polygon contracts and share no transactions.

Out:

- Multi-platform extension. V1 is Polymarket-specific.
- Backwards-compatibility shims beyond the deprecated `--subgraph-id`
  alias (preserves the desktop's existing scripts; emits a deprecation
  log event).

## Background: V1 vs V2 (post-investigation)

**The Stage 0 investigation (2026-05-26) overturned the original
schema-difference assumption baked into the issue body and CLAUDE.md.**
The V1 subgraph still serves the *pre-#151* `OrderFilledEvent` schema —
the schema pscanner used until April 2026. The "re-pushed with an
entirely different schema" claim was incorrect. What follows is the
verified reality.

| | V1 (`7fu2DWYK…`) | V2 (`B9mm21DK…`) |
|---|---|---|
| Coverage | 2025-04-07 → 2026-04-28 | 2026-04-03 → live |
| Entity | `orderFilledEvents` | `orderFilledEvents` |
| Maker/Taker shape | flat `Bytes` hex string | nested `Account { id }` |
| Asset identifiers | `makerAssetId` + `takerAssetId` | `tokenId` + `side` (int 0/1) |
| Filter field | `makerAssetId_in` and `takerAssetId_in` (two queries) | `market_in` (one query) |
| Amount fields | `makerAmountFilled` + `takerAmountFilled` (BigInt, 6-decimal) | `makerAmountFilled` + `takerAmountFilled` (BigInt, 6-decimal) — identical |
| Side encoding | `side` (string "buy"/"sell") | `side` (int 0/1) |
| Price field | `price` (decimal string, ratio) | (derived from amounts) |

**Key insight:** because the amount fields and the underlying
`(transactionHash, orderHash)` keys are identical across V1 and V2, the
amount-derivation logic in the existing `event_to_corpus_trade` can be
reused **without any change**. The V1 adapter's only job is parsing
flat-address rows into `OrderFilledEvent`, and the maker-POV BUY/SELL
derivation falls out of `makerAssetId=="0"` (BUY) vs `takerAssetId=="0"`
(SELL) — the same logic V2 uses.

The overlap window (April 3 – April 28) is *temporal*, not
transactional — both subgraphs were running concurrently but indexed
disjoint sets of transactions from disjoint Polygon Exchange contracts.
Stage 1 uses the window to sample real V1 production data for the
fixture, but cannot use it for cross-subgraph reconciliation.

The pre-#151 V1 module in git history (`a809378^:src/pscanner/corpus/
subgraph_ingest.py`) is the direct ancestor of the V1 adapter we need
to build today. Its query string and paginator lift almost verbatim;
the only adjustment is the `side` enum (now a string in V1) and `price`
field (now present in V1, ignored by the adapter).

## Implementation stages

### Stage 0 — Investigation script (complete)

Status: DONE 2026-05-26 (commit `e430e54` on
`feat/issue-193-v1-subgraph-adapter`).

`scripts/investigate_v1_schema.py` introspects the V1 subgraph schema
and reports format distributions for `makerAssetId` / `takerAssetId`
against the local `asset_index`. Output at
`scripts/v1_investigation_report.md`. Key findings recorded in the
"Background" section above.

### Stage 1 — Invariant verification on real V1 data (revised post-Task-2)

`scripts/verify_v1_units.py` (committed as `c8c726a` on the worktree branch).

**Background note:** the originally-spec'd verification — find the same
`(transactionHash, orderHash)` in both V1 and V2, assert identical
amounts — turned out to be impossible. V1 (`7fu2DWYK…`) and V2
(`B9mm21DK…`) index *different* Polygon Exchange contracts. A trade
exists in exactly one subgraph, never both. Empirically verified: 200
V1 tx hashes checked against V2 → 0 intersection. The
"overlap window" is temporal (both subgraphs were running 2026-04-03
through 2026-04-28), not transactional.

The revised Stage 1 instead proves the **invariants** the adapter
depends on, on real production V1 data:

1. Find a candidate market that traded across the overlap window via
   the local `corpus_trades` (a market with both pre-V2 and post-V2
   trades).
2. Pull ~200 V1 BUY rows (server-side filter `makerAssetId_in:
   <market's asset_ids>`) and ~200 V1 SELL rows (filter
   `takerAssetId_in: ...`) from the overlap window.
3. Pick representative samples (≥2 BUY, ≥2 SELL) and assert:
   - Every BUY row has `makerAssetId == "0"` and `side == "buy"`.
   - Every SELL row has `takerAssetId == "0"` and `side == "sell"`.
   - Both BUY and SELL shapes are represented in the kept sample.
4. Also commit a small set of V2 reference rows (different market —
   one V2 indexes — same overlap window) so the fixture documents the
   schema-shape delta the adapter must bridge (flat `maker`/`taker`
   vs nested `Account { id }`, `makerAssetId` + `takerAssetId` vs
   `tokenId` + `side`).
5. Commit as `tests/corpus/fixtures/v1_v2_overlap.json` with the
   structure `{condition_id, asset_index, cross_subgraph_match: false,
   v1_buy_rows: [...], v1_sell_rows: [...], v2_reference_rows: [...]}`.

The downstream Task 4 parser test asserts that the adapter, fed each
`v1_buy_rows` entry, produces an `OrderFilledEvent` with
`maker_asset_id == 0` and the correct `taker_asset_id`, and analogously
for `v1_sell_rows`. The V2 reference rows are used to document — not
assert against — the cross-schema shape difference.

If the V1 fixture's invariants fail (e.g. a BUY row has non-`"0"`
makerAssetId), the design needs revision. They held when Task 2 ran on
production V1 data (commit `c8c726a`).

### Stage 2 — Adapter, orchestrator, dispatcher

New files; the existing 519-line `subgraph_ingest.py` is not touched.

- `src/pscanner/corpus/subgraph_ingest_v1.py`:
  - V1 query string targeting `orderFilledEvents` with
    `makerAssetId_in: $assets` (one pass) plus a second pass with
    `takerAssetId_in: $assets` (no `_or` operator in The Graph means
    two paginated queries per market, like the pre-#151 code did).
  - `_paginate_v1_side` cursor-paginator (uses `id_gt` cursors, same
    pattern as the V2 path).
  - `subgraph_v1_row_to_event(row) -> OrderFilledEvent`.
  - `iter_v1_market_trades(client, asset_ids, page_size)` generator
    that drives both passes.
  - `run_v1_subgraph_backfill(conn, client, ...)` orchestrator.
- `src/pscanner/corpus/subgraph_dispatch.py`:
  - `run_subgraph_backfill_dispatched(...)` — drives each enabled
    version through its own orchestrator, then calls
    `_clear_truncation_flags` once at the end.

The V1 query (one pass per side per market):

```graphql
query MakerFills($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker
    makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled
    fee
  }
}

# A second identical query with `takerAssetId_in` in place of
# `makerAssetId_in`, called as a separate paginator pass.
```

`assets` is the list of CTF token ids for the market (from
`asset_index`). De-duplication across the two passes is handled by the
unique constraint on `corpus_trades` (`INSERT OR IGNORE`).

The adapter:

- `maker`, `taker` → flat lowercase hex.
- `makerAssetId`, `takerAssetId` → ints (already decimal-string CTF
  token ids; one will be `"0"` for the USDC side).
- `makerAmountFilled`, `takerAmountFilled` → ints (BigInt, 6-decimal,
  same as V2).
- `orderHash` → flat str (V1 stores it directly, unlike the pre-#151
  V2 code that had it too).
- `fee` → BigInt int.
- `side` and `price` from V1 are **ignored** — the maker-POV BUY/SELL
  derivation is already done downstream by `event_to_corpus_trade`
  based on which of `maker_asset_id`/`taker_asset_id` is zero.

### Stage 3 — Schema migration

Add `onchain_v1_processed_at INTEGER` (nullable) to `corpus_markets`:

- Declared on the `CREATE TABLE` in `db.py:_SCHEMA_STATEMENTS`.
- Idempotent `ALTER TABLE corpus_markets ADD COLUMN
  onchain_v1_processed_at INTEGER` entry in `_MIGRATIONS`, swallowed by
  the existing `OperationalError("duplicate column name")` handler.

V1 adapter sentinel-write rules:

- On successful market drain with ≥1 row inserted: same transaction
  sets `onchain_v1_processed_at = now` AND `v1_history_pending = 0`.
- On 0-row drain: leave both columns alone, log
  `subgraph.v1.zero_events`.
- On adapter exception (per row): log per-row, continue; do not stamp.
- On transport/GraphQL failure: log per-market, continue to next; do
  not stamp.

`v1_history_pending` is **only** cleared by a successful V1 pass.
Unrecoverable markets stay at `1` forever — they're the operator's
re-run signal.

### Stage 4 — CLI surface

Extend `pscanner corpus subgraph-backfill`:

- Add `--subgraph-version v1|v2|both` (default `both`).
- Add `--v1-subgraph-id` (default
  `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`).
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
_load_pending_v1_markets (ORDER BY total_volume_usd DESC, LIMIT)
        |
        v
asset_index lookup: asset_ids list for the condition_id
        |
        v
per market:
    iter_v1_market_trades(client, asset_ids)
        |
        v
    pass A: paginate orderFilledEvents WHERE makerAssetId_in: assets, id_gt: cursor
        |
        v
    pass B: paginate orderFilledEvents WHERE takerAssetId_in: assets, id_gt: cursor
        |
        v
    subgraph_v1_row_to_event(row) -> OrderFilledEvent
        |
        v
    event_to_corpus_trade(event, asset_index_lookup) -> CorpusTrade
        |
        v
    CorpusTradesRepo.insert_batch  (INSERT OR IGNORE; dups absorb pass-A/B overlap)
        |
        v
    on success: UPDATE corpus_markets SET onchain_v1_processed_at=?,
                v1_history_pending=0 WHERE condition_id=?
        |
        v
    next market

after all markets:
    _clear_truncation_flags(conn)  (shared helper; re-evaluates V2 flag too)
```

The dispatcher runs the V2 path first (existing code), then the V1
path (new code), then `_clear_truncation_flags` once. V1 only modifies
its own sentinel column and `v1_history_pending`, so V2 state is
undisturbed.

## Error handling

| Failure | Behavior |
|---|---|
| HTTP/transport error on V1 query | Per-market: log `subgraph.v1.market_failed` with `condition_id`, status, attempt. Skip on persistent failure; sentinel stays NULL. |
| GraphQL error in 200 response body | Treat like HTTP error. Log the GraphQL error message. |
| Adapter `KeyError` / `ValueError` on one row | Log `subgraph.v1.row_skipped` with row id + reason. Continue with the page. |
| `UnresolvableAsset` from `event_to_corpus_trade` | Shared with V2's path. Log `subgraph.v1.unresolvable_asset`. |
| `UnsupportedFill` | Same as V2: log and skip. |
| Empty `asset_index` lookup for the market | Early-exit the market with `subgraph.v1.no_asset_index`. Sentinel stays NULL. |
| Subgraph returns 0 rows for a market | Log `subgraph.v1.zero_events`. Sentinel **not stamped**. |
| Duplicate-key inserts (pass-B re-finds pass-A rows on internal swaps) | `INSERT OR IGNORE` absorbs. Per-market dup count logged as `subgraph.v1.dups_dropped`. Cross-run dups (May-5 OLD-backfill leftover) absorb the same way. |
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
  - Adapter unit tests: maker-zero (BUY) row, taker-zero (SELL) row.
    The Stage 1 fixture (`v1_v2_overlap.json`) carries `v1_buy_rows`
    and `v1_sell_rows` from production V1; the test feeds each into
    the adapter and asserts the derived `(maker_asset_id,
    taker_asset_id)` pair matches the row's `(makerAssetId,
    takerAssetId)`, and `making`/`taking` equal `makerAmountFilled`/
    `takerAmountFilled`. (Cross-subgraph amount equality cannot be
    asserted — the fixture's `v2_reference_rows` are present for
    schema-shape documentation only.)
  - Sentinel hygiene: 0-row drain leaves columns unchanged; successful
    drain stamps both `onchain_v1_processed_at` and
    `v1_history_pending`.
  - Hybrid-market integration: V2 ran (sentinel set), then V1 runs;
    both sentinels are set, `v1_history_pending=0`, trade-count delta
    matches.
  - Two-pass paginator: a fake client serves separate row sets to the
    `makerAssetId_in` and `takerAssetId_in` queries; the iterator
    yields the union. Cursor advancement verified per pass.
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

## Acceptance criteria

- [x] Stage 0 investigation report committed; V1 schema documented.
- [ ] Stage 1 overlap-window verification passes; fixture committed.
- [ ] V1 adapter parses real V1 rows into `OrderFilledEvent` matching
      V2's maker-POV BUY/SELL convention (proven by the fixture test).
- [ ] `onchain_v1_processed_at` column migration applied idempotently.
- [ ] Dispatcher routes V1-pending markets to V1 and V2-eligible
      markets to V2.
- [ ] A backfill run on production V1-pending markets inserts trades
      for the recoverable subset, leaves sentinels unchanged on 0-row
      drains, and reports a duplicate count.
- [ ] CLI `--subgraph-version`, `--v1-subgraph-id`, `--v2-subgraph-id`
      flags work; deprecated `--subgraph-id` alias still functions and
      emits the deprecation event.
- [ ] Tests cover adapter, dispatcher, sentinel hygiene, hybrid
      markets, two-pass paginator, and the dup-handling path.

## References

- Issue: [#193](https://github.com/jm709/polymarketScanner/issues/193)
- Pre-#151 V1 module (recyclable scaffolding): `a809378^:src/pscanner/
  corpus/subgraph_ingest.py`
- Stage 0 investigation report: `scripts/v1_investigation_report.md`
- V1 subgraph: `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/
  7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`
- V2 subgraph: `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/
  B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`
- Related: `_V2_SUBGRAPH_START_TS = 1775220779` in
  `pscanner.corpus.repos`.

## Revision history

- 2026-05-26 v1: original draft (assumed `OrderFill` entity, `marketId`
  + `outcomeIndex`, BigInt `price × size`).
- 2026-05-26 v2: rewritten after Stage 0 investigation proved the V1
  schema is the pre-#151 `OrderFilledEvent` shape.
  Dropped `marketId="0"` scope; dropped `price × size` unit-conversion
  concern; simplified Stage 1 verification to amount equality on
  matched `(tx_hash, orderHash)` pairs; switched the query plan from
  one `marketId_in` query to two `makerAssetId_in` / `takerAssetId_in`
  queries.
- 2026-05-26 v3 (current): Task 2 proved V1 and V2 index *different*
  Polygon Exchange contracts and share no transactions. Replaced
  Stage 1's cross-subgraph amount-equality assertion with on-V1
  invariant checks (BUY rows have `makerAssetId="0" + side="buy"`;
  SELL rows have `takerAssetId="0" + side="sell"`). Updated the
  fixture structure to `{v1_buy_rows, v1_sell_rows,
  v2_reference_rows, cross_subgraph_match: false}` and the
  downstream parser test to assert against `v1_buy_rows`/`v1_sell_rows`
  instead of an impossible `pairs` list.
