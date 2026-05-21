# Subgraph migration to the current Polymarket Orderbook subgraph — design

**Status:** Approved 2026-05-21. Implementation pending.
**Related issues:** #151 (this migration), #152 (downstream `SubgraphTradeCollector`).

## Goal

Migrate `pscanner corpus subgraph-backfill` from the stale subgraph `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY` (stopped indexing `OrderFilledEvent` rows on 2026-04-28) to the current real-time subgraph `B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`. The migration must produce `corpus_trades` rows byte-identical to what the old path produced for the same on-chain events, so that the larger backfill is fully idempotent against the existing corpus.

## Non-goals

- **No deletion of the eth_getLogs paths.** `onchain-backfill` / `onchain-backfill-targeted` and their modules stay in place. CLAUDE.md flags them for removal in a separate follow-up; not part of this PR.
- **No `OrderFilledEvent` dataclass changes.** Internal field names and types stay the same so every downstream consumer (`event_to_corpus_trade`, `_backfill_one_market`, `CorpusTradesRepo.insert_batch`) needs zero edits.
- **No new pip dependencies.** All work happens inside existing modules with existing imports.
- **No daemon/CLI surface changes.** The `pscanner corpus subgraph-backfill` flag set, exit codes, and resumability semantics stay identical.
- **No new tests for already-tested downstream logic.** `event_to_corpus_trade` has existing coverage; we don't re-test it.

## Architecture

The migration is contained to one module + one constant. Public function signatures stay unchanged so every caller and every downstream piece of code is unaffected.

### Modified files

| file | nature of change |
|---|---|
| `src/pscanner/corpus/subgraph_ingest.py` | Rewrite the `_TRADES_QUERY_*` constant(s), `_QUERY_TEMPLATE_FIELDS`, and `subgraph_row_to_event`. Collapse the existing maker/taker side-split into a single query. |
| `src/pscanner/corpus/cli.py` | One-line default change — the `--subgraph-id` flag's default flips from `7fu2DWYK...` to `B9mm21D...`. CLI surface and flag set unchanged. |
| `tests/corpus/test_subgraph_ingest.py` (if extant) | Update existing fixtures to the new schema shape; add two new unit tests pinning down the side-reconstruction convention. |

### Unchanged (verified by exploration)

- `pscanner.poly.subgraph.SubgraphClient` — generic GraphQL client, schema-agnostic.
- `OrderFilledEvent` dataclass at `pscanner.poly.onchain` — internal fields stay the same.
- `event_to_corpus_trade` at `pscanner.poly.onchain_ingest` — reads only `maker`, `maker_asset_id`, `taker_asset_id`, `making`, `taking`, `tx_hash`. As long as those four asset-related fields are reconstructed correctly, downstream is invariant.
- `_paginate_side`, `_backfill_one_market`, `_mark_processed`, `run_subgraph_backfill` — public signatures unchanged. The only body change is in `iter_market_trades`: it currently invokes `_paginate_side` twice (once with the maker-side query, once with the taker-side query) and merges the results; after the migration it invokes `_paginate_side` once with the single new query. Same public iterator semantics — still yields `(OrderFilledEvent, ts)` tuples in the same order.
- `_cmd_subgraph_backfill` CLI handler and dispatch table.
- `corpus_trades` SQLite schema — no migration needed.

### Subgraph endpoint

- **New ID:** `B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`
- **URL pattern:** `https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{subgraph_id}` (unchanged)
- **Auth:** existing `$GRAPH_API_KEY` env var (or `--api-key` flag) is sufficient.

## Schema mapping & parity invariant

The parity contract goes `OrderFilledEvent → event_to_corpus_trade → CorpusTrade`. The dataclass shape and the conversion function both stay unchanged. What changes is **how we populate the dataclass** from the new schema fields.

### Field-by-field mapping

| `OrderFilledEvent` field | OLD subgraph | NEW subgraph |
|---|---|---|
| `tx_hash` | `transactionHash` | `transactionHash` |
| `order_hash` | `orderHash` | `orderHash` |
| `maker` | `maker` (Bytes — direct address) | `maker.id` (nested `Account.id`) |
| `taker` | `taker` (Bytes — direct address) | `taker.id` (nested `Account.id`) |
| `maker_asset_id` | `makerAssetId` (BigInt) | **derived from `side` + `tokenId`** — see below |
| `taker_asset_id` | `takerAssetId` (BigInt) | **derived from `side` + `tokenId`** — see below |
| `making` | `makerAmountFilled` (BigInt) | `makerAmountFilled` (BigInt) — same convention, used directly |
| `taking` | `takerAmountFilled` (BigInt) | `takerAmountFilled` (BigInt) — same convention, used directly |
| `fee` | `fee` (BigInt) | `fee` (BigInt) |
| `block_number` | `blockNumber` (often absent) | `blockNumber` — present in new schema |
| `log_index` | not exposed (= 0) | not exposed (= 0) — matches old behavior |

### Derived reconstruction

The new schema collapses `makerAssetId` and `takerAssetId` into `tokenId` (= `Market.id`, the conditional token being traded) + `side` (Int: 0=BUY, 1=SELL, indicating the maker's order direction). Empirically verified against live subgraph rows: `makerAmountFilled` and `takerAmountFilled` carry the same maker/taker convention as the old schema, so `making` / `taking` flow through directly without any reconstruction. Only the two asset-id fields need derivation:

```python
side = int(row["side"])
token_id = int(row["tokenId"])

if side == 0:
    # Maker placed a BUY order: gave USDC, took conditional tokens.
    maker_asset_id = 0  # CTF Exchange convention for USDC
    taker_asset_id = token_id
elif side == 1:
    # Maker placed a SELL order: gave conditional tokens, took USDC.
    maker_asset_id = token_id
    taker_asset_id = 0
else:
    raise ValueError(f"unexpected side: {side}")

making = int(row["makerAmountFilled"])
taking = int(row["takerAmountFilled"])
```

`event_to_corpus_trade` then derives BUY/SELL via `maker_gives_usdc = (maker_asset_id == 0)`, picks `usdc_amount` / `ctf_amount` accordingly, and computes `price = usdc / ctf`, `size = ctf / 1e6`, `notional_usd = usdc / 1e6`. The resulting `CorpusTrade` is byte-identical to what the old path emitted.

### What we intentionally ignore from the new schema

The new schema also exposes `price` (BigDecimal), `size` (BigDecimal), `collateralAmount` (BigInt), and `tokenAmount` (BigInt). We **deliberately ignore** all four — the existing `makerAmountFilled` / `takerAmountFilled` BigInts give us exactly what the old path used, and `event_to_corpus_trade`'s downstream integer-division math is preserved. Importing the subgraph's pre-computed decimals would risk floating-point divergence vs the existing corpus rows.

`exchange` (CTF / NEG_RISK enum), `builder`, `metadata` are also new fields we ignore — they don't affect the `CorpusTrade` row.

## Query strategy

### Single query, server-side `market_in` filter

```graphql
{
  orderFilledEvents(
    where: { market_in: $assets, id_gt: $cursor }
    first: 1000
    orderBy: id
    orderDirection: asc
  ) {
    id
    orderHash
    transactionHash
    timestamp
    blockNumber
    maker { id }
    taker { id }
    market { id }
    tokenId
    side
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
```

Replaces both `_TRADES_QUERY_MAKER_SIDE` and `_TRADES_QUERY_TAKER_SIDE`. Filter verified live: `market_in: [<tokenId>]` returns events for that specific conditional token. To get every fill on a binary market, pass both YES and NO tokenIds (which our local `asset_index` already enumerates).

### Pagination

Unchanged from the old path — keyset cursor on `id` (`id_gt: $cursor`, `orderBy: id`). The new schema's `OrderFilledEvent.id` is a unique String ID per event and lexically orderable.

### Asset-id list construction

`_load_market_asset_ids(conn, condition_id)` already returns a list of asset IDs for a condition_id from the corpus's `asset_index` table. No change. The list passes straight into `market_in: $assets`.

## Error handling

Most failures are at the GraphQL client layer (existing `SubgraphClient` handles 429s, 5xx, network timeouts via retry-with-backoff). The migration introduces three new per-row failure modes in the rewritten `subgraph_row_to_event`:

| failure | handling |
|---|---|
| Missing nested field (`maker.id`, `taker.id`, `market.id` absent) | Raise `KeyError("missing required field: <name>")`. Caller's per-row loop catches and skips. |
| Unparseable `side` (not 0 or 1) | Raise `ValueError("unexpected side: <value>")`. Same skip path. Defensive — schema says Int but neg-risk markets might surprise us. |
| `collateralAmount` / `tokenAmount` not BigInt-parseable | Raise `ValueError("unparseable amount: <field>")`. Same skip path. |
| `tokenAmount == 0` (zero-size fill) | Already handled downstream in `event_to_corpus_trade` via `UnsupportedFill`. No change. |
| Maker == exchange contract address | Already handled downstream in `event_to_corpus_trade` via `UnsupportedFill`. No change. |
| `block_number` absent | New schema does expose `blockNumber`; if it's somehow missing, default to 0 (matches old behavior — the field was never load-bearing). |

The existing GraphQL-error crash path stays — if the subgraph itself rejects the query (schema drift, malformed query), `SubgraphClient.query` raises `RuntimeError("GraphQL errors: ...")` and the operator sees a hard failure. Don't silently fall back.

## Testing

### Existing tests to update

| test category | action |
|---|---|
| `subgraph_row_to_event` unit tests | Rewrite fixtures to the new schema shape (nested `maker.id` / `taker.id`, `market { id }`, `tokenId`, `side`, `collateralAmount`, `tokenAmount`). Same assertions on the resulting `OrderFilledEvent`. Both BUY (`side=0`) and SELL (`side=1`) cases required. |
| Query string assertion tests | Update any test that compares the GraphQL body string — they now assert the single-query shape with `market_in: $assets`. |
| `iter_market_trades` integration tests | Update fake-client fixtures to return new-schema rows. Behavior assertions (pagination, filter coverage) stay the same. |

### New unit tests

Two new tests in `tests/corpus/test_subgraph_ingest.py` (or wherever the existing tests live) pin down the side-reconstruction convention so a future schema tweak can't silently flip directions:

1. **`test_subgraph_row_to_event_buy_side`** — given `side=0`, `tokenId=T`, `makerAmountFilled=100`, `takerAmountFilled=200`, assert `OrderFilledEvent.maker_asset_id == 0`, `.taker_asset_id == T`, `.making == 100`, `.taking == 200`.
2. **`test_subgraph_row_to_event_sell_side`** — given `side=1`, `tokenId=T`, `makerAmountFilled=100`, `takerAmountFilled=200`, assert `.maker_asset_id == T`, `.taker_asset_id == 0`, `.making == 100`, `.taking == 200`.

### What we do NOT add

- **No parity unit test against the old subgraph.** That comparison happens at the validation gate level (live one-market run, see below). Adding a unit-level mock-old-vs-mock-new comparison duplicates effort without catching anything the integration step wouldn't.
- **No CI-bound end-to-end run** against the live subgraph. The validation gate is a manual one-market run.

## Validation gate

Before kicking off the larger backfill against the new subgraph, run a parity check on one already-corpused market:

1. Pick a market with `corpus_markets.onchain_processed_at IS NOT NULL` whose trades all landed before 2026-04-28 (fully backfilled via the old subgraph, predates the indexing freeze).
2. Snapshot its existing rows from `corpus_trades` — `SELECT * FROM corpus_trades WHERE condition_id = ? ORDER BY tx_hash, asset_id, wallet_address INTO /tmp/parity_old.csv`.
3. Re-run the new-subgraph backfill against just that market — temporarily clear its `onchain_processed_at` (`UPDATE corpus_markets SET onchain_processed_at = NULL WHERE condition_id = ?`), then `pscanner corpus subgraph-backfill --limit 1` after positioning the resume cursor at that condition.
4. Snapshot the rows again into `/tmp/parity_new.csv`.
5. `diff /tmp/parity_old.csv /tmp/parity_new.csv` — expected: zero diff.
6. If clean: kick off the larger backfill.
7. If a column diff surfaces, the failure mode is localized to the side-mapping reconstruction. Most likely cause is a sign-convention or amount-direction error.

## CLI / docs touch-ups

- `src/pscanner/corpus/cli.py` — change the `--subgraph-id` default constant value from `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY` to `B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`. Help text stays the same.
- `CLAUDE.md` — update the existing bullet on the subgraph migration. Today it flags the old ID as stale and mentions the new ID as a pending swap behind `--subgraph-id`. After this PR lands: the new ID becomes the default, the old ID becomes "deprecated — pass `--subgraph-id 7fu2DWYK...` if you specifically need pre-migration history." Drop the "#151 will obsolete these" forward-references that are now resolved.

## Implementation order (suggested for the plan)

1. Update tests/fixtures first (TDD): rewrite the existing `subgraph_row_to_event` tests to the new schema, watch them fail.
2. Rewrite the GraphQL query constant + `subgraph_row_to_event` to make them pass.
3. Add the two new side-mapping unit tests, watch them pass (or fail then fix the reconstruction).
4. Update any query-string-assertion tests + integration tests.
5. Flip the `--subgraph-id` default in the CLI.
6. Update CLAUDE.md.
7. Manual validation gate per the section above.
8. Commit each step separately so the diff is reviewable.

## Out-of-scope follow-ups

- **Deletion of eth_getLogs paths** — tracked separately, CLAUDE.md flags them. Not part of this PR.
- **`SubgraphTradeCollector` daemon collector** — #152, depends on this migration shipping but is its own scope.
- **Backwards-compat shim for the old subgraph ID** — none planned. Operators who pin the old ID via `--subgraph-id` will hit the new parser and get a schema mismatch; that's the right failure mode (old subgraph is broken anyway since April 28).
