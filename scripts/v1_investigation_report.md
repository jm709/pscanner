# V1 schema investigation report

Subgraph: `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY` (V1)
V2 start timestamp: `1775220779` (2026-04-03T00:00:00Z)

## Confirmed field names on `OrderFilledEvent`

```
id, transactionHash, timestamp, orderHash, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee, blockNumber, side, price
```

**Key schema difference from V2:** V1 uses `makerAssetId` + `takerAssetId`
as separate columns. V2 collapses these into `tokenId` + `side`.
The `marketId` / `outcomeIndex` fields from CLAUDE.md do NOT exist —
the CLAUDE.md note referred to the V1 entity name, not field names.

## Sample A: 1000 most-recent rows (desc by timestamp)

- Oldest in sample: `1777374022` (Unix seconds)
- Newest in sample: `1777374040` (Unix seconds)
- **Note:** V1 subgraph is NOT frozen — these rows are from
  AFTER V2 start.
  The subgraph continues to index new events alongside V2.

### `makerAssetId` format distribution

| format | count | pct |
|---|---|---|
| zero | 866 | 86.6% |
| decimal | 134 | 13.4% |

### `takerAssetId` format distribution

| format | count | pct |
|---|---|---|
| decimal | 866 | 86.6% |
| zero | 134 | 13.4% |

### `side` distribution (sanity-check)

| side | count |
|---|---|
| 'buy' | 866 |
| 'sell' | 134 |

### asset_index recovery (asset_index size: 8,626)

| field | hits | total | pct |
|---|---|---|---|
| makerAssetId | 0 | 1000 | 0.0% |
| takerAssetId | 0 | 1000 | 0.0% |
| union (either side) | 0 | 1000 | 0.0% |

**Expected 0% for most-recent sample:** `makerAssetId` is `"0"` on BUY orders
(the other side is the token). The decimal `takerAssetId` values are CTF token IDs
for markets not yet in our `asset_index` (which only covers 8,626 markets we have
already indexed). Format is identical to V2 — the 0% rate is a coverage gap, not
a format mismatch.

## Sample B: 200 historical rows (ts < 1775220779)

- Oldest in sample: `1775220775`
- Newest in sample: `1775220777`

### asset_index recovery (historical pre-V2 rows)

| field | hits | total | pct |
|---|---|---|---|
| makerAssetId | 0 | 200 | 0.0% |
| takerAssetId | 2 | 200 | 1.0% |
| union (either side) | 2 | 200 | 1.0% |

## Findings and adapter design implications

1. **V1 is not frozen.** It continues indexing new trades. The historical gap
   for #193 is specifically the `ts < V2_START` window — markets that only
   appear in V1 because they closed before V2 launched.

2. **No `marketId` field.** The plan's concern about `marketId="0"` recovery
   is moot — V1 has no `marketId` field at all. Drop that concern from the
   follow-up issue scope.

3. **Token ID format is identical.** V1 `makerAssetId`/`takerAssetId` are
   the same large-decimal CTF token IDs as V2 `tokenId`. The Stage 2 adapter
   can look up token IDs in `asset_index` using the same key format.

4. **BUY orders encode as makerAssetId="0" + takerAssetId=<token>.**
   The token to look up is always on the non-zero side. The adapter should
   use `takerAssetId if makerAssetId == "0" else makerAssetId` for routing.

5. **`asset_index` coverage gap.** The 0% recovery rate reflects that our
   `asset_index` only covers 8,626 markets we've already processed via V2.
   Historical V1 markets have different token IDs (different markets). The
   adapter will need to build `asset_index` entries for V1 markets via the
   existing `_resolve_outcome_side_and_persist` pathway — same as V2.

**Decision on `marketId="0"` follow-up issue:** NOT needed. No such field.