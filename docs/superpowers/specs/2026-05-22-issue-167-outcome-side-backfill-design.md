# Design — Outcome-side backfill (#167)

Date: 2026-05-22
Issue: [#167](https://github.com/jm709/pscanner/issues/167)
Predecessor: PR #166 (#159 forward-fix)

## Problem

PR #166 forward-fixed `market_walker._parse_trade` so new walks derive
`outcome_side` from `clob_token_ids` position via gamma. Existing
`corpus_trades` and `asset_index` rows written before the fix are still
wrong on team-name binary markets: both legs are stored as `outcome_side=NO`,
which inverts the supervised-learning label for half the YES-side trades
(`pscanner.corpus.examples.py:174`).

Empirical scope on the 2026-05-22 desktop corpus:

- 5,340 binary markets in `asset_index`
- **3,643 correct** (YES+NO leg pair)
- **1,697 buggy** (NO+NO leg pair) — 31.8% of all binary markets
- **95% of the bug is concentrated in `sports` (1,245) + `esports` (364)**, with a tail of `thesis` (81), `elections` (3), `culture` (2), `crypto` (1)
- V1 and V2 eras are about equally affected (V1 34%, V2 40%) — the V2
  subgraph backfill path inherits the bug because
  `event_to_corpus_trade` reads `outcome_side` from `AssetIndexRepo.get(asset_id)`,
  which in turn comes from `backfill_from_corpus_trades()` over buggy
  V1-walked rows. The bug propagates forward unless `asset_index` is
  corrected.

## Goals

1. Correct `asset_index.outcome_side` (+ `outcome_index`) for every binary
   market currently storing both legs as `NO`.
2. Correct `corpus_trades.outcome_side` for every trade on those markets.
3. Make the backfill resumable, idempotent, and re-runnable without state cleanup.
4. Keep Python RSS bounded (low MB) regardless of trade count.
5. Coexist with the live daemon — no need to pause it during the backfill.

## Non-goals

- ML retrain. Filed as a separate follow-up issue after Phase 4 (build-features) verification.
- `market_resolutions.outcome_yes_won` — already correct (written by `record_resolutions` from gamma's authoritative resolved outcome; bug doesn't apply).
- Multi-outcome (>2 leg) markets. Out of scope per #159; `outcome_side` doesn't apply cleanly.
- Periodic self-healing job. Forward-fix in #166 prevents reintroduction, so this is a one-shot.

## Architecture overview

A new CLI subcommand `pscanner corpus backfill-outcome-side` that:

1. Discovers buggy markets via a `GROUP BY condition_id HAVING ...` query on `asset_index`.
2. For each unbackfilled buggy market:
   a. `data.get_market_slug_by_condition_id(condition_id)` → slug
   b. `gamma.get_market_by_slug(slug)` → `Market`
   c. Build `{clob_token_ids[0]: "YES" (idx 0), clob_token_ids[1]: "NO" (idx 1)}`
   d. `UPDATE asset_index` for both rows (defensive — idempotent on the already-correct leg)
   e. `UPDATE corpus_trades` for both `(condition_id, asset_id)` pairs
   f. `UPDATE corpus_markets SET outcome_side_backfilled_at=NOW()`
   g. `COMMIT`
3. Logs progress every N markets.

```
pscanner corpus backfill-outcome-side
  [--db PATH] [--rpm N] [--limit N] [--dry-run] [--rebuild-after]

  ┌──────────────────┐
  │ Discover buggy   │  GROUP BY condition_id
  │ markets          │  HAVING COUNT(*)=2
  │                  │   AND COUNT(DISTINCT outcome_side)=1
  │                  │   AND MIN(outcome_side)='NO'
  │                  │   AND condition_id NOT IN
  │                  │     (SELECT condition_id FROM corpus_markets
  │                  │       WHERE outcome_side_backfilled_at IS NOT NULL)
  └────────┬─────────┘
           ▼
  ┌──────────────────┐  per market (one transaction):
  │ Resolve mapping  │   slug ← data.get_market_slug_by_condition_id
  │ via gamma        │   market ← gamma.get_market_by_slug(slug)
  │                  │   mapping = {token0: (YES, 0), token1: (NO, 1)}
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ UPDATE           │   asset_index   × 2 rows
  │ asset_index +    │   corpus_trades × ~2-3K rows (index-seek)
  │ corpus_trades +  │   corpus_markets.outcome_side_backfilled_at
  │ corpus_markets   │
  └────────┬─────────┘
           ▼
        COMMIT
```

## CLI surface

```
pscanner corpus backfill-outcome-side
  --db PATH            corpus DB (default: data/corpus.sqlite3)
  --rpm INT            gamma rate limit (default: 50)
  --limit INT          stop after N markets (default: no limit)
  --dry-run            log planned UPDATEs without writing
  --rebuild-after      run `build-features --rebuild` after success
```

`--dry-run` does the full discovery + gamma fetches + diff computation,
but skips the SQL UPDATEs. Used for a sanity-check pass before the real run.

`--rebuild-after` is a convenience flag that triggers
`pscanner corpus build-features --rebuild --engine duckdb` immediately on
successful completion. Optional — operators can run the rebuild manually.

## Schema additions

Add an idempotent migration to the `_MIGRATIONS` tuple in
`pscanner.corpus.db` (the place CLAUDE.md calls out for additive
`ALTER TABLE` statements):

```sql
ALTER TABLE corpus_markets ADD COLUMN outcome_side_backfilled_at INTEGER;
```

The `_apply_migrations` wrapper already swallows `OperationalError`
on `"duplicate column name"`, so re-running `init_corpus_db` after the
migration is a no-op. The column is `NULL` for every existing row and
gets `int(time.time())` written on a successful backfill. The work
queue uses `outcome_side_backfilled_at IS NULL` as the resumability
gate.

No new tables needed.

## Per-market transaction shape

```python
with corpus_conn:                # auto-commit on success, rollback on exception
    asset_repo.upsert(           # 2 UPDATEs against asset_index PK
        condition_id=condition_id,
        asset_id=clob_token_ids[0],
        outcome_side="YES",
        outcome_index=0,
    )
    asset_repo.upsert(
        condition_id=condition_id,
        asset_id=clob_token_ids[1],
        outcome_side="NO",
        outcome_index=1,
    )
    corpus_conn.execute(
        "UPDATE corpus_trades SET outcome_side = ? "
        "WHERE condition_id = ? AND asset_id = ?",
        ("YES", condition_id, clob_token_ids[0]),
    )
    corpus_conn.execute(
        "UPDATE corpus_trades SET outcome_side = ? "
        "WHERE condition_id = ? AND asset_id = ?",
        ("NO", condition_id, clob_token_ids[1]),
    )
    corpus_conn.execute(
        "UPDATE corpus_markets SET outcome_side_backfilled_at = ? "
        "WHERE condition_id = ?",
        (int(time.time()), condition_id),
    )
```

Verified via `EXPLAIN QUERY PLAN`: the `corpus_trades` UPDATE uses
`idx_corpus_trades_market_ts (condition_id=?)` as an index seek, not a
table scan. Wall time per market ~milliseconds.

`outcome_side` is part of the `corpus_trades` PRIMARY KEY composite, so
the UPDATE semantically does delete-then-reinsert at the storage level.
`tx_hash` is in the PK and unique per trade, so no collisions on the
new YES key — the rewrite is safe.

## Error handling per market

The catch-and-skip points:

| Failure | Behavior |
|---|---|
| `data.get_market_slug_by_condition_id` returns `None` | Log `corpus.backfill_outcome_side.no_slug`, leave market untouched (no sentinel write), continue. |
| `gamma.get_market_by_slug` returns `None` | Log `corpus.backfill_outcome_side.gamma_missing`, leave market untouched, continue. |
| Gamma raises | Same handling as above; log + skip. |
| `len(market.clob_token_ids) != 2` | Log `corpus.backfill_outcome_side.not_binary`, skip — would only fire if the market schema changed mid-flight. |
| Any of the UPDATEs raises | Transaction rolls back; market stays in the work queue for a future re-run. |

Markets that fail get re-attempted on every subsequent run. Successful
markets are gated out by the sentinel.

## Validation

Built-in post-run validation that re-runs the asset_index health query:

```python
buggy_remaining = conn.execute("""
    SELECT COUNT(*) FROM (
      SELECT condition_id FROM asset_index
       GROUP BY condition_id
      HAVING COUNT(*) = 2 AND COUNT(DISTINCT outcome_side) = 1
         AND MIN(outcome_side) = 'NO'
    )
""").fetchone()[0]
```

After a complete run, this should be `0`. Logged at INFO with the count.
Non-zero values indicate markets that failed resolution — operator should
re-run.

## Memory and wall-time analysis

- 1697 markets × 1 gamma call each, at 50 RPM = **~34 min** wall (dominant cost)
- 1697 × (2 asset_index UPDATEs + 2 corpus_trades UPDATEs) = ~6800 UPDATEs
- corpus_trades has 22M rows; the UPDATEs touch ~3-5M of them
  (~2-3K rows per market × 1697 markets)
- SQLite UPDATE on indexed rows: ~50-100K rows/sec ⇒ **~30-100s** total
- WAL growth bounded to one market's pages (a few MB), checkpointed on each commit
- Python RSS stays in single-digit MB throughout (one gamma response dict held at a time)

The gamma rate limit is the bottleneck; the disk work is negligible.

## Coexistence with the live daemon

The desktop's live copy-trading daemon writes to `data/pscanner.sqlite3`
and only READS from `data/corpus.sqlite3` (via `AssetIndexRepo` in the
token resolver). SQLite WAL mode handles one writer + many readers
cleanly. The daemon's only writes to `corpus.sqlite3` are token_resolver
upserts on first-sighting; those target rows we don't touch (new
tokens not yet in asset_index, populated via `clob_token_ids` position
correctly per #157/#158).

**No daemon pause needed.** Per-market backfill transactions take
milliseconds and serialize cleanly against the daemon's occasional
writes.

## Phase 4 — build-features rebuild

After the backfill completes (`buggy_remaining == 0`), run:

```
pscanner corpus build-features --rebuild --engine duckdb
```

This regenerates `training_examples` from `corpus_trades`, picking up
the corrected `outcome_side` and recomputing every `label_won` per
`examples.py:174`. Expected wall: ~1h on the 22M-trade corpus.

Spot-check after rebuild: pick a known-bad market from the issue body
(e.g. `nba-mil-ind-2025-04-22`), inspect the training_examples rows for
each leg, verify that YES-leg buy-trade labels reflect the actual winner.

## Phase 5 — separate issue for retrain

Filing a follow-up issue immediately after Phase 4 verifies the labels
look right. The retrain itself (`pscanner ml train --device cuda --n-jobs 1 --n-trials 100`)
is mechanical — ~1h 18m on the desktop, comparable to the 2026-05-15
baseline. The output is a new model artifact + metrics.json that gets
compared per-category against the prior baseline.

This is deferred so the user can kick off the retrain at a convenient
time independent of the backfill PR's merge cycle.

## Test surface

Unit tests in `tests/corpus/test_backfill_outcome_side.py`:

1. **Buggy-market discovery** — seed asset_index with 3 markets: NO+NO (target), YES+NO (skip), single-leg (skip). Assert discovery returns only the first.
2. **Sentinel respected** — same setup, mark the buggy market as already-backfilled via the sentinel column, assert it's excluded from the work queue.
3. **Per-market UPDATE flow** — seed corpus_trades + asset_index in the buggy state. Stub gamma to return a binary Market with known `clob_token_ids`. Assert the UPDATEs land correctly on both tables.
4. **Idempotent re-run** — run the backfill twice on the same market; assert results are identical (no double-rewrite).
5. **`--dry-run` mode** — same setup, assert no DB writes happen.
6. **Gamma missing → skip-and-continue** — stub gamma to return `None`; assert market is left untouched (no sentinel set) and the run continues to the next market.
7. **Non-binary market → skip-and-continue** — gamma returns a 3-outcome Market; assert skip.
8. **Validation query** — pre-/post-backfill counts on asset_index, assert the post-run count is `0`.

Integration test in `tests/corpus/test_backfill_outcome_side_integration.py`:

9. **End-to-end** — real `init_corpus_db`, seed one buggy sports-like market with 50 corpus_trades rows, stub gamma + data clients, run the CLI command via `Click`/argparse entry point, assert final state of both tables.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Gamma rate-limit spike during the run | Default `--rpm 50` shared with other corpus paths; configurable down via `--rpm 20` if needed. The run is resumable so partial completion is fine. |
| Market schema drift mid-run | Per-market transaction rolls back on any UPDATE failure; the market stays in the queue for a future re-run with updated code. |
| Live daemon write contention | SQLite WAL handles concurrent reader + occasional writer; per-market transactions take milliseconds; daemon's token_resolver writes target disjoint rows. Verified safe to leave the daemon running. |
| Backfilling a market whose corpus_trades were already correct via #166 | The discovery query gates on NO+NO in asset_index; markets walked post-#166 are YES+NO and won't be picked up. Defensive idempotence: re-running the UPDATE on a correct row is a no-op. |
| ALTER TABLE on the 63 GB corpus_markets in production | The column is nullable with no default. SQLite handles this as a metadata-only operation (no row rewrite). Should complete in seconds. |

## Open follow-ups (out of scope)

- ML retrain + per-category metric comparison (separate issue, filed after Phase 4 validation)
- A periodic self-check job that scans for NO+NO regressions (YAGNI — forward fix in #166 prevents this; revisit only if a regression appears)
- Migrating `outcome_side` derivation to be JOINed from `asset_index` rather than read from `corpus_trades` (architectural cleanup, would obviate the need for the corpus_trades UPDATE but is much bigger scope; explicitly out of scope per the brainstorming decision)
