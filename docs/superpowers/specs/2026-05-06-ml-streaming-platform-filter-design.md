# ML streaming pipeline `--platform` filter design

Date: 2026-05-06
Status: pending review
Follow-up to: `2026-05-06-corpus-platform-column-design.md` (RFC #35 PR A, merged as #82)

---

## Goal

Thread a single-platform filter through `pscanner.ml.streaming.open_dataset` so `pscanner ml train --platform <name>` trains a model on rows for that platform only. Single-platform-per-run by design; multi-platform aggregation (training a shared model across platforms) is an explicit non-goal here and a future follow-up.

This design is the missing half of RFC #35 PR A. PR A added the `platform` column on shared corpus tables; PR A's spec called for an ML CLI flag, but the implementation deferred it because PR A landed against a corpus that had been forked before PR #39's streaming pipeline. This spec is the platform filter against the post-#39 streaming code path.

## Non-goals

- Multi-platform aggregation. Cases B (`--platform all`) and C (`--platform polymarket,manifold`) are deliberate non-goals. The design picks parameter shapes and SQL idioms that don't preclude them, but doesn't implement them.
- Removing `platform` from `pscanner.ml.preprocessing._NEVER_LOAD_COLS`. Keeping it excluded matches single-platform behavior; the column is a constant within a run and would only bloat the feature matrix.
- Tightening the now-latent `USING (condition_id)` joins in `pscanner.corpus.cli`, `pscanner.corpus.onchain_*`, `pscanner.corpus.subgraph_ingest`. The PR A final review flagged these (issues I1/I2); they affect Polymarket-only paths today and are an independent corpus-side hygiene PR.
- Schema changes. None.

## Refinement of PR A's "deferred ML filter"

PR A merged with this Codebase-conventions note:

> Filtering ML training to a single platform via a `--platform` flag on `pscanner ml train` is a follow-up against the streaming pipeline (#39).

This spec is that follow-up. It does NOT change any corpus-side or schema-side behavior; it only thread `platform` through `pscanner.ml.streaming` and `pscanner.ml.cli`, and tightens one SQL JOIN that became latent at PR A.

## Surface signatures

Every helper that touches platform-scoped data gains a `platform: str = "polymarket"` keyword-only parameter. The default preserves every existing call site (tests, `scripts/analyze_model.py`, internal callers) unchanged.

| Layer | File | Signature change |
|---|---|---|
| CLI flag | `src/pscanner/ml/cli.py:build_ml_parser` | adds `--platform` with `choices=["polymarket","kalshi","manifold"]`, default `"polymarket"` |
| CLI dispatch | `src/pscanner/ml/cli.py:_cmd_train` | reads `args.platform`, passes to `run_study(platform=...)` |
| Training entry | `src/pscanner/ml/training.py:run_study` | gains `platform: str = "polymarket"` keyword arg, passes to `open_dataset(platform=...)` |
| Streaming entry | `src/pscanner/ml/streaming.py:open_dataset` | gains `platform: str = "polymarket"`, threads through to all helpers, stores on `StreamingDataset` |
| Streaming dataset state | `src/pscanner/ml/streaming.py:StreamingDataset` | gains `_platform: str` private field; constructed by `open_dataset` and read by `dtrain`/`dval`/`val_aux`/`materialize_test` to build `_SplitIter` instances |
| Partition helper | `src/pscanner/ml/streaming.py:_partition_markets` | gains `platform: str = "polymarket"`, adds `WHERE platform = ?` to the `market_resolutions` SELECT |
| Encoder fit helper | `src/pscanner/ml/streaming.py:_fit_encoder_on_train` | gains `platform: str = "polymarket"`, adds `WHERE te.platform = ?` to the categorical-discovery SELECT |
| Row counter | `src/pscanner/ml/streaming.py:_count_split_rows` | gains `platform: str = "polymarket"`, adds `WHERE te.platform = ?` to each per-split count |
| Chunk iterator | `src/pscanner/ml/streaming.py:_SplitIter` | gains `platform: str` field (no default — supplied by `open_dataset`); JOIN to `market_resolutions` tightens to composite key; SELECT adds `WHERE te.platform = ?` |
| Analysis script | `scripts/analyze_model.py` | adds `--platform` arg, forwards to `open_dataset` |

**Why `str` and not a sequence today.** Single-platform-per-run is the contract. When multi-platform aggregation lands, the parameter type widens to `tuple[str, ...]`, the SQL `=` becomes `IN (...)`, and the `_SplitIter` JOIN already uses composite-key form so it doesn't need to change. The CLI `choices=` list is the only thing that gains an `"all"` value or multi-value support. The change is mechanical, not architectural.

## SQL changes

### `_partition_markets`

```sql
-- Today
SELECT condition_id, resolved_at FROM market_resolutions
ORDER BY resolved_at, condition_id

-- After
SELECT condition_id, resolved_at FROM market_resolutions
WHERE platform = ?
ORDER BY resolved_at, condition_id
```

### `_fit_encoder_on_train`

```sql
-- Today
SELECT DISTINCT side, top_category, market_category
FROM training_examples te
JOIN _p2_train tm USING (condition_id)

-- After
SELECT DISTINCT side, top_category, market_category
FROM training_examples te
JOIN _p2_train tm USING (condition_id)
WHERE te.platform = ?
```

`_p2_train` already holds only this platform's `condition_id`s (because `_partition_markets` filtered them), but adding the explicit `WHERE` is belt-and-suspenders: after PR A's composite PK, two platforms can hold markets with the same `condition_id` string. The temp-table partition isolates split membership; the `WHERE` clause isolates platform membership. Both are needed for correctness when non-Polymarket data exists.

### `_count_split_rows`

```sql
-- Today
SELECT COUNT(*) FROM training_examples te
JOIN {label} sm USING (condition_id)

-- After
SELECT COUNT(*) FROM training_examples te
JOIN {label} sm USING (condition_id)
WHERE te.platform = ?
```

Same belt-and-suspenders rationale.

### `_SplitIter.__iter__` — the load-bearing query

```sql
-- Today
SELECT {select_list}, mr.resolved_at
FROM training_examples te
JOIN market_resolutions mr USING (condition_id)
JOIN _split_markets sm USING (condition_id)
ORDER BY te.id

-- After
SELECT {select_list}, mr.resolved_at
FROM training_examples te
JOIN market_resolutions mr ON mr.platform = te.platform AND mr.condition_id = te.condition_id
JOIN _split_markets sm USING (condition_id)
WHERE te.platform = ?
ORDER BY te.id
```

Two changes:

1. **Composite-key JOIN to `market_resolutions`.** Replaces `USING (condition_id)` with explicit `ON` clause matching both columns of the composite PK. Required for correctness — without this, training silently cross-platform-joins once non-Polymarket data exists in either table.
2. **`WHERE te.platform = ?`.** Scopes the iteration to the requested platform.

The `_split_markets` temp table stays `condition_id`-only. Forward-comp note: multi-platform mode needs `(platform, condition_id)` tuples in the temp table and a corresponding JOIN tightening; flagged in code comment.

## `_NEVER_LOAD_COLS` (unchanged)

Stays at `frozenset({"id", "platform", *LEAKAGE_COLS})`. Within a single training run `platform` is constant; including it in the materialized DataFrame would bloat each chunk with a useless TEXT column and require special-casing in `build_feature_matrix` to drop it before the float cast.

The comment block updates to point at the active filter:

> `platform` is RFC #35's cross-platform tag. The streaming pipeline filters by platform via `WHERE te.platform = ?` (see `pscanner.ml.streaming.open_dataset`); the column is excluded at SELECT time because within a single training run it is a constant. Multi-platform aggregation (a future follow-up) would remove `platform` from this set and add it to `CATEGORICAL_COLS` so the encoder can one-hot it.

## Tests

### New file: `tests/ml/test_streaming_platform_filter.py`

Three behavioral tests using the existing `make_synthetic_examples_db` fixture, extended to seed both polymarket and kalshi rows:

1. **`test_open_dataset_defaults_to_polymarket`** — DB seeded with both platforms; `open_dataset(db)` returns row counts equal to the polymarket subset, not the union.
2. **`test_open_dataset_filters_to_kalshi`** — same DB; `open_dataset(db, platform="kalshi")` returns row counts equal to the kalshi subset.
3. **`test_split_iter_does_not_leak_other_platform`** — same DB; iterate every chunk via `ds.dtrain(device="cpu")`; assert the materialized X/y row counts match the platform-filtered counts, no other-platform rows appear.

### `tests/ml/test_cli.py` additions

Three small parser tests, mirroring PR A's analogous now-discarded Task 12:

- `test_train_parser_accepts_platform_flag`
- `test_train_parser_default_platform_is_polymarket`
- `test_train_parser_rejects_unknown_platform`

### `tests/ml/conftest.py` extension

`make_synthetic_examples_db` (and its `_seed_db_from_synthetic` helper) gains an optional `platform: str = "polymarket"` parameter. Existing call sites (no kwarg) continue inserting polymarket rows. The new platform-filter tests pass `platform="kalshi"` to seed kalshi rows, calling the helper twice to build a mixed-platform corpus.

### Existing tests stay unchanged

`tests/ml/test_streaming.py` (16 tests) and `tests/ml/test_training.py` (5+ tests) make no platform-related assertions. The new `platform: str = "polymarket"` defaults preserve their behavior — they continue testing the polymarket-only happy path.

## Forward-compatibility markers

Two code comments to drop, so a future engineer landing case B/C knows what to widen without re-deriving the design:

1. **`_populate_temp_table` docstring addendum** — notes that the table stores `condition_id` only and that multi-platform mode needs `(platform, condition_id)` tuples and a corresponding JOIN tightening in `_SplitIter`.
2. **`open_dataset` docstring** — documents that `platform: str = "polymarket"` is single-platform-per-run by design; multi-platform aggregation widens the parameter to `tuple[str, ...]` (or sequence) and switches the SQL filter from `=` to `IN`.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Composite-key JOIN syntax in `_SplitIter` regresses the polymarket-only happy path | Existing 16 streaming tests + 5+ training tests all run on polymarket-default fixtures; any regression on the JOIN surfaces immediately. |
| `_partition_markets` filter changes the train/val/test split for an existing polymarket-only corpus | It can't: filtering by platform on a single-platform corpus is a no-op. Verified by the unchanged-tests-stay-green property. |
| `_NEVER_LOAD_COLS` decision misaligns with multi-platform future | Documented in the updated comment block. Multi-platform mode is a future follow-up; this PR doesn't constrain the choice it'll make. |
| Test fixture extension breaks existing tests | `platform: str = "polymarket"` default on `make_synthetic_examples_db` means every existing call site (`tests/ml/test_streaming.py`, `tests/ml/test_training.py`) keeps inserting polymarket rows. |

## Affected files (estimate)

- `src/pscanner/ml/streaming.py` — most of the change (4 helpers + `_SplitIter` + `open_dataset` + `StreamingDataset`)
- `src/pscanner/ml/training.py` — one signature change, one call-site forward
- `src/pscanner/ml/cli.py` — flag definition + forward to `run_study`
- `src/pscanner/ml/preprocessing.py` — one comment update on `_NEVER_LOAD_COLS`
- `scripts/analyze_model.py` — flag + forward
- `tests/ml/conftest.py` — `make_synthetic_examples_db` parameter
- `tests/ml/test_cli.py` — 3 new parser tests
- `tests/ml/test_streaming_platform_filter.py` — new file, 3 behavioral tests
- `CLAUDE.md` — refresh the "ML pipeline platform filter" follow-up bullet to point at this PR rather than describing the work as pending

Roughly 9 files, ~250 lines of source change, ~150 lines of test additions.
