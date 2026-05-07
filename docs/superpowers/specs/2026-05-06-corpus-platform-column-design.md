# Corpus `platform` column — RFC #35 PR A design

Date: 2026-05-06
Status: pending review
Refines: `2026-05-04-multi-platform-rfc.md` decision #4

---

## Goal

Make the five shared corpus tables platform-aware so future Kalshi and Manifold rows can coexist with the existing Polymarket corpus. Zero changes to daemon tables, zero changes to detector logic, zero changes to feature engineering. The deliverable is one focused PR: schema migration + repo signature changes + an ML preprocessing filter.

This spec is a prerequisite for the larger Kalshi-corpus and Manifold-corpus integration specs to follow. Landing it independently leaves `main` in a clean intermediate state — the corpus is ready for any future platform whether or not those integration specs ship.

## Non-goals

- Renaming legacy columns (`condition_id`, `tx_hash`, `asset_id`, `wallet_address`). Renaming would cascade through every detector, repo, evaluator, and ML feature — far outside this PR's scope. Convention: for non-Polymarket platforms, `condition_id` holds the platform-native market identifier (Kalshi ticker, Manifold market hash). This matches the existing `event_tag_cache.event_id` precedent (a column that actually stores slugs).
- Changes to `pscanner corpus *` CLI commands. They stay Polymarket-hardcoded; the integration specs add `--platform` flags where relevant.
- Cross-platform ML training. The ML pipeline gains a `--platform` filter that defaults to `polymarket` (preserves current behavior); cross-platform aggregation is a follow-up once Kalshi/Manifold features stabilize.
- Daemon tables (`kalshi_*`, `manifold_*`, `market_cache`, `wallet_trades`, etc.). Per RFC decision #1, these stay namespaced and are untouched here.
- The `alerts` table. The RFC mentions adding `platform` there too; that's daemon-side and lands when a detector first emits a non-Polymarket alert, not as part of corpus migration.

## Refinement of RFC decision #4

The original RFC specified a simple `ALTER TABLE ADD COLUMN platform TEXT NOT NULL DEFAULT 'polymarket'`. This spec refines that decision: **`platform` becomes part of the primary key on every shared corpus table**, not just an indexed column.

Rationale: the PK guarantees no implicit "trust me, the strings won't collide" between Polymarket condition_ids (66-char hex), Kalshi tickers (`KX...`), and Manifold market hashes (~12 chars). The cost is a one-time table-copy migration on `corpus_trades` (~15.9M rows) and `training_examples` (~15.5M rows) because SQLite cannot `ALTER PRIMARY KEY`. On the dev WSL2 box this is roughly 5–15 minutes of disk I/O, run once.

## Tables touched

| Table | Old PK | New PK | Migration shape |
|---|---|---|---|
| `corpus_markets` | `condition_id` | `(platform, condition_id)` | Table-copy |
| `corpus_trades` | UNIQUE`(tx_hash, asset_id, wallet_address)` | `(platform, tx_hash, asset_id, wallet_address)` | Table-copy (~15.9M rows) |
| `market_resolutions` | `condition_id` | `(platform, condition_id)` | Table-copy |
| `training_examples` | UNIQUE`(tx_hash, asset_id, wallet_address)` | `(platform, tx_hash, asset_id, wallet_address)` | Table-copy (~15.5M rows) |
| `asset_index` | `asset_id` | `(platform, asset_id)` | Table-copy |

Tables explicitly **not** touched:

- `event_tag_cache` — Polymarket-specific by design (slugs, neg-risk flags, gamma cache).
- `corpus_state` — key/value table; if cross-platform state ever matters, keys can be prefixed (`kalshi:onchain_last_block`).
- All `kalshi_*`, `manifold_*`, and other daemon tables.

## Column shape

```sql
platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold'))
```

Existing rows are backfilled to `'polymarket'` during the migration. The `CHECK` constraint catches typos at insert time; new platforms require a one-line schema bump.

## Migration mechanics

SQLite cannot alter primary keys, so each migration is a four-step dance inside a per-table transaction:

```python
def _migrate_add_platform_to_pk(conn, table, old_cols, new_pk_cols, indexes):
    """Idempotent: skip if `platform` column already exists on `table`."""
    if _column_exists(conn, table, "platform"):
        return
    # Inside one transaction:
    #   CREATE TABLE {table}__new (... platform ..., PRIMARY KEY(<new_pk_cols>))
    #   INSERT INTO {table}__new SELECT 'polymarket', <old_cols> FROM {table}
    #   DROP TABLE {table}
    #   ALTER TABLE {table}__new RENAME TO {table}
    #   Recreate non-PK indexes
```

**Idempotency:** the migration checks `PRAGMA table_info(<table>)` for the `platform` column. Present → skip. This means existing on-disk corpora migrate exactly once; fresh DBs go straight to `_SCHEMA_STATEMENTS` (which contains the new shape inline) with no migration step.

**Order** — small independent tables first, then the heavy two:

1. `asset_index`
2. `corpus_markets`
3. `market_resolutions`
4. `corpus_trades` (~15.9M rows)
5. `training_examples` (~15.5M rows)

**Per-table transactions**, not one giant transaction. SQLite's WAL handles per-table fine; one giant transaction can't roll back atomically across `DROP TABLE` boundaries anyway. Each migration logs `info("migrated N rows in Xs", table=...)` so progress is visible.

**Operational notes** for the two corpora (WSL2 dev, desktop training box):

- Migration runs on next `init_corpus_db()` call.
- Estimated 5–15 min for the two big tables on a typical SSD.
- Recommend backup before first migrated run: `cp data/corpus.sqlite3 data/corpus.sqlite3.pre-pra-bak`.

## Repo changes

Each method that touches a market key gains a `platform: str = "polymarket"` parameter. Existing callers (corpus refresh, build-features, ml train) continue working unchanged via the default.

| Repo | Methods needing `platform` |
|---|---|
| `CorpusMarketsRepo` | `insert_pending`, `next_pending`, `get_last_offset`, `mark_in_progress`, `record_progress`, `mark_complete`, `mark_failed` |
| `CorpusTradesRepo` | `insert_batch` (read `platform` from row), `iter_chronological` |
| `MarketResolutionsRepo` | `upsert`, `get`, `missing_for` |
| `TrainingExamplesRepo` | `insert_or_ignore` (read `platform` from row), `truncate` (gains `platform` arg, per-platform truncation), `existing_keys` (gains `platform` arg, filters internally — return tuple shape unchanged) |
| `AssetIndexRepo` | `upsert`, `get`, `backfill_from_corpus_trades` |

For row-carrying repos (`corpus_trades`, `training_examples`), the dataclass row type gains a `platform: str = "polymarket"` field rather than passing `platform` as a method parameter. Cleaner `insert_batch` signature; the platform travels with the data.

## ML preprocessing changes

`pscanner.ml.preprocessing.load_dataset`:

- New parameter `platform: str = "polymarket"`.
- Adds `WHERE training_examples.platform = ?` to the load query.
- Default preserves current behavior — ML training stays Polymarket-only unless explicitly overridden.

`pscanner ml train` CLI:

- New `--platform polymarket|kalshi|manifold` flag, defaults to `polymarket`.
- Forwarded to `load_dataset(platform=...)`.

## Out of scope (deferred to integration specs)

- `pscanner corpus backfill`, `corpus refresh`, `corpus build-features`, etc. stay Polymarket-hardcoded. The Kalshi and Manifold integration specs add `--platform` flags and per-platform code paths where appropriate.
- Cross-platform model training. The current ML pipeline assumes a fixed feature schema; mixing Polymarket (with wallet history) and Kalshi (anonymous trades) requires schema reconciliation that belongs in the Kalshi integration spec.
- Removing `polymarket` defaults. Until Kalshi/Manifold collectors actually run, defaults preserve every existing call site. A future cleanup PR can remove the default once every caller passes `platform` explicitly.

## Testing

- **Migration test**: build a temp DB with the old schema, insert sample rows in each table, call `init_corpus_db()`, assert every row has `platform='polymarket'` and PK is composite (verify via `PRAGMA table_info` and `PRAGMA index_list`).
- **Idempotency test**: call `init_corpus_db()` twice on a fresh DB and on a migrated DB; assert no errors, row counts unchanged, PK shape unchanged.
- **Cross-platform insert test**: after migration, insert rows with `platform='kalshi'` and `platform='manifold'`; assert they coexist with polymarket rows and `WHERE platform=?` queries segregate correctly.
- **Repo backwards-compat test**: existing repo tests run unchanged (default `platform='polymarket'` preserves behavior).
- **ML preprocessing test**: `load_dataset` with no platform arg returns only polymarket rows; with `platform='kalshi'` returns only kalshi rows (empty in PR A but the SQL filter is exercised on a seeded test DB).
- **CHECK constraint test**: inserting a row with `platform='nonsense'` raises `IntegrityError`.

All tests use `tmp_db`-style in-memory SQLite fixtures so the migration runs without touching the real corpus on disk.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Migration corrupts a 15M-row table on the dev or desktop corpus | Per-table transaction, backup recommendation in operational notes, run on dev box first before desktop. |
| Some caller already passes a Polymarket-shaped tuple/positional arg that breaks when `platform` is added to row dataclasses | Search-and-update at PR time; ty type checking surfaces these. Repo dataclass field has a default so positional inserts still work unless the caller relied on field ordering. |
| `ty` complaints from new `platform: str` parameter on dozens of repo methods | Default value makes most existing callers happy without edits; remaining ones get explicit `platform="polymarket"`. |
| The `PRAGMA index_list` / `PRAGMA table_info` checks miss an edge case and the migration runs twice on a real corpus | Idempotency test on a previously-migrated DB confirms no-op behavior; the `_column_exists` check is the gate. |

## Affected files (estimate)

- `src/pscanner/corpus/db.py` — schema statements + migration helpers (the bulk of the change).
- `src/pscanner/corpus/repos.py` — repo signatures + row dataclasses.
- `src/pscanner/corpus/examples.py` — passes `platform` through to `TrainingExamplesRepo.insert_batch` if it materializes platform on the row (likely).
- `src/pscanner/ml/preprocessing.py` — `load_dataset` filter.
- `src/pscanner/ml/cli.py` — `--platform` flag.
- `tests/corpus/test_db.py`, `tests/corpus/test_repos.py`, `tests/ml/test_preprocessing.py` — test additions.
- `CLAUDE.md` — short note documenting the `platform` column and the legacy `condition_id`-as-market-key convention.

Roughly 8–12 files. No new modules.
