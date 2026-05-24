# Refactor plan — corpus-ml slice (Phase 2)

Worktree: `/home/macph/projects/pscanner-worktrees/review-corpus-ml`. Branch: `refactor-review/corpus-ml` off clean main.

Per the orchestrator's Phase 2 brief: plan-and-execute T1.2a→T1.2b→T1.2c→T1.2d→T1.2e→T1.2f→T1.7→T2.18→T2.19→T2.20→T2.24, then plan-only T3.22, T3.26, T3.28. Each PR is one branch + one logical change unless commit-shape notes say otherwise.

Verify command (run before every commit): `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

Out-of-scope guardrails:

- No touches to `corpus/repos.py` or `corpus/db.py` (store-layer).
- No touches to `daemon/live_history.py` (strategies).
- No `--no-verify`, no force pushes, no rebases.
- Load-bearing tests that MUST stay green: `tests/corpus/test_feature_projection_parity.py`, `tests/daemon/test_live_history_parity.py`, `tests/corpus/test_db.py::test_apply_migrations_adds_platform_to_existing_corpus`.

---

## T1.2a — Delete eth_getLogs corpus path

**PR title:** `chore(corpus): delete eth_getLogs backfill path (#46 supersedes)`

**Commit shape:** Single commit. The relocation of `clear_truncation_flags` happens in the same commit because deleting the file without first relocating breaks `subgraph_ingest.py`.

**Files touched:**

- Delete `src/pscanner/corpus/onchain_backfill.py` (173 LOC).
- Delete `src/pscanner/corpus/onchain_targeted.py` (348 LOC).
- Delete `tests/corpus/test_onchain_backfill.py` (336 LOC).
- Delete `tests/corpus/test_onchain_targeted.py` (287 LOC).
- Edit `src/pscanner/corpus/subgraph_ingest.py`: inline `clear_truncation_flags` body (~50 LOC of SQL + log). Drop the `from pscanner.corpus.onchain_backfill import clear_truncation_flags` import. Rename the inline helper to `_clear_truncation_flags` (module-private since only `run_subgraph_backfill` uses it).
- Edit `src/pscanner/corpus/cli.py`:
  - Drop imports: `from pscanner.corpus.onchain_backfill import (clear_truncation_flags, run_onchain_backfill)` (lines 34-37), `from pscanner.corpus.onchain_targeted import run_targeted_backfill` (line 38), `from pscanner.poly.onchain_rpc import OnchainRpcClient` (line 59).
  - Drop constants (lines 64-69): `_DEFAULT_RPC_URL`, `_DEFAULT_FROM_BLOCK`, `_DEFAULT_CHUNK_SIZE`, `_DEFAULT_MAX_BLOCKS`, `_DEFAULT_TARGETED_CHUNK_SIZE`, `_DEFAULT_BLOCK_SLACK`.
  - Drop subparsers (lines 177-263): `onchain-backfill` and `onchain-backfill-targeted` definitions in `build_corpus_parser`.
  - Drop handlers (lines 766-842): `_cmd_onchain_backfill` and `_cmd_onchain_backfill_targeted`.
  - Drop `_HANDLERS` entries (lines 931-932).
  - Update module docstring (line 1-7) to drop the two command names.
- Edit `CLAUDE.md`:
  - Line 24: strip the "On-chain ingest fills the gap: `pscanner corpus onchain-backfill` …" sentence and what follows up to the next bullet. The remainder (the offset cap fact) stays.
  - Line 25: collapse the multi-sentence "**Phase 3 (subgraph) supersedes …**" bullet — drop the "stay during the transition" caveat and the cross-references to the deleted commands. Keep the schema notes and `decode_order_filled`/`AssetIndexRepo` survival note.
  - Line 117: drop the `pscanner corpus onchain-backfill / onchain-backfill-targeted` CLI surface bullet entirely.
  - Lines 120-121: drop the full per-command CLI documentation for both commands.
  - Line 175: drop the "Open follow-ups" entry for "Delete eth_getLogs corpus path" — done.

**Commit message draft:**

```
chore(corpus): delete eth_getLogs backfill path (#46 supersedes)

The subgraph backfill path (#46) has been live-validated; the eth_getLogs
corpus orchestrators (onchain_backfill.py, onchain_targeted.py) and their
CLI surface are no longer needed.

Inline clear_truncation_flags into subgraph_ingest.py (its only caller)
so it stays available for the subgraph path's post-ingest cleanup.

Keeps pscanner.poly.onchain_ingest (UnresolvableAsset, UnsupportedFill,
event_to_corpus_trade) and pscanner.poly.onchain_rpc (OnchainRpcClient) —
subgraph_ingest still imports the first three; the second is now
orphan-but-out-of-scope (platform-clients owns).

Strip CLAUDE.md references to the deleted commands and the corresponding
"Open follow-ups" entry that tracked this deletion.
```

**Test plan:**

- Existing `tests/corpus/test_subgraph_ingest.py` (617 LOC) covers the subgraph path including `clear_truncation_flags`-equivalent behaviour after the inline. Trace the test cases that touch the flag-clearing branch.
- Existing `tests/corpus/test_cli.py` covers `build_corpus_parser` for the surviving commands — confirm no test names `onchain-backfill` directly.
- No new tests required: pure deletion + mechanical relocation of one function inside a single file.

**Risk:** Low. Subgraph path is the live-validated replacement. Two concrete things to watch:

1. The inline relocation of `clear_truncation_flags` is the only behaviour-affecting edit. The function reads from `corpus_markets` + `corpus_trades` and updates two columns; verify the existing subgraph test that exercises the truncation-clearance branch (`subgraph.run_done` `truncation_flags_cleared` field) still passes post-inline.
2. `pscanner.poly.onchain_rpc.OnchainRpcClient` becomes orphaned. CLAUDE.md "Open follow-ups" mentions also removing it; that's out of my slice (platform-clients owns `poly/`). Leave it; flag the orphan to the orchestrator in the PR DONE message.

---

## T1.2b — Delete watch_subgraph_copy.py

**PR title:** `chore(scripts): delete watch_subgraph_copy.py (#152 daemon supersedes)`

**Commit shape:** Single commit.

**Files touched:**

- Delete `scripts/watch_subgraph_copy.py` (618 LOC).
- Delete `tests/scripts/test_watch_subgraph_copy.py` (127 LOC).
- Edit `CLAUDE.md`:
  - The "Tracked work in flight" section mentions: "The earlier research script `scripts/watch_subgraph_copy.py` stays during rollout for parity comparison; delete after the daemon path is live-validated for a 24h window." Strip that caveat — the window has passed. Line is part of the SubgraphTradeCollector entry.

**Commit message draft:**

```
chore(scripts): delete watch_subgraph_copy.py (#152 daemon supersedes)

The SubgraphTradeCollector daemon path on the feat/issue-152-subgraph-
trade-collector branch has been live-validated for >24h
(project_subgraph_copy_live_2026-05-21 memory note). Per CLAUDE.md's
tracked deletion item, the research script can now go.

Drop the parity-window caveat from CLAUDE.md's "Tracked work in flight"
SubgraphTradeCollector entry.
```

**Test plan:**

- The script's test file (`tests/scripts/test_watch_subgraph_copy.py`) is the only thing in `tests/scripts/`; deleting it leaves the `__init__.py` alone. Existing `tests/scripts/__init__.py` is empty (0 LOC); keep it for future scripts.
- No other test imports the script.

**Risk:** None. Pure script + test deletion. CLAUDE.md's project memory confirms the 24h window has elapsed.

---

## T1.2c — Delete parity scripts

**PR title:** `chore(scripts): delete one-shot parity scripts (covered by parity test)`

**Commit shape:** Single commit.

**Files touched:**

- Delete `scripts/parity_build_features.py` (222 LOC).
- Delete `scripts/feature_projection_byte_compare.py` (206 LOC).
- No CLAUDE.md edits needed (neither is referenced).

**Commit message draft:**

```
chore(scripts): delete one-shot parity scripts (covered by parity test)

Both scripts compared Python and DuckDB build-features engine outputs
row-by-row on a real corpus. They were manual pre-merge gates that have
served their purpose; tests/corpus/test_feature_projection_parity.py
(Hypothesis-driven) now enforces engine-vs-engine row equality
automatically.

The two scripts had different float tolerances (1e-9 vs 1e-5) and
different invocation modes (in-process vs subprocess); keeping either
risks a future operator re-running the wrong one.
```

**Test plan:**

- No tests reference either script. `tests/corpus/test_feature_projection_parity.py` remains the canonical parity gate.

**Risk:** None.

---

## T1.2d — Delete backfill_close_times.py

**PR title:** `chore(scripts): delete backfill_close_times.py (one-time fix shipped)`

**Commit shape:** Single commit.

**Files touched:**

- Delete `scripts/backfill_close_times.py` (119 LOC).

**Commit message draft:**

```
chore(scripts): delete backfill_close_times.py (one-time fix shipped)

Issue #40 was fixed in the live pipeline: CorpusMarketsRepo.mark_complete
rewrites corpus_markets.closed_at to MAX(corpus_trades.ts) on backfill
completion, and record_resolutions propagates that into
market_resolutions.resolved_at. The script has been run on the
production corpus and no test exercises it.

The SQL is preserved in git for any future recovery scenario.
```

**Test plan:** No tests reference the script.

**Risk:** None.

---

## T1.2e — Delete backfill_asset_index.py

**PR title:** `chore(scripts): delete backfill_asset_index.py (Phase 1 shipped)`

**Commit shape:** Single commit.

**Files touched:**

- Delete `scripts/backfill_asset_index.py` (61 LOC).

**Commit message draft:**

```
chore(scripts): delete backfill_asset_index.py (Phase 1 shipped)

Phase 1 of #42 has shipped; AssetIndexRepo.backfill_from_corpus_trades
remains available for ad-hoc recovery as a one-liner via uv run. The
script body is preserved in git history.
```

**Test plan:** No tests reference the script.

**Risk:** None.

---

## T1.2f — Delete 4 dead constants in features.py

**PR title:** `chore(corpus): drop dead constants in features.py`

**Commit shape:** Single commit.

**Files touched:**

- Edit `src/pscanner/corpus/features.py`: delete lines 358-363 (and the comment block at lines 358-364 that introduces them):
  - `_SECONDS_PER_DAY = 86_400`
  - `_MIN_PRICES_FOR_VOLATILITY = 2`
  - `_CONFIDENCE_N_MIN = 20`
  - `_HIGH_QUALITY_WIN_RATE_THRESHOLD = 0.55`
- Keep the live constant `_RECENT_WINDOW_SECONDS = 30 * 86_400` (used by `_trim_and_append`).

**Commit message draft:**

```
chore(corpus): drop dead constants in features.py

These 4 constants were orphaned when the feature_projection.py registry
took over the formula definitions (#145). The canonical values now live
as CONFIDENCE_N_MIN, HIGH_QUALITY_WIN_RATE_THRESHOLD, SECONDS_PER_DAY,
and MIN_PRICES_FOR_VOLATILITY in pscanner.corpus.feature_projection.

Verified unused by repo-wide rg before deletion.
```

**Test plan:** No behaviour change. Existing tests cover the constants' real homes in `feature_projection.py`. Parity test (`test_feature_projection_parity.py`) provides indirect coverage.

**Risk:** None — verified unused by `rg` before deletion.

---

## T1.7 — Extract `_iso_to_epoch` to shared module

**PR title:** `refactor(kalshi): extract _iso_to_epoch to shared module`

**Commit shape:** Single commit.

**Files touched:**

- New `src/pscanner/kalshi/shared.py` (or extend an existing in-scope module — if `pscanner.kalshi.shared` doesn't exist, create it with just this function; CLAUDE.md does not constrain this namespace).
- Edit `src/pscanner/corpus/kalshi_walker.py`: drop the local `_iso_to_epoch` definition (lines 116-131); import from `pscanner.kalshi.shared`. Drop the `datetime` import if unused.
- Edit `src/pscanner/corpus/kalshi_enumerator.py`: same — drop local definition (lines 113-128), import from `pscanner.kalshi.shared`. Drop the `datetime` import if unused.

**Decision:** the helper module home. Two viable options:

1. `pscanner.kalshi.shared` — kalshi-platform-specific, signals "only kalshi callers should reach for this".
2. `pscanner.util.time_parsing` — generic, signals "any platform that needs ISO→epoch can use it".

Going with **(1) `pscanner.kalshi.shared`**: keeps the helper next to its current callers, doesn't claim cross-platform applicability we haven't verified. If a third caller emerges, promotion is a one-line move.

**Commit message draft:**

```
refactor(kalshi): extract _iso_to_epoch to shared module

corpus/kalshi_walker.py and corpus/kalshi_enumerator.py defined the same
16-line _iso_to_epoch helper line-for-line. Move to pscanner.kalshi.shared
so a future drift can't happen.
```

**Test plan:**

- `tests/corpus/test_kalshi_walker.py` and `tests/corpus/test_kalshi_enumerator.py` cover the call sites indirectly via their public functions. Add a 5-line unit test `tests/kalshi/test_shared.py` (create if needed) covering the helper's three branches: empty string → fallback, valid ISO → epoch, malformed → fallback.
- If `tests/kalshi/` doesn't exist as a package, drop the test next to `tests/kalshi/test_client.py` or wherever `pscanner.kalshi.*` tests already live (verify path before writing).

**Risk:** None. Behaviour-identical extraction with a small unit test gate.

---

## T2.18 — `_client_ctx` helper (Corpus-ml M5)

**PR title:** `refactor(corpus): unify gamma/data client context managers`

**Commit shape:** Single commit.

**Files touched:**

- Edit `src/pscanner/corpus/cli.py`:
  - Drop `_GammaCM`, `_DataCM`, `_make_gamma_client`, `_make_data_client` (lines 336-363).
  - Add `_client_ctx(client_cls, *, rpm: int = 50)` as an `@asynccontextmanager` (~10 lines).
  - Replace 2 callers using the wrappers (`_run_polymarket_backfill`, `_run_polymarket_refresh`) with `await stack.enter_async_context(_client_ctx(GammaClient, rpm=50))` etc.
  - For `_cmd_backfill_gamma_tags` (line 875) and `_cmd_backfill_outcome_side` (line 900), replace the manual `GammaClient(rpm=args.rpm)` / try-finally pair with `async with _client_ctx(GammaClient, rpm=args.rpm) as gamma:` — same behaviour, less plumbing. `_cmd_backfill_outcome_side` also takes a DataClient; same treatment.
- Edit `tests/corpus/test_cli.py`:
  - Lines 61, 62, 90, 91: `patch("pscanner.corpus.cli._make_data_client", ...)` and `patch("pscanner.corpus.cli._make_gamma_client", ...)` no longer have targets. Repoint to `patch("pscanner.corpus.cli._client_ctx", ...)` returning a side-effect that yields the right fake client based on `client_cls`. Easiest path: keep the patches at the factory layer and have `_client_ctx` itself delegate to a tiny `_build_client(client_cls, rpm)` helper that's the patch surface. Decide during execution; the goal is to minimize test churn.

**Decision:** Make `_client_ctx` patch-friendly. The test uses two distinct fake CMs (data + gamma) and patches both wrappers. Cleanest path: `_client_ctx` becomes the only public surface but has an internal `_build_client(client_cls, rpm)` whose patches stand in for both old factory functions. The test patches one call (`_build_client`) twice with `side_effect` keyed off the `client_cls` argument.

**Commit message draft:**

```
refactor(corpus): unify gamma/data client context managers

_GammaCM and _DataCM were two near-identical async context managers with
hardcoded rpm=50, plus factory wrapper functions. _cmd_backfill_gamma_tags
and _cmd_backfill_outcome_side had to bypass them entirely with manual
try/finally because they needed --rpm to propagate.

One @asynccontextmanager _client_ctx(client_cls, *, rpm) covers all 6
callsites uniformly and lets --rpm flow through every command.

Test patches repoint from _make_*_client to _build_client (the
patch-friendly seam inside _client_ctx).
```

**Test plan:**

- `tests/corpus/test_cli.py` — repoint patches as described. Verify all existing tests pass.
- The 2 commands that previously bypassed the wrappers (`_cmd_backfill_gamma_tags`, `_cmd_backfill_outcome_side`) get the same `_client_ctx` shape; their existing tests (`tests/corpus/test_gamma_tags_backfill.py`, `tests/corpus/test_outcome_side_backfill_cli.py`) must continue to pass — they patch the client class itself, not the factory, so this should be transparent.

**Risk:** Low. The patch-friendly seam decision is the one judgement call. If `_build_client` turns out to be awkward, fall back to passing the constructor (`lambda: ClientClass(rpm=rpm)`) instead of `client_cls` and patch the lambda.

---

## T2.19 — `_temp_split_conn` context manager (Corpus-ml M1+M2)

**PR title:** `refactor(ml): collapse temp-table boilerplate in streaming.py`

**Commit shape:** Two-commit chain on the same branch for reviewability:

1. **`refactor(ml): fold materialize_test 3 connections to 1`** — pure M1.
2. **`refactor(ml): extract _temp_split_conn context manager`** — M2.

**Files touched:**

Commit 1 (M1):

- Edit `src/pscanner/ml/streaming.py`: rewrite `materialize_test()` (lines 135-212). Open one `sqlite3.connect`, populate one `_split_markets` temp table, run all 3 SELECTs (`_SplitIter`-chained for X/y/implied, then `top_category` SELECT, then `total_volume_usd` SELECT) against that connection, close once. The `_SplitIter` block is unchanged; the two trailing SELECT blocks share one connection.

Commit 2 (M2):

- Edit `src/pscanner/ml/streaming.py`: define `@contextmanager _temp_split_conn(db_path, condition_ids, *, table_name="_split_markets")` near `_populate_temp_table`. Convert 4 sites: `val_aux`, `materialize_test` (now 1 conn from commit 1), `_SplitIter.__iter__`, plus push the temp-table prep out of `_populate_temp_table` (turn the existing function into pure populate-existing-conn behaviour; the new CM owns the create+populate+close).
- Edit `scripts/analyze_model.py`: `_load_test_cat_columns` (lines 144-180) uses the same pattern; refactor to use `_temp_split_conn`. This adds a `from pscanner.ml.streaming import _temp_split_conn` import. The underscore prefix is OK because both files are in this slice; but if cross-slice cleanliness becomes a concern, lift to `pscanner.ml.streaming.temp_split_conn` (drop underscore) — the helper is small and pure. Decide during execution.

**Commit message drafts:**

```
refactor(ml): fold materialize_test 3 connections to 1

materialize_test() opened three separate sqlite3 connections and built
three copies of the _split_markets temp table from the same frozenset.
Replace with one connection and one temp table, three SELECTs against it.

Saves the duplicated populate cost (~10K INSERTs × 2 redundant builds at
production scale) plus 25 LOC.
```

```
refactor(ml): extract _temp_split_conn context manager

The DROP/CREATE/INSERT/SELECT pattern around _split_markets was repeated
across val_aux, materialize_test, _SplitIter.__iter__, and (in scripts/
analyze_model.py) _load_test_cat_columns. Lift to a @contextmanager that
owns connection lifecycle and temp-table prep.

Removes the conn = sqlite3.connect / try / finally: conn.close boilerplate
from 4 callsites; the populate logic stays in _populate_temp_table for the
existing test that asserts the table contents directly.
```

**Test plan:**

- `tests/ml/test_streaming.py` (530 LOC) covers `materialize_test`, `val_aux`, `_SplitIter` behaviour. Run as-is; pass.
- The temp-table prep behaviour (DROP, CREATE, INSERT) is opaque to existing tests (no test inspects the temp table contents directly).
- `scripts/analyze_model.py` has no test coverage; trust the existing public-shape tests for `materialize_test`.

**Risk:** None for commit 1 (pure folding). For commit 2: the CM is a pass-through context manager around the existing public `_populate_temp_table` — same SQL, same parameters, same ordering. The reach into `_temp_split_conn` from `scripts/analyze_model.py` is the only cross-module surface change.

---

## T2.20 — `_record_resolutions_loop` + `resolve_binary_outcome_map` (Corpus-ml H4+H5)

**PR title:** `refactor(corpus): share resolution writer scaffolding and binary-outcome lookup`

**Commit shape:** Two-commit chain:

1. **`refactor(corpus): extract _record_resolutions_loop for manifold + kalshi`** — H4.
2. **`refactor(corpus): share binary-outcome resolver between walker and outcome_side_backfill`** — H5.

**Files touched:**

Commit 1 (H4):

- Edit `src/pscanner/corpus/resolutions.py`:
  - Add `_record_resolutions_loop(*, targets, fetch_market, classify, source: str, platform: str, repo, now_ts: int) -> int`. `classify` is a callable `(ident, market) -> tuple[int, int] | None`. Loop body: fetch, classify, skip-or-upsert.
  - Refactor `record_manifold_resolutions` to use it: classifier returns `(1, 0)` for `"YES"`, `(0, 1)` for `"NO"`, else logs + returns `None`.
  - Refactor `record_kalshi_resolutions` similarly. The 4-way branch (disputed → skip with reason log, `"yes"`, `"no"`, `"scalar"` → skip with reason log, else → skip with reason log) all collapse into the classifier — the loop body owns the upsert + counter.
  - Leave `record_resolutions` (Polymarket) alone — different signature (`(condition_id, slug, resolved_at)` tuples), different fetch path (`gamma.get_market_by_slug` not `client.get_market`).

Commit 2 (H5):

- Add `src/pscanner/corpus/outcome_resolver.py`: `async def resolve_binary_outcome_map(condition_id, *, data, gamma) -> dict[str, tuple[str, int]] | None`. Body lifted from `outcome_side_backfill.resolve_correct_mapping` (the richer of the two return shapes). Uses a single shared log-event prefix `corpus.binary_outcome_resolver`.
- Edit `src/pscanner/corpus/market_walker.py`:
  - Drop local `_resolve_outcome_side_index` (lines 31-72).
  - Drop the local `_BINARY_MARKET_OUTCOME_COUNT = 2` constant (line 28).
  - Use `resolve_binary_outcome_map(...)` and narrow: `outcome_side_by_asset_id = {k: side for k, (side, _) in (mapping or {}).items()}`.
- Edit `src/pscanner/corpus/outcome_side_backfill.py`:
  - Drop local `resolve_correct_mapping` (lines 62-110).
  - Drop the local `_BINARY_MARKET_OUTCOME_COUNT = 2` constant (line 59).
  - Update `run_backfill` to call `resolve_binary_outcome_map(...)` directly.

**Commit message drafts:**

```
refactor(corpus): extract _record_resolutions_loop for manifold + kalshi

record_manifold_resolutions and record_kalshi_resolutions shared the
for/fetch/upsert/counter scaffolding around different outcome classifiers
(2-way for Manifold; 4-way with status check for Kalshi).

Lift the loop into _record_resolutions_loop and pass the classifier as
a callable. record_resolutions (Polymarket) stays separate — different
signature shape (with slug) and a price-threshold classifier.
```

```
refactor(corpus): share binary-outcome resolver between walker and outcome_side_backfill

market_walker._resolve_outcome_side_index and
outcome_side_backfill.resolve_correct_mapping were 90% identical lookups
("slug -> market -> binary token map"). Lift to
pscanner.corpus.outcome_resolver.resolve_binary_outcome_map and have
both callers narrow the return shape locally.

Log event prefix is now corpus.binary_outcome_resolver.* — structlog
consumers that grep on corpus.outcome_side_index.* or
corpus.backfill_outcome_side.slug_lookup_failed need to update (none
known in-tree).
```

**Test plan:**

- `tests/corpus/test_resolutions.py` (301 LOC) covers the polymarket path + manifold + kalshi. Re-run; existing assertions should pass — public function signatures are unchanged. The skip-with-log branches are tested via log-capture; verify capture_logs assertions still match (commit 1 may move log event names — minimize churn by emitting the *same* event name from inside the classifier).
- `tests/corpus/test_outcome_side_backfill.py` (445 LOC) + `tests/corpus/test_market_walker.py` (322 LOC) cover the binary-outcome resolver consumers. Existing assertions test public behaviour; the new shared helper has a tighter contract (returns `(side, index)` tuples; callers narrow). Add a small `tests/corpus/test_outcome_resolver.py` covering the 6 return paths (slug lookup fails / None, gamma lookup fails / None, market not binary, success).

**Risk:** Low. The risk is log-event-name churn on the structured-log surface; mitigated by keeping event names stable inside the classifier callbacks for H4, and accepting a one-time rename for H5 (which is documented in the commit message).

---

## T2.24 — Expose `iter_wallet_states` / `iter_market_states` on `StreamingHistoryProvider`

**PR title:** `feat(corpus): expose iter_wallet_states and iter_market_states`

**Commit shape:** Single commit.

**Files touched:**

- Edit `src/pscanner/corpus/features.py`: add two public methods on `StreamingHistoryProvider`:

  ```python
  def iter_wallet_states(self) -> Iterator[tuple[str, WalletState]]:
      for wallet_address, accum in self._wallets.items():
          yield wallet_address, accum.state

  def iter_market_states(self) -> Iterator[tuple[str, MarketState]]:
      yield from self._markets.items()
  ```

  Both are additive; no existing caller is touched.

**Commit message draft:**

```
feat(corpus): expose iter_wallet_states and iter_market_states

Strategies' cold-start bootstrap currently reaches into the streaming
provider's private _wallets / _markets / _market_traders dicts to
serialize state into wallet_state_live / market_state_live. Expose a
public iter API so the provider's storage shape can evolve without
silently breaking cold-start.

Producer side of the cross-slice cleanup (corpus-ml T2.24 + strategies
T3.24). Strategies will switch consumers to the new methods in their
own PR.
```

**Test plan:**

- Add 5-line test in `tests/corpus/test_features_streaming.py`: observe a couple of trades, then iterate; assert the iterated states equal the per-wallet/per-market lookups via the existing public `wallet_state(...)` / `market_state(...)` methods.
- Existing tests pass unchanged (purely additive).

**Risk:** Low. The API contract: `iter_wallet_states()` yields per-wallet snapshots at the moment of call (resolution-drain semantics differ from `wallet_state(..., as_of_ts=...)` which performs heap drain to that ts). Document on the docstring: "Iterates the provider's currently-resolved wallet state — does NOT drain pending resolutions. Callers that need point-in-time state for a specific ts should use wallet_state(addr, as_of_ts)."

---

## T3.22 — `_example_from_features` via `asdict` (Corpus-ml H1) [PLAN ONLY]

**PR title:** `refactor(corpus): rebuild _example_from_features via asdict`

**Commit shape:** Single commit with regression test included.

**Files touched:**

- Edit `src/pscanner/corpus/examples.py`:
  - Add module-level: `_FEATURE_ONLY_FIELDS = frozenset(f.name for f in fields(TrainingExample)) & frozenset(f.name for f in fields(FeatureRow))`.
  - Replace `_example_from_features` body with `**asdict(features)` filtered to `_FEATURE_ONLY_FIELDS`, plus the identity fields and `label_won`/`built_at`/`platform`. ~10 LOC instead of 58.
- Add `tests/corpus/test_examples_field_parity.py`: assert
  `set(FeatureRow.__dataclass_fields__) - {"market_categories"} == _FEATURE_ONLY_FIELDS` and
  `set(TrainingExample.__dataclass_fields__) - {"tx_hash", "asset_id", "wallet_address", "condition_id", "trade_ts", "built_at", "platform", "label_won"} == _FEATURE_ONLY_FIELDS`.

**Commit message draft:**

```
refactor(corpus): rebuild _example_from_features via asdict

Adding a new feature requires editing FeatureRow + TrainingExample +
the FEATURES registry + TRAINING_EXAMPLES_COLUMNS. _example_from_features
was the 5th hand-copy, and the easiest to miss — a missed field becomes
a silent NULL in the schema.

Drive the copy from dataclass introspection (asdict + field-name
intersection). Add a regression test that asserts the two dataclasses'
field sets agree, so future drift surfaces immediately.
```

**Test plan:**

- New `tests/corpus/test_examples_field_parity.py` (the regression gate). The two assertions document the contract: every TrainingExample column with a counterpart on FeatureRow is in the intersection; identity fields and `label_won` are not.
- Existing `tests/corpus/test_examples.py` (744 LOC) covers the function's behaviour end-to-end against real trade fixtures; the new internal shape is invisible to those tests.

**Risk:** Low (with the regression test). Without the test, a future field added to FeatureRow but missing from TrainingExample would silently drop. With the test, that drift fails CI immediately.

---

## T3.26 — Rehome `_build_synthetic_trades` / `_build_metadata` (X5) [PLAN ONLY]

**PR title:** `refactor(tests): rehome synthetic trade builders to shared fixture`

**Commit shape:** Single commit; cross-slice scope (touches `tests/daemon/` which is strategies territory but the rehome is in-slice-coordinated per orchestrator's brief).

**Files touched:**

- New `tests/fixtures/synthetic_trades.py`:
  ```python
  def build_synthetic_trades(seed: int, n: int, *, n_wallets: int = 6, n_markets: int = 4) -> list[Trade]: ...
  def build_metadata(trades: list[Trade], *, with_categories: bool = True) -> dict[str, MarketMetadata]: ...
  ```
  Parameterized to support both existing shapes:
  - `tests/corpus/conftest.py` shape: `n_wallets=6, n_markets=4, with_categories=True`.
  - `tests/daemon/test_live_history_parity.py` shape: `n_wallets=8, n_markets=5, with_categories=False`.
- Edit `tests/corpus/conftest.py`: drop the local `_build_synthetic_trades` (lines 34-79) and `_build_metadata` (lines 66-79); import from the new fixture module. Update `trade_stream` and `metadata_for_stream` fixtures to call the new functions.
- Edit `tests/daemon/test_live_history_parity.py`: drop the local helpers (lines 21-63); import + parameterize accordingly.
- Edit `scripts/profile_live_history.py`:
  - Drop the `sys.path.insert` hack (lines 18) and the test-module import (lines 19-22).
  - Import from the new fixture module via a clean `from tests.fixtures.synthetic_trades import ...`. **Or** (preferable) promote the helpers further to `pscanner.testing.fixtures.synthetic_trades` and import via the package — this lets the script avoid reaching into tests/ entirely. Decide during execution; the `pscanner.testing` option is cleaner long-term but adds a new package.

**Decision:** Promote to `pscanner.testing.synthetic_trades` (under `src/pscanner/testing/`). The script-reach justifies a non-test home; this matches the `pscanner.util.clock` precedent (testing helpers that ship with the package).

**Commit message draft:**

```
refactor(tests): rehome synthetic trade builders to shared fixture

_build_synthetic_trades and _build_metadata were defined three times:
in tests/corpus/conftest.py, tests/daemon/test_live_history_parity.py,
and imported via sys.path hackery from the latter by
scripts/profile_live_history.py.

Lift to pscanner.testing.synthetic_trades (a new package home that
lets the script import cleanly without reaching into tests/). Both
fixture shapes (6w/4m with categories; 8w/5m without) are covered
by optional parameters.

Cross-slice: touches tests/daemon/. Coordinated via orchestrator
brief's T3.26 + strategies' T3.23 sequencing.
```

**Test plan:**

- All three call sites must produce identical synthetic streams to the originals — parameterize defaults to match. Run `tests/corpus/test_features_*` (currently using `trade_stream` fixture) and `tests/daemon/test_live_history_parity.py` to confirm.
- `scripts/profile_live_history.py` has no test; smoke-run via `uv run python scripts/profile_live_history.py --n 100` to confirm imports resolve.

**Risk:** Low. Behaviour-preserving if defaults are set right. Cross-slice surface is the only friction.

---

## T3.28 — Manifold/Kalshi backfill+refresh handler collapse (Corpus-ml H3) [PLAN ONLY]

**PR title:** `refactor(corpus): collapse alt-platform backfill+refresh handlers`

**Commit shape:** Single commit.

**Files touched:**

- Edit `src/pscanner/corpus/cli.py`:
  - Add `_AltPlatformBinding(platform, client_factory, enumerate_fn, record_fn)` dataclass.
  - Add `_BINDINGS: Mapping[str, _AltPlatformBinding] = {...}` with manifold + kalshi entries.
  - Add `_run_alt_platform_backfill(args, binding)` and `_run_alt_platform_refresh(args, binding)` parameterized helpers.
  - Collapse `_run_manifold_backfill`, `_run_kalshi_backfill`, `_run_manifold_refresh`, `_run_kalshi_refresh` (lines 468-613) to 4 two-line dispatcher shims OR delete the 4 handler functions entirely and dispatch directly from `_cmd_backfill` / `_cmd_refresh`. Decide based on tests-patch surface.

**Commit message draft:**

```
refactor(corpus): collapse alt-platform backfill+refresh handlers

_run_manifold_backfill, _run_kalshi_backfill, _run_manifold_refresh,
_run_kalshi_refresh shared the same shape with platform-specific
client/enumerator/recorder. Parameterize via _AltPlatformBinding
table; the 4 handlers become 4 two-line shims.

Polymarket stays separate — gamma+data dual clients,
_drain_pending, _register_missing_polymarket_resolutions don't fit
the alt-platform shape.
```

**Test plan:**

- `tests/corpus/test_cli.py` (505 LOC) covers `_cmd_backfill` / `_cmd_refresh` dispatch. Existing tests should pass unchanged if dispatch surface is preserved.
- Integration tests `test_kalshi_e2e.py` (288 LOC), `test_manifold_e2e.py` (206 LOC) provide end-to-end coverage; expect to pass.

**Risk:** Low. Behaviour-preserving collapse. The judgement call is whether to keep the per-platform handler functions as 2-line shims or delete them entirely and dispatch from `_cmd_backfill` directly — depends on which has cleaner test patches.

---

## Sequencing summary

Plan-and-execute:

1. T1.2a — eth_getLogs deletion (~520 src + 620 test LOC + CLI surface)
2. T1.2b — watch_subgraph_copy.py deletion (618 + 127 LOC)
3. T1.2c — parity scripts deletion (428 LOC)
4. T1.2d — backfill_close_times.py deletion (119 LOC)
5. T1.2e — backfill_asset_index.py deletion (61 LOC)
6. T1.2f — 4 dead constants deletion (6 LOC)
7. T1.7 — `_iso_to_epoch` extraction (~16 LOC consolidated)
8. T2.18 — `_client_ctx` helper (~15 LOC)
9. T2.19 — `_temp_split_conn` (M1+M2, ~55 LOC over 2 commits)
10. T2.20 — `_record_resolutions_loop` + `resolve_binary_outcome_map` (H4+H5, ~80 LOC over 2 commits)
11. T2.24 — `iter_wallet_states` / `iter_market_states` (additive)

Plan-only:

12. T3.22 — `_example_from_features` via `asdict`
13. T3.26 — rehome synthetic trade builders
14. T3.28 — alt-platform handler collapse

Estimated total deletion: ~2,000 LOC src + ~620 LOC tests across T1.2a-f. Estimated additions: 4 new files (~150 LOC total) — `kalshi/shared.py`, `corpus/outcome_resolver.py`, `pscanner/testing/synthetic_trades.py` (T3.26), and one regression test (T3.22).

After all T1+T2 PRs land clean, send `ALL DONE: corpus-ml`.
