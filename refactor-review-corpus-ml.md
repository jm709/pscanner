# Refactor review — corpus pipeline + ML training + analysis scripts

Slice: `src/pscanner/corpus/` (minus `repos.py`, `db.py`), `src/pscanner/ml/`, `src/pscanner/util/`, `scripts/`. Worktree: `/home/macph/projects/pscanner-worktrees/review-corpus-ml` on branch `refactor-review/corpus-ml` (off clean main, read-only).

Production LOC actually present: corpus ~5.9K (excluding repos/db), ml ~1.5K, util ~0.1K, scripts ~2.3K = ~9.8K. (Brief estimated ~6.5K; the gap is mostly the multi-platform walker/enumerator files added since the brief was written — they are in scope by the "entire dir EXCEPT" rule.)

Findings are ranked by impact. "Risk" calls out anything that could affect FeatureRow parity, the registry contract, or other invariants in CLAUDE.md.

---

## High impact

### H1. `_example_from_features` shim is a hand-copy of 40 FeatureRow fields → TrainingExample

**Problem.** `src/pscanner/corpus/examples.py:92-149` is a 58-line manual field copy from `FeatureRow` → `TrainingExample`. Every new feature already requires editing four places: `FeatureRow` (`features.py:282`), `TrainingExample` (`repos.py:653`), the `FEATURES` registry (`feature_projection.py:197`), and `TRAINING_EXAMPLES_COLUMNS` (`corpus/db.py`). This shim is the **fifth** place, and the one with the lowest "what could go wrong" visibility — a missed field is silently default-NULL'd in the schema.

The two dataclasses already share every non-identity field name. `FeatureRow` adds `market_categories` (the compute-only feature, `project_to_sql=False`); `TrainingExample` adds identity fields (`tx_hash`, `asset_id`, …) plus `label_won` and `built_at`.

**Before** (`examples.py:92-149`):

```python
def _example_from_features(
    *,
    trade: Trade,
    features: FeatureRow,
    label_won: int,
    now_ts: int,
    platform: str = "polymarket",
) -> TrainingExample:
    return TrainingExample(
        tx_hash=trade.tx_hash,
        asset_id=trade.asset_id,
        # ... 50 more lines of `field=features.field` ...
        cat_culture=features.cat_culture,
        label_won=label_won,
    )
```

**After:**

```python
from dataclasses import asdict, fields

_FEATURE_ONLY_FIELDS = frozenset(f.name for f in fields(TrainingExample)) & \
                       frozenset(f.name for f in fields(FeatureRow))

def _example_from_features(
    *,
    trade: Trade,
    features: FeatureRow,
    label_won: int,
    now_ts: int,
    platform: str = "polymarket",
) -> TrainingExample:
    feature_values = {k: v for k, v in asdict(features).items() if k in _FEATURE_ONLY_FIELDS}
    return TrainingExample(
        **feature_values,
        tx_hash=trade.tx_hash,
        asset_id=trade.asset_id,
        wallet_address=trade.wallet_address,
        condition_id=trade.condition_id,
        trade_ts=trade.ts,
        built_at=now_ts,
        platform=platform,
        label_won=label_won,
    )
```

**Tradeoff.** Adds a per-call `asdict` allocation (~1–2 μs on the BUY hot path); offsets the manual-copy fragility cost.

**Risk.** Behaviour-neutral: the intersection guards against `market_categories` and any FeatureRow field that doesn't exist on TrainingExample. Add a regression test asserting `_FEATURE_ONLY_FIELDS == set of FeatureRow fields minus {"market_categories"}` so a stray future divergence surfaces immediately.

---

### H2. Eth_getLogs path is ~520 LOC of dead code, plus its CLI surface

**Problem.** CLAUDE.md "Open follow-ups" already names this:

> **Delete eth_getLogs corpus path** now that #46 is live-validated. Remove `onchain_backfill.py`, `onchain_targeted.py`, `onchain_rpc.py`, `onchain_ingest.py`, and their CLI dispatch entries (`onchain-backfill`, `onchain-backfill-targeted`). Keep `pscanner.poly.onchain.decode_order_filled` and `AssetIndexRepo`.

Live in-tree files in this slice:

- `src/pscanner/corpus/onchain_backfill.py` — 173 LOC. Still imported by `cli.py:34` AND by `subgraph_ingest.py:19` (`clear_truncation_flags`).
- `src/pscanner/corpus/onchain_targeted.py` — 348 LOC. Imported only by `cli.py:38`.
- `src/pscanner/corpus/cli.py:177-263` — argparse subparsers for both commands (~85 LOC).
- `src/pscanner/corpus/cli.py:766-842` — `_cmd_onchain_backfill` and `_cmd_onchain_backfill_targeted` handlers (~75 LOC).
- `src/pscanner/corpus/cli.py:64-69` — `_DEFAULT_RPC_URL`, `_DEFAULT_FROM_BLOCK`, `_DEFAULT_CHUNK_SIZE`, `_DEFAULT_MAX_BLOCKS`, `_DEFAULT_TARGETED_CHUNK_SIZE`, `_DEFAULT_BLOCK_SLACK`.
- `src/pscanner/corpus/cli.py:59` — `OnchainRpcClient` import.
- `_HANDLERS` entries (`cli.py:931-932`).

**After:** Delete those files + the CLI plumbing in one PR. Three follow-up edits:

1. `subgraph_ingest.py:19` currently does `from pscanner.corpus.onchain_backfill import clear_truncation_flags`. The function (`onchain_backfill.py:126-173`) is ~50 LOC of SQL + logging that doesn't actually depend on RPC. Either inline into `subgraph_ingest.run_subgraph_backfill` (it's only called once, on line 460) or relocate to `pscanner.corpus.repos` next to the `truncated_at_offset_cap` column owner.
2. `tests/corpus/test_onchain_backfill.py` (336 LOC) and `tests/corpus/test_onchain_targeted.py` (287 LOC) delete with the modules.
3. CLAUDE.md's "Polymarket API quirks" section still names these commands as examples — strip those once the code is gone.

**Tradeoff.** Per CLAUDE.md the subgraph path is live-validated. The eth_getLogs path is operator-runnable but no longer recommended; pre-cutoff backfill is still gated on it (CLAUDE.md mentions the "pre-2026-04-28" subgraph has a different schema). The pre-cutoff use case appears to be one-shot and not regression-tested in CI; if a future operator needs it they can `git checkout` an old SHA.

**Risk.** Low — pure deletion. The follow-up imports are the only operational footgun; both are easy to fix.

---

### H3. Manifold / Kalshi backfill + refresh handlers are 4-way near-duplicates

**Problem.** `src/pscanner/corpus/cli.py:468-613` defines four async handlers (`_run_manifold_backfill`, `_run_kalshi_backfill`, `_run_manifold_refresh`, `_run_kalshi_refresh`) with near-identical shape. The two refresh handlers especially: open conn, enter client, enumerate, SELECT condition_id + closed_at WHERE platform AND backfill_state='complete', `missing_for`, build targets list comprehension, call `record_*_resolutions`. Lines 560-577 and 593-610 are line-for-line equivalent modulo client class / table prefix / record_fn.

**Before** (cli.py:550-613, two ~30-line handlers; only `client` ctor, the platform string, the enumerator, and the record_fn differ).

**After:** One handler factory parameterised by platform binding:

```python
@dataclass(frozen=True)
class _AltPlatformBinding:
    platform: Literal["manifold", "kalshi"]
    client_factory: Callable[[], AbstractAsyncContextManager[ManifoldClient | KalshiClient]]
    enumerate_fn: Callable  # enumerator (client, repo, *, now_ts) -> int
    record_fn: Callable     # record_*_resolutions

_BINDINGS = {"manifold": _AltPlatformBinding(...), "kalshi": _AltPlatformBinding(...)}

async def _run_alt_platform_refresh(args, binding: _AltPlatformBinding) -> int:
    db_path = Path(args.db)
    conn = init_corpus_db(db_path)
    markets_repo = CorpusMarketsRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    now_ts = int(time.time())
    try:
        async with binding.client_factory() as client:
            await binding.enumerate_fn(client, markets_repo, now_ts=now_ts)
            rows = conn.execute(
                "SELECT condition_id, closed_at FROM corpus_markets "
                "WHERE platform = ? AND backfill_state = 'complete'",
                (binding.platform,),
            ).fetchall()
            condition_ids = [r["condition_id"] for r in rows]
            missing = set(resolutions_repo.missing_for(condition_ids, platform=binding.platform))
            targets = [(r["condition_id"], int(r["closed_at"])) for r in rows if r["condition_id"] in missing]
            await binding.record_fn(client=client, repo=resolutions_repo, targets=targets, now_ts=now_ts)
    finally:
        conn.close()
    return 0
```

`_run_manifold_refresh`/`_run_kalshi_refresh` collapse to two-line `return await _run_alt_platform_refresh(args, _BINDINGS["manifold"])` shims. Same shape for the backfill pair.

**Tradeoff.** The Polymarket path stays separate — it has gamma+data dual clients, `_drain_pending`, and `_register_missing_polymarket_resolutions`, which don't fit the alt-platform shape. So this only collapses 4 handlers to 2 + a parameter table, not all 6.

**Risk.** Low. The binding table is closed (only manifold/kalshi land in `--platform` choices); a future platform addition gets a single new entry instead of two new handlers.

---

### H4. `record_manifold_resolutions` and `record_kalshi_resolutions` share scaffolding around different outcome classifiers

**Problem.** `src/pscanner/corpus/resolutions.py:91-142` and `145-213` follow the same pattern: for each target, fetch market, classify the outcome string into `(outcome_yes_won, winning_outcome_index)` or skip-with-log, upsert `MarketResolution(...)`, counter increment. Only the outcome-classification branch differs:

- Manifold (lines 117-128): `if resolution == "YES" → (1,0); elif "NO" → (0,1); else → skip`.
- Kalshi (lines 173-200): 4-way branch (`disputed` → skip; `result=="yes"` → (1,0); `result=="no"` → (0,1); `"scalar"` → skip; else → skip with status log).

**After:** Extract a `_record_resolutions_loop` helper that takes a classifier:

```python
async def _record_resolutions_loop(
    *, targets, fetch_market, classify, source: str, platform: str, repo, now_ts: int
) -> int:
    written = 0
    for ident, resolved_at in targets:
        market = await fetch_market(ident)
        outcome = classify(ident, market)  # returns (yes_won, win_idx) or None
        if outcome is None:
            continue
        yes_won, win_idx = outcome
        repo.upsert(MarketResolution(
            condition_id=ident, winning_outcome_index=win_idx, outcome_yes_won=yes_won,
            resolved_at=resolved_at, source=source, platform=platform,
        ), recorded_at=now_ts)
        written += 1
    return written
```

The two public functions shrink to building their classifier closure + calling the loop. `record_resolutions` (Polymarket) doesn't fit — it threads `slug` separately and uses a price-threshold classifier — keep it as is.

**Tradeoff.** Two callers, not three; it's the borderline case. The win is that the skip-with-log branches are visibly tied to one place, so a future operator adding "Kalshi voided market" handling can't accidentally diverge the upsert payload.

**Risk.** Low. Each classifier handles its own logging; upsert payload is type-checked.

---

### H5. `_resolve_outcome_side_index` and `resolve_correct_mapping` duplicate the "slug → market → binary token map" lookup

**Problem.** `src/pscanner/corpus/market_walker.py:31-72` and `src/pscanner/corpus/outcome_side_backfill.py:62-110` are ~90% identical. Both:

1. Try `data.get_market_slug_by_condition_id(condition_id)`; catch any Exception → log warning, return empty/None.
2. If slug is None → return empty/None.
3. Try `gamma.get_market_by_slug(slug)`; catch any Exception → log warning, return empty/None.
4. If market is None → return empty/None.
5. Check `len(market.clob_token_ids) == 2`.
6. Return `{token_id_0: ("YES", 0), token_id_1: ("NO", 1)}`.

Differences:

- `_resolve_outcome_side_index` returns `dict[str, str]` (just side, no index).
- `resolve_correct_mapping` returns `dict[str, tuple[str, int]]` (side + index).
- Different `_log.warning` event names.

Both functions also live next to a `_BINARY_MARKET_OUTCOME_COUNT = 2` constant declared once per file (`market_walker.py:28`, `outcome_side_backfill.py:59`).

**After:** A shared `pscanner.poly.binary_outcome_resolver` (or, less ambitious, a free function inside `pscanner.poly.gamma`):

```python
async def resolve_binary_outcome_map(
    condition_id: str, *, data: DataClient, gamma: GammaClient,
) -> dict[str, tuple[str, int]] | None:
    # ... the shared body ...
```

Both call sites then do their narrowing: `market_walker` reads `{k: side for k, (side, _) in mapping.items()}`, `outcome_side_backfill` uses it directly.

**Tradeoff.** The shared module brings two corpus modules to a common helper — the kind of consolidation `pscanner.poly` is naturally the right home for. The slight risk is forcing every future callsite onto the tuple shape (instead of the smaller dict-of-str shape), but that's the more informative return so it's fine.

**Risk.** Low. Both functions return-on-error to `None`/`{}`; the unified version returns `None`. The error-log event name churns; structured log consumers that pattern-match on `corpus.outcome_side_index.*` would need to update (the daemon's structlog doesn't currently grep on these). Worth a comment in the migration commit.

---

## Medium impact

### M1. `materialize_test` opens two SQLite connections + temp tables for the same split

**Problem.** `src/pscanner/ml/streaming.py:135-212` materializes the test split. It runs the streaming `_SplitIter` (one connection) for `(x, y, implied)`, then opens a **second** connection (line 177-183) to fetch unencoded `top_category`, then a **third** connection (line 198-204) to fetch `total_volume_usd`. Each of the parallel-select branches builds + populates `_split_markets` from the same `_test_markets` frozenset.

**After:** One connection, one temp table, both parallel SELECTs:

```python
conn = sqlite3.connect(str(self._db_path))
try:
    _populate_temp_table(conn, "_split_markets", self._test_markets)
    top_rows = conn.execute(top_sql, (self._platform,)).fetchall()
    volume_rows = conn.execute(volume_sql, (self._platform,)).fetchall()
finally:
    conn.close()
```

Saves ~25 LOC and the cost of building the temp table twice (~10K INSERTs each pass per CLAUDE.md's `_populate_temp_table` docstring).

**Tradeoff.** None notable.

**Risk.** Order-of-results: both SELECTs already `ORDER BY te.id`, so they'll line up with `materialize_test`'s `(x, y, implied)` arrays (per their existing comment-of-trust). No behaviour change.

---

### M2. Temp-table pattern repeated 4× across `streaming.py`

**Problem.** The "DROP IF EXISTS / CREATE TEMP TABLE / executemany INSERT / SELECT JOIN / close conn" pattern shows up in `_populate_temp_table` itself (defined once) plus its callers in `val_aux`, `materialize_test` (×2 after M1 lands), `_SplitIter.__iter__`, and a fifth time in `scripts/analyze_model.py:_load_test_cat_columns`. Each caller is `conn = sqlite3.connect(...) ; try: _populate ; cursor = conn.execute ; finally: conn.close()`.

**After:** Either a small context manager:

```python
@contextmanager
def _temp_split_conn(db_path, condition_ids, table_name="_split_markets"):
    conn = sqlite3.connect(str(db_path))
    try:
        _populate_temp_table(conn, table_name, condition_ids)
        yield conn
    finally:
        conn.close()
```

Or — since `_SplitIter` already takes `condition_ids` + `db_path` and could expose a `__call__`-style chunked fetcher — fold both `val_aux` and `materialize_test`'s helper SELECTs into `_SplitIter` as alternative output adapters (`val_aux_iter()`, `_id_columns_iter()`).

The lighter context-manager version is the safer move; the second is invasive.

**Tradeoff.** Saves ~10 LOC × 4 callsites and removes the chance of a future call site forgetting `conn.close()`.

**Risk.** None.

---

### M3. `per_decile_edge_breakdown` and `per_volume_bucket_edge_breakdown` differ only in bucket definitions

**Problem.** `src/pscanner/ml/metrics.py:44-81` and `93-132` have line-equivalent bodies:

```python
take = y_pred_proba > implied_prob
out = {}
for label, lo, hi in BUCKETS:           # decile: derived from `range(10)`. volume: literal tuple.
    in_bucket = (implied_prob >= lo) & (implied_prob < hi)  # decile: special-cased last bucket. volume: special-cased inf.
    mask = take & in_bucket
    n = int(mask.sum())
    if n == 0: continue
    out[label] = {"n": float(n), "mean_edge": float((y_true[mask] - implied_prob[mask]).mean())}
return out
```

**After:** Extract:

```python
def _edge_breakdown_by_buckets(
    y_true, y_pred_proba, implied_prob, *, buckets: Iterable[tuple[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    take = y_pred_proba > implied_prob
    out: dict[str, dict[str, float]] = {}
    for label, in_bucket in buckets:
        mask = take & in_bucket
        n = int(mask.sum())
        if n == 0: continue
        out[label] = {"n": float(n), "mean_edge": float((y_true[mask] - implied_prob[mask]).mean())}
    return out
```

Each public function precomputes its `(label, mask)` pairs from its own bucket definition, then defers. ~30 LOC of body → ~10 LOC of setup per function.

**Tradeoff.** A third "edge breakdown" surface (per-category-any in `scripts/analyze_model.py:_print_per_category_any_breakdown`) would also use this helper — see L5.

**Risk.** None — pure refactor of two functions that already have parity tests in `tests/ml/test_metrics.py`.

---

### M4. Four dead constants in `features.py`

**Problem.** `src/pscanner/corpus/features.py:358-363`:

```python
_SECONDS_PER_DAY = 86_400
_MIN_PRICES_FOR_VOLATILITY = 2
_CONFIDENCE_N_MIN = 20
_HIGH_QUALITY_WIN_RATE_THRESHOLD = 0.55
```

Verified with `rg "_SECONDS_PER_DAY|_MIN_PRICES_FOR_VOLATILITY|_CONFIDENCE_N_MIN|_HIGH_QUALITY_WIN_RATE_THRESHOLD" src tests scripts`: only definition matches. No callers anywhere in the repo. They were used before the `feature_projection.py` registry took over the formulas; the registry re-defines them as `CONFIDENCE_N_MIN`, `HIGH_QUALITY_WIN_RATE_THRESHOLD`, `SECONDS_PER_DAY`, `MIN_PRICES_FOR_VOLATILITY` (`feature_projection.py:27-31`).

**After:** Delete the four lines. The `_RECENT_WINDOW_SECONDS = 30 * 86_400` directly above is still live (used by `_trim_and_append`) — keep it.

**Risk.** None — confirmed unused.

---

### M5. `_GammaCM` / `_DataCM` are two near-identical 7-line async context managers plus factory wrappers

**Problem.** `src/pscanner/corpus/cli.py:336-363`. Both classes do `__aenter__`: build client with hardcoded `rpm=50`; `__aexit__`: `await client.aclose()`. The factory functions `_make_gamma_client()` / `_make_data_client()` exist solely to construct one. Total: 28 LOC.

Also worth noting: `_cmd_backfill_gamma_tags` (cli.py:881) and `_cmd_backfill_outcome_side` (cli.py:908) **don't** use these — they instantiate `GammaClient(rpm=args.rpm)` directly with a try/finally, because the wrapper hardcodes rpm=50 and operators need `--rpm` to propagate. So the wrappers are usable in exactly 2/6 commands.

**After:**

```python
@asynccontextmanager
async def _client_ctx(client_cls, *, rpm: int = 50):
    client = client_cls(rpm=rpm)
    try:
        yield client
    finally:
        await client.aclose()
```

And every callsite — including the `--rpm`-respecting ones — collapses to `async with _client_ctx(GammaClient, rpm=args.rpm) as gamma: ...`. The factory wrappers go away; the `AsyncExitStack` plumbing in `_run_polymarket_backfill`/`_run_polymarket_refresh` stays.

**Tradeoff.** None — pure code reduction.

**Risk.** None.

---

### M6. `_iso_to_epoch` is line-for-line duplicated in two files

**Problem.** `src/pscanner/corpus/kalshi_walker.py:116-131` and `src/pscanner/corpus/kalshi_enumerator.py:113-128` define the **identical** 16-line `_iso_to_epoch(iso, *, fallback)` helper.

**After:** Move to `pscanner.kalshi.shared` or `pscanner.util.time_parsing` (a `pscanner.util.kalshi_time` is the smaller surface). Both call sites import the shared function.

**Tradeoff.** A two-callsite shared helper is the borderline case, but these are line-identical (down to the comment) and the parsing is non-trivial enough (TZ handling, fromisoformat compatibility, fallback semantics) that drift would be a real footgun.

**Risk.** None.

---

### M7. `_build_synthetic_trades` + `_build_metadata` are TRIPLICATED across test/script files

**Problem.** Same function name, same shape, near-identical body, three locations:

1. `tests/corpus/conftest.py:34-79` — 6 wallets / 4 markets, 70/30 BUY/SELL.
2. `tests/daemon/test_live_history_parity.py:21-63` — 8 wallets / 5 markets, 70/30 BUY/SELL.
3. `scripts/profile_live_history.py:19-22` — imports #2 via `sys.path.insert + from tests.daemon.test_live_history_parity import _build_synthetic_trades, _build_metadata`.

The script importing private (underscore-prefixed) helpers from a test module via path manipulation is itself a structural smell — see L4.

The brief notes that `daemon/live_history.py` is out of this slice. The duplication crosses the scope boundary: my-scope `tests/corpus/conftest.py` and `scripts/profile_live_history.py` both reproduce/depend on the test helper from the daemon slice. I'm flagging it so the orchestrator can wire the fix into whichever slice owns it.

**After:** Single source-of-truth helper at `tests/fixtures/synthetic_trades.py` (or `pscanner.testing.fixtures` if we want script reuse). Parameterize wallet/market counts so both seed shapes round-trip:

```python
def build_synthetic_trades(seed: int, n: int, *, n_wallets: int = 6, n_markets: int = 4) -> list[Trade]: ...
def build_metadata(trades: list[Trade]) -> dict[str, MarketMetadata]: ...
```

Both test files import from the fixture module; `profile_live_history.py` imports from there too, no `sys.path` hack.

**Tradeoff.** Cross-slice change; needs coordination with whoever owns `tests/daemon/`. If that's costly, the minimum-impact partial fix is just rehoming inside this slice (consolidate `conftest.py`'s copies + have the script import from `conftest.py`, since `conftest.py` is already importable from tests).

**Risk.** Behaviour-neutral if the parameter defaults preserve both shapes. The `tests/daemon/test_live_history_parity.py` build also lacks the `categories=(t.category,)` field that `tests/corpus/conftest.py:77` sets — that's a real divergence; check whether the daemon test ever exercises `meta.categories` (likely not, since it predates #122 multi-label).

---

## Low impact / cleanup

### L1. Parity scripts are two different scripts doing the same job

**Problem.** `scripts/parity_build_features.py` (222 LOC) and `scripts/feature_projection_byte_compare.py` (206 LOC) both diff Python-vs-DuckDB engine outputs row-by-row on a real corpus. Differences:

- `parity_build_features.py` runs both engines IN-PROCESS via direct imports.
- `feature_projection_byte_compare.py` runs both via `subprocess` invoking the CLI.
- Float tolerances differ: `parity_build_features.py` uses `rel=1e-9 / abs=1e-12`; `feature_projection_byte_compare.py` uses `rel=1e-5 / abs=1e-7`.
- Different output formatting / mismatch logging.

CLAUDE.md confirms `tests/corpus/test_feature_projection_parity.py` runs Hypothesis-driven engine-vs-engine row equality automatically, so the byte-compare gate is structurally already covered.

**After.** Delete both. If a one-shot pre-merge byte gate is still wanted, keep `parity_build_features.py` (in-process is faster and avoids subprocess flakiness) and drop the loose-tolerance subprocess variant.

**Tradeoff.** Removing both removes the manual operator-visible gate evidence step from PR descriptions. If that's load-bearing for some reviewer's workflow, keep one.

**Risk.** None — these scripts are dev-only.

---

### L2. `scripts/backfill_close_times.py` is a one-time fix that has shipped

**Problem.** Top-of-file docstring (`scripts/backfill_close_times.py:1`):

> "One-time fix for issue #40: rewrite `closed_at`/`resolved_at` from observed trades."

CLAUDE.md confirms the fix is now in the live pipeline:

> "`CorpusMarketsRepo.mark_complete` rewrites `corpus_markets.closed_at` to `MAX(corpus_trades.ts)` when backfill finishes; `record_resolutions` then propagates that into `market_resolutions.resolved_at`."

The script is idempotent so re-running is safe, but it has done its job on the production corpus and no test exercises it.

**After.** Delete. If the operator needs the SQL one more time it's preserved in git.

**Risk.** None.

---

### L3. `scripts/backfill_asset_index.py` is also one-shot but marginal

**Problem.** Top docstring: "One-shot backfill of `asset_index` from existing `corpus_trades` data. Phase 1 of #42 on-chain backfill."

Phase 1 has shipped (`AssetIndexRepo.backfill_from_corpus_trades` exists; phases 2+ are in production via the subgraph path). The script is small (61 LOC) and might still be useful as a manual recovery tool after a corpus reset. Lower priority than L2.

**After.** Either delete (favoured — operators can re-call `AssetIndexRepo.backfill_from_corpus_trades` via a one-liner) or move under `scripts/operator/` to signal "occasional human-driven recovery".

**Risk.** None.

---

### L4. `scripts/profile_live_history.py` imports private test helpers via path manipulation

**Problem.** `scripts/profile_live_history.py:18-22`:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.daemon.test_live_history_parity import (
    _build_metadata,
    _build_synthetic_trades,
)  # type: ignore[import-not-found]
```

This script:

- Inserts the repo root into `sys.path` at runtime.
- Imports two underscore-prefixed (private-by-convention) functions from a `tests/` module.
- Requires a `type: ignore` because tests/ isn't a real package on the import path.

Hard to use, hard to type-check, and fragile to test refactors.

**After.** Resolve via M7 — once the helpers live in a shared fixtures module, this script imports them cleanly. As an interim, copy the two helpers into the script body (~30 LOC duplicated, but no test-module reach).

**Risk.** None.

---

### L5. `scripts/analyze_model.py` reaches into `StreamingDataset._test_markets` and duplicates the temp-table pattern

**Problem.** `scripts/analyze_model.py:247`: `test_markets_set = frozenset(ds._test_markets)`. Reaches across an underscore-prefixed boundary; the script needs the test split's condition_ids for the parallel `_load_test_cat_columns` SELECT.

`_load_test_cat_columns` (`scripts/analyze_model.py:144-180`) is the **fifth** copy of the temp-table-and-JOIN pattern (after `_populate_temp_table` + `val_aux` + `materialize_test`'s two SELECTs + `_SplitIter`). With M1+M2 it'd be a sixth.

Two adjacent functions `_print_per_category_breakdown` (lines 111-141) and `_print_per_category_any_breakdown` (lines 183-223) are also near-identical loops differing only in mask construction (`cat_array == cat` vs `cat_columns_test[cat_col].astype(bool)`).

**After.**

1. Add `StreamingDataset.test_markets` as a public property returning `self._test_markets`.
2. Push `_load_test_cat_columns`'s SQL into `StreamingDataset` (e.g., a `materialize_test_with_cat_columns()` method that bundles X, y, implied, cat_*, top_category, total_volume into a richer `TestSplit`). Or expose a thin helper that takes the SQL projection and returns a numpy array; the script and `materialize_test` then both use it.
3. Unify the two per-category breakdown functions either by passing in the mask iterator (paralleling M3's `_edge_breakdown_by_buckets` extraction) or by collapsing into one function that takes a `Mapping[label, mask]` argument.

**Risk.** None for #1 (additive). #2 grows the `StreamingDataset` surface; ok because the script is the one downstream consumer that justifies it. #3 has parity tests under `tests/ml/test_metrics.py` if you reuse M3's helper.

---

### L6. `market_walker.py` calls `data._fetch_market_trades_page` — leaky private method

**Problem.** `src/pscanner/corpus/market_walker.py:212`: `page = await data._fetch_market_trades_page(condition_id, offset=offset)`. The `_`-prefixed method on `DataClient` indicates module-private intent; `market_walker.py` is the sole external caller.

**After.** Either rename to `data.fetch_market_trades_page(...)` and make it part of the `DataClient` public surface, or extract the pagination loop into `DataClient` itself (e.g., `async for page in data.iter_market_trades(condition_id, start_offset=...)`).

The pagination loop in `_fetch_all_pages` (lines 193-234) is already a small adapter shell on top of the underscore method; pushing it into `DataClient` would let the walker focus on parse + insert + repo bookkeeping. Reasonable to ship as a follow-up rather than bundled with this review.

**Risk.** Low. `DataClient` lives in `pscanner.poly` (out of this slice strictly), so the public-rename is the cheapest path inside scope.

---

### L7. `_create_v2_via_sqlite3` and `_atomic_swap` repeat the (condition, wallet, label) index tuple

**Problem.** `src/pscanner/corpus/_duckdb_engine.py:305-310` and `883-901` both list out the same three index suffixes (`condition`, `wallet`, `label`) with hand-written SQL. The `_V2_INDEX_PREFIX` constant exists (line 36) and is referenced from both, but the *suffix list* is repeated.

**After.** Add `_INDEX_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (("condition", "condition_id"), ("wallet", "wallet_address"), ("label", "label_won"))` and loop in both functions. Saves ~10 LOC, removes a future-maintenance drift surface when a new index is added.

**Risk.** None — pure mechanical refactor.

---

### L8. `_run_stage`'s `poll_table` dict is awkwardly positioned inside the helper

**Problem.** `src/pscanner/corpus/_duckdb_engine.py:207-219`: `_run_stage` looks up which DuckDB table to count via a dict-literal indexed by `name`. The dict is rebuilt on every call (~6 stages = 6 negligible allocations), but the bigger issue is that the helper is now coupled to its callers' stage names — adding a new stage means editing two places (the call site at lines 136-176 AND the poll_table dict). The lambdas already encapsulate the stage function calls; the poll-table mapping could ride alongside.

**After.** Either pass `poll_table` as a parameter (`_run_stage(scratch, name=..., fn=..., poll_table="trades")`), or make the stage list at lines 136-176 data-driven so name + fn + poll_table travel together:

```python
_STAGES: tuple[tuple[str, str | None, Callable[..., None]], ...] = (
    ("materialize_trades", "trades", _materialize_trades),
    ("stage1_events", "events", _stage1_events),
    # ...
)
```

`build_features_duckdb` loops `_STAGES`. The currying for `platform`/`now_ts`/`scratch` arguments needs handling — partials or a closure capture — but the call site shrinks substantially.

**Tradeoff.** The data-driven version is more rigid (everything must have the same signature). Parameter-passing in `_run_stage` is the safer incremental fix.

**Risk.** None — internal.

---

### L9. `scripts/watch_subgraph_copy.py` is explicitly slated for deletion

**Problem.** CLAUDE.md "Tracked work in flight":

> "The earlier research script `scripts/watch_subgraph_copy.py` stays during rollout for parity comparison; delete after the daemon path is live-validated for a 24h window."

The 24h window has passed (the project memory note `project_subgraph_copy_live_2026-05-21.md` predates today, 2026-05-24). Verify with the orchestrator whether the live-validation window is satisfied; if yes, the 618-LOC script can be removed.

**Risk.** None — it's a research artefact. Deletion is reversible via git.

---

## Notes on what I did NOT recommend

- **`feature_projection.py`'s registry pattern is correct.** CLAUDE.md says "Don't propose splitting this; do flag if more could fold in." I considered whether the `_example_from_features` shim (H1) should fold into the registry — it shouldn't, because TrainingExample is the SQLite-row dataclass, the registry should not know about the database. H1's `**asdict` approach keeps the registry the SSOT for formulas while letting the shim disappear into native dataclass introspection.
- **The 6-stage DuckDB pipeline is well-decomposed.** The per-stage SQL is genuinely different and each stage's invariants are documented; abstracting further would obscure the SQL rather than clarify it. L7+L8 are the only meaningful cleanup opportunities.
- **Per-platform walker/enumerator pairs (Manifold, Kalshi)** look duplicative on first read but each has different pagination + filtering quirks (Manifold cursor on `id`, Kalshi cursor on status walk, Polymarket offset + truncation handling). 2 occurrences of similar-but-not-identical shape; the right abstraction will emerge if a fourth platform lands.
- **`_partition_markets`, `_fit_encoder_on_train`, `_count_split_rows`, `_kept_columns`** in `streaming.py` look like they could be lambdas-with-a-conn but they're each doing a discrete, named, testable pre-pass step. Leaving them.
- **`pscanner.util.clock`** is clean; no findings.

---

## Summary table

| ID | Where | Estimated saving | Risk |
|----|-------|-----------------|------|
| H1 | `examples.py:92-149` shim | ~50 LOC + fragility | Low (add regression test) |
| H2 | `onchain_*.py` + cli surface | ~520 LOC + CLI cruft | Low (pure deletion) |
| H3 | `cli.py:468-613` four handlers | ~60 LOC + future platforms | Low |
| H4 | `resolutions.py:91-213` two writers | ~40 LOC | Low |
| H5 | `market_walker.py:31-72` + `outcome_side_backfill.py:62-110` | ~40 LOC | Low |
| M1 | `streaming.py:135-212` materialize_test | ~25 LOC | None |
| M2 | temp-table pattern × 4-5 callers | ~30 LOC | None |
| M3 | `metrics.py` two breakdowns | ~30 LOC | None |
| M4 | `features.py:358-363` dead constants | 6 LOC | None |
| M5 | `cli.py:336-363` _GammaCM/_DataCM | ~15 LOC | None |
| M6 | `_iso_to_epoch` × 2 | 16 LOC × 1 copy | None |
| M7 | `_build_synthetic_trades` × 3 | ~50 LOC across files | Low (cross-slice) |
| L1 | parity scripts × 2 | 428 LOC of one-shots | None |
| L2 | `backfill_close_times.py` | 119 LOC | None |
| L3 | `backfill_asset_index.py` | 61 LOC | None |
| L4 | `profile_live_history.py` test reach | structural | None |
| L5 | `analyze_model.py` ds._test_markets + ×2 funcs | ~30 LOC | None |
| L6 | `market_walker.py:212` private method | structural | Low |
| L7 | `_duckdb_engine.py` index tuple | ~10 LOC | None |
| L8 | `_duckdb_engine.py` poll_table dict | structural | None |
| L9 | `watch_subgraph_copy.py` | 618 LOC | None (verify rollout window) |

**Most leveraged:** H2 (pure-deletion 520 LOC + CLI cruft) and H1 (fix the fragility hotspot for future feature additions) together pay back the most maintenance burden for the least risk. The cluster H3+H4+H5+M5 around the platform-handler shell are a natural single-PR follow-up that flattens the cli.py surface considerably.
