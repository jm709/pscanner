# pscanner — Claude Code notes

Multi-platform prediction-market data daemon. Polymarket is the primary
source (on-chain trade backfill, paper trading via the subgraph-copy
signal); Kalshi and Manifold ship Stage 1 REST/WS clients with no
signal wiring yet. Python 3.13 + uv + ruff + ty + pytest.
Subgraph-copy is the only live signal source — alerts come from
`SubgraphTradeCollector`; `PaperTrader` books them via
`SubgraphCopyEvaluator`; `PaperResolver` (the lone surviving
`PollingDetector`) closes positions on resolution.

Multi-platform architecture decisions live in
`docs/superpowers/specs/2026-05-04-multi-platform-rfc.md` (#35).

## Quick verify
`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`

## Read the docs for your task area FIRST

The detailed notes are split by topic under `docs/claude/`. They contain hard-won API quirks
and pipeline gotchas that WILL bite you — Read the matching file(s) before touching code in
that area. Open follow-ups and in-flight tracked work live in their topic file.

| Working on… | Read |
|---|---|
| Polymarket clients (gamma / data / leaderboard / subgraph), market-cache refresh, resolution detection | `docs/claude/polymarket-api.md` |
| Kalshi or Manifold (clients, ingestion, multi-platform schema/`platform` column) | `docs/claude/kalshi-manifold.md` |
| Daemon: scheduler, collectors, detectors, alerts, PaperTrader / evaluators, PaperResolver | `docs/claude/daemon.md` |
| Corpus: backfill, enumerator, `asset_index`, volume gates, build-features / DuckDB engine, corpus CLIs | `docs/claude/corpus.md` |
| ML training (`pscanner.ml`), `training_examples` features, model analysis, baselines | `docs/claude/ml.md` |
| Writing or changing ANY tests | `docs/claude/testing.md` |
| `scripts/*.py` ops & research tools, wallet/copy-trading research | `docs/claude/operator-scripts.md` |
| Wallet-cluster investigations, watchlist curation | `docs/claude/wallet-clusters.md` |

Cross-area tasks: read every matching file (e.g. a new evaluator that books off Polymarket
resolution data → `daemon.md` + `polymarket-api.md` + `testing.md`).

## Cross-cutting conventions
- **Identifiers**: 5 distinct types in `pscanner.poly.ids` — `MarketId`, `ConditionId`, `AssetId`, `EventId`, `EventSlug`. They're `NewType[str]`; `ty check` catches mis-uses. Kalshi/Manifold have parallel per-platform `ids.py` modules — never alias across platforms.
- Every long-running loop accepts `clock: Clock | None = None`.
- **Schema migrations**: idempotent `ALTER TABLE` in `_MIGRATIONS` tuple, wrapped by `_apply_migrations` swallowing `"duplicate column name"` and `"no such column"` `OperationalError`s. CREATE statements are `IF NOT EXISTS` in `_SCHEMA_STATEMENTS`. **Always open the corpus DB via `init_corpus_db()`**, never raw `sqlite3.connect()` — the latter skips the migration step and pre-existing on-disk corpora won't auto-pick-up new tables.
- **Categories**: single source of truth in `pscanner.categories.DEFAULT_TAXONOMY`. `categorize_tags(tags) -> frozenset[Category]` returns every matching category (multi-label, issue #120); `primary_category(tags) -> Category` returns the priority-ordered first match. `categorize_event` remains a single-Category wrapper. `CategorySettings.tag_exclusions` blocks matching when an exclusion tag is present (used to keep `Crypto Prices` recurring markets out of CRYPTO).
- DBs: `./data/pscanner.sqlite3` (daemon) and `./data/corpus.sqlite3` (corpus). Drop either for a clean smoke run of its respective domain.
- Smoke verification idiom: `rm -f data/pscanner.sqlite3 && timeout NNNN uv run pscanner run > /tmp/smoke.log 2>&1; echo exit=$?`

## Build orchestration (when shipping multi-issue waves)
- Use `git worktree add /home/macph/projects/pscanner-worktrees/<name> -b <branch>` for parallel sub-agents (per global standards).
- Avoid `Closes #N` in commit bodies if the issue number is uncertain — direct push to main auto-closes the wrong issue without a PR review step.
