# Issue #110 — `build-features` Perf (deque + Read Connection) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `pscanner corpus build-features --rebuild` wall time from ~9-10h to ~1-2h on the 2026-05-08 corpus by eliminating the two O(N²) hot paths in `WalletState` updates and giving the chronological cursor a dedicated read connection.

**Architecture:** Three small, pure-in-memory changes plus one wiring change. (1) `WalletState.recent_30d_trades` becomes a `deque[int]` mutated in-place inside `apply_buy_to_state` / `apply_sell_to_state`; the per-trade tuple rebuild becomes O(1) amortized via `popleft` + `append`. (2) `category_counts` is mutated in place inside `apply_buy_to_state` instead of being copied via `dict(...)`. (3) `build_features` opens a second SQLite connection in read-only mode for `iter_chronological` so writes (`INSERT OR IGNORE`) don't contend with the streaming read cursor's WAL snapshot. (4) No DB schema change — `recent_30d_trades_json` already round-trips `[ts1, ts2, ...]` for both tuple and deque via `list(...)` serialization.

**Tech Stack:** Python 3.13, sqlite3 (WAL), `collections.deque`. No new deps.

---

## Scope Check

The issue covers three composable in-memory fixes plus a wiring change. They land together (per the issue's explicit "All three changes land together" acceptance criterion) but the plan splits them into 4 tasks for review granularity. Each task leaves the test suite green.

---

## File Structure

- Modify: `src/pscanner/corpus/features.py:64-91` — `WalletState` dataclass field type.
- Modify: `src/pscanner/corpus/features.py:123-140` — `empty_wallet_state` factory.
- Modify: `src/pscanner/corpus/features.py:154-201` — drop `_trim_recent_trades`, inline trim into `apply_buy_to_state` + `apply_sell_to_state`, mutate-in-place.
- Modify: `src/pscanner/daemon/live_history.py:120-131` — deserialize `recent_30d_trades` as `deque(...)` not `tuple(...)`.
- Modify: `tests/corpus/test_features_state.py:47, 77` — assert against deque, not tuple.
- Modify: `tests/corpus/test_features_compute.py:102` — construct `WalletState(recent_30d_trades=deque([900_000]))`.
- Modify: `tests/corpus/test_features_interaction.py:100-111` — construct with deque.
- Modify: `tests/daemon/test_live_history.py:81` — assert against deque.
- Modify: `src/pscanner/corpus/examples.py:167-246` — pass a separate read connection to `CorpusTradesRepo`.
- Modify: `src/pscanner/corpus/cli.py:426-442` — open the read connection alongside the existing write connection.
- Modify: `CLAUDE.md` — clear the open-follow-ups bullet for #58/#110 once the perf work lands.

Tasks 1, 2, 3, 4 are sequential because each later task may exercise APIs touched earlier. The parity test (`tests/daemon/test_live_history_parity.py`) is the safety net — it runs unchanged at every step and must stay green.

---

## Task 1: Convert `recent_30d_trades` to `deque` with in-place trim

**Files:**
- Modify: `src/pscanner/corpus/features.py:64-91, 123-140, 154-201`
- Modify: `src/pscanner/daemon/live_history.py:120-131`
- Modify: `tests/corpus/test_features_state.py:47, 77`
- Modify: `tests/corpus/test_features_compute.py:102`
- Modify: `tests/corpus/test_features_interaction.py:100-111`
- Modify: `tests/daemon/test_live_history.py:81`

The dataclass stays `frozen=True` — frozen blocks reassignment of the field reference, but the `deque` object itself is mutable, just like the existing `dict[str, int]` `category_counts` field already is. `apply_buy_to_state` / `apply_sell_to_state` call `popleft` + `append` on the existing deque rather than constructing a new one. The function still returns the (same) `WalletState` instance via `replace(...)` so callers that hold the post-update reference see the mutation.

**Why we can mutate in place:** every caller in the codebase (`StreamingHistoryProvider.observe`, `LiveHistoryProvider.observe`, `bootstrap_wallet`) immediately overwrites its prior `state` reference with the function's return value. No caller holds a snapshot of the pre-update state expecting it to stay frozen. The parity test compares `compute_features` output (a fresh `FeatureRow`), not `WalletState` snapshots, so internal representation changes are invisible to the parity contract.

- [ ] **Step 1: Confirm baseline tests pass on the worktree**

Run: `uv run pytest tests/corpus/test_features_state.py tests/corpus/test_features_compute.py tests/corpus/test_features_interaction.py tests/daemon/test_live_history.py tests/daemon/test_live_history_parity.py -v`

Expected: all pass on origin/main (pre-change baseline).

- [ ] **Step 2: Update the `WalletState` field type**

Edit `src/pscanner/corpus/features.py`. Add to the existing imports near the top (look for the `from dataclasses import ...` line — if `field` is already imported, the existing import block may be `from dataclasses import dataclass, field, replace`):

```python
from collections import deque
```

Replace the `WalletState` dataclass body (currently around line 64-91) — change ONLY the `recent_30d_trades` field annotation. Keep everything else identical:

```python
@dataclass(frozen=True)
class WalletState:
    """Running per-wallet aggregate at some point in time.

    Holds enough state to derive every trader feature in
    ``training_examples``. Updated by ``apply_*_to_state`` functions.

    ``recent_30d_trades`` is mutated in place by the apply_* functions
    (see issue #110 — the previous immutable-tuple rebuild was O(N) per
    trade and dominated the build-features wall time on heavy wallets).
    The dataclass stays frozen — only the deque's contents change, not
    the field reference.
    """

    first_seen_ts: int
    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    cumulative_buy_price_sum: float
    cumulative_buy_count: int
    realized_pnl_usd: float
    last_trade_ts: int | None
    recent_30d_trades: deque[int]
    # Running totals for avg_bet_size_usd. Storing the raw bet_sizes
    # tuple would cost O(N) per fold and O(N) per feature read on
    # heavy-hitter wallets — a streaming sum/count keeps both at O(1).
    # ``median_bet_size_usd`` is no longer derived (always None in
    # FeatureRow) — accepted v1 cost; could be revived via a bounded
    # rolling window if a model needs it.
    bet_size_sum: float
    bet_size_count: int
    category_counts: dict[str, int] = field(default_factory=dict)
```

- [ ] **Step 3: Update `empty_wallet_state`**

Edit `src/pscanner/corpus/features.py:123-140`. Replace `recent_30d_trades=()` with `recent_30d_trades=deque()`:

```python
def empty_wallet_state(*, first_seen_ts: int) -> WalletState:
    """Construct an initial WalletState for a wallet's first seen ts."""
    return WalletState(
        first_seen_ts=first_seen_ts,
        prior_trades_count=0,
        prior_buys_count=0,
        prior_resolved_buys=0,
        prior_wins=0,
        prior_losses=0,
        cumulative_buy_price_sum=0.0,
        cumulative_buy_count=0,
        realized_pnl_usd=0.0,
        last_trade_ts=None,
        recent_30d_trades=deque(),
        bet_size_sum=0.0,
        bet_size_count=0,
        category_counts={},
    )
```

- [ ] **Step 4: Delete `_trim_recent_trades` and inline the trim into `apply_buy_to_state` + `apply_sell_to_state`**

Edit `src/pscanner/corpus/features.py:154-201`. Replace the entire block from the `_RECENT_WINDOW_SECONDS` comment through the end of `apply_sell_to_state`:

```python
# Rolling-window for `recent_30d_trades` storage. The deque holds only
# trades within this many seconds of the most recent fold, so the
# accumulator's per-wallet memory stays bounded for very-active wallets.
# The window matches what `compute_features` reads (30 days), so trimmed
# entries are exactly the ones a feature query would have discarded.
_RECENT_WINDOW_SECONDS = 30 * 86_400


def _trim_and_append(window: deque[int], current_ts: int) -> None:
    """Drop entries older than ``current_ts - _RECENT_WINDOW_SECONDS`` and append.

    Mutates ``window`` in place. O(1) amortized per call (popleft + append),
    versus O(N) for the old tuple rebuild — the change that drives most of
    issue #110's wall-time reduction.
    """
    cutoff = current_ts - _RECENT_WINDOW_SECONDS
    while window and window[0] < cutoff:
        window.popleft()
    window.append(current_ts)


def apply_buy_to_state(state: WalletState, trade: Trade) -> WalletState:
    """Apply a BUY fill to wallet state. Returns a new WalletState.

    Mutates ``state.recent_30d_trades`` and ``state.category_counts`` in
    place — see :class:`WalletState` for why frozen+mutate is safe.
    """
    state.category_counts[trade.category] = state.category_counts.get(trade.category, 0) + 1
    _trim_and_append(state.recent_30d_trades, trade.ts)
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        prior_buys_count=state.prior_buys_count + 1,
        cumulative_buy_price_sum=state.cumulative_buy_price_sum + trade.price,
        cumulative_buy_count=state.cumulative_buy_count + 1,
        last_trade_ts=trade.ts,
        bet_size_sum=state.bet_size_sum + trade.notional_usd,
        bet_size_count=state.bet_size_count + 1,
    )


def apply_sell_to_state(state: WalletState, trade: _TradeFields) -> WalletState:
    """Apply a SELL fill to wallet state. Returns a new WalletState.

    Sells contribute to total trade count and recency but not to BUY
    aggregates (avg price paid, bet sizes, win/loss ledger). Accepts any
    object with the SELL-relevant fields so callers can pass either
    ``Trade`` or the bare repo ``CorpusTrade`` without rebuilding.
    Mutates ``state.recent_30d_trades`` in place.
    """
    _trim_and_append(state.recent_30d_trades, trade.ts)
    return replace(
        state,
        prior_trades_count=state.prior_trades_count + 1,
        last_trade_ts=trade.ts,
    )
```

Note: `category_counts` is also mutated in place above (the `state.category_counts[trade.category] = ...` line). That's the second perf fix from the issue (drop `dict(state.category_counts)`); since it's already in this same block, it lands in this commit. The plan's "Task 2: drop dict copy" effectively becomes a sub-bullet of Task 1.

- [ ] **Step 5: Update `LiveHistoryProvider` JSON deserialization**

Edit `src/pscanner/daemon/live_history.py:120-131`. The line that reconstructs the field from JSON currently reads:

```python
recent_30d_trades=tuple(json.loads(row["recent_30d_trades_json"])),
```

Change to:

```python
recent_30d_trades=deque(json.loads(row["recent_30d_trades_json"])),
```

Add `from collections import deque` to the top of `src/pscanner/daemon/live_history.py` if not already imported (search the existing imports first).

The serialization side at `daemon/live_history.py:320` (`json.dumps(list(state.recent_30d_trades))`) and `daemon/bootstrap.py:191` (same) does NOT need changes — `list(deque)` and `list(tuple)` both produce the same JSON output. The on-disk format is unchanged; rows written by the pre-#110 daemon read back fine post-#110.

- [ ] **Step 6: Update test fixtures**

Update each test file that constructs `WalletState(recent_30d_trades=...)` or asserts against the field. Run:

```bash
grep -rn "recent_30d_trades=()\|recent_30d_trades == ()\|recent_30d_trades=(" tests/
```

For each match, swap to `deque()` (empty) or `deque([...])` (with values). Specifically:

- `tests/corpus/test_features_state.py:47`: `assert state.recent_30d_trades == ()` → `assert state.recent_30d_trades == deque()`. Add `from collections import deque` to the top.
- `tests/corpus/test_features_state.py:77`: `assert state.recent_30d_trades == (1_000, 2_000)` → `assert state.recent_30d_trades == deque([1_000, 2_000])`.
- `tests/corpus/test_features_compute.py:102`: `recent_30d_trades=(900_000,)` → `recent_30d_trades=deque([900_000])`. Add the import.
- `tests/corpus/test_features_interaction.py:111`: `recent_30d_trades=()` → `recent_30d_trades=deque()`. Add the import.
- `tests/daemon/test_live_history.py:81`: `assert state.recent_30d_trades == ()` → `assert state.recent_30d_trades == deque()`. Add the import.

- [ ] **Step 7: Run the affected tests**

```bash
uv run pytest tests/corpus/test_features_state.py tests/corpus/test_features_compute.py tests/corpus/test_features_interaction.py tests/daemon/test_live_history.py tests/daemon/test_live_history_parity.py -v
```

Expected: all pass.

- [ ] **Step 8: Run full corpus + daemon + ml suites for regression**

```bash
uv run pytest tests/corpus/ tests/daemon/ tests/ml/ -q
```

Expected: all pass.

- [ ] **Step 9: Lint + format + types**

```bash
uv run ruff check src/pscanner/corpus/features.py src/pscanner/daemon/live_history.py tests/
uv run ruff format --check src/pscanner/corpus/features.py src/pscanner/daemon/live_history.py tests/
uv run ty check src/pscanner/corpus/features.py src/pscanner/daemon/live_history.py
```

Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/pscanner/corpus/features.py src/pscanner/daemon/live_history.py tests/
git commit -m "perf(corpus): deque-backed recent_30d_trades + in-place state mutation (#110)"
```

---

## Task 2: Verify the in-place `category_counts` mutation

**Files:**
- (No new files; this task is a verification + commit-message check.)

The dict-copy elision happened inside Task 1's edit (the `state.category_counts[trade.category] = ...` line replaced `new_categories = dict(state.category_counts); new_categories[...] = ...`). This task is a safety check — confirm the existing tests cover the in-place semantics correctly and the parity contract still holds.

- [ ] **Step 1: Confirm `apply_buy_to_state` mutates the input's `category_counts`**

Add a regression test to `tests/corpus/test_features_state.py`. Append:

```python
def test_apply_buy_mutates_category_counts_in_place() -> None:
    """category_counts is mutated in place — the post-change semantics from #110."""
    state = empty_wallet_state(first_seen_ts=1_000)
    trade = Trade(
        tx_hash="tx1",
        asset_id="a1",
        wallet_address="0xw",
        condition_id="0xc",
        outcome_side="YES",
        bs="BUY",
        price=0.5,
        size=10.0,
        notional_usd=5.0,
        ts=1_500,
        category="esports",
    )
    new_state = apply_buy_to_state(state, trade)

    # The same dict object is shared across both states post-replace.
    assert new_state.category_counts is state.category_counts
    assert state.category_counts == {"esports": 1}
    # The same deque object is shared too.
    assert new_state.recent_30d_trades is state.recent_30d_trades
    assert list(state.recent_30d_trades) == [1_500]
```

The imports needed at the top: `from pscanner.corpus.features import Trade, apply_buy_to_state, empty_wallet_state` (likely already present — check first).

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/corpus/test_features_state.py::test_apply_buy_mutates_category_counts_in_place -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/corpus/test_features_state.py
git commit -m "test(corpus): pin in-place mutation contract for WalletState (#110)"
```

---

## Task 3: Separate read connection in `build_features`

**Files:**
- Modify: `src/pscanner/corpus/examples.py:167-246`
- Modify: `src/pscanner/corpus/cli.py:426-442`

`iter_chronological` already uses keyset-paginated `fetchall` (per its docstring) to release the read transaction between chunks, so the pinning problem is partially mitigated. But a dedicated read connection still wins because:
- The chunk-boundary release is mid-batch, not mid-row. With one connection, `examples_repo.insert_or_ignore(pending_examples)` (every 500 rows) acquires the write lock and serializes against the chunk-boundary read transaction even when both are short.
- A separate connection lets the WAL checkpointer make progress without waiting for the read transaction to release.

`build_features`'s contract changes: the function now takes the read connection separately from `markets_conn`. Caller (CLI) opens both.

- [ ] **Step 1: Update `build_features` signature**

Edit `src/pscanner/corpus/examples.py:167`. Replace the function signature + body's first lines:

```python
def build_features(
    *,
    trades_repo: CorpusTradesRepo,
    resolutions_repo: MarketResolutionsRepo,
    examples_repo: TrainingExamplesRepo,
    markets_conn: sqlite3.Connection,
    now_ts: int,
    rebuild: bool = False,
    platform: str = "polymarket",
) -> int:
    """Build the training_examples table from corpus_trades + resolutions.

    Args:
        trades_repo: Source of raw trades (chronological). Should be
            constructed against a dedicated read-only connection — see
            issue #110. Sharing the write connection with ``examples_repo``
            serializes WAL checkpoints behind the chunk-boundary read
            transaction; a separate connection lets the cursor stream
            without contention.
        resolutions_repo: Kept for API compat; the per-trade resolution
            check now reads from the provider's in-memory map (seeded by
            ``_register_resolutions`` at startup) so the hot loop avoids a
            per-row SQLite SELECT.
        examples_repo: Sink for materialized rows.
        markets_conn: Connection used to load corpus_markets metadata
            and the one-shot ``market_resolutions`` snapshot.
        now_ts: ``built_at`` for new rows.
        rebuild: If True, drop training_examples before walking.
        platform: Which platform's rows to read and write. Default
            ``"polymarket"`` preserves the pre-multi-platform behavior
            for existing callers. Scopes ``corpus_trades``,
            ``corpus_markets``, ``market_resolutions``, and
            ``training_examples`` reads/writes to that platform.

    Returns:
        Number of rows actually written (deduped via INSERT OR IGNORE).
    """
```

The body itself is unchanged — `build_features` already takes `trades_repo` and uses it for `iter_chronological`. The change is purely in HOW the caller constructs `trades_repo` (against a separate read connection).

- [ ] **Step 2: Update the CLI to open a second read-only connection**

Edit `src/pscanner/corpus/cli.py:426-442`. Replace `_cmd_build_features`:

```python
async def _cmd_build_features(args: argparse.Namespace) -> int:
    """Rebuild the training_examples table from raw corpus_trades + resolutions."""
    db_path = Path(args.db)
    write_conn = init_corpus_db(db_path)
    # Dedicated read-only connection for the streaming chronological cursor
    # so writes (INSERT OR IGNORE) don't contend with the read txn (#110).
    read_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    read_conn.row_factory = sqlite3.Row
    try:
        written = build_features(
            trades_repo=CorpusTradesRepo(read_conn),
            resolutions_repo=MarketResolutionsRepo(write_conn),
            examples_repo=TrainingExamplesRepo(write_conn),
            markets_conn=write_conn,
            now_ts=int(time.time()),
            rebuild=bool(getattr(args, "rebuild", False)),
            platform=args.platform,
        )
        _log.info("corpus.build_features_done", written=written)
        return 0
    finally:
        read_conn.close()
        write_conn.close()
```

The `import sqlite3` at the top of `src/pscanner/corpus/cli.py` may or may not already be present — verify via:

```bash
grep -n "^import sqlite3\|^from sqlite3" src/pscanner/corpus/cli.py
```

If absent, add it.

- [ ] **Step 3: Add a test for the separate-connection wiring**

Many existing build_features tests construct `CorpusTradesRepo` against the same connection as the writer. Those tests still need to pass — `build_features` accepts whatever connections the caller hands it and doesn't enforce that they're separate. The contract is at the CLI level, not at `build_features` itself.

Add a smoke test confirming the CLI path uses two distinct connections. In `tests/corpus/test_examples.py` (or wherever `build_features` is tested — check via `grep -rln "build_features" tests/`):

```python
@pytest.mark.asyncio
async def test_cli_build_features_uses_separate_read_connection(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The CLI opens a read-only connection for the chronological cursor (#110)."""
    import argparse
    import sqlite3
    from unittest.mock import patch

    from pscanner.corpus.cli import _cmd_build_features

    db_path = tmp_path / "corpus.sqlite3"
    # Initialize the schema so the CLI can open the file.
    from pscanner.corpus.db import init_corpus_db
    init_corpus_db(db_path).close()

    args = argparse.Namespace(
        db=str(db_path),
        rebuild=False,
        platform="polymarket",
    )

    seen_uris: list[str | None] = []
    real_connect = sqlite3.connect

    def _spy_connect(database: str, *cargs, **ckwargs):  # type: ignore[no-untyped-def]
        # Capture the URI for the read-only open; ignore plain-path opens.
        uri = ckwargs.get("uri", False)
        if uri and "mode=ro" in str(database):
            seen_uris.append(str(database))
        return real_connect(database, *cargs, **ckwargs)

    with patch("pscanner.corpus.cli.sqlite3.connect", side_effect=_spy_connect):
        rc = await _cmd_build_features(args)
    assert rc == 0
    assert seen_uris, "_cmd_build_features must open a read-only connection"
    assert any("mode=ro" in u for u in seen_uris)
```

(`patch` target is `pscanner.corpus.cli.sqlite3.connect` because that's the name resolved inside the CLI module. If `sqlite3` is imported as `import sqlite3 as _sqlite3` or similar, adjust the patch target accordingly — verify by reading the import line at the top of the CLI module.)

- [ ] **Step 4: Run the new test plus the existing build_features tests**

```bash
uv run pytest tests/corpus/test_examples.py -v
```

Expected: all pass.

- [ ] **Step 5: Run the full corpus suite for regression**

```bash
uv run pytest tests/corpus/ -q
```

Expected: all pass.

- [ ] **Step 6: Lint + format + types**

```bash
uv run ruff check src/pscanner/corpus/examples.py src/pscanner/corpus/cli.py tests/corpus/test_examples.py
uv run ruff format --check src/pscanner/corpus/examples.py src/pscanner/corpus/cli.py tests/corpus/test_examples.py
uv run ty check src/pscanner/corpus/examples.py src/pscanner/corpus/cli.py
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/examples.py src/pscanner/corpus/cli.py tests/corpus/test_examples.py
git commit -m "perf(corpus): dedicated read-only connection for build_features cursor (#110)"
```

---

## Task 4: Update CLAUDE.md to clear the open-follow-ups bullet

**Files:**
- Modify: `CLAUDE.md` — open-follow-ups bullet for #58/#110

The CLAUDE.md `## Open follow-ups (no issues filed)` section currently has a bullet `- **`build-features` deque + read-connection follow-ups (#58 fixes 3-5).** ...`. With this PR, fixes 3-5 land. Update or remove that bullet.

- [ ] **Step 1: Locate the bullet**

```bash
grep -n "deque\|read-connection follow-ups\|recent_30d_trades" CLAUDE.md
```

- [ ] **Step 2: Replace the bullet with a status note**

Find the bullet starting with `**`build-features` deque + read-connection follow-ups`. Replace its body with a record of completion (keep the bullet so the surrounding context is preserved):

```
- **`build-features` perf — RESOLVED.** #58 fix #1 (in-memory resolutions cache + skip Trade rebuild for SELLs) shipped in PR #59. Fixes #2-#4 — deque-backed `recent_30d_trades`, in-place `category_counts` mutation, separate read connection — shipped via #110. Expect rebuild times to drop from ~9-10 hours to ~1-2 hours on the 2026-05-08 corpus.
```

(The exact wording can match the surrounding bullets' style. The point is to clear the open follow-up.)

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): mark build-features perf follow-ups resolved (#110)"
```

---

## Verification

After all 4 tasks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: zero failures, zero new warnings.

---

## Operational Validation (operator-driven, post-merge)

Acceptance criterion 4 ("rebuild target < 2h on the 2026-05-08 corpus") and acceptance criterion 3 ("byte-identical `training_examples`") require a real rebuild on the desktop training box. These are NOT agent tasks. Steps for the operator:

```bash
# 1. Pre-rebuild snapshot of current training_examples shape (to verify parity)
sqlite3 data/corpus.sqlite3 \
  "SELECT platform, COUNT(*) AS n, SUM(label_won) AS wins,
          AVG(realized_edge_pp) AS edge
   FROM training_examples
   GROUP BY platform;" > /tmp/te-pre-110.txt

# 2. Run the rebuild on the new branch
time uv run pscanner corpus build-features --rebuild

# 3. Post-rebuild snapshot
sqlite3 data/corpus.sqlite3 \
  "SELECT platform, COUNT(*) AS n, SUM(label_won) AS wins,
          AVG(realized_edge_pp) AS edge
   FROM training_examples
   GROUP BY platform;" > /tmp/te-post-110.txt

# 4. Compare — should be identical (deterministic walk over the same trades)
diff /tmp/te-pre-110.txt /tmp/te-post-110.txt

# 5. Confirm wall time < 2h (target per the issue)
```

If wall time misses the < 2h target, profile via `python -X importtime -m cProfile -o /tmp/build_features.prof -- $(which pscanner) corpus build-features --rebuild` and look at the residual hot path. The likely candidates: SQLite write lock held longer than expected (try `PRAGMA wal_autocheckpoint`), or the per-resolution drain inside `register_resolution` (O(W·U) where W is wallets, U is unscheduled buys per wallet — typically small but worth checking).

If parity diff is non-empty, that's a correctness regression — abort the merge. Most likely cause: in-place mutation of `recent_30d_trades` exposed a caller that holds the pre-update reference. Audit `StreamingHistoryProvider.observe`, `LiveHistoryProvider.observe`, `bootstrap_wallet`. The parity test should have caught this — file a follow-up to add coverage for whatever case slipped through.

---

## Self-Review

**Spec coverage:**
- Acceptance criterion 1 (all three changes land together): Tasks 1+2+3 in a single PR.
- Acceptance criterion 2 (no DB schema change): verified — `recent_30d_trades_json` column unchanged; tuple/deque both serialize as `[ts1, ts2, ...]`.
- Acceptance criterion 3 (output parity): operational validation via SQL snapshot diff.
- Acceptance criterion 4 (wall-clock < 2h on the 2026-05-08 corpus): operational validation.
- Acceptance criterion 5 (peak RSS no worse): operational validation; expected to be a strict win because dict-copy elision and deque are both lower-allocation than the current path.

**Placeholder scan:** None. All code blocks are concrete. Test fixture updates are listed by file:line.

**Type consistency:**
- `WalletState.recent_30d_trades: deque[int]` (Task 1) propagates to `LiveHistoryProvider` deserialization (Task 1, same commit) and all test fixtures (Task 1).
- `_trim_and_append(window: deque[int], current_ts: int) -> None` returns None because it mutates; old `_trim_recent_trades` returned a new tuple. Replacing one with the other shifts the API from "pure functional" to "mutating helper" — acceptable because all call sites are inside `apply_*_to_state` which were already replacing the field.
- `from collections import deque` added at the top of every file that constructs or asserts against `WalletState.recent_30d_trades`. Consistent across both `src` and `tests`.
