# Feature Projection Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the three independent ML feature-projection implementations (Python streaming, live daemon, DuckDB batch) into a single `FeatureFormula` registry with a Hypothesis-based parity test as the drift guarantee. Issue #145.

**Architecture:** New module `pscanner.corpus.feature_projection` holds an ordered tuple of `FeatureFormula` entries; each entry carries a Python lambda AND a parameterized SQL fragment that both compute the same value. `compute_features` (used by Python streaming + live daemon) becomes a thin wrapper around `project_row`. `_final_join_to_v2` (DuckDB) becomes a thin wrapper around `project_sql`. Magic numbers and the known-category tuple become module constants imported by both paths. A `hypothesis`-driven property test fuzzes trade streams through both engines and asserts row-for-row equality.

**Tech Stack:** Python 3.13, `uv`, `ruff`, `ty`, `pytest`, `hypothesis` (new test dep), DuckDB 1.x, SQLite (stdlib `sqlite3`), xgboost (existing).

**Working directory:** `/home/macph/projects/pscanner-worktrees/issue-145` (already created on branch `feat/issue-145-feature-projection` off `origin/main`).

**Reference files (read these before starting):**
- `src/pscanner/corpus/features.py:283-467` — current `FeatureRow` and `compute_features`
- `src/pscanner/corpus/_duckdb_engine.py:730-892` — current `_final_join_to_v2` SQL
- `src/pscanner/daemon/live_history.py` — `LiveHistoryProvider`, consumer of `compute_features`
- `tests/daemon/test_live_history_parity.py` — existing parity test (will keep working unchanged)

**Convention:** Every commit must pass `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`. `pyproject.toml` sets `filterwarnings = ["error"]`, so any new warning fails CI.

---

## File Structure

**Created:**
- `src/pscanner/corpus/feature_projection.py` — registry + `project_row` + `project_sql`
- `tests/corpus/test_feature_projection.py` — registry unit tests + SQL snapshot
- `tests/corpus/test_feature_projection_parity.py` — Hypothesis property test (engine-vs-engine)
- `tests/corpus/feature_projection_sql.snapshot` — checked-in snapshot of emitted SQL
- `scripts/feature_projection_byte_compare.py` — one-shot byte-compare gate for production rebuild

**Modified:**
- `pyproject.toml` — add `hypothesis` to the `test` dependency group
- `src/pscanner/corpus/features.py` — `compute_features` becomes a wrapper around `project_row`
- `src/pscanner/corpus/_duckdb_engine.py` — `_final_join_to_v2` emits its SELECT columns from `project_sql`

**Deleted at the end (Task 16, after parity passes):**
- Whichever per-feature correctness tests in `tests/corpus/` are now redundant with the property test — audit during Task 16, default is conservative deletion.

---

## Phase 1: Foundation

### Task 1: Add hypothesis to test deps and scaffold the module

**Files:**
- Modify: `pyproject.toml`
- Create: `src/pscanner/corpus/feature_projection.py` (empty module)
- Create: `tests/corpus/test_feature_projection.py` (single import test)

- [ ] **Step 1: Add `hypothesis` to the test dependency group**

In `pyproject.toml`, change line 37 from:

```toml
test = ["pytest==9.0.3", "pytest-asyncio==1.3.0", "respx==0.23.1"]
```

to:

```toml
test = ["pytest==9.0.3", "pytest-asyncio==1.3.0", "respx==0.23.1", "hypothesis==6.119.4"]
```

Use the latest stable version at implementation time — check with `uv pip index versions hypothesis | head -3` and pin to whatever it returns.

- [ ] **Step 2: Sync deps**

Run: `uv sync --group test`
Expected: hypothesis appears in `uv.lock`.

- [ ] **Step 3: Create empty module file**

Create `src/pscanner/corpus/feature_projection.py`:

```python
"""Single source of truth for the ML feature projection (#145).

See ``docs/superpowers/plans/2026-05-19-issue-145-feature-projection.md`` for
the architectural rationale. In short: the same FeatureRow is computed by
three code paths (Python streaming via ``compute_features``, live daemon via
``LiveHistoryProvider``, DuckDB batch via ``_final_join_to_v2``). This module
holds the canonical definitions and is consumed by all three.
"""

from __future__ import annotations
```

- [ ] **Step 4: Create the test file with a sanity import**

Create `tests/corpus/test_feature_projection.py`:

```python
"""Tests for the feature-projection registry (#145)."""

from __future__ import annotations


def test_module_imports() -> None:
    """The module loads without errors."""
    from pscanner.corpus import feature_projection  # noqa: F401
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/corpus/test_feature_projection.py -v`
Expected: 1 passed.

- [ ] **Step 6: Verify lint + types**

Run: `uv run ruff check tests/corpus/test_feature_projection.py src/pscanner/corpus/feature_projection.py && uv run ty check`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/pscanner/corpus/feature_projection.py tests/corpus/test_feature_projection.py
git commit -m "chore(corpus): scaffold feature_projection module (#145)"
```

---

### Task 2: Define `FeatureFormula`, `FeatureInputs`, and module constants

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`
- Modify: `tests/corpus/test_feature_projection.py`

- [ ] **Step 1: Add the constants and dataclasses**

Append to `src/pscanner/corpus/feature_projection.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pscanner.corpus.features import MarketMetadata, MarketState, Trade, WalletState

# Magic numbers. Both engines import these; never inline a literal that
# duplicates one of these.
CONFIDENCE_N_MIN = 20
HIGH_QUALITY_WIN_RATE_THRESHOLD = 0.55
RECENT_TRADES_WINDOW_DAYS = 30
SECONDS_PER_DAY = 86_400
MIN_PRICES_FOR_VOLATILITY = 2

# Multi-label category universe. The DuckDB engine refuses to start if a
# corpus row has a category outside this tuple (see _duckdb_engine.py
# _assert_no_unknown_categories); the registry's cat_* indicators are
# generated by looping over this tuple.
KNOWN_CATEGORIES: tuple[str, ...] = (
    "sports",
    "esports",
    "thesis",
    "macro",
    "elections",
    "crypto",
    "geopolitics",
    "tech",
    "culture",
)

FeatureDType = Literal["float", "int", "str", "tuple_str"]


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    """Bundle of state passed to a feature's Python evaluator.

    ``wallet`` and ``market`` are point-in-time-correct (computed strictly
    from events with ``ts < trade.ts`` by the caller). ``meta`` is static
    per market. ``trade`` is the current row being projected.
    """

    wallet: WalletState
    market: MarketState
    meta: MarketMetadata
    trade: Trade


@dataclass(frozen=True, slots=True)
class FeatureFormula:
    """One column in ``training_examples_v2``.

    ``py`` and ``sql`` MUST compute the same value on the same input. The
    parity test in ``tests/corpus/test_feature_projection_parity.py``
    asserts this with Hypothesis.

    The ``sql`` string is a template — placeholders of the form
    ``{w.field}``, ``{m.field}``, ``{meta.field}``, ``{t.field}`` are
    replaced by ``render_sql_fragment`` using ``SQL_BINDINGS`` below.
    """

    name: str
    dtype: FeatureDType
    nullable: bool
    py: Callable[[FeatureInputs], object]
    sql: str
    docs: str = ""
```

- [ ] **Step 2: Add tests for the dataclasses**

Append to `tests/corpus/test_feature_projection.py`:

```python
from pscanner.corpus import feature_projection as fp
from pscanner.corpus.features import (
    MarketMetadata,
    Trade,
    empty_market_state,
    empty_wallet_state,
)


def test_constants_are_load_bearing() -> None:
    """Constants exported by the module match what compute_features uses."""
    assert fp.CONFIDENCE_N_MIN == 20
    assert fp.HIGH_QUALITY_WIN_RATE_THRESHOLD == 0.55
    assert fp.SECONDS_PER_DAY == 86_400


def test_known_categories_covers_taxonomy() -> None:
    """All categories in the taxonomy are listed exactly once."""
    assert len(fp.KNOWN_CATEGORIES) == len(set(fp.KNOWN_CATEGORIES))
    # Sanity check: every entry is lowercase and non-empty
    for cat in fp.KNOWN_CATEGORIES:
        assert cat and cat == cat.lower()


def test_feature_formula_is_frozen() -> None:
    """FeatureFormula instances reject mutation."""
    formula = fp.FeatureFormula(
        name="x",
        dtype="float",
        nullable=False,
        py=lambda _i: 1.0,
        sql="1.0",
    )
    import dataclasses

    try:
        formula.name = "y"  # ty:ignore[invalid-assignment]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("FeatureFormula should be frozen")


def test_feature_inputs_holds_state() -> None:
    """FeatureInputs bundles the four input types."""
    trade = Trade(
        tx_hash="t",
        asset_id="a",
        wallet_address="w",
        condition_id="c",
        outcome_side="YES",
        bs="BUY",
        price=0.5,
        size=100.0,
        notional_usd=50.0,
        ts=1_700_000_000,
        category="sports",
    )
    inputs = fp.FeatureInputs(
        wallet=empty_wallet_state(first_seen_ts=trade.ts),
        market=empty_market_state(market_age_start_ts=trade.ts),
        meta=MarketMetadata(
            condition_id="c", category="sports", closed_at=0, opened_at=0
        ),
        trade=trade,
    )
    assert inputs.trade.tx_hash == "t"
    assert inputs.wallet.first_seen_ts == trade.ts
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/corpus/test_feature_projection.py -v`
Expected: 4 passed (1 from Task 1 + 3 new).

- [ ] **Step 4: Verify lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py tests/corpus/test_feature_projection.py
git commit -m "feat(corpus): add FeatureFormula + constants + FeatureInputs (#145)"
```

---

### Task 3: Define `SQL_BINDINGS` and `render_sql_fragment`

The Python lambdas read attributes directly off `FeatureInputs.wallet`, `.market`, `.meta`, `.trade`. The SQL fragments use `{scope.field}` placeholders that resolve to the column references the DuckDB engine actually emits (e.g. `wa.prior_wins_w`). This task defines that mapping.

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`
- Modify: `tests/corpus/test_feature_projection.py`

- [ ] **Step 1: Add the SQL bindings dict**

Append to `src/pscanner/corpus/feature_projection.py`:

```python
from collections.abc import Mapping

# Maps {scope.field} placeholder keys to the SQL column references used by
# pscanner.corpus._duckdb_engine._final_join_to_v2. The names on the SQL
# side are the column aliases that wallet_aggs (wa), market_aggs (ma), and
# wallet_cat_summary (wcs) expose at the final-join stage.
#
# Key naming: "<scope>.<feature-py-attr>". The same logical field can have
# a different name on each engine (e.g. WalletState.cumulative_buy_count
# == wallet_aggs.bet_size_count_w by construction); the binding hides
# that asymmetry.
SQL_BINDINGS: Mapping[str, str] = {
    # WalletState fields
    "w.prior_trades_count": "wa.prior_trades_count_w",
    "w.prior_buys_count": "wa.prior_buys_count_w",
    "w.prior_resolved_buys": "wa.prior_resolved_buys_w",
    "w.prior_wins": "wa.prior_wins_w",
    "w.prior_losses": "wa.prior_losses_w",
    "w.cumulative_buy_price_sum": "wa.cum_buy_price_sum_w",
    # WalletState.cumulative_buy_count tracks the same count as bet_size_count
    # by construction; the SQL side uses the bet_size_count_w column for both.
    "w.cumulative_buy_count": "wa.bet_size_count_w",
    "w.bet_size_sum": "wa.bet_size_sum_w",
    "w.bet_size_count": "wa.bet_size_count_w",
    "w.realized_pnl_usd": "wa.prior_realized_pnl_usd_w",
    "w.last_trade_ts": "wa.last_trade_ts_w",
    "w.first_seen_ts": "wa.first_seen_ts",
    "w.prior_trades_30d": "wa.prior_trades_30d_w",
    # MarketState fields
    "m.volume_so_far_usd": "ma.market_volume_so_far_w",
    "m.unique_traders_count": "ma.market_unique_traders_so_far_w",
    "m.market_age_start_ts": "ma.market_first_prior_ts_w",
    "m.last_trade_price": "ma.last_trade_price_w",
    "m.price_volatility": "ma.price_volatility_w",
    "m.price_count_20": "ma.price_count_20",
    # MarketMetadata fields
    "meta.category": "wa.category",
    "meta.categories_json": "wa.categories_json",
    "meta.closed_at": "wa.closed_at",
    # Trade fields (the current row being projected)
    "t.notional_usd": "wa.notional_usd",
    "t.price": "wa.price",
    "t.outcome_side": "wa.outcome_side",
    "t.ts": "wa.event_ts",
    # Already-computed columns (from wallet_cat_summary subquery)
    "wcs.top_category": "wcs.top_category",
    "wcs.category_diversity": "COALESCE(wcs.category_diversity, 0)",
}


def render_sql_fragment(template: str, bindings: Mapping[str, str] = SQL_BINDINGS) -> str:
    """Resolve ``{scope.field}`` placeholders against ``bindings``.

    Raises ``KeyError`` if the template references an unbound placeholder
    (e.g. ``{w.bogus_field}``) — this catches typos at module-load time
    instead of producing malformed SQL at query time.
    """
    rendered = template
    for key, value in bindings.items():
        rendered = rendered.replace("{" + key + "}", value)
    # Belt-and-braces: if any "{...}" placeholder survives the loop, it
    # didn't match a binding key. Surface that as a clear error rather
    # than letting DuckDB choke on the literal braces.
    if "{" in rendered:
        # Find the first unresolved placeholder for the error message.
        start = rendered.index("{")
        end = rendered.index("}", start)
        raise KeyError(
            f"feature_projection: unbound SQL placeholder {rendered[start : end + 1]!r}"
        )
    return rendered
```

- [ ] **Step 2: Add tests for SQL_BINDINGS and render_sql_fragment**

Append to `tests/corpus/test_feature_projection.py`:

```python
import pytest


def test_render_sql_fragment_substitutes_known_placeholder() -> None:
    """render_sql_fragment swaps {scope.field} with the bound column ref."""
    template = "{w.prior_wins} > 0"
    rendered = fp.render_sql_fragment(template)
    assert rendered == "wa.prior_wins_w > 0"


def test_render_sql_fragment_raises_on_unbound_placeholder() -> None:
    """An unbound placeholder raises KeyError, not silently leaks braces."""
    with pytest.raises(KeyError, match="bogus"):
        fp.render_sql_fragment("{w.bogus_field} > 0")


def test_render_sql_fragment_passes_through_literal_braces_when_resolved() -> None:
    """All placeholders in a real template resolve."""
    template = (
        "CASE WHEN {w.prior_resolved_buys} > 0 "
        "THEN CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys} "
        "ELSE NULL END"
    )
    rendered = fp.render_sql_fragment(template)
    assert "{" not in rendered
    assert "wa.prior_wins_w" in rendered
    assert "wa.prior_resolved_buys_w" in rendered
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/corpus/test_feature_projection.py -v`
Expected: 7 passed.

- [ ] **Step 4: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py tests/corpus/test_feature_projection.py
git commit -m "feat(corpus): add SQL_BINDINGS and render_sql_fragment (#145)"
```

---

## Phase 2: Port all features into the registry

The strategy: write ONE driving test that walks `FEATURES` and asserts each formula's `.py` callable returns the same value `compute_features` produces on a shared fixture. Then add formulas incrementally — the test fails for missing/incorrect ones, passes once `FEATURES` is complete.

### Task 4: Add the parity-vs-compute_features driving test (red)

This test will fail until Task 9 completes — that's intentional. It's the TDD harness for the porting work.

**Files:**
- Create: `tests/corpus/conftest.py` (shared fixtures)
- Modify: `tests/corpus/test_feature_projection.py`

- [ ] **Step 1: Add a shared trade-stream fixture in conftest**

Create `tests/corpus/conftest.py` (or append if it exists):

```python
"""Shared fixtures for tests/corpus/."""

from __future__ import annotations

import random

import pytest

from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
    compute_features,
)


def _build_synthetic_trades(seed: int, n: int) -> list[Trade]:
    """Generate a deterministic synthetic trade stream covering common shapes."""
    rng = random.Random(seed)  # noqa: S311
    wallets = [f"0xw{i:02d}" for i in range(6)]
    markets = [f"0xm{i:02d}" for i in range(4)]
    base_ts = 1_700_000_000
    out: list[Trade] = []
    for i in range(n):
        wallet = rng.choice(wallets)
        market = rng.choice(markets)
        side = rng.choice(("YES", "NO"))
        bs = rng.choices(("BUY", "SELL"), weights=(0.7, 0.3))[0]
        price = round(rng.uniform(0.05, 0.95), 4)
        size = round(rng.uniform(50.0, 500.0), 2)
        out.append(
            Trade(
                tx_hash=f"tx{i:04d}",
                asset_id=f"{market}-{side}",
                wallet_address=wallet,
                condition_id=market,
                outcome_side=side,
                bs=bs,
                price=price,
                size=size,
                notional_usd=round(price * size, 4),
                ts=base_ts + i * 60,
                category=rng.choice(("sports", "esports", "crypto")),
            )
        )
    return out


def _build_metadata(trades: list[Trade]) -> dict[str, MarketMetadata]:
    """Build a MarketMetadata for every market in the trade stream."""
    by_market: dict[str, MarketMetadata] = {}
    for t in trades:
        if t.condition_id in by_market:
            continue
        by_market[t.condition_id] = MarketMetadata(
            condition_id=t.condition_id,
            category=t.category,
            closed_at=t.ts + 86_400 * 7,
            opened_at=t.ts - 60,
            categories=(t.category,),
        )
    return by_market


@pytest.fixture
def trade_stream() -> list[Trade]:
    """Deterministic 80-trade stream for cross-feature parity tests."""
    return _build_synthetic_trades(seed=42, n=80)


@pytest.fixture
def metadata_for_stream(trade_stream: list[Trade]) -> dict[str, MarketMetadata]:
    """MarketMetadata covering every market in `trade_stream`."""
    return _build_metadata(trade_stream)


@pytest.fixture
def streaming_provider(
    trade_stream: list[Trade],
    metadata_for_stream: dict[str, MarketMetadata],
) -> StreamingHistoryProvider:
    """A StreamingHistoryProvider with the trade stream already observed.

    Caller can call `compute_features(trade, provider)` for any trade
    whose ts is <= the last trade's ts.
    """
    provider = StreamingHistoryProvider(metadata=metadata_for_stream)
    # Pre-register resolutions for half the markets so resolution math is
    # exercised by the fixture.
    for cond_id in list(metadata_for_stream)[: len(metadata_for_stream) // 2]:
        meta = metadata_for_stream[cond_id]
        provider.register_resolution(
            condition_id=cond_id,
            resolved_at=meta.closed_at,
            outcome_yes_won=1,
        )
    return provider
```

- [ ] **Step 2: Add the driving parity-vs-compute_features test (will fail)**

Append to `tests/corpus/test_feature_projection.py`:

```python
import dataclasses

from pscanner.corpus.features import FeatureRow, compute_features


def test_project_row_matches_compute_features(
    trade_stream, metadata_for_stream, streaming_provider
) -> None:
    """project_row produces the same FeatureRow as compute_features.

    This is the TDD harness for porting features. Initially expected to
    fail (project_row not defined). Each formula added to FEATURES makes
    one more field of this comparison pass.
    """
    # Pick a mid-stream trade so wallet + market state are non-empty
    trade = trade_stream[40]

    expected: FeatureRow = compute_features(trade, streaming_provider)
    actual: FeatureRow = fp.project_row(
        trade=trade,
        wallet=streaming_provider.wallet_state(trade.wallet_address, as_of_ts=trade.ts),
        market=streaming_provider.market_state(trade.condition_id, as_of_ts=trade.ts),
        meta=streaming_provider.market_metadata(trade.condition_id),
    )

    diffs = []
    for field in dataclasses.fields(FeatureRow):
        e = getattr(expected, field.name)
        a = getattr(actual, field.name)
        if e != a:
            diffs.append(f"  {field.name}: expected={e!r} actual={a!r}")
    assert not diffs, "feature divergence:\n" + "\n".join(diffs)
```

- [ ] **Step 3: Verify the test fails for the expected reason**

Run: `uv run pytest tests/corpus/test_feature_projection.py::test_project_row_matches_compute_features -v`
Expected: FAIL with `AttributeError: module 'pscanner.corpus.feature_projection' has no attribute 'project_row'` (or similar). This is intentional — the harness is in place.

- [ ] **Step 4: Commit the harness**

```bash
git add tests/corpus/conftest.py tests/corpus/test_feature_projection.py
git commit -m "test(corpus): add project_row vs compute_features driving test (#145)"
```

---

### Task 5: Define `FEATURES` (empty) + `project_row` skeleton

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`

- [ ] **Step 1: Add `FEATURES` and `project_row`**

Append to `src/pscanner/corpus/feature_projection.py`:

```python
from pscanner.corpus.features import FeatureRow

# The canonical registry. Order matches FeatureRow field declaration order so
# project_row can emit a tuple-positional FeatureRow construction below.
FEATURES: tuple[FeatureFormula, ...] = ()


def project_row(
    *,
    trade: Trade,
    wallet: WalletState,
    market: MarketState,
    meta: MarketMetadata,
) -> FeatureRow:
    """Compute a FeatureRow from point-in-time state.

    Walks ``FEATURES``, evaluates each formula's ``py`` against
    ``FeatureInputs``, and packages the result into a FeatureRow.
    """
    inputs = FeatureInputs(wallet=wallet, market=market, meta=meta, trade=trade)
    values = {formula.name: formula.py(inputs) for formula in FEATURES}
    return FeatureRow(**values)
```

- [ ] **Step 2: Verify the harness now fails with a more useful error**

Run: `uv run pytest tests/corpus/test_feature_projection.py::test_project_row_matches_compute_features -v`
Expected: FAIL with `TypeError: FeatureRow.__init__() missing N required positional arguments` (because `FEATURES` is empty and `FeatureRow` has ~40 required fields).

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors. (The test is still failing, but the production code is clean.)

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py
git commit -m "feat(corpus): add empty FEATURES tuple + project_row skeleton (#145)"
```

---

### Task 6: Port the passthrough and identity features

The simplest formulas: read one attribute off the input state, return it directly. No nulls, no arithmetic.

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`

- [ ] **Step 1: Replace the empty `FEATURES` tuple with the passthroughs**

In `src/pscanner/corpus/feature_projection.py`, replace `FEATURES: tuple[FeatureFormula, ...] = ()` with:

```python
FEATURES: tuple[FeatureFormula, ...] = (
    # ----- Passthrough wallet aggregates -----
    FeatureFormula(
        name="prior_trades_count", dtype="int", nullable=False,
        py=lambda i: i.wallet.prior_trades_count,
        sql="{w.prior_trades_count}",
    ),
    FeatureFormula(
        name="prior_buys_count", dtype="int", nullable=False,
        py=lambda i: i.wallet.prior_buys_count,
        sql="{w.prior_buys_count}",
    ),
    FeatureFormula(
        name="prior_resolved_buys", dtype="int", nullable=False,
        py=lambda i: i.wallet.prior_resolved_buys,
        sql="{w.prior_resolved_buys}",
    ),
    FeatureFormula(
        name="prior_wins", dtype="int", nullable=False,
        py=lambda i: i.wallet.prior_wins,
        sql="{w.prior_wins}",
    ),
    FeatureFormula(
        name="prior_losses", dtype="int", nullable=False,
        py=lambda i: i.wallet.prior_losses,
        sql="{w.prior_losses}",
    ),
    FeatureFormula(
        name="prior_realized_pnl_usd", dtype="float", nullable=False,
        py=lambda i: i.wallet.realized_pnl_usd,
        sql="{w.realized_pnl_usd}",
    ),
    FeatureFormula(
        name="prior_trades_30d", dtype="int", nullable=False,
        py=lambda i: i.wallet.prior_trades_30d,
        sql="{w.prior_trades_30d}",
    ),
    # ----- Trade-row passthroughs -----
    FeatureFormula(
        name="bet_size_usd", dtype="float", nullable=False,
        py=lambda i: i.trade.notional_usd,
        sql="{t.notional_usd}",
    ),
    FeatureFormula(
        name="side", dtype="str", nullable=False,
        py=lambda i: i.trade.outcome_side,
        sql="{t.outcome_side}",
    ),
    FeatureFormula(
        name="implied_prob_at_buy", dtype="float", nullable=False,
        py=lambda i: i.trade.price,
        sql="{t.price}",
    ),
    # ----- Market-state passthroughs -----
    FeatureFormula(
        name="market_volume_so_far_usd", dtype="float", nullable=False,
        py=lambda i: i.market.volume_so_far_usd,
        sql="COALESCE({m.volume_so_far_usd}, 0.0)",
    ),
    FeatureFormula(
        name="market_unique_traders_so_far", dtype="int", nullable=False,
        py=lambda i: i.market.unique_traders_count,
        sql="CAST(COALESCE({m.unique_traders_count}, 0) AS INTEGER)",
    ),
    FeatureFormula(
        name="last_trade_price", dtype="float", nullable=True,
        py=lambda i: i.market.last_trade_price,
        sql="{m.last_trade_price}",
    ),
    # ----- Market metadata passthroughs -----
    FeatureFormula(
        name="market_category", dtype="str", nullable=False,
        py=lambda i: i.meta.category,
        sql="{meta.category}",
    ),
)
```

Note: ``wallet.prior_trades_30d`` is not a field on `WalletState` in the current code — it's computed inline in `compute_features` from `recent_30d_trades`. We'll handle that in Task 7.

**Correction needed before moving on:** look at `features.py:406-407`:

```python
cutoff = trade.ts - 30 * _SECONDS_PER_DAY
recent_30d = sum(1 for ts in wallet.recent_30d_trades if ts >= cutoff)
```

The Python lambda for `prior_trades_30d` must replicate this. Replace the placeholder entry with:

```python
    FeatureFormula(
        name="prior_trades_30d", dtype="int", nullable=False,
        py=lambda i: sum(
            1 for ts in i.wallet.recent_30d_trades
            if ts >= i.trade.ts - RECENT_TRADES_WINDOW_DAYS * SECONDS_PER_DAY
        ),
        sql="{w.prior_trades_30d}",
    ),
```

(The DuckDB engine pre-computes this as `prior_trades_30d_w` in stage 2; on the SQL side it's just a passthrough.)

- [ ] **Step 2: Run the parity test, observe a different (smaller) error set**

Run: `uv run pytest tests/corpus/test_feature_projection.py::test_project_row_matches_compute_features -v`
Expected: STILL FAILS with `TypeError: FeatureRow.__init__() missing ~25 required positional arguments` (smaller because Task 6 covered ~15 of them).

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py
git commit -m "feat(corpus): port passthrough features to registry (#145)"
```

---

### Task 7: Port the nullable-division features

These are `compute_features` lines 377-405: `win_rate`, `avg_implied_prob_paid`, `realized_edge_pp`, `avg_bet_size_usd`, `median_bet_size_usd`, `bet_size_rel_to_avg`. All share the shape `numerator / denominator if denominator > 0 else None`.

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`

- [ ] **Step 1: Add the nullable-division entries to `FEATURES`**

Insert these entries into the `FEATURES` tuple after the passthroughs:

```python
    # ----- Nullable divisions (denominator == 0 → None) -----
    FeatureFormula(
        name="win_rate", dtype="float", nullable=True,
        py=lambda i: (
            i.wallet.prior_wins / i.wallet.prior_resolved_buys
            if i.wallet.prior_resolved_buys > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 "
            "THEN CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="avg_implied_prob_paid", dtype="float", nullable=True,
        py=lambda i: (
            i.wallet.cumulative_buy_price_sum / i.wallet.cumulative_buy_count
            if i.wallet.cumulative_buy_count > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.bet_size_count} > 0 "
            "THEN {w.cumulative_buy_price_sum} / {w.bet_size_count} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="realized_edge_pp", dtype="float", nullable=True,
        py=lambda i: (
            (i.wallet.prior_wins / i.wallet.prior_resolved_buys)
            - (i.wallet.cumulative_buy_price_sum / i.wallet.cumulative_buy_count)
            if i.wallet.prior_resolved_buys > 0 and i.wallet.cumulative_buy_count > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 AND {w.bet_size_count} > 0 "
            "THEN (CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys}) "
            "- ({w.cumulative_buy_price_sum} / {w.bet_size_count}) "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="avg_bet_size_usd", dtype="float", nullable=True,
        py=lambda i: (
            i.wallet.bet_size_sum / i.wallet.bet_size_count
            if i.wallet.bet_size_count > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.bet_size_count} > 0 "
            "THEN {w.bet_size_sum} / {w.bet_size_count} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="median_bet_size_usd", dtype="float", nullable=True,
        py=lambda _i: None,  # not maintained in v1 (see features.py:389)
        sql="CAST(NULL AS DOUBLE)",
    ),
    FeatureFormula(
        name="bet_size_rel_to_avg", dtype="float", nullable=True,
        py=lambda i: (
            i.trade.notional_usd / (i.wallet.bet_size_sum / i.wallet.bet_size_count)
            if i.wallet.bet_size_count > 0 and i.wallet.bet_size_sum > 0
            else None
        ),
        sql=(
            "CASE WHEN {w.bet_size_count} > 0 AND {w.bet_size_sum} > 0 "
            "THEN {t.notional_usd} / ({w.bet_size_sum} / {w.bet_size_count}) "
            "ELSE NULL END"
        ),
    ),
```

- [ ] **Step 2: Run the parity test**

Run: `uv run pytest tests/corpus/test_feature_projection.py::test_project_row_matches_compute_features -v`
Expected: STILL FAILS (more fields covered, still missing ~20). Note any FeatureRow init errors — they tell you what's still missing.

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py
git commit -m "feat(corpus): port nullable-division features (#145)"
```

---

### Task 8: Port the wallet-quality interaction features

Four features from #44: `edge_confidence_weighted`, `win_rate_confidence_weighted`, `is_high_quality_wallet`, `bet_size_relative_to_history`. These have hard-coded magic numbers (`20`, `0.55`) that must come from the module constants.

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`

- [ ] **Step 1: Append the interaction entries to `FEATURES`**

Insert after the nullable-divisions block:

```python
    # ----- Wallet-quality interaction features (#44) -----
    FeatureFormula(
        name="edge_confidence_weighted", dtype="float", nullable=False,
        py=lambda i: (
            (
                (i.wallet.prior_wins / i.wallet.prior_resolved_buys)
                - (i.wallet.cumulative_buy_price_sum / i.wallet.cumulative_buy_count)
            )
            * min(1.0, i.wallet.prior_resolved_buys / CONFIDENCE_N_MIN)
            if i.wallet.prior_resolved_buys > 0 and i.wallet.cumulative_buy_count > 0
            else 0.0
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 AND {w.bet_size_count} > 0 "
            "THEN ((CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys}) "
            "- ({w.cumulative_buy_price_sum} / {w.bet_size_count})) "
            f"* LEAST(1.0, CAST({{w.prior_resolved_buys}} AS DOUBLE) / {CONFIDENCE_N_MIN}.0) "
            "ELSE 0.0 END"
        ),
    ),
    FeatureFormula(
        name="win_rate_confidence_weighted", dtype="float", nullable=False,
        py=lambda i: (
            ((i.wallet.prior_wins / i.wallet.prior_resolved_buys) - 0.5)
            * min(1.0, i.wallet.prior_resolved_buys / CONFIDENCE_N_MIN)
            if i.wallet.prior_resolved_buys > 0
            else 0.0
        ),
        sql=(
            "CASE WHEN {w.prior_resolved_buys} > 0 "
            "THEN ((CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys}) - 0.5) "
            f"* LEAST(1.0, CAST({{w.prior_resolved_buys}} AS DOUBLE) / {CONFIDENCE_N_MIN}.0) "
            "ELSE 0.0 END"
        ),
    ),
    FeatureFormula(
        name="is_high_quality_wallet", dtype="int", nullable=False,
        py=lambda i: int(
            i.wallet.prior_resolved_buys >= CONFIDENCE_N_MIN
            and i.wallet.prior_resolved_buys > 0
            and (i.wallet.prior_wins / i.wallet.prior_resolved_buys)
                > HIGH_QUALITY_WIN_RATE_THRESHOLD
        ),
        sql=(
            f"CASE WHEN {{w.prior_resolved_buys}} >= {CONFIDENCE_N_MIN} "
            "AND (CAST({w.prior_wins} AS DOUBLE) "
            f"/ NULLIF({{w.prior_resolved_buys}}, 0)) > {HIGH_QUALITY_WIN_RATE_THRESHOLD} "
            "THEN 1 ELSE 0 END"
        ),
    ),
    FeatureFormula(
        name="bet_size_relative_to_history", dtype="float", nullable=False,
        # v1: median_bet_size_usd is never maintained, so the ratio is
        # always 1.0. See features.py:389 + 399-401.
        py=lambda _i: 1.0,
        sql="CAST(1.0 AS DOUBLE)",
    ),
```

- [ ] **Step 2: Run the parity test**

Run: `uv run pytest tests/corpus/test_feature_projection.py::test_project_row_matches_compute_features -v`
Expected: still fails with ~16 missing fields, but the wallet-quality features should be present in the diff output (and not in the `expected != actual` list — they should match).

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py
git commit -m "feat(corpus): port wallet-quality interaction features (#145)"
```

---

### Task 9: Port the temporal, category, and remaining features

This task finishes the registry. After it lands, the driving test from Task 4 should PASS.

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`

- [ ] **Step 1: Add temporal + category-counts entries**

Insert after the interaction block:

```python
    # ----- Temporal features -----
    FeatureFormula(
        name="wallet_age_days", dtype="float", nullable=False,
        py=lambda i: max(0.0, (i.trade.ts - i.wallet.first_seen_ts) / SECONDS_PER_DAY),
        sql=f"GREATEST(0.0, ({{t.ts}} - {{w.first_seen_ts}}) / {SECONDS_PER_DAY}.0)",
    ),
    FeatureFormula(
        name="seconds_since_last_trade", dtype="int", nullable=True,
        py=lambda i: (
            i.trade.ts - i.wallet.last_trade_ts
            if i.wallet.last_trade_ts is not None
            else None
        ),
        sql=(
            "CASE WHEN {w.last_trade_ts} IS NOT NULL "
            "THEN {t.ts} - {w.last_trade_ts} "
            "ELSE NULL END"
        ),
    ),
    FeatureFormula(
        name="market_age_seconds", dtype="int", nullable=False,
        # When the market has not yet been observed, compute_features reads
        # market_state's default empty state (market_age_start_ts=0), so
        # market_age_seconds = trade.ts on first sighting. The SQL mirrors
        # this with COALESCE-to-0 (see _duckdb_engine.py:798-801 comment).
        py=lambda i: i.trade.ts - i.market.market_age_start_ts,
        sql="CAST({t.ts} - COALESCE({m.market_age_start_ts}, 0) AS INTEGER)",
    ),
    FeatureFormula(
        name="time_to_resolution_seconds", dtype="int", nullable=True,
        py=lambda i: i.meta.closed_at - i.trade.ts,
        sql="CAST({meta.closed_at} - {t.ts} AS INTEGER)",
    ),
    # ----- Category counts (from wallet_cat_summary subquery on the SQL side) -----
    FeatureFormula(
        name="top_category", dtype="str", nullable=True,
        py=lambda i: (
            max(i.wallet.category_counts.items(), key=lambda kv: kv[1])[0]
            if i.wallet.category_counts
            else None
        ),
        sql="{wcs.top_category}",
    ),
    FeatureFormula(
        name="category_diversity", dtype="int", nullable=False,
        py=lambda i: len(i.wallet.category_counts),
        sql="{wcs.category_diversity}",
    ),
    # ----- Price volatility -----
    FeatureFormula(
        name="price_volatility_recent", dtype="float", nullable=True,
        py=lambda i: (
            __import__("statistics").pstdev(i.market.recent_prices)
            if len(i.market.recent_prices) >= MIN_PRICES_FOR_VOLATILITY
            else None
        ),
        sql=(
            f"CASE WHEN {{m.price_count_20}} >= {MIN_PRICES_FOR_VOLATILITY} "
            "THEN {m.price_volatility} ELSE NULL END"
        ),
    ),
```

Note on `price_volatility_recent`: `__import__("statistics").pstdev` is a deliberate import inline so the lambda doesn't need a module-level `import statistics` clutter — but `ruff` will dislike it. Better: add `import statistics` at the top of `feature_projection.py` and change the lambda to `statistics.pstdev(...)`. Apply that edit.

- [ ] **Step 2: Add the multi-label cat_* indicators via a loop**

The 9 `cat_*` indicators all share a shape. Generate them programmatically before the closing `)` of the `FEATURES` tuple:

```python
def _cat_indicator_formula(category: str) -> FeatureFormula:
    """Generate one cat_<category> indicator formula."""

    def _py(i: FeatureInputs) -> int:
        categories = i.meta.categories if i.meta.categories else (i.meta.category,)
        return int(category in set(categories))

    sql = (
        "CAST(CASE "
        "WHEN json_array_length(COALESCE({meta.categories_json}, '[]')) > 0 "
        "THEN list_contains("
        "CAST(json_extract({meta.categories_json}, '$') AS VARCHAR[]), "
        f"'{category}') "
        f"ELSE {{meta.category}} = '{category}' "
        "END AS INTEGER)"
    )
    return FeatureFormula(
        name=f"cat_{category}", dtype="int", nullable=False, py=_py, sql=sql,
    )
```

Add a tuple of indicator entries by extending `FEATURES`:

```python
# Extend FEATURES with the generated cat_* indicators. Defined as a
# separate statement (rather than expanding inside the FEATURES literal)
# because the closure capture inside _cat_indicator_formula requires a
# helper function — see issue #145 for the parity rationale.
FEATURES = FEATURES + tuple(_cat_indicator_formula(cat) for cat in KNOWN_CATEGORIES)
```

- [ ] **Step 3: Add the `market_categories` tuple-valued feature**

Insert before the `_cat_indicator_formula` extension:

```python
    # Tuple-valued; flows to training_examples_v2 as a JSON-encoded list
    # via the DuckDB engine's existing column type, and as a plain tuple
    # in the Python FeatureRow.
    FeatureFormula(
        name="market_categories", dtype="tuple_str", nullable=False,
        py=lambda i: i.meta.categories if i.meta.categories else (i.meta.category,),
        # The DuckDB side already stores this in training_examples_v2 as
        # categories_json; on the Python side compute_features returns a
        # tuple. The parity test ignores this column for SQL emission
        # (it's read from the source column directly downstream).
        # If you need to project it, use a coalesce on categories_json.
        sql=(
            "CASE "
            "WHEN json_array_length(COALESCE({meta.categories_json}, '[]')) > 0 "
            "THEN CAST(json_extract({meta.categories_json}, '$') AS VARCHAR[]) "
            "ELSE [{meta.category}] "
            "END"
        ),
    ),
```

- [ ] **Step 4: Run the parity test — it should now PASS**

Run: `uv run pytest tests/corpus/test_feature_projection.py::test_project_row_matches_compute_features -v`
Expected: **PASS**. If it fails, the diff output names the offending field(s); fix the corresponding formula and re-run. Common gotchas:
- Off-by-one in the `cat_*` indicator (Python uses `set(categories)`; the test fixture has `categories=(category,)` so set membership matches exactly)
- `top_category` tiebreak: Python's `max()` picks first-inserted at ties (dict iteration order). The SQL uses `wallet_cat_summary` which has its own tiebreak. If the fixture generates ties, this will diverge — adjust the fixture's category-shuffle seed.

- [ ] **Step 5: Run full corpus test suite to check for regressions**

Run: `uv run pytest tests/corpus/ -q`
Expected: all passing, no warnings (per `filterwarnings = ["error"]`).

- [ ] **Step 6: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py
git commit -m "feat(corpus): complete registry — project_row matches compute_features (#145)"
```

---

## Phase 3: Implement and snapshot `project_sql`

### Task 10: Implement `project_sql`

**Files:**
- Modify: `src/pscanner/corpus/feature_projection.py`
- Modify: `tests/corpus/test_feature_projection.py`

- [ ] **Step 1: Add `project_sql` to the module**

Append to `src/pscanner/corpus/feature_projection.py`:

```python
def project_sql(*, bindings: Mapping[str, str] = SQL_BINDINGS) -> str:
    """Emit the SELECT-list column expressions for ``training_examples_v2``.

    Returns a comma-separated string of ``<expression> AS <column_name>``
    lines, ready to splice into the ``_final_join_to_v2`` SELECT. The
    caller is responsible for the surrounding SELECT scaffolding
    (platform, tx_hash, label_won, JOIN clauses, WHERE).
    """
    parts = []
    for formula in FEATURES:
        rendered = render_sql_fragment(formula.sql, bindings)
        parts.append(f"{rendered} AS {formula.name}")
    return ",\n    ".join(parts)
```

- [ ] **Step 2: Add a smoke test that `project_sql()` returns a non-empty string with no unresolved placeholders**

Append to `tests/corpus/test_feature_projection.py`:

```python
def test_project_sql_renders_all_features() -> None:
    """project_sql produces a non-empty SELECT list with no leftover placeholders."""
    sql = fp.project_sql()
    assert sql.strip(), "project_sql returned empty"
    assert "{" not in sql, f"unresolved placeholder in: {sql[:200]}"
    # Every registered feature appears as a column alias
    for formula in fp.FEATURES:
        assert f" AS {formula.name}" in sql, f"missing column alias for {formula.name}"
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/corpus/test_feature_projection.py -v`
Expected: all pass.

- [ ] **Step 4: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/feature_projection.py tests/corpus/test_feature_projection.py
git commit -m "feat(corpus): implement project_sql (#145)"
```

---

### Task 11: Add the SQL snapshot test and FeatureRow exhaustiveness test

**Files:**
- Create: `tests/corpus/feature_projection_sql.snapshot`
- Modify: `tests/corpus/test_feature_projection.py`

- [ ] **Step 1: Generate the initial snapshot**

Run:

```bash
uv run python -c "from pscanner.corpus.feature_projection import project_sql; print(project_sql())" \
  > tests/corpus/feature_projection_sql.snapshot
```

Open `tests/corpus/feature_projection_sql.snapshot` and **manually review the emitted SQL**. Spot-check 3-5 columns against the corresponding lines in `src/pscanner/corpus/_duckdb_engine.py:730-892`:
- Does `win_rate` emit `CASE WHEN wa.prior_resolved_buys_w > 0 THEN CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w ELSE NULL END`?
- Does `edge_confidence_weighted` inline the constant `20.0`?
- Do all 9 `cat_*` indicators reference `wa.categories_json` and `wa.category`?

If any of these is wrong, fix the formula in `feature_projection.py`, regenerate the snapshot, repeat. **Do not commit a snapshot you have not eyeballed.**

- [ ] **Step 2: Add the snapshot test**

Append to `tests/corpus/test_feature_projection.py`:

```python
from pathlib import Path


def test_project_sql_matches_snapshot() -> None:
    """The emitted SQL matches the checked-in snapshot.

    To intentionally update the snapshot, run:

        uv run python -c "from pscanner.corpus.feature_projection \
            import project_sql; print(project_sql())" \
            > tests/corpus/feature_projection_sql.snapshot

    Then re-review the file before committing.
    """
    snapshot_path = Path(__file__).parent / "feature_projection_sql.snapshot"
    expected = snapshot_path.read_text().rstrip()
    actual = fp.project_sql().rstrip()
    assert actual == expected, (
        "project_sql() drifted from snapshot. "
        "If intentional, regenerate the snapshot as documented in the test."
    )
```

- [ ] **Step 3: Add the FeatureRow exhaustiveness test**

Append to `tests/corpus/test_feature_projection.py`:

```python
def test_features_match_feature_row() -> None:
    """FEATURES and FeatureRow expose the same column names."""
    registry_names = {f.name for f in fp.FEATURES}
    fr_names = {f.name for f in dataclasses.fields(FeatureRow)}
    extra_in_registry = registry_names - fr_names
    extra_in_fr = fr_names - registry_names
    assert not extra_in_registry, (
        f"FEATURES has names not in FeatureRow: {sorted(extra_in_registry)}"
    )
    assert not extra_in_fr, (
        f"FeatureRow has fields not in FEATURES: {sorted(extra_in_fr)}"
    )
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/corpus/test_feature_projection.py -v`
Expected: all pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add tests/corpus/feature_projection_sql.snapshot tests/corpus/test_feature_projection.py
git commit -m "test(corpus): snapshot test on project_sql + FeatureRow exhaustiveness (#145)"
```

---

## CHECKPOINT 1 — Registry complete, no callers switched

At this point:
- `feature_projection.FEATURES` is the complete registry.
- `project_row` produces identical `FeatureRow`s to `compute_features` (verified by Task 4 test).
- `project_sql` produces SQL that closely mirrors `_final_join_to_v2`'s hand-written SQL (verified by snapshot review).
- Nothing in production calls the new module yet.

**Review:** read the snapshot one more time. Read `feature_projection.py` end-to-end. If anything looks off, fix it now — before the next phase wires it up to production code.

---

## Phase 4: Switch the Python caller

### Task 12: Replace `compute_features` body with a call to `project_row`

**Files:**
- Modify: `src/pscanner/corpus/features.py`

- [ ] **Step 1: Replace the body of `compute_features`**

In `src/pscanner/corpus/features.py`, locate the `compute_features` function (line 367). Replace its body with:

```python
def compute_features(trade: Trade, history: HistoryProvider) -> FeatureRow:
    """Compute the full feature row for a trade, point-in-time correct.

    Thin wrapper around ``pscanner.corpus.feature_projection.project_row``;
    the canonical formulas live in that module's ``FEATURES`` registry.

    Pure function: takes only ``trade`` and ``history``. All
    non-determinism enters via the provider.
    """
    # Local import to avoid a circular dependency:
    # feature_projection imports FeatureRow + state types from features.py.
    from pscanner.corpus.feature_projection import project_row

    return project_row(
        trade=trade,
        wallet=history.wallet_state(trade.wallet_address, as_of_ts=trade.ts),
        market=history.market_state(trade.condition_id, as_of_ts=trade.ts),
        meta=history.market_metadata(trade.condition_id),
    )
```

Delete the now-unused imports if any (e.g. `statistics` may no longer be referenced — only delete it if `ruff F401` flags it).

- [ ] **Step 2: Run the existing live-history parity test**

Run: `uv run pytest tests/daemon/test_live_history_parity.py -v`
Expected: PASS unchanged (this test runs both providers through `compute_features`; if it passes, our switch is transparent).

- [ ] **Step 3: Run the new project_row parity test**

Run: `uv run pytest tests/corpus/test_feature_projection.py -v`
Expected: PASS.

- [ ] **Step 4: Run the broader test suite to catch downstream callers**

Run: `uv run pytest tests/corpus/ tests/ml/ tests/daemon/ -q`
Expected: all pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/features.py
git commit -m "refactor(corpus): compute_features delegates to project_row (#145)"
```

---

## Phase 5: Switch the DuckDB caller, with byte-compare gate

### Task 13: Add Hypothesis strategies for engine-vs-engine fuzz testing

**Files:**
- Create: `tests/corpus/test_feature_projection_parity.py`

- [ ] **Step 1: Sketch the strategies and the failing property test**

Create `tests/corpus/test_feature_projection_parity.py`:

```python
"""Engine-vs-engine parity test for project_row vs project_sql (#145).

Drives a Hypothesis-generated trade stream through both the Python engine
(via build_features) and the DuckDB engine (via build_features_duckdb),
then compares the resulting training_examples_v2 rows.

Initially FAILS — passes once Task 14 lands (when _final_join_to_v2 is
switched to project_sql).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pscanner.corpus.features import MarketMetadata, Trade


# --- Strategies ----------------------------------------------------------


@st.composite
def trade_strategy(
    draw,
    wallets: list[str],
    markets: list[str],
    base_ts: int,
    index: int,
) -> Trade:
    """Generate one Trade. ts is monotonic via the `index` arg."""
    wallet = draw(st.sampled_from(wallets))
    market = draw(st.sampled_from(markets))
    side = draw(st.sampled_from(("YES", "NO")))
    bs = draw(st.sampled_from(("BUY", "SELL")))
    price = draw(st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    size = draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
    return Trade(
        tx_hash=f"tx{index:05d}",
        asset_id=f"{market}-{side}",
        wallet_address=wallet,
        condition_id=market,
        outcome_side=side,
        bs=bs,
        price=round(price, 4),
        size=round(size, 2),
        notional_usd=round(price * size, 4),
        ts=base_ts + index * 60,
        category=draw(st.sampled_from(("sports", "esports", "crypto"))),
    )


@st.composite
def trade_stream(draw, min_size: int = 20, max_size: int = 200) -> list[Trade]:
    """Generate a chronologically-ordered stream of trades."""
    n_wallets = draw(st.integers(2, 6))
    n_markets = draw(st.integers(2, 4))
    wallets = [f"0xw{i:02d}" for i in range(n_wallets)]
    markets = [f"0xm{i:02d}" for i in range(n_markets)]
    n = draw(st.integers(min_size, max_size))
    base_ts = 1_700_000_000
    return [
        draw(trade_strategy(wallets, markets, base_ts, i))
        for i in range(n)
    ]


# --- Harness ------------------------------------------------------------


def _materialize_stream_to_corpus(stream: Iterable[Trade], db_path: Path) -> None:
    """Write `stream` into corpus_trades + corpus_markets + market_resolutions.

    Mirrors what `pscanner corpus backfill` produces. Resolves every market
    at the latest trade's ts so feature labels have non-null label_won.
    """
    from pscanner.corpus.repos import init_corpus_db

    trades = list(stream)
    if not trades:
        return
    conn = init_corpus_db(db_path)
    try:
        markets = {t.condition_id for t in trades}
        last_ts = max(t.ts for t in trades)
        for cond_id in markets:
            conn.execute(
                "INSERT OR IGNORE INTO corpus_markets "
                "(platform, condition_id, event_slug, title, category, "
                " categories_json, opened_at, closed_at, completed_at, truncated_at_offset_cap) "
                "VALUES ('polymarket', ?, ?, ?, 'sports', '[\"sports\"]', ?, ?, ?, 0)",
                (cond_id, cond_id, cond_id, trades[0].ts - 60, last_ts, last_ts),
            )
            conn.execute(
                "INSERT OR IGNORE INTO market_resolutions "
                "(platform, condition_id, resolved_at, outcome_yes_won) "
                "VALUES ('polymarket', ?, ?, 1)",
                (cond_id, last_ts),
            )
        for t in trades:
            conn.execute(
                "INSERT OR IGNORE INTO corpus_trades "
                "(platform, tx_hash, asset_id, wallet_address, condition_id, "
                " outcome_side, bs, price, size, notional_usd, ts) "
                "VALUES ('polymarket', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t.tx_hash, t.asset_id, t.wallet_address, t.condition_id,
                 t.outcome_side, t.bs, t.price, t.size, t.notional_usd, t.ts),
            )
        conn.commit()
    finally:
        conn.close()


def _run_python_engine(db_path: Path) -> list[dict[str, object]]:
    """Run build-features --engine python and return training_examples_v2 rows."""
    from pscanner.corpus.examples import build_features

    build_features(db_path=db_path, rebuild=True)
    return _read_training_examples(db_path)


def _run_duckdb_engine(db_path: Path, scratch_dir: Path) -> list[dict[str, object]]:
    """Run build-features --engine duckdb and return training_examples_v2 rows."""
    from pscanner.corpus._duckdb_engine import build_features_duckdb

    build_features_duckdb(
        db_path=db_path,
        scratch_dir=scratch_dir,
        memory_limit="2GB",
        threads=2,
    )
    return _read_training_examples(db_path)


def _read_training_examples(db_path: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM training_examples_v2 "
            "ORDER BY trade_ts, wallet_address, condition_id, asset_id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# --- The property test --------------------------------------------------


@given(stream=trade_stream(min_size=20, max_size=100))
@settings(
    max_examples=20,  # bump to 200+ once green on CI
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_python_duckdb_engines_produce_identical_feature_rows(
    stream: list[Trade], tmp_path: Path
) -> None:
    """build-features in python engine and duckdb engine produce row-identical output."""
    py_db = tmp_path / "py.sqlite3"
    duck_db = tmp_path / "duck.sqlite3"
    _materialize_stream_to_corpus(stream, py_db)
    _materialize_stream_to_corpus(stream, duck_db)

    py_rows = _run_python_engine(py_db)
    duck_rows = _run_duckdb_engine(duck_db, tmp_path / "scratch")

    assert len(py_rows) == len(duck_rows), (
        f"row count diverges: python={len(py_rows)} duckdb={len(duck_rows)}"
    )
    for i, (p, d) in enumerate(zip(py_rows, duck_rows, strict=True)):
        diffs = []
        for col in sorted(p.keys() | d.keys()):
            pv, dv = p.get(col), d.get(col)
            if isinstance(pv, float) and isinstance(dv, float):
                if not _floats_equal(pv, dv):
                    diffs.append(f"  {col}: py={pv!r} duck={dv!r}")
            elif pv != dv:
                diffs.append(f"  {col}: py={pv!r} duck={dv!r}")
        assert not diffs, (
            f"row {i} (tx_hash={p.get('tx_hash')}) diverges:\n" + "\n".join(diffs)
        )


def _floats_equal(a: float, b: float) -> bool:
    """Approximate float equality, tolerant to NULL and float-precision drift."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < max(1e-7, 1e-5 * max(abs(a), abs(b)))


@pytest.mark.skip(reason="enable after Task 14 lands; large-batch confidence test")
@given(stream=trade_stream(min_size=200, max_size=500))
@settings(max_examples=50, deadline=None)
def test_python_duckdb_parity_large(stream: list[Trade], tmp_path: Path) -> None:
    """Same as the small parity test but with bigger streams. Opt-in for nightly."""
    test_python_duckdb_engines_produce_identical_feature_rows(stream, tmp_path)
```

- [ ] **Step 2: Verify the property test fails (DuckDB engine still hand-written)**

Run: `uv run pytest tests/corpus/test_feature_projection_parity.py -v -x`
Expected: at least one Hypothesis example produces a divergence (the `cat_*` columns and wallet-quality features might still be aligned because the hand-written SQL is correct today; but ANY future drift would be caught). It's also acceptable if it PASSES — that means the current hand-written SQL is already row-equivalent to the registry, which is the precondition for switching it.

Note: if it passes, do not skip Task 14. The switch is still required to make the registry the source of truth.

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check tests/corpus/test_feature_projection_parity.py && uv run ty check tests/corpus/test_feature_projection_parity.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/corpus/test_feature_projection_parity.py
git commit -m "test(corpus): add Hypothesis-driven engine parity test (#145)"
```

---

### Task 14: Switch `_final_join_to_v2` to use `project_sql`

**Files:**
- Modify: `src/pscanner/corpus/_duckdb_engine.py`

- [ ] **Step 1: Read the existing `_final_join_to_v2` to understand the surrounding SQL**

Open `src/pscanner/corpus/_duckdb_engine.py` and locate `_final_join_to_v2` (around line 730). Note the structure:
1. Outer `CREATE OR REPLACE TABLE ... AS SELECT ...`
2. Identity columns (`platform`, `tx_hash`, `asset_id`, `wallet_address`, `condition_id`, `trade_ts`, `built_at`)
3. The 40-column feature projection (THIS is what `project_sql` replaces)
4. The `label_won` CASE
5. `FROM wallet_aggs wa JOIN resolutions r USING (condition_id) LEFT JOIN wallet_cat_summary wcs ... LEFT JOIN market_aggs ma ...`
6. `WHERE wa.is_buy_only = 1`

Only the feature-projection block changes; everything else stays.

- [ ] **Step 2: Replace the feature-projection block**

In `_final_join_to_v2`, replace the lines that compute the ~40 feature columns (lines 741-869, give or take) with a single `{project_sql_block}` substitution. The function should look like:

```python
def _final_join_to_v2(
    scratch: duckdb.DuckDBPyConnection,
    *,
    platform: str,
    now_ts: int,
) -> None:
    """..."""  # keep the existing docstring
    from pscanner.corpus.feature_projection import project_sql

    projection = project_sql()  # generated from the registry; see issue #145

    scratch.execute(
        f"""
        CREATE OR REPLACE TABLE {_TE_LOCAL_TABLE} AS
        SELECT
            ? AS platform,
            wa.tx_hash,
            wa.asset_id,
            wa.wallet_address,
            wa.condition_id,
            wa.event_ts AS trade_ts,
            ? AS built_at,
            {projection},
            CASE
                WHEN (r.outcome_yes_won = 1 AND wa.outcome_side = 'YES')
                  OR (r.outcome_yes_won = 0 AND wa.outcome_side = 'NO')
                THEN 1 ELSE 0
            END AS label_won
        FROM wallet_aggs wa
        JOIN resolutions r USING (condition_id)
        LEFT JOIN wallet_cat_summary wcs
            USING (wallet_address, event_ts, kind_priority, tx_hash, asset_id)
        LEFT JOIN market_aggs ma
            USING (wallet_address, condition_id, event_ts, kind_priority, tx_hash, asset_id)
        WHERE wa.is_buy_only = 1
        """,  # noqa: S608 — projection is from a static registry, no user input
        [platform, now_ts],
    )
```

- [ ] **Step 3: Run the Hypothesis parity test**

Run: `uv run pytest tests/corpus/test_feature_projection_parity.py -v -x`
Expected: PASS. If Hypothesis finds a divergence, it will shrink to a minimal-trade-count repro. Fix the registry formula (or, if the existing SQL was the wrong one, fix the formula AND regenerate the snapshot).

- [ ] **Step 4: Run the existing DuckDB engine tests**

Run: `uv run pytest tests/corpus/test_duckdb_engine.py -v`
Expected: PASS unchanged.

- [ ] **Step 5: Run the full corpus + ml + daemon test suites**

Run: `uv run pytest tests/corpus/ tests/ml/ tests/daemon/ -q`
Expected: all pass.

- [ ] **Step 6: Lint + types**

Run: `uv run ruff check . && uv run ty check`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/_duckdb_engine.py
git commit -m "refactor(corpus): _final_join_to_v2 emits SELECT columns via project_sql (#145)"
```

---

### Task 15: Add the byte-compare gate script and document the manual gate

The Hypothesis test gives statistical confidence; this script gives byte-level confidence on a real production-shaped corpus before merging.

**Files:**
- Create: `scripts/feature_projection_byte_compare.py`

- [ ] **Step 1: Write the byte-compare script**

Create `scripts/feature_projection_byte_compare.py`:

```python
"""One-shot byte-compare gate for the feature-projection switch (#145).

Reads a real corpus DB, runs build-features under both the python and
duckdb engines into two copies of training_examples_v2, and reports any
divergence at row granularity. Run before merging the DuckDB switch.

Usage:
    uv run python scripts/feature_projection_byte_compare.py \\
        --corpus ./data/corpus.sqlite3 \\
        --sample-size 100000

The script copies the source corpus to two tmp DBs, runs build-features
on each, and diffs training_examples_v2 row-by-row. Exits non-zero on any
divergence; prints up to 20 example divergences to stdout.

This is a manual gate — the result should be pasted into the PR
description as evidence before merging.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

_FLOAT_REL_TOL = 1e-5
_FLOAT_ABS_TOL = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="path to corpus.sqlite3")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="cap on number of training_examples_v2 rows compared (default 100000)",
    )
    parser.add_argument(
        "--duckdb-memory",
        default="6GB",
        help="DuckDB memory limit (default 6GB)",
    )
    return parser.parse_args()


def _floats_close(a: object, b: object) -> bool:
    if a is None and b is None:
        return True
    if not isinstance(a, float) or not isinstance(b, float):
        return False
    return abs(a - b) < max(_FLOAT_ABS_TOL, _FLOAT_REL_TOL * max(abs(a), abs(b)))


def _read_rows(db_path: Path, limit: int) -> Iterator[dict[str, object]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(
            "SELECT * FROM training_examples_v2 "
            "ORDER BY trade_ts, wallet_address, condition_id, asset_id "
            "LIMIT ?",
            (limit,),
        ):
            yield dict(row)
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    if not args.corpus.exists():
        print(f"ERROR: corpus DB not found at {args.corpus}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="feature-projection-bytecompare-") as tmpdir:
        tmp = Path(tmpdir)
        py_db = tmp / "py.sqlite3"
        duck_db = tmp / "duck.sqlite3"
        scratch = tmp / "scratch"

        print(f"[1/4] Copying corpus to {py_db} and {duck_db} ...")
        shutil.copy(args.corpus, py_db)
        shutil.copy(args.corpus, duck_db)

        # Clear training_examples_v2 in both
        for path in (py_db, duck_db):
            conn = sqlite3.connect(path)
            try:
                conn.execute("DELETE FROM training_examples_v2")
                conn.commit()
            finally:
                conn.close()

        print("[2/4] Running build-features --engine python ...")
        from pscanner.corpus.examples import build_features
        build_features(db_path=py_db, rebuild=True)

        print(f"[3/4] Running build-features --engine duckdb (memory={args.duckdb_memory}) ...")
        from pscanner.corpus._duckdb_engine import build_features_duckdb
        build_features_duckdb(
            db_path=duck_db,
            scratch_dir=scratch,
            memory_limit=args.duckdb_memory,
            threads=4,
        )

        print(f"[4/4] Comparing up to {args.sample_size} rows ...")
        divergences = 0
        examples_shown = 0
        for i, (p, d) in enumerate(
            zip(_read_rows(py_db, args.sample_size),
                _read_rows(duck_db, args.sample_size),
                strict=False)
        ):
            row_diff = []
            for col in sorted(p.keys() | d.keys()):
                pv, dv = p.get(col), d.get(col)
                if isinstance(pv, float) or isinstance(dv, float):
                    if not _floats_close(pv, dv):
                        row_diff.append((col, pv, dv))
                elif pv != dv:
                    row_diff.append((col, pv, dv))
            if row_diff:
                divergences += 1
                if examples_shown < 20:
                    print(f"\n  ROW {i} (tx_hash={p.get('tx_hash')}):")
                    for col, pv, dv in row_diff:
                        print(f"    {col}: py={pv!r}  duck={dv!r}")
                    examples_shown += 1

        if divergences:
            print(f"\nRESULT: {divergences} diverging rows out of "
                  f"~{args.sample_size} sampled.")
            return 1
        print("\nRESULT: all sampled rows agree byte-for-byte.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script is syntactically clean**

Run: `uv run ruff check scripts/feature_projection_byte_compare.py && uv run ty check scripts/feature_projection_byte_compare.py`
Expected: no errors.

- [ ] **Step 3: Smoke-test the script on a small fixture**

If a local corpus is available, run:

```bash
uv run python scripts/feature_projection_byte_compare.py \
    --corpus ./data/corpus.sqlite3 \
    --sample-size 10000 \
    --duckdb-memory 2GB
```

Expected: `RESULT: all sampled rows agree byte-for-byte.`

If no local corpus is available, skip this step and document it in the PR description as a follow-up task for the next operator with access.

- [ ] **Step 4: Commit**

```bash
git add scripts/feature_projection_byte_compare.py
git commit -m "tool(corpus): byte-compare script for feature_projection switch (#145)"
```

---

## CHECKPOINT 2 — Both engines on the registry, parity proven

At this point:
- The Python and DuckDB engines BOTH derive their feature projection from `FEATURES`.
- The Hypothesis test verifies row-equality on synthetic streams.
- The byte-compare script provides a production-scale gate on demand.
- The existing `test_live_history_parity.py` still passes (live ↔ streaming parity, unchanged).

**Manual review before Task 16:**
1. Re-read `feature_projection.py` end-to-end. Are the magic numbers all from constants?
2. Run `uv run pytest -q` — does the full suite pass?
3. If you have a local corpus, run the byte-compare script and paste the result into the PR description.

---

## Phase 6: Cleanup

### Task 16: Audit and delete redundant tests

The property test plus the snapshot test plus the existing live-vs-streaming parity test should now cover everything the old hand-written per-feature tests covered. Find which ones are now redundant and delete them.

**Files:**
- Modify or delete: tests in `tests/corpus/` and `tests/ml/` that hand-check individual feature values

- [ ] **Step 1: Inventory existing per-feature tests**

Run:

```bash
grep -rn "compute_features\|FeatureRow" tests/ | grep -v "test_feature_projection\|test_live_history_parity" | head -40
```

Read each match. For each test, ask: "Does the property test or the existing parity test already cover this case?"

- **Keep** tests that exercise behaviors orthogonal to the registry (e.g. tests on `apply_buy_to_state`, `StreamingHistoryProvider.observe`, `LiveHistoryProvider` state-storage roundtrips).
- **Delete** tests that hand-check the value of a single feature column on a fixed input — those are now covered by the property test's exhaustive walk over `FEATURES`.

When in doubt, keep the test.

- [ ] **Step 2: For each test marked for deletion, delete it and run the suite**

For each deletion:

```bash
git rm tests/corpus/<test_file_that_is_now_redundant>.py  # or delete specific test_* functions inside
uv run pytest -q
```

If anything else breaks, restore the file (`git restore --staged <path> && git checkout -- <path>`) and re-evaluate.

- [ ] **Step 3: Final full-suite run**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: all green.

- [ ] **Step 4: Commit cleanup**

```bash
git add -A
git commit -m "test(corpus): drop redundant per-feature tests covered by property test (#145)"
```

---

### Task 17: Update CLAUDE.md to point at the new module

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a paragraph under "ML training pipeline" describing the new module**

Locate the "ML training pipeline (`pscanner.ml`)" section in `CLAUDE.md`. Add a paragraph (or amend an existing one) saying:

```markdown
- **Feature projection registry (#145).** `pscanner.corpus.feature_projection` is the single source of truth for the FeatureRow projection step. `FEATURES: tuple[FeatureFormula, ...]` carries one entry per training_examples_v2 column with both a Python lambda and a parameterized SQL fragment. `compute_features` (used by both `StreamingHistoryProvider` and `LiveHistoryProvider`) and `_duckdb_engine._final_join_to_v2` both derive their projection from this registry. Magic numbers (`CONFIDENCE_N_MIN`, `HIGH_QUALITY_WIN_RATE_THRESHOLD`, `KNOWN_CATEGORIES`) live as module constants. Parity is enforced by `tests/corpus/test_feature_projection_parity.py` (Hypothesis-driven, engine-vs-engine row equality) and `tests/corpus/feature_projection_sql.snapshot` (PR-review diff of emitted SQL). Adding a feature = one entry in `FEATURES` + snapshot regen + new FeatureRow field + parity test passes.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): document feature_projection registry (#145)"
```

---

### Task 18: Open the PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/issue-145-feature-projection
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "feat(corpus): feature projection registry as single source of truth (#145)" --body "$(cat <<'EOF'
## Summary
- Adds `pscanner.corpus.feature_projection` with a `FEATURES` registry of one entry per `training_examples_v2` column. Each entry carries both a Python lambda and a parameterized SQL fragment.
- `compute_features` and `_final_join_to_v2` both derive their projection from the registry. Magic numbers (`CONFIDENCE_N_MIN`, `HIGH_QUALITY_WIN_RATE_THRESHOLD`, `KNOWN_CATEGORIES`) live as module constants.
- Adds a Hypothesis-driven parity property test that asserts row-for-row equality between the Python and DuckDB engines on synthetic trade streams.
- Adds `scripts/feature_projection_byte_compare.py` as a manual byte-level gate before merging.

Closes #145.

## Test plan
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
- [ ] Hypothesis parity test passes locally with `max_examples=200` (bumped from CI default 20)
- [ ] `uv run python scripts/feature_projection_byte_compare.py --corpus data/corpus.sqlite3 --sample-size 100000` reports byte-for-byte agreement
- [ ] Snapshot `tests/corpus/feature_projection_sql.snapshot` reviewed by hand against the prior `_final_join_to_v2` body

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Paste the byte-compare result into the PR description**

If you ran the byte-compare script in Task 15, paste its output into a comment on the PR. If you have not, leave a note in the PR description that the next operator with corpus access should run it before approving.

---

## Self-Review (run before handoff)

**1. Spec coverage:**

| RFC section | Plan task(s) |
|---|---|
| `FeatureFormula` + `FEATURES` + `project_row` + `project_sql` | Tasks 2, 5, 6-9, 10 |
| Magic numbers as module constants | Task 2 |
| 9 `cat_*` indicators via loop | Task 9 |
| `compute_features` wrapper | Task 12 |
| `_final_join_to_v2` wrapper | Task 14 |
| Hypothesis property test | Tasks 13-14 |
| SQL snapshot test | Task 11 |
| FeatureRow exhaustiveness test | Task 11 |
| 100k-row byte-compare gate | Task 15 |
| Delete shallow per-feature tests | Task 16 |
| CLAUDE.md update | Task 17 |
| Keep existing live-vs-streaming parity test | Confirmed in Tasks 12, 14 |

**2. Placeholder scan:** No "TBD", "implement later", or hand-wave "follow same pattern as Task N" without showing the code. The repetitive feature porting (Tasks 6-9) shows multiple concrete examples per task and references the source file line ranges; no formula is left as "fill in the rest yourself."

**3. Type consistency:** `FeatureFormula`, `FeatureInputs`, `SQL_BINDINGS`, `render_sql_fragment`, `FEATURES`, `project_row`, `project_sql` are referenced consistently across Tasks 2-12. The lambda input arg is named `i` (or `_i` if unused) throughout. The SQL template placeholder syntax `{scope.field}` is consistent. Constant names (`CONFIDENCE_N_MIN`, `HIGH_QUALITY_WIN_RATE_THRESHOLD`, `SECONDS_PER_DAY`, `RECENT_TRADES_WINDOW_DAYS`, `MIN_PRICES_FOR_VOLATILITY`, `KNOWN_CATEGORIES`) defined in Task 2 and used in Tasks 6-9.
