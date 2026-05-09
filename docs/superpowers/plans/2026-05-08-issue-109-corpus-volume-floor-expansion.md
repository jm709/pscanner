# Issue #109 — Corpus Volume Floor Expansion (Per-Category) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the corpus enumerator's volume floor into alignment with the live daemon's `gate_model_market_filter.min_volume_24h_usd = 100_000`. Replace the single `VOLUME_GATE_USD = 1_000_000` constant with a per-category map (esports → $100K lifetime, default → $1M), and add a per-volume-bucket edge metric to `metrics.json` so post-retrain validation can tell whether the new sub-$1M esports cohort actually carries edge or contributes noise.

**Architecture:** Two surgical code changes plus one telemetry addition. (1) `pscanner.corpus.enumerator` — swap the constant for a `VOLUME_GATE_BY_CATEGORY_USD: Mapping[Category, float]` lookup with a `_DEFAULT_VOLUME_GATE_USD` fallback. (2) `pscanner.ml.metrics` — add `per_volume_bucket_edge_breakdown(...)` mirroring `per_decile_edge_breakdown`. (3) `pscanner.ml.training` + `pscanner.ml.streaming` — wire `total_volume_usd` through `TestSplit` (parallel SELECT against `corpus_markets`) so `evaluate_on_test` can compute the new metric. The corpus DB schema is unchanged — the metric joins `training_examples` to `corpus_markets` at materialize time.

The operational steps (re-enumerate, subgraph-backfill, build-features --rebuild, ml train, validate) are document-only; they run on the desktop training box and are listed in the Operational Handoff section after the code tasks.

**Tech Stack:** Python 3.13, sqlite3, numpy, xgboost, pytest. No new deps.

---

## Scope Check

The issue has both code and operational components. The plan covers code changes only — they're a single coherent PR. The operational steps (long-running on the desktop) are listed in the Operational Handoff section so the operator has a checklist; they are NOT bite-sized tasks.

---

## File Structure

- Modify: `src/pscanner/corpus/enumerator.py:1-50` — replace `VOLUME_GATE_USD` constant with per-category mapping; update `_qualifying_markets`.
- Modify: `tests/corpus/test_enumerator.py` — update existing tests, add per-category test.
- Create: `src/pscanner/ml/metrics.py` (extend) — add `per_volume_bucket_edge_breakdown`.
- Create: `tests/ml/test_metrics_volume_bucket.py` — unit tests for the new function.
- Modify: `src/pscanner/ml/streaming.py:46-52` and `:134-184` — add `total_volume_usd` field to `TestSplit`; pull it via parallel SELECT in `materialize_test`.
- Modify: `src/pscanner/ml/training.py:196-251` (`evaluate_on_test`) — accept `total_volume_usd_test` and call the new breakdown.
- Modify: `src/pscanner/ml/training.py:422-446` — pass volume array; add `test_per_volume_bucket` to `metrics.json`.
- Modify: `CLAUDE.md` — append a note about the per-category gate and the new `metrics.json` field.

Tasks 1, 2, 3, 4 are sequential because each later task uses APIs introduced earlier. Task 5 (CLAUDE.md) is independent.

---

## Task 1: Per-category volume gate in enumerator

**Files:**
- Modify: `src/pscanner/corpus/enumerator.py:1-50`
- Test: `tests/corpus/test_enumerator.py` — update + add

The single `VOLUME_GATE_USD` constant becomes a mapping. Existing tests reference the constant by name; they break and need updating.

- [ ] **Step 1: Write a failing per-category test**

Append to `tests/corpus/test_enumerator.py` (after `test_enumerate_inserts_above_gate`):

```python
@pytest.mark.asyncio
async def test_enumerate_uses_per_category_gate(tmp_corpus_db: sqlite3.Connection) -> None:
    """Esports markets clear at $100K, but thesis markets at the same volume don't.

    Per issue #109: live daemon polls esports at $100K 24h; corpus must
    train on the same band to avoid OOD inference.
    """
    repo = CorpusMarketsRepo(tmp_corpus_db)
    esports_event = Event.model_validate(
        {
            "id": "ev-e",
            "title": "T",
            "slug": "ev-e",
            "markets": [_market("c-esports", 200_000.0).model_dump(by_alias=True)],
            "active": False,
            "closed": True,
            "tags": [{"label": "Esports"}],
        }
    )
    thesis_event = Event.model_validate(
        {
            "id": "ev-t",
            "title": "T",
            "slug": "ev-t",
            "markets": [_market("c-thesis", 200_000.0).model_dump(by_alias=True)],
            "active": False,
            "closed": True,
            "tags": [],
        }
    )
    inserted = await enumerate_closed_markets(
        gamma=_fake_gamma([esports_event, thesis_event]),
        repo=repo,
        now_ts=1_000,
        since_ts=None,
    )
    assert inserted == 1
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["c-esports"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/corpus/test_enumerator.py::test_enumerate_uses_per_category_gate -v`

Expected: FAIL — both events get filtered (or both pass) because the gate is uniform.

- [ ] **Step 3: Replace the constant with a mapping**

Edit `src/pscanner/corpus/enumerator.py`. Replace lines 1-50 (everything up to and including `_qualifying_markets`):

```python
"""Enumerate closed Polymarket markets above the corpus volume gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx
import structlog

from pscanner.categories import Category, categorize_event
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event

_log = structlog.get_logger(__name__)

_DEFAULT_VOLUME_GATE_USD: Final[float] = 1_000_000.0
"""Lifetime-volume floor for any category not in :data:`VOLUME_GATE_BY_CATEGORY_USD`.

Defaults remain at $1M (the historical corpus floor) so existing categories
are unaffected. New per-category overrides go in the mapping below.
"""

VOLUME_GATE_BY_CATEGORY_USD: Final[Mapping[Category, float]] = {
    Category.ESPORTS: 100_000.0,
}
"""Per-category lifetime-volume floors.

Esports drops to ``$100K`` to match the live daemon's
``gate_model_market_filter.min_volume_24h_usd = 100_000`` floor — the
previous $1M corpus floor put the live polling target out of distribution
relative to the training set (issue #109).

Categories not listed fall through to :data:`_DEFAULT_VOLUME_GATE_USD`.
"""

_HTTP_SERVER_ERROR_FLOOR: Final[int] = 500
# Polymarket's gamma `/events` uses 422 to signal a deep-offset overflow
# (mirroring the documented 400 cap on `/trades`). Some deployments
# return 500 instead. Both terminate enumeration cleanly with whatever
# pages succeeded.
_DEEP_OFFSET_STATUS: Final[int] = 422


def _volume_gate_for(category: Category) -> float:
    """Return the lifetime-volume floor for ``category``."""
    return VOLUME_GATE_BY_CATEGORY_USD.get(category, _DEFAULT_VOLUME_GATE_USD)


def _qualifying_markets(event: Event, now_ts: int) -> list[CorpusMarket]:
    """Return CorpusMarket rows for every market on ``event`` that qualifies."""
    if not event.closed:
        return []
    category = categorize_event(event)
    gate = _volume_gate_for(category)
    out: list[CorpusMarket] = []
    for market in event.markets:
        if not market.closed:
            continue
        volume = market.volume or 0.0
        if volume < gate:
            continue
        if market.condition_id is None:
            continue
        out.append(
            CorpusMarket(
                condition_id=str(market.condition_id),
                event_slug=event.slug,
                category=str(category),
                closed_at=now_ts,  # placeholder; mark_complete rewrites this to MAX(trade_ts) once backfill finishes  # noqa: E501
                total_volume_usd=volume,
                enumerated_at=now_ts,
                market_slug=market.slug,
            )
        )
    return out
```

- [ ] **Step 4: Update existing test imports + references**

The existing tests import and reference `VOLUME_GATE_USD` (a single number). Replace those with explicit numeric values that make the test's intent clear, and update the import.

Edit `tests/corpus/test_enumerator.py`. Replace the import (line 12-15):

```python
from pscanner.corpus.enumerator import (
    _DEFAULT_VOLUME_GATE_USD,
    enumerate_closed_markets,
)
```

Then in every test that currently uses `VOLUME_GATE_USD + 1` or `VOLUME_GATE_USD - 1`, swap to `_DEFAULT_VOLUME_GATE_USD + 1` / `_DEFAULT_VOLUME_GATE_USD - 1` (the existing tests construct events with no esports tags, so the default gate is the right one to test against). Use ripgrep to find them:

```bash
grep -n "VOLUME_GATE_USD" tests/corpus/test_enumerator.py
```

For each match, replace `VOLUME_GATE_USD` with `_DEFAULT_VOLUME_GATE_USD`.

- [ ] **Step 5: Run all enumerator tests to verify they pass**

Run: `uv run pytest tests/corpus/test_enumerator.py -v`

Expected: all pass (existing tests + the new per-category test).

- [ ] **Step 6: Lint + format + types**

Run:

```bash
uv run ruff check src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py
uv run ruff format --check src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py
uv run ty check src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py
git commit -m "feat(corpus): per-category volume gate (#109)"
```

---

## Task 2: Add `per_volume_bucket_edge_breakdown` to ml.metrics

**Files:**
- Modify: `src/pscanner/ml/metrics.py`
- Test: `tests/ml/test_metrics_volume_bucket.py` (create)

A pure-numpy diagnostic mirroring `per_decile_edge_breakdown`. Buckets total lifetime volume into log-spaced bins so the issue's central question — "do sub-$1M esports markets carry the same edge as $1M+?" — gets a direct answer.

Bucket boundaries (closed-open except top, which is closed-closed):
- `<$250K`
- `$250K-$1M`
- `$1M-$5M`
- `$5M-$25M`
- `≥$25M`

Five buckets gives enough resolution to see if the new sub-$1M cohort behaves differently from the existing $1M-$5M cohort, without overstratifying.

- [ ] **Step 1: Write failing tests**

Create `tests/ml/test_metrics_volume_bucket.py`:

```python
"""Unit tests for per_volume_bucket_edge_breakdown (#109)."""

from __future__ import annotations

import numpy as np

from pscanner.ml.metrics import per_volume_bucket_edge_breakdown


def test_buckets_emitted_only_when_taken_bets_present() -> None:
    """Buckets with no taken bets are omitted from the result."""
    y_true = np.array([1, 0])
    y_pred = np.array([0.6, 0.4])
    implied = np.array([0.5, 0.5])  # only first row has y_pred > implied
    volume = np.array([2_000_000.0, 50_000.0])

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    assert list(result.keys()) == ["1M-5M"]
    assert result["1M-5M"]["n"] == 1.0
    assert result["1M-5M"]["mean_edge"] == 0.5  # (1 - 0.5)


def test_volume_bucket_boundaries() -> None:
    """Boundary values land in the lower-bound-inclusive bucket."""
    # 5 rows, all taken (y_pred > implied), one in each bucket
    y_true = np.array([1, 1, 1, 1, 1])
    y_pred = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    implied = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    volume = np.array([
        100_000.0,     # <250K
        250_000.0,     # 250K-1M
        1_000_000.0,   # 1M-5M
        5_000_000.0,   # 5M-25M
        25_000_000.0,  # >=25M
    ])

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    assert sorted(result.keys()) == ["1M-5M", "25M+", "250K-1M", "5M-25M", "<250K"]
    for bucket in result.values():
        assert bucket["n"] == 1.0
        assert bucket["mean_edge"] == 0.5


def test_only_taken_bets_counted() -> None:
    """Bets where y_pred <= implied are excluded from the breakdown."""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0.6, 0.4, 0.6])
    implied = np.array([0.5, 0.5, 0.5])
    volume = np.array([2_000_000.0, 2_000_000.0, 2_000_000.0])

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    # Only rows 0 and 2 are taken (y_pred=0.6 > implied=0.5).
    assert result["1M-5M"]["n"] == 2.0


def test_empty_inputs_return_empty_dict() -> None:
    """No rows yields an empty dict, not an error."""
    y_true = np.array([], dtype=np.int32)
    y_pred = np.array([], dtype=np.float32)
    implied = np.array([], dtype=np.float32)
    volume = np.array([], dtype=np.float32)

    result = per_volume_bucket_edge_breakdown(y_true, y_pred, implied, volume)

    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_metrics_volume_bucket.py -v`

Expected: 4 FAILs — `ImportError: cannot import name 'per_volume_bucket_edge_breakdown'`.

- [ ] **Step 3: Implement the function**

Edit `src/pscanner/ml/metrics.py`. Append after `per_decile_edge_breakdown`:

```python
_VOLUME_BUCKETS_USD: tuple[tuple[str, float, float], ...] = (
    ("<250K", 0.0, 250_000.0),
    ("250K-1M", 250_000.0, 1_000_000.0),
    ("1M-5M", 1_000_000.0, 5_000_000.0),
    ("5M-25M", 5_000_000.0, 25_000_000.0),
    ("25M+", 25_000_000.0, float("inf")),
)


def per_volume_bucket_edge_breakdown(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
    total_volume_usd: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Stratify realized edge over taken bets by market lifetime volume.

    Diagnostic for issue #109. The corpus floor was lowered from $1M to
    $100K for esports; this metric tells us whether the newly-included
    sub-$1M esports cohort carries the same edge as the established $1M+
    cohort or contributes noise.

    Buckets are closed-open (``[lo, hi)``) except the top bucket which is
    closed-closed (``[lo, inf]``). Buckets with zero taken bets are omitted.

    Args:
        y_true: 1D array of binary labels.
        y_pred_proba: 1D array of model probabilities.
        implied_prob: 1D array of implied probabilities at trade time.
        total_volume_usd: 1D array of per-row market lifetime volume.

    Returns:
        Mapping from bucket label (e.g. ``"1M-5M"``) to
        ``{"n": <count>, "mean_edge": <mean_realized_edge>}``.
    """
    take = y_pred_proba > implied_prob
    out: dict[str, dict[str, float]] = {}
    for label, lo, hi in _VOLUME_BUCKETS_USD:
        if hi == float("inf"):
            in_bucket = total_volume_usd >= lo
        else:
            in_bucket = (total_volume_usd >= lo) & (total_volume_usd < hi)
        mask = take & in_bucket
        n = int(mask.sum())
        if n == 0:
            continue
        mean_edge = float((y_true[mask] - implied_prob[mask]).mean())
        out[label] = {"n": float(n), "mean_edge": mean_edge}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ml/test_metrics_volume_bucket.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Lint + format + types**

Run:

```bash
uv run ruff check src/pscanner/ml/metrics.py tests/ml/test_metrics_volume_bucket.py
uv run ruff format --check src/pscanner/ml/metrics.py tests/ml/test_metrics_volume_bucket.py
uv run ty check src/pscanner/ml/metrics.py tests/ml/test_metrics_volume_bucket.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/metrics.py tests/ml/test_metrics_volume_bucket.py
git commit -m "feat(ml): per_volume_bucket_edge_breakdown for issue #109"
```

---

## Task 3: Plumb `total_volume_usd` through `TestSplit` and `evaluate_on_test`

**Files:**
- Modify: `src/pscanner/ml/streaming.py:46-52` and `:134-184`
- Modify: `src/pscanner/ml/training.py:196-251`
- Test: existing test files cover the path; add a unit test for the new `evaluate_on_test` branch.

`TestSplit` already carries `top_categories` via a parallel SELECT pattern; `total_volume_usd` joins the same way against `corpus_markets` (key: `(platform, condition_id)`). Adding the field doesn't change any other consumer because `evaluate_on_test` is the only reader.

- [ ] **Step 1: Add the field to `TestSplit`**

Edit `src/pscanner/ml/streaming.py`. Locate `TestSplit` (around line 46-52). Add `total_volume_usd` alongside `top_categories`:

```python
@dataclass(frozen=True)
class TestSplit:
    """Materialized test split for ``evaluate_on_test``."""

    x: np.ndarray  # float32 (n, n_features)
    y: np.ndarray  # int32 (n,)
    implied_prob: np.ndarray  # float32 (n,)
    top_categories: np.ndarray  # object (str), unencoded — for per-category breakdowns
    total_volume_usd: np.ndarray  # float32 (n,) — for per-volume-bucket breakdowns (#109)
```

(The exact preceding declaration above `TestSplit` may differ — read lines 40-55 first to match the file's existing dataclass-decorator style.)

- [ ] **Step 2: Populate the field in `materialize_test`**

Edit `src/pscanner/ml/streaming.py:134-184`. Right after the `top_categories` parallel SELECT (current lines 169-182), add a parallel SELECT for `total_volume_usd` that JOINs `training_examples` to `corpus_markets`:

```python
        # Parallel small SELECT for total_volume_usd, JOINed to corpus_markets
        # via (platform, condition_id). Used by per_volume_bucket_edge_breakdown
        # (#109) to stratify the test edge by market lifetime volume.
        sql_volume = (
            "SELECT COALESCE(cm.total_volume_usd, 0.0) "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "LEFT JOIN corpus_markets cm "
            "  ON cm.condition_id = te.condition_id "
            " AND cm.platform = te.platform "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._test_markets)
            volume_rows = conn.execute(sql_volume, (self._platform,)).fetchall()
        finally:
            conn.close()
        total_volume_usd = np.array(
            [r[0] for r in volume_rows], dtype=np.float32
        )

        return TestSplit(
            x=x,
            y=y,
            implied_prob=implied,
            top_categories=top_categories,
            total_volume_usd=total_volume_usd,
        )
```

The existing `return TestSplit(...)` line gets replaced by the multi-line version above.

- [ ] **Step 3: Update existing TestSplit constructions in tests**

Run:

```bash
grep -rn "TestSplit(" tests/
```

For every match that constructs a `TestSplit` directly (test fixtures), add `total_volume_usd=np.zeros(n, dtype=np.float32)` (where `n` matches the existing `y`/`x` shape). The default zeros are fine for tests that don't exercise the new metric.

- [ ] **Step 4: Verify the streaming tests still pass**

Run: `uv run pytest tests/ml/ -v -k "streaming or materialize"`

Expected: all pass. If any fail with `TypeError: missing keyword argument 'total_volume_usd'`, fix the corresponding `TestSplit(...)` construction site.

- [ ] **Step 5: Add a unit test for `evaluate_on_test`'s new branch**

Append to `tests/ml/test_metrics_volume_bucket.py` (or create a `tests/ml/test_evaluate_on_test_volume.py` if you prefer per-function test files — match whatever `evaluate_on_test` already has for tests; check via `grep -rn "evaluate_on_test" tests/`):

```python
def test_evaluate_on_test_includes_per_volume_bucket() -> None:
    """`evaluate_on_test` returns ``per_volume_bucket`` when volume array is given."""
    import numpy as np
    import xgboost as xgb

    from pscanner.ml.training import evaluate_on_test

    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, size=(n, 1))
    y = (x[:, 0] > 0.5).astype(int)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "max_depth": 2,
            "tree_method": "hist",
            "verbosity": 0,
        },
        dtrain=xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    implied = np.full(n, 0.4, dtype=np.float32)
    volume = np.full(n, 2_000_000.0, dtype=np.float32)

    result = evaluate_on_test(
        booster=booster,
        X_test=x.astype(np.float32),
        y_test=y,
        implied_prob_test=implied,
        n_min=20,
        total_volume_usd_test=volume,
    )

    assert "per_volume_bucket" in result
    assert "1M-5M" in result["per_volume_bucket"]
```

- [ ] **Step 6: Update `evaluate_on_test` signature and behavior**

Edit `src/pscanner/ml/training.py:196-251`. Add `total_volume_usd_test: np.ndarray | None = None` as the last optional kwarg. Inside the function, after `decile = per_decile_edge_breakdown(...)`, add:

```python
    if total_volume_usd_test is not None:
        result["per_volume_bucket"] = per_volume_bucket_edge_breakdown(
            y_test, p_test, implied_prob_test, total_volume_usd_test
        )
```

The full updated signature + relevant body block:

```python
def evaluate_on_test(
    booster: xgb.Booster,
    X_test: np.ndarray,  # noqa: N803 -- ML matrix convention
    y_test: np.ndarray,
    implied_prob_test: np.ndarray,
    n_min: int,
    top_category_test: np.ndarray | None = None,
    accepted_categories: tuple[str, ...] | None = None,
    total_volume_usd_test: np.ndarray | None = None,
) -> dict[str, object]:
    """Score the booster on the held-out test split.

    Args:
        booster: Fitted XGBoost booster.
        X_test: Test feature matrix.
        y_test: Test labels.
        implied_prob_test: Implied probabilities per test row.
        n_min: Anti-overfit guard threshold for ``realized_edge_metric``.
        top_category_test: Optional string array (parallel to ``y_test``)
            of per-row ``top_category`` values. When provided together
            with ``accepted_categories``, an ``edge_filtered`` metric is
            computed over the accepted-category subset of taken bets.
        accepted_categories: Category strings to include in the filtered
            edge computation. Ignored when ``top_category_test`` is None.
        total_volume_usd_test: Optional float array (parallel to ``y_test``)
            of per-row market lifetime volume. When provided, a
            ``per_volume_bucket`` breakdown is added to the result so we
            can tell whether sub-$1M esports markets carry edge after
            the corpus floor expansion (issue #109).

    Returns:
        Dict with keys ``"edge"``, ``"accuracy"``, ``"logloss"``,
        ``"per_decile"``. When both ``top_category_test`` and
        ``accepted_categories`` are supplied, also includes
        ``"edge_filtered"``. When ``total_volume_usd_test`` is supplied,
        also includes ``"per_volume_bucket"``.
    """
    dtest = xgb.DMatrix(X_test)
    p_test = booster.predict(dtest)
    edge = realized_edge_metric(y_test, p_test, implied_prob_test, n_min=n_min)
    accuracy = float(((p_test >= _BINARY_DECISION_THRESHOLD).astype(int) == y_test).mean())
    eps = 1e-9
    logloss = float(
        -(y_test * np.log(p_test + eps) + (1 - y_test) * np.log(1 - p_test + eps)).mean()
    )
    decile = per_decile_edge_breakdown(y_test, p_test, implied_prob_test)
    result: dict[str, object] = {
        "edge": edge,
        "accuracy": accuracy,
        "logloss": logloss,
        "per_decile": decile,
    }
    if top_category_test is not None and accepted_categories is not None:
        cat_mask = np.isin(top_category_test, accepted_categories)
        result["edge_filtered"] = realized_edge_metric(
            y_test[cat_mask],
            p_test[cat_mask],
            implied_prob_test[cat_mask],
            n_min=n_min,
        )
    if total_volume_usd_test is not None:
        result["per_volume_bucket"] = per_volume_bucket_edge_breakdown(
            y_test, p_test, implied_prob_test, total_volume_usd_test
        )
    return result
```

Add the import at the top of `src/pscanner/ml/training.py` alongside the existing imports from `pscanner.ml.metrics`:

```python
from pscanner.ml.metrics import (
    per_decile_edge_breakdown,
    per_volume_bucket_edge_breakdown,
    realized_edge_metric,
)
```

(Verify the existing import shape via `grep -n "from pscanner.ml.metrics" src/pscanner/ml/training.py` and merge the new name into it.)

- [ ] **Step 7: Wire the new field at the call site in `_run_optimization_phase`**

Edit `src/pscanner/ml/training.py:422-446` (around the `evaluate_on_test(...)` call inside the main pipeline). Pass `total_volume_usd_test=test.total_volume_usd` and add the result to `metrics`:

```python
    test_metrics = evaluate_on_test(
        booster=booster,
        X_test=test.x,
        y_test=test.y,
        implied_prob_test=test.implied_prob,
        n_min=n_min,
        top_category_test=test.top_categories,
        accepted_categories=resolved_categories,
        total_volume_usd_test=test.total_volume_usd,
    )

    metrics: dict[str, object] = {
        "best_params": best_params,
        "best_iteration": best_iteration,
        "best_val_edge": best_value,
        "test_edge": test_metrics["edge"],
        "test_accuracy": test_metrics["accuracy"],
        "test_logloss": test_metrics["logloss"],
        "test_per_decile": test_metrics["per_decile"],
        "test_per_volume_bucket": test_metrics["per_volume_bucket"],
        "split_label_won_rate": rates,
        "seed": seed,
        "accepted_categories": list(resolved_categories),
    }
    if "edge_filtered" in test_metrics:
        metrics["test_edge_filtered"] = test_metrics["edge_filtered"]
```

- [ ] **Step 8: Run all touched test suites**

Run:

```bash
uv run pytest tests/ml/ tests/corpus/ -v
```

Expected: all pass.

- [ ] **Step 9: Lint + format + types**

Run:

```bash
uv run ruff check src/pscanner/ml/streaming.py src/pscanner/ml/training.py src/pscanner/ml/metrics.py tests/ml/
uv run ruff format --check src/pscanner/ml/streaming.py src/pscanner/ml/training.py src/pscanner/ml/metrics.py tests/ml/
uv run ty check src/pscanner/ml/streaming.py src/pscanner/ml/training.py
```

Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/pscanner/ml/streaming.py src/pscanner/ml/training.py tests/ml/
git commit -m "feat(ml): per_volume_bucket in metrics.json + TestSplit (#109)"
```

---

## Task 4: Document the per-category gate + new metric in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — corpus / training section

- [ ] **Step 1: Locate the corpus-volume bullet**

CLAUDE.md doesn't currently document the `VOLUME_GATE_USD` floor explicitly. The natural anchor is either the "Build features" bullet or the "Production baseline" bullet under `## ML training pipeline`. Search:

```bash
grep -n "VOLUME_GATE\|volume floor\|corpus floor\|enumerator" CLAUDE.md
```

If no anchor exists, add a new bullet under `## Codebase conventions` titled `**Corpus volume gates.**`.

- [ ] **Step 2: Append the documentation**

Add this bullet under `## Codebase conventions` (or merge into an adjacent bullet if one fits):

```
- **Corpus volume gates.** `pscanner.corpus.enumerator` applies a per-category lifetime-volume floor (`VOLUME_GATE_BY_CATEGORY_USD`): esports = $100K (matches the live daemon's `gate_model_market_filter.min_volume_24h_usd`), all other categories = $1M (`_DEFAULT_VOLUME_GATE_USD`). Lowering esports brought the corpus into alignment with the live polling target — see #109. The retrained metrics.json includes `test_per_volume_bucket` (5 log-spaced buckets from `<250K` to `25M+`) so post-rebuild validation can tell whether the new sub-$1M esports cohort carries the same edge as the existing $1M+ cohort.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): per-category corpus volume gates + per-volume-bucket metric (#109)"
```

---

## Verification

After all 4 tasks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: zero failures, zero new warnings.

---

## Self-Review

**Spec coverage:**
- Acceptance criterion 1 (`VOLUME_GATE_USD` replaced by per-category mapping): Task 1.
- Acceptance criterion 2 (esports floor at $100K): Task 1 (`Category.ESPORTS: 100_000.0`).
- Acceptance criterion 3 (re-enumeration completes; report counts): Operational handoff (operator runs).
- Acceptance criterion 4 (backfill completes for new markets): Operational handoff.
- Acceptance criterion 5 (`build-features --rebuild` produces expanded `training_examples`): Operational handoff.
- Acceptance criterion 6 (retrained model's `test_edge_filtered` ≥ 0.069): Operational handoff (validation step after retrain).
- Acceptance criterion 7 (per-volume-bucket edge breakdown in `metrics.json`): Tasks 2 + 3.

The plan covers every code-level acceptance criterion. The remaining acceptance criteria are operational and require running CLI commands on the desktop training box; they're listed below.

**Placeholder scan:** None. All code blocks are concrete; the only "find via grep" instruction is for locating an import line that the implementer will see immediately.

**Type consistency:**
- `Category` enum imported in enumerator.py (Task 1) — verified via the existing `categorize_event` import.
- `Mapping[Category, float]` for the gate dict — covariant in value, matches the codebase's preference per CLAUDE.md.
- `total_volume_usd: np.ndarray` on `TestSplit` (Task 3) is `float32`, matching how the SQL coerces via `dtype=np.float32`.
- `total_volume_usd_test: np.ndarray | None = None` is consistent with the existing `top_category_test: np.ndarray | None = None` pattern in `evaluate_on_test`.
- `per_volume_bucket_edge_breakdown(y_true, y_pred_proba, implied_prob, total_volume_usd)` signature matches the existing `per_decile_edge_breakdown` triple plus the new volume parameter.

---

## Operational Handoff

Once Tasks 1-4 land on `main` and the new model artifact is wanted on the desktop training box (`desktop-htrj0nn`, RTX 3070 — see memory `reference_desktop_training_box.md`), run these in sequence. None of these are agent tasks; they're operator commands.

**Pre-flight: capture the baseline numbers** (so the post-retrain validation has something to compare against):

```bash
sqlite3 data/corpus.sqlite3 \
  "SELECT category, COUNT(*) AS n, MIN(total_volume_usd) AS min_v, MAX(total_volume_usd) AS max_v
   FROM corpus_markets
   GROUP BY category;"
```

Save the output to a file for the PR description.

### Step 1: Re-enumerate

```bash
uv run pscanner corpus enumerate
```

Estimated time: ~1-5 minutes (gamma `/events` walk; cheap).

After it finishes, capture the new per-category counts (same SQL as the baseline) and diff. Expected: esports row count grows; other categories unchanged.

### Step 2: Subgraph backfill

Newly-enumerated esports markets are in `pending` state and need their trade history fetched.

```bash
export GRAPH_API_KEY=<your_key>
uv run pscanner corpus subgraph-backfill
```

Estimated time: ~1-2 hours per the issue (depends on count). The `--limit` flag can chunk the work if needed.

### Step 3: Build features

```bash
uv run pscanner corpus build-features --rebuild
```

Estimated time: ~9-10 hours on the desktop (per CLAUDE.md). The `#58` follow-ups are NOT a prerequisite — accept the long runtime for v1.

After it finishes:

```bash
sqlite3 data/corpus.sqlite3 \
  "SELECT platform, COUNT(*) AS n FROM training_examples GROUP BY platform;"
```

Compare to the baseline to confirm the row count grew.

### Step 4: Retrain

```bash
uv run pscanner ml train --device cuda --n-jobs 1 --n-trials 100
```

Estimated time: ~42 minutes on the desktop (per CLAUDE.md production-baseline note).

### Step 5: Validate

Open `models/<run_id>/metrics.json` and confirm:

- `test_edge_filtered ≥ 0.069` (the 2026-05-08 baseline, per CLAUDE.md). If lower, document the regression in the PR description and decide whether to ship anyway or revert.
- `test_per_volume_bucket["<250K"]` and `test_per_volume_bucket["250K-1M"]` exist and have meaningful `n` (≥ several thousand). If `n` is small, the new cohort isn't large enough to justify the corpus expansion — note this in the PR.
- The mean edges across volume buckets shouldn't differ by more than a factor of 2-3. If sub-$1M edge is much lower than $1M+, that signals the new cohort contributes noise; consider raising the esports floor to $250K or $500K instead of $100K.

### Step 6: Point the daemon at the new artifact

Update the daemon's `[gate_model] artifact_dir` to point at the new `models/<run_id>` directory. Smoke-run for one polling cycle to confirm the model loads and emits `gate_model.loaded` with the new model_version hash.

---

## Risks (per the issue, restated for the operator)

- **Backfill volume.** ~1-2 hours.
- **Build-features rebuild.** ~9-10 hours. Acceptable for v1.
- **Distribution shift.** `last_trade_price` and `price_volatility_recent` may behave differently in sub-$1M markets — the `test_per_volume_bucket` metric is the diagnostic for this.
- **Wallet feature quality.** Sub-$1M markets attract more retail; `is_high_quality_wallet` frequency may drop. Worth a sanity check on the new feature distribution after build-features completes:

```bash
sqlite3 data/corpus.sqlite3 \
  "SELECT
     CASE
       WHEN cm.total_volume_usd < 1000000 THEN 'sub-1M'
       ELSE '1M+'
     END AS volume_bucket,
     AVG(te.is_high_quality_wallet) AS frac_hq
   FROM training_examples te
   JOIN corpus_markets cm USING (platform, condition_id)
   GROUP BY volume_bucket;"
```

If `sub-1M` `frac_hq` is dramatically lower than `1M+` (e.g. 0.05 vs 0.20), that's expected and the model will learn to weight the indicator accordingly. If it's near-zero, the wallet-quality features are effectively absent in the new cohort and the per-bucket edge metric will tell us whether that costs us alpha.
