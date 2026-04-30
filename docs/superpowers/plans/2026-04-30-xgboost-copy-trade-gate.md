# XGBoost Copy-Trade Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `pscanner.ml` package that trains an XGBoost classifier from `training_examples`, optimizing realized edge per copied bet on a temporal validation split, evaluates once on a held-out test split, and persists model + preprocessor + study + metrics to `models/<run-name>/`.

**Architecture:** New `pscanner.ml` package parallel to `pscanner.corpus`. Reads `data/corpus.sqlite3` via stdlib sqlite3 → Polars DataFrame, applies leakage-column drops, fits a one-hot encoder on the train split, runs an Optuna study (TPE + median pruner, outer parallelism with `nthread=1` per trial) optimizing a custom realized-edge metric, refits the winning hyperparameters on train alone for `best_iteration` rounds, evaluates on test, dumps artifacts. Wired into the existing argparse-based `pscanner` CLI as `pscanner ml train`.

**Tech Stack:** Python 3.13, Polars (DataFrame I/O), XGBoost (model), Optuna with SQLite storage (search), numpy (feature matrices), sqlite3 (stdlib, source DB), argparse (CLI). Test stack: pytest, in-memory SQLite, synthetic Polars DataFrames.

**Spec:** `docs/superpowers/specs/2026-04-30-xgboost-copy-trade-gate-design.md`

---

## File Structure

**Created:**
- `src/pscanner/ml/__init__.py` — empty package marker
- `src/pscanner/ml/preprocessing.py` — column constants, `drop_leakage_cols`, `OneHotEncoder`, `temporal_split`, `load_dataset`
- `src/pscanner/ml/metrics.py` — `realized_edge_metric`, `per_decile_edge_breakdown`
- `src/pscanner/ml/training.py` — `run_single_trial`, `fit_winning_model`, `evaluate_on_test`, `run_study`, `dump_artifacts`
- `src/pscanner/ml/cli.py` — argparse parser, `run_ml_command(argv)` dispatcher
- `tests/ml/__init__.py` — empty package marker
- `tests/ml/conftest.py` — `make_synthetic_examples` fixture
- `tests/ml/test_metrics.py`
- `tests/ml/test_preprocessing.py`
- `tests/ml/test_training.py`
- `tests/ml/test_cli.py`

**Modified:**
- `pyproject.toml` — add `polars`, `xgboost`, `optuna` to `dependencies`
- `src/pscanner/cli.py` — register the `ml` subparser; dispatch to `run_ml_command`

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the three deps with `uv add`**

This resolves the latest stable line of each package against Python 3.13 and pins exact versions in `pyproject.toml` + `uv.lock`. Existing deps in this project use `==` exact pins (see lines 10-15 of `pyproject.toml`); `uv add` defaults to `>=` so we'll convert to `==` after.

Run:
```bash
uv add polars xgboost optuna
```
Expected: lockfile updates; the three new packages install. No errors.

- [ ] **Step 2: Convert the new deps to exact pins**

Open `pyproject.toml` and find the freshly-added entries near the bottom of the `dependencies` list. They will look like `"polars>=1.x.x"`. Replace each `>=` with `==` and the resolved version stays the same.

Use the version strings already present in `uv.lock` for the three new packages — find them via:
```bash
grep -E "^name = \"(polars|xgboost|optuna)\"" -A 1 uv.lock | head -12
```

Edit `pyproject.toml` so the three new lines look like:
```toml
    "polars==<RESOLVED_VERSION>",
    "xgboost==<RESOLVED_VERSION>",
    "optuna==<RESOLVED_VERSION>",
```

- [ ] **Step 3: Re-lock with the exact pins**

Run: `uv lock && uv sync`
Expected: lockfile is consistent; no version churn.

- [ ] **Step 4: Smoke-import each package**

Run: `uv run python -c "import polars, xgboost, optuna; print(polars.__version__, xgboost.__version__, optuna.__version__)"`
Expected: three version strings print. No errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(ml): add polars, xgboost, optuna deps for training pipeline"
```

---

## Task 2: Scaffold `pscanner.ml` Package

**Files:**
- Create: `src/pscanner/ml/__init__.py`
- Create: `tests/ml/__init__.py`

- [ ] **Step 1: Create the package marker**

Create `src/pscanner/ml/__init__.py`:

```python
"""ML training pipeline for the copy-trade gate model.

Consumes ``training_examples`` from ``data/corpus.sqlite3`` and produces
a versioned XGBoost model artifact. Inference is out of scope for v1.
"""
```

- [ ] **Step 2: Create the test package marker**

Create `tests/ml/__init__.py` as an empty file.

- [ ] **Step 3: Verify ruff and ty pass on the new package**

Run: `uv run ruff check src/pscanner/ml tests/ml && uv run ty check src/pscanner/ml`
Expected: no warnings or errors.

- [ ] **Step 4: Commit**

```bash
git add src/pscanner/ml/__init__.py tests/ml/__init__.py
git commit -m "chore(ml): scaffold pscanner.ml and tests/ml package markers"
```

---

## Task 3: Synthetic Training-Examples Fixture

**Files:**
- Create: `tests/ml/conftest.py`

- [ ] **Step 1: Create the fixture**

Create `tests/ml/conftest.py`:

```python
"""Shared test fixtures for the ml suite.

``make_synthetic_examples`` builds a Polars DataFrame whose schema
matches a join of ``training_examples`` + ``market_resolutions.resolved_at``.
Used by preprocessing, training, and CLI tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

_BASE_RESOLVED_AT = 1_700_000_000
_DAY_SECONDS = 86_400


def _make_synthetic_examples(
    n_markets: int = 30,
    rows_per_market: int = 20,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a synthetic ``training_examples`` DataFrame for tests.

    Markets resolve at evenly spaced timestamps over ~60 days so the
    temporal split has clean cutoffs. ``label_won`` is computed
    consistently with ``side`` and the market's outcome to keep the
    edge metric meaningful.

    Args:
        n_markets: Distinct ``condition_id`` values to generate.
        rows_per_market: BUYs per market.
        seed: Numpy RNG seed for reproducibility.

    Returns:
        Polars DataFrame with all 34 ``training_examples`` columns plus
        ``resolved_at`` (joined from ``market_resolutions``).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for m_idx in range(n_markets):
        cond_id = f"0xmarket{m_idx:03d}"
        resolved_at = _BASE_RESOLVED_AT + (m_idx * 60 * _DAY_SECONDS // n_markets)
        outcome_yes = bool(rng.integers(0, 2))
        for i in range(rows_per_market):
            implied_prob = float(rng.uniform(0.1, 0.95))
            side = "YES" if rng.random() < 0.6 else "NO"
            won_yes_market = side == "YES"
            won = won_yes_market if outcome_yes else (side == "NO")
            top_cat_choice = rng.choice(["sports", "esports", "thesis", None])
            top_cat = None if top_cat_choice is None else str(top_cat_choice)
            rows.append(
                {
                    "tx_hash": f"0xtx{m_idx:03d}{i:02d}",
                    "asset_id": f"asset{m_idx}{i}",
                    "wallet_address": f"0xwallet{int(rng.integers(0, 50)):03d}",
                    "condition_id": cond_id,
                    "trade_ts": resolved_at - int(rng.integers(_DAY_SECONDS, _DAY_SECONDS * 30)),
                    "built_at": _BASE_RESOLVED_AT,
                    "prior_trades_count": int(rng.integers(0, 100)),
                    "prior_buys_count": int(rng.integers(0, 80)),
                    "prior_resolved_buys": int(rng.integers(0, 50)),
                    "prior_wins": int(rng.integers(0, 30)),
                    "prior_losses": int(rng.integers(0, 20)),
                    "win_rate": (float(rng.uniform(0, 1)) if rng.random() < 0.8 else None),
                    "avg_implied_prob_paid": (
                        float(rng.uniform(0.3, 0.9)) if rng.random() < 0.8 else None
                    ),
                    "realized_edge_pp": (
                        float(rng.uniform(-0.2, 0.3)) if rng.random() < 0.6 else None
                    ),
                    "prior_realized_pnl_usd": float(rng.normal(0, 1000)),
                    "avg_bet_size_usd": (
                        float(rng.uniform(20, 1000)) if rng.random() < 0.8 else None
                    ),
                    "median_bet_size_usd": (
                        float(rng.uniform(20, 800)) if rng.random() < 0.8 else None
                    ),
                    "wallet_age_days": float(rng.uniform(0, 365)),
                    "seconds_since_last_trade": (
                        int(rng.integers(0, _DAY_SECONDS)) if rng.random() < 0.9 else None
                    ),
                    "prior_trades_30d": int(rng.integers(0, 30)),
                    "top_category": top_cat,
                    "category_diversity": int(rng.integers(0, 4)),
                    "bet_size_usd": float(rng.uniform(10, 500)),
                    "bet_size_rel_to_avg": (
                        float(rng.uniform(0.5, 3.0)) if rng.random() < 0.8 else None
                    ),
                    "side": side,
                    "implied_prob_at_buy": implied_prob,
                    "market_category": str(
                        rng.choice(["sports", "esports", "thesis", "unknown"])
                    ),
                    "market_volume_so_far_usd": float(rng.uniform(1000, 1e6)),
                    "market_unique_traders_so_far": int(rng.integers(1, 500)),
                    "market_age_seconds": int(rng.integers(60, _DAY_SECONDS * 30)),
                    "time_to_resolution_seconds": int(rng.integers(60, _DAY_SECONDS * 30)),
                    "last_trade_price": (
                        float(rng.uniform(0.05, 0.95)) if rng.random() < 0.95 else None
                    ),
                    "price_volatility_recent": (
                        float(rng.uniform(0, 0.1)) if rng.random() < 0.95 else None
                    ),
                    "label_won": int(won),
                    "resolved_at": resolved_at,
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def make_synthetic_examples() -> Callable[..., pl.DataFrame]:
    """Return the synthetic-examples builder."""
    return _make_synthetic_examples
```

- [ ] **Step 2: Smoke-test the fixture in a one-liner**

Create a temporary file `tests/ml/test_smoke_fixture.py`:

```python
"""Smoke test for the synthetic-examples fixture."""

from collections.abc import Callable

import polars as pl


def test_fixture_produces_dataframe(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=5, rows_per_market=3)
    assert df.shape == (15, 35)
    assert df["condition_id"].n_unique() == 5
    assert set(df["side"].unique().to_list()).issubset({"YES", "NO"})
```

Run: `uv run pytest tests/ml/test_smoke_fixture.py -v`
Expected: PASS.

- [ ] **Step 3: Delete the smoke test (kept the fixture; smoke is no longer needed)**

```bash
rm tests/ml/test_smoke_fixture.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/ml/conftest.py
git commit -m "test(ml): add synthetic training_examples fixture"
```

---

## Task 4: `realized_edge_metric`

**Files:**
- Create: `src/pscanner/ml/metrics.py`
- Create: `tests/ml/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/ml/test_metrics.py`:

```python
"""Tests for ml.metrics."""

from __future__ import annotations

import numpy as np

from pscanner.ml.metrics import realized_edge_metric


def test_returns_minus_one_below_n_min() -> None:
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.9, 0.9])
    implied = np.array([0.5, 0.5, 0.5])
    assert realized_edge_metric(y, p, implied, n_min=10) == -1.0


def test_mean_realized_edge_over_taken_bets() -> None:
    # Three bets total. Take only the two where p > implied.
    y = np.array([1, 0, 1])
    p = np.array([0.7, 0.3, 0.8])
    implied = np.array([0.5, 0.5, 0.5])
    # Taken: indices 0 and 2. realized edges: (1 - 0.5)=0.5, (1 - 0.5)=0.5.
    assert realized_edge_metric(y, p, implied, n_min=2) == 0.5


def test_negative_edge_when_model_wrong() -> None:
    y = np.array([0, 0, 0])
    p = np.array([0.9, 0.9, 0.9])
    implied = np.array([0.5, 0.5, 0.5])
    # Take all three; realized edges all (0 - 0.5)=-0.5.
    assert realized_edge_metric(y, p, implied, n_min=3) == -0.5


def test_no_taken_bets_returns_minus_one() -> None:
    y = np.array([1, 1, 1])
    p = np.array([0.1, 0.2, 0.3])
    implied = np.array([0.5, 0.5, 0.5])
    # Nothing passes p > implied.
    assert realized_edge_metric(y, p, implied, n_min=1) == -1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pscanner.ml.metrics'`.

- [ ] **Step 3: Write the implementation**

Create `src/pscanner/ml/metrics.py`:

```python
"""Edge-based metrics for copy-trade gate model evaluation.

The optimization target is ``realized_edge_metric`` — mean realized
edge across bets the model would copy. ``per_decile_edge_breakdown``
is a diagnostic stratification by ``implied_prob_at_buy`` decile.
"""

from __future__ import annotations

import numpy as np


def realized_edge_metric(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
    n_min: int = 20,
) -> float:
    """Return mean realized edge over bets the model would copy.

    A bet is "copied" iff ``y_pred_proba > implied_prob`` (the model
    predicts positive expected edge). Realized edge per copied bet is
    ``label_won - implied_prob_at_buy``.

    Args:
        y_true: 1D array of binary labels (``label_won``).
        y_pred_proba: 1D array of model probabilities for ``label_won=1``.
        implied_prob: 1D array of implied probabilities at trade time.
        n_min: Anti-overfit guard. If fewer than ``n_min`` bets pass
            the gate, return ``-1.0`` so trial configurations that
            overfit to a tiny lucky subset are penalized.

    Returns:
        Mean realized edge over taken bets, or ``-1.0`` if too few.
    """
    take = y_pred_proba > implied_prob
    if int(take.sum()) < n_min:
        return -1.0
    return float((y_true[take] - implied_prob[take]).mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ml/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/metrics.py tests/ml/test_metrics.py && uv run ty check src/pscanner/ml/metrics.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/metrics.py tests/ml/test_metrics.py
git commit -m "feat(ml): realized_edge_metric with n_min guard"
```

---

## Task 5: `per_decile_edge_breakdown`

**Files:**
- Modify: `src/pscanner/ml/metrics.py`
- Modify: `tests/ml/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ml/test_metrics.py`:

```python
from pscanner.ml.metrics import per_decile_edge_breakdown


def test_per_decile_breakdown_groups_by_implied_prob() -> None:
    # 20 trades spread across two implied-prob deciles.
    # First 10: implied=0.05 (decile 0). Take all (p=0.9). 7 wins.
    # Next 10: implied=0.55 (decile 5). Take all (p=0.9). 9 wins.
    y = np.array([1] * 7 + [0] * 3 + [1] * 9 + [0])
    p = np.array([0.9] * 20)
    implied = np.array([0.05] * 10 + [0.55] * 10)
    result = per_decile_edge_breakdown(y, p, implied)
    # Decile 0: 7/10 wins, mean edge = 0.7 - 0.05 = 0.65
    assert result["0.0-0.1"]["n"] == 10
    assert result["0.0-0.1"]["mean_edge"] == 0.65
    # Decile 5: 9/10 wins, mean edge = 0.9 - 0.55 = 0.35
    assert result["0.5-0.6"]["n"] == 10
    assert abs(result["0.5-0.6"]["mean_edge"] - 0.35) < 1e-9


def test_per_decile_skips_empty_deciles() -> None:
    y = np.array([1, 1])
    p = np.array([0.9, 0.9])
    implied = np.array([0.05, 0.05])
    result = per_decile_edge_breakdown(y, p, implied)
    assert "0.0-0.1" in result
    assert "0.5-0.6" not in result


def test_per_decile_only_counts_taken_bets() -> None:
    # First bet not taken (p < implied), second taken.
    y = np.array([1, 1])
    p = np.array([0.1, 0.9])
    implied = np.array([0.05, 0.05])
    result = per_decile_edge_breakdown(y, p, implied)
    # Both in decile 0, but only second is taken.
    assert result["0.0-0.1"]["n"] == 1
    assert abs(result["0.0-0.1"]["mean_edge"] - (1 - 0.05)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_metrics.py -v -k per_decile`
Expected: 3 failures with `ImportError: cannot import name 'per_decile_edge_breakdown'`.

- [ ] **Step 3: Implement `per_decile_edge_breakdown`**

Append to `src/pscanner/ml/metrics.py`:

```python
def per_decile_edge_breakdown(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Stratify realized edge over taken bets by implied-prob decile.

    Diagnostic only — not the optimization target. Reveals whether
    edge is concentrated in cheap-side bets (longshot finder) or
    distributed across the implied-prob range (mispricing detector).

    Args:
        y_true: 1D array of binary labels.
        y_pred_proba: 1D array of model probabilities.
        implied_prob: 1D array of implied probabilities at trade time.

    Returns:
        Mapping from decile label (e.g. ``"0.0-0.1"``) to
        ``{"n": <count>, "mean_edge": <mean_realized_edge>}``. Deciles
        with zero taken bets are omitted.
    """
    take = y_pred_proba > implied_prob
    out: dict[str, dict[str, float]] = {}
    for decile in range(10):
        lo = decile / 10
        hi = (decile + 1) / 10
        in_decile = (implied_prob >= lo) & (implied_prob < hi)
        mask = take & in_decile
        n = int(mask.sum())
        if n == 0:
            continue
        mean_edge = float((y_true[mask] - implied_prob[mask]).mean())
        label = f"{lo:.1f}-{hi:.1f}"
        out[label] = {"n": float(n), "mean_edge": mean_edge}
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_metrics.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/metrics.py tests/ml/test_metrics.py && uv run ty check src/pscanner/ml/metrics.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/metrics.py tests/ml/test_metrics.py
git commit -m "feat(ml): per_decile_edge_breakdown diagnostic"
```

---

## Task 6: Column Constants and `drop_leakage_cols`

**Files:**
- Create: `src/pscanner/ml/preprocessing.py`
- Create: `tests/ml/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/ml/test_preprocessing.py`:

```python
"""Tests for ml.preprocessing."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from pscanner.ml.preprocessing import (
    CARRIER_COLS,
    CATEGORICAL_COLS,
    LEAKAGE_COLS,
    drop_leakage_cols,
)


def test_leakage_cols_lists_documented_drops() -> None:
    expected = {
        "tx_hash",
        "asset_id",
        "wallet_address",
        "built_at",
        "time_to_resolution_seconds",
    }
    assert set(LEAKAGE_COLS) == expected


def test_carrier_cols_lists_documented_carriers() -> None:
    assert set(CARRIER_COLS) == {"condition_id", "trade_ts", "resolved_at"}


def test_categorical_cols_lists_documented_categoricals() -> None:
    assert set(CATEGORICAL_COLS) == {"side", "top_category", "market_category"}


def test_drop_leakage_cols_removes_each_documented_col(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=3, rows_per_market=2)
    out = drop_leakage_cols(df)
    for col in LEAKAGE_COLS:
        assert col not in out.columns
    # Carrier cols must survive the drop.
    for col in CARRIER_COLS:
        assert col in out.columns
    # Categorical cols and label must survive.
    assert "label_won" in out.columns
    for col in CATEGORICAL_COLS:
        assert col in out.columns


def test_drop_leakage_cols_is_idempotent(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=3, rows_per_market=2)
    once = drop_leakage_cols(df)
    twice = drop_leakage_cols(once)
    assert once.columns == twice.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_preprocessing.py -v`
Expected: 5 failures with `ModuleNotFoundError: No module named 'pscanner.ml.preprocessing'`.

- [ ] **Step 3: Write the implementation**

Create `src/pscanner/ml/preprocessing.py`:

```python
"""Preprocessing for the copy-trade gate training pipeline.

Exposes:

* ``LEAKAGE_COLS`` / ``CARRIER_COLS`` / ``CATEGORICAL_COLS`` — column
  membership constants documented in the design spec.
* ``drop_leakage_cols`` — pure column removal.
* ``OneHotEncoder`` — fit-on-train, transform-each-split. Handles
  the ``__none__`` level for nullable categoricals.
* ``temporal_split`` — assigns each ``condition_id`` to one of
  ``{train, val, test}`` by ``resolved_at`` percentile.
* ``load_dataset`` — sqlite3 → Polars DataFrame, joining
  ``training_examples`` with ``market_resolutions.resolved_at``.
"""

from __future__ import annotations

import polars as pl

LEAKAGE_COLS: tuple[str, ...] = (
    "tx_hash",
    "asset_id",
    "wallet_address",
    "built_at",
    "time_to_resolution_seconds",
)

CARRIER_COLS: tuple[str, ...] = ("condition_id", "trade_ts", "resolved_at")

CATEGORICAL_COLS: tuple[str, ...] = ("side", "top_category", "market_category")


def drop_leakage_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that risk identity leakage or future-information leakage.

    See the design spec for per-column reasoning. Idempotent — drops only
    columns that exist on the input frame.
    """
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    return df.drop(to_drop)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_preprocessing.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py && uv run ty check src/pscanner/ml/preprocessing.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git commit -m "feat(ml): leakage column constants and drop helper"
```

---

## Task 7: `OneHotEncoder` — Fit and Transform

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py`
- Modify: `tests/ml/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ml/test_preprocessing.py`:

```python
import json

from pscanner.ml.preprocessing import OneHotEncoder


def test_one_hot_encoder_fits_on_train_levels() -> None:
    df = pl.DataFrame(
        {
            "side": ["YES", "NO", "YES"],
            "top_category": ["sports", None, "thesis"],
            "market_category": ["sports", "esports", "thesis"],
            "implied_prob_at_buy": [0.5, 0.5, 0.5],
            "label_won": [1, 0, 1],
        }
    )
    enc = OneHotEncoder.fit(df, columns=("side", "top_category", "market_category"))
    assert enc.levels["side"] == ("NO", "YES")
    assert enc.levels["top_category"] == ("__none__", "sports", "thesis")
    assert enc.levels["market_category"] == ("esports", "sports", "thesis")


def test_one_hot_encoder_transform_emits_indicator_columns() -> None:
    df = pl.DataFrame(
        {
            "side": ["YES", "NO", "YES"],
            "top_category": ["sports", None, "thesis"],
            "market_category": ["sports", "esports", "thesis"],
            "implied_prob_at_buy": [0.5, 0.5, 0.5],
            "label_won": [1, 0, 1],
        }
    )
    enc = OneHotEncoder.fit(df, columns=("side", "top_category", "market_category"))
    out = enc.transform(df)
    # Original categoricals dropped.
    for col in ("side", "top_category", "market_category"):
        assert col not in out.columns
    # New indicator columns present.
    assert "side__YES" in out.columns
    assert "side__NO" in out.columns
    assert "top_category____none__" in out.columns
    # Indicators carry correct values for the first row (YES, sports, sports).
    assert out["side__YES"][0] == 1
    assert out["side__NO"][0] == 0
    assert out["top_category__sports"][0] == 1
    assert out["top_category____none__"][0] == 0
    # Second row had top_category=None → __none__.
    assert out["top_category____none__"][1] == 1


def test_one_hot_encoder_handles_unseen_levels_at_transform() -> None:
    train = pl.DataFrame({"side": ["YES", "NO"]})
    val = pl.DataFrame({"side": ["YES", "DRAW"]})  # DRAW not seen at fit
    enc = OneHotEncoder.fit(train, columns=("side",))
    out = enc.transform(val)
    # Both fit-time levels exist on the output.
    assert "side__YES" in out.columns
    assert "side__NO" in out.columns
    # Unseen value gets all zeros across known levels.
    assert out["side__YES"][1] == 0
    assert out["side__NO"][1] == 0


def test_one_hot_encoder_round_trips_through_json() -> None:
    df = pl.DataFrame({"side": ["YES", "NO"], "top_category": ["sports", None]})
    enc = OneHotEncoder.fit(df, columns=("side", "top_category"))
    payload = enc.to_json()
    rendered = json.dumps(payload)
    parsed = json.loads(rendered)
    enc2 = OneHotEncoder.from_json(parsed)
    assert enc2.levels == enc.levels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_preprocessing.py -v -k one_hot`
Expected: 4 failures with `ImportError: cannot import name 'OneHotEncoder'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pscanner/ml/preprocessing.py`:

```python
from collections.abc import Iterable
from dataclasses import dataclass

_NONE_TOKEN = "__none__"


@dataclass(frozen=True)
class OneHotEncoder:
    """Fit-once, transform-many one-hot encoder.

    Only encodes the columns passed to ``fit``. At ``fit`` time, nulls
    are mapped to the explicit ``"__none__"`` level so first-time
    wallets become a learnable signal rather than a missing value.
    Levels are stored sorted for deterministic column ordering.
    """

    levels: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, df: pl.DataFrame, columns: Iterable[str]) -> OneHotEncoder:
        """Discover the level set per column on a (training) DataFrame."""
        levels: dict[str, tuple[str, ...]] = {}
        for col in columns:
            uniq = (
                df.select(pl.col(col).fill_null(_NONE_TOKEN))
                .to_series()
                .unique()
                .sort()
                .to_list()
            )
            levels[col] = tuple(str(v) for v in uniq)
        return cls(levels=levels)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace each fit column with one ``{col}__{level}`` int8 indicator
        per known level, dropping the original column.

        Unseen levels at transform time are silently mapped to
        all-zeros across the known indicators.
        """
        out = df
        for col, lvls in self.levels.items():
            if col not in out.columns:
                continue
            filled = out.with_columns(pl.col(col).fill_null(_NONE_TOKEN).alias(col))
            indicator_exprs = [
                (pl.col(col) == lvl).cast(pl.Int8).alias(f"{col}__{lvl}") for lvl in lvls
            ]
            out = filled.with_columns(indicator_exprs).drop(col)
        return out

    def to_json(self) -> dict[str, dict[str, list[str]]]:
        """Serialise level state to a JSON-safe dict."""
        return {"levels": {k: list(v) for k, v in self.levels.items()}}

    @classmethod
    def from_json(cls, payload: dict[str, dict[str, list[str]]]) -> OneHotEncoder:
        """Rebuild an encoder from ``to_json`` output."""
        return cls(levels={k: tuple(v) for k, v in payload["levels"].items()})
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_preprocessing.py -v`
Expected: 9 passed (5 from previous task + 4 new).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py && uv run ty check src/pscanner/ml/preprocessing.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git commit -m "feat(ml): OneHotEncoder for low-cardinality categoricals"
```

---

## Task 8: `temporal_split`

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py`
- Modify: `tests/ml/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ml/test_preprocessing.py`:

```python
from collections.abc import Callable

from pscanner.ml.preprocessing import Split, temporal_split


def test_temporal_split_partitions_by_resolved_at_percentiles(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    assert isinstance(split, Split)
    # All rows must be assigned exactly once.
    total = split.train.height + split.val.height + split.test.height
    assert total == df.height


def test_temporal_split_no_market_in_two_splits(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    train_markets = set(split.train["condition_id"].unique().to_list())
    val_markets = set(split.val["condition_id"].unique().to_list())
    test_markets = set(split.test["condition_id"].unique().to_list())
    assert train_markets.isdisjoint(val_markets)
    assert train_markets.isdisjoint(test_markets)
    assert val_markets.isdisjoint(test_markets)


def test_temporal_split_train_precedes_val_precedes_test(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    train_max = split.train["resolved_at"].max()
    val_min = split.val["resolved_at"].min()
    val_max = split.val["resolved_at"].max()
    test_min = split.test["resolved_at"].min()
    assert train_max <= val_min
    assert val_max <= test_min


def test_temporal_split_60_20_20_proportion(
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=30, rows_per_market=10)
    split = temporal_split(df, train_frac=0.6, val_frac=0.2)
    # 30 markets at 60/20/20 → 18 / 6 / 6 markets.
    assert split.train["condition_id"].n_unique() == 18
    assert split.val["condition_id"].n_unique() == 6
    assert split.test["condition_id"].n_unique() == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_preprocessing.py -v -k temporal_split`
Expected: 4 failures with `ImportError: cannot import name 'temporal_split'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pscanner/ml/preprocessing.py`:

```python
@dataclass(frozen=True)
class Split:
    """Three-way temporal split of a training-examples DataFrame."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def temporal_split(
    df: pl.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Split:
    """Split rows into train/val/test by ``resolved_at`` market percentiles.

    Each ``condition_id`` lands in exactly one split; trades for a market
    cannot leak across splits. The split key is the market's
    ``resolved_at``, sorted ascending. Tie-break on ``condition_id``
    lexically for a stable order.

    Args:
        df: Polars DataFrame with at least ``condition_id`` and
            ``resolved_at`` columns.
        train_frac: Fraction of distinct markets in train.
        val_frac: Fraction of distinct markets in val. ``test_frac`` is
            ``1 - train_frac - val_frac``.

    Returns:
        A ``Split`` with three disjoint DataFrames.
    """
    markets = (
        df.select(["condition_id", "resolved_at"])
        .unique()
        .sort(["resolved_at", "condition_id"])
    )
    n = markets.height
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_ids = set(markets["condition_id"].slice(0, n_train).to_list())
    val_ids = set(markets["condition_id"].slice(n_train, n_val).to_list())
    test_ids = set(markets["condition_id"].slice(n_train + n_val, n - n_train - n_val).to_list())
    return Split(
        train=df.filter(pl.col("condition_id").is_in(train_ids)),
        val=df.filter(pl.col("condition_id").is_in(val_ids)),
        test=df.filter(pl.col("condition_id").is_in(test_ids)),
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_preprocessing.py -v`
Expected: 13 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py && uv run ty check src/pscanner/ml/preprocessing.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git commit -m "feat(ml): temporal_split by market resolved_at percentiles"
```

---

## Task 9: `load_dataset` from SQLite

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py`
- Modify: `tests/ml/test_preprocessing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_preprocessing.py`:

```python
import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.ml.preprocessing import load_dataset


def _seed_db_from_synthetic(
    conn: sqlite3.Connection,
    df: pl.DataFrame,
) -> None:
    # Populate corpus_markets, market_resolutions, training_examples
    # from a synthetic Polars frame so load_dataset has matching rows.
    markets = df.select(["condition_id", "resolved_at"]).unique()
    for row in markets.iter_rows(named=True):
        conn.execute(
            """
            INSERT INTO corpus_markets (
              condition_id, event_slug, category, closed_at,
              total_volume_usd, market_slug, backfill_state, enumerated_at
            ) VALUES (?, '', 'sports', ?, 1000.0, '', 'complete', ?)
            """,
            (row["condition_id"], int(row["resolved_at"]), int(row["resolved_at"]) - 1),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions (
              condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, 0, 1, ?, 'gamma', ?)
            """,
            (row["condition_id"], int(row["resolved_at"]), int(row["resolved_at"])),
        )
    # Drop resolved_at from the example rows (it lives on market_resolutions).
    examples = df.drop("resolved_at")
    for row in examples.iter_rows(named=True):
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO training_examples ({cols}) VALUES ({placeholders})",
            tuple(row.values()),
        )
    conn.commit()


def test_load_dataset_joins_resolved_at(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    try:
        synthetic = make_synthetic_examples(n_markets=4, rows_per_market=3)
        _seed_db_from_synthetic(conn, synthetic)
    finally:
        conn.close()
    out = load_dataset(db_path)
    assert out.height == 12
    assert "resolved_at" in out.columns
    assert "label_won" in out.columns
    # Inner join: every row has a resolved_at.
    assert out["resolved_at"].null_count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_preprocessing.py::test_load_dataset_joins_resolved_at -v`
Expected: FAIL with `ImportError: cannot import name 'load_dataset'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pscanner/ml/preprocessing.py`:

```python
import sqlite3
from pathlib import Path


def load_dataset(db_path: Path) -> pl.DataFrame:
    """Load ``training_examples`` joined with ``market_resolutions.resolved_at``.

    Inner join is correct: ``build-features`` only emits rows for
    markets with a resolutions entry, so the join is row-preserving.

    Args:
        db_path: Path to the corpus SQLite file.

    Returns:
        A Polars DataFrame with all 34 ``training_examples`` columns
        plus ``resolved_at``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT te.*, mr.resolved_at
            FROM training_examples te
            JOIN market_resolutions mr USING (condition_id)
            """
        )
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()
    return pl.DataFrame(rows, schema=cols, orient="row")
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/ml/test_preprocessing.py::test_load_dataset_joins_resolved_at -v`
Expected: PASS.

- [ ] **Step 5: Run the entire test suite**

Run: `uv run pytest tests/ml -v`
Expected: 14 passed.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml tests/ml && uv run ty check src/pscanner/ml`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git commit -m "feat(ml): load_dataset joins training_examples with resolved_at"
```

---

## Task 10: `_build_feature_matrix` Helper

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py`
- Modify: `tests/ml/test_preprocessing.py`

This is the bridge from a Polars DataFrame (post-encoding) to numpy arrays for XGBoost. Carrier columns and the label are split out; everything else becomes the feature matrix.

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_preprocessing.py`:

```python
import numpy as np

from pscanner.ml.preprocessing import build_feature_matrix


def test_build_feature_matrix_extracts_arrays() -> None:
    df = pl.DataFrame(
        {
            "condition_id": ["a", "b"],
            "trade_ts": [1, 2],
            "resolved_at": [10, 20],
            "implied_prob_at_buy": [0.4, 0.7],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            "label_won": [1, 0],
        }
    )
    X, y, implied = build_feature_matrix(df)
    assert X.shape == (2, 3)  # implied_prob_at_buy, feature_a, feature_b
    assert y.tolist() == [1, 0]
    assert implied.tolist() == [0.4, 0.7]


def test_build_feature_matrix_preserves_nan() -> None:
    df = pl.DataFrame(
        {
            "condition_id": ["a"],
            "trade_ts": [1],
            "resolved_at": [10],
            "implied_prob_at_buy": [0.5],
            "win_rate": [None],
            "label_won": [1],
        }
    )
    X, _, _ = build_feature_matrix(df)
    # Polars null → numpy nan.
    assert np.isnan(X[0, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_preprocessing.py -v -k build_feature_matrix`
Expected: 2 failures with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `src/pscanner/ml/preprocessing.py`:

```python
import numpy as np


def build_feature_matrix(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(X, y, implied_prob)`` numpy arrays from a preprocessed split.

    Drops carrier columns and the ``label_won`` column from ``X``. The
    feature column ordering is the surviving column order on ``df``.
    Polars nulls become ``np.nan`` in float columns — XGBoost's
    missing-direction rule handles them at split time.

    Args:
        df: A preprocessed Polars DataFrame (post-drop, post-encoding).

    Returns:
        ``(X, y, implied_prob)`` tuple.
    """
    feature_cols = [
        c for c in df.columns if c not in (*CARRIER_COLS, "label_won")
    ]
    X = df.select(feature_cols).to_numpy()
    y = df["label_won"].to_numpy()
    implied = df["implied_prob_at_buy"].to_numpy()
    return X, y, implied
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml -v`
Expected: 16 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml tests/ml && uv run ty check src/pscanner/ml`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git commit -m "feat(ml): build_feature_matrix extracts (X, y, implied) arrays"
```

---

## Task 11: `run_single_trial`

**Files:**
- Create: `src/pscanner/ml/training.py`
- Create: `tests/ml/test_training.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_training.py`:

```python
"""Tests for ml.training."""

from __future__ import annotations

import numpy as np
import optuna

from pscanner.ml.training import run_single_trial


def _toy_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 200
    # X is a single feature that is correlated with y.
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    implied = np.full(n, 0.5)  # All bets at 50% implied prob.
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    implied_val = implied[150:]
    return X_train, y_train, X_val, y_val, implied_val


def test_run_single_trial_returns_finite_edge() -> None:
    X_train, y_train, X_val, y_val, implied_val = _toy_problem()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        return run_single_trial(
            trial=trial,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            implied_prob_val=implied_val,
            n_min=5,
            seed=42,
        )

    study.optimize(objective, n_trials=2)
    assert len(study.trials) == 2
    # Each trial must record best_iteration as a user attr.
    for trial in study.trials:
        assert "best_iteration" in trial.user_attrs
        assert isinstance(trial.user_attrs["best_iteration"], int)


def test_run_single_trial_is_deterministic_under_same_seed() -> None:
    X_train, y_train, X_val, y_val, implied_val = _toy_problem()

    def study_value() -> float:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=7),
        )
        study.optimize(
            lambda t: run_single_trial(
                trial=t,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                implied_prob_val=implied_val,
                n_min=5,
                seed=7,
            ),
            n_trials=2,
        )
        return study.best_value

    assert study_value() == study_value()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pscanner.ml.training'`.

- [ ] **Step 3: Write the implementation**

Create `src/pscanner/ml/training.py`:

```python
"""Optuna-driven XGBoost training for the copy-trade gate model.

Single-trial fitting, study orchestration, winning-model refit, test
evaluation, and artifact dump. The optimization target is the custom
``realized_edge_metric``; ``binary:logistic`` keeps ``model_prob``
calibrated against ``implied_prob_at_buy``.
"""

from __future__ import annotations

import numpy as np
import optuna
import xgboost as xgb

from pscanner.ml.metrics import realized_edge_metric

_NUM_BOOST_ROUND = 2000
_EARLY_STOPPING_ROUNDS = 50


def run_single_trial(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    implied_prob_val: np.ndarray,
    n_min: int,
    seed: int,
) -> float:
    """Fit one XGBoost trial and return its validation realized edge.

    Sampled hyperparameters: ``learning_rate``, ``max_depth``,
    ``min_child_weight``, ``subsample``, ``colsample_bytree``,
    ``reg_alpha``, ``reg_lambda``, ``gamma``. Boosting rounds are
    capped at 2000 with 50-round early stopping on val log-loss; the
    actual rounds used for prediction is ``best_iteration + 1``. The
    chosen ``best_iteration`` is recorded on the trial's user attrs so
    the winning model can be refit later without re-running the study.

    Args:
        trial: Optuna trial object for parameter suggestion.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        implied_prob_val: Implied probability per validation row.
        n_min: Minimum copied bets for the edge metric guard.
        seed: XGBoost RNG seed.

    Returns:
        The trial's realized edge on val (or ``-1.0`` if too few bets).
    """
    params: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        "nthread": 1,
        "seed": seed,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=_NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = booster.best_iteration
    p_val = booster.predict(dval, iteration_range=(0, best_iter + 1))
    edge = realized_edge_metric(y_val, p_val, implied_prob_val, n_min=n_min)
    trial.set_user_attr("best_iteration", int(best_iter))
    return edge
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_training.py -v`
Expected: 2 passed (this may take 15-30 seconds — XGBoost trains real models).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/training.py tests/ml/test_training.py && uv run ty check src/pscanner/ml/training.py`
Expected: clean. If `ty` complains about the optuna or xgboost stubs, add `# ty: ignore[unresolved-import]` with a justification comment on the offending line.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/training.py tests/ml/test_training.py
git commit -m "feat(ml): run_single_trial fits an XGBoost trial against val edge"
```

---

## Task 12: `fit_winning_model` and `evaluate_on_test`

**Files:**
- Modify: `src/pscanner/ml/training.py`
- Modify: `tests/ml/test_training.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/ml/test_training.py`:

```python
from pscanner.ml.training import evaluate_on_test, fit_winning_model


def test_fit_winning_model_returns_booster_with_expected_iterations() -> None:
    X_train, y_train, _, _, _ = _toy_problem()
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }
    booster = fit_winning_model(
        best_params=params,
        best_iteration=10,
        X_train=X_train,
        y_train=y_train,
        seed=42,
    )
    # 11 trees corresponds to best_iteration + 1.
    assert booster.num_boosted_rounds() == 11


def test_evaluate_on_test_returns_metric_dict() -> None:
    X_train, y_train, X_val, y_val, _ = _toy_problem()
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }
    booster = fit_winning_model(
        best_params=params,
        best_iteration=20,
        X_train=X_train,
        y_train=y_train,
        seed=42,
    )
    implied_test = np.full(len(y_val), 0.5)
    result = evaluate_on_test(booster, X_val, y_val, implied_test, n_min=5)
    assert "edge" in result
    assert "accuracy" in result
    assert "logloss" in result
    assert "per_decile" in result
    assert isinstance(result["per_decile"], dict)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ml/test_training.py -v -k "fit_winning_model or evaluate_on_test"`
Expected: 2 failures with `ImportError`.

- [ ] **Step 3: Write the implementations**

Append to `src/pscanner/ml/training.py`:

```python
from collections.abc import Mapping

from pscanner.ml.metrics import per_decile_edge_breakdown


def fit_winning_model(
    best_params: Mapping[str, object],
    best_iteration: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> xgb.Booster:
    """Refit the winning hyperparams on train alone for ``best_iteration+1`` rounds.

    Avoids retraining on ``train + val`` (per the spec): the val set
    has already been used for model selection. Determinism is preserved
    by the shared ``seed`` + ``nthread=1``; this gives the same booster
    the winning trial produced.

    Args:
        best_params: Optuna's ``study.best_params`` dict.
        best_iteration: From the winning trial's user attrs.
        X_train: Training feature matrix.
        y_train: Training labels.
        seed: XGBoost RNG seed.

    Returns:
        The fitted XGBoost booster.
    """
    params: dict[str, object] = dict(best_params)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "nthread": 1,
            "seed": seed,
            "verbosity": 0,
        }
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(params, dtrain, num_boost_round=best_iteration + 1)


def evaluate_on_test(
    booster: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    implied_prob_test: np.ndarray,
    n_min: int,
) -> dict[str, object]:
    """Score the booster on the held-out test split.

    Returns:
        ``{"edge": float, "accuracy": float, "logloss": float,
        "per_decile": {decile_label: {"n": float, "mean_edge": float}}}``.
    """
    dtest = xgb.DMatrix(X_test)
    p_test = booster.predict(dtest)
    edge = realized_edge_metric(y_test, p_test, implied_prob_test, n_min=n_min)
    accuracy = float(((p_test >= 0.5).astype(int) == y_test).mean())
    eps = 1e-9
    logloss = float(
        -(y_test * np.log(p_test + eps) + (1 - y_test) * np.log(1 - p_test + eps)).mean()
    )
    decile = per_decile_edge_breakdown(y_test, p_test, implied_prob_test)
    return {
        "edge": edge,
        "accuracy": accuracy,
        "logloss": logloss,
        "per_decile": decile,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ml/test_training.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml/training.py tests/ml/test_training.py && uv run ty check src/pscanner/ml/training.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/ml/training.py tests/ml/test_training.py
git commit -m "feat(ml): fit_winning_model + evaluate_on_test"
```

---

## Task 13: `run_study` and `dump_artifacts`

**Files:**
- Modify: `src/pscanner/ml/training.py`
- Modify: `tests/ml/test_training.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/ml/test_training.py`:

```python
import json
from collections.abc import Callable
from pathlib import Path

import polars as pl

from pscanner.ml.training import run_study


def test_run_study_writes_all_artifacts(
    tmp_path: Path,
    make_synthetic_examples: Callable[..., pl.DataFrame],
) -> None:
    df = make_synthetic_examples(n_markets=20, rows_per_market=15, seed=3)
    output_dir = tmp_path / "run"
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=3,
        n_jobs=1,
        n_min=5,
        seed=42,
    )
    assert (output_dir / "model.json").exists()
    assert (output_dir / "preprocessor.json").exists()
    assert (output_dir / "study.db").exists()
    assert (output_dir / "metrics.json").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "best_params" in metrics
    assert "best_iteration" in metrics
    assert "best_val_edge" in metrics
    assert "test_edge" in metrics
    assert "test_accuracy" in metrics
    assert "test_logloss" in metrics
    assert "test_per_decile" in metrics
    assert "split_label_won_rate" in metrics
    rates = metrics["split_label_won_rate"]
    assert {"train", "val", "test"} == set(rates.keys())
    assert metrics["seed"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_training.py::test_run_study_writes_all_artifacts -v`
Expected: FAIL with `ImportError: cannot import name 'run_study'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pscanner/ml/training.py`:

```python
import json
import logging
from pathlib import Path

import polars as pl
import structlog

from pscanner.ml.preprocessing import (
    CATEGORICAL_COLS,
    OneHotEncoder,
    build_feature_matrix,
    drop_leakage_cols,
    temporal_split,
)

_log = structlog.get_logger(__name__)


def _dump_artifacts(
    output_dir: Path,
    booster: xgb.Booster,
    encoder: OneHotEncoder,
    metrics: dict[str, object],
) -> None:
    """Write model, preprocessor, and metrics to ``output_dir``."""
    booster.save_model(str(output_dir / "model.json"))
    (output_dir / "preprocessor.json").write_text(json.dumps(encoder.to_json(), indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def run_study(
    df: pl.DataFrame,
    output_dir: Path,
    n_trials: int,
    n_jobs: int,
    n_min: int,
    seed: int,
) -> None:
    """End-to-end study: preprocess, run Optuna, refit, evaluate, dump.

    Mutates ``output_dir`` (created if missing). Writes ``model.json``,
    ``preprocessor.json``, ``study.db``, ``metrics.json``.

    Args:
        df: Output of ``load_dataset``.
        output_dir: Per-run artifact directory.
        n_trials: Optuna trial budget.
        n_jobs: Parallel trials. Must be >=1.
        n_min: Edge-metric anti-overfit guard threshold.
        seed: Master RNG seed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = drop_leakage_cols(df)
    splits = temporal_split(df)
    encoder = OneHotEncoder.fit(splits.train, columns=CATEGORICAL_COLS)
    train_df = encoder.transform(splits.train)
    val_df = encoder.transform(splits.val)
    test_df = encoder.transform(splits.test)

    X_train, y_train, _ = build_feature_matrix(train_df)
    X_val, y_val, implied_val = build_feature_matrix(val_df)
    X_test, y_test, implied_test = build_feature_matrix(test_df)

    rates = {
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    }
    _log.info("ml.split_label_won_rate", **rates)

    storage_url = f"sqlite:///{output_dir / 'study.db'}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    # Silence optuna's per-trial chatter on stderr while the test suite
    # is running with filterwarnings=error.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
        storage=storage,
        study_name="copy_trade_gate",
    )
    study.optimize(
        lambda t: run_single_trial(
            trial=t,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            implied_prob_val=implied_val,
            n_min=n_min,
            seed=seed,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    best_iteration = int(study.best_trial.user_attrs["best_iteration"])
    booster = fit_winning_model(
        best_params=study.best_params,
        best_iteration=best_iteration,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
    )
    test_metrics = evaluate_on_test(
        booster=booster,
        X_test=X_test,
        y_test=y_test,
        implied_prob_test=implied_test,
        n_min=n_min,
    )

    metrics = {
        "best_params": dict(study.best_params),
        "best_iteration": best_iteration,
        "best_val_edge": float(study.best_value),
        "test_edge": test_metrics["edge"],
        "test_accuracy": test_metrics["accuracy"],
        "test_logloss": test_metrics["logloss"],
        "test_per_decile": test_metrics["per_decile"],
        "split_label_won_rate": rates,
        "seed": seed,
    }
    _dump_artifacts(output_dir, booster, encoder, metrics)
    _log.info(
        "ml.study_complete",
        best_val_edge=metrics["best_val_edge"],
        test_edge=metrics["test_edge"],
        n_trials=n_trials,
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/ml/test_training.py::test_run_study_writes_all_artifacts -v`
Expected: PASS (this may take 30-60 seconds — full pipeline on 300 synthetic rows).

If the test fails because XGBoost or Optuna emit deprecation warnings under `filterwarnings = ["error"]`, the fix is a targeted suppression in the test only — never globally:

```python
import warnings
import pytest

@pytest.mark.filterwarnings("ignore::DeprecationWarning:xgboost")
@pytest.mark.filterwarnings("ignore::DeprecationWarning:optuna")
def test_run_study_writes_all_artifacts(...):
    ...
```

Apply only as needed and add a comment naming the offending dep + version.

- [ ] **Step 5: Run the entire ml test suite**

Run: `uv run pytest tests/ml -v`
Expected: 17 passed.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/pscanner/ml tests/ml && uv run ty check src/pscanner/ml`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/ml/training.py tests/ml/test_training.py
git commit -m "feat(ml): run_study orchestrates preprocess + Optuna + dump"
```

---

## Task 14: CLI Wiring

**Files:**
- Create: `src/pscanner/ml/cli.py`
- Modify: `src/pscanner/cli.py`
- Create: `tests/ml/test_cli.py`

- [ ] **Step 1: Write the failing test for the ml CLI parser**

Create `tests/ml/test_cli.py`:

```python
"""Tests for the pscanner ml CLI parser."""

from __future__ import annotations

from pscanner.ml.cli import build_ml_parser


def test_train_subcommand_defaults() -> None:
    parser = build_ml_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.n_trials == 100
    assert args.seed == 42
    assert args.n_min == 20
    assert args.n_jobs == 10
    assert args.db == "data/corpus.sqlite3"


def test_train_subcommand_overrides() -> None:
    parser = build_ml_parser()
    args = parser.parse_args(
        [
            "train",
            "--n-trials",
            "5",
            "--seed",
            "7",
            "--n-min",
            "1",
            "--n-jobs",
            "2",
            "--db",
            "/tmp/x.sqlite3",
            "--output-dir",
            "/tmp/out",
        ]
    )
    assert args.n_trials == 5
    assert args.seed == 7
    assert args.n_min == 1
    assert args.n_jobs == 2
    assert args.db == "/tmp/x.sqlite3"
    assert args.output_dir == "/tmp/out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ml/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pscanner.ml.cli'`.

- [ ] **Step 3: Implement the parser and dispatcher**

Create `src/pscanner/ml/cli.py`:

```python
"""argparse handler for ``pscanner ml {train}``.

Mirrors the structure of ``pscanner.corpus.cli`` but synchronous —
training has no network I/O.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import structlog

from pscanner.ml.preprocessing import load_dataset
from pscanner.ml.training import run_study

_log = structlog.get_logger(__name__)


def build_ml_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ml subcommand group."""
    parser = argparse.ArgumentParser(prog="pscanner ml")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train an XGBoost copy-trade gate model")
    train.add_argument("--n-trials", type=int, default=100, help="Optuna trial budget")
    train.add_argument("--seed", type=int, default=42, help="RNG seed")
    train.add_argument(
        "--n-min", type=int, default=20, help="Min copied bets for the edge metric guard"
    )
    train.add_argument("--n-jobs", type=int, default=10, help="Parallel Optuna trials")
    train.add_argument(
        "--db",
        type=str,
        default="data/corpus.sqlite3",
        help="Path to the corpus SQLite database",
    )
    train.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Per-run artifact directory (default: models/<YYYY-MM-DD>-copy_trade_gate)",
    )
    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    """Run the training pipeline end-to-end."""
    db_path = Path(args.db)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        output_dir = Path("models") / f"{today}-copy_trade_gate"
    df = load_dataset(db_path)
    _log.info("ml.dataset_loaded", rows=df.height, output_dir=str(output_dir))
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        n_min=args.n_min,
        seed=args.seed,
    )
    return 0


_HANDLERS = {"train": _cmd_train}


def run_ml_command(argv: list[str]) -> int:
    """Parse ``argv`` (excluding the leading ``ml``) and dispatch."""
    parser = build_ml_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    return handler(args)
```

- [ ] **Step 4: Run the parser test**

Run: `uv run pytest tests/ml/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Wire `ml` into the top-level CLI**

Edit `src/pscanner/cli.py`. After the `from pscanner.corpus.cli import run_corpus_command` import (around line 36), add:

```python
from pscanner.ml.cli import run_ml_command
```

In the `main` function, after the `corpus` dispatch (around lines 67-68):

```python
    if args.command == "corpus":
        return asyncio.run(run_corpus_command(args.corpus_argv))
```

Add an analogous block:

```python
    if args.command == "ml":
        return run_ml_command(args.ml_argv)
```

In the `_build_parser` function, after the `corpus` subparser block (around lines 156-164):

```python
    corpus = sub.add_parser(
        "corpus",
        help="historical trade corpus subcommands",
    )
    corpus.add_argument(
        "corpus_argv",
        nargs=argparse.REMAINDER,
        help="forwarded to `pscanner corpus --help`",
    )
```

Add an analogous block:

```python
    ml = sub.add_parser(
        "ml",
        help="machine-learning training pipeline subcommands",
    )
    ml.add_argument(
        "ml_argv",
        nargs=argparse.REMAINDER,
        help="forwarded to `pscanner ml --help`",
    )
```

- [ ] **Step 6: Smoke-test the wired CLI**

Run: `uv run pscanner ml --help`
Expected: usage prints, with `train` listed as a subcommand.

Run: `uv run pscanner ml train --help`
Expected: usage prints, listing `--n-trials`, `--seed`, `--n-min`, `--n-jobs`, `--db`, `--output-dir`.

- [ ] **Step 7: Run the full project test suite to ensure nothing broke**

Run: `uv run pytest -q`
Expected: previously-passing tests still pass + new `tests/ml/` passes.

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: clean across the project.

- [ ] **Step 9: Commit**

```bash
git add src/pscanner/ml/cli.py src/pscanner/cli.py tests/ml/test_cli.py
git commit -m "feat(ml): wire pscanner ml train CLI"
```

---

## Task 15: End-to-End Smoke on Real Corpus

**Files:** None modified — just verifies the pipeline works against `data/corpus.sqlite3`.

This task is verification, not implementation. It runs the real CLI against the live corpus and inspects the artifacts.

- [ ] **Step 1: Confirm the corpus has training_examples populated**

Run:
```bash
uv run python -c "
import sqlite3
c = sqlite3.connect('data/corpus.sqlite3')
n = c.execute('SELECT COUNT(*) FROM training_examples').fetchone()[0]
r = c.execute('SELECT COUNT(*) FROM market_resolutions').fetchone()[0]
print(f'training_examples={n} market_resolutions={r}')
"
```
Expected: `training_examples=<N>` where `N > 0`. If `N == 0`, run `uv run pscanner corpus build-features` first and re-check.

- [ ] **Step 2: Run a small smoke training run**

Run:
```bash
uv run pscanner ml train --n-trials 5 --n-jobs 2 --output-dir /tmp/ml-smoke
```
Expected: command exits 0; the run takes a few minutes depending on corpus size. Watch for the `ml.split_label_won_rate` log line — the three rates should be roughly comparable. Sharp divergence (e.g. train=0.65, test=0.40) is informative, not a fix needed.

- [ ] **Step 3: Inspect the artifacts**

Run:
```bash
ls -la /tmp/ml-smoke/
cat /tmp/ml-smoke/metrics.json
```
Expected: four files (`model.json`, `preprocessor.json`, `study.db`, `metrics.json`). The metrics JSON should contain a positive or slightly-negative `test_edge` (a small smoke run is unlikely to learn anything meaningful — that's fine, the test is that the pipeline runs cleanly end-to-end).

- [ ] **Step 4: If `test_edge` is suspiciously high (>0.05) on a 5-trial smoke**

Inspect the per-decile breakdown in `metrics.json`. If edge is concentrated in a single decile with very few `n` taken bets, the model is winning by accident. This is expected on a tiny smoke run; just note for the real training run later.

- [ ] **Step 5: Cleanup**

```bash
rm -rf /tmp/ml-smoke
```

- [ ] **Step 6: Commit (no-op — nothing to commit; this task is verification only)**

Skip the commit step.

---

## Self-Review Checklist (run after Task 15)

- [ ] Spec coverage: every section in the spec is implemented.
  - Lazy load → Task 9 (`load_dataset`)
  - Column drops → Task 6 (`drop_leakage_cols`)
  - One-hot encoding → Task 7 (`OneHotEncoder`)
  - Temporal split → Task 8 (`temporal_split`)
  - Feature matrix extraction → Task 10 (`build_feature_matrix`)
  - Edge metric + decile breakdown → Tasks 4, 5
  - Single trial → Task 11
  - Refit + test eval → Task 12
  - Study orchestration + artifacts → Task 13
  - CLI surface → Task 14
  - End-to-end smoke → Task 15
- [ ] No placeholders, no "TBD" / "TODO" anywhere in the plan.
- [ ] Type/method names are consistent across tasks (e.g. `Split.train/val/test`, `OneHotEncoder.fit/transform`, `realized_edge_metric` everywhere).
- [ ] Every code-changing step shows the actual code, not a description.

---

## Open Risks (informational — not blocking implementation)

1. **Optuna with SQLite storage + `n_jobs > 1`** sometimes warns about
   thread safety. SQLite-backed storage uses connection pooling per
   process, so single-process parallelism is fine; if Optuna emits a
   warning under `filterwarnings = ["error"]`, file a targeted
   suppression in the test (Task 13 Step 4 has the pattern).
2. **XGBoost API drift** — `iteration_range` and `best_iteration` are
   the modern API; older snippets used `ntree_limit`. If `xgboost==2.x`
   resolves, the API in this plan is correct.
3. **Polars `to_numpy()` of int columns containing nulls** raises
   because numpy ints can't be NaN. The synthetic fixture has null-able
   counts but stores them as ints (no nulls injected). For the real
   corpus, `prior_*` count columns are `INTEGER NOT NULL` per
   `repos.py`, so this is fine. If a future schema change introduces a
   nullable int column, cast it to float in `build_feature_matrix`
   before `to_numpy()`.
