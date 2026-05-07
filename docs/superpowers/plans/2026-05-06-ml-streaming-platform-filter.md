# ML streaming pipeline platform filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread `platform: str = "polymarket"` through `pscanner.ml.streaming` and `pscanner.ml.cli` so `pscanner ml train --platform <name>` trains a single-platform model. Single-platform-per-run by design; the parameter shape and SQL idioms preserve forward-compatibility for future multi-platform aggregation.

**Architecture:** Every helper in `streaming.py` gains a `platform: str = "polymarket"` keyword-only parameter; `open_dataset` is the entry point that propagates it; `StreamingDataset` stashes it for downstream `_SplitIter` constructions; SQL gets `WHERE te.platform = ?` predicates and the `_SplitIter` JOIN tightens to `ON mr.platform = te.platform AND mr.condition_id = te.condition_id`. Defaults preserve every existing call site (tests, `scripts/analyze_model.py`, internal callers) unchanged.

**Tech Stack:** Python 3.13, SQLite, Polars, XGBoost, pytest. Quick verify: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Spec:** `docs/superpowers/specs/2026-05-06-ml-streaming-platform-filter-design.md`

---

## File map

- **Modify** `tests/ml/conftest.py` — `_seed_db_from_synthetic` and `make_synthetic_examples_db` gain `platform: str = "polymarket"` kwargs.
- **Modify** `src/pscanner/ml/streaming.py` — most of the change. `_partition_markets`, `_fit_encoder_on_train`, `_count_split_rows`, `_SplitIter`, `StreamingDataset.val_aux`, `StreamingDataset.materialize_test`, `StreamingDataset._build_dmatrix`, `open_dataset` all thread `platform`.
- **Modify** `src/pscanner/ml/training.py` — `run_study` accepts `platform`, forwards to `open_dataset`.
- **Modify** `src/pscanner/ml/cli.py` — `build_ml_parser` adds `--platform` flag; `_cmd_train` forwards `args.platform`.
- **Modify** `src/pscanner/ml/preprocessing.py` — small docstring/comment refresh on `_NEVER_LOAD_COLS` to point at the active filter (no functional change).
- **Modify** `scripts/analyze_model.py` — `--platform` flag, forwards to `open_dataset`.
- **Modify** `tests/ml/test_cli.py` — 3 new parser tests.
- **Create** `tests/ml/test_streaming_platform_filter.py` — 3 behavioral tests on a mixed-platform corpus.
- **Modify** `CLAUDE.md` — refresh the "ML pipeline platform filter" follow-up bullet to reflect the now-landed work.

---

### Task 1: Extend `make_synthetic_examples_db` fixture with `platform` kwarg

**Files:**
- Modify: `tests/ml/conftest.py:24-58, 165-189`

The existing fixture seeds a single-platform corpus (everything defaults to `polymarket` via the schema CHECK/DEFAULT). The new `tests/ml/test_streaming_platform_filter.py` needs to seed a mixed-platform corpus by calling the helper twice — once with `platform="polymarket"`, once with `platform="kalshi"`. This task adds the kwarg.

- [ ] **Step 1: Write a failing test that uses the new kwarg**

Add to `tests/ml/test_cli.py` (sanity test for the fixture extension; the real platform-filter tests come in Task 11):

```python
def test_make_synthetic_examples_db_accepts_platform_kwarg(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    import sqlite3

    db_path = make_synthetic_examples_db(
        n_markets=2, rows_per_market=2, seed=0, platform="kalshi"
    )
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT platform FROM training_examples"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("kalshi",)]
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/ml/test_cli.py::test_make_synthetic_examples_db_accepts_platform_kwarg -v`
Expected: FAIL — `_build()` got an unexpected keyword argument `platform` (or rows are `polymarket` because the fixture ignored the kwarg).

- [ ] **Step 3: Update `_seed_db_from_synthetic` and the fixture**

In `tests/ml/conftest.py`, replace the entire `_seed_db_from_synthetic` function (lines 24-59) with:

```python
def _seed_db_from_synthetic(
    conn: sqlite3.Connection,
    df: pl.DataFrame,
    *,
    platform: str = "polymarket",
) -> None:
    """Populate corpus_markets, market_resolutions, training_examples from
    a synthetic-examples Polars frame so load_dataset / open_dataset see
    matching rows."""
    markets = df.select(["condition_id", "resolved_at"]).unique()
    for row in markets.iter_rows(named=True):
        conn.execute(
            """
            INSERT INTO corpus_markets (
              platform, condition_id, event_slug, category, closed_at,
              total_volume_usd, market_slug, backfill_state, enumerated_at
            ) VALUES (?, ?, '', 'sports', ?, 1000.0, '', 'complete', ?)
            """,
            (
                platform,
                row["condition_id"],
                int(row["resolved_at"]),
                int(row["resolved_at"]) - 1,
            ),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, ?, 0, 1, ?, 'gamma', ?)
            """,
            (
                platform,
                row["condition_id"],
                int(row["resolved_at"]),
                int(row["resolved_at"]),
            ),
        )
    examples = df.drop("resolved_at")
    for row in examples.iter_rows(named=True):
        # Inject platform if the synthetic frame didn't set it (most tests).
        if "platform" not in row:
            row = {"platform": platform, **row}
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT INTO training_examples ({cols}) VALUES ({placeholders})",  # noqa: S608 -- column names are statically derived from synthetic frame
            tuple(row.values()),
        )
    conn.commit()
```

In `tests/ml/conftest.py`, replace the `_build` callable inside `make_synthetic_examples_db` (lines 172-187) with:

```python
    def _build(
        *,
        n_markets: int = 30,
        rows_per_market: int = 20,
        seed: int = 0,
        platform: str = "polymarket",
        db_path: Path | None = None,
    ) -> Path:
        df = make_synthetic_examples(
            n_markets=n_markets, rows_per_market=rows_per_market, seed=seed
        )
        if db_path is None:
            db_path = (
                tmp_path
                / f"corpus_n{n_markets}_r{rows_per_market}_s{seed}_{platform}.sqlite3"
            )
        # init_corpus_db is idempotent; re-opening lets a caller layer a second
        # platform's rows onto a corpus the helper already created.
        conn = init_corpus_db(db_path)
        try:
            _seed_db_from_synthetic(conn, df, platform=platform)
        finally:
            conn.close()
        return db_path
```

The new `db_path` kwarg lets the platform-filter test layer kalshi rows on top of a polymarket-seeded DB by passing the same path twice.

- [ ] **Step 4: Run the test, expect pass**

Run: `uv run pytest tests/ml/test_cli.py::test_make_synthetic_examples_db_accepts_platform_kwarg -v`
Expected: PASS.

Run: `uv run pytest tests/ml/ -q`
Expected: all existing tests still pass — defaults preserve behavior.

- [ ] **Step 5: Commit**

```bash
git add tests/ml/conftest.py tests/ml/test_cli.py
git commit -m "test(ml): seed fixture accepts platform kwarg + db_path override"
```

---

### Task 2: Thread `platform` through `_partition_markets`

**Files:**
- Modify: `src/pscanner/ml/streaming.py:204-221`

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_streaming.py`:

```python
def test_partition_markets_filters_by_platform(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    """`_partition_markets` returns only the requested platform's condition_ids."""
    import sqlite3

    from pscanner.ml.streaming import _partition_markets

    poly_db = make_synthetic_examples_db(n_markets=4, rows_per_market=2, seed=0)
    # Layer kalshi rows on top of the same DB.
    make_synthetic_examples_db(
        n_markets=4, rows_per_market=2, seed=1, platform="kalshi", db_path=poly_db
    )
    conn = sqlite3.connect(str(poly_db))
    try:
        train_p, val_p, test_p = _partition_markets(conn, platform="polymarket")
        train_k, val_k, test_k = _partition_markets(conn, platform="kalshi")
    finally:
        conn.close()
    poly_total = len(train_p) + len(val_p) + len(test_p)
    kalshi_total = len(train_k) + len(val_k) + len(test_k)
    assert poly_total == 4, "polymarket has 4 markets"
    assert kalshi_total == 4, "kalshi has 4 markets"
    assert (train_p | val_p | test_p).isdisjoint(train_k | val_k | test_k), (
        "polymarket and kalshi condition_id sets must not overlap"
    )
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_streaming.py::test_partition_markets_filters_by_platform -v`
Expected: FAIL — `_partition_markets()` got an unexpected keyword argument `platform`.

- [ ] **Step 3: Update `_partition_markets`**

In `src/pscanner/ml/streaming.py`, replace the function (lines 204-221) with:

```python
def _partition_markets(
    conn: sqlite3.Connection,
    *,
    platform: str = "polymarket",
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Run P1: SELECT condition_id, resolved_at FROM market_resolutions ORDER BY...

    Slice the sorted list at 70% / 85% into train, val, test. Filters
    ``market_resolutions`` by ``platform`` so each split is platform-scoped;
    after RFC #35 PR A, ``condition_id`` is no longer unique across
    platforms (composite PK), so the filter is required for correctness.
    """
    rows = conn.execute(
        "SELECT condition_id, resolved_at FROM market_resolutions "
        "WHERE platform = ? "
        "ORDER BY resolved_at, condition_id",
        (platform,),
    ).fetchall()
    n = len(rows)
    n_train = round(_TRAIN_FRAC * n)
    n_val = round(_VAL_FRAC * n)
    train = frozenset(r[0] for r in rows[:n_train])
    val = frozenset(r[0] for r in rows[n_train : n_train + n_val])
    test = frozenset(r[0] for r in rows[n_train + n_val :])
    return train, val, test
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming.py -q`
Expected: all pass (the new test plus all existing ones — `_partition_markets()` callers that don't pass `platform` get the polymarket default).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "feat(ml): platform filter on _partition_markets"
```

---

### Task 3: Thread `platform` through `_fit_encoder_on_train`

**Files:**
- Modify: `src/pscanner/ml/streaming.py:243-267`

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_streaming.py`:

```python
def test_fit_encoder_on_train_filters_by_platform(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    """`_fit_encoder_on_train` only sees rows with the requested platform."""
    import sqlite3

    from pscanner.ml.streaming import _fit_encoder_on_train, _partition_markets

    poly_db = make_synthetic_examples_db(n_markets=4, rows_per_market=2, seed=0)
    make_synthetic_examples_db(
        n_markets=4, rows_per_market=2, seed=1, platform="kalshi", db_path=poly_db
    )
    conn = sqlite3.connect(str(poly_db))
    try:
        train_poly, _, _ = _partition_markets(conn, platform="polymarket")
        encoder = _fit_encoder_on_train(conn, train_poly, platform="polymarket")
    finally:
        conn.close()
    # The encoder fits over the categorical levels of training_examples joined to
    # the train condition_ids. Even seeding two platforms, the train markets are
    # platform-scoped — encoder.levels reflects exactly the polymarket train rows.
    assert "side" in encoder.levels
    assert set(encoder.levels["side"]).issubset({"YES", "NO"})
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_streaming.py::test_fit_encoder_on_train_filters_by_platform -v`
Expected: FAIL — `_fit_encoder_on_train()` got an unexpected keyword argument `platform`.

- [ ] **Step 3: Update `_fit_encoder_on_train`**

In `src/pscanner/ml/streaming.py`, replace the function (lines 243-267) with:

```python
def _fit_encoder_on_train(
    conn: sqlite3.Connection,
    train_markets: frozenset[str],
    *,
    platform: str = "polymarket",
) -> OneHotEncoder:
    """Run P2: fit a OneHotEncoder on the train split's categorical levels.

    SELECTs DISTINCT side, top_category, market_category from training_examples
    joined on the _p2_train temp table. Belt-and-suspenders ``WHERE platform = ?``
    is added because after RFC #35 PR A, two platforms can share the same
    ``condition_id`` string; the temp-table partition isolates split membership,
    the platform predicate isolates platform membership.
    """
    _populate_temp_table(conn, "_p2_train", train_markets)
    rows = conn.execute(
        "SELECT DISTINCT side, top_category, market_category "
        "FROM training_examples te "
        "JOIN _p2_train tm USING (condition_id) "
        "WHERE te.platform = ?",
        (platform,),
    ).fetchall()
    df = pl.DataFrame(
        rows,
        schema={
            "side": pl.String,
            "top_category": pl.String,
            "market_category": pl.String,
        },
        orient="row",
    )
    return OneHotEncoder.fit(df, columns=CATEGORICAL_COLS)
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "feat(ml): platform filter on _fit_encoder_on_train"
```

---

### Task 4: Thread `platform` through `_count_split_rows`

**Files:**
- Modify: `src/pscanner/ml/streaming.py:270-285`

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_streaming.py`:

```python
def test_count_split_rows_filters_by_platform(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    """`_count_split_rows` counts only the requested platform's rows."""
    import sqlite3

    from pscanner.ml.streaming import _count_split_rows, _partition_markets

    poly_db = make_synthetic_examples_db(n_markets=10, rows_per_market=3, seed=0)
    make_synthetic_examples_db(
        n_markets=10, rows_per_market=3, seed=1, platform="kalshi", db_path=poly_db
    )
    conn = sqlite3.connect(str(poly_db))
    try:
        train, val, test = _partition_markets(conn, platform="polymarket")
        n_train, n_val, n_test = _count_split_rows(
            conn, train, val, test, platform="polymarket"
        )
    finally:
        conn.close()
    # Per-platform corpus has 10 markets x 3 rows = 30 rows total.
    assert n_train + n_val + n_test == 30
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_streaming.py::test_count_split_rows_filters_by_platform -v`
Expected: FAIL.

- [ ] **Step 3: Update `_count_split_rows`**

In `src/pscanner/ml/streaming.py`, replace the function (lines 270-285) with:

```python
def _count_split_rows(
    conn: sqlite3.Connection,
    train: frozenset[str],
    val: frozenset[str],
    test: frozenset[str],
    *,
    platform: str = "polymarket",
) -> tuple[int, int, int]:
    """Run P3: COUNT(*) per split via temp tables.

    ``WHERE te.platform = ?`` is belt-and-suspenders against cross-platform
    ``condition_id`` collisions (see ``_fit_encoder_on_train`` docstring).
    """
    counts = []
    for label, markets in (("_p3_train", train), ("_p3_val", val), ("_p3_test", test)):
        _populate_temp_table(conn, label, markets)
        (n,) = conn.execute(
            f"SELECT COUNT(*) FROM training_examples te "  # noqa: S608 -- label is a literal
            f"JOIN {label} sm USING (condition_id) "
            "WHERE te.platform = ?",
            (platform,),
        ).fetchone()
        counts.append(int(n))
    return counts[0], counts[1], counts[2]
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "feat(ml): platform filter on _count_split_rows"
```

---

### Task 5: Add `platform` field to `_SplitIter` + composite-key JOIN

**Files:**
- Modify: `src/pscanner/ml/streaming.py:317-353`

This is the load-bearing query. Two changes inside `_SplitIter.__iter__`: tighten the JOIN to `market_resolutions` to use the composite key, and add `WHERE te.platform = ?`.

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_streaming.py`:

```python
def test_split_iter_filters_by_platform(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    """`_SplitIter` yields only rows with the configured platform."""
    from pscanner.ml.streaming import _SplitIter, open_dataset

    poly_db = make_synthetic_examples_db(n_markets=10, rows_per_market=3, seed=0)
    make_synthetic_examples_db(
        n_markets=10, rows_per_market=3, seed=1, platform="kalshi", db_path=poly_db
    )
    with open_dataset(poly_db) as ds:
        # _SplitIter is constructed inside StreamingDataset; here we instantiate
        # it directly with the polymarket train markets to verify the platform
        # field gates the SQL.
        iterator = _SplitIter(
            db_path=poly_db,
            condition_ids=ds._train_markets,  # type: ignore[arg-type]
            encoder=ds.encoder,  # type: ignore[arg-type]
            kept_cols=ds._kept_cols,
            chunk_size=64,
            platform="polymarket",
        )
        chunks = list(iter(iterator))
    total_rows = sum(x.shape[0] for x, _, _ in chunks)
    # The train markets came from polymarket; rows count must equal 10*3 * 0.7 (n_train_rows).
    assert total_rows == ds.n_train_rows
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_streaming.py::test_split_iter_filters_by_platform -v`
Expected: FAIL — `_SplitIter` got an unexpected keyword argument `platform`.

- [ ] **Step 3: Update `_SplitIter`**

In `src/pscanner/ml/streaming.py`, replace the dataclass definition and `__iter__` body (lines 317-353) with:

```python
@dataclass
class _SplitIter:
    """Yields (x, y, implied) numpy tuples per chunk for one split.

    Each iter() opens a fresh sqlite3.Connection (XGBoost's DataIter may
    iterate from worker threads; sqlite3 connections aren't thread-safe).
    The connection's TEMP TABLE is populated from condition_ids on first
    iteration; the connection closes when iteration finishes or raises.

    The ``platform`` field gates both the JOIN to ``market_resolutions``
    (composite key after RFC #35 PR A) and the ``WHERE te.platform = ?``
    predicate on ``training_examples``. Defaults to ``"polymarket"`` to
    preserve existing test fixtures that construct ``_SplitIter`` directly.
    """

    db_path: Path
    condition_ids: frozenset[str]
    encoder: OneHotEncoder
    kept_cols: tuple[str, ...]
    chunk_size: int
    platform: str = "polymarket"

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        select_list = ", ".join(f"te.{c}" for c in self.kept_cols)
        sql = (
            f"SELECT {select_list}, mr.resolved_at "  # noqa: S608 -- kept_cols derived from PRAGMA
            "FROM training_examples te "
            "JOIN market_resolutions mr "
            "  ON mr.platform = te.platform AND mr.condition_id = te.condition_id "
            "JOIN _split_markets sm USING (condition_id) "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        col_names = (*self.kept_cols, "resolved_at")
        conn = sqlite3.connect(str(self.db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self.condition_ids)
            cursor = conn.execute(sql, (self.platform,))
            while True:
                rows = cursor.fetchmany(self.chunk_size)
                if not rows:
                    return
                yield self._encode_chunk(rows, col_names)
        finally:
            conn.close()
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming.py -q`
Expected: all pass — the new test plus the existing 16 streaming tests (all of which use the polymarket-default fixture and the polymarket-default `_SplitIter.platform`).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "feat(ml): platform field + composite-key JOIN in _SplitIter"
```

---

### Task 6: `StreamingDataset` stashes `_platform` and threads it to `_SplitIter` constructions

**Files:**
- Modify: `src/pscanner/ml/streaming.py:55-201`

`StreamingDataset` constructs `_SplitIter` instances in three places (`_build_dmatrix` for `dtrain`/`dval`, `materialize_test`, and the `val_aux` direct SQL). All need to know the platform.

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_streaming.py`:

```python
def test_streaming_dataset_stores_platform(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    """`StreamingDataset` exposes the platform it was opened for."""
    from pscanner.ml.streaming import open_dataset

    db = make_synthetic_examples_db(n_markets=4, rows_per_market=2, seed=0)
    with open_dataset(db) as ds:
        assert ds._platform == "polymarket"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_streaming.py::test_streaming_dataset_stores_platform -v`
Expected: FAIL — `StreamingDataset` has no attribute `_platform`.

- [ ] **Step 3: Update `StreamingDataset`**

In `src/pscanner/ml/streaming.py`, replace the dataclass body (lines 55-74) with:

```python
@dataclass
class StreamingDataset:
    """Two-pass streaming view over training_examples.

    Constructed by :func:`open_dataset`. Pre-scan results live on the
    instance; per-split chunked reads are deferred until ``dtrain`` /
    ``dval`` / ``val_aux`` / ``materialize_test`` is called.
    """

    _db_path: Path
    _chunk_size: int
    _train_markets: frozenset[str] = field(default_factory=frozenset)
    _val_markets: frozenset[str] = field(default_factory=frozenset)
    _test_markets: frozenset[str] = field(default_factory=frozenset)
    encoder: OneHotEncoder | None = None
    feature_names: tuple[str, ...] = ()
    _kept_cols: tuple[str, ...] = ()
    n_train_rows: int = 0
    n_val_rows: int = 0
    n_test_rows: int = 0
    _platform: str = "polymarket"
```

Replace `val_aux` (lines 97-130) with a version that adds `WHERE te.platform = ?` and binds the platform parameter:

```python
    def val_aux(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_val, implied_prob_val)`` numpy arrays.

        Streamed via the same chunked path as ``dval`` but pulls only
        the two columns the edge-metric closure needs, into pre-allocated
        arrays sized from the P3 row count. No feature matrix is built.
        """
        y = np.empty(self.n_val_rows, dtype=np.int32)
        implied = np.empty(self.n_val_rows, dtype=np.float32)

        sql = (
            "SELECT te.label_won, te.implied_prob_at_buy "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._val_markets)
            cursor = conn.execute(sql, (self._platform,))
            offset = 0
            while True:
                rows = cursor.fetchmany(self._chunk_size)
                if not rows:
                    break
                end = offset + len(rows)
                for i, (label, prob) in enumerate(rows):
                    y[offset + i] = int(label)
                    implied[offset + i] = float(prob)
                offset = end
        finally:
            conn.close()

        return y, implied
```

Replace `materialize_test` (lines 132-180) with a version that passes platform through to `_SplitIter` and adds `WHERE te.platform = ?` to the parallel top_category SELECT:

```python
    def materialize_test(self) -> TestSplit:
        """Stream the test split into pre-allocated numpy arrays.

        Uses the same _SplitIter as dtrain/dval for X/y/implied; pulls
        the unencoded ``top_category`` column via a parallel SELECT for
        per-category metrics in evaluate_on_test.
        """
        if self.encoder is None:
            raise RuntimeError("StreamingDataset.encoder is None; was open_dataset used?")

        x = np.empty((self.n_test_rows, len(self.feature_names)), dtype=np.float32)
        y = np.empty(self.n_test_rows, dtype=np.int32)
        implied = np.empty(self.n_test_rows, dtype=np.float32)

        source = _SplitIter(
            db_path=self._db_path,
            condition_ids=self._test_markets,
            encoder=self.encoder,
            kept_cols=self._kept_cols,
            chunk_size=self._chunk_size,
            platform=self._platform,
        )
        offset = 0
        for x_chunk, y_chunk, implied_chunk in iter(source):
            end = offset + x_chunk.shape[0]
            x[offset:end] = x_chunk
            y[offset:end] = y_chunk
            implied[offset:end] = implied_chunk
            offset = end

        # Parallel small SELECT for unencoded top_category strings.
        # Mirrors the deleted _extract_top_category: nulls become "".
        # Bulk fetchall + np.array is multiple orders of magnitude faster than
        # the per-row loop pattern at corpus scale (~2M rows takes seconds vs
        # ~10+ minutes for the row-at-a-time numpy assignment).
        sql = (
            "SELECT COALESCE(te.top_category, '') "
            "FROM training_examples te "
            "JOIN _split_markets sm USING (condition_id) "
            "WHERE te.platform = ? "
            "ORDER BY te.id"
        )
        conn = sqlite3.connect(str(self._db_path))
        try:
            _populate_temp_table(conn, "_split_markets", self._test_markets)
            rows = conn.execute(sql, (self._platform,)).fetchall()
        finally:
            conn.close()
        top_categories = np.array([r[0] for r in rows], dtype=object)

        return TestSplit(x=x, y=y, implied_prob=implied, top_categories=top_categories)
```

Replace `_build_dmatrix` (lines 182-201) with a version that passes platform through to `_SplitIter`:

```python
    def _build_dmatrix(
        self,
        condition_ids: frozenset[str],
        *,
        device: str,  # forwarded to booster via xgb.train params, not DMatrix constructor
        ref: xgb.QuantileDMatrix | None,
    ) -> xgb.QuantileDMatrix:
        if self.encoder is None:
            raise RuntimeError("StreamingDataset.encoder is None; was open_dataset used?")
        source = _SplitIter(
            db_path=self._db_path,
            condition_ids=condition_ids,
            encoder=self.encoder,
            kept_cols=self._kept_cols,
            chunk_size=self._chunk_size,
            platform=self._platform,
        )
        kwargs: dict[str, object] = {"max_bin": 256}
        if ref is not None:
            kwargs["ref"] = ref
        return xgb.QuantileDMatrix(SplitDataIter(source), **kwargs)
```

Note: `open_dataset` doesn't yet supply `_platform` when constructing `StreamingDataset`. That gap is closed in Task 7. Until Task 7 lands, `_platform` falls back to its default `"polymarket"` — the test from this task's Step 1 still passes because the default is what we'd pass anyway.

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "feat(ml): StreamingDataset stashes platform + threads to _SplitIter"
```

---

### Task 7: `open_dataset` accepts `platform` and propagates it

**Files:**
- Modify: `src/pscanner/ml/streaming.py:431-468`

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_streaming.py`:

```python
def test_open_dataset_records_requested_platform(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    """`open_dataset(platform=...)` is reflected on the dataset."""
    from pscanner.ml.streaming import open_dataset

    db = make_synthetic_examples_db(n_markets=4, rows_per_market=2, seed=0)
    make_synthetic_examples_db(
        n_markets=4, rows_per_market=2, seed=1, platform="kalshi", db_path=db
    )
    with open_dataset(db, platform="kalshi") as ds:
        assert ds._platform == "kalshi"
        # Train markets came from kalshi rows.
        kalshi_total = ds.n_train_rows + ds.n_val_rows + ds.n_test_rows
        assert kalshi_total == 8  # 4 markets x 2 rows
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_streaming.py::test_open_dataset_records_requested_platform -v`
Expected: FAIL — either the kwarg is unknown, or `_platform` ends up `"polymarket"` because nothing is propagated.

- [ ] **Step 3: Update `open_dataset`**

In `src/pscanner/ml/streaming.py`, replace the function (lines 431-468) with:

```python
@contextmanager
def open_dataset(
    db_path: Path,
    *,
    chunk_size: int = 100_000,
    platform: str = "polymarket",
) -> Iterator[StreamingDataset]:
    """Open the corpus for streaming training.

    Args:
        db_path: Path to the corpus SQLite database.
        chunk_size: Rows per chunk fed into xgboost's DataIter. Default
            100_000; see Issue #39 for the memory / overhead trade-off.
        platform: RFC #35 PR A platform tag. Filters every SELECT against
            ``training_examples`` and ``market_resolutions`` to rows with
            this tag. Defaults to ``"polymarket"`` so existing callers
            (tests, ``scripts/analyze_model.py``, internal callers) see
            no behavior change. Single-platform-per-run by design;
            multi-platform aggregation widens this to a sequence type.

    Yields:
        A :class:`StreamingDataset` whose pre-scan has completed.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        train, val, test = _partition_markets(conn, platform=platform)
        encoder = _fit_encoder_on_train(conn, train, platform=platform)
        n_train, n_val, n_test = _count_split_rows(
            conn, train, val, test, platform=platform
        )
        kept = _kept_columns(conn)
        ds = StreamingDataset(
            _db_path=db_path,
            _chunk_size=chunk_size,
            _train_markets=train,
            _val_markets=val,
            _test_markets=test,
            encoder=encoder,
            feature_names=_derive_feature_names(kept, encoder),
            _kept_cols=kept,
            n_train_rows=n_train,
            n_val_rows=n_val,
            n_test_rows=n_test,
            _platform=platform,
        )
        yield ds
    finally:
        conn.close()
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/streaming.py tests/ml/test_streaming.py
git commit -m "feat(ml): open_dataset accepts platform and threads to all helpers"
```

---

### Task 8: `run_study` accepts `platform` and forwards to `open_dataset`

**Files:**
- Modify: `src/pscanner/ml/training.py` (locate `run_study` and the `open_dataset` call)

- [ ] **Step 1: Write a failing test**

Add to `tests/ml/test_training.py`:

```python
def test_run_study_accepts_platform_kwarg(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """`run_study` accepts platform kwarg and forwards to open_dataset."""
    from pscanner.ml.training import run_study

    poly_db = make_synthetic_examples_db(n_markets=10, rows_per_market=4, seed=0)
    make_synthetic_examples_db(
        n_markets=10, rows_per_market=4, seed=1, platform="kalshi", db_path=poly_db
    )
    output_dir = tmp_path / "out"
    # Smoke-train on kalshi rows only; using a tiny budget so this stays fast.
    run_study(
        db_path=poly_db,
        output_dir=output_dir,
        n_trials=1,
        n_jobs=1,
        n_min=1,
        seed=0,
        device="cpu",
        chunk_size=64,
        platform="kalshi",
    )
    # Read back the artifact and confirm the platform was honored.
    import json

    with (output_dir / "preprocessor.json").open() as fh:
        cfg = json.load(fh)
    assert cfg.get("platform") == "kalshi"
```

If the existing `preprocessor.json` doesn't yet record `platform`, the writer in `run_study` must be updated. Find the `json.dump` site in `src/pscanner/ml/training.py` (search `preprocessor.json`) and add `"platform": platform` to the dict.

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_training.py::test_run_study_accepts_platform_kwarg -v`
Expected: FAIL — `run_study()` got an unexpected keyword argument `platform`.

- [ ] **Step 3: Update `run_study`**

Open `src/pscanner/ml/training.py`. Locate the `run_study` signature and the `open_dataset(...)` call. Make these changes:

1. Add `platform: str = "polymarket"` keyword-only argument to `run_study`'s signature (place it after the existing kwargs, with a default).
2. Forward to `open_dataset(db_path, chunk_size=chunk_size, platform=platform)`.
3. Find the `preprocessor.json` writer. Add `"platform": platform` to the dict that gets serialized.

Concretely, the signature change at the function definition looks like (insert one line into the existing kwarg list):

```python
def run_study(
    db_path: Path,
    output_dir: Path,
    n_trials: int,
    n_jobs: int,
    n_min: int,
    seed: int,
    device: str = "cpu",
    chunk_size: int = 100_000,
    accepted_categories: tuple[str, ...] | None = None,
    platform: str = "polymarket",
) -> None:
```

The `open_dataset` call site changes from:

```python
with open_dataset(db_path, chunk_size=chunk_size) as ds:
```

to:

```python
with open_dataset(db_path, chunk_size=chunk_size, platform=platform) as ds:
```

The `preprocessor.json` write looks something like (find the existing writer; add the `"platform"` key):

```python
preprocessor_payload = {
    # ... existing keys (encoder, feature_names, accepted_categories, etc.) ...
    "platform": platform,
}
with (output_dir / "preprocessor.json").open("w") as fh:
    json.dump(preprocessor_payload, fh, indent=2)
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_training.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/training.py tests/ml/test_training.py
git commit -m "feat(ml): run_study accepts platform + records in preprocessor.json"
```

---

### Task 9: `pscanner ml train --platform` CLI flag

**Files:**
- Modify: `src/pscanner/ml/cli.py:21-90`
- Modify: `tests/ml/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/ml/test_cli.py`:

```python
def test_train_parser_accepts_platform_flag() -> None:
    from pscanner.ml.cli import build_ml_parser

    parser = build_ml_parser()
    args = parser.parse_args(["train", "--platform", "kalshi", "--db", "x"])
    assert args.platform == "kalshi"


def test_train_parser_default_platform_is_polymarket() -> None:
    from pscanner.ml.cli import build_ml_parser

    parser = build_ml_parser()
    args = parser.parse_args(["train", "--db", "x"])
    assert args.platform == "polymarket"


def test_train_parser_rejects_unknown_platform() -> None:
    from pscanner.ml.cli import build_ml_parser

    parser = build_ml_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--platform", "ftx", "--db", "x"])
```

If `pytest` isn't already imported at the top of `test_cli.py`, add `import pytest`.

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_cli.py -v -k platform`
Expected: 3 fails — `--platform` is unknown.

- [ ] **Step 3: Add the flag to the parser and forward in `_cmd_train`**

In `src/pscanner/ml/cli.py`, find the `build_ml_parser` function. Add a new `train.add_argument` block (after the existing flags, before `return parser`):

```python
    train.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "kalshi", "manifold"],
        default="polymarket",
        help=(
            "Filter training_examples to rows with this platform tag. "
            "Single-platform per run; multi-platform aggregation is a future "
            "follow-up. Defaults to polymarket."
        ),
    )
```

In the same file, find `_cmd_train`. Add `platform=args.platform` to the `run_study(...)` call. Also add `platform=args.platform` to the `_log.info("ml.dataset_loaded", ...)` event (if that event exists in the current `_cmd_train`).

The `_cmd_train` body becomes:

```python
def _cmd_train(args: argparse.Namespace) -> int:
    """Run the training pipeline end-to-end."""
    db_path = Path(args.db)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        output_dir = Path("models") / f"{today}-copy_trade_gate"
    run_study(
        db_path=db_path,
        output_dir=output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        n_min=args.n_min,
        seed=args.seed,
        device=args.device,
        chunk_size=args.chunk_size,
        platform=args.platform,
    )
    return 0
```

(If the existing `_cmd_train` builds a DataFrame via `load_dataset` and then calls `run_study(df=...)`, that's the pre-#39 shape. Read the actual file first; the post-#39 `_cmd_train` calls `run_study` with `db_path=` directly. Mirror what's there.)

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/ml/test_cli.py -v`
Expected: all pass (3 new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/cli.py tests/ml/test_cli.py
git commit -m "feat(ml): --platform flag on \`pscanner ml train\`"
```

---

### Task 10: `scripts/analyze_model.py` accepts `--platform`

**Files:**
- Modify: `scripts/analyze_model.py`

- [ ] **Step 1: Read the file**

Read `scripts/analyze_model.py` to find the argparse parser and the `open_dataset(db_path)` call site. The script's `main` typically defines flags like `--model-dir`, `--db`, `--top-k`. Add `--platform` to the same parser.

- [ ] **Step 2: Add `--platform` argument and forward it**

In the argparse section of `scripts/analyze_model.py`, add:

```python
    parser.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "kalshi", "manifold"],
        default="polymarket",
        help="Filter to rows with this platform tag (matches `pscanner ml train --platform`).",
    )
```

In the `analyze` (or equivalent) function, change the `open_dataset` call from:

```python
with open_dataset(db_path) as ds:
```

to:

```python
with open_dataset(db_path, platform=platform) as ds:
```

Thread `platform` through any call signature that's needed (likely `analyze(model_dir, db_path, top_k, platform="polymarket")`).

There's no test for `scripts/analyze_model.py`; verify by:

```bash
uv run python scripts/analyze_model.py --help 2>&1 | grep -A1 platform
```

Expected output: a `--platform` line in the help text.

- [ ] **Step 3: Lint check**

Run: `uv run ruff check scripts/analyze_model.py && uv run ruff format --check scripts/analyze_model.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_model.py
git commit -m "feat(scripts): --platform flag on analyze_model.py"
```

---

### Task 11: New behavioral tests against a mixed-platform corpus

**Files:**
- Create: `tests/ml/test_streaming_platform_filter.py`

- [ ] **Step 1: Write the new test file**

Create `tests/ml/test_streaming_platform_filter.py`:

```python
"""Behavioral tests for the streaming platform filter (RFC #35 follow-up).

Each test seeds a corpus with both polymarket and kalshi rows, then verifies
that ``open_dataset`` and the underlying ``_SplitIter`` honor the platform
parameter — no cross-platform leakage in either direction.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pscanner.ml.streaming import open_dataset


def _build_two_platform_db(
    builder: Callable[..., Path],
    *,
    n_markets: int = 10,
    rows_per_market: int = 4,
) -> Path:
    """Seed n_markets * 2 markets — half polymarket, half kalshi."""
    db = builder(
        n_markets=n_markets,
        rows_per_market=rows_per_market,
        seed=0,
        platform="polymarket",
    )
    builder(
        n_markets=n_markets,
        rows_per_market=rows_per_market,
        seed=1,
        platform="kalshi",
        db_path=db,
    )
    return db


def test_open_dataset_defaults_to_polymarket(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """No platform kwarg => row counts equal the polymarket subset only."""
    db = _build_two_platform_db(
        make_synthetic_examples_db, n_markets=10, rows_per_market=4
    )
    with open_dataset(db) as ds:
        total = ds.n_train_rows + ds.n_val_rows + ds.n_test_rows
    # Polymarket has 10 markets x 4 rows = 40 rows.
    assert total == 40


def test_open_dataset_filters_to_kalshi(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """`platform='kalshi'` => row counts equal the kalshi subset only."""
    db = _build_two_platform_db(
        make_synthetic_examples_db, n_markets=10, rows_per_market=4
    )
    with open_dataset(db, platform="kalshi") as ds:
        total = ds.n_train_rows + ds.n_val_rows + ds.n_test_rows
    assert total == 40


def test_split_iter_does_not_leak_other_platform(
    make_synthetic_examples_db: Callable[..., Path],
) -> None:
    """Iterating dtrain on a polymarket-opened dataset never yields kalshi rows.

    Because the synthetic frame uses different ``condition_id`` namespaces per
    seed (`0xmarket000` etc. for both, but with platform-disjoint membership
    in market_resolutions), the row-count equality is the proxy for no-leak:
    if the WHERE clause leaked, the materialized X would have more rows than
    `n_train_rows`.
    """
    db = _build_two_platform_db(
        make_synthetic_examples_db, n_markets=10, rows_per_market=4
    )
    with open_dataset(db, platform="polymarket") as ds:
        dtrain = ds.dtrain(device="cpu")
    # Polymarket has 10 markets x 4 rows = 40 rows; train fraction is 0.7.
    assert dtrain.num_row() == ds.n_train_rows
    expected = round(0.7 * 40)
    assert dtrain.num_row() == expected
```

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/ml/test_streaming_platform_filter.py -v`
Expected: all 3 pass on first run because the prior tasks already wired up the platform filter end-to-end.

If a test fails: re-read the failure carefully. Common cause — the synthetic frame uses the same `condition_id` strings across seeds (e.g., `0xmarket000`), so polymarket and kalshi share keys. After RFC #35 PR A, that's a feature (composite PK allows it). But the test relies on `_partition_markets` filtering at SELECT time — which it does. If the count is off by 2x, the WHERE clause is missing somewhere; bisect by re-running each Task's test.

- [ ] **Step 3: Commit**

```bash
git add tests/ml/test_streaming_platform_filter.py
git commit -m "test(ml): platform-filter behavioral tests on mixed-platform corpus"
```

---

### Task 12: Forward-compat markers, `_NEVER_LOAD_COLS` docstring, CLAUDE.md note, final verify

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py:154-164` (the `_NEVER_LOAD_COLS` comment block)
- Modify: `src/pscanner/ml/streaming.py` (`_populate_temp_table` docstring)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Refresh the `_NEVER_LOAD_COLS` comment**

In `src/pscanner/ml/preprocessing.py`, find the comment block on `_NEVER_LOAD_COLS` (around lines 154-164). Replace it with:

```python
# Columns excluded at SELECT time so the full-fat DataFrame is never
# materialized. ``id`` is the autoincrement primary key, useless for
# training. ``platform`` is RFC #35 PR A's cross-platform tag; the
# streaming pipeline filters by platform via ``WHERE te.platform = ?``
# (see ``pscanner.ml.streaming.open_dataset``); the column is excluded
# at SELECT time because within a single training run it is a constant.
# Multi-platform aggregation (a future follow-up) would remove
# ``platform`` from this set and add it to ``CATEGORICAL_COLS`` so the
# encoder can one-hot it.
# ``LEAKAGE_COLS`` are dropped downstream by ``drop_leakage_cols``
# anyway — excluding them at the SQL boundary avoids loading ~1+ GB of
# hex-string columns (tx_hash, asset_id, wallet_address are 42-66 chars
# x 5M rows each) only to drop them.
_NEVER_LOAD_COLS: frozenset[str] = frozenset({"id", "platform", *LEAKAGE_COLS})
```

- [ ] **Step 2: Add forward-comp marker on `_populate_temp_table`**

In `src/pscanner/ml/streaming.py`, replace the `_populate_temp_table` docstring with:

```python
def _populate_temp_table(
    conn: sqlite3.Connection,
    table_name: str,
    condition_ids: frozenset[str],
) -> None:
    """Create + populate a per-connection TEMP TABLE for split-membership joins.

    Used in place of an ``IN (?, ?, ...)`` parameterized query so the
    SQLite ``SQLITE_MAX_VARIABLE_NUMBER`` limit (32766 in 3.32+) doesn't
    bite as the corpus grows. Cost: ~10K INSERTs at startup, < 100 ms.

    Single-platform note: the table stores ``condition_id`` only because
    each training run is single-platform. Multi-platform aggregation
    (future follow-up) needs ``(platform, condition_id)`` tuples here and
    a corresponding JOIN tightening to ``USING (platform, condition_id)``
    in ``_SplitIter`` / ``val_aux`` / ``materialize_test``.
    """
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TEMP TABLE {table_name} (condition_id TEXT PRIMARY KEY)")
    conn.executemany(
        f"INSERT INTO {table_name} VALUES (?)",  # noqa: S608 -- table_name is a module-internal literal
        [(cid,) for cid in condition_ids],
    )
```

- [ ] **Step 3: Refresh the CLAUDE.md follow-up bullet**

In `/home/macph/projects/polymarketScanner/CLAUDE.md`, find the bullet that begins with `- **ML pipeline platform filter.** RFC PR A added the \`platform\` column...`. Replace it with:

```markdown
- **ML pipeline platform filter.** `pscanner ml train --platform <name>` filters training to a single platform (default `polymarket`); the streaming `open_dataset` threads the parameter through `_partition_markets`, `_fit_encoder_on_train`, `_count_split_rows`, and `_SplitIter` (whose JOIN to `market_resolutions` uses the composite `(platform, condition_id)` key). Single-platform per run by design; multi-platform aggregation (`--platform all` or comma-sep subset) is the next follow-up — the SQL idioms and parameter shape preserve forward compat (widen `platform: str` to `tuple[str, ...]`, swap `=` for `IN`).
```

- [ ] **Step 4: Run the full quick-verify**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: clean. ty count should match the post-PR-A baseline (13 diagnostics in tracked files; the ML-side changes here shouldn't introduce new ones because everything is keyword arguments with defaults).

If `ruff format --check` reports differences, run `uv run ruff format .` and commit those changes alongside.

If `ty check` reports new diagnostics, investigate. Most likely cause: the `tuple[str, ...] | None` defaults from `accepted_categories` plus `platform: str` don't compose cleanly in some annotation; promote to explicit named keyword arguments or use `Final` if needed.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/preprocessing.py src/pscanner/ml/streaming.py CLAUDE.md
git commit -m "docs: refresh ML platform-filter comments + CLAUDE.md note"
```

---

## Self-review

**Spec coverage:**
- ✅ CLI flag added → Task 9
- ✅ `_cmd_train` forwards → Task 9
- ✅ `run_study` accepts platform → Task 8
- ✅ `open_dataset` accepts platform → Task 7
- ✅ `_partition_markets` filters → Task 2
- ✅ `_fit_encoder_on_train` filters → Task 3
- ✅ `_count_split_rows` filters → Task 4
- ✅ `_SplitIter` composite-key JOIN + WHERE → Task 5
- ✅ `StreamingDataset.val_aux` adds WHERE → Task 6
- ✅ `StreamingDataset.materialize_test` passes platform to `_SplitIter` + WHERE on aux SELECT → Task 6
- ✅ `scripts/analyze_model.py` flag → Task 10
- ✅ Test fixture extension → Task 1
- ✅ Behavioral tests on mixed-platform corpus → Task 11
- ✅ Parser tests → Task 9
- ✅ `_NEVER_LOAD_COLS` docstring → Task 12
- ✅ Forward-comp markers → Task 12
- ✅ CLAUDE.md note refresh → Task 12

**Placeholder scan:** No "TBD" or "implement later" anywhere. The training.py preprocessor.json hook in Task 8 says "Find the existing writer" without pasting it — the writer's exact line varies by post-#39 file shape. Acceptable because the implementer is told what to add (`"platform": platform` key) and how to verify (the test reads the written JSON). If the writer doesn't yet exist as a `dict.dump`, that's fine — Task 8 introduces the write.

**Type consistency:** `platform: str = "polymarket"` is used uniformly across all signatures. Tests and assertions use the same string literals.

---

## Out of scope

- Multi-platform aggregation (`--platform all` or comma-separated subset) — explicit future follow-up.
- Tightening latent `USING (condition_id)` joins in `pscanner.corpus.cli` / `onchain_*.py` / `subgraph_ingest.py` — flagged in PR A's final review (I1/I2); independent corpus-side hygiene PR.
- Removing `platform` from `_NEVER_LOAD_COLS` — would only matter for multi-platform mode; deferred with the multi-platform aggregation work.
- Changes to `pscanner.corpus.*`, schema, or migrations — none. This PR is ML-side only.

---

## Risks

- **`_SplitIter.platform` default vs. test fixture defaults.** Several existing tests construct `_SplitIter` directly (visible from `tests/ml/test_streaming.py` greps). They rely on `platform` defaulting to `"polymarket"` — same as the synthetic fixture. The default in Task 5 keeps every existing test green.
- **`materialize_test` parallel SELECT for top_categories.** Easy to miss the platform predicate on this auxiliary query. Task 6 explicitly adds it; verify by running the existing `test_materialize_test_returns_unencoded_top_categories` test before and after Task 6 — count should be unchanged for the polymarket-only fixture.
- **`run_study` adding `platform` to `preprocessor.json`.** Loading model artifacts at inference time may key off the `preprocessor.json` schema. Adding a new field is forwards-compatible (loaders that don't read it still work), but check if anything strictly validates the schema before merging — if so, update accordingly.
- **CLI flag default = "polymarket" vs. cross-platform corpus.** Once a future PR adds Kalshi/Manifold rows to the same corpus DB, anyone who omits `--platform` continues training a Polymarket-only model. That's the correct behavior (preserves backward compat for the existing pipeline) but worth a CHANGELOG entry on the multi-platform follow-up PR so people don't get surprised.
