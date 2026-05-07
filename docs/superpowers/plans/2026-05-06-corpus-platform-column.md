# Corpus `platform` column Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five shared corpus tables platform-aware (corpus_markets, corpus_trades, market_resolutions, training_examples, asset_index) so future Kalshi and Manifold rows can coexist with the existing Polymarket corpus.

**Architecture:** Add `platform TEXT NOT NULL CHECK (platform IN ('polymarket','kalshi','manifold'))` as part of the primary key on each shared table. Fresh DBs get the new shape via `_SCHEMA_STATEMENTS`. Existing on-disk corpora migrate via per-table table-copy functions wired into `_apply_migrations`. Dataclass row types and repo methods gain a `platform: str = "polymarket"` field/parameter, preserving every existing caller. ML preprocessing gains a platform filter that defaults to `"polymarket"`.

**Tech Stack:** Python 3.13, SQLite (with WAL), pytest. Quick verify command: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Spec:** `docs/superpowers/specs/2026-05-06-corpus-platform-column-design.md`

---

## File map

- **Modify** `src/pscanner/corpus/db.py` — schema statements + migration helpers (the bulk of the change).
- **Modify** `src/pscanner/corpus/repos.py` — dataclass fields + repo method signatures + SQL.
- **Modify** `src/pscanner/corpus/resolutions.py` — pass `platform` through `record_resolutions`.
- **Modify** `src/pscanner/corpus/examples.py` — thread `platform` through to `TrainingExample`.
- **Modify** `src/pscanner/ml/preprocessing.py` — add `platform` parameter to `load_dataset`.
- **Modify** `src/pscanner/ml/cli.py` — add `--platform` flag, forward to `load_dataset`.
- **Modify** `tests/corpus/test_db.py` — update existing assertions for new schema; add migration tests.
- **Modify** `tests/corpus/test_repos_markets.py` / `test_repos_trades.py` / `test_repos_resolutions.py` / `test_repos_examples.py` / `test_repos_asset_index.py` — add cross-platform insert tests.
- **Modify** `tests/corpus/test_resolutions.py` — confirm platform threads through.
- **Modify** `tests/ml/test_cli.py` — add `--platform` flag test.
- **Create** `tests/ml/test_preprocessing.py` (if not present; otherwise modify) — add platform-filter test.
- **Modify** `CLAUDE.md` — short note on platform column convention.

---

### Task 1: Fresh-DB schema with `platform` on all five tables

**Files:**
- Modify: `src/pscanner/corpus/db.py:14-131` (_SCHEMA_STATEMENTS)
- Modify: `tests/corpus/test_db.py:51-60, 85-113` (update assertions for new PK shapes)

- [ ] **Step 1: Write the new failing test for fresh-DB `corpus_markets` shape**

In `tests/corpus/test_db.py`, modify `test_init_corpus_db_creates_asset_index_table` to also cover the new `corpus_markets` shape, and add a new test:

```python
def test_init_corpus_db_corpus_markets_has_platform_pk() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        info = conn.execute("PRAGMA table_info(corpus_markets)").fetchall()
        cols = {row[1] for row in info}
        assert "platform" in cols
        platform_row = next(r for r in info if r[1] == "platform")
        assert platform_row[3] == 1, "platform must be NOT NULL"
        pk_cols = sorted([row[1] for row in info if row[5] > 0])
        assert pk_cols == ["condition_id", "platform"]
    finally:
        conn.close()
```

Add parallel tests for the other four tables: `corpus_trades` (PK `platform, tx_hash, asset_id, wallet_address`), `market_resolutions` (PK `platform, condition_id`), `training_examples` (PK `platform, tx_hash, asset_id, wallet_address` — note: drop the existing `id INTEGER PRIMARY KEY AUTOINCREMENT` column entirely; the composite is the new PK), `asset_index` (PK `platform, asset_id`).

- [ ] **Step 2: Run new tests, expect failures**

Run: `uv run pytest tests/corpus/test_db.py -v`
Expected: 5 new tests fail because `platform` column doesn't exist yet.

- [ ] **Step 3: Update `_SCHEMA_STATEMENTS` to add platform + composite PK + CHECK on all five tables**

Replace the five CREATE TABLE blocks in `src/pscanner/corpus/db.py:14-131`. The full new schema:

```python
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS corpus_markets (
      platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
      condition_id TEXT NOT NULL,
      event_slug TEXT NOT NULL,
      category TEXT,
      closed_at INTEGER NOT NULL,
      total_volume_usd REAL NOT NULL,
      backfill_state TEXT NOT NULL,
      last_offset_seen INTEGER,
      trades_pulled_count INTEGER NOT NULL DEFAULT 0,
      truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
      error_message TEXT,
      enumerated_at INTEGER NOT NULL,
      backfill_started_at INTEGER,
      backfill_completed_at INTEGER,
      market_slug TEXT,
      onchain_trades_count INTEGER,
      onchain_processed_at INTEGER,
      PRIMARY KEY (platform, condition_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corpus_markets_state ON corpus_markets(backfill_state)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_markets_volume ON corpus_markets(total_volume_usd DESC)",
    """
    CREATE TABLE IF NOT EXISTS corpus_trades (
      platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
      tx_hash TEXT NOT NULL,
      asset_id TEXT NOT NULL,
      wallet_address TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      outcome_side TEXT NOT NULL,
      bs TEXT NOT NULL,
      price REAL NOT NULL,
      size REAL NOT NULL,
      notional_usd REAL NOT NULL,
      ts INTEGER NOT NULL,
      PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_market_ts ON corpus_trades(condition_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_wallet_ts ON corpus_trades(wallet_address, ts)",
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_platform_ts_tx_asset "
    "ON corpus_trades(platform, ts, tx_hash, asset_id)",
    """
    CREATE TABLE IF NOT EXISTS market_resolutions (
      platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
      condition_id TEXT NOT NULL,
      winning_outcome_index INTEGER NOT NULL,
      outcome_yes_won INTEGER NOT NULL,
      resolved_at INTEGER NOT NULL,
      source TEXT NOT NULL,
      recorded_at INTEGER NOT NULL,
      PRIMARY KEY (platform, condition_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_examples (
      platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
      tx_hash TEXT NOT NULL,
      asset_id TEXT NOT NULL,
      wallet_address TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      trade_ts INTEGER NOT NULL,
      built_at INTEGER NOT NULL,
      prior_trades_count INTEGER NOT NULL,
      prior_buys_count INTEGER NOT NULL,
      prior_resolved_buys INTEGER NOT NULL,
      prior_wins INTEGER NOT NULL,
      prior_losses INTEGER NOT NULL,
      win_rate REAL,
      avg_implied_prob_paid REAL,
      realized_edge_pp REAL,
      prior_realized_pnl_usd REAL NOT NULL DEFAULT 0,
      avg_bet_size_usd REAL,
      median_bet_size_usd REAL,
      wallet_age_days REAL NOT NULL,
      seconds_since_last_trade INTEGER,
      prior_trades_30d INTEGER NOT NULL,
      top_category TEXT,
      category_diversity INTEGER NOT NULL,
      bet_size_usd REAL NOT NULL,
      bet_size_rel_to_avg REAL,
      edge_confidence_weighted REAL NOT NULL DEFAULT 0,
      win_rate_confidence_weighted REAL NOT NULL DEFAULT 0,
      is_high_quality_wallet INTEGER NOT NULL DEFAULT 0,
      bet_size_relative_to_history REAL NOT NULL DEFAULT 1,
      side TEXT NOT NULL,
      implied_prob_at_buy REAL NOT NULL,
      market_category TEXT NOT NULL,
      market_volume_so_far_usd REAL NOT NULL,
      market_unique_traders_so_far INTEGER NOT NULL,
      market_age_seconds INTEGER NOT NULL,
      time_to_resolution_seconds INTEGER,
      last_trade_price REAL,
      price_volatility_recent REAL,
      label_won INTEGER NOT NULL,
      PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_training_examples_condition ON training_examples(condition_id)",
    "CREATE INDEX IF NOT EXISTS idx_training_examples_wallet ON training_examples(wallet_address)",
    "CREATE INDEX IF NOT EXISTS idx_training_examples_label ON training_examples(label_won)",
    """
    CREATE TABLE IF NOT EXISTS corpus_state (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS asset_index (
      platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
      asset_id TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      outcome_side TEXT NOT NULL,
      outcome_index INTEGER NOT NULL,
      PRIMARY KEY (platform, asset_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_asset_index_condition ON asset_index(condition_id)",
)
```

Notes:
- `training_examples` loses its `id INTEGER PRIMARY KEY AUTOINCREMENT` column. The composite PK replaces it. Search the codebase for any reader that reads `id`: `rg "training_examples" -l` then grep each for `\.id\b` or `SELECT id` — there should be none (only `_NEVER_LOAD_COLS` references `id`, which still works because it's an unused column).
- The Phase 2 `onchain_trades_count` and `onchain_processed_at` columns are now inlined into the `CREATE TABLE corpus_markets` statement. The corresponding `_MIGRATIONS` entries become no-ops on fresh DBs but stay valid for old DBs that haven't yet hit those migrations.
- `idx_corpus_trades_ts_tx_asset` is renamed `idx_corpus_trades_platform_ts_tx_asset` and prefixed with `platform` so per-platform `iter_chronological` is index-friendly.

Update existing tests in `tests/corpus/test_db.py` that assert old shapes:

`test_init_corpus_db_creates_asset_index_table` — change `assert pk_cols == ["asset_id"]` to `assert pk_cols == ["asset_id", "platform"]` (PRAGMA returns PK columns in column-position order; verify with a print first).

`test_corpus_trades_unique_key` — the old test inserted twice and expected `IntegrityError`. The new shape has `PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)` so the double-insert still raises `IntegrityError`. Add `'polymarket'` as the first VALUES element on both inserts.

- [ ] **Step 4: Run all `tests/corpus/test_db.py` tests, all pass**

Run: `uv run pytest tests/corpus/test_db.py -v`
Expected: all tests pass, including the 5 new platform PK tests.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/db.py tests/corpus/test_db.py
git commit -m "feat(corpus): platform column + composite PK on shared tables (fresh-DB path)"
```

---

### Task 2: Migration path for existing on-disk corpora

**Files:**
- Modify: `src/pscanner/corpus/db.py:139-171` (helpers + `_apply_migrations`)
- Modify: `tests/corpus/test_db.py` (add migration tests)

- [ ] **Step 1: Write the failing migration test**

Add to `tests/corpus/test_db.py`:

```python
def test_apply_migrations_adds_platform_to_existing_corpus() -> None:
    """A pre-existing on-disk corpus (old schema) gets migrated in place
    with every existing row backfilled to platform='polymarket'."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "old.sqlite3"
        # Build a pre-PR-A DB by hand using the OLD schema.
        old_conn = sqlite3.connect(str(db_path))
        old_conn.row_factory = sqlite3.Row
        old_conn.executescript(
            """
            CREATE TABLE corpus_markets (
              condition_id TEXT PRIMARY KEY,
              event_slug TEXT NOT NULL,
              category TEXT,
              closed_at INTEGER NOT NULL,
              total_volume_usd REAL NOT NULL,
              backfill_state TEXT NOT NULL,
              last_offset_seen INTEGER,
              trades_pulled_count INTEGER NOT NULL DEFAULT 0,
              truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
              error_message TEXT,
              enumerated_at INTEGER NOT NULL,
              backfill_started_at INTEGER,
              backfill_completed_at INTEGER,
              market_slug TEXT,
              onchain_trades_count INTEGER,
              onchain_processed_at INTEGER
            );
            CREATE TABLE corpus_trades (
              tx_hash TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              wallet_address TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              bs TEXT NOT NULL,
              price REAL NOT NULL,
              size REAL NOT NULL,
              notional_usd REAL NOT NULL,
              ts INTEGER NOT NULL,
              UNIQUE(tx_hash, asset_id, wallet_address)
            );
            CREATE TABLE market_resolutions (
              condition_id TEXT PRIMARY KEY,
              winning_outcome_index INTEGER NOT NULL,
              outcome_yes_won INTEGER NOT NULL,
              resolved_at INTEGER NOT NULL,
              source TEXT NOT NULL,
              recorded_at INTEGER NOT NULL
            );
            CREATE TABLE asset_index (
              asset_id TEXT PRIMARY KEY,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              outcome_index INTEGER NOT NULL
            );
            INSERT INTO corpus_markets(condition_id, event_slug, closed_at, total_volume_usd,
                                       backfill_state, enumerated_at)
              VALUES ('cond1', 'slug1', 1000, 5_000_000.0, 'complete', 999);
            INSERT INTO corpus_trades VALUES
              ('0xtx', 'asset1', '0xw', 'cond1', 'YES', 'BUY', 0.5, 100.0, 50.0, 1000);
            INSERT INTO market_resolutions VALUES ('cond1', 0, 1, 1500, 'gamma', 1500);
            INSERT INTO asset_index VALUES ('asset1', 'cond1', 'YES', 0);
            """
        )
        old_conn.commit()
        old_conn.close()

        # init_corpus_db should detect the missing `platform` column and migrate.
        conn = init_corpus_db(db_path)
        try:
            for table in ("corpus_markets", "corpus_trades", "market_resolutions", "asset_index"):
                rows = conn.execute(f"SELECT platform FROM {table}").fetchall()
                assert len(rows) == 1, f"{table}: expected 1 row, got {len(rows)}"
                assert rows[0]["platform"] == "polymarket"
            # PK is composite
            info = conn.execute("PRAGMA table_info(corpus_markets)").fetchall()
            pk_cols = sorted([r[1] for r in info if r[5] > 0])
            assert pk_cols == ["condition_id", "platform"]
        finally:
            conn.close()
```

Note: `training_examples` is omitted from this test because old-shape `training_examples` had an `id` autoincrement column; covering its migration is task 2's same test — add identical INSERT/assertion blocks for `training_examples` too. (The plan calls for migrating all five, so test all five.)

- [ ] **Step 2: Run the test, expect failure**

Run: `uv run pytest tests/corpus/test_db.py::test_apply_migrations_adds_platform_to_existing_corpus -v`
Expected: fail. Probably an `OperationalError` because `init_corpus_db` runs `CREATE TABLE IF NOT EXISTS` first (which is a no-op on the existing tables), then `_apply_migrations` runs old `ALTER TABLE` statements but does no PK migration.

- [ ] **Step 3: Add `_column_exists` helper and the per-table migration functions**

In `src/pscanner/corpus/db.py`, add above `_apply_migrations`:

```python
def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if ``table`` has a column named ``column``."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


def _migrate_corpus_markets_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "corpus_markets", "platform"):
        return
    with conn:
        conn.execute(
            """
            CREATE TABLE corpus_markets__new (
              platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              condition_id TEXT NOT NULL,
              event_slug TEXT NOT NULL,
              category TEXT,
              closed_at INTEGER NOT NULL,
              total_volume_usd REAL NOT NULL,
              backfill_state TEXT NOT NULL,
              last_offset_seen INTEGER,
              trades_pulled_count INTEGER NOT NULL DEFAULT 0,
              truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
              error_message TEXT,
              enumerated_at INTEGER NOT NULL,
              backfill_started_at INTEGER,
              backfill_completed_at INTEGER,
              market_slug TEXT,
              onchain_trades_count INTEGER,
              onchain_processed_at INTEGER,
              PRIMARY KEY (platform, condition_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO corpus_markets__new (
              platform, condition_id, event_slug, category, closed_at,
              total_volume_usd, backfill_state, last_offset_seen,
              trades_pulled_count, truncated_at_offset_cap, error_message,
              enumerated_at, backfill_started_at, backfill_completed_at,
              market_slug, onchain_trades_count, onchain_processed_at
            )
            SELECT
              'polymarket', condition_id, event_slug, category, closed_at,
              total_volume_usd, backfill_state, last_offset_seen,
              trades_pulled_count, truncated_at_offset_cap, error_message,
              enumerated_at, backfill_started_at, backfill_completed_at,
              market_slug, onchain_trades_count, onchain_processed_at
            FROM corpus_markets
            """
        )
        conn.execute("DROP TABLE corpus_markets")
        conn.execute("ALTER TABLE corpus_markets__new RENAME TO corpus_markets")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_markets_state "
            "ON corpus_markets(backfill_state)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_markets_volume "
            "ON corpus_markets(total_volume_usd DESC)"
        )


def _migrate_corpus_trades_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "corpus_trades", "platform"):
        return
    with conn:
        conn.execute(
            """
            CREATE TABLE corpus_trades__new (
              platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              tx_hash TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              wallet_address TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              bs TEXT NOT NULL,
              price REAL NOT NULL,
              size REAL NOT NULL,
              notional_usd REAL NOT NULL,
              ts INTEGER NOT NULL,
              PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO corpus_trades__new (
              platform, tx_hash, asset_id, wallet_address, condition_id,
              outcome_side, bs, price, size, notional_usd, ts
            )
            SELECT
              'polymarket', tx_hash, asset_id, wallet_address, condition_id,
              outcome_side, bs, price, size, notional_usd, ts
            FROM corpus_trades
            """
        )
        conn.execute("DROP TABLE corpus_trades")
        conn.execute("ALTER TABLE corpus_trades__new RENAME TO corpus_trades")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_trades_market_ts "
            "ON corpus_trades(condition_id, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_trades_wallet_ts "
            "ON corpus_trades(wallet_address, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_trades_platform_ts_tx_asset "
            "ON corpus_trades(platform, ts, tx_hash, asset_id)"
        )


def _migrate_market_resolutions_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "market_resolutions", "platform"):
        return
    with conn:
        conn.execute(
            """
            CREATE TABLE market_resolutions__new (
              platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              condition_id TEXT NOT NULL,
              winning_outcome_index INTEGER NOT NULL,
              outcome_yes_won INTEGER NOT NULL,
              resolved_at INTEGER NOT NULL,
              source TEXT NOT NULL,
              recorded_at INTEGER NOT NULL,
              PRIMARY KEY (platform, condition_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO market_resolutions__new (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            )
            SELECT
              'polymarket', condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            FROM market_resolutions
            """
        )
        conn.execute("DROP TABLE market_resolutions")
        conn.execute("ALTER TABLE market_resolutions__new RENAME TO market_resolutions")


def _migrate_training_examples_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "training_examples", "platform"):
        return
    with conn:
        conn.execute(
            """
            CREATE TABLE training_examples__new (
              platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              tx_hash TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              wallet_address TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              trade_ts INTEGER NOT NULL,
              built_at INTEGER NOT NULL,
              prior_trades_count INTEGER NOT NULL,
              prior_buys_count INTEGER NOT NULL,
              prior_resolved_buys INTEGER NOT NULL,
              prior_wins INTEGER NOT NULL,
              prior_losses INTEGER NOT NULL,
              win_rate REAL,
              avg_implied_prob_paid REAL,
              realized_edge_pp REAL,
              prior_realized_pnl_usd REAL NOT NULL DEFAULT 0,
              avg_bet_size_usd REAL,
              median_bet_size_usd REAL,
              wallet_age_days REAL NOT NULL,
              seconds_since_last_trade INTEGER,
              prior_trades_30d INTEGER NOT NULL,
              top_category TEXT,
              category_diversity INTEGER NOT NULL,
              bet_size_usd REAL NOT NULL,
              bet_size_rel_to_avg REAL,
              edge_confidence_weighted REAL NOT NULL DEFAULT 0,
              win_rate_confidence_weighted REAL NOT NULL DEFAULT 0,
              is_high_quality_wallet INTEGER NOT NULL DEFAULT 0,
              bet_size_relative_to_history REAL NOT NULL DEFAULT 1,
              side TEXT NOT NULL,
              implied_prob_at_buy REAL NOT NULL,
              market_category TEXT NOT NULL,
              market_volume_so_far_usd REAL NOT NULL,
              market_unique_traders_so_far INTEGER NOT NULL,
              market_age_seconds INTEGER NOT NULL,
              time_to_resolution_seconds INTEGER,
              last_trade_price REAL,
              price_volatility_recent REAL,
              label_won INTEGER NOT NULL,
              PRIMARY KEY (platform, tx_hash, asset_id, wallet_address)
            )
            """
        )
        # Note: drop the old `id` autoincrement column at copy time. None of the
        # readers use it (LEAKAGE_COLS / _NEVER_LOAD_COLS already exclude it).
        conn.execute(
            """
            INSERT INTO training_examples__new (
              platform, tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
              prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, win_rate, avg_implied_prob_paid,
              realized_edge_pp, prior_realized_pnl_usd,
              avg_bet_size_usd, median_bet_size_usd, wallet_age_days,
              seconds_since_last_trade, prior_trades_30d, top_category,
              category_diversity, bet_size_usd, bet_size_rel_to_avg,
              edge_confidence_weighted, win_rate_confidence_weighted,
              is_high_quality_wallet, bet_size_relative_to_history,
              side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds,
              time_to_resolution_seconds, last_trade_price, price_volatility_recent,
              label_won
            )
            SELECT
              'polymarket', tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
              prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, win_rate, avg_implied_prob_paid,
              realized_edge_pp, prior_realized_pnl_usd,
              avg_bet_size_usd, median_bet_size_usd, wallet_age_days,
              seconds_since_last_trade, prior_trades_30d, top_category,
              category_diversity, bet_size_usd, bet_size_rel_to_avg,
              edge_confidence_weighted, win_rate_confidence_weighted,
              is_high_quality_wallet, bet_size_relative_to_history,
              side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds,
              time_to_resolution_seconds, last_trade_price, price_volatility_recent,
              label_won
            FROM training_examples
            """
        )
        conn.execute("DROP TABLE training_examples")
        conn.execute("ALTER TABLE training_examples__new RENAME TO training_examples")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_examples_condition "
            "ON training_examples(condition_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_examples_wallet "
            "ON training_examples(wallet_address)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_examples_label "
            "ON training_examples(label_won)"
        )


def _migrate_asset_index_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "asset_index", "platform"):
        return
    with conn:
        conn.execute(
            """
            CREATE TABLE asset_index__new (
              platform TEXT NOT NULL CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              asset_id TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              outcome_index INTEGER NOT NULL,
              PRIMARY KEY (platform, asset_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO asset_index__new (
              platform, asset_id, condition_id, outcome_side, outcome_index
            )
            SELECT
              'polymarket', asset_id, condition_id, outcome_side, outcome_index
            FROM asset_index
            """
        )
        conn.execute("DROP TABLE asset_index")
        conn.execute("ALTER TABLE asset_index__new RENAME TO asset_index")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_asset_index_condition "
            "ON asset_index(condition_id)"
        )
```

Wire all five into `_apply_migrations` (run BEFORE the existing `_MIGRATIONS` ALTER statements so the additive migrations target the new schema):

```python
def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply migrations. Idempotent.

    Runs the platform-column migrations first (which copy old tables into
    new ones with composite PKs), then the additive ALTER TABLE migrations
    in ``_MIGRATIONS``. The platform migrations are idempotent via
    ``_column_exists`` checks; the additive ones swallow ``duplicate column
    name`` / ``no such column`` errors.
    """
    _migrate_corpus_markets_add_platform(conn)
    _migrate_corpus_trades_add_platform(conn)
    _migrate_market_resolutions_add_platform(conn)
    _migrate_training_examples_add_platform(conn)
    _migrate_asset_index_add_platform(conn)
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "duplicate column name" in msg or "no such column" in msg:
                continue
            raise
    conn.commit()
```

Important: `init_corpus_db` runs `_SCHEMA_STATEMENTS` first (CREATE TABLE IF NOT EXISTS). On an old DB, those CREATE statements are no-ops because the tables already exist (with old shapes). Then `_apply_migrations` runs and migrates them. On a fresh DB, the CREATE statements create the new shape directly, and the migration `_column_exists` check returns True immediately, skipping migration. Both paths converge.

- [ ] **Step 4: Run the migration test, all pass**

Run: `uv run pytest tests/corpus/test_db.py -v`
Expected: all tests pass, including the new migration test.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/db.py tests/corpus/test_db.py
git commit -m "feat(corpus): table-copy migration to add platform column to existing DBs"
```

---

### Task 3: Migration idempotency + CHECK constraint tests

**Files:**
- Modify: `tests/corpus/test_db.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/corpus/test_db.py`:

```python
def test_apply_migrations_is_idempotent_on_already_migrated_db() -> None:
    """Calling init_corpus_db twice on a migrated DB is a no-op."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "fresh.sqlite3"
        conn1 = init_corpus_db(db_path)
        conn1.execute(
            """
            INSERT INTO corpus_markets(platform, condition_id, event_slug, closed_at,
                                       total_volume_usd, backfill_state, enumerated_at)
              VALUES ('polymarket', 'cond1', 'slug1', 1000, 5_000_000.0, 'complete', 999)
            """
        )
        conn1.commit()
        conn1.close()
        # Second init must not raise and must preserve the row.
        conn2 = init_corpus_db(db_path)
        try:
            count = conn2.execute("SELECT COUNT(*) AS n FROM corpus_markets").fetchone()["n"]
            assert count == 1
        finally:
            conn2.close()


def test_check_constraint_rejects_invalid_platform() -> None:
    """A row with platform NOT IN ('polymarket','kalshi','manifold') is rejected."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        try:
            conn.execute(
                """
                INSERT INTO corpus_markets(platform, condition_id, event_slug, closed_at,
                                           total_volume_usd, backfill_state, enumerated_at)
                  VALUES ('nonsense', 'cond1', 'slug1', 1000, 5_000_000.0, 'complete', 999)
                """
            )
            conn.commit()
            raise AssertionError("expected CHECK constraint to reject")
        except sqlite3.IntegrityError:
            pass
    finally:
        conn.close()


def test_cross_platform_rows_coexist() -> None:
    """Same condition_id under different platforms = two distinct rows."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        for platform in ("polymarket", "kalshi", "manifold"):
            conn.execute(
                """
                INSERT INTO corpus_markets(platform, condition_id, event_slug, closed_at,
                                           total_volume_usd, backfill_state, enumerated_at)
                  VALUES (?, 'shared-id', 'slug', 1000, 1.0, 'pending', 999)
                """,
                (platform,),
            )
        conn.commit()
        rows = conn.execute(
            "SELECT platform FROM corpus_markets ORDER BY platform"
        ).fetchall()
        assert [r["platform"] for r in rows] == ["kalshi", "manifold", "polymarket"]
    finally:
        conn.close()
```

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/corpus/test_db.py -v`
Expected: all three new tests pass on the first run because the schema and migrations from Tasks 1+2 already support these properties.

If any test fails, debug — the implementation from Tasks 1/2 is incomplete. Common failure: idempotency test fails because `_column_exists` returns False after a successful migration (shouldn't happen — verify the migration committed).

- [ ] **Step 3: Commit**

```bash
git add tests/corpus/test_db.py
git commit -m "test(corpus): platform column idempotency + CHECK + cross-platform coexistence"
```

---

### Task 4: `CorpusMarket` + `CorpusMarketsRepo` platform threading

**Files:**
- Modify: `src/pscanner/corpus/repos.py:16-32` (CorpusMarket dataclass)
- Modify: `src/pscanner/corpus/repos.py:34-206` (CorpusMarketsRepo)
- Modify: `tests/corpus/test_repos_markets.py` (add cross-platform test)

- [ ] **Step 1: Write the failing cross-platform test**

Add to `tests/corpus/test_repos_markets.py`:

```python
def test_repo_isolates_polymarket_and_kalshi(tmp_corpus_db: sqlite3.Connection) -> None:
    """Inserting markets with different platforms keeps them isolated by `platform` arg."""
    from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo

    repo = CorpusMarketsRepo(tmp_corpus_db)
    repo.insert_pending(CorpusMarket(
        condition_id="0xpoly", event_slug="poly-event", category=None,
        closed_at=1000, total_volume_usd=1_000_000.0, enumerated_at=900,
        market_slug="poly-slug", platform="polymarket",
    ))
    repo.insert_pending(CorpusMarket(
        condition_id="KX-1", event_slug="kx-event", category=None,
        closed_at=1100, total_volume_usd=2_000_000.0, enumerated_at=950,
        market_slug="kx-slug", platform="kalshi",
    ))
    poly = repo.next_pending(limit=10, platform="polymarket")
    kalshi = repo.next_pending(limit=10, platform="kalshi")
    assert [m.condition_id for m in poly] == ["0xpoly"]
    assert [m.condition_id for m in kalshi] == ["KX-1"]
    assert all(m.platform == "polymarket" for m in poly)
    assert all(m.platform == "kalshi" for m in kalshi)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_repos_markets.py::test_repo_isolates_polymarket_and_kalshi -v`
Expected: fail — `CorpusMarket` has no `platform` field; `next_pending` has no `platform` parameter.

- [ ] **Step 3: Update dataclass and repo**

In `src/pscanner/corpus/repos.py`, modify `CorpusMarket`:

```python
@dataclass(frozen=True)
class CorpusMarket:
    """A market that qualifies for the corpus (volume gate passed).

    Identifies a closed market by ``(platform, condition_id)``. The
    ``backfill_state`` is tracked separately on the row and progresses
    ``pending → in_progress → complete | failed``.
    """

    condition_id: str
    event_slug: str
    category: str | None
    closed_at: int
    total_volume_usd: float
    enumerated_at: int
    market_slug: str
    platform: str = "polymarket"
```

(Default keeps every existing `CorpusMarket(...)` call site working without edits.)

Update `CorpusMarketsRepo` methods. Each method that scopes to a market gains a `platform: str = "polymarket"` parameter (keyword-only for safety) and embeds it in WHERE / VALUES clauses.

`insert_pending` — read platform from the dataclass row:

```python
def insert_pending(self, market: CorpusMarket) -> int:
    cur = self._conn.execute(
        """
        INSERT OR IGNORE INTO corpus_markets (
          platform, condition_id, event_slug, category, closed_at, total_volume_usd,
          market_slug, backfill_state, enumerated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
        (
            market.platform,
            market.condition_id,
            market.event_slug,
            market.category,
            market.closed_at,
            market.total_volume_usd,
            market.market_slug,
            market.enumerated_at,
        ),
    )
    inserted = cur.rowcount or 0
    self._conn.execute(
        """
        UPDATE corpus_markets
        SET market_slug = ?
        WHERE platform = ? AND condition_id = ? AND market_slug IS NULL
        """,
        (market.market_slug, market.platform, market.condition_id),
    )
    self._conn.commit()
    return inserted
```

`next_pending` — gain `platform` keyword arg, filter, and populate the returned `CorpusMarket.platform`:

```python
def next_pending(self, *, limit: int, platform: str = "polymarket") -> list[CorpusMarket]:
    rows = self._conn.execute(
        """
        SELECT condition_id, event_slug, category, closed_at,
               total_volume_usd, market_slug, enumerated_at
        FROM corpus_markets
        WHERE platform = ? AND backfill_state IN ('pending', 'in_progress', 'failed')
        ORDER BY total_volume_usd DESC, closed_at DESC
        LIMIT ?
        """,
        (platform, limit),
    ).fetchall()
    return [
        CorpusMarket(
            condition_id=row["condition_id"],
            event_slug=row["event_slug"],
            category=row["category"],
            closed_at=row["closed_at"],
            total_volume_usd=row["total_volume_usd"],
            market_slug=row["market_slug"] or "",
            enumerated_at=row["enumerated_at"],
            platform=platform,
        )
        for row in rows
    ]
```

`get_last_offset`, `mark_in_progress`, `record_progress`, `mark_complete`, `mark_failed` — each gains `platform: str = "polymarket"` keyword arg and adds `AND platform = ?` to the WHERE clause. For `mark_complete`, the inner subquery (`SELECT MAX(ts) FROM corpus_trades WHERE condition_id = ?`) also needs `AND platform = ?`. Example:

```python
def mark_complete(
    self,
    condition_id: str,
    *,
    completed_at: int,
    truncated: bool,
    platform: str = "polymarket",
) -> None:
    self._conn.execute(
        """
        UPDATE corpus_markets
        SET backfill_state = 'complete',
            backfill_completed_at = ?,
            truncated_at_offset_cap = ?,
            error_message = NULL,
            closed_at = COALESCE(
                (SELECT MAX(ts) FROM corpus_trades
                 WHERE platform = ? AND condition_id = ?),
                closed_at
            )
        WHERE platform = ? AND condition_id = ?
        """,
        (completed_at, 1 if truncated else 0, platform, condition_id, platform, condition_id),
    )
    self._conn.commit()
```

Apply the same pattern to all the other `CorpusMarketsRepo` mutators.

- [ ] **Step 4: Run all `tests/corpus/test_repos_markets.py` tests, all pass**

Run: `uv run pytest tests/corpus/test_repos_markets.py -v`
Expected: all existing tests pass (defaults preserve behavior) + new cross-platform test passes.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_markets.py
git commit -m "feat(corpus): platform threading in CorpusMarket + CorpusMarketsRepo"
```

---

### Task 5: `CorpusTrade` + `CorpusTradesRepo` platform threading

**Files:**
- Modify: `src/pscanner/corpus/repos.py:246-382`
- Modify: `tests/corpus/test_repos_trades.py`

- [ ] **Step 1: Write the failing cross-platform test**

Add to `tests/corpus/test_repos_trades.py`:

```python
def test_trades_repo_isolates_platforms(tmp_corpus_db: sqlite3.Connection) -> None:
    """Insert trades with different platforms; iter_chronological filters correctly."""
    from pscanner.corpus.repos import CorpusTrade, CorpusTradesRepo

    repo = CorpusTradesRepo(tmp_corpus_db)
    poly = CorpusTrade(
        tx_hash="0xtx", asset_id="poly-asset", wallet_address="0xw",
        condition_id="0xcond", outcome_side="YES", bs="BUY",
        price=0.5, size=200.0, notional_usd=100.0, ts=1000,
        platform="polymarket",
    )
    kalshi = CorpusTrade(
        tx_hash="kx-trade-1", asset_id="KX-1-Y", wallet_address="anon",
        condition_id="KX-1", outcome_side="YES", bs="BUY",
        price=0.5, size=200.0, notional_usd=100.0, ts=1100,
        platform="kalshi",
    )
    assert repo.insert_batch([poly, kalshi]) == 2
    poly_only = list(repo.iter_chronological(platform="polymarket"))
    kalshi_only = list(repo.iter_chronological(platform="kalshi"))
    assert [t.condition_id for t in poly_only] == ["0xcond"]
    assert [t.condition_id for t in kalshi_only] == ["KX-1"]
    assert poly_only[0].platform == "polymarket"
    assert kalshi_only[0].platform == "kalshi"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_repos_trades.py::test_trades_repo_isolates_platforms -v`
Expected: fail.

- [ ] **Step 3: Update dataclass and repo**

In `src/pscanner/corpus/repos.py`, modify `CorpusTrade`:

```python
@dataclass(frozen=True)
class CorpusTrade:
    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    outcome_side: str
    bs: str
    price: float
    size: float
    notional_usd: float
    ts: int
    platform: str = "polymarket"
```

`insert_batch` — read platform from each row:

```python
def insert_batch(self, trades: Iterable[CorpusTrade]) -> int:
    rows = []
    for t in trades:
        if t.notional_usd < _NOTIONAL_FLOOR_USD:
            continue
        rows.append(
            (
                t.platform,
                t.tx_hash,
                t.asset_id,
                t.wallet_address.lower(),
                t.condition_id,
                t.outcome_side,
                t.bs,
                t.price,
                t.size,
                t.notional_usd,
                t.ts,
            )
        )
    if not rows:
        return 0
    cur = self._conn.executemany(
        """
        INSERT OR IGNORE INTO corpus_trades (
          platform, tx_hash, asset_id, wallet_address, condition_id,
          outcome_side, bs, price, size, notional_usd, ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    self._conn.commit()
    return cur.rowcount or 0
```

`iter_chronological` — gain `platform: str = "polymarket"` keyword arg, filter rows, populate the yielded `CorpusTrade.platform`. Use the new index `idx_corpus_trades_platform_ts_tx_asset` for index-friendly iteration:

```python
def iter_chronological(
    self, *, chunk_size: int = 50_000, platform: str = "polymarket"
) -> Iterator[CorpusTrade]:
    last: tuple[int, str, str] | None = None
    while True:
        if last is None:
            rows = self._conn.execute(
                """
                SELECT tx_hash, asset_id, wallet_address, condition_id,
                       outcome_side, bs, price, size, notional_usd, ts
                FROM corpus_trades
                WHERE platform = ?
                ORDER BY ts, tx_hash, asset_id
                LIMIT ?
                """,
                (platform, chunk_size),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT tx_hash, asset_id, wallet_address, condition_id,
                       outcome_side, bs, price, size, notional_usd, ts
                FROM corpus_trades
                WHERE platform = ? AND (ts, tx_hash, asset_id) > (?, ?, ?)
                ORDER BY ts, tx_hash, asset_id
                LIMIT ?
                """,
                (platform, last[0], last[1], last[2], chunk_size),
            ).fetchall()
        if not rows:
            return
        for row in rows:
            yield CorpusTrade(
                tx_hash=row["tx_hash"],
                asset_id=row["asset_id"],
                wallet_address=row["wallet_address"],
                condition_id=row["condition_id"],
                outcome_side=row["outcome_side"],
                bs=row["bs"],
                price=row["price"],
                size=row["size"],
                notional_usd=row["notional_usd"],
                ts=row["ts"],
                platform=platform,
            )
        tail = rows[-1]
        last = (tail["ts"], tail["tx_hash"], tail["asset_id"])
```

- [ ] **Step 4: Run all `tests/corpus/test_repos_trades.py` tests, all pass**

Run: `uv run pytest tests/corpus/test_repos_trades.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_trades.py
git commit -m "feat(corpus): platform threading in CorpusTrade + CorpusTradesRepo"
```

---

### Task 6: `MarketResolution` + `MarketResolutionsRepo` platform threading

**Files:**
- Modify: `src/pscanner/corpus/repos.py:385-463`
- Modify: `tests/corpus/test_repos_resolutions.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/corpus/test_repos_resolutions.py`:

```python
def test_resolutions_repo_isolates_platforms(tmp_corpus_db: sqlite3.Connection) -> None:
    from pscanner.corpus.repos import MarketResolution, MarketResolutionsRepo

    repo = MarketResolutionsRepo(tmp_corpus_db)
    repo.upsert(
        MarketResolution(
            condition_id="0xpoly", winning_outcome_index=0, outcome_yes_won=1,
            resolved_at=1000, source="gamma", platform="polymarket",
        ),
        recorded_at=1000,
    )
    repo.upsert(
        MarketResolution(
            condition_id="KX-1", winning_outcome_index=0, outcome_yes_won=1,
            resolved_at=1100, source="kalshi-rest", platform="kalshi",
        ),
        recorded_at=1100,
    )
    assert repo.get("0xpoly", platform="polymarket") is not None
    assert repo.get("0xpoly", platform="kalshi") is None
    assert repo.get("KX-1", platform="kalshi") is not None
    assert repo.missing_for(["0xpoly", "0xother"], platform="polymarket") == ["0xother"]
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_repos_resolutions.py::test_resolutions_repo_isolates_platforms -v`
Expected: fail.

- [ ] **Step 3: Update dataclass and repo**

```python
@dataclass(frozen=True)
class MarketResolution:
    condition_id: str
    winning_outcome_index: int
    outcome_yes_won: int
    resolved_at: int
    source: str
    platform: str = "polymarket"


class MarketResolutionsRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, resolution: MarketResolution, *, recorded_at: int) -> None:
        self._conn.execute(
            """
            INSERT INTO market_resolutions (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, condition_id) DO UPDATE SET
              winning_outcome_index = excluded.winning_outcome_index,
              outcome_yes_won = excluded.outcome_yes_won,
              resolved_at = excluded.resolved_at,
              source = excluded.source,
              recorded_at = excluded.recorded_at
            """,
            (
                resolution.platform,
                resolution.condition_id,
                resolution.winning_outcome_index,
                resolution.outcome_yes_won,
                resolution.resolved_at,
                resolution.source,
                recorded_at,
            ),
        )
        self._conn.commit()

    def get(
        self, condition_id: str, *, platform: str = "polymarket"
    ) -> MarketResolution | None:
        row = self._conn.execute(
            """
            SELECT condition_id, winning_outcome_index, outcome_yes_won,
                   resolved_at, source
            FROM market_resolutions
            WHERE platform = ? AND condition_id = ?
            """,
            (platform, condition_id),
        ).fetchone()
        if row is None:
            return None
        return MarketResolution(
            condition_id=row["condition_id"],
            winning_outcome_index=row["winning_outcome_index"],
            outcome_yes_won=row["outcome_yes_won"],
            resolved_at=row["resolved_at"],
            source=row["source"],
            platform=platform,
        )

    def missing_for(
        self, condition_ids: Iterable[str], *, platform: str = "polymarket"
    ) -> list[str]:
        ids = list(condition_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"""
            SELECT condition_id FROM market_resolutions
            WHERE platform = ? AND condition_id IN ({placeholders})
            """,  # noqa: S608 — placeholder count is fixed to len(ids)
            (platform, *ids),
        ).fetchall()
        present = {row["condition_id"] for row in rows}
        return [cid for cid in ids if cid not in present]
```

- [ ] **Step 4: Run all `tests/corpus/test_repos_resolutions.py` tests**

Run: `uv run pytest tests/corpus/test_repos_resolutions.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_resolutions.py
git commit -m "feat(corpus): platform threading in MarketResolution + repo"
```

---

### Task 7: `TrainingExample` + `TrainingExamplesRepo` platform threading

**Files:**
- Modify: `src/pscanner/corpus/repos.py:466-603`
- Modify: `tests/corpus/test_repos_examples.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/corpus/test_repos_examples.py`:

```python
def test_examples_repo_isolates_platforms(tmp_corpus_db: sqlite3.Connection) -> None:
    from pscanner.corpus.repos import TrainingExample, TrainingExamplesRepo

    base_kwargs: dict[str, object] = dict(
        condition_id="cond1", trade_ts=1000, built_at=1500,
        prior_trades_count=10, prior_buys_count=8, prior_resolved_buys=5,
        prior_wins=3, prior_losses=2, win_rate=0.6,
        avg_implied_prob_paid=0.5, realized_edge_pp=0.05,
        prior_realized_pnl_usd=10.0, avg_bet_size_usd=50.0,
        median_bet_size_usd=40.0, wallet_age_days=30.0,
        seconds_since_last_trade=3600, prior_trades_30d=5,
        top_category="sports", category_diversity=2, bet_size_usd=100.0,
        bet_size_rel_to_avg=2.0, edge_confidence_weighted=0.025,
        win_rate_confidence_weighted=0.05, is_high_quality_wallet=0,
        bet_size_relative_to_history=2.5, side="YES",
        implied_prob_at_buy=0.5, market_category="sports",
        market_volume_so_far_usd=1_000_000.0,
        market_unique_traders_so_far=500, market_age_seconds=86_400,
        time_to_resolution_seconds=86_400, last_trade_price=0.51,
        price_volatility_recent=0.02, label_won=1,
    )
    repo = TrainingExamplesRepo(tmp_corpus_db)
    repo.insert_or_ignore([
        TrainingExample(
            tx_hash="0xtx", asset_id="a1", wallet_address="0xw",
            **base_kwargs,
            platform="polymarket",
        ),
        TrainingExample(
            tx_hash="kx-1", asset_id="KX-Y", wallet_address="anon",
            **base_kwargs,
            platform="kalshi",
        ),
    ])
    poly_keys = repo.existing_keys(platform="polymarket")
    kalshi_keys = repo.existing_keys(platform="kalshi")
    assert ("0xtx", "a1", "0xw") in poly_keys
    assert ("0xtx", "a1", "0xw") not in kalshi_keys
    assert ("kx-1", "KX-Y", "anon") in kalshi_keys

    repo.truncate(platform="kalshi")
    assert repo.existing_keys(platform="kalshi") == set()
    assert repo.existing_keys(platform="polymarket") == {("0xtx", "a1", "0xw")}
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_repos_examples.py::test_examples_repo_isolates_platforms -v`
Expected: fail.

- [ ] **Step 3: Update dataclass, `_example_to_row`, and repo**

Add `platform: str = "polymarket"` to `TrainingExample`. Update `_example_to_row` to put `platform` first:

```python
def _example_to_row(ex: TrainingExample) -> tuple[object, ...]:
    return (
        ex.platform,
        ex.tx_hash,
        ex.asset_id,
        ex.wallet_address,
        ex.condition_id,
        ex.trade_ts,
        ex.built_at,
        ex.prior_trades_count,
        ex.prior_buys_count,
        ex.prior_resolved_buys,
        ex.prior_wins,
        ex.prior_losses,
        ex.win_rate,
        ex.avg_implied_prob_paid,
        ex.realized_edge_pp,
        ex.prior_realized_pnl_usd,
        ex.avg_bet_size_usd,
        ex.median_bet_size_usd,
        ex.wallet_age_days,
        ex.seconds_since_last_trade,
        ex.prior_trades_30d,
        ex.top_category,
        ex.category_diversity,
        ex.bet_size_usd,
        ex.bet_size_rel_to_avg,
        ex.edge_confidence_weighted,
        ex.win_rate_confidence_weighted,
        ex.is_high_quality_wallet,
        ex.bet_size_relative_to_history,
        ex.side,
        ex.implied_prob_at_buy,
        ex.market_category,
        ex.market_volume_so_far_usd,
        ex.market_unique_traders_so_far,
        ex.market_age_seconds,
        ex.time_to_resolution_seconds,
        ex.last_trade_price,
        ex.price_volatility_recent,
        ex.label_won,
    )
```

Update `insert_or_ignore` to add `platform` to the column list (39 columns total, 39 placeholders):

```python
def insert_or_ignore(self, examples: Iterable[TrainingExample]) -> int:
    rows = [_example_to_row(ex) for ex in examples]
    if not rows:
        return 0
    cur = self._conn.executemany(
        """
        INSERT OR IGNORE INTO training_examples (
          platform, tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
          prior_trades_count, prior_buys_count, prior_resolved_buys,
          prior_wins, prior_losses, win_rate, avg_implied_prob_paid,
          realized_edge_pp, prior_realized_pnl_usd,
          avg_bet_size_usd, median_bet_size_usd, wallet_age_days,
          seconds_since_last_trade, prior_trades_30d, top_category,
          category_diversity, bet_size_usd, bet_size_rel_to_avg,
          edge_confidence_weighted, win_rate_confidence_weighted,
          is_high_quality_wallet, bet_size_relative_to_history,
          side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
          market_unique_traders_so_far, market_age_seconds,
          time_to_resolution_seconds, last_trade_price, price_volatility_recent,
          label_won
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    self._conn.commit()
    return cur.rowcount or 0


def truncate(self, *, platform: str = "polymarket") -> None:
    """Delete every row in ``training_examples`` for the given platform."""
    self._conn.execute("DELETE FROM training_examples WHERE platform = ?", (platform,))
    self._conn.commit()


def existing_keys(self, *, platform: str = "polymarket") -> set[tuple[str, str, str]]:
    """Return (tx_hash, asset_id, wallet_address) for the given platform."""
    rows = self._conn.execute(
        """
        SELECT tx_hash, asset_id, wallet_address
        FROM training_examples WHERE platform = ?
        """,
        (platform,),
    ).fetchall()
    return {(row["tx_hash"], row["asset_id"], row["wallet_address"]) for row in rows}
```

- [ ] **Step 4: Run all `tests/corpus/test_repos_examples.py` tests**

Run: `uv run pytest tests/corpus/test_repos_examples.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_examples.py
git commit -m "feat(corpus): platform threading in TrainingExample + repo"
```

---

### Task 8: `AssetEntry` + `AssetIndexRepo` platform threading

**Files:**
- Modify: `src/pscanner/corpus/repos.py:606-696`
- Modify: `tests/corpus/test_repos_asset_index.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/corpus/test_repos_asset_index.py`:

```python
def test_asset_index_isolates_platforms(tmp_corpus_db: sqlite3.Connection) -> None:
    from pscanner.corpus.repos import AssetEntry, AssetIndexRepo

    repo = AssetIndexRepo(tmp_corpus_db)
    repo.upsert(AssetEntry(
        asset_id="a1", condition_id="0xpoly",
        outcome_side="YES", outcome_index=0,
        platform="polymarket",
    ))
    repo.upsert(AssetEntry(
        asset_id="a1", condition_id="KX-1",
        outcome_side="YES", outcome_index=0,
        platform="kalshi",
    ))
    poly = repo.get("a1", platform="polymarket")
    kalshi = repo.get("a1", platform="kalshi")
    assert poly is not None and poly.condition_id == "0xpoly"
    assert kalshi is not None and kalshi.condition_id == "KX-1"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_repos_asset_index.py::test_asset_index_isolates_platforms -v`

- [ ] **Step 3: Update dataclass and repo**

```python
@dataclass(frozen=True)
class AssetEntry:
    asset_id: str
    condition_id: str
    outcome_side: str
    outcome_index: int
    platform: str = "polymarket"


class AssetIndexRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, entry: AssetEntry) -> None:
        self._conn.execute(
            """
            INSERT INTO asset_index (platform, asset_id, condition_id, outcome_side, outcome_index)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(platform, asset_id) DO UPDATE SET
              condition_id = excluded.condition_id,
              outcome_side = excluded.outcome_side,
              outcome_index = excluded.outcome_index
            """,
            (entry.platform, entry.asset_id, entry.condition_id,
             entry.outcome_side, entry.outcome_index),
        )
        self._conn.commit()

    def get(self, asset_id: str, *, platform: str = "polymarket") -> AssetEntry | None:
        row = self._conn.execute(
            "SELECT asset_id, condition_id, outcome_side, outcome_index "
            "FROM asset_index WHERE platform = ? AND asset_id = ?",
            (platform, asset_id),
        ).fetchone()
        if row is None:
            return None
        return AssetEntry(
            asset_id=row["asset_id"],
            condition_id=row["condition_id"],
            outcome_side=row["outcome_side"],
            outcome_index=row["outcome_index"],
            platform=platform,
        )

    def backfill_from_corpus_trades(self, *, platform: str = "polymarket") -> int:
        """Populate `asset_index` from existing `corpus_trades` rows for ``platform``."""
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO asset_index (
              platform, asset_id, condition_id, outcome_side, outcome_index
            )
            SELECT
              ?, asset_id, condition_id, outcome_side,
              CASE outcome_side WHEN 'YES' THEN 0 ELSE 1 END
            FROM (
              SELECT asset_id, condition_id, outcome_side
              FROM corpus_trades
              WHERE platform = ?
              GROUP BY asset_id
            )
            """,
            (platform, platform),
        )
        inserted = cursor.rowcount
        self._conn.commit()
        return inserted
```

- [ ] **Step 4: Run all `tests/corpus/test_repos_asset_index.py` tests**

Run: `uv run pytest tests/corpus/test_repos_asset_index.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_asset_index.py
git commit -m "feat(corpus): platform threading in AssetEntry + AssetIndexRepo"
```

---

### Task 9: Thread `platform` through `record_resolutions`

**Files:**
- Modify: `src/pscanner/corpus/resolutions.py:42-90`
- Modify: `tests/corpus/test_resolutions.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/corpus/test_resolutions.py` (or update an existing test):

```python
@pytest.mark.asyncio
async def test_record_resolutions_records_platform(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """`record_resolutions` writes the platform onto each MarketResolution row."""
    from pscanner.corpus.repos import MarketResolutionsRepo
    from pscanner.corpus.resolutions import record_resolutions

    repo = MarketResolutionsRepo(tmp_corpus_db)

    class _StubGamma:
        async def get_market_by_slug(self, slug: str):
            from pscanner.poly.models import Market
            # Return a market that resolves YES (price >= 0.99 on outcome 0).
            return Market.model_validate({
                "id": "1", "conditionId": slug, "slug": slug,
                "question": "?", "outcomePrices": '["0.99","0.01"]',
                "outcomes": '["Yes","No"]', "active": False, "closed": True,
            })

    written = await record_resolutions(
        gamma=_StubGamma(),
        repo=repo,
        targets=[("0xpoly", "poly-slug", 1500)],
        now_ts=1500,
    )
    assert written == 1
    row = repo.get("0xpoly", platform="polymarket")
    assert row is not None and row.platform == "polymarket"
```

(If the test setup for `_StubGamma` doesn't quite match the existing `pscanner.poly.models.Market`, adapt — the point is to exercise the platform threading.)

- [ ] **Step 2: Run, expect failure or success**

Run: `uv run pytest tests/corpus/test_resolutions.py::test_record_resolutions_records_platform -v`
Expected: pass on the first try after the prior tasks (`MarketResolution.platform` defaults to `"polymarket"`, so even a `record_resolutions` that doesn't pass the kwarg already records it). If pass, that's fine — the test still anchors the behavior.

If you want tighter assertion, also add a Kalshi-platform call site test that requires `record_resolutions` to accept and pass through a `platform` argument.

- [ ] **Step 3: Update `record_resolutions`**

In `src/pscanner/corpus/resolutions.py`:

```python
async def record_resolutions(
    *,
    gamma: GammaClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, str, int]],
    now_ts: int,
    platform: str = "polymarket",
) -> int:
    """Fetch resolutions and persist them under ``platform``.

    Args:
        platform: The platform tag to write onto every MarketResolution row.
            Defaults to ``"polymarket"`` so the existing call site is
            unchanged.
        ... (rest of the existing docstring)
    """
    written = 0
    for condition_id, slug, resolved_at in targets:
        market = await gamma.get_market_by_slug(slug)
        if market is None:
            _log.warning("corpus.resolution_market_not_found", condition_id=condition_id, slug=slug)
            continue
        yes_won = determine_outcome_yes_won(market)
        if yes_won is None:
            _log.warning("corpus.resolution_disputed", condition_id=condition_id, slug=slug)
            continue
        repo.upsert(
            MarketResolution(
                condition_id=condition_id,
                winning_outcome_index=0 if yes_won == 1 else 1,
                outcome_yes_won=yes_won,
                resolved_at=resolved_at,
                source="gamma",
                platform=platform,
            ),
            recorded_at=now_ts,
        )
        written += 1
    return written
```

- [ ] **Step 4: Run all `tests/corpus/test_resolutions.py` tests**

Run: `uv run pytest tests/corpus/test_resolutions.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/resolutions.py tests/corpus/test_resolutions.py
git commit -m "feat(corpus): platform threading in record_resolutions"
```

---

### Task 10: Thread `platform` through `build_features`

**Files:**
- Modify: `src/pscanner/corpus/examples.py:54-219` (`_example_from_features`, `_maybe_make_example`, `build_features`)
- Modify: `tests/corpus/test_examples.py`

The build-features pipeline currently produces `TrainingExample` rows from `corpus_trades`. After Task 7, `TrainingExample.platform` defaults to `"polymarket"`, so existing call sites still work. But to support future Kalshi/Manifold feature pipelines, `build_features` needs an explicit `platform` parameter that flows down to every `TrainingExample` it constructs and every `iter_chronological()` call it makes.

- [ ] **Step 1: Write the failing test**

In `tests/corpus/test_examples.py`, add (or extend) a test that calls `build_features(..., platform="polymarket")` end-to-end and asserts the produced rows have `platform="polymarket"`. Then call again with `platform="kalshi"` against an empty kalshi corpus slice and assert no rows are produced (because the iter_chronological call returns nothing). The test verifies that `platform` flows through.

The test will use the existing `tmp_corpus_db` fixture, seeded `corpus_trades` and `corpus_markets` rows under both platforms, and assert isolation. Use the structure of the existing tests in this file as a template (look for the longest existing test in `tests/corpus/test_examples.py` and copy its setup).

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/corpus/test_examples.py -v`
Expected: at least one new test fails because `build_features` doesn't accept a `platform` kwarg or doesn't thread it.

- [ ] **Step 3: Add `platform` parameter to `build_features` and threading**

In `src/pscanner/corpus/examples.py`:

- Add `platform: str = "polymarket"` keyword arg to `build_features`.
- Pass `platform=platform` through to `iter_chronological(...)` calls.
- Pass `platform=platform` through to `TrainingExamplesRepo.existing_keys(...)`, `truncate(...)` (if used).
- When constructing `TrainingExample(...)` instances inside `_example_from_features` / `_maybe_make_example`, pass `platform=platform` (likely thread the parameter through these helpers).
- Also thread `platform=platform` into `MarketResolutionsRepo.get(...)` / `missing_for(...)` / `iter_*` calls if any.

The exact signature changes depend on the current shape of the helpers — read `src/pscanner/corpus/examples.py` carefully and mirror the parameter in each function from the call site backward. Default of `"polymarket"` preserves existing callers.

- [ ] **Step 4: Run all `tests/corpus/test_examples.py` tests**

Run: `uv run pytest tests/corpus/test_examples.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/corpus/examples.py tests/corpus/test_examples.py
git commit -m "feat(corpus): platform threading in build_features pipeline"
```

---

### Task 11: ML preprocessing — add `platform` filter to `load_dataset`

**Files:**
- Modify: `src/pscanner/ml/preprocessing.py:214-262`
- Create or modify: `tests/ml/test_preprocessing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ml/test_preprocessing.py` (if it doesn't exist):

```python
"""Tests for ``pscanner.ml.preprocessing.load_dataset``."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.ml.preprocessing import load_dataset


def _seed_two_platforms(conn: sqlite3.Connection) -> None:
    """Insert one resolved training example each under polymarket and kalshi."""
    base = (
        "cond1", 1000, 1500,                                # condition_id, trade_ts, built_at
        10, 8, 5, 3, 2, 0.6, 0.5, 0.05, 10.0, 50.0, 40.0,
        30.0, 3600, 5, "sports", 2, 100.0, 2.0,
        0.025, 0.05, 0, 2.5,
        "YES", 0.5, "sports", 1_000_000.0, 500, 86_400,
        86_400, 0.51, 0.02, 1,
    )
    for platform, tx in (("polymarket", "0xpoly"), ("kalshi", "kx-1")):
        conn.execute(
            """
            INSERT INTO training_examples (
              platform, tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
              prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, win_rate, avg_implied_prob_paid,
              realized_edge_pp, prior_realized_pnl_usd,
              avg_bet_size_usd, median_bet_size_usd, wallet_age_days,
              seconds_since_last_trade, prior_trades_30d, top_category,
              category_diversity, bet_size_usd, bet_size_rel_to_avg,
              edge_confidence_weighted, win_rate_confidence_weighted,
              is_high_quality_wallet, bet_size_relative_to_history,
              side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds,
              time_to_resolution_seconds, last_trade_price, price_volatility_recent,
              label_won
            ) VALUES (?, ?, 'asset1', '0xw', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (platform, tx, *base),
        )
        conn.execute(
            """
            INSERT INTO market_resolutions (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, 'cond1', 0, 1, 2000, 'gamma', 2000)
            ON CONFLICT(platform, condition_id) DO NOTHING
            """,
            (platform,),
        )
    conn.commit()


def test_load_dataset_defaults_to_polymarket() -> None:
    """No `platform` arg => only polymarket rows are returned."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "corpus.sqlite3"
        conn = init_corpus_db(db_path)
        _seed_two_platforms(conn)
        conn.close()
        df = load_dataset(db_path)
    assert df.height == 1


def test_load_dataset_filters_by_platform_kalshi() -> None:
    """`platform='kalshi'` returns only kalshi rows."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "corpus.sqlite3"
        conn = init_corpus_db(db_path)
        _seed_two_platforms(conn)
        conn.close()
        df = load_dataset(db_path, platform="kalshi")
    assert df.height == 1
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_preprocessing.py -v`
Expected: fail — `load_dataset` doesn't accept a `platform` parameter.

- [ ] **Step 3: Update `load_dataset`**

In `src/pscanner/ml/preprocessing.py:214`:

```python
def load_dataset(db_path: Path, *, platform: str = "polymarket") -> pl.DataFrame:
    """Load ``training_examples`` joined with ``market_resolutions.resolved_at``.

    Args:
        db_path: Path to the corpus SQLite file.
        platform: Filter to rows with this platform tag. Defaults to
            ``"polymarket"`` so existing callers see no behavior change.

    ... (rest of docstring unchanged)
    """
    conn = sqlite3.connect(str(db_path))
    try:
        all_cols = [r[1] for r in conn.execute("PRAGMA table_info(training_examples)").fetchall()]
        keep_cols = [c for c in all_cols if c not in _NEVER_LOAD_COLS]
        select_list = ", ".join(f"te.{c}" for c in keep_cols)
        df = pl.read_database(
            query=(
                f"SELECT {select_list}, mr.resolved_at "  # noqa: S608 -- col names from PRAGMA
                "FROM training_examples te "
                "JOIN market_resolutions mr "
                "  ON mr.platform = te.platform AND mr.condition_id = te.condition_id "
                "WHERE te.platform = :platform"
            ),
            connection=conn,
            batch_size=100_000,
            execute_options={"parameters": {"platform": platform}},
        )
    finally:
        conn.close()
    # ... cast_exprs unchanged
```

Notes:
- The JOIN now uses `(platform, condition_id)` because PR A's PK on `market_resolutions` is composite. The old `JOIN market_resolutions mr USING (condition_id)` would join across platforms, which is wrong.
- Polars' `pl.read_database` parameter binding via `execute_options["parameters"]` works for SQLite via `connectorx`/SQLAlchemy backends. If your local Polars version doesn't accept that form, use a string-format approach with strict input validation: `assert platform in ("polymarket", "kalshi", "manifold")` then f-string. (Verify by running the test once and iterating.)

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/ml/test_preprocessing.py -v`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/preprocessing.py tests/ml/test_preprocessing.py
git commit -m "feat(ml): platform filter on load_dataset (default polymarket)"
```

---

### Task 12: ML CLI — add `--platform` flag

**Files:**
- Modify: `src/pscanner/ml/cli.py:21-90`
- Modify: `tests/ml/test_cli.py`

- [ ] **Step 1: Write the failing test**

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
    import pytest as _pytest

    from pscanner.ml.cli import build_ml_parser

    parser = build_ml_parser()
    with _pytest.raises(SystemExit):
        parser.parse_args(["train", "--platform", "ftx", "--db", "x"])
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/ml/test_cli.py -v`

- [ ] **Step 3: Add the flag and forward it**

In `src/pscanner/ml/cli.py`:

```python
def build_ml_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pscanner ml")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train an XGBoost copy-trade gate model")
    # ... existing flags ...
    train.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "kalshi", "manifold"],
        default="polymarket",
        help="Filter training_examples to rows with this platform tag.",
    )
    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        output_dir = Path("models") / f"{today}-copy_trade_gate"
    df = load_dataset(db_path, platform=args.platform)
    _log.info(
        "ml.dataset_loaded",
        rows=df.height,
        cols=len(df.columns),
        platform=args.platform,
        output_dir=str(output_dir),
        rss_mb=_rss_mb(),
    )
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        n_min=args.n_min,
        seed=args.seed,
        device=args.device,
    )
    return 0
```

- [ ] **Step 4: Run new tests + full ml suite**

Run: `uv run pytest tests/ml/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/ml/cli.py tests/ml/test_cli.py
git commit -m "feat(ml): --platform flag on `pscanner ml train`"
```

---

### Task 13: Final verification + CLAUDE.md note

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full quick-verify suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: zero errors, zero warnings, all tests pass. The CLAUDE.md `filterwarnings = ["error"]` setting means any new pytest warning is a fail.

If ruff or ty surfaces issues:
- `ruff format` — run `uv run ruff format .` to fix.
- `ty` — most likely complaint is around the `pl.read_database` parameter binding; if the `execute_options` form isn't accepted by the installed Polars version, fall back to f-string with validated input.

- [ ] **Step 2: Add a CLAUDE.md note**

Add a short paragraph under "Codebase conventions" in `/home/macph/projects/polymarketScanner/CLAUDE.md`:

```markdown
- **`platform` column on shared corpus tables.** `corpus_markets`, `corpus_trades`, `market_resolutions`, `training_examples`, and `asset_index` carry a `platform TEXT NOT NULL CHECK (platform IN ('polymarket','kalshi','manifold'))` column that is part of the composite primary key on each table. Repo methods and dataclass row types take a `platform: str = "polymarket"` parameter/field so existing Polymarket call sites are unchanged. The legacy column `condition_id` holds platform-native market identifiers for non-Polymarket platforms (Kalshi tickers, Manifold market hashes) — it was not renamed at PR A time. ML training filters to one platform at a time via `pscanner ml train --platform <name>`.
```

- [ ] **Step 3: Re-run the quick-verify suite**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: still all pass.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document corpus platform column convention"
```

- [ ] **Step 5: Manual smoke against real corpus (optional but recommended)**

If the user has `data/corpus.sqlite3` on disk:

```bash
cp data/corpus.sqlite3 data/corpus.sqlite3.pre-pra-bak
uv run python -c "from pathlib import Path; from pscanner.corpus.db import init_corpus_db; conn = init_corpus_db(Path('data/corpus.sqlite3')); print(conn.execute('SELECT platform, COUNT(*) FROM corpus_markets GROUP BY platform').fetchall()); conn.close()"
```

Expected output: a single tuple `('polymarket', N)` where N matches the pre-migration row count. The migration runs on the on-disk corpus once, then subsequent calls are no-ops (idempotency check).

If the user is willing, run the same smoke on the desktop training box's corpus.

---

## Self-review

**Spec coverage:**
- ✅ Add `platform` column to 5 shared tables → Task 1 (fresh-DB) + Task 2 (migration)
- ✅ Composite PK including `platform` → Tasks 1, 2
- ✅ CHECK constraint → Tasks 1, 3
- ✅ Existing rows backfill to `'polymarket'` → Task 2
- ✅ Migration is idempotent → Task 3
- ✅ Repo signatures gain `platform` parameter → Tasks 4–8
- ✅ Row dataclasses gain `platform` field → Tasks 4–8
- ✅ `record_resolutions` threads platform → Task 9
- ✅ `build_features` threads platform → Task 10
- ✅ `load_dataset(platform=...)` filter → Task 11
- ✅ `pscanner ml train --platform` → Task 12
- ✅ CLAUDE.md note → Task 13

**Placeholder scan:** Task 10 deliberately leaves the helper-thread-through detail to the implementer because the helper signatures aren't fully visible at plan-write time without re-reading every line of `examples.py`. The task is concrete enough — name the helpers, name the calls, name the kwarg — but doesn't paste the full code. Acceptable.

**Type consistency:** `platform: str = "polymarket"` is used uniformly across all dataclasses, repo methods, and CLI flags. Tests use the same string literals. No drift.

---

## Out of scope (explicit, do not do)

- Renaming `condition_id` to `market_key` or any other neutral name. Big blast radius (every detector, every paper-trading evaluator, every ML feature). Spec calls this out explicitly.
- Updating any `pscanner corpus *` CLI command to take `--platform`. Those stay Polymarket-hardcoded and are reshaped later by the integration spec.
- Cross-platform ML training. The plan only adds a single-platform filter; mixing platforms in one model requires schema reconciliation that belongs in the Kalshi/Manifold integration spec.
- Daemon tables (`kalshi_*`, `manifold_*`, `market_cache`, `wallet_trades`, `alerts`, etc.). These stay namespaced per the RFC's hybrid decision.

---

## Risks

- **Long migration on the real corpus.** `corpus_trades` (~15.9M) and `training_examples` (~15.5M) are table-copy migrations. Each takes 5–15 minutes on the WSL2 dev box. Backup before running. If interrupted mid-table, the half-renamed `__new` table remains; recovery: drop the `__new` table and re-run init_corpus_db (idempotency on the rest of the tables already-migrated continues to work).
- **`pl.read_database` parameter form.** Different Polars versions accept different parameter binding shapes. If the `execute_options={"parameters": ...}` form fails, fall back to validated f-string interpolation (`assert platform in {...}` then format).
- **Existing repo callers with positional args.** The dataclass `platform` field has a default and is appended after all existing fields, so positional construction continues to work. ty type checking surfaces any caller that didn't migrate cleanly.
