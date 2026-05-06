"""SQLite bootstrap for the corpus subsystem.

Creates ``data/corpus.sqlite3`` (idempotently), applies WAL pragmas, and
sets ``row_factory = sqlite3.Row``. The schema is deliberately separate
from ``pscanner.store.db`` — corpus tables never live in the live DB,
and the live DB never holds corpus tables.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS corpus_markets (
      platform TEXT NOT NULL DEFAULT 'polymarket'
        CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
      platform TEXT NOT NULL DEFAULT 'polymarket'
        CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
    # Composite covers chronological keyset pagination
    # (``CorpusTradesRepo.iter_chronological``) without a temp B-tree sort.
    # The leading ``platform`` column scopes per-platform iteration to a
    # contiguous index range; the trailing ``ts, tx_hash, asset_id`` columns
    # satisfy the keyset-tiebreak ordering.
    "CREATE INDEX IF NOT EXISTS idx_corpus_trades_platform_ts_tx_asset "
    "ON corpus_trades(platform, ts, tx_hash, asset_id)",
    """
    CREATE TABLE IF NOT EXISTS market_resolutions (
      platform TEXT NOT NULL DEFAULT 'polymarket'
        CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
      platform TEXT NOT NULL DEFAULT 'polymarket'
        CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
      platform TEXT NOT NULL DEFAULT 'polymarket'
        CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
      asset_id TEXT NOT NULL,
      condition_id TEXT NOT NULL,
      outcome_side TEXT NOT NULL,
      outcome_index INTEGER NOT NULL,
      PRIMARY KEY (platform, asset_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_asset_index_condition ON asset_index(condition_id)",
)

_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)

_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE corpus_markets ADD COLUMN market_slug TEXT",
    # Superseded by ``idx_corpus_trades_platform_ts_tx_asset``, which covers
    # platform/ts-prefix queries (the ``ts``-only single-column index it once
    # replaced is no longer adequate after PR A's platform-aware indexing).
    "DROP INDEX IF EXISTS idx_corpus_trades_ts",
    "ALTER TABLE corpus_markets ADD COLUMN onchain_trades_count INTEGER",
    # Resume cursor for the per-market targeted on-chain backfill: NULL means
    # the market has not been processed yet; an integer Unix-second timestamp
    # marks completion. Cleared at runtime if a market needs to be re-processed.
    "ALTER TABLE corpus_markets ADD COLUMN onchain_processed_at INTEGER",
    # Wallet-quality x confidence interaction features (issue #44).
    "ALTER TABLE training_examples ADD COLUMN edge_confidence_weighted REAL NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN win_rate_confidence_weighted REAL NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN is_high_quality_wallet INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN bet_size_relative_to_history REAL NOT NULL DEFAULT 1",
)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if ``table`` has a column named ``column``.

    Returns False if the table itself does not exist (so a fresh-DB run
    where the table is created later by ``_SCHEMA_STATEMENTS`` falls
    through cleanly without raising).
    """
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


def _migrate_corpus_markets_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "corpus_markets", "platform"):
        return
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='corpus_markets'"
    ).fetchone():
        return
    start = time.monotonic()
    row_count = conn.execute("SELECT COUNT(*) AS n FROM corpus_markets").fetchone()[0]
    _log.info("corpus.migration_started", table="corpus_markets", rows=row_count)
    with conn:
        conn.execute(
            """
            CREATE TABLE corpus_markets__new (
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
            "CREATE INDEX IF NOT EXISTS idx_corpus_markets_state ON corpus_markets(backfill_state)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_markets_volume "
            "ON corpus_markets(total_volume_usd DESC)"
        )
    duration_s = time.monotonic() - start
    _log.info(
        "corpus.migration_completed",
        table="corpus_markets",
        rows=row_count,
        duration_s=round(duration_s, 2),
    )


def _migrate_corpus_trades_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "corpus_trades", "platform"):
        return
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='corpus_trades'"
    ).fetchone():
        return
    start = time.monotonic()
    row_count = conn.execute("SELECT COUNT(*) AS n FROM corpus_trades").fetchone()[0]
    _log.info("corpus.migration_started", table="corpus_trades", rows=row_count)
    with conn:
        conn.execute(
            """
            CREATE TABLE corpus_trades__new (
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
    duration_s = time.monotonic() - start
    _log.info(
        "corpus.migration_completed",
        table="corpus_trades",
        rows=row_count,
        duration_s=round(duration_s, 2),
    )


def _migrate_market_resolutions_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "market_resolutions", "platform"):
        return
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_resolutions'"
    ).fetchone():
        return
    start = time.monotonic()
    row_count = conn.execute("SELECT COUNT(*) AS n FROM market_resolutions").fetchone()[0]
    _log.info("corpus.migration_started", table="market_resolutions", rows=row_count)
    with conn:
        conn.execute(
            """
            CREATE TABLE market_resolutions__new (
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
    duration_s = time.monotonic() - start
    _log.info(
        "corpus.migration_completed",
        table="market_resolutions",
        rows=row_count,
        duration_s=round(duration_s, 2),
    )


def _migrate_training_examples_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "training_examples", "platform"):
        return
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_examples'"
    ).fetchone():
        return
    start = time.monotonic()
    row_count = conn.execute("SELECT COUNT(*) AS n FROM training_examples").fetchone()[0]
    _log.info("corpus.migration_started", table="training_examples", rows=row_count)
    with conn:
        conn.execute(
            """
            CREATE TABLE training_examples__new (
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
        # Drop the old `id` autoincrement column at copy time. None of the
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
            "CREATE INDEX IF NOT EXISTS idx_training_examples_label ON training_examples(label_won)"
        )
    duration_s = time.monotonic() - start
    _log.info(
        "corpus.migration_completed",
        table="training_examples",
        rows=row_count,
        duration_s=round(duration_s, 2),
    )


def _migrate_asset_index_add_platform(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "asset_index", "platform"):
        return
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='asset_index'"
    ).fetchone():
        return
    start = time.monotonic()
    row_count = conn.execute("SELECT COUNT(*) AS n FROM asset_index").fetchone()[0]
    _log.info("corpus.migration_started", table="asset_index", rows=row_count)
    with conn:
        conn.execute(
            """
            CREATE TABLE asset_index__new (
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
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
            "CREATE INDEX IF NOT EXISTS idx_asset_index_condition ON asset_index(condition_id)"
        )
    duration_s = time.monotonic() - start
    _log.info(
        "corpus.migration_completed",
        table="asset_index",
        rows=row_count,
        duration_s=round(duration_s, 2),
    )


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
            if "duplicate column name" in msg or "no such column" in msg or "no such table" in msg:
                continue
            raise
    conn.commit()


def init_corpus_db(path: Path) -> sqlite3.Connection:
    """Open the corpus SQLite database, creating dirs/schema as needed.

    Idempotent: every CREATE statement uses ``IF NOT EXISTS``. The returned
    connection has ``row_factory = sqlite3.Row`` and is in WAL mode.

    Migration order matters: the platform-column migrations run BEFORE
    ``_SCHEMA_STATEMENTS`` so that index-create statements which reference
    the new ``platform`` column don't trip on a still-old table shape.
    On a fresh DB the migration helpers no-op (their target tables don't
    exist yet) and ``_SCHEMA_STATEMENTS`` creates the new shape directly.

    Args:
        path: Filesystem path to the corpus database, or ``":memory:"``.
            Parent directories are created for non-memory paths.

    Returns:
        An open ``sqlite3.Connection``. Caller owns the lifecycle.

    Raises:
        sqlite3.DatabaseError: If pragma application or schema creation fails.
    """
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        for pragma in _PRAGMAS:
            conn.execute(pragma)
        _apply_migrations(conn)
        with conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn
