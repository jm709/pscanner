"""SQLite bootstrap for the corpus subsystem.

Creates ``data/corpus.sqlite3`` (idempotently), applies WAL pragmas, and
sets ``row_factory = sqlite3.Row``. The schema is deliberately separate
from ``pscanner.store.db`` — corpus tables never live in the live DB,
and the live DB never holds corpus tables.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from pscanner.store.migrations import apply_additive_migrations

_log = structlog.get_logger(__name__)


TRAINING_EXAMPLES_COLUMNS: tuple[str, ...] = (
    "platform",
    "tx_hash",
    "asset_id",
    "wallet_address",
    "condition_id",
    "trade_ts",
    "built_at",
    "prior_trades_count",
    "prior_buys_count",
    "prior_resolved_buys",
    "prior_wins",
    "prior_losses",
    "win_rate",
    "avg_implied_prob_paid",
    "realized_edge_pp",
    "prior_realized_pnl_usd",
    "avg_bet_size_usd",
    "median_bet_size_usd",
    "wallet_age_days",
    "seconds_since_last_trade",
    "prior_trades_30d",
    "top_category",
    "category_diversity",
    "bet_size_usd",
    "bet_size_rel_to_avg",
    "edge_confidence_weighted",
    "win_rate_confidence_weighted",
    "is_high_quality_wallet",
    "bet_size_relative_to_history",
    "side",
    "implied_prob_at_buy",
    "market_category",
    "market_volume_so_far_usd",
    "market_unique_traders_so_far",
    "market_age_seconds",
    "time_to_resolution_seconds",
    "last_trade_price",
    "price_volatility_recent",
    "cat_sports",
    "cat_esports",
    "cat_thesis",
    "cat_macro",
    "cat_elections",
    "cat_crypto",
    "cat_geopolitics",
    "cat_tech",
    "cat_culture",
    "label_won",
)


def training_examples_ddl(table_name: str) -> str:
    """Canonical CREATE TABLE statement for ``training_examples``.

    Parametrized on ``table_name`` so the DuckDB engine can build
    ``training_examples_v2`` with identical constraints. Keep in sync
    with ``TRAINING_EXAMPLES_COLUMNS``: any new column here must also be
    added there (the engine's column list relies on tuple order).
    """
    return f"""
    CREATE TABLE {table_name} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
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
      cat_sports INTEGER NOT NULL DEFAULT 0,
      cat_esports INTEGER NOT NULL DEFAULT 0,
      cat_thesis INTEGER NOT NULL DEFAULT 0,
      cat_macro INTEGER NOT NULL DEFAULT 0,
      cat_elections INTEGER NOT NULL DEFAULT 0,
      cat_crypto INTEGER NOT NULL DEFAULT 0,
      cat_geopolitics INTEGER NOT NULL DEFAULT 0,
      cat_tech INTEGER NOT NULL DEFAULT 0,
      cat_culture INTEGER NOT NULL DEFAULT 0,
      label_won INTEGER NOT NULL,
      UNIQUE (platform, tx_hash, asset_id, wallet_address)
    )
    """


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
      tags_json TEXT NOT NULL DEFAULT '[]',
      categories_json TEXT NOT NULL DEFAULT '[]',
      outcome_side_backfilled_at INTEGER,
      v1_history_pending INTEGER NOT NULL DEFAULT 0,
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
    # Covering index for the build-features chronological scan (#114).
    # The leading ``(platform, ts, tx_hash, asset_id)`` prefix satisfies the
    # keyset-paginated WHERE clause and ORDER BY so no temp B-tree is needed.
    # The trailing columns cover every column the ``iter_chronological`` SELECT
    # reads, eliminating the per-row heap rowid lookup on an existing B-tree
    # descent entirely.
    (
        "CREATE INDEX IF NOT EXISTS idx_corpus_trades_chrono_covering "
        "ON corpus_trades("
        "  platform, ts, tx_hash, asset_id, "
        "  wallet_address, condition_id, outcome_side, bs, "
        "  price, size, notional_usd"
        ")"
    ),
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
    training_examples_ddl("training_examples").replace(
        "CREATE TABLE training_examples", "CREATE TABLE IF NOT EXISTS training_examples"
    ),
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
    # Path A perf (#114): ~2 GB write-side page cache (-2000000 KiB ≈
    # 1.91 GB), and a hard 256 MB cap on the WAL. The 12 GB WAL symptom
    # from issue #114's 6.7h rebuild was checkpoint starvation under a
    # long-lived read cursor — auto-checkpoint cadence is a sanity floor,
    # not the bound. journal_size_limit is the hard cap; Task 4 adds
    # manual wal_checkpoint(TRUNCATE) calls on the build-features write
    # path to actively reclaim WAL pages between batches.
    "PRAGMA cache_size=-2000000",
    "PRAGMA journal_size_limit=268435456",
    "PRAGMA temp_store=MEMORY",
)

# Read-connection PRAGMAs for the build-features chronological cursor.
# A separate set is needed because the read connection is opened in
# URI ?mode=ro mode (issue #110) and bypasses ``init_corpus_db``. The
# ~4 GB cache (-4000000 KiB ≈ 3.81 GB) + 8 GB mmap is sized for a 32 GB corpus on a desktop with
# 12+ GB RAM allocated to WSL2; ``temp_store=MEMORY`` prevents temp-
# btree spill onto the same vhdx that already pressures the source
# read. ``query_only=1`` is belt-and-braces against a future caller
# accidentally writing through the read connection.
_READ_PRAGMAS: tuple[str, ...] = (
    "PRAGMA cache_size=-4000000",
    "PRAGMA mmap_size=8589934592",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA query_only=1",
)


def apply_read_pragmas(conn: sqlite3.Connection) -> None:
    """Apply Path A read-side PRAGMAs to a connection (#114).

    Idempotent. Use on read-only connections opened outside
    ``init_corpus_db`` (the build-features chronological cursor is the
    only current caller).
    """
    for pragma in _READ_PRAGMAS:
        conn.execute(pragma)


_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE corpus_markets ADD COLUMN market_slug TEXT",
    # Superseded by ``idx_corpus_trades_platform_ts_tx_asset``, which itself
    # is later superseded in this same migration list by
    # ``idx_corpus_trades_chrono_covering`` (Path A perf, #114) — see the
    # DROP+CREATE pair appended at the end of _MIGRATIONS.
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
    # Raw gamma tag list (JSON-encoded) and the derived multi-label category set
    # for the corpus market. Populated by `pscanner corpus backfill-gamma-tags`
    # (issue #121) and by `enumerate_closed_markets` on new inserts going forward.
    "ALTER TABLE corpus_markets ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE corpus_markets ADD COLUMN categories_json TEXT NOT NULL DEFAULT '[]'",
    # Sentinel tracking when outcome_side was backfilled for this market (issue #167).
    "ALTER TABLE corpus_markets ADD COLUMN outcome_side_backfilled_at INTEGER",
    # Multi-label category indicator columns (issue #122). One per Category
    # enum member; build-features writes 0/1 per row from `features.market_categories`.
    "ALTER TABLE training_examples ADD COLUMN cat_sports INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_esports INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_thesis INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_macro INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_elections INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_crypto INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_geopolitics INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_tech INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE training_examples ADD COLUMN cat_culture INTEGER NOT NULL DEFAULT 0",
    # Path A perf (#114): the existing 4-column index was non-covering for the
    # build-features chronological scan, so each fetched row triggered a heap
    # rowid lookup. Replace with an 11-column index that covers every column the
    # build-features SELECT reads, leaving the keyset-paginated WHERE-clause
    # prefix (platform, ts, tx_hash, asset_id) unchanged.
    "DROP INDEX IF EXISTS idx_corpus_trades_platform_ts_tx_asset",
    (
        "CREATE INDEX IF NOT EXISTS idx_corpus_trades_chrono_covering "
        "ON corpus_trades("
        "  platform, ts, tx_hash, asset_id, "
        "  wallet_address, condition_id, outcome_side, bs, "
        "  price, size, notional_usd"
        ")"
    ),
    # V1 subgraph fill-in queue (issue #193). 1 = this market has pre-V2
    # trade history that the current V2 subgraph cannot reach (the
    # Polymarket V1 contract was indexed by `7fu2DWYK…` only through
    # 2026-04-28, then re-deployed with a different schema). Set by
    # `walk_market` when truncated_at_offset_cap=1 AND the oldest
    # corpus_trades.ts for the market is < V2_START (1775220779,
    # 2026-04-03). Cleared by the future V1 adapter.
    "ALTER TABLE corpus_markets ADD COLUMN v1_history_pending INTEGER NOT NULL DEFAULT 0",
)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if ``table`` has a column named ``column``.

    Returns False if the table itself does not exist (so a fresh-DB run
    where the table is created later by ``_SCHEMA_STATEMENTS`` falls
    through cleanly without raising).
    """
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


@dataclass(frozen=True)
class _PlatformMigrationSpec:
    """One-time PR-A platform-column migration spec for a single table.

    ``new_table_ddl`` is the pre-PR-A table shape augmented with the
    composite platform PK. It is deliberately *not* refreshed from current
    ``_SCHEMA_STATEMENTS`` — additive columns added after PR-A layer on via
    ``_MIGRATIONS`` (``ALTER TABLE``) once the copy completes. Refreshing
    the DDL would break the upgrade-from-pre-PR-A path.

    The ``training_examples`` spec is the one exception: its DDL is the
    canonical :func:`training_examples_ddl` output because that function is
    parametrized on the destination table name. The corresponding INSERT
    projection supplies ``0`` defaults for the ``cat_*`` columns that exist
    in the canonical DDL but not in pre-PR-A on-disk rows.

    Each ``(insert_col, select_expr)`` pair in ``column_projections``
    maps a column on the destination ``<table>__new`` table to its source
    expression in the SELECT — typically the same column name on the legacy
    table, or a literal like ``'polymarket'`` for the platform column.
    """

    table: str
    new_table_ddl: str
    column_projections: tuple[tuple[str, str], ...]
    post_swap_index_ddl: tuple[str, ...] = ()


_PLATFORM_MIGRATIONS: tuple[_PlatformMigrationSpec, ...] = (
    _PlatformMigrationSpec(
        table="corpus_markets",
        new_table_ddl="""
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
            """,
        column_projections=(
            ("platform", "'polymarket'"),
            ("condition_id", "condition_id"),
            ("event_slug", "event_slug"),
            ("category", "category"),
            ("closed_at", "closed_at"),
            ("total_volume_usd", "total_volume_usd"),
            ("backfill_state", "backfill_state"),
            ("last_offset_seen", "last_offset_seen"),
            ("trades_pulled_count", "trades_pulled_count"),
            ("truncated_at_offset_cap", "truncated_at_offset_cap"),
            ("error_message", "error_message"),
            ("enumerated_at", "enumerated_at"),
            ("backfill_started_at", "backfill_started_at"),
            ("backfill_completed_at", "backfill_completed_at"),
            ("market_slug", "market_slug"),
            ("onchain_trades_count", "onchain_trades_count"),
            ("onchain_processed_at", "onchain_processed_at"),
        ),
        post_swap_index_ddl=(
            "CREATE INDEX IF NOT EXISTS idx_corpus_markets_state ON corpus_markets(backfill_state)",
            (
                "CREATE INDEX IF NOT EXISTS idx_corpus_markets_volume "
                "ON corpus_markets(total_volume_usd DESC)"
            ),
        ),
    ),
    _PlatformMigrationSpec(
        table="corpus_trades",
        new_table_ddl="""
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
            """,
        column_projections=(
            ("platform", "'polymarket'"),
            ("tx_hash", "tx_hash"),
            ("asset_id", "asset_id"),
            ("wallet_address", "wallet_address"),
            ("condition_id", "condition_id"),
            ("outcome_side", "outcome_side"),
            ("bs", "bs"),
            ("price", "price"),
            ("size", "size"),
            ("notional_usd", "notional_usd"),
            ("ts", "ts"),
        ),
        post_swap_index_ddl=(
            (
                "CREATE INDEX IF NOT EXISTS idx_corpus_trades_market_ts "
                "ON corpus_trades(condition_id, ts)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_corpus_trades_wallet_ts "
                "ON corpus_trades(wallet_address, ts)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_corpus_trades_chrono_covering "
                "ON corpus_trades("
                "  platform, ts, tx_hash, asset_id, "
                "  wallet_address, condition_id, outcome_side, bs, "
                "  price, size, notional_usd"
                ")"
            ),
        ),
    ),
    _PlatformMigrationSpec(
        table="market_resolutions",
        new_table_ddl="""
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
            """,
        column_projections=(
            ("platform", "'polymarket'"),
            ("condition_id", "condition_id"),
            ("winning_outcome_index", "winning_outcome_index"),
            ("outcome_yes_won", "outcome_yes_won"),
            ("resolved_at", "resolved_at"),
            ("source", "source"),
            ("recorded_at", "recorded_at"),
        ),
    ),
    _PlatformMigrationSpec(
        table="training_examples",
        # Canonical DDL with the current shape, not a frozen pre-PR-A shape —
        # `training_examples_ddl` is the one parametrized helper. The 9
        # `cat_*` columns added post-PR-A are defaulted to 0 in the
        # projection below; the legacy `id` column is preserved as a stable
        # rowid for `pscanner.ml.streaming`'s chunk iteration.
        new_table_ddl=training_examples_ddl("training_examples__new"),
        column_projections=(
            ("id", "id"),
            ("platform", "'polymarket'"),
            ("tx_hash", "tx_hash"),
            ("asset_id", "asset_id"),
            ("wallet_address", "wallet_address"),
            ("condition_id", "condition_id"),
            ("trade_ts", "trade_ts"),
            ("built_at", "built_at"),
            ("prior_trades_count", "prior_trades_count"),
            ("prior_buys_count", "prior_buys_count"),
            ("prior_resolved_buys", "prior_resolved_buys"),
            ("prior_wins", "prior_wins"),
            ("prior_losses", "prior_losses"),
            ("win_rate", "win_rate"),
            ("avg_implied_prob_paid", "avg_implied_prob_paid"),
            ("realized_edge_pp", "realized_edge_pp"),
            ("prior_realized_pnl_usd", "prior_realized_pnl_usd"),
            ("avg_bet_size_usd", "avg_bet_size_usd"),
            ("median_bet_size_usd", "median_bet_size_usd"),
            ("wallet_age_days", "wallet_age_days"),
            ("seconds_since_last_trade", "seconds_since_last_trade"),
            ("prior_trades_30d", "prior_trades_30d"),
            ("top_category", "top_category"),
            ("category_diversity", "category_diversity"),
            ("bet_size_usd", "bet_size_usd"),
            ("bet_size_rel_to_avg", "bet_size_rel_to_avg"),
            ("edge_confidence_weighted", "edge_confidence_weighted"),
            ("win_rate_confidence_weighted", "win_rate_confidence_weighted"),
            ("is_high_quality_wallet", "is_high_quality_wallet"),
            ("bet_size_relative_to_history", "bet_size_relative_to_history"),
            ("side", "side"),
            ("implied_prob_at_buy", "implied_prob_at_buy"),
            ("market_category", "market_category"),
            ("market_volume_so_far_usd", "market_volume_so_far_usd"),
            ("market_unique_traders_so_far", "market_unique_traders_so_far"),
            ("market_age_seconds", "market_age_seconds"),
            ("time_to_resolution_seconds", "time_to_resolution_seconds"),
            ("last_trade_price", "last_trade_price"),
            ("price_volatility_recent", "price_volatility_recent"),
            ("cat_sports", "0"),
            ("cat_esports", "0"),
            ("cat_thesis", "0"),
            ("cat_macro", "0"),
            ("cat_elections", "0"),
            ("cat_crypto", "0"),
            ("cat_geopolitics", "0"),
            ("cat_tech", "0"),
            ("cat_culture", "0"),
            ("label_won", "label_won"),
        ),
        post_swap_index_ddl=(
            (
                "CREATE INDEX IF NOT EXISTS idx_training_examples_condition "
                "ON training_examples(condition_id)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_training_examples_wallet "
                "ON training_examples(wallet_address)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_training_examples_label "
                "ON training_examples(label_won)"
            ),
        ),
    ),
    _PlatformMigrationSpec(
        table="asset_index",
        new_table_ddl="""
            CREATE TABLE asset_index__new (
              platform TEXT NOT NULL DEFAULT 'polymarket'
                CHECK (platform IN ('polymarket', 'kalshi', 'manifold')),
              asset_id TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome_side TEXT NOT NULL,
              outcome_index INTEGER NOT NULL,
              PRIMARY KEY (platform, asset_id)
            )
            """,
        column_projections=(
            ("platform", "'polymarket'"),
            ("asset_id", "asset_id"),
            ("condition_id", "condition_id"),
            ("outcome_side", "outcome_side"),
            ("outcome_index", "outcome_index"),
        ),
        post_swap_index_ddl=(
            "CREATE INDEX IF NOT EXISTS idx_asset_index_condition ON asset_index(condition_id)",
        ),
    ),
)


def _apply_platform_migration(
    conn: sqlite3.Connection,
    spec: _PlatformMigrationSpec,
) -> None:
    """Apply one platform-column migration. Idempotent.

    Skips silently when the target table already carries ``platform``, and
    when the table doesn't exist yet (a fresh DB hits ``_SCHEMA_STATEMENTS``
    directly and never enters this path).
    """
    if _column_exists(conn, spec.table, "platform"):
        return
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (spec.table,),
    ).fetchone():
        return
    start = time.monotonic()
    row_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM {spec.table}"  # noqa: S608
    ).fetchone()[0]
    _log.info("corpus.migration_started", table=spec.table, rows=row_count)
    insert_cols = ", ".join(col for col, _ in spec.column_projections)
    select_exprs = ", ".join(expr for _, expr in spec.column_projections)
    with conn:
        conn.executescript(spec.new_table_ddl + ";")
        conn.execute(
            f"INSERT INTO {spec.table}__new ({insert_cols}) "  # noqa: S608
            f"SELECT {select_exprs} FROM {spec.table}"
        )
        conn.execute(f"DROP TABLE {spec.table}")
        conn.execute(f"ALTER TABLE {spec.table}__new RENAME TO {spec.table}")
        for stmt in spec.post_swap_index_ddl:
            conn.execute(stmt)
    duration_s = time.monotonic() - start
    _log.info(
        "corpus.migration_completed",
        table=spec.table,
        rows=row_count,
        duration_s=round(duration_s, 2),
    )


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply migrations. Idempotent.

    Runs the platform-column migrations first (which copy old tables into
    new ones with composite PKs), then the additive ALTER TABLE migrations
    in ``_MIGRATIONS``. The platform migrations are idempotent via
    ``_column_exists`` checks; the additive ones swallow the standard
    idempotent-failure error messages via
    :func:`pscanner.store.migrations.apply_additive_migrations`.
    """
    for spec in _PLATFORM_MIGRATIONS:
        _apply_platform_migration(conn, spec)
    apply_additive_migrations(conn, _MIGRATIONS)


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
