"""DuckDB-based engine for ``pscanner corpus build-features``.

Pure SQL pipeline that produces ``training_examples`` rows bit-equivalent
(within ``rtol=1e-9``) to the Python ``StreamingHistoryProvider`` fold,
in 5-25 min vs 6h. See ``docs/superpowers/plans/2026-05-11-issue-116-duckdb-engine.md``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Final

import duckdb
import structlog

_log = structlog.get_logger(__name__)

_V2_TABLE: Final[str] = "training_examples_v2"
_V2_INDEX_PREFIX: Final[str] = "idx_te_v2_"


def build_features_duckdb(
    *,
    db_path: Path,
    platform: str,
    now_ts: int,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> int:
    """Rebuild ``training_examples`` for ``platform`` via DuckDB.

    Args:
        db_path: Path to ``corpus.sqlite3``.
        platform: Single platform to rebuild (``polymarket``/``manifold``/etc).
        now_ts: Value written to every row's ``built_at`` column.
        memory_limit: DuckDB ``memory_limit`` PRAGMA value (e.g. ``"6GB"``).
        temp_dir: Spill directory for partitions that don't fit in memory.
        threads: ``threads`` PRAGMA value.

    Returns:
        Number of rows in the new ``training_examples`` table.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # Pre-create v2 via stdlib sqlite3 BEFORE DuckDB opens the file.
    # DuckDB's attached-SQLite CREATE TABLE rewrites types and strips
    # DEFAULT / CHECK clauses, so we must own schema creation here.
    _create_v2_via_sqlite3(db_path=db_path)

    duck = duckdb.connect(":memory:")
    try:
        _configure_duckdb(duck, memory_limit=memory_limit, temp_dir=temp_dir, threads=threads)
        _attach_corpus(duck, db_path=db_path)
        _materialize_trades(duck, platform=platform)
        _build_training_examples_v2(duck, platform=platform, now_ts=now_ts)
        n_rows = _count_v2(duck)
        corpus_path = _detach_corpus(duck)
        _atomic_swap(corpus_path)
        _log.info(
            "corpus.build_features_duckdb_done",
            rows=n_rows,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        return n_rows
    finally:
        duck.close()


def _configure_duckdb(
    duck: duckdb.DuckDBPyConnection,
    *,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> None:
    duck.execute(f"SET memory_limit = '{memory_limit}'")
    duck.execute(f"SET temp_directory = '{temp_dir}'")
    duck.execute(f"SET threads = {threads}")
    duck.execute("INSTALL sqlite")
    duck.execute("LOAD sqlite")


def _attach_corpus(duck: duckdb.DuckDBPyConnection, *, db_path: Path) -> None:
    duck.execute(f"ATTACH '{db_path}' AS corpus (TYPE sqlite)")


def _create_v2_via_sqlite3(*, db_path: Path) -> None:
    """Create training_examples_v2 with the canonical SQLite DDL.

    Run via stdlib sqlite3 (NOT DuckDB) because DuckDB's attached-SQLite
    CREATE TABLE rewrites types and strips ``DEFAULT`` / ``CHECK`` clauses.
    Pre-creating v2 here means DuckDB's INSERT can target the canonical
    schema, and the post-swap production table keeps its constraints.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Drop any stale artifacts from a prior crashed run.
        conn.execute(f"DROP TABLE IF EXISTS {_V2_TABLE}")
        for suffix in ("condition", "wallet", "label"):
            conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}{suffix}")

        # Canonical training_examples DDL with table+index names suffixed.
        # Keep in sync with src/pscanner/corpus/db.py:_SCHEMA_STATEMENTS.
        conn.execute(
            f"""
            CREATE TABLE {_V2_TABLE} (
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
              label_won INTEGER NOT NULL,
              UNIQUE (platform, tx_hash, asset_id, wallet_address)
            )
            """
        )
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}condition ON {_V2_TABLE}(condition_id)")
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}wallet ON {_V2_TABLE}(wallet_address)")
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}label ON {_V2_TABLE}(label_won)")
        conn.commit()
    finally:
        conn.close()


def _materialize_trades(duck: duckdb.DuckDBPyConnection, *, platform: str) -> None:
    """Pull corpus_trades + corpus_markets + market_resolutions into DuckDB TEMP."""
    duck.execute(
        f"""
        CREATE TEMP TABLE trades AS
        SELECT
            t.tx_hash, t.asset_id, t.wallet_address, t.condition_id,
            t.outcome_side, t.bs, t.price, t.size, t.notional_usd, t.ts,
            m.category, m.closed_at, m.enumerated_at
        FROM corpus.corpus_trades t
        JOIN corpus.corpus_markets m
          ON m.platform = t.platform AND m.condition_id = t.condition_id
        WHERE t.platform = '{platform}' AND m.platform = '{platform}'
        """  # noqa: S608 — platform is caller-controlled, not user input
    )
    duck.execute(
        f"""
        CREATE TEMP TABLE resolutions AS
        SELECT condition_id, resolved_at, outcome_yes_won
        FROM corpus.market_resolutions
        WHERE platform = '{platform}'
        """  # noqa: S608 — platform is caller-controlled, not user input
    )


def _build_training_examples_v2(
    duck: duckdb.DuckDBPyConnection, *, platform: str, now_ts: int
) -> None:
    """INSERT into training_examples_v2 via a single CTE chain.

    Stages:
      1. ``events`` — UNION of BUYs/SELLs + synthetic RESOLUTION events.
         RESOLUTIONs carry the original BUY's notional/size/side so realized_pnl
         can sum in the same window as the wins/losses counters. ``kind_priority``
         is 0 for RESOLUTION and 1 for BUY/SELL so same-ts ties put resolutions
         first (matching Python heap-drain semantics: ``wallet_state(W, as_of_ts=T)``
         drains heap entries with ``resolution_ts < T`` BEFORE folding in
         the current trade).
      2. ``wallet_acc`` — windowed running aggregates per wallet over events.
      3. Final SELECT — joins to wallet_first_seen + market_first_seen, computes
         interaction features (later tasks will fill in category and market
         aggregates), filters to BUYs whose market has resolved.

    Tasks 8-11 will extend this with: avg_prob/edge formulas (T8), top_category
    + diversity CTE (T9), market-side aggregates (T10), interaction features (T11).
    """
    duck.execute(
        f"""
        INSERT INTO corpus.{_V2_TABLE} (
            platform, tx_hash, asset_id, wallet_address, condition_id,
            trade_ts, built_at,
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
        WITH
        wallet_first_seen AS (
            SELECT wallet_address, MIN(ts) AS first_seen_ts
            FROM trades
            GROUP BY wallet_address
        ),
        buy_events AS (
            SELECT
                t.wallet_address, t.condition_id, t.ts AS event_ts,
                t.tx_hash, t.asset_id, t.bs, t.outcome_side,
                t.price, t.size, t.notional_usd, t.category,
                t.closed_at, t.enumerated_at,
                -- kind_priority=0 for trades (BUY/SELL) so they sort BEFORE
                -- same-ts resolution events (kind_priority=1).  Python's heap
                -- drain uses strict-< on resolution_ts, meaning a resolution at
                -- the same ts as a BUY is NOT in the prior state for that BUY.
                CAST(0 AS INTEGER) AS kind_priority,
                CAST(0 AS INTEGER) AS is_resolution,
                CAST(NULL AS INTEGER) AS res_won_for_this_buy,
                CAST(0.0 AS DOUBLE) AS payout_pnl_increment,
                CAST(1 AS INTEGER) AS is_trade,
                CAST(CASE WHEN t.bs = 'BUY' THEN 1 ELSE 0 END AS INTEGER) AS is_buy_only,
                CAST(CASE WHEN t.bs = 'BUY' THEN t.price ELSE NULL END AS DOUBLE) AS buy_price,
                CAST(CASE WHEN t.bs = 'BUY' THEN t.notional_usd ELSE NULL END AS DOUBLE)
                    AS buy_notional
            FROM trades t
        ),
        resolution_events AS (
            SELECT
                t.wallet_address, t.condition_id, r.resolved_at AS event_ts,
                t.tx_hash, t.asset_id,
                CAST(NULL AS VARCHAR) AS bs,
                CAST(NULL AS VARCHAR) AS outcome_side,
                CAST(NULL AS DOUBLE) AS price,
                CAST(NULL AS DOUBLE) AS size,
                CAST(NULL AS DOUBLE) AS notional_usd,
                CAST(NULL AS VARCHAR) AS category,
                CAST(NULL AS INTEGER) AS closed_at,
                CAST(NULL AS INTEGER) AS enumerated_at,
                -- kind_priority=1 for resolutions: sorts AFTER same-ts BUYs.
                -- Matches Python's heap_drain condition (resolution_ts < as_of_ts),
                -- where same-ts resolutions are excluded from prior state.
                CAST(1 AS INTEGER) AS kind_priority,
                CAST(1 AS INTEGER) AS is_resolution,
                CAST(
                    CASE
                        WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES')
                          OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO')
                        THEN 1 ELSE 0
                    END AS INTEGER
                ) AS res_won_for_this_buy,
                CAST(
                    CASE
                        WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES')
                          OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO')
                        THEN t.size - t.notional_usd
                        ELSE -t.notional_usd
                    END AS DOUBLE
                ) AS payout_pnl_increment,
                CAST(0 AS INTEGER) AS is_trade,
                CAST(0 AS INTEGER) AS is_buy_only,
                CAST(NULL AS DOUBLE) AS buy_price,
                CAST(NULL AS DOUBLE) AS buy_notional
            FROM trades t
            JOIN resolutions r USING (condition_id)
            WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at
        ),
        events AS (
            SELECT * FROM buy_events
            UNION ALL
            SELECT * FROM resolution_events
        ),
        wallet_acc AS (
            SELECT
                e.*,
                COALESCE(SUM(is_trade) OVER w_strict, 0) AS prior_trades_count_w,
                COALESCE(SUM(is_buy_only) OVER w_strict, 0) AS prior_buys_count_w,
                COALESCE(SUM(is_resolution) OVER w_strict, 0) AS prior_resolved_buys_w,
                COALESCE(SUM(res_won_for_this_buy) OVER w_strict, 0) AS prior_wins_w,
                COALESCE(
                    SUM(is_resolution) OVER w_strict
                    - SUM(res_won_for_this_buy) OVER w_strict, 0
                ) AS prior_losses_w,
                COALESCE(SUM(payout_pnl_increment) OVER w_strict, 0.0) AS prior_realized_pnl_w,
                SUM(buy_price) OVER w_strict AS cumulative_buy_price_sum_w,
                SUM(buy_notional) OVER w_strict AS bet_size_sum_w,
                SUM(is_buy_only) OVER w_strict AS bet_size_count_w,
                MAX(CASE WHEN is_trade = 1 THEN event_ts END) OVER w_strict
                    AS last_trade_ts_w,
                -- prior_trades_30d: count trades strictly in the preceding window (w_strict)
                -- that are within 30 days.  RANGE windows can't tiebreak by tx_hash, so
                -- we compute total-prior minus older-than-30d-prior instead.
                -- w_pre_30d counts is_trade=1 rows with event_ts <= current_ts - 2592001,
                -- i.e. strictly older than 30 days (integer ts: < current_ts - 2592000).
                COALESCE(SUM(is_trade) OVER w_strict, 0)
                    - COALESCE(
                        COUNT(*) FILTER (WHERE is_trade = 1) OVER w_pre_30d, 0
                    ) AS prior_trades_30d_w,
                -- Market-age start: minimum trade ts strictly prior to this event on
                -- the same market.  NULL when this is the first event, in which case
                -- the Python engine returns empty_market_state(market_age_start_ts=0).
                MIN(CASE WHEN is_trade = 1 THEN event_ts END) OVER w_market_strict
                    AS market_first_prior_ts_w
            FROM events e
            WINDOW
                w_strict AS (
                    PARTITION BY wallet_address
                    ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                -- Counts is_trade=1 rows strictly older than 30 days (event_ts
                -- <= current_ts - 2592001, i.e. < current_ts - 2592000 for
                -- integer seconds).  Used to derive prior_trades_30d via
                -- total_prior - over_30d_prior, preserving the same-ts
                -- tie-breaking that ROWS-based w_strict provides.
                w_pre_30d AS (
                    PARTITION BY wallet_address
                    ORDER BY event_ts
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 2592001 PRECEDING
                ),
                w_market_strict AS (
                    PARTITION BY condition_id
                    ORDER BY event_ts, kind_priority, tx_hash, asset_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                )
        ),
        -- Per-(wallet, category) first-seen metadata used for tie-breaking.
        -- Only BUYs contribute to category_counts in Python (see
        -- apply_buy_to_state vs apply_sell_to_state in features.py).
        -- insertion_rank assigns 1 to the earliest-inserted category per wallet,
        -- matching Python's dict-iteration-order tiebreak: when two categories
        -- have the same count, the one with the lower rank (inserted first) wins.
        wallet_cat_buys AS (
            SELECT wallet_address, event_ts, kind_priority, tx_hash, asset_id, category,
                   MIN(event_ts) OVER (PARTITION BY wallet_address, category) AS cat_first_ts,
                   MIN(tx_hash) OVER (PARTITION BY wallet_address, category) AS cat_first_tx
            FROM events
            WHERE is_buy_only = 1 AND is_trade = 1
        ),
        wallet_cat_rank AS (
            SELECT DISTINCT
                wallet_address, category,
                DENSE_RANK() OVER (
                    PARTITION BY wallet_address
                    ORDER BY cat_first_ts ASC, cat_first_tx ASC
                ) AS insertion_rank
            FROM wallet_cat_buys
        ),
        -- For each target event, count prior BUYs per category by joining all
        -- BUY events that are strictly before this event (strict-< on the
        -- canonical sort key: event_ts, kind_priority, tx_hash, asset_id).
        wallet_cat_prior_counts AS (
            SELECT
                te.wallet_address, te.event_ts, te.kind_priority,
                te.tx_hash, te.asset_id,
                wcr.category, wcr.insertion_rank,
                COUNT(*) AS cat_count_prior
            FROM events te
            JOIN wallet_cat_buys b
                ON  b.wallet_address = te.wallet_address
                AND (
                    b.event_ts < te.event_ts
                    OR (b.event_ts = te.event_ts AND b.kind_priority < te.kind_priority)
                    OR (b.event_ts = te.event_ts AND b.kind_priority = te.kind_priority
                        AND b.tx_hash < te.tx_hash)
                    OR (b.event_ts = te.event_ts AND b.kind_priority = te.kind_priority
                        AND b.tx_hash = te.tx_hash AND b.asset_id < te.asset_id)
                )
            JOIN wallet_cat_rank wcr
                ON  wcr.wallet_address = b.wallet_address
                AND wcr.category = b.category
            WHERE te.is_trade = 1
            GROUP BY
                te.wallet_address, te.event_ts, te.kind_priority,
                te.tx_hash, te.asset_id,
                wcr.category, wcr.insertion_rank
        ),
        -- Collapse per-(event, category) rows to per-event (top_category, diversity).
        -- ARG_MAX key: primary = cat_count_prior DESC (higher count wins),
        -- secondary = insertion_rank DESC (lower rank = earlier insertion, so
        -- negating means higher neg_rank → larger key → ARG_MAX picks the winner).
        wallet_cat_summary AS (
            SELECT
                wallet_address, event_ts, kind_priority, tx_hash, asset_id,
                ARG_MAX(category, STRUCT_PACK(
                    cnt := cat_count_prior,
                    neg_rank := -insertion_rank
                )) AS top_category,
                COUNT(DISTINCT category) AS category_diversity
            FROM wallet_cat_prior_counts
            GROUP BY wallet_address, event_ts, kind_priority, tx_hash, asset_id
        )
        SELECT
            '{platform}' AS platform,
            wa.tx_hash, wa.asset_id, wa.wallet_address, wa.condition_id,
            wa.event_ts AS trade_ts,
            {now_ts} AS built_at,
            CAST(wa.prior_trades_count_w AS INTEGER) AS prior_trades_count,
            CAST(wa.prior_buys_count_w AS INTEGER) AS prior_buys_count,
            CAST(wa.prior_resolved_buys_w AS INTEGER) AS prior_resolved_buys,
            CAST(wa.prior_wins_w AS INTEGER) AS prior_wins,
            CAST(wa.prior_losses_w AS INTEGER) AS prior_losses,
            CASE WHEN wa.prior_resolved_buys_w > 0
                 THEN CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w
                 ELSE NULL END AS win_rate,
            CASE WHEN COALESCE(wa.bet_size_count_w, 0) > 0
                 THEN wa.cumulative_buy_price_sum_w / wa.bet_size_count_w
                 ELSE NULL END AS avg_implied_prob_paid,
            CASE
                WHEN wa.prior_resolved_buys_w > 0
                 AND COALESCE(wa.bet_size_count_w, 0) > 0
                THEN (CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w)
                     - (wa.cumulative_buy_price_sum_w / wa.bet_size_count_w)
                ELSE NULL
            END AS realized_edge_pp,
            wa.prior_realized_pnl_w AS prior_realized_pnl_usd,
            CASE WHEN COALESCE(wa.bet_size_count_w, 0) > 0
                 THEN wa.bet_size_sum_w / wa.bet_size_count_w
                 ELSE NULL END AS avg_bet_size_usd,
            CAST(NULL AS DOUBLE) AS median_bet_size_usd,
            GREATEST(0.0, (wa.event_ts - wfs.first_seen_ts) / 86400.0)
                AS wallet_age_days,
            CASE WHEN wa.last_trade_ts_w IS NOT NULL
                 THEN CAST(wa.event_ts - wa.last_trade_ts_w AS INTEGER)
                 ELSE NULL END AS seconds_since_last_trade,
            CAST(COALESCE(wa.prior_trades_30d_w, 0) AS INTEGER) AS prior_trades_30d,
            wcs.top_category,
            CAST(COALESCE(wcs.category_diversity, 0) AS INTEGER) AS category_diversity,
            wa.notional_usd AS bet_size_usd,
            CAST(NULL AS DOUBLE) AS bet_size_rel_to_avg,  -- Task 11
            CAST(0.0 AS DOUBLE) AS edge_confidence_weighted,  -- Task 11
            CAST(0.0 AS DOUBLE) AS win_rate_confidence_weighted,  -- Task 11
            CAST(0 AS INTEGER) AS is_high_quality_wallet,  -- Task 11
            CAST(1.0 AS DOUBLE) AS bet_size_relative_to_history,
            wa.outcome_side AS side,
            wa.price AS implied_prob_at_buy,
            wa.category AS market_category,
            CAST(0.0 AS DOUBLE) AS market_volume_so_far_usd,  -- Task 10
            CAST(0 AS INTEGER) AS market_unique_traders_so_far,  -- Task 10
            CAST(wa.event_ts - COALESCE(wa.market_first_prior_ts_w, 0) AS INTEGER)
                AS market_age_seconds,
            CAST(wa.closed_at - wa.event_ts AS INTEGER) AS time_to_resolution_seconds,
            CAST(NULL AS DOUBLE) AS last_trade_price,        -- Task 10
            CAST(NULL AS DOUBLE) AS price_volatility_recent, -- Task 10
            CASE
                WHEN (r.outcome_yes_won = 1 AND wa.outcome_side = 'YES')
                  OR (r.outcome_yes_won = 0 AND wa.outcome_side = 'NO')
                THEN 1 ELSE 0
            END AS label_won
        FROM wallet_acc wa
        JOIN wallet_first_seen wfs USING (wallet_address)
        JOIN resolutions r USING (condition_id)
        LEFT JOIN wallet_cat_summary wcs
            USING (wallet_address, event_ts, kind_priority, tx_hash, asset_id)
        WHERE wa.is_buy_only = 1
        """  # noqa: S608
    )


def _count_v2(duck: duckdb.DuckDBPyConnection) -> int:
    row = duck.execute(f"SELECT COUNT(*) FROM corpus.{_V2_TABLE}").fetchone()  # noqa: S608 — _V2_TABLE is a module-level literal
    return int(row[0]) if row else 0


def _detach_corpus(duck: duckdb.DuckDBPyConnection) -> str:
    """Retrieve the corpus DB path, then detach to release file locks.

    Returns:
        Absolute path to the SQLite file.
    """
    row = duck.execute("FROM duckdb_databases() WHERE database_name = 'corpus'").fetchone()
    if row is None:
        raise RuntimeError("corpus database not attached")
    # duckdb_databases() columns: database_name, database_oid, path, ...
    corpus_path = str(row[2])
    duck.execute("DETACH corpus")
    return corpus_path


def _atomic_swap(corpus_path: str) -> None:
    """Swap training_examples_v2 → training_examples inside one transaction.

    SQLite does not support ``ALTER INDEX ... RENAME TO``, so the swap drops
    the old table's named indexes first (before renaming, so DROP INDEX sees
    the canonical names), then renames v2 into place, then recreates the
    canonical indexes on the new table. This is safe because DuckDB has been
    detached before this function is called.
    """
    swap_conn = sqlite3.connect(corpus_path, isolation_level=None)
    try:
        swap_conn.execute("BEGIN IMMEDIATE")
        # Drop old named indexes before renaming (SQLite tracks indexes by name,
        # not by table, so they survive the rename and would collide on recreate).
        swap_conn.execute("DROP INDEX IF EXISTS idx_training_examples_condition")
        swap_conn.execute("DROP INDEX IF EXISTS idx_training_examples_wallet")
        swap_conn.execute("DROP INDEX IF EXISTS idx_training_examples_label")
        # Also drop any stale v2 indexes (shouldn't exist here, but be safe).
        swap_conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}condition")
        swap_conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}wallet")
        swap_conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}label")
        swap_conn.execute("ALTER TABLE training_examples RENAME TO training_examples_old")
        swap_conn.execute(f"ALTER TABLE {_V2_TABLE} RENAME TO training_examples")
        # Recreate canonical indexes on the new table.
        swap_conn.execute(
            "CREATE INDEX idx_training_examples_condition ON training_examples(condition_id)"
        )
        swap_conn.execute(
            "CREATE INDEX idx_training_examples_wallet ON training_examples(wallet_address)"
        )
        swap_conn.execute(
            "CREATE INDEX idx_training_examples_label ON training_examples(label_won)"
        )
        # Drop the old table (its auto-index is dropped automatically).
        swap_conn.execute("DROP TABLE training_examples_old")
        swap_conn.execute("DELETE FROM corpus_state WHERE key = 'build_features_in_progress'")
        swap_conn.execute("COMMIT")
    except Exception:
        swap_conn.execute("ROLLBACK")
        raise
    finally:
        swap_conn.close()
