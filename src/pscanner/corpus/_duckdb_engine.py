"""DuckDB-based engine for ``pscanner corpus build-features``.

Pure SQL pipeline that produces ``training_examples`` rows bit-equivalent
(within ``rtol=1e-9``) to the Python ``StreamingHistoryProvider`` fold,
in 5-25 min vs 6h. See ``docs/superpowers/plans/2026-05-11-issue-116-duckdb-engine.md``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import duckdb
import structlog

from pscanner.corpus.db import TRAINING_EXAMPLES_COLUMNS, training_examples_ddl

_log = structlog.get_logger(__name__)

_V2_TABLE: Final[str] = "training_examples_v2"
_V2_INDEX_PREFIX: Final[str] = "idx_te_v2_"
_PLATFORMS: Final[frozenset[str]] = frozenset({"polymarket", "kalshi", "manifold"})
_SCRATCH_FILENAME: Final[str] = "build_scratch.duckdb"

# Categories that may appear in events.category. Drawn from
# pscanner.categories.Category enum plus the "unknown" sentinel used
# for markets without a recognized tag. The forward-compat guard
# (_assert_no_unknown_categories) refuses to run if events.category
# contains anything outside this set.
_KNOWN_CATEGORIES: Final[tuple[str, ...]] = (
    "sports",
    "esports",
    "thesis",
    "macro",
    "elections",
    "crypto",
    "geopolitics",
    "tech",
    "culture",
    "unknown",
)


def _validate_platform(platform: str) -> None:
    """Raise ValueError if ``platform`` is not in the allowlist.

    Prevents SQL-injection footgun in f-string-interpolated table queries.
    Caller-validated input upstream (argparse ``choices=``) is still
    enforced here for library callers and to drop the noqa markers.
    """
    if platform not in _PLATFORMS:
        raise ValueError(f"unknown platform: {platform!r}; must be one of {sorted(_PLATFORMS)}")


def _heartbeat_loop(
    *,
    stop: threading.Event,
    poll_fn: Callable[[], int],
    interval_seconds: float,
    stage: str,
) -> None:
    """Emit a heartbeat every ``interval_seconds`` until ``stop`` is set.

    ``poll_fn`` returns a snapshot of "how many rows landed so far" (or
    any integer progress signal). Errors in ``poll_fn`` are caught so a
    transient hiccup doesn't kill observability.
    """
    started = time.monotonic()
    while not stop.wait(interval_seconds):
        try:
            n = poll_fn()
        except Exception as exc:  # observability must not propagate
            _log.warning("corpus.build_features.heartbeat_poll_failed", error=str(exc))
            continue
        _log.info(
            "corpus.build_features.heartbeat",
            stage=stage,
            elapsed_seconds=round(time.monotonic() - started, 1),
            rows=n,
        )


def build_features_duckdb(
    *,
    db_path: Path,
    platform: str,
    now_ts: int,
    memory_limit: str,
    temp_dir: Path,
    threads: int,
) -> int:
    """Rebuild ``training_examples`` for ``platform`` via the 4-stage DuckDB pipeline.

    Args:
        db_path: Path to ``corpus.sqlite3``.
        platform: Single platform to rebuild. Validated against the
            allowlist; ValueError on unknown.
        now_ts: Value written to every row's ``built_at`` column.
        memory_limit: DuckDB ``memory_limit`` PRAGMA value (e.g. ``"6GB"``).
        temp_dir: Directory for the scratch DuckDB file and spill.
        threads: ``threads`` PRAGMA value.

    Returns:
        Row count of training_examples_v2 after the build, before swap.
    """
    _validate_platform(platform)
    started = time.monotonic()
    _create_v2_via_sqlite3(db_path=db_path)
    scratch_path = _scratch_path(temp_dir)
    _wipe_scratch(scratch_path)

    scratch = _open_scratch(scratch_path, memory_limit=memory_limit, threads=threads)
    try:
        scratch.execute("INSTALL sqlite")
        scratch.execute("LOAD sqlite")
        _attach_corpus(scratch, db_path=db_path)

        _run_stage(
            scratch,
            name="materialize_trades",
            fn=lambda: _materialize_trades(scratch, platform=platform),
        )
        _run_stage(
            scratch,
            name="stage1_events",
            fn=lambda: _stage1_events(scratch),
        )
        _run_stage(
            scratch,
            name="stage2_wallet_aggs",
            fn=lambda: _stage2_wallet_aggs(scratch),
        )
        _run_stage(
            scratch,
            name="stage3_market_aggs",
            fn=lambda: _stage3_market_aggs(scratch),
        )
        _run_stage(
            scratch,
            name="stage4_wallet_cat",
            fn=lambda: _stage4_wallet_cat(scratch),
        )
        _run_stage(
            scratch,
            name="final_join",
            fn=lambda: _final_join_to_v2(scratch, platform=platform, now_ts=now_ts),
        )

        n_rows = _count_v2(scratch)
        corpus_path = _detach_corpus(scratch)
    finally:
        scratch.close()

    _atomic_swap(corpus_path)
    _wipe_scratch(scratch_path)

    _log.info(
        "corpus.build_features_duckdb_done",
        rows=n_rows,
        elapsed_seconds=round(time.monotonic() - started, 1),
    )
    return n_rows


def _run_stage(
    scratch: duckdb.DuckDBPyConnection,
    *,
    name: str,
    fn: Callable[[], None],
) -> None:
    """Run a single stage with logging + heartbeat thread.

    Heartbeat polls the stage's likely-output table for row counts. If
    the table doesn't exist yet (early in the stage), poll returns 0.
    """
    _log.info("corpus.build_features.stage_start", stage=name)
    stop = threading.Event()
    poll_table = {
        "materialize_trades": "trades",
        "stage1_events": "events",
        "stage2_wallet_aggs": "wallet_aggs",
        "stage3_market_aggs": "market_aggs",
        "stage4_wallet_cat": "wallet_cat_summary",
        "final_join": f"corpus.{_V2_TABLE}",
    }.get(name)
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "stop": stop,
            "poll_fn": lambda: _count_table_safe(scratch, poll_table) if poll_table else 0,
            "interval_seconds": 30.0,
            "stage": name,
        },
        daemon=True,
        name=f"build_features_heartbeat_{name}",
    )
    heartbeat.start()
    started = time.monotonic()
    try:
        fn()
    except Exception:
        elapsed = round(time.monotonic() - started, 1)
        _log.error("corpus.build_features.stage_failed", stage=name, elapsed_seconds=elapsed)
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=5.0)
    _log.info(
        "corpus.build_features.stage_done",
        stage=name,
        elapsed_seconds=round(time.monotonic() - started, 1),
    )


def _count_table_safe(duck: duckdb.DuckDBPyConnection, table: str | None) -> int:
    """Row count for ``table``; 0 if the table doesn't exist yet."""
    if table is None:
        return 0
    try:
        row = duck.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
        return int(row[0]) if row else 0
    except duckdb.Error:
        return 0


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


def _scratch_path(temp_dir: Path) -> Path:
    """Path to the scratch DuckDB file under the spill dir."""
    return temp_dir / _SCRATCH_FILENAME


def _open_scratch(path: Path, *, memory_limit: str, threads: int) -> duckdb.DuckDBPyConnection:
    """Open (or create) a persistent scratch DuckDB file with the given budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(f"SET memory_limit = '{memory_limit}'")
    conn.execute(f"SET threads = {threads}")
    return conn


def _wipe_scratch(path: Path) -> None:
    """Remove the scratch file. No-op if it doesn't exist."""
    if path.exists():
        path.unlink()
    wal = path.with_suffix(path.suffix + ".wal")
    if wal.exists():
        wal.unlink()


def _create_v2_via_sqlite3(*, db_path: Path) -> None:
    """Create training_examples_v2 with the canonical SQLite DDL.

    Run via stdlib sqlite3 (NOT DuckDB) because DuckDB's attached-SQLite
    CREATE TABLE rewrites types and strips ``DEFAULT`` / ``CHECK`` clauses.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {_V2_TABLE}")
        for suffix in ("condition", "wallet", "label"):
            conn.execute(f"DROP INDEX IF EXISTS {_V2_INDEX_PREFIX}{suffix}")
        conn.execute(training_examples_ddl(_V2_TABLE))
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}condition ON {_V2_TABLE}(condition_id)")
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}wallet ON {_V2_TABLE}(wallet_address)")
        conn.execute(f"CREATE INDEX {_V2_INDEX_PREFIX}label ON {_V2_TABLE}(label_won)")
        conn.commit()
    finally:
        conn.close()


def _materialize_trades(duck: duckdb.DuckDBPyConnection, *, platform: str) -> None:
    """Pull corpus_trades + corpus_markets + market_resolutions into DuckDB TEMP."""
    duck.execute(
        """
        CREATE TEMP TABLE trades AS
        SELECT
            t.tx_hash, t.asset_id, t.wallet_address, t.condition_id,
            t.outcome_side, t.bs, t.price, t.size, t.notional_usd, t.ts,
            m.category, m.categories_json, m.closed_at, m.enumerated_at
        FROM corpus.corpus_trades t
        JOIN corpus.corpus_markets m
          ON m.platform = t.platform AND m.condition_id = t.condition_id
        WHERE t.platform = ? AND m.platform = ?
        """,
        [platform, platform],
    )
    duck.execute(
        """
        CREATE TEMP TABLE resolutions AS
        SELECT condition_id, resolved_at, outcome_yes_won
        FROM corpus.market_resolutions
        WHERE platform = ?
        """,
        [platform],
    )


def _stage1_events(scratch: duckdb.DuckDBPyConnection) -> None:
    """Materialize the UNION of trade events + synthetic resolution events.

    Output table ``events`` columns mirror the previous monolithic
    pipeline's ``events`` CTE: ``wallet_address``, ``condition_id``,
    ``event_ts``, ``tx_hash``, ``asset_id``, ``bs``, ``outcome_side``,
    ``price``, ``size``, ``notional_usd``, ``category``, ``categories_json``,
    ``closed_at``, ``enumerated_at``, ``kind_priority`` (0 for trades, 1 for
    resolutions), ``is_resolution``, ``res_won_for_this_buy``,
    ``payout_pnl_increment``, ``is_trade``, ``is_buy_only``, ``buy_price``,
    ``buy_notional``.

    Reads from the TEMP ``trades`` and ``resolutions`` tables produced by
    ``_materialize_trades``. Resolution events are emitted only for BUYs on
    resolved markets where ``t.ts <= r.resolved_at``.
    """
    scratch.execute(
        """
        CREATE OR REPLACE TABLE events AS
        WITH buy_events AS (
            SELECT
                t.wallet_address, t.condition_id, t.ts AS event_ts,
                t.tx_hash, t.asset_id, t.bs, t.outcome_side,
                t.price, t.size, t.notional_usd, t.category, t.categories_json,
                t.closed_at, t.enumerated_at,
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
                CAST(NULL AS VARCHAR) AS categories_json,
                CAST(NULL AS INTEGER) AS closed_at,
                CAST(NULL AS INTEGER) AS enumerated_at,
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
        )
        SELECT * FROM buy_events
        UNION ALL
        SELECT * FROM resolution_events
        """
    )


def _stage2_wallet_aggs(scratch: duckdb.DuckDBPyConnection) -> None:
    """Per-wallet running aggregates over the events table.

    Output table ``wallet_aggs`` has one row per event with windowed
    cumulative columns strictly preceding the event (the python engine's
    ``wallet_state(W, as_of_ts=T)`` semantics).

    Two non-obvious invariants:
      * ``last_trade_ts_w`` uses MAX-CASE not LAG — resolution events do
        not update python's ``wallet.last_trade_ts``.
      * ``prior_trades_30d_w`` is computed as ``total_prior - over_30d_prior``
        because DuckDB RANGE windows cannot multi-sort (so a direct
        RANGE-based 30d window loses same-ts tiebreak).
    """
    scratch.execute(
        """
        CREATE OR REPLACE TABLE wallet_aggs AS
        WITH wallet_first_seen AS (
            SELECT wallet_address, MIN(event_ts) AS first_seen_ts
            FROM events
            WHERE is_trade = 1
            GROUP BY wallet_address
        )
        SELECT
            e.wallet_address,
            e.event_ts,
            e.kind_priority,
            e.tx_hash,
            e.asset_id,
            e.condition_id,
            e.bs,
            e.outcome_side,
            e.price,
            e.size,
            e.notional_usd,
            e.category,
            e.categories_json,
            e.closed_at,
            e.is_buy_only,
            e.is_trade,
            wfs.first_seen_ts,
            COALESCE(SUM(e.is_trade) OVER w_strict, 0) AS prior_trades_count_w,
            COALESCE(SUM(e.is_buy_only) OVER w_strict, 0) AS prior_buys_count_w,
            COALESCE(SUM(e.is_resolution) OVER w_strict, 0) AS prior_resolved_buys_w,
            COALESCE(SUM(e.res_won_for_this_buy) OVER w_strict, 0) AS prior_wins_w,
            COALESCE(
                SUM(e.is_resolution) OVER w_strict
                - SUM(e.res_won_for_this_buy) OVER w_strict,
                0
            ) AS prior_losses_w,
            COALESCE(SUM(e.payout_pnl_increment) OVER w_strict, 0.0)
                AS prior_realized_pnl_usd_w,
            SUM(e.buy_price) OVER w_strict AS cum_buy_price_sum_w,
            SUM(e.buy_notional) OVER w_strict AS bet_size_sum_w,
            SUM(e.is_buy_only) OVER w_strict AS bet_size_count_w,
            MAX(CASE WHEN e.is_trade = 1 THEN e.event_ts END) OVER w_strict
                AS last_trade_ts_w,
            -- prior_trades_30d via total_prior - over_30d_prior. RANGE
            -- windows can't multi-sort, so we count is_trade=1 rows in
            -- (-inf, ts - 2592001] and subtract from the total preceding
            -- ROWS-based count. Cutoff: 2592001 PRECEDING covers anything
            -- with event_ts <= current_ts - 2592001 (i.e. > 30 days old).
            COALESCE(SUM(e.is_trade) OVER w_strict, 0)
                - COALESCE(
                    COUNT(*) FILTER (WHERE e.is_trade = 1) OVER w_pre_30d, 0
                ) AS prior_trades_30d_w
        FROM events e
        JOIN wallet_first_seen wfs USING (wallet_address)
        WINDOW
            w_strict AS (
                PARTITION BY e.wallet_address
                ORDER BY e.event_ts, e.kind_priority, e.tx_hash, e.asset_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            w_pre_30d AS (
                PARTITION BY e.wallet_address
                ORDER BY e.event_ts
                RANGE BETWEEN UNBOUNDED PRECEDING AND 2592001 PRECEDING
            )
        """
    )


def _stage3_market_aggs(scratch: duckdb.DuckDBPyConnection) -> None:
    """Per-market running aggregates over trade events.

    Inputs are filtered to ``is_trade = 1`` so resolution NULLs cannot
    pollute LAST_VALUE / STDDEV_POP / COUNT(price) windows. Also owns
    ``market_first_prior_ts_w`` (moved from stage 2 to keep stage 2
    leaf-independent on the wallet partition).
    """
    scratch.execute(
        """
        CREATE OR REPLACE TABLE market_aggs AS
        WITH market_trades AS (
            SELECT
                e.condition_id, e.wallet_address, e.event_ts,
                e.kind_priority, e.tx_hash, e.asset_id,
                e.price, e.notional_usd,
                CAST(
                    ROW_NUMBER() OVER (
                        PARTITION BY e.condition_id, e.wallet_address
                        ORDER BY e.event_ts, e.kind_priority, e.tx_hash, e.asset_id
                    ) = 1
                    AS INTEGER
                ) AS is_first_trade_in_market
            FROM events e
            WHERE e.is_trade = 1
        )
        SELECT
            mt.condition_id,
            mt.wallet_address,
            mt.event_ts,
            mt.kind_priority,
            mt.tx_hash,
            mt.asset_id,
            COALESCE(SUM(mt.notional_usd) OVER w_market_strict, 0.0)
                AS market_volume_so_far_w,
            COALESCE(SUM(mt.is_first_trade_in_market) OVER w_market_strict, 0)
                AS market_unique_traders_so_far_w,
            LAST_VALUE(mt.price IGNORE NULLS) OVER w_market_strict
                AS last_trade_price_w,
            STDDEV_POP(mt.price) OVER w_market_recent_20 AS price_volatility_w,
            COUNT(mt.price) OVER w_market_recent_20 AS price_count_20,
            MIN(mt.event_ts) OVER w_market_strict AS market_first_prior_ts_w
        FROM market_trades mt
        WINDOW
            w_market_strict AS (
                PARTITION BY mt.condition_id
                ORDER BY mt.event_ts, mt.kind_priority, mt.tx_hash, mt.asset_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            w_market_recent_20 AS (
                PARTITION BY mt.condition_id
                ORDER BY mt.event_ts, mt.kind_priority, mt.tx_hash, mt.asset_id
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            )
        """
    )


def _assert_no_unknown_categories(scratch: duckdb.DuckDBPyConnection) -> None:
    """Refuse to proceed if events.category contains a value outside _KNOWN_CATEGORIES.

    Forward-compat guard: the rewrite hardcodes the category universe in
    the CASE chain. If a new category appears in the corpus (e.g.
    pscanner.categories adds Category.WEATHER) and is missing here, the
    rewrite would silently drop affected wallets' top_category. The
    baseline ``DENSE_RANK OVER (cat_first_ts ASC, cat_first_tx ASC)``
    handled new categories implicitly; we trade that for explicit fail-fast.
    """
    placeholders = ", ".join(f"'{c}'" for c in _KNOWN_CATEGORIES)
    rows = scratch.execute(
        f"""
        SELECT DISTINCT category
        FROM events
        WHERE is_buy_only = 1 AND is_trade = 1
          AND category IS NOT NULL
          AND category NOT IN ({placeholders})
        LIMIT 5
        """  # noqa: S608 — _KNOWN_CATEGORIES is a module-level literal tuple
    ).fetchall()
    if rows:
        unknown = sorted(r[0] for r in rows)
        raise RuntimeError(
            f"events table contains unknown categories not in _KNOWN_CATEGORIES: "
            f"{unknown}. Update _KNOWN_CATEGORIES in _duckdb_engine.py to match "
            f"the current pscanner.categories.Category enum, then re-run."
        )


def _stage4_wallet_cat(scratch: duckdb.DuckDBPyConnection) -> None:
    """Per-wallet per-category running counts via FILTER windows.

    Replaces the O(k²) wallet_cat_prior_counts OR-chain self-join with
    a window-only pattern over the known 10-category universe.
    DuckDB plans this as one sort + (3 windows x N categories) cheap counters.

    Tiebreak for top_category (matching baseline):
      1. cat_count_X DESC (highest count wins)
      2. cat_first_ts_X ASC (earliest first-trade ts wins)
      3. cat_first_tx_X ASC (lex-min tx_hash within same ts)
    The CASE chain below encodes ``cat_X wins iff for every other cat_Y:
    (count_X > count_Y) OR (tie by count AND (first_ts_X < first_ts_Y
    OR (tie by ts AND first_tx_X <= first_tx_Y)))``.
    """
    _assert_no_unknown_categories(scratch)

    count_cols = ",\n            ".join(
        f"COALESCE(SUM(CASE WHEN e.is_buy_only = 1 AND e.category = '{cat}' "
        f"THEN 1 ELSE 0 END) OVER w_strict, 0) AS cat_count_{cat}"
        for cat in _KNOWN_CATEGORIES
    )
    first_ts_cols = ",\n            ".join(
        f"MIN(CASE WHEN e.is_buy_only = 1 AND e.category = '{cat}' "
        f"THEN e.event_ts END) OVER w_strict AS cat_first_ts_{cat}"
        for cat in _KNOWN_CATEGORIES
    )
    first_tx_cols = ",\n            ".join(
        f"MIN(CASE WHEN e.is_buy_only = 1 AND e.category = '{cat}' "
        f"THEN e.tx_hash END) OVER w_strict AS cat_first_tx_{cat}"
        for cat in _KNOWN_CATEGORIES
    )
    diversity_terms = " + ".join(
        f"CASE WHEN cat_count_{cat} > 0 THEN 1 ELSE 0 END" for cat in _KNOWN_CATEGORIES
    )

    _ts_sentinel = "9223372036854775807"  # INT64 max — NULL ranks last
    _tx_sentinel = "'~'"  # ASCII > any hex-digit, so NULL ranks last
    top_cat_branches: list[str] = []
    for cat in _KNOWN_CATEGORIES:
        conds: list[str] = [f"cat_count_{cat} > 0"]
        for other in _KNOWN_CATEGORIES:
            if other == cat:
                continue
            # cat wins over other when:
            #   count_cat > count_other
            #   OR (count_cat = count_other AND first_ts_cat < first_ts_other)
            #   OR (count_cat = count_other AND first_ts_cat = first_ts_other
            #       AND first_tx_cat <= first_tx_other)
            conds.append(
                f"(cat_count_{cat} > cat_count_{other} "
                f"OR (cat_count_{cat} = cat_count_{other} AND "
                f"COALESCE(cat_first_ts_{cat}, {_ts_sentinel}) < "
                f"COALESCE(cat_first_ts_{other}, {_ts_sentinel})) "
                f"OR (cat_count_{cat} = cat_count_{other} AND "
                f"COALESCE(cat_first_ts_{cat}, {_ts_sentinel}) = "
                f"COALESCE(cat_first_ts_{other}, {_ts_sentinel}) AND "
                f"COALESCE(cat_first_tx_{cat}, {_tx_sentinel}) <= "
                f"COALESCE(cat_first_tx_{other}, {_tx_sentinel})))"
            )
        top_cat_branches.append(f"WHEN {' AND '.join(conds)} THEN '{cat}'")
    top_cat_expr = "CASE " + " ".join(top_cat_branches) + " ELSE NULL END"

    scratch.execute(
        f"""
        CREATE OR REPLACE TABLE wallet_cat_summary AS
        WITH per_event_counts AS (
            SELECT
                e.wallet_address,
                e.event_ts,
                e.kind_priority,
                e.tx_hash,
                e.asset_id,
                {count_cols},
                {first_ts_cols},
                {first_tx_cols}
            FROM events e
            WINDOW w_strict AS (
                PARTITION BY e.wallet_address
                ORDER BY e.event_ts, e.kind_priority, e.tx_hash, e.asset_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
        )
        SELECT
            wallet_address,
            event_ts,
            kind_priority,
            tx_hash,
            asset_id,
            {top_cat_expr} AS top_category,
            ({diversity_terms}) AS category_diversity
        FROM per_event_counts
        """  # noqa: S608 — column names from _KNOWN_CATEGORIES module-level literal
    )


def _final_join_to_v2(scratch: duckdb.DuckDBPyConnection, *, platform: str, now_ts: int) -> None:
    """Join the four stage outputs and INSERT into corpus.training_examples_v2.

    All staged inputs live in the scratch DuckDB. ``resolutions`` is the
    TEMP table from _materialize_trades; ``corpus`` is the attached
    SQLite database the v2 table lives in.

    ``platform`` and ``now_ts`` are passed via DuckDB ? bindings, NOT
    f-string interpolation, preserving Task 3's parameterization defense.
    The only f-string interpolation in this SQL is _V2_TABLE (table name)
    and col_list (derived from canonical TRAINING_EXAMPLES_COLUMNS).
    """
    col_list = ", ".join(TRAINING_EXAMPLES_COLUMNS)
    scratch.execute(
        f"""
        INSERT INTO corpus.{_V2_TABLE} ({col_list})
        SELECT
            ? AS platform,
            wa.tx_hash,
            wa.asset_id,
            wa.wallet_address,
            wa.condition_id,
            wa.event_ts AS trade_ts,
            ? AS built_at,
            wa.prior_trades_count_w AS prior_trades_count,
            wa.prior_buys_count_w AS prior_buys_count,
            wa.prior_resolved_buys_w AS prior_resolved_buys,
            wa.prior_wins_w AS prior_wins,
            wa.prior_losses_w AS prior_losses,
            CASE WHEN wa.prior_resolved_buys_w > 0
                 THEN CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w
                 ELSE NULL END AS win_rate,
            CASE WHEN wa.bet_size_count_w > 0
                 THEN wa.cum_buy_price_sum_w / wa.bet_size_count_w
                 ELSE NULL END AS avg_implied_prob_paid,
            CASE WHEN wa.prior_resolved_buys_w > 0 AND wa.bet_size_count_w > 0
                 THEN (CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w)
                      - (wa.cum_buy_price_sum_w / wa.bet_size_count_w)
                 ELSE NULL END AS realized_edge_pp,
            wa.prior_realized_pnl_usd_w AS prior_realized_pnl_usd,
            CASE WHEN wa.bet_size_count_w > 0
                 THEN wa.bet_size_sum_w / wa.bet_size_count_w
                 ELSE NULL END AS avg_bet_size_usd,
            CAST(NULL AS DOUBLE) AS median_bet_size_usd,
            GREATEST(0.0, (wa.event_ts - wa.first_seen_ts) / 86400.0) AS wallet_age_days,
            CASE WHEN wa.last_trade_ts_w IS NOT NULL
                 THEN wa.event_ts - wa.last_trade_ts_w
                 ELSE NULL END AS seconds_since_last_trade,
            wa.prior_trades_30d_w AS prior_trades_30d,
            wcs.top_category AS top_category,
            COALESCE(wcs.category_diversity, 0) AS category_diversity,
            wa.notional_usd AS bet_size_usd,
            CASE WHEN wa.bet_size_count_w > 0 AND wa.bet_size_sum_w > 0
                 THEN wa.notional_usd / (wa.bet_size_sum_w / wa.bet_size_count_w)
                 ELSE NULL END AS bet_size_rel_to_avg,
            CASE WHEN wa.prior_resolved_buys_w > 0 AND wa.bet_size_count_w > 0
                 THEN ((CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w)
                       - (wa.cum_buy_price_sum_w / wa.bet_size_count_w))
                      * LEAST(1.0, CAST(wa.prior_resolved_buys_w AS DOUBLE) / 20.0)
                 ELSE 0.0 END AS edge_confidence_weighted,
            CASE WHEN wa.prior_resolved_buys_w > 0
                 THEN ((CAST(wa.prior_wins_w AS DOUBLE) / wa.prior_resolved_buys_w) - 0.5)
                      * LEAST(1.0, CAST(wa.prior_resolved_buys_w AS DOUBLE) / 20.0)
                 ELSE 0.0 END AS win_rate_confidence_weighted,
            CASE WHEN wa.prior_resolved_buys_w >= 20
                      AND (CAST(wa.prior_wins_w AS DOUBLE)
                           / NULLIF(wa.prior_resolved_buys_w, 0)) > 0.55
                 THEN 1 ELSE 0 END AS is_high_quality_wallet,
            CAST(1.0 AS DOUBLE) AS bet_size_relative_to_history,
            wa.outcome_side AS side,
            wa.price AS implied_prob_at_buy,
            wa.category AS market_category,
            COALESCE(ma.market_volume_so_far_w, 0.0) AS market_volume_so_far_usd,
            CAST(COALESCE(ma.market_unique_traders_so_far_w, 0) AS INTEGER)
                AS market_unique_traders_so_far,
            -- market_first_prior_ts_w now lives on stage 3 (market_aggs).
            -- When NULL (first observed trade on the market), python's
            -- compute_features reads market_state(condition_id, as_of_ts)
            -- BEFORE observe() folds the trade in. The pre-observe state is
            -- empty_market_state(market_age_start_ts=0) per features.py:643-645,
            -- so market_age_seconds = trade.ts - 0 = trade.ts on first sighting.
            -- COALESCE-to-0 (NOT to wa.event_ts) matches python.
            CAST(
                wa.event_ts - COALESCE(ma.market_first_prior_ts_w, 0)
                AS INTEGER
            ) AS market_age_seconds,
            CAST(wa.closed_at - wa.event_ts AS INTEGER) AS time_to_resolution_seconds,
            ma.last_trade_price_w AS last_trade_price,
            CASE WHEN ma.price_count_20 >= 2 THEN ma.price_volatility_w ELSE NULL END
                AS price_volatility_recent,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'sports'
                )
                ELSE wa.category = 'sports'
            END AS INTEGER) AS cat_sports,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'esports'
                )
                ELSE wa.category = 'esports'
            END AS INTEGER) AS cat_esports,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'thesis'
                )
                ELSE wa.category = 'thesis'
            END AS INTEGER) AS cat_thesis,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'macro'
                )
                ELSE wa.category = 'macro'
            END AS INTEGER) AS cat_macro,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'elections'
                )
                ELSE wa.category = 'elections'
            END AS INTEGER) AS cat_elections,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'crypto'
                )
                ELSE wa.category = 'crypto'
            END AS INTEGER) AS cat_crypto,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'geopolitics'
                )
                ELSE wa.category = 'geopolitics'
            END AS INTEGER) AS cat_geopolitics,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'tech'
                )
                ELSE wa.category = 'tech'
            END AS INTEGER) AS cat_tech,
            CAST(CASE
                WHEN json_array_length(COALESCE(wa.categories_json, '[]')) > 0
                THEN list_contains(
                    CAST(json_extract(wa.categories_json, '$') AS VARCHAR[]), 'culture'
                )
                ELSE wa.category = 'culture'
            END AS INTEGER) AS cat_culture,
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
            USING (condition_id, event_ts, kind_priority, tx_hash, asset_id)
        WHERE wa.is_buy_only = 1
        """,  # noqa: S608 — _V2_TABLE is a module-level literal; platform/now_ts bound below
        [platform, now_ts],
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
