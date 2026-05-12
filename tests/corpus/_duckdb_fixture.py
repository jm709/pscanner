"""Synthetic-corpus fixture for build-features parity tests.

Produces a small SQLite file (~8 trades, 4 markets, 3 resolutions) that
exercises the edge cases the DuckDB engine must handle bit-equivalent
to the Python engine:

- Multiple trades by same wallet at same ts (tiebreak on tx_hash, asset_id)
- A market resolution at the same ts as a buy on that market (RESOLUTION
  vs BUY tiebreak)
- Wallets with multi-category trades (top_category, category_diversity)
- A wallet with both wins and losses across markets
- BUYs on markets that never resolve (must not produce training_examples)
- SELLs interleaved with BUYs (must update recency but not BUY aggregates)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
)


def build_fixture_db(path: Path) -> None:
    """Build a deterministic corpus DB at ``path``. Overwrites if it exists."""
    if path.exists():
        path.unlink()
    conn = init_corpus_db(path)
    try:
        _insert_markets(conn)
        _insert_trades(conn)
        _insert_resolutions(conn)
        conn.commit()
    finally:
        conn.close()


def _insert_markets(conn: sqlite3.Connection) -> None:
    """Four markets: 3 resolved (sports/esports/politics), 1 unresolved.

    ``categories_json`` is populated for MKT_A (multi-label: sports + thesis)
    and MKT_B (single-label: esports) to exercise both paths through the
    ``cat_*`` indicator logic. MKT_C and MKT_D use the default ``'[]'`` so the
    fallback-to-primary-category path is covered too.
    """
    rows = [
        (
            "polymarket",
            "MKT_A",
            "ev-a",
            "sports",
            '["sports","thesis"]',
            2_000_000,
            5000.0,
            "complete",
            None,
            0,
            0,
            None,
            1_700_000_000,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "polymarket",
            "MKT_B",
            "ev-b",
            "esports",
            '["esports"]',
            2_000_500,
            7000.0,
            "complete",
            None,
            0,
            0,
            None,
            1_700_000_000,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "polymarket",
            "MKT_C",
            "ev-c",
            "politics",
            "[]",
            2_001_000,
            3000.0,
            "complete",
            None,
            0,
            0,
            None,
            1_700_000_000,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "polymarket",
            "MKT_D",
            "ev-d",
            "sports",
            "[]",
            0,
            1000.0,
            "complete",
            None,
            0,
            0,
            None,
            1_700_000_000,
            None,
            None,
            None,
            None,
            None,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO corpus_markets (
          platform, condition_id, event_slug, category, categories_json, closed_at,
          total_volume_usd, backfill_state, last_offset_seen,
          trades_pulled_count, truncated_at_offset_cap, error_message,
          enumerated_at, backfill_started_at, backfill_completed_at,
          market_slug, onchain_trades_count, onchain_processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_trades(conn: sqlite3.Connection) -> None:
    """Hand-crafted trades exercising every parity edge case.

    Layout (ts in seconds, ordered chronologically):
      1_000_000 W1 BUY  MKT_A  YES @ 0.40  size=100 notional=40
      1_000_000 W1 BUY  MKT_B  YES @ 0.30  size=100 notional=30   (same ts; tiebreak by tx_hash)
      1_000_500 W2 BUY  MKT_A  NO  @ 0.55  size=50  notional=27.5
      1_001_000 W1 SELL MKT_A  YES @ 0.45  size=30  notional=13.5
      1_500_000 W3 BUY  MKT_C  YES @ 0.20  size=200 notional=40
      1_900_000 W1 BUY  MKT_C  NO  @ 0.85  size=50  notional=42.5
      2_000_000 (RESOLUTION MKT_A yes_won=1)                       (same ts as next BUY)
      2_000_000 W2 BUY  MKT_B  NO  @ 0.65  size=80  notional=52
      2_000_500 (RESOLUTION MKT_B yes_won=0)
      2_001_000 (RESOLUTION MKT_C yes_won=1)
      2_100_000 W1 BUY  MKT_D  YES @ 0.50  size=40  notional=20   (MKT_D never resolves)

    All trades pass the $10 notional floor. The SELL (tx_d) exercises
    the edge case of recency-only updates that don't affect BUY aggregates.
    """

    def _t(
        tx: str,
        asset: str,
        wallet: str,
        cid: str,
        side: str,
        bs: str,
        price: float,
        size: float,
        notional: float,
        ts: int,
    ) -> CorpusTrade:
        return CorpusTrade(tx, asset, wallet, cid, side, bs, price, size, notional, ts)

    trades = [
        _t("tx_a", "asset_a_yes", "0xw1", "MKT_A", "YES", "BUY", 0.40, 100.0, 40.0, 1_000_000),
        _t("tx_b", "asset_b_yes", "0xw1", "MKT_B", "YES", "BUY", 0.30, 100.0, 30.0, 1_000_000),
        _t("tx_c", "asset_a_no", "0xw2", "MKT_A", "NO", "BUY", 0.55, 50.0, 27.5, 1_000_500),
        _t("tx_d", "asset_a_yes", "0xw1", "MKT_A", "YES", "SELL", 0.45, 30.0, 13.5, 1_001_000),
        _t("tx_e", "asset_c_yes", "0xw3", "MKT_C", "YES", "BUY", 0.20, 200.0, 40.0, 1_500_000),
        _t("tx_f", "asset_c_no", "0xw1", "MKT_C", "NO", "BUY", 0.85, 50.0, 42.5, 1_900_000),
        _t("tx_g", "asset_b_no", "0xw2", "MKT_B", "NO", "BUY", 0.65, 80.0, 52.0, 2_000_000),
        _t("tx_h", "asset_d_yes", "0xw1", "MKT_D", "YES", "BUY", 0.50, 40.0, 20.0, 2_100_000),
    ]
    repo = CorpusTradesRepo(conn)
    repo.insert_batch(trades)


def _insert_resolutions(conn: sqlite3.Connection) -> None:
    """Three markets resolve; MKT_D does not."""
    repo = MarketResolutionsRepo(conn)
    repo.upsert(
        MarketResolution(
            condition_id="MKT_A",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_000_000,
            source="test",
        ),
        recorded_at=2_000_100,
    )
    repo.upsert(
        MarketResolution(
            condition_id="MKT_B",
            winning_outcome_index=1,
            outcome_yes_won=0,
            resolved_at=2_000_500,
            source="test",
        ),
        recorded_at=2_000_600,
    )
    repo.upsert(
        MarketResolution(
            condition_id="MKT_C",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=2_001_000,
            source="test",
        ),
        recorded_at=2_001_100,
    )
