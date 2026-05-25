"""End-to-end Manifold ingestion -> build_features test on a mixed-platform corpus.

Seeds a synthetic corpus with both Polymarket and Manifold rows (one Manifold
market YES-resolved, one CANCEL-only). Runs
``build_features_duckdb(platform="manifold")`` and asserts that only the
YES-resolved Manifold market's bets produce training_examples rows, and
that those rows carry ``platform='manifold'``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus._duckdb_engine import build_features_duckdb
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
)


def _seed_polymarket_row(conn: sqlite3.Connection) -> None:
    """Drop in one polymarket market with one resolved YES outcome."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="0xpoly",
            event_slug="poly-event",
            category="sports",
            closed_at=1_700_000_500,
            total_volume_usd=1_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="poly-slug",
            platform="polymarket",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xpoly-tx",
                asset_id="poly-asset",
                wallet_address="0xwallet",
                condition_id="0xpoly",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=200.0,
                notional_usd=100.0,
                ts=1_700_000_100,
                platform="polymarket",
            ),
        ]
    )
    resolutions_repo.upsert(
        MarketResolution(
            condition_id="0xpoly",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_000_500,
            source="gamma",
            platform="polymarket",
        ),
        recorded_at=1_700_000_500,
    )


def _seed_manifold_yes(conn: sqlite3.Connection) -> None:
    """Drop in one manifold market that resolves YES with two bets.

    Manifold's notional floor is $100 (mana-denominated), so the bets must
    clear that to land in ``corpus_trades``.
    """
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="m-yes",
            event_slug="m-yes-slug",
            category="sports",
            closed_at=1_700_000_600,
            total_volume_usd=5_000.0,
            enumerated_at=1_700_000_000,
            market_slug="m-yes-slug",
            platform="manifold",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="b-yes-1",
                asset_id="m-yes:YES",
                wallet_address="user-yes",
                condition_id="m-yes",
                outcome_side="YES",
                bs="BUY",
                price=0.4,
                size=500.0,
                notional_usd=200.0,
                ts=1_700_000_200,
                platform="manifold",
            ),
            CorpusTrade(
                tx_hash="b-yes-2",
                asset_id="m-yes:NO",
                wallet_address="user-no",
                condition_id="m-yes",
                outcome_side="NO",
                bs="BUY",
                price=0.6,
                size=500.0,
                notional_usd=300.0,
                ts=1_700_000_300,
                platform="manifold",
            ),
        ]
    )
    resolutions_repo.upsert(
        MarketResolution(
            condition_id="m-yes",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_000_600,
            source="manifold-rest",
            platform="manifold",
        ),
        recorded_at=1_700_000_600,
    )


def _seed_manifold_cancel(conn: sqlite3.Connection) -> None:
    """Drop in one manifold market that gets CANCELed (no market_resolutions row)."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="m-cancel",
            event_slug="m-cancel-slug",
            category="sports",
            closed_at=1_700_000_700,
            total_volume_usd=5_000.0,
            enumerated_at=1_700_000_000,
            market_slug="m-cancel-slug",
            platform="manifold",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="b-cancel-1",
                asset_id="m-cancel:YES",
                wallet_address="user-cancel",
                condition_id="m-cancel",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=500.0,
                notional_usd=200.0,
                ts=1_700_000_400,
                platform="manifold",
            ),
        ]
    )


def test_build_features_manifold_only_emits_manifold_training_examples(
    tmp_path: Path,
) -> None:
    """``build_features_duckdb(platform="manifold")`` produces only Manifold rows.

    Only the YES-resolved Manifold market should yield training_examples;
    the CANCEL market drops out via the resolution lookup, and the
    polymarket rows are filtered by the ``platform`` parameter.
    """
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    try:
        _seed_polymarket_row(conn)
        _seed_manifold_yes(conn)
        _seed_manifold_cancel(conn)
    finally:
        conn.close()

    written = build_features_duckdb(
        db_path=db_path,
        platform="manifold",
        now_ts=1_700_001_000,
        memory_limit="512MB",
        temp_dir=tmp_path / "duckdb_spill",
        threads=1,
    )
    assert written >= 1, "at least one BUY on the YES-resolved manifold market"

    read_conn = sqlite3.connect(db_path)
    read_conn.row_factory = sqlite3.Row
    try:
        rows = read_conn.execute(
            "SELECT platform, condition_id, tx_hash FROM training_examples ORDER BY tx_hash"
        ).fetchall()
    finally:
        read_conn.close()
    assert all(r["platform"] == "manifold" for r in rows)
    assert {r["condition_id"] for r in rows} == {"m-yes"}
    tx_hashes = {r["tx_hash"] for r in rows}
    assert "b-cancel-1" not in tx_hashes
    assert "0xpoly-tx" not in tx_hashes
