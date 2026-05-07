"""End-to-end test: Kalshi corpus ingestion alongside Polymarket and Manifold.

Seeds a synthetic three-platform corpus DB by hand, then verifies:
- Kalshi trades land in corpus_trades with platform='kalshi' and wallet_address=''.
- Kalshi resolutions land in market_resolutions with platform='kalshi'.
- A market with `result==''` (voided) lands its trades in corpus_trades but
  has no market_resolutions row.
- `build_features(platform='manifold')` produces only manifold rows; no
  Kalshi or Polymarket rows leak.
"""

from __future__ import annotations

import sqlite3

from pscanner.corpus.examples import build_features
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
    TrainingExamplesRepo,
)


def _seed_polymarket_row(conn: sqlite3.Connection) -> None:
    """One polymarket market with a YES resolution and one BUY."""
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
    """One manifold YES market with two bets."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="m-yes",
            event_slug="m-yes-slug",
            category="BINARY",
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
                size=200.0,
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
                size=300.0,
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


def _seed_kalshi_yes(conn: sqlite3.Connection) -> None:
    """One Kalshi YES market with two trades + a YES resolution."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    resolutions_repo = MarketResolutionsRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="KX-YES",
            event_slug="KX",
            category="binary",
            closed_at=1_700_000_700,
            total_volume_usd=50_000.0,
            enumerated_at=1_700_000_000,
            market_slug="KX-YES",
            platform="kalshi",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="kx-yes-tx1",
                asset_id="KX-YES:yes",
                wallet_address="",
                condition_id="KX-YES",
                outcome_side="YES",
                bs="BUY",
                price=0.4,
                size=100.0,
                notional_usd=40.0,
                ts=1_700_000_400,
                platform="kalshi",
            ),
            CorpusTrade(
                tx_hash="kx-yes-tx2",
                asset_id="KX-YES:no",
                wallet_address="",
                condition_id="KX-YES",
                outcome_side="NO",
                bs="BUY",
                price=0.6,
                size=200.0,
                notional_usd=120.0,
                ts=1_700_000_500,
                platform="kalshi",
            ),
        ]
    )
    resolutions_repo.upsert(
        MarketResolution(
            condition_id="KX-YES",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_000_700,
            source="kalshi-rest",
            platform="kalshi",
        ),
        recorded_at=1_700_000_700,
    )


def _seed_kalshi_voided(conn: sqlite3.Connection) -> None:
    """One Kalshi market that landed in corpus but has no resolution row (result='')."""
    markets_repo = CorpusMarketsRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    markets_repo.insert_pending(
        CorpusMarket(
            condition_id="KX-VOID",
            event_slug="KX",
            category="binary",
            closed_at=1_700_000_800,
            total_volume_usd=50_000.0,
            enumerated_at=1_700_000_000,
            market_slug="KX-VOID",
            platform="kalshi",
        )
    )
    trades_repo.insert_batch(
        [
            CorpusTrade(
                tx_hash="kx-void-tx1",
                asset_id="KX-VOID:yes",
                wallet_address="",
                condition_id="KX-VOID",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=100.0,
                notional_usd=50.0,
                ts=1_700_000_600,
                platform="kalshi",
            ),
        ]
    )


def test_kalshi_data_isolated_from_other_platforms(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Kalshi rows carry platform='kalshi' and don't leak into other-platform queries."""
    _seed_polymarket_row(tmp_corpus_db)
    _seed_manifold_yes(tmp_corpus_db)
    _seed_kalshi_yes(tmp_corpus_db)
    _seed_kalshi_voided(tmp_corpus_db)

    # Both Kalshi markets have trades in corpus_trades.
    kalshi_trade_rows = tmp_corpus_db.execute(
        "SELECT condition_id, tx_hash FROM corpus_trades WHERE platform = 'kalshi' ORDER BY tx_hash"
    ).fetchall()
    kalshi_tx_hashes = {r["tx_hash"] for r in kalshi_trade_rows}
    assert kalshi_tx_hashes == {"kx-yes-tx1", "kx-yes-tx2", "kx-void-tx1"}

    # Wallet addresses are all empty strings on Kalshi rows.
    for row in kalshi_trade_rows:
        wallet = tmp_corpus_db.execute(
            "SELECT wallet_address FROM corpus_trades WHERE tx_hash = ?",
            (row["tx_hash"],),
        ).fetchone()
        assert wallet["wallet_address"] == ""

    # Only the YES-resolved Kalshi market has a market_resolutions row.
    kalshi_res_rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM market_resolutions WHERE platform = 'kalshi'"
    ).fetchall()
    assert {r["condition_id"] for r in kalshi_res_rows} == {"KX-YES"}


def test_build_features_manifold_does_not_leak_kalshi_or_polymarket(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """`build_features(platform='manifold')` produces only manifold-tagged rows."""
    _seed_polymarket_row(tmp_corpus_db)
    _seed_manifold_yes(tmp_corpus_db)
    _seed_kalshi_yes(tmp_corpus_db)
    _seed_kalshi_voided(tmp_corpus_db)

    examples_repo = TrainingExamplesRepo(tmp_corpus_db)
    written = build_features(
        trades_repo=CorpusTradesRepo(tmp_corpus_db),
        resolutions_repo=MarketResolutionsRepo(tmp_corpus_db),
        examples_repo=examples_repo,
        markets_conn=tmp_corpus_db,
        now_ts=2_000_000_000,
        rebuild=True,
        platform="manifold",
    )
    assert written >= 1

    rows = tmp_corpus_db.execute("SELECT platform, condition_id FROM training_examples").fetchall()
    platforms = {r["platform"] for r in rows}
    condition_ids = {r["condition_id"] for r in rows}
    assert platforms == {"manifold"}, f"unexpected platforms: {platforms}"
    assert "KX-YES" not in condition_ids
    assert "KX-VOID" not in condition_ids
    assert "0xpoly" not in condition_ids
