"""Tests for `AssetIndexRepo`."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusTrade,
    CorpusTradesRepo,
)


def test_upsert_inserts_new_entry(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    repo.upsert(
        AssetEntry(
            asset_id="999",
            condition_id="0xabc",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    got = repo.get("999")
    assert got == AssetEntry(
        asset_id="999",
        condition_id="0xabc",
        outcome_side="YES",
        outcome_index=0,
    )


def test_get_returns_none_for_unknown_asset(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    assert repo.get("nope") is None


def test_upsert_updates_existing_entry(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    repo.upsert(AssetEntry(asset_id="1", condition_id="0xa", outcome_side="YES", outcome_index=0))
    repo.upsert(AssetEntry(asset_id="1", condition_id="0xa", outcome_side="NO", outcome_index=1))
    got = repo.get("1")
    assert got is not None
    assert got.outcome_side == "NO"
    assert got.outcome_index == 1


def test_backfill_from_corpus_trades_populates_index(tmp_corpus_db: sqlite3.Connection) -> None:
    trades = CorpusTradesRepo(tmp_corpus_db)
    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xt1",
                asset_id="100",
                wallet_address="0xw",
                condition_id="0xc1",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=100.0,
                notional_usd=50.0,
                ts=1000,
            ),
            CorpusTrade(
                tx_hash="0xt2",
                asset_id="200",
                wallet_address="0xw",
                condition_id="0xc1",
                outcome_side="NO",
                bs="BUY",
                price=0.5,
                size=100.0,
                notional_usd=50.0,
                ts=1001,
            ),
            # duplicate asset 100 — should not double-insert
            CorpusTrade(
                tx_hash="0xt3",
                asset_id="100",
                wallet_address="0xother",
                condition_id="0xc1",
                outcome_side="YES",
                bs="SELL",
                price=0.6,
                size=50.0,
                notional_usd=30.0,
                ts=1002,
            ),
        ]
    )
    repo = AssetIndexRepo(tmp_corpus_db)
    n = repo.backfill_from_corpus_trades()
    assert n == 2  # asset 100 and asset 200
    assert repo.get("100") == AssetEntry(
        asset_id="100",
        condition_id="0xc1",
        outcome_side="YES",
        outcome_index=0,
    )
    assert repo.get("200") == AssetEntry(
        asset_id="200",
        condition_id="0xc1",
        outcome_side="NO",
        outcome_index=1,
    )


def test_asset_index_isolates_platforms(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    repo.upsert(
        AssetEntry(
            asset_id="a1",
            condition_id="0xpoly",
            outcome_side="YES",
            outcome_index=0,
            platform="polymarket",
        )
    )
    repo.upsert(
        AssetEntry(
            asset_id="a1",
            condition_id="KX-1",
            outcome_side="YES",
            outcome_index=0,
            platform="kalshi",
        )
    )
    poly = repo.get("a1", platform="polymarket")
    kalshi = repo.get("a1", platform="kalshi")
    assert poly is not None
    assert poly.condition_id == "0xpoly"
    assert kalshi is not None
    assert kalshi.condition_id == "KX-1"


def test_backfill_is_idempotent(tmp_corpus_db: sqlite3.Connection) -> None:
    trades = CorpusTradesRepo(tmp_corpus_db)
    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xt1",
                asset_id="100",
                wallet_address="0xw",
                condition_id="0xc1",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=100.0,
                notional_usd=50.0,
                ts=1000,
            ),
        ]
    )
    repo = AssetIndexRepo(tmp_corpus_db)
    n1 = repo.backfill_from_corpus_trades()
    n2 = repo.backfill_from_corpus_trades()
    assert n1 == 1
    n_total = tmp_corpus_db.execute("SELECT COUNT(*) FROM asset_index").fetchone()[0]
    assert n_total == 1
    assert n2 in (0, 1)
