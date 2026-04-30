"""Tests for ``CorpusTradesRepo``."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import CorpusTrade, CorpusTradesRepo


def _trade(**kwargs: object) -> CorpusTrade:
    base = {
        "tx_hash": "0xtx",
        "asset_id": "asset1",
        "wallet_address": "0xw",
        "condition_id": "cond1",
        "outcome_side": "YES",
        "bs": "BUY",
        "price": 0.5,
        "size": 100.0,
        "notional_usd": 50.0,
        "ts": 1_000,
    }
    base.update(kwargs)
    return CorpusTrade(**base)  # type: ignore[arg-type]


def test_insert_batch_persists_trades(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    inserted = repo.insert_batch(
        [
            _trade(tx_hash="0xa"),
            _trade(tx_hash="0xb"),
        ]
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades ORDER BY tx_hash").fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xa", "0xb"]


def test_insert_batch_dedupes_on_unique_key(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch([_trade(tx_hash="0xa")])
    inserted = repo.insert_batch([_trade(tx_hash="0xa"), _trade(tx_hash="0xb")])
    assert inserted == 1
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM corpus_trades").fetchone()["c"]
    assert count == 2


def test_insert_batch_normalizes_wallet_to_lowercase(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch([_trade(wallet_address="0xABCDEF")])
    row = tmp_corpus_db.execute("SELECT wallet_address FROM corpus_trades").fetchone()
    assert row["wallet_address"] == "0xabcdef"


def test_insert_batch_filters_below_notional_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    inserted = repo.insert_batch(
        [
            _trade(tx_hash="0xbig", notional_usd=50.0),
            _trade(tx_hash="0xsmall", notional_usd=4.99),
        ]
    )
    assert inserted == 1
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades").fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xbig"]


def test_iter_chronological_yields_in_ts_order(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch(
        [
            _trade(tx_hash="0xc", ts=3_000),
            _trade(tx_hash="0xa", ts=1_000),
            _trade(tx_hash="0xb", ts=2_000),
        ]
    )
    seen = [t.tx_hash for t in repo.iter_chronological()]
    assert seen == ["0xa", "0xb", "0xc"]


def test_iter_chronological_breaks_ties_deterministically(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch(
        [
            _trade(tx_hash="0xa", ts=1_000),
            _trade(tx_hash="0xb", ts=1_000),
        ]
    )
    seen_first = [t.tx_hash for t in repo.iter_chronological()]
    seen_second = [t.tx_hash for t in repo.iter_chronological()]
    assert seen_first == seen_second


def test_iter_chronological_chunk_boundary_no_dup_or_skip(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Pagination across chunk boundaries must not drop or duplicate rows."""
    repo = CorpusTradesRepo(tmp_corpus_db)
    rows = [_trade(tx_hash=f"0x{i:04x}", ts=1_000 + i) for i in range(25)]
    repo.insert_batch(rows)
    seen = [t.tx_hash for t in repo.iter_chronological(chunk_size=7)]
    assert seen == [f"0x{i:04x}" for i in range(25)]


def test_iter_chronological_chunk_boundary_handles_ties(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """When a chunk ends mid-tie, the next chunk must continue past the tie."""
    repo = CorpusTradesRepo(tmp_corpus_db)
    repo.insert_batch(
        [
            _trade(tx_hash="0xa", asset_id="a", ts=1),
            _trade(tx_hash="0xa", asset_id="b", ts=1),
            _trade(tx_hash="0xa", asset_id="c", ts=1),
            _trade(tx_hash="0xa", asset_id="d", ts=1),
        ]
    )
    pairs = [(t.tx_hash, t.asset_id) for t in repo.iter_chronological(chunk_size=2)]
    assert pairs == [("0xa", "a"), ("0xa", "b"), ("0xa", "c"), ("0xa", "d")]
