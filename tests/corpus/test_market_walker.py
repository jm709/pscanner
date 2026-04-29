"""Tests for the per-market trade walker."""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.market_walker import walk_market
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTradesRepo,
)


def _trade_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "transactionHash": "0xa",
        "asset": "asset1",
        "proxyWallet": "0xWALLET",
        "conditionId": "cond1",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.5,
        "size": 100.0,
        "timestamp": 1_000,
    }
    base.update(overrides)
    return base


def _seed_market(repo: CorpusMarketsRepo, condition_id: str) -> None:
    repo.insert_pending(
        CorpusMarket(
            condition_id=condition_id,
            event_slug="evt",
            category="crypto",
            closed_at=2_000,
            total_volume_usd=50_000.0,
            enumerated_at=500,
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_trades_and_marks_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")

    fake_data = AsyncMock()
    fake_data._fetch_market_trades_page = AsyncMock(
        side_effect=[
            [_trade_dict(transactionHash="0xa", price=0.5, size=100.0)],
            [],
        ]
    )
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute(
        "SELECT backfill_state, trades_pulled_count, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()
    assert rows["backfill_state"] == "complete"
    assert rows["trades_pulled_count"] == 1
    assert rows["truncated_at_offset_cap"] == 0
    trade_count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM corpus_trades").fetchone()["c"]
    assert trade_count == 1


@pytest.mark.asyncio
async def test_walk_normalizes_wallet_lowercases(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    fake_data._fetch_market_trades_page = AsyncMock(
        side_effect=[
            [_trade_dict(proxyWallet="0xMIXED")],
            [],
        ]
    )
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute("SELECT wallet_address FROM corpus_trades").fetchone()
    assert row["wallet_address"] == "0xmixed"


@pytest.mark.asyncio
async def test_walk_filters_below_notional_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    fake_data._fetch_market_trades_page = AsyncMock(
        side_effect=[
            [
                _trade_dict(transactionHash="0xbig", price=0.5, size=100.0),
                _trade_dict(transactionHash="0xsmall", price=0.05, size=1.0),
            ],
            [],
        ]
    )
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades").fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xbig"]


@pytest.mark.asyncio
async def test_walk_truncates_at_offset_cap(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    full_page = [_trade_dict(transactionHash=f"0x{i}") for i in range(500)]

    async def _fetch(condition_id: str, *, offset: int) -> list[dict[str, Any]]:
        del condition_id
        if offset >= 3500:
            return []
        return full_page

    fake_data._fetch_market_trades_page = AsyncMock(side_effect=_fetch)
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        markets_repo=markets,
        trades_repo=trades,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets"
    ).fetchone()
    assert row["backfill_state"] == "complete"
    assert row["truncated_at_offset_cap"] == 1
