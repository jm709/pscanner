"""Tests for the Kalshi per-market trades walker."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.kalshi_walker import walk_kalshi_market
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTradesRepo,
)
from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import KalshiTrade, KalshiTradesPage


class _FakeKalshiClient:
    def __init__(self, pages: list[list[KalshiTrade]]) -> None:
        self._pages = pages
        self._call_count = 0

    async def get_market_trades(
        self,
        *,
        ticker: KalshiMarketTicker,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiTradesPage:
        if self._call_count >= len(self._pages):
            return KalshiTradesPage(trades=[], cursor="")
        idx = self._call_count
        self._call_count += 1
        page = self._pages[idx]
        next_cursor = "next" if idx + 1 < len(self._pages) else ""
        return KalshiTradesPage(trades=page, cursor=next_cursor)


def _trade(
    *,
    trade_id: str,
    ticker: str = "KX-1",
    taker_side: str = "yes",
    yes_price: float = 0.50,
    no_price: float = 0.50,
    count: float = 100.0,
    created_time: str = "2026-05-04T12:00:00Z",
) -> KalshiTrade:
    return KalshiTrade.model_validate(
        {
            "trade_id": trade_id,
            "ticker": ticker,
            "taker_side": taker_side,
            "yes_price_dollars": yes_price,
            "no_price_dollars": no_price,
            "count_fp": count,
            "created_time": created_time,
        }
    )


def _seed_market(repo: CorpusMarketsRepo, *, ticker: str = "KX-1") -> None:
    repo.insert_pending(
        CorpusMarket(
            condition_id=ticker,
            event_slug="KX",
            category="binary",
            closed_at=1_700_000_000,
            total_volume_usd=50_000.0,
            enumerated_at=1_700_000_000 - 1,
            market_slug=ticker,
            platform="kalshi",
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_trades_with_kalshi_platform(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Trades land in corpus_trades with platform='kalshi' and empty wallet_address."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, ticker="KX-1")

    pages = [
        [
            _trade(trade_id="t1", taker_side="yes", yes_price=0.40, count=100.0),
            _trade(trade_id="t2", taker_side="no", no_price=0.60, count=200.0),
        ],
        [],
    ]
    client = _FakeKalshiClient(pages)

    inserted = await walk_kalshi_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_ticker=KalshiMarketTicker("KX-1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT platform, tx_hash, asset_id, wallet_address, outcome_side, "
        "price, size, notional_usd, ts FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["platform"] == "kalshi" for r in rows)
    assert all(r["wallet_address"] == "" for r in rows)
    assert {r["tx_hash"] for r in rows} == {"t1", "t2"}
    by_id = {r["tx_hash"]: r for r in rows}
    # t1: yes-side at $0.40 x 100 contracts = $40 notional
    assert by_id["t1"]["asset_id"] == "KX-1:yes"
    assert by_id["t1"]["outcome_side"] == "YES"
    assert by_id["t1"]["price"] == pytest.approx(0.40)
    assert by_id["t1"]["size"] == pytest.approx(100.0)
    assert by_id["t1"]["notional_usd"] == pytest.approx(40.0)
    # t2: no-side at $0.60 x 200 = $120 notional
    assert by_id["t2"]["asset_id"] == "KX-1:no"
    assert by_id["t2"]["outcome_side"] == "NO"
    assert by_id["t2"]["price"] == pytest.approx(0.60)
    assert by_id["t2"]["notional_usd"] == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_walk_drops_below_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Trades below the $10 Kalshi floor are dropped by CorpusTradesRepo.insert_batch."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, ticker="KX-1")

    pages = [
        [
            _trade(trade_id="dust", yes_price=0.50, count=10.0),  # $5 notional
            _trade(trade_id="real", yes_price=0.50, count=100.0),  # $50 notional
        ],
        [],
    ]
    client = _FakeKalshiClient(pages)

    inserted = await walk_kalshi_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_ticker=KalshiMarketTicker("KX-1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 1


@pytest.mark.asyncio
async def test_walk_marks_market_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """After walk completes, the corpus_markets row transitions to backfill_state='complete'."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, ticker="KX-1")

    pages = [[_trade(trade_id="t1", count=100.0)], []]
    client = _FakeKalshiClient(pages)

    await walk_kalshi_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_ticker=KalshiMarketTicker("KX-1"),
        now_ts=2_000_000_000,
    )
    state = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets "
        "WHERE platform = 'kalshi' AND condition_id = 'KX-1'"
    ).fetchone()
    assert state["backfill_state"] == "complete"
    assert state["truncated_at_offset_cap"] == 0
