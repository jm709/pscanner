"""Tests for the Kalshi corpus enumerator."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.kalshi_enumerator import enumerate_resolved_kalshi_markets
from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.kalshi.models import KalshiMarket, KalshiMarketsPage


class _FakeKalshiClient:
    """Stub returning fixed pages keyed by status, ignoring cursor."""

    def __init__(self, pages_by_status: dict[str, list[list[KalshiMarket]]]) -> None:
        self._pages_by_status = pages_by_status
        self._call_counts: dict[str, int] = dict.fromkeys(pages_by_status, 0)

    async def get_markets(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> KalshiMarketsPage:
        if status is None or status not in self._pages_by_status:
            return KalshiMarketsPage(markets=[], cursor="")
        idx = self._call_counts[status]
        pages = self._pages_by_status[status]
        if idx >= len(pages):
            return KalshiMarketsPage(markets=[], cursor="")
        self._call_counts[status] = idx + 1
        page = pages[idx]
        next_cursor = "next" if idx + 1 < len(pages) else ""
        return KalshiMarketsPage(markets=page, cursor=next_cursor)


def _market(
    *,
    ticker: str,
    status: str = "finalized",
    result: str = "yes",
    market_type: str = "binary",
    volume_fp: float = 50_000.0,
    close_time: str = "2026-05-04T12:00:00Z",
    event_ticker: str = "KX",
) -> KalshiMarket:
    return KalshiMarket.model_validate(
        {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "title": f"Q for {ticker}",
            "status": status,
            "market_type": market_type,
            "result": result,
            "volume_fp": volume_fp,
            "close_time": close_time,
        }
    )


@pytest.mark.asyncio
async def test_enumerate_inserts_only_qualifying_kalshi_markets(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """A page with mixed conditions yields only resolved+binary+above-volume rows."""
    pages_by_status = {
        "determined": [
            [
                _market(ticker="keep1", status="determined", result="yes"),
                _market(ticker="scalar-typed", market_type="scalar", result="yes"),
                _market(ticker="lowvol", volume_fp=1000.0, result="no"),
                _market(ticker="empty-result", result=""),
            ]
        ],
        "amended": [
            [_market(ticker="keep2", status="amended", result="no")],
        ],
        "finalized": [
            [_market(ticker="keep3", status="finalized", result="yes")],
        ],
    }
    client = _FakeKalshiClient(pages_by_status)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_kalshi_markets(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
        min_volume_contracts=10_000.0,
    )
    assert inserted == 3
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets WHERE platform = 'kalshi' ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["keep1", "keep2", "keep3"]


@pytest.mark.asyncio
async def test_enumerate_paginates_until_cursor_empty(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Multiple pages within a status are walked until cursor is empty."""
    pages_by_status = {
        "determined": [
            [_market(ticker="d_a"), _market(ticker="d_b")],
            [_market(ticker="d_c")],
        ],
        "amended": [],
        "finalized": [],
    }
    client = _FakeKalshiClient(pages_by_status)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_kalshi_markets(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
    )
    assert inserted == 3


@pytest.mark.asyncio
async def test_enumerate_idempotent_on_rerun(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Re-running the enumerator over the same markets inserts zero new rows."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    pages_by_status = {
        "determined": [[_market(ticker="m1"), _market(ticker="m2")]],
        "amended": [],
        "finalized": [],
    }
    client1 = _FakeKalshiClient(pages_by_status)
    first = await enumerate_resolved_kalshi_markets(
        client1,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
    )
    assert first == 2

    pages_by_status_2 = {
        "determined": [[_market(ticker="m1"), _market(ticker="m2")]],
        "amended": [],
        "finalized": [],
    }
    client2 = _FakeKalshiClient(pages_by_status_2)
    second = await enumerate_resolved_kalshi_markets(
        client2,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_001,
    )
    assert second == 0
