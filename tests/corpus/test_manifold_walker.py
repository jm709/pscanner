"""Tests for the Manifold per-market bet walker."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.manifold_walker import walk_manifold_market
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo, CorpusTradesRepo
from pscanner.manifold.ids import ManifoldMarketId
from pscanner.manifold.models import ManifoldBet


class _FakeManifoldClient:
    def __init__(self, pages: list[list[ManifoldBet]]) -> None:
        self._pages = pages
        self._call_count = 0

    async def get_bets(
        self,
        *,
        market_id: ManifoldMarketId | None = None,
        user_id: object = None,
        limit: int = 1000,
        before: str | None = None,
    ) -> list[ManifoldBet]:
        if self._call_count >= len(self._pages):
            return []
        page = self._pages[self._call_count]
        self._call_count += 1
        return page


def _bet(
    *,
    bet_id: str,
    market_id: str = "m1",
    outcome: str = "YES",
    amount: float = 200.0,
    prob_before: float = 0.5,
    is_filled: bool | None = None,
    is_cancelled: bool | None = None,
    limit_prob: float | None = None,
    user_id: str = "user1",
) -> ManifoldBet:
    return ManifoldBet.model_validate(
        {
            "id": bet_id,
            "userId": user_id,
            "contractId": market_id,
            "outcome": outcome,
            "amount": amount,
            "probBefore": prob_before,
            "probAfter": prob_before + 0.01,
            "createdTime": 1_700_000_000,
            "isFilled": is_filled,
            "isCancelled": is_cancelled,
            "limitProb": limit_prob,
        }
    )


def _seed_market(repo: CorpusMarketsRepo, *, market_id: str = "m1") -> None:
    """Insert a pending market so the walker can mark it in_progress + complete."""
    repo.insert_pending(
        CorpusMarket(
            condition_id=market_id,
            event_slug=market_id,
            category="BINARY",
            closed_at=1_700_000_000,
            total_volume_usd=5_000.0,
            enumerated_at=1_700_000_000 - 1,
            market_slug=market_id,
            platform="manifold",
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_filled_bets_with_manifold_platform(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Filled bets land in corpus_trades with platform='manifold'."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [
        [_bet(bet_id="b1", amount=200.0), _bet(bet_id="b2", amount=300.0)],
        [],
    ]
    client = _FakeManifoldClient(pages)

    inserted = await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT platform, tx_hash, asset_id, wallet_address, price, size, notional_usd "
        "FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["platform"] == "manifold" for r in rows)
    assert {r["tx_hash"] for r in rows} == {"b1", "b2"}
    assert all(r["asset_id"] == "m1:YES" for r in rows)
    assert all(r["wallet_address"] == "user1" for r in rows)


@pytest.mark.asyncio
async def test_walk_skips_cancelled_and_unfilled_limit_orders(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Cancelled bets and unfilled limit orders never land in corpus_trades."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [
        [
            _bet(bet_id="ok", amount=200.0),
            _bet(bet_id="cancelled", amount=200.0, is_cancelled=True),
            _bet(bet_id="unfilled-limit", amount=200.0, limit_prob=0.6, is_filled=False),
            _bet(bet_id="filled-limit", amount=200.0, limit_prob=0.6, is_filled=True),
        ],
        [],
    ]
    client = _FakeManifoldClient(pages)

    inserted = await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 2  # ok + filled-limit
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades ORDER BY tx_hash").fetchall()
    assert {r["tx_hash"] for r in rows} == {"filled-limit", "ok"}


@pytest.mark.asyncio
async def test_walk_drops_below_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Bets below the 100-mana floor are dropped by CorpusTradesRepo.insert_batch."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [
        [
            _bet(bet_id="dust", amount=50.0),
            _bet(bet_id="real", amount=200.0),
        ],
        [],
    ]
    client = _FakeManifoldClient(pages)

    inserted = await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    assert inserted == 1
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades").fetchall()
    assert [r["tx_hash"] for r in rows] == ["real"]


@pytest.mark.asyncio
async def test_walk_marks_market_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """After walk completes, the corpus_markets row transitions to backfill_state='complete'."""
    markets_repo = CorpusMarketsRepo(tmp_corpus_db)
    trades_repo = CorpusTradesRepo(tmp_corpus_db)
    _seed_market(markets_repo, market_id="m1")

    pages = [[_bet(bet_id="b1", amount=200.0)], []]
    client = _FakeManifoldClient(pages)

    await walk_manifold_market(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        markets_repo,
        trades_repo,
        market_id=ManifoldMarketId("m1"),
        now_ts=2_000_000_000,
    )
    state = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets "
        "WHERE platform = 'manifold' AND condition_id = 'm1'"
    ).fetchone()
    assert state["backfill_state"] == "complete"
    assert state["truncated_at_offset_cap"] == 0
