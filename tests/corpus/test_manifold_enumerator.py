"""Tests for the Manifold corpus enumerator."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus.manifold_enumerator import enumerate_resolved_manifold_markets
from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.manifold.models import ManifoldMarket


class _FakeManifoldClient:
    """Tiny stub that yields fixed pages of markets per call.

    Indexes through `pages` per get_markets call; cursor parameter is ignored
    because pages are pre-built with the desired filter mix.
    """

    def __init__(self, pages: list[list[ManifoldMarket]]) -> None:
        self._pages = pages
        self._call_count = 0

    async def get_markets(
        self,
        *,
        limit: int = 1000,
        before: str | None = None,
    ) -> list[ManifoldMarket]:
        if self._call_count >= len(self._pages):
            return []
        page = self._pages[self._call_count]
        self._call_count += 1
        return page


def _market(
    *,
    market_id: str,
    is_resolved: bool = True,
    outcome_type: str = "BINARY",
    volume: float = 5_000.0,
    resolution_time: int = 1_700_000_000,
    slug: str | None = None,
) -> ManifoldMarket:
    return ManifoldMarket.model_validate(
        {
            "id": market_id,
            "creatorId": "creator1",
            "question": f"Question for {market_id}?",
            "outcomeType": outcome_type,
            "mechanism": "cpmm-1",
            "volume": volume,
            "isResolved": is_resolved,
            "resolutionTime": resolution_time,
            "slug": slug or market_id,
        }
    )


@pytest.mark.asyncio
async def test_enumerate_inserts_only_resolved_binary_above_volume(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """A page with mixed conditions yields only resolved+binary+above-volume rows."""
    pages = [
        [
            _market(market_id="keep1"),
            _market(market_id="unresolved", is_resolved=False),
            _market(market_id="cfmm", outcome_type="MULTIPLE_CHOICE"),
            _market(market_id="lowvol", volume=100.0),
            _market(market_id="keep2"),
        ],
    ]
    client = _FakeManifoldClient(pages)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_manifold_markets(
        client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
        min_volume_mana=1000.0,
    )
    assert inserted == 2
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets WHERE platform = 'manifold' ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["keep1", "keep2"]


@pytest.mark.asyncio
async def test_enumerate_paginates_until_empty(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Multiple pages are walked until the client returns an empty list."""
    pages = [
        [_market(market_id="p0_a"), _market(market_id="p0_b")],
        [_market(market_id="p1_a")],
        [],
    ]
    client = _FakeManifoldClient(pages)
    repo = CorpusMarketsRepo(tmp_corpus_db)
    inserted = await enumerate_resolved_manifold_markets(
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
    pages = [[_market(market_id="m1"), _market(market_id="m2")], []]
    repo = CorpusMarketsRepo(tmp_corpus_db)

    client1 = _FakeManifoldClient(pages)
    first = await enumerate_resolved_manifold_markets(
        client1,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_000,
    )
    assert first == 2

    pages2 = [[_market(market_id="m1"), _market(market_id="m2")], []]
    client2 = _FakeManifoldClient(pages2)
    second = await enumerate_resolved_manifold_markets(
        client2,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo,
        now_ts=2_000_000_001,
    )
    assert second == 0
