"""Tests for ``pscanner.corpus.enumerator``."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import httpx
import pytest

from pscanner.corpus.enumerator import (
    VOLUME_GATE_USD,
    enumerate_closed_markets,
)
from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.poly.models import Event, Market


def _event(slug: str, markets: list[Market], closed: bool = True) -> Event:
    return Event.model_validate(
        {
            "id": slug + "-id",
            "title": "T",
            "slug": slug,
            "markets": [m.model_dump(by_alias=True) for m in markets],
            "active": False,
            "closed": closed,
            "tags": [],
        }
    )


def _market(condition_id: str, volume: float, closed: bool = True) -> Market:
    return Market.model_validate(
        {
            "id": condition_id,
            "conditionId": condition_id,
            "question": "?",
            "slug": condition_id + "-slug",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "volume": volume,
            "closed": closed,
            "active": False,
        }
    )


async def _async_events(events: list[Event]) -> AsyncIterator[Event]:
    for ev in events:
        yield ev


def _fake_gamma(events: list[Event]) -> MagicMock:
    """Return a minimal gamma stub whose ``iter_events`` yields ``events``."""
    stub = MagicMock()
    stub.iter_events = lambda **_kw: _async_events(events)
    return stub


@pytest.mark.asyncio
async def test_enumerate_inserts_above_gate(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    events = [
        _event("e1", [_market("c1", VOLUME_GATE_USD + 1)]),
        _event("e2", [_market("c2", VOLUME_GATE_USD - 1)]),
    ]
    inserted = await enumerate_closed_markets(
        gamma=_fake_gamma(events),
        repo=repo,
        now_ts=1_000,
        since_ts=None,
    )
    assert inserted == 1
    rows = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets ORDER BY condition_id"
    ).fetchall()
    assert [r["condition_id"] for r in rows] == ["c1"]


@pytest.mark.asyncio
async def test_enumerate_is_idempotent(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    events = [_event("e1", [_market("c1", VOLUME_GATE_USD + 1)])]
    await enumerate_closed_markets(
        gamma=_fake_gamma(events), repo=repo, now_ts=1_000, since_ts=None
    )
    second = await enumerate_closed_markets(
        gamma=_fake_gamma(events), repo=repo, now_ts=1_000, since_ts=None
    )
    assert second == 0
    count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM corpus_markets").fetchone()["c"]
    assert count == 1


@pytest.mark.asyncio
async def test_enumerate_skips_open_markets(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusMarketsRepo(tmp_corpus_db)
    events = [
        _event(
            "e1",
            [_market("c1", VOLUME_GATE_USD + 1, closed=False)],
            closed=True,
        )
    ]
    inserted = await enumerate_closed_markets(
        gamma=_fake_gamma(events), repo=repo, now_ts=1_000, since_ts=None
    )
    assert inserted == 0


async def _events_then_500(events: list[Event]) -> AsyncIterator[Event]:
    """Yield ``events``, then raise ``HTTPStatusError(500)`` mid-stream."""
    for ev in events:
        yield ev
    request = httpx.Request("GET", "https://gamma-api.polymarket.com/events")
    response = httpx.Response(500, request=request)
    raise httpx.HTTPStatusError("server error", request=request, response=response)


@pytest.mark.asyncio
async def test_enumerate_treats_5xx_as_end_of_catalog(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Polymarket returns 500 past a deep ``/events`` offset; the enumerator
    must persist the markets it already saw rather than aborting the run.
    """
    repo = CorpusMarketsRepo(tmp_corpus_db)
    events = [_event("e1", [_market("c1", VOLUME_GATE_USD + 1)])]
    stub = MagicMock()
    stub.iter_events = lambda **_kw: _events_then_500(events)
    inserted = await enumerate_closed_markets(gamma=stub, repo=repo, now_ts=1_000, since_ts=None)
    assert inserted == 1
    row = tmp_corpus_db.execute(
        "SELECT condition_id FROM corpus_markets WHERE condition_id = 'c1'"
    ).fetchone()
    assert row is not None


async def _events_then_4xx(events: list[Event]) -> AsyncIterator[Event]:
    for ev in events:
        yield ev
    request = httpx.Request("GET", "https://gamma-api.polymarket.com/events")
    response = httpx.Response(404, request=request)
    raise httpx.HTTPStatusError("not found", request=request, response=response)


@pytest.mark.asyncio
async def test_enumerate_propagates_4xx(tmp_corpus_db: sqlite3.Connection) -> None:
    """4xx errors from gamma should not be swallowed — they signal a real
    client problem and must surface to the caller.
    """
    repo = CorpusMarketsRepo(tmp_corpus_db)
    events = [_event("e1", [_market("c1", VOLUME_GATE_USD + 1)])]
    stub = MagicMock()
    stub.iter_events = lambda **_kw: _events_then_4xx(events)
    with pytest.raises(httpx.HTTPStatusError):
        await enumerate_closed_markets(gamma=stub, repo=repo, now_ts=1_000, since_ts=None)
