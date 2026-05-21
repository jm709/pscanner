"""Tests for ``pscanner.poly.token_resolver.resolve_token``."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.models import Market
from pscanner.poly.token_resolver import resolve_token
from pscanner.store.db import init_db
from pscanner.store.repo import MarketCacheRepo

_CID = "0x6622b219b4b0796651304f85f279b18797edd2a818be471ecd82dd2d7d8d4ac7"
_TOKEN_YES = "52361899659273746688310711154377414983286902715574343626305571157053514347311"  # noqa: S105
_TOKEN_NO = "42619408858710407426561352918056373492522898259895604902369103847017159614170"  # noqa: S105


def _gamma_market_payload(*, condition_id: str = _CID) -> dict[str, Any]:
    """Build a gamma /markets payload matching the real wire shape."""
    return {
        "id": "2289432",
        "conditionId": condition_id,
        "question": "Cavaliers vs. Knicks",
        "slug": "nba-cle-nyk-2026-05-21",
        "outcomes": '["Cavaliers", "Knicks"]',
        "outcomePrices": '["0.355", "0.645"]',
        "clobTokenIds": f'["{_TOKEN_YES}", "{_TOKEN_NO}"]',
        "active": True,
        "closed": False,
    }


@pytest.fixture
def corpus_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def daemon_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = init_db(tmp_path / "pscanner.sqlite3")
    try:
        yield conn
    finally:
        conn.close()


def _make_gamma_returning(markets: list[Market]) -> AsyncMock:
    """Build an ``AsyncMock`` for ``GammaClient`` whose ``list_markets`` returns ``markets``."""
    gamma = AsyncMock(spec=GammaClient)
    gamma.list_markets.return_value = markets
    return gamma


@pytest.mark.asyncio
async def test_resolve_token_local_hit_no_gamma_call(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Local hit on both tables: gamma must not be touched."""
    asset_index = AssetIndexRepo(corpus_conn)
    asset_index.upsert(
        AssetEntry(
            asset_id=_TOKEN_NO,
            condition_id=_CID,
            outcome_side="NO",
            outcome_index=1,
        )
    )
    market_cache = MarketCacheRepo(daemon_conn)
    market_cache.upsert(Market.model_validate(_gamma_market_payload()))
    gamma = _make_gamma_returning([])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    assert result is not None
    assert result.condition_id == ConditionId(_CID)
    assert result.asset_id == AssetId(_TOKEN_NO)
    assert result.outcome_name == "Knicks"
    assert result.outcome_index == 1
    gamma.list_markets.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_token_gamma_fallback_populates_both_tables(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Local miss + gamma hit: both tables upserted, resolver returns correct tuple."""
    asset_index = AssetIndexRepo(corpus_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    gamma_market = Market.model_validate(_gamma_market_payload())
    gamma = _make_gamma_returning([gamma_market])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    # Resolver returned the right tuple.
    assert result is not None
    assert result.condition_id == ConditionId(_CID)
    assert result.outcome_name == "Knicks"
    assert result.outcome_index == 1

    # Asset_index now has both YES (idx 0) and NO (idx 1) rows.
    yes_entry = asset_index.get(_TOKEN_YES)
    no_entry = asset_index.get(_TOKEN_NO)
    assert yes_entry is not None
    assert yes_entry.outcome_side == "YES"
    assert yes_entry.outcome_index == 0
    assert no_entry is not None
    assert no_entry.outcome_side == "NO"
    assert no_entry.outcome_index == 1

    # Market_cache has the market.
    cached = market_cache.get_by_condition_id(ConditionId(_CID))
    assert cached is not None
    assert cached.condition_id == ConditionId(_CID)
    assert cached.outcomes == ["Cavaliers", "Knicks"]

    # Exactly one gamma call with the queried token id.
    gamma.list_markets.assert_awaited_once_with(clob_token_ids=_TOKEN_NO, limit=5)


@pytest.mark.asyncio
async def test_resolve_token_gamma_empty_returns_none(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Local miss + gamma 0-results: returns None, no DB writes."""
    asset_index = AssetIndexRepo(corpus_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    gamma = _make_gamma_returning([])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    assert result is None
    # No DB writes.
    assert asset_index.get(_TOKEN_NO) is None
    assert market_cache.get_by_condition_id(ConditionId(_CID)) is None


@pytest.mark.asyncio
async def test_resolve_token_gamma_payload_missing_token_returns_none(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Defensive: gamma returns a Market whose clob_token_ids don't include the queried token.

    Would indicate a gamma indexing inconsistency. Resolver returns None and
    does NOT persist any rows from the inconsistent response — neither the
    queried token, the foreign tokens, nor the market itself.
    """
    asset_index = AssetIndexRepo(corpus_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    payload = _gamma_market_payload()
    payload["clobTokenIds"] = '["1111", "2222"]'  # neither matches _TOKEN_NO
    gamma = _make_gamma_returning([Market.model_validate(payload)])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    assert result is None
    # Neither the queried token nor the foreign tokens land in asset_index.
    assert asset_index.get(_TOKEN_NO) is None
    assert asset_index.get(AssetId("1111")) is None
    assert asset_index.get(AssetId("2222")) is None
    # And the market itself is not persisted to market_cache.
    assert market_cache.get_by_condition_id(ConditionId(_CID)) is None
