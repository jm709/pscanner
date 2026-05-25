"""Tests for refresh_market_cache_row."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from pscanner.poly.ids import ConditionId
from pscanner.poly.models import Market
from pscanner.store.repo import MarketCacheRepo
from pscanner.strategies.market_cache_refresh import refresh_market_cache_row

_RESOLVED_MARKET = Market.model_validate(
    {
        "id": "mkt-1",
        "conditionId": "0xcond-1",
        "question": "Will X happen?",
        "slug": "will-x-happen",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["1.0", "0.0"],
        "clobTokenIds": ["asset-yes", "asset-no"],
        "active": False,
        "closed": True,
    },
)


@pytest.mark.asyncio
async def test_happy_path_upserts(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = "will-x-happen"
    gamma_client = AsyncMock()
    gamma_client.get_market_by_slug.return_value = _RESOLVED_MARKET

    ok = await refresh_market_cache_row(
        data_client=data_client,
        gamma_client=gamma_client,
        market_cache=cache,
        condition_id=ConditionId("0xcond-1"),
    )

    assert ok is True
    cached = cache.get_by_condition_id(ConditionId("0xcond-1"))
    assert cached is not None
    assert cached.active is False
    assert cached.outcome_prices == [1.0, 0.0]


@pytest.mark.asyncio
async def test_slug_miss_returns_false(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = None
    gamma_client = AsyncMock()

    ok = await refresh_market_cache_row(
        data_client=data_client,
        gamma_client=gamma_client,
        market_cache=cache,
        condition_id=ConditionId("0xcond-1"),
    )

    assert ok is False
    gamma_client.get_market_by_slug.assert_not_awaited()
    assert cache.get_by_condition_id(ConditionId("0xcond-1")) is None


@pytest.mark.asyncio
async def test_gamma_miss_returns_false(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = "will-x-happen"
    gamma_client = AsyncMock()
    gamma_client.get_market_by_slug.return_value = None

    ok = await refresh_market_cache_row(
        data_client=data_client,
        gamma_client=gamma_client,
        market_cache=cache,
        condition_id=ConditionId("0xcond-1"),
    )

    assert ok is False
    assert cache.get_by_condition_id(ConditionId("0xcond-1")) is None


@pytest.mark.asyncio
async def test_exception_logged_and_swallowed(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.side_effect = RuntimeError("boom")
    gamma_client = AsyncMock()

    with capture_logs() as logs:
        ok = await refresh_market_cache_row(
            data_client=data_client,
            gamma_client=gamma_client,
            market_cache=cache,
            condition_id=ConditionId("0xcond-1"),
        )

    assert ok is False
    assert any(entry["event"] == "market_cache.refresh.failed" for entry in logs)
