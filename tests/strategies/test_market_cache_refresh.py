"""Tests for refresh_market_cache_row and refresh_market_cache_by_asset_id."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.models import Market
from pscanner.store.repo import MarketCacheRepo
from pscanner.strategies.market_cache_refresh import (
    refresh_market_cache_by_asset_id,
    refresh_market_cache_row,
)

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
    assert any(
        entry["event"] == "market_cache.refresh.failed" and entry.get("log_level") == "warning"
        for entry in logs
    )


_MARKET_WITH_EVENT = Market.model_validate(
    {
        "id": "mkt-rec",
        "conditionId": "0xcond-rec",
        "question": "Iran closes its airspace by May 24?",
        "slug": "iran-closes-its-airspace-by-may-24",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.5", "0.5"],
        "clobTokenIds": ["asset-y", "asset-n"],
        "active": True,
        "closed": False,
        "events": [{"id": "evt-iran", "slug": "iran-closes-its-airspace-by"}],
    },
)


@pytest.mark.asyncio
async def test_by_asset_id_happy_path(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    gamma_client = AsyncMock()
    gamma_client.list_markets.return_value = [_MARKET_WITH_EVENT]

    ok = await refresh_market_cache_by_asset_id(
        gamma_client=gamma_client,
        market_cache=cache,
        asset_id=AssetId("asset-y"),
    )

    assert ok is True
    gamma_client.list_markets.assert_awaited_once_with(clob_token_ids="asset-y", limit=5)  # noqa: S106
    cached = cache.get_by_condition_id(ConditionId("0xcond-rec"))
    assert cached is not None
    # The validator hoisted event_slug from events[0].slug.
    assert str(cached.event_slug) == "iran-closes-its-airspace-by"


@pytest.mark.asyncio
async def test_by_asset_id_no_matches_returns_false(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    gamma_client = AsyncMock()
    gamma_client.list_markets.return_value = []

    ok = await refresh_market_cache_by_asset_id(
        gamma_client=gamma_client,
        market_cache=cache,
        asset_id=AssetId("asset-unknown"),
    )

    assert ok is False
    assert cache.get_by_condition_id(ConditionId("0xcond-rec")) is None


@pytest.mark.asyncio
async def test_by_asset_id_indexing_drift_no_upsert(tmp_db) -> None:
    """Gamma sometimes returns a market whose clob_token_ids don't include the queried token."""
    cache = MarketCacheRepo(tmp_db)
    gamma_client = AsyncMock()
    gamma_client.list_markets.return_value = [_MARKET_WITH_EVENT]  # has asset-y / asset-n

    with capture_logs() as logs:
        ok = await refresh_market_cache_by_asset_id(
            gamma_client=gamma_client,
            market_cache=cache,
            asset_id=AssetId("asset-other"),  # NOT in the returned market
        )

    assert ok is False
    assert cache.get_by_condition_id(ConditionId("0xcond-rec")) is None
    assert any(
        entry["event"] == "market_cache.refresh_by_asset.indexing_drift"
        and entry.get("log_level") == "warning"
        for entry in logs
    )


@pytest.mark.asyncio
async def test_by_asset_id_exception_swallowed(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    gamma_client = AsyncMock()
    gamma_client.list_markets.side_effect = RuntimeError("boom")

    with capture_logs() as logs:
        ok = await refresh_market_cache_by_asset_id(
            gamma_client=gamma_client,
            market_cache=cache,
            asset_id=AssetId("asset-y"),
        )

    assert ok is False
    assert any(
        entry["event"] == "market_cache.refresh_by_asset.failed"
        and entry.get("log_level") == "warning"
        for entry in logs
    )
