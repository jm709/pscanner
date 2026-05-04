"""Tests for ``pscanner.manifold.client.ManifoldClient``."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId

_BASE = "https://api.manifold.markets"

_MARKET_PAYLOAD = {
    "id": "mkABC123",
    "creatorId": "userXYZ",
    "question": "Will the market resolve YES?",
    "outcomeType": "BINARY",
    "mechanism": "cpmm-1",
    "prob": 0.5,
    "volume": 1000.0,
    "totalLiquidity": 200.0,
    "isResolved": False,
    "resolutionTime": None,
    "closeTime": 1_800_000_000_000,
}

_BET_PAYLOAD = {
    "id": "betXYZ",
    "userId": "userXYZ",
    "contractId": "mkABC123",
    "outcome": "YES",
    "amount": 10.0,
    "probBefore": 0.49,
    "probAfter": 0.50,
    "createdTime": 1_714_000_000_000,
    "isFilled": True,
    "isCancelled": False,
    "limitProb": None,
}


@pytest.fixture
def client() -> ManifoldClient:
    """Fresh client per test — high implicit timeout, won't hit real network."""
    return ManifoldClient(base_url=_BASE, timeout_seconds=5.0)


@respx.mock
async def test_get_markets_returns_list(client: ManifoldClient) -> None:
    respx.get(f"{_BASE}/v0/markets").mock(
        return_value=httpx.Response(200, json=[_MARKET_PAYLOAD]),
    )
    try:
        markets = await client.get_markets(limit=10)
    finally:
        await client.aclose()

    assert len(markets) == 1
    assert markets[0].id == "mkABC123"
    assert markets[0].is_binary is True


@respx.mock
async def test_get_markets_passes_before_cursor(client: ManifoldClient) -> None:
    route = respx.get(f"{_BASE}/v0/markets", params={"limit": "5", "before": "cursor99"}).mock(
        return_value=httpx.Response(200, json=[]),
    )
    try:
        result = await client.get_markets(limit=5, before="cursor99")
    finally:
        await client.aclose()

    assert result == []
    assert route.called


@respx.mock
async def test_get_market_single(client: ManifoldClient) -> None:
    mid = ManifoldMarketId("mkABC123")
    respx.get(f"{_BASE}/v0/market/{mid}").mock(
        return_value=httpx.Response(200, json=_MARKET_PAYLOAD),
    )
    try:
        market = await client.get_market(mid)
    finally:
        await client.aclose()

    assert market.id == mid
    assert market.question == "Will the market resolve YES?"


@respx.mock
async def test_search_markets(client: ManifoldClient) -> None:
    route = respx.get(f"{_BASE}/v0/search-markets", params={"term": "AI", "limit": "5"}).mock(
        return_value=httpx.Response(200, json=[_MARKET_PAYLOAD]),
    )
    try:
        results = await client.search_markets("AI", limit=5)
    finally:
        await client.aclose()

    assert len(results) == 1
    assert route.called


@respx.mock
async def test_get_bets_no_filter(client: ManifoldClient) -> None:
    respx.get(f"{_BASE}/v0/bets").mock(
        return_value=httpx.Response(200, json=[_BET_PAYLOAD]),
    )
    try:
        bets = await client.get_bets(limit=10)
    finally:
        await client.aclose()

    assert len(bets) == 1
    assert bets[0].id == "betXYZ"
    assert bets[0].amount == 10.0


@respx.mock
async def test_get_bets_with_market_filter(client: ManifoldClient) -> None:
    mid = ManifoldMarketId("mkABC123")
    route = respx.get(
        f"{_BASE}/v0/bets",
        params={"contractId": mid, "limit": "1000"},
    ).mock(return_value=httpx.Response(200, json=[_BET_PAYLOAD]))
    try:
        bets = await client.get_bets(market_id=mid)
    finally:
        await client.aclose()

    assert len(bets) == 1
    assert route.called


@respx.mock
async def test_get_bets_with_user_filter(client: ManifoldClient) -> None:
    uid = ManifoldUserId("userXYZ")
    route = respx.get(
        f"{_BASE}/v0/bets",
        params={"userId": uid, "limit": "1000"},
    ).mock(return_value=httpx.Response(200, json=[_BET_PAYLOAD]))
    try:
        bets = await client.get_bets(user_id=uid)
    finally:
        await client.aclose()

    assert len(bets) == 1
    assert route.called


@respx.mock
async def test_get_bets_with_before_cursor(client: ManifoldClient) -> None:
    route = respx.get(
        f"{_BASE}/v0/bets",
        params={"before": "bet-prev", "limit": "1000"},
    ).mock(return_value=httpx.Response(200, json=[]))
    try:
        await client.get_bets(before="bet-prev")
    finally:
        await client.aclose()

    assert route.called


@respx.mock
async def test_429_retries_and_succeeds(client: ManifoldClient) -> None:
    route = respx.get(f"{_BASE}/v0/markets").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rl"}),
            httpx.Response(200, json=[_MARKET_PAYLOAD]),
        ],
    )
    try:
        markets = await client.get_markets(limit=1)
    finally:
        await client.aclose()

    assert markets[0].id == "mkABC123"
    assert route.call_count == 2


@respx.mock
async def test_503_retries_then_succeeds(client: ManifoldClient) -> None:
    route = respx.get(f"{_BASE}/v0/markets").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=[_MARKET_PAYLOAD]),
        ],
    )
    try:
        markets = await client.get_markets()
    finally:
        await client.aclose()

    assert markets[0].id == "mkABC123"
    assert route.call_count == 2


@respx.mock
async def test_persistent_503_raises_after_max_attempts(client: ManifoldClient) -> None:
    route = respx.get(f"{_BASE}/v0/markets").mock(
        return_value=httpx.Response(503, json={"err": "down"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_markets()
    finally:
        await client.aclose()

    assert exc_info.value.response.status_code == 503
    assert route.call_count == 5


@respx.mock
async def test_404_not_retried(client: ManifoldClient) -> None:
    route = respx.get(f"{_BASE}/v0/market/gone").mock(
        return_value=httpx.Response(404, json={"error": "not found"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_market(ManifoldMarketId("gone"))
    finally:
        await client.aclose()

    assert exc_info.value.response.status_code == 404
    assert route.call_count == 1


@respx.mock
async def test_rate_limit_token_bucket_blocks_when_exhausted() -> None:
    """Draining bucket forces acquire to wait until tokens refill."""
    client = ManifoldClient(base_url=_BASE, timeout_seconds=5.0)
    respx.get(f"{_BASE}/v0/markets").mock(
        return_value=httpx.Response(200, json=[]),
    )
    try:
        _, bucket = await client._ensure_ready()
        loop = asyncio.get_running_loop()
        bucket._tokens = 0.0  # type: ignore[attr-defined]
        bucket._last_refill = loop.time()  # type: ignore[attr-defined]
        # With rate=500/60 ≈ 8.33/s, waiting for 1 token takes ~0.12s
        start = loop.time()
        await client.get_markets()
        elapsed = loop.time() - start
    finally:
        await client.aclose()

    assert elapsed >= 0.05


async def test_aclose_idempotent() -> None:
    client = ManifoldClient(base_url=_BASE)
    await client.aclose()
    await client.aclose()


async def test_get_after_close_raises() -> None:
    client = ManifoldClient(base_url=_BASE)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.get_markets()


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ManifoldClient(timeout_seconds=0.0)


@respx.mock
async def test_async_context_manager_closes_on_exit() -> None:
    respx.get(f"{_BASE}/v0/markets").mock(
        return_value=httpx.Response(200, json=[]),
    )
    async with ManifoldClient(base_url=_BASE) as client:
        await client.get_markets()
        underlying = client._http
    assert underlying is not None
    assert underlying.is_closed is True
