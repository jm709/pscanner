"""Tests for :mod:`pscanner.kalshi.client.KalshiClient`.

All HTTP calls are intercepted by respx. Tests verify pagination, 429/retry,
and error-propagation paths.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import KalshiMarketsPage, KalshiOrderbook, KalshiTradesPage

_BASE = "https://test.kalshi.invalid/v2"

_MARKET_PAYLOAD = {
    "ticker": "KXELONMARS-99",
    "event_ticker": "KXELONMARS-99",
    "title": "Will Elon Musk visit Mars?",
    "status": "active",
    "last_price_dollars": "0.0900",
    "yes_bid_dollars": "0.0800",
    "yes_ask_dollars": "0.1000",
    "no_bid_dollars": "0.9000",
    "no_ask_dollars": "0.9200",
    "volume_fp": "100.00",
    "volume_24h_fp": "10.00",
    "open_interest_fp": "50.00",
}

_TRADE_PAYLOAD = {
    "trade_id": "abc-123",
    "ticker": "KXELONMARS-99",
    "taker_side": "yes",
    "yes_price_dollars": "0.0900",
    "no_price_dollars": "0.9100",
    "count_fp": "1.00",
    "created_time": "2026-05-04T12:00:00Z",
}


@pytest.fixture
def client() -> KalshiClient:
    """A fresh client per test with a mock base URL (high rpm to avoid rate delays)."""
    return KalshiClient(rpm=600, timeout_seconds=5.0, base_url=_BASE)


# ---------------------------------------------------------------------------
# get_markets
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_markets_returns_page(client: KalshiClient) -> None:
    respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(200, json={"markets": [_MARKET_PAYLOAD], "cursor": "next_abc"})
    )
    try:
        page = await client.get_markets()
    finally:
        await client.aclose()
    assert isinstance(page, KalshiMarketsPage)
    assert len(page.markets) == 1
    assert page.markets[0].ticker == "KXELONMARS-99"
    assert page.cursor == "next_abc"


@respx.mock
async def test_get_markets_with_status_filter(client: KalshiClient) -> None:
    route = respx.get(f"{_BASE}/markets", params={"status": "active", "limit": 100}).mock(
        return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    try:
        await client.get_markets(status="active")
    finally:
        await client.aclose()
    assert route.called


@respx.mock
async def test_get_markets_with_cursor(client: KalshiClient) -> None:
    route = respx.get(f"{_BASE}/markets", params={"cursor": "page2", "limit": 100}).mock(
        return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    try:
        await client.get_markets(cursor="page2")
    finally:
        await client.aclose()
    assert route.called


@respx.mock
async def test_get_markets_pagination_exhausted_returns_empty_cursor(
    client: KalshiClient,
) -> None:
    respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    try:
        page = await client.get_markets()
    finally:
        await client.aclose()
    assert page.cursor == ""


# ---------------------------------------------------------------------------
# get_market
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_market_single(client: KalshiClient) -> None:
    ticker = KalshiMarketTicker("KXELONMARS-99")
    respx.get(f"{_BASE}/markets/{ticker}").mock(
        return_value=httpx.Response(200, json={"market": _MARKET_PAYLOAD})
    )
    try:
        market = await client.get_market(ticker)
    finally:
        await client.aclose()
    assert market.ticker == "KXELONMARS-99"
    assert market.last_price_cents == 9


@respx.mock
async def test_get_market_404_raises(client: KalshiClient) -> None:
    ticker = KalshiMarketTicker("KXNOSUCH-1")
    respx.get(f"{_BASE}/markets/{ticker}").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_market(ticker)
    finally:
        await client.aclose()
    assert exc_info.value.response.status_code == 404


# ---------------------------------------------------------------------------
# get_orderbook
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_orderbook(client: KalshiClient) -> None:
    ticker = KalshiMarketTicker("KXELONMARS-99")
    payload = {
        "orderbook_fp": {
            "yes_dollars": [["0.0800", "100.00"]],
            "no_dollars": [["0.9000", "200.00"]],
        }
    }
    respx.get(f"{_BASE}/markets/{ticker}/orderbook").mock(
        return_value=httpx.Response(200, json=payload)
    )
    try:
        ob = await client.get_orderbook(ticker)
    finally:
        await client.aclose()
    assert isinstance(ob, KalshiOrderbook)
    assert ob.yes_bids[0] == ["0.0800", "100.00"]
    assert ob.no_bids[0] == ["0.9000", "200.00"]


@respx.mock
async def test_get_orderbook_empty(client: KalshiClient) -> None:
    ticker = KalshiMarketTicker("KXELONMARS-99")
    respx.get(f"{_BASE}/markets/{ticker}/orderbook").mock(
        return_value=httpx.Response(
            200, json={"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
        )
    )
    try:
        ob = await client.get_orderbook(ticker)
    finally:
        await client.aclose()
    assert ob.yes_bids == []
    assert ob.no_bids == []


# ---------------------------------------------------------------------------
# get_market_trades
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_market_trades_returns_page(client: KalshiClient) -> None:
    ticker = KalshiMarketTicker("KXELONMARS-99")
    respx.get(f"{_BASE}/markets/trades", params={"ticker": ticker, "limit": 100}).mock(
        return_value=httpx.Response(200, json={"trades": [_TRADE_PAYLOAD], "cursor": "next_trade"})
    )
    try:
        page = await client.get_market_trades(ticker)
    finally:
        await client.aclose()
    assert isinstance(page, KalshiTradesPage)
    assert len(page.trades) == 1
    assert page.trades[0].trade_id == "abc-123"
    assert page.cursor == "next_trade"


@respx.mock
async def test_get_market_trades_with_cursor(client: KalshiClient) -> None:
    ticker = KalshiMarketTicker("KXELONMARS-99")
    route = respx.get(
        f"{_BASE}/markets/trades",
        params={"ticker": ticker, "limit": 50, "cursor": "cursor_xyz"},
    ).mock(return_value=httpx.Response(200, json={"trades": [], "cursor": ""}))
    try:
        await client.get_market_trades(ticker, limit=50, cursor="cursor_xyz")
    finally:
        await client.aclose()
    assert route.called


def _trade(trade_id: str) -> dict[str, str]:
    """Minimal trade dict with the required fields per KalshiTrade."""
    return {
        "trade_id": trade_id,
        "ticker": "KXFOO-99",
        "taker_side": "yes",
        "yes_price_dollars": "0.0900",
        "no_price_dollars": "0.9100",
        "count_fp": "1.00",
        "created_time": "2026-05-04T12:00:00Z",
    }


@respx.mock
async def test_get_market_trades_paginates_across_two_pages(
    client: KalshiClient,
) -> None:
    """Cursor in the first response feeds into the second call."""
    page_1 = {"trades": [_trade("t1"), _trade("t2")], "cursor": "page2"}
    page_2 = {"trades": [_trade("t3"), _trade("t4")], "cursor": ""}

    def _route(request: httpx.Request) -> httpx.Response:
        if "cursor=page2" in str(request.url):
            return httpx.Response(200, json=page_2)
        return httpx.Response(200, json=page_1)

    respx.get(url__regex=r".*/markets/trades.*").mock(side_effect=_route)
    try:
        first = await client.get_market_trades(KalshiMarketTicker("KXFOO-99"), limit=2)
        second = await client.get_market_trades(
            KalshiMarketTicker("KXFOO-99"), limit=2, cursor=first.cursor
        )
    finally:
        await client.aclose()

    assert len(first.trades) == 2
    assert first.cursor == "page2"
    assert len(second.trades) == 2
    assert second.cursor in ("", None)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_429_retries_then_succeeds(client: KalshiClient) -> None:
    route = respx.get(f"{_BASE}/markets").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rate limited"}),
            httpx.Response(200, json={"markets": [], "cursor": ""}),
        ]
    )
    try:
        page = await client.get_markets()
    finally:
        await client.aclose()
    assert page.markets == []
    assert route.call_count == 2


@respx.mock
async def test_persistent_503_raises_after_max_attempts(client: KalshiClient) -> None:
    route = respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(503, json={"err": "down"})
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_markets()
    finally:
        await client.aclose()
    assert exc_info.value.response.status_code == 503
    assert route.call_count == 5


@respx.mock
async def test_500_not_retried(client: KalshiClient) -> None:
    route = respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(500, json={"err": "internal"})
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_markets()
    finally:
        await client.aclose()
    assert exc_info.value.response.status_code == 500
    assert route.call_count == 1


@respx.mock
async def test_read_timeout_retries_then_succeeds(client: KalshiClient) -> None:
    route = respx.get(f"{_BASE}/markets").mock(
        side_effect=[
            httpx.ReadTimeout("stalled"),
            httpx.Response(200, json={"markets": [], "cursor": ""}),
        ]
    )
    try:
        page = await client.get_markets()
    finally:
        await client.aclose()
    assert page.markets == []
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_context_manager_closes_client() -> None:
    respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(200, json={"markets": [], "cursor": ""})
    )
    async with KalshiClient(rpm=600, base_url=_BASE) as client:
        await client.get_markets()
        underlying = client._client  # type: ignore[attr-defined]
    assert underlying is not None
    assert underlying.is_closed is True


async def test_aclose_is_idempotent() -> None:
    client = KalshiClient(rpm=600, base_url=_BASE)
    await client.aclose()
    await client.aclose()


async def test_get_after_close_raises() -> None:
    client = KalshiClient(rpm=600, base_url=_BASE)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.get_markets()


def test_invalid_rpm_raises() -> None:
    with pytest.raises(ValueError, match="rpm"):
        KalshiClient(rpm=0)


def test_invalid_timeout_raises() -> None:
    with pytest.raises(ValueError, match="timeout"):
        KalshiClient(rpm=60, timeout_seconds=0.0)
