"""Tests for :mod:`pscanner.poly.http`."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pscanner.poly.http import PolyHttpClient

_BASE = "https://example.test"


@pytest.fixture
def client() -> PolyHttpClient:
    """A fresh client per test (high rpm so rate-limiting isn't on the path)."""
    return PolyHttpClient(base_url=_BASE, rpm=600, timeout_seconds=5.0)


@respx.mock
async def test_get_returns_parsed_json(client: PolyHttpClient) -> None:
    route = respx.get(f"{_BASE}/v1/foo").mock(
        return_value=httpx.Response(200, json={"hello": "world"}),
    )
    try:
        result = await client.get("/v1/foo")
    finally:
        await client.aclose()

    assert result == {"hello": "world"}
    assert route.called
    assert route.calls.last.request.headers["user-agent"] == "pscanner/0.1"


@respx.mock
async def test_get_returns_list_payload(client: PolyHttpClient) -> None:
    respx.get(f"{_BASE}/v1/list").mock(
        return_value=httpx.Response(200, json=[1, 2, 3]),
    )
    try:
        result = await client.get("/v1/list")
    finally:
        await client.aclose()
    assert result == [1, 2, 3]


@respx.mock
async def test_get_passes_query_params(client: PolyHttpClient) -> None:
    route = respx.get(f"{_BASE}/v1/q", params={"a": "1", "b": "two"}).mock(
        return_value=httpx.Response(200, json={}),
    )
    try:
        await client.get("/v1/q", params={"a": 1, "b": "two"})
    finally:
        await client.aclose()
    assert route.called


@respx.mock
async def test_429_with_retry_after_zero_retries_and_succeeds(
    client: PolyHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/rl").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rl"}),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        result = await client.get("/v1/rl")
    finally:
        await client.aclose()

    assert result == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_persistent_503_raises_after_max_attempts(
    client: PolyHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/down").mock(
        return_value=httpx.Response(503, json={"err": "boom"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get("/v1/down")
    finally:
        await client.aclose()

    assert exc_info.value.response.status_code == 503
    assert route.call_count == 5


@respx.mock
async def test_500_is_not_retried(client: PolyHttpClient) -> None:
    """Per spec only 502/503/504 retry on 5xx; 500 propagates immediately."""
    route = respx.get(f"{_BASE}/v1/internal").mock(
        return_value=httpx.Response(500, json={"err": "boom"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get("/v1/internal")
    finally:
        await client.aclose()
    assert exc_info.value.response.status_code == 500
    assert route.call_count == 1


@respx.mock
async def test_404_is_not_retried(client: PolyHttpClient) -> None:
    route = respx.get(f"{_BASE}/v1/nope").mock(
        return_value=httpx.Response(404, json={"err": "not found"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get("/v1/nope")
    finally:
        await client.aclose()

    assert exc_info.value.response.status_code == 404
    assert route.call_count == 1


@respx.mock
async def test_503_retries_then_succeeds(client: PolyHttpClient) -> None:
    route = respx.get(f"{_BASE}/v1/flap").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"recovered": True}),
        ],
    )
    try:
        result = await client.get("/v1/flap")
    finally:
        await client.aclose()
    assert result == {"recovered": True}
    assert route.call_count == 2


@respx.mock
async def test_token_bucket_blocks_when_exhausted() -> None:
    """With rpm=60 (1 token/s, capacity 60), draining capacity forces a wait."""
    client = PolyHttpClient(base_url=_BASE, rpm=60)
    respx.get(f"{_BASE}/v1/ping").mock(
        return_value=httpx.Response(200, json={"ok": True}),
    )
    try:
        # Force-drain bucket: pre-warm by calling _ensure_ready then mutate.
        _, bucket = await client._ensure_ready()
        loop = asyncio.get_running_loop()
        bucket._tokens = 0.0  # type: ignore[attr-defined]
        bucket._last_refill = loop.time()  # type: ignore[attr-defined]

        start = loop.time()
        await client.get("/v1/ping")
        elapsed = loop.time() - start
    finally:
        await client.aclose()

    assert elapsed >= 0.5


@respx.mock
async def test_token_bucket_allows_burst_up_to_capacity() -> None:
    """rpm=120 (capacity 120) — three rapid calls do not block."""
    client = PolyHttpClient(base_url=_BASE, rpm=120)
    respx.get(f"{_BASE}/v1/burst").mock(
        return_value=httpx.Response(200, json={"ok": True}),
    )
    try:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await client.get("/v1/burst")
        await client.get("/v1/burst")
        await client.get("/v1/burst")
        elapsed = loop.time() - start
    finally:
        await client.aclose()

    assert elapsed < 0.25


@respx.mock
async def test_async_with_closes_underlying_client() -> None:
    respx.get(f"{_BASE}/v1/ctx").mock(
        return_value=httpx.Response(200, json={"x": 1}),
    )
    async with PolyHttpClient(base_url=_BASE, rpm=600) as client:
        result = await client.get("/v1/ctx")
        assert result == {"x": 1}
        underlying = client._client
    assert underlying is not None
    assert underlying.is_closed is True


async def test_aclose_is_idempotent() -> None:
    client = PolyHttpClient(base_url=_BASE, rpm=600)
    await client.aclose()
    await client.aclose()


async def test_get_after_close_raises() -> None:
    client = PolyHttpClient(base_url=_BASE, rpm=600)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.get("/v1/anything")


def test_invalid_rpm_rejected() -> None:
    with pytest.raises(ValueError, match="rpm"):
        PolyHttpClient(base_url=_BASE, rpm=0)


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        PolyHttpClient(base_url=_BASE, rpm=60, timeout_seconds=0.0)


@respx.mock
async def test_429_with_http_date_retry_after(client: PolyHttpClient) -> None:
    """``Retry-After`` as an HTTP-date in the past parses as zero wait."""
    route = respx.get(f"{_BASE}/v1/date").mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"},
            ),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        result = await client.get("/v1/date")
    finally:
        await client.aclose()
    assert result == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_429_with_unparseable_retry_after_still_retries(
    client: PolyHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/badheader").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "garbage"}),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        result = await client.get("/v1/badheader")
    finally:
        await client.aclose()
    assert result == {"ok": True}
    assert route.call_count == 2
