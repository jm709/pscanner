"""Tests for :mod:`pscanner.util.http_client`."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
import structlog
from structlog.testing import capture_logs

from pscanner.util.http_client import RateLimitedHttpClient, TokenBucket

_BASE = "https://example.test"


@pytest.fixture
def client() -> RateLimitedHttpClient:
    """A fresh client per test (high rpm so rate-limiting isn't on the path)."""
    return RateLimitedHttpClient(base_url=_BASE, rpm=600, timeout_seconds=5.0)


# ---------------------------------------------------------------------------
# Happy-path requests
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_returns_response_with_parsed_json(
    client: RateLimitedHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/foo").mock(
        return_value=httpx.Response(200, json={"hello": "world"}),
    )
    try:
        response = await client.get("/v1/foo")
    finally:
        await client.aclose()
    assert response.status_code == 200
    assert response.json() == {"hello": "world"}
    assert route.called
    assert route.calls.last.request.headers["user-agent"] == "pscanner/0.1"


@respx.mock
async def test_get_passes_query_params(client: RateLimitedHttpClient) -> None:
    route = respx.get(f"{_BASE}/v1/q", params={"a": "1", "b": "two"}).mock(
        return_value=httpx.Response(200, json={}),
    )
    try:
        await client.get("/v1/q", params={"a": 1, "b": "two"})
    finally:
        await client.aclose()
    assert route.called


@respx.mock
async def test_post_with_json_body_sets_content_type(
    client: RateLimitedHttpClient,
) -> None:
    """`json=` triggers httpx's auto Content-Type — no need to set explicitly."""
    route = respx.post(f"{_BASE}/v1/rpc").mock(
        return_value=httpx.Response(200, json={"result": 42}),
    )
    try:
        response = await client.post("/v1/rpc", json={"method": "ping"})
    finally:
        await client.aclose()
    assert response.json() == {"result": 42}
    request = route.calls.last.request
    assert request.headers["content-type"] == "application/json"


@respx.mock
async def test_absolute_url_works_without_base_url() -> None:
    """When base_url is None, request() accepts absolute URLs in `path`."""
    client = RateLimitedHttpClient(rpm=600)
    full_url = "https://example.test/v1/abs"
    respx.get(full_url).mock(return_value=httpx.Response(200, json={"ok": True}))
    try:
        response = await client.get(full_url)
    finally:
        await client.aclose()
    assert response.json() == {"ok": True}


@respx.mock
async def test_custom_user_agent_header() -> None:
    client = RateLimitedHttpClient(base_url=_BASE, rpm=600, user_agent="myagent/1.0")
    route = respx.get(f"{_BASE}/v1/ua").mock(return_value=httpx.Response(200, json={}))
    try:
        await client.get("/v1/ua")
    finally:
        await client.aclose()
    assert route.calls.last.request.headers["user-agent"] == "myagent/1.0"


# ---------------------------------------------------------------------------
# Retry-Status behaviour (429 / 5xx)
# ---------------------------------------------------------------------------


@respx.mock
async def test_429_with_retry_after_zero_retries_and_succeeds(
    client: RateLimitedHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/rl").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rl"}),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        response = await client.get("/v1/rl")
    finally:
        await client.aclose()
    assert response.json() == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_persistent_503_raises_after_max_attempts(
    client: RateLimitedHttpClient,
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
async def test_500_is_not_retried(client: RateLimitedHttpClient) -> None:
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
async def test_404_is_not_retried(client: RateLimitedHttpClient) -> None:
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
async def test_503_retries_then_succeeds(client: RateLimitedHttpClient) -> None:
    route = respx.get(f"{_BASE}/v1/flap").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"recovered": True}),
        ],
    )
    try:
        response = await client.get("/v1/flap")
    finally:
        await client.aclose()
    assert response.json() == {"recovered": True}
    assert route.call_count == 2


@respx.mock
async def test_429_with_http_date_retry_after(
    client: RateLimitedHttpClient,
) -> None:
    """`Retry-After` as an HTTP-date in the past parses as zero wait."""
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
        response = await client.get("/v1/date")
    finally:
        await client.aclose()
    assert response.json() == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_429_with_unparseable_retry_after_still_retries(
    client: RateLimitedHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/badheader").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "garbage"}),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        response = await client.get("/v1/badheader")
    finally:
        await client.aclose()
    assert response.json() == {"ok": True}
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# Transport-error retries
# ---------------------------------------------------------------------------


@respx.mock
async def test_read_timeout_retries_then_succeeds(
    client: RateLimitedHttpClient,
) -> None:
    """Transient ``httpx.ReadTimeout`` is retried, not propagated."""
    route = respx.get(f"{_BASE}/v1/slow").mock(
        side_effect=[
            httpx.ReadTimeout("upstream stalled"),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        response = await client.get("/v1/slow")
    finally:
        await client.aclose()
    assert response.json() == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_persistent_read_timeout_raises_after_max_attempts(
    client: RateLimitedHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/dead").mock(
        side_effect=httpx.ReadTimeout("upstream gone"),
    )
    try:
        with pytest.raises(httpx.ReadTimeout):
            await client.get("/v1/dead")
    finally:
        await client.aclose()
    assert route.call_count == 5


@respx.mock
async def test_connect_error_retries_then_succeeds(
    client: RateLimitedHttpClient,
) -> None:
    route = respx.get(f"{_BASE}/v1/reset").mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        response = await client.get("/v1/reset")
    finally:
        await client.aclose()
    assert response.json() == {"ok": True}
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


@respx.mock
async def test_token_bucket_blocks_when_exhausted() -> None:
    """With rpm=60 (1 token/s, capacity 60), draining capacity forces a wait."""
    client = RateLimitedHttpClient(base_url=_BASE, rpm=60)
    respx.get(f"{_BASE}/v1/ping").mock(
        return_value=httpx.Response(200, json={"ok": True}),
    )
    try:
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
    client = RateLimitedHttpClient(base_url=_BASE, rpm=120)
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


async def test_token_bucket_tokens_property_reflects_capacity() -> None:
    bucket = TokenBucket(capacity=10, rate_per_second=1.0)
    assert bucket.tokens == 10.0
    await bucket.acquire()
    assert bucket.tokens == 9.0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_with_closes_underlying_client() -> None:
    respx.get(f"{_BASE}/v1/ctx").mock(
        return_value=httpx.Response(200, json={"x": 1}),
    )
    async with RateLimitedHttpClient(base_url=_BASE, rpm=600) as client:
        response = await client.get("/v1/ctx")
        assert response.json() == {"x": 1}
        underlying = client._client
    assert underlying is not None
    assert underlying.is_closed is True


async def test_aclose_is_idempotent() -> None:
    client = RateLimitedHttpClient(base_url=_BASE, rpm=600)
    await client.aclose()
    await client.aclose()


async def test_get_after_close_raises() -> None:
    client = RateLimitedHttpClient(base_url=_BASE, rpm=600)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.get("/v1/anything")


def test_invalid_rpm_rejected() -> None:
    with pytest.raises(ValueError, match="rpm"):
        RateLimitedHttpClient(base_url=_BASE, rpm=0)


def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        RateLimitedHttpClient(base_url=_BASE, rpm=60, timeout_seconds=0.0)


# ---------------------------------------------------------------------------
# log_event constructor argument
# ---------------------------------------------------------------------------


@respx.mock
async def test_log_event_constructor_arg_drives_retry_event_name() -> None:
    """`log_event` is the structlog event name emitted on each retry sleep."""
    # Capture-logs requires the WARN-level event to flow through structlog;
    # ensure a stdlib handler-friendly config is active for this test.
    structlog.reset_defaults()
    client = RateLimitedHttpClient(base_url=_BASE, rpm=600, log_event="my_custom_retry")
    respx.get(f"{_BASE}/v1/retry").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ],
    )
    try:
        with capture_logs() as logs:
            await client.get("/v1/retry")
    finally:
        await client.aclose()
    assert any(entry["event"] == "my_custom_retry" for entry in logs)
