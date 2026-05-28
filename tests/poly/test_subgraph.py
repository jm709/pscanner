"""Tests for `pscanner.poly.subgraph` — async GraphQL client."""

from __future__ import annotations

import json as _json

import httpx
import pytest
import respx

from pscanner.poly.subgraph import SubgraphClient, SubgraphQuotaExhaustedError

_URL = "https://gateway.example.test/api/key/subgraphs/id/abc"
_FALLBACK_URL = "https://gateway.example.test/api/key2/subgraphs/id/abc"


@pytest.fixture
def client() -> SubgraphClient:
    return SubgraphClient(url=_URL, rpm=600, timeout_seconds=5.0)


@respx.mock
async def test_query_returns_data_payload(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"orderFilledEvents": [{"id": "0xabc"}]}},
        )
    )
    try:
        result = await client.query(
            "query Q($x: String!) { orderFilledEvents(where: {id: $x}) { id } }",
            {"x": "0xabc"},
        )
    finally:
        await client.aclose()
    assert result == {"orderFilledEvents": [{"id": "0xabc"}]}


@respx.mock
async def test_429_then_200_succeeds(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rl"}),
            httpx.Response(200, json={"data": {"x": 1}}),
        ]
    )
    try:
        result = await client.query("{ x }", {})
    finally:
        await client.aclose()
    assert result == {"x": 1}


@respx.mock
async def test_persistent_503_raises(client: SubgraphClient) -> None:
    route = respx.post(_URL).mock(return_value=httpx.Response(503, json={"err": "down"}))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.query("{ x }", {})
    finally:
        await client.aclose()
    assert route.call_count == 5


@respx.mock
async def test_graphql_errors_surface_as_runtime_error(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(200, json={"errors": [{"message": "bad query"}]})
    )
    try:
        with pytest.raises(RuntimeError, match="GraphQL errors"):
            await client.query("{ broken }", {})
    finally:
        await client.aclose()


@respx.mock
async def test_payment_required_raises_quota_exhausted(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={"errors": [{"message": "auth error: payment required for subsequent requests"}]},
        )
    )
    try:
        with pytest.raises(SubgraphQuotaExhaustedError):
            await client.query("{ x }", {})
    finally:
        await client.aclose()


@respx.mock
async def test_fallback_url_swaps_on_quota_and_retries() -> None:
    client = SubgraphClient(url=_URL, rpm=600, fallback_url=_FALLBACK_URL)
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"errors": [{"message": "auth error: payment required"}]}
        )
    )
    fallback_route = respx.post(_FALLBACK_URL).mock(
        return_value=httpx.Response(200, json={"data": {"ok": True}})
    )
    try:
        result = await client.query("{ ok }", {})
    finally:
        await client.aclose()
    assert result == {"ok": True}
    assert fallback_route.called
    assert client.url == _FALLBACK_URL  # one-shot swap is permanent


@respx.mock
async def test_fallback_url_double_quota_exhaustion_propagates() -> None:
    client = SubgraphClient(url=_URL, rpm=600, fallback_url=_FALLBACK_URL)
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"errors": [{"message": "auth error: payment required"}]}
        )
    )
    respx.post(_FALLBACK_URL).mock(
        return_value=httpx.Response(
            200, json={"errors": [{"message": "auth error: payment required"}]}
        )
    )
    try:
        with pytest.raises(SubgraphQuotaExhaustedError):
            await client.query("{ x }", {})
    finally:
        await client.aclose()


@respx.mock
async def test_no_fallback_propagates_quota_error() -> None:
    client = SubgraphClient(url=_URL, rpm=600)
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"errors": [{"message": "auth error: payment required"}]}
        )
    )
    try:
        with pytest.raises(SubgraphQuotaExhaustedError):
            await client.query("{ x }", {})
    finally:
        await client.aclose()


@respx.mock
async def test_query_sends_variables_in_body(client: SubgraphClient) -> None:
    captured: list[dict[str, object]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(_json.loads(request.read()))
        return httpx.Response(200, json={"data": {"ok": True}})

    respx.post(_URL).mock(side_effect=_capture)
    try:
        await client.query("query Q($a: Int!) { ok }", {"a": 7})
    finally:
        await client.aclose()
    assert captured[0]["query"] == "query Q($a: Int!) { ok }"
    assert captured[0]["variables"] == {"a": 7}
