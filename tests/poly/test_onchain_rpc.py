"""Tests for `pscanner.poly.onchain_rpc` — Polygon JSON-RPC client."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pscanner.poly.onchain_rpc import OnchainRpcClient

_RPC_URL = "https://example-rpc.test/"


@pytest.fixture
def client() -> OnchainRpcClient:
    """A fresh RPC client per test (high rpm to neutralise rate limiting)."""
    return OnchainRpcClient(rpc_url=_RPC_URL, rpm=600, timeout_seconds=5.0)


@respx.mock
async def test_get_block_number_returns_int(client: OnchainRpcClient) -> None:
    respx.post(_RPC_URL).mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x1f4abcd"}),
    )
    try:
        head = await client.get_block_number()
    finally:
        await client.aclose()
    assert head == 0x1F4ABCD


@respx.mock
async def test_get_logs_passes_hex_block_bounds(client: OnchainRpcClient) -> None:
    captured: list[dict[str, object]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.read()))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": []})

    respx.post(_RPC_URL).mock(side_effect=_capture)
    try:
        logs = await client.get_logs(
            address="0xabc",
            topics=["0xdeadbeef"],
            from_block=100,
            to_block=200,
        )
    finally:
        await client.aclose()

    assert logs == []
    assert captured[0]["method"] == "eth_getLogs"
    params = captured[0]["params"][0]  # type: ignore[index]
    assert params["address"] == "0xabc"
    assert params["topics"] == ["0xdeadbeef"]
    assert params["fromBlock"] == "0x64"
    assert params["toBlock"] == "0xc8"


@respx.mock
async def test_get_logs_returns_payload(client: OnchainRpcClient) -> None:
    log = {
        "address": "0xabc",
        "topics": ["0xdeadbeef"],
        "data": "0x" + "00" * 256,
        "transactionHash": "0x" + "ab" * 32,
        "blockNumber": "0x10",
        "logIndex": "0x0",
    }
    respx.post(_RPC_URL).mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": [log]}),
    )
    try:
        logs = await client.get_logs(
            address="0xabc", topics=["0xdeadbeef"], from_block=16, to_block=16
        )
    finally:
        await client.aclose()

    assert len(logs) == 1
    assert logs[0]["transactionHash"] == "0x" + "ab" * 32
