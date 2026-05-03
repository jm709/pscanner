"""Tests for `pscanner.poly.onchain_rpc` — Polygon JSON-RPC client."""

from __future__ import annotations

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
