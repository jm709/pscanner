"""Tests for `pscanner.poly.onchain_ingest` — event→trade conversion + paginator."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import ORDER_FILLED_TOPIC0, OrderFilledEvent
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
    iter_order_filled_logs,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient


@pytest.fixture
def asset_repo(tmp_path: Path) -> Iterator[AssetIndexRepo]:
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        repo = AssetIndexRepo(conn)
        repo.upsert(
            AssetEntry(
                asset_id="123456789",
                condition_id="0xCONDITION",
                outcome_side="YES",
                outcome_index=0,
            )
        )
        yield repo
    finally:
        conn.close()


def _ev(
    *,
    maker: str = "0x" + "11" * 20,
    taker: str = "0x" + "22" * 20,
    maker_asset_id: int = 0,
    taker_asset_id: int = 123_456_789,
    making: int = 700_000,
    taking: int = 1_000_000,
) -> OrderFilledEvent:
    return OrderFilledEvent(
        order_hash="0x" + "ab" * 32,
        maker=maker,
        taker=taker,
        maker_asset_id=maker_asset_id,
        taker_asset_id=taker_asset_id,
        making=making,
        taking=taking,
        fee=0,
        tx_hash="0x" + "cd" * 32,
        block_number=42,
        log_index=0,
    )


def test_event_to_corpus_trade_buy_taker_gives_usdc(
    asset_repo: AssetIndexRepo,
) -> None:
    """Taker giving USDC for CTF tokens is a BUY from the taker's POV."""
    event = _ev(
        maker_asset_id=123_456_789,
        taker_asset_id=0,
        making=1_000_000,  # 1.0 CTF the maker is giving
        taking=700_000,  # 0.70 USDC the taker is giving
    )
    trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=1_700_000_000)
    assert isinstance(trade, CorpusTrade)
    assert trade.tx_hash == "0x" + "cd" * 32
    assert trade.asset_id == "123456789"
    assert trade.condition_id == "0xCONDITION"
    assert trade.outcome_side == "YES"
    assert trade.wallet_address == "0x" + "22" * 20  # taker
    assert trade.bs == "BUY"
    assert trade.price == pytest.approx(0.70)
    assert trade.size == pytest.approx(1.0)
    assert trade.notional_usd == pytest.approx(0.70)
    assert trade.ts == 1_700_000_000


def test_event_to_corpus_trade_sell_taker_gives_ctf(
    asset_repo: AssetIndexRepo,
) -> None:
    """Taker giving CTF tokens for USDC is a SELL from the taker's POV."""
    event = _ev(
        maker_asset_id=0,
        taker_asset_id=123_456_789,
        making=420_000,  # 0.42 USDC the maker gives
        taking=1_000_000,  # 1.0 CTF the taker gives
    )
    trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=1_700_000_000)
    assert trade.bs == "SELL"
    assert trade.wallet_address == "0x" + "22" * 20  # taker
    assert trade.price == pytest.approx(0.42)
    assert trade.size == pytest.approx(1.0)
    assert trade.notional_usd == pytest.approx(0.42)


def test_event_to_corpus_trade_raises_when_both_assets_zero(
    asset_repo: AssetIndexRepo,
) -> None:
    event = _ev(maker_asset_id=0, taker_asset_id=0)
    with pytest.raises(UnsupportedFill, match="both-zero"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)


def test_event_to_corpus_trade_raises_when_both_assets_nonzero(
    asset_repo: AssetIndexRepo,
) -> None:
    event = _ev(maker_asset_id=42, taker_asset_id=99)
    with pytest.raises(UnsupportedFill, match="both-zero or both-non-zero"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)


def test_event_to_corpus_trade_raises_when_asset_unknown(
    asset_repo: AssetIndexRepo,
) -> None:
    event = _ev(maker_asset_id=0, taker_asset_id=999_999_999)  # not in repo
    with pytest.raises(UnresolvableAsset, match="999999999"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)


def _synthetic_log(
    *,
    block_number: int,
    log_index: int,
    tx_hash: str,
    maker_asset_id: int,
    taker_asset_id: int,
    making: int,
    taking: int,
    maker: str = "0x" + "11" * 20,
    taker: str = "0x" + "22" * 20,
) -> dict[str, object]:
    parts = [
        maker_asset_id.to_bytes(32, "big"),
        taker_asset_id.to_bytes(32, "big"),
        making.to_bytes(32, "big"),
        taking.to_bytes(32, "big"),
        (0).to_bytes(32, "big"),  # fee
    ]
    return {
        "data": "0x" + b"".join(parts).hex(),
        "topics": [
            ORDER_FILLED_TOPIC0,
            "0x" + "00" * 32,  # orderHash
            "0x" + "00" * 12 + maker[2:],
            "0x" + "00" * 12 + taker[2:],
        ],
        "transactionHash": tx_hash,
        "blockNumber": hex(block_number),
        "logIndex": hex(log_index),
    }


@respx.mock
async def test_iter_order_filled_logs_chunks_and_yields() -> None:
    rpc_url = "https://example-rpc.test/"

    log_a = _synthetic_log(
        block_number=100,
        log_index=0,
        tx_hash="0x" + "aa" * 32,
        maker_asset_id=0,
        taker_asset_id=42,
        making=500_000,
        taking=1_000_000,
    )
    log_b = _synthetic_log(
        block_number=200,
        log_index=1,
        tx_hash="0x" + "bb" * 32,
        maker_asset_id=42,
        taker_asset_id=0,
        making=1_000_000,
        taking=500_000,
    )

    posts: list[dict[str, object]] = []

    def _route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        posts.append(body)
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x12c"})
        if method == "eth_getLogs":
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            if from_b == 0:
                return httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log_a]}
                )
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log_b]})
        if method == "eth_getBlockByNumber":
            block = int(body["params"][0], 16)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": hex(1_700_000_000 + block)},
                },
            )
        raise AssertionError(f"unexpected method: {method}")

    respx.post(rpc_url).mock(side_effect=_route)

    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    yielded: list[tuple[int, int]] = []
    try:
        async for event, ts in iter_order_filled_logs(
            rpc=client,
            from_block=0,
            to_block=300,
            chunk_size=200,
        ):
            yielded.append((event.block_number, ts))
    finally:
        await client.aclose()

    assert yielded == [(100, 1_700_000_100), (200, 1_700_000_200)]
    methods = [p["method"] for p in posts]
    assert methods.count("eth_getLogs") == 2
    assert methods.count("eth_getBlockByNumber") == 2


async def test_iter_order_filled_logs_from_block_greater_than_to_block() -> None:
    """Early return when from_block > to_block yields nothing (no RPC calls)."""
    client = OnchainRpcClient(rpc_url="https://unused.test/", rpm=600)
    yielded: list[tuple[OrderFilledEvent, int]] = []
    try:
        async for item in iter_order_filled_logs(
            rpc=client, from_block=100, to_block=50, chunk_size=200
        ):
            yielded.append(item)
    finally:
        await client.aclose()
    assert yielded == []


async def test_iter_order_filled_logs_rejects_nonpositive_chunk_size() -> None:
    """Raises ValueError for non-positive chunk_size before yielding."""
    client = OnchainRpcClient(rpc_url="https://unused.test/", rpm=600)
    try:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            async for _ in iter_order_filled_logs(
                rpc=client, from_block=0, to_block=100, chunk_size=0
            ):
                pass
    finally:
        await client.aclose()
