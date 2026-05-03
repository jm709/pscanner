"""End-to-end test for `pscanner.corpus.onchain_backfill`."""

from __future__ import annotations

import json as _json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.onchain_backfill import (
    clear_truncation_flags,
    run_onchain_backfill,
)
from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusStateRepo,
)
from pscanner.poly.onchain import ORDER_FILLED_TOPIC0
from pscanner.poly.onchain_rpc import OnchainRpcClient


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
        (0).to_bytes(32, "big"),
    ]
    return {
        "data": "0x" + b"".join(parts).hex(),
        "topics": [
            ORDER_FILLED_TOPIC0,
            "0x" + "00" * 32,
            "0x" + "00" * 12 + maker[2:],
            "0x" + "00" * 12 + taker[2:],
        ],
        "transactionHash": tx_hash,
        "blockNumber": hex(block_number),
        "logIndex": hex(log_index),
    }


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = init_corpus_db(tmp_path / "corpus.sqlite3")
    AssetIndexRepo(db).upsert(
        AssetEntry(
            asset_id="42",
            condition_id="0xCONDITION",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    try:
        yield db
    finally:
        db.close()


@respx.mock
async def test_run_onchain_backfill_inserts_trades_and_advances_cursor(
    conn: sqlite3.Connection,
) -> None:
    rpc_url = "https://example-rpc.test/"
    # making=40e6 (40 CTF tokens), taking=20e6 (20 USDC @ 6 decimals).
    # Price = 20/40 = 0.50, notional = $20 (above the $10 insert floor).
    log = _synthetic_log(
        block_number=150,
        log_index=0,
        tx_hash="0x" + "aa" * 32,
        maker_asset_id=42,
        taker_asset_id=0,
        making=40_000_000,
        taking=20_000_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0xc8"})
        if method == "eth_getLogs":
            # Only return the log for the chunk that contains block 150.
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            to_b = int(params["toBlock"], 16)
            result = [log] if from_b <= 150 <= to_b else []
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        if method == "eth_getBlockByNumber":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": "0x65f0a000"},
                },
            )
        raise AssertionError(method)

    respx.post(rpc_url).mock(side_effect=_route)

    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        summary = await run_onchain_backfill(
            conn=conn,
            rpc=client,
            from_block=0,
            to_block=200,
            chunk_size=100,
        )
    finally:
        await client.aclose()

    assert summary.events_decoded == 1
    assert summary.trades_inserted == 1
    assert summary.skipped_unresolvable == 0
    assert summary.last_block == 200

    rows = conn.execute(
        "SELECT bs, price, notional_usd, wallet_address FROM corpus_trades"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["bs"] == "BUY"
    assert rows[0]["price"] == pytest.approx(0.50)
    assert rows[0]["notional_usd"] == pytest.approx(20.0)
    assert rows[0]["wallet_address"] == "0x" + "22" * 20

    cursor = CorpusStateRepo(conn).get_int("onchain_last_block")
    assert cursor == 200


@respx.mock
async def test_run_onchain_backfill_skips_unknown_asset(
    conn: sqlite3.Connection,
) -> None:
    rpc_url = "https://example-rpc.test/"
    log = _synthetic_log(
        block_number=10,
        log_index=0,
        tx_hash="0x" + "ee" * 32,
        maker_asset_id=999_999_999,
        taker_asset_id=0,
        making=1_000_000,
        taking=500_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x32"})
        if method == "eth_getLogs":
            # Only return the log for the chunk that contains block 10.
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            to_b = int(params["toBlock"], 16)
            result = [log] if from_b <= 10 <= to_b else []
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        if method == "eth_getBlockByNumber":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": "0x65f0a000"},
                },
            )
        raise AssertionError(method)

    respx.post(rpc_url).mock(side_effect=_route)
    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        summary = await run_onchain_backfill(
            conn=conn, rpc=client, from_block=0, to_block=50, chunk_size=50
        )
    finally:
        await client.aclose()

    assert summary.events_decoded == 1
    assert summary.trades_inserted == 0
    assert summary.skipped_unresolvable == 1
    assert conn.execute("SELECT COUNT(*) FROM corpus_trades").fetchone()[0] == 0


@respx.mock
async def test_run_onchain_backfill_is_idempotent(
    conn: sqlite3.Connection,
) -> None:
    """Re-running with the same range produces no new inserts."""
    rpc_url = "https://example-rpc.test/"
    # making=40e6 (40 CTF tokens), taking=20e6 (20 USDC @ 6 decimals).
    # Notional = $20, above the $10 insert floor.
    log = _synthetic_log(
        block_number=10,
        log_index=0,
        tx_hash="0x" + "ff" * 32,
        maker_asset_id=42,
        taker_asset_id=0,
        making=40_000_000,
        taking=20_000_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x32"})
        if method == "eth_getLogs":
            # Only return the log for the chunk that contains block 10.
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            to_b = int(params["toBlock"], 16)
            result = [log] if from_b <= 10 <= to_b else []
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        if method == "eth_getBlockByNumber":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": "0x65f0a000"},
                },
            )
        raise AssertionError(method)

    respx.post(rpc_url).mock(side_effect=_route)
    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        first = await run_onchain_backfill(
            conn=conn, rpc=client, from_block=0, to_block=50, chunk_size=50
        )
        second = await run_onchain_backfill(
            conn=conn, rpc=client, from_block=0, to_block=50, chunk_size=50
        )
    finally:
        await client.aclose()

    assert first.trades_inserted == 1
    assert second.events_decoded == 1
    assert second.trades_inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM corpus_trades").fetchone()[0] == 1


def test_clear_truncation_flags_clears_market_above_threshold(
    conn: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(conn)
    cid = "0xMARKET_BIG"
    markets.insert_pending(
        CorpusMarket(
            condition_id=cid,
            event_slug="evt",
            category="politics",
            closed_at=1_700_000_000,
            total_volume_usd=2_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="some-slug",
        )
    )
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1 WHERE condition_id = ?",
        (cid,),
    )
    conn.executemany(
        """
        INSERT INTO corpus_trades (
          tx_hash, asset_id, wallet_address, condition_id, outcome_side,
          bs, price, size, notional_usd, ts
        ) VALUES (?, ?, ?, ?, 'YES', 'BUY', 0.5, 100.0, 50.0, ?)
        """,
        [(f"0xtx{i:04x}", "42", f"0x{i:040x}", cid, 1_700_000_000 + i) for i in range(3500)],
    )
    conn.commit()

    cleared = clear_truncation_flags(conn=conn, threshold=3000)
    assert cleared == 1

    row = conn.execute(
        "SELECT truncated_at_offset_cap, onchain_trades_count "
        "FROM corpus_markets WHERE condition_id = ?",
        (cid,),
    ).fetchone()
    assert row["truncated_at_offset_cap"] == 0
    assert row["onchain_trades_count"] == 3500


def test_clear_truncation_flags_skips_below_threshold(
    conn: sqlite3.Connection,
) -> None:
    """Markets with corpus_trades count < threshold keep the flag set."""
    markets = CorpusMarketsRepo(conn)
    cid = "0xMARKET_SHORT"
    markets.insert_pending(
        CorpusMarket(
            condition_id=cid,
            event_slug="evt",
            category="politics",
            closed_at=1_700_000_000,
            total_volume_usd=2_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="some-slug",
        )
    )
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1 WHERE condition_id = ?",
        (cid,),
    )
    cleared = clear_truncation_flags(conn=conn, threshold=3000)
    assert cleared == 0
    row = conn.execute(
        "SELECT truncated_at_offset_cap, onchain_trades_count "
        "FROM corpus_markets WHERE condition_id = ?",
        (cid,),
    ).fetchone()
    assert row["truncated_at_offset_cap"] == 1
    assert row["onchain_trades_count"] == 0
