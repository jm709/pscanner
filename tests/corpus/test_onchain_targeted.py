"""Tests for `pscanner.corpus.onchain_targeted`."""

from __future__ import annotations

import json as _json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.onchain_targeted import (
    DEPLOYMENT_BLOCK,
    TargetedRunSummary,
    run_targeted_backfill,
    ts_to_block,
)
from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
)
from pscanner.poly.onchain import ORDER_FILLED_TOPIC0
from pscanner.poly.onchain_rpc import OnchainRpcClient


def test_ts_to_block_interpolates_within_anchors() -> None:
    """Linear interpolation between two anchors lands at the right block."""
    anchors = [(1000, 1_700_000_000), (2000, 1_700_002_000)]  # 1 block per 2 sec
    # midpoint timestamp → midpoint block
    assert ts_to_block(anchors, 1_700_001_000) == 1500
    # exact match on lower anchor
    assert ts_to_block(anchors, 1_700_000_000) == 1000
    # exact match on upper anchor
    assert ts_to_block(anchors, 1_700_002_000) == 2000


def test_ts_to_block_clamps_outside_anchor_range() -> None:
    """Targets before/after the anchor range clamp to the bounds."""
    anchors = [(1000, 1_700_000_000), (2000, 1_700_002_000)]
    assert ts_to_block(anchors, 1_500_000_000) == 1000  # before range → first
    assert ts_to_block(anchors, 1_800_000_000) == 2000  # after range → last


def test_ts_to_block_raises_on_empty_anchors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ts_to_block([], 1_700_000_000)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """In-memory corpus DB with one truncated market + asset_index entries."""
    db = init_corpus_db(tmp_path / "corpus.sqlite3")
    # Two asset ids on the same condition (YES + NO)
    repo = AssetIndexRepo(db)
    repo.upsert(
        AssetEntry(
            asset_id="111",
            condition_id="0xMARKET_A",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    repo.upsert(
        AssetEntry(
            asset_id="222",
            condition_id="0xMARKET_A",
            outcome_side="NO",
            outcome_index=1,
        )
    )
    # Mark the market as truncated so it shows up as pending
    markets = CorpusMarketsRepo(db)
    markets.insert_pending(
        CorpusMarket(
            condition_id="0xMARKET_A",
            event_slug="evt-a",
            category="politics",
            closed_at=1_700_002_000,
            total_volume_usd=2_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="some-slug-a",
        )
    )
    db.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1, "
        "backfill_state = 'complete' WHERE condition_id = ?",
        ("0xMARKET_A",),
    )
    # Seed two REST-collected trades so min/max ts is non-null
    db.executemany(
        """
        INSERT INTO corpus_trades (tx_hash, asset_id, wallet_address, condition_id,
              outcome_side, bs, price, size, notional_usd, ts)
        VALUES (?, ?, ?, '0xMARKET_A', 'NO', 'BUY', 0.5, 100.0, 50.0, ?)
        """,
        [
            ("0x" + "00" * 32, "222", "0x" + "aa" * 20, 1_700_001_000),
            ("0x" + "01" * 32, "222", "0x" + "bb" * 20, 1_700_001_500),
        ],
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()


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


@respx.mock
async def test_run_targeted_backfill_processes_pending_markets(
    conn: sqlite3.Connection,
) -> None:
    """End-to-end happy path: one pending market, RPC mocked, DB updated."""
    rpc_url = "https://example-rpc.test/"
    head_block = DEPLOYMENT_BLOCK + 100_000
    # The trade timestamps in the fixture are ts=1_700_001_000 and 1_700_001_500.
    # Block timestamps use 2 sec/block from DEPLOYMENT_BLOCK at ts=1_700_000_000,
    # so trades land at blocks DEPLOYMENT_BLOCK+500 and +750 respectively.
    # Place the target log inside that window plus a small offset.
    target_block = DEPLOYMENT_BLOCK + 600  # within the trade window

    # Maker buying NO (asset 222) — matches our market
    target_log = _synthetic_log(
        block_number=target_block,
        log_index=0,
        tx_hash="0x" + "ee" * 32,
        maker_asset_id=0,
        taker_asset_id=222,
        making=20_000_000,
        taking=40_000_000,
        maker="0x" + "ff" * 20,
    )
    # Decoy event on a different asset — should be filtered out
    decoy_log = _synthetic_log(
        block_number=target_block + 1,
        log_index=0,
        tx_hash="0x" + "dd" * 32,
        maker_asset_id=0,
        taker_asset_id=999,  # not in market_assets {111, 222}
        making=20_000_000,
        taking=40_000_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": hex(head_block)}
            )
        if method == "eth_getBlockByNumber":
            block = int(body["params"][0], 16)
            # Synthesise timestamps consistent with 2 sec/block
            ts = 1_700_000_000 + (block - DEPLOYMENT_BLOCK) * 2
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": hex(ts)},
                },
            )
        if method == "eth_getLogs":
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            to_b = int(params["toBlock"], 16)
            logs = [
                log
                for log, log_block in (
                    (target_log, target_block),
                    (decoy_log, target_block + 1),
                )
                if from_b <= log_block <= to_b
            ]
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": logs})
        raise AssertionError(f"unexpected method: {method}")

    respx.post(rpc_url).mock(side_effect=_route)

    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        summary = await run_targeted_backfill(
            conn=conn,
            rpc=client,
            chunk_size=20_000,
            block_slack=10_000,
        )
    finally:
        await client.aclose()

    assert isinstance(summary, TargetedRunSummary)
    assert summary.markets_processed == 1
    assert summary.markets_failed == 0
    assert summary.events_decoded >= 1
    assert summary.trades_inserted == 1

    # Confirm the trade was inserted on the right asset
    rows = conn.execute(
        """
        SELECT bs, price, asset_id, wallet_address
        FROM corpus_trades
        WHERE condition_id = '0xMARKET_A' AND tx_hash = ?
        """,
        ("0x" + "ee" * 32,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["bs"] == "BUY"
    assert rows[0]["asset_id"] == "222"
    assert rows[0]["wallet_address"] == "0x" + "ff" * 20

    # Confirm the market was marked processed
    row = conn.execute(
        "SELECT onchain_processed_at, onchain_trades_count, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id = '0xMARKET_A'"
    ).fetchone()
    assert row["onchain_processed_at"] is not None
    assert row["onchain_processed_at"] <= int(time.time())
    assert row["onchain_trades_count"] == 3  # 2 seeded + 1 inserted
    # Below the 3000 threshold; flag stays at 1
    assert row["truncated_at_offset_cap"] == 1


@respx.mock
async def test_run_targeted_backfill_resumes_skipping_processed_markets(
    conn: sqlite3.Connection,
) -> None:
    """Markets with onchain_processed_at set must be skipped on subsequent runs."""
    # Pre-mark the market as processed
    conn.execute(
        "UPDATE corpus_markets SET onchain_processed_at = ? WHERE condition_id = ?",
        (int(time.time()) - 60, "0xMARKET_A"),
    )
    conn.commit()

    rpc_url = "https://example-rpc.test/"

    def _route(request: httpx.Request) -> httpx.Response:
        # Should never be called — no pending markets
        raise AssertionError("RPC should not be called when nothing is pending")

    respx.post(rpc_url).mock(side_effect=_route)

    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        summary = await run_targeted_backfill(conn=conn, rpc=client)
    finally:
        await client.aclose()

    assert summary == TargetedRunSummary(0, 0, 0, 0, 0, 0)
