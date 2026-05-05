from __future__ import annotations

import json as _json
import sqlite3
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.corpus.subgraph_ingest import (
    SubgraphRunSummary,
    iter_market_trades,
    run_subgraph_backfill,
    subgraph_row_to_event,
)
from pscanner.poly.subgraph import SubgraphClient

_GATEWAY_URL = "https://gateway.example.test/api/k/subgraphs/id/abc"


def test_subgraph_row_to_event_parses_buy_side_row() -> None:
    """Maker BUY: maker gives USDC ('0'), taker gives CTF token."""
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0xMaker_Address_NOT_LowerCased",
        "taker": "0x" + "22" * 20,
        "makerAssetId": "0",
        "takerAssetId": "222",
        "makerAmountFilled": "20000000",
        "takerAmountFilled": "40000000",
        "fee": "0",
    }
    event = subgraph_row_to_event(row)
    assert event.tx_hash == "0xee" * 32
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 222
    assert event.making == 20_000_000
    assert event.taking == 40_000_000
    assert event.fee == 0
    assert event.block_number == 0
    assert event.log_index == 0
    # event_to_corpus_trade lowercases the maker; the dataclass preserves whatever's passed in
    assert event.maker == "0xMaker_Address_NOT_LowerCased"


def test_subgraph_row_to_event_rejects_missing_field() -> None:
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        # orderHash deliberately missing
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": "0",
        "takerAssetId": "222",
        "makerAmountFilled": "1",
        "takerAmountFilled": "1",
        "fee": "0",
    }
    with pytest.raises(KeyError, match="orderHash"):
        subgraph_row_to_event(row)


def test_subgraph_row_to_event_rejects_non_numeric_amount() -> None:
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": "0",
        "takerAssetId": "222",
        "makerAmountFilled": "not-a-number",
        "takerAmountFilled": "40000000",
        "fee": "0",
    }
    with pytest.raises(ValueError, match="makerAmountFilled could not be parsed"):
        subgraph_row_to_event(row)


def test_subgraph_row_to_event_parses_sell_side_row() -> None:
    """Maker SELL: maker gives CTF token, taker gives USDC ('0')."""
    # Realistic large CTF token-id (2**255 + something)
    ctf_token_id = 57896044618658097711785492504343953926634992332820282019728792003956564819968
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xff" * 32,
        "timestamp": "1700001500",
        "orderHash": "0x" + "cd" * 32,
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": str(ctf_token_id),
        "takerAssetId": "0",
        "makerAmountFilled": "40000000",
        "takerAmountFilled": "20000000",
        "fee": "0",
    }
    event = subgraph_row_to_event(row)
    assert event.maker_asset_id == ctf_token_id
    assert event.taker_asset_id == 0
    assert event.making == 40_000_000
    assert event.taking == 20_000_000


def test_subgraph_row_to_event_accepts_int_values_for_bigints() -> None:
    """Some GraphQL clients deserialize BigInts as native ints, not strings."""
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": 0,  # int, not string
        "takerAssetId": 222,
        "makerAmountFilled": 20_000_000,
        "takerAmountFilled": 40_000_000,
        "fee": 0,
    }
    event = subgraph_row_to_event(row)
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 222
    assert event.making == 20_000_000
    assert event.taking == 40_000_000


# ---------------------------------------------------------------------------
# iter_market_trades paginator tests
# ---------------------------------------------------------------------------


async def test_iter_market_trades_paginates_both_sides() -> None:
    """Paginator runs maker-side then taker-side, yields decoded events."""
    side_responses = {
        # Maker-side page 1 (full page → another page expected)
        ("makerAssetId_in", ""): [
            _row(
                id_="0x01_a", maker_asset="111", taker_asset="0", making=1_000_000, taking=2_000_000
            ),
        ],
        # Maker-side page 2 (smaller than 'first' → done)
        ("makerAssetId_in", "0x01_a"): [],
        # Taker-side page 1
        ("takerAssetId_in", ""): [
            _row(
                id_="0x02_b", maker_asset="0", taker_asset="111", making=1_000_000, taking=2_000_000
            ),
        ],
        ("takerAssetId_in", "0x02_b"): [],
    }

    async def fake_query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        side = "makerAssetId_in" if "makerAssetId_in" in graphql else "takerAssetId_in"
        cursor = variables.get("cursor", "")
        rows = side_responses[(side, cursor)]
        return {"orderFilledEvents": rows}

    client = AsyncMock()
    client.query.side_effect = fake_query

    yielded = []
    async for event, ts in iter_market_trades(
        client=client,
        asset_ids=["111"],
        page_size=1,
    ):
        yielded.append((event, ts))

    assert len(yielded) == 2
    # Maker-side first
    assert yielded[0][0].maker_asset_id == 111
    assert yielded[0][0].taker_asset_id == 0
    # Then taker-side
    assert yielded[1][0].maker_asset_id == 0
    assert yielded[1][0].taker_asset_id == 111
    # Timestamps preserved
    assert yielded[0][1] == 1_700_000_000


async def test_iter_market_trades_empty_asset_ids_skips_query() -> None:
    client = AsyncMock()
    yielded = []
    async for ev, ts in iter_market_trades(client=client, asset_ids=[], page_size=10):
        yielded.append((ev, ts))
    assert yielded == []
    client.query.assert_not_called()


async def test_iter_market_trades_short_first_page_exits_without_second_query() -> None:
    """When the first page is shorter than page_size, no further query runs for that side."""
    side_calls: dict[str, int] = {"maker": 0, "taker": 0}

    async def fake_query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        side = "maker" if "makerAssetId_in" in graphql else "taker"
        side_calls[side] += 1
        if side == "maker":
            return {
                "orderFilledEvents": [
                    _row(
                        id_="0x01_a",
                        maker_asset="111",
                        taker_asset="0",
                        making=1_000_000,
                        taking=2_000_000,
                    )
                ]
            }
        return {"orderFilledEvents": []}

    client = AsyncMock()
    client.query.side_effect = fake_query

    yielded = []
    async for ev, ts in iter_market_trades(client=client, asset_ids=["111"], page_size=100):
        yielded.append((ev, ts))

    assert len(yielded) == 1
    # Maker side: 1 query (short page → no follow-up). Taker side: 1 query (empty).
    assert side_calls == {"maker": 1, "taker": 1}


@pytest.mark.parametrize("bad_size", [0, -1, 1001])
async def test_iter_market_trades_rejects_invalid_page_size(bad_size: int) -> None:
    """page_size must be in 1..1000 (subgraph hard limit)."""
    client = AsyncMock()
    with pytest.raises(ValueError, match="page_size must be in 1"):
        async for _ in iter_market_trades(client=client, asset_ids=["111"], page_size=bad_size):
            pass
    client.query.assert_not_called()


def _row(
    *, id_: str, maker_asset: str, taker_asset: str, making: int, taking: int
) -> dict[str, str]:
    return {
        "id": id_,
        "transactionHash": "0x" + "ee" * 32,
        "timestamp": "1700000000",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": maker_asset,
        "takerAssetId": taker_asset,
        "makerAmountFilled": str(making),
        "takerAmountFilled": str(taking),
        "fee": "0",
    }


# ---------------------------------------------------------------------------
# run_subgraph_backfill orchestrator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Corpus DB seeded with one truncated market and its asset_index entries."""
    db = init_corpus_db(tmp_path / "c.sqlite3")
    try:
        markets = CorpusMarketsRepo(db)
        markets.insert_pending(
            CorpusMarket(
                condition_id="0xMARKET_A",
                event_slug="some-event",
                category=None,
                closed_at=1_700_001_000,
                total_volume_usd=42_000.0,
                enumerated_at=1_700_000_000,
                market_slug="some-market",
            )
        )
        db.execute(
            """
            UPDATE corpus_markets
            SET truncated_at_offset_cap = 1, backfill_state = 'complete'
            WHERE condition_id = ?
            """,
            ("0xMARKET_A",),
        )
        db.commit()
        AssetIndexRepo(db).upsert(
            AssetEntry(
                asset_id="111",
                condition_id="0xMARKET_A",
                outcome_side="YES",
                outcome_index=0,
            )
        )
        AssetIndexRepo(db).upsert(
            AssetEntry(
                asset_id="222",
                condition_id="0xMARKET_A",
                outcome_side="NO",
                outcome_index=1,
            )
        )
        CorpusTradesRepo(db).insert_batch(
            [
                CorpusTrade(
                    tx_hash="0x" + "aa" * 32,
                    asset_id="111",
                    wallet_address="0x" + "11" * 20,
                    condition_id="0xMARKET_A",
                    outcome_side="YES",
                    bs="BUY",
                    price=0.5,
                    size=20.0,
                    notional_usd=10.0,
                    ts=1_700_000_500,
                ),
                CorpusTrade(
                    tx_hash="0x" + "bb" * 32,
                    asset_id="111",
                    wallet_address="0x" + "11" * 20,
                    condition_id="0xMARKET_A",
                    outcome_side="YES",
                    bs="BUY",
                    price=0.5,
                    size=20.0,
                    notional_usd=10.0,
                    ts=1_700_000_900,
                ),
            ]
        )
        yield db
    finally:
        db.close()


@respx.mock
async def test_run_subgraph_backfill_processes_pending_market(
    conn: sqlite3.Connection,
) -> None:
    """End-to-end: one pending market → 1 trade inserted → market marked processed."""

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        side = "maker" if "makerAssetId_in" in body["query"] else "taker"
        cursor = body["variables"]["cursor"]
        if side == "maker" and cursor == "":
            # SELL from maker POV: maker gives CTF (asset 111), taker gives USDC
            return httpx.Response(
                200,
                json={
                    "data": {
                        "orderFilledEvents": [
                            {
                                "id": "0xtx1_0xord1",
                                "transactionHash": "0x" + "ee" * 32,
                                "timestamp": "1700001500",
                                "orderHash": "0x" + "ab" * 32,
                                "maker": "0x" + "ff" * 20,
                                "taker": "0x" + "22" * 20,
                                "makerAssetId": "111",
                                "takerAssetId": "0",
                                "makerAmountFilled": "40000000",
                                "takerAmountFilled": "20000000",
                                "fee": "0",
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"data": {"orderFilledEvents": []}})

    respx.post(_GATEWAY_URL).mock(side_effect=_route)

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client)
    finally:
        await client.aclose()

    assert isinstance(summary, SubgraphRunSummary)
    assert summary.markets_processed == 1
    assert summary.markets_failed == 0
    assert summary.trades_inserted == 1

    rows = conn.execute(
        """
        SELECT bs, asset_id, wallet_address, ts FROM corpus_trades
        WHERE condition_id = '0xMARKET_A' AND tx_hash = ?
        """,
        ("0x" + "ee" * 32,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["bs"] == "SELL"  # maker gave CTF → SELL from maker POV
    assert rows[0]["asset_id"] == "111"
    assert rows[0]["wallet_address"] == "0x" + "ff" * 20
    assert rows[0]["ts"] == 1_700_001_500

    row = conn.execute(
        "SELECT onchain_processed_at, onchain_trades_count, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id = '0xMARKET_A'"
    ).fetchone()
    assert row["onchain_processed_at"] is not None
    assert row["onchain_processed_at"] <= int(time.time())
    assert row["onchain_trades_count"] == 3  # 2 seeded + 1 inserted
    assert row["truncated_at_offset_cap"] == 1  # below 3000 threshold


@respx.mock
async def test_run_subgraph_backfill_skips_already_processed_markets(
    conn: sqlite3.Connection,
) -> None:
    """Markets with onchain_processed_at set must be skipped on subsequent runs."""
    conn.execute(
        "UPDATE corpus_markets SET onchain_processed_at = ? WHERE condition_id = ?",
        (int(time.time()) - 60, "0xMARKET_A"),
    )
    conn.commit()

    route = respx.post(_GATEWAY_URL).mock(
        return_value=httpx.Response(200, json={"data": {"orderFilledEvents": []}})
    )

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client)
    finally:
        await client.aclose()

    assert summary.markets_processed == 0
    assert route.call_count == 0  # no markets pending → no queries fired


@respx.mock
async def test_run_subgraph_backfill_respects_limit(
    conn: sqlite3.Connection,
) -> None:
    """`limit=N` processes at most N markets even if more are pending."""
    CorpusMarketsRepo(conn).insert_pending(
        CorpusMarket(
            condition_id="0xMARKET_B",
            event_slug="event-b",
            category=None,
            closed_at=1_700_001_000,
            total_volume_usd=10_000.0,
            enumerated_at=1_700_000_000,
            market_slug="market-b",
        )
    )
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1, backfill_state = 'complete' "
        "WHERE condition_id = ?",
        ("0xMARKET_B",),
    )
    conn.commit()
    AssetIndexRepo(conn).upsert(
        AssetEntry(
            asset_id="333",
            condition_id="0xMARKET_B",
            outcome_side="YES",
            outcome_index=0,
        )
    )

    respx.post(_GATEWAY_URL).mock(
        return_value=httpx.Response(200, json={"data": {"orderFilledEvents": []}})
    )

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client, limit=1)
    finally:
        await client.aclose()

    assert summary.markets_processed == 1

    # Volume-ordering: MARKET_A (42k) should run before MARKET_B (10k),
    # so with limit=1 only MARKET_A gets onchain_processed_at set.
    row_a = conn.execute(
        "SELECT onchain_processed_at FROM corpus_markets WHERE condition_id = ?",
        ("0xMARKET_A",),
    ).fetchone()
    row_b = conn.execute(
        "SELECT onchain_processed_at FROM corpus_markets WHERE condition_id = ?",
        ("0xMARKET_B",),
    ).fetchone()
    assert row_a["onchain_processed_at"] is not None
    assert row_b["onchain_processed_at"] is None


@respx.mock
async def test_run_subgraph_backfill_records_market_failure_and_continues(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A market whose paginator raises is recorded as failed, run continues."""
    # Add a second pending market so we can verify the run continues past the failure.
    CorpusMarketsRepo(conn).insert_pending(
        CorpusMarket(
            condition_id="0xMARKET_B",
            event_slug="event-b",
            category=None,
            closed_at=1_700_001_000,
            total_volume_usd=10_000.0,
            enumerated_at=1_700_000_000,
            market_slug="market-b",
        )
    )
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1, backfill_state = 'complete' "
        "WHERE condition_id = ?",
        ("0xMARKET_B",),
    )
    conn.commit()
    AssetIndexRepo(conn).upsert(
        AssetEntry(asset_id="333", condition_id="0xMARKET_B", outcome_side="YES", outcome_index=0)
    )

    # First call raises (the highest-volume market, MARKET_A — volume 42k).
    # Second call succeeds with empty result (MARKET_B — volume 10k).
    call_count = {"n": 0}

    async def fake_iter(**kwargs: object):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated subgraph failure")
        # Empty async generator
        if False:
            yield None  # pragma: no cover

    monkeypatch.setattr("pscanner.corpus.subgraph_ingest.iter_market_trades", fake_iter)

    respx.post(_GATEWAY_URL).mock(
        return_value=httpx.Response(200, json={"data": {"orderFilledEvents": []}})
    )

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client)
    finally:
        await client.aclose()

    assert summary.markets_failed == 1
    assert summary.markets_processed == 1  # the second market still processed

    # Failed market should NOT have onchain_processed_at set
    row_a = conn.execute(
        "SELECT onchain_processed_at FROM corpus_markets WHERE condition_id = ?",
        ("0xMARKET_A",),
    ).fetchone()
    assert row_a["onchain_processed_at"] is None

    # Successful market SHOULD
    row_b = conn.execute(
        "SELECT onchain_processed_at FROM corpus_markets WHERE condition_id = ?",
        ("0xMARKET_B",),
    ).fetchone()
    assert row_b["onchain_processed_at"] is not None
