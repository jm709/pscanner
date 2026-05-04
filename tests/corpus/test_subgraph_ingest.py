from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.subgraph_ingest import iter_market_trades, subgraph_row_to_event


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
