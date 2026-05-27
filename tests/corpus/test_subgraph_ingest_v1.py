"""Tests for the V1 subgraph adapter (issue #193)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pscanner.corpus.subgraph_ingest_v1 import subgraph_v1_row_to_event

_FIXTURE = Path(__file__).parent / "fixtures" / "v1_v2_overlap.json"


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def test_parser_handles_v1_buy_rows_from_production_fixture():
    """V1 BUY rows: makerAssetId='0', takerAssetId=<token>, side='buy'.

    The fixture's `v1_buy_rows` are real production rows from the V1
    Polymarket subgraph (see Task 2's `scripts/verify_v1_units.py`).
    The adapter must produce an OrderFilledEvent where maker_asset_id=0
    and taker_asset_id matches the row's takerAssetId.
    """
    data = _load_fixture()
    assert data["cross_subgraph_match"] is False, (
        "fixture must declare cross_subgraph_match=False — V1/V2 index "
        "different Polygon contracts and share no transactions"
    )
    buys = data["v1_buy_rows"]
    assert len(buys) >= 1, "fixture must contain at least one V1 BUY row"
    for row in buys:
        assert row["makerAssetId"] == "0"
        assert row["side"] == "buy"
        event = subgraph_v1_row_to_event(row)
        assert event.maker_asset_id == 0
        assert event.taker_asset_id == int(row["takerAssetId"])
        assert event.making == int(row["makerAmountFilled"])
        assert event.taking == int(row["takerAmountFilled"])
        assert event.maker == row["maker"].lower()
        assert event.taker == row["taker"].lower()
        assert event.tx_hash == row["transactionHash"]
        assert event.order_hash == row["orderHash"]


def test_parser_handles_v1_sell_rows_from_production_fixture():
    """V1 SELL rows: takerAssetId='0', makerAssetId=<token>, side='sell'."""
    data = _load_fixture()
    sells = data["v1_sell_rows"]
    assert len(sells) >= 1, "fixture must contain at least one V1 SELL row"
    for row in sells:
        assert row["takerAssetId"] == "0"
        assert row["side"] == "sell"
        event = subgraph_v1_row_to_event(row)
        assert event.taker_asset_id == 0
        assert event.maker_asset_id == int(row["makerAssetId"])
        assert event.making == int(row["makerAmountFilled"])
        assert event.taking == int(row["takerAmountFilled"])


def test_parser_buy_row_maker_zero():
    row = {
        "id": "tx-1_hash-1",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "1" * 64,
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": "12345",
        "makerAmountFilled": "500000",
        "takerAmountFilled": "1000000",
        "fee": "0",
        "side": "buy",
        "price": "0.5",
    }
    event = subgraph_v1_row_to_event(row)
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 12345
    assert event.making == 500000
    assert event.taking == 1000000
    assert event.maker == "0x" + "b" * 40
    assert event.taker == "0x" + "c" * 40
    assert event.tx_hash == "0x" + "a" * 64
    assert event.order_hash == "0x" + "1" * 64
    assert event.block_number == 0
    assert event.log_index == 0


def test_parser_sell_row_taker_zero():
    row = {
        "id": "tx-2_hash-2",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "2" * 64,
        "maker": "0x" + "B" * 40,  # uppercase to verify normalization
        "taker": "0x" + "c" * 40,
        "makerAssetId": "67890",
        "takerAssetId": "0",
        "makerAmountFilled": "2000000",
        "takerAmountFilled": "600000",
        "fee": "5000",
        "side": "sell",
        "price": "0.3",
    }
    event = subgraph_v1_row_to_event(row)
    assert event.maker_asset_id == 67890
    assert event.taker_asset_id == 0
    assert event.making == 2000000
    assert event.taking == 600000
    assert event.maker == "0x" + "b" * 40  # normalized
    assert event.fee == 5000
    assert event.block_number == 0
    assert event.log_index == 0


def test_parser_rejects_both_zero():
    row = {
        "id": "tx-3_hash-3",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "3" * 64,
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": "0",
        "makerAmountFilled": "0",
        "takerAmountFilled": "0",
        "fee": "0",
        "side": "buy",
        "price": "0",
    }
    with pytest.raises(ValueError, match="both-zero or both-non-zero"):
        subgraph_v1_row_to_event(row)


def test_parser_rejects_both_nonzero():
    row = {
        "id": "tx-4_hash-4",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "4" * 64,
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "12345",
        "takerAssetId": "67890",
        "makerAmountFilled": "100",
        "takerAmountFilled": "100",
        "fee": "0",
        "side": "buy",
        "price": "1",
    }
    with pytest.raises(ValueError, match="both-zero or both-non-zero"):
        subgraph_v1_row_to_event(row)


# ---------------------------------------------------------------------------
# Paginator + iterator tests
# ---------------------------------------------------------------------------


from pscanner.corpus.subgraph_ingest_v1 import iter_v1_market_trades  # noqa: E402


class _FakeSubgraphClient:
    """Records every query() invocation and yields canned responses in order.

    `pages_by_query` is keyed by 'maker' (queries containing
    'makerAssetId_in') or 'taker' (queries containing 'takerAssetId_in').
    """

    def __init__(self, pages_by_query: dict[str, list[list[dict[str, Any]]]]) -> None:
        self._pages = pages_by_query
        self.calls: list[dict[str, Any]] = []

    async def query(self, graphql: str, variables: dict[str, Any]) -> dict[str, Any]:
        side = "maker" if "makerAssetId_in" in graphql else "taker"
        self.calls.append({"side": side, "variables": dict(variables)})
        pages = self._pages.get(side, [])
        if not pages:
            return {"orderFilledEvents": []}
        return {"orderFilledEvents": pages.pop(0)}


def _buy_row(idx: int, asset_id: str) -> dict[str, str]:
    return {
        "id": f"tx-{idx}_hash-{idx}",
        "transactionHash": "0x" + str(idx).rjust(64, "0"),
        "timestamp": str(1_700_000_000 + idx),
        "orderHash": "0x" + str(idx).rjust(64, "1"),
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": asset_id,
        "makerAmountFilled": "500000",
        "takerAmountFilled": "1000000",
        "fee": "0",
        "side": "buy",
        "price": "0.5",
    }


def _sell_row(idx: int, asset_id: str) -> dict[str, str]:
    return {
        "id": f"tx-{idx}_hash-{idx}",
        "transactionHash": "0x" + str(idx).rjust(64, "0"),
        "timestamp": str(1_700_000_000 + idx),
        "orderHash": "0x" + str(idx).rjust(64, "1"),
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": asset_id,
        "takerAssetId": "0",
        "makerAmountFilled": "1000000",
        "takerAmountFilled": "300000",
        "fee": "0",
        "side": "sell",
        "price": "0.3",
    }


@pytest.mark.asyncio
async def test_paginator_returns_empty_on_no_rows():
    client = _FakeSubgraphClient(pages_by_query={})
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=2)
    ]
    assert out == []
    sides = {c["side"] for c in client.calls}
    assert sides == {"maker", "taker"}


@pytest.mark.asyncio
async def test_paginator_yields_union_of_buy_and_sell_passes():
    buys = [_buy_row(i, "100") for i in range(3)]
    sells = [_sell_row(10 + i, "100") for i in range(2)]
    client = _FakeSubgraphClient(
        pages_by_query={
            "maker": [sells],  # maker query catches SELL rows (maker holds CTF)
            "taker": [buys],  # taker query catches BUY rows (taker holds CTF)
        }
    )
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=10)
    ]
    assert len(out) == 5


@pytest.mark.asyncio
async def test_paginator_advances_cursor_per_side():
    buys = [_buy_row(i, "100") for i in range(5)]
    client = _FakeSubgraphClient(
        pages_by_query={
            "maker": [],
            "taker": [buys[0:2], buys[2:4], buys[4:5]],
        }
    )
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=2)
    ]
    assert len(out) == 5
    taker_calls = [c for c in client.calls if c["side"] == "taker"]
    assert len(taker_calls) == 3
    assert taker_calls[0]["variables"]["cursor"] == ""
    assert taker_calls[1]["variables"]["cursor"] == "tx-1_hash-1"
    assert taker_calls[2]["variables"]["cursor"] == "tx-3_hash-3"


@pytest.mark.asyncio
async def test_paginator_rejects_invalid_page_size():
    client = _FakeSubgraphClient(pages_by_query={})
    with pytest.raises(ValueError, match="page_size"):
        async for _ in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=0):
            pass


@pytest.mark.asyncio
async def test_paginator_short_circuits_on_empty_asset_ids():
    client = _FakeSubgraphClient(pages_by_query={})
    out = [x async for x in iter_v1_market_trades(client=client, asset_ids=[], page_size=10)]
    assert out == []
    assert client.calls == []
