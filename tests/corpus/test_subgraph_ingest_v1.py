"""Tests for the V1 subgraph adapter (issue #193)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pscanner.corpus.subgraph_ingest_v1 import subgraph_v1_row_to_event

_FIXTURE = Path(__file__).parent / "fixtures" / "v1_v2_overlap.json"


def _load_fixture() -> dict:  # type: ignore[type-arg]
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
