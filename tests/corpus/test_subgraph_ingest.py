from __future__ import annotations

import pytest

from pscanner.corpus.subgraph_ingest import subgraph_row_to_event


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
    with pytest.raises(ValueError, match="invalid literal"):
        subgraph_row_to_event(row)
