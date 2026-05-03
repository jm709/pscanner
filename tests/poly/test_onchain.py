"""Tests for `pscanner.poly.onchain` — OrderFilled log decoder."""

from __future__ import annotations

import pytest

from pscanner.poly.onchain import OrderFilledEvent, decode_order_filled

_ORDER_FILLED_TOPIC0 = (
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
)


def _make_log(
    *,
    order_hash: str,
    maker: str,
    taker: str,
    maker_asset_id: int,
    taker_asset_id: int,
    making: int,
    taking: int,
    fee: int,
    tx_hash: str = "0x" + "ab" * 32,
    block_number: int = 0x1234567,
    log_index: int = 5,
) -> dict[str, object]:
    """Build a synthetic eth_getLogs response entry for an OrderFilled event.

    Mirrors the CTF Exchange's actual log layout: 4 topics
    (signature + indexed orderHash, maker, taker) and 5 data slots
    (makerAssetId, takerAssetId, making, taking, fee).
    """
    parts = [
        maker_asset_id.to_bytes(32, "big"),
        taker_asset_id.to_bytes(32, "big"),
        making.to_bytes(32, "big"),
        taking.to_bytes(32, "big"),
        fee.to_bytes(32, "big"),
    ]
    data = b"".join(parts)
    assert len(data) == 5 * 32
    return {
        "data": "0x" + data.hex(),
        "topics": [
            _ORDER_FILLED_TOPIC0,
            order_hash,
            "0x" + "00" * 12 + maker[2:],
            "0x" + "00" * 12 + taker[2:],
        ],
        "transactionHash": tx_hash,
        "blockNumber": hex(block_number),
        "logIndex": hex(log_index),
    }


def test_decode_order_filled_extracts_all_fields() -> None:
    log = _make_log(
        order_hash="0x" + "cd" * 32,
        maker="0x" + "11" * 20,
        taker="0x" + "22" * 20,
        maker_asset_id=42,
        taker_asset_id=10**40,
        making=1_000_000,
        taking=500_000,
        fee=125,
    )
    event = decode_order_filled(log)
    assert isinstance(event, OrderFilledEvent)
    assert event.order_hash == "0x" + "cd" * 32
    assert event.maker == "0x" + "11" * 20
    assert event.taker == "0x" + "22" * 20
    assert event.maker_asset_id == 42
    assert event.taker_asset_id == 10**40
    assert event.making == 1_000_000
    assert event.taking == 500_000
    assert event.fee == 125
    assert event.tx_hash == "0x" + "ab" * 32
    assert event.block_number == 0x1234567
    assert event.log_index == 5


def test_decode_order_filled_rejects_short_data() -> None:
    log = {
        "topics": [
            _ORDER_FILLED_TOPIC0,
            "0x" + "00" * 32,
            "0x" + "00" * 32,
            "0x" + "00" * 32,
        ],
        "data": "0x" + "00" * 50,  # 50 bytes — well under 5 * 32 = 160
        "transactionHash": "0x" + "00" * 32,
        "blockNumber": "0x1",
        "logIndex": "0x0",
    }
    with pytest.raises(ValueError, match="too short"):
        decode_order_filled(log)


def test_decode_order_filled_rejects_missing_prefix() -> None:
    log = {
        "topics": [
            _ORDER_FILLED_TOPIC0,
            "0x" + "00" * 32,
            "0x" + "00" * 32,
            "0x" + "00" * 32,
        ],
        "data": "ab" * 160,
        "transactionHash": "0x" + "00" * 32,
        "blockNumber": "0x1",
        "logIndex": "0x0",
    }
    with pytest.raises(ValueError, match="0x prefix"):
        decode_order_filled(log)


def test_decode_order_filled_rejects_wrong_topic_count() -> None:
    log = {
        "topics": [_ORDER_FILLED_TOPIC0],  # signature only, missing 3 indexed params
        "data": "0x" + "00" * 160,
        "transactionHash": "0x" + "00" * 32,
        "blockNumber": "0x1",
        "logIndex": "0x0",
    }
    with pytest.raises(ValueError, match="must be a list of 4"):
        decode_order_filled(log)


def test_decode_order_filled_handles_int_block_number() -> None:
    """Some RPC providers return blockNumber as int, others as hex-string."""
    log = _make_log(
        order_hash="0x" + "00" * 32,
        maker="0x" + "00" * 20,
        taker="0x" + "00" * 20,
        maker_asset_id=0,
        taker_asset_id=0,
        making=0,
        taking=0,
        fee=0,
        block_number=0,
        log_index=0,
    )
    log["blockNumber"] = 12345
    log["logIndex"] = 7
    event = decode_order_filled(log)
    assert event.block_number == 12345
    assert event.log_index == 7
