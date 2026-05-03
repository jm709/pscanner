"""On-chain event decoding for Polymarket's CTF Exchange contract.

Phase 1 scope: pure decoding from raw `eth_getLogs` payloads to typed
events. No RPC client lives here — Phase 2 wires `eth_getLogs` calls
and a backfill CLI on top of these primitives.

The CTF Exchange (Polygon mainnet `0x4bFb41d5...`) emits
`OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)`
on every match. All 8 parameters are unindexed, so they land in the
log's `data` field as a flat 256-byte (8 x 32) ABI-encoded sequence.
Addresses are right-aligned in their 32-byte slots; uint256/bytes32
take the full slot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

# Polymarket CTF Exchange on Polygon mainnet. Verified from
# https://github.com/Polymarket/ctf-exchange README.
CTF_EXCHANGE_ADDRESS: Final[str] = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Topic0 is the keccak256 hash of the OrderFilled event signature
# OrderFilled bytes32 address address uint256 uint256 uint256 uint256 uint256.
# Hard-coded to avoid pulling in a Keccak dependency for one constant.
# Phase 2 should add an integration test that compares this against a
# real log's topics[0] when first hitting the live RPC.
ORDER_FILLED_TOPIC0: Final[str] = (
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
)

_DATA_BYTE_LEN: Final[int] = 8 * 32  # 8 fields x 32 bytes
_SLOT: Final[int] = 32
_ADDRESS_BYTES: Final[int] = 20


@dataclass(frozen=True)
class OrderFilledEvent:
    """Decoded `OrderFilled` event with log-position metadata.

    Field semantics (per CTF Exchange `Trading.sol`):
        order_hash: unique hash of the matched order.
        maker: address of the resting (limit) side.
        taker: address of the aggressor side.
        maker_asset_id: ERC1155 token id (or 0 for USDC collateral) the
            maker is giving up.
        taker_asset_id: ERC1155 token id (or 0) the taker is giving up.
        making: amount the maker gives, in the maker asset's smallest unit.
        taking: amount the taker gives, in the taker asset's smallest unit.
        fee: protocol fee in the taker-side asset.
        tx_hash: the containing transaction hash.
        block_number: Polygon block number containing the log.
        log_index: position of this log within the transaction receipt.
    """

    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    making: int
    taking: int
    fee: int
    tx_hash: str
    block_number: int
    log_index: int


def _hex_to_int(value: str | int) -> int:
    """Coerce an eth_getLogs numeric field (often hex-string) to int."""
    if isinstance(value, int):
        return value
    return int(value, 16)


def decode_order_filled(log: Mapping[str, object]) -> OrderFilledEvent:
    """Decode a raw `eth_getLogs` response entry into a typed event.

    Args:
        log: A single entry from an `eth_getLogs` JSON-RPC response. Must
            have ``data`` (hex string with ``0x`` prefix, 256 bytes payload),
            ``transactionHash``, ``blockNumber`` (int or hex string), and
            ``logIndex`` (int or hex string).

    Returns:
        Decoded `OrderFilledEvent`.

    Raises:
        ValueError: If ``data`` is malformed or shorter than 256 bytes.
        KeyError: If a required log field is missing.
    """
    raw = log["data"]
    if not isinstance(raw, str) or not raw.startswith("0x"):
        raise ValueError(f"log.data must be hex string with 0x prefix, got: {raw!r}")
    payload = bytes.fromhex(raw[2:])
    if len(payload) < _DATA_BYTE_LEN:
        raise ValueError(f"log.data too short: {len(payload)} bytes (expected {_DATA_BYTE_LEN})")

    def slot(i: int) -> bytes:
        return payload[i * _SLOT : (i + 1) * _SLOT]

    def slot_address(i: int) -> str:
        return "0x" + slot(i)[_SLOT - _ADDRESS_BYTES :].hex()

    def slot_uint(i: int) -> int:
        return int.from_bytes(slot(i), "big")

    tx_hash = log["transactionHash"]
    if not isinstance(tx_hash, str):
        raise ValueError(f"transactionHash must be str, got: {type(tx_hash).__name__}")

    return OrderFilledEvent(
        order_hash="0x" + slot(0).hex(),
        maker=slot_address(1),
        taker=slot_address(2),
        maker_asset_id=slot_uint(3),
        taker_asset_id=slot_uint(4),
        making=slot_uint(5),
        taking=slot_uint(6),
        fee=slot_uint(7),
        tx_hash=tx_hash,
        block_number=_hex_to_int(log["blockNumber"]),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        log_index=_hex_to_int(log["logIndex"]),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
