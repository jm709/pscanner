"""Subgraph-driven backfill of `corpus_trades` (Phase 3).

Adapter, paginator, and orchestrator that replace the eth_getLogs path
in ``pscanner.corpus.onchain_targeted``. Reuses the Phase 2 decoder
output type (``OrderFilledEvent``) and ``event_to_corpus_trade`` so the
maker-POV BUY/SELL semantics stay identical.
"""

from __future__ import annotations

from collections.abc import Mapping

from pscanner.poly.onchain import OrderFilledEvent

_REQUIRED_KEYS = (
    "transactionHash",
    "timestamp",
    "orderHash",
    "maker",
    "taker",
    "makerAssetId",
    "takerAssetId",
    "makerAmountFilled",
    "takerAmountFilled",
    "fee",
)


def subgraph_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one GraphQL ``OrderFilledEvent`` row to a Phase 2 dataclass.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list. Must
            carry every key in ``_REQUIRED_KEYS``.

    Returns:
        ``OrderFilledEvent`` with ``block_number=0`` and ``log_index=0``
        (subgraph payloads do not include these; downstream
        ``event_to_corpus_trade`` does not read those fields).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable as int, or a string
            field has the wrong type.
    """
    for key in _REQUIRED_KEYS:
        if key not in row:
            raise KeyError(key)

    def as_int(key: str) -> int:
        raw = row[key]
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            return int(raw)
        raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")

    def as_str(key: str) -> str:
        raw = row[key]
        if not isinstance(raw, str):
            raise ValueError(f"{key} must be str, got {type(raw).__name__}")
        return raw

    return OrderFilledEvent(
        order_hash=as_str("orderHash"),
        maker=as_str("maker"),
        taker=as_str("taker"),
        maker_asset_id=as_int("makerAssetId"),
        taker_asset_id=as_int("takerAssetId"),
        making=as_int("makerAmountFilled"),
        taking=as_int("takerAmountFilled"),
        fee=as_int("fee"),
        tx_hash=as_str("transactionHash"),
        block_number=0,
        log_index=0,
    )
