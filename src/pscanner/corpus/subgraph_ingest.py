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
    "timestamp",  # consumed by iter_market_trades (Task 4), not by the adapter
    "orderHash",
    "maker",
    "taker",
    "makerAssetId",
    "takerAssetId",
    "makerAmountFilled",
    "takerAmountFilled",
    "fee",
)


def _parse_int_field(key: str, raw: object) -> int:
    """Parse a GraphQL BigInt field (string or native int) to Python int."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} could not be parsed as int: {raw!r}") from exc
    raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")


def _parse_str_field(key: str, raw: object) -> str:
    """Validate that a GraphQL field is a plain string."""
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be str, got {type(raw).__name__}")
    return raw


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
        return _parse_int_field(key, row[key])

    def as_str(key: str) -> str:
        return _parse_str_field(key, row[key])

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
