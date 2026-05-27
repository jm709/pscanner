"""V1 subgraph adapter for `corpus_trades` backfill (issue #193).

The V1 Polymarket Orderbook subgraph (`7fu2DWYK…`) emits the pre-#151
`OrderFilledEvent` schema: flat `maker`/`taker` hex addresses,
`makerAssetId`/`takerAssetId` (one is `"0"` for USDC), and
`makerAmountFilled`/`takerAmountFilled` in 6-decimal base units —
identical to V2's amount conventions. This module owns the V1-specific
row adapter. The paginator and orchestrator land in later tasks.

The adapter emits the same `OrderFilledEvent` dataclass V2 produces so
the downstream `event_to_corpus_trade` insert path is shared verbatim.

Verified against `tests/corpus/fixtures/v1_v2_overlap.json` (Stage 1).
"""

from __future__ import annotations

from collections.abc import Mapping

from pscanner.poly.onchain import OrderFilledEvent


def _parse_int(key: str, raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} could not be parsed as int: {raw!r}") from exc
    raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")


def _parse_str(key: str, raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be str, got {type(raw).__name__}")
    return raw


def subgraph_v1_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one V1 `orderFilledEvents` row to the existing `OrderFilledEvent`.

    V1 stores `makerAssetId` and `takerAssetId` directly as decimal-string
    CTF token ids (one will be `"0"` for the USDC side). `maker` and
    `taker` are flat hex strings (lowercased on output). Amount fields are
    in 6-decimal base units, matching V2 exactly.

    V1's `side` (string) and `price` (decimal string) fields are present
    but ignored — the downstream `event_to_corpus_trade` derives maker-POV
    BUY/SELL from `(maker_asset_id, taker_asset_id)` alone.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list.

    Returns:
        ``OrderFilledEvent`` (block_number=0, log_index=0).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable, a string field has
            the wrong type, or both/neither asset id is zero.
    """
    maker_asset = _parse_int("makerAssetId", row["makerAssetId"])
    taker_asset = _parse_int("takerAssetId", row["takerAssetId"])
    if (maker_asset == 0) == (taker_asset == 0):
        raise ValueError(
            f"both-zero or both-non-zero asset ids: maker={maker_asset}, taker={taker_asset}"
        )

    return OrderFilledEvent(
        order_hash=_parse_str("orderHash", row["orderHash"]),
        maker=_parse_str("maker", row["maker"]).lower(),
        taker=_parse_str("taker", row["taker"]).lower(),
        maker_asset_id=maker_asset,
        taker_asset_id=taker_asset,
        making=_parse_int("makerAmountFilled", row["makerAmountFilled"]),
        taking=_parse_int("takerAmountFilled", row["takerAmountFilled"]),
        fee=_parse_int("fee", row["fee"]),
        tx_hash=_parse_str("transactionHash", row["transactionHash"]),
        block_number=0,
        log_index=0,
    )
