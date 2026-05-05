"""Convert decoded `OrderFilled` events to `CorpusTrade` rows.

Pure functions and the block-range paginator. The orchestration loop
that drives state mutations on `corpus_markets` lives in
`pscanner.corpus.onchain_backfill`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import (
    CTF_EXCHANGE_ADDRESS,
    ORDER_FILLED_TOPIC0,
    OrderFilledEvent,
    decode_order_filled,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient

_LOG = structlog.get_logger(__name__)

# USDC and Polymarket CTF tokens both use 6 decimals on Polygon mainnet.
_DECIMALS = 1_000_000.0


class UnsupportedFill(Exception):  # noqa: N818 — name prescribed by task spec
    """Raised when an OrderFilled event does not represent a CTF↔USDC swap."""


class UnresolvableAsset(Exception):  # noqa: N818 — name prescribed by task spec
    """Raised when neither asset id is known to `AssetIndexRepo`."""


_EXCHANGE_ADDRESS_LOWER = CTF_EXCHANGE_ADDRESS.lower()


def event_to_corpus_trade(
    event: OrderFilledEvent,
    *,
    asset_repo: AssetIndexRepo,
    ts: int,
) -> CorpusTrade:
    """Convert one `OrderFilledEvent` to a `CorpusTrade` from the maker's POV.

    On Polymarket's CTF Exchange, every user-facing trade is recorded with
    the user as ``maker`` (their resting order is the one that was filled);
    the ``taker`` field carries either a counterparty user address or the
    exchange contract itself when the fill is settled via a merge/split
    of YES/NO complementary tokens. We therefore record the trade from
    the maker's perspective and skip events whose maker is the exchange
    contract (those are bookkeeping side-effects of merges, not real
    user trades).

    Args:
        event: Decoded event payload.
        asset_repo: Lookup for `(asset_id → condition_id, outcome_side)`.
        ts: Block timestamp in Unix seconds.

    Returns:
        A `CorpusTrade` row. Caller inserts via `CorpusTradesRepo`.

    Raises:
        UnsupportedFill: Both asset ids are zero, both are non-zero
            (split/merge between two CTF tokens), or the maker is the
            exchange contract itself.
        UnresolvableAsset: The CTF asset id is not in `asset_repo`.
    """
    maker_lower = event.maker.lower()
    if maker_lower == _EXCHANGE_ADDRESS_LOWER:
        raise UnsupportedFill(f"maker is exchange contract in fill {event.tx_hash}")

    maker_id, taker_id = event.maker_asset_id, event.taker_asset_id
    maker_gives_usdc = maker_id == 0
    taker_gives_usdc = taker_id == 0
    if maker_gives_usdc == taker_gives_usdc:
        raise UnsupportedFill(
            f"both-zero or both-non-zero asset ids: maker={maker_id}, taker={taker_id}"
        )

    # The user is the maker; their side determines BUY/SELL.
    if maker_gives_usdc:
        # Maker gave USDC, received CTF token → maker BUY.
        bs = "BUY"
        usdc_amount = event.making
        ctf_amount = event.taking
        ctf_asset_id = taker_id
    else:
        # Maker gave CTF token, received USDC → maker SELL.
        bs = "SELL"
        usdc_amount = event.taking
        ctf_amount = event.making
        ctf_asset_id = maker_id

    asset_id_str = str(ctf_asset_id)
    entry = asset_repo.get(asset_id_str)
    if entry is None:
        raise UnresolvableAsset(asset_id_str)

    if ctf_amount == 0:
        # Defensive: zero-size fill (shouldn't reach the contract, but guard).
        raise UnsupportedFill(f"zero ctf amount in fill {event.tx_hash}")

    price = usdc_amount / ctf_amount
    size = ctf_amount / _DECIMALS
    notional_usd = usdc_amount / _DECIMALS

    return CorpusTrade(
        tx_hash=event.tx_hash,
        asset_id=asset_id_str,
        wallet_address=maker_lower,
        condition_id=entry.condition_id,
        outcome_side=entry.outcome_side,
        bs=bs,
        price=price,
        size=size,
        notional_usd=notional_usd,
        ts=ts,
    )


@dataclass(frozen=True)
class IngestRunSummary:
    """Per-run accumulator returned by the orchestrator."""

    chunks_processed: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    last_block: int


async def iter_order_filled_logs(
    *,
    rpc: OnchainRpcClient,
    from_block: int,
    to_block: int,
    chunk_size: int = 5_000,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield decoded `OrderFilled` events from `from_block`..`to_block` inclusive.

    Each yielded tuple is `(event, block_timestamp)`. Walks in fixed-size
    chunks; pre-fetches block timestamps via the RPC client's cache.

    Args:
        rpc: Async RPC client (caller owns lifecycle).
        from_block: Inclusive start block.
        to_block: Inclusive end block.
        chunk_size: Blocks per `eth_getLogs` call (default 5,000).

    Yields:
        `(OrderFilledEvent, unix_timestamp_seconds)` in `(blockNumber, logIndex)`
        order within each chunk.
    """
    if from_block > to_block:
        return
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    cursor = from_block
    while cursor <= to_block:
        chunk_end = min(cursor + chunk_size - 1, to_block)
        logs = await rpc.get_logs(
            address=CTF_EXCHANGE_ADDRESS,
            topics=[ORDER_FILLED_TOPIC0],
            from_block=cursor,
            to_block=chunk_end,
        )
        for log in logs:
            event = decode_order_filled(log)
            ts = await rpc.get_block_timestamp(event.block_number)
            yield event, ts
        cursor = chunk_end + 1
