"""Convert decoded `OrderFilled` events to `CorpusTrade` rows.

Pure event-to-trade conversion. Consumed today by
`pscanner.corpus.subgraph_ingest`, which decodes GraphQL rows into
`OrderFilledEvent` and calls `event_to_corpus_trade` to derive the
maker-POV BUY/SELL semantics + USDC notional.
"""

from __future__ import annotations

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import CTF_EXCHANGE_ADDRESS, OrderFilledEvent

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
