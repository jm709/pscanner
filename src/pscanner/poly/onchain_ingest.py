"""Convert decoded `OrderFilled` events to `CorpusTrade` rows.

Pure functions and the block-range paginator. The orchestration loop
that drives state mutations on `corpus_markets` lives in
`pscanner.corpus.onchain_backfill`.
"""

from __future__ import annotations

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import OrderFilledEvent

_LOG = structlog.get_logger(__name__)

# USDC and Polymarket CTF tokens both use 6 decimals on Polygon mainnet.
_DECIMALS = 1_000_000.0


class UnsupportedFill(Exception):  # noqa: N818 — name prescribed by task spec
    """Raised when an OrderFilled event does not represent a CTF↔USDC swap."""


class UnresolvableAsset(Exception):  # noqa: N818 — name prescribed by task spec
    """Raised when neither asset id is known to `AssetIndexRepo`."""


def event_to_corpus_trade(
    event: OrderFilledEvent,
    *,
    asset_repo: AssetIndexRepo,
    ts: int,
) -> CorpusTrade:
    """Convert one `OrderFilledEvent` to a `CorpusTrade` from the taker's POV.

    Args:
        event: Decoded event payload.
        asset_repo: Lookup for `(asset_id → condition_id, outcome_side)`.
        ts: Block timestamp in Unix seconds.

    Returns:
        A `CorpusTrade` row. Caller inserts via `CorpusTradesRepo`.

    Raises:
        UnsupportedFill: Both asset ids zero, or both non-zero (split/merge).
        UnresolvableAsset: The CTF asset id is not in `asset_repo`.
    """
    maker_id, taker_id = event.maker_asset_id, event.taker_asset_id
    maker_is_usdc = maker_id == 0
    taker_is_usdc = taker_id == 0
    if maker_is_usdc == taker_is_usdc:
        raise UnsupportedFill(
            f"both-zero or both-non-zero asset ids: maker={maker_id}, taker={taker_id}"
        )

    # Taker initiated; their side determines BUY/SELL.
    if taker_is_usdc:
        # Taker gave USDC, received CTF token → taker BUY.
        bs = "BUY"
        usdc_amount = event.taking
        ctf_amount = event.making
        ctf_asset_id = maker_id
    else:
        # Taker gave CTF token, received USDC → taker SELL.
        bs = "SELL"
        usdc_amount = event.making
        ctf_amount = event.taking
        ctf_asset_id = taker_id

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
        wallet_address=event.taker.lower(),
        condition_id=entry.condition_id,
        outcome_side=entry.outcome_side,
        bs=bs,
        price=price,
        size=size,
        notional_usd=notional_usd,
        ts=ts,
    )
