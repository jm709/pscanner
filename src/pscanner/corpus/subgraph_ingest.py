"""Subgraph-driven backfill of `corpus_trades` (Phase 3).

Adapter, paginator, and orchestrator that replace the eth_getLogs path
in ``pscanner.corpus.onchain_targeted``. Reuses the Phase 2 decoder
output type (``OrderFilledEvent``) and ``event_to_corpus_trade`` so the
maker-POV BUY/SELL semantics stay identical.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any

from pscanner.poly.onchain import OrderFilledEvent
from pscanner.poly.subgraph import SubgraphClient

_REQUIRED_KEYS = (
    "id",  # consumed by _paginate_side (Task 4) cursor logic, not by the adapter
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


# ---------------------------------------------------------------------------
# Paginator
# ---------------------------------------------------------------------------

# The Graph's hard cap on a single page of results.
_MAX_PAGE_SIZE = 1000

# Two separate query constants rather than one template: the filter field
# name (makerAssetId_in vs takerAssetId_in) is a structural part of the
# GraphQL document, not a $variable, so it can't be parameterised. Keeping
# them as separate string literals avoids any string-formatting on GraphQL
# document text — which would be a query-injection foot-gun.
_TRADES_QUERY_MAKER_SIDE = """
query MarketTradesMakerSide($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    orderBy: id
    first: $first
  ) {
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }
}
""".strip()

_TRADES_QUERY_TAKER_SIDE = """
query MarketTradesTakerSide($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $assets, id_gt: $cursor }
    orderBy: id
    first: $first
  ) {
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }
}
""".strip()


async def _paginate_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from one filter side, paginated by id_gt.

    Args:
        client: Open ``SubgraphClient``.
        graphql: One of ``_TRADES_QUERY_MAKER_SIDE`` or ``_TRADES_QUERY_TAKER_SIDE``.
        asset_ids: CTF token ids (as strings) to filter on.
        page_size: Rows per query page (≤ ``_MAX_PAGE_SIZE``).

    Yields:
        ``(event, ts)`` tuples where ``ts`` is the Unix timestamp integer
        from the subgraph ``timestamp`` field.
    """
    cursor = ""
    while True:
        result = await client.query(
            graphql,
            {"assets": list(asset_ids), "cursor": cursor, "first": page_size},
        )
        rows: list[dict[str, Any]] = result.get("orderFilledEvents") or []
        if not rows:
            return
        for row in rows:
            event = subgraph_row_to_event(row)
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            # Short page guarantees no more rows exist for this cursor range.
            return
        cursor = str(rows[-1]["id"])


async def iter_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every ``OrderFilledEvent`` whose maker- or taker-side asset is in ``asset_ids``.

    Runs maker-side queries to exhaustion, then taker-side queries.  Each side
    uses ``id_gt`` cursor pagination so restarts are safe (no duplicates on
    resume, only forward progress).

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as decimal strings) belonging to one condition.
            Pass both YES and NO token ids for a binary market.
        page_size: Rows per query, capped at ``_MAX_PAGE_SIZE`` (1000) by
            The Graph.  Reduce for lower memory pressure during tests.

    Yields:
        ``(event, ts)`` tuples — same shape as ``iter_order_filled_logs``
        from Phase 2 so the orchestrator can mirror its loop body.

    Raises:
        ValueError: ``page_size`` is out of the ``1.._MAX_PAGE_SIZE`` range.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY_MAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY_TAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
