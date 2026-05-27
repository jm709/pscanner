"""V1 subgraph adapter for `corpus_trades` backfill (issue #193).

The V1 Polymarket Orderbook subgraph (`7fu2DWYK…`) emits the pre-#151
`OrderFilledEvent` schema: flat `maker`/`taker` hex addresses,
`makerAssetId`/`takerAssetId` (one is `"0"` for USDC), and
`makerAmountFilled`/`takerAmountFilled` in 6-decimal base units —
identical to V2's amount conventions. This module owns the V1-specific
row adapter and paginator. The orchestrator lands in a later task.

The adapter emits the same `OrderFilledEvent` dataclass V2 produces so
the downstream `event_to_corpus_trade` insert path is shared verbatim.

Verified against `tests/corpus/fixtures/v1_v2_overlap.json` (Stage 1).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any

from pscanner.poly.onchain import OrderFilledEvent
from pscanner.poly.subgraph import SubgraphClient


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


# ---------------------------------------------------------------------------
# Paginator + iterator
# ---------------------------------------------------------------------------

_MAX_PAGE_SIZE = 1000

_V1_QUERY_MAKER_SIDE = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee
  }
}
"""

_V1_QUERY_TAKER_SIDE = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee
  }
}
"""


async def _paginate_v1_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from one V1 query (maker or taker side).

    Args:
        client: Open ``SubgraphClient``.
        graphql: GraphQL query string (``_V1_QUERY_MAKER_SIDE`` or
            ``_V1_QUERY_TAKER_SIDE``).
        asset_ids: CTF token ids (as decimal strings) to filter on.
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
            event = subgraph_v1_row_to_event(row)
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            return
        cursor = str(rows[-1]["id"])


async def iter_v1_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every V1 fill involving any of ``asset_ids``.

    V1's schema has no ``_or`` operator, so two separate paginated passes
    are required: one on ``makerAssetId_in`` (catches SELL rows where the
    maker held a CTF token) and one on ``takerAssetId_in`` (catches BUY
    rows where the taker held a CTF token). The two passes are disjoint for
    valid CTF↔USDC fills; the downstream ``CorpusTradesRepo.insert_batch``
    ``INSERT OR IGNORE`` absorbs any accidental overlap.

    Empty ``asset_ids`` short-circuits to an empty iterator (no query).

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as bare decimal strings) belonging to one
            condition. Pass both YES and NO token ids for a binary market.
        page_size: Rows per query, capped at ``_MAX_PAGE_SIZE`` (1000).
            Reduce for lower memory pressure during tests.

    Yields:
        ``(event, ts)`` tuples.

    Raises:
        ValueError: ``page_size`` is out of the ``1.._MAX_PAGE_SIZE`` range.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return
    async for ev, ts in _paginate_v1_side(
        client=client,
        graphql=_V1_QUERY_MAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
    async for ev, ts in _paginate_v1_side(
        client=client,
        graphql=_V1_QUERY_TAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
