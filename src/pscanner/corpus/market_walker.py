"""Per-market `/trades` pagination walker.

Pages all trades on one market, normalizes them into ``CorpusTrade``,
inserts via ``CorpusTradesRepo``, and updates the ``corpus_markets``
progress + state columns. Idempotent: re-running on a market that's
already complete is a no-op (the trades unique key bounces duplicates).
"""

from __future__ import annotations

from typing import Any, Final

import structlog

from pscanner.corpus.outcome_resolver import resolve_binary_outcome_map
from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)
_PAGE_SIZE: Final[int] = 500
_OFFSET_CAP: Final[int] = (
    3000  # Polymarket /trades hard cap (server: "max historical activity offset of 3000 exceeded")  # noqa: E501
)


async def _resolve_outcome_side_index(
    condition_id: str,
    *,
    data: DataClient,
    gamma: GammaClient,
) -> dict[str, str]:
    """Build ``{asset_id: "YES" | "NO"}`` from the market's ``clob_token_ids``.

    Thin shim over :func:`pscanner.corpus.outcome_resolver.resolve_binary_outcome_map`
    that narrows the shared `(side, index)` tuple to just the side. Returns
    an empty mapping when the resolver yields ``None`` so the caller can
    fall back to the legacy outcome-name heuristic.
    """
    mapping = await resolve_binary_outcome_map(condition_id, data=data, gamma=gamma)
    if mapping is None:
        return {}
    return {token_id: side for token_id, (side, _index) in mapping.items()}


def _parse_trade(
    item: dict[str, Any],
    condition_id: str,
    *,
    outcome_side_by_asset_id: dict[str, str],
) -> CorpusTrade | None:
    """Best-effort parse of a `/trades` JSON item to ``CorpusTrade``.

    ``outcome_side`` is derived from ``outcome_side_by_asset_id`` (built
    once per ``walk_market`` from the market's ``clob_token_ids``). When
    the trade's ``asset`` is not present in the map (e.g. resolution
    failed or this is a non-binary market), falls back to the legacy
    outcome-name heuristic, which collapses non-``yes`` labels to ``NO``
    (the #159 bug, kept as the fallback because it preserves earlier
    behavior for unresolvable markets).

    Returns ``None`` if required fields are missing or malformed.
    """
    tx = item.get("transactionHash")
    asset = item.get("asset")
    wallet = item.get("proxyWallet")
    side = item.get("side")
    outcome = item.get("outcome")
    price = item.get("price")
    size = item.get("size")
    ts = item.get("timestamp")
    if not isinstance(tx, str) or not isinstance(asset, str):
        return None
    if not isinstance(wallet, str) or not isinstance(side, str):
        return None
    if not isinstance(outcome, str) or not isinstance(ts, int):
        return None
    try:
        price_f = float(price) if price is not None else None
        size_f = float(size) if size is not None else None
    except (TypeError, ValueError):
        return None
    if price_f is None or size_f is None:
        return None
    resolved_side = outcome_side_by_asset_id.get(asset)
    if resolved_side is None:
        resolved_side = "YES" if outcome.lower() == "yes" else "NO"
    return CorpusTrade(
        tx_hash=tx,
        asset_id=asset,
        wallet_address=wallet,
        condition_id=condition_id,
        outcome_side=resolved_side,
        bs="BUY" if side.upper() == "BUY" else "SELL",
        price=price_f,
        size=size_f,
        notional_usd=price_f * size_f,
        ts=ts,
    )


async def walk_market(
    *,
    condition_id: str,
    data: DataClient,
    gamma: GammaClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    now_ts: int,
) -> int:
    """Pull every trade on ``condition_id``; record progress and final state.

    Pre-fetches the parent ``Market`` once via gamma so each trade's
    ``outcome_side`` can be derived from the position of its ``asset`` in
    ``clob_token_ids`` (#159). When the gamma lookup fails, every trade
    falls back to the legacy outcome-name heuristic; the walk still
    proceeds with the older (known-buggy) labelling rather than aborting.

    Args:
        condition_id: Polymarket market identifier.
        data: Data client used for ``/trades`` pagination + slug resolution.
        gamma: Gamma client used to fetch ``clob_token_ids`` once.
        markets_repo: Markets repo to update progress/state on.
        trades_repo: Trades repo for inserts.
        now_ts: Unix seconds for state-machine timestamps.

    Returns:
        Number of trades inserted (post-floor, post-dedupe).
    """
    markets_repo.mark_in_progress(condition_id, started_at=now_ts)
    offset = markets_repo.get_last_offset(condition_id)
    total_inserted = 0
    truncated = False

    outcome_side_by_asset_id = await _resolve_outcome_side_index(
        condition_id, data=data, gamma=gamma
    )

    try:
        total_inserted, truncated = await _fetch_all_pages(
            condition_id=condition_id,
            data=data,
            markets_repo=markets_repo,
            trades_repo=trades_repo,
            start_offset=offset,
            outcome_side_by_asset_id=outcome_side_by_asset_id,
        )
    except Exception as exc:
        markets_repo.mark_failed(condition_id, error_message=str(exc))
        _log.warning("corpus.walk_market_failed", condition_id=condition_id, error=str(exc))
        raise

    markets_repo.mark_complete(condition_id, completed_at=now_ts, truncated=truncated)
    _log.info(
        "corpus.walk_market_complete",
        condition_id=condition_id,
        trades_inserted=total_inserted,
        truncated=truncated,
        outcome_side_resolved=bool(outcome_side_by_asset_id),
    )
    return total_inserted


async def _fetch_all_pages(
    *,
    condition_id: str,
    data: DataClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    start_offset: int,
    outcome_side_by_asset_id: dict[str, str],
) -> tuple[int, bool]:
    """Fetch and store all pages of trades for one market.

    Returns:
        Tuple of (total_trades_inserted, truncated_at_offset_cap).
    """
    offset = start_offset
    total_inserted = 0
    truncated = False

    while True:
        page = await data._fetch_market_trades_page(condition_id, offset=offset)
        if not page:
            truncated = offset >= _OFFSET_CAP
            break
        parsed = [
            t
            for item in page
            if (
                t := _parse_trade(
                    item, condition_id, outcome_side_by_asset_id=outcome_side_by_asset_id
                )
            )
            is not None
        ]
        inserted = trades_repo.insert_batch(parsed)
        total_inserted += inserted
        offset += len(page)
        markets_repo.record_progress(condition_id, last_offset=offset, inserted_delta=inserted)
        if len(page) < _PAGE_SIZE or offset >= _OFFSET_CAP:
            truncated = offset >= _OFFSET_CAP
            break

    return total_inserted, truncated
