"""Per-market `/trades` pagination walker.

Pages all trades on one market, normalizes them into ``CorpusTrade``,
inserts via ``CorpusTradesRepo``, and updates the ``corpus_markets``
progress + state columns. Idempotent: re-running on a market that's
already complete is a no-op (the trades unique key bounces duplicates).
"""

from __future__ import annotations

from typing import Any, Final

import structlog

from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.poly.data import DataClient

_log = structlog.get_logger(__name__)
_PAGE_SIZE: Final[int] = 500
_OFFSET_CAP: Final[int] = (
    3000  # Polymarket /trades hard cap (server: "max historical activity offset of 3000 exceeded")  # noqa: E501
)


def _parse_trade(item: dict[str, Any], condition_id: str) -> CorpusTrade | None:
    """Best-effort parse of a `/trades` JSON item to ``CorpusTrade``.

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
    return CorpusTrade(
        tx_hash=tx,
        asset_id=asset,
        wallet_address=wallet,
        condition_id=condition_id,
        outcome_side="YES" if outcome.lower() == "yes" else "NO",
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
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    now_ts: int,
) -> int:
    """Pull every trade on ``condition_id``; record progress and final state.

    Args:
        condition_id: Polymarket market identifier.
        data: Data client used for ``/trades`` pagination.
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

    try:
        total_inserted, truncated = await _fetch_all_pages(
            condition_id=condition_id,
            data=data,
            markets_repo=markets_repo,
            trades_repo=trades_repo,
            start_offset=offset,
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
    )
    return total_inserted


async def _fetch_all_pages(
    *,
    condition_id: str,
    data: DataClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    start_offset: int,
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
        parsed = [t for item in page if (t := _parse_trade(item, condition_id)) is not None]
        inserted = trades_repo.insert_batch(parsed)
        total_inserted += inserted
        offset += len(page)
        markets_repo.record_progress(condition_id, last_offset=offset, inserted_delta=inserted)
        if len(page) < _PAGE_SIZE or offset >= _OFFSET_CAP:
            truncated = offset >= _OFFSET_CAP
            break

    return total_inserted, truncated
