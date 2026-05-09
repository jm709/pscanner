"""Per-market trades backfill from Kalshi REST into corpus_trades.

Cursor-paginates ``/markets/trades?ticker=<ticker>``, projects each fill
into ``CorpusTrade``, and upserts via ``CorpusTradesRepo``. Marks the
corpus_markets row in_progress at start and complete on successful
exhaustion.

Kalshi's cursor pagination has no offset cap (unlike Polymarket's 3000-offset
limit), so ``truncated`` is always ``False`` when ``mark_complete`` runs.

Anonymous identity convention: ``corpus_trades.wallet_address = ""`` for
every Kalshi row. Kalshi public REST trades carry no taker identity. The
L3-enabling social-API path is tracked separately as #95.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.ids import KalshiMarketTicker
from pscanner.kalshi.models import KalshiTrade

_log = structlog.get_logger(__name__)


async def walk_kalshi_market(
    client: KalshiClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    *,
    market_ticker: KalshiMarketTicker,
    now_ts: int,
    page_size: int = 100,
) -> int:
    """Backfill all fills for one Kalshi market into corpus_trades.

    Args:
        client: Open ``KalshiClient``.
        markets_repo: Corpus markets repo (for state transitions).
        trades_repo: Corpus trades repo (for trade upserts).
        market_ticker: Kalshi market ticker.
        now_ts: Unix seconds, recorded as ``backfill_started_at`` /
            ``backfill_completed_at`` on the ``corpus_markets`` row.
        page_size: ``limit`` parameter on ``client.get_market_trades``.

    Returns:
        Count of inserted ``CorpusTrade`` rows (after the platform-aware
        notional floor in ``CorpusTradesRepo.insert_batch``).
    """
    markets_repo.mark_in_progress(market_ticker, started_at=now_ts, platform="kalshi")
    inserted_total = 0
    examined_total = 0
    cursor: str | None = None
    while True:
        page = await client.get_market_trades(ticker=market_ticker, limit=page_size, cursor=cursor)
        if not page.trades:
            break
        examined_total += len(page.trades)
        trades = [_to_corpus_trade(t, market_ticker=market_ticker) for t in page.trades]
        if trades:
            inserted_total += trades_repo.insert_batch(trades)
        if not page.cursor:
            break
        cursor = page.cursor
    markets_repo.mark_complete(
        market_ticker,
        completed_at=now_ts,
        truncated=False,
        platform="kalshi",
    )
    _log.info(
        "kalshi.walk_complete",
        market_ticker=market_ticker,
        examined=examined_total,
        inserted=inserted_total,
    )
    return inserted_total


def _to_corpus_trade(trade: KalshiTrade, *, market_ticker: KalshiMarketTicker) -> CorpusTrade:
    """Project a Kalshi trade into the corpus dataclass.

    Synthetic ``asset_id = f"{ticker}:{taker_side}"`` names the position;
    ``corpus_trades.asset_id`` is NOT NULL and Kalshi has no separate asset
    identifier. The taker price is the dollar price for the taker's side
    (yes_price for yes-takers, no_price for no-takers). ``notional_usd`` is
    real USD: contracts x price/contract.

    ``wallet_address = ""`` is the documented anonymous-trade convention for
    the L1+L2 path; #95 will surface real attribution via the social API.
    """
    price = trade.yes_price_dollars if trade.taker_side == "yes" else trade.no_price_dollars
    return CorpusTrade(
        tx_hash=trade.trade_id,
        asset_id=f"{market_ticker}:{trade.taker_side}",
        wallet_address="",
        condition_id=market_ticker,
        outcome_side=trade.taker_side.upper(),
        bs="BUY",
        price=price,
        size=trade.count_fp,
        notional_usd=trade.count_fp * price,
        ts=_iso_to_epoch(trade.created_time, fallback=0),
        platform="kalshi",
    )


def _iso_to_epoch(iso: str, *, fallback: int) -> int:
    """Parse an ISO 8601 datetime string to epoch seconds.

    Returns ``fallback`` if the input is empty or unparseable. Kalshi wire
    format is ``"2026-05-04T12:00:00Z"``; ``datetime.fromisoformat`` handles
    the trailing ``Z`` since Python 3.11.
    """
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())
