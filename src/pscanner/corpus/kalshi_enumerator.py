"""Corpus enumerator for settled Kalshi markets.

Walks ``/markets?status=<status>&cursor=...`` for each terminal status
(``determined``, ``amended``, ``finalized``), filters to binary markets with
a clean ``yes``/``no`` result above the volume gate, and inserts
``(platform='kalshi')`` rows into ``corpus_markets``. Idempotent — repeated
runs are no-ops on already-known markets thanks to
``CorpusMarketsRepo.insert_pending``'s ``INSERT OR IGNORE`` semantics.

The three-pass status walk is necessary because Kalshi's ``/markets`` filter
takes a single ``status`` value at a time. ``disputed`` is intentionally
skipped — contested resolutions land on a future refresh once Kalshi moves
them to a clean terminal state.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.kalshi.client import KalshiClient
from pscanner.kalshi.models import KalshiMarket

_log = structlog.get_logger(__name__)

_TERMINAL_STATUSES: tuple[str, ...] = ("determined", "amended", "finalized")


async def enumerate_resolved_kalshi_markets(
    client: KalshiClient,
    repo: CorpusMarketsRepo,
    *,
    now_ts: int,
    min_volume_contracts: float = 10_000.0,
    page_size: int = 100,
) -> int:
    """Walk Kalshi markets and insert qualifying rows into corpus_markets.

    Iterates each terminal status (`determined`, `amended`, `finalized`)
    via cursor pagination on ``/markets?status=<value>``. Skips ``disputed``
    (contested resolution) and ``closed`` (trading halted, no determination yet).

    Args:
        client: Open ``KalshiClient`` with rate-limit budget available.
        repo: Corpus markets repo bound to a platform-aware corpus DB.
        now_ts: Unix seconds, recorded as ``enumerated_at`` on each row.
        min_volume_contracts: Minimum ``KalshiMarket.volume_fp`` to qualify.
            Contract count, not USD. Defaults to 10_000.
        page_size: ``limit`` parameter on ``client.get_markets``.

    Returns:
        Count of newly-inserted ``corpus_markets`` rows. Does not include
        rows that already existed (idempotent re-enumeration).
    """
    inserted_total = 0
    examined_total = 0
    for status in _TERMINAL_STATUSES:
        cursor: str | None = None
        while True:
            page = await client.get_markets(status=status, limit=page_size, cursor=cursor)
            if not page.markets:
                break
            examined_total += len(page.markets)
            for market in page.markets:
                if not _qualifies(market, min_volume_contracts=min_volume_contracts):
                    continue
                corpus_market = _to_corpus_market(market, now_ts=now_ts)
                inserted_total += repo.insert_pending(corpus_market)
            if not page.cursor:
                break
            cursor = page.cursor
    _log.info(
        "kalshi.enumerate_complete",
        examined=examined_total,
        inserted=inserted_total,
        min_volume_contracts=min_volume_contracts,
    )
    return inserted_total


def _qualifies(market: KalshiMarket, *, min_volume_contracts: float) -> bool:
    """True iff the market should land in the corpus."""
    return (
        market.market_type == "binary"
        and market.result in ("yes", "no")
        and market.volume_fp >= min_volume_contracts
    )


def _to_corpus_market(market: KalshiMarket, *, now_ts: int) -> CorpusMarket:
    """Project a ``KalshiMarket`` into the corpus dataclass.

    ``closed_at`` is parsed from ``market.close_time`` (ISO datetime) into
    epoch seconds. Falls back to ``now_ts`` if parsing fails. The corpus
    invariant is that ``mark_complete`` rewrites ``closed_at`` to
    ``MAX(corpus_trades.ts)`` after the walker runs, so this initial value is
    a placeholder anyway.
    """
    return CorpusMarket(
        condition_id=market.ticker,
        event_slug=market.event_ticker,
        category=market.market_type,
        closed_at=_iso_to_epoch(market.close_time, fallback=now_ts),
        total_volume_usd=market.volume_fp,
        enumerated_at=now_ts,
        market_slug=market.ticker,
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
