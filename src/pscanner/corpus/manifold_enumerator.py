"""Corpus enumerator for closed Manifold Markets.

Walks ``/v0/markets`` paginated via ``before=<id>`` cursor, filters to resolved
binary markets above the volume gate, and inserts ``(platform='manifold')`` rows
into ``corpus_markets``. Idempotent — repeated runs are no-ops on already-known
markets thanks to ``CorpusMarketsRepo.insert_pending``'s ``INSERT OR IGNORE``
semantics.
"""

from __future__ import annotations

import structlog

from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.models import ManifoldMarket

_log = structlog.get_logger(__name__)


async def enumerate_resolved_manifold_markets(
    client: ManifoldClient,
    repo: CorpusMarketsRepo,
    *,
    now_ts: int,
    min_volume_mana: float = 1000.0,
    page_size: int = 1000,
) -> int:
    """Walk Manifold markets and insert resolved+binary+above-volume rows.

    Args:
        client: Open ``ManifoldClient`` with rate-limit budget available.
        repo: Corpus markets repo bound to a platform-aware corpus DB.
        now_ts: Unix seconds, recorded as ``enumerated_at`` on each row.
        min_volume_mana: Minimum ``ManifoldMarket.volume`` to qualify
            (mana, not USD). Defaults to 1000.
        page_size: ``limit`` parameter on ``client.get_markets``.

    Returns:
        Count of newly-inserted ``corpus_markets`` rows. Does not include
        rows that already existed (idempotent re-enumeration).
    """
    inserted_total = 0
    examined_total = 0
    cursor: str | None = None
    while True:
        page = await client.get_markets(limit=page_size, before=cursor)
        if not page:
            break
        examined_total += len(page)
        for market in page:
            if not _qualifies(market, min_volume_mana=min_volume_mana):
                continue
            corpus_market = _to_corpus_market(market, now_ts=now_ts)
            inserted_total += repo.insert_pending(corpus_market)
        cursor = page[-1].id
    _log.info(
        "manifold.enumerate_complete",
        examined=examined_total,
        inserted=inserted_total,
        min_volume_mana=min_volume_mana,
    )
    return inserted_total


def _qualifies(market: ManifoldMarket, *, min_volume_mana: float) -> bool:
    """True iff the market should land in the corpus."""
    return market.is_resolved and market.is_binary and market.volume >= min_volume_mana


def _to_corpus_market(market: ManifoldMarket, *, now_ts: int) -> CorpusMarket:
    """Project a ``ManifoldMarket`` into the corpus dataclass."""
    return CorpusMarket(
        condition_id=market.id,
        event_slug=market.slug or market.id,
        category=market.outcome_type,
        closed_at=market.resolution_time or now_ts,
        total_volume_usd=market.volume,
        enumerated_at=now_ts,
        market_slug=market.slug or market.id,
        platform="manifold",
    )
