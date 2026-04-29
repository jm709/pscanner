"""Enumerate closed Polymarket markets above the corpus volume gate."""

from __future__ import annotations

from typing import Final

import structlog

from pscanner.categories import categorize_event
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)
VOLUME_GATE_USD: Final[float] = 10_000.0


async def enumerate_closed_markets(
    *,
    gamma: GammaClient,
    repo: CorpusMarketsRepo,
    now_ts: int,
    since_ts: int | None,
) -> int:
    """Walk gamma closed events; insert qualifying markets as ``pending``.

    Args:
        gamma: Gamma client with ``iter_events``.
        repo: Markets repo to insert into.
        now_ts: Unix seconds at enumeration time (recorded on rows).
        since_ts: Reserved for future use; currently ignored. Kept in the
            signature so refresh and backfill share an interface.

    Returns:
        Count of markets actually inserted (excluding duplicates).
    """
    del since_ts  # not yet used; gamma /events doesn't expose a precise close ts
    inserted = 0
    async for event in gamma.iter_events(active=False, closed=True, page_size=100):
        if not event.closed:
            continue
        category = str(categorize_event(event))
        for market in event.markets:
            if not market.closed:
                continue
            volume = market.volume or 0.0
            if volume < VOLUME_GATE_USD:
                continue
            if market.condition_id is None:
                continue
            corpus = CorpusMarket(
                condition_id=str(market.condition_id),
                event_slug=event.slug,
                category=category,
                closed_at=now_ts,
                total_volume_usd=volume,
                enumerated_at=now_ts,
            )
            inserted += repo.insert_pending(corpus)
    _log.info("corpus.enumerated", inserted=inserted)
    return inserted
