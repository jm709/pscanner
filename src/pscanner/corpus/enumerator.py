"""Enumerate closed Polymarket markets above the corpus volume gate."""

from __future__ import annotations

from typing import Final

import httpx
import structlog

from pscanner.categories import categorize_event
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event

_log = structlog.get_logger(__name__)
VOLUME_GATE_USD: Final[float] = 1_000_000.0
_HTTP_SERVER_ERROR_FLOOR: Final[int] = 500


def _qualifying_markets(event: Event, now_ts: int) -> list[CorpusMarket]:
    """Return CorpusMarket rows for every market on ``event`` that qualifies."""
    if not event.closed:
        return []
    category = str(categorize_event(event))
    out: list[CorpusMarket] = []
    for market in event.markets:
        if not market.closed:
            continue
        volume = market.volume or 0.0
        if volume < VOLUME_GATE_USD:
            continue
        if market.condition_id is None:
            continue
        out.append(
            CorpusMarket(
                condition_id=str(market.condition_id),
                event_slug=event.slug,
                category=category,
                closed_at=now_ts,
                total_volume_usd=volume,
                enumerated_at=now_ts,
            )
        )
    return out


async def enumerate_closed_markets(
    *,
    gamma: GammaClient,
    repo: CorpusMarketsRepo,
    now_ts: int,
    since_ts: int | None,
) -> int:
    """Walk gamma closed events; insert qualifying markets as ``pending``.

    A ``5xx`` from gamma during pagination is treated as the end of the
    catalog and logged at warn-level. Polymarket's ``/events`` endpoint
    returns ``500`` past a deep offset (mirroring the documented
    ``400`` cap on ``/trades``), so this lets enumeration finish cleanly
    on whatever pages did succeed rather than aborting the whole run.

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
    try:
        async for event in gamma.iter_events(active=False, closed=True, page_size=100):
            for corpus in _qualifying_markets(event, now_ts):
                inserted += repo.insert_pending(corpus)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status < _HTTP_SERVER_ERROR_FLOOR:
            raise
        _log.warning(
            "corpus.enumerate_pagination_capped",
            status=status,
            url=str(exc.request.url),
        )
    _log.info("corpus.enumerated", inserted=inserted)
    return inserted
