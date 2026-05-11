"""Enumerate closed Polymarket markets above the corpus volume gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import httpx
import structlog

from pscanner.categories import Category, categorize_tags, primary_category
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Event

_log = structlog.get_logger(__name__)

_DEFAULT_VOLUME_GATE_USD: Final[float] = 1_000_000.0
"""Lifetime-volume floor for any category not in :data:`VOLUME_GATE_BY_CATEGORY_USD`.

Defaults remain at $1M (the historical corpus floor) so existing categories
are unaffected. New per-category overrides go in the mapping below.
"""

VOLUME_GATE_BY_CATEGORY_USD: Final[Mapping[Category, float]] = {
    Category.ESPORTS: 100_000.0,
}
"""Per-category lifetime-volume floors.

Esports drops to ``$100K`` to match the live daemon's
``gate_model_market_filter.min_volume_24h_usd = 100_000`` floor — the
previous $1M corpus floor put the live polling target out of distribution
relative to the training set (issue #109).

Categories not listed fall through to :data:`_DEFAULT_VOLUME_GATE_USD`.
"""

_HTTP_SERVER_ERROR_FLOOR: Final[int] = 500
# Polymarket's gamma `/events` uses 422 to signal a deep-offset overflow
# (mirroring the documented 400 cap on `/trades`). Some deployments
# return 500 instead. Both terminate enumeration cleanly with whatever
# pages succeeded.
_DEEP_OFFSET_STATUS: Final[int] = 422


def _volume_gate_for(category: Category) -> float:
    """Return the lifetime-volume floor for ``category``."""
    return VOLUME_GATE_BY_CATEGORY_USD.get(category, _DEFAULT_VOLUME_GATE_USD)


def _qualifying_markets(event: Event, now_ts: int) -> list[CorpusMarket]:
    """Return CorpusMarket rows for every market on ``event`` that qualifies."""
    if not event.closed:
        return []
    tags = list(event.tags)
    primary = primary_category(tags)
    matched = categorize_tags(tags)
    gate = _volume_gate_for(primary)
    tags_json = json.dumps(tags)
    categories_json = json.dumps(sorted(c.value for c in matched))
    out: list[CorpusMarket] = []
    for market in event.markets:
        if not market.closed:
            continue
        volume = market.volume or 0.0
        if volume < gate:
            continue
        if market.condition_id is None:
            continue
        out.append(
            CorpusMarket(
                condition_id=str(market.condition_id),
                event_slug=event.slug,
                category=str(primary),
                closed_at=now_ts,  # placeholder; mark_complete rewrites this to MAX(trade_ts) once backfill finishes  # noqa: E501
                total_volume_usd=volume,
                enumerated_at=now_ts,
                market_slug=market.slug,
                tags_json=tags_json,
                categories_json=categories_json,
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
        if status != _DEEP_OFFSET_STATUS and status < _HTTP_SERVER_ERROR_FLOOR:
            raise
        _log.warning(
            "corpus.enumerate_pagination_capped",
            status=status,
            url=str(exc.request.url),
        )
    _log.info("corpus.enumerated", inserted=inserted)
    return inserted
