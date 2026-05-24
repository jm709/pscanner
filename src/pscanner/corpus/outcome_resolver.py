"""Binary-outcome token-id resolution via gamma slug lookup.

Shared helper for ``market_walker`` (records per-trade outcome_side at
ingest time) and ``outcome_side_backfill`` (repairs historical NO+NO
rows from the pre-#166 parser bug). Both follow the same lookup chain:
``data.get_market_slug_by_condition_id`` → ``gamma.get_market_by_slug`` →
2-element ``clob_token_ids`` mapping.
"""

from __future__ import annotations

import structlog

from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)

_BINARY_MARKET_OUTCOME_COUNT = 2


async def resolve_binary_outcome_map(
    condition_id: str,
    *,
    data: DataClient,
    gamma: GammaClient,
) -> dict[str, tuple[str, int]] | None:
    """Return ``{token_id: (outcome_side, outcome_index)}`` for ``condition_id``.

    Polymarket convention: ``clob_token_ids[0]`` is the YES-equivalent leg,
    ``clob_token_ids[1]`` is the NO-equivalent leg (parallel to ``outcomes``).

    Returns ``None`` when:
    - either client raises
    - the slug lookup returns ``None``
    - the gamma market lookup returns ``None``
    - the market has ``len(clob_token_ids) != 2`` (non-binary)

    Callers narrow the return shape — ``market_walker`` reads just the
    ``outcome_side`` field, ``outcome_side_backfill`` uses both fields.
    """
    try:
        slug = await data.get_market_slug_by_condition_id(condition_id)
    except Exception:
        _log.warning("corpus.binary_outcome_resolver.slug_lookup_failed", condition_id=condition_id)
        return None
    if slug is None:
        return None
    try:
        market = await gamma.get_market_by_slug(slug)
    except Exception:
        _log.warning(
            "corpus.binary_outcome_resolver.gamma_lookup_failed",
            condition_id=condition_id,
            slug=slug,
        )
        return None
    if market is None:
        return None
    if len(market.clob_token_ids) != _BINARY_MARKET_OUTCOME_COUNT:
        _log.info(
            "corpus.binary_outcome_resolver.not_binary",
            condition_id=condition_id,
            n_outcomes=len(market.clob_token_ids),
        )
        return None
    return {
        str(market.clob_token_ids[0]): ("YES", 0),
        str(market.clob_token_ids[1]): ("NO", 1),
    }
