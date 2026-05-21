"""Just-in-time resolver: tokenId → ResolvedToken via local tables + gamma fallback.

Local-first: try ``AssetIndexRepo.get(tokenId)`` + ``MarketCacheRepo.get_by_condition_id``.
On miss, call gamma ``/markets?clob_token_ids=<tokenId>`` exactly once, upsert both
tables, then resolve from the just-written rows.

Used by:
  - ``scripts/watch_subgraph_copy.py`` (the subgraph copy-trader)
  - ``pscanner.collectors.subgraph_trades.SubgraphTradeCollector`` (issue #152, pending)

Out of scope: negative caching for tokens gamma can't resolve. If the daemon sees
the same unresolvable token on every poll, the log volume signals a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import AssetEntry, AssetIndexRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.models import Market
from pscanner.store.repo import MarketCacheRepo

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ResolvedToken:
    """Result of resolving a tokenId to its market + outcome position.

    ``condition_id`` is the on-chain conditional-token id; ``outcome_index`` is
    the position in the market's parallel ``outcomes`` / ``clob_token_ids``
    arrays (0 = YES-equivalent, 1 = NO-equivalent per Polymarket convention).
    """

    condition_id: ConditionId
    asset_id: AssetId
    outcome_name: str
    outcome_index: int


async def resolve_token(
    *,
    token_id: AssetId,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    gamma: GammaClient,
) -> ResolvedToken | None:
    """Resolve ``token_id`` to a ``ResolvedToken``, falling back to gamma on local miss.

    Lookup order:
        1. ``AssetIndexRepo.get(token_id)`` for ``(condition_id, outcome_side)``.
        2. ``MarketCacheRepo.get_by_condition_id`` for the cached ``Market``.
        3. If either misses, call gamma ``/markets?clob_token_ids=<token_id>``,
           upsert both tables, then re-resolve from the freshly-written rows.

    Args:
        token_id: Subgraph ``tokenId`` (== Polymarket CTF position id, decimal string).
        asset_index: Corpus-side asset_id → condition_id index.
        market_cache: Daemon-side market metadata cache.
        gamma: Gamma client for the on-miss fallback.

    Returns:
        ``ResolvedToken`` on success. ``None`` if gamma returns 0 markets for the
        token id, or if gamma returns a market whose ``clob_token_ids`` don't
        include the queried token (defensive — would indicate gamma indexing drift).
    """
    cached_resolution = _try_local(token_id, asset_index, market_cache)
    if cached_resolution is not None:
        return cached_resolution

    market = await _fetch_market_from_gamma(token_id, gamma)
    if market is None:
        return None

    return _upsert_and_resolve(token_id, market, asset_index, market_cache)


def _try_local(
    token_id: AssetId,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
) -> ResolvedToken | None:
    """Local lookup. Returns None on any miss (asset_index OR market_cache OR position check)."""
    entry = asset_index.get(token_id)
    if entry is None:
        return None
    condition_id = ConditionId(entry.condition_id)
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None:
        return None
    try:
        idx = cached.asset_ids.index(AssetId(token_id))
    except ValueError:
        return None
    return ResolvedToken(
        condition_id=condition_id,
        asset_id=AssetId(token_id),
        outcome_name=cached.outcomes[idx],
        outcome_index=idx,
    )


async def _fetch_market_from_gamma(token_id: AssetId, gamma: GammaClient) -> Market | None:
    """One-shot gamma fetch by ``clob_token_ids``. Returns the first match or None."""
    matches = await gamma.list_markets(clob_token_ids=token_id, limit=5)
    if not matches:
        return None
    # Take the first match. Gamma returns at most a handful of markets per
    # clob_token_ids query; in practice it's always 1.
    return matches[0]


def _upsert_and_resolve(
    token_id: AssetId,
    market: Market,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
) -> ResolvedToken | None:
    """Write the gamma Market into both tables, then return the resolved tuple.

    Returns None (without writing anything) if the queried ``token_id`` is not
    present in ``market.clob_token_ids`` — defensive against gamma indexing
    drift, where a clob_token_ids filter returns a market that doesn't actually
    contain the queried token. We don't trust the response in that case.
    """
    if market.condition_id is None:
        _LOG.warning("token_resolver.gamma_market_missing_condition_id", token_id=token_id)
        return None
    # Containment check first: if the queried token isn't actually in this
    # market's clob_token_ids, the gamma response is inconsistent. Don't
    # persist anything from it.
    asset_id_strs = [str(a) for a in market.clob_token_ids]
    try:
        target_idx = asset_id_strs.index(str(token_id))
    except ValueError:
        _LOG.warning(
            "token_resolver.gamma_market_missing_token",
            token_id=token_id,
            condition_id=str(market.condition_id),
            clob_token_ids=asset_id_strs,
        )
        return None
    # Containment confirmed; write both tables.
    market_cache.upsert(market)
    for idx, asset_id in enumerate(market.clob_token_ids):
        side = "YES" if idx == 0 else "NO"
        asset_index.upsert(
            AssetEntry(
                asset_id=str(asset_id),
                condition_id=str(market.condition_id),
                outcome_side=side,
                outcome_index=idx,
            )
        )
    return ResolvedToken(
        condition_id=ConditionId(str(market.condition_id)),
        asset_id=AssetId(str(token_id)),
        outcome_name=market.outcomes[target_idx],
        outcome_index=target_idx,
    )
