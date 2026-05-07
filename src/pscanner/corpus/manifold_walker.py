"""Per-market bet backfill from Manifold REST into corpus_trades.

Cursor-paginates ``/v0/bets?contractId=<market_id>``, filters out cancelled bets
and unfilled limit orders, projects each fillable bet into ``CorpusTrade``, and
upserts via ``CorpusTradesRepo``. Marks the corpus_markets row in_progress at
start and complete on successful exhaustion.

Manifold's cursor pagination has no offset cap (unlike Polymarket's 3000-offset
limit), so ``truncated`` is always ``False`` when ``mark_complete`` runs.
"""

from __future__ import annotations

import structlog

from pscanner.corpus.repos import (
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.manifold.client import ManifoldClient
from pscanner.manifold.ids import ManifoldMarketId
from pscanner.manifold.models import ManifoldBet

_log = structlog.get_logger(__name__)


async def walk_manifold_market(
    client: ManifoldClient,
    markets_repo: CorpusMarketsRepo,
    trades_repo: CorpusTradesRepo,
    *,
    market_id: ManifoldMarketId,
    now_ts: int,
    page_size: int = 1000,
) -> int:
    """Backfill all fillable bets for one Manifold market into corpus_trades.

    Args:
        client: Open ``ManifoldClient``.
        markets_repo: Corpus markets repo (for state transitions).
        trades_repo: Corpus trades repo (for bet upserts).
        market_id: Manifold market hash ID.
        now_ts: Unix seconds, recorded as ``backfill_started_at`` /
            ``backfill_completed_at`` on the ``corpus_markets`` row.
        page_size: ``limit`` parameter on ``client.get_bets``.

    Returns:
        Count of inserted ``CorpusTrade`` rows (after the manifold-floor filter
        in ``CorpusTradesRepo.insert_batch``).
    """
    markets_repo.mark_in_progress(market_id, started_at=now_ts, platform="manifold")
    inserted_total = 0
    examined_total = 0
    cursor: str | None = None
    while True:
        page = await client.get_bets(market_id=market_id, limit=page_size, before=cursor)
        if not page:
            break
        examined_total += len(page)
        trades = [_to_corpus_trade(bet, market_id=market_id) for bet in page if _is_fillable(bet)]
        if trades:
            inserted_total += trades_repo.insert_batch(trades)
        cursor = page[-1].id
    markets_repo.mark_complete(
        market_id,
        completed_at=now_ts,
        truncated=False,
        platform="manifold",
    )
    _log.info(
        "manifold.walk_complete",
        market_id=market_id,
        examined=examined_total,
        inserted=inserted_total,
    )
    return inserted_total


def _is_fillable(bet: ManifoldBet) -> bool:
    """True iff the bet represents a real fill (not cancelled, not unfilled-limit)."""
    if bet.is_cancelled is True:
        return False
    return not (bet.limit_prob is not None and bet.is_filled is not True)


def _to_corpus_trade(bet: ManifoldBet, *, market_id: ManifoldMarketId) -> CorpusTrade:
    """Project a Manifold bet into the corpus dataclass.

    The synthetic ``asset_id = f"{market_id}:{outcome}"`` names the position;
    Manifold has no separate asset id but ``corpus_trades.asset_id`` is NOT NULL.

    Mana goes into ``notional_usd`` as platform-native units (per the spec's
    convention; downstream readers must group by ``platform`` before any
    USD-aggregating math).
    """
    return CorpusTrade(
        tx_hash=bet.id,
        asset_id=f"{market_id}:{bet.outcome}",
        wallet_address=bet.user_id,
        condition_id=market_id,
        outcome_side=bet.outcome,
        bs="BUY",
        price=bet.prob_before,
        size=bet.amount,
        notional_usd=bet.amount,
        ts=bet.created_time,
        platform="manifold",
    )
