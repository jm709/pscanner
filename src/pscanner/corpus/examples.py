"""Streaming ``build-features`` orchestrator.

Walks ``corpus_trades`` chronologically, registers known resolutions up
front, drives a ``StreamingHistoryProvider``, and writes one
``TrainingExample`` per qualifying BUY whose market has resolved.

Incremental: ``INSERT OR IGNORE`` on
``(tx_hash, asset_id, wallet_address)`` makes re-runs cheap. The
streaming walk itself is full each time (sub-minute on expected corpus
size) — true watermark-incremental is deferred to v2.
"""

from __future__ import annotations

import sqlite3
from typing import Final

import structlog

from pscanner.corpus.features import (
    FeatureRow,
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
    compute_features,
)
from pscanner.corpus.repos import (
    CorpusTradesRepo,
    MarketResolutionsRepo,
    TrainingExample,
    TrainingExamplesRepo,
)

_log = structlog.get_logger(__name__)
_BATCH_SIZE: Final[int] = 500


def _load_market_metadata(
    conn: sqlite3.Connection, *, platform: str = "polymarket"
) -> dict[str, MarketMetadata]:
    """Load ``MarketMetadata`` for every row in ``corpus_markets`` for ``platform``."""
    rows = conn.execute(
        """
        SELECT condition_id, category, closed_at, enumerated_at
        FROM corpus_markets
        WHERE platform = ?
        """,
        (platform,),
    ).fetchall()
    return {
        row["condition_id"]: MarketMetadata(
            condition_id=row["condition_id"],
            category=row["category"] or "unknown",
            closed_at=row["closed_at"],
            opened_at=row["enumerated_at"],
        )
        for row in rows
    }


def _example_from_features(
    *,
    trade: Trade,
    features: FeatureRow,
    label_won: int,
    now_ts: int,
    platform: str = "polymarket",
) -> TrainingExample:
    return TrainingExample(
        tx_hash=trade.tx_hash,
        asset_id=trade.asset_id,
        wallet_address=trade.wallet_address,
        condition_id=trade.condition_id,
        trade_ts=trade.ts,
        built_at=now_ts,
        platform=platform,
        prior_trades_count=features.prior_trades_count,
        prior_buys_count=features.prior_buys_count,
        prior_resolved_buys=features.prior_resolved_buys,
        prior_wins=features.prior_wins,
        prior_losses=features.prior_losses,
        win_rate=features.win_rate,
        avg_implied_prob_paid=features.avg_implied_prob_paid,
        realized_edge_pp=features.realized_edge_pp,
        prior_realized_pnl_usd=features.prior_realized_pnl_usd,
        avg_bet_size_usd=features.avg_bet_size_usd,
        median_bet_size_usd=features.median_bet_size_usd,
        wallet_age_days=features.wallet_age_days,
        seconds_since_last_trade=features.seconds_since_last_trade,
        prior_trades_30d=features.prior_trades_30d,
        top_category=features.top_category,
        category_diversity=features.category_diversity,
        bet_size_usd=features.bet_size_usd,
        bet_size_rel_to_avg=features.bet_size_rel_to_avg,
        edge_confidence_weighted=features.edge_confidence_weighted,
        win_rate_confidence_weighted=features.win_rate_confidence_weighted,
        is_high_quality_wallet=features.is_high_quality_wallet,
        bet_size_relative_to_history=features.bet_size_relative_to_history,
        side=features.side,
        implied_prob_at_buy=features.implied_prob_at_buy,
        market_category=features.market_category,
        market_volume_so_far_usd=features.market_volume_so_far_usd,
        market_unique_traders_so_far=features.market_unique_traders_so_far,
        market_age_seconds=features.market_age_seconds,
        time_to_resolution_seconds=features.time_to_resolution_seconds,
        last_trade_price=features.last_trade_price,
        price_volatility_recent=features.price_volatility_recent,
        label_won=label_won,
    )


def _maybe_make_example(
    trade: Trade,
    provider: StreamingHistoryProvider,
    now_ts: int,
    platform: str = "polymarket",
) -> TrainingExample | None:
    """Return a TrainingExample for a resolved BUY, or None to skip.

    Caller must guarantee ``trade.bs == "BUY"`` — the loop short-circuits
    SELL rows before allocating ``Trade``, so this function does not
    re-check the side.

    Reads resolution status from the provider's in-memory map rather
    than ``MarketResolutionsRepo`` so the hot path avoids a per-trade
    SQLite SELECT. The provider is seeded once from
    ``market_resolutions`` at startup via ``_register_resolutions``.
    """
    resolution = provider.get_resolution(trade.condition_id)
    if resolution is None:
        return None
    _, outcome_yes_won = resolution
    features = compute_features(trade, provider)
    won = outcome_yes_won == 1 if trade.outcome_side == "YES" else outcome_yes_won == 0
    return _example_from_features(
        trade=trade,
        features=features,
        label_won=1 if won else 0,
        now_ts=now_ts,
        platform=platform,
    )


def _register_resolutions(
    provider: StreamingHistoryProvider,
    markets_conn: sqlite3.Connection,
    *,
    platform: str = "polymarket",
) -> None:
    """Seed the provider with all known market resolutions for ``platform``."""
    rows = markets_conn.execute(
        """
        SELECT condition_id, resolved_at, outcome_yes_won
        FROM market_resolutions
        WHERE platform = ?
        """,
        (platform,),
    ).fetchall()
    for row in rows:
        provider.register_resolution(
            condition_id=row["condition_id"],
            resolved_at=row["resolved_at"],
            outcome_yes_won=row["outcome_yes_won"],
        )


def build_features(
    *,
    trades_repo: CorpusTradesRepo,
    resolutions_repo: MarketResolutionsRepo,
    examples_repo: TrainingExamplesRepo,
    markets_conn: sqlite3.Connection,
    now_ts: int,
    rebuild: bool = False,
    platform: str = "polymarket",
) -> int:
    """Build the training_examples table from corpus_trades + resolutions.

    Args:
        trades_repo: Source of raw trades (chronological). Should be
            constructed against a dedicated read-only connection — see
            issue #110. Sharing the write connection with ``examples_repo``
            serializes WAL checkpoints behind the chunk-boundary read
            transaction; a separate connection lets the cursor stream
            without contention.
        resolutions_repo: Kept for API compat; the per-trade resolution
            check now reads from the provider's in-memory map (seeded by
            ``_register_resolutions`` at startup) so the hot loop avoids a
            per-row SQLite SELECT.
        examples_repo: Sink for materialized rows.
        markets_conn: Connection used to load corpus_markets metadata
            and the one-shot ``market_resolutions`` snapshot.
        now_ts: ``built_at`` for new rows.
        rebuild: If True, drop training_examples before walking.
        platform: Which platform's rows to read and write. Default
            ``"polymarket"`` preserves the pre-multi-platform behavior
            for existing callers. Scopes ``corpus_trades``,
            ``corpus_markets``, ``market_resolutions``, and
            ``training_examples`` reads/writes to that platform.

    Returns:
        Number of rows actually written (deduped via INSERT OR IGNORE).
    """
    del resolutions_repo  # signature kept; reads come from provider/markets_conn
    if rebuild:
        examples_repo.truncate(platform=platform)

    metadata = _load_market_metadata(markets_conn, platform=platform)
    provider = StreamingHistoryProvider(metadata=metadata)
    _register_resolutions(provider, markets_conn, platform=platform)

    written = 0
    pending_examples: list[TrainingExample] = []

    for ct in trades_repo.iter_chronological(platform=platform):
        meta = metadata.get(ct.condition_id)
        if meta is None:
            continue
        # Half the corpus is SELLs; they only need to enter wallet/market
        # state for recency + volume bookkeeping. Skip the Trade rebuild
        # (saves the per-row dataclass allocation + the unused
        # ``category`` lookup) and route through ``observe_sell``.
        if ct.bs != "BUY":
            provider.observe_sell(ct)
            continue
        trade = Trade(
            tx_hash=ct.tx_hash,
            asset_id=ct.asset_id,
            wallet_address=ct.wallet_address,
            condition_id=ct.condition_id,
            outcome_side=ct.outcome_side,
            bs=ct.bs,
            price=ct.price,
            size=ct.size,
            notional_usd=ct.notional_usd,
            ts=ct.ts,
            category=meta.category,
        )
        example = _maybe_make_example(trade, provider, now_ts, platform=platform)
        if example is not None:
            pending_examples.append(example)
        provider.observe(trade)
        if len(pending_examples) >= _BATCH_SIZE:
            written += examples_repo.insert_or_ignore(pending_examples)
            pending_examples.clear()

    if pending_examples:
        written += examples_repo.insert_or_ignore(pending_examples)

    _log.info("corpus.build_features_complete", written=written, rebuild=rebuild)
    return written
