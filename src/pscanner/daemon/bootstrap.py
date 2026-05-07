"""``pscanner daemon bootstrap-features`` — cold-start the live history tables.

Walks ``corpus_trades`` chronologically, folding every BUY/SELL into
``wallet_state_live`` + ``market_state_live`` via :class:`LiveHistoryProvider`.
Resolutions are registered up-front from ``market_resolutions`` so the
buy-then-resolve drain fires correctly during the walk.

After this completes, the daemon can start with O(1) state load.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.features import MarketMetadata, Trade
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db

_LOG = structlog.get_logger(__name__)


def run_bootstrap(
    *,
    corpus_db: Path,
    daemon_db: Path,
    log_every: int = 100_000,
) -> int:
    """Cold-start ``wallet_state_live`` + ``market_state_live`` from corpus.

    Args:
        corpus_db: Path to the corpus SQLite (``data/corpus.sqlite3``).
        daemon_db: Path to the daemon SQLite (``data/pscanner.sqlite3``).
        log_every: Emit a progress log every N trades.

    Returns:
        Total trade count folded.
    """
    corpus_conn = init_corpus_db(corpus_db)
    daemon_conn = init_db(daemon_db)
    try:
        metadata = _load_metadata(corpus_conn)
        provider = LiveHistoryProvider(conn=daemon_conn, metadata=metadata)
        for cond_id, resolved_at, yes_won in corpus_conn.execute(
            "SELECT condition_id, resolved_at, outcome_yes_won FROM market_resolutions"
        ):
            provider.register_resolution(
                condition_id=cond_id,
                resolved_at=int(resolved_at),
                outcome_yes_won=int(yes_won),
            )
        rows = corpus_conn.execute(
            """
            SELECT ct.tx_hash, ct.asset_id, ct.wallet_address, ct.condition_id,
                   ct.outcome_side, ct.bs, ct.price, ct.size, ct.notional_usd, ct.ts
            FROM corpus_trades ct
            ORDER BY ct.ts ASC, ct.tx_hash ASC
            """
        )
        n = 0
        for row in rows:
            trade = Trade(
                tx_hash=row[0],
                asset_id=row[1],
                wallet_address=row[2],
                condition_id=row[3],
                outcome_side=row[4],
                bs=row[5],
                price=float(row[6]),
                size=float(row[7]),
                notional_usd=float(row[8]),
                ts=int(row[9]),
                category="",
            )
            if trade.bs == "BUY":
                provider.observe(trade)
            elif trade.bs == "SELL":
                provider.observe_sell(trade)
            n += 1
            if n % log_every == 0:
                _LOG.info("daemon.bootstrap.progress", trades_folded=n)
        _LOG.info("daemon.bootstrap.done", trades_folded=n)
        return n
    finally:
        daemon_conn.close()
        corpus_conn.close()


def _load_metadata(conn: sqlite3.Connection) -> dict[str, MarketMetadata]:
    out: dict[str, MarketMetadata] = {}
    for cond_id, category, closed_at, opened_at in conn.execute(
        """
        SELECT condition_id,
               COALESCE(category, ''),
               COALESCE(closed_at, 0),
               COALESCE(enumerated_at, 0)
        FROM corpus_markets
        """
    ):
        out[cond_id] = MarketMetadata(
            condition_id=cond_id,
            category=category,
            closed_at=int(closed_at),
            opened_at=int(opened_at),
        )
    return out
