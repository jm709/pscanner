"""Smoke test: pscanner daemon bootstrap-features (#78)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
    MarketResolution,
    MarketResolutionsRepo,
)
from pscanner.daemon.bootstrap import run_bootstrap
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


def _seed_corpus(conn: sqlite3.Connection) -> None:
    markets = CorpusMarketsRepo(conn)
    markets.insert_pending(
        CorpusMarket(
            condition_id="0xcond",
            event_slug="evt",
            category="esports",
            closed_at=1_700_001_000,
            total_volume_usd=1000.0,
            enumerated_at=1_699_900_000,
            market_slug="m",
        )
    )
    trades = CorpusTradesRepo(conn)
    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="tx1",
                asset_id="0xa",
                wallet_address="0xabc",
                condition_id="0xcond",
                outcome_side="YES",
                bs="BUY",
                price=0.40,
                size=100.0,
                notional_usd=40.0,
                ts=1_700_000_000,
            )
        ]
    )
    resolutions = MarketResolutionsRepo(conn)
    resolutions.upsert(
        MarketResolution(
            condition_id="0xcond",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_001_000,
            source="gamma",
        ),
        recorded_at=1_700_001_500,
    )


def test_run_bootstrap_populates_live_tables(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.sqlite3"
    daemon_path = tmp_path / "daemon.sqlite3"
    corpus_conn = init_corpus_db(corpus_path)
    try:
        _seed_corpus(corpus_conn)
    finally:
        corpus_conn.close()
    daemon_conn = init_db(daemon_path)
    daemon_conn.close()  # close so run_bootstrap opens its own connection
    n = run_bootstrap(corpus_db=corpus_path, daemon_db=daemon_path)
    assert n == 1
    daemon_conn = init_db(daemon_path)
    try:
        provider = LiveHistoryProvider(conn=daemon_conn, metadata={})
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_000_500)
    finally:
        daemon_conn.close()
    assert wallet.prior_trades_count == 1
    assert wallet.prior_buys_count == 1
