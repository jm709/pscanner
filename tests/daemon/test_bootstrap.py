"""Smoke test: pscanner daemon bootstrap-features (#78)."""

from __future__ import annotations

import json
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


def _seed_mixed_platform_corpus(conn: sqlite3.Connection) -> None:
    """Insert one Polymarket trade + one Manifold trade against the same
    wallet-address-style key (different namespaces in practice, but the test
    proves the platform filter even if a string happened to collide).
    """
    markets = CorpusMarketsRepo(conn)
    markets.insert_pending(
        CorpusMarket(
            condition_id="0xpoly",
            event_slug="poly-evt",
            category="esports",
            closed_at=1_700_001_000,
            total_volume_usd=1000.0,
            enumerated_at=1_699_900_000,
            market_slug="poly-m",
            platform="polymarket",
        )
    )
    markets.insert_pending(
        CorpusMarket(
            condition_id="manifold-cond",
            event_slug="m-evt",
            category="politics",
            closed_at=1_700_001_500,
            total_volume_usd=2000.0,
            enumerated_at=1_699_950_000,
            market_slug="m-m",
            platform="manifold",
        )
    )
    trades = CorpusTradesRepo(conn)
    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="tx-poly",
                asset_id="0xpoly-yes",
                wallet_address="0xabc",
                condition_id="0xpoly",
                outcome_side="YES",
                bs="BUY",
                price=0.40,
                size=100.0,
                notional_usd=40.0,
                ts=1_700_000_000,
                platform="polymarket",
            ),
            CorpusTrade(
                tx_hash="tx-manifold",
                asset_id="manifold-yes",
                wallet_address="manifold-user",
                condition_id="manifold-cond",
                outcome_side="YES",
                bs="BUY",
                price=0.55,
                # 200 mana — above the 100-mana Manifold floor so the row lands.
                size=200.0,
                notional_usd=200.0,
                ts=1_700_000_500,
                platform="manifold",
            ),
        ]
    )
    resolutions = MarketResolutionsRepo(conn)
    resolutions.upsert(
        MarketResolution(
            condition_id="0xpoly",
            winning_outcome_index=0,
            outcome_yes_won=1,
            resolved_at=1_700_001_000,
            source="gamma",
            platform="polymarket",
        ),
        recorded_at=1_700_001_500,
    )
    resolutions.upsert(
        MarketResolution(
            condition_id="manifold-cond",
            winning_outcome_index=0,
            outcome_yes_won=0,
            resolved_at=1_700_001_500,
            source="manifold",
            platform="manifold",
        ),
        recorded_at=1_700_001_700,
    )


def test_run_bootstrap_filters_by_platform_default_polymarket(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.sqlite3"
    daemon_path = tmp_path / "daemon.sqlite3"
    corpus_conn = init_corpus_db(corpus_path)
    try:
        _seed_mixed_platform_corpus(corpus_conn)
    finally:
        corpus_conn.close()
    daemon_conn = init_db(daemon_path)
    daemon_conn.close()
    n = run_bootstrap(corpus_db=corpus_path, daemon_db=daemon_path)
    # Only the Polymarket trade should land; the Manifold one is filtered.
    assert n == 1
    daemon_conn = init_db(daemon_path)
    try:
        rows = daemon_conn.execute(
            "SELECT wallet_address FROM wallet_state_live ORDER BY wallet_address"
        ).fetchall()
        market_rows = daemon_conn.execute(
            "SELECT condition_id FROM market_state_live ORDER BY condition_id"
        ).fetchall()
    finally:
        daemon_conn.close()
    addresses = [r[0] for r in rows]
    cond_ids = [r[0] for r in market_rows]
    assert addresses == ["0xabc"]
    assert "manifold-user" not in addresses
    assert cond_ids == ["0xpoly"]
    assert "manifold-cond" not in cond_ids


def test_run_bootstrap_explicit_manifold_platform(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.sqlite3"
    daemon_path = tmp_path / "daemon.sqlite3"
    corpus_conn = init_corpus_db(corpus_path)
    try:
        _seed_mixed_platform_corpus(corpus_conn)
    finally:
        corpus_conn.close()
    daemon_conn = init_db(daemon_path)
    daemon_conn.close()
    n = run_bootstrap(corpus_db=corpus_path, daemon_db=daemon_path, platform="manifold")
    assert n == 1
    daemon_conn = init_db(daemon_path)
    try:
        addresses = [
            r[0]
            for r in daemon_conn.execute("SELECT wallet_address FROM wallet_state_live").fetchall()
        ]
    finally:
        daemon_conn.close()
    assert addresses == ["manifold-user"]


def test_run_bootstrap_handles_heavy_unresolved_buys(tmp_path: Path) -> None:
    """Pin perf path: a wallet with many unresolved buys must dump cleanly.

    The pre-2026-05-07 implementation called ``LiveHistoryProvider.observe``
    per trade, which re-serialized the wallet's full ``unresolved_buys_json``
    on every fold — O(N^2) per heavy wallet. Bootstrap on a 22M-trade corpus
    projected ~15 hours. The new path walks ``StreamingHistoryProvider`` in
    memory and bulk-writes once at the end. This test pins the bulk-write
    path against a single wallet with 200 unresolved buys (none of the
    markets resolve) to confirm the dump preserves the unresolved list.
    """
    corpus_path = tmp_path / "corpus.sqlite3"
    daemon_path = tmp_path / "daemon.sqlite3"
    corpus_conn = init_corpus_db(corpus_path)
    n_unresolved = 200
    try:
        markets = CorpusMarketsRepo(corpus_conn)
        for i in range(n_unresolved):
            markets.insert_pending(
                CorpusMarket(
                    condition_id=f"0xcond-{i:03d}",
                    event_slug=f"evt-{i}",
                    category="esports",
                    closed_at=1_700_001_000 + i,
                    total_volume_usd=100.0,
                    enumerated_at=1_699_900_000,
                    market_slug=f"m-{i}",
                )
            )
        trades = CorpusTradesRepo(corpus_conn)
        trades.insert_batch(
            [
                CorpusTrade(
                    tx_hash=f"tx-{i:03d}",
                    asset_id=f"0xa-{i}",
                    wallet_address="0xheavy",
                    condition_id=f"0xcond-{i:03d}",
                    outcome_side="YES",
                    bs="BUY",
                    price=0.50,
                    size=20.0,
                    notional_usd=10.0,
                    ts=1_700_000_000 + i,
                )
                for i in range(n_unresolved)
            ]
        )
        # No resolutions registered — every buy stays unresolved.
    finally:
        corpus_conn.close()
    daemon_conn = init_db(daemon_path)
    daemon_conn.close()
    n = run_bootstrap(corpus_db=corpus_path, daemon_db=daemon_path)
    assert n == n_unresolved
    daemon_conn = init_db(daemon_path)
    try:
        row = daemon_conn.execute(
            "SELECT prior_buys_count, prior_resolved_buys, unresolved_buys_json "
            "FROM wallet_state_live WHERE wallet_address = '0xheavy'"
        ).fetchone()
    finally:
        daemon_conn.close()
    assert row is not None
    prior_buys, prior_resolved, unresolved_json = row
    assert prior_buys == n_unresolved
    assert prior_resolved == 0  # no resolutions registered, so nothing drained
    unresolved = json.loads(unresolved_json)
    assert len(unresolved) == n_unresolved
    cond_ids = {entry["condition_id"] for entry in unresolved}
    assert len(cond_ids) == n_unresolved  # every distinct market preserved
