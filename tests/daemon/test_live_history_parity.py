"""Parity test: LiveHistoryProvider vs StreamingHistoryProvider (#78)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus.features import (
    StreamingHistoryProvider,
    compute_features,
)
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db
from pscanner.testing.synthetic_trades import build_metadata, build_synthetic_trades


@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_compute_features_matches_streaming_provider(seed: int) -> None:
    trades = build_synthetic_trades(seed=seed, n=100, n_wallets=8, n_markets=5)
    metadata = build_metadata(trades, with_categories=False)
    streaming = StreamingHistoryProvider(metadata=metadata)
    conn: sqlite3.Connection = init_db(Path(":memory:"))
    try:
        live = LiveHistoryProvider(conn=conn, metadata=metadata)
        for trade in trades:
            streaming_row = compute_features(trade, streaming)
            live_row = compute_features(trade, live)
            assert streaming_row == live_row, (
                f"feature divergence at {trade.tx_hash}: streaming={streaming_row} live={live_row}"
            )
            streaming.observe(trade)
            live.observe(trade)
    finally:
        conn.close()
