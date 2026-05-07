"""Parity test: LiveHistoryProvider vs StreamingHistoryProvider (#78)."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
    compute_features,
)
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


def _build_synthetic_trades(seed: int, n: int) -> list[Trade]:
    rng = random.Random(seed)  # noqa: S311  -- deterministic test seed, not crypto
    wallets = [f"0xw{i:02d}" for i in range(8)]
    markets = [f"0xm{i:02d}" for i in range(5)]
    trades: list[Trade] = []
    base_ts = 1_700_000_000
    for i in range(n):
        wallet = rng.choice(wallets)
        market = rng.choice(markets)
        side = rng.choice(("YES", "NO"))
        bs = rng.choices(("BUY", "SELL"), weights=(0.7, 0.3))[0]
        price = round(rng.uniform(0.05, 0.95), 4)
        size = round(rng.uniform(50.0, 500.0), 2)
        trades.append(
            Trade(
                tx_hash=f"tx{i:04d}",
                asset_id=f"{market}-{side}",
                wallet_address=wallet,
                condition_id=market,
                outcome_side=side,
                bs=bs,
                price=price,
                size=size,
                notional_usd=round(price * size, 4),
                ts=base_ts + i * 60,
                category=rng.choice(("sports", "esports", "crypto")),
            )
        )
    return trades


def _build_metadata(trades: list[Trade]) -> dict[str, MarketMetadata]:
    by_market: dict[str, MarketMetadata] = {}
    for t in trades:
        if t.condition_id in by_market:
            continue
        by_market[t.condition_id] = MarketMetadata(
            condition_id=t.condition_id,
            category=t.category,
            closed_at=t.ts + 86_400 * 7,
            opened_at=t.ts - 60,
        )
    return by_market


@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_compute_features_matches_streaming_provider(seed: int) -> None:
    trades = _build_synthetic_trades(seed=seed, n=100)
    metadata = _build_metadata(trades)
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
