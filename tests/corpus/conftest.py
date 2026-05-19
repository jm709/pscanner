"""Shared pytest fixtures for the corpus test suite.

Mirrors the ``tmp_db`` pattern in ``tests/conftest.py`` but applies
``pscanner.corpus.db.init_corpus_db`` instead.
"""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
)


@pytest.fixture
def tmp_corpus_db() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection with the corpus schema applied."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        yield conn
    finally:
        conn.close()


def _build_synthetic_trades(seed: int, n: int) -> list[Trade]:
    """Generate a deterministic synthetic trade stream covering common shapes."""
    rng = random.Random(seed)  # noqa: S311
    wallets = [f"0xw{i:02d}" for i in range(6)]
    markets = [f"0xm{i:02d}" for i in range(4)]
    base_ts = 1_700_000_000
    out: list[Trade] = []
    for i in range(n):
        wallet = rng.choice(wallets)
        market = rng.choice(markets)
        side = rng.choice(("YES", "NO"))
        bs = rng.choices(("BUY", "SELL"), weights=(0.7, 0.3))[0]
        price = round(rng.uniform(0.05, 0.95), 4)
        size = round(rng.uniform(50.0, 500.0), 2)
        out.append(
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
    return out


def _build_metadata(trades: list[Trade]) -> dict[str, MarketMetadata]:
    """Build a MarketMetadata for every market in the trade stream."""
    by_market: dict[str, MarketMetadata] = {}
    for t in trades:
        if t.condition_id in by_market:
            continue
        by_market[t.condition_id] = MarketMetadata(
            condition_id=t.condition_id,
            category=t.category,
            closed_at=t.ts + 86_400 * 7,
            opened_at=t.ts - 60,
            categories=(t.category,),
        )
    return by_market


@pytest.fixture
def trade_stream() -> list[Trade]:
    """Deterministic 80-trade stream for cross-feature parity tests."""
    return _build_synthetic_trades(seed=42, n=80)


@pytest.fixture
def metadata_for_stream(trade_stream: list[Trade]) -> dict[str, MarketMetadata]:
    """MarketMetadata covering every market in `trade_stream`."""
    return _build_metadata(trade_stream)


@pytest.fixture
def streaming_provider(
    trade_stream: list[Trade],
    metadata_for_stream: dict[str, MarketMetadata],
) -> StreamingHistoryProvider:
    """A StreamingHistoryProvider with the trade stream already observed.

    Caller can call `compute_features(trade, provider)` for any trade
    whose ts is <= the last trade's ts.
    """
    provider = StreamingHistoryProvider(metadata=metadata_for_stream)
    # Pre-register resolutions for half the markets so resolution math is
    # exercised by the fixture.
    for cond_id in list(metadata_for_stream)[: len(metadata_for_stream) // 2]:
        meta = metadata_for_stream[cond_id]
        provider.register_resolution(
            condition_id=cond_id,
            resolved_at=meta.closed_at,
            outcome_yes_won=1,
        )
    # Observe every trade so the provider has non-empty state when
    # callers ask for wallet_state / market_state.
    for trade in trade_stream:
        provider.observe(trade)
    return provider
