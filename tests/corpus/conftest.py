"""Shared pytest fixtures for the corpus test suite.

Mirrors the ``tmp_db`` pattern in ``tests/conftest.py`` but applies
``pscanner.corpus.db.init_corpus_db`` instead.
"""

from __future__ import annotations

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
from pscanner.testing.synthetic_trades import build_metadata, build_synthetic_trades


@pytest.fixture
def tmp_corpus_db() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection with the corpus schema applied."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def trade_stream() -> list[Trade]:
    """Deterministic 80-trade stream for cross-feature parity tests."""
    return build_synthetic_trades(seed=42, n=80)


@pytest.fixture
def metadata_for_stream(trade_stream: list[Trade]) -> dict[str, MarketMetadata]:
    """MarketMetadata covering every market in `trade_stream`."""
    return build_metadata(trade_stream)


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
