"""Unit tests for LiveHistoryProvider (#78)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


def _new_conn() -> sqlite3.Connection:
    return init_db(Path(":memory:"))


def test_wallet_state_returns_empty_for_unknown_wallet() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        state = provider.wallet_state("0xabc", as_of_ts=1_700_000_000)
    finally:
        conn.close()
    assert state.first_seen_ts == 1_700_000_000
    assert state.prior_trades_count == 0
    assert state.prior_buys_count == 0
    assert state.prior_wins == 0
    assert state.recent_30d_trades == ()
    assert state.category_counts == {}
