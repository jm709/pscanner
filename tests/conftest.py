"""Shared pytest fixtures for the pscanner test suite.

Wave 2 detector and store tests reuse:

* ``tmp_db`` — in-memory ``sqlite3.Connection`` with the schema applied.
* ``sample_*_json`` — pre-loaded JSON fixtures captured from real Polymarket
  responses (or synthesised for the websocket).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pscanner.store.db import init_db
from pscanner.util.clock import FakeClock

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> Any:
    """Read and parse a JSON fixture from ``tests/fixtures/``."""
    return json.loads((_FIXTURE_DIR / name).read_text())


@pytest.fixture
def tmp_db() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection with pscanner's schema applied."""
    conn = init_db(Path(":memory:"))
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def sample_event_json() -> dict[str, Any]:
    """Realistic gamma ``/events`` response (single event)."""
    return _load_fixture("event.json")


@pytest.fixture
def sample_market_json() -> dict[str, Any]:
    """Realistic gamma ``/markets`` response (single market)."""
    return _load_fixture("market.json")


@pytest.fixture
def sample_position_json() -> list[dict[str, Any]]:
    """Realistic data-api ``/positions`` response."""
    return _load_fixture("positions.json")


@pytest.fixture
def sample_closed_positions_json() -> list[dict[str, Any]]:
    """Realistic data-api ``/closed-positions`` response."""
    return _load_fixture("closed_positions.json")


@pytest.fixture
def sample_activity_json() -> list[dict[str, Any]]:
    """Realistic data-api ``/activity`` response."""
    return _load_fixture("activity.json")


@pytest.fixture
def sample_leaderboard_json() -> list[dict[str, Any]]:
    """Realistic ``lb-api`` ``/profit`` leaderboard response."""
    return _load_fixture("leaderboard.json")


@pytest.fixture
def sample_trade_ws_json() -> dict[str, Any]:
    """Synthetic ``CONFIRMED`` trade message for the CLOB websocket."""
    return _load_fixture("ws_trade.json")


@pytest.fixture
def fake_clock() -> FakeClock:
    """Shared :class:`FakeClock` for tests that drive time-based loops."""
    return FakeClock()
