"""Tests for the SubgraphWatchStateRepo."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from pscanner.store.db import init_db
from pscanner.store.repo import SubgraphWatchStateRepo


@pytest.fixture
def conn(tmp_path) -> Iterator[sqlite3.Connection]:
    db = init_db(tmp_path / "pscanner.sqlite3")
    yield db
    db.close()


def test_get_returns_none_when_no_row(conn: sqlite3.Connection) -> None:
    repo = SubgraphWatchStateRepo(conn)
    assert repo.get_last_seen_ts() is None


def test_set_then_get_roundtrip(conn: sqlite3.Connection) -> None:
    repo = SubgraphWatchStateRepo(conn)
    repo.set_last_seen_ts(1_700_000_000)
    assert repo.get_last_seen_ts() == 1_700_000_000


def test_set_overwrites_existing_row(conn: sqlite3.Connection) -> None:
    repo = SubgraphWatchStateRepo(conn)
    repo.set_last_seen_ts(1_700_000_000)
    repo.set_last_seen_ts(1_700_000_500)
    assert repo.get_last_seen_ts() == 1_700_000_500
