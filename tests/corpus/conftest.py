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


@pytest.fixture
def tmp_corpus_db() -> Iterator[sqlite3.Connection]:
    """Yield an in-memory SQLite connection with the corpus schema applied."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        yield conn
    finally:
        conn.close()
