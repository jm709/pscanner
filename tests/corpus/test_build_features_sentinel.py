"""Tests for the build-features in-progress sentinel."""

from __future__ import annotations

import sqlite3

import pytest

from pscanner.corpus._build_features_sentinel import (
    SENTINEL_KEY,
    SentinelAlreadySetError,
    check_and_set_sentinel,
    clear_sentinel,
)
from pscanner.corpus.repos import CorpusStateRepo


def test_check_and_set_writes_sentinel(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    check_and_set_sentinel(repo, now_ts=1_700_000_000, force=False)
    assert repo.get(SENTINEL_KEY) == "1700000000"


def test_check_and_set_refuses_when_present(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set(SENTINEL_KEY, "1699900000", updated_at=1_699_900_000)

    with pytest.raises(SentinelAlreadySetError) as exc:
        check_and_set_sentinel(repo, now_ts=1_700_000_000, force=False)

    assert "1699900000" in str(exc.value)


def test_check_and_set_force_overrides(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set(SENTINEL_KEY, "1699900000", updated_at=1_699_900_000)

    check_and_set_sentinel(repo, now_ts=1_700_000_000, force=True)

    assert repo.get(SENTINEL_KEY) == "1700000000"


def test_clear_sentinel_removes_key(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = CorpusStateRepo(tmp_corpus_db)
    repo.set(SENTINEL_KEY, "1700000000", updated_at=1_700_000_000)

    clear_sentinel(repo)

    assert repo.get(SENTINEL_KEY) is None
