"""Confirms the corpus_markets sentinel column lands via _MIGRATIONS."""

from __future__ import annotations

from pscanner.corpus.db import init_corpus_db


def test_corpus_markets_has_outcome_side_backfilled_at_column(tmp_path) -> None:
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cols = [row[1] for row in conn.execute("PRAGMA table_info(corpus_markets)")]
    assert "outcome_side_backfilled_at" in cols
    conn.close()
