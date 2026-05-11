"""Tests for ``pscanner.corpus.gamma_tags_backfill``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.gamma_tags_backfill import run_backfill_gamma_tags
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.poly.models import Event


def _event(slug: str, tags: list[str]) -> Event:
    return Event.model_validate(
        {
            "id": f"evt-{slug}",
            "slug": slug,
            "title": slug,
            "tags": tags,
            "markets": [],
        }
    )


def _seed_unbackfilled(repo: CorpusMarketsRepo, slug: str, condition_id: str) -> None:
    repo.insert_pending(
        CorpusMarket(
            condition_id=condition_id,
            event_slug=slug,
            category="thesis",
            closed_at=0,
            total_volume_usd=1.0,
            enumerated_at=0,
            market_slug="",
        )
    )


async def test_backfill_writes_tags_categories_and_primary() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        repo = CorpusMarketsRepo(conn)
        _seed_unbackfilled(repo, "fed-decision-jan", "0xfed")

        gamma = AsyncMock()
        gamma.get_event_by_slug.return_value = _event("fed-decision-jan", ["Fed Rates", "Economy"])

        summary = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=None)

        row = conn.execute(
            "SELECT tags_json, categories_json, category FROM corpus_markets "
            "WHERE condition_id = '0xfed'"
        ).fetchone()
        assert row["tags_json"] == '["Fed Rates", "Economy"]'
        assert "macro" in row["categories_json"]
        assert row["category"] == "macro"
        assert summary.markets_processed == 1
        assert summary.markets_quarantined == 0
    finally:
        conn.close()


async def test_backfill_quarantines_dead_slugs() -> None:
    """When ``get_event_by_slug`` returns ``None``, the row is quarantined."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        repo = CorpusMarketsRepo(conn)
        _seed_unbackfilled(repo, "deleted-event", "0xdead")

        gamma = AsyncMock()
        gamma.get_event_by_slug.return_value = None

        summary = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=None)

        row = conn.execute(
            "SELECT tags_json FROM corpus_markets WHERE condition_id = '0xdead'"
        ).fetchone()
        assert row["tags_json"] == "__ERROR__"
        assert summary.markets_processed == 0
        assert summary.markets_quarantined == 1
    finally:
        conn.close()


async def test_backfill_quarantines_on_http_error() -> None:
    """An HTTP error from gamma is caught and the row is quarantined."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        repo = CorpusMarketsRepo(conn)
        _seed_unbackfilled(repo, "broken-event", "0xbork")

        gamma = AsyncMock()
        gamma.get_event_by_slug.side_effect = httpx.HTTPStatusError(
            "gamma 422",
            request=httpx.Request("GET", "https://gamma.example/events"),
            response=httpx.Response(422),
        )

        summary = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=None)

        row = conn.execute(
            "SELECT tags_json FROM corpus_markets WHERE condition_id = '0xbork'"
        ).fetchone()
        assert row["tags_json"] == "__ERROR__"
        assert summary.markets_quarantined == 1
    finally:
        conn.close()


async def test_backfill_quarantines_on_malformed_payload() -> None:
    """A TypeError / ValidationError from gamma is caught and quarantined."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        repo = CorpusMarketsRepo(conn)
        _seed_unbackfilled(repo, "malformed-event", "0xmal")

        gamma = AsyncMock()
        gamma.get_event_by_slug.side_effect = TypeError("expected JSON object, got list")

        summary = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=None)

        row = conn.execute(
            "SELECT tags_json FROM corpus_markets WHERE condition_id = '0xmal'"
        ).fetchone()
        assert row["tags_json"] == "__ERROR__"
        assert summary.markets_quarantined == 1
    finally:
        conn.close()


async def test_backfill_idempotent_skip() -> None:
    """A second run is a no-op when all rows are populated or quarantined."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        repo = CorpusMarketsRepo(conn)
        _seed_unbackfilled(repo, "ev", "0xc1")

        gamma = AsyncMock()
        gamma.get_event_by_slug.return_value = _event("ev", ["Sports"])

        first = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=None)
        second = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=None)

        assert first.markets_processed == 1
        assert second.markets_processed == 0
        # `get_event_by_slug` is called once across both runs.
        assert gamma.get_event_by_slug.call_count == 1
    finally:
        conn.close()


async def test_backfill_honors_limit() -> None:
    conn = init_corpus_db(Path(":memory:"))
    try:
        repo = CorpusMarketsRepo(conn)
        for i in range(5):
            _seed_unbackfilled(repo, f"ev-{i}", f"0x{i}")

        gamma = AsyncMock()
        gamma.get_event_by_slug.side_effect = [_event(f"ev-{i}", ["Sports"]) for i in range(5)]

        summary = await run_backfill_gamma_tags(conn=conn, gamma=gamma, limit=2)
        assert summary.markets_processed == 2
        assert gamma.get_event_by_slug.call_count == 2
    finally:
        conn.close()
