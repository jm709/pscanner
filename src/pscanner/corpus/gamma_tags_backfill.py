"""Backfill ``corpus_markets.tags_json`` + ``categories_json`` from gamma.

Walks every row where ``tags_json = '[]'`` (i.e. not yet backfilled),
fetches the event's tag list from gamma via ``GammaClient.get_event_by_slug``,
and persists three columns atomically per row:

- ``tags_json`` — raw gamma tag list as JSON-encoded text
- ``categories_json`` — multi-label category set from
  :func:`pscanner.categories.categorize_tags`
- ``category`` — priority-first category from
  :func:`pscanner.categories.primary_category`

A gamma 422 or transport error, or a ``None`` return from
``get_event_by_slug`` (dead slug), quarantines the row with
``tags_json = '__ERROR__'`` so operators can re-run those specifically
for triage. Successful rows are skipped on re-runs via the
``WHERE tags_json = '[]'`` predicate in
:meth:`CorpusMarketsRepo.iter_unbackfilled_tags` (issue #121, Decision B).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import httpx
import structlog

from pscanner.categories import categorize_tags, primary_category
from pscanner.corpus.repos import CorpusMarketsRepo
from pscanner.poly.gamma import GammaClient

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BackfillSummary:
    """Outcome counts for a single ``run_backfill_gamma_tags`` invocation."""

    markets_processed: int
    markets_quarantined: int


async def run_backfill_gamma_tags(
    *,
    conn: sqlite3.Connection,
    gamma: GammaClient,
    limit: int | None,
) -> BackfillSummary:
    """Walk unbackfilled corpus markets and populate their gamma tag data.

    Idempotent: re-running after a successful or quarantined row is a
    no-op for that row (the iterator filters ``tags_json = '[]'``).

    Args:
        conn: Open corpus SQLite connection.
        gamma: Initialised :class:`GammaClient`. Caller owns its lifecycle
            (typically an ``async with`` block).
        limit: Cap the number of rows processed in this run, or ``None``
            to drain the queue.

    Returns:
        A :class:`BackfillSummary` with success and quarantine counts.
    """
    repo = CorpusMarketsRepo(conn)
    processed = 0
    quarantined = 0
    for market in repo.iter_unbackfilled_tags(limit=limit):
        try:
            event = await gamma.get_event_by_slug(market.event_slug)
        except httpx.HTTPError as exc:
            _log.warning(
                "gamma_tags_backfill.fetch_failed",
                condition_id=market.condition_id,
                event_slug=market.event_slug,
                error=str(exc),
            )
            repo.set_gamma_tags_error(condition_id=market.condition_id)
            quarantined += 1
            continue
        if event is None:
            _log.info(
                "gamma_tags_backfill.dead_slug",
                condition_id=market.condition_id,
                event_slug=market.event_slug,
            )
            repo.set_gamma_tags_error(condition_id=market.condition_id)
            quarantined += 1
            continue
        tags = list(event.tags)
        matched = categorize_tags(tags)
        primary = primary_category(tags)
        repo.set_gamma_tags(
            condition_id=market.condition_id,
            tags_json=json.dumps(tags),
            categories_json=json.dumps(sorted(c.value for c in matched)),
            category=str(primary),
        )
        processed += 1
    _log.info(
        "gamma_tags_backfill.summary",
        markets_processed=processed,
        markets_quarantined=quarantined,
    )
    return BackfillSummary(
        markets_processed=processed,
        markets_quarantined=quarantined,
    )
