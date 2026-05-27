"""V1+V2 subgraph-backfill dispatcher (issue #193).

Runs the V2 backfill (existing) then the V1 backfill (new), then the
shared truncation-flag clearance. Both versions share the corpus DB
connection but use independent ``SubgraphClient`` instances pointed at
their respective subgraph deployments.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import structlog

from pscanner.corpus.subgraph_ingest import (
    SubgraphRunSummary,
    # `_clear_truncation_flags` is module-private but shared deliberately —
    # both the V2 orchestrator and this dispatcher need it as the post-run
    # cleanup primitive. Promoting it to the public API would invite ad-hoc
    # callers; keeping it private + cross-imported keeps the call sites
    # explicit and greppable.
    _clear_truncation_flags,
    run_subgraph_backfill,
)
from pscanner.corpus.subgraph_ingest_v1 import V1SubgraphRunSummary, run_v1_subgraph_backfill
from pscanner.poly.subgraph import SubgraphClient

_LOG = structlog.get_logger(__name__)

SubgraphVersion = Literal["v1", "v2"]


@dataclass(frozen=True)
class DispatchedRunSummary:
    """Result of one dispatched backfill run.

    ``v1_summary``/``v2_summary`` are ``None`` when that version was
    excluded by the ``versions`` argument.
    """

    v2_summary: SubgraphRunSummary | None
    v1_summary: V1SubgraphRunSummary | None
    truncation_flags_cleared: int


async def run_subgraph_backfill_dispatched(
    *,
    conn: sqlite3.Connection,
    v1_client: SubgraphClient,
    v2_client: SubgraphClient,
    versions: Sequence[SubgraphVersion] = ("v2", "v1"),
    page_size: int = 1000,
    limit: int | None = None,
    truncation_threshold: int = 3000,
) -> DispatchedRunSummary:
    """Run each requested subgraph version's backfill in order.

    Args:
        conn: Open corpus DB connection.
        v1_client: ``SubgraphClient`` for the V1 endpoint.
        v2_client: ``SubgraphClient`` for the V2 endpoint.
        versions: Ordered list of versions to run. Default
            ``("v2", "v1")`` preserves V2-first ordering.
        page_size: GraphQL ``first:`` per query (max 1000), passed to both.
        limit: Process at most ``N`` markets per version in this run.
        truncation_threshold: Trade-count threshold below which
            ``truncated_at_offset_cap`` stays set. Passed to V2 only;
            ``_clear_truncation_flags`` runs once at the end with the
            same value.

    Returns:
        ``DispatchedRunSummary`` with per-version summaries and the
        final truncation-clearance count.

    Raises:
        ValueError: ``versions`` contains a value other than ``"v1"`` or ``"v2"``.
    """
    unknown = [v for v in versions if v not in ("v1", "v2")]
    if unknown:
        raise ValueError(f"unknown subgraph versions: {unknown}")

    v2_summary: SubgraphRunSummary | None = None
    v1_summary: V1SubgraphRunSummary | None = None

    for version in versions:
        if version == "v2":
            _LOG.info("subgraph.dispatch.v2_start")
            v2_summary = await run_subgraph_backfill(
                conn=conn,
                client=v2_client,
                page_size=page_size,
                limit=limit,
                truncation_threshold=truncation_threshold,
            )
        elif version == "v1":
            _LOG.info("subgraph.dispatch.v1_start")
            v1_summary = await run_v1_subgraph_backfill(
                conn=conn,
                client=v1_client,
                page_size=page_size,
                limit=limit,
            )

    cleared = _clear_truncation_flags(conn, threshold=truncation_threshold)
    summary = DispatchedRunSummary(
        v2_summary=v2_summary,
        v1_summary=v1_summary,
        truncation_flags_cleared=cleared,
    )
    _LOG.info("subgraph.dispatch.done", cleared=cleared)
    return summary
