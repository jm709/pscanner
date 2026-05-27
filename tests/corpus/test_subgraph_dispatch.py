"""Tests for the V1+V2 subgraph dispatcher (issue #193)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.subgraph_dispatch import (
    DispatchedRunSummary,
    run_subgraph_backfill_dispatched,
)


class _NoopClient:
    async def query(self, graphql: str, variables: dict[str, Any]) -> dict[str, Any]:
        return {"orderFilledEvents": []}


@pytest.mark.asyncio
async def test_dispatcher_runs_both_versions_by_default(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        summary = await run_subgraph_backfill_dispatched(
            conn=conn,
            v1_client=_NoopClient(),
            v2_client=_NoopClient(),
            versions=("v2", "v1"),
        )
        assert isinstance(summary, DispatchedRunSummary)
        assert summary.v2_summary is not None
        assert summary.v1_summary is not None
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_dispatcher_skips_v1_when_only_v2_requested(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        summary = await run_subgraph_backfill_dispatched(
            conn=conn,
            v1_client=_NoopClient(),
            v2_client=_NoopClient(),
            versions=("v2",),
        )
        assert summary.v2_summary is not None
        assert summary.v1_summary is None
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_dispatcher_skips_v2_when_only_v1_requested(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        summary = await run_subgraph_backfill_dispatched(
            conn=conn,
            v1_client=_NoopClient(),
            v2_client=_NoopClient(),
            versions=("v1",),
        )
        assert summary.v2_summary is None
        assert summary.v1_summary is not None
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_dispatcher_rejects_unknown_version(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        with pytest.raises(ValueError, match="unknown subgraph versions"):
            await run_subgraph_backfill_dispatched(
                conn=conn,
                v1_client=_NoopClient(),
                v2_client=_NoopClient(),
                versions=("v3",),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            )
    finally:
        conn.close()
