"""Tests for `pscanner.corpus.outcome_resolver.resolve_binary_outcome_map`.

The two callers (``market_walker`` and ``outcome_side_backfill``) already
have integration-style coverage; this module exercises the 6 return-path
branches directly so the helper's contract surfaces in CI even if a future
caller is added.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.corpus.outcome_resolver import resolve_binary_outcome_map


def _make_market(clob_token_ids: list[str]) -> Any:
    market = MagicMock()
    market.clob_token_ids = clob_token_ids
    return market


@pytest.mark.asyncio
async def test_returns_yes_no_map_for_binary_market() -> None:
    data = AsyncMock()
    gamma = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "the-slug"
    gamma.get_market_by_slug.return_value = _make_market(["111", "222"])
    mapping = await resolve_binary_outcome_map("cond1", data=data, gamma=gamma)
    assert mapping == {"111": ("YES", 0), "222": ("NO", 1)}


@pytest.mark.asyncio
async def test_returns_none_on_missing_slug() -> None:
    data = AsyncMock()
    gamma = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = None
    assert await resolve_binary_outcome_map("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_returns_none_on_missing_market() -> None:
    data = AsyncMock()
    gamma = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "the-slug"
    gamma.get_market_by_slug.return_value = None
    assert await resolve_binary_outcome_map("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_returns_none_on_non_binary_market() -> None:
    data = AsyncMock()
    gamma = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "the-slug"
    gamma.get_market_by_slug.return_value = _make_market(["a", "b", "c"])
    assert await resolve_binary_outcome_map("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_returns_none_when_data_raises() -> None:
    data = AsyncMock()
    gamma = AsyncMock()
    data.get_market_slug_by_condition_id.side_effect = RuntimeError("boom")
    assert await resolve_binary_outcome_map("cond1", data=data, gamma=gamma) is None


@pytest.mark.asyncio
async def test_returns_none_when_gamma_raises() -> None:
    data = AsyncMock()
    gamma = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "the-slug"
    gamma.get_market_by_slug.side_effect = RuntimeError("boom")
    assert await resolve_binary_outcome_map("cond1", data=data, gamma=gamma) is None
