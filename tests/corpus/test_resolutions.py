"""Tests for ``pscanner.corpus.resolutions``."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.repos import (
    MarketResolutionsRepo,
)
from pscanner.corpus.resolutions import (
    determine_outcome_yes_won,
    record_resolutions,
)
from pscanner.poly.models import Market


def _market(condition_id: str, outcome_prices: list[float], closed: bool = True) -> Market:
    return Market.model_validate(
        {
            "id": condition_id,
            "conditionId": condition_id,
            "question": "?",
            "slug": "s",
            "outcomes": ["Yes", "No"],
            "outcomePrices": [str(p) for p in outcome_prices],
            "closed": closed,
            "active": False,
        }
    )


def test_determine_outcome_yes_won_yes() -> None:
    m = _market("c1", [1.0, 0.0])
    assert determine_outcome_yes_won(m) == 1


def test_determine_outcome_yes_won_no() -> None:
    m = _market("c1", [0.0, 1.0])
    assert determine_outcome_yes_won(m) == 0


def test_determine_outcome_disputed_returns_none() -> None:
    m = _market("c1", [0.5, 0.5])
    assert determine_outcome_yes_won(m) is None


def test_determine_outcome_empty_prices_returns_none() -> None:
    m = _market("c1", [])
    assert determine_outcome_yes_won(m) is None


def test_determine_outcome_yes_won_with_legacy_decimal_price() -> None:
    """Old Polymarket markets store the winning price as ~0.9999... rather
    than crisp 1.0. The threshold-based check handles both formats.
    """
    m = _market("c1", [0.9999996501077740101437594120537861, 0.0])
    assert determine_outcome_yes_won(m) == 1


def test_determine_outcome_no_won_with_legacy_decimal_price() -> None:
    m = _market("c1", [0.0000003498922259898562405879462138714832, 0.999999650107774])
    assert determine_outcome_yes_won(m) == 0


@pytest.mark.asyncio
async def test_record_resolutions_writes_resolved_markets(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(
        side_effect=lambda slug: {
            "evt-c1": _market("c1", [1.0, 0.0]),
            "evt-c2": _market("c2", [0.0, 1.0]),
        }[slug]
    )

    await record_resolutions(
        gamma=fake_gamma,
        repo=repo,
        targets=[("c1", "evt-c1", 1_000), ("c2", "evt-c2", 2_000)],
        now_ts=3_000,
    )
    assert repo.get("c1") is not None
    assert repo.get("c2") is not None
    res_c1 = repo.get("c1")
    res_c2 = repo.get("c2")
    assert res_c1 is not None
    assert res_c1.outcome_yes_won == 1
    assert res_c2 is not None
    assert res_c2.outcome_yes_won == 0


@pytest.mark.asyncio
async def test_record_resolutions_skips_disputed(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    repo = MarketResolutionsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(return_value=_market("c1", [0.5, 0.5]))
    await record_resolutions(
        gamma=fake_gamma,
        repo=repo,
        targets=[("c1", "evt-c1", 1_000)],
        now_ts=3_000,
    )
    assert repo.get("c1") is None


@pytest.mark.asyncio
async def test_record_resolutions_records_platform(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """The default ``platform="polymarket"`` is written onto each row."""
    repo = MarketResolutionsRepo(tmp_corpus_db)
    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(return_value=_market("0xpoly", [0.99, 0.01]))

    written = await record_resolutions(
        gamma=fake_gamma,
        repo=repo,
        targets=[("0xpoly", "poly-slug", 1_500)],
        now_ts=1_500,
    )
    assert written == 1
    res = repo.get("0xpoly", platform="polymarket")
    assert res is not None
    assert res.platform == "polymarket"
    assert res.outcome_yes_won == 1
