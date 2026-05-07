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
    record_manifold_resolutions,
    record_resolutions,
)
from pscanner.manifold.models import ManifoldMarket
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


class _FakeManifoldClient:
    """Tiny stub that returns a fixed market by id."""

    def __init__(self, markets: dict[str, ManifoldMarket]) -> None:
        self._markets = markets

    async def get_market(self, market_id: str) -> ManifoldMarket:
        return self._markets[market_id]


def _resolved_manifold_market(*, market_id: str, resolution: str | None) -> ManifoldMarket:
    return ManifoldMarket.model_validate(
        {
            "id": market_id,
            "creatorId": "creator",
            "question": f"Question for {market_id}?",
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "isResolved": True,
            "resolutionTime": 1_700_000_000,
            "resolution": resolution,
        }
    )


@pytest.mark.asyncio
async def test_record_manifold_resolutions_writes_yes_no(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """YES and NO resolutions land in market_resolutions with platform='manifold'."""
    repo = MarketResolutionsRepo(tmp_corpus_db)
    client = _FakeManifoldClient(
        {
            "yes-market": _resolved_manifold_market(market_id="yes-market", resolution="YES"),
            "no-market": _resolved_manifold_market(market_id="no-market", resolution="NO"),
        }
    )
    written = await record_manifold_resolutions(
        client=client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo=repo,
        targets=[("yes-market", 1_700_000_000), ("no-market", 1_700_000_001)],
        now_ts=2_000_000_000,
    )
    assert written == 2
    yes_row = repo.get("yes-market", platform="manifold")
    no_row = repo.get("no-market", platform="manifold")
    assert yes_row is not None
    assert yes_row.outcome_yes_won == 1
    assert no_row is not None
    assert no_row.outcome_yes_won == 0
    assert yes_row.platform == "manifold"
    assert yes_row.source == "manifold-rest"


@pytest.mark.asyncio
async def test_record_manifold_resolutions_skips_mkt_and_cancel(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """MKT and CANCEL resolutions are logged + skipped — no market_resolutions row."""
    repo = MarketResolutionsRepo(tmp_corpus_db)
    client = _FakeManifoldClient(
        {
            "mkt-market": _resolved_manifold_market(market_id="mkt-market", resolution="MKT"),
            "cancel-market": _resolved_manifold_market(
                market_id="cancel-market", resolution="CANCEL"
            ),
            "null-market": _resolved_manifold_market(market_id="null-market", resolution=None),
        }
    )
    written = await record_manifold_resolutions(
        client=client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        repo=repo,
        targets=[
            ("mkt-market", 1_700_000_000),
            ("cancel-market", 1_700_000_001),
            ("null-market", 1_700_000_002),
        ],
        now_ts=2_000_000_000,
    )
    assert written == 0
    assert repo.get("mkt-market", platform="manifold") is None
    assert repo.get("cancel-market", platform="manifold") is None
    assert repo.get("null-market", platform="manifold") is None
