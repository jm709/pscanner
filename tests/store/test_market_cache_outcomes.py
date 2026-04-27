"""Tests for ``MarketCacheRepo`` outcome/asset persistence and lookup."""

from __future__ import annotations

import sqlite3

from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import CachedMarket, MarketCacheRepo


def _build_market(
    *,
    market_id: str = "mkt-1",
    condition_id: str = "0xcond-1",
    outcomes: list[str] | None = None,
    asset_ids: list[str] | None = None,
    outcome_prices: list[float] | None = None,
) -> CachedMarket:
    """Build a :class:`CachedMarket` with the new outcome fields populated."""
    return CachedMarket(
        market_id=MarketId(market_id),
        event_id=None,
        title="t",
        liquidity_usd=1.0,
        volume_usd=1.0,
        outcome_prices=outcome_prices or [0.6, 0.4],
        outcomes=outcomes or ["Yes", "No"],
        asset_ids=[AssetId(a) for a in (asset_ids or ["asset-yes", "asset-no"])],
        active=True,
        cached_at=1700000000,
        condition_id=ConditionId(condition_id),
        event_slug=None,
    )


def test_market_cache_persists_outcomes_and_asset_ids(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market())
    got = repo.get_by_condition_id(ConditionId("0xcond-1"))
    assert got is not None
    assert got.outcomes == ["Yes", "No"]
    assert got.asset_ids == [AssetId("asset-yes"), AssetId("asset-no")]


def test_outcome_to_asset_exact_match(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market(outcomes=["Yes", "No"], asset_ids=["asset-yes", "asset-no"]))
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "Yes") == AssetId("asset-yes")
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "No") == AssetId("asset-no")


def test_outcome_to_asset_case_and_whitespace_tolerant(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market(outcomes=["Oilers", "Ducks"], asset_ids=["a-oil", "a-duck"]))
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "oilers") == AssetId("a-oil")
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), " DUCKS ") == AssetId("a-duck")


def test_outcome_to_asset_returns_none_when_market_missing(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    assert repo.outcome_to_asset(ConditionId("0xnope"), "Yes") is None


def test_outcome_to_asset_returns_none_when_outcome_missing(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    repo.upsert(_build_market(outcomes=["Yes", "No"], asset_ids=["a-y", "a-n"]))
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "Maybe") is None


def test_outcome_to_asset_returns_none_when_lengths_mismatch(tmp_db: sqlite3.Connection) -> None:
    repo = MarketCacheRepo(tmp_db)
    m = _build_market(outcomes=["Yes", "No"], asset_ids=["only-one"])
    repo.upsert(m)
    assert repo.outcome_to_asset(ConditionId("0xcond-1"), "Yes") is None
