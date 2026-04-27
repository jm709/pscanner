"""Tests for PaperResolver."""

from __future__ import annotations

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    PaperTradesRepo,
)
from pscanner.strategies.paper_resolver import (
    PaperResolver,
    _check_resolution,
    _compute_payout,
)
from pscanner.util.clock import FakeClock

_NOW = 1700000000


def _cache_market(
    repo: MarketCacheRepo,
    *,
    condition_id: str = "0xcond-1",
    active: bool = True,
    outcome_prices: list[float] | None = None,
    outcomes: list[str] | None = None,
    asset_ids: list[str] | None = None,
) -> None:
    repo.upsert(
        CachedMarket(
            market_id=MarketId(f"mkt-{condition_id}"),
            event_id=None,
            title="Test market",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=outcome_prices or [0.5, 0.5],
            outcomes=outcomes or ["Yes", "No"],
            asset_ids=[AssetId(a) for a in (asset_ids or ["asset-yes", "asset-no"])],
            active=active,
            cached_at=_NOW,
            condition_id=ConditionId(condition_id),
            event_slug=None,
        ),
    )


def _open_position(
    repo: PaperTradesRepo,
    *,
    condition_id: str = "0xcond-1",
    asset_id: str = "asset-yes",
    outcome: str = "yes",
    cost_usd: float = 10.0,
    shares: float = 20.0,
    fill_price: float = 0.5,
) -> int:
    return repo.insert_entry(
        triggering_alert_key=f"k-{condition_id}-{outcome}",
        source_wallet="0xw1",
        condition_id=ConditionId(condition_id),
        asset_id=AssetId(asset_id),
        outcome=outcome,
        shares=shares,
        fill_price=fill_price,
        cost_usd=cost_usd,
        nav_after_usd=1000.0 - cost_usd,
        ts=_NOW,
    )


def test_check_resolution_active_market_returns_none(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    assert _check_resolution(cache, ConditionId("0xcond-1")) is None


def test_check_resolution_yes_won(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    res = _check_resolution(cache, ConditionId("0xcond-1"))
    assert res == AssetId("asset-yes")


def test_check_resolution_no_won(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[0.0, 1.0])
    res = _check_resolution(cache, ConditionId("0xcond-1"))
    assert res == AssetId("asset-no")


def test_check_resolution_ambiguous_outcomes_returns_none(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[0.5, 0.5])
    assert _check_resolution(cache, ConditionId("0xcond-1")) is None


def test_check_resolution_market_missing_returns_none(tmp_db) -> None:
    cache = MarketCacheRepo(tmp_db)
    assert _check_resolution(cache, ConditionId("0xnope")) is None


def test_compute_payout_winner() -> None:
    assert (
        _compute_payout(
            position_asset_id=AssetId("asset-yes"),
            winning_asset_id=AssetId("asset-yes"),
        )
        == 1.0
    )


def test_compute_payout_loser() -> None:
    assert (
        _compute_payout(
            position_asset_id=AssetId("asset-yes"),
            winning_asset_id=AssetId("asset-no"),
        )
        == 0.0
    )


@pytest.mark.asyncio
async def test_resolver_books_winning_exit(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    sink = AlertSink(AlertsRepo(tmp_db))
    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        clock=clock,
    )
    await resolver._scan(sink)
    assert paper.list_open_positions() == []
    nav = paper.compute_cost_basis_nav(starting_bankroll=1000.0)
    assert nav == 1010.0


@pytest.mark.asyncio
async def test_resolver_books_losing_exit(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[0.0, 1.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    sink = AlertSink(AlertsRepo(tmp_db))
    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        clock=clock,
    )
    await resolver._scan(sink)
    assert paper.list_open_positions() == []
    nav = paper.compute_cost_basis_nav(starting_bankroll=1000.0)
    assert nav == 990.0


@pytest.mark.asyncio
async def test_resolver_skips_unresolved(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    _open_position(paper)
    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))
    assert len(paper.list_open_positions()) == 1


@pytest.mark.asyncio
async def test_resolver_books_multiple_in_one_scan(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(
        cache,
        condition_id="0xcond-1",
        active=False,
        outcome_prices=[1.0, 0.0],
        asset_ids=["a-y1", "a-n1"],
    )
    _cache_market(
        cache,
        condition_id="0xcond-2",
        active=False,
        outcome_prices=[0.0, 1.0],
        asset_ids=["a-y2", "a-n2"],
    )
    _open_position(paper, condition_id="0xcond-1", asset_id="a-y1", outcome="yes")
    _open_position(paper, condition_id="0xcond-2", asset_id="a-y2", outcome="yes")
    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))
    assert paper.list_open_positions() == []


def test_resolver_interval_from_config(tmp_db) -> None:
    cfg = PaperTradingConfig(enabled=True, resolver_scan_interval_seconds=120.0)
    resolver = PaperResolver(
        config=cfg,
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=PaperTradesRepo(tmp_db),
    )
    assert resolver._interval_seconds() == 120.0
