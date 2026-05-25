"""Tests for PaperResolver."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.poly.models import Market
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


def _async_mocks() -> tuple[AsyncMock, AsyncMock]:
    """Return (data_client, gamma_client) AsyncMocks suitable for tests
    whose cache rows are pre-seeded ``active=False`` (refresh never fires).

    ``get_market_slug_by_condition_id`` returns ``None`` as the safe default
    so that if the refresh path fires unexpectedly it exits early rather
    than crashing on an upsert with a mock object.
    """
    data = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = None
    return data, AsyncMock()


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
        triggering_alert_detector="smart_money",
        rule_variant=None,
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
    data, gamma = _async_mocks()
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
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
    data, gamma = _async_mocks()
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
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
    data, gamma = _async_mocks()
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
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
    data, gamma = _async_mocks()
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
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
        data_client=AsyncMock(),
        gamma_client=AsyncMock(),
    )
    assert resolver._interval_seconds() == 120.0


@pytest.mark.asyncio
async def test_resolver_keeps_position_open_when_insert_exit_raises(tmp_db, monkeypatch) -> None:
    """If insert_exit raises, the position must stay open for retry next cycle."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)

    def boom(**_kwargs: object) -> int:
        raise sqlite3.OperationalError("simulated transient DB failure")

    monkeypatch.setattr(paper, "insert_exit", boom)
    clock = FakeClock(start=float(_NOW + 100))
    data, gamma = _async_mocks()
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    # Must not raise
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))
    # Position is still open (retry will happen next cycle)
    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    # NAV unchanged because no exit was booked
    assert paper.compute_cost_basis_nav(starting_bankroll=1000.0) == 1000.0


@pytest.mark.asyncio
async def test_resolver_refreshes_stale_active_market_then_books_exit(tmp_db) -> None:
    """A market whose cache row is still active=1 gets refreshed via gamma
    and, if gamma reports it resolved, an exit row is booked in the same scan.
    """
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    # Pre-seed cache as still-active (the bug condition).
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)

    resolved_market = Market.model_validate(
        {
            "id": "mkt-0xcond-1",
            "conditionId": "0xcond-1",
            "question": "Test market",
            "slug": "test-market",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["asset-yes", "asset-no"],
            "active": False,
            "closed": True,
        },
    )
    data = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "test-market"
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = resolved_market

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    data.get_market_slug_by_condition_id.assert_awaited_once_with(ConditionId("0xcond-1"))
    gamma.get_market_by_slug.assert_awaited_once_with("test-market")
    assert paper.list_open_positions() == []
    assert paper.compute_cost_basis_nav(starting_bankroll=1000.0) == 1010.0


@pytest.mark.asyncio
async def test_resolver_dedups_refresh_per_scan(tmp_db) -> None:
    """Two open positions on the same condition_id trigger exactly one refresh."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=True, outcome_prices=[0.6, 0.4])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    # Second open position on same condition_id, different fill (e.g. twin trade).
    paper.insert_entry(
        triggering_alert_key="k-0xcond-1-no",
        triggering_alert_detector="velocity",
        rule_variant="fade",
        source_wallet="0xw2",
        condition_id=ConditionId("0xcond-1"),
        asset_id=AssetId("asset-no"),
        outcome="no",
        shares=10.0,
        fill_price=0.4,
        cost_usd=4.0,
        nav_after_usd=996.0,
        ts=_NOW,
    )

    resolved_market = Market.model_validate(
        {
            "id": "mkt-0xcond-1",
            "conditionId": "0xcond-1",
            "question": "Test market",
            "slug": "test-market",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["asset-yes", "asset-no"],
            "active": False,
            "closed": True,
        },
    )
    data = AsyncMock()
    data.get_market_slug_by_condition_id.return_value = "test-market"
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = resolved_market

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    assert data.get_market_slug_by_condition_id.await_count == 1
    assert gamma.get_market_by_slug.await_count == 1
    # Both positions resolved in the same scan.
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_resolver_skips_refresh_when_cache_already_inactive(tmp_db) -> None:
    """Cache rows that already say active=False are not re-fetched."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(cache, active=False, outcome_prices=[1.0, 0.0])
    _open_position(paper, outcome="yes", cost_usd=10.0, shares=20.0)
    data = AsyncMock()
    gamma = AsyncMock()

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    data.get_market_slug_by_condition_id.assert_not_awaited()
    gamma.get_market_by_slug.assert_not_awaited()
    # Exit still booked via the unchanged resolution path.
    assert paper.list_open_positions() == []


@pytest.mark.asyncio
async def test_resolver_refresh_failure_does_not_block_other_positions(tmp_db) -> None:
    """One market's refresh raising must not prevent another from being booked."""
    cfg = PaperTradingConfig(enabled=True)
    cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _cache_market(
        cache,
        condition_id="0xcond-1",
        active=True,
        outcome_prices=[0.5, 0.5],
        asset_ids=["a-y1", "a-n1"],
    )
    _cache_market(
        cache,
        condition_id="0xcond-2",
        active=True,
        outcome_prices=[0.5, 0.5],
        asset_ids=["a-y2", "a-n2"],
    )
    _open_position(paper, condition_id="0xcond-1", asset_id="a-y1", outcome="yes")
    _open_position(paper, condition_id="0xcond-2", asset_id="a-y2", outcome="yes")

    resolved_cond2 = Market.model_validate(
        {
            "id": "mkt-0xcond-2",
            "conditionId": "0xcond-2",
            "question": "Second market",
            "slug": "second-market",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["a-y2", "a-n2"],
            "active": False,
            "closed": True,
        },
    )

    async def slug_side_effect(cid: ConditionId) -> str | None:
        if cid == ConditionId("0xcond-1"):
            raise RuntimeError("transient gamma upstream failure")
        return "second-market"

    data = AsyncMock()
    data.get_market_slug_by_condition_id.side_effect = slug_side_effect
    gamma = AsyncMock()
    gamma.get_market_by_slug.return_value = resolved_cond2

    clock = FakeClock(start=float(_NOW + 100))
    resolver = PaperResolver(
        config=cfg,
        market_cache=cache,
        paper_trades=paper,
        data_client=data,
        gamma_client=gamma,
        clock=clock,
    )
    await resolver._scan(AlertSink(AlertsRepo(tmp_db)))

    open_after = paper.list_open_positions()
    assert len(open_after) == 1
    assert open_after[0].condition_id == ConditionId("0xcond-1")
