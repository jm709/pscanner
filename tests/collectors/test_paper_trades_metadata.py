"""Tests for PaperTradesMetadataCollector."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from pscanner.collectors.paper_trades_metadata import PaperTradesMetadataCollector
from pscanner.poly.ids import AssetId, ConditionId, EventSlug, MarketId
from pscanner.poly.models import Event, Market
from pscanner.store.repo import (
    CachedMarket,
    EventTagCacheRepo,
    MarketCacheRepo,
    PaperTradesRepo,
)

_NOW = 1700000000


def _seed_paper_entry(
    paper: PaperTradesRepo,
    *,
    condition_id: str,
    detector: str = "subgraph_copy",
) -> None:
    paper.insert_entry(
        triggering_alert_key=f"k-{condition_id}",
        triggering_alert_detector=detector,
        rule_variant=None,
        source_wallet="0xwallet",
        condition_id=ConditionId(condition_id),
        asset_id=AssetId(f"asset-{condition_id}"),
        outcome="Yes",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=995.0,
        ts=_NOW,
    )


def _seed_cached_market(
    cache: MarketCacheRepo,
    *,
    condition_id: str,
    event_slug: str | None,
) -> None:
    cache.upsert(
        CachedMarket(
            market_id=MarketId(f"mkt-{condition_id}"),
            event_id=None,
            title="Test",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.5, 0.5],
            outcomes=["Yes", "No"],
            asset_ids=[AssetId("a-y"), AssetId("a-n")],
            active=True,
            cached_at=_NOW,
            condition_id=ConditionId(condition_id),
            event_slug=EventSlug(event_slug) if event_slug else None,
        ),
    )


def _build_market(*, condition_id: str, slug: str = "mkt-slug") -> Market:
    return Market.model_validate(
        {
            "id": f"mkt-{condition_id}",
            "conditionId": condition_id,
            "question": "Test",
            "slug": slug,
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["a-y", "a-n"],
            "active": True,
            "closed": False,
            "events": [{"id": "evt-1", "slug": "evt-slug"}],
        },
    )


def _build_event(slug: str = "evt-slug", tags: list[str] | None = None) -> Event:
    return Event.model_validate(
        {
            "id": "evt-1",
            "title": "Test Event",
            "slug": slug,
            "markets": [],
            "tags": tags or ["Sports", "NBA"],
        },
    )


def _make_collector(
    tmp_db: sqlite3.Connection,
    *,
    data: AsyncMock | None = None,
    gamma: AsyncMock | None = None,
) -> tuple[
    PaperTradesMetadataCollector,
    MarketCacheRepo,
    EventTagCacheRepo,
    PaperTradesRepo,
    AsyncMock,
    AsyncMock,
]:
    cache = MarketCacheRepo(tmp_db)
    tag_cache = EventTagCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    d = data if data is not None else AsyncMock()
    g = gamma if gamma is not None else AsyncMock()
    collector = PaperTradesMetadataCollector(
        db=tmp_db,
        market_cache=cache,
        event_tag_cache=tag_cache,
        data_client=d,
        gamma_client=g,
    )
    return collector, cache, tag_cache, paper, d, g


@pytest.mark.asyncio
async def test_no_paper_trades_no_calls(tmp_db) -> None:
    collector, _cache, _tag_cache, _paper, data, gamma = _make_collector(tmp_db)
    await collector.poll_once()
    data.get_market_slug_by_condition_id.assert_not_awaited()
    gamma.get_market_by_slug.assert_not_awaited()
    gamma.get_event_by_slug.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_market_cache_triggers_refresh(tmp_db) -> None:
    """A paper-trade condition_id without a market_cache row triggers a refresh."""
    collector, cache, _tag_cache, paper, data, gamma = _make_collector(tmp_db)
    _seed_paper_entry(paper, condition_id="0xcond-1")
    data.get_market_slug_by_condition_id.return_value = "mkt-slug"
    gamma.get_market_by_slug.return_value = _build_market(condition_id="0xcond-1")
    # Event lookup for the event_slug surfaced by the new market_cache row.
    gamma.get_event_by_slug.return_value = _build_event(slug="evt-slug")

    await collector.poll_once()

    data.get_market_slug_by_condition_id.assert_awaited_once_with(ConditionId("0xcond-1"))
    cached = cache.get_by_condition_id(ConditionId("0xcond-1"))
    assert cached is not None
    assert cached.event_slug == EventSlug("evt-slug")
    gamma.get_event_by_slug.assert_awaited_once_with(EventSlug("evt-slug"))


@pytest.mark.asyncio
async def test_null_event_slug_triggers_asset_id_refresh(tmp_db) -> None:
    """Cache rows with event_slug=NULL hit the token-id refresh path, not the slug path.

    The token-id response includes the events array that the Market model's
    _hoist_event_fields validator uses to populate event_slug. The slug
    response sometimes omits it for recurring/date-rolled markets.
    """
    collector, cache, _tag_cache, paper, data, gamma = _make_collector(tmp_db)
    _seed_paper_entry(paper, condition_id="0xcond-2")
    _seed_cached_market(cache, condition_id="0xcond-2", event_slug=None)
    # The seeded paper entry's asset_id is "asset-0xcond-2" (see _seed_paper_entry).
    # Build a Market whose clob_token_ids include that asset_id.
    market = Market.model_validate(
        {
            # Same market_id as the seeded row so upsert hits the same PK
            # (would otherwise create a second row, leaving the seeded null
            # row to win get_by_condition_id's LIMIT 1).
            "id": "mkt-0xcond-2",
            "conditionId": "0xcond-2",
            "question": "Test",
            "slug": "mkt-slug-2",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["asset-0xcond-2", "asset-other"],
            "active": True,
            "closed": False,
            "events": [{"id": "evt-1", "slug": "evt-recurring"}],
        },
    )
    gamma.list_markets.return_value = [market]
    gamma.get_event_by_slug.return_value = _build_event(slug="evt-recurring")

    await collector.poll_once()

    # Token-id path, not slug path.
    data.get_market_slug_by_condition_id.assert_not_awaited()
    gamma.list_markets.assert_awaited_once_with(clob_token_ids="asset-0xcond-2", limit=5)  # noqa: S106
    cached = cache.get_by_condition_id(ConditionId("0xcond-2"))
    assert cached is not None
    assert str(cached.event_slug) == "evt-recurring"


@pytest.mark.asyncio
async def test_populated_caches_skip_all_calls(tmp_db) -> None:
    """Fully-cached condition_ids trigger no gamma/data calls."""
    collector, cache, tag_cache, paper, data, gamma = _make_collector(tmp_db)
    _seed_paper_entry(paper, condition_id="0xcond-3")
    _seed_cached_market(cache, condition_id="0xcond-3", event_slug="evt-3")
    tag_cache.upsert(EventSlug("evt-3"), ["Sports"])

    await collector.poll_once()

    data.get_market_slug_by_condition_id.assert_not_awaited()
    gamma.get_market_by_slug.assert_not_awaited()
    gamma.get_event_by_slug.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_event_tag_cache_only_phase(tmp_db) -> None:
    """Market_cache populated but event_tag_cache empty → only phase 2 fires."""
    collector, cache, tag_cache, paper, data, gamma = _make_collector(tmp_db)
    _seed_paper_entry(paper, condition_id="0xcond-4")
    _seed_cached_market(cache, condition_id="0xcond-4", event_slug="evt-4")
    gamma.get_event_by_slug.return_value = _build_event(slug="evt-4", tags=["Sports", "NBA"])

    await collector.poll_once()

    data.get_market_slug_by_condition_id.assert_not_awaited()
    gamma.get_event_by_slug.assert_awaited_once_with(EventSlug("evt-4"))
    assert tag_cache.get(EventSlug("evt-4")) == ["Sports", "NBA"]


@pytest.mark.asyncio
async def test_event_fetch_failure_logged_and_skipped(tmp_db) -> None:
    """get_event_by_slug raising must not block other slugs."""
    collector, cache, tag_cache, paper, _data, gamma = _make_collector(tmp_db)
    _seed_paper_entry(paper, condition_id="0xcond-5")
    _seed_paper_entry(paper, condition_id="0xcond-6")
    _seed_cached_market(cache, condition_id="0xcond-5", event_slug="evt-5")
    _seed_cached_market(cache, condition_id="0xcond-6", event_slug="evt-6")

    async def slug_side_effect(slug: EventSlug) -> Event | None:
        if slug == EventSlug("evt-5"):
            raise RuntimeError("transient gamma failure")
        return _build_event(slug=slug, tags=["Politics"])

    gamma.get_event_by_slug.side_effect = slug_side_effect

    await collector.poll_once()

    # evt-5 stays unpopulated; evt-6 succeeds.
    assert tag_cache.get(EventSlug("evt-5")) is None
    assert tag_cache.get(EventSlug("evt-6")) == ["Politics"]


@pytest.mark.asyncio
async def test_dedup_across_multiple_entries_same_cid(tmp_db) -> None:
    """Two paper-trade entries on the same condition_id trigger one refresh."""
    collector, _cache, _tag_cache, paper, data, gamma = _make_collector(tmp_db)
    paper.insert_entry(
        triggering_alert_key="k-1",
        triggering_alert_detector="subgraph_copy",
        rule_variant=None,
        source_wallet="0xa",
        condition_id=ConditionId("0xcond-7"),
        asset_id=AssetId("a-y"),
        outcome="Yes",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=995.0,
        ts=_NOW,
    )
    paper.insert_entry(
        triggering_alert_key="k-2",
        triggering_alert_detector="subgraph_copy",
        rule_variant=None,
        source_wallet="0xb",
        condition_id=ConditionId("0xcond-7"),
        asset_id=AssetId("a-n"),
        outcome="No",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=990.0,
        ts=_NOW,
    )
    data.get_market_slug_by_condition_id.return_value = "mkt-slug-7"
    gamma.get_market_by_slug.return_value = _build_market(
        condition_id="0xcond-7", slug="mkt-slug-7"
    )
    gamma.get_event_by_slug.return_value = _build_event()

    await collector.poll_once()

    assert data.get_market_slug_by_condition_id.await_count == 1
