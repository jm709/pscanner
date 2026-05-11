"""Unit tests for MarketScopedTradeCollector (#79)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector
from pscanner.config import GateModelMarketFilterConfig
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.poly.ids import ConditionId
from pscanner.poly.models import Event, Market
from pscanner.store.db import init_db
from pscanner.store.repo import MarketCacheRepo, WalletTrade
from pscanner.util.clock import FakeClock


def _make_market(*, condition_id: str, volume: float, mid: str | None = None) -> Market:
    return Market.model_validate(
        {
            "id": mid or condition_id,
            "conditionId": condition_id,
            "question": f"q-{condition_id}",
            "slug": f"slug-{condition_id}",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.5","0.5"]',
            "liquidity": "100",
            "volume": str(volume),
            "active": True,
            "closed": False,
            "clobTokenIds": '["a1","a2"]',
        }
    )


def _make_event(*, slug: str, tags: list[str], markets: list[Market]) -> Event:
    # Event model needs id + slug + tags + markets; build via direct kwargs.
    return Event.model_validate(
        {
            "id": slug,
            "slug": slug,
            "title": slug,
            "tags": [{"label": t} for t in tags],
            "markets": [m.model_dump(by_alias=True) for m in markets],
            "active": True,
            "closed": False,
        }
    )


class _FakeGammaClient:
    def __init__(self, events: list[Event]) -> None:
        self._events = events

    async def iter_events(
        self, *, active: bool = True, closed: bool = False, page_size: int = 100
    ) -> AsyncIterator[Event]:
        del active, closed, page_size
        for event in self._events:
            yield event


class _FakeDataClient:
    def __init__(self, by_market: dict[str, list[dict[str, Any]]]) -> None:
        self._by_market = by_market
        self.calls: list[tuple[str, int]] = []

    async def get_market_trades(
        self, condition_id: str, *, since_ts: int, until_ts: int
    ) -> list[dict[str, Any]]:
        del until_ts
        self.calls.append((condition_id, since_ts))
        rows = self._by_market.get(condition_id, [])
        return [r for r in rows if int(r["timestamp"]) > since_ts]


@pytest.mark.asyncio
async def test_refresh_filters_by_category_and_volume() -> None:
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=100_000,
        max_markets=50,
    )
    esports_event = _make_event(
        slug="ev-esports",
        tags=["Esports"],
        markets=[
            _make_market(condition_id="0xc1", volume=200_000),
            _make_market(condition_id="0xc3", volume=50_000),  # below floor
        ],
    )
    sports_event = _make_event(
        slug="ev-sports",
        tags=["Sports"],
        markets=[_make_market(condition_id="0xc2", volume=500_000)],
    )
    esports_event_2 = _make_event(
        slug="ev-esports-2",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xc4", volume=1_000_000)],
    )
    gamma = _FakeGammaClient([esports_event, sports_event, esports_event_2])
    collector = MarketScopedTradeCollector(
        config=cfg,
        gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=None,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    selected = await collector.refresh_market_set()
    # Only esports markets above the volume floor; sorted desc by volume.
    assert selected == ["0xc4", "0xc1"]


@pytest.mark.asyncio
async def test_refresh_caps_at_max_markets() -> None:
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=0,
        max_markets=2,
    )
    markets = [_make_market(condition_id=f"0xc{i}", volume=float(1000 - i)) for i in range(5)]
    event = _make_event(slug="ev", tags=["Esports"], markets=markets)
    gamma = _FakeGammaClient([event])
    collector = MarketScopedTradeCollector(
        config=cfg,
        gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=None,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    selected = await collector.refresh_market_set()
    assert len(selected) == 2
    # Top-2 by volume (descending): c0 (1000), c1 (999).
    assert selected == ["0xc0", "0xc1"]


@pytest.mark.asyncio
async def test_poll_once_dispatches_new_trades() -> None:
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=0,
    )
    event = _make_event(
        slug="ev",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xc1", volume=100_000)],
    )
    gamma = _FakeGammaClient([event])
    data = _FakeDataClient(
        by_market={
            "0xc1": [
                {
                    "tx_hash": "tx1",
                    "asset_id": "0xa1",
                    "bs": "BUY",
                    "wallet_address": "0xabc",
                    "condition_id": "0xc1",
                    "outcome_side": "YES",
                    "price": 0.42,
                    "size": 100.0,
                    "notional_usd": 42.0,
                    "timestamp": 1_700_000_100,
                }
            ]
        }
    )
    collector = MarketScopedTradeCollector(
        config=cfg,
        gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=data,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    received: list[WalletTrade] = []
    collector.subscribe_new_trade(received.append)
    await collector.refresh_market_set()
    n = await collector.poll_once()
    assert n == 1
    assert len(received) == 1
    assert received[0].condition_id == "0xc1"
    assert received[0].transaction_hash == "tx1"
    assert received[0].side == "BUY"  # NB: WalletTrade.side is BUY/SELL


@pytest.mark.asyncio
async def test_poll_once_handles_polymarket_camelcase_keys() -> None:
    """Pin the live Polymarket ``/trades?market=`` row shape.

    Surfaced during the 2026-05-08 smoke run on the desktop: the API
    returns camelCase keys (``transactionHash``, ``proxyWallet``,
    ``conditionId``, ``asset``, ``usdcSize``) — the original
    ``_row_to_wallet_trade`` was looking up snake_case (``tx_hash``,
    ``wallet_address``, ``condition_id``, ``asset_id``, ``notional_usd``)
    and warning ``market_scoped.bad_row`` on every observed trade. This
    test pins the canonical keys so that regression can't recur.
    """
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=0,
    )
    event = _make_event(
        slug="ev",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xc1", volume=100_000)],
    )
    gamma = _FakeGammaClient([event])
    data = _FakeDataClient(
        by_market={
            "0xc1": [
                {
                    # Real Polymarket /trades response shape (camelCase).
                    "transactionHash": "0xabc123",
                    "asset": "0xasset1",
                    "side": "BUY",
                    "proxyWallet": "0xwallet",
                    "conditionId": "0xc1",
                    "outcome": "Yes",
                    "outcomeIndex": 0,
                    "price": 0.42,
                    "size": 100.0,
                    "usdcSize": 42.0,
                    "timestamp": 1_700_000_100,
                    "title": "irrelevant",
                    "slug": "ev-slug",
                    "name": "Trader",
                    "pseudonym": "Trader",
                    "bio": "",
                    "profileImage": "",
                    "profileImageOptimized": "",
                    "icon": "",
                    "eventSlug": "ev",
                }
            ]
        }
    )
    collector = MarketScopedTradeCollector(
        config=cfg,
        gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=data,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    received: list[WalletTrade] = []
    collector.subscribe_new_trade(received.append)
    await collector.refresh_market_set()
    n = await collector.poll_once()
    assert n == 1
    assert len(received) == 1
    trade = received[0]
    assert trade.transaction_hash == "0xabc123"
    assert trade.asset_id == "0xasset1"
    assert trade.side == "BUY"
    assert trade.wallet == "0xwallet"
    assert trade.condition_id == "0xc1"
    assert trade.price == 0.42
    assert trade.size == 100.0
    assert trade.usd_value == 42.0
    assert trade.timestamp == 1_700_000_100


@pytest.mark.asyncio
async def test_poll_once_advances_last_seen_ts() -> None:
    cfg = GateModelMarketFilterConfig(
        enabled=True, accepted_categories=("esports",), min_volume_24h_usd=0
    )
    event = _make_event(
        slug="ev",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xc1", volume=100_000)],
    )
    gamma = _FakeGammaClient([event])
    data = _FakeDataClient(
        by_market={
            "0xc1": [
                {
                    "tx_hash": "tx1",
                    "asset_id": "0xa1",
                    "bs": "BUY",
                    "wallet_address": "0xabc",
                    "condition_id": "0xc1",
                    "outcome_side": "YES",
                    "price": 0.42,
                    "size": 100.0,
                    "notional_usd": 42.0,
                    "timestamp": 1_700_000_100,
                }
            ]
        }
    )
    collector = MarketScopedTradeCollector(
        config=cfg,
        gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=data,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    await collector.refresh_market_set()
    await collector.poll_once()
    await collector.poll_once()  # second poll: nothing new since last_seen_ts advanced
    # First call had since_ts=0, second should have since_ts >= 1_700_000_100.
    assert data.calls[0][1] == 0
    assert data.calls[1][1] >= 1_700_000_100


@pytest.mark.asyncio
async def test_run_loop_polls_on_cadence() -> None:
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=0,
        poll_interval_seconds=30,
    )
    event = _make_event(
        slug="ev",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xc1", volume=100_000)],
    )
    gamma = _FakeGammaClient([event])
    data = _FakeDataClient(by_market={"0xc1": []})
    collector = MarketScopedTradeCollector(
        config=cfg,
        gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=data,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    clk = FakeClock()
    stop_event = asyncio.Event()
    task = asyncio.create_task(collector.run(stop_event, clock=clk))
    try:
        await asyncio.sleep(0)
        await clk.advance(30)
        await asyncio.sleep(0)
        await clk.advance(30)
        await asyncio.sleep(0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    # At least 2 poll iterations completed.
    assert len(data.calls) >= 2


@pytest.mark.asyncio
async def test_refresh_populates_provider_metadata_for_every_candidate(
    tmp_path: Path,
) -> None:
    """`refresh_market_set` writes MarketMetadata for every category-matching market.

    Issue #102: live open markets aren't in `corpus_markets`, so the
    collector is the only thing that can teach the provider about them
    before a trade arrives.
    """
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=10.0,
        max_markets=2,
    )
    esports_a = _make_event(
        slug="ev-a",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xMA", volume=500.0)],
    )
    esports_b = _make_event(
        slug="ev-b",
        tags=["Esports"],
        markets=[
            _make_market(condition_id="0xMB1", volume=400.0),
            _make_market(condition_id="0xMB2", volume=15.0),  # passes floor; below top-N
        ],
    )
    politics = _make_event(
        slug="ev-p",
        tags=["Politics"],
        markets=[_make_market(condition_id="0xMP", volume=99999.0)],
    )
    gamma = _FakeGammaClient([esports_a, esports_b, politics])
    data_client = _FakeDataClient(by_market={})

    db_path = tmp_path / "daemon.sqlite3"
    conn = init_db(db_path)
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        collector = MarketScopedTradeCollector(
            config=cfg,
            gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            data_client=data_client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            provider=provider,
        )
        selected = await collector.refresh_market_set()
    finally:
        conn.close()

    assert selected == ["0xMA", "0xMB1"]  # top-2 by volume
    # All three esports markets seeded — even the one below top-N.
    assert provider.market_metadata("0xMA").category == "esports"
    assert provider.market_metadata("0xMB1").category == "esports"
    assert provider.market_metadata("0xMB2").category == "esports"
    # Politics market is filtered out at the category gate.
    with pytest.raises(KeyError):
        provider.market_metadata("0xMP")


@pytest.mark.asyncio
async def test_refresh_populates_market_metadata_categories() -> None:
    """Every market that passes the filter lands with ``MarketMetadata.categories``
    populated to the multi-label set, not just a single primary string.
    """
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=100_000,
        max_markets=50,
    )
    event = _make_event(
        slug="ev-esports",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xc1", volume=200_000)],
    )
    gamma = _FakeGammaClient([event])
    data = _FakeDataClient({})
    conn = init_db(Path(":memory:"))
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        collector = MarketScopedTradeCollector(
            config=cfg,
            gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            data_client=data,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            provider=provider,
        )
        await collector.refresh_market_set()
        meta = provider.market_metadata("0xc1")
        # categories should be populated from categorize_tags(event.tags)
        assert "esports" in meta.categories
        # primary category string still single-valued (priority-first)
        assert meta.category == "esports"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_refresh_accepts_multi_label_event_via_intersection() -> None:
    """A multi-tag event passes the filter when any of its categories
    intersects ``accepted_categories``, even if the primary doesn't.

    Differentiating case: tags = ["Fed Rates", "Global Elections"] ->
    primary_category = MACRO, categorize_tags = {MACRO, ELECTIONS}.
    With accepted_categories = ("elections",), the legacy single-string
    filter rejects (primary "macro" not in accepted); the new set-intersection
    filter accepts.
    """
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("elections",),
        min_volume_24h_usd=100_000,
        max_markets=50,
    )
    event = _make_event(
        slug="ev-fed-during-election",
        tags=["Fed Rates", "Global Elections"],
        markets=[_make_market(condition_id="0xc1", volume=200_000)],
    )
    gamma = _FakeGammaClient([event])
    data = _FakeDataClient({})
    conn = init_db(Path(":memory:"))
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        collector = MarketScopedTradeCollector(
            config=cfg,
            gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            data_client=data,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            provider=provider,
        )
        selected = await collector.refresh_market_set()
        assert "0xc1" in selected
        meta = provider.market_metadata("0xc1")
        assert "macro" in meta.categories
        assert "elections" in meta.categories
        # Primary remains the highest-priority match (MACRO precedes ELECTIONS).
        assert meta.category == "macro"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_refresh_upserts_market_cache_for_every_candidate(
    tmp_path: Path,
) -> None:
    """`refresh_market_set` upserts MarketCacheRepo for every category-matching market.

    Issue #103: in a gate-model-only config, no other code path populates
    market_cache, so `GateModelDetector._resolve_outcome_side` finds no cached
    market and silently drops every trade. The collector must seed the cache
    from the same gamma response it already has in hand.
    """
    cfg = GateModelMarketFilterConfig(
        enabled=True,
        accepted_categories=("esports",),
        min_volume_24h_usd=10.0,
        max_markets=2,
    )
    esports_a = _make_event(
        slug="ev-a",
        tags=["Esports"],
        markets=[_make_market(condition_id="0xMA", volume=500.0)],
    )
    esports_b = _make_event(
        slug="ev-b",
        tags=["Esports"],
        markets=[
            _make_market(condition_id="0xMB1", volume=400.0),
            _make_market(condition_id="0xMB2", volume=15.0),  # passes floor; below top-N
        ],
    )
    politics = _make_event(
        slug="ev-p",
        tags=["Politics"],
        markets=[_make_market(condition_id="0xMP", volume=99999.0)],
    )
    gamma = _FakeGammaClient([esports_a, esports_b, politics])
    data_client = _FakeDataClient(by_market={})

    db_path = tmp_path / "daemon.sqlite3"
    conn = init_db(db_path)
    try:
        market_cache = MarketCacheRepo(conn)
        collector = MarketScopedTradeCollector(
            config=cfg,
            gamma=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            data_client=data_client,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            market_cache=market_cache,
        )
        await collector.refresh_market_set()

        # All three esports markets cached — including the one below top-N.
        assert market_cache.get_by_condition_id(ConditionId("0xMA")) is not None
        assert market_cache.get_by_condition_id(ConditionId("0xMB1")) is not None
        assert market_cache.get_by_condition_id(ConditionId("0xMB2")) is not None
        # Politics market is filtered out at the category gate.
        assert market_cache.get_by_condition_id(ConditionId("0xMP")) is None
    finally:
        conn.close()
