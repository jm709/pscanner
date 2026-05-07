"""Unit tests for MarketScopedTradeCollector (#79)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from pscanner.collectors.market_scoped_trades import MarketScopedTradeCollector
from pscanner.config import GateModelMarketFilterConfig
from pscanner.poly.models import Event, Market
from pscanner.store.repo import WalletTrade
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
        gamma=gamma,
        data_client=None,  # type: ignore[arg-type]
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
        gamma=gamma,
        data_client=None,  # type: ignore[arg-type]
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
    collector = MarketScopedTradeCollector(config=cfg, gamma=gamma, data_client=data)
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
    collector = MarketScopedTradeCollector(config=cfg, gamma=gamma, data_client=data)
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
    collector = MarketScopedTradeCollector(config=cfg, gamma=gamma, data_client=data)
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
