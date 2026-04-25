"""Tests for ``MarketTickCollector`` (DC-4 Wave 2).

The watchlist registry, data client, gamma client, and websocket are all
mocked. The :class:`MarketTicksRepo` runs against an in-memory SQLite via
the ``tmp_db`` conftest fixture so end-to-end persistence is verified.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.ticks import MarketTickCollector, _Orderbook
from pscanner.config import TicksConfig
from pscanner.poly.models import Market, Position, WsBookMessage
from pscanner.store.repo import MarketTicksRepo

_BASE_TS = 1_700_000_000


def _async_iter_markets(items: list[Any]) -> AsyncIterator[Any]:
    """Wrap a list in an async iterator matching ``iter_markets`` shape."""

    async def _gen() -> AsyncIterator[Any]:
        for item in items:
            yield item

    return _gen()


def _make_registry(addresses: set[str] | None = None) -> MagicMock:
    """Build a ``WatchlistRegistry`` mock returning ``addresses``."""
    addrs = set(addresses or set())
    registry = MagicMock()
    registry.addresses = MagicMock(side_effect=lambda: set(addrs))
    return registry


def _make_position(
    *,
    asset: str,
    condition_id: str,
    proxy_wallet: str = "0xWallet",
) -> Position:
    """Build a ``Position`` with the fields the collector reads."""
    return Position.model_validate(
        {
            "proxyWallet": proxy_wallet,
            "asset": asset,
            "conditionId": condition_id,
            "outcome": "Yes",
            "outcomeIndex": 0,
            "size": 100.0,
            "avgPrice": 0.5,
        },
    )


def _make_market(
    *,
    market_id: str,
    condition_id: str,
    clob_token_ids: list[str],
    volume: float,
    enable_order_book: bool = True,
) -> Market:
    """Build a lightweight ``Market`` stand-in via direct pydantic validation."""
    payload: dict[str, Any] = {
        "id": market_id,
        "conditionId": condition_id,
        "question": f"q-{market_id}",
        "slug": f"m-{market_id}",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.5", "0.5"],
        "volume": volume,
        "active": True,
        "closed": False,
        "enableOrderBook": enable_order_book,
        "clobTokenIds": clob_token_ids,
    }
    return Market.model_validate(payload)


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    registry: MagicMock | None = None,
    data_client: MagicMock | None = None,
    gamma_client: MagicMock | None = None,
    ws: MagicMock | None = None,
    config: TicksConfig | None = None,
) -> tuple[MarketTickCollector, MarketTicksRepo, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Wire up a collector with mocks plus a real ``MarketTicksRepo``."""
    repo = MarketTicksRepo(tmp_db)
    reg = registry or _make_registry()
    dc = data_client or MagicMock()
    if not hasattr(dc, "get_positions") or not isinstance(dc.get_positions, AsyncMock):
        dc.get_positions = AsyncMock(return_value=[])
    gc = gamma_client or MagicMock()
    if not hasattr(gc, "iter_markets") or not callable(gc.iter_markets):
        gc.iter_markets = lambda **_: _async_iter_markets([])
    fake_ws = ws or MagicMock()
    if not hasattr(fake_ws, "connect") or not isinstance(fake_ws.connect, AsyncMock):
        fake_ws.connect = AsyncMock(return_value=None)
    if not hasattr(fake_ws, "subscribe") or not isinstance(fake_ws.subscribe, AsyncMock):
        fake_ws.subscribe = AsyncMock(return_value=None)
    if not hasattr(fake_ws, "close") or not isinstance(fake_ws.close, AsyncMock):
        fake_ws.close = AsyncMock(return_value=None)
    cfg = config or TicksConfig()
    collector = MarketTickCollector(
        config=cfg,
        ws=fake_ws,  # type: ignore[arg-type]
        gamma_client=gc,  # type: ignore[arg-type]
        data_client=dc,  # type: ignore[arg-type]
        registry=reg,  # type: ignore[arg-type]
        ticks_repo=repo,
    )
    return collector, repo, reg, dc, gc, fake_ws


def _book_msg(
    *,
    asset_id: str,
    market: str | None,
    bids: list[dict[str, str]],
    asks: list[dict[str, str]],
    last_trade_price: str | None = None,
) -> WsBookMessage:
    """Build a ``book`` event with bids/asks/last_trade_price under ``data``."""
    data: dict[str, Any] = {"bids": bids, "asks": asks}
    if last_trade_price is not None:
        data["last_trade_price"] = last_trade_price
    return WsBookMessage.model_validate(
        {
            "event_type": "book",
            "asset_id": asset_id,
            "market": market,
            "data": data,
        },
    )


def _price_change_msg(
    *,
    market: str | None,
    changes: list[dict[str, Any]],
) -> WsBookMessage:
    """Build a ``price_change`` event with a ``price_changes`` array under data."""
    return WsBookMessage.model_validate(
        {
            "event_type": "price_change",
            "market": market,
            "data": {"price_changes": changes},
        },
    )


async def test_book_message_updates_orderbook(tmp_db: sqlite3.Connection) -> None:
    """A ``book`` snapshot replaces bids/asks and updates last_trade_price."""
    collector, _, _, _, _, _ = _make_collector(tmp_db=tmp_db)
    msg = _book_msg(
        asset_id="A1",
        market="0xcond",
        bids=[{"price": "0.40", "size": "100"}],
        asks=[{"price": "0.42", "size": "50"}],
        last_trade_price="0.41",
    )
    await collector._handle_message(msg)
    book = collector._books["A1"]
    assert book.bids == {0.40: 100.0}
    assert book.asks == {0.42: 50.0}
    assert book.last_trade_price == 0.41
    assert book.condition_id == "0xcond"


async def test_price_change_delta_updates_levels(tmp_db: sqlite3.Connection) -> None:
    """A ``price_change`` updates an existing bid level in place."""
    collector, _, _, _, _, _ = _make_collector(tmp_db=tmp_db)
    seed = _book_msg(
        asset_id="A1",
        market="0xcond",
        bids=[{"price": "0.40", "size": "100"}],
        asks=[{"price": "0.42", "size": "50"}],
    )
    await collector._handle_message(seed)
    delta = _price_change_msg(
        market="0xcond",
        changes=[
            {"asset_id": "A1", "price": "0.40", "size": "50", "side": "BUY"},
        ],
    )
    await collector._handle_message(delta)
    assert collector._books["A1"].bids == {0.40: 50.0}


async def test_price_change_size_zero_deletes_level(tmp_db: sqlite3.Connection) -> None:
    """A price-change with ``size=0`` removes the matching level."""
    collector, _, _, _, _, _ = _make_collector(tmp_db=tmp_db)
    seed = _book_msg(
        asset_id="A1",
        market="0xcond",
        bids=[
            {"price": "0.40", "size": "100"},
            {"price": "0.39", "size": "75"},
        ],
        asks=[{"price": "0.42", "size": "50"}],
    )
    await collector._handle_message(seed)
    delta = _price_change_msg(
        market="0xcond",
        changes=[{"asset_id": "A1", "price": "0.40", "size": "0", "side": "BUY"}],
    )
    await collector._handle_message(delta)
    assert collector._books["A1"].bids == {0.39: 75.0}


async def test_snapshot_computes_mid_spread_depth(tmp_db: sqlite3.Connection) -> None:
    """``snapshot_once`` derives mid/spread/depth and writes one row."""
    collector, repo, _, _, _, _ = _make_collector(tmp_db=tmp_db)
    bids = [
        {"price": "0.50", "size": "10"},
        {"price": "0.49", "size": "20"},
        {"price": "0.48", "size": "30"},
        {"price": "0.47", "size": "40"},
        {"price": "0.46", "size": "50"},
        {"price": "0.45", "size": "60"},
    ]
    asks = [
        {"price": "0.52", "size": "5"},
        {"price": "0.53", "size": "15"},
        {"price": "0.54", "size": "25"},
        {"price": "0.55", "size": "35"},
        {"price": "0.56", "size": "45"},
        {"price": "0.57", "size": "55"},
    ]
    await collector._handle_message(
        _book_msg(asset_id="A1", market="0xcond", bids=bids, asks=asks),
    )
    inserted = await collector.snapshot_once()
    assert inserted == 1
    rows = repo.recent_for_asset("A1")
    assert len(rows) == 1
    row = rows[0]
    assert row.best_bid == pytest.approx(0.50)
    assert row.best_ask == pytest.approx(0.52)
    assert row.mid_price == pytest.approx(0.51)
    assert row.spread == pytest.approx(0.02)
    # Top 5 bids by price descending: 0.50,0.49,0.48,0.47,0.46 → 10+20+30+40+50
    assert row.bid_depth_top5 == pytest.approx(150.0)
    # Top 5 asks by price ascending: 0.52,0.53,0.54,0.55,0.56 → 5+15+25+35+45
    assert row.ask_depth_top5 == pytest.approx(125.0)
    assert row.condition_id == "0xcond"


async def test_snapshot_skips_assets_with_no_condition_and_no_mid(
    tmp_db: sqlite3.Connection,
) -> None:
    """An empty orderbook with no condition lookup is skipped entirely."""
    collector, repo, _, _, _, _ = _make_collector(tmp_db=tmp_db)
    # Seed an empty orderbook for X1 directly (no message ingested).
    collector._books["X1"] = _Orderbook()
    inserted = await collector.snapshot_once()
    assert inserted == 0
    assert "X1" not in repo.count_by_asset()


async def test_subscription_refresh_wallet_positions(
    tmp_db: sqlite3.Connection,
) -> None:
    """Wallet positions seed the asset universe and the condition lookup."""
    registry = _make_registry({"0xWallet"})
    data_client = MagicMock()
    data_client.get_positions = AsyncMock(
        return_value=[
            _make_position(asset="A1", condition_id="0xcond1"),
            _make_position(asset="A2", condition_id="0xcond1"),
        ],
    )
    gamma = MagicMock()
    gamma.iter_markets = lambda **_: _async_iter_markets([])
    collector, _, _, _, _, ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data_client,
        gamma_client=gamma,
    )
    await collector._refresh_subscriptions()
    assert ws.subscribe.await_count == 1
    subscribed_arg = ws.subscribe.await_args.args[0]
    assert sorted(subscribed_arg) == ["A1", "A2"]
    assert collector.subscribed_asset_ids() == {"A1", "A2"}
    assert collector._asset_to_condition["A1"] == "0xcond1"
    assert collector._asset_to_condition["A2"] == "0xcond1"


async def test_subscription_refresh_volume_floor(
    tmp_db: sqlite3.Connection,
) -> None:
    """Markets below the volume floor contribute no asset ids."""
    markets = [
        _make_market(
            market_id="m1",
            condition_id="0xc1",
            clob_token_ids=["A1", "A2"],
            volume=50_000.0,
        ),
        _make_market(
            market_id="m2",
            condition_id="0xc2",
            clob_token_ids=["B1", "B2"],
            volume=5_000.0,  # below floor
        ),
        _make_market(
            market_id="m3",
            condition_id="0xc3",
            clob_token_ids=["C1", "C2"],
            volume=100_000.0,
        ),
    ]
    gamma = MagicMock()
    gamma.iter_markets = lambda **_: _async_iter_markets(markets)
    collector, _, _, _, _, ws = _make_collector(
        tmp_db=tmp_db,
        gamma_client=gamma,
        config=TicksConfig(tick_volume_floor_usd=10_000.0),
    )
    await collector._refresh_subscriptions()
    assert ws.subscribe.await_count == 1
    subscribed = ws.subscribe.await_args.args[0]
    assert sorted(subscribed) == ["A1", "A2", "C1", "C2"]


async def test_max_assets_cap(tmp_db: sqlite3.Connection) -> None:
    """The total subscribed set is hard-capped at ``max_assets``."""
    registry = _make_registry({"0xWallet"})
    data_client = MagicMock()
    data_client.get_positions = AsyncMock(
        return_value=[_make_position(asset=f"A{i}", condition_id="0xcond") for i in range(5)],
    )
    gamma = MagicMock()
    gamma.iter_markets = lambda **_: _async_iter_markets([])
    collector, _, _, _, _, ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data_client,
        gamma_client=gamma,
        config=TicksConfig(max_assets=3),
    )
    await collector._refresh_subscriptions()
    assert ws.subscribe.await_count == 1
    subscribed = ws.subscribe.await_args.args[0]
    assert len(subscribed) == 3
    assert len(collector.subscribed_asset_ids()) == 3


async def test_subscription_is_incremental(tmp_db: sqlite3.Connection) -> None:
    """Refreshing twice with the same data does not re-subscribe."""
    registry = _make_registry({"0xWallet"})
    data_client = MagicMock()
    data_client.get_positions = AsyncMock(
        return_value=[
            _make_position(asset="A1", condition_id="0xcond"),
            _make_position(asset="A2", condition_id="0xcond"),
        ],
    )
    gamma = MagicMock()
    gamma.iter_markets = lambda **_: _async_iter_markets([])
    collector, _, _, _, _, ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data_client,
        gamma_client=gamma,
    )
    await collector._refresh_subscriptions()
    await collector._refresh_subscriptions()
    assert ws.subscribe.await_count == 1


async def test_get_recent_mids_filters_by_window(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only mid pairs newer than ``now - window_seconds`` are returned."""
    collector, _, _, _, _, _ = _make_collector(tmp_db=tmp_db)
    # Seed an orderbook so snapshots produce mid prices.
    await collector._handle_message(
        _book_msg(
            asset_id="A1",
            market="0xcond",
            bids=[{"price": "0.40", "size": "100"}],
            asks=[{"price": "0.42", "size": "50"}],
        ),
    )
    timestamps = iter([_BASE_TS, _BASE_TS + 60])
    monkeypatch.setattr(
        "pscanner.collectors.ticks.time.time",
        lambda: next(timestamps),
    )
    await collector.snapshot_once()
    await collector.snapshot_once()

    # Now stamp wall-clock to just after the second snapshot so a 30-second
    # window only captures the second.
    monkeypatch.setattr(
        "pscanner.collectors.ticks.time.time",
        lambda: _BASE_TS + 65,
    )
    pairs = collector.get_recent_mids("A1", window_seconds=30)
    assert len(pairs) == 1
    ts, mid = pairs[0]
    assert ts == _BASE_TS + 60
    assert mid == pytest.approx(0.41)


async def test_volume_floor_skips_markets_with_orderbook_disabled(
    tmp_db: sqlite3.Connection,
) -> None:
    """Markets with ``enable_order_book=False`` are skipped from volume scope."""
    markets = [
        _make_market(
            market_id="m1",
            condition_id="0xc1",
            clob_token_ids=["A1"],
            volume=1_000_000.0,
            enable_order_book=False,
        ),
        _make_market(
            market_id="m2",
            condition_id="0xc2",
            clob_token_ids=["B1"],
            volume=1_000_000.0,
            enable_order_book=True,
        ),
    ]
    gamma = MagicMock()
    gamma.iter_markets = lambda **_: _async_iter_markets(markets)
    collector, _, _, _, _, ws = _make_collector(
        tmp_db=tmp_db,
        gamma_client=gamma,
    )
    await collector._refresh_subscriptions()
    subscribed = ws.subscribe.await_args.args[0]
    assert sorted(subscribed) == ["B1"]


async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` exits within a second after the stop event is set."""
    ws = MagicMock()
    ws.connect = AsyncMock(return_value=None)
    ws.subscribe = AsyncMock(return_value=None)
    ws.close = AsyncMock(return_value=None)

    async def _empty_messages() -> AsyncIterator[Any]:
        # Stay alive until cancelled so ``run`` does not exit prematurely.
        await asyncio.sleep(10)
        if False:
            yield  # pragma: no cover

    ws.messages = _empty_messages
    collector, _, _, _, _, _ = _make_collector(
        tmp_db=tmp_db,
        ws=ws,
        config=TicksConfig(
            tick_interval_seconds=0.05,
            subscription_refresh_seconds=0.05,
        ),
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.1)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=2.0,
    )
    ws.close.assert_awaited()
