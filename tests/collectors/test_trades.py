"""Tests for ``TradeCollector``.

These exercise ``_record_trade``, ``_refresh_subscriptions``,
``_on_watchlist_add``, and ``run`` directly with synthesised dependencies.
The :class:`WatchlistRegistry` and websocket are fully mocked; the
:class:`WalletTradesRepo` runs against an in-memory SQLite to verify
end-to-end persistence and dedupe behaviour.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.trades import TradeCollector
from pscanner.poly.models import Position, WsBookMessage, WsTradeMessage
from pscanner.store.repo import WalletTradesRepo

_NOW = 1_700_000_000
_WALLET = "0xwatched"
_OTHER_WALLET = "0xunwatched"
_ASSET_ID = "asset-1"
_CONDITION_ID = "cond-1"


def _make_trade(**overrides: Any) -> WsTradeMessage:
    """Build a ``CONFIRMED`` trade message with sensible defaults."""
    base: dict[str, Any] = {
        "event_type": "trade",
        "condition_id": _CONDITION_ID,
        "asset_id": _ASSET_ID,
        "side": "BUY",
        "size": 100.0,
        "price": 0.4,
        "taker_proxy": _WALLET,
        "status": "CONFIRMED",
        "transaction_hash": "0xtx1",
        "timestamp": _NOW,
    }
    base.update(overrides)
    return WsTradeMessage.model_validate(base)


def _make_position(asset_id: str, *, address: str = _WALLET) -> Position:
    """Build a ``Position`` with the given asset id."""
    return Position.model_validate(
        {
            "proxyWallet": address,
            "asset": asset_id,
            "conditionId": _CONDITION_ID,
            "outcome": "Yes",
            "outcomeIndex": 0,
            "size": 1.0,
            "avgPrice": 0.5,
        },
    )


def _make_registry(addresses: set[str] | None = None) -> MagicMock:
    """Build a ``WatchlistRegistry`` mock with realistic membership semantics."""
    addrs = set(addresses or set())
    registry = MagicMock()
    registry.__contains__ = MagicMock(side_effect=lambda addr: addr in addrs)
    registry.addresses = MagicMock(side_effect=lambda: set(addrs))
    registry.subscribe = MagicMock()
    registry._addrs = addrs  # exposed for tests that mutate it
    return registry


def _make_ws() -> MagicMock:
    """Build a websocket mock with async ``connect``, ``subscribe``, ``close``."""
    ws = MagicMock()
    ws.connect = AsyncMock()
    ws.subscribe = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    registry: MagicMock | None = None,
    data_client: MagicMock | None = None,
    ws: MagicMock | None = None,
    refresh_seconds: float = 300.0,
) -> tuple[TradeCollector, WalletTradesRepo, MagicMock, MagicMock, MagicMock]:
    """Wire up a ``TradeCollector`` with default mocks plus a real repo."""
    repo = WalletTradesRepo(tmp_db)
    reg = registry or _make_registry({_WALLET})
    dc = data_client or MagicMock()
    if not hasattr(dc, "get_positions") or not isinstance(dc.get_positions, AsyncMock):
        dc.get_positions = AsyncMock(return_value=[])
    socket = ws or _make_ws()
    collector = TradeCollector(
        registry=reg,  # type: ignore[arg-type]
        data_client=dc,  # type: ignore[arg-type]
        ws=socket,  # type: ignore[arg-type]
        trades_repo=repo,
        subscription_refresh_seconds=refresh_seconds,
    )
    return collector, repo, reg, dc, socket


@pytest.mark.asyncio
async def test_record_trade_inserts_row_for_watched_wallet(
    tmp_db: sqlite3.Connection,
) -> None:
    collector, repo, _reg, _dc, _ws = _make_collector(tmp_db=tmp_db)

    collector._record_trade(_make_trade(size=200.0, price=0.6))

    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 1
    row = rows[0]
    assert row.transaction_hash == "0xtx1"
    assert row.asset_id == _ASSET_ID
    assert row.side == "BUY"
    assert row.wallet == _WALLET
    assert row.condition_id == _CONDITION_ID
    assert row.size == pytest.approx(200.0)
    assert row.price == pytest.approx(0.6)
    assert row.usd_value == pytest.approx(120.0)
    assert row.status == "CONFIRMED"
    assert row.source == "ws"
    assert row.timestamp == _NOW


@pytest.mark.asyncio
async def test_consume_loop_skips_unwatched_wallet(tmp_db: sqlite3.Connection) -> None:
    """A trade from a wallet outside the registry must not be persisted."""
    registry = _make_registry({_WALLET})
    ws = _make_ws()
    trade = _make_trade(taker_proxy=_OTHER_WALLET, transaction_hash="0xtxOther")

    async def _messages() -> AsyncIterator[Any]:
        yield trade

    ws.messages = _messages
    collector, repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        ws=ws,
    )

    stop_event = asyncio.Event()
    await collector._consume_loop(stop_event)

    assert repo.recent_for_wallet(_OTHER_WALLET) == []
    assert repo.recent_for_wallet(_WALLET) == []


@pytest.mark.asyncio
async def test_consume_loop_persists_watched_trade(tmp_db: sqlite3.Connection) -> None:
    """Trades from watched wallets flow through the consume loop into the repo."""
    registry = _make_registry({_WALLET})
    ws = _make_ws()
    trade = _make_trade(transaction_hash="0xtxConsume")
    book_msg = WsBookMessage.model_validate(
        {"event_type": "book", "asset_id": "x", "data": {}},
    )

    async def _messages() -> AsyncIterator[Any]:
        yield book_msg  # non-trade messages must be skipped, not crash
        yield trade

    ws.messages = _messages
    collector, repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        ws=ws,
    )

    await collector._consume_loop(asyncio.Event())

    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 1
    assert rows[0].transaction_hash == "0xtxConsume"


@pytest.mark.asyncio
async def test_record_trade_dedupes_on_composite_key(
    tmp_db: sqlite3.Connection,
) -> None:
    """Inserting the same (tx, asset, side) twice yields a single row."""
    collector, repo, _reg, _dc, _ws = _make_collector(tmp_db=tmp_db)

    trade = _make_trade(transaction_hash="0xdup")
    collector._record_trade(trade)
    collector._record_trade(trade)

    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_record_trade_skips_when_tx_hash_missing(
    tmp_db: sqlite3.Connection,
) -> None:
    """Without a tx_hash the dedupe key is unsafe; the row must not be written."""
    collector, repo, _reg, _dc, _ws = _make_collector(tmp_db=tmp_db)

    collector._record_trade(_make_trade(transaction_hash=None))

    assert repo.recent_for_wallet(_WALLET) == []
    assert repo.count_by_wallet() == {}


@pytest.mark.asyncio
async def test_refresh_subscriptions_unions_assets_across_wallets(
    tmp_db: sqlite3.Connection,
) -> None:
    """``_refresh_subscriptions`` unions every wallet's open-position assets."""
    registry = _make_registry({"0xaaa", "0xbbb"})
    data = MagicMock()

    async def _get_positions(address: str, **_kwargs: Any) -> list[Position]:
        if address == "0xaaa":
            return [_make_position("A1", address=address), _make_position("A2", address=address)]
        return [_make_position("A3", address=address), _make_position("A4", address=address)]

    data.get_positions = AsyncMock(side_effect=_get_positions)
    ws = _make_ws()
    collector, _repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        ws=ws,
    )

    await collector._refresh_subscriptions()

    subscribed = {aid for call in ws.subscribe.await_args_list for aid in call.args[0]}
    assert subscribed == {"A1", "A2", "A3", "A4"}
    assert collector._subscribed_assets == {"A1", "A2", "A3", "A4"}


@pytest.mark.asyncio
async def test_refresh_subscriptions_batches_in_chunks_of_100(
    tmp_db: sqlite3.Connection,
) -> None:
    """Asset ids are sent in batches of at most 100 per ``ws.subscribe`` call."""
    registry = _make_registry({_WALLET})
    data = MagicMock()
    positions = [_make_position(f"asset-{i}") for i in range(250)]
    data.get_positions = AsyncMock(return_value=positions)
    ws = _make_ws()
    collector, _repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        ws=ws,
    )

    await collector._refresh_subscriptions()

    batch_sizes = [len(call.args[0]) for call in ws.subscribe.await_args_list]
    assert batch_sizes == [100, 100, 50]
    flat = [aid for call in ws.subscribe.await_args_list for aid in call.args[0]]
    assert sorted(flat) == sorted({f"asset-{i}" for i in range(250)})


@pytest.mark.asyncio
async def test_refresh_subscriptions_caps_huge_wallet_at_500(
    tmp_db: sqlite3.Connection,
) -> None:
    """A wallet with more than 500 positions is sliced to the cap (no crash)."""
    registry = _make_registry({_WALLET})
    data = MagicMock()
    positions = [_make_position(f"asset-{i}") for i in range(600)]
    data.get_positions = AsyncMock(return_value=positions)
    ws = _make_ws()
    collector, _repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        ws=ws,
    )

    await collector._refresh_subscriptions()

    assert len(collector._subscribed_assets) == 500
    flat = {aid for call in ws.subscribe.await_args_list for aid in call.args[0]}
    assert flat == {f"asset-{i}" for i in range(500)}


@pytest.mark.asyncio
async def test_on_watchlist_add_subscribes_immediately(
    tmp_db: sqlite3.Connection,
) -> None:
    """A new watchlist entry triggers an off-cycle subscribe for its assets."""
    registry = _make_registry(set())
    data = MagicMock()
    new_addr = "0xnew"
    data.get_positions = AsyncMock(
        return_value=[
            _make_position("N1", address=new_addr),
            _make_position("N2", address=new_addr),
        ],
    )
    ws = _make_ws()
    collector, _repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        ws=ws,
    )

    collector._on_watchlist_add(new_addr)
    # Yield until the spawned task has finished.
    for _ in range(5):
        await asyncio.sleep(0)

    assert not collector._pending_add_tasks
    flat = {aid for call in ws.subscribe.await_args_list for aid in call.args[0]}
    assert flat == {"N1", "N2"}
    data.get_positions.assert_awaited_once_with(new_addr)


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` returns within a few seconds once ``stop_event`` is set."""
    registry = _make_registry({_WALLET})
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=[])
    ws = _make_ws()

    async def _messages() -> AsyncIterator[Any]:
        # Idle forever — the consume task should be cancelled on stop.
        await asyncio.sleep(60)
        if False:  # pragma: no cover  -- generator typing helper
            yield None

    ws.messages = _messages
    collector, _repo, _reg, _dc, _ws = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
        ws=ws,
        refresh_seconds=60.0,
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.05)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=2.0,
    )

    ws.connect.assert_awaited_once()
    ws.close.assert_awaited_once()
    registry.subscribe.assert_called_once_with(collector._on_watchlist_add)


@pytest.mark.asyncio
async def test_consume_loop_continues_after_record_trade_exception(
    tmp_db: sqlite3.Connection,
) -> None:
    """If the repo raises on insert, the consume loop logs and keeps going."""
    registry = _make_registry({_WALLET})
    failing_repo = MagicMock()
    failing_repo.insert = MagicMock(side_effect=[RuntimeError("boom"), True])
    ws = _make_ws()
    first = _make_trade(transaction_hash="0xtxFail", asset_id="A1")
    second = _make_trade(transaction_hash="0xtxOk", asset_id="A2")

    async def _messages() -> AsyncIterator[Any]:
        yield first
        yield second

    ws.messages = _messages

    collector = TradeCollector(
        registry=registry,  # type: ignore[arg-type]
        data_client=MagicMock(),  # type: ignore[arg-type]
        ws=ws,  # type: ignore[arg-type]
        trades_repo=failing_repo,  # type: ignore[arg-type]
    )

    await collector._consume_loop(asyncio.Event())

    assert failing_repo.insert.call_count == 2
