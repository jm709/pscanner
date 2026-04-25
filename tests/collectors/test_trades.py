"""Tests for ``TradeCollector`` (REST polling against ``/activity``).

The watchlist registry is mocked; the data client is mocked; the
:class:`WalletTradesRepo` runs against an in-memory SQLite to verify
end-to-end persistence and dedupe behaviour.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.trades import TradeCollector
from pscanner.store.repo import WalletTrade, WalletTradesRepo

_NOW = 1_700_000_000
_WALLET = "0xwatched"
_OTHER_WALLET = "0xunwatched"
_ASSET_ID = "asset-1"
_CONDITION_ID = "cond-1"


def _make_activity_event(**overrides: Any) -> dict[str, Any]:
    """Build one ``TRADE`` activity dict shaped like the real API response."""
    base: dict[str, Any] = {
        "type": "TRADE",
        "transactionHash": "0xtx1",
        "asset": _ASSET_ID,
        "side": "BUY",
        "size": 100.0,
        "price": 0.4,
        "conditionId": _CONDITION_ID,
        "timestamp": _NOW,
        "usdcSize": 40.0,
    }
    base.update(overrides)
    return base


def _make_registry(addresses: set[str] | None = None) -> MagicMock:
    """Build a ``WatchlistRegistry`` mock with realistic membership semantics."""
    addrs = set(addresses or set())
    registry = MagicMock()
    registry.__contains__ = MagicMock(side_effect=lambda addr: addr in addrs)
    registry.addresses = MagicMock(side_effect=lambda: set(addrs))
    registry.subscribe = MagicMock()
    registry._addrs = addrs
    return registry


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    registry: MagicMock | None = None,
    data_client: MagicMock | None = None,
    poll_interval_seconds: float = 60.0,
) -> tuple[TradeCollector, WalletTradesRepo, MagicMock, MagicMock]:
    """Wire up a ``TradeCollector`` with default mocks plus a real repo."""
    repo = WalletTradesRepo(tmp_db)
    reg = registry or _make_registry({_WALLET})
    dc = data_client or MagicMock()
    if not hasattr(dc, "get_activity") or not isinstance(dc.get_activity, AsyncMock):
        dc.get_activity = AsyncMock(return_value=[])
    collector = TradeCollector(
        registry=reg,  # type: ignore[arg-type]
        data_client=dc,  # type: ignore[arg-type]
        trades_repo=repo,
        poll_interval_seconds=poll_interval_seconds,
    )
    return collector, repo, reg, dc


@pytest.mark.asyncio
async def test_poll_all_wallets_inserts_three_trades(tmp_db: sqlite3.Connection) -> None:
    """A single watched wallet with 3 TRADE events yields 3 rows."""
    events = [
        _make_activity_event(transactionHash="0xa", size=10.0, price=0.5, usdcSize=5.0),
        _make_activity_event(transactionHash="0xb", size=20.0, price=0.6, usdcSize=12.0),
        _make_activity_event(transactionHash="0xc", size=30.0, price=0.7, usdcSize=21.0),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 3
    rows = repo.recent_for_wallet(_WALLET)
    assert len(rows) == 3
    hashes = {row.transaction_hash for row in rows}
    assert hashes == {"0xa", "0xb", "0xc"}
    assert all(row.source == "activity_api" for row in rows)
    assert all(row.status == "CONFIRMED" for row in rows)
    assert all(row.wallet == _WALLET for row in rows)


@pytest.mark.asyncio
async def test_poll_all_wallets_dedupes_on_repeat_calls(
    tmp_db: sqlite3.Connection,
) -> None:
    """Polling the same wallet twice with overlapping events does not double-count."""
    events = [
        _make_activity_event(transactionHash="0xa"),
        _make_activity_event(transactionHash="0xb"),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    first = await collector.poll_all_wallets()
    second = await collector.poll_all_wallets()

    assert first == 2
    assert second == 0
    assert len(repo.recent_for_wallet(_WALLET)) == 2


@pytest.mark.asyncio
async def test_poll_skips_event_with_empty_transaction_hash(
    tmp_db: sqlite3.Connection,
) -> None:
    """An activity entry without a transaction hash is dropped, not persisted."""
    events = [
        _make_activity_event(transactionHash=""),
        _make_activity_event(transactionHash="0xok"),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    rows = repo.recent_for_wallet(_WALLET)
    assert {row.transaction_hash for row in rows} == {"0xok"}


@pytest.mark.asyncio
async def test_poll_only_fetches_for_watched_wallets(
    tmp_db: sqlite3.Connection,
) -> None:
    """The collector calls ``get_activity`` only for addresses in the registry."""
    registry = _make_registry({_WALLET})
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=[])
    collector, _repo, _reg, dc = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
    )

    await collector.poll_all_wallets()

    called_wallets = {call.args[0] for call in dc.get_activity.await_args_list}
    assert called_wallets == {_WALLET}
    assert _OTHER_WALLET not in called_wallets


@pytest.mark.asyncio
async def test_on_watchlist_add_polls_immediately(tmp_db: sqlite3.Connection) -> None:
    """A new watchlist entry triggers an off-cycle poll for that wallet."""
    new_addr = "0xnew"
    events = [_make_activity_event(transactionHash="0xnew1")]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, repo, _reg, _dc = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry(set()),
        data_client=data,
    )

    collector._on_watchlist_add(new_addr)
    for _ in range(10):
        await asyncio.sleep(0)

    assert not collector._pending_add_tasks
    rows = repo.recent_for_wallet(new_addr)
    assert len(rows) == 1
    assert rows[0].transaction_hash == "0xnew1"
    data.get_activity.assert_awaited()


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` returns within the poll interval after ``stop_event`` is set."""
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=[])
    collector, _repo, registry, _dc = _make_collector(
        tmp_db=tmp_db,
        data_client=data,
        poll_interval_seconds=60.0,
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.05)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=2.0,
    )

    registry.subscribe.assert_called_once_with(collector._on_watchlist_add)


@pytest.mark.asyncio
async def test_per_wallet_exception_does_not_crash_loop(
    tmp_db: sqlite3.Connection,
) -> None:
    """A failing ``get_activity`` for one wallet does not stop the next."""
    registry = _make_registry({"0xfail", "0xok"})
    data = MagicMock()

    async def _get_activity(address: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if address == "0xfail":
            msg = "boom"
            raise RuntimeError(msg)
        return [_make_activity_event(transactionHash="0xok1")]

    data.get_activity = AsyncMock(side_effect=_get_activity)
    collector, repo, _reg, _dc = _make_collector(
        tmp_db=tmp_db,
        registry=registry,
        data_client=data,
    )

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    assert len(repo.recent_for_wallet("0xok")) == 1
    assert repo.recent_for_wallet("0xfail") == []


@pytest.mark.asyncio
async def test_source_field_is_activity_api(tmp_db: sqlite3.Connection) -> None:
    """Every persisted row has ``source='activity_api'`` (REST provenance)."""
    events = [_make_activity_event(transactionHash=f"0x{i}") for i in range(3)]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    await collector.poll_all_wallets()

    rows = repo.recent_for_wallet(_WALLET)
    assert {row.source for row in rows} == {"activity_api"}


@pytest.mark.asyncio
async def test_non_trade_activity_entries_are_dropped(
    tmp_db: sqlite3.Connection,
) -> None:
    """Activity entries with ``type != 'TRADE'`` are skipped defensively."""
    events = [
        {"type": "REWARD", "transactionHash": "0xreward", "asset": "x"},
        _make_activity_event(transactionHash="0xtrade"),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    rows = repo.recent_for_wallet(_WALLET)
    assert {row.transaction_hash for row in rows} == {"0xtrade"}


@pytest.mark.asyncio
async def test_subscribe_new_trade_fires_on_insert(tmp_db: sqlite3.Connection) -> None:
    """A registered callback is invoked once per newly-inserted trade."""
    events = [
        _make_activity_event(transactionHash="0xa"),
        _make_activity_event(transactionHash="0xb"),
    ]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, _repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    received: list[WalletTrade] = []
    collector.subscribe_new_trade(received.append)

    inserted = await collector.poll_all_wallets()

    assert inserted == 2
    assert len(received) == 2
    assert {trade.transaction_hash for trade in received} == {"0xa", "0xb"}
    assert all(trade.wallet == _WALLET for trade in received)


@pytest.mark.asyncio
async def test_subscribe_new_trade_skips_duplicates(tmp_db: sqlite3.Connection) -> None:
    """Callbacks fire only on real inserts; PK collisions are silent."""
    events = [_make_activity_event(transactionHash="0xa")]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, _repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    received: list[WalletTrade] = []
    collector.subscribe_new_trade(received.append)

    first = await collector.poll_all_wallets()
    second = await collector.poll_all_wallets()

    assert first == 1
    assert second == 0
    # First poll fired once; the second poll's duplicate must NOT re-fire.
    assert len(received) == 1
    assert received[0].transaction_hash == "0xa"


@pytest.mark.asyncio
async def test_subscribe_new_trade_isolates_callback_failures(
    tmp_db: sqlite3.Connection,
) -> None:
    """One callback raising must not prevent others from firing."""
    events = [_make_activity_event(transactionHash="0xa")]
    data = MagicMock()
    data.get_activity = AsyncMock(return_value=events)
    collector, _repo, _reg, _dc = _make_collector(tmp_db=tmp_db, data_client=data)

    received: list[WalletTrade] = []

    def _broken(_trade: WalletTrade) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    collector.subscribe_new_trade(_broken)
    collector.subscribe_new_trade(received.append)

    inserted = await collector.poll_all_wallets()

    assert inserted == 1
    assert len(received) == 1
    assert received[0].transaction_hash == "0xa"
