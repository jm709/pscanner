"""Tests for ``PositionCollector`` (REST polling against ``/positions``).

The watchlist registry is mocked; the data client is mocked; the
:class:`WalletPositionsHistoryRepo` runs against an in-memory SQLite to
verify end-to-end persistence and append-only semantics.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.collectors.positions import PositionCollector
from pscanner.poly.models import Position
from pscanner.store.repo import WalletPositionsHistoryRepo

_WALLET_A = "0xaaaa"
_WALLET_B = "0xbbbb"
_WALLET_C = "0xcccc"


def _make_position(**overrides: Any) -> Position:
    """Build a realistic ``Position`` from a Polymarket-shaped dict."""
    base: dict[str, Any] = {
        "proxyWallet": _WALLET_A,
        "asset": "asset-1",
        "conditionId": "cond-1",
        "outcome": "Yes",
        "outcomeIndex": 0,
        "size": 100.0,
        "avgPrice": 0.42,
        "currentValue": 50.0,
        "cashPnl": 8.0,
        "percentPnl": 19.0,
        "realizedPnl": 0.0,
        "redeemable": False,
        "mergeable": False,
    }
    base.update(overrides)
    return Position.model_validate(base)


def _make_registry(addresses: set[str]) -> MagicMock:
    """Build a ``WatchlistRegistry`` mock returning ``addresses()``."""
    registry = MagicMock()
    registry.addresses = MagicMock(side_effect=lambda: set(addresses))
    return registry


def _make_collector(
    *,
    tmp_db: sqlite3.Connection,
    registry: MagicMock,
    data_client: MagicMock,
    snapshot_interval_seconds: float = 300.0,
) -> tuple[PositionCollector, WalletPositionsHistoryRepo]:
    """Wire up a ``PositionCollector`` with mocks and a real repo."""
    repo = WalletPositionsHistoryRepo(tmp_db)
    collector = PositionCollector(
        registry=registry,  # type: ignore[arg-type]
        data_client=data_client,  # type: ignore[arg-type]
        positions_repo=repo,
        snapshot_interval_seconds=snapshot_interval_seconds,
    )
    return collector, repo


@pytest.mark.asyncio
async def test_snapshot_single_wallet_three_positions(
    tmp_db: sqlite3.Connection,
) -> None:
    """One watched wallet with 3 positions yields 3 rows with mapped fields."""
    positions = [
        _make_position(conditionId="cond-A", outcome="Yes", size=10.0, avgPrice=0.4),
        _make_position(
            conditionId="cond-B",
            outcome="No",
            size=25.5,
            avgPrice=0.6,
            currentValue=15.0,
            cashPnl=-1.5,
            realizedPnl=None,
            redeemable=True,
        ),
        _make_position(
            conditionId="cond-C",
            outcome="Yes",
            size=300.0,
            avgPrice=0.85,
            realizedPnl=12.34,
        ),
    ]
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=positions)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A}),
        data_client=data,
    )

    inserted = await collector.snapshot_all_wallets()

    assert inserted == 3
    rows = repo.recent_for_wallet(_WALLET_A)
    assert len(rows) == 3
    by_cond: dict[str, Any] = {str(row.condition_id): row for row in rows}
    row_b = by_cond["cond-B"]
    assert row_b.outcome == "No"
    assert row_b.size == 25.5
    assert row_b.avg_price == 0.6
    assert row_b.current_value == 15.0
    assert row_b.cash_pnl == -1.5
    assert row_b.realized_pnl is None
    assert row_b.redeemable is True
    assert by_cond["cond-C"].realized_pnl == 12.34
    assert all(row.wallet == _WALLET_A for row in rows)


@pytest.mark.asyncio
async def test_snapshot_multiple_wallets_sums_correctly(
    tmp_db: sqlite3.Connection,
) -> None:
    """Two wallets with different counts produce a correct total + per-wallet split."""
    pos_a = [
        _make_position(proxyWallet=_WALLET_A, conditionId="cond-A1", outcome="Yes"),
        _make_position(proxyWallet=_WALLET_A, conditionId="cond-A2", outcome="No"),
    ]
    pos_b = [
        _make_position(proxyWallet=_WALLET_B, conditionId="cond-B1", outcome="Yes"),
        _make_position(proxyWallet=_WALLET_B, conditionId="cond-B2", outcome="No"),
        _make_position(proxyWallet=_WALLET_B, conditionId="cond-B3", outcome="Yes"),
    ]

    async def _by_wallet(address: str, **_kwargs: Any) -> list[Position]:
        return {_WALLET_A: pos_a, _WALLET_B: pos_b}[address]

    data = MagicMock()
    data.get_positions = AsyncMock(side_effect=_by_wallet)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A, _WALLET_B}),
        data_client=data,
    )

    inserted = await collector.snapshot_all_wallets()

    assert inserted == 5
    counts = repo.count_by_wallet()
    assert counts == {_WALLET_A: 2, _WALLET_B: 3}


@pytest.mark.asyncio
async def test_two_consecutive_snapshots_produce_independent_rows(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling twice across distinct timestamps yields 2 rows per position."""
    positions = [
        _make_position(conditionId="cond-A", outcome="Yes"),
        _make_position(conditionId="cond-B", outcome="No"),
    ]
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=positions)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A}),
        data_client=data,
    )

    clock = {"now": 1_700_000_000}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    first = await collector.snapshot_all_wallets()
    clock["now"] += 60
    second = await collector.snapshot_all_wallets()

    assert first == 2
    assert second == 2
    rows = repo.recent_for_wallet(_WALLET_A)
    assert len(rows) == 4
    snapshot_times = {row.snapshot_at for row in rows}
    assert snapshot_times == {1_700_000_000, 1_700_000_060}


@pytest.mark.asyncio
async def test_per_wallet_exception_does_not_break_others(
    tmp_db: sqlite3.Connection,
) -> None:
    """A failing ``get_positions`` for one wallet does not stop the others."""
    pos_a = [_make_position(proxyWallet=_WALLET_A, conditionId="cond-A", outcome="Yes")]
    pos_c = [
        _make_position(proxyWallet=_WALLET_C, conditionId="cond-C1", outcome="Yes"),
        _make_position(proxyWallet=_WALLET_C, conditionId="cond-C2", outcome="No"),
    ]

    async def _by_wallet(address: str, **_kwargs: Any) -> list[Position]:
        if address == _WALLET_B:
            msg = "boom"
            raise RuntimeError(msg)
        return {_WALLET_A: pos_a, _WALLET_C: pos_c}[address]

    data = MagicMock()
    data.get_positions = AsyncMock(side_effect=_by_wallet)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A, _WALLET_B, _WALLET_C}),
        data_client=data,
    )

    inserted = await collector.snapshot_all_wallets()

    assert inserted == 3
    counts = repo.count_by_wallet()
    assert counts == {_WALLET_A: 1, _WALLET_C: 2}
    assert repo.recent_for_wallet(_WALLET_B) == []


@pytest.mark.asyncio
async def test_empty_positions_list_inserts_nothing(
    tmp_db: sqlite3.Connection,
) -> None:
    """A wallet returning no positions inserts 0 rows without raising."""
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=[])
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A}),
        data_client=data,
    )

    inserted = await collector.snapshot_all_wallets()

    assert inserted == 0
    assert repo.recent_for_wallet(_WALLET_A) == []


@pytest.mark.asyncio
async def test_skips_position_with_empty_condition_id(
    tmp_db: sqlite3.Connection,
) -> None:
    """A position with an empty ``conditionId`` is dropped defensively."""
    positions = [
        _make_position(conditionId="", outcome="Yes"),
        _make_position(conditionId="cond-ok", outcome="No"),
    ]
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=positions)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A}),
        data_client=data,
    )

    inserted = await collector.snapshot_all_wallets()

    assert inserted == 1
    rows = repo.recent_for_wallet(_WALLET_A)
    assert {row.condition_id for row in rows} == {"cond-ok"}


@pytest.mark.asyncio
async def test_run_exits_cleanly_when_stop_event_set(
    tmp_db: sqlite3.Connection,
) -> None:
    """``run`` returns within 1s after ``stop_event`` is set."""
    positions = [_make_position(conditionId="cond-A", outcome="Yes")]
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=positions)
    collector, repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A}),
        data_client=data,
        snapshot_interval_seconds=0.05,
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.1)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=1.0,
    )

    data.get_positions.assert_awaited()
    assert len(repo.recent_for_wallet(_WALLET_A)) >= 1


@pytest.mark.asyncio
async def test_per_iteration_exception_does_not_break_run(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised inside ``snapshot_all_wallets`` is swallowed by run()."""
    data = MagicMock()
    data.get_positions = AsyncMock(return_value=[])
    collector, _repo = _make_collector(
        tmp_db=tmp_db,
        registry=_make_registry({_WALLET_A}),
        data_client=data,
        snapshot_interval_seconds=0.05,
    )

    calls = {"n": 0}

    async def _flaky() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "first call boom"
            raise RuntimeError(msg)
        return 0

    monkeypatch.setattr(collector, "snapshot_all_wallets", _flaky)

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.25)
        stop_event.set()

    await asyncio.wait_for(
        asyncio.gather(collector.run(stop_event), _trigger_stop()),
        timeout=1.5,
    )

    assert calls["n"] >= 2
