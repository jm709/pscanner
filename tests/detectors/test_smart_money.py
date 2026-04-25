"""Behaviour tests for the smart-money detector.

The detector's collaborators (``DataClient``, ``TrackedWalletsRepo``,
``PositionSnapshotsRepo``, ``AlertSink``) are all mocked: this test module
exercises only the orchestration logic in
:class:`pscanner.detectors.smart_money.SmartMoneyDetector`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import SmartMoneyConfig
from pscanner.detectors.base import Detector
from pscanner.detectors.smart_money import SmartMoneyDetector
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position
from pscanner.store.repo import TrackedWallet


def _config(**overrides: Any) -> SmartMoneyConfig:
    """Build a config with sane defaults plus overrides."""
    base: dict[str, Any] = {
        "leaderboard_top_n": 10,
        "min_resolved_positions": 5,
        "min_winrate": 0.6,
        "refresh_interval_seconds": 60,
        "position_poll_interval_seconds": 30,
        "new_position_min_usd": 1000.0,
    }
    base.update(overrides)
    return SmartMoneyConfig(**base)


def _leaderboard_entry(address: str = "0xabc", pnl: float = 1234.5) -> LeaderboardEntry:
    """Build a leaderboard entry."""
    return LeaderboardEntry.model_validate(
        {
            "proxyWallet": address,
            "amount": pnl,
            "period": "all",
        },
    )


def _closed_position(*, won: bool) -> ClosedPosition:
    """Build a closed position with a known win/loss outcome."""
    return ClosedPosition.model_validate(
        {
            "proxyWallet": "0xabc",
            "asset": "tok",
            "conditionId": "0xcond",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "avgPrice": 0.5,
            "realizedPnl": 10.0 if won else -5.0,
        },
    )


def _open_position(
    *,
    address: str = "0xabc",
    condition_id: str = "0xcond",
    outcome: str = "Yes",
    size: float = 10000.0,
    avg_price: float = 0.5,
    title: str = "Will it rain?",
) -> Position:
    """Build an open position payload."""
    return Position.model_validate(
        {
            "proxyWallet": address,
            "asset": "tok",
            "conditionId": condition_id,
            "outcome": outcome,
            "outcomeIndex": 0,
            "size": size,
            "avgPrice": avg_price,
            "title": title,
        },
    )


def _tracked_wallet(address: str = "0xabc") -> TrackedWallet:
    """Build a tracked wallet."""
    return TrackedWallet(
        address=address,
        closed_position_count=10,
        closed_position_wins=8,
        winrate=0.8,
        leaderboard_pnl=500.0,
        last_refreshed_at=1,
    )


def _build_detector(
    *,
    config: SmartMoneyConfig | None = None,
    data_client: AsyncMock | None = None,
    tracked_repo: MagicMock | None = None,
    snapshots_repo: MagicMock | None = None,
) -> tuple[SmartMoneyDetector, AsyncMock, MagicMock, MagicMock]:
    """Wire a detector with mocked collaborators and return all four handles."""
    cfg = config or _config()
    data = data_client or AsyncMock()
    tracked = tracked_repo or MagicMock()
    snapshots = snapshots_repo or MagicMock()
    detector = SmartMoneyDetector(
        config=cfg,
        data_client=data,
        tracked_repo=tracked,
        snapshots_repo=snapshots,
    )
    return detector, data, tracked, snapshots


@pytest.mark.asyncio
async def test_refresh_upserts_qualifying_wallet() -> None:
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_closed_positions.return_value = [
        _closed_position(won=True),
        _closed_position(won=True),
        _closed_position(won=True),
        _closed_position(won=True),
        _closed_position(won=False),
    ]  # 4/5 = 0.8 winrate, 5 resolved (>= min 5)
    detector, _, tracked, _ = _build_detector(data_client=data_client)

    await detector._refresh_tracked_wallets()

    tracked.upsert.assert_called_once()
    kwargs = tracked.upsert.call_args.kwargs
    assert kwargs["address"] == "0xabc"
    assert kwargs["closed_position_count"] == 5
    assert kwargs["closed_position_wins"] == 4
    assert kwargs["winrate"] == pytest.approx(0.8)
    assert kwargs["leaderboard_pnl"] == 1234.5


@pytest.mark.asyncio
async def test_refresh_skips_wallet_below_count_threshold() -> None:
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_closed_positions.return_value = [
        _closed_position(won=True),
        _closed_position(won=True),
    ]  # Only 2 resolved (< min 5)
    detector, _, tracked, _ = _build_detector(data_client=data_client)

    await detector._refresh_tracked_wallets()

    tracked.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_skips_wallet_below_winrate_threshold() -> None:
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_closed_positions.return_value = [
        _closed_position(won=True),
        _closed_position(won=False),
        _closed_position(won=False),
        _closed_position(won=False),
        _closed_position(won=False),
    ]  # 1/5 = 0.2 (< min 0.6)
    detector, _, tracked, _ = _build_detector(data_client=data_client)

    await detector._refresh_tracked_wallets()

    tracked.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_handles_empty_leaderboard() -> None:
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = []
    detector, _, tracked, _ = _build_detector(data_client=data_client)

    await detector._refresh_tracked_wallets()

    tracked.upsert.assert_not_called()
    data_client.get_closed_positions.assert_not_called()


@pytest.mark.asyncio
async def test_poll_emits_alert_for_new_position(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pscanner.detectors.smart_money.time.time", lambda: 1_700_000_000)
    monkeypatch.setattr(
        "pscanner.detectors.smart_money.time.strftime",
        lambda fmt, _t=None: "20231114",
    )
    data_client = AsyncMock()
    data_client.get_positions.return_value = [_open_position(size=5000.0, avg_price=0.4)]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = None  # Never seen before.
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    sink.emit.assert_awaited_once()
    alert = sink.emit.await_args.args[0]
    assert isinstance(alert, Alert)
    assert alert.detector == "smart_money"
    assert alert.alert_key == "smart:0xabc:0xcond:yes:20231114"
    assert alert.body["new_size"] == 5000.0
    assert alert.body["prev_size"] == 0.0
    assert alert.body["winrate"] == 0.8
    assert alert.body["closed_position_count"] == 10
    snapshots_repo.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_poll_no_alert_when_size_unchanged() -> None:
    data_client = AsyncMock()
    data_client.get_positions.return_value = [_open_position(size=5000.0, avg_price=0.4)]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = 5000.0  # Same as current.
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    sink.emit.assert_not_called()
    snapshots_repo.upsert.assert_called_once()  # Still refreshes the snapshot.


@pytest.mark.asyncio
async def test_poll_no_alert_when_delta_below_threshold() -> None:
    # prev=5000 shares, new=5100 shares, avg_price=0.4 -> delta_usd = 100*0.4 = 40 USD.
    data_client = AsyncMock()
    data_client.get_positions.return_value = [_open_position(size=5100.0, avg_price=0.4)]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = 5000.0
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    sink.emit.assert_not_called()
    snapshots_repo.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_poll_emits_alert_when_delta_meets_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pscanner.detectors.smart_money.time.time", lambda: 1_700_000_000)
    monkeypatch.setattr(
        "pscanner.detectors.smart_money.time.strftime",
        lambda fmt, _t=None: "20231114",
    )
    # prev=5000 -> new=10000 at price 0.4 -> delta_usd = 5000*0.4 = 2000 (>= 1000 threshold).
    data_client = AsyncMock()
    data_client.get_positions.return_value = [_open_position(size=10000.0, avg_price=0.4)]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = 5000.0
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    sink.emit.assert_awaited_once()
    alert = sink.emit.await_args.args[0]
    assert alert.body["delta_usd"] == pytest.approx(2000.0)
    assert alert.body["prev_size"] == 5000.0


@pytest.mark.asyncio
async def test_poll_handles_no_tracked_wallets() -> None:
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = []
    sink = AsyncMock()
    data_client = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
    )

    await detector._poll_positions(sink)

    sink.emit.assert_not_called()
    data_client.get_positions.assert_not_called()


@pytest.mark.asyncio
async def test_poll_handles_empty_positions() -> None:
    data_client = AsyncMock()
    data_client.get_positions.return_value = []
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    sink.emit.assert_not_called()
    snapshots_repo.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_poll_continues_after_get_positions_error() -> None:
    """If one wallet's fetch raises, the detector logs and moves on."""
    wallet_a = _tracked_wallet("0xaaa")
    wallet_b = _tracked_wallet("0xbbb")
    data_client = AsyncMock()
    data_client.get_positions.side_effect = [
        RuntimeError("boom"),
        [_open_position(address="0xbbb", size=5000.0, avg_price=0.4)],
    ]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [wallet_a, wallet_b]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = None
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    # Wallet B still produced an alert (prev was None).
    sink.emit.assert_awaited_once()
    assert data_client.get_positions.await_count == 2


@pytest.mark.asyncio
async def test_alert_key_uses_lowercased_outcome_as_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pscanner.detectors.smart_money.time.time", lambda: 1_700_000_000)
    monkeypatch.setattr(
        "pscanner.detectors.smart_money.time.strftime",
        lambda fmt, _t=None: "20231114",
    )
    data_client = AsyncMock()
    data_client.get_positions.return_value = [
        _open_position(outcome="NO", size=5000.0, avg_price=0.4),
    ]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = None
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector._poll_positions(sink)

    alert = sink.emit.await_args.args[0]
    assert alert.alert_key.endswith(":no:20231114")
    assert alert.body["side"] == "no"


def test_detector_implements_protocol() -> None:
    detector, _, _, _ = _build_detector()
    assert isinstance(detector, Detector)
    assert detector.name == "smart_money"
