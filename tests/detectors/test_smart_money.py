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
from pscanner.detectors.smart_money import (
    SmartMoneyDetector,
    _compute_metrics,
    _severity,
)
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position
from pscanner.store.repo import TrackedWallet


def _config(**overrides: Any) -> SmartMoneyConfig:
    """Build a config with sane defaults plus overrides."""
    base: dict[str, Any] = {
        "leaderboard_top_n": 10,
        "min_resolved_positions": 5,
        "min_edge": 0.05,
        "min_excess_pnl_usd": 1000.0,
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


def _closed_position(
    *,
    won: bool,
    avg_price: float = 0.5,
    size: float = 100.0,
    realized_pnl: float | None = None,
) -> ClosedPosition:
    """Build a closed position with a known win/loss outcome."""
    if realized_pnl is None:
        realized_pnl = 1000.0 if won else -100.0
    return ClosedPosition.model_validate(
        {
            "proxyWallet": "0xabc",
            "asset": "tok",
            "conditionId": "0xcond",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "size": size,
            "avgPrice": avg_price,
            "realizedPnl": realized_pnl,
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


def _tracked_wallet(
    address: str = "0xabc",
    *,
    mean_edge: float | None = 0.15,
    excess_pnl_usd: float | None = 5000.0,
) -> TrackedWallet:
    """Build a tracked wallet."""
    return TrackedWallet(
        address=address,
        closed_position_count=10,
        closed_position_wins=8,
        winrate=0.8,
        leaderboard_pnl=500.0,
        last_refreshed_at=1,
        mean_edge=mean_edge,
        weighted_edge=mean_edge,
        excess_pnl_usd=excess_pnl_usd,
        total_stake_usd=20000.0,
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


def test_compute_metrics_mean_edge_arithmetic() -> None:
    """3 positions: avg_price 0.5/0.5/0.5, won True/True/False -> mean_edge ≈ 0.167."""
    closed = [
        _closed_position(won=True, avg_price=0.5, size=100.0, realized_pnl=50.0),
        _closed_position(won=True, avg_price=0.5, size=100.0, realized_pnl=50.0),
        _closed_position(won=False, avg_price=0.5, size=100.0, realized_pnl=-50.0),
    ]
    metrics = _compute_metrics(closed)
    assert metrics.count == 3
    assert metrics.wins == 2
    assert metrics.winrate == pytest.approx(2 / 3)
    assert metrics.mean_edge == pytest.approx(1 / 6)  # 0.5/3
    assert metrics.weighted_edge == pytest.approx(1 / 6)
    assert metrics.excess_pnl_usd == pytest.approx(50.0)
    assert metrics.total_stake_usd == pytest.approx(150.0)


def test_compute_metrics_skips_degenerate_prices() -> None:
    """Positions with avg_price <= 0 or >= 1 are excluded entirely."""
    closed = [
        _closed_position(won=True, avg_price=0.0, size=100.0, realized_pnl=10.0),
        _closed_position(won=True, avg_price=1.0, size=100.0, realized_pnl=10.0),
        _closed_position(won=True, avg_price=0.4, size=100.0, realized_pnl=60.0),
    ]
    metrics = _compute_metrics(closed)
    assert metrics.count == 1
    assert metrics.mean_edge == pytest.approx(0.6)  # 1.0 - 0.4
    assert metrics.excess_pnl_usd == pytest.approx(60.0)


def test_compute_metrics_empty_returns_zeroed() -> None:
    metrics = _compute_metrics([])
    assert metrics.count == 0
    assert metrics.wins == 0
    assert metrics.winrate == 0.0
    assert metrics.mean_edge == 0.0
    assert metrics.weighted_edge == 0.0
    assert metrics.excess_pnl_usd == 0.0
    assert metrics.total_stake_usd == 0.0


def test_compute_metrics_weighted_edge_differs_with_uneven_stakes() -> None:
    """Bigger losers should drag the weighted edge below the mean edge."""
    closed = [
        # Big losing stake: edge = -0.4, stake = 100*0.4 = 40
        _closed_position(won=False, avg_price=0.4, size=100.0, realized_pnl=-40.0),
        # Small winning stake: edge = +0.5, stake = 1*0.5 = 0.5
        _closed_position(won=True, avg_price=0.5, size=1.0, realized_pnl=0.5),
    ]
    metrics = _compute_metrics(closed)
    assert metrics.mean_edge == pytest.approx((-0.4 + 0.5) / 2)
    expected_weighted = (-0.4 * 40.0 + 0.5 * 0.5) / (40.0 + 0.5)
    assert metrics.weighted_edge == pytest.approx(expected_weighted)


def test_severity_high_med_low_boundaries() -> None:
    # delta_usd=20000, edge=0.10 -> score = 2.0 * 0.10 = 0.20 -> med
    # Use boundary for high: 50000 * 0.10 = 0.5 -> high
    assert _severity(50_000.0, 0.10) == "high"
    # delta_usd=5000, edge=0.05 -> score = 0.5 * 0.05 = 0.025 -> low
    # boundary med: 20000 * 0.05 = 0.1 -> med
    assert _severity(20_000.0, 0.05) == "med"
    assert _severity(1_000.0, 0.01) == "low"


def test_severity_handles_none_and_negative_edge() -> None:
    assert _severity(100_000.0, None) == "low"
    assert _severity(100_000.0, -0.1) == "low"


@pytest.mark.asyncio
async def test_refresh_upserts_qualifying_wallet() -> None:
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    # 5 resolved positions, 4 winners, avg_price=0.5 each, size=100, realized_pnl=±50.
    data_client.get_closed_positions.return_value = [
        _closed_position(won=True, realized_pnl=500.0),
        _closed_position(won=True, realized_pnl=500.0),
        _closed_position(won=True, realized_pnl=500.0),
        _closed_position(won=True, realized_pnl=500.0),
        _closed_position(won=False, realized_pnl=-50.0),
    ]
    detector, _, tracked, _ = _build_detector(data_client=data_client)

    await detector._refresh_tracked_wallets()

    tracked.upsert.assert_called_once()
    kwargs = tracked.upsert.call_args.kwargs
    assert kwargs["address"] == "0xabc"
    assert kwargs["closed_position_count"] == 5
    assert kwargs["closed_position_wins"] == 4
    assert kwargs["winrate"] == pytest.approx(0.8)
    assert kwargs["leaderboard_pnl"] == 1234.5
    assert kwargs["mean_edge"] == pytest.approx(0.3)  # (4*0.5 + (-0.5))/5
    assert kwargs["weighted_edge"] == pytest.approx(0.3)
    assert kwargs["excess_pnl_usd"] == pytest.approx(1950.0)
    assert kwargs["total_stake_usd"] == pytest.approx(250.0)  # 5*100*0.5


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
async def test_refresh_skips_wallet_below_edge_threshold() -> None:
    """Wallet with sufficient size but mean_edge < min_edge is skipped."""
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    # avg_price=0.95 won True -> edge=0.05; avg_price=0.95 won False -> edge=-0.95.
    # Construct mean_edge ≈ 0.0 with plenty of resolved positions.
    data_client.get_closed_positions.return_value = [
        _closed_position(won=True, avg_price=0.5, realized_pnl=50.0),
        _closed_position(won=False, avg_price=0.5, realized_pnl=-50.0),
        _closed_position(won=True, avg_price=0.5, realized_pnl=50.0),
        _closed_position(won=False, avg_price=0.5, realized_pnl=-50.0),
        _closed_position(won=True, avg_price=0.5, realized_pnl=50.0),
        _closed_position(won=False, avg_price=0.5, realized_pnl=-50.0),
    ]  # 3/6 wins -> mean_edge = 0 (< 0.05)
    detector, _, tracked, _ = _build_detector(
        data_client=data_client,
        config=_config(min_edge=0.05, min_excess_pnl_usd=0.0),
    )

    await detector._refresh_tracked_wallets()

    tracked.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_skips_wallet_below_excess_pnl_threshold() -> None:
    """Wallet with high edge but low excess_pnl_usd is skipped."""
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    # 5 winners at avg_price=0.4, realized_pnl=10 each -> excess_pnl = 50 (< 1000).
    data_client.get_closed_positions.return_value = [
        _closed_position(won=True, avg_price=0.4, size=10.0, realized_pnl=10.0),
        _closed_position(won=True, avg_price=0.4, size=10.0, realized_pnl=10.0),
        _closed_position(won=True, avg_price=0.4, size=10.0, realized_pnl=10.0),
        _closed_position(won=True, avg_price=0.4, size=10.0, realized_pnl=10.0),
        _closed_position(won=True, avg_price=0.4, size=10.0, realized_pnl=10.0),
    ]
    detector, _, tracked, _ = _build_detector(
        data_client=data_client,
        config=_config(min_edge=0.05, min_excess_pnl_usd=1000.0),
    )

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
async def test_poll_uses_edge_kwargs_for_list_active() -> None:
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = []
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(tracked_repo=tracked_repo)

    await detector.poll_positions(sink)

    tracked_repo.list_active.assert_called_once_with(
        min_edge=0.05,
        min_excess_pnl_usd=1000.0,
        min_resolved=5,
    )


@pytest.mark.asyncio
async def test_poll_first_observation_silently_upserts_no_alert() -> None:
    """Cold-start bootstrap: first time we see a position, snapshot only."""
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

    await detector.poll_positions(sink)

    sink.emit.assert_not_called()
    snapshots_repo.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_poll_emits_alert_when_growth_meets_threshold_after_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second observation: prev is set, delta_usd >= threshold -> emit."""
    monkeypatch.setattr("pscanner.detectors.smart_money.time.time", lambda: 1_700_000_000)
    monkeypatch.setattr(
        "pscanner.detectors.smart_money.time.strftime",
        lambda fmt, _t=None: "20231114",
    )
    # prev=2500 shares, new=5000 shares at avg_price=0.4 -> delta_usd = 2500*0.4 = 1000.
    data_client = AsyncMock()
    data_client.get_positions.return_value = [_open_position(size=5000.0, avg_price=0.4)]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = 2500.0
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector.poll_positions(sink)

    sink.emit.assert_awaited_once()
    alert = sink.emit.await_args.args[0]
    assert isinstance(alert, Alert)
    assert alert.detector == "smart_money"
    assert alert.alert_key == "smart:0xabc:0xcond:yes:20231114"
    assert alert.body["new_size"] == 5000.0
    assert alert.body["prev_size"] == 2500.0
    assert alert.body["delta_usd"] == pytest.approx(1000.0)
    assert alert.body["mean_edge"] == 0.15
    snapshots_repo.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_poll_alert_severity_grades_by_size_and_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """delta_usd=20000 + mean_edge=0.10 -> score 0.20 -> med."""
    monkeypatch.setattr("pscanner.detectors.smart_money.time.time", lambda: 1_700_000_000)
    monkeypatch.setattr(
        "pscanner.detectors.smart_money.time.strftime",
        lambda fmt, _t=None: "20231114",
    )
    data_client = AsyncMock()
    # prev=0 -> new=50000 at price=0.4 -> delta_usd = 20000.
    data_client.get_positions.return_value = [_open_position(size=50000.0, avg_price=0.4)]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet(mean_edge=0.10)]
    snapshots_repo = MagicMock()
    snapshots_repo.previous_size.return_value = 0.0
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector.poll_positions(sink)

    alert = sink.emit.await_args.args[0]
    assert alert.severity == "med"


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

    await detector.poll_positions(sink)

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

    await detector.poll_positions(sink)

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
    # prev=5000 -> new=10000 at price 0.4 -> delta_usd = 5000*0.4 = 2000.
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

    await detector.poll_positions(sink)

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

    await detector.poll_positions(sink)

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

    await detector.poll_positions(sink)

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
        [_open_position(address="0xbbb", size=10000.0, avg_price=0.4)],
    ]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [wallet_a, wallet_b]
    snapshots_repo = MagicMock()
    # Prev=2500 -> new=10000 at 0.4 -> delta_usd = 7500*0.4 = 3000 (>= 1000) -> emit.
    snapshots_repo.previous_size.return_value = 2500.0
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector.poll_positions(sink)

    # Wallet B still produced an alert (delta_usd cleared the threshold).
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
        _open_position(outcome="NO", size=10000.0, avg_price=0.4),
    ]
    tracked_repo = MagicMock()
    tracked_repo.list_active.return_value = [_tracked_wallet()]
    snapshots_repo = MagicMock()
    # Prev=5000 -> new=10000 at 0.4 -> delta_usd = 5000*0.4 = 2000.
    snapshots_repo.previous_size.return_value = 5000.0
    sink = AsyncMock()
    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        snapshots_repo=snapshots_repo,
    )

    await detector.poll_positions(sink)

    alert = sink.emit.await_args.args[0]
    assert alert.alert_key.endswith(":no:20231114")
    assert alert.body["side"] == "no"


def test_detector_implements_protocol() -> None:
    detector, _, _, _ = _build_detector()
    assert isinstance(detector, Detector)
    assert detector.name == "smart_money"
