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
    event_id: str | None = None,
    condition_id: str = "0xcond",
) -> ClosedPosition:
    """Build a closed position with a known win/loss outcome.

    ``event_id`` here is the *event slug* — closed-position payloads expose
    ``eventSlug`` rather than a numeric event id, and the tag cache is keyed
    on the same slug.
    """
    if realized_pnl is None:
        realized_pnl = 1000.0 if won else -100.0
    payload: dict[str, Any] = {
        "proxyWallet": "0xabc",
        "asset": "tok",
        "conditionId": condition_id,
        "outcome": "Yes",
        "outcomeIndex": 0,
        "size": size,
        "avgPrice": avg_price,
        "realizedPnl": realized_pnl,
    }
    if event_id is not None:
        payload["eventSlug"] = event_id
    return ClosedPosition.model_validate(payload)


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
    gamma_client: AsyncMock | None = None,
    tracked_repo: MagicMock | None = None,
    snapshots_repo: MagicMock | None = None,
    categories_repo: MagicMock | None = None,
    event_tag_cache: MagicMock | None = None,
) -> tuple[SmartMoneyDetector, AsyncMock, MagicMock, MagicMock]:
    """Wire a detector with mocked collaborators and return all four handles."""
    cfg = config or _config()
    data = data_client or AsyncMock()
    tracked = tracked_repo or MagicMock()
    snapshots = snapshots_repo or MagicMock()
    gamma = gamma_client or AsyncMock()
    categories = categories_repo or MagicMock()
    if event_tag_cache is None:
        event_tag_cache = MagicMock()
        event_tag_cache.get.return_value = []
    detector = SmartMoneyDetector(
        config=cfg,
        data_client=data,
        gamma_client=gamma,
        tracked_repo=tracked,
        snapshots_repo=snapshots,
        categories_repo=categories,
        event_tag_cache=event_tag_cache,
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


def test_compute_metrics_skips_zero_pnl_zero_value_artifacts() -> None:
    """Split/merge/convert artifacts (realized_pnl == 0 AND current_value == 0) are skipped."""
    closed = [
        # Real winning trade: kept.
        _closed_position(won=True, avg_price=0.4, size=100.0, realized_pnl=60.0),
        # Artifact: zero pnl + zero current value -> dropped.
        ClosedPosition.model_validate(
            {
                "proxyWallet": "0xabc",
                "asset": "tok",
                "conditionId": "0xartifact",
                "outcome": "Yes",
                "outcomeIndex": 0,
                "size": 999.0,
                "avgPrice": 0.5,
                "currentValue": 0.0,
                "realizedPnl": 0.0,
            },
        ),
        # Real losing trade: kept (negative pnl is a real signal).
        _closed_position(won=False, avg_price=0.5, size=100.0, realized_pnl=-50.0),
    ]
    metrics = _compute_metrics(closed)
    assert metrics.count == 2
    assert metrics.wins == 1
    assert metrics.winrate == pytest.approx(0.5)
    # mean_edge = ((1 - 0.4) + (0 - 0.5)) / 2 = (0.6 - 0.5) / 2 = 0.05.
    assert metrics.mean_edge == pytest.approx(0.05)
    # excess_pnl_usd ignores the artifact.
    assert metrics.excess_pnl_usd == pytest.approx(10.0)


def test_compute_metrics_mixed_wins_losses_bounds_winrate_and_edge() -> None:
    """Mixed wins/losses must yield winrate < 1.0 and a bounded (non-inflated) mean_edge."""
    closed = [
        _closed_position(won=True, avg_price=0.4, size=100.0, realized_pnl=60.0),
        _closed_position(won=True, avg_price=0.5, size=100.0, realized_pnl=50.0),
        _closed_position(won=False, avg_price=0.6, size=100.0, realized_pnl=-60.0),
        _closed_position(won=False, avg_price=0.7, size=100.0, realized_pnl=-70.0),
    ]
    metrics = _compute_metrics(closed)
    assert 0.0 < metrics.winrate < 1.0
    assert metrics.winrate == pytest.approx(0.5)
    # mean_edge = ((1-0.4) + (1-0.5) + (0-0.6) + (0-0.7)) / 4 = (0.6+0.5-0.6-0.7)/4 = -0.05.
    assert metrics.mean_edge == pytest.approx(-0.05)
    # Far below the inflated 0.885 value the legacy /v1/closed-positions endpoint produced.
    assert metrics.mean_edge < 0.5


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
    data_client.get_settled_positions.return_value = [
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
    data_client.get_settled_positions.return_value = [
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
    data_client.get_settled_positions.return_value = [
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
    data_client.get_settled_positions.return_value = [
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
    data_client.get_settled_positions.assert_not_called()


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


def _winning_position(*, event_id: str, condition_id: str) -> ClosedPosition:
    """Synthetic winning position at avg_price=0.4 (edge=0.6) with realized_pnl=600."""
    return _closed_position(
        won=True,
        avg_price=0.4,
        size=1000.0,
        realized_pnl=600.0,
        event_id=event_id,
        condition_id=condition_id,
    )


def _losing_position(*, event_id: str, condition_id: str) -> ClosedPosition:
    """Synthetic losing position at avg_price=0.5 (edge=-0.5) with realized_pnl=-500."""
    return _closed_position(
        won=False,
        avg_price=0.5,
        size=1000.0,
        realized_pnl=-500.0,
        event_id=event_id,
        condition_id=condition_id,
    )


def _make_event_tag_cache_mock(tags_by_event: dict[str, list[str]]) -> MagicMock:
    """Build a MagicMock returning provided tag lists per event id (None for misses)."""
    cache = MagicMock()
    cache.get.side_effect = tags_by_event.get
    return cache


@pytest.mark.asyncio
async def test_refresh_categorizes_closed_positions_by_tags() -> None:
    """5 thesis + 5 sports + 5 esports closed positions → 3 category rows upserted."""
    closed: list[ClosedPosition] = []
    for i in range(5):
        closed.append(_winning_position(event_id=f"evt-th-{i}", condition_id=f"c-th-{i}"))
    for i in range(5):
        closed.append(_winning_position(event_id=f"evt-sp-{i}", condition_id=f"c-sp-{i}"))
    for i in range(5):
        closed.append(_winning_position(event_id=f"evt-es-{i}", condition_id=f"c-es-{i}"))
    tags_by_event: dict[str, list[str]] = {}
    for i in range(5):
        tags_by_event[f"evt-th-{i}"] = ["Politics"]
        tags_by_event[f"evt-sp-{i}"] = ["Sports", "NFL"]
        tags_by_event[f"evt-es-{i}"] = ["Esports"]
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_settled_positions.return_value = closed
    categories_repo = MagicMock()
    event_tag_cache = _make_event_tag_cache_mock(tags_by_event)

    detector, _, _, _ = _build_detector(
        data_client=data_client,
        categories_repo=categories_repo,
        event_tag_cache=event_tag_cache,
    )

    await detector._refresh_tracked_wallets()

    assert categories_repo.upsert.call_count == 3
    by_category = {
        call.kwargs["category"]: call.kwargs for call in categories_repo.upsert.call_args_list
    }
    assert set(by_category) == {"thesis", "sports", "esports"}
    for category, kwargs in by_category.items():
        assert kwargs["wallet"] == "0xabc", f"wallet for {category}"
        assert kwargs["position_count"] == 5
        assert kwargs["win_count"] == 5
        # All wins at avg_price=0.4 → mean_edge = 0.6 for every category.
        assert kwargs["mean_edge"] == pytest.approx(0.6)
        assert kwargs["excess_pnl_usd"] == pytest.approx(3000.0)  # 5 * 600
        assert kwargs["total_stake_usd"] == pytest.approx(2000.0)  # 5 * 1000 * 0.4


@pytest.mark.asyncio
async def test_refresh_cache_miss_falls_back_to_gamma() -> None:
    """A position whose event tag isn't cached triggers a gamma fetch + cache write."""
    closed = [_winning_position(event_id=f"evt-{i}", condition_id=f"c-{i}") for i in range(6)]
    # Cache hits for the first 5; miss on evt-5 → gamma fallback.
    tags_by_event: dict[str, list[str]] = {f"evt-{i}": ["Politics"] for i in range(5)}
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_settled_positions.return_value = closed
    event_tag_cache = _make_event_tag_cache_mock(tags_by_event)
    gamma_client = AsyncMock()
    fetched_event = MagicMock()
    fetched_event.tags = ["Sports", "NBA"]
    gamma_client.get_event_by_slug.return_value = fetched_event
    categories_repo = MagicMock()

    detector, _, _, _ = _build_detector(
        data_client=data_client,
        gamma_client=gamma_client,
        categories_repo=categories_repo,
        event_tag_cache=event_tag_cache,
        config=_config(min_resolved_positions=1),
    )

    await detector._refresh_tracked_wallets()

    gamma_client.get_event_by_slug.assert_awaited_once_with("evt-5")
    event_tag_cache.upsert.assert_called_once_with("evt-5", ["Sports", "NBA"])
    by_category = {
        call.kwargs["category"]: call.kwargs["position_count"]
        for call in categories_repo.upsert.call_args_list
    }
    assert by_category == {"thesis": 5, "sports": 1}


@pytest.mark.asyncio
async def test_refresh_skips_category_below_min_resolved() -> None:
    """A category with fewer than ``min_resolved_positions`` positions is NOT upserted."""
    closed: list[ClosedPosition] = []
    # 5 thesis (>= min_resolved=5).
    for i in range(5):
        closed.append(_winning_position(event_id=f"evt-th-{i}", condition_id=f"c-th-{i}"))
    # 2 sports (< min_resolved=5).
    for i in range(2):
        closed.append(_winning_position(event_id=f"evt-sp-{i}", condition_id=f"c-sp-{i}"))
    tags_by_event: dict[str, list[str]] = {}
    for i in range(5):
        tags_by_event[f"evt-th-{i}"] = ["Politics"]
    for i in range(2):
        tags_by_event[f"evt-sp-{i}"] = ["Sports"]
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_settled_positions.return_value = closed
    categories_repo = MagicMock()
    event_tag_cache = _make_event_tag_cache_mock(tags_by_event)

    detector, _, _, _ = _build_detector(
        data_client=data_client,
        categories_repo=categories_repo,
        event_tag_cache=event_tag_cache,
        config=_config(min_resolved_positions=5),
    )

    await detector._refresh_tracked_wallets()

    categories = [call.kwargs["category"] for call in categories_repo.upsert.call_args_list]
    assert categories == ["thesis"]


@pytest.mark.asyncio
async def test_refresh_drops_position_when_gamma_fails() -> None:
    """Cache miss + gamma error → log warn, skip position from category breakdown."""
    closed = [_winning_position(event_id=f"evt-{i}", condition_id=f"c-{i}") for i in range(5)]
    tags_by_event: dict[str, list[str]] = {f"evt-{i}": ["Politics"] for i in range(4)}
    # Index 4 → cache miss → gamma raises.
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_settled_positions.return_value = closed
    event_tag_cache = _make_event_tag_cache_mock(tags_by_event)
    gamma_client = AsyncMock()
    gamma_client.get_event_by_slug.side_effect = RuntimeError("gamma 503")
    categories_repo = MagicMock()

    detector, _, _, _ = _build_detector(
        data_client=data_client,
        gamma_client=gamma_client,
        categories_repo=categories_repo,
        event_tag_cache=event_tag_cache,
        config=_config(min_resolved_positions=4),
    )

    await detector._refresh_tracked_wallets()

    # The dropped position is not categorized; the other four feed thesis.
    by_category = {
        call.kwargs["category"]: call.kwargs["position_count"]
        for call in categories_repo.upsert.call_args_list
    }
    assert by_category == {"thesis": 4}
    event_tag_cache.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_position_without_event_id_is_thesis() -> None:
    """Closed positions lacking an event_id default to ``thesis``."""
    closed = [_winning_position(event_id="evt-th", condition_id=f"c-th-{i}") for i in range(4)]
    closed.append(_closed_position(won=True, avg_price=0.4, size=100.0, realized_pnl=60.0))
    tags_by_event = {"evt-th": ["Politics"]}
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_settled_positions.return_value = closed
    categories_repo = MagicMock()
    event_tag_cache = _make_event_tag_cache_mock(tags_by_event)

    detector, _, _, _ = _build_detector(
        data_client=data_client,
        categories_repo=categories_repo,
        event_tag_cache=event_tag_cache,
        config=_config(min_resolved_positions=5),
    )

    await detector._refresh_tracked_wallets()

    by_category = {
        call.kwargs["category"]: call.kwargs["position_count"]
        for call in categories_repo.upsert.call_args_list
    }
    assert by_category == {"thesis": 5}


@pytest.mark.asyncio
async def test_refresh_overall_upsert_unchanged_with_categories() -> None:
    """The overall ``tracked_wallets`` upsert still fires (backwards compat)."""
    closed = [_losing_position(event_id=f"evt-{i}", condition_id=f"c-{i}") for i in range(3)] + [
        _winning_position(event_id=f"evt-w-{i}", condition_id=f"c-w-{i}") for i in range(4)
    ]
    tags_by_event: dict[str, list[str]] = {}
    for i in range(3):
        tags_by_event[f"evt-{i}"] = ["Sports"]
    for i in range(4):
        tags_by_event[f"evt-w-{i}"] = ["Politics"]
    data_client = AsyncMock()
    data_client.get_leaderboard.return_value = [_leaderboard_entry()]
    data_client.get_settled_positions.return_value = closed
    tracked_repo = MagicMock()
    categories_repo = MagicMock()
    event_tag_cache = _make_event_tag_cache_mock(tags_by_event)

    detector, _, _, _ = _build_detector(
        data_client=data_client,
        tracked_repo=tracked_repo,
        categories_repo=categories_repo,
        event_tag_cache=event_tag_cache,
        config=_config(min_resolved_positions=3, min_excess_pnl_usd=500.0),
    )

    await detector._refresh_tracked_wallets()

    tracked_repo.upsert.assert_called_once()
    overall_kwargs = tracked_repo.upsert.call_args.kwargs
    assert overall_kwargs["address"] == "0xabc"
    assert overall_kwargs["closed_position_count"] == 7
