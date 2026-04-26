"""Tests for MoveAttributionDetector and its pure helpers."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import MoveAttributionConfig
from pscanner.detectors.move_attribution import (
    BurstHit,
    MoveAttributionDetector,
    _backwalk,
    _detect_burst,
)
from pscanner.poly.data import DataClient
from pscanner.poly.http import PolyHttpClient
from pscanner.store.repo import AlertsRepo, WatchlistRepo

_DATA = "https://data-api.polymarket.com"


def _trade(
    *,
    wallet: str,
    ts: int,
    side: str = "BUY",
    outcome: str = "Yes",
    size: float = 100.0,
    price: float = 0.5,
) -> dict:
    return {
        "proxyWallet": wallet,
        "timestamp": ts,
        "side": side,
        "outcome": outcome,
        "size": size,
        "price": price,
    }


def _cfg(**overrides) -> MoveAttributionConfig:
    return MoveAttributionConfig(**overrides)


def test_detect_burst_happy_path_fires() -> None:
    trades = [
        _trade(wallet=f"0x{i}", ts=1000, size=500.0 + i)  # CV ~0
        for i in range(4)
    ]
    hits = _detect_burst(trades, cfg=_cfg())
    assert len(hits) == 1
    h = hits[0]
    assert isinstance(h, BurstHit)
    assert h.outcome == "Yes"
    assert h.side == "BUY"
    assert set(h.wallets) == {"0x0", "0x1", "0x2", "0x3"}


def test_detect_burst_below_wallet_threshold_no_hit() -> None:
    trades = [_trade(wallet=f"0x{i}", ts=1000) for i in range(3)]
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_high_cv_no_hit() -> None:
    sizes = [10.0, 100.0, 1000.0, 5000.0]  # CV >> 0.4
    trades = [_trade(wallet=f"0x{i}", ts=1000, size=s) for i, s in enumerate(sizes)]
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_mixed_sides_split_buckets_no_hit() -> None:
    trades = [
        _trade(wallet="0x1", ts=1000, side="BUY"),
        _trade(wallet="0x2", ts=1000, side="BUY"),
        _trade(wallet="0x3", ts=1000, side="SELL"),
        _trade(wallet="0x4", ts=1000, side="SELL"),
    ]
    # 2 BUY + 2 SELL — neither side hits min_burst_wallets=4
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_cross_bucket_no_hit() -> None:
    # 4 wallets but split across two 60s buckets (2 + 2)
    trades = [
        _trade(wallet="0x1", ts=1000),
        _trade(wallet="0x2", ts=1010),
        _trade(wallet="0x3", ts=1080),  # next bucket
        _trade(wallet="0x4", ts=1090),
    ]
    assert _detect_burst(trades, cfg=_cfg()) == []


def test_detect_burst_truncates_at_max_burst_hits() -> None:
    # 8 distinct buckets each with 4 wallets at uniform size
    trades: list[dict] = []
    for bucket in range(8):
        ts = 1000 + bucket * 60
        trades.extend(_trade(wallet=f"0x{bucket}-{w}", ts=ts) for w in range(4))
    hits = _detect_burst(trades, cfg=_cfg(max_burst_hits_per_alert=5))
    assert len(hits) == 5  # truncated


def test_detect_burst_truncates_contributors_to_top_50() -> None:
    # 80 wallets in one bucket, sizes 1..80 — keep 50 closest to median
    sizes = list(range(1, 81))
    trades = [_trade(wallet=f"0x{i}", ts=1000, size=float(s)) for i, s in enumerate(sizes)]
    hits = _detect_burst(
        trades,
        cfg=_cfg(min_burst_wallets=4, max_burst_size_cv=2.0, max_contributors_per_burst=50),
    )
    assert len(hits) == 1
    assert len(hits[0].wallets) == 50
    kept_indices = {int(w.removeprefix("0x")) for w in hits[0].wallets}
    assert kept_indices == set(range(15, 65))


def test_detect_burst_drops_zero_size_wallets_from_contributors() -> None:
    # 4 valid wallets at size=500 + 50 wallets with missing size; median is 500.
    # Without the fix, the 50 noise wallets would rank above the cap and crowd
    # out the genuine signal in the kept set.
    valid = [_trade(wallet=f"0xv{i}", ts=1000, size=500.0) for i in range(4)]
    noise = [
        {
            "proxyWallet": f"0xn{i}",
            "timestamp": 1000,
            "side": "BUY",
            "outcome": "Yes",
            "size": 0.0,
            "price": 0.5,
        }
        for i in range(50)
    ]
    hits = _detect_burst(valid + noise, cfg=_cfg(max_contributors_per_burst=50))
    assert len(hits) == 1
    assert set(hits[0].wallets) == {f"0xv{i}" for i in range(4)}


def _backwalk_client() -> DataClient:
    return DataClient(http=PolyHttpClient(base_url=_DATA, rpm=600, timeout_seconds=5.0))


def _gen_trades(*, ts_start: int, ts_end: int, gap: int) -> list[dict]:
    """Generate trades with uniform gap, newest-first."""
    out: list[dict] = []
    ts = ts_end
    i = 0
    while ts >= ts_start:
        out.append(
            {
                "proxyWallet": f"0x{i:04d}",
                "timestamp": ts,
                "side": "BUY",
                "outcome": "Yes",
                "size": 50.0,
                "price": 0.5,
            }
        )
        ts -= gap
        i += 1
    return out


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_stops_on_quiescence() -> None:
    # Baseline: 1 trade per 60s for 24h prior
    # Burst: 1 trade per 5s for 30 min ending at alert_ts
    alert_ts = 1_700_086_400
    baseline_start = alert_ts - 86_400
    baseline = _gen_trades(ts_start=baseline_start, ts_end=alert_ts - 1800, gap=60)
    burst = _gen_trades(ts_start=alert_ts - 1800, ts_end=alert_ts, gap=5)
    page = sorted(burst + baseline, key=lambda t: -t["timestamp"])[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    client = _backwalk_client()
    try:
        since_ts, until_ts, burst_trades = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg()
        )
    finally:
        await client.aclose()
    assert until_ts == alert_ts
    # Backwalk should stop within ~30 min of the alert (the burst window)
    assert alert_ts - 7200 < since_ts < alert_ts - 1500
    assert all(since_ts <= t["timestamp"] <= until_ts for t in burst_trades)


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_caps_at_max_backwalk_seconds() -> None:
    # Constant-rate market: burst rate never drops, must hit the 7200s cap
    alert_ts = 1_700_000_000
    # gap=20 yields ~366 trades spanning the full 7200s cap in one short page,
    # mirroring real-API pagination termination (page len < 500 → stop).
    page = _gen_trades(ts_start=alert_ts - 7300, ts_end=alert_ts, gap=20)
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    client = _backwalk_client()
    try:
        since_ts, until_ts, _ = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg()
        )
    finally:
        await client.aclose()
    assert until_ts - since_ts == 7200


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_requires_two_consecutive_quiescent_windows() -> None:
    # Pattern: burst, single 5-min lull, burst again, then quiescence.
    # Single dip must NOT stop the backwalk.
    alert_ts = 1_700_086_400
    parts: list[dict] = []
    parts += _gen_trades(ts_start=alert_ts - 600, ts_end=alert_ts, gap=5)
    parts += _gen_trades(ts_start=alert_ts - 900, ts_end=alert_ts - 600, gap=180)
    parts += _gen_trades(ts_start=alert_ts - 1800, ts_end=alert_ts - 900, gap=5)
    parts += _gen_trades(ts_start=alert_ts - 86400, ts_end=alert_ts - 1800, gap=60)
    page = sorted(parts, key=lambda t: -t["timestamp"])[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))
    client = _backwalk_client()
    try:
        since_ts, _, _ = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg()
        )
    finally:
        await client.aclose()
    # Window must extend back past the single dip (>= 1800s)
    assert alert_ts - since_ts >= 1800


@respx.mock
@pytest.mark.asyncio
async def test_backwalk_empty_trades_returns_full_cap() -> None:
    alert_ts = 1_700_000_000
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=[]))
    client = _backwalk_client()
    try:
        since_ts, until_ts, burst = await _backwalk(
            client, condition_id="0xabc", alert_ts=alert_ts, cfg=_cfg()
        )
    finally:
        await client.aclose()
    assert (since_ts, until_ts) == (alert_ts - 7200, alert_ts)
    assert burst == []


def _build_velocity_alert(
    *,
    condition_id: str = "0xabc",
    alert_key: str = "velocity:0xabc:1",
    created_at: int = 1_700_086_400,
) -> Alert:
    return Alert(
        detector="velocity",
        alert_key=alert_key,
        severity="med",
        title="market moved",
        body={"condition_id": condition_id, "delta_price": 0.07},
        created_at=created_at,
    )


@respx.mock
@pytest.mark.asyncio
async def test_detector_emits_candidate_and_watchlists_contributors(tmp_db) -> None:
    alert_ts = 1_700_086_400
    burst = [
        {
            "proxyWallet": f"0x{i:04d}",
            "timestamp": alert_ts - 30,
            "side": "BUY",
            "outcome": "Yes",
            "size": 500.0 + i,
            "price": 0.95,
        }
        for i in range(6)
    ]
    baseline = _gen_trades(ts_start=alert_ts - 86400, ts_end=alert_ts - 1800, gap=60)
    page = sorted(burst + baseline, key=lambda t: -t["timestamp"])[:500]
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(200, json=page))

    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        # The detector needs sink set so its async path can call sink.emit.
        detector._sink = sink  # type: ignore[assignment]
        await sink.emit(_build_velocity_alert(created_at=alert_ts))
        for _ in range(10):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    rendered = AlertsRepo(tmp_db).recent(limit=10)
    candidate_alerts = [a for a in rendered if a.detector == "move_attribution"]
    assert len(candidate_alerts) == 1
    assert candidate_alerts[0].alert_key.startswith("cluster.candidate:")
    watchlisted = [w.address for w in watchlist.list_active()]
    assert sum(1 for a in watchlisted if a.startswith("0x")) >= 6


@pytest.mark.asyncio
async def test_detector_ignores_non_trigger_detectors(tmp_db) -> None:
    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        detector._sink = sink  # type: ignore[assignment]
        a = Alert(
            detector="whales",
            alert_key="whales:1",
            severity="med",
            title="whale",
            body={"condition_id": "0xabc"},
            created_at=1_700_000_000,
        )
        await sink.emit(a)
        for _ in range(10):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    assert watchlist.list_active() == []


@pytest.mark.asyncio
async def test_detector_skips_alert_without_condition_id(tmp_db) -> None:
    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        detector._sink = sink  # type: ignore[assignment]
        a = Alert(
            detector="velocity",
            alert_key="velocity:nocond",
            severity="med",
            title="event-level",
            body={"event_id": "evt-1"},
            created_at=1_700_000_000,
        )
        await sink.emit(a)
        for _ in range(10):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    assert watchlist.list_active() == []


@respx.mock
@pytest.mark.asyncio
async def test_detector_swallows_trades_http_error(tmp_db) -> None:
    respx.get(f"{_DATA}/trades").mock(return_value=httpx.Response(500))
    sink = AlertSink(AlertsRepo(tmp_db))
    watchlist = WatchlistRepo(tmp_db)
    client = _backwalk_client()
    try:
        detector = MoveAttributionDetector(
            config=_cfg(),
            data_client=client,
            watchlist_repo=watchlist,
        )
        sink.subscribe(detector.handle_alert_sync)
        detector._sink = sink  # type: ignore[assignment]
        await sink.emit(_build_velocity_alert())
        for _ in range(10):
            await asyncio.sleep(0)
        await detector.aclose()
    finally:
        await client.aclose()
    assert watchlist.list_active() == []
    rendered = AlertsRepo(tmp_db).recent(limit=10)
    assert [a for a in rendered if a.detector == "move_attribution"] == []
