"""Tests for MoveAttributionDetector and its pure helpers."""

from __future__ import annotations

from pscanner.config import MoveAttributionConfig
from pscanner.detectors.move_attribution import BurstHit, _detect_burst


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
