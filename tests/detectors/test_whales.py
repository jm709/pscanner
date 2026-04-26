"""Tests for ``WhalesDetector`` (DC-1.5 callback-driven).

These exercise :meth:`WhalesDetector.evaluate` directly with synthesised
``WalletTrade`` objects and verify :meth:`_refresh_market_cache` populates
the in-memory ``condition_id -> CachedMarket`` map. All collaborators are
stubbed in-process — no network, no SQLite.

Shared trade-callback plumbing (``handle_trade_sync`` task tracking, the
defensive ``_sink``-unwired short-circuit on the base class) is covered
once in :mod:`tests.detectors.test_trade_driven`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import WhalesConfig
from pscanner.detectors.whales import WhalesDetector
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.poly.models import Market
from pscanner.store.repo import CachedMarket, WalletFirstSeen, WalletTrade

# Pin "now" to wall-clock at import. Tests treat this as the trade timestamp so
# the wallet-first-seen cache TTL check (which uses real time) sees a fresh row.
_NOW = int(time.time())
_WALLET = "0xnewwhale"
_ASSET_ID = AssetId("asset-1")
_MARKET_ID = MarketId("market-1")
_CONDITION_ID = ConditionId("cond-1")


def _make_config(**overrides: Any) -> WhalesConfig:
    """Return a ``WhalesConfig`` with test-friendly defaults plus overrides."""
    base: dict[str, Any] = {
        "enabled": True,
        "new_account_max_age_days": 30,
        "new_account_max_trades": 50,
        "small_market_max_liquidity_usd": 50000.0,
        "big_bet_min_pct_of_liquidity": 0.05,
        "big_bet_min_usd": 2000.0,
        "ws_resubscribe_interval_seconds": 1800,
    }
    base.update(overrides)
    return WhalesConfig(**base)


def _make_trade(**overrides: Any) -> WalletTrade:
    """Build a ``CONFIRMED`` ``WalletTrade`` with sensible defaults."""
    base: dict[str, Any] = {
        "transaction_hash": "0xtxhash1",
        "asset_id": _ASSET_ID,
        "side": "BUY",
        "wallet": _WALLET,
        "condition_id": _CONDITION_ID,
        "size": 1000.0,
        "price": 0.50,
        "usd_value": 500.0,
        "status": "CONFIRMED",
        "source": "activity_api",
        "timestamp": _NOW,
        "recorded_at": _NOW,
    }
    base.update(overrides)
    return WalletTrade(**base)


def _make_cached_market(**overrides: Any) -> CachedMarket:
    """Build a ``CachedMarket`` with sensible test defaults."""
    base: dict[str, Any] = {
        "market_id": _MARKET_ID,
        "event_id": "event-1",
        "title": "Will X happen?",
        "liquidity_usd": 10000.0,
        "volume_usd": 50000.0,
        "outcome_prices": [],
        "active": True,
        "cached_at": _NOW,
    }
    base.update(overrides)
    return CachedMarket(**base)


def _make_first_seen(**overrides: Any) -> WalletFirstSeen:
    """Build a ``WalletFirstSeen`` row with sensible defaults."""
    base: dict[str, Any] = {
        "address": _WALLET,
        "first_activity_at": _NOW - 5 * 86400,
        "total_trades": 12,
        "cached_at": _NOW,
    }
    base.update(overrides)
    return WalletFirstSeen(**base)


class StubGammaClient:
    """Yields a fixed sequence of ``Market`` instances from ``iter_markets``."""

    def __init__(self, markets: list[Market]) -> None:
        self._markets = markets

    async def iter_markets(self, **_kwargs: Any) -> AsyncIterator[Market]:
        for market in self._markets:
            yield market


class StubDataClient:
    """Records calls and returns canned responses."""

    def __init__(
        self,
        *,
        first_activity_at: int | None = None,
        activity: list[dict[str, Any]] | None = None,
    ) -> None:
        self.first_calls: list[str] = []
        self.activity_calls: list[tuple[str, int | None]] = []
        self._first_activity_at = first_activity_at
        self._activity = activity or []

    async def get_first_activity_timestamp(self, address: str) -> int | None:
        self.first_calls.append(address)
        return self._first_activity_at

    async def get_activity(
        self,
        address: str,
        *,
        limit: int = 500,
        type: str | None = None,  # noqa: A002
    ) -> list[dict[str, Any]]:
        del type
        self.activity_calls.append((address, limit))
        return self._activity


class StubMarketCache:
    """In-memory ``MarketCacheRepo`` with explicit upsert tracking."""

    def __init__(self, prepop: dict[str, CachedMarket] | None = None) -> None:
        self._cache: dict[str, CachedMarket] = dict(prepop or {})
        self.upserts: list[Market] = []

    def upsert(self, market: Market) -> None:
        self.upserts.append(market)
        # Mirror what MarketCacheRepo does: ensure the next .get() returns a
        # CachedMarket reflecting the upsert so _refresh_market_cache can
        # rebuild its condition map from .get().
        self._cache[market.id] = CachedMarket(
            market_id=market.id,
            event_id=market.event_id,
            title=market.question,
            liquidity_usd=market.liquidity,
            volume_usd=market.volume,
            outcome_prices=list(market.outcome_prices),
            active=market.active,
            cached_at=_NOW,
        )

    def get(self, market_id: str) -> CachedMarket | None:
        return self._cache.get(market_id)


class StubFirstSeen:
    """In-memory ``WalletFirstSeenRepo``."""

    def __init__(self, prepop: dict[str, WalletFirstSeen] | None = None) -> None:
        self._rows: dict[str, WalletFirstSeen] = dict(prepop or {})
        self.upserts: list[tuple[str, int | None, int | None]] = []
        self.now_ts = _NOW

    def get(self, address: str) -> WalletFirstSeen | None:
        return self._rows.get(address)

    def upsert(
        self,
        address: str,
        first_activity_at: int | None,
        total_trades: int | None,
    ) -> None:
        self.upserts.append((address, first_activity_at, total_trades))
        self._rows[address] = WalletFirstSeen(
            address=address,
            first_activity_at=first_activity_at,
            total_trades=total_trades,
            cached_at=self.now_ts,
        )


class CapturingSink:
    """Collects every alert ``emit`` is called with."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    async def emit(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


def _make_detector(
    *,
    config: WhalesConfig | None = None,
    gamma: StubGammaClient | None = None,
    data: StubDataClient | None = None,
    market_cache: StubMarketCache | None = None,
    first_seen: StubFirstSeen | None = None,
    sink: CapturingSink | None = None,
    seed_condition_map: bool = True,
) -> WhalesDetector:
    """Construct a ``WhalesDetector`` wired to the given stubs (with defaults)."""
    cache = market_cache or StubMarketCache()
    detector = WhalesDetector(
        config=config or _make_config(),
        gamma_client=gamma or StubGammaClient([]),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=data or StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=cache,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=first_seen or StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    if sink is not None:
        detector._sink = sink  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
    if seed_condition_map:
        cached = cache.get(_MARKET_ID)
        if cached is not None:
            detector._condition_to_market[_CONDITION_ID] = cached
    return detector


@pytest.mark.asyncio
async def test_evaluate_emits_when_thresholds_met() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    trade = _make_trade(size=4000.0, price=1.0, usd_value=4000.0)
    await detector.evaluate(trade)

    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.severity == "high"  # >$10k or pct>0.20 -> high
    assert alert.body["usd_value"] == pytest.approx(4000.0)
    assert alert.body["market_liquidity"] == pytest.approx(10000.0)
    assert alert.body["age_days"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_evaluate_skips_aged_wallet() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    aged_seen = _make_first_seen(first_activity_at=_NOW - 60 * 86400)
    first_seen = StubFirstSeen({_WALLET: aged_seen})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_evaluate_skips_when_usd_below_min() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    await detector.evaluate(_make_trade(size=10.0, price=0.10, usd_value=1.0))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_evaluate_skips_when_market_too_liquid() -> None:
    sink = CapturingSink()
    big_market = _make_cached_market(liquidity_usd=200000.0)
    market_cache = StubMarketCache({_MARKET_ID: big_market})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_evaluate_skips_unknown_condition() -> None:
    sink = CapturingSink()
    detector = _make_detector(sink=sink, seed_condition_map=False)

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_evaluate_refreshes_when_first_seen_missing() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen()  # empty
    data = StubDataClient(
        first_activity_at=_NOW - 5 * 86400,
        activity=[{"type": "TRADE"}] * 7,
    )
    detector = _make_detector(
        market_cache=market_cache,
        first_seen=first_seen,
        data=data,
        sink=sink,
    )

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert data.first_calls == [_WALLET]
    assert data.activity_calls == [(_WALLET, 200)]
    assert first_seen.upserts == [(_WALLET, _NOW - 5 * 86400, 7)]
    assert len(sink.alerts) == 1
    assert sink.alerts[0].body["total_trades"] == 7


@pytest.mark.asyncio
async def test_evaluate_refreshes_stale_cache() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    # cached_at is 2 days ago — TTL is 1 day, so this row is stale.
    stale = replace(_make_first_seen(), cached_at=int(time.time()) - 2 * 86400)
    first_seen = StubFirstSeen({_WALLET: stale})
    data = StubDataClient(
        first_activity_at=_NOW - 5 * 86400,
        activity=[{"type": "TRADE"}] * 9,
    )
    detector = _make_detector(
        market_cache=market_cache,
        first_seen=first_seen,
        data=data,
        sink=sink,
    )

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert data.first_calls == [_WALLET]
    assert first_seen.upserts == [(_WALLET, _NOW - 5 * 86400, 9)]
    assert len(sink.alerts) == 1


@pytest.mark.asyncio
async def test_evaluate_skips_when_first_activity_unknown() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen()
    data = StubDataClient(first_activity_at=None, activity=[])
    detector = _make_detector(
        market_cache=market_cache,
        first_seen=first_seen,
        data=data,
        sink=sink,
    )

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_evaluate_skips_when_total_trades_exceeds_cap() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    seen = _make_first_seen(total_trades=999)
    first_seen = StubFirstSeen({_WALLET: seen})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    await detector.evaluate(_make_trade(size=4000.0, price=1.0, usd_value=4000.0))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_evaluate_alert_key_falls_back_when_no_tx_hash() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    trade = _make_trade(size=4000.0, price=1.0, usd_value=4000.0, transaction_hash="")
    await detector.evaluate(trade)

    assert sink.alerts[0].alert_key == f"whale:{_CONDITION_ID}:{_WALLET}:{_NOW}"


@pytest.mark.asyncio
async def test_evaluate_severity_med_for_modest_bet() -> None:
    sink = CapturingSink()
    # Liquidity 50000 (right at the small-market boundary); usd=2500 -> 5% pct, < $10k abs.
    market = _make_cached_market(liquidity_usd=50000.0)
    market_cache = StubMarketCache({_MARKET_ID: market})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, sink=sink)

    await detector.evaluate(_make_trade(size=2500.0, price=1.0, usd_value=2500.0))

    assert sink.alerts[0].severity == "med"


@pytest.mark.asyncio
async def test_refresh_market_cache_populates_condition_map() -> None:
    """``_refresh_market_cache`` builds ``_condition_to_market`` from gamma iter."""
    market_a = Market.model_validate(
        {
            "id": "m-a",
            "conditionId": "cond-a",
            "question": "A?",
            "slug": "a",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["a1", "a2"],
            "volume": 1000.0,
            "liquidity": 5000.0,
        }
    )
    market_b = Market.model_validate(
        {
            "id": "m-b",
            "conditionId": "cond-b",
            "question": "B?",
            "slug": "b",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.4", "0.6"],
            "clobTokenIds": ["b1", "b2"],
            "volume": 1000.0,
            "liquidity": 7000.0,
        }
    )
    gamma = StubGammaClient([market_a, market_b])
    market_cache = StubMarketCache()
    detector = WhalesDetector(
        config=_make_config(),
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    await detector._refresh_market_cache()

    assert [m.id for m in market_cache.upserts] == ["m-a", "m-b"]
    assert set(detector._condition_to_market.keys()) == {"cond-a", "cond-b"}


@pytest.mark.asyncio
async def test_refresh_market_cache_skips_markets_without_order_book() -> None:
    """Markets with ``enable_order_book=False`` aren't cached."""
    bookless = Market.model_validate(
        {
            "id": "m-x",
            "conditionId": "cond-x",
            "question": "X?",
            "slug": "x",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["x1", "x2"],
            "volume": 5000.0,
            "enableOrderBook": False,
        }
    )
    booked = Market.model_validate(
        {
            "id": "m-y",
            "conditionId": "cond-y",
            "question": "Y?",
            "slug": "y",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["y1", "y2"],
            "volume": 5000.0,
            "enableOrderBook": True,
        }
    )
    gamma = StubGammaClient([bookless, booked])
    market_cache = StubMarketCache()
    detector = WhalesDetector(
        config=_make_config(),
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    await detector._refresh_market_cache()

    assert [m.id for m in market_cache.upserts] == ["m-y"]
    assert set(detector._condition_to_market.keys()) == {"cond-y"}


@pytest.mark.asyncio
async def test_refresh_market_cache_stops_at_max_markets_cap() -> None:
    """Loop stops once ``subscription_max_markets`` markets accumulated."""

    def _make_market_payload(idx: int) -> dict[str, Any]:
        return {
            "id": f"m-{idx}",
            "conditionId": f"cond-{idx}",
            "question": "Q?",
            "slug": f"slug-{idx}",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": [f"a{idx}", f"b{idx}"],
            "volume": 1000.0,
        }

    markets = [Market.model_validate(_make_market_payload(i)) for i in range(10)]
    gamma = StubGammaClient(markets)
    market_cache = StubMarketCache()
    detector = WhalesDetector(
        config=_make_config(subscription_max_markets=3),
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    await detector._refresh_market_cache()

    assert len(detector._condition_to_market) == 3


@pytest.mark.asyncio
async def test_refresh_market_cache_skips_low_volume_markets() -> None:
    """Markets with volume below ``subscription_min_volume_usd`` are skipped."""
    quiet = Market.model_validate(
        {
            "id": "m-quiet",
            "conditionId": "cond-quiet",
            "question": "Q?",
            "slug": "quiet",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["q1", "q2"],
            "volume": 5.0,
        }
    )
    busy = Market.model_validate(
        {
            "id": "m-busy",
            "conditionId": "cond-busy",
            "question": "B?",
            "slug": "busy",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["w1", "w2"],
            "volume": 5000.0,
        }
    )
    gamma = StubGammaClient([quiet, busy])
    market_cache = StubMarketCache()
    detector = WhalesDetector(
        config=_make_config(subscription_min_volume_usd=100.0),
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    await detector._refresh_market_cache()

    assert [m.id for m in market_cache.upserts] == ["m-busy"]
    assert set(detector._condition_to_market.keys()) == {"cond-busy"}


@pytest.mark.asyncio
async def test_run_sets_sink_and_runs_until_cancelled() -> None:
    """``run`` stores the sink and refreshes the cache, then sleeps."""
    sink = CapturingSink()
    gamma = StubGammaClient([])
    detector = WhalesDetector(
        config=_make_config(ws_resubscribe_interval_seconds=3600),
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=StubMarketCache(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    # Yield control briefly so the task hits its first await.
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert detector._sink is sink
