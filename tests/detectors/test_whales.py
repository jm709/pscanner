"""Tests for ``WhalesDetector``.

These exercise ``_handle_trade`` directly with synthesised dependencies and
verify ``_subscription_loop`` issues batched ``ws.subscribe`` calls. All
collaborators are stubbed in-process — no network, no SQLite.
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
from pscanner.poly.models import Market, WsBookMessage, WsTradeMessage
from pscanner.store.repo import CachedMarket, WalletFirstSeen

# Pin "now" to wall-clock at import. Tests treat this as the trade timestamp so
# the wallet-first-seen cache TTL check (which uses real time) sees a fresh row.
_NOW = int(time.time())
_WALLET = "0xnewwhale"
_ASSET_ID = "asset-1"
_MARKET_ID = "market-1"
_CONDITION_ID = "cond-1"


def _make_config(**overrides: Any) -> WhalesConfig:
    """Return a ``WhalesConfig`` with test-friendly defaults plus overrides."""
    base: dict[str, Any] = {
        "enabled": True,
        "new_account_max_age_days": 30,
        "new_account_max_trades": 50,
        "small_market_max_liquidity_usd": 50000.0,
        "big_bet_min_pct_of_liquidity": 0.05,
        "big_bet_min_usd": 2000.0,
        "ws_subscribe_batch_size": 50,
        "ws_resubscribe_interval_seconds": 1800,
    }
    base.update(overrides)
    return WhalesConfig(**base)


def _make_trade(**overrides: Any) -> WsTradeMessage:
    """Build a ``CONFIRMED`` trade message with sensible defaults."""
    base: dict[str, Any] = {
        "event_type": "trade",
        "condition_id": _CONDITION_ID,
        "asset_id": _ASSET_ID,
        "side": "BUY",
        "size": 1000.0,
        "price": 0.50,
        "taker_proxy": _WALLET,
        "status": "CONFIRMED",
        "transaction_hash": "0xtxhash1",
        "timestamp": _NOW,
    }
    base.update(overrides)
    return WsTradeMessage.model_validate(base)


def _make_cached_market(**overrides: Any) -> CachedMarket:
    """Build a ``CachedMarket`` with sensible test defaults."""
    base: dict[str, Any] = {
        "market_id": _MARKET_ID,
        "event_id": "event-1",
        "title": "Will X happen?",
        "liquidity_usd": 10000.0,
        "volume_usd": 50000.0,
        "outcome_prices_json": "[]",
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


class StubWebSocket:
    """Minimal in-memory ``MarketWebSocket`` replacement."""

    def __init__(self) -> None:
        self.connected = False
        self.subscribe_calls: list[list[str]] = []

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(self, asset_ids: Any) -> None:
        self.subscribe_calls.append(list(asset_ids))

    async def messages(self) -> AsyncIterator[Any]:
        if False:  # pragma: no cover - generator typing helper
            yield None


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
    ws: StubWebSocket | None = None,
    gamma: StubGammaClient | None = None,
    data: StubDataClient | None = None,
    market_cache: StubMarketCache | None = None,
    first_seen: StubFirstSeen | None = None,
) -> WhalesDetector:
    """Construct a ``WhalesDetector`` wired to the given stubs (with defaults)."""
    detector = WhalesDetector(
        config=config or _make_config(),
        ws=ws or StubWebSocket(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        gamma_client=gamma or StubGammaClient([]),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=data or StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache or StubMarketCache(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=first_seen or StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )
    detector._asset_to_market[_ASSET_ID] = _MARKET_ID
    return detector


@pytest.mark.asyncio
async def test_handle_trade_emits_when_thresholds_met() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    trade = _make_trade(size=4000.0, price=1.0)  # $4000 > $2000 min, 40% of $10k liquidity
    await detector._handle_trade(trade, sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.severity == "high"  # >$10k or pct>0.20 -> high
    assert alert.body["usd_value"] == pytest.approx(4000.0)
    assert alert.body["market_liquidity"] == pytest.approx(10000.0)
    assert alert.body["age_days"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_handle_trade_skips_aged_wallet() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    aged_seen = _make_first_seen(first_activity_at=_NOW - 60 * 86400)
    first_seen = StubFirstSeen({_WALLET: aged_seen})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_handle_trade_skips_when_usd_below_min() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    await detector._handle_trade(_make_trade(size=10.0, price=0.10), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_handle_trade_skips_when_market_too_liquid() -> None:
    sink = CapturingSink()
    big_market = _make_cached_market(liquidity_usd=200000.0)
    market_cache = StubMarketCache({_MARKET_ID: big_market})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_handle_trade_skips_unknown_market() -> None:
    sink = CapturingSink()
    detector = _make_detector()
    detector._asset_to_market.clear()  # forget the mapping

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_handle_trade_refreshes_when_first_seen_missing() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen()  # empty
    data = StubDataClient(
        first_activity_at=_NOW - 5 * 86400,
        activity=[{"type": "TRADE"}] * 7,
    )
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, data=data)

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert data.first_calls == [_WALLET]
    assert data.activity_calls == [(_WALLET, 200)]
    assert first_seen.upserts == [(_WALLET, _NOW - 5 * 86400, 7)]
    assert len(sink.alerts) == 1
    assert sink.alerts[0].body["total_trades"] == 7


@pytest.mark.asyncio
async def test_handle_trade_refreshes_stale_cache() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    # cached_at is 2 days ago — TTL is 1 day, so this row is stale.
    stale = replace(_make_first_seen(), cached_at=int(time.time()) - 2 * 86400)
    first_seen = StubFirstSeen({_WALLET: stale})
    data = StubDataClient(
        first_activity_at=_NOW - 5 * 86400,
        activity=[{"type": "TRADE"}] * 9,
    )
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, data=data)

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert data.first_calls == [_WALLET]
    assert first_seen.upserts == [(_WALLET, _NOW - 5 * 86400, 9)]
    assert len(sink.alerts) == 1


@pytest.mark.asyncio
async def test_handle_trade_skips_when_first_activity_unknown() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen()
    data = StubDataClient(first_activity_at=None, activity=[])
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen, data=data)

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_handle_trade_skips_when_total_trades_exceeds_cap() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    seen = _make_first_seen(total_trades=999)
    first_seen = StubFirstSeen({_WALLET: seen})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    await detector._handle_trade(_make_trade(size=4000.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_handle_trade_alert_key_falls_back_when_no_tx_hash() -> None:
    sink = CapturingSink()
    market_cache = StubMarketCache({_MARKET_ID: _make_cached_market()})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    trade = _make_trade(size=4000.0, price=1.0, transaction_hash=None)
    await detector._handle_trade(trade, sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts[0].alert_key == f"whale:{_CONDITION_ID}:{_WALLET}:{_NOW}"


@pytest.mark.asyncio
async def test_handle_trade_severity_med_for_modest_bet() -> None:
    sink = CapturingSink()
    # Liquidity 50000 (right at the small-market boundary); usd=2500 -> 5% pct, < $10k abs.
    market = _make_cached_market(liquidity_usd=50000.0)
    market_cache = StubMarketCache({_MARKET_ID: market})
    first_seen = StubFirstSeen({_WALLET: _make_first_seen()})
    detector = _make_detector(market_cache=market_cache, first_seen=first_seen)

    await detector._handle_trade(_make_trade(size=2500.0, price=1.0), sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts[0].severity == "med"


@pytest.mark.asyncio
async def test_refresh_subscriptions_batches_asset_ids() -> None:
    market_a = Market.model_validate(
        {
            "id": "m-a",
            "question": "A?",
            "slug": "a",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": ["a1", "a2"],
        }
    )
    market_b = Market.model_validate(
        {
            "id": "m-b",
            "question": "B?",
            "slug": "b",
            "outcomes": ["YES", "NO"],
            "outcomePrices": ["0.4", "0.6"],
            "clobTokenIds": ["b1", "b2", "b3"],
        }
    )
    ws = StubWebSocket()
    gamma = StubGammaClient([market_a, market_b])
    market_cache = StubMarketCache()
    detector = WhalesDetector(
        config=_make_config(ws_subscribe_batch_size=2),
        ws=ws,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    await detector._refresh_subscriptions()

    assert [m.id for m in market_cache.upserts] == ["m-a", "m-b"]
    assert detector._asset_to_market == {
        "a1": "m-a",
        "a2": "m-a",
        "b1": "m-b",
        "b2": "m-b",
        "b3": "m-b",
    }
    # 5 asset ids, batch size 2 -> 3 calls of sizes 2, 2, 1.
    assert [len(batch) for batch in ws.subscribe_calls] == [2, 2, 1]
    assert ws.subscribe_calls[0] == ["a1", "a2"]
    assert ws.subscribe_calls[1] == ["b1", "b2"]
    assert ws.subscribe_calls[2] == ["b3"]


@pytest.mark.asyncio
async def test_consume_loop_skips_non_trade_messages() -> None:
    sink = CapturingSink()
    book_msg = WsBookMessage.model_validate({"event_type": "book", "asset_id": "x", "data": {}})

    class WsWithMessages(StubWebSocket):
        async def messages(self) -> AsyncIterator[Any]:
            yield book_msg

    detector = _make_detector(ws=WsWithMessages())

    await detector._consume_loop(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_run_connects_and_dispatches_taskgroup() -> None:
    """``run`` must call ``ws.connect`` and start the two loops concurrently."""
    sink = CapturingSink()

    class OneShotWs(StubWebSocket):
        async def messages(self) -> AsyncIterator[Any]:
            if False:  # pragma: no cover
                yield None

    ws = OneShotWs()
    gamma = StubGammaClient([])
    detector = WhalesDetector(
        config=_make_config(ws_resubscribe_interval_seconds=3600),
        ws=ws,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        gamma_client=gamma,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        data_client=StubDataClient(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=StubMarketCache(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        wallet_first_seen=StubFirstSeen(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    )

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    # Yield control briefly so the task hits its first await.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
        await task

    assert ws.connected is True
    # Subscription loop ran at least one iteration.
    assert ws.subscribe_calls == [] or ws.subscribe_calls
