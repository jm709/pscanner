"""Tests for ``ConvergenceDetector`` (DC-1.8.B).

Exercise :meth:`ConvergenceDetector.evaluate` with synthesised collaborators.
All collaborators are stubbed in-process — no network, no SQLite. Shared
trade-callback plumbing (``handle_trade_sync`` task tracking, ``run``
parking) is covered once in :mod:`tests.detectors.test_trade_driven`.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import ConvergenceConfig, SmartMoneyConfig
from pscanner.detectors.convergence import ConvergenceDetector
from pscanner.store.repo import (
    CachedMarket,
    TrackedWalletCategory,
    WalletTrade,
)

_NOW = int(time.time())
_CONDITION_ID = "0xcond-1"
_EVENT_SLUG = "will-x-happen-2026"
_MARKET_ID = "market-1"
_TRADER = "0xtrader"
_OTHER_WALLET = "0xother"
_THIRD_WALLET = "0xthird"


def _make_trade(**overrides: Any) -> WalletTrade:
    """Build a ``CONFIRMED`` ``WalletTrade`` with sensible defaults."""
    base: dict[str, Any] = {
        "transaction_hash": "0xtxhash1",
        "asset_id": "asset-1",
        "side": "BUY",
        "wallet": _TRADER,
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
        "condition_id": _CONDITION_ID,
        "event_slug": _EVENT_SLUG,
    }
    base.update(overrides)
    return CachedMarket(**base)


def _make_category_row(wallet: str, category: str = "thesis") -> TrackedWalletCategory:
    """Build a ``TrackedWalletCategory`` row for a given wallet."""
    return TrackedWalletCategory(
        wallet=wallet,
        category=category,
        position_count=30,
        win_count=20,
        mean_edge=0.10,
        weighted_edge=0.12,
        excess_pnl_usd=5000.0,
        total_stake_usd=20000.0,
        last_refreshed_at=_NOW,
    )


class StubMarketCache:
    """In-memory ``MarketCacheRepo`` keyed by condition_id."""

    def __init__(self, by_condition: dict[str, CachedMarket] | None = None) -> None:
        self._by_condition: dict[str, CachedMarket] = dict(by_condition or {})

    def get_by_condition_id(self, condition_id: str) -> CachedMarket | None:
        return self._by_condition.get(condition_id)


class StubEventTagCache:
    """In-memory ``EventTagCacheRepo`` keyed by whatever string the detector passes."""

    def __init__(self, tags_by_key: dict[str, list[str]] | None = None) -> None:
        self._tags: dict[str, list[str]] = dict(tags_by_key or {})
        self.calls: list[str] = []

    def get(self, key: str) -> list[str] | None:
        self.calls.append(key)
        return self._tags.get(key)


class StubCategoryRepo:
    """In-memory ``TrackedWalletCategoriesRepo.list_by_category`` stub."""

    def __init__(self, by_category: dict[str, list[TrackedWalletCategory]] | None = None) -> None:
        self._rows: dict[str, list[TrackedWalletCategory]] = dict(by_category or {})
        self.calls: list[tuple[str, float, float, int]] = []

    def list_by_category(
        self,
        category: str,
        *,
        min_edge: float,
        min_excess_pnl_usd: float,
        min_resolved: int,
    ) -> list[TrackedWalletCategory]:
        self.calls.append((category, min_edge, min_excess_pnl_usd, min_resolved))
        return list(self._rows.get(category, []))


class StubTradesRepo:
    """In-memory ``WalletTradesRepo`` stub for the methods the detector touches."""

    def __init__(
        self,
        *,
        distinct_by_condition: dict[str, set[str]] | None = None,
        recent_by_wallet: dict[str, list[WalletTrade]] | None = None,
    ) -> None:
        self._distinct: dict[str, set[str]] = dict(distinct_by_condition or {})
        self._recent: dict[str, list[WalletTrade]] = dict(recent_by_wallet or {})
        self.distinct_calls: list[tuple[str, int]] = []

    def distinct_wallets_for_condition(self, condition_id: str, *, since: int) -> set[str]:
        self.distinct_calls.append((condition_id, since))
        return set(self._distinct.get(condition_id, set()))

    def recent_for_wallet(self, wallet: str, *, limit: int = 100) -> list[WalletTrade]:
        del limit
        return list(self._recent.get(wallet, []))


class CapturingSink:
    """Collects every alert ``emit`` is called with."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    async def emit(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


def _make_detector(
    *,
    config: ConvergenceConfig | None = None,
    smart_config: SmartMoneyConfig | None = None,
    market_cache: StubMarketCache | None = None,
    event_tag_cache: StubEventTagCache | None = None,
    category_repo: StubCategoryRepo | None = None,
    trades_repo: StubTradesRepo | None = None,
    sink: CapturingSink | None = None,
) -> ConvergenceDetector:
    """Build a ``ConvergenceDetector`` wired to stubs (with defaults)."""
    detector = ConvergenceDetector(
        config=config or ConvergenceConfig(),
        trades_repo=trades_repo or StubTradesRepo(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        category_repo=category_repo or StubCategoryRepo(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache or StubMarketCache(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        event_tag_cache=event_tag_cache or StubEventTagCache(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        smart_money_config=smart_config or SmartMoneyConfig(),
    )
    if sink is not None:
        detector._sink = sink  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
    return detector


async def test_two_thesis_wallets_same_condition_within_window_emits_med_alert() -> None:
    sink = CapturingSink()
    cached = _make_cached_market()
    market_cache = StubMarketCache({_CONDITION_ID: cached})
    event_tag_cache = StubEventTagCache({_EVENT_SLUG: ["Politics"]})
    category_repo = StubCategoryRepo(
        {"thesis": [_make_category_row(_TRADER), _make_category_row(_OTHER_WALLET)]},
    )
    trades_repo = StubTradesRepo(
        distinct_by_condition={_CONDITION_ID: {_TRADER, _OTHER_WALLET}},
    )
    detector = _make_detector(
        market_cache=market_cache,
        event_tag_cache=event_tag_cache,
        category_repo=category_repo,
        trades_repo=trades_repo,
        sink=sink,
    )

    await detector.evaluate(_make_trade())

    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.detector == "convergence"
    assert alert.severity == "med"
    assert alert.body["category"] == "thesis"
    assert set(alert.body["convergent_wallets"]) == {_TRADER, _OTHER_WALLET}
    assert alert.body["convergent_count"] == 2
    assert alert.body["condition_id"] == _CONDITION_ID
    assert alert.body["event_slug"] == _EVENT_SLUG


async def test_three_or_more_convergent_wallets_emits_high_severity() -> None:
    sink = CapturingSink()
    cached = _make_cached_market()
    market_cache = StubMarketCache({_CONDITION_ID: cached})
    event_tag_cache = StubEventTagCache({_EVENT_SLUG: ["Politics"]})
    wallets = {_TRADER, _OTHER_WALLET, _THIRD_WALLET}
    category_repo = StubCategoryRepo(
        {"thesis": [_make_category_row(w) for w in wallets]},
    )
    trades_repo = StubTradesRepo(distinct_by_condition={_CONDITION_ID: wallets})
    detector = _make_detector(
        market_cache=market_cache,
        event_tag_cache=event_tag_cache,
        category_repo=category_repo,
        trades_repo=trades_repo,
        sink=sink,
    )

    await detector.evaluate(_make_trade())

    assert len(sink.alerts) == 1
    assert sink.alerts[0].severity == "high"
    assert sink.alerts[0].body["convergent_count"] == 3


async def test_market_cache_miss_skips_silently() -> None:
    sink = CapturingSink()
    detector = _make_detector(market_cache=StubMarketCache(), sink=sink)

    await detector.evaluate(_make_trade())

    assert sink.alerts == []


async def test_cached_market_without_event_slug_skips() -> None:
    sink = CapturingSink()
    cached = _make_cached_market(event_slug=None)
    market_cache = StubMarketCache({_CONDITION_ID: cached})
    detector = _make_detector(market_cache=market_cache, sink=sink)

    await detector.evaluate(_make_trade())

    assert sink.alerts == []


async def test_event_tag_cache_miss_skips() -> None:
    sink = CapturingSink()
    cached = _make_cached_market()
    market_cache = StubMarketCache({_CONDITION_ID: cached})
    event_tag_cache = StubEventTagCache()  # empty -> get returns None
    detector = _make_detector(
        market_cache=market_cache,
        event_tag_cache=event_tag_cache,
        sink=sink,
    )

    await detector.evaluate(_make_trade())

    assert sink.alerts == []
    # Confirm the detector did consult the cache for the right key.
    assert event_tag_cache.calls == [_EVENT_SLUG]


async def test_sports_market_with_only_one_smart_wallet_does_not_alert() -> None:
    """Sports tag classifies the event as ``sports``; only the trader is in
    that roster so the convergent set is below the minimum."""
    sink = CapturingSink()
    cached = _make_cached_market()
    market_cache = StubMarketCache({_CONDITION_ID: cached})
    event_tag_cache = StubEventTagCache({_EVENT_SLUG: ["Sports"]})
    category_repo = StubCategoryRepo(
        {"sports": [_make_category_row(_TRADER, category="sports")]},
    )
    trades_repo = StubTradesRepo(distinct_by_condition={_CONDITION_ID: {_TRADER}})
    detector = _make_detector(
        market_cache=market_cache,
        event_tag_cache=event_tag_cache,
        category_repo=category_repo,
        trades_repo=trades_repo,
        sink=sink,
    )

    await detector.evaluate(_make_trade())

    assert sink.alerts == []
    # Confirm category lookup used "sports" with sports-tier min_edge.
    assert len(category_repo.calls) == 1
    category, min_edge, _excess, _resolved = category_repo.calls[0]
    assert category == "sports"
    assert min_edge == pytest.approx(0.10)


async def test_window_filter_excludes_old_co_traders() -> None:
    """Only one wallet falls inside the 48h thesis window so no alert fires."""
    sink = CapturingSink()
    cached = _make_cached_market()
    market_cache = StubMarketCache({_CONDITION_ID: cached})
    event_tag_cache = StubEventTagCache({_EVENT_SLUG: ["Politics"]})
    category_repo = StubCategoryRepo(
        {"thesis": [_make_category_row(_TRADER), _make_category_row(_OTHER_WALLET)]},
    )
    # Distinct query inside the window returns only the trader; the other
    # wallet's trade was older than 48h.
    trades_repo = StubTradesRepo(distinct_by_condition={_CONDITION_ID: {_TRADER}})
    detector = _make_detector(
        market_cache=market_cache,
        event_tag_cache=event_tag_cache,
        category_repo=category_repo,
        trades_repo=trades_repo,
        sink=sink,
    )

    trade = _make_trade()
    await detector.evaluate(trade)

    assert sink.alerts == []
    # Verify the detector queried the trades repo with the thesis window.
    assert trades_repo.distinct_calls == [(_CONDITION_ID, trade.timestamp - 48 * 3600)]
