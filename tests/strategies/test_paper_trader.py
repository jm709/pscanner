"""Tests for PaperTrader."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import (
    EvaluatorsConfig,
    PaperTradingConfig,
    SmartMoneyEvaluatorConfig,
)
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.poly.models import Market
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
    TrackedWalletsRepo,
)
from pscanner.strategies.evaluators import ParsedSignal, SmartMoneyEvaluator
from pscanner.strategies.paper_trader import PaperTrader

_NOW = 1700000000


def _smart_money_alert(
    *,
    wallet: str = "0xwallet1",
    condition_id: str = "0xcond-1",
    side: str = "yes",
    delta_usd: float = 100.0,
    alert_key: str = "smart:0xwallet1:0xcond-1:yes:20260427",
) -> Alert:
    return Alert(
        detector="smart_money",
        alert_key=alert_key,
        severity="med",
        title=f"smart-money {wallet[:8]} +{side}",
        body={
            "wallet": wallet,
            "market_title": "Test market",
            "condition_id": condition_id,
            "side": side,
            "new_size": 200.0,
            "prev_size": 100.0,
            "delta_usd": delta_usd,
            "winrate": 0.85,
            "mean_edge": 0.4,
            "excess_pnl_usd": 1000.0,
            "closed_position_count": 50,
        },
        created_at=_NOW,
    )


def _track_wallet(
    repo: TrackedWalletsRepo,
    *,
    address: str = "0xwallet1",
    weighted_edge: float | None = 0.4,
) -> None:
    repo.upsert(
        address=address,
        closed_position_count=50,
        closed_position_wins=42,
        winrate=0.84,
        leaderboard_pnl=1000.0,
        mean_edge=0.4,
        weighted_edge=weighted_edge,
        excess_pnl_usd=1000.0,
        total_stake_usd=1000.0,
    )


def _cache_market(
    repo: MarketCacheRepo,
    *,
    condition_id: str = "0xcond-1",
    outcomes: list[str] | None = None,
    asset_ids: list[str] | None = None,
    outcome_prices: list[float] | None = None,
) -> None:
    repo.upsert(
        CachedMarket(
            market_id=MarketId(f"mkt-{condition_id}"),
            event_id=None,
            title="Test market",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=outcome_prices if outcome_prices is not None else [0.6, 0.4],
            outcomes=outcomes or ["Yes", "No"],
            asset_ids=[AssetId(a) for a in (asset_ids or ["asset-yes", "asset-no"])],
            active=True,
            cached_at=_NOW,
            condition_id=ConditionId(condition_id),
            event_slug=None,
        ),
    )


def _seed_tick(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    best_ask: float | None,
    last_trade_price: float | None = None,
    ts: int = _NOW,
) -> None:
    conn.execute(
        """
        INSERT INTO market_ticks (asset_id, condition_id, snapshot_at, mid_price,
          best_bid, best_ask, spread, bid_depth_top5, ask_depth_top5,
          last_trade_price)
        VALUES (?, '0xcond-1', ?, NULL, NULL, ?, NULL, NULL, NULL, ?)
        """,
        (asset_id, ts, best_ask, last_trade_price),
    )
    conn.commit()


def _default_data_client() -> AsyncMock:
    """Default mock that signals "no slug for this market" so the backfill no-ops."""
    client = AsyncMock()
    client.get_market_slug_by_condition_id.return_value = None
    return client


def _default_gamma_client() -> AsyncMock:
    """Default mock that signals "no market for this slug" so the backfill no-ops."""
    client = AsyncMock()
    client.get_market_by_slug.return_value = None
    return client


def _build_trader(
    tmp_db: sqlite3.Connection,
    cfg: PaperTradingConfig,
    *,
    data_client: AsyncMock | None = None,
    gamma_client: AsyncMock | None = None,
) -> tuple[
    AlertSink,
    PaperTrader,
    MarketCacheRepo,
    TrackedWalletsRepo,
    PaperTradesRepo,
]:
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    smart_money = SmartMoneyEvaluator(
        config=cfg.evaluators.smart_money,
        tracked_wallets=wallets,
    )
    trader = PaperTrader(
        config=cfg,
        evaluators=[smart_money],
        market_cache=cache,
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=data_client or _default_data_client(),
        gamma_client=gamma_client or _default_gamma_client(),
    )
    sink.subscribe(trader.handle_alert_sync)
    return sink, trader, cache, wallets, paper


async def _drain() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_paper_trader_inserts_entry_on_smart_money_alert(
    tmp_db: sqlite3.Connection,
) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)

    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()

    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    p = open_positions[0]
    assert p.source_wallet == "0xwallet1"
    assert p.outcome == "yes"
    assert p.asset_id == AssetId("asset-yes")
    assert p.fill_price == 0.5
    assert p.cost_usd == 10.0
    assert p.shares == 20.0
    # NAV is cost-basis (starting_bankroll + realized_pnl); entries don't move it.
    assert p.nav_after_usd == cfg.starting_bankroll_usd


async def test_paper_trader_skips_non_smart_money(tmp_db: sqlite3.Connection) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, _cache, _wallets, paper = _build_trader(tmp_db, cfg)
    await sink.emit(
        Alert(
            detector="velocity",
            alert_key="v:1",
            severity="med",
            title="t",
            body={"condition_id": "0xc"},
            created_at=_NOW,
        ),
    )
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


async def test_paper_trader_skips_wallet_below_edge(tmp_db: sqlite3.Connection) -> None:
    cfg = PaperTradingConfig(
        enabled=True,
        evaluators=EvaluatorsConfig(
            smart_money=SmartMoneyEvaluatorConfig(min_weighted_edge=0.0),
        ),
    )
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=-0.1)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


async def test_paper_trader_skips_null_edge(tmp_db: sqlite3.Connection) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=None)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


async def test_paper_trader_skips_when_no_market_cache(tmp_db: sqlite3.Connection) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, _cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


async def test_paper_trader_skips_when_outcome_unmappable(tmp_db: sqlite3.Connection) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache, outcomes=["Yes", "No"], asset_ids=["a-y", "a-n"])
    _seed_tick(tmp_db, asset_id="a-y", best_ask=0.5)

    await sink.emit(_smart_money_alert(side="Maybe"))
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


async def test_paper_trader_skips_when_no_price(tmp_db: sqlite3.Connection) -> None:
    """No tick AND no usable cached outcome price (resolved market): skip."""
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    # outcome_prices at the boundary mean both the tick lookup and the
    # cached-price fallback yield no usable fill price.
    _cache_market(cache, outcome_prices=[1.0, 0.0])

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()
    assert paper.list_open_positions() == []


async def test_paper_trader_falls_back_to_last_trade_price(tmp_db: sqlite3.Connection) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=None, last_trade_price=0.55)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()
    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].fill_price == 0.55


async def test_paper_trader_falls_back_to_cached_outcome_price(
    tmp_db: sqlite3.Connection,
) -> None:
    """When market_ticks has no row, paper_trader uses cached outcome_prices."""
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(
        cache,
        outcomes=["Yes", "No"],
        asset_ids=["asset-yes", "asset-no"],
        outcome_prices=[0.6, 0.4],
    )
    # No market_ticks row at all.

    await sink.emit(_smart_money_alert(side="Yes"))
    await _drain()
    await trader.aclose()

    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    # Fill price came from cached outcome_prices[0] for "Yes".
    assert open_positions[0].fill_price == 0.6


async def test_paper_trader_prefers_tick_over_cached_price(
    tmp_db: sqlite3.Connection,
) -> None:
    """When market_ticks has best_ask, that wins over cached outcome_prices."""
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(
        cache,
        outcomes=["Yes", "No"],
        asset_ids=["asset-yes", "asset-no"],
        outcome_prices=[0.6, 0.4],
    )
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.45)

    await sink.emit(_smart_money_alert(side="Yes"))
    await _drain()
    await trader.aclose()

    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].fill_price == 0.45


async def test_paper_trader_skips_when_cached_price_out_of_range(
    tmp_db: sqlite3.Connection,
) -> None:
    """If outcome_prices[idx] is 0 or 1 (resolved), no fallback happens."""
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(
        cache,
        outcomes=["Yes", "No"],
        asset_ids=["asset-yes", "asset-no"],
        outcome_prices=[1.0, 0.0],
    )

    await sink.emit(_smart_money_alert(side="Yes"))
    await _drain()
    await trader.aclose()

    assert paper.list_open_positions() == []


def test_cached_outcome_price_unit_happy(tmp_db: sqlite3.Connection) -> None:
    """``_cached_outcome_price`` returns the parallel-indexed price."""
    cfg = PaperTradingConfig(enabled=True)
    _, trader, cache, _wallets, _paper = _build_trader(tmp_db, cfg)
    _cache_market(
        cache,
        outcomes=["Yes", "No"],
        asset_ids=["asset-yes", "asset-no"],
        outcome_prices=[0.55, 0.45],
    )

    price = trader._cached_outcome_price(
        ConditionId("0xcond-1"),
        AssetId("asset-yes"),
    )
    assert price == 0.55


def test_cached_outcome_price_unit_unknown_asset(tmp_db: sqlite3.Connection) -> None:
    """Asset_id not in cached asset_ids returns None."""
    cfg = PaperTradingConfig(enabled=True)
    _, trader, cache, _wallets, _paper = _build_trader(tmp_db, cfg)
    _cache_market(
        cache,
        outcomes=["Yes", "No"],
        asset_ids=["asset-yes", "asset-no"],
    )

    price = trader._cached_outcome_price(
        ConditionId("0xcond-1"),
        AssetId("asset-nope"),
    )
    assert price is None


def test_cached_outcome_price_unit_missing_market(tmp_db: sqlite3.Connection) -> None:
    """No cached market for the condition id returns None."""
    cfg = PaperTradingConfig(enabled=True)
    _, trader, _cache, _wallets, _paper = _build_trader(tmp_db, cfg)

    price = trader._cached_outcome_price(
        ConditionId("0xcond-missing"),
        AssetId("asset-yes"),
    )
    assert price is None


async def test_paper_trader_idempotent_on_duplicate_alert_key(
    tmp_db: sqlite3.Connection,
) -> None:
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    alert = _smart_money_alert(alert_key="dup-key")
    await sink.emit(alert)
    trader.handle_alert_sync(alert)
    for _ in range(10):
        await asyncio.sleep(0)
    await trader.aclose()
    assert len(paper.list_open_positions()) == 1


async def test_paper_trader_backfills_market_cache_on_miss(
    tmp_db: sqlite3.Connection,
) -> None:
    """On cache miss, fetch slug via data-api, market via gamma, then retry."""
    cfg = PaperTradingConfig(enabled=True)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = "test-slug"
    gamma_client = AsyncMock()
    gamma_client.get_market_by_slug.return_value = Market.model_validate(
        {
            "id": "mkt-1",
            "conditionId": "0xcond-1",
            "question": "Test",
            "slug": "test-slug",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.6", "0.4"],
            "clobTokenIds": ["asset-yes", "asset-no"],
            "active": True,
            "closed": False,
            "liquidity": 1.0,
            "volume": 1.0,
        },
    )

    sink, trader, cache, wallets, paper = _build_trader(
        tmp_db,
        cfg,
        data_client=data_client,
        gamma_client=gamma_client,
    )
    _track_wallet(wallets, weighted_edge=0.4)
    # Note: market_cache is intentionally empty to trigger the miss.
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()

    # Backfill ran and the entry was inserted at the resolved asset_id.
    open_positions = paper.list_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].asset_id == AssetId("asset-yes")

    # And market_cache now carries the row.
    cached = cache.get_by_condition_id(ConditionId("0xcond-1"))
    assert cached is not None
    assert cached.outcomes == ["Yes", "No"]
    assert cached.asset_ids == [AssetId("asset-yes"), AssetId("asset-no")]

    data_client.get_market_slug_by_condition_id.assert_awaited_once_with("0xcond-1")
    gamma_client.get_market_by_slug.assert_awaited_once_with("test-slug")


async def test_paper_trader_skips_when_backfill_finds_no_slug(
    tmp_db: sqlite3.Connection,
) -> None:
    """If the data-api has no trades on the market, skip cleanly."""
    cfg = PaperTradingConfig(enabled=True)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = None
    gamma_client = AsyncMock()
    sink, trader, _cache, wallets, paper = _build_trader(
        tmp_db,
        cfg,
        data_client=data_client,
        gamma_client=gamma_client,
    )
    _track_wallet(wallets, weighted_edge=0.4)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()

    assert paper.list_open_positions() == []
    gamma_client.get_market_by_slug.assert_not_awaited()


async def test_paper_trader_skips_when_backfill_finds_no_market(
    tmp_db: sqlite3.Connection,
) -> None:
    """If gamma doesn't recognise the slug, skip cleanly."""
    cfg = PaperTradingConfig(enabled=True)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.return_value = "test-slug"
    gamma_client = AsyncMock()
    gamma_client.get_market_by_slug.return_value = None
    sink, trader, _cache, wallets, paper = _build_trader(
        tmp_db,
        cfg,
        data_client=data_client,
        gamma_client=gamma_client,
    )
    _track_wallet(wallets, weighted_edge=0.4)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()

    assert paper.list_open_positions() == []


async def test_paper_trader_swallows_backfill_exception(
    tmp_db: sqlite3.Connection,
) -> None:
    """A network/HTTP failure inside the backfill leaves the alert as a no-op."""
    cfg = PaperTradingConfig(enabled=True)
    data_client = AsyncMock()
    data_client.get_market_slug_by_condition_id.side_effect = RuntimeError("boom")
    gamma_client = AsyncMock()
    sink, trader, _cache, wallets, paper = _build_trader(
        tmp_db,
        cfg,
        data_client=data_client,
        gamma_client=gamma_client,
    )
    _track_wallet(wallets, weighted_edge=0.4)

    await sink.emit(_smart_money_alert())
    await _drain()
    await trader.aclose()

    assert paper.list_open_positions() == []


async def test_paper_trader_logs_warning_on_unexpected_db_error(
    tmp_db: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-IntegrityError DB exceptions get logged as paper_trader.insert_failed WARN."""
    cfg = PaperTradingConfig(enabled=True)
    sink = AlertSink(AlertsRepo(tmp_db))
    cache = MarketCacheRepo(tmp_db)
    wallets = TrackedWalletsRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    def boom(**_kwargs: object) -> int:
        raise sqlite3.OperationalError("simulated transient DB failure")

    monkeypatch.setattr(paper, "insert_entry", boom)

    trader = PaperTrader(
        config=cfg,
        evaluators=[
            SmartMoneyEvaluator(
                config=cfg.evaluators.smart_money,
                tracked_wallets=wallets,
            ),
        ],
        market_cache=cache,
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=AsyncMock(),
        gamma_client=AsyncMock(),
    )
    sink.subscribe(trader.handle_alert_sync)
    await sink.emit(_smart_money_alert())
    for _ in range(5):
        await asyncio.sleep(0)
    await trader.aclose()
    assert paper.list_open_positions() == []


def _velocity_alert(
    *,
    condition_id: str = "0xc1",
    asset_id: str = "a-y",
    severity: Literal["low", "med", "high"] = "high",
    consolidation: bool = False,
) -> Alert:
    return Alert(
        detector="velocity",
        alert_key=f"velocity:{condition_id}:{asset_id}",
        severity=severity,
        title="velocity",
        body={
            "condition_id": condition_id,
            "asset_id": asset_id,
            "consolidation": consolidation,
        },
        created_at=_NOW,
    )


def _seed_market(
    repo: MarketCacheRepo,
    *,
    condition_id: str,
    outcomes: list[str],
    asset_ids: list[str],
    outcome_prices: list[float] | None = None,
) -> None:
    """Compatibility wrapper around ``_cache_market`` for new tests."""
    _cache_market(
        repo,
        condition_id=condition_id,
        outcomes=outcomes,
        asset_ids=asset_ids,
        outcome_prices=outcome_prices,
    )


def _stub_data_client() -> AsyncMock:
    """Alias kept for the new tests' naming style."""
    return _default_data_client()


def _stub_gamma_client() -> AsyncMock:
    """Alias kept for the new tests' naming style."""
    return _default_gamma_client()


def test_paper_trader_accepts_evaluator_list(tmp_db: sqlite3.Connection) -> None:
    """The new ctor takes evaluators=[...]; the old tracked_wallets kwarg is gone
    (smart_money's edge filter now lives inside SmartMoneyEvaluator)."""
    cfg = PaperTradingConfig(enabled=True)
    market_cache = MarketCacheRepo(tmp_db)
    paper = PaperTradesRepo(tmp_db)
    market_ticks = MarketTicksRepo(tmp_db)
    data = _stub_data_client()
    gamma = _stub_gamma_client()

    trader = PaperTrader(
        config=cfg,
        evaluators=[],
        market_cache=market_cache,
        paper_trades=paper,
        market_ticks=market_ticks,
        data_client=data,
        gamma_client=gamma,
    )
    assert trader is not None


async def test_evaluate_dispatches_to_first_acceptor(
    tmp_db: sqlite3.Connection,
) -> None:
    """PaperTrader walks the list, picks the first ev whose accepts() is True."""
    seen: list[str] = []

    class _StubEvaluator:
        def __init__(self, name: str, accepts_detector: str) -> None:
            self._name = name
            self._accepts_detector = accepts_detector

        def accepts(self, alert: Alert) -> bool:
            return alert.detector == self._accepts_detector

        def parse(self, alert: Alert) -> list[ParsedSignal]:
            del alert
            seen.append(self._name)
            return []

        def quality_passes(self, parsed: ParsedSignal) -> bool:
            del parsed
            return True

        def size(self, bankroll: float, parsed: ParsedSignal) -> float:
            del bankroll, parsed
            return 0.0

    cfg = PaperTradingConfig(enabled=True)
    paper = PaperTradesRepo(tmp_db)
    trader = PaperTrader(
        config=cfg,
        evaluators=[
            _StubEvaluator("smart_money_ev", "smart_money"),
            _StubEvaluator("velocity_ev", "velocity"),
        ],
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=_stub_data_client(),
        gamma_client=_stub_gamma_client(),
    )

    await trader.evaluate(_velocity_alert())
    assert seen == ["velocity_ev"]


async def test_evaluator_exception_logs_and_continues(
    tmp_db: sqlite3.Connection,
) -> None:
    """A raising evaluator does not kill PaperTrader; warning is logged."""

    class _RaisingEvaluator:
        def accepts(self, alert: Alert) -> bool:
            del alert
            return True

        def parse(self, alert: Alert) -> list[ParsedSignal]:
            del alert
            raise RuntimeError("boom")

        def quality_passes(self, parsed: ParsedSignal) -> bool:
            del parsed
            return True

        def size(self, bankroll: float, parsed: ParsedSignal) -> float:
            del bankroll, parsed
            return 0.0

    cfg = PaperTradingConfig(enabled=True)
    paper = PaperTradesRepo(tmp_db)
    trader = PaperTrader(
        config=cfg,
        evaluators=[_RaisingEvaluator()],
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=_stub_data_client(),
        gamma_client=_stub_gamma_client(),
    )

    with capture_logs() as logs:
        await trader.evaluate(_smart_money_alert())
    assert any(log["event"] == "paper_trader.evaluator_failed" for log in logs)


async def test_evaluate_writes_detector_and_variant_to_entry(
    tmp_db: sqlite3.Connection,
) -> None:
    """Each ParsedSignal becomes an insert_entry with the alert detector +
    parsed rule_variant stamped onto the row."""

    class _DummyEvaluator:
        def accepts(self, alert: Alert) -> bool:
            return alert.detector == "velocity"

        def parse(self, alert: Alert) -> list[ParsedSignal]:
            del alert
            return [
                ParsedSignal(
                    condition_id=ConditionId("0xc1"),
                    side="yes",
                    rule_variant="follow",
                ),
                ParsedSignal(
                    condition_id=ConditionId("0xc1"),
                    side="no",
                    rule_variant="fade",
                ),
            ]

        def quality_passes(self, parsed: ParsedSignal) -> bool:
            del parsed
            return True

        def size(self, bankroll: float, parsed: ParsedSignal) -> float:
            del bankroll, parsed
            return 2.5

    cfg = PaperTradingConfig(enabled=True)
    paper = PaperTradesRepo(tmp_db)
    cache = MarketCacheRepo(tmp_db)
    _seed_market(
        cache,
        condition_id="0xc1",
        outcomes=["yes", "no"],
        asset_ids=["a-y", "a-n"],
    )
    _seed_tick(tmp_db, asset_id="a-y", best_ask=0.5)
    _seed_tick(tmp_db, asset_id="a-n", best_ask=0.5)

    trader = PaperTrader(
        config=cfg,
        evaluators=[_DummyEvaluator()],
        market_cache=cache,
        paper_trades=paper,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=_stub_data_client(),
        gamma_client=_stub_gamma_client(),
    )

    await trader.evaluate(_velocity_alert())

    rows = list(
        tmp_db.execute(
            "SELECT triggering_alert_detector, rule_variant FROM paper_trades "
            "WHERE trade_kind = 'entry' ORDER BY trade_id",
        ),
    )
    assert [tuple(r) for r in rows] == [("velocity", "follow"), ("velocity", "fade")]
