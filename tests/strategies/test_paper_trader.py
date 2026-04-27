"""Tests for PaperTrader."""

from __future__ import annotations

import asyncio
import sqlite3

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import PaperTradingConfig
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    PaperTradesRepo,
    TrackedWalletsRepo,
)
from pscanner.strategies.paper_trader import (
    PaperTrader,
    _size_trade,
)

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
) -> None:
    repo.upsert(
        CachedMarket(
            market_id=MarketId(f"mkt-{condition_id}"),
            event_id=None,
            title="Test market",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.6, 0.4],
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


def _build_trader(
    tmp_db: sqlite3.Connection, cfg: PaperTradingConfig
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
    trader = PaperTrader(
        config=cfg,
        market_cache=cache,
        tracked_wallets=wallets,
        paper_trades=paper,
        conn=tmp_db,
    )
    sink.subscribe(trader.handle_alert_sync)
    return sink, trader, cache, wallets, paper


async def _drain() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def test_size_trade_happy_path() -> None:
    cfg = PaperTradingConfig(
        enabled=True,
        starting_bankroll_usd=1000.0,
        position_fraction=0.01,
        min_position_cost_usd=0.5,
    )
    result = _size_trade(nav=1000.0, fill_price=0.5, cfg=cfg)
    assert result is not None
    cost, shares = result
    assert cost == 10.0
    assert shares == 20.0


def test_size_trade_below_minimum_returns_none() -> None:
    cfg = PaperTradingConfig(min_position_cost_usd=0.50, position_fraction=0.01)
    assert _size_trade(nav=40.0, fill_price=0.5, cfg=cfg) is None


def test_size_trade_bad_fill_price_returns_none() -> None:
    cfg = PaperTradingConfig()
    assert _size_trade(nav=1000.0, fill_price=0.0, cfg=cfg) is None
    assert _size_trade(nav=1000.0, fill_price=1.0, cfg=cfg) is None
    assert _size_trade(nav=1000.0, fill_price=-0.1, cfg=cfg) is None
    assert _size_trade(nav=1000.0, fill_price=1.5, cfg=cfg) is None


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
    cfg = PaperTradingConfig(enabled=True, min_weighted_edge=0.0)
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
    cfg = PaperTradingConfig(enabled=True)
    sink, trader, cache, wallets, paper = _build_trader(tmp_db, cfg)
    _track_wallet(wallets, weighted_edge=0.4)
    _cache_market(cache)

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
