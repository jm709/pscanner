"""Tests for :class:`PaperTrader` against the surviving evaluator chain.

The 6 per-detector evaluators were deleted alongside their detector classes;
only :class:`SubgraphCopyEvaluator` survives. These tests exercise the
PaperTrader pipeline (evaluate -> parse -> quality -> resolve -> size ->
insert) end-to-end using a subgraph_copy alert as the sole input shape.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest.mock import AsyncMock

import pytest

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import (
    EvaluatorsConfig,
    PaperTradingConfig,
    SubgraphCopyEvaluatorConfig,
)
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    AlertsRepo,
    CachedMarket,
    MarketCacheRepo,
    MarketTicksRepo,
    PaperTradesRepo,
    WatchlistRepo,
)
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator
from pscanner.strategies.paper_trader import PaperTrader

_NOW = 1_700_000_000


def _subgraph_copy_alert(
    *,
    wallet: str = "0xwallet1",
    condition_id: str = "0xcond-1",
    side: str = "Yes",
    tx_hash: str = "0xtxabc",
    alert_key: str = "subgraph:0xwallet1:0xcond-1:Yes",
) -> Alert:
    return Alert(
        detector="subgraph_copy",
        alert_key=alert_key,
        severity="med",
        title=f"copy {wallet[:8]} {side}",
        body={
            "condition_id": condition_id,
            "outcome": side,
            "source_wallet": wallet,
            "tx_hash": tx_hash,
            "ts": _NOW,
        },
        created_at=_NOW,
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
) -> PaperTrader:
    paper_trades = PaperTradesRepo(tmp_db)
    watchlist = WatchlistRepo(tmp_db)
    evaluator = SubgraphCopyEvaluator(
        config=cfg.evaluators.subgraph_copy,
        watchlist_repo=watchlist,
        paper_trades=paper_trades,
    )
    return PaperTrader(
        config=cfg,
        evaluators=[evaluator],
        market_cache=MarketCacheRepo(tmp_db),
        paper_trades=paper_trades,
        market_ticks=MarketTicksRepo(tmp_db),
        data_client=data_client or _default_data_client(),
        gamma_client=gamma_client or _default_gamma_client(),
        alerts_repo=AlertsRepo(tmp_db),
    )


def _enabled_cfg(**overrides: object) -> PaperTradingConfig:
    base = {
        "enabled": True,
        "starting_bankroll_usd": 1000.0,
        "min_position_cost_usd": 0.50,
        "evaluators": EvaluatorsConfig(
            subgraph_copy=SubgraphCopyEvaluatorConfig(
                enabled=True,
                position_fraction=0.01,
            ),
        ),
    }
    base.update(overrides)
    return PaperTradingConfig(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_evaluate_books_subgraph_copy_entry(tmp_db: sqlite3.Connection) -> None:
    """End-to-end: subgraph_copy alert -> ParsedSignal -> entry row inserted."""
    _cache_market(MarketCacheRepo(tmp_db))
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = _build_trader(tmp_db, _enabled_cfg())

    await trader.evaluate(_subgraph_copy_alert())

    rows = tmp_db.execute(
        "SELECT triggering_alert_detector, source_wallet, outcome, fill_price, cost_usd "
        "FROM paper_trades WHERE trade_kind='entry'"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["triggering_alert_detector"] == "subgraph_copy"
    assert row["source_wallet"] == "0xwallet1"
    assert row["outcome"] == "Yes"
    assert row["fill_price"] == 0.5
    assert row["cost_usd"] == 10.0  # 1000 * 1%


@pytest.mark.asyncio
async def test_evaluate_skips_when_market_uncached(tmp_db: sqlite3.Connection) -> None:
    """Cache miss + no backfill data => no entry row, no crash."""
    trader = _build_trader(tmp_db, _enabled_cfg())

    await trader.evaluate(_subgraph_copy_alert())

    rows = tmp_db.execute("SELECT 1 FROM paper_trades").fetchall()
    assert rows == []


@pytest.mark.asyncio
async def test_evaluate_skips_on_no_evaluator(tmp_db: sqlite3.Connection) -> None:
    """Alert from an unknown detector falls through with no rows inserted."""
    _cache_market(MarketCacheRepo(tmp_db))
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = _build_trader(tmp_db, _enabled_cfg())

    # Build an alert whose detector is not in the SubgraphCopyEvaluator
    # accept-set. We bypass the DetectorName Literal narrowing by
    # constructing the Alert through the dataclass directly.
    alien = Alert(
        detector="subgraph_copy",  # only valid literal value
        alert_key="x:alien",
        severity="low",
        title="alien",
        body={},  # SubgraphCopyEvaluator.parse returns [] on body-shape mismatch
        created_at=_NOW,
    )

    await trader.evaluate(alien)

    rows = tmp_db.execute("SELECT 1 FROM paper_trades").fetchall()
    assert rows == []


@pytest.mark.asyncio
async def test_evaluate_dedupes_on_duplicate_alert_key(
    tmp_db: sqlite3.Connection,
) -> None:
    """Two evaluations of the same alert_key yield exactly one entry row."""
    _cache_market(MarketCacheRepo(tmp_db))
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = _build_trader(tmp_db, _enabled_cfg())

    alert = _subgraph_copy_alert(alert_key="dedupe-me")
    await trader.evaluate(alert)
    await trader.evaluate(alert)

    rows = tmp_db.execute("SELECT 1 FROM paper_trades WHERE trade_kind='entry'").fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_handle_alert_sync_spawns_evaluate_task(
    tmp_db: sqlite3.Connection,
) -> None:
    """The sync callback hook spawns an async evaluation that books a row."""
    _cache_market(MarketCacheRepo(tmp_db))
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)
    trader = _build_trader(tmp_db, _enabled_cfg())
    sink = AlertSink(AlertsRepo(tmp_db))
    sink.subscribe(trader.handle_alert_sync)

    await sink.emit(_subgraph_copy_alert())
    # let the spawned task finish
    for _ in range(20):
        await asyncio.sleep(0)
    await trader.aclose()

    rows = tmp_db.execute("SELECT 1 FROM paper_trades WHERE trade_kind='entry'").fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_replay_unbooked_drives_unbooked_alerts_through_pipeline(
    tmp_db: sqlite3.Connection,
) -> None:
    """``replay_unbooked`` re-runs in-window alerts that have no paper_trades row."""
    _cache_market(MarketCacheRepo(tmp_db))
    _seed_tick(tmp_db, asset_id="asset-yes", best_ask=0.5)

    cfg = _enabled_cfg(replay_lookback_seconds=3600)
    # Pre-insert the alert into the AlertsRepo so replay_unbooked finds it.
    alert = _subgraph_copy_alert(alert_key="replay-me")
    AlertsRepo(tmp_db).insert_if_new(alert)
    # Backdate created_at within the replay window so it qualifies.
    tmp_db.execute(
        "UPDATE alerts SET created_at = ? WHERE alert_key = ?",
        (int(time.time()) - 60, alert.alert_key),
    )
    tmp_db.commit()

    trader = _build_trader(tmp_db, cfg)
    n = await trader.replay_unbooked()

    assert n == 1
    rows = tmp_db.execute("SELECT 1 FROM paper_trades WHERE trade_kind='entry'").fetchall()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_replay_unbooked_returns_zero_when_disabled(
    tmp_db: sqlite3.Connection,
) -> None:
    """``replay_lookback_seconds=0`` short-circuits without touching the repo."""
    trader = _build_trader(tmp_db, _enabled_cfg(replay_lookback_seconds=0))
    assert await trader.replay_unbooked() == 0
