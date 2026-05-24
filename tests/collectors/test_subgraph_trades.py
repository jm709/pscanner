"""Unit tests for SubgraphTradeCollector internals (#152)."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.collectors import subgraph_trades as subgraph_trades_mod
from pscanner.collectors.subgraph_trades import (
    DETECTOR_TAG,
    SUBGRAPH_ID,
    SubgraphTradeCollector,
    _build_where_clause,
    _compute_copy_direction,
    _serialize_where_inline,
)
from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import SubgraphTradeCollectorConfig
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.token_resolver import ResolvedToken
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    SubgraphWatchStateRepo,
    WatchlistRepo,
)
from pscanner.util.clock import FakeClock


def test_compute_copy_direction_maker_buy() -> None:
    direction = _compute_copy_direction(maker="0xAA", taker="0xBB", side=0, watchlist={"0xaa"})
    assert direction == "BUY"


def test_compute_copy_direction_taker_sell_is_buy() -> None:
    # watchlist == taker AND side == 1 -> taker bought (hit a sell order).
    direction = _compute_copy_direction(maker="0xAA", taker="0xBB", side=1, watchlist={"0xbb"})
    assert direction == "BUY"


def test_compute_copy_direction_maker_sell_is_skip() -> None:
    direction = _compute_copy_direction(maker="0xAA", taker="0xBB", side=1, watchlist={"0xaa"})
    assert direction == "SKIP"


def test_compute_copy_direction_taker_buy_is_skip() -> None:
    direction = _compute_copy_direction(maker="0xAA", taker="0xBB", side=0, watchlist={"0xbb"})
    assert direction == "SKIP"


def test_compute_copy_direction_neither_side_watched() -> None:
    direction = _compute_copy_direction(maker="0xAA", taker="0xBB", side=0, watchlist={"0xcc"})
    assert direction == "SKIP"


def test_serialize_where_inline_renders_object_literal() -> None:
    rendered = _serialize_where_inline({"timestamp_gte": "100", "maker_in": ["0xaa", "0xbb"]})
    # GraphQL object literals do NOT quote keys.
    assert rendered == '{timestamp_gte:"100",maker_in:["0xaa","0xbb"]}'


def test_build_where_clause_repeats_timestamp_inside_or() -> None:
    where = _build_where_clause(["0xaa", "0xbb"], 1_700_000_000)
    assert where == {
        "or": [
            {"timestamp_gte": "1700000000", "maker_in": ["0xaa", "0xbb"]},
            {"timestamp_gte": "1700000000", "taker_in": ["0xaa", "0xbb"]},
        ],
    }


def test_constants_exposed_for_callers() -> None:
    assert DETECTOR_TAG == "subgraph_copy"
    assert SUBGRAPH_ID == "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"


@pytest.fixture
def daemon_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = init_db(tmp_path / "pscanner.sqlite3")
    yield db
    db.close()


@pytest.fixture
def corpus_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    db = init_corpus_db(tmp_path / "corpus.sqlite3")
    yield db
    db.close()


def _make_sink(daemon_conn: sqlite3.Connection) -> AlertSink:
    return AlertSink(AlertsRepo(daemon_conn))


def _make_event(
    *, tx: str, ts: int, maker: str, taker: str, side: int, token_id: str
) -> dict[str, object]:
    return {
        "transactionHash": tx,
        "timestamp": str(ts),
        "maker": {"id": maker},
        "taker": {"id": taker},
        "market": {"id": token_id},
        "tokenId": token_id,
        "side": side,
        "price": "0.5",
        "size": "1.0",
    }


async def test_poll_once_empty_watchlist_short_circuits(
    daemon_conn: sqlite3.Connection, corpus_conn: sqlite3.Connection
) -> None:
    registry = WatchlistRegistry(WatchlistRepo(daemon_conn))  # empty
    cfg = SubgraphTradeCollectorConfig(enabled=True)
    sub_client = MagicMock()
    sub_client.query = AsyncMock()
    collector = SubgraphTradeCollector(
        config=cfg,
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=_make_sink(daemon_conn),
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    with capture_logs() as logs:
        await collector.poll_once()
    sub_client.query.assert_not_called()
    assert any(log["event"] == "subgraph_trades.empty_watchlist" for log in logs)


async def test_poll_once_emits_alert_for_watchlist_buy(
    daemon_conn: sqlite3.Connection,
    corpus_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = WatchlistRepo(daemon_conn)
    repo.upsert(address="0xaa", source="manual", reason="test")
    registry = WatchlistRegistry(repo)

    cfg = SubgraphTradeCollectorConfig(enabled=True)
    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                _make_event(
                    tx="0xtx1",
                    ts=1_700_000_100,
                    maker="0xAA",
                    taker="0xBB",
                    side=0,  # maker BUY -> watchlist wallet accumulates
                    token_id="9999",  # noqa: S106 (false positive — not a credential)
                ),
            ],
            "_meta": {"block": {"number": "1", "timestamp": str(int(time.time()))}},
        }
    )

    async def fake_resolve(
        *,
        token_id: AssetId,
        asset_index: AssetIndexRepo,
        market_cache: MarketCacheRepo,
        gamma: object,
    ) -> ResolvedToken:
        return ResolvedToken(
            condition_id=ConditionId("0xcond"),
            asset_id=AssetId(str(token_id)),
            outcome_name="Yes",
            outcome_index=0,
        )

    monkeypatch.setattr(subgraph_trades_mod, "resolve_token", fake_resolve)

    sink = _make_sink(daemon_conn)
    emitted: list[Alert] = []
    sink.subscribe(emitted.append)
    collector = SubgraphTradeCollector(
        config=cfg,
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=sink,
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    await collector.poll_once()

    assert len(emitted) == 1
    alert = emitted[0]
    assert alert.detector == "subgraph_copy"
    assert alert.alert_key == "subgraph:0xtx1:Yes"
    assert alert.body["source_wallet"].lower() == "0xaa"
    assert alert.body["condition_id"] == "0xcond"
    assert alert.body["outcome"] == "Yes"
    assert alert.body["ts"] == 1_700_000_100


async def test_poll_once_persists_new_last_seen_ts(
    daemon_conn: sqlite3.Connection,
    corpus_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    WatchlistRepo(daemon_conn).upsert(address="0xaa", source="manual", reason="t")
    registry = WatchlistRegistry(WatchlistRepo(daemon_conn))
    state_repo = SubgraphWatchStateRepo(daemon_conn)
    state_repo.set_last_seen_ts(1_700_000_000)

    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                _make_event(
                    tx="0xtx",
                    ts=1_700_000_200,
                    maker="0xAA",
                    taker="0xBB",
                    side=0,
                    token_id="t1",  # noqa: S106 (false positive — not a credential)
                ),
            ],
            "_meta": {"block": {"number": "1", "timestamp": str(int(time.time()))}},
        }
    )

    async def fake_resolve(
        *,
        token_id: AssetId,
        asset_index: AssetIndexRepo,
        market_cache: MarketCacheRepo,
        gamma: object,
    ) -> ResolvedToken:
        return ResolvedToken(
            condition_id=ConditionId("0xcond"),
            asset_id=AssetId(str(token_id)),
            outcome_name="Yes",
            outcome_index=0,
        )

    monkeypatch.setattr(subgraph_trades_mod, "resolve_token", fake_resolve)

    collector = SubgraphTradeCollector(
        config=SubgraphTradeCollectorConfig(enabled=True),
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=_make_sink(daemon_conn),
        state_repo=state_repo,
        clock=FakeClock(),
    )
    await collector.poll_once()
    assert state_repo.get_last_seen_ts() == 1_700_000_200


async def test_poll_once_skips_sells_silently(
    daemon_conn: sqlite3.Connection, corpus_conn: sqlite3.Connection
) -> None:
    WatchlistRepo(daemon_conn).upsert(address="0xaa", source="manual", reason="t")
    registry = WatchlistRegistry(WatchlistRepo(daemon_conn))
    sub_client = MagicMock()
    sub_client.query = AsyncMock(
        return_value={
            "orderFilledEvents": [
                _make_event(
                    tx="0xtx",
                    ts=1_700_000_300,
                    maker="0xAA",
                    taker="0xBB",
                    side=1,  # maker SELL -> watchlist reduces -> SKIP
                    token_id="t1",  # noqa: S106 (false positive — not a credential)
                ),
            ],
            "_meta": {"block": {"number": "1", "timestamp": str(int(time.time()))}},
        }
    )
    sink = _make_sink(daemon_conn)
    emitted: list[Alert] = []
    sink.subscribe(emitted.append)
    collector = SubgraphTradeCollector(
        config=SubgraphTradeCollectorConfig(enabled=True),
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=registry,
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=sink,
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    await collector.poll_once()
    assert emitted == []


async def test_run_stops_on_stop_event(
    daemon_conn: sqlite3.Connection, corpus_conn: sqlite3.Connection
) -> None:
    cfg = SubgraphTradeCollectorConfig(enabled=True, poll_interval_seconds=0.01)
    sub_client = MagicMock()
    sub_client.query = AsyncMock(return_value={"orderFilledEvents": [], "_meta": {"block": {}}})
    collector = SubgraphTradeCollector(
        config=cfg,
        subgraph_client=sub_client,
        gamma_client=MagicMock(),
        watchlist=WatchlistRegistry(WatchlistRepo(daemon_conn)),
        asset_index=AssetIndexRepo(corpus_conn),
        market_cache=MarketCacheRepo(daemon_conn),
        sink=_make_sink(daemon_conn),
        state_repo=SubgraphWatchStateRepo(daemon_conn),
        clock=FakeClock(),
    )
    stop = asyncio.Event()

    async def _stopper() -> None:
        await asyncio.sleep(0.05)
        stop.set()

    await asyncio.gather(collector.run(stop), _stopper())
