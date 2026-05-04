"""Tests for :mod:`pscanner.kalshi.repos` — CRUD against ``tmp_db``."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from pscanner.kalshi.models import KalshiMarket, KalshiOrderbook, KalshiTrade
from pscanner.kalshi.repos import (
    KalshiMarketsRepo,
    KalshiOrderbookSnapshotsRepo,
    KalshiTradesRepo,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MARKET_PAYLOAD = {
    "ticker": "KXELONMARS-99",
    "event_ticker": "KXELONMARS-99",
    "title": "Will Elon Musk visit Mars?",
    "status": "active",
    "market_type": "binary",
    "open_time": "2026-01-01T00:00:00Z",
    "close_time": "2099-01-01T00:00:00Z",
    "expected_expiration_time": "2099-01-01T00:00:00Z",
    "yes_sub_title": "Yes",
    "no_sub_title": "No",
    "last_price_dollars": "0.0900",
    "yes_bid_dollars": "0.0800",
    "yes_ask_dollars": "0.1000",
    "no_bid_dollars": "0.9000",
    "no_ask_dollars": "0.9200",
    "volume_fp": "12345.00",
    "volume_24h_fp": "500.00",
    "open_interest_fp": "1000.00",
}

_TRADE_PAYLOAD_1 = {
    "trade_id": "trade-aaa",
    "ticker": "KXELONMARS-99",
    "taker_side": "yes",
    "yes_price_dollars": "0.0900",
    "no_price_dollars": "0.9100",
    "count_fp": "1.00",
    "created_time": "2026-05-04T12:36:03Z",
}

_TRADE_PAYLOAD_2 = {
    "trade_id": "trade-bbb",
    "ticker": "KXELONMARS-99",
    "taker_side": "no",
    "yes_price_dollars": "0.0800",
    "no_price_dollars": "0.9200",
    "count_fp": "2.50",
    "created_time": "2026-05-04T13:00:00Z",
}

_ORDERBOOK_PAYLOAD = {
    "orderbook_fp": {
        "yes_dollars": [["0.0800", "100.00"], ["0.0700", "200.00"]],
        "no_dollars": [["0.9000", "50.00"]],
    }
}


def _sample_market() -> KalshiMarket:
    return KalshiMarket.model_validate(_MARKET_PAYLOAD)


def _sample_trade(payload: dict) -> KalshiTrade:
    return KalshiTrade.model_validate(payload)


def _sample_orderbook() -> KalshiOrderbook:
    return KalshiOrderbook.model_validate(_ORDERBOOK_PAYLOAD)


# ---------------------------------------------------------------------------
# KalshiMarketsRepo
# ---------------------------------------------------------------------------


def test_markets_repo_upsert_and_get(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    repo.upsert(_sample_market())

    row = repo.get("KXELONMARS-99")
    assert row is not None
    assert row.ticker == "KXELONMARS-99"
    assert row.status == "active"
    assert row.last_price_cents == 9
    assert row.yes_bid_cents == 8
    assert row.yes_ask_cents == 10
    assert row.no_bid_cents == 90
    assert row.no_ask_cents == 92


def test_markets_repo_get_missing_returns_none(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    assert repo.get("KXNOSUCH-1") is None


def test_markets_repo_upsert_replaces_on_duplicate(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    repo.upsert(_sample_market())

    updated_payload = dict(_MARKET_PAYLOAD)
    updated_payload["status"] = "closed"
    updated_payload["last_price_dollars"] = "0.9500"
    repo.upsert(KalshiMarket.model_validate(updated_payload))

    row = repo.get("KXELONMARS-99")
    assert row is not None
    assert row.status == "closed"
    assert row.last_price_cents == 95


def test_markets_repo_list_by_status(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    repo.upsert(_sample_market())

    other_payload = dict(_MARKET_PAYLOAD)
    other_payload["ticker"] = "KXOTHER-1"
    other_payload["status"] = "closed"
    repo.upsert(KalshiMarket.model_validate(other_payload))

    active = repo.list_by_status("active")
    assert len(active) == 1
    assert active[0].ticker == "KXELONMARS-99"

    closed = repo.list_by_status("closed")
    assert len(closed) == 1
    assert closed[0].ticker == "KXOTHER-1"


def test_markets_repo_list_by_status_empty(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    assert repo.list_by_status("active") == []


def test_markets_repo_list_by_event(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    repo.upsert(_sample_market())

    other_payload = dict(_MARKET_PAYLOAD)
    other_payload["ticker"] = "KXOTHER-EVENT"
    other_payload["event_ticker"] = "KXOTHER-EVENT"
    repo.upsert(KalshiMarket.model_validate(other_payload))

    rows = repo.list_by_event("KXELONMARS-99")
    assert len(rows) == 1
    assert rows[0].ticker == "KXELONMARS-99"


def test_markets_repo_cached_at_populated(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiMarketsRepo(tmp_db)
    repo.upsert(_sample_market())
    row = repo.get("KXELONMARS-99")
    assert row is not None
    assert row.cached_at >= int(time.time()) - 5


# ---------------------------------------------------------------------------
# KalshiTradesRepo
# ---------------------------------------------------------------------------


def test_trades_repo_insert_batch_and_get(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    inserted = repo.insert_batch([_sample_trade(_TRADE_PAYLOAD_1)])
    assert inserted == 1

    row = repo.get("trade-aaa")
    assert row is not None
    assert row.trade_id == "trade-aaa"
    assert row.taker_side == "yes"
    assert row.yes_price_cents == 9
    assert row.no_price_cents == 91
    assert row.count_fp == pytest.approx(1.0)


def test_trades_repo_get_missing_returns_none(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    assert repo.get("no-such-id") is None


def test_trades_repo_insert_batch_deduplicates(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    first = repo.insert_batch([_sample_trade(_TRADE_PAYLOAD_1)])
    assert first == 1
    # Second insert of same trade_id is a no-op
    second = repo.insert_batch([_sample_trade(_TRADE_PAYLOAD_1)])
    assert second == 0


def test_trades_repo_insert_multiple(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    trades = [_sample_trade(_TRADE_PAYLOAD_1), _sample_trade(_TRADE_PAYLOAD_2)]
    inserted = repo.insert_batch(trades)
    assert inserted == 2


def test_trades_repo_list_by_ticker(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    repo.insert_batch([_sample_trade(_TRADE_PAYLOAD_1), _sample_trade(_TRADE_PAYLOAD_2)])

    rows = repo.list_by_ticker("KXELONMARS-99")
    assert len(rows) == 2
    # Newest-first ordering (trade-bbb was created_time 13:00, trade-aaa 12:36)
    assert rows[0].trade_id == "trade-bbb"
    assert rows[1].trade_id == "trade-aaa"


def test_trades_repo_list_by_ticker_empty(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    assert repo.list_by_ticker("KXNOSUCH-99") == []


def test_trades_repo_list_by_ticker_limit(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiTradesRepo(tmp_db)
    repo.insert_batch([_sample_trade(_TRADE_PAYLOAD_1), _sample_trade(_TRADE_PAYLOAD_2)])
    rows = repo.list_by_ticker("KXELONMARS-99", limit=1)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# KalshiOrderbookSnapshotsRepo
# ---------------------------------------------------------------------------


def test_orderbook_repo_insert_and_latest(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiOrderbookSnapshotsRepo(tmp_db)
    repo.insert("KXELONMARS-99", _sample_orderbook())

    snap = repo.latest("KXELONMARS-99")
    assert snap is not None
    assert snap.ticker == "KXELONMARS-99"
    # yes_bids_json round-trips through JSON
    yes_bids = json.loads(snap.yes_bids_json)
    assert yes_bids[0] == ["0.0800", "100.00"]


def test_orderbook_repo_latest_missing_returns_none(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiOrderbookSnapshotsRepo(tmp_db)
    assert repo.latest("KXNOSUCH-1") is None


def test_orderbook_repo_latest_returns_newest(tmp_db: sqlite3.Connection) -> None:
    """latest() returns the row with the highest id when ts values are equal."""
    repo = KalshiOrderbookSnapshotsRepo(tmp_db)
    ob1 = KalshiOrderbook.model_validate(
        {"orderbook_fp": {"yes_dollars": [["0.05", "1.00"]], "no_dollars": []}}
    )
    ob2 = KalshiOrderbook.model_validate(
        {"orderbook_fp": {"yes_dollars": [["0.06", "2.00"]], "no_dollars": []}}
    )
    repo.insert("KXELONMARS-99", ob1)
    repo.insert("KXELONMARS-99", ob2)

    # list_by_ticker uses ORDER BY ts DESC, id DESC so the second insert (higher id)
    # sorts first even when ts is the same.
    snaps = repo.list_by_ticker("KXELONMARS-99", limit=10)
    assert len(snaps) >= 2
    assert snaps[0].id is not None
    assert snaps[1].id is not None
    assert snaps[0].id > snaps[1].id
    # latest() should return the same as snaps[0]
    snap = repo.latest("KXELONMARS-99")
    assert snap is not None
    assert snap.id == snaps[0].id


def test_orderbook_repo_list_by_ticker(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiOrderbookSnapshotsRepo(tmp_db)
    repo.insert("KXELONMARS-99", _sample_orderbook())
    repo.insert("KXELONMARS-99", _sample_orderbook())

    snaps = repo.list_by_ticker("KXELONMARS-99", limit=10)
    assert len(snaps) == 2


def test_orderbook_repo_id_is_set(tmp_db: sqlite3.Connection) -> None:
    repo = KalshiOrderbookSnapshotsRepo(tmp_db)
    repo.insert("KXELONMARS-99", _sample_orderbook())
    snap = repo.latest("KXELONMARS-99")
    assert snap is not None
    assert snap.id is not None
    assert snap.id >= 1
