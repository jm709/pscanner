"""Tests for :mod:`pscanner.kalshi.models` — pydantic round-trip validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pscanner.kalshi.models import (
    KalshiEvent,
    KalshiMarket,
    KalshiMarketsPage,
    KalshiOrderbook,
    KalshiSeries,
    KalshiTrade,
    KalshiTradesPage,
)

# ---------------------------------------------------------------------------
# Representative payloads captured from live API on 2026-05-04
# ---------------------------------------------------------------------------

_MARKET_PAYLOAD: dict = {
    "ticker": "KXELONMARS-99",
    "event_ticker": "KXELONMARS-99",
    "title": "Will Elon Musk visit Mars in his lifetime?",
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

_ORDERBOOK_PAYLOAD: dict = {
    "orderbook_fp": {
        "yes_dollars": [["0.0100", "49524.50"], ["0.0200", "4195.00"]],
        "no_dollars": [["0.0100", "13551.00"], ["0.0400", "555.00"]],
    }
}

_TRADE_PAYLOAD: dict = {
    "trade_id": "5422e44d-d87a-67ad-2d8e-bc570f95d5da",
    "ticker": "KXELONMARS-99",
    "taker_side": "yes",
    "yes_price_dollars": "0.0900",
    "no_price_dollars": "0.9100",
    "count_fp": "1.00",
    "created_time": "2026-05-04T12:36:03.956406Z",
}


# ---------------------------------------------------------------------------
# KalshiMarket
# ---------------------------------------------------------------------------


def test_market_basic_round_trip() -> None:
    market = KalshiMarket.model_validate(_MARKET_PAYLOAD)
    assert market.ticker == "KXELONMARS-99"
    assert market.status == "active"
    assert market.title == "Will Elon Musk visit Mars in his lifetime?"


def test_market_prices_coerced_from_string() -> None:
    market = KalshiMarket.model_validate(_MARKET_PAYLOAD)
    assert abs(market.last_price_dollars - 0.09) < 1e-9
    assert abs(market.yes_bid_dollars - 0.08) < 1e-9
    assert abs(market.yes_ask_dollars - 0.10) < 1e-9


def test_market_cents_computed_correctly() -> None:
    market = KalshiMarket.model_validate(_MARKET_PAYLOAD)
    assert market.last_price_cents == 9
    assert market.yes_bid_cents == 8
    assert market.yes_ask_cents == 10


def test_market_volume_coerced_from_string() -> None:
    market = KalshiMarket.model_validate(_MARKET_PAYLOAD)
    assert market.volume_fp == pytest.approx(12345.0)
    assert market.volume_24h_fp == pytest.approx(500.0)


def test_market_extra_fields_ignored() -> None:
    payload = dict(_MARKET_PAYLOAD)
    payload["unknown_field_xyz"] = "ignored"
    market = KalshiMarket.model_validate(payload)
    assert not hasattr(market, "unknown_field_xyz")


def test_market_defaults_for_optional_fields() -> None:
    minimal = {
        "ticker": "KXTEST-1",
        "event_ticker": "KXTEST-1",
        "title": "Test",
        "status": "active",
    }
    market = KalshiMarket.model_validate(minimal)
    assert market.market_type == ""
    assert market.last_price_dollars == 0.0
    assert market.last_price_cents == 0


# ---------------------------------------------------------------------------
# KalshiOrderbook
# ---------------------------------------------------------------------------


def test_orderbook_round_trip() -> None:
    ob = KalshiOrderbook.model_validate(_ORDERBOOK_PAYLOAD)
    assert ob.yes_bids[0] == ["0.0100", "49524.50"]
    assert ob.no_bids[0] == ["0.0100", "13551.00"]


def test_orderbook_empty_levels() -> None:
    ob = KalshiOrderbook.model_validate({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}})
    assert ob.yes_bids == []
    assert ob.no_bids == []


# ---------------------------------------------------------------------------
# KalshiTrade
# ---------------------------------------------------------------------------


def test_trade_round_trip() -> None:
    trade = KalshiTrade.model_validate(_TRADE_PAYLOAD)
    assert trade.trade_id == "5422e44d-d87a-67ad-2d8e-bc570f95d5da"
    assert trade.taker_side == "yes"
    assert trade.yes_price_cents == 9
    assert trade.no_price_cents == 91


def test_trade_count_fp_coerced() -> None:
    trade = KalshiTrade.model_validate(_TRADE_PAYLOAD)
    assert trade.count_fp == pytest.approx(1.0)


def test_trade_missing_required_field_raises() -> None:
    payload = dict(_TRADE_PAYLOAD)
    del payload["trade_id"]
    with pytest.raises(ValidationError):
        KalshiTrade.model_validate(payload)


# ---------------------------------------------------------------------------
# KalshiMarketsPage
# ---------------------------------------------------------------------------


def test_markets_page_round_trip() -> None:
    payload = {
        "markets": [_MARKET_PAYLOAD],
        "cursor": "abc123",
    }
    page = KalshiMarketsPage.model_validate(payload)
    assert len(page.markets) == 1
    assert page.cursor == "abc123"
    assert page.markets[0].ticker == "KXELONMARS-99"


def test_markets_page_empty() -> None:
    page = KalshiMarketsPage.model_validate({"markets": [], "cursor": ""})
    assert page.markets == []
    assert page.cursor == ""


def test_markets_page_cursor_defaults_to_empty() -> None:
    page = KalshiMarketsPage.model_validate({"markets": []})
    assert page.cursor == ""


# ---------------------------------------------------------------------------
# KalshiTradesPage
# ---------------------------------------------------------------------------


def test_trades_page_round_trip() -> None:
    payload = {
        "cursor": "XYZ",
        "trades": [_TRADE_PAYLOAD],
    }
    page = KalshiTradesPage.model_validate(payload)
    assert len(page.trades) == 1
    assert page.cursor == "XYZ"


# ---------------------------------------------------------------------------
# KalshiEvent
# ---------------------------------------------------------------------------


def test_event_round_trip() -> None:
    payload = {
        "event_ticker": "KXELONMARS-99",
        "series_ticker": "KXELONMARS",
        "title": "Will Elon Musk visit Mars in his lifetime?",
        "sub_title": "Before 2099",
        "category": "World",
        "mutually_exclusive": False,
    }
    event = KalshiEvent.model_validate(payload)
    assert event.event_ticker == "KXELONMARS-99"
    assert event.series_ticker == "KXELONMARS"
    assert event.category == "World"


# ---------------------------------------------------------------------------
# KalshiSeries
# ---------------------------------------------------------------------------


def test_series_round_trip() -> None:
    payload = {
        "ticker": "KXELONMARS",
        "title": "Elon Mars",
        "category": "Politics",
        "frequency": "custom",
    }
    series = KalshiSeries.model_validate(payload)
    assert series.ticker == "KXELONMARS"
    assert series.category == "Politics"
