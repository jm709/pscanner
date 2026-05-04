"""Tests for :mod:`pscanner.kalshi.ids`."""

from __future__ import annotations

import pscanner.kalshi.ids as kalshi_ids_module
import pscanner.poly.ids as poly_ids_module
from pscanner.kalshi.ids import KalshiEventTicker, KalshiMarketTicker, KalshiSeriesTicker
from pscanner.poly.ids import MarketId


def test_market_ticker_is_string() -> None:
    ticker = KalshiMarketTicker("KXELONMARS-99")
    assert ticker == "KXELONMARS-99"
    assert isinstance(ticker, str)


def test_event_ticker_is_string() -> None:
    ticker = KalshiEventTicker("KXELONMARS-99")
    assert ticker == "KXELONMARS-99"
    assert isinstance(ticker, str)


def test_series_ticker_is_string() -> None:
    ticker = KalshiSeriesTicker("KXELONMARS")
    assert ticker == "KXELONMARS"
    assert isinstance(ticker, str)


def test_types_have_str_supertype() -> None:
    """All three Kalshi ID types are NewType wrappers over str."""
    assert KalshiMarketTicker.__supertype__ is str  # type: ignore[attr-defined]
    assert KalshiEventTicker.__supertype__ is str  # type: ignore[attr-defined]
    assert KalshiSeriesTicker.__supertype__ is str  # type: ignore[attr-defined]


def test_kalshi_market_ticker_is_str_at_runtime() -> None:
    """Kalshi and Poly IDs are both plain str at runtime; NewType is a type-check-only primitive."""
    poly_id = MarketId("540817")
    kalshi_id = KalshiMarketTicker("KXELONMARS-99")
    assert type(poly_id).__name__ == "str"
    assert type(kalshi_id).__name__ == "str"
    # Different module of origin — the modules are separate
    assert KalshiMarketTicker.__supertype__ is MarketId.__supertype__  # type: ignore[attr-defined]


def test_kalshi_ids_module_is_separate_from_poly() -> None:
    """Kalshi IDs live in pscanner.kalshi.ids, not pscanner.poly.ids."""
    assert kalshi_ids_module is not poly_ids_module
    assert not hasattr(kalshi_ids_module, "MarketId")
    assert not hasattr(poly_ids_module, "KalshiMarketTicker")
