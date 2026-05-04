"""Pydantic models for Kalshi REST API payloads.

Field names follow Kalshi's JSON casing where reasonable. Prices are expressed
as ``str`` dollar amounts by the API (e.g. ``"0.0900"`` = 9 cents); the models
expose them as ``float`` in dollar terms and provide a ``_cents`` computed
property for the integer-cent representation used in DB storage.

Verified against live ``https://api.elections.kalshi.com/trade-api/v2/``
responses on 2026-05-04.

Note on pricing units:
    Kalshi prices are in dollars on the wire (``"0.09"`` = 9 cents = $0.09 per
    share). Contracts settle to $0 or $1. The ``yes_price_cents`` and
    ``no_price_cents`` helpers return the rounded integer-cent value (1-99).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_BASE_CONFIG: ConfigDict = ConfigDict(populate_by_name=True, extra="ignore")


def _dollars_to_cents(dollars: float) -> int:
    """Convert a dollar price (0.0-1.0) to integer cents (0-100)."""
    return round(dollars * 100)


class KalshiMarket(BaseModel):
    """A single Kalshi binary market.

    Maps to one row in the ``kalshi_markets`` table.
    """

    model_config = _BASE_CONFIG

    ticker: str
    event_ticker: str
    title: str
    status: str
    market_type: str = ""
    open_time: str = ""
    close_time: str = ""
    expected_expiration_time: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""

    # Prices are returned as dollar strings like "0.0900"; coerce to float.
    last_price_dollars: Annotated[float, Field(default=0.0)]
    yes_bid_dollars: Annotated[float, Field(default=0.0)]
    yes_ask_dollars: Annotated[float, Field(default=0.0)]
    no_bid_dollars: Annotated[float, Field(default=0.0)]
    no_ask_dollars: Annotated[float, Field(default=0.0)]

    # Volume and open-interest come back as fixed-point strings ("0.00").
    volume_fp: Annotated[float, Field(default=0.0)]
    volume_24h_fp: Annotated[float, Field(default=0.0)]
    open_interest_fp: Annotated[float, Field(default=0.0)]

    @property
    def last_price_cents(self) -> int:
        """Integer-cent representation of ``last_price_dollars`` (0-100)."""
        return _dollars_to_cents(self.last_price_dollars)

    @property
    def yes_bid_cents(self) -> int:
        """Integer cents for the YES bid price."""
        return _dollars_to_cents(self.yes_bid_dollars)

    @property
    def yes_ask_cents(self) -> int:
        """Integer cents for the YES ask price."""
        return _dollars_to_cents(self.yes_ask_dollars)

    @property
    def no_bid_cents(self) -> int:
        """Integer cents for the NO bid price."""
        return _dollars_to_cents(self.no_bid_dollars)

    @property
    def no_ask_cents(self) -> int:
        """Integer cents for the NO ask price."""
        return _dollars_to_cents(self.no_ask_dollars)


class _OrderbookInner(BaseModel):
    """Inner object under the ``orderbook_fp`` key in orderbook responses."""

    model_config = _BASE_CONFIG

    yes_dollars: list[list[str]] = Field(default_factory=list)
    no_dollars: list[list[str]] = Field(default_factory=list)


class KalshiOrderbook(BaseModel):
    """Kalshi orderbook snapshot for a single market.

    Each level in ``yes_dollars`` / ``no_dollars`` is a ``[price, size]``
    pair of dollar-string values. The ``yes_bids`` property returns the list
    directly; callers that need cents should convert with ``round(float(p)*100)``.
    """

    model_config = _BASE_CONFIG

    orderbook_fp: _OrderbookInner

    @property
    def yes_bids(self) -> list[list[str]]:
        """YES-side bid levels as ``[price_dollars, size]`` string pairs."""
        return self.orderbook_fp.yes_dollars

    @property
    def no_bids(self) -> list[list[str]]:
        """NO-side bid levels as ``[price_dollars, size]`` string pairs."""
        return self.orderbook_fp.no_dollars


class KalshiTrade(BaseModel):
    """A single executed trade returned by ``/markets/trades``.

    ``taker_side`` is ``"yes"`` or ``"no"``.
    ``count_fp`` is the contract count as a fixed-point string (``"1.00"``).
    Prices are dollar strings (``"0.0900"``).
    """

    model_config = _BASE_CONFIG

    trade_id: str
    ticker: str
    taker_side: str
    yes_price_dollars: Annotated[float, Field(default=0.0)]
    no_price_dollars: Annotated[float, Field(default=0.0)]
    count_fp: Annotated[float, Field(default=0.0)]
    created_time: str = ""

    @property
    def yes_price_cents(self) -> int:
        """Integer-cent representation of the YES price."""
        return _dollars_to_cents(self.yes_price_dollars)

    @property
    def no_price_cents(self) -> int:
        """Integer-cent representation of the NO price."""
        return _dollars_to_cents(self.no_price_dollars)


class KalshiMarketsPage(BaseModel):
    """Paginated response from ``GET /markets``."""

    model_config = _BASE_CONFIG

    markets: list[KalshiMarket] = Field(default_factory=list)
    cursor: str = ""


class KalshiTradesPage(BaseModel):
    """Paginated response from ``GET /markets/trades``."""

    model_config = _BASE_CONFIG

    trades: list[KalshiTrade] = Field(default_factory=list)
    cursor: str = ""


class KalshiEvent(BaseModel):
    """A Kalshi event grouping one or more markets."""

    model_config = _BASE_CONFIG

    event_ticker: str
    series_ticker: str = ""
    title: str = ""
    sub_title: str = ""
    category: str = ""
    mutually_exclusive: bool = False


class KalshiSeries(BaseModel):
    """A Kalshi series grouping related events."""

    model_config = _BASE_CONFIG

    ticker: str
    title: str = ""
    category: str = ""
    frequency: str = ""
