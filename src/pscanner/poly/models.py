"""Pydantic models for the Polymarket REST and WebSocket payloads.

These types are the contract every Polymarket client (gamma, data, clob_ws) must
return. Where Polymarket sends list-shaped data as JSON-encoded strings — for
example ``outcomePrices`` on the gamma ``markets`` endpoint — we parse defensively
so callers always see the typed Python list.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

_BASE_MODEL_CONFIG: ConfigDict = ConfigDict(populate_by_name=True, extra="ignore")


def _parse_json_string_list(value: Any) -> list[Any]:
    r"""Coerce a JSON-encoded string list into a Python list.

    Polymarket's gamma API often serialises list fields as JSON strings
    (e.g. ``outcomePrices = "[\"0.42\", \"0.58\"]"``). Already-decoded lists
    pass through untouched.

    Args:
        value: A list, a JSON-encoded string, or ``None``/empty string.

    Returns:
        A Python list (empty if input was ``None`` or empty).

    Raises:
        ValueError: If ``value`` is a string but not valid JSON-list.
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            msg = f"expected JSON-encoded list, got {value!r}"
            raise ValueError(msg) from exc
        if not isinstance(decoded, list):
            msg = f"expected JSON list, decoded to {type(decoded).__name__}"
            raise ValueError(msg)
        return decoded
    msg = f"expected list or JSON string, got {type(value).__name__}"
    raise ValueError(msg)


def _coerce_optional_float(value: Any) -> float | None:
    """Coerce strings like ``"123.4"`` to float; pass through None/numbers."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            msg = f"could not coerce {value!r} to float"
            raise ValueError(msg) from exc
    msg = f"unsupported type for float coercion: {type(value).__name__}"
    raise ValueError(msg)


class Outcome(BaseModel):
    """A single outcome within a market (e.g. ``YES`` at price 0.42)."""

    model_config = _BASE_MODEL_CONFIG

    name: str
    price: float


class Market(BaseModel):
    """A Polymarket market — one binary or multi-outcome question."""

    model_config = _BASE_MODEL_CONFIG

    id: str
    condition_id: Annotated[str | None, Field(alias="conditionId", default=None)] = None
    question: str
    slug: str
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: Annotated[
        list[float],
        Field(alias="outcomePrices", default_factory=list),
    ] = Field(default_factory=list)
    liquidity: float | None = None
    volume: float | None = None
    enable_order_book: Annotated[bool, Field(alias="enableOrderBook")] = True
    active: bool = True
    closed: bool = False
    clob_token_ids: Annotated[
        list[str],
        Field(alias="clobTokenIds", default_factory=list),
    ] = Field(default_factory=list)
    event_id: str | None = None

    @field_validator("outcomes", "clob_token_ids", mode="before")
    @classmethod
    def _decode_string_list(cls, value: Any) -> list[Any]:
        """Decode JSON-string lists into Python lists."""
        return _parse_json_string_list(value)

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def _decode_outcome_prices(cls, value: Any) -> list[float]:
        """Decode prices to floats; the wire format is ``["0.42", "0.58"]``."""
        decoded = _parse_json_string_list(value)
        return [float(item) for item in decoded]

    @field_validator("liquidity", "volume", mode="before")
    @classmethod
    def _decode_money(cls, value: Any) -> float | None:
        """Coerce ``"123.4"``-style strings to float."""
        return _coerce_optional_float(value)


class Event(BaseModel):
    """A Polymarket event grouping one or more mutex markets."""

    model_config = _BASE_MODEL_CONFIG

    id: str
    title: str
    slug: str
    markets: list[Market] = Field(default_factory=list)
    liquidity: float | None = None
    volume: float | None = None
    active: bool = True
    closed: bool = False

    @field_validator("liquidity", "volume", mode="before")
    @classmethod
    def _decode_money(cls, value: Any) -> float | None:
        """Coerce numeric strings to float."""
        return _coerce_optional_float(value)


class Position(BaseModel):
    """An open position held by a wallet on a Polymarket market.

    Fields not present on the ``/closed-positions`` response (``size``,
    ``current_value``, ``cash_pnl``, ``percent_pnl``) default to ``0.0`` so the
    same model parses both ``/positions`` and ``/closed-positions`` payloads.
    """

    model_config = _BASE_MODEL_CONFIG

    proxy_wallet: Annotated[str, Field(alias="proxyWallet")]
    asset: str
    condition_id: Annotated[str, Field(alias="conditionId")]
    market_id: str | None = None
    outcome: str
    outcome_index: Annotated[int, Field(alias="outcomeIndex")]
    size: float = 0.0
    avg_price: Annotated[float, Field(alias="avgPrice")]
    current_value: Annotated[float, Field(alias="currentValue", default=0.0)] = 0.0
    cash_pnl: Annotated[float, Field(alias="cashPnl", default=0.0)] = 0.0
    percent_pnl: Annotated[float, Field(alias="percentPnl", default=0.0)] = 0.0
    realized_pnl: Annotated[float | None, Field(alias="realizedPnl", default=None)] = None
    redeemable: bool = False
    mergeable: bool = False
    title: str | None = None
    slug: str | None = None
    end_date: Annotated[str | None, Field(alias="endDate", default=None)] = None


class ClosedPosition(Position):
    """A resolved (closed) position with optional final value and computed win flag."""

    final_value: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def won(self) -> bool:
        """True if the position resolved profitably (PnL>0) or is redeemable."""
        if self.realized_pnl is not None and self.realized_pnl > 0:
            return True
        return self.redeemable


class Trade(BaseModel):
    """A single trade fill emitted by the data API or CLOB websocket."""

    model_config = _BASE_MODEL_CONFIG

    transaction_hash: Annotated[str, Field(alias="transactionHash")]
    proxy_wallet: Annotated[str, Field(alias="proxyWallet")]
    condition_id: Annotated[str, Field(alias="conditionId")]
    asset: str
    side: Literal["BUY", "SELL"]
    size: float
    price: float
    timestamp: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def usd_value(self) -> float:
        """USD notional of the fill (``size * price``)."""
        return self.size * self.price


class LeaderboardEntry(BaseModel):
    """A single row from the data-api leaderboard endpoint."""

    model_config = _BASE_MODEL_CONFIG

    proxy_wallet: Annotated[str, Field(alias="proxyWallet")]
    name: str | None = None
    pseudonym: str | None = None
    pnl: Annotated[float, Field(alias="amount")]
    volume: float | None = None
    period: str


class WsTradeMessage(BaseModel):
    """WebSocket ``trade`` event from ``wss://.../ws/market``.

    Polymarket emits two messages per fill — ``MATCHED`` then ``CONFIRMED`` —
    so consumers must dedupe on ``transaction_hash`` and only act on the
    ``CONFIRMED`` event.
    """

    model_config = _BASE_MODEL_CONFIG

    event_type: Literal["trade"]
    condition_id: str
    asset_id: str
    side: str
    size: float
    price: float
    taker_proxy: str
    status: Literal["MATCHED", "CONFIRMED"]
    transaction_hash: str | None = None
    timestamp: int


class WsBookMessage(BaseModel):
    """WebSocket book/price-change event — opaque payload kept as ``data``."""

    model_config = _BASE_MODEL_CONFIG

    event_type: Literal["book", "price_change", "tick_size_change", "last_trade_price"]
    asset_id: str
    data: dict[str, Any] = Field(default_factory=dict)
