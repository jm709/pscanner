"""SQLite repository classes for Kalshi tables.

Each repo accepts an already-initialised ``sqlite3.Connection`` (obtained via
``pscanner.store.db.init_db``, which also runs the Kalshi schema statements)
and uses parameterised SQL exclusively. Writes commit on the bound connection.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from pscanner.kalshi.models import KalshiMarket, KalshiOrderbook, KalshiTrade


def _now_seconds() -> int:
    """Return the current Unix timestamp in whole seconds."""
    return int(time.time())


@dataclass(frozen=True, slots=True)
class KalshiMarketRow:
    """A cached Kalshi market as stored in ``kalshi_markets``."""

    ticker: str
    event_ticker: str
    title: str
    status: str
    market_type: str
    open_time: str
    close_time: str
    expected_expiration_time: str
    yes_sub_title: str
    no_sub_title: str
    last_price_cents: int
    yes_bid_cents: int
    yes_ask_cents: int
    no_bid_cents: int
    no_ask_cents: int
    volume_fp: float
    volume_24h_fp: float
    open_interest_fp: float
    cached_at: int


@dataclass(frozen=True, slots=True)
class KalshiTradeRow:
    """A single Kalshi trade as stored in ``kalshi_trades``."""

    trade_id: str
    ticker: str
    taker_side: str
    yes_price_cents: int
    no_price_cents: int
    count_fp: float
    created_time: str
    recorded_at: int


@dataclass(frozen=True, slots=True)
class KalshiOrderbookSnapshotRow:
    """A kalshi orderbook snapshot as stored in ``kalshi_orderbook_snapshots``."""

    id: int | None
    ticker: str
    ts: int
    yes_bids_json: str
    no_bids_json: str


class KalshiMarketsRepo:
    """CRUD for the ``kalshi_markets`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, market: KalshiMarket) -> None:
        """Insert or replace a market, refreshing ``cached_at``.

        Args:
            market: Validated :class:`~pscanner.kalshi.models.KalshiMarket`.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO kalshi_markets (
              ticker, event_ticker, title, status, market_type,
              open_time, close_time, expected_expiration_time,
              yes_sub_title, no_sub_title,
              last_price_cents, yes_bid_cents, yes_ask_cents,
              no_bid_cents, no_ask_cents,
              volume_fp, volume_24h_fp, open_interest_fp, cached_at
            ) VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                market.ticker,
                market.event_ticker,
                market.title,
                market.status,
                market.market_type,
                market.open_time,
                market.close_time,
                market.expected_expiration_time,
                market.yes_sub_title,
                market.no_sub_title,
                market.last_price_cents,
                market.yes_bid_cents,
                market.yes_ask_cents,
                _dollars_to_cents(market.no_bid_dollars),
                _dollars_to_cents(market.no_ask_dollars),
                market.volume_fp,
                market.volume_24h_fp,
                market.open_interest_fp,
                _now_seconds(),
            ),
        )
        self._conn.commit()

    def get(self, ticker: str) -> KalshiMarketRow | None:
        """Fetch a single market by ticker, or ``None`` if not cached.

        Args:
            ticker: Kalshi market ticker.

        Returns:
            The cached market row, or ``None``.
        """
        row = self._conn.execute(
            "SELECT * FROM kalshi_markets WHERE ticker = ?", (ticker,)
        ).fetchone()
        return _market_row(row) if row is not None else None

    def list_by_status(self, status: str) -> list[KalshiMarketRow]:
        """Return all cached markets with the given status.

        Args:
            status: Market status string (e.g. ``"active"``, ``"closed"``).

        Returns:
            List of matching market rows.
        """
        rows = self._conn.execute(
            "SELECT * FROM kalshi_markets WHERE status = ? ORDER BY ticker", (status,)
        ).fetchall()
        return [_market_row(r) for r in rows]

    def list_by_event(self, event_ticker: str) -> list[KalshiMarketRow]:
        """Return all cached markets for the given event ticker.

        Args:
            event_ticker: Kalshi event ticker.

        Returns:
            List of matching market rows.
        """
        rows = self._conn.execute(
            "SELECT * FROM kalshi_markets WHERE event_ticker = ? ORDER BY ticker",
            (event_ticker,),
        ).fetchall()
        return [_market_row(r) for r in rows]


class KalshiTradesRepo:
    """CRUD for the ``kalshi_trades`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_batch(self, trades: list[KalshiTrade]) -> int:
        """Insert trades, skipping duplicates by ``trade_id``.

        Args:
            trades: List of validated :class:`~pscanner.kalshi.models.KalshiTrade`.

        Returns:
            Number of rows actually inserted (existing rows are ignored).
        """
        now = _now_seconds()
        inserted = 0
        for trade in trades:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO kalshi_trades (
                  trade_id, ticker, taker_side, yes_price_cents, no_price_cents,
                  count_fp, created_time, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    trade.ticker,
                    trade.taker_side,
                    trade.yes_price_cents,
                    trade.no_price_cents,
                    trade.count_fp,
                    trade.created_time,
                    now,
                ),
            )
            inserted += cursor.rowcount
        self._conn.commit()
        return inserted

    def get(self, trade_id: str) -> KalshiTradeRow | None:
        """Fetch a single trade by trade ID, or ``None`` if not found.

        Args:
            trade_id: UUID string identifying the trade.

        Returns:
            The trade row, or ``None``.
        """
        row = self._conn.execute(
            "SELECT * FROM kalshi_trades WHERE trade_id = ?", (trade_id,)
        ).fetchone()
        return _trade_row(row) if row is not None else None

    def list_by_ticker(self, ticker: str, *, limit: int = 500) -> list[KalshiTradeRow]:
        """Return the most recent trades for a given market ticker.

        Args:
            ticker: Kalshi market ticker.
            limit: Maximum number of rows to return (default 500).

        Returns:
            List of trade rows ordered newest-first.
        """
        rows = self._conn.execute(
            "SELECT * FROM kalshi_trades WHERE ticker = ? ORDER BY created_time DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        return [_trade_row(r) for r in rows]


class KalshiOrderbookSnapshotsRepo:
    """CRUD for the ``kalshi_orderbook_snapshots`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, ticker: str, orderbook: KalshiOrderbook) -> None:
        """Insert an orderbook snapshot for the given ticker.

        Args:
            ticker: Kalshi market ticker.
            orderbook: Validated :class:`~pscanner.kalshi.models.KalshiOrderbook`.
        """
        self._conn.execute(
            """
            INSERT INTO kalshi_orderbook_snapshots (ticker, ts, yes_bids_json, no_bids_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                ticker,
                _now_seconds(),
                json.dumps(orderbook.yes_bids),
                json.dumps(orderbook.no_bids),
            ),
        )
        self._conn.commit()

    def latest(self, ticker: str) -> KalshiOrderbookSnapshotRow | None:
        """Return the most recent orderbook snapshot for a ticker, or ``None``.

        Args:
            ticker: Kalshi market ticker.

        Returns:
            The most recent snapshot row, or ``None`` if none exist.
        """
        row = self._conn.execute(
            "SELECT * FROM kalshi_orderbook_snapshots WHERE ticker = ? "
            "ORDER BY ts DESC, id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        return _snapshot_row(row) if row is not None else None

    def list_by_ticker(self, ticker: str, *, limit: int = 10) -> list[KalshiOrderbookSnapshotRow]:
        """Return recent orderbook snapshots for a ticker, newest-first.

        Args:
            ticker: Kalshi market ticker.
            limit: Maximum number of snapshots (default 10).

        Returns:
            List of snapshot rows, newest first.
        """
        rows = self._conn.execute(
            "SELECT * FROM kalshi_orderbook_snapshots WHERE ticker = ? "
            "ORDER BY ts DESC, id DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        return [_snapshot_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Internal row-mapper helpers
# ---------------------------------------------------------------------------


def _dollars_to_cents(dollars: float) -> int:
    return round(dollars * 100)


def _market_row(row: sqlite3.Row) -> KalshiMarketRow:
    return KalshiMarketRow(
        ticker=row["ticker"],
        event_ticker=row["event_ticker"],
        title=row["title"],
        status=row["status"],
        market_type=row["market_type"],
        open_time=row["open_time"],
        close_time=row["close_time"],
        expected_expiration_time=row["expected_expiration_time"],
        yes_sub_title=row["yes_sub_title"],
        no_sub_title=row["no_sub_title"],
        last_price_cents=row["last_price_cents"],
        yes_bid_cents=row["yes_bid_cents"],
        yes_ask_cents=row["yes_ask_cents"],
        no_bid_cents=row["no_bid_cents"],
        no_ask_cents=row["no_ask_cents"],
        volume_fp=row["volume_fp"],
        volume_24h_fp=row["volume_24h_fp"],
        open_interest_fp=row["open_interest_fp"],
        cached_at=row["cached_at"],
    )


def _trade_row(row: sqlite3.Row) -> KalshiTradeRow:
    return KalshiTradeRow(
        trade_id=row["trade_id"],
        ticker=row["ticker"],
        taker_side=row["taker_side"],
        yes_price_cents=row["yes_price_cents"],
        no_price_cents=row["no_price_cents"],
        count_fp=row["count_fp"],
        created_time=row["created_time"],
        recorded_at=row["recorded_at"],
    )


def _snapshot_row(row: sqlite3.Row) -> KalshiOrderbookSnapshotRow:
    return KalshiOrderbookSnapshotRow(
        id=row["id"],
        ticker=row["ticker"],
        ts=row["ts"],
        yes_bids_json=row["yes_bids_json"],
        no_bids_json=row["no_bids_json"],
    )
