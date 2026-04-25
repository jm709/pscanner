"""SQLite repository classes — one per table.

Wave 1 freezes the public method signatures and the small dataclasses that the
repos hand back. Wave 2's ``repo-layer`` agent implements the bodies.

All repos accept an already-initialised ``sqlite3.Connection`` (see
``pscanner.store.db.init_db``) and use parameterised SQL exclusively. Writes
commit on the bound connection so callers do not need to manage transactions
for the simple upsert-and-read patterns the detectors require.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.poly.models import Market


@dataclass(frozen=True, slots=True)
class TrackedWallet:
    """A wallet retained for smart-money monitoring."""

    address: str
    closed_position_count: int
    closed_position_wins: int
    winrate: float
    leaderboard_pnl: float | None
    last_refreshed_at: int


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """A point-in-time record of a wallet's position on a market+side."""

    address: str
    market_id: str
    side: str
    size: float
    avg_price: float
    snapshot_at: int


@dataclass(frozen=True, slots=True)
class WalletFirstSeen:
    """Cached first-activity metadata for whale-detector age checks."""

    address: str
    first_activity_at: int | None
    total_trades: int | None
    cached_at: int


@dataclass(frozen=True, slots=True)
class CachedMarket:
    """Cached subset of ``Market`` used by detectors at runtime.

    The on-disk row stores ``outcome_prices_json`` as a JSON-encoded list of
    floats; this dataclass surfaces it pre-parsed so detectors do not need to
    re-decode the column on every read.
    """

    market_id: str
    event_id: str | None
    title: str | None
    liquidity_usd: float | None
    volume_usd: float | None
    outcome_prices: list[float]
    active: bool
    cached_at: int


def _now_seconds() -> int:
    """Return the current Unix timestamp in whole seconds."""
    return int(time.time())


class TrackedWalletsRepo:
    """CRUD for the ``tracked_wallets`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(
        self,
        address: str,
        closed_position_count: int,
        closed_position_wins: int,
        winrate: float,
        leaderboard_pnl: float | None,
    ) -> None:
        """Insert or update a tracked wallet, refreshing ``last_refreshed_at``.

        Args:
            address: 0x-prefixed proxy wallet address.
            closed_position_count: Total resolved positions observed.
            closed_position_wins: Resolved positions with PnL > 0.
            winrate: ``wins / count`` precomputed by the caller.
            leaderboard_pnl: All-time PnL from the leaderboard (may be ``None``).
        """
        now = _now_seconds()
        self._conn.execute(
            """
            INSERT INTO tracked_wallets (
              address, closed_position_count, closed_position_wins,
              winrate, leaderboard_pnl, last_refreshed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
              closed_position_count = excluded.closed_position_count,
              closed_position_wins = excluded.closed_position_wins,
              winrate = excluded.winrate,
              leaderboard_pnl = excluded.leaderboard_pnl,
              last_refreshed_at = excluded.last_refreshed_at
            """,
            (
                address,
                closed_position_count,
                closed_position_wins,
                winrate,
                leaderboard_pnl,
                now,
            ),
        )
        self._conn.commit()

    def list_active(self, min_winrate: float, min_resolved: int) -> list[TrackedWallet]:
        """Return wallets meeting the smart-money quality bar.

        Args:
            min_winrate: Inclusive winrate threshold.
            min_resolved: Inclusive minimum closed-position count.

        Returns:
            Wallets passing both filters, ordered by winrate desc.
        """
        rows = self._conn.execute(
            """
            SELECT address, closed_position_count, closed_position_wins,
                   winrate, leaderboard_pnl, last_refreshed_at
              FROM tracked_wallets
             WHERE winrate >= ? AND closed_position_count >= ?
             ORDER BY winrate DESC
            """,
            (min_winrate, min_resolved),
        ).fetchall()
        return [_row_to_tracked_wallet(row) for row in rows]

    def list_all(self) -> list[TrackedWallet]:
        """Return every row in the table (no filtering)."""
        rows = self._conn.execute(
            """
            SELECT address, closed_position_count, closed_position_wins,
                   winrate, leaderboard_pnl, last_refreshed_at
              FROM tracked_wallets
             ORDER BY winrate DESC
            """,
        ).fetchall()
        return [_row_to_tracked_wallet(row) for row in rows]


def _row_to_tracked_wallet(row: sqlite3.Row) -> TrackedWallet:
    """Convert a ``tracked_wallets`` row to a ``TrackedWallet`` dataclass."""
    return TrackedWallet(
        address=row["address"],
        closed_position_count=row["closed_position_count"],
        closed_position_wins=row["closed_position_wins"],
        winrate=row["winrate"],
        leaderboard_pnl=row["leaderboard_pnl"],
        last_refreshed_at=row["last_refreshed_at"],
    )


class PositionSnapshotsRepo:
    """CRUD for the ``wallet_position_snapshots`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(
        self,
        address: str,
        market_id: str,
        side: str,
        size: float,
        avg_price: float,
    ) -> None:
        """Insert or update the (address, market, side) snapshot row."""
        now = _now_seconds()
        self._conn.execute(
            """
            INSERT INTO wallet_position_snapshots (
              address, market_id, side, size, avg_price, snapshot_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(address, market_id, side) DO UPDATE SET
              size = excluded.size,
              avg_price = excluded.avg_price,
              snapshot_at = excluded.snapshot_at
            """,
            (address, market_id, side, size, avg_price, now),
        )
        self._conn.commit()

    def get_for_wallet(self, address: str) -> list[PositionSnapshot]:
        """Return all snapshots for a single wallet."""
        rows = self._conn.execute(
            """
            SELECT address, market_id, side, size, avg_price, snapshot_at
              FROM wallet_position_snapshots
             WHERE address = ?
             ORDER BY market_id ASC, side ASC
            """,
            (address,),
        ).fetchall()
        return [
            PositionSnapshot(
                address=row["address"],
                market_id=row["market_id"],
                side=row["side"],
                size=row["size"],
                avg_price=row["avg_price"],
                snapshot_at=row["snapshot_at"],
            )
            for row in rows
        ]

    def previous_size(self, address: str, market_id: str, side: str) -> float | None:
        """Return the most-recently-stored size, or ``None`` if no row exists."""
        row = self._conn.execute(
            """
            SELECT size FROM wallet_position_snapshots
             WHERE address = ? AND market_id = ? AND side = ?
            """,
            (address, market_id, side),
        ).fetchone()
        if row is None:
            return None
        return float(row["size"])


class WalletFirstSeenRepo:
    """CRUD for the ``wallet_first_seen`` cache."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def get(self, address: str) -> WalletFirstSeen | None:
        """Return the cached first-seen row for a wallet, or ``None``."""
        row = self._conn.execute(
            """
            SELECT address, first_activity_at, total_trades, cached_at
              FROM wallet_first_seen
             WHERE address = ?
            """,
            (address,),
        ).fetchone()
        if row is None:
            return None
        return WalletFirstSeen(
            address=row["address"],
            first_activity_at=row["first_activity_at"],
            total_trades=row["total_trades"],
            cached_at=row["cached_at"],
        )

    def upsert(
        self,
        address: str,
        first_activity_at: int | None,
        total_trades: int | None,
    ) -> None:
        """Insert or update the cache row, stamping ``cached_at`` to now."""
        now = _now_seconds()
        self._conn.execute(
            """
            INSERT INTO wallet_first_seen (
              address, first_activity_at, total_trades, cached_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
              first_activity_at = excluded.first_activity_at,
              total_trades = excluded.total_trades,
              cached_at = excluded.cached_at
            """,
            (address, first_activity_at, total_trades, now),
        )
        self._conn.commit()


class MarketCacheRepo:
    """CRUD for the ``market_cache`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, market: Market) -> None:
        """Insert or update a cache row from a freshly-fetched ``Market``.

        Args:
            market: Source-of-truth market model (gamma response).
        """
        now = _now_seconds()
        prices_json = json.dumps(list(market.outcome_prices))
        self._conn.execute(
            """
            INSERT INTO market_cache (
              market_id, event_id, title, liquidity_usd, volume_usd,
              outcome_prices_json, active, cached_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
              event_id = excluded.event_id,
              title = excluded.title,
              liquidity_usd = excluded.liquidity_usd,
              volume_usd = excluded.volume_usd,
              outcome_prices_json = excluded.outcome_prices_json,
              active = excluded.active,
              cached_at = excluded.cached_at
            """,
            (
                market.id,
                market.event_id,
                market.question,
                market.liquidity,
                market.volume,
                prices_json,
                1 if market.active else 0,
                now,
            ),
        )
        self._conn.commit()

    def get(self, market_id: str) -> CachedMarket | None:
        """Return the cached row for a market, or ``None``."""
        row = self._conn.execute(
            """
            SELECT market_id, event_id, title, liquidity_usd, volume_usd,
                   outcome_prices_json, active, cached_at
              FROM market_cache
             WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_cached_market(row)

    def list_active(self) -> list[CachedMarket]:
        """Return every cached market with ``active = 1``."""
        rows = self._conn.execute(
            """
            SELECT market_id, event_id, title, liquidity_usd, volume_usd,
                   outcome_prices_json, active, cached_at
              FROM market_cache
             WHERE active = 1
             ORDER BY market_id ASC
            """,
        ).fetchall()
        return [_row_to_cached_market(row) for row in rows]


def _row_to_cached_market(row: sqlite3.Row) -> CachedMarket:
    """Convert a ``market_cache`` row to a ``CachedMarket`` dataclass."""
    raw_prices = row["outcome_prices_json"]
    prices: list[float] = []
    if raw_prices:
        decoded = json.loads(raw_prices)
        if not isinstance(decoded, list):
            msg = (
                "market_cache.outcome_prices_json must decode to list, "
                f"got {type(decoded).__name__}"
            )
            raise ValueError(msg)
        prices = [float(item) for item in decoded]
    return CachedMarket(
        market_id=row["market_id"],
        event_id=row["event_id"],
        title=row["title"],
        liquidity_usd=row["liquidity_usd"],
        volume_usd=row["volume_usd"],
        outcome_prices=prices,
        active=bool(row["active"]),
        cached_at=row["cached_at"],
    )


class AlertsRepo:
    """CRUD for the ``alerts`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_if_new(self, alert: Alert) -> bool:
        """Insert ``alert`` if its ``alert_key`` is unseen.

        Args:
            alert: The alert to persist.

        Returns:
            ``True`` if the row was inserted, ``False`` if the key was already
            present (idempotency dedupe hit).
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO alerts (
              alert_key, detector, severity, title, body_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                alert.alert_key,
                alert.detector,
                alert.severity,
                alert.title,
                json.dumps(alert.body),
                alert.created_at,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent(self, detector: str | None = None, limit: int = 100) -> list[Alert]:
        """Return the most recent alerts, optionally filtered by detector.

        Args:
            detector: If set, only return alerts where ``detector = <value>``.
            limit: Max rows to return.

        Returns:
            Alerts in descending ``created_at`` order.
        """
        if detector is None:
            rows = self._conn.execute(
                """
                SELECT alert_key, detector, severity, title, body_json, created_at
                  FROM alerts
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT alert_key, detector, severity, title, body_json, created_at
                  FROM alerts
                 WHERE detector = ?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (detector, limit),
            ).fetchall()
        return [_row_to_alert(row) for row in rows]


def _row_to_alert(row: sqlite3.Row) -> Alert:
    """Convert an ``alerts`` row into an ``Alert`` dataclass."""
    body = json.loads(row["body_json"])
    if not isinstance(body, dict):
        msg = f"alerts.body_json must decode to dict, got {type(body).__name__}"
        raise ValueError(msg)
    detector_name: DetectorName = row["detector"]
    severity: Severity = row["severity"]
    return Alert(
        detector=detector_name,
        alert_key=row["alert_key"],
        severity=severity,
        title=row["title"],
        body=body,
        created_at=row["created_at"],
    )
