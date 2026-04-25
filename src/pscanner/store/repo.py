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
    mean_edge: float | None = None
    weighted_edge: float | None = None
    excess_pnl_usd: float | None = None
    total_stake_usd: float | None = None


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
        *,
        address: str,
        closed_position_count: int,
        closed_position_wins: int,
        winrate: float,
        leaderboard_pnl: float | None = None,
        mean_edge: float | None = None,
        weighted_edge: float | None = None,
        excess_pnl_usd: float | None = None,
        total_stake_usd: float | None = None,
    ) -> None:
        """Insert or update a tracked wallet, refreshing ``last_refreshed_at``.

        Args:
            address: 0x-prefixed proxy wallet address.
            closed_position_count: Total resolved positions observed.
            closed_position_wins: Resolved positions with PnL > 0.
            winrate: ``wins / count`` precomputed by the caller.
            leaderboard_pnl: All-time PnL from the leaderboard (may be ``None``).
            mean_edge: Average per-position edge ``(outcome - avg_price)``.
            weighted_edge: Stake-USD-weighted mean edge.
            excess_pnl_usd: Realized PnL summed across resolved positions
                (the wallet's dollar alpha versus market-rate expected PnL).
            total_stake_usd: Sum of ``size * avg_price`` over scored positions.
        """
        now = _now_seconds()
        self._conn.execute(
            """
            INSERT INTO tracked_wallets (
              address, closed_position_count, closed_position_wins,
              winrate, leaderboard_pnl, last_refreshed_at,
              mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
              closed_position_count = excluded.closed_position_count,
              closed_position_wins = excluded.closed_position_wins,
              winrate = excluded.winrate,
              leaderboard_pnl = excluded.leaderboard_pnl,
              last_refreshed_at = excluded.last_refreshed_at,
              mean_edge = excluded.mean_edge,
              weighted_edge = excluded.weighted_edge,
              excess_pnl_usd = excluded.excess_pnl_usd,
              total_stake_usd = excluded.total_stake_usd
            """,
            (
                address,
                closed_position_count,
                closed_position_wins,
                winrate,
                leaderboard_pnl,
                now,
                mean_edge,
                weighted_edge,
                excess_pnl_usd,
                total_stake_usd,
            ),
        )
        self._conn.commit()

    def list_active(
        self,
        *,
        min_edge: float,
        min_excess_pnl_usd: float,
        min_resolved: int,
    ) -> list[TrackedWallet]:
        """Return wallets meeting the smart-money quality bar.

        Args:
            min_edge: Inclusive minimum ``mean_edge`` threshold. Wallets with
                a NULL ``mean_edge`` are excluded.
            min_excess_pnl_usd: Inclusive minimum ``excess_pnl_usd`` threshold.
                Wallets with a NULL ``excess_pnl_usd`` are excluded.
            min_resolved: Inclusive minimum closed-position count.

        Returns:
            Wallets passing all filters, ordered by ``excess_pnl_usd`` desc.
        """
        rows = self._conn.execute(
            """
            SELECT address, closed_position_count, closed_position_wins,
                   winrate, leaderboard_pnl, last_refreshed_at,
                   mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd
              FROM tracked_wallets
             WHERE closed_position_count >= :min_resolved
               AND mean_edge IS NOT NULL AND mean_edge >= :min_edge
               AND excess_pnl_usd IS NOT NULL
               AND excess_pnl_usd >= :min_excess_pnl_usd
             ORDER BY excess_pnl_usd DESC
            """,
            {
                "min_resolved": min_resolved,
                "min_edge": min_edge,
                "min_excess_pnl_usd": min_excess_pnl_usd,
            },
        ).fetchall()
        return [_row_to_tracked_wallet(row) for row in rows]

    def list_all(self) -> list[TrackedWallet]:
        """Return every row in the table (no filtering)."""
        rows = self._conn.execute(
            """
            SELECT address, closed_position_count, closed_position_wins,
                   winrate, leaderboard_pnl, last_refreshed_at,
                   mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd
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
        mean_edge=row["mean_edge"],
        weighted_edge=row["weighted_edge"],
        excess_pnl_usd=row["excess_pnl_usd"],
        total_stake_usd=row["total_stake_usd"],
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


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    """A wallet enrolled in the data-collection watchlist."""

    address: str
    source: str
    reason: str | None
    added_at: int
    active: bool


class WatchlistRepo:
    """CRUD for the ``wallet_watchlist`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(
        self,
        *,
        address: str,
        source: str,
        reason: str | None = None,
    ) -> bool:
        """Insert a new watchlist entry; never overwrite an existing one.

        On conflict the existing ``source`` and ``reason`` are preserved (we
        keep the first-recorded provenance).

        Args:
            address: 0x-prefixed proxy wallet address (primary key).
            source: How the address was discovered (``smart_money``,
                ``whale_alert``, or ``manual``).
            reason: Optional free-form note explaining why the wallet was
                added (e.g. an alert key).

        Returns:
            ``True`` if a new row was inserted, ``False`` if ``address`` was
            already present.
        """
        now = _now_seconds()
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO wallet_watchlist (
              address, source, reason, added_at, active
            ) VALUES (?, ?, ?, ?, 1)
            """,
            (address, source, reason, now),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def set_active(self, address: str, active: bool) -> None:
        """Toggle an entry's ``active`` flag in place.

        Args:
            address: 0x-prefixed proxy wallet address.
            active: New active state. ``False`` deactivates without removing
                the row, preserving its provenance.
        """
        self._conn.execute(
            "UPDATE wallet_watchlist SET active = ? WHERE address = ?",
            (1 if active else 0, address),
        )
        self._conn.commit()

    def get(self, address: str) -> WatchlistEntry | None:
        """Return the row for ``address``, or ``None`` if absent."""
        row = self._conn.execute(
            """
            SELECT address, source, reason, added_at, active
              FROM wallet_watchlist
             WHERE address = ?
            """,
            (address,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_watchlist_entry(row)

    def list_active(self) -> list[WatchlistEntry]:
        """Return every row with ``active = 1``, ordered by address."""
        rows = self._conn.execute(
            """
            SELECT address, source, reason, added_at, active
              FROM wallet_watchlist
             WHERE active = 1
             ORDER BY address ASC
            """,
        ).fetchall()
        return [_row_to_watchlist_entry(row) for row in rows]

    def list_all(self) -> list[WatchlistEntry]:
        """Return every row in the table (active + inactive)."""
        rows = self._conn.execute(
            """
            SELECT address, source, reason, added_at, active
              FROM wallet_watchlist
             ORDER BY address ASC
            """,
        ).fetchall()
        return [_row_to_watchlist_entry(row) for row in rows]


def _row_to_watchlist_entry(row: sqlite3.Row) -> WatchlistEntry:
    """Convert a ``wallet_watchlist`` row to a ``WatchlistEntry``."""
    return WatchlistEntry(
        address=row["address"],
        source=row["source"],
        reason=row["reason"],
        added_at=row["added_at"],
        active=bool(row["active"]),
    )


@dataclass(frozen=True, slots=True)
class WalletTrade:
    """A single CONFIRMED trade by a watched wallet (append-only)."""

    transaction_hash: str
    asset_id: str
    side: str
    wallet: str
    condition_id: str
    size: float
    price: float
    usd_value: float
    status: str
    source: str
    timestamp: int
    recorded_at: int


class WalletTradesRepo:
    """Append-only store for ``wallet_trades`` rows.

    Rows are never updated or deleted post-insert: the composite primary key
    ``(transaction_hash, asset_id, side)`` is the dedupe key and ``insert``
    uses ``INSERT OR IGNORE``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, trade: WalletTrade) -> bool:
        """Insert ``trade`` if its composite PK is unseen.

        Args:
            trade: The fully-populated trade row to persist.

        Returns:
            ``True`` if the row was inserted, ``False`` if the composite key
            already existed (dedupe hit).
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO wallet_trades (
              transaction_hash, asset_id, side, wallet, condition_id,
              size, price, usd_value, status, source, timestamp, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.transaction_hash,
                trade.asset_id,
                trade.side,
                trade.wallet,
                trade.condition_id,
                trade.size,
                trade.price,
                trade.usd_value,
                trade.status,
                trade.source,
                trade.timestamp,
                trade.recorded_at,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent_for_wallet(self, wallet: str, *, limit: int = 100) -> list[WalletTrade]:
        """Return the wallet's most-recent trades, newest first.

        Args:
            wallet: 0x-prefixed proxy wallet address.
            limit: Max rows to return.

        Returns:
            Trades with ``wallet = <address>`` ordered by ``timestamp DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT transaction_hash, asset_id, side, wallet, condition_id,
                   size, price, usd_value, status, source, timestamp, recorded_at
              FROM wallet_trades
             WHERE wallet = ?
             ORDER BY timestamp DESC
             LIMIT ?
            """,
            (wallet, limit),
        ).fetchall()
        return [_row_to_wallet_trade(row) for row in rows]

    def count_by_wallet(self) -> dict[str, int]:
        """Return ``{wallet: trade_count}`` across the full table."""
        rows = self._conn.execute(
            "SELECT wallet, COUNT(*) AS c FROM wallet_trades GROUP BY wallet",
        ).fetchall()
        return {row["wallet"]: int(row["c"]) for row in rows}


def _row_to_wallet_trade(row: sqlite3.Row) -> WalletTrade:
    """Convert a ``wallet_trades`` row to a ``WalletTrade`` dataclass."""
    return WalletTrade(
        transaction_hash=row["transaction_hash"],
        asset_id=row["asset_id"],
        side=row["side"],
        wallet=row["wallet"],
        condition_id=row["condition_id"],
        size=float(row["size"]),
        price=float(row["price"]),
        usd_value=float(row["usd_value"]),
        status=row["status"],
        source=row["source"],
        timestamp=int(row["timestamp"]),
        recorded_at=int(row["recorded_at"]),
    )


@dataclass(frozen=True, slots=True)
class WalletPositionsHistoryRow:
    """One row per (wallet, market-position) at a given timestamp."""

    wallet: str
    condition_id: str
    outcome: str
    size: float
    avg_price: float
    current_value: float | None
    cash_pnl: float | None
    realized_pnl: float | None
    redeemable: bool | None
    snapshot_at: int


class WalletPositionsHistoryRepo:
    """Append-only history of position snapshots per watched wallet.

    Rows are inserted with ``INSERT OR IGNORE`` against the composite primary
    key ``(wallet, condition_id, outcome, snapshot_at)``. Because callers
    stamp ``snapshot_at`` at second resolution, two snapshots taken in the
    same second for the same position collapse to a single row — that is
    the intended idempotency behaviour.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, row: WalletPositionsHistoryRow) -> bool:
        """Insert ``row`` if its composite PK is unseen.

        Args:
            row: Fully-populated history row to persist.

        Returns:
            ``True`` if the row was newly inserted, ``False`` on PK collision.
        """
        redeemable = _bool_to_int_or_none(row.redeemable)
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO wallet_positions_history (
              wallet, condition_id, outcome, size, avg_price,
              current_value, cash_pnl, realized_pnl, redeemable, snapshot_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.wallet,
                row.condition_id,
                row.outcome,
                row.size,
                row.avg_price,
                row.current_value,
                row.cash_pnl,
                row.realized_pnl,
                redeemable,
                row.snapshot_at,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent_for_wallet(
        self,
        wallet: str,
        *,
        limit: int = 200,
    ) -> list[WalletPositionsHistoryRow]:
        """Return the wallet's most-recent rows, newest first.

        Args:
            wallet: 0x-prefixed proxy wallet address.
            limit: Max rows to return.

        Returns:
            History rows ordered by ``snapshot_at DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT wallet, condition_id, outcome, size, avg_price,
                   current_value, cash_pnl, realized_pnl, redeemable, snapshot_at
              FROM wallet_positions_history
             WHERE wallet = ?
             ORDER BY snapshot_at DESC
             LIMIT ?
            """,
            (wallet, limit),
        ).fetchall()
        return [_row_to_positions_history(row) for row in rows]

    def count_by_wallet(self) -> dict[str, int]:
        """Return ``{wallet: row_count}`` across the full table."""
        rows = self._conn.execute(
            "SELECT wallet, COUNT(*) AS c FROM wallet_positions_history GROUP BY wallet",
        ).fetchall()
        return {row["wallet"]: int(row["c"]) for row in rows}


def _row_to_positions_history(row: sqlite3.Row) -> WalletPositionsHistoryRow:
    """Convert a ``wallet_positions_history`` row to its dataclass."""
    redeemable_raw = row["redeemable"]
    redeemable = None if redeemable_raw is None else bool(redeemable_raw)
    return WalletPositionsHistoryRow(
        wallet=row["wallet"],
        condition_id=row["condition_id"],
        outcome=row["outcome"],
        size=float(row["size"]),
        avg_price=float(row["avg_price"]),
        current_value=_optional_float(row["current_value"]),
        cash_pnl=_optional_float(row["cash_pnl"]),
        realized_pnl=_optional_float(row["realized_pnl"]),
        redeemable=redeemable,
        snapshot_at=int(row["snapshot_at"]),
    )


def _optional_float(value: float | int | None) -> float | None:
    """Return ``float(value)`` or ``None`` when the value is missing."""
    if value is None:
        return None
    return float(value)


def _bool_to_int_or_none(value: bool | None) -> int | None:
    """Encode an optional bool as 0/1/None for SQLite storage."""
    if value is None:
        return None
    return 1 if value else 0


@dataclass(frozen=True, slots=True)
class WalletActivityEvent:
    """One activity-stream event from the Polymarket data API."""

    wallet: str
    event_type: str
    payload_json: str
    timestamp: int
    recorded_at: int
    source: str


class WalletActivityEventsRepo:
    """Append-only activity stream per watched wallet.

    Dedupes by composite primary key ``(wallet, timestamp, event_type)`` via
    ``INSERT OR IGNORE`` — the data API returns overlapping pages across
    polls, so the repo silently swallows redundant inserts.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, event: WalletActivityEvent) -> bool:
        """Insert ``event`` if its composite PK is unseen.

        Args:
            event: Fully-populated activity event to persist.

        Returns:
            ``True`` if the row was newly inserted, ``False`` on PK collision.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO wallet_activity_events (
              wallet, event_type, payload_json, timestamp, recorded_at, source
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.wallet,
                event.event_type,
                event.payload_json,
                event.timestamp,
                event.recorded_at,
                event.source,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent_for_wallet(
        self,
        wallet: str,
        *,
        limit: int = 200,
        event_type: str | None = None,
    ) -> list[WalletActivityEvent]:
        """Return the wallet's most-recent events, newest first.

        Args:
            wallet: 0x-prefixed proxy wallet address.
            limit: Max rows to return.
            event_type: When set, only rows with ``event_type = <value>``.

        Returns:
            Activity events ordered by ``timestamp DESC``.
        """
        if event_type is None:
            rows = self._conn.execute(
                """
                SELECT wallet, event_type, payload_json, timestamp, recorded_at, source
                  FROM wallet_activity_events
                 WHERE wallet = ?
                 ORDER BY timestamp DESC
                 LIMIT ?
                """,
                (wallet, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT wallet, event_type, payload_json, timestamp, recorded_at, source
                  FROM wallet_activity_events
                 WHERE wallet = ? AND event_type = ?
                 ORDER BY timestamp DESC
                 LIMIT ?
                """,
                (wallet, event_type, limit),
            ).fetchall()
        return [_row_to_activity_event(row) for row in rows]

    def count_by_wallet(self) -> dict[str, int]:
        """Return ``{wallet: event_count}`` across the full table."""
        rows = self._conn.execute(
            "SELECT wallet, COUNT(*) AS c FROM wallet_activity_events GROUP BY wallet",
        ).fetchall()
        return {row["wallet"]: int(row["c"]) for row in rows}


def _row_to_activity_event(row: sqlite3.Row) -> WalletActivityEvent:
    """Convert a ``wallet_activity_events`` row to its dataclass."""
    return WalletActivityEvent(
        wallet=row["wallet"],
        event_type=row["event_type"],
        payload_json=row["payload_json"],
        timestamp=int(row["timestamp"]),
        recorded_at=int(row["recorded_at"]),
        source=row["source"],
    )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """One row per (market, snapshot_at) — full market state at point in time.

    ``outcome_prices_json`` is a JSON-encoded ``list[float]`` (the same shape
    used by ``market_cache``) preserved verbatim so callers can re-decode on
    read without forcing this layer to know the ordering convention.
    """

    market_id: str
    event_id: str | None
    outcome_prices_json: str
    liquidity_usd: float | None
    volume_usd: float | None
    active: bool
    snapshot_at: int


class MarketSnapshotsRepo:
    """Append-only history of market state snapshots.

    Inserts dedupe on the composite primary key ``(market_id, snapshot_at)``
    via ``INSERT OR IGNORE`` — two snapshots of the same market in the same
    second collapse to one row, which is the intended idempotency contract.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, snapshot: MarketSnapshot) -> bool:
        """Insert ``snapshot`` if its composite PK is unseen.

        Args:
            snapshot: Fully-populated market snapshot to persist.

        Returns:
            ``True`` if the row was newly inserted, ``False`` on PK collision.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO market_snapshots (
              market_id, event_id, outcome_prices_json, liquidity_usd,
              volume_usd, active, snapshot_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.market_id,
                snapshot.event_id,
                snapshot.outcome_prices_json,
                snapshot.liquidity_usd,
                snapshot.volume_usd,
                1 if snapshot.active else 0,
                snapshot.snapshot_at,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent_for_market(
        self,
        market_id: str,
        *,
        limit: int = 200,
    ) -> list[MarketSnapshot]:
        """Return the market's most-recent snapshots, newest first.

        Args:
            market_id: Polymarket market identifier.
            limit: Max rows to return.

        Returns:
            Snapshots ordered by ``snapshot_at DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT market_id, event_id, outcome_prices_json, liquidity_usd,
                   volume_usd, active, snapshot_at
              FROM market_snapshots
             WHERE market_id = ?
             ORDER BY snapshot_at DESC
             LIMIT ?
            """,
            (market_id, limit),
        ).fetchall()
        return [_row_to_market_snapshot(row) for row in rows]

    def distinct_snapshot_count(self) -> int:
        """Return ``COUNT(DISTINCT snapshot_at)`` — total sweeps recorded."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT snapshot_at) AS c FROM market_snapshots",
        ).fetchone()
        if row is None:
            return 0
        return int(row["c"])

    def count_by_market(self) -> dict[str, int]:
        """Return ``{market_id: snapshot_count}`` across the full table."""
        rows = self._conn.execute(
            "SELECT market_id, COUNT(*) AS c FROM market_snapshots GROUP BY market_id",
        ).fetchall()
        return {row["market_id"]: int(row["c"]) for row in rows}


def _row_to_market_snapshot(row: sqlite3.Row) -> MarketSnapshot:
    """Convert a ``market_snapshots`` row to a ``MarketSnapshot`` dataclass."""
    return MarketSnapshot(
        market_id=row["market_id"],
        event_id=row["event_id"],
        outcome_prices_json=row["outcome_prices_json"],
        liquidity_usd=_optional_float(row["liquidity_usd"]),
        volume_usd=_optional_float(row["volume_usd"]),
        active=bool(row["active"]),
        snapshot_at=int(row["snapshot_at"]),
    )


@dataclass(frozen=True, slots=True)
class EventSnapshot:
    """One row per (event, snapshot_at) — event-level metadata at a point in time."""

    event_id: str
    title: str
    slug: str
    liquidity_usd: float | None
    volume_usd: float | None
    active: bool
    closed: bool
    market_count: int
    snapshot_at: int


class EventSnapshotsRepo:
    """Append-only history of event-level state snapshots.

    Inserts dedupe on the composite primary key ``(event_id, snapshot_at)``
    via ``INSERT OR IGNORE`` — two snapshots of the same event in the same
    second collapse to one row.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, snapshot: EventSnapshot) -> bool:
        """Insert ``snapshot`` if its composite PK is unseen.

        Args:
            snapshot: Fully-populated event snapshot to persist.

        Returns:
            ``True`` if the row was newly inserted, ``False`` on PK collision.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO event_snapshots (
              event_id, title, slug, liquidity_usd, volume_usd,
              active, closed, market_count, snapshot_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.event_id,
                snapshot.title,
                snapshot.slug,
                snapshot.liquidity_usd,
                snapshot.volume_usd,
                1 if snapshot.active else 0,
                1 if snapshot.closed else 0,
                snapshot.market_count,
                snapshot.snapshot_at,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent_for_event(
        self,
        event_id: str,
        *,
        limit: int = 200,
    ) -> list[EventSnapshot]:
        """Return the event's most-recent snapshots, newest first.

        Args:
            event_id: Polymarket event identifier.
            limit: Max rows to return.

        Returns:
            Snapshots ordered by ``snapshot_at DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT event_id, title, slug, liquidity_usd, volume_usd,
                   active, closed, market_count, snapshot_at
              FROM event_snapshots
             WHERE event_id = ?
             ORDER BY snapshot_at DESC
             LIMIT ?
            """,
            (event_id, limit),
        ).fetchall()
        return [_row_to_event_snapshot(row) for row in rows]

    def distinct_snapshot_count(self) -> int:
        """Return ``COUNT(DISTINCT snapshot_at)`` — total sweeps recorded."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT snapshot_at) AS c FROM event_snapshots",
        ).fetchone()
        if row is None:
            return 0
        return int(row["c"])

    def count_by_event(self) -> dict[str, int]:
        """Return ``{event_id: snapshot_count}`` across the full table."""
        rows = self._conn.execute(
            "SELECT event_id, COUNT(*) AS c FROM event_snapshots GROUP BY event_id",
        ).fetchall()
        return {row["event_id"]: int(row["c"]) for row in rows}


def _row_to_event_snapshot(row: sqlite3.Row) -> EventSnapshot:
    """Convert an ``event_snapshots`` row to an ``EventSnapshot`` dataclass."""
    return EventSnapshot(
        event_id=row["event_id"],
        title=row["title"],
        slug=row["slug"],
        liquidity_usd=_optional_float(row["liquidity_usd"]),
        volume_usd=_optional_float(row["volume_usd"]),
        active=bool(row["active"]),
        closed=bool(row["closed"]),
        market_count=int(row["market_count"]),
        snapshot_at=int(row["snapshot_at"]),
    )
