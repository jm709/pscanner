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
from dataclasses import dataclass, field

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.poly.ids import AssetId, ConditionId, EventId, EventSlug, MarketId
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
    market_id: ConditionId
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
    re-decode the column on every read. ``outcomes`` and ``asset_ids`` are
    parallel lists: index ``i`` of ``outcomes`` is the human-readable outcome
    name (e.g. ``"Yes"``) whose CLOB asset id sits at ``asset_ids[i]``.
    """

    market_id: MarketId
    event_id: EventId | None
    title: str | None
    liquidity_usd: float | None
    volume_usd: float | None
    outcome_prices: list[float]
    active: bool
    cached_at: int
    condition_id: ConditionId | None = None
    event_slug: EventSlug | None = None
    outcomes: list[str] = field(default_factory=list)
    asset_ids: list[AssetId] = field(default_factory=list)


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

    def get(self, address: str) -> TrackedWallet | None:
        """Return the tracked wallet for ``address``, or ``None`` if not tracked.

        Args:
            address: 0x-prefixed proxy wallet address.

        Returns:
            The matching :class:`TrackedWallet`, or ``None`` if no row exists.
        """
        row = self._conn.execute(
            """
            SELECT address, closed_position_count, closed_position_wins,
                   winrate, leaderboard_pnl, last_refreshed_at,
                   mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd
              FROM tracked_wallets
             WHERE address = ?
            """,
            (address,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_tracked_wallet(row)


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
        market_id: ConditionId,
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
                market_id=ConditionId(row["market_id"]),
                side=row["side"],
                size=row["size"],
                avg_price=row["avg_price"],
                snapshot_at=row["snapshot_at"],
            )
            for row in rows
        ]

    def previous_size(self, address: str, market_id: ConditionId, side: str) -> float | None:
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

    def list_recent(self, *, within: int) -> list[WalletFirstSeen]:
        """Return rows whose ``first_activity_at`` is within the trailing window.

        Args:
            within: Trailing window length in days. Rows whose
                ``first_activity_at`` is older than ``now - within * 86400``
                seconds are excluded. Rows with NULL ``first_activity_at``
                are also excluded — the cluster detector only cares about
                wallets we have a creation timestamp for.

        Returns:
            Matching rows ordered by ``first_activity_at`` ascending.
        """
        now = _now_seconds()
        cutoff = now - within * 86400
        rows = self._conn.execute(
            """
            SELECT address, first_activity_at, total_trades, cached_at
              FROM wallet_first_seen
             WHERE first_activity_at IS NOT NULL
               AND first_activity_at >= ?
             ORDER BY first_activity_at ASC
            """,
            (cutoff,),
        ).fetchall()
        return [
            WalletFirstSeen(
                address=row["address"],
                first_activity_at=row["first_activity_at"],
                total_trades=row["total_trades"],
                cached_at=row["cached_at"],
            )
            for row in rows
        ]


class MarketCacheRepo:
    """CRUD for the ``market_cache`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, market: Market | CachedMarket) -> None:
        """Insert or update a cache row from a ``Market`` or ``CachedMarket``.

        Accepts either the gamma-API ``Market`` model (most production callers)
        or a pre-built ``CachedMarket`` dataclass (tests + callers that have
        already mapped to the cached shape). Both end up as the same row.

        Args:
            market: Source-of-truth market model or its cached projection.
        """
        row = _market_to_cache_row(market)
        self._conn.execute(
            """
            INSERT INTO market_cache (
              market_id, event_id, title, liquidity_usd, volume_usd,
              outcome_prices_json, active, cached_at,
              condition_id, event_slug, outcomes_json, asset_ids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
              event_id = excluded.event_id,
              title = excluded.title,
              liquidity_usd = excluded.liquidity_usd,
              volume_usd = excluded.volume_usd,
              outcome_prices_json = excluded.outcome_prices_json,
              active = excluded.active,
              cached_at = excluded.cached_at,
              condition_id = excluded.condition_id,
              event_slug = excluded.event_slug,
              outcomes_json = excluded.outcomes_json,
              asset_ids_json = excluded.asset_ids_json
            """,
            row,
        )
        self._conn.commit()

    def get(self, market_id: MarketId) -> CachedMarket | None:
        """Return the cached row for a market, or ``None``."""
        row = self._conn.execute(
            """
            SELECT market_id, event_id, title, liquidity_usd, volume_usd,
                   outcome_prices_json, active, cached_at,
                   condition_id, event_slug, outcomes_json, asset_ids_json
              FROM market_cache
             WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_cached_market(row)

    def get_by_condition_id(self, condition_id: ConditionId) -> CachedMarket | None:
        """Return the cached row whose ``condition_id`` matches, or ``None``.

        Args:
            condition_id: Polymarket condition identifier (the on-chain id used
                by the CLOB and the data-api activity stream).

        Returns:
            The first matching ``CachedMarket``, or ``None`` if no row carries
            this condition id. Multiple rows would be a data-corruption signal
            but are tolerated — only the first is returned.
        """
        row = self._conn.execute(
            """
            SELECT market_id, event_id, title, liquidity_usd, volume_usd,
                   outcome_prices_json, active, cached_at,
                   condition_id, event_slug, outcomes_json, asset_ids_json
              FROM market_cache
             WHERE condition_id = ?
             LIMIT 1
            """,
            (condition_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_cached_market(row)

    def list_active(self) -> list[CachedMarket]:
        """Return every cached market with ``active = 1``."""
        rows = self._conn.execute(
            """
            SELECT market_id, event_id, title, liquidity_usd, volume_usd,
                   outcome_prices_json, active, cached_at,
                   condition_id, event_slug, outcomes_json, asset_ids_json
              FROM market_cache
             WHERE active = 1
             ORDER BY market_id ASC
            """,
        ).fetchall()
        return [_row_to_cached_market(row) for row in rows]

    def outcome_to_asset(
        self,
        condition_id: ConditionId,
        outcome_name: str,
    ) -> AssetId | None:
        """Resolve an outcome name to its CLOB ``AssetId``.

        Matching is case-insensitive and tolerant of leading/trailing
        whitespace. When the cached market has mismatched ``outcomes`` and
        ``asset_ids`` lengths (an upstream data bug) the lookup gives up and
        returns ``None`` rather than guessing.

        Args:
            condition_id: The market's on-chain condition identifier.
            outcome_name: Human-readable outcome label (e.g. ``"Yes"``).

        Returns:
            The matching ``AssetId``, or ``None`` if the market is unknown,
            the outcome is unknown, or the parallel lists disagree on length.
        """
        cached = self.get_by_condition_id(condition_id)
        if cached is None:
            return None
        if len(cached.outcomes) != len(cached.asset_ids):
            return None
        target = outcome_name.strip().casefold()
        for name, asset_id in zip(cached.outcomes, cached.asset_ids, strict=True):
            if name.strip().casefold() == target:
                return asset_id
        return None


def _decode_json_string_list(raw: str | None, column: str) -> list[str]:
    """Decode a ``TEXT`` column expected to hold a JSON list of strings.

    A null/missing/empty column value yields ``[]``. A non-list payload is a
    schema violation and raises rather than silently dropping data.
    """
    if not raw:
        return []
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        msg = f"market_cache.{column} must decode to list, got {type(decoded).__name__}"
        raise ValueError(msg)
    return [str(item) for item in decoded]


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
    raw_event_id = row["event_id"]
    raw_condition_id = row["condition_id"]
    raw_event_slug = row["event_slug"]
    outcomes = _decode_json_string_list(row["outcomes_json"], "outcomes_json")
    asset_ids = [
        AssetId(item) for item in _decode_json_string_list(row["asset_ids_json"], "asset_ids_json")
    ]
    return CachedMarket(
        market_id=MarketId(row["market_id"]),
        event_id=EventId(raw_event_id) if raw_event_id is not None else None,
        title=row["title"],
        liquidity_usd=row["liquidity_usd"],
        volume_usd=row["volume_usd"],
        outcome_prices=prices,
        active=bool(row["active"]),
        cached_at=row["cached_at"],
        condition_id=ConditionId(raw_condition_id) if raw_condition_id is not None else None,
        event_slug=EventSlug(raw_event_slug) if raw_event_slug is not None else None,
        outcomes=outcomes,
        asset_ids=asset_ids,
    )


def _market_to_cache_row(
    market: Market | CachedMarket,
) -> tuple[
    str,
    str | None,
    str | None,
    float | None,
    float | None,
    str,
    int,
    int,
    str | None,
    str | None,
    str,
    str,
]:
    """Project a ``Market`` or ``CachedMarket`` into a ``market_cache`` row tuple."""
    if isinstance(market, CachedMarket):
        return (
            market.market_id,
            market.event_id,
            market.title,
            market.liquidity_usd,
            market.volume_usd,
            json.dumps(list(market.outcome_prices)),
            1 if market.active else 0,
            market.cached_at,
            market.condition_id,
            market.event_slug,
            json.dumps(list(market.outcomes)),
            json.dumps(list(market.asset_ids)),
        )
    return (
        market.id,
        market.event_id,
        market.question,
        market.liquidity,
        market.volume,
        json.dumps(list(market.outcome_prices)),
        1 if market.active else 0,
        _now_seconds(),
        market.condition_id,
        market.event_slug,
        json.dumps(list(market.outcomes)),
        json.dumps(list(market.clob_token_ids)),
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
    asset_id: AssetId
    side: str
    wallet: str
    condition_id: ConditionId
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

    def distinct_wallets_for_condition(
        self,
        condition_id: ConditionId,
        *,
        since: int,
    ) -> set[str]:
        """Return distinct wallets that traded ``condition_id`` since a timestamp.

        Args:
            condition_id: Polymarket condition identifier.
            since: Inclusive lower bound on ``timestamp`` (Unix seconds).

        Returns:
            Distinct wallet addresses with at least one ``wallet_trades`` row
            for ``condition_id`` whose ``timestamp >= since``.
        """
        rows = self._conn.execute(
            """
            SELECT DISTINCT wallet
              FROM wallet_trades
             WHERE condition_id = ? AND timestamp >= ?
            """,
            (condition_id, since),
        ).fetchall()
        return {row["wallet"] for row in rows}


def _row_to_wallet_trade(row: sqlite3.Row) -> WalletTrade:
    """Convert a ``wallet_trades`` row to a ``WalletTrade`` dataclass."""
    return WalletTrade(
        transaction_hash=row["transaction_hash"],
        asset_id=AssetId(row["asset_id"]),
        side=row["side"],
        wallet=row["wallet"],
        condition_id=ConditionId(row["condition_id"]),
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
    condition_id: ConditionId
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
        condition_id=ConditionId(row["condition_id"]),
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

    market_id: MarketId
    event_id: EventId | None
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
        market_id: MarketId,
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
    raw_event_id = row["event_id"]
    return MarketSnapshot(
        market_id=MarketId(row["market_id"]),
        event_id=EventId(raw_event_id) if raw_event_id is not None else None,
        outcome_prices_json=row["outcome_prices_json"],
        liquidity_usd=_optional_float(row["liquidity_usd"]),
        volume_usd=_optional_float(row["volume_usd"]),
        active=bool(row["active"]),
        snapshot_at=int(row["snapshot_at"]),
    )


@dataclass(frozen=True, slots=True)
class EventSnapshot:
    """One row per (event, snapshot_at) — event-level metadata at a point in time."""

    event_id: EventId
    title: str
    slug: EventSlug
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
        event_id: EventId,
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
        event_id=EventId(row["event_id"]),
        title=row["title"],
        slug=EventSlug(row["slug"]),
        liquidity_usd=_optional_float(row["liquidity_usd"]),
        volume_usd=_optional_float(row["volume_usd"]),
        active=bool(row["active"]),
        closed=bool(row["closed"]),
        market_count=int(row["market_count"]),
        snapshot_at=int(row["snapshot_at"]),
    )


@dataclass(frozen=True, slots=True)
class EventOutcomeSumRow:
    """One row per (event, snapshot_at) — captured even when no alert fires."""

    event_id: EventId
    market_count: int
    price_sum: float
    deviation: float
    snapshot_at: int


class EventOutcomeSumRepo:
    """Append-only history of event-level Σ-of-outcomes per scan.

    Captures the YES-leg price sum across every market in a mispricing-eligible
    event on each scan, regardless of whether the event triggered an alert.
    Lets analysts retroactively study high-Σ multi-outcome layouts (checkbox
    events) that the alert path now silently filters past
    ``MispricingConfig.alert_max_deviation``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, row: EventOutcomeSumRow) -> bool:
        """Insert ``row`` if its composite PK ``(event_id, snapshot_at)`` is unseen.

        Args:
            row: Fully-populated outcome-sum row to persist.

        Returns:
            ``True`` if the row was newly inserted, ``False`` on PK collision.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO event_outcome_sum_history (
              event_id, market_count, price_sum, deviation, snapshot_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row.event_id,
                row.market_count,
                row.price_sum,
                row.deviation,
                row.snapshot_at,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def recent(self, *, limit: int = 200) -> list[EventOutcomeSumRow]:
        """Return the most recent rows across every event.

        Args:
            limit: Max rows to return.

        Returns:
            Rows ordered by ``snapshot_at DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT event_id, market_count, price_sum, deviation, snapshot_at
              FROM event_outcome_sum_history
             ORDER BY snapshot_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_event_outcome_sum(row) for row in rows]

    def by_event_id(self, event_id: EventId, *, limit: int = 200) -> list[EventOutcomeSumRow]:
        """Return all rows for one event, newest first.

        Args:
            event_id: Polymarket event identifier.
            limit: Max rows to return.

        Returns:
            Rows ordered by ``snapshot_at DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT event_id, market_count, price_sum, deviation, snapshot_at
              FROM event_outcome_sum_history
             WHERE event_id = ?
             ORDER BY snapshot_at DESC
             LIMIT ?
            """,
            (event_id, limit),
        ).fetchall()
        return [_row_to_event_outcome_sum(row) for row in rows]

    def with_high_deviation(
        self,
        *,
        min_abs_deviation: float,
        limit: int = 200,
    ) -> list[EventOutcomeSumRow]:
        """Return rows whose ``|deviation| >= min_abs_deviation``.

        Args:
            min_abs_deviation: Inclusive minimum absolute deviation threshold.
            limit: Max rows to return.

        Returns:
            Rows ordered by ``ABS(deviation) DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT event_id, market_count, price_sum, deviation, snapshot_at
              FROM event_outcome_sum_history
             WHERE ABS(deviation) >= ?
             ORDER BY ABS(deviation) DESC
             LIMIT ?
            """,
            (min_abs_deviation, limit),
        ).fetchall()
        return [_row_to_event_outcome_sum(row) for row in rows]


def _row_to_event_outcome_sum(row: sqlite3.Row) -> EventOutcomeSumRow:
    """Convert an ``event_outcome_sum_history`` row to its dataclass."""
    return EventOutcomeSumRow(
        event_id=EventId(row["event_id"]),
        market_count=int(row["market_count"]),
        price_sum=float(row["price_sum"]),
        deviation=float(row["deviation"]),
        snapshot_at=int(row["snapshot_at"]),
    )


@dataclass(frozen=True, slots=True)
class TrackedWalletCategory:
    """Per-category edge metrics for a tracked wallet.

    A wallet can have up to one row per category (``thesis``, ``sports``,
    ``esports``). The metrics mirror the fields stored on ``tracked_wallets``
    but are computed over the subset of closed positions that fall in the
    given category. ``last_refreshed_at`` is set by the repo on every upsert.
    """

    wallet: str
    category: str
    position_count: int
    win_count: int
    mean_edge: float | None
    weighted_edge: float | None
    excess_pnl_usd: float | None
    total_stake_usd: float | None
    last_refreshed_at: int


class TrackedWalletCategoriesRepo:
    """CRUD for the ``tracked_wallet_categories`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(
        self,
        *,
        wallet: str,
        category: str,
        position_count: int,
        win_count: int,
        mean_edge: float | None,
        weighted_edge: float | None,
        excess_pnl_usd: float | None,
        total_stake_usd: float | None,
    ) -> None:
        """Insert or update one ``(wallet, category)`` row.

        Args:
            wallet: 0x-prefixed proxy wallet address.
            category: One of ``"thesis"``, ``"sports"``, ``"esports"``.
            position_count: Resolved positions in the category.
            win_count: Resolved positions with PnL > 0 in the category.
            mean_edge: Average per-position edge over this category's
                positions, or ``None`` if not computable.
            weighted_edge: Stake-weighted mean edge, or ``None``.
            excess_pnl_usd: Realized PnL summed for the category, or ``None``.
            total_stake_usd: Sum of ``size * avg_price`` for the category,
                or ``None``.
        """
        now = _now_seconds()
        self._conn.execute(
            """
            INSERT INTO tracked_wallet_categories (
              wallet, category, position_count, win_count,
              mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd,
              last_refreshed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wallet, category) DO UPDATE SET
              position_count = excluded.position_count,
              win_count = excluded.win_count,
              mean_edge = excluded.mean_edge,
              weighted_edge = excluded.weighted_edge,
              excess_pnl_usd = excluded.excess_pnl_usd,
              total_stake_usd = excluded.total_stake_usd,
              last_refreshed_at = excluded.last_refreshed_at
            """,
            (
                wallet,
                category,
                position_count,
                win_count,
                mean_edge,
                weighted_edge,
                excess_pnl_usd,
                total_stake_usd,
                now,
            ),
        )
        self._conn.commit()

    def list_by_category(
        self,
        category: str,
        *,
        min_edge: float,
        min_excess_pnl_usd: float,
        min_resolved: int,
    ) -> list[TrackedWalletCategory]:
        """Return rows for ``category`` passing the per-category quality bar.

        Args:
            category: Category to filter on.
            min_edge: Inclusive minimum ``mean_edge``. Rows with NULL
                ``mean_edge`` are excluded.
            min_excess_pnl_usd: Inclusive minimum ``excess_pnl_usd``. Rows
                with NULL ``excess_pnl_usd`` are excluded.
            min_resolved: Inclusive minimum ``position_count``.

        Returns:
            Matching rows ordered by ``excess_pnl_usd`` desc.
        """
        rows = self._conn.execute(
            """
            SELECT wallet, category, position_count, win_count,
                   mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd,
                   last_refreshed_at
              FROM tracked_wallet_categories
             WHERE category = :category
               AND position_count >= :min_resolved
               AND mean_edge IS NOT NULL AND mean_edge >= :min_edge
               AND excess_pnl_usd IS NOT NULL
               AND excess_pnl_usd >= :min_excess_pnl_usd
             ORDER BY excess_pnl_usd DESC
            """,
            {
                "category": category,
                "min_resolved": min_resolved,
                "min_edge": min_edge,
                "min_excess_pnl_usd": min_excess_pnl_usd,
            },
        ).fetchall()
        return [_row_to_tracked_wallet_category(row) for row in rows]

    def list_for_wallet(self, wallet: str) -> list[TrackedWalletCategory]:
        """Return every category row for ``wallet``, ordered by category."""
        rows = self._conn.execute(
            """
            SELECT wallet, category, position_count, win_count,
                   mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd,
                   last_refreshed_at
              FROM tracked_wallet_categories
             WHERE wallet = ?
             ORDER BY category ASC
            """,
            (wallet,),
        ).fetchall()
        return [_row_to_tracked_wallet_category(row) for row in rows]

    def list_all(self) -> list[TrackedWalletCategory]:
        """Return every row in the table (no filtering)."""
        rows = self._conn.execute(
            """
            SELECT wallet, category, position_count, win_count,
                   mean_edge, weighted_edge, excess_pnl_usd, total_stake_usd,
                   last_refreshed_at
              FROM tracked_wallet_categories
             ORDER BY wallet ASC, category ASC
            """,
        ).fetchall()
        return [_row_to_tracked_wallet_category(row) for row in rows]


def _row_to_tracked_wallet_category(row: sqlite3.Row) -> TrackedWalletCategory:
    """Convert a ``tracked_wallet_categories`` row to its dataclass."""
    return TrackedWalletCategory(
        wallet=row["wallet"],
        category=row["category"],
        position_count=int(row["position_count"]),
        win_count=int(row["win_count"]),
        mean_edge=_optional_float(row["mean_edge"]),
        weighted_edge=_optional_float(row["weighted_edge"]),
        excess_pnl_usd=_optional_float(row["excess_pnl_usd"]),
        total_stake_usd=_optional_float(row["total_stake_usd"]),
        last_refreshed_at=int(row["last_refreshed_at"]),
    )


class EventTagCacheRepo:
    """CRUD for the ``event_tag_cache`` table.

    Persists the gamma ``tags`` array per event slug so detectors can
    categorise closed positions without re-fetching every event on every
    refresh. The cache is keyed on ``EventSlug`` (not numeric event id)
    because closed-position payloads expose ``eventSlug`` rather than a
    numeric id; the column was renamed from ``event_id`` to ``event_slug``
    by the typed-ids migration.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, event_slug: EventSlug, tags: list[str]) -> None:
        """Persist ``tags`` for ``event_slug`` as JSON, stamping ``cached_at``.

        Args:
            event_slug: Polymarket event slug (URL fragment).
            tags: Tag labels (e.g. ``["Sports", "NFL"]``).
        """
        now = _now_seconds()
        tags_json = json.dumps(list(tags))
        self._conn.execute(
            """
            INSERT INTO event_tag_cache (event_slug, tags_json, cached_at)
            VALUES (?, ?, ?)
            ON CONFLICT(event_slug) DO UPDATE SET
              tags_json = excluded.tags_json,
              cached_at = excluded.cached_at
            """,
            (event_slug, tags_json, now),
        )
        self._conn.commit()

    def get(self, event_slug: EventSlug) -> list[str] | None:
        """Return cached tags for ``event_slug``, or ``None`` if absent.

        Args:
            event_slug: Polymarket event slug (URL fragment).

        Returns:
            The decoded tag list, or ``None`` if no cached row exists.

        Raises:
            ValueError: If the stored JSON does not decode to a list.
        """
        row = self._conn.execute(
            "SELECT tags_json FROM event_tag_cache WHERE event_slug = ?",
            (event_slug,),
        ).fetchone()
        if row is None:
            return None
        decoded = json.loads(row["tags_json"])
        if not isinstance(decoded, list):
            msg = f"event_tag_cache.tags_json must decode to list, got {type(decoded).__name__}"
            raise ValueError(msg)
        return [str(item) for item in decoded]


@dataclass(frozen=True, slots=True)
class MarketTick:
    """One per-asset orderbook snapshot row in the ``market_ticks`` table.

    All numeric fields beyond the composite primary key are nullable: an asset
    with one-sided liquidity (no bids, only asks) will have a ``best_bid`` of
    ``None``, propagating through ``mid_price``/``spread``. Detectors that read
    these rows are expected to filter NULLs explicitly.
    """

    asset_id: AssetId
    condition_id: ConditionId
    snapshot_at: int
    mid_price: float | None
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    bid_depth_top5: float | None
    ask_depth_top5: float | None
    last_trade_price: float | None


class MarketTicksRepo:
    """Append-only per-asset tick history.

    Rows dedupe on the composite primary key ``(asset_id, snapshot_at)`` via
    ``INSERT OR IGNORE`` — two snapshots of the same asset in the same second
    collapse to one row, matching the idempotency contract used by the other
    snapshot tables.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert(self, tick: MarketTick) -> bool:
        """Insert ``tick`` if its composite PK is unseen.

        Args:
            tick: Fully-populated tick row to persist.

        Returns:
            ``True`` if the row was newly inserted, ``False`` on PK collision.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO market_ticks (
              asset_id, condition_id, snapshot_at, mid_price, best_bid, best_ask,
              spread, bid_depth_top5, ask_depth_top5, last_trade_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tick.asset_id,
                tick.condition_id,
                tick.snapshot_at,
                tick.mid_price,
                tick.best_bid,
                tick.best_ask,
                tick.spread,
                tick.bid_depth_top5,
                tick.ask_depth_top5,
                tick.last_trade_price,
            ),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def latest_for_asset(self, asset_id: AssetId) -> MarketTick | None:
        """Return the most recent ``market_ticks`` row for an asset, or ``None``."""
        row = self._conn.execute(
            """
            SELECT asset_id, condition_id, snapshot_at, mid_price, best_bid,
                   best_ask, spread, bid_depth_top5, ask_depth_top5,
                   last_trade_price
              FROM market_ticks
             WHERE asset_id = ?
             ORDER BY snapshot_at DESC
             LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_market_tick(row)

    def recent_for_asset(self, asset_id: AssetId, *, limit: int = 200) -> list[MarketTick]:
        """Return the asset's most recent ticks, newest first.

        Args:
            asset_id: CLOB token id.
            limit: Max rows to return.

        Returns:
            Tick rows ordered by ``snapshot_at DESC``.
        """
        rows = self._conn.execute(
            """
            SELECT asset_id, condition_id, snapshot_at, mid_price, best_bid, best_ask,
                   spread, bid_depth_top5, ask_depth_top5, last_trade_price
              FROM market_ticks
             WHERE asset_id = ?
             ORDER BY snapshot_at DESC
             LIMIT ?
            """,
            (asset_id, limit),
        ).fetchall()
        return [_row_to_market_tick(row) for row in rows]

    def recent_mids_in_window(
        self,
        asset_id: AssetId,
        *,
        window_seconds: int,
        now_ts: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(snapshot_at, mid_price)`` pairs within the trailing window.

        Filters out rows whose ``mid_price`` is NULL so callers can compute a
        first/last delta without re-checking. Ordered ascending by
        ``snapshot_at`` (oldest → newest).

        Args:
            asset_id: CLOB token id.
            window_seconds: Inclusive trailing window length.
            now_ts: Override for the upper bound (Unix seconds). ``None`` uses
                the current wall-clock time. Tests pass an explicit value to
                avoid sleep-induced flakes.

        Returns:
            Pairs ordered by ``snapshot_at`` ascending.
        """
        upper = _now_seconds() if now_ts is None else now_ts
        lower = upper - window_seconds
        rows = self._conn.execute(
            """
            SELECT snapshot_at, mid_price
              FROM market_ticks
             WHERE asset_id = ?
               AND mid_price IS NOT NULL
               AND snapshot_at > ?
               AND snapshot_at <= ?
             ORDER BY snapshot_at ASC
            """,
            (asset_id, lower, upper),
        ).fetchall()
        return [(int(row["snapshot_at"]), float(row["mid_price"])) for row in rows]

    def distinct_snapshot_count(self) -> int:
        """Return ``COUNT(DISTINCT snapshot_at)`` — total tick cycles recorded."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT snapshot_at) AS c FROM market_ticks",
        ).fetchone()
        if row is None:
            return 0
        return int(row["c"])

    def count_by_asset(self) -> dict[str, int]:
        """Return ``{asset_id: tick_count}`` across the full table."""
        rows = self._conn.execute(
            "SELECT asset_id, COUNT(*) AS c FROM market_ticks GROUP BY asset_id",
        ).fetchall()
        return {row["asset_id"]: int(row["c"]) for row in rows}


def _row_to_market_tick(row: sqlite3.Row) -> MarketTick:
    """Convert a ``market_ticks`` row to a ``MarketTick`` dataclass."""
    return MarketTick(
        asset_id=AssetId(row["asset_id"]),
        condition_id=ConditionId(row["condition_id"]),
        snapshot_at=int(row["snapshot_at"]),
        mid_price=_optional_float(row["mid_price"]),
        best_bid=_optional_float(row["best_bid"]),
        best_ask=_optional_float(row["best_ask"]),
        spread=_optional_float(row["spread"]),
        bid_depth_top5=_optional_float(row["bid_depth_top5"]),
        ask_depth_top5=_optional_float(row["ask_depth_top5"]),
        last_trade_price=_optional_float(row["last_trade_price"]),
    )


@dataclass(frozen=True, slots=True)
class WalletCluster:
    """A coordinated-wallet cluster persisted by the cluster detector."""

    cluster_id: str
    member_count: int
    first_member_created_at: int
    last_member_created_at: int
    shared_market_count: int
    behavior_tag: str | None
    detection_score: int
    first_detected_at: int
    last_active_at: int


class WalletClustersRepo:
    """CRUD for the ``wallet_clusters`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, cluster: WalletCluster) -> None:
        """Insert or update one ``wallet_clusters`` row by ``cluster_id``.

        Args:
            cluster: Fully-populated ``WalletCluster`` to persist.
        """
        self._conn.execute(
            """
            INSERT INTO wallet_clusters (
              cluster_id, member_count,
              first_member_created_at, last_member_created_at,
              shared_market_count, behavior_tag, detection_score,
              first_detected_at, last_active_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cluster_id) DO UPDATE SET
              member_count = excluded.member_count,
              first_member_created_at = excluded.first_member_created_at,
              last_member_created_at = excluded.last_member_created_at,
              shared_market_count = excluded.shared_market_count,
              behavior_tag = excluded.behavior_tag,
              detection_score = excluded.detection_score,
              first_detected_at = excluded.first_detected_at,
              last_active_at = excluded.last_active_at
            """,
            (
                cluster.cluster_id,
                cluster.member_count,
                cluster.first_member_created_at,
                cluster.last_member_created_at,
                cluster.shared_market_count,
                cluster.behavior_tag,
                cluster.detection_score,
                cluster.first_detected_at,
                cluster.last_active_at,
            ),
        )
        self._conn.commit()

    def get(self, cluster_id: str) -> WalletCluster | None:
        """Return the cluster row for ``cluster_id``, or ``None`` if absent."""
        row = self._conn.execute(
            """
            SELECT cluster_id, member_count,
                   first_member_created_at, last_member_created_at,
                   shared_market_count, behavior_tag, detection_score,
                   first_detected_at, last_active_at
              FROM wallet_clusters
             WHERE cluster_id = ?
            """,
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_wallet_cluster(row)

    def list_all(self) -> list[WalletCluster]:
        """Return every cluster row, ordered by ``first_detected_at`` desc."""
        rows = self._conn.execute(
            """
            SELECT cluster_id, member_count,
                   first_member_created_at, last_member_created_at,
                   shared_market_count, behavior_tag, detection_score,
                   first_detected_at, last_active_at
              FROM wallet_clusters
             ORDER BY first_detected_at DESC
            """,
        ).fetchall()
        return [_row_to_wallet_cluster(row) for row in rows]

    def update_last_active(self, cluster_id: str, ts: int) -> None:
        """Stamp ``last_active_at`` for ``cluster_id`` to ``ts``."""
        self._conn.execute(
            "UPDATE wallet_clusters SET last_active_at = ? WHERE cluster_id = ?",
            (ts, cluster_id),
        )
        self._conn.commit()


def _row_to_wallet_cluster(row: sqlite3.Row) -> WalletCluster:
    """Convert a ``wallet_clusters`` row to a ``WalletCluster`` dataclass."""
    return WalletCluster(
        cluster_id=row["cluster_id"],
        member_count=int(row["member_count"]),
        first_member_created_at=int(row["first_member_created_at"]),
        last_member_created_at=int(row["last_member_created_at"]),
        shared_market_count=int(row["shared_market_count"]),
        behavior_tag=row["behavior_tag"],
        detection_score=int(row["detection_score"]),
        first_detected_at=int(row["first_detected_at"]),
        last_active_at=int(row["last_active_at"]),
    )


class WalletClusterMembersRepo:
    """CRUD for the ``wallet_cluster_members`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def add_member(self, cluster_id: str, wallet: str) -> None:
        """Add ``wallet`` to ``cluster_id``. No-ops if already a member."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO wallet_cluster_members (cluster_id, wallet)
            VALUES (?, ?)
            """,
            (cluster_id, wallet),
        )
        self._conn.commit()

    def members_of(self, cluster_id: str) -> list[str]:
        """Return every wallet in ``cluster_id``, sorted ascending."""
        rows = self._conn.execute(
            """
            SELECT wallet FROM wallet_cluster_members
             WHERE cluster_id = ?
             ORDER BY wallet ASC
            """,
            (cluster_id,),
        ).fetchall()
        return [row["wallet"] for row in rows]

    def cluster_for_wallet(self, wallet: str) -> str | None:
        """Return the first cluster_id ``wallet`` belongs to, or ``None``.

        A wallet should only belong to one cluster in the current detector
        design, but the schema allows membership in multiple. The repo
        returns the first match (by insertion order) for callers that need
        a single answer.
        """
        row = self._conn.execute(
            """
            SELECT cluster_id FROM wallet_cluster_members
             WHERE wallet = ?
             LIMIT 1
            """,
            (wallet,),
        ).fetchone()
        if row is None:
            return None
        return row["cluster_id"]


@dataclass(frozen=True, slots=True)
class OpenPaperPosition:
    """An entry row in ``paper_trades`` with no matching exit."""

    trade_id: int
    triggering_alert_key: str | None
    source_wallet: str | None
    condition_id: ConditionId
    asset_id: AssetId
    outcome: str
    shares: float
    fill_price: float
    cost_usd: float
    nav_after_usd: float
    ts: int
    triggering_alert_detector: str | None = None
    rule_variant: str | None = None


@dataclass(frozen=True, slots=True)
class PaperSummary:
    """Aggregate stats for the ``paper status`` CLI."""

    starting_bankroll: float
    current_nav: float
    total_return_pct: float
    realized_pnl: float
    open_positions: int
    closed_positions: int


@dataclass(frozen=True, slots=True)
class SourceSummary:
    """Per-(detector, rule_variant) PnL aggregate for the paper status CLI."""

    detector: str | None
    rule_variant: str | None
    open_count: int
    resolved_count: int
    realized_pnl: float
    win_rate: float


class PaperTradesRepo:
    """CRUD + aggregates for the ``paper_trades`` table.

    ``paper_trades`` stores both entries (``trade_kind = 'entry'``) and exits
    (``trade_kind = 'exit'``). An exit row's ``parent_trade_id`` points at the
    entry it closes; an entry with no matching exit is an open position.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_entry(
        self,
        *,
        triggering_alert_key: str | None,
        triggering_alert_detector: str | None,
        rule_variant: str | None,
        source_wallet: str | None,
        condition_id: ConditionId,
        asset_id: AssetId,
        outcome: str,
        shares: float,
        fill_price: float,
        cost_usd: float,
        nav_after_usd: float,
        ts: int,
    ) -> int:
        """Insert an entry row and return its ``trade_id``.

        A non-null ``triggering_alert_key`` is unique among entries on the
        ``(triggering_alert_key, COALESCE(rule_variant, ''))`` index, so the
        same alert key with two distinct ``rule_variant`` values (e.g. velocity
        twin trades) coexists, but re-inserting the same ``(key, variant)``
        pair raises ``sqlite3.IntegrityError`` so callers can dedupe without an
        extra SELECT.
        """
        cur = self._conn.execute(
            """
            INSERT INTO paper_trades (
              trade_kind, triggering_alert_key, parent_trade_id, source_wallet,
              condition_id, asset_id, outcome, shares, fill_price, cost_usd,
              nav_after_usd, ts, triggering_alert_detector, rule_variant
            ) VALUES ('entry', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                triggering_alert_key,
                source_wallet,
                condition_id,
                asset_id,
                outcome,
                shares,
                fill_price,
                cost_usd,
                nav_after_usd,
                ts,
                triggering_alert_detector,
                rule_variant,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def insert_exit(
        self,
        *,
        parent_trade_id: int,
        condition_id: ConditionId,
        asset_id: AssetId,
        outcome: str,
        shares: float,
        fill_price: float,
        cost_usd: float,
        nav_after_usd: float,
        ts: int,
    ) -> int:
        """Insert an exit row linked to ``parent_trade_id``; return its ``trade_id``."""
        cur = self._conn.execute(
            """
            INSERT INTO paper_trades (
              trade_kind, triggering_alert_key, parent_trade_id, source_wallet,
              condition_id, asset_id, outcome, shares, fill_price, cost_usd,
              nav_after_usd, ts
            ) VALUES ('exit', NULL, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent_trade_id,
                condition_id,
                asset_id,
                outcome,
                shares,
                fill_price,
                cost_usd,
                nav_after_usd,
                ts,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def list_open_positions(self) -> list[OpenPaperPosition]:
        """Return entries with no matching exit, oldest first."""
        rows = self._conn.execute(
            """
            SELECT e.trade_id, e.triggering_alert_key, e.source_wallet,
                   e.condition_id, e.asset_id, e.outcome, e.shares,
                   e.fill_price, e.cost_usd, e.nav_after_usd, e.ts,
                   e.triggering_alert_detector, e.rule_variant
              FROM paper_trades e
             WHERE e.trade_kind = 'entry'
               AND NOT EXISTS (
                 SELECT 1 FROM paper_trades x
                  WHERE x.parent_trade_id = e.trade_id
               )
             ORDER BY e.ts ASC
            """,
        ).fetchall()
        return [
            OpenPaperPosition(
                trade_id=int(r["trade_id"]),
                triggering_alert_key=r["triggering_alert_key"],
                source_wallet=r["source_wallet"],
                condition_id=ConditionId(r["condition_id"]),
                asset_id=AssetId(r["asset_id"]),
                outcome=r["outcome"],
                shares=float(r["shares"]),
                fill_price=float(r["fill_price"]),
                cost_usd=float(r["cost_usd"]),
                nav_after_usd=float(r["nav_after_usd"]),
                ts=int(r["ts"]),
                triggering_alert_detector=r["triggering_alert_detector"],
                rule_variant=r["rule_variant"],
            )
            for r in rows
        ]

    def compute_cost_basis_nav(self, *, starting_bankroll: float) -> float:
        """Return ``starting_bankroll + realized_pnl``.

        Realized PnL is the sum of ``exit.cost_usd - parent_entry.cost_usd``
        across resolved trades. Open positions sit at cost basis and contribute
        nothing to NAV until they are closed.
        """
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(x.cost_usd - e.cost_usd), 0.0) AS realized_pnl
              FROM paper_trades x
              JOIN paper_trades e ON e.trade_id = x.parent_trade_id
             WHERE x.trade_kind = 'exit' AND e.trade_kind = 'entry'
            """,
        ).fetchone()
        realized = float(row["realized_pnl"] or 0.0)
        return starting_bankroll + realized

    def summary_stats(self, *, starting_bankroll: float) -> PaperSummary:
        """Aggregate stats for the ``paper status`` CLI."""
        nav = self.compute_cost_basis_nav(starting_bankroll=starting_bankroll)
        realized = nav - starting_bankroll
        open_n = self._conn.execute(
            """
            SELECT COUNT(*) AS n FROM paper_trades e
             WHERE e.trade_kind = 'entry'
               AND NOT EXISTS (
                 SELECT 1 FROM paper_trades x
                  WHERE x.parent_trade_id = e.trade_id
               )
            """,
        ).fetchone()
        closed_n = self._conn.execute(
            "SELECT COUNT(*) AS n FROM paper_trades WHERE trade_kind = 'exit'",
        ).fetchone()
        return PaperSummary(
            starting_bankroll=starting_bankroll,
            current_nav=nav,
            total_return_pct=(realized / starting_bankroll * 100.0) if starting_bankroll else 0.0,
            realized_pnl=realized,
            open_positions=int(open_n["n"]),
            closed_positions=int(closed_n["n"]),
        )

    def summary_by_source(self) -> list[SourceSummary]:
        """Per-source aggregate of open count, resolved count, realized PnL, win rate."""
        rows = self._conn.execute(
            """
            SELECT
              e.triggering_alert_detector AS detector,
              e.rule_variant AS rule_variant,
              SUM(CASE WHEN x.trade_id IS NULL THEN 1 ELSE 0 END) AS open_count,
              SUM(CASE WHEN x.trade_id IS NOT NULL THEN 1 ELSE 0 END) AS resolved_count,
              COALESCE(SUM(x.cost_usd - e.cost_usd), 0.0) AS realized_pnl,
              AVG(CASE WHEN x.trade_id IS NOT NULL
                       THEN CASE WHEN x.cost_usd > e.cost_usd THEN 1.0 ELSE 0.0 END
                       ELSE NULL END) AS win_rate
            FROM paper_trades e
            LEFT JOIN paper_trades x
              ON x.parent_trade_id = e.trade_id AND x.trade_kind = 'exit'
            WHERE e.trade_kind = 'entry'
            GROUP BY e.triggering_alert_detector, e.rule_variant
            ORDER BY e.triggering_alert_detector, e.rule_variant
            """,
        ).fetchall()
        return [
            SourceSummary(
                detector=r["detector"],
                rule_variant=r["rule_variant"],
                open_count=int(r["open_count"] or 0),
                resolved_count=int(r["resolved_count"] or 0),
                realized_pnl=float(r["realized_pnl"] or 0.0),
                win_rate=float(r["win_rate"] or 0.0),
            )
            for r in rows
        ]
