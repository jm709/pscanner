"""SQLite repository classes — one per table.

Wave 1 freezes the public method signatures and the small dataclasses that the
repos hand back. Wave 2's ``repo-layer`` agent implements the bodies.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pscanner.alerts.models import Alert
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
    """Cached subset of ``Market`` used by detectors at runtime."""

    market_id: str
    event_id: str | None
    title: str | None
    liquidity_usd: float | None
    volume_usd: float | None
    outcome_prices_json: str | None
    active: bool
    cached_at: int


class TrackedWalletsRepo:
    """CRUD for the ``tracked_wallets`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        raise NotImplementedError("Wave 2: repo-layer")

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
        raise NotImplementedError("Wave 2: repo-layer")

    def list_active(self, min_winrate: float, min_resolved: int) -> list[TrackedWallet]:
        """Return wallets meeting the smart-money quality bar.

        Args:
            min_winrate: Inclusive winrate threshold.
            min_resolved: Inclusive minimum closed-position count.

        Returns:
            Wallets passing both filters, ordered by winrate desc.
        """
        raise NotImplementedError("Wave 2: repo-layer")

    def list_all(self) -> list[TrackedWallet]:
        """Return every row in the table (no filtering)."""
        raise NotImplementedError("Wave 2: repo-layer")


class PositionSnapshotsRepo:
    """CRUD for the ``wallet_position_snapshots`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        raise NotImplementedError("Wave 2: repo-layer")

    def upsert(
        self,
        address: str,
        market_id: str,
        side: str,
        size: float,
        avg_price: float,
    ) -> None:
        """Insert or update the (address, market, side) snapshot row."""
        raise NotImplementedError("Wave 2: repo-layer")

    def get_for_wallet(self, address: str) -> list[PositionSnapshot]:
        """Return all snapshots for a single wallet."""
        raise NotImplementedError("Wave 2: repo-layer")

    def previous_size(self, address: str, market_id: str, side: str) -> float | None:
        """Return the most-recently-stored size, or ``None`` if no row exists."""
        raise NotImplementedError("Wave 2: repo-layer")


class WalletFirstSeenRepo:
    """CRUD for the ``wallet_first_seen`` cache."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        raise NotImplementedError("Wave 2: repo-layer")

    def get(self, address: str) -> WalletFirstSeen | None:
        """Return the cached first-seen row for a wallet, or ``None``."""
        raise NotImplementedError("Wave 2: repo-layer")

    def upsert(
        self,
        address: str,
        first_activity_at: int | None,
        total_trades: int | None,
    ) -> None:
        """Insert or update the cache row, stamping ``cached_at`` to now."""
        raise NotImplementedError("Wave 2: repo-layer")


class MarketCacheRepo:
    """CRUD for the ``market_cache`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        raise NotImplementedError("Wave 2: repo-layer")

    def upsert(self, market: Market) -> None:
        """Insert or update a cache row from a freshly-fetched ``Market``.

        Args:
            market: Source-of-truth market model (gamma response).
        """
        raise NotImplementedError("Wave 2: repo-layer")

    def get(self, market_id: str) -> CachedMarket | None:
        """Return the cached row for a market, or ``None``."""
        raise NotImplementedError("Wave 2: repo-layer")

    def list_active(self) -> list[CachedMarket]:
        """Return every cached market with ``active = 1``."""
        raise NotImplementedError("Wave 2: repo-layer")


class AlertsRepo:
    """CRUD for the ``alerts`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        raise NotImplementedError("Wave 2: repo-layer")

    def insert_if_new(self, alert: Alert) -> bool:
        """Insert ``alert`` if its ``alert_key`` is unseen.

        Args:
            alert: The alert to persist.

        Returns:
            ``True`` if the row was inserted, ``False`` if the key was already
            present (idempotency dedupe hit).
        """
        raise NotImplementedError("Wave 2: repo-layer")

    def recent(self, detector: str | None = None, limit: int = 100) -> list[Alert]:
        """Return the most recent alerts, optionally filtered by detector.

        Args:
            detector: If set, only return alerts where ``detector = <value>``.
            limit: Max rows to return.

        Returns:
            Alerts in descending ``created_at`` order.
        """
        raise NotImplementedError("Wave 2: repo-layer")
