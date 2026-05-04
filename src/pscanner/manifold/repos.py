"""Repositories for Manifold Markets daemon tables.

Each repo wraps a single table with typed insert/get/query methods. All writes
commit immediately. Reads use the connection's current transaction state.

All repos receive a ``sqlite3.Connection`` at construction — the connection must
have the Manifold schema already applied (via ``init_manifold_tables``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId
from pscanner.manifold.models import ManifoldBet, ManifoldMarket, ManifoldUser


class ManifoldMarketsRepo:
    """Manage the ``manifold_markets`` table.

    Stores Manifold contracts (markets) by their opaque hash ID. ``raw_json``
    preserves the full API payload for forward compatibility.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind to an already-initialised connection.

        Args:
            conn: Open ``sqlite3.Connection`` with manifold schema applied.
        """
        self._conn = conn

    def insert_or_replace(self, market: ManifoldMarket) -> None:
        """Upsert a market row. Replaces all columns on conflict.

        Args:
            market: ``ManifoldMarket`` model to persist.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO manifold_markets (
              id, creator_id, question, outcome_type, mechanism,
              prob_at_last_seen, volume, total_liquidity, is_resolved,
              resolution_time, close_time, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market.id,
                market.creator_id,
                market.question,
                market.outcome_type,
                market.mechanism,
                market.prob,
                market.volume,
                market.total_liquidity,
                int(market.is_resolved),
                market.resolution_time,
                market.close_time,
                market.model_dump_json(by_alias=True),
            ),
        )
        self._conn.commit()

    def get_by_id(self, market_id: ManifoldMarketId) -> ManifoldMarket | None:
        """Fetch a single market by ID, or ``None`` if absent.

        Args:
            market_id: Manifold market hash ID.

        Returns:
            ``ManifoldMarket`` model or ``None``.
        """
        row = self._conn.execute(
            "SELECT raw_json FROM manifold_markets WHERE id = ?",
            (market_id,),
        ).fetchone()
        if row is None:
            return None
        return ManifoldMarket.model_validate_json(row[0])

    def iter_chronological(self) -> Iterator[ManifoldMarket]:
        """Yield all markets ordered by ``close_time`` ascending (nulls last).

        Yields:
            ``ManifoldMarket`` models in close-time order.
        """
        rows = self._conn.execute(
            "SELECT raw_json FROM manifold_markets ORDER BY close_time ASC NULLS LAST"
        ).fetchall()
        for (raw,) in rows:
            yield ManifoldMarket.model_validate_json(raw)


class ManifoldBetsRepo:
    """Manage the ``manifold_bets`` table.

    Stores individual bets (trades) by their opaque ID. Amounts are in mana.

    Note: ``ManifoldBet.shares`` and ``ManifoldBet.fees`` are NOT persisted by
    this repo (the table schema omits them). After a DB round-trip both will be
    ``None``. If a future use case needs CFMM position sizing analysis, add the
    columns to the schema migration before relying on these fields downstream.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind to an already-initialised connection.

        Args:
            conn: Open ``sqlite3.Connection`` with manifold schema applied.
        """
        self._conn = conn

    def insert_or_replace(self, bet: ManifoldBet) -> None:
        """Upsert a bet row. Replaces all columns on conflict.

        Args:
            bet: ``ManifoldBet`` model to persist.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO manifold_bets (
              id, user_id, contract_id, outcome, amount,
              prob_before, prob_after, created_time,
              is_filled, is_cancelled, limit_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bet.id,
                bet.user_id,
                bet.contract_id,
                bet.outcome,
                bet.amount,
                bet.prob_before,
                bet.prob_after,
                bet.created_time,
                int(bet.is_filled) if bet.is_filled is not None else None,
                int(bet.is_cancelled) if bet.is_cancelled is not None else None,
                bet.limit_prob,
            ),
        )
        self._conn.commit()

    def get_by_id(self, bet_id: str) -> ManifoldBet | None:
        """Fetch a single bet by ID, or ``None`` if absent.

        Args:
            bet_id: Bet ID string.

        Returns:
            ``ManifoldBet`` model or ``None``.
        """
        row = self._conn.execute(
            """
            SELECT id, user_id, contract_id, outcome, amount,
                   prob_before, prob_after, created_time,
                   is_filled, is_cancelled, limit_prob
            FROM manifold_bets WHERE id = ?
            """,
            (bet_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_bet(row)

    def iter_chronological(
        self,
        *,
        market_id: ManifoldMarketId | None = None,
    ) -> Iterator[ManifoldBet]:
        """Yield bets ordered by ``created_time`` ascending.

        Args:
            market_id: If provided, restrict to bets on this market.

        Yields:
            ``ManifoldBet`` models in creation-time order.
        """
        if market_id is not None:
            rows = self._conn.execute(
                """
                SELECT id, user_id, contract_id, outcome, amount,
                       prob_before, prob_after, created_time,
                       is_filled, is_cancelled, limit_prob
                FROM manifold_bets
                WHERE contract_id = ?
                ORDER BY created_time ASC
                """,
                (market_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, user_id, contract_id, outcome, amount,
                       prob_before, prob_after, created_time,
                       is_filled, is_cancelled, limit_prob
                FROM manifold_bets
                ORDER BY created_time ASC
                """
            ).fetchall()
        for row in rows:
            yield _row_to_bet(row)


class ManifoldUsersRepo:
    """Manage the ``manifold_users`` table.

    Stores Manifold user profiles by their opaque hash ID. ``raw_json``
    preserves the full API payload.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind to an already-initialised connection.

        Args:
            conn: Open ``sqlite3.Connection`` with manifold schema applied.
        """
        self._conn = conn

    def insert_or_replace(self, user: ManifoldUser) -> None:
        """Upsert a user row. Replaces all columns on conflict.

        Args:
            user: ``ManifoldUser`` model to persist.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO manifold_users (
              id, username, name, created_time, raw_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                user.id,
                user.username,
                user.name,
                user.created_time,
                user.model_dump_json(by_alias=True),
            ),
        )
        self._conn.commit()

    def get_by_id(self, user_id: ManifoldUserId) -> ManifoldUser | None:
        """Fetch a single user by ID, or ``None`` if absent.

        Args:
            user_id: Manifold user hash ID.

        Returns:
            ``ManifoldUser`` model or ``None``.
        """
        row = self._conn.execute(
            "SELECT raw_json FROM manifold_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return ManifoldUser.model_validate_json(row[0])

    def iter_chronological(self) -> Iterator[ManifoldUser]:
        """Yield all users ordered by ``created_time`` ascending.

        Yields:
            ``ManifoldUser`` models in creation-time order.
        """
        rows = self._conn.execute(
            "SELECT raw_json FROM manifold_users ORDER BY created_time ASC"
        ).fetchall()
        for (raw,) in rows:
            yield ManifoldUser.model_validate_json(raw)


def _row_to_bet(row: sqlite3.Row) -> ManifoldBet:
    """Reconstruct a ``ManifoldBet`` from a raw DB row.

    Args:
        row: Row from ``manifold_bets`` with columns in the standard select order.

    Returns:
        ``ManifoldBet`` model.
    """
    (
        bet_id,
        user_id,
        contract_id,
        outcome,
        amount,
        prob_before,
        prob_after,
        created_time,
        is_filled_raw,
        is_cancelled_raw,
        limit_prob,
    ) = row
    data = {
        "id": bet_id,
        "userId": user_id,
        "contractId": contract_id,
        "outcome": outcome,
        "amount": amount,
        "probBefore": prob_before,
        "probAfter": prob_after,
        "createdTime": created_time,
        "isFilled": bool(is_filled_raw) if is_filled_raw is not None else None,
        "isCancelled": bool(is_cancelled_raw) if is_cancelled_raw is not None else None,
        "limitProb": limit_prob,
    }
    return ManifoldBet.model_validate(data)
