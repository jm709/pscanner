"""Repositories for the corpus subsystem.

One file mirrors ``pscanner.store.repo``. Each repo wraps a single table
(or two related tables) with typed insert/get/update methods. All methods
take a ``sqlite3.Connection`` injected at construction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class CorpusMarket:
    """A market that qualifies for the corpus (volume gate passed).

    Identifies a closed Polymarket market by its ``condition_id``. The
    ``backfill_state`` is tracked separately on the row and progresses
    ``pending → in_progress → complete | failed``.
    """

    condition_id: str
    event_slug: str
    category: str | None
    closed_at: int
    total_volume_usd: float
    enumerated_at: int


class CorpusMarketsRepo:
    """Manage the ``corpus_markets`` work-queue table.

    All write methods commit immediately. Reads use the connection's
    default transaction state.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_pending(self, market: CorpusMarket) -> None:
        """Insert a market in ``pending`` state. Idempotent (INSERT OR IGNORE)."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO corpus_markets (
              condition_id, event_slug, category, closed_at, total_volume_usd,
              backfill_state, enumerated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                market.condition_id,
                market.event_slug,
                market.category,
                market.closed_at,
                market.total_volume_usd,
                market.enumerated_at,
            ),
        )
        self._conn.commit()

    def next_pending(self, *, limit: int) -> list[CorpusMarket]:
        """Return up to ``limit`` markets needing work, largest-volume-first.

        Includes both ``pending`` and ``in_progress`` rows (the latter cover
        the resume case after a crash). Failed rows are also included so
        re-running ``backfill`` retries them. Tied volume breaks by
        ``closed_at`` descending.
        """
        rows = self._conn.execute(
            """
            SELECT condition_id, event_slug, category, closed_at,
                   total_volume_usd, enumerated_at
            FROM corpus_markets
            WHERE backfill_state IN ('pending', 'in_progress', 'failed')
            ORDER BY total_volume_usd DESC, closed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            CorpusMarket(
                condition_id=row["condition_id"],
                event_slug=row["event_slug"],
                category=row["category"],
                closed_at=row["closed_at"],
                total_volume_usd=row["total_volume_usd"],
                enumerated_at=row["enumerated_at"],
            )
            for row in rows
        ]

    def get_last_offset(self, condition_id: str) -> int:
        """Return the last offset seen for a market, or 0 if not started."""
        row = self._conn.execute(
            "SELECT last_offset_seen FROM corpus_markets WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None or row["last_offset_seen"] is None:
            return 0
        return int(row["last_offset_seen"])

    def mark_in_progress(self, condition_id: str, *, started_at: int) -> None:
        """Transition a market to ``in_progress`` and record the start timestamp."""
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'in_progress',
                backfill_started_at = COALESCE(backfill_started_at, ?)
            WHERE condition_id = ?
            """,
            (started_at, condition_id),
        )
        self._conn.commit()

    def record_progress(
        self,
        condition_id: str,
        *,
        last_offset: int,
        inserted_delta: int,
    ) -> None:
        """Advance the offset cursor and accumulate the inserted-trade count."""
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET last_offset_seen = ?,
                trades_pulled_count = trades_pulled_count + ?
            WHERE condition_id = ?
            """,
            (last_offset, inserted_delta, condition_id),
        )
        self._conn.commit()

    def mark_complete(
        self,
        condition_id: str,
        *,
        completed_at: int,
        truncated: bool,
    ) -> None:
        """Mark a market as fully backfilled.

        Sets the truncation flag when the offset cap was hit before all trades
        were retrieved.
        """
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'complete',
                backfill_completed_at = ?,
                truncated_at_offset_cap = ?,
                error_message = NULL
            WHERE condition_id = ?
            """,
            (completed_at, 1 if truncated else 0, condition_id),
        )
        self._conn.commit()

    def mark_failed(self, condition_id: str, *, error_message: str) -> None:
        """Record a terminal error; backfill will retry this row on next run."""
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'failed',
                error_message = ?
            WHERE condition_id = ?
            """,
            (error_message, condition_id),
        )
        self._conn.commit()


class CorpusStateRepo:
    """Tiny key/value cursor table for cross-cutting orchestrator state."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def get(self, key: str) -> str | None:
        """Return the stored string value for ``key``, or ``None`` if absent."""
        row = self._conn.execute(
            "SELECT value FROM corpus_state WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row["value"])

    def get_int(self, key: str) -> int | None:
        """Return the stored value parsed as ``int``, or ``None`` if absent."""
        value = self.get(key)
        return None if value is None else int(value)

    def set(self, key: str, value: str, *, updated_at: int) -> None:
        """Upsert ``key`` with a new ``value`` and timestamp."""
        self._conn.execute(
            """
            INSERT INTO corpus_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )
        self._conn.commit()


_NOTIONAL_FLOOR_USD: Final[float] = 10.0


@dataclass(frozen=True)
class CorpusTrade:
    """One BUY or SELL fill captured by the market-walker.

    Wallet addresses are normalized to lowercase at insert time. ``bs`` is
    ``BUY`` or ``SELL``; ``outcome_side`` is ``YES`` or ``NO``. ``price``
    is the implied probability paid (already normalized so YES@0.7 and
    NO@0.3 are equivalent buys of the same outcome).
    """

    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    outcome_side: str
    bs: str
    price: float
    size: float
    notional_usd: float
    ts: int


class CorpusTradesRepo:
    """Append-only writes + chronological streaming reads on ``corpus_trades``.

    The notional floor (``$10``) is applied at insert time — sub-floor
    trades never land. The unique constraint
    ``(tx_hash, asset_id, wallet_address)`` makes ``insert_batch``
    idempotent.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_batch(self, trades: Iterable[CorpusTrade]) -> int:
        """Insert trades, skipping duplicates and sub-floor notionals.

        Returns the number of rows actually inserted.
        """
        rows = []
        for t in trades:
            if t.notional_usd < _NOTIONAL_FLOOR_USD:
                continue
            rows.append(
                (
                    t.tx_hash,
                    t.asset_id,
                    t.wallet_address.lower(),
                    t.condition_id,
                    t.outcome_side,
                    t.bs,
                    t.price,
                    t.size,
                    t.notional_usd,
                    t.ts,
                )
            )
        if not rows:
            return 0
        cur = self._conn.executemany(
            """
            INSERT OR IGNORE INTO corpus_trades (
              tx_hash, asset_id, wallet_address, condition_id,
              outcome_side, bs, price, size, notional_usd, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return cur.rowcount or 0

    def iter_chronological(self) -> Iterator[CorpusTrade]:
        """Yield every trade in (ts, tx_hash, asset_id) order.

        Tie-breaking on ``(tx_hash, asset_id)`` makes the iteration
        order deterministic for the streaming feature pipeline.
        """
        cursor = self._conn.execute(
            """
            SELECT tx_hash, asset_id, wallet_address, condition_id,
                   outcome_side, bs, price, size, notional_usd, ts
            FROM corpus_trades
            ORDER BY ts, tx_hash, asset_id
            """
        )
        for row in cursor:
            yield CorpusTrade(
                tx_hash=row["tx_hash"],
                asset_id=row["asset_id"],
                wallet_address=row["wallet_address"],
                condition_id=row["condition_id"],
                outcome_side=row["outcome_side"],
                bs=row["bs"],
                price=row["price"],
                size=row["size"],
                notional_usd=row["notional_usd"],
                ts=row["ts"],
            )
