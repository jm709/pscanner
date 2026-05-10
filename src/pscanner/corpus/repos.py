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

    Identifies a closed market by ``(platform, condition_id)``. The
    ``backfill_state`` is tracked separately on the row and progresses
    ``pending → in_progress → complete | failed``.
    """

    condition_id: str
    event_slug: str
    category: str | None
    closed_at: int
    total_volume_usd: float
    enumerated_at: int
    market_slug: str
    platform: str = "polymarket"


class CorpusMarketsRepo:
    """Manage the ``corpus_markets`` work-queue table.

    All write methods commit immediately. Reads use the connection's
    default transaction state.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_pending(self, market: CorpusMarket) -> int:
        """Insert a market in ``pending`` state. Idempotent (INSERT OR IGNORE).

        Also backfills ``market_slug`` on existing rows where it was NULL —
        this lets a re-enumeration after the schema migration populate the
        new column without touching other fields.

        Returns:
            1 if a new row was inserted, 0 if the market was already present.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO corpus_markets (
              platform, condition_id, event_slug, category, closed_at, total_volume_usd,
              market_slug, backfill_state, enumerated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                market.platform,
                market.condition_id,
                market.event_slug,
                market.category,
                market.closed_at,
                market.total_volume_usd,
                market.market_slug,
                market.enumerated_at,
            ),
        )
        inserted = cur.rowcount or 0
        # Backfill market_slug on rows that pre-date the migration.
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET market_slug = ?
            WHERE platform = ? AND condition_id = ? AND market_slug IS NULL
            """,
            (market.market_slug, market.platform, market.condition_id),
        )
        self._conn.commit()
        return inserted

    def next_pending(self, *, limit: int, platform: str = "polymarket") -> list[CorpusMarket]:
        """Return up to ``limit`` markets needing work, largest-volume-first.

        Includes both ``pending`` and ``in_progress`` rows (the latter cover
        the resume case after a crash). Failed rows are also included so
        re-running ``backfill`` retries them. Tied volume breaks by
        ``closed_at`` descending. Scoped to a single ``platform``.
        """
        rows = self._conn.execute(
            """
            SELECT condition_id, event_slug, category, closed_at,
                   total_volume_usd, market_slug, enumerated_at
            FROM corpus_markets
            WHERE platform = ? AND backfill_state IN ('pending', 'in_progress', 'failed')
            ORDER BY total_volume_usd DESC, closed_at DESC
            LIMIT ?
            """,
            (platform, limit),
        ).fetchall()
        return [
            CorpusMarket(
                condition_id=row["condition_id"],
                event_slug=row["event_slug"],
                category=row["category"],
                closed_at=row["closed_at"],
                total_volume_usd=row["total_volume_usd"],
                market_slug=row["market_slug"] or "",
                enumerated_at=row["enumerated_at"],
                platform=platform,
            )
            for row in rows
        ]

    def get_last_offset(self, condition_id: str, *, platform: str = "polymarket") -> int:
        """Return the last offset seen for a market, or 0 if not started."""
        row = self._conn.execute(
            """
            SELECT last_offset_seen FROM corpus_markets
            WHERE platform = ? AND condition_id = ?
            """,
            (platform, condition_id),
        ).fetchone()
        if row is None or row["last_offset_seen"] is None:
            return 0
        return int(row["last_offset_seen"])

    def mark_in_progress(
        self,
        condition_id: str,
        *,
        started_at: int,
        platform: str = "polymarket",
    ) -> None:
        """Transition a market to ``in_progress`` and record the start timestamp."""
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'in_progress',
                backfill_started_at = COALESCE(backfill_started_at, ?)
            WHERE platform = ? AND condition_id = ?
            """,
            (started_at, platform, condition_id),
        )
        self._conn.commit()

    def record_progress(
        self,
        condition_id: str,
        *,
        last_offset: int,
        inserted_delta: int,
        platform: str = "polymarket",
    ) -> None:
        """Advance the offset cursor and accumulate the inserted-trade count."""
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET last_offset_seen = ?,
                trades_pulled_count = trades_pulled_count + ?
            WHERE platform = ? AND condition_id = ?
            """,
            (last_offset, inserted_delta, platform, condition_id),
        )
        self._conn.commit()

    def mark_complete(
        self,
        condition_id: str,
        *,
        completed_at: int,
        truncated: bool,
        platform: str = "polymarket",
    ) -> None:
        """Mark a market as fully backfilled.

        Sets the truncation flag when the offset cap was hit before all trades
        were retrieved.

        Also rewrites ``closed_at`` to ``MAX(corpus_trades.ts)`` for this
        market, replacing the placeholder ``now_ts`` written at enumeration
        time. Without this, every row gets the enumerator's run time and the
        downstream ``temporal_split`` collapses to a hash split. Falls back
        to leaving ``closed_at`` unchanged if the market has no observed
        trades (shouldn't happen given the $1M volume gate, but guarded).
        """
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'complete',
                backfill_completed_at = ?,
                truncated_at_offset_cap = ?,
                error_message = NULL,
                closed_at = COALESCE(
                    (SELECT MAX(ts) FROM corpus_trades
                     WHERE platform = ? AND condition_id = ?),
                    closed_at
                )
            WHERE platform = ? AND condition_id = ?
            """,
            (
                completed_at,
                1 if truncated else 0,
                platform,
                condition_id,
                platform,
                condition_id,
            ),
        )
        self._conn.commit()

    def mark_failed(
        self,
        condition_id: str,
        *,
        error_message: str,
        platform: str = "polymarket",
    ) -> None:
        """Record a terminal error; backfill will retry this row on next run."""
        self._conn.execute(
            """
            UPDATE corpus_markets
            SET backfill_state = 'failed',
                error_message = ?
            WHERE platform = ? AND condition_id = ?
            """,
            (error_message, platform, condition_id),
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


_NOTIONAL_FLOORS: Final[dict[str, float]] = {
    "polymarket": 10.0,
    "manifold": 100.0,
    "kalshi": 10.0,  # placeholder; revisit when Kalshi ingestion ships
}
# Backward-compat alias for the existing module-private constant.
_NOTIONAL_FLOOR_USD: Final[float] = _NOTIONAL_FLOORS["polymarket"]


def _canonicalize_wallet_address(platform: str, address: str) -> str:
    """Normalize a wallet address for storage.

    Polymarket addresses are EVM-derived hex strings — case-insensitive,
    canonically stored as lowercase to dedupe writes that round-trip through
    different sources. Manifold ``user_id`` and any future platform-native
    identifier types are case-sensitive opaque hashes; preserve them verbatim.
    """
    if platform == "polymarket":
        return address.lower()
    return address


@dataclass(frozen=True)
class CorpusTrade:
    """One BUY or SELL fill captured by the market-walker.

    Polymarket wallet addresses are canonicalized to lowercase at insert
    time (EVM hex is case-insensitive). Manifold ``user_id`` hashes are
    case-sensitive and preserved verbatim. ``bs`` is ``BUY`` or ``SELL``;
    ``outcome_side`` is ``YES`` or ``NO``. ``price`` is the implied
    probability paid (already normalized so YES@0.7 and NO@0.3 are
    equivalent buys of the same outcome).
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
    platform: str = "polymarket"


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
            floor = _NOTIONAL_FLOORS.get(t.platform, _NOTIONAL_FLOORS["polymarket"])
            if t.notional_usd < floor:
                continue
            rows.append(
                (
                    t.platform,
                    t.tx_hash,
                    t.asset_id,
                    _canonicalize_wallet_address(t.platform, t.wallet_address),
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
              platform, tx_hash, asset_id, wallet_address, condition_id,
              outcome_side, bs, price, size, notional_usd, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return cur.rowcount or 0

    def iter_chronological(
        self, *, chunk_size: int = 50_000, platform: str = "polymarket"
    ) -> Iterator[CorpusTrade]:
        """Yield every trade in (ts, tx_hash, asset_id) order for ``platform``.

        Pages via keyset pagination on ``(ts, tx_hash, asset_id)`` so no
        long-lived read cursor is held across yields. This matters in
        WAL mode: an open read transaction pins the WAL snapshot and
        prevents the checkpointer from reclaiming pages, which causes
        unbounded WAL growth when callers (e.g. ``build_features``)
        write to other tables on the same connection during iteration.
        Materialising one chunk at a time with ``fetchall`` releases
        the read txn between chunks.

        Tie-breaking on ``(tx_hash, asset_id)`` makes the iteration
        order deterministic for the streaming feature pipeline.

        Performance depends on
        ``idx_corpus_trades_chrono_covering`` covering the
        platform-prefixed ORDER BY tuple and every SELECT column — without
        it SQLite falls back to a ``USE TEMP B-TREE FOR ORDER BY`` plan
        and performs a heap rowid lookup for each fetched row.

        Args:
            chunk_size: Rows per page. Default 50,000 (~5MB resident).
            platform: Platform to scope iteration to. Default
                ``"polymarket"`` preserves single-platform behavior for
                existing callers.
        """
        last: tuple[int, str, str] | None = None
        while True:
            if last is None:
                rows = self._conn.execute(
                    """
                    SELECT tx_hash, asset_id, wallet_address, condition_id,
                           outcome_side, bs, price, size, notional_usd, ts
                    FROM corpus_trades
                    WHERE platform = ?
                    ORDER BY ts, tx_hash, asset_id
                    LIMIT ?
                    """,
                    (platform, chunk_size),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT tx_hash, asset_id, wallet_address, condition_id,
                           outcome_side, bs, price, size, notional_usd, ts
                    FROM corpus_trades
                    WHERE platform = ? AND (ts, tx_hash, asset_id) > (?, ?, ?)
                    ORDER BY ts, tx_hash, asset_id
                    LIMIT ?
                    """,
                    (platform, last[0], last[1], last[2], chunk_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
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
                    platform=platform,
                )
            tail = rows[-1]
            last = (tail["ts"], tail["tx_hash"], tail["asset_id"])


@dataclass(frozen=True)
class MarketResolution:
    """Resolved outcome for a closed market."""

    condition_id: str
    winning_outcome_index: int
    outcome_yes_won: int  # 1 if YES won, 0 if NO won
    resolved_at: int
    source: str
    platform: str = "polymarket"


class MarketResolutionsRepo:
    """Upserts and lookups against ``market_resolutions``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, resolution: MarketResolution, *, recorded_at: int) -> None:
        """Insert or replace a resolution row for ``resolution.condition_id``."""
        self._conn.execute(
            """
            INSERT INTO market_resolutions (
              platform, condition_id, winning_outcome_index, outcome_yes_won,
              resolved_at, source, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, condition_id) DO UPDATE SET
              winning_outcome_index = excluded.winning_outcome_index,
              outcome_yes_won = excluded.outcome_yes_won,
              resolved_at = excluded.resolved_at,
              source = excluded.source,
              recorded_at = excluded.recorded_at
            """,
            (
                resolution.platform,
                resolution.condition_id,
                resolution.winning_outcome_index,
                resolution.outcome_yes_won,
                resolution.resolved_at,
                resolution.source,
                recorded_at,
            ),
        )
        self._conn.commit()

    def get(self, condition_id: str, *, platform: str = "polymarket") -> MarketResolution | None:
        """Return the resolution for ``(platform, condition_id)`` or ``None``."""
        row = self._conn.execute(
            """
            SELECT condition_id, winning_outcome_index, outcome_yes_won,
                   resolved_at, source
            FROM market_resolutions
            WHERE platform = ? AND condition_id = ?
            """,
            (platform, condition_id),
        ).fetchone()
        if row is None:
            return None
        return MarketResolution(
            condition_id=row["condition_id"],
            winning_outcome_index=row["winning_outcome_index"],
            outcome_yes_won=row["outcome_yes_won"],
            resolved_at=row["resolved_at"],
            source=row["source"],
            platform=platform,
        )

    def missing_for(
        self, condition_ids: Iterable[str], *, platform: str = "polymarket"
    ) -> list[str]:
        """Return the subset of ``condition_ids`` without a resolution row."""
        ids = list(condition_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"""
            SELECT condition_id FROM market_resolutions
            WHERE platform = ? AND condition_id IN ({placeholders})
            """,  # noqa: S608 — placeholder count is fixed to len(ids)
            (platform, *ids),
        ).fetchall()
        present = {row["condition_id"] for row in rows}
        return [cid for cid in ids if cid not in present]


@dataclass(frozen=True)
class TrainingExample:
    """One materialized row in the training_examples table.

    The full feature set computed at the trade's timestamp from prior
    trades only. ``label_won`` is the binary target.
    """

    tx_hash: str
    asset_id: str
    wallet_address: str
    condition_id: str
    trade_ts: int
    built_at: int
    prior_trades_count: int
    prior_buys_count: int
    prior_resolved_buys: int
    prior_wins: int
    prior_losses: int
    win_rate: float | None
    avg_implied_prob_paid: float | None
    realized_edge_pp: float | None
    prior_realized_pnl_usd: float
    avg_bet_size_usd: float | None
    median_bet_size_usd: float | None
    wallet_age_days: float
    seconds_since_last_trade: int | None
    prior_trades_30d: int
    top_category: str | None
    category_diversity: int
    bet_size_usd: float
    bet_size_rel_to_avg: float | None
    edge_confidence_weighted: float
    win_rate_confidence_weighted: float
    is_high_quality_wallet: int
    bet_size_relative_to_history: float
    side: str
    implied_prob_at_buy: float
    market_category: str
    market_volume_so_far_usd: float
    market_unique_traders_so_far: int
    market_age_seconds: int
    time_to_resolution_seconds: int | None
    last_trade_price: float | None
    price_volatility_recent: float | None
    label_won: int
    platform: str = "polymarket"


class TrainingExamplesRepo:
    """Inserts, truncate, and uniqueness lookups against ``training_examples``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def insert_or_ignore(self, examples: Iterable[TrainingExample]) -> int:
        """Insert examples; rows hitting the unique constraint silently skip."""
        rows = [_example_to_row(ex) for ex in examples]
        if not rows:
            return 0
        cur = self._conn.executemany(
            """
            INSERT OR IGNORE INTO training_examples (
              platform, tx_hash, asset_id, wallet_address, condition_id, trade_ts, built_at,
              prior_trades_count, prior_buys_count, prior_resolved_buys,
              prior_wins, prior_losses, win_rate, avg_implied_prob_paid,
              realized_edge_pp, prior_realized_pnl_usd,
              avg_bet_size_usd, median_bet_size_usd, wallet_age_days,
              seconds_since_last_trade, prior_trades_30d, top_category,
              category_diversity, bet_size_usd, bet_size_rel_to_avg,
              edge_confidence_weighted, win_rate_confidence_weighted,
              is_high_quality_wallet, bet_size_relative_to_history,
              side, implied_prob_at_buy, market_category, market_volume_so_far_usd,
              market_unique_traders_so_far, market_age_seconds,
              time_to_resolution_seconds, last_trade_price, price_volatility_recent,
              label_won
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return cur.rowcount or 0

    def truncate(self, *, platform: str = "polymarket") -> None:
        """Delete every row in ``training_examples`` for the given platform."""
        self._conn.execute("DELETE FROM training_examples WHERE platform = ?", (platform,))
        self._conn.commit()

    def existing_keys(self, *, platform: str = "polymarket") -> set[tuple[str, str, str]]:
        """Return (tx_hash, asset_id, wallet_address) for the given platform."""
        rows = self._conn.execute(
            """
            SELECT tx_hash, asset_id, wallet_address
            FROM training_examples WHERE platform = ?
            """,
            (platform,),
        ).fetchall()
        return {(row["tx_hash"], row["asset_id"], row["wallet_address"]) for row in rows}


def _example_to_row(ex: TrainingExample) -> tuple[object, ...]:
    return (
        ex.platform,
        ex.tx_hash,
        ex.asset_id,
        ex.wallet_address,
        ex.condition_id,
        ex.trade_ts,
        ex.built_at,
        ex.prior_trades_count,
        ex.prior_buys_count,
        ex.prior_resolved_buys,
        ex.prior_wins,
        ex.prior_losses,
        ex.win_rate,
        ex.avg_implied_prob_paid,
        ex.realized_edge_pp,
        ex.prior_realized_pnl_usd,
        ex.avg_bet_size_usd,
        ex.median_bet_size_usd,
        ex.wallet_age_days,
        ex.seconds_since_last_trade,
        ex.prior_trades_30d,
        ex.top_category,
        ex.category_diversity,
        ex.bet_size_usd,
        ex.bet_size_rel_to_avg,
        ex.edge_confidence_weighted,
        ex.win_rate_confidence_weighted,
        ex.is_high_quality_wallet,
        ex.bet_size_relative_to_history,
        ex.side,
        ex.implied_prob_at_buy,
        ex.market_category,
        ex.market_volume_so_far_usd,
        ex.market_unique_traders_so_far,
        ex.market_age_seconds,
        ex.time_to_resolution_seconds,
        ex.last_trade_price,
        ex.price_volatility_recent,
        ex.label_won,
    )


@dataclass(frozen=True)
class AssetEntry:
    """One row in `asset_index`: asset_id -> (condition_id, outcome_side, outcome_index).

    `outcome_index` is 0 for YES / first outcome, 1 for NO / second outcome
    on standard binary markets. We persist it explicitly for parity with
    Polymarket's `outcome_prices` array ordering.
    """

    asset_id: str
    condition_id: str
    outcome_side: str
    outcome_index: int
    platform: str = "polymarket"


class AssetIndexRepo:
    """Lookups and upserts against `asset_index`.

    Phase 2's on-chain ingest needs to map a decoded `OrderFilledEvent`'s
    `makerAssetId` / `takerAssetId` (uint256, the CTF position id) to the
    parent market's `condition_id` and the outcome side. We persist that
    mapping here so the lookup is local-only (no gamma round-trip per event).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repo to an already-initialised connection."""
        self._conn = conn

    def upsert(self, entry: AssetEntry) -> None:
        """Insert or replace the row for `(entry.platform, entry.asset_id)`."""
        self._conn.execute(
            """
            INSERT INTO asset_index (platform, asset_id, condition_id, outcome_side, outcome_index)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(platform, asset_id) DO UPDATE SET
              condition_id = excluded.condition_id,
              outcome_side = excluded.outcome_side,
              outcome_index = excluded.outcome_index
            """,
            (
                entry.platform,
                entry.asset_id,
                entry.condition_id,
                entry.outcome_side,
                entry.outcome_index,
            ),
        )
        self._conn.commit()

    def get(self, asset_id: str, *, platform: str = "polymarket") -> AssetEntry | None:
        """Look up an entry by `(platform, asset_id)`, or `None` if not present."""
        row = self._conn.execute(
            "SELECT asset_id, condition_id, outcome_side, outcome_index "
            "FROM asset_index WHERE platform = ? AND asset_id = ?",
            (platform, asset_id),
        ).fetchone()
        if row is None:
            return None
        return AssetEntry(
            asset_id=row["asset_id"],
            condition_id=row["condition_id"],
            outcome_side=row["outcome_side"],
            outcome_index=row["outcome_index"],
            platform=platform,
        )

    def backfill_from_corpus_trades(self, *, platform: str = "polymarket") -> int:
        """Populate `asset_index` from existing `corpus_trades` rows for ``platform``.

        Each `corpus_trades` row already carries (asset_id, condition_id,
        outcome_side). We derive `outcome_index` from the side: YES -> 0,
        NO -> 1 (matches Polymarket's `outcome_prices` array ordering).

        Returns:
            Number of distinct `asset_id`s inserted (excludes existing rows
            that conflicted on the PRIMARY KEY).
        """
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO asset_index (
              platform, asset_id, condition_id, outcome_side, outcome_index
            )
            SELECT
              ?,
              asset_id,
              condition_id,
              outcome_side,
              CASE outcome_side WHEN 'YES' THEN 0 ELSE 1 END
            FROM (
              SELECT asset_id, condition_id, outcome_side
              FROM corpus_trades
              WHERE platform = ?
              GROUP BY asset_id
            )
            """,
            (platform, platform),
        )
        inserted = cursor.rowcount
        self._conn.commit()
        return inserted
