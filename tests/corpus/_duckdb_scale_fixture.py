"""Synthetic ~500K-trade fixture for at-scale memory tests.

Deterministic via seeded RNG. Generates a corpus shaped to exercise
the OR-chain blowup in the old engine and the FILTER-window
replacement in the rewrite:
  - ~5,000 wallets with power-law trade-count distribution
    (most trade <10 times; ~50 wallets trade 1000+ times)
  - ~1,000 markets across all 9 categories
  - resolved-binary-only (so every BUY gets a label)
  - chronological timestamps spanning ~6 months
  - trade notionals are bounded below at $10 (matching production's
    CorpusTradesRepo.insert_batch floor)
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db

_CATEGORIES: tuple[str, ...] = (
    "sports",
    "esports",
    "thesis",
    "macro",
    "elections",
    "crypto",
    "geopolitics",
    "tech",
    "culture",
)


def build_scale_fixture_db(
    db_path: Path,
    *,
    seed: int = 20260513,
    n_wallets: int = 5_000,
    n_markets: int = 1_000,
    target_trades: int = 500_000,
) -> dict[str, int]:
    """Build a corpus DB at ``db_path`` with synthetic data.

    Returns a dict of {table: row_count} for the test to assert on.
    """
    rng = random.Random(seed)  # noqa: S311 — non-cryptographic test fixture
    init_corpus_db(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.executescript("PRAGMA synchronous=OFF; PRAGMA journal_mode=MEMORY;")
    try:
        markets = _gen_markets(rng, n_markets)
        _insert_markets(conn, markets)
        _insert_resolutions(conn, markets)
        wallets = [f"0x{rng.randrange(2**160):040x}" for _ in range(n_wallets)]
        n_trades = _insert_trades(conn, rng, markets, wallets, target_trades)
        conn.commit()
    finally:
        conn.close()
    return {"markets": len(markets), "wallets": n_wallets, "trades": n_trades}


def _gen_markets(rng: random.Random, n: int) -> list[dict[str, object]]:
    """Markets carry both market-table fields and resolution-table fields.

    outcome_yes_won and winning_outcome_index are present on every row.
    _insert_markets and _insert_resolutions each pick the columns they need.
    """
    out: list[dict[str, object]] = []
    base_ts = 1_700_000_000
    six_months_seconds = 6 * 30 * 86_400
    min_market_lifetime_s = 86_400
    max_market_lifetime_s = 30 * 86_400
    for i in range(n):
        cat = rng.choice(_CATEGORIES)
        opened = base_ts + rng.randrange(0, six_months_seconds)
        closed = opened + rng.randrange(min_market_lifetime_s, max_market_lifetime_s)
        yes_won = rng.randrange(2)
        out.append(
            {
                "platform": "polymarket",
                "condition_id": f"0x{i:064x}",
                "event_slug": f"event-{i}",
                "category": cat,
                "categories_json": f'["{cat}"]',
                "enumerated_at": opened,
                "closed_at": closed,
                "total_volume_usd": 10_000.0,
                "backfill_state": "complete",
                # Resolution-only fields (consumed by _insert_resolutions):
                "outcome_yes_won": yes_won,
                "winning_outcome_index": 0 if yes_won else 1,
            }
        )
    return out


def _insert_markets(conn: sqlite3.Connection, markets: list[dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT INTO corpus_markets
            (platform, condition_id, event_slug, category, categories_json,
             enumerated_at, closed_at, total_volume_usd, backfill_state)
        VALUES (:platform, :condition_id, :event_slug, :category,
                :categories_json, :enumerated_at, :closed_at,
                :total_volume_usd, :backfill_state)
        """,
        markets,
    )


def _insert_resolutions(conn: sqlite3.Connection, markets: list[dict[str, object]]) -> None:
    conn.executemany(
        """
        INSERT INTO market_resolutions
            (platform, condition_id, resolved_at, winning_outcome_index,
             outcome_yes_won, source, recorded_at)
        VALUES (:platform, :condition_id, :closed_at, :winning_outcome_index,
                :outcome_yes_won, 'synthetic', :closed_at)
        """,
        markets,
    )


def _insert_trades(
    conn: sqlite3.Connection,
    rng: random.Random,
    markets: list[dict[str, object]],
    wallets: list[str],
    target_trades: int,
) -> int:
    """Generate trades with a power-law wallet-activity distribution.

    Most wallets trade 1-10 times, a few whales trade 1000+. The sum
    of per-wallet activities is rescaled to approximately ``target_trades``.
    """
    activities: list[int] = []
    for _ in wallets:
        # Pareto-ish: floor(50 / U^0.6) clipped to [1, 5000]
        u = rng.random() or 1e-9
        a = max(1, min(5000, int(50.0 / (u**0.6))))
        activities.append(a)
    total = sum(activities)
    scale = target_trades / total
    activities = [max(1, int(a * scale)) for a in activities]

    rows: list[tuple[object, ...]] = []
    seq = 0
    buy_probability = 0.55
    price_span = 0.95
    price_floor = 0.025
    size_span = 100.0
    size_floor = 400.0
    for wallet, n_trades in zip(wallets, activities, strict=True):
        for _ in range(n_trades):
            m = rng.choice(markets)
            opened = int(m["enumerated_at"])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            closed = int(m["closed_at"])  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            ts = rng.randrange(opened, closed)
            outcome_side = rng.choice(("YES", "NO"))
            bs = "BUY" if rng.random() < buy_probability else "SELL"
            price = round(rng.random() * price_span + price_floor, 4)
            size = round(rng.random() * size_span + size_floor, 2)
            notional = round(size * price, 2)
            tx_hash = f"0x{seq:064x}"
            seq += 1
            rows.append(
                (
                    "polymarket",
                    tx_hash,
                    f"a_{m['condition_id']}_{outcome_side}",
                    wallet,
                    m["condition_id"],
                    outcome_side,
                    bs,
                    price,
                    size,
                    notional,
                    ts,
                )
            )
    conn.executemany(
        """
        INSERT INTO corpus_trades
            (platform, tx_hash, asset_id, wallet_address, condition_id,
             outcome_side, bs, price, size, notional_usd, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)
