"""``pscanner daemon bootstrap-features`` — cold-start the live history tables.

Walks ``corpus_trades`` chronologically through an in-memory
:class:`StreamingHistoryProvider`, then bulk-writes the final wallet
and market state to ``wallet_state_live`` + ``market_state_live`` in a
single transaction. After this completes, the daemon can start with
O(1) state load.

The in-memory walk is the same one ``pscanner.corpus.examples.build_features``
uses, so bootstrap and training share the exact accumulator semantics.
A naive per-trade ``LiveHistoryProvider.observe`` would re-serialize
each wallet's growing JSON unresolved-buys list / category-counts on
every fold — at corpus scale (~22M trades, ~700K wallets) that is
``O(N^2)`` per heavy wallet and would take 15+ hours. The streaming
walk holds state in dataclasses + sets / heaps in memory and dumps
once at the end; the same workload finishes in ~10 minutes with peak
RSS in the low single GB.

Reads are scoped to a single ``platform`` so the gate model is only fed
rows from the platform it was trained on. Manifold trades are
mana-denominated (CLAUDE.md: "Never aggregate Manifold bet amounts into
real-money totals"); mixing them into a Polymarket-trained model would
poison ``cumulative_buy_price_sum`` / ``bet_size_sum`` / ``realized_pnl_usd``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import structlog

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
    _UnresolvedBuy,
    _WalletAccumulator,
)
from pscanner.store.db import init_db

_LOG = structlog.get_logger(__name__)


def run_bootstrap(
    *,
    corpus_db: Path,
    daemon_db: Path,
    log_every: int = 1_000_000,
    platform: str = "polymarket",
) -> int:
    """Cold-start ``wallet_state_live`` + ``market_state_live`` from corpus.

    Args:
        corpus_db: Path to the corpus SQLite (``data/corpus.sqlite3``).
        daemon_db: Path to the daemon SQLite (``data/pscanner.sqlite3``).
        log_every: Emit a progress log every N trades during the walk.
        platform: Scope all corpus reads to this platform. Defaults to
            ``"polymarket"`` because every shipped gate-model artifact
            today was trained on Polymarket data; mixing in non-Polymarket
            rows poisons the wallet/market accumulators.

    Returns:
        Total trade count folded.
    """
    corpus_conn = init_corpus_db(corpus_db)
    daemon_conn = init_db(daemon_db)
    try:
        metadata = _load_metadata(corpus_conn, platform=platform)
        provider = StreamingHistoryProvider(metadata=metadata)
        for cond_id, resolved_at, yes_won in corpus_conn.execute(
            "SELECT condition_id, resolved_at, outcome_yes_won "
            "FROM market_resolutions WHERE platform = ?",
            (platform,),
        ):
            provider.register_resolution(
                condition_id=cond_id,
                resolved_at=int(resolved_at),
                outcome_yes_won=int(yes_won),
            )
        n = _walk_corpus_trades(corpus_conn, provider, platform=platform, log_every=log_every)
        _LOG.info("daemon.bootstrap.dump_started", wallets=len(provider._wallets))
        _bulk_write_wallets(daemon_conn, provider)
        _LOG.info("daemon.bootstrap.dump_markets", markets=len(provider._markets))
        _bulk_write_markets(daemon_conn, provider)
        _LOG.info("daemon.bootstrap.done", trades_folded=n, platform=platform)
        return n
    finally:
        daemon_conn.close()
        corpus_conn.close()


def _walk_corpus_trades(
    conn: sqlite3.Connection,
    provider: StreamingHistoryProvider,
    *,
    platform: str,
    log_every: int,
) -> int:
    rows = conn.execute(
        """
        SELECT ct.tx_hash, ct.asset_id, ct.wallet_address, ct.condition_id,
               ct.outcome_side, ct.bs, ct.price, ct.size, ct.notional_usd, ct.ts
        FROM corpus_trades ct
        WHERE ct.platform = ?
        ORDER BY ct.ts ASC, ct.tx_hash ASC
        """,
        (platform,),
    )
    n = 0
    for row in rows:
        trade = Trade(
            tx_hash=row[0],
            asset_id=row[1],
            wallet_address=row[2],
            condition_id=row[3],
            outcome_side=row[4],
            bs=row[5],
            price=float(row[6]),
            size=float(row[7]),
            notional_usd=float(row[8]),
            ts=int(row[9]),
            category="",
        )
        if trade.bs == "BUY":
            provider.observe(trade)
        elif trade.bs == "SELL":
            provider.observe_sell(trade)
        n += 1
        if n % log_every == 0:
            _LOG.info("daemon.bootstrap.progress", trades_folded=n, platform=platform)
    return n


def _bulk_write_wallets(
    daemon_conn: sqlite3.Connection,
    provider: StreamingHistoryProvider,
) -> None:
    rows = [
        _wallet_row(wallet_address, accum) for wallet_address, accum in provider._wallets.items()
    ]
    daemon_conn.executemany(
        """
        INSERT INTO wallet_state_live (
          wallet_address, first_seen_ts, prior_trades_count, prior_buys_count,
          prior_resolved_buys, prior_wins, prior_losses,
          cumulative_buy_price_sum, cumulative_buy_count, realized_pnl_usd,
          last_trade_ts, bet_size_sum, bet_size_count,
          recent_30d_trades_json, category_counts_json, unresolved_buys_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    daemon_conn.commit()


def _wallet_row(wallet_address: str, accum: _WalletAccumulator) -> tuple[object, ...]:
    state = accum.state
    unresolved: list[_UnresolvedBuy] = list(accum.unscheduled)
    unresolved.extend(buy for _resolved_at, _seq, buy in accum.heap)
    unresolved_json = json.dumps(
        [
            {
                "condition_id": buy.condition_id,
                "notional_usd": buy.notional_usd,
                "size": buy.size,
                "side_yes": buy.side_yes,
                # ts is informational; the live drain reads resolved_at
                # from the in-memory _resolutions map, not from the buy.
                "ts": 0,
            }
            for buy in unresolved
        ]
    )
    return (
        wallet_address,
        state.first_seen_ts,
        state.prior_trades_count,
        state.prior_buys_count,
        state.prior_resolved_buys,
        state.prior_wins,
        state.prior_losses,
        state.cumulative_buy_price_sum,
        state.cumulative_buy_count,
        state.realized_pnl_usd,
        state.last_trade_ts,
        state.bet_size_sum,
        state.bet_size_count,
        json.dumps(list(state.recent_30d_trades)),
        json.dumps(state.category_counts),
        unresolved_json,
    )


def _bulk_write_markets(
    daemon_conn: sqlite3.Connection,
    provider: StreamingHistoryProvider,
) -> None:
    rows: list[tuple[object, ...]] = []
    for cond_id, market in provider._markets.items():
        traders = sorted(provider._market_traders.get(cond_id, set()))
        rows.append(
            (
                cond_id,
                market.market_age_start_ts,
                market.volume_so_far_usd,
                market.unique_traders_count,
                market.last_trade_price,
                json.dumps(list(market.recent_prices)),
                json.dumps(traders),
            )
        )
    daemon_conn.executemany(
        """
        INSERT INTO market_state_live (
          condition_id, market_age_start_ts, volume_so_far_usd,
          unique_traders_count, last_trade_price, recent_prices_json,
          traders_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    daemon_conn.commit()


def _load_metadata(
    conn: sqlite3.Connection, *, platform: str = "polymarket"
) -> dict[str, MarketMetadata]:
    out: dict[str, MarketMetadata] = {}
    for cond_id, category, closed_at, opened_at in conn.execute(
        """
        SELECT condition_id,
               COALESCE(category, ''),
               COALESCE(closed_at, 0),
               COALESCE(enumerated_at, 0)
        FROM corpus_markets
        WHERE platform = ?
        """,
        (platform,),
    ):
        out[cond_id] = MarketMetadata(
            condition_id=cond_id,
            category=category,
            closed_at=int(closed_at),
            opened_at=int(opened_at),
        )
    return out
