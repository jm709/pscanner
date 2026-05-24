"""Column-list-driven SQLite serializers for ``wallet_state_live`` + ``market_state_live``.

The same ``WalletState`` / ``MarketState`` snapshots used by the streaming
provider (``pscanner.corpus.features``) are persisted by two distinct
write paths:

* :class:`pscanner.daemon.live_history.LiveHistoryProvider` (the live
  daemon) writes per-observe via UPSERT.
* :mod:`pscanner.daemon.bootstrap` (the cold-start CLI) writes the final
  in-memory state via bulk INSERT.

Both paths used to hand-write the column list + tuple unpacking, which
made adding a new ``WalletState`` field a 3-place lockstep edit and
risked silent column-drops. This module centralises:

* the column tuples (single source of truth for column order),
* the upsert / insert SQL (built from the column tuples),
* the dataclass ↔ tuple converters used by every writer + reader.

A parity test (``tests/daemon/test_state_persistence_parity.py``) asserts
that every dataclass field has a corresponding column entry so the
column tuples can't drift behind the dataclass.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from collections.abc import Iterable

from pscanner.corpus.features import (
    _RECENT_PRICES_MAX,
    MarketState,
    WalletState,
)

_WALLET_STATE_TABLE = "wallet_state_live"
_MARKET_STATE_TABLE = "market_state_live"

WALLET_STATE_COLUMNS: tuple[str, ...] = (
    "wallet_address",
    "first_seen_ts",
    "prior_trades_count",
    "prior_buys_count",
    "prior_resolved_buys",
    "prior_wins",
    "prior_losses",
    "cumulative_buy_price_sum",
    "cumulative_buy_count",
    "realized_pnl_usd",
    "last_trade_ts",
    "bet_size_sum",
    "bet_size_count",
    "recent_30d_trades_json",
    "category_counts_json",
    "unresolved_buys_json",
)
"""Order is load-bearing — matches every row tuple this module produces."""

MARKET_STATE_COLUMNS: tuple[str, ...] = (
    "condition_id",
    "market_age_start_ts",
    "volume_so_far_usd",
    "unique_traders_count",
    "last_trade_price",
    "recent_prices_json",
    "traders_json",
)
"""Order is load-bearing — matches every row tuple this module produces."""


def _build_upsert_sql(table: str, columns: tuple[str, ...], *, conflict_on: str) -> str:
    """Build the INSERT...ON CONFLICT(...) DO UPDATE SET... SQL for a column tuple.

    The table and column names are module-level constants (not caller input), so
    the f-string interpolation is safe from injection.
    """
    placeholders = ", ".join(["?"] * len(columns))
    sets = ",\n          ".join(f"{col} = excluded.{col}" for col in columns if col != conflict_on)
    cols_joined = ", ".join(columns)
    return (
        f"INSERT INTO {table} ({cols_joined})\n"
        f"VALUES ({placeholders})\n"
        f"ON CONFLICT({conflict_on}) DO UPDATE SET\n"
        f"          {sets}"
    )


def _build_insert_sql(table: str, columns: tuple[str, ...]) -> str:
    """Build the plain INSERT SQL used by the cold-start bulk writer.

    Table and column names are module-level constants — safe from injection.
    """
    placeholders = ", ".join(["?"] * len(columns))
    cols_joined = ", ".join(columns)
    # `table` / `columns` are module-level constants — see module docstring.
    return f"INSERT INTO {table} ({cols_joined}) VALUES ({placeholders})"  # noqa: S608


WALLET_STATE_UPSERT_SQL = _build_upsert_sql(
    _WALLET_STATE_TABLE, WALLET_STATE_COLUMNS, conflict_on="wallet_address"
)
WALLET_STATE_INSERT_SQL = _build_insert_sql(_WALLET_STATE_TABLE, WALLET_STATE_COLUMNS)
MARKET_STATE_UPSERT_SQL = _build_upsert_sql(
    _MARKET_STATE_TABLE, MARKET_STATE_COLUMNS, conflict_on="condition_id"
)
MARKET_STATE_INSERT_SQL = _build_insert_sql(_MARKET_STATE_TABLE, MARKET_STATE_COLUMNS)


def wallet_state_to_row(
    wallet_address: str,
    state: WalletState,
    *,
    unresolved_buys_json: str,
) -> tuple[object, ...]:
    """Build the ``wallet_state_live`` row tuple for ``state``.

    Order matches :data:`WALLET_STATE_COLUMNS`. ``unresolved_buys_json``
    is supplied by the caller because it lives outside ``WalletState``
    (per-wallet pending-buy queue is owned by the provider, not the
    state dataclass).
    """
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
        unresolved_buys_json,
    )


def wallet_state_from_row(row: sqlite3.Row) -> WalletState:
    """Rebuild a :class:`WalletState` from a ``wallet_state_live`` row.

    Caller is responsible for reading ``row["unresolved_buys_json"]``
    separately when it needs the queue — see
    :meth:`LiveHistoryProvider.wallet_state` for the drain path.
    """
    return WalletState(
        first_seen_ts=row["first_seen_ts"],
        prior_trades_count=row["prior_trades_count"],
        prior_buys_count=row["prior_buys_count"],
        prior_resolved_buys=row["prior_resolved_buys"],
        prior_wins=row["prior_wins"],
        prior_losses=row["prior_losses"],
        cumulative_buy_price_sum=row["cumulative_buy_price_sum"],
        cumulative_buy_count=row["cumulative_buy_count"],
        realized_pnl_usd=row["realized_pnl_usd"],
        last_trade_ts=row["last_trade_ts"],
        recent_30d_trades=deque(json.loads(row["recent_30d_trades_json"])),
        bet_size_sum=row["bet_size_sum"],
        bet_size_count=row["bet_size_count"],
        category_counts=dict(json.loads(row["category_counts_json"])),
    )


def market_state_to_row(
    condition_id: str,
    state: MarketState,
    *,
    traders: Iterable[str],
) -> tuple[object, ...]:
    """Build the ``market_state_live`` row tuple for ``state``.

    Order matches :data:`MARKET_STATE_COLUMNS`. ``traders`` is the
    market's distinct-wallet set; serialised as a sorted JSON array.
    """
    return (
        condition_id,
        state.market_age_start_ts,
        state.volume_so_far_usd,
        state.unique_traders_count,
        state.last_trade_price,
        json.dumps(list(state.recent_prices)),
        json.dumps(sorted(traders)),
    )


def market_state_from_row(row: sqlite3.Row) -> MarketState:
    """Rebuild a :class:`MarketState` from a ``market_state_live`` row.

    ``recent_prices`` is reconstructed with the same ``maxlen`` enforced
    by the streaming provider so the deque doesn't grow unbounded once
    the live daemon resumes folding trades.
    """
    return MarketState(
        market_age_start_ts=row["market_age_start_ts"],
        volume_so_far_usd=row["volume_so_far_usd"],
        unique_traders_count=row["unique_traders_count"],
        last_trade_price=row["last_trade_price"],
        recent_prices=deque(
            json.loads(row["recent_prices_json"]),
            maxlen=_RECENT_PRICES_MAX,
        ),
    )


def market_traders_from_row(row: sqlite3.Row) -> set[str]:
    """Read the trader-address set serialized at ``traders_json``."""
    return set(json.loads(row["traders_json"]))
