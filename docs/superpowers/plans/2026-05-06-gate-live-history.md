# Live history provider (Issue #78) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a `LiveHistoryProvider` that mirrors `StreamingHistoryProvider`'s feature-state semantics but persists to SQLite, plus a `pscanner daemon bootstrap-features` CLI that pre-warms the tables from `corpus_trades`. This is the foundation for #79 (gate-model detector) and #80 (evaluator).

**Architecture:** Two new SQLite tables (`wallet_state_live`, `market_state_live`) store the per-wallet and per-market accumulators. `LiveHistoryProvider` (`src/pscanner/daemon/live_history.py`) implements the `HistoryProvider` Protocol from `pscanner.corpus.features` so `compute_features` can consume it unchanged. `observe()` becomes SELECT-then-UPDATE on the indexed primary keys. A new CLI command walks `corpus_trades` chronologically and folds into the live provider — same pure functions as `build-features`, persisted result. `bootstrap_wallet(addr)` fills cold wallets seen for the first time post-bootstrap by calling `DataClient.get_positions(user=..., closed=true)`.

**Tech Stack:** Python 3.13, SQLite, pytest. Quick verify: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`.

**Spec:** GitHub issue #78 (under RFC #77).

---

## File map

- **Create** `src/pscanner/daemon/__init__.py` — package marker for the daemon-side code introduced by #77/#78/#79.
- **Create** `src/pscanner/daemon/live_history.py` — `LiveHistoryProvider` class plus the `_pack/_unpack` helpers for the JSON-encoded heap/traders fields. ~250 lines.
- **Modify** `src/pscanner/store/db.py` — append `wallet_state_live` + `market_state_live` `CREATE TABLE IF NOT EXISTS` to `_SCHEMA_STATEMENTS`. No migrations needed (these are new tables).
- **Modify** `src/pscanner/cli.py` — add `pscanner daemon bootstrap-features` subcommand wired to `pscanner.daemon.bootstrap.run_bootstrap`.
- **Create** `src/pscanner/daemon/bootstrap.py` — `run_bootstrap(corpus_db, daemon_db, *, log_every=100_000)` walks `corpus_trades` chronologically, calls `provider.observe`/`observe_sell`/`register_resolution`, prints a progress line every N trades.
- **Create** `tests/daemon/__init__.py` — package marker.
- **Create** `tests/daemon/test_live_history.py` — round-trip + restart parity unit tests.
- **Create** `tests/daemon/test_live_history_parity.py` — parity test against `StreamingHistoryProvider` over a 100-trade synthetic fixture.
- **Create** `tests/daemon/test_bootstrap.py` — bootstrap CLI test against a small corpus fixture.
- **Modify** `tests/conftest.py` — add a `tmp_corpus_db` fixture that returns a `sqlite3.Connection` from `init_corpus_db(":memory:")`. (Already exists for daemon db; this mirrors it for the corpus side.)
- **Modify** `CLAUDE.md` — add a paragraph under "Codebase conventions" describing the new tables and the bootstrap requirement.

Out of scope for this plan: the gate-model detector itself (#79), the market-scoped trade collector (#79), the paper-trade evaluator (#80). Those land in their own plans once this one is merged.

---

### Task 1: Add the schema for `wallet_state_live` and `market_state_live`

**Files:**
- Modify: `src/pscanner/store/db.py:15` (top of `_SCHEMA_STATEMENTS`)
- Test: `tests/store/test_db.py` (existing file, add a new test)

The tables mirror the in-memory accumulators in `StreamingHistoryProvider` (see `pscanner.corpus.features:443-587`). Column rationale:

- `wallet_state_live` columns map 1:1 to `WalletState` (`features.py:64-92`) plus three accumulator-only fields that live outside `WalletState` in the streaming provider (`recent_30d_trades`, `category_counts`, plus the heap+unscheduled buys held in `_WalletAccumulator`). The heap/unscheduled state collapses into one `unresolved_buys_json` column for v1.
- `market_state_live` mirrors `MarketState` (`features.py:94-111`) plus the per-market traders set held outside the dataclass (`StreamingHistoryProvider._market_traders`).

- [ ] **Step 1: Write the failing schema test**

Add to `tests/store/test_db.py` (after the existing tests):

```python
def test_init_db_creates_wallet_state_live_table() -> None:
    conn = init_db(Path(":memory:"))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(wallet_state_live)")}
    finally:
        conn.close()
    assert {
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
    }.issubset(cols)


def test_init_db_creates_market_state_live_table() -> None:
    conn = init_db(Path(":memory:"))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(market_state_live)")}
    finally:
        conn.close()
    assert {
        "condition_id",
        "market_age_start_ts",
        "volume_so_far_usd",
        "unique_traders_count",
        "last_trade_price",
        "recent_prices_json",
        "traders_json",
    }.issubset(cols)
```

- [ ] **Step 2: Run the tests, expect failure**

Run: `uv run pytest tests/store/test_db.py::test_init_db_creates_wallet_state_live_table tests/store/test_db.py::test_init_db_creates_market_state_live_table -v`
Expected: FAIL — `PRAGMA table_info(...)` returns an empty set because the tables don't exist.

- [ ] **Step 3: Append the two new statements to `_SCHEMA_STATEMENTS`**

Open `src/pscanner/store/db.py`. Find the closing `)` of the `_SCHEMA_STATEMENTS` tuple (search for `_SCHEMA_STATEMENTS: tuple[str, ...] = (` and locate its terminator). Insert these two strings just before the trailing `)`:

```python
    """
    CREATE TABLE IF NOT EXISTS wallet_state_live (
      wallet_address TEXT PRIMARY KEY,
      first_seen_ts INTEGER NOT NULL,
      prior_trades_count INTEGER NOT NULL DEFAULT 0,
      prior_buys_count INTEGER NOT NULL DEFAULT 0,
      prior_resolved_buys INTEGER NOT NULL DEFAULT 0,
      prior_wins INTEGER NOT NULL DEFAULT 0,
      prior_losses INTEGER NOT NULL DEFAULT 0,
      cumulative_buy_price_sum REAL NOT NULL DEFAULT 0,
      cumulative_buy_count INTEGER NOT NULL DEFAULT 0,
      realized_pnl_usd REAL NOT NULL DEFAULT 0,
      last_trade_ts INTEGER,
      bet_size_sum REAL NOT NULL DEFAULT 0,
      bet_size_count INTEGER NOT NULL DEFAULT 0,
      recent_30d_trades_json TEXT NOT NULL DEFAULT '[]',
      category_counts_json TEXT NOT NULL DEFAULT '{}',
      unresolved_buys_json TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_state_live (
      condition_id TEXT PRIMARY KEY,
      market_age_start_ts INTEGER NOT NULL,
      volume_so_far_usd REAL NOT NULL DEFAULT 0,
      unique_traders_count INTEGER NOT NULL DEFAULT 0,
      last_trade_price REAL,
      recent_prices_json TEXT NOT NULL DEFAULT '[]',
      traders_json TEXT NOT NULL DEFAULT '[]'
    )
    """,
```

- [ ] **Step 4: Re-run the tests, expect pass**

Run: `uv run pytest tests/store/test_db.py::test_init_db_creates_wallet_state_live_table tests/store/test_db.py::test_init_db_creates_market_state_live_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/store/db.py tests/store/test_db.py
git commit -m "feat(store): add wallet_state_live + market_state_live schema for #78"
```

---

### Task 2: Skeleton `LiveHistoryProvider` with the `HistoryProvider` Protocol shape

**Files:**
- Create: `src/pscanner/daemon/__init__.py`
- Create: `src/pscanner/daemon/live_history.py`
- Create: `tests/daemon/__init__.py`
- Create: `tests/daemon/test_live_history.py`

The class will satisfy `pscanner.corpus.features.HistoryProvider` (the Protocol at `features.py:288-307`) so the same `compute_features` function works for both providers. We'll start with `wallet_state` returning the empty default for an unseen wallet, then layer in `observe()` in the next task.

- [ ] **Step 1: Write the failing test for `wallet_state` on an unseen wallet**

Create `tests/daemon/__init__.py` as an empty file. Then create `tests/daemon/test_live_history.py`:

```python
"""Unit tests for LiveHistoryProvider (#78)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.features import MarketMetadata
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


def _new_conn() -> sqlite3.Connection:
    return init_db(Path(":memory:"))


def test_wallet_state_returns_empty_for_unknown_wallet() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        state = provider.wallet_state("0xabc", as_of_ts=1_700_000_000)
    finally:
        conn.close()
    assert state.first_seen_ts == 1_700_000_000
    assert state.prior_trades_count == 0
    assert state.prior_buys_count == 0
    assert state.prior_wins == 0
    assert state.recent_30d_trades == ()
    assert state.category_counts == {}
```

- [ ] **Step 2: Run the test, expect failure**

Run: `uv run pytest tests/daemon/test_live_history.py::test_wallet_state_returns_empty_for_unknown_wallet -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pscanner.daemon'`.

- [ ] **Step 3: Create the skeleton class**

Create `src/pscanner/daemon/__init__.py` as an empty file. Create `src/pscanner/daemon/live_history.py`:

```python
"""SQLite-backed history provider for the live daemon (#78).

Mirrors ``StreamingHistoryProvider`` from ``pscanner.corpus.features`` but
persists state to ``wallet_state_live`` + ``market_state_live`` so daemon
restarts are O(1) instead of O(corpus). Implements the ``HistoryProvider``
Protocol so ``compute_features`` can consume it unchanged.

The accumulator semantics are point-for-point identical to the streaming
provider — the only difference is storage. The same pure
``apply_*_to_state`` functions in ``pscanner.corpus.features`` drive both
providers, which is the parity contract validated by
``tests/daemon/test_live_history_parity.py``.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from pscanner.corpus.features import (
    MarketMetadata,
    MarketState,
    WalletState,
    empty_market_state,
    empty_wallet_state,
)

if TYPE_CHECKING:
    from pscanner.corpus.features import Trade, _TradeFields


class LiveHistoryProvider:
    """Persistent ``HistoryProvider`` backed by SQLite.

    The provider holds an open ``sqlite3.Connection`` for the daemon DB
    (the same one returned by ``init_db``). All reads/writes happen on
    that connection — caller owns the connection lifecycle.

    Args:
        conn: Open daemon-DB connection (schema applied). Must remain
            open for the lifetime of the provider.
        metadata: Map of ``condition_id -> MarketMetadata`` used by
            ``market_metadata`` and the time-to-resolution feature. The
            daemon refreshes this dict from ``market_resolutions`` on a
            cadence (out of scope for this class).
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        metadata: dict[str, MarketMetadata],
    ) -> None:
        self._conn = conn
        self._metadata = metadata

    def market_metadata(self, condition_id: str) -> MarketMetadata:
        """Return static metadata for ``condition_id``; raises KeyError if unknown."""
        return self._metadata[condition_id]

    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        """Return the wallet's state at ``as_of_ts``.

        ``as_of_ts`` is used only as the seed ``first_seen_ts`` for an
        unknown wallet (matching ``StreamingHistoryProvider`` semantics).
        Resolution drain is implemented in Task 4.
        """
        row = self._conn.execute(
            "SELECT * FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if row is None:
            return empty_wallet_state(first_seen_ts=as_of_ts)
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
            recent_30d_trades=tuple(json.loads(row["recent_30d_trades_json"])),
            bet_size_sum=row["bet_size_sum"],
            bet_size_count=row["bet_size_count"],
            category_counts=dict(json.loads(row["category_counts_json"])),
        )

    def market_state(self, condition_id: str, as_of_ts: int) -> MarketState:
        """Return per-market running state.

        ``as_of_ts`` is unused — caller must query before observing the
        next event for the same market (parity with streaming provider).
        """
        del as_of_ts
        row = self._conn.execute(
            "SELECT * FROM market_state_live WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if row is None:
            return empty_market_state(market_age_start_ts=0)
        return MarketState(
            market_age_start_ts=row["market_age_start_ts"],
            volume_so_far_usd=row["volume_so_far_usd"],
            unique_traders_count=row["unique_traders_count"],
            last_trade_price=row["last_trade_price"],
            recent_prices=tuple(json.loads(row["recent_prices_json"])),
        )

    def observe(self, trade: Trade) -> None:
        """Fold a trade into running wallet + market state. Implemented in Task 3."""
        raise NotImplementedError

    def observe_sell(self, trade: _TradeFields) -> None:
        """Fold a SELL fill into wallet + market state. Implemented in Task 3."""
        raise NotImplementedError

    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None:
        """Record a market's resolution. Implemented in Task 4."""
        raise NotImplementedError
```

- [ ] **Step 4: Re-run the test, expect pass**

Run: `uv run pytest tests/daemon/test_live_history.py::test_wallet_state_returns_empty_for_unknown_wallet -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/daemon/ tests/daemon/__init__.py tests/daemon/test_live_history.py
git commit -m "feat(daemon): skeleton LiveHistoryProvider with read-side decode for #78"
```

---

### Task 3: Implement `observe()` and `observe_sell()` for the BUY/SELL fold

**Files:**
- Modify: `src/pscanner/daemon/live_history.py`
- Test: `tests/daemon/test_live_history.py`

The pure fold logic already lives in `pscanner.corpus.features` as `apply_buy_to_state` / `apply_sell_to_state` / `apply_trade_to_market`. The new code is just the SELECT-then-UPDATE persistence wrapper around those calls, plus the JSON encoding for `recent_30d_trades`, `category_counts`, `recent_prices`, `traders`, and the unresolved-buys list.

- [ ] **Step 1: Write the failing test for a single BUY observation**

Add to `tests/daemon/test_live_history.py`:

```python
import pytest

from pscanner.corpus.features import Trade


def _make_trade(
    *,
    bs: str = "BUY",
    wallet: str = "0xabc",
    condition_id: str = "0xcond",
    side: str = "YES",
    price: float = 0.42,
    size: float = 100.0,
    notional_usd: float = 42.0,
    ts: int = 1_700_000_000,
    category: str = "esports",
) -> Trade:
    return Trade(
        tx_hash=f"tx-{ts}-{bs}",
        asset_id="0xasset",
        wallet_address=wallet,
        condition_id=condition_id,
        outcome_side=side,
        bs=bs,
        price=price,
        size=size,
        notional_usd=notional_usd,
        ts=ts,
        category=category,
    )


def test_observe_buy_persists_wallet_and_market_state() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        trade = _make_trade(bs="BUY", price=0.42, notional_usd=42.0)
        provider.observe(trade)
        wallet = provider.wallet_state("0xabc", as_of_ts=trade.ts + 1)
        market = provider.market_state("0xcond", as_of_ts=trade.ts + 1)
    finally:
        conn.close()
    assert wallet.prior_trades_count == 1
    assert wallet.prior_buys_count == 1
    assert wallet.cumulative_buy_count == 1
    assert wallet.cumulative_buy_price_sum == pytest.approx(0.42)
    assert wallet.bet_size_count == 1
    assert wallet.bet_size_sum == pytest.approx(42.0)
    assert wallet.category_counts == {"esports": 1}
    assert market.unique_traders_count == 1
    assert market.last_trade_price == pytest.approx(0.42)
    assert market.volume_so_far_usd == pytest.approx(42.0)


def test_observe_sell_updates_wallet_and_market_state() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        provider.observe(_make_trade(bs="BUY", ts=1_700_000_000))
        provider.observe(_make_trade(bs="SELL", ts=1_700_000_100, price=0.55))
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_000_200)
        market = provider.market_state("0xcond", as_of_ts=1_700_000_200)
    finally:
        conn.close()
    assert wallet.prior_trades_count == 2
    assert wallet.prior_buys_count == 1  # SELL doesn't increment buys
    assert market.last_trade_price == pytest.approx(0.55)
```

- [ ] **Step 2: Run the tests, expect failure**

Run: `uv run pytest tests/daemon/test_live_history.py -v`
Expected: FAIL — `NotImplementedError` from the `observe` placeholder.

- [ ] **Step 3: Implement `observe`/`observe_sell`/`_persist_wallet`/`_persist_market`**

Replace the `observe`, `observe_sell`, and `register_resolution` placeholders in `src/pscanner/daemon/live_history.py` with the full implementation. Add these imports at the top:

```python
import heapq

from pscanner.corpus.features import (
    Trade,
    _TradeFields,  # noqa: PLC2701  -- re-using the structural Protocol from features
    apply_buy_to_state,
    apply_resolution_to_state,
    apply_sell_to_state,
    apply_trade_to_market,
)
```

(Move `Trade` and `_TradeFields` out of the `TYPE_CHECKING` guard now that they're used at runtime.) Then add these methods to the class — replacing the `observe`/`observe_sell` placeholders:

```python
    def observe(self, trade: Trade) -> None:
        """Fold a trade into running wallet + market state.

        BUY rows update the wallet's running aggregates AND append an
        ``_UnresolvedBuy`` to the wallet's serialized heap so it can be
        drained when ``register_resolution`` fires.
        """
        wallet = self.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
        unresolved = self._load_unresolved(trade.wallet_address)
        if trade.bs == "BUY":
            new_state = apply_buy_to_state(wallet, trade)
            unresolved.append(
                {
                    "condition_id": trade.condition_id,
                    "notional_usd": trade.notional_usd,
                    "size": trade.size,
                    "side_yes": trade.outcome_side == "YES",
                    "ts": trade.ts,
                }
            )
        elif trade.bs == "SELL":
            new_state = apply_sell_to_state(wallet, trade)
        else:
            return
        self._persist_wallet(trade.wallet_address, new_state, unresolved)
        self._observe_market(trade)

    def observe_sell(self, trade: _TradeFields) -> None:
        """Fold a SELL fill (no ``category`` required) into wallet + market state."""
        wallet = self.wallet_state(trade.wallet_address, as_of_ts=trade.ts)
        new_state = apply_sell_to_state(wallet, trade)
        unresolved = self._load_unresolved(trade.wallet_address)
        self._persist_wallet(trade.wallet_address, new_state, unresolved)
        self._observe_market(trade)

    def _observe_market(self, trade: _TradeFields) -> None:
        market_row = self._conn.execute(
            "SELECT * FROM market_state_live WHERE condition_id = ?",
            (trade.condition_id,),
        ).fetchone()
        if market_row is None:
            current = empty_market_state(market_age_start_ts=trade.ts)
            traders: set[str] = set()
        else:
            current = MarketState(
                market_age_start_ts=market_row["market_age_start_ts"],
                volume_so_far_usd=market_row["volume_so_far_usd"],
                unique_traders_count=market_row["unique_traders_count"],
                last_trade_price=market_row["last_trade_price"],
                recent_prices=tuple(json.loads(market_row["recent_prices_json"])),
            )
            traders = set(json.loads(market_row["traders_json"]))
        is_new_trader = trade.wallet_address not in traders
        if is_new_trader:
            traders.add(trade.wallet_address)
        new_state = apply_trade_to_market(current, trade, is_new_trader=is_new_trader)
        self._persist_market(trade.condition_id, new_state, traders)

    def _load_unresolved(self, wallet_address: str) -> list[dict[str, object]]:
        row = self._conn.execute(
            "SELECT unresolved_buys_json FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if row is None:
            return []
        return list(json.loads(row["unresolved_buys_json"]))

    def _persist_wallet(
        self,
        wallet_address: str,
        state: WalletState,
        unresolved: list[dict[str, object]],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO wallet_state_live (
              wallet_address, first_seen_ts, prior_trades_count, prior_buys_count,
              prior_resolved_buys, prior_wins, prior_losses,
              cumulative_buy_price_sum, cumulative_buy_count, realized_pnl_usd,
              last_trade_ts, bet_size_sum, bet_size_count,
              recent_30d_trades_json, category_counts_json, unresolved_buys_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wallet_address) DO UPDATE SET
              first_seen_ts = excluded.first_seen_ts,
              prior_trades_count = excluded.prior_trades_count,
              prior_buys_count = excluded.prior_buys_count,
              prior_resolved_buys = excluded.prior_resolved_buys,
              prior_wins = excluded.prior_wins,
              prior_losses = excluded.prior_losses,
              cumulative_buy_price_sum = excluded.cumulative_buy_price_sum,
              cumulative_buy_count = excluded.cumulative_buy_count,
              realized_pnl_usd = excluded.realized_pnl_usd,
              last_trade_ts = excluded.last_trade_ts,
              bet_size_sum = excluded.bet_size_sum,
              bet_size_count = excluded.bet_size_count,
              recent_30d_trades_json = excluded.recent_30d_trades_json,
              category_counts_json = excluded.category_counts_json,
              unresolved_buys_json = excluded.unresolved_buys_json
            """,
            (
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
                json.dumps(unresolved),
            ),
        )
        self._conn.commit()

    def _persist_market(
        self, condition_id: str, state: MarketState, traders: set[str]
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO market_state_live (
              condition_id, market_age_start_ts, volume_so_far_usd,
              unique_traders_count, last_trade_price, recent_prices_json,
              traders_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(condition_id) DO UPDATE SET
              market_age_start_ts = excluded.market_age_start_ts,
              volume_so_far_usd = excluded.volume_so_far_usd,
              unique_traders_count = excluded.unique_traders_count,
              last_trade_price = excluded.last_trade_price,
              recent_prices_json = excluded.recent_prices_json,
              traders_json = excluded.traders_json
            """,
            (
                condition_id,
                state.market_age_start_ts,
                state.volume_so_far_usd,
                state.unique_traders_count,
                state.last_trade_price,
                json.dumps(list(state.recent_prices)),
                json.dumps(sorted(traders)),
            ),
        )
        self._conn.commit()
```

- [ ] **Step 4: Re-run the tests, expect pass**

Run: `uv run pytest tests/daemon/test_live_history.py -v`
Expected: PASS — both `test_observe_buy_*` and `test_observe_sell_*`.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/daemon/live_history.py tests/daemon/test_live_history.py
git commit -m "feat(daemon): persist wallet/market state on observe() for #78"
```

---

### Task 4: Implement `register_resolution` + the resolution drain in `wallet_state`

**Files:**
- Modify: `src/pscanner/daemon/live_history.py`
- Test: `tests/daemon/test_live_history.py`

`StreamingHistoryProvider.wallet_state` (`features.py:555-574`) drains the per-wallet heap whenever the queried `as_of_ts` crosses a resolution time. The live provider does the same — but the heap is rehydrated from `unresolved_buys_json` and re-serialized after the drain. We also need a `condition_id -> resolved_at` map so the drain knows which buys to apply; that map persists in a tiny in-memory dict on the provider for v1 (resolutions land via the daemon's market-resolution polling loop, which is out of scope for this plan).

- [ ] **Step 1: Write the failing test for resolve-then-query draining**

Add to `tests/daemon/test_live_history.py`:

```python
def test_register_resolution_drains_buy_to_win() -> None:
    conn = _new_conn()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        provider.observe(
            _make_trade(
                bs="BUY",
                side="YES",
                price=0.40,
                size=100.0,
                notional_usd=40.0,
                ts=1_700_000_000,
            )
        )
        provider.register_resolution(
            condition_id="0xcond",
            resolved_at=1_700_001_000,
            outcome_yes_won=1,
        )
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_001_500)
    finally:
        conn.close()
    assert wallet.prior_resolved_buys == 1
    assert wallet.prior_wins == 1
    assert wallet.prior_losses == 0
    assert wallet.realized_pnl_usd == pytest.approx(60.0)  # 100 - 40
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/daemon/test_live_history.py::test_register_resolution_drains_buy_to_win -v`
Expected: FAIL — `register_resolution` raises `NotImplementedError`.

- [ ] **Step 3: Implement `register_resolution` + drain in `wallet_state`**

Add a resolution map to `__init__`:

```python
        self._resolutions: dict[str, tuple[int, int]] = {}
```

Add `_drain_resolved_buys` helper and replace `register_resolution` placeholder in `live_history.py`:

```python
    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None:
        """Record a market's resolution.

        Subsequent ``wallet_state`` queries past ``resolved_at`` drain
        any unresolved buys against this market for the queried wallet.
        """
        self._resolutions[condition_id] = (resolved_at, outcome_yes_won)

    def get_resolution(self, condition_id: str) -> tuple[int, int] | None:
        """Return ``(resolved_at, outcome_yes_won)`` if known, else ``None``."""
        return self._resolutions.get(condition_id)
```

Now make `wallet_state` apply the drain. Replace the existing `wallet_state` body so that after fetching the row, it walks through `unresolved` buys and applies any whose resolution_ts < as_of_ts:

```python
    def wallet_state(self, wallet_address: str, as_of_ts: int) -> WalletState:
        """Return the wallet's state at ``as_of_ts``, draining ready resolutions."""
        row = self._conn.execute(
            "SELECT * FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if row is None:
            return empty_wallet_state(first_seen_ts=as_of_ts)
        state = WalletState(
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
            recent_30d_trades=tuple(json.loads(row["recent_30d_trades_json"])),
            bet_size_sum=row["bet_size_sum"],
            bet_size_count=row["bet_size_count"],
            category_counts=dict(json.loads(row["category_counts_json"])),
        )
        unresolved = list(json.loads(row["unresolved_buys_json"]))
        state, remaining = self._drain_resolved_buys(state, unresolved, as_of_ts)
        if remaining is not unresolved:
            self._persist_wallet(wallet_address, state, remaining)
        return state

    def _drain_resolved_buys(
        self,
        state: WalletState,
        unresolved: list[dict[str, object]],
        as_of_ts: int,
    ) -> tuple[WalletState, list[dict[str, object]]]:
        # Build a heap of (resolved_at, idx, buy) for buys whose market
        # has a known resolution. Drain anything with resolved_at < as_of_ts.
        ready: list[tuple[int, int, dict[str, object]]] = []
        deferred: list[dict[str, object]] = []
        for idx, buy in enumerate(unresolved):
            cond_id = buy["condition_id"]
            assert isinstance(cond_id, str)
            resolution = self._resolutions.get(cond_id)
            if resolution is None:
                deferred.append(buy)
                continue
            resolved_at, _ = resolution
            heapq.heappush(ready, (resolved_at, idx, buy))
        if not ready:
            return state, unresolved
        leftover: list[dict[str, object]] = list(deferred)
        while ready:
            resolved_at, _, buy = heapq.heappop(ready)
            if resolved_at >= as_of_ts:
                leftover.append(buy)
                continue
            cond_id = buy["condition_id"]
            assert isinstance(cond_id, str)
            _, yes_won = self._resolutions[cond_id]
            side_yes = bool(buy["side_yes"])
            won = (yes_won == 1) if side_yes else (yes_won == 0)
            size = float(buy["size"])  # type: ignore[arg-type]
            notional = float(buy["notional_usd"])  # type: ignore[arg-type]
            payout = size if won else 0.0
            state = apply_resolution_to_state(
                state,
                won=won,
                notional_usd=notional,
                payout_usd=payout,
            )
        return state, leftover
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/daemon/test_live_history.py::test_register_resolution_drains_buy_to_win -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pscanner/daemon/live_history.py tests/daemon/test_live_history.py
git commit -m "feat(daemon): resolution drain in LiveHistoryProvider for #78"
```

---

### Task 5: Restart-parity test — close & re-open the connection mid-stream

**Files:**
- Modify: `tests/daemon/test_live_history.py`

The whole point of #78 is that daemon restarts don't lose state. Add a test that observes some trades, closes the connection, opens a new one, and asserts the post-restart `wallet_state` matches the pre-restart one byte-for-byte.

- [ ] **Step 1: Write the failing test (will pass first try if Task 3 was correct, but worth pinning)**

Add to `tests/daemon/test_live_history.py`. We back the test by an on-disk file rather than `:memory:` (the latter loses state on close):

```python
def test_restart_preserves_wallet_state(tmp_path: Path) -> None:
    db_path = tmp_path / "daemon.sqlite3"
    conn1 = init_db(db_path)
    try:
        provider1 = LiveHistoryProvider(conn=conn1, metadata={})
        provider1.observe(_make_trade(bs="BUY", ts=1_700_000_000, price=0.40))
        provider1.observe(
            _make_trade(bs="BUY", ts=1_700_000_100, price=0.45, condition_id="0xcond2")
        )
        before = provider1.wallet_state("0xabc", as_of_ts=1_700_000_200)
    finally:
        conn1.close()
    conn2 = init_db(db_path)
    try:
        provider2 = LiveHistoryProvider(conn=conn2, metadata={})
        after = provider2.wallet_state("0xabc", as_of_ts=1_700_000_200)
    finally:
        conn2.close()
    assert before == after
```

- [ ] **Step 2: Run, expect pass**

Run: `uv run pytest tests/daemon/test_live_history.py::test_restart_preserves_wallet_state -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/daemon/test_live_history.py
git commit -m "test(daemon): pin restart-parity behavior for LiveHistoryProvider"
```

---

### Task 6: Parity test against `StreamingHistoryProvider` over a synthetic 100-trade fixture

**Files:**
- Create: `tests/daemon/test_live_history_parity.py`

This is the load-bearing parity contract: for any sequence of trades + resolutions, `compute_features(trade, live_provider)` must equal `compute_features(trade, streaming_provider)` exactly.

- [ ] **Step 1: Write the parity test**

Create `tests/daemon/test_live_history_parity.py`:

```python
"""Parity test: LiveHistoryProvider vs StreamingHistoryProvider (#78)."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus.features import (
    MarketMetadata,
    StreamingHistoryProvider,
    Trade,
    compute_features,
)
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db


def _build_synthetic_trades(seed: int, n: int) -> list[Trade]:
    rng = random.Random(seed)
    wallets = [f"0xw{i:02d}" for i in range(8)]
    markets = [f"0xm{i:02d}" for i in range(5)]
    trades: list[Trade] = []
    base_ts = 1_700_000_000
    for i in range(n):
        wallet = rng.choice(wallets)
        market = rng.choice(markets)
        side = rng.choice(("YES", "NO"))
        bs = rng.choices(("BUY", "SELL"), weights=(0.7, 0.3))[0]
        price = round(rng.uniform(0.05, 0.95), 4)
        size = round(rng.uniform(50.0, 500.0), 2)
        trades.append(
            Trade(
                tx_hash=f"tx{i:04d}",
                asset_id=f"{market}-{side}",
                wallet_address=wallet,
                condition_id=market,
                outcome_side=side,
                bs=bs,
                price=price,
                size=size,
                notional_usd=round(price * size, 4),
                ts=base_ts + i * 60,
                category=rng.choice(("sports", "esports", "crypto")),
            )
        )
    return trades


def _build_metadata(trades: list[Trade]) -> dict[str, MarketMetadata]:
    by_market: dict[str, MarketMetadata] = {}
    for t in trades:
        if t.condition_id in by_market:
            continue
        by_market[t.condition_id] = MarketMetadata(
            condition_id=t.condition_id,
            category=t.category,
            closed_at=t.ts + 86_400 * 7,
            opened_at=t.ts - 60,
        )
    return by_market


@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_compute_features_matches_streaming_provider(seed: int) -> None:
    trades = _build_synthetic_trades(seed=seed, n=100)
    metadata = _build_metadata(trades)
    streaming = StreamingHistoryProvider(metadata=metadata)
    conn: sqlite3.Connection = init_db(Path(":memory:"))
    try:
        live = LiveHistoryProvider(conn=conn, metadata=metadata)
        for trade in trades:
            streaming_row = compute_features(trade, streaming)
            live_row = compute_features(trade, live)
            assert streaming_row == live_row, (
                f"feature divergence at {trade.tx_hash}: "
                f"streaming={streaming_row} live={live_row}"
            )
            streaming.observe(trade)
            live.observe(trade)
    finally:
        conn.close()
```

- [ ] **Step 2: Run, expect pass on all four seeds**

Run: `uv run pytest tests/daemon/test_live_history_parity.py -v`
Expected: PASS for all 4 seeds.

If any seed diverges, the failure message points at the trade where streaming and live disagree. Most likely root causes: (a) JSON-encoded list ordering for `recent_30d_trades` differs (it shouldn't — append-only), (b) `recent_prices` truncation off-by-one, (c) `category_counts` key ordering inside the JSON survives but the `==` comparison ignores ordering. If divergence is real, fix the live provider — never the test.

- [ ] **Step 3: Commit**

```bash
git add tests/daemon/test_live_history_parity.py
git commit -m "test(daemon): parity vs StreamingHistoryProvider on 4 seeds"
```

---

### Task 7: `bootstrap_wallet` — pre-warm an unknown wallet from `/positions`

**Files:**
- Modify: `src/pscanner/daemon/live_history.py`
- Test: `tests/daemon/test_live_history.py`

When the gate detector observes a trade from a wallet not in `wallet_state_live` (post-bootstrap), it scores with null wallet-quality features (the model's `__none__` encoder token handles this) AND enqueues a job to bootstrap the wallet from `/positions?user=X&closed=true&limit=500`. This task adds the bootstrap method; the enqueue/worker pool lives in the detector plan (#79).

- [ ] **Step 1: Write the failing test using a stub data client**

Add to `tests/daemon/test_live_history.py`:

```python
import dataclasses


@dataclasses.dataclass(frozen=True)
class _FakePosition:
    condition_id: str
    side: str  # YES | NO
    avg_price: float  # implied prob paid
    size: float  # # shares bought
    notional_usd: float
    opened_at: int
    closed_at: int
    won: bool


class _FakeDataClient:
    def __init__(self, positions: list[_FakePosition]) -> None:
        self._positions = positions

    async def get_closed_positions_for_bootstrap(
        self, address: str, *, limit: int = 500
    ) -> list[_FakePosition]:
        del address, limit
        return list(self._positions)


@pytest.mark.asyncio
async def test_bootstrap_wallet_folds_closed_positions() -> None:
    conn = _new_conn()
    try:
        positions = [
            _FakePosition(
                condition_id="0xc1",
                side="YES",
                avg_price=0.40,
                size=100.0,
                notional_usd=40.0,
                opened_at=1_699_000_000,
                closed_at=1_699_500_000,
                won=True,
            ),
            _FakePosition(
                condition_id="0xc2",
                side="NO",
                avg_price=0.20,
                size=50.0,
                notional_usd=10.0,
                opened_at=1_699_000_500,
                closed_at=1_699_500_500,
                won=False,
            ),
        ]
        provider = LiveHistoryProvider(conn=conn, metadata={})
        await provider.bootstrap_wallet("0xabc", data_client=_FakeDataClient(positions))
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_000_000)
    finally:
        conn.close()
    assert wallet.prior_buys_count == 2
    assert wallet.prior_resolved_buys == 2
    assert wallet.prior_wins == 1
    assert wallet.prior_losses == 1
    assert wallet.realized_pnl_usd == pytest.approx(60.0 - 10.0)
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/daemon/test_live_history.py::test_bootstrap_wallet_folds_closed_positions -v`
Expected: FAIL — `LiveHistoryProvider has no attribute 'bootstrap_wallet'`.

- [ ] **Step 3: Add the bootstrap method**

Add to `src/pscanner/daemon/live_history.py`. Define a `BootstrapDataClient` Protocol so the production `pscanner.poly.data.DataClient` and the test fake satisfy the same shape:

```python
from typing import Protocol


class BootstrapDataClient(Protocol):
    """Subset of ``DataClient`` needed for wallet bootstrap."""

    async def get_closed_positions_for_bootstrap(
        self, address: str, *, limit: int = 500
    ) -> list[object]: ...
```

Then add the bootstrap method to the class:

```python
    async def bootstrap_wallet(
        self,
        wallet_address: str,
        *,
        data_client: BootstrapDataClient,
        limit: int = 500,
    ) -> None:
        """Pre-warm wallet state from ``/positions?user=X&closed=true``.

        Folds historical closed positions into the wallet's running state
        so subsequent feature reads have prior_resolved_buys / win_rate /
        realized_pnl populated for previously-unseen wallets.

        No-ops if the wallet already has a row in ``wallet_state_live``
        (idempotent).
        """
        existing = self._conn.execute(
            "SELECT 1 FROM wallet_state_live WHERE wallet_address = ?",
            (wallet_address,),
        ).fetchone()
        if existing is not None:
            return
        positions = await data_client.get_closed_positions_for_bootstrap(
            wallet_address, limit=limit
        )
        positions_sorted = sorted(positions, key=lambda p: p.opened_at)  # type: ignore[attr-defined]
        first_seen = (
            positions_sorted[0].opened_at if positions_sorted else 0  # type: ignore[attr-defined]
        )
        state = empty_wallet_state(first_seen_ts=first_seen)
        for position in positions_sorted:
            buy_trade = Trade(
                tx_hash=f"bootstrap:{wallet_address}:{position.condition_id}",  # type: ignore[attr-defined]
                asset_id=f"{position.condition_id}-{position.side}",  # type: ignore[attr-defined]
                wallet_address=wallet_address,
                condition_id=position.condition_id,  # type: ignore[attr-defined]
                outcome_side=position.side,  # type: ignore[attr-defined]
                bs="BUY",
                price=position.avg_price,  # type: ignore[attr-defined]
                size=position.size,  # type: ignore[attr-defined]
                notional_usd=position.notional_usd,  # type: ignore[attr-defined]
                ts=position.opened_at,  # type: ignore[attr-defined]
                category="",
            )
            state = apply_buy_to_state(state, buy_trade)
            payout = position.size if position.won else 0.0  # type: ignore[attr-defined]
            state = apply_resolution_to_state(
                state,
                won=position.won,  # type: ignore[attr-defined]
                notional_usd=position.notional_usd,  # type: ignore[attr-defined]
                payout_usd=payout,
            )
        self._persist_wallet(wallet_address, state, [])
```

- [ ] **Step 4: Re-run, expect pass**

Run: `uv run pytest tests/daemon/test_live_history.py::test_bootstrap_wallet_folds_closed_positions -v`
Expected: PASS.

- [ ] **Step 5: Add the production `DataClient.get_closed_positions_for_bootstrap` method**

Open `src/pscanner/poly/data.py`. Find `get_positions` (the existing method) and add a sibling:

```python
    async def get_closed_positions_for_bootstrap(
        self,
        address: str,
        *,
        limit: int = 500,
    ) -> list[ClosedPosition]:
        """Fetch closed positions for wallet bootstrap (#78).

        Calls ``/positions?user=X&closed=true&limit=N`` — winners + losers,
        per the CLAUDE.md note distinguishing this from the winners-only
        ``/v1/closed-positions`` endpoint.
        """
        # Implementation follows the same pattern as get_positions but
        # passes closed=true. ClosedPosition is a new pydantic model
        # mapping the response fields used by bootstrap_wallet:
        # condition_id, side, avg_price, size, notional_usd, opened_at,
        # closed_at, won.
        ...
```

(Stub the body so tests don't depend on a live API. Real implementation can land in a separate small commit alongside this task — it's a thin pydantic wrapper around `/positions?closed=true`.)

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/daemon/live_history.py src/pscanner/poly/data.py tests/daemon/test_live_history.py
git commit -m "feat(daemon): bootstrap_wallet from /positions for #78"
```

---

### Task 8: `pscanner daemon bootstrap-features` CLI — cold-start from `corpus_trades`

**Files:**
- Create: `src/pscanner/daemon/bootstrap.py`
- Modify: `src/pscanner/cli.py`
- Create: `tests/daemon/test_bootstrap.py`

The bootstrap walks `corpus_trades` chronologically, calling `provider.observe` / `observe_sell` / `register_resolution` to fold every historical trade into `wallet_state_live` + `market_state_live`. Subsequent daemon starts skip this — state is preserved.

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/daemon/test_bootstrap.py`:

```python
"""Smoke test: pscanner daemon bootstrap-features (#78)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pscanner.corpus.repos import CorpusMarketsRepo, CorpusTradesRepo
from pscanner.daemon.bootstrap import run_bootstrap
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db
from pscanner.store.repo import init_corpus_db


def _seed_corpus(conn: sqlite3.Connection) -> None:
    markets = CorpusMarketsRepo(conn)
    markets.upsert_market(
        platform="polymarket",
        condition_id="0xcond",
        event_slug="evt",
        category="esports",
        closed_at=1_700_001_000,
        total_volume_usd=1000.0,
        market_slug="m",
    )
    trades = CorpusTradesRepo(conn)
    trades.insert_batch(
        [
            {
                "platform": "polymarket",
                "tx_hash": "tx1",
                "asset_id": "0xa",
                "wallet_address": "0xabc",
                "condition_id": "0xcond",
                "outcome_side": "YES",
                "bs": "BUY",
                "price": 0.40,
                "size": 100.0,
                "notional_usd": 40.0,
                "ts": 1_700_000_000,
                "category": "esports",
            }
        ]
    )


def test_run_bootstrap_populates_live_tables(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.sqlite3"
    daemon_path = tmp_path / "daemon.sqlite3"
    corpus_conn = init_corpus_db(corpus_path)
    try:
        _seed_corpus(corpus_conn)
    finally:
        corpus_conn.close()
    daemon_conn = init_db(daemon_path)
    daemon_conn.close()  # close so run_bootstrap opens its own connection
    n = run_bootstrap(corpus_db=corpus_path, daemon_db=daemon_path)
    assert n == 1
    daemon_conn = init_db(daemon_path)
    try:
        provider = LiveHistoryProvider(conn=daemon_conn, metadata={})
        wallet = provider.wallet_state("0xabc", as_of_ts=1_700_000_500)
    finally:
        daemon_conn.close()
    assert wallet.prior_trades_count == 1
    assert wallet.prior_buys_count == 1
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/daemon/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError: pscanner.daemon.bootstrap`.

- [ ] **Step 3: Implement `run_bootstrap`**

Create `src/pscanner/daemon/bootstrap.py`:

```python
"""``pscanner daemon bootstrap-features`` — cold-start the live history tables.

Walks ``corpus_trades`` chronologically, folding every BUY/SELL into
``wallet_state_live`` + ``market_state_live`` via :class:`LiveHistoryProvider`.
Resolutions are registered up-front from ``market_resolutions`` so the
buy-then-resolve drain fires correctly during the walk.

After this completes, the daemon can start with O(1) state load.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

from pscanner.corpus.features import MarketMetadata, Trade
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db
from pscanner.store.repo import init_corpus_db

_LOG = structlog.get_logger(__name__)


def run_bootstrap(
    *,
    corpus_db: Path,
    daemon_db: Path,
    log_every: int = 100_000,
) -> int:
    """Cold-start ``wallet_state_live`` + ``market_state_live`` from corpus.

    Args:
        corpus_db: Path to the corpus SQLite (``data/corpus.sqlite3``).
        daemon_db: Path to the daemon SQLite (``data/pscanner.sqlite3``).
        log_every: Emit a progress log every N trades.

    Returns:
        Total trade count folded.
    """
    corpus_conn = init_corpus_db(corpus_db)
    daemon_conn = init_db(daemon_db)
    try:
        metadata = _load_metadata(corpus_conn)
        provider = LiveHistoryProvider(conn=daemon_conn, metadata=metadata)
        for cond_id, resolved_at, yes_won in corpus_conn.execute(
            "SELECT condition_id, resolved_at, outcome_yes_won FROM market_resolutions"
        ):
            provider.register_resolution(
                condition_id=cond_id,
                resolved_at=int(resolved_at),
                outcome_yes_won=int(yes_won),
            )
        rows = corpus_conn.execute(
            """
            SELECT tx_hash, asset_id, wallet_address, condition_id,
                   outcome_side, bs, price, size, notional_usd, ts, category
            FROM corpus_trades
            ORDER BY ts ASC, tx_hash ASC
            """
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
                category=row[10] or "",
            )
            if trade.bs == "BUY":
                provider.observe(trade)
            elif trade.bs == "SELL":
                provider.observe_sell(trade)
            n += 1
            if n % log_every == 0:
                _LOG.info("daemon.bootstrap.progress", trades_folded=n)
        _LOG.info("daemon.bootstrap.done", trades_folded=n)
        return n
    finally:
        daemon_conn.close()
        corpus_conn.close()


def _load_metadata(conn: sqlite3.Connection) -> dict[str, MarketMetadata]:
    out: dict[str, MarketMetadata] = {}
    for cond_id, category, closed_at, opened_at in conn.execute(
        """
        SELECT condition_id,
               COALESCE(category, ''),
               COALESCE(closed_at, 0),
               COALESCE(enumerated_at, 0)
        FROM corpus_markets
        """
    ):
        out[cond_id] = MarketMetadata(
            condition_id=cond_id,
            category=category,
            closed_at=int(closed_at),
            opened_at=int(opened_at),
        )
    return out
```

- [ ] **Step 4: Wire the CLI subcommand**

Open `src/pscanner/cli.py`. Find the existing `daemon` subparser group (or add one if it doesn't exist; mirror the `corpus` group pattern). Add:

```python
def _add_daemon_subcommands(subparsers: argparse._SubParsersAction[Any]) -> None:
    daemon_parser = subparsers.add_parser("daemon", help="Daemon-side ops")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_command", required=True)
    bootstrap_parser = daemon_sub.add_parser(
        "bootstrap-features",
        help="Cold-start wallet_state_live + market_state_live from corpus_trades.",
    )
    bootstrap_parser.add_argument(
        "--corpus-db", type=Path, default=Path("data/corpus.sqlite3")
    )
    bootstrap_parser.add_argument(
        "--daemon-db", type=Path, default=Path("data/pscanner.sqlite3")
    )
    bootstrap_parser.set_defaults(func=_cmd_daemon_bootstrap)


def _cmd_daemon_bootstrap(args: argparse.Namespace) -> int:
    from pscanner.daemon.bootstrap import run_bootstrap

    n = run_bootstrap(corpus_db=args.corpus_db, daemon_db=args.daemon_db)
    print(f"bootstrap-features: folded {n} trades")  # noqa: T201
    return 0
```

Then call `_add_daemon_subcommands(subparsers)` from the parser-build function (search for `_add_corpus_subcommands` and add the new call right after it).

- [ ] **Step 5: Re-run the test, expect pass**

Run: `uv run pytest tests/daemon/test_bootstrap.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/daemon/bootstrap.py src/pscanner/cli.py tests/daemon/test_bootstrap.py
git commit -m "feat(daemon): pscanner daemon bootstrap-features for #78"
```

---

### Task 9: Profile per-trade `observe` cost

**Files:**
- Create: `scripts/profile_live_history.py`

The DoD requires `<5 ms p99` per `observe` on production scale. Capture a quick benchmark so future regressions are visible.

- [ ] **Step 1: Write the profile script**

Create `scripts/profile_live_history.py`:

```python
"""Quick profile: LiveHistoryProvider.observe() median + p99 latency.

Usage: uv run python scripts/profile_live_history.py [--n 10000]

Synthetic workload — same shape as the parity fixture but at production
scale. Reports median + p99 in microseconds.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.store.db import init_db
from tests.daemon.test_live_history_parity import _build_metadata, _build_synthetic_trades


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trades = _build_synthetic_trades(seed=args.seed, n=args.n)
    metadata = _build_metadata(trades)
    conn = init_db(Path(":memory:"))
    try:
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        observe_times: list[int] = []
        for trade in trades:
            t0 = time.perf_counter_ns()
            provider.observe(trade)
            observe_times.append(time.perf_counter_ns() - t0)
    finally:
        conn.close()
    median = statistics.median(observe_times) / 1_000
    p99 = sorted(observe_times)[int(0.99 * len(observe_times))] / 1_000
    print(f"observe median={median:.1f} us p99={p99:.1f} us  (n={args.n})")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the profile**

Run: `uv run python scripts/profile_live_history.py --n 10000`
Expected output: `observe median=<X> us p99=<Y> us  (n=10000)` — both well under the 5_000 us target.

If p99 exceeds 5_000 us, the most likely cause is the JSON encode/decode on `unresolved_buys_json` for hot wallets with many open buys. Mitigation: batch the unresolved-buys load+save into a single round-trip per `observe` call (already the case in Task 3) and consider promoting the heap to a relational table if profiling shows it's the bottleneck.

- [ ] **Step 3: Commit**

```bash
git add scripts/profile_live_history.py
git commit -m "perf(daemon): profile script for LiveHistoryProvider.observe()"
```

---

### Task 10: CLAUDE.md note + close the loop

**Files:**
- Modify: `CLAUDE.md`

Add a brief note in the "Codebase conventions" section so future agents know about the new tables and the bootstrap requirement.

- [ ] **Step 1: Add the paragraph**

Find the "Codebase conventions" section in `CLAUDE.md` (look for the `## Codebase conventions` header). Insert this bullet near the existing `pscanner.daemon.live_history` would naturally sit (alphabetically among the bullets):

```markdown
- **`wallet_state_live` / `market_state_live`** are the persistent backing for `pscanner.daemon.live_history.LiveHistoryProvider` (#78). They mirror the in-memory accumulators from `StreamingHistoryProvider` and let the daemon restart in O(1) instead of replaying ~15M corpus trades. Cold-start populates them via `pscanner daemon bootstrap-features`. The same pure `apply_*_to_state` helpers in `pscanner.corpus.features` drive both providers — that's the parity contract validated by `tests/daemon/test_live_history_parity.py`. New wallets observed live (post-bootstrap) get `LiveHistoryProvider.bootstrap_wallet` invoked off the hot path; their first trade scores with null wallet-quality features and the model's `__none__` encoder token covers the gap.
```

- [ ] **Step 2: Run the full verify gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: ALL PASS, no warnings (`pyproject.toml` has `filterwarnings = ["error"]`).

If `ty check` flags the `# type: ignore[attr-defined]` comments in Task 7 around `position.opened_at`/etc., the right fix is to define a `BootstrapPosition` Protocol in `live_history.py` matching the fields used and switch the ignores to `position: BootstrapPosition` typing. That keeps the Protocol approach end-to-end and avoids the ignores.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note wallet_state_live + bootstrap-features in CLAUDE.md"
```

---

## Self-review checklist

- **Spec coverage:** every DoD item in #78 has a task. Schema (Task 1), provider class (Tasks 2-4), restart parity (Task 5), streaming parity (Task 6), bootstrap_wallet (Task 7), bootstrap CLI (Task 8), perf check (Task 9), docs (Task 10).
- **No placeholders:** the `get_closed_positions_for_bootstrap` body in Task 7 Step 5 is a stub-by-design (the test uses `_FakeDataClient`). The real HTTP wrapper is one paragraph of httpx; it's noted as in-scope-of-the-task to keep the plan from sprawling into a `/positions` API rework.
- **Type consistency:** `LiveHistoryProvider.observe` matches the streaming provider's signature (`Trade`); `observe_sell` accepts `_TradeFields` (the Protocol from `features.py:43-61`); `wallet_state` returns `WalletState` (frozen dataclass from `features.py:64-92`).

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-06-gate-live-history.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batched checkpoints for review.

Once #78 lands, the next plan (`2026-05-06-gate-detector.md`) builds on `LiveHistoryProvider` to add the gate-model detector and market-scoped trade collector.
