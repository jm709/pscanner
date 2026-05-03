# On-chain Trades — Phase 1: Skeleton + Event Decoder

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a self-contained, tested foundation for indexing Polymarket's on-chain `OrderFilled` events — the event decoder + the asset-id → condition-id lookup layer — so a follow-up Phase 2 can plug in `eth_getLogs` pagination and a CLI command without revisiting the data structures.

**Architecture:** Phase 1 is purely local Python + SQLite work, no RPC. We add a `pscanner.poly.onchain` module with the event decoder (data class + ABI-decoding from a raw log), and a new `asset_index` table + `AssetIndexRepo` for the asset_id → condition_id mapping that Phase 2 will need to convert decoded events into `CorpusTrade` rows. The asset index is backfilled from existing `corpus_trades` rows, so we have ~99% coverage before any RPC calls happen.

**Tech Stack:** Python 3.13, sqlite3, no new dependencies. Decoder uses manual ABI decoding (8 fixed-size 32-byte words) — `eth_abi` is overkill for a single event with no dynamic fields. Tests use existing `tmp_corpus_db` fixture and synthetic log payloads.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/pscanner/poly/onchain.py` | create | Constants (contract address, OrderFilled topic0, event signature) and `OrderFilledEvent` dataclass + `decode_order_filled` function |
| `tests/poly/test_onchain.py` | create | Tests for decoder against synthetic log payloads |
| `src/pscanner/corpus/db.py` | modify | Add `asset_index` table to `_SCHEMA_STATEMENTS` |
| `src/pscanner/corpus/repos.py` | modify | Add `AssetEntry` dataclass and `AssetIndexRepo` (upsert, get, bulk_backfill_from_corpus_trades) |
| `tests/corpus/test_repos_asset_index.py` | create | Tests for `AssetIndexRepo` |
| `scripts/backfill_asset_index.py` | create | One-shot CLI to populate `asset_index` from existing `corpus_trades` data |

## What's deferred to Phase 2 (do not implement here)

- The `eth_getLogs` HTTP RPC client and block-range pagination
- The `pscanner corpus onchain-backfill` CLI command
- Resumability via `corpus_state` for the on-chain cursor
- Negative-Risk adapter contract handling (separate event source)
- Live integration tests against a Polygon RPC

---

## Task 1: `OrderFilledEvent` dataclass + ABI decoder

**Files:**
- Create: `src/pscanner/poly/onchain.py`
- Test: `tests/poly/test_onchain.py`

The OrderFilled event has 8 unindexed parameters, so all data lands in the log's `data` field as 8 × 32-byte ABI-encoded words. No dynamic fields, so decoding is fixed-offset slicing.

- [ ] **Step 1.1: Create the test file with one failing decoder test**

```python
# tests/poly/test_onchain.py
"""Tests for `pscanner.poly.onchain` — OrderFilled log decoder."""

from __future__ import annotations

from pscanner.poly.onchain import OrderFilledEvent, decode_order_filled


def _make_log(
    *,
    order_hash: str,
    maker: str,
    taker: str,
    maker_asset_id: int,
    taker_asset_id: int,
    making: int,
    taking: int,
    fee: int,
    tx_hash: str = "0x" + "ab" * 32,
    block_number: int = 0x1234567,
    log_index: int = 5,
) -> dict[str, object]:
    """Build a synthetic eth_getLogs response entry for an OrderFilled event."""
    # Each of the 8 unindexed params gets a 32-byte slot, big-endian.
    # Addresses are right-aligned (12 zero bytes prefix + 20-byte address).
    parts = [
        bytes.fromhex(order_hash[2:]),
        bytes(12) + bytes.fromhex(maker[2:]),
        bytes(12) + bytes.fromhex(taker[2:]),
        maker_asset_id.to_bytes(32, "big"),
        taker_asset_id.to_bytes(32, "big"),
        making.to_bytes(32, "big"),
        taking.to_bytes(32, "big"),
        fee.to_bytes(32, "big"),
    ]
    data = b"".join(parts)
    assert len(data) == 8 * 32
    return {
        "data": "0x" + data.hex(),
        "topics": ["0x" + "00" * 32],  # decoder doesn't use topic0; Phase 2 filters by it
        "transactionHash": tx_hash,
        "blockNumber": hex(block_number),
        "logIndex": hex(log_index),
    }


def test_decode_order_filled_extracts_all_fields() -> None:
    log = _make_log(
        order_hash="0x" + "cd" * 32,
        maker="0x" + "11" * 20,
        taker="0x" + "22" * 20,
        maker_asset_id=42,
        taker_asset_id=10**40,  # large uint256 to exercise full 32-byte width
        making=1_000_000,
        taking=500_000,
        fee=125,
    )
    event = decode_order_filled(log)
    assert isinstance(event, OrderFilledEvent)
    assert event.order_hash == "0x" + "cd" * 32
    assert event.maker == "0x" + "11" * 20
    assert event.taker == "0x" + "22" * 20
    assert event.maker_asset_id == 42
    assert event.taker_asset_id == 10**40
    assert event.making == 1_000_000
    assert event.taking == 500_000
    assert event.fee == 125
    assert event.tx_hash == "0x" + "ab" * 32
    assert event.block_number == 0x1234567
    assert event.log_index == 5
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `uv run pytest tests/poly/test_onchain.py -q`
Expected: FAIL with ModuleNotFoundError: No module named `pscanner.poly.onchain`

- [ ] **Step 1.3: Create the module with the decoder**

```python
# src/pscanner/poly/onchain.py
"""On-chain event decoding for Polymarket's CTF Exchange contract.

Phase 1 scope: pure decoding from raw `eth_getLogs` payloads to typed
events. No RPC client lives here — Phase 2 wires `eth_getLogs` calls
and a backfill CLI on top of these primitives.

The CTF Exchange (Polygon mainnet `0x4bFb41d5...`) emits
`OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)`
on every match. All 8 parameters are unindexed, so they land in the
log's `data` field as a flat 256-byte (8 × 32) ABI-encoded sequence.
Addresses are right-aligned in their 32-byte slots; uint256/bytes32
take the full slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Polymarket CTF Exchange on Polygon mainnet. Verified from
# https://github.com/Polymarket/ctf-exchange README.
CTF_EXCHANGE_ADDRESS: Final[str] = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# keccak256("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
# Hard-coded to avoid pulling in a Keccak dependency for one constant.
# Phase 2 should add an integration test that compares this against a
# real log's topics[0] when first hitting the live RPC.
ORDER_FILLED_TOPIC0: Final[str] = (
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"
)

_DATA_BYTE_LEN: Final[int] = 8 * 32  # 8 fields × 32 bytes
_SLOT: Final[int] = 32
_ADDRESS_BYTES: Final[int] = 20


@dataclass(frozen=True)
class OrderFilledEvent:
    """Decoded `OrderFilled` event with log-position metadata.

    Field semantics (per CTF Exchange `Trading.sol`):
        order_hash: unique hash of the matched order.
        maker: address of the resting (limit) side.
        taker: address of the aggressor side.
        maker_asset_id: ERC1155 token id (or 0 for USDC collateral) the
            maker is giving up.
        taker_asset_id: ERC1155 token id (or 0) the taker is giving up.
        making: amount the maker gives, in the maker asset's smallest unit.
        taking: amount the taker gives, in the taker asset's smallest unit.
        fee: protocol fee in the taker-side asset.
        tx_hash: the containing transaction hash.
        block_number: Polygon block number containing the log.
        log_index: position of this log within the transaction receipt.
    """

    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    making: int
    taking: int
    fee: int
    tx_hash: str
    block_number: int
    log_index: int


def _hex_to_int(value: str | int) -> int:
    """Coerce an eth_getLogs numeric field (often hex-string) to int."""
    if isinstance(value, int):
        return value
    return int(value, 16)


def decode_order_filled(log: dict[str, object]) -> OrderFilledEvent:
    """Decode a raw `eth_getLogs` response entry into a typed event.

    Args:
        log: A single entry from an `eth_getLogs` JSON-RPC response. Must
            have ``data`` (hex string with ``0x`` prefix, 256 bytes payload),
            ``transactionHash``, ``blockNumber`` (int or hex string), and
            ``logIndex`` (int or hex string).

    Returns:
        Decoded `OrderFilledEvent`.

    Raises:
        ValueError: If ``data`` is malformed or shorter than 256 bytes.
        KeyError: If a required log field is missing.
    """
    raw = log["data"]
    if not isinstance(raw, str) or not raw.startswith("0x"):
        raise ValueError(f"log.data must be hex string with 0x prefix, got: {raw!r}")
    payload = bytes.fromhex(raw[2:])
    if len(payload) < _DATA_BYTE_LEN:
        raise ValueError(
            f"log.data too short: {len(payload)} bytes (expected {_DATA_BYTE_LEN})"
        )

    def slot(i: int) -> bytes:
        return payload[i * _SLOT : (i + 1) * _SLOT]

    def slot_address(i: int) -> str:
        # Right-aligned address: last 20 bytes of the slot
        return "0x" + slot(i)[_SLOT - _ADDRESS_BYTES :].hex()

    def slot_uint(i: int) -> int:
        return int.from_bytes(slot(i), "big")

    tx_hash = log["transactionHash"]
    if not isinstance(tx_hash, str):
        raise ValueError(f"transactionHash must be str, got: {type(tx_hash).__name__}")

    return OrderFilledEvent(
        order_hash="0x" + slot(0).hex(),
        maker=slot_address(1),
        taker=slot_address(2),
        maker_asset_id=slot_uint(3),
        taker_asset_id=slot_uint(4),
        making=slot_uint(5),
        taking=slot_uint(6),
        fee=slot_uint(7),
        tx_hash=tx_hash,
        block_number=_hex_to_int(log["blockNumber"]),  # type: ignore[arg-type]
        log_index=_hex_to_int(log["logIndex"]),  # type: ignore[arg-type]
    )
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `uv run pytest tests/poly/test_onchain.py -q`
Expected: PASS — 1 test passes

- [ ] **Step 1.5: Add error-path tests**

Append to `tests/poly/test_onchain.py`:

```python
import pytest


def test_decode_order_filled_rejects_short_data() -> None:
    log = {
        "data": "0x" + "00" * 100,  # only 100 bytes, need 256
        "transactionHash": "0x" + "00" * 32,
        "blockNumber": "0x1",
        "logIndex": "0x0",
    }
    with pytest.raises(ValueError, match="too short"):
        decode_order_filled(log)


def test_decode_order_filled_rejects_missing_prefix() -> None:
    log = {
        "data": "ab" * 256,  # missing 0x prefix
        "transactionHash": "0x" + "00" * 32,
        "blockNumber": "0x1",
        "logIndex": "0x0",
    }
    with pytest.raises(ValueError, match="0x prefix"):
        decode_order_filled(log)


def test_decode_order_filled_handles_int_block_number() -> None:
    """Some RPC providers return blockNumber as int, others as hex-string."""
    log = _make_log(
        order_hash="0x" + "00" * 32,
        maker="0x" + "00" * 20,
        taker="0x" + "00" * 20,
        maker_asset_id=0,
        taker_asset_id=0,
        making=0,
        taking=0,
        fee=0,
        block_number=0,
        log_index=0,
    )
    log["blockNumber"] = 12345  # int, not hex
    log["logIndex"] = 7
    event = decode_order_filled(log)
    assert event.block_number == 12345
    assert event.log_index == 7
```

- [ ] **Step 1.6: Run all tests in the new file, format, lint, typecheck**

Run:
```
uv run pytest tests/poly/test_onchain.py -q
uv run ruff format src/pscanner/poly/onchain.py tests/poly/test_onchain.py
uv run ruff check src/pscanner/poly/onchain.py tests/poly/test_onchain.py
uv run ty check src/pscanner/poly/onchain.py
```
Expected: 4 tests pass; format/lint/ty all clean.

- [ ] **Step 1.7: Commit**

```bash
git add src/pscanner/poly/onchain.py tests/poly/test_onchain.py
git commit -m "feat(poly): add OrderFilled event decoder for on-chain trade indexing

Phase 1 of #42 on-chain backfill. Pure-Python ABI decoder for the CTF
Exchange's OrderFilled event; no RPC client yet. Manual fixed-offset
decoding of the 8 × 32-byte data payload — no eth_abi dependency.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `asset_index` schema migration

**Files:**
- Modify: `src/pscanner/corpus/db.py:14-118` (extend `_SCHEMA_STATEMENTS`)
- Test: `tests/corpus/test_db.py` (add one new test)

The `asset_index` table maps every Polymarket CTF asset_id (uint256 stored as TEXT for sqlite ergonomics — same convention as `corpus_trades.asset_id`) to its parent market and outcome side. Phase 2's on-chain ingest needs this lookup to convert decoded `OrderFilledEvent`s into `CorpusTrade` rows (which require `condition_id` and `outcome_side`).

- [ ] **Step 2.1: Add a failing test that the new table exists after init**

Append to `tests/corpus/test_db.py`:

```python
def test_init_corpus_db_creates_asset_index_table(tmp_path: pytest.TempPathFactory) -> None:
    db_path = tmp_path / "init_asset_idx.sqlite3"  # type: ignore[attr-defined]
    conn = init_corpus_db(db_path)  # type: ignore[arg-type]
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(asset_index)").fetchall()}
        assert cols == {"asset_id", "condition_id", "outcome_side", "outcome_index"}
        # PRIMARY KEY check
        info = conn.execute("PRAGMA table_info(asset_index)").fetchall()
        pk_cols = [r[1] for r in info if r[5] == 1]
        assert pk_cols == ["asset_id"]
    finally:
        conn.close()
```

(Note: existing tests in `test_db.py` use the `tmp_path` fixture pattern — match whatever's already there. Look at one of the existing schema tests in the file before writing this one.)

- [ ] **Step 2.2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_db.py -q -k asset_index`
Expected: FAIL — `no such table: asset_index`

- [ ] **Step 2.3: Add the table to `_SCHEMA_STATEMENTS`**

In `src/pscanner/corpus/db.py`, append to the `_SCHEMA_STATEMENTS` tuple (just before the closing parenthesis on line 118, after the `corpus_state` table):

```python
    """
    CREATE TABLE IF NOT EXISTS asset_index (
      asset_id TEXT PRIMARY KEY,
      condition_id TEXT NOT NULL,
      outcome_side TEXT NOT NULL,
      outcome_index INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_asset_index_condition ON asset_index(condition_id)",
```

- [ ] **Step 2.4: Run the test to verify it passes**

Run: `uv run pytest tests/corpus/test_db.py -q -k asset_index`
Expected: PASS

- [ ] **Step 2.5: Commit**

```bash
git add src/pscanner/corpus/db.py tests/corpus/test_db.py
git commit -m "feat(corpus): add asset_index table for asset_id -> condition_id lookup

Phase 1 of #42 on-chain backfill. New table maps Polymarket CTF asset
IDs (uint256 stored as TEXT) to their parent market and outcome side.
Phase 2's on-chain ingest needs this to convert decoded OrderFilled
events into CorpusTrade rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `AssetEntry` dataclass + `AssetIndexRepo`

**Files:**
- Modify: `src/pscanner/corpus/repos.py` (append at end)
- Create: `tests/corpus/test_repos_asset_index.py`

The repo needs three operations: upsert a single entry, look up by asset_id, and bulk-backfill from existing `corpus_trades` rows (which already carry asset_id + condition_id + outcome_side together).

- [ ] **Step 3.1: Write the failing test**

Create `tests/corpus/test_repos_asset_index.py`:

```python
"""Tests for `AssetIndexRepo`."""

from __future__ import annotations

import sqlite3

from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusTrade,
    CorpusTradesRepo,
)


def test_upsert_inserts_new_entry(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    repo.upsert(
        AssetEntry(
            asset_id="999",
            condition_id="0xabc",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    got = repo.get("999")
    assert got == AssetEntry(
        asset_id="999",
        condition_id="0xabc",
        outcome_side="YES",
        outcome_index=0,
    )


def test_get_returns_none_for_unknown_asset(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    assert repo.get("nope") is None


def test_upsert_updates_existing_entry(tmp_corpus_db: sqlite3.Connection) -> None:
    repo = AssetIndexRepo(tmp_corpus_db)
    repo.upsert(
        AssetEntry(asset_id="1", condition_id="0xa", outcome_side="YES", outcome_index=0)
    )
    repo.upsert(
        AssetEntry(asset_id="1", condition_id="0xa", outcome_side="NO", outcome_index=1)
    )
    got = repo.get("1")
    assert got is not None
    assert got.outcome_side == "NO"
    assert got.outcome_index == 1


def test_backfill_from_corpus_trades_populates_index(tmp_corpus_db: sqlite3.Connection) -> None:
    trades = CorpusTradesRepo(tmp_corpus_db)
    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xt1",
                asset_id="100",
                wallet_address="0xw",
                condition_id="0xc1",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=10.0,
                notional_usd=5.0,
                ts=1000,
            ),
            CorpusTrade(
                tx_hash="0xt2",
                asset_id="200",
                wallet_address="0xw",
                condition_id="0xc1",
                outcome_side="NO",
                bs="BUY",
                price=0.5,
                size=10.0,
                notional_usd=5.0,
                ts=1001,
            ),
            # duplicate (asset 100 again) — should not double-insert
            CorpusTrade(
                tx_hash="0xt3",
                asset_id="100",
                wallet_address="0xother",
                condition_id="0xc1",
                outcome_side="YES",
                bs="SELL",
                price=0.6,
                size=5.0,
                notional_usd=3.0,
                ts=1002,
            ),
        ]
    )
    repo = AssetIndexRepo(tmp_corpus_db)
    n = repo.backfill_from_corpus_trades()
    assert n == 2  # asset 100 and asset 200
    assert repo.get("100") == AssetEntry(
        asset_id="100",
        condition_id="0xc1",
        outcome_side="YES",
        outcome_index=0,
    )
    assert repo.get("200") == AssetEntry(
        asset_id="200",
        condition_id="0xc1",
        outcome_side="NO",
        outcome_index=1,
    )


def test_backfill_is_idempotent(tmp_corpus_db: sqlite3.Connection) -> None:
    trades = CorpusTradesRepo(tmp_corpus_db)
    trades.insert_batch(
        [
            CorpusTrade(
                tx_hash="0xt1",
                asset_id="100",
                wallet_address="0xw",
                condition_id="0xc1",
                outcome_side="YES",
                bs="BUY",
                price=0.5,
                size=10.0,
                notional_usd=5.0,
                ts=1000,
            ),
        ]
    )
    repo = AssetIndexRepo(tmp_corpus_db)
    n1 = repo.backfill_from_corpus_trades()
    n2 = repo.backfill_from_corpus_trades()
    assert n1 == 1
    # Second run is a no-op (or at minimum doesn't grow the table beyond 1 row).
    n_total = tmp_corpus_db.execute("SELECT COUNT(*) FROM asset_index").fetchone()[0]
    assert n_total == 1
    # n2 may be 0 (fully idempotent) or 1 (re-upsert). Either is acceptable.
    assert n2 in (0, 1)
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/corpus/test_repos_asset_index.py -q`
Expected: FAIL with ImportError on `AssetEntry` / `AssetIndexRepo`.

- [ ] **Step 3.3: Add the dataclass + repo to `repos.py`**

Append at the end of `src/pscanner/corpus/repos.py`:

```python
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
        """Insert or replace the row for `entry.asset_id`."""
        self._conn.execute(
            """
            INSERT INTO asset_index (asset_id, condition_id, outcome_side, outcome_index)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
              condition_id = excluded.condition_id,
              outcome_side = excluded.outcome_side,
              outcome_index = excluded.outcome_index
            """,
            (entry.asset_id, entry.condition_id, entry.outcome_side, entry.outcome_index),
        )
        self._conn.commit()

    def get(self, asset_id: str) -> AssetEntry | None:
        """Look up an entry by its `asset_id`, or `None` if not present."""
        row = self._conn.execute(
            "SELECT asset_id, condition_id, outcome_side, outcome_index "
            "FROM asset_index WHERE asset_id = ?",
            (asset_id,),
        ).fetchone()
        if row is None:
            return None
        return AssetEntry(
            asset_id=row["asset_id"],
            condition_id=row["condition_id"],
            outcome_side=row["outcome_side"],
            outcome_index=row["outcome_index"],
        )

    def backfill_from_corpus_trades(self) -> int:
        """Populate `asset_index` from existing `corpus_trades` rows.

        Each `corpus_trades` row already carries (asset_id, condition_id,
        outcome_side). We derive `outcome_index` from the side: YES → 0,
        NO → 1 (matches Polymarket's `outcome_prices` array ordering).

        Returns:
            Number of distinct `asset_id`s inserted (excludes existing rows
            that conflicted on the PRIMARY KEY).
        """
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO asset_index (
              asset_id, condition_id, outcome_side, outcome_index
            )
            SELECT
              asset_id,
              condition_id,
              outcome_side,
              CASE outcome_side WHEN 'YES' THEN 0 ELSE 1 END
            FROM (
              SELECT asset_id, condition_id, outcome_side
              FROM corpus_trades
              GROUP BY asset_id
            )
            """
        )
        inserted = cursor.rowcount
        self._conn.commit()
        return inserted
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `uv run pytest tests/corpus/test_repos_asset_index.py -q`
Expected: 5 tests pass

- [ ] **Step 3.5: Run lint, format, typecheck**

Run:
```
uv run ruff format src/pscanner/corpus/repos.py tests/corpus/test_repos_asset_index.py
uv run ruff check src/pscanner/corpus/repos.py tests/corpus/test_repos_asset_index.py
uv run ty check src/pscanner/corpus/repos.py
```
Expected: all clean

- [ ] **Step 3.6: Commit**

```bash
git add src/pscanner/corpus/repos.py tests/corpus/test_repos_asset_index.py
git commit -m "feat(corpus): add AssetIndexRepo for asset_id -> condition_id lookup

Phase 1 of #42 on-chain backfill. AssetEntry dataclass + repo with
upsert/get/backfill_from_corpus_trades. Phase 2 will use this to
convert decoded OrderFilled events into CorpusTrade rows without a
per-event gamma roundtrip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Backfill script + run on real corpus

**Files:**
- Create: `scripts/backfill_asset_index.py`

Same shape as `scripts/backfill_close_times.py` (already in tree) — argparse + sqlite3 connection + before/after counts + idempotent backfill call.

- [ ] **Step 4.1: Create the script**

```python
# scripts/backfill_asset_index.py
"""One-shot backfill of `asset_index` from existing `corpus_trades` data.

Phase 1 of #42 on-chain backfill. Populates the `asset_index` lookup
table from the (asset_id, condition_id, outcome_side) tuples already
present in `corpus_trades`. Idempotent: re-runs are a no-op for rows
already indexed.

Usage:
    uv run python scripts/backfill_asset_index.py
    uv run python scripts/backfill_asset_index.py --db data/corpus.sqlite3
"""

# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import argparse
import sqlite3

from pscanner.corpus.repos import AssetIndexRepo


def _row_count(conn: sqlite3.Connection) -> int:
    """Return total rows currently in `asset_index`."""
    return int(conn.execute("SELECT COUNT(*) FROM asset_index").fetchone()[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=str,
        default="data/corpus.sqlite3",
        help="Path to the corpus SQLite database",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: connect, log before/after counts, run backfill."""
    args = _parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        before = _row_count(conn)
        print(f"before: asset_index rows = {before:,}")

        repo = AssetIndexRepo(conn)
        inserted = repo.backfill_from_corpus_trades()
        print(f"inserted: {inserted:,} new asset_index rows")

        after = _row_count(conn)
        print(f"after:  asset_index rows = {after:,}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4.2: Lint, format, typecheck**

Run:
```
uv run ruff format scripts/backfill_asset_index.py
uv run ruff check scripts/backfill_asset_index.py
uv run ty check scripts/backfill_asset_index.py
```
Expected: all clean

- [ ] **Step 4.3: Run against the real corpus and verify output**

Run: `uv run python scripts/backfill_asset_index.py --db data/corpus.sqlite3`

Expected output: `before: asset_index rows = 0`, then a non-zero `inserted` count (likely on the order of 8-10k, since each market has 2 binary outcomes and we have ~4,400 markets), then `after` matching `inserted`.

Sanity-check the result manually:

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
n = conn.execute('SELECT COUNT(*) FROM asset_index').fetchone()[0]
n_per_market = conn.execute('SELECT condition_id, COUNT(*) FROM asset_index GROUP BY condition_id LIMIT 5').fetchall()
print(f'total: {n:,}')
print('sample markets:', [tuple(r) for r in n_per_market])
"
```

Expected: total ~8-10k. Sample shows 1-2 assets per market (2 if both YES and NO appeared in trades, 1 if only one side traded).

- [ ] **Step 4.4: Commit**

```bash
git add scripts/backfill_asset_index.py
git commit -m "feat(scripts): add asset_index backfill script

Phase 1 of #42 on-chain backfill. One-shot CLI to populate the new
asset_index table from existing corpus_trades rows. Idempotent — safe
to re-run after corpus rebuilds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Final verification

- [ ] **Step 5.1: Run the full Quick Verify**

Run (from CLAUDE.md): `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`

Expected: all clean. Existing 734 tests + the new ones added in Tasks 1, 2, 3 pass.

- [ ] **Step 5.2: Update issue #42 status comment**

```bash
gh issue comment 42 --body "Phase 1 landed: \`pscanner.poly.onchain\` decoder + \`asset_index\` table + \`AssetIndexRepo\` + backfill script. Decoder is fully unit-tested against synthetic ABI-encoded log payloads; no RPC client yet. \`scripts/backfill_asset_index.py\` populated <N> asset entries from existing corpus_trades data.

Phase 2 (deferred): \`eth_getLogs\` HTTP client, block-range pagination with resumability, \`pscanner corpus onchain-backfill\` CLI command, and the asset-index gap-fill via gamma when on-chain encounters an unknown asset_id."
```

(Replace `<N>` with the actual count from Task 4.3.)

---

## Self-Review

**Spec coverage:** Phase 1 was scoped as "skeleton + event decoder, defer CLI + backfill to next session." All four pieces present:
- Constants (contract address + topic0): Task 1, ✓
- Event decoder: Task 1, ✓
- Asset-index storage layer (table + repo): Tasks 2-3, ✓
- Asset-index seed from existing data: Task 4, ✓
- Phase 2 boundaries clearly enumerated in the file structure section. ✓

**Placeholder scan:** No "TBD", "implement later", or "similar to". Each task has full file paths, complete code, exact commands, and expected outputs. The one acknowledged constant-without-derivation is `ORDER_FILLED_TOPIC0` — but that's documented in code as "verify in Phase 2 against a live log", which is a concrete plan, not a placeholder.

**Type consistency:** `AssetEntry` is referenced consistently in Task 3.1 (test) and 3.3 (impl). `OrderFilledEvent` field names match between Task 1.1 (test), 1.3 (impl), 1.5 (test). `backfill_from_corpus_trades` signature is consistent between Task 3.1 (test) and 3.3 (impl). No drift.
