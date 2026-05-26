# V1 Subgraph Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a V1-subgraph adapter that fills `corpus_trades` for the 2,769 markets carrying `corpus_markets.v1_history_pending = 1`, then wire a dispatcher that drives the V1 and V2 backfills independently from one `pscanner corpus subgraph-backfill` invocation.

**Architecture:** The Stage 0 investigation (committed 2026-05-26, see `scripts/v1_investigation_report.md`) proved that the V1 subgraph still serves the *pre-#151* `OrderFilledEvent` schema — flat `maker`/`taker` addresses, `makerAssetId` + `takerAssetId` (one is `"0"` for USDC), and `makerAmountFilled` + `takerAmountFilled` in the same 6-decimal base units V2 uses. A new `subgraph_ingest_v1.py` module mirrors V2's shape but uses two paginated passes per market (`makerAssetId_in` and `takerAssetId_in`, like the pre-#151 code) since the V1 schema has no `_or` operator. The V1 adapter emits the existing `OrderFilledEvent` dataclass so the downstream `event_to_corpus_trade` path is shared with V2 unchanged. A thin `subgraph_dispatch.py` runs V2 first, then V1, then the shared `_clear_truncation_flags`. The new `onchain_v1_processed_at` column is the V1-side sentinel.

**Tech Stack:** Python 3.13 + uv + ruff + ty + pytest · `pscanner.poly.subgraph.SubgraphClient` for GraphQL · `sqlite3` + `pscanner.corpus.db.init_corpus_db` for storage · `structlog` for events.

**Spec reference:** `docs/superpowers/specs/2026-05-26-issue-193-v1-subgraph-adapter-design.md` (v2, post-investigation).

**Recycled scaffolding (almost direct lift):** `git show a809378^:src/pscanner/corpus/subgraph_ingest.py` — pre-#151 module. The query string, paginator, and orchestrator skeleton match V1's actual schema almost exactly. The adapter needs a small tweak to read `orderHash` directly from the row (V1 has it as a top-level field) and to ignore V1's `side` and `price` fields (the downstream `event_to_corpus_trade` derives BUY/SELL from `(maker_asset_id, taker_asset_id)` directly).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/investigate_v1_schema.py` | ✅ Done (Task 1) | Stage 0 probe; produced `scripts/v1_investigation_report.md` confirming V1 still serves the pre-#151 schema. |
| `scripts/v1_investigation_report.md` | ✅ Done (Task 1) | Committed evidence; cited in the spec's Background section. |
| `scripts/verify_v1_units.py` | Create (Task 2) | Stage 1: pick a market with fills in the April 3 – April 28 overlap window, reconcile V1 vs V2 row-by-row on `(transactionHash, orderHash)`, emit `tests/corpus/fixtures/v1_v2_overlap.json`. |
| `tests/corpus/fixtures/v1_v2_overlap.json` | Create (Task 2) | Committed ground-truth fixture: ~5 paired V1+V2 rows. Frozen post Stage 1. |
| `src/pscanner/corpus/db.py` | Modify (Task 3) | Add `onchain_v1_processed_at INTEGER` to `_SCHEMA_STATEMENTS` and `_MIGRATIONS`. |
| `src/pscanner/corpus/subgraph_ingest_v1.py` | Create (Tasks 4–6) | V1 GraphQL query strings (one per side), `subgraph_v1_row_to_event` adapter, `_paginate_v1_side`, `iter_v1_market_trades`, `_load_pending_v1_markets`, `_mark_v1_processed`, `_backfill_one_v1_market`, `run_v1_subgraph_backfill`, `V1SubgraphRunSummary`. |
| `src/pscanner/corpus/subgraph_dispatch.py` | Create (Task 7) | `run_subgraph_backfill_dispatched(...)` — runs V2 first, then V1, then `_clear_truncation_flags`. |
| `src/pscanner/corpus/cli.py` | Modify (Task 8) | Add `--subgraph-version`, `--v1-subgraph-id`, `--v2-subgraph-id` flags; preserve `--subgraph-id` as a deprecated alias that emits `subgraph.cli.deprecated_flag`. Replace the direct `run_subgraph_backfill` call with `run_subgraph_backfill_dispatched`. |
| `tests/corpus/test_subgraph_ingest_v1.py` | Create (Tasks 4–6, 9) | Adapter unit tests (against the fixture), paginator unit tests (two-pass), orchestrator sentinel-hygiene tests, hybrid-market integration. |
| `tests/corpus/test_subgraph_dispatch.py` | Create (Task 7) | Dispatcher routing tests + CLI flag tests. |
| `CLAUDE.md` | Modify (Task 10) | Update the V1 subgraph entry — note the schema reality (pre-#151 shape, not the originally-claimed `OrderFill` schema). |

---

## Task 1: Stage 0 — V1 schema investigation script

**Status:** ✅ DONE. Committed as `e430e54` on the worktree branch.

The investigation found that the V1 subgraph still serves the
pre-#151 `OrderFilledEvent` schema (flat addresses, `makerAssetId` +
`takerAssetId`, same amount fields as V2). The original spec's
`marketId="0"` cohort concern was based on a misreading — no such
field exists. See `scripts/v1_investigation_report.md` for the full
schema dump and recovery-rate analysis.

---

## Task 2: Stage 1 — Overlap-window verification script + fixture

**Files:**
- Create: `scripts/verify_v1_units.py`
- Create: `tests/corpus/fixtures/v1_v2_overlap.json` (script output, committed)

Stage 1 proves that V1 and V2 produce identical `(maker, taker,
makerAmountFilled, takerAmountFilled)` for the same
`(transactionHash, orderHash)` keys. The committed fixture is what
every adapter unit test in Tasks 4–6 asserts against.

- [ ] **Step 1: Identify an overlap-window candidate market**

Run this SQL to pick a candidate (one that traded across the April
3–28 overlap):

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('/home/macph/projects/polymarketScanner/data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
row = conn.execute('''
  SELECT m.condition_id, m.market_slug, COUNT(t.tx_hash) AS trades,
         MIN(t.ts) AS first_ts, MAX(t.ts) AS last_ts
  FROM corpus_markets m JOIN corpus_trades t USING (condition_id)
  WHERE m.platform = 'polymarket'
    AND m.v1_history_pending = 1
    AND m.onchain_processed_at IS NOT NULL
  GROUP BY m.condition_id
  HAVING first_ts < 1775220779 AND last_ts > 1775220779
  ORDER BY trades DESC LIMIT 5
''').fetchall()
for r in row: print(dict(r))
"
```

Expected: up to 5 candidate condition_ids printed. Record the top one;
the script will receive it as `--condition-id`. If 0 rows, fall back
to dropping the `v1_history_pending=1` filter (some hybrid markets may
not be flagged).

- [ ] **Step 2: Write the verification script**

```python
# scripts/verify_v1_units.py
"""Stage 1 of issue #193: verify V1 amounts equal V2 amounts.

Reconciles V1 and V2 `orderFilledEvents` for one shared condition_id,
row-by-row on (transactionHash, orderHash). Emits the matched pairs as
`tests/corpus/fixtures/v1_v2_overlap.json` so the V1 adapter unit
tests have ground truth.

Run:
    GRAPH_API_KEY=... uv run python scripts/verify_v1_units.py \\
        --condition-id <hex> --db /home/macph/projects/polymarketScanner/data/corpus.sqlite3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.poly.subgraph import SubgraphClient

_V1 = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
_V2 = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
_GATEWAY = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{id}"
# April 3 2026 -> April 28 2026 overlap window (Unix seconds).
_OVERLAP_MIN = 1775220779
_OVERLAP_MAX = 1777374040

_V1_Q_MAKER = """
query($ids: [String!]!, $tmin: BigInt!, $tmax: BigInt!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $ids, timestamp_gte: $tmin, timestamp_lte: $tmax }
    first: $first orderBy: timestamp orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee side price
  }
}
"""

_V1_Q_TAKER = """
query($ids: [String!]!, $tmin: BigInt!, $tmax: BigInt!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $ids, timestamp_gte: $tmin, timestamp_lte: $tmax }
    first: $first orderBy: timestamp orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee side price
  }
}
"""

_V2_Q = """
query($ids: [String!]!, $tmin: BigInt!, $tmax: BigInt!, $first: Int!) {
  orderFilledEvents(
    where: { market_in: $ids, timestamp_gte: $tmin, timestamp_lte: $tmax }
    first: $first orderBy: timestamp orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker { id } taker { id } market { id }
    tokenId side makerAmountFilled takerAmountFilled fee
  }
}
"""


async def _amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="V1/V2 overlap reconciliation (issue #193, Stage 1)"
    )
    parser.add_argument("--condition-id", required=True)
    parser.add_argument(
        "--db", default="/home/macph/projects/polymarketScanner/data/corpus.sqlite3"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--output", default="tests/corpus/fixtures/v1_v2_overlap.json"
    )
    parser.add_argument("--per-side", type=int, default=200)
    args = parser.parse_args(argv)
    api_key = args.api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        sys.stderr.write("error: --api-key or $GRAPH_API_KEY required\n")
        return 2

    conn = init_corpus_db(Path(args.db))
    try:
        asset_rows = conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id = ?",
            (args.condition_id,),
        ).fetchall()
    finally:
        conn.close()
    asset_ids = [r["asset_id"] for r in asset_rows]
    if not asset_ids:
        sys.stderr.write(f"error: no asset_index rows for {args.condition_id}\n")
        return 3

    common = {
        "ids": asset_ids,
        "tmin": str(_OVERLAP_MIN),
        "tmax": str(_OVERLAP_MAX),
        "first": args.per_side,
    }

    async with (
        SubgraphClient(url=_GATEWAY.format(key=api_key, id=_V1), rpm=60) as v1,
        SubgraphClient(url=_GATEWAY.format(key=api_key, id=_V2), rpm=60) as v2,
    ):
        v1_maker = (await v1.query(_V1_Q_MAKER, common)).get("orderFilledEvents") or []
        v1_taker = (await v1.query(_V1_Q_TAKER, common)).get("orderFilledEvents") or []
        v2_rows = (await v2.query(_V2_Q, common)).get("orderFilledEvents") or []

    v1_by_key: dict[tuple[str, str], dict] = {}
    for r in v1_maker + v1_taker:
        key = (r["transactionHash"].lower(), r["orderHash"].lower())
        v1_by_key.setdefault(key, r)

    by_tx_v2: dict[tuple[str, str], dict] = {
        (r["transactionHash"].lower(), r["orderHash"].lower()): r for r in v2_rows
    }

    matched: list[dict] = []
    for key, v1_row in v1_by_key.items():
        if key in by_tx_v2:
            matched.append({"v1": v1_row, "v2": by_tx_v2[key]})

    keep: list[dict] = []
    seen_sides: set[str] = set()
    for pair in matched:
        side = pair["v1"]["side"]
        if side not in seen_sides or len(keep) < 4:
            seen_sides.add(side)
            keep.append(pair)
        if len(keep) >= 8:
            break

    failures: list[str] = []
    for pair in keep:
        v1_row, v2_row = pair["v1"], pair["v2"]
        if int(v1_row["makerAmountFilled"]) != int(v2_row["makerAmountFilled"]):
            failures.append(
                f"makerAmount mismatch on {v1_row['transactionHash']}: "
                f"v1={v1_row['makerAmountFilled']} v2={v2_row['makerAmountFilled']}"
            )
        if int(v1_row["takerAmountFilled"]) != int(v2_row["takerAmountFilled"]):
            failures.append(
                f"takerAmount mismatch on {v1_row['transactionHash']}: "
                f"v1={v1_row['takerAmountFilled']} v2={v2_row['takerAmountFilled']}"
            )
        v1_buy = v1_row["makerAssetId"] == "0"
        v2_buy = int(v2_row["side"]) == 0
        if v1_buy != v2_buy:
            failures.append(
                f"side mismatch on {v1_row['transactionHash']}: "
                f"v1_buy={v1_buy} v2_buy={v2_buy}"
            )
    if failures:
        sys.stderr.write("VERIFICATION FAILED:\n  " + "\n  ".join(failures) + "\n")
        return 4

    out = {
        "condition_id": args.condition_id,
        "asset_index": {r["asset_id"]: r["outcome_side"] for r in asset_rows},
        "pairs": keep,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {args.output}: {len(keep)} matched pairs from {len(matched)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain(sys.argv[1:])))
```

- [ ] **Step 3: Run the verification script with a candidate condition_id from Step 1**

Run: `GRAPH_API_KEY=9e5231bfb63603ff576b3b0ce1b58913 uv run python scripts/verify_v1_units.py --condition-id <hex from Step 1>`
Expected: `wrote tests/corpus/fixtures/v1_v2_overlap.json: N matched pairs from M candidates` where `N >= 2` (at least one BUY pair and one SELL pair).

If `N < 2`: try a different candidate from Step 1's list. If no candidate yields ≥ 2 pairs, the overlap-window approach is dead and you should STOP and report BLOCKED.

If the script exits with `VERIFICATION FAILED:` output, the V1 and V2 amount fields are NOT identical — the spec's central assumption is wrong. STOP and report BLOCKED so the controller can re-open the design.

- [ ] **Step 4: Commit the script and fixture**

```bash
git add scripts/verify_v1_units.py tests/corpus/fixtures/v1_v2_overlap.json
git commit -m "$(cat <<'EOF'
feat(corpus): V1/V2 overlap reconciliation + ground-truth fixture (#193, Stage 1)

Verifies V1 makerAmountFilled / takerAmountFilled equal V2's for the
same (transactionHash, orderHash) keys, and V1 maker-zero pattern
matches V2 side==0. Downstream adapter unit tests assert against this
file forever.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Schema migration — `onchain_v1_processed_at` column

**Files:**
- Modify: `src/pscanner/corpus/db.py`
- Test: `tests/corpus/test_db.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/corpus/test_db.py`:

```python
def test_corpus_markets_has_onchain_v1_processed_at_column(tmp_path):
    from pscanner.corpus.db import init_corpus_db

    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(corpus_markets)").fetchall()}
        assert "onchain_v1_processed_at" in cols
    finally:
        conn.close()


def test_corpus_markets_migration_is_idempotent_for_v1_column(tmp_path):
    from pscanner.corpus.db import init_corpus_db

    db = tmp_path / "corpus.sqlite3"
    init_corpus_db(db).close()
    conn = init_corpus_db(db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(corpus_markets)").fetchall()}
        assert "onchain_v1_processed_at" in cols
    finally:
        conn.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_db.py::test_corpus_markets_has_onchain_v1_processed_at_column -v`
Expected: FAIL — column not yet declared.

- [ ] **Step 3: Add the column to the schema**

In `src/pscanner/corpus/db.py`, find the `corpus_markets` `CREATE TABLE` statement in `_SCHEMA_STATEMENTS` (around line 142–167). Add `onchain_v1_processed_at INTEGER,` between `onchain_processed_at INTEGER,` and `tags_json TEXT NOT NULL DEFAULT '[]',`:

```python
      onchain_trades_count INTEGER,
      onchain_processed_at INTEGER,
      onchain_v1_processed_at INTEGER,
      tags_json TEXT NOT NULL DEFAULT '[]',
```

- [ ] **Step 4: Add the migration**

In `src/pscanner/corpus/db.py`, append to `_MIGRATIONS`:

```python
    # V1 adapter sentinel (issue #193). Stamped by `run_v1_subgraph_backfill`
    # when a market's V1 pages have been successfully drained. Separate
    # column from `onchain_processed_at` so hybrid markets (V2 + V1 both
    # ran) carry both sentinels independently.
    "ALTER TABLE corpus_markets ADD COLUMN onchain_v1_processed_at INTEGER",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_db.py -v -k v1_processed`
Expected: PASS for both new tests.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/db.py tests/corpus/test_db.py
git commit -m "$(cat <<'EOF'
feat(corpus): add onchain_v1_processed_at column (#193)

Separate V1 adapter sentinel so hybrid markets carry both V1 and V2
processing timestamps independently. Idempotent ALTER TABLE migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: V1 adapter — `subgraph_v1_row_to_event`

**Files:**
- Create: `src/pscanner/corpus/subgraph_ingest_v1.py` (parser only)
- Create: `tests/corpus/test_subgraph_ingest_v1.py`

The adapter receives one `orderFilledEvents` row from the V1 subgraph
and produces an `OrderFilledEvent`. V1's `maker`/`taker` are flat hex,
`makerAssetId`/`takerAssetId` carry the asset ids (one is `"0"`),
amounts are `makerAmountFilled`/`takerAmountFilled` in 6-decimal base
units (identical to V2). V1's `side` and `price` fields are ignored —
the maker-POV BUY/SELL derivation is done downstream by
`event_to_corpus_trade`.

- [ ] **Step 1: Write the failing test that uses the overlap fixture**

```python
# tests/corpus/test_subgraph_ingest_v1.py
"""Tests for the V1 subgraph adapter (issue #193)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pscanner.corpus.subgraph_ingest_v1 import subgraph_v1_row_to_event

_FIXTURE = Path(__file__).parent / "fixtures" / "v1_v2_overlap.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def test_parser_matches_v2_amounts_on_overlap_fixture():
    data = _load_fixture()
    for pair in data["pairs"]:
        v1, v2 = pair["v1"], pair["v2"]
        event = subgraph_v1_row_to_event(v1)

        v2_side = int(v2["side"])
        v2_token = int(v2["tokenId"])
        if v2_side == 0:  # maker BUY: USDC -> CTF
            assert event.maker_asset_id == 0
            assert event.taker_asset_id == v2_token
        else:  # maker SELL: CTF -> USDC
            assert event.maker_asset_id == v2_token
            assert event.taker_asset_id == 0
        assert event.making == int(v2["makerAmountFilled"])
        assert event.taking == int(v2["takerAmountFilled"])
        assert event.maker == v2["maker"]["id"].lower()
        assert event.taker == v2["taker"]["id"].lower()
        assert event.tx_hash == v2["transactionHash"]
        assert event.order_hash == v2["orderHash"]


def test_parser_buy_row_maker_zero():
    row = {
        "id": "tx-1_hash-1",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "1" * 64,
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": "12345",
        "makerAmountFilled": "500000",
        "takerAmountFilled": "1000000",
        "fee": "0",
        "side": "buy",
        "price": "0.5",
    }
    event = subgraph_v1_row_to_event(row)
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 12345
    assert event.making == 500000
    assert event.taking == 1000000
    assert event.maker == "0x" + "b" * 40
    assert event.taker == "0x" + "c" * 40
    assert event.tx_hash == "0x" + "a" * 64
    assert event.order_hash == "0x" + "1" * 64


def test_parser_sell_row_taker_zero():
    row = {
        "id": "tx-2_hash-2",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "2" * 64,
        "maker": "0x" + "B" * 40,  # uppercase to verify normalization
        "taker": "0x" + "c" * 40,
        "makerAssetId": "67890",
        "takerAssetId": "0",
        "makerAmountFilled": "2000000",
        "takerAmountFilled": "600000",
        "fee": "5000",
        "side": "sell",
        "price": "0.3",
    }
    event = subgraph_v1_row_to_event(row)
    assert event.maker_asset_id == 67890
    assert event.taker_asset_id == 0
    assert event.making == 2000000
    assert event.taking == 600000
    assert event.maker == "0x" + "b" * 40  # normalized
    assert event.fee == 5000


def test_parser_rejects_both_zero():
    row = {
        "id": "tx-3_hash-3",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "3" * 64,
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": "0",
        "makerAmountFilled": "0",
        "takerAmountFilled": "0",
        "fee": "0",
        "side": "buy",
        "price": "0",
    }
    with pytest.raises(ValueError):
        subgraph_v1_row_to_event(row)


def test_parser_rejects_both_nonzero():
    row = {
        "id": "tx-4_hash-4",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "orderHash": "0x" + "4" * 64,
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "12345",
        "takerAssetId": "67890",
        "makerAmountFilled": "100",
        "takerAmountFilled": "100",
        "fee": "0",
        "side": "buy",
        "price": "1",
    }
    with pytest.raises(ValueError):
        subgraph_v1_row_to_event(row)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the adapter**

```python
# src/pscanner/corpus/subgraph_ingest_v1.py
"""V1 subgraph adapter for `corpus_trades` backfill (issue #193).

The V1 Polymarket Orderbook subgraph (`7fu2DWYK…`) emits the pre-#151
`OrderFilledEvent` schema: flat `maker`/`taker` hex addresses,
`makerAssetId`/`takerAssetId` (one is `"0"` for USDC), and
`makerAmountFilled`/`takerAmountFilled` in 6-decimal base units —
identical to V2's amount conventions. This module owns the V1-specific
query, paginator, and orchestrator. The adapter emits the same
`OrderFilledEvent` dataclass V2 produces so the downstream
`event_to_corpus_trade` insert path is shared verbatim.

Verified against `tests/corpus/fixtures/v1_v2_overlap.json` (Stage 1).
"""

from __future__ import annotations

from collections.abc import Mapping

from pscanner.poly.onchain import OrderFilledEvent


def _parse_int(key: str, raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} could not be parsed as int: {raw!r}") from exc
    raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")


def _parse_str(key: str, raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be str, got {type(raw).__name__}")
    return raw


def subgraph_v1_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one V1 `orderFilledEvents` row to the existing `OrderFilledEvent`.

    V1 stores `makerAssetId` and `takerAssetId` directly as decimal-string
    CTF token ids (one will be `"0"` for the USDC side). `maker` and
    `taker` are flat lowercase-able hex strings. Amount fields are in
    6-decimal base units, matching V2 exactly.

    V1's `side` (string) and `price` (decimal string) fields are present
    but ignored — the downstream `event_to_corpus_trade` derives maker-POV
    BUY/SELL from `(maker_asset_id, taker_asset_id)` alone.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list.

    Returns:
        ``OrderFilledEvent`` (block_number=0, log_index=0).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable, a string field has
            the wrong type, or both/neither asset id is zero.
    """
    maker_asset = _parse_int("makerAssetId", row["makerAssetId"])
    taker_asset = _parse_int("takerAssetId", row["takerAssetId"])
    if (maker_asset == 0) == (taker_asset == 0):
        raise ValueError(
            f"both-zero or both-non-zero asset ids: "
            f"maker={maker_asset}, taker={taker_asset}"
        )

    return OrderFilledEvent(
        order_hash=_parse_str("orderHash", row["orderHash"]),
        maker=_parse_str("maker", row["maker"]).lower(),
        taker=_parse_str("taker", row["taker"]).lower(),
        maker_asset_id=maker_asset,
        taker_asset_id=taker_asset,
        making=_parse_int("makerAmountFilled", row["makerAmountFilled"]),
        taking=_parse_int("takerAmountFilled", row["takerAmountFilled"]),
        fee=_parse_int("fee", row["fee"]),
        tx_hash=_parse_str("transactionHash", row["transactionHash"]),
        block_number=0,
        log_index=0,
    )
```

- [ ] **Step 4: Run the parser tests**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v`
Expected: All five parser tests PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py && uv run ruff format --check src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py && uv run ty check src/pscanner/corpus/subgraph_ingest_v1.py`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py
git commit -m "$(cat <<'EOF'
feat(corpus): V1 subgraph row -> OrderFilledEvent adapter (#193)

Parses V1 orderFilledEvents rows (flat addresses, makerAssetId +
takerAssetId, BigInt amounts) into the existing OrderFilledEvent
dataclass. Downstream event_to_corpus_trade derives BUY/SELL from
asset-id pair alone; V1's side/price fields ignored. Verified against
the Stage 1 overlap fixture.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: V1 paginator + iterator (two-pass)

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest_v1.py`
- Modify: `tests/corpus/test_subgraph_ingest_v1.py`

V1's schema has no `_or` operator, so per-market we paginate two
separate queries: one filtering on `makerAssetId_in` (catches BUY rows
where maker=USDC=0) and one on `takerAssetId_in` (catches SELL rows
where taker=USDC=0). The iterator emits the union. De-dupe is handled
by `corpus_trades`'s unique constraint downstream — but in practice
the two queries shouldn't overlap (a row has makerAssetId=0 XOR
takerAssetId=0 for valid CTF↔USDC fills).

- [ ] **Step 1: Write the failing paginator tests**

Append to `tests/corpus/test_subgraph_ingest_v1.py`:

```python
import pytest

from pscanner.corpus.subgraph_ingest_v1 import iter_v1_market_trades


class _FakeSubgraphClient:
    """Records every query() invocation and yields canned responses in order."""

    def __init__(self, pages_by_query: dict[str, list[list[dict]]]) -> None:
        self._pages = pages_by_query
        self.calls: list[dict] = []

    async def query(self, graphql: str, variables: dict) -> dict:
        side = "maker" if "makerAssetId_in" in graphql else "taker"
        self.calls.append({"side": side, "variables": dict(variables)})
        pages = self._pages.get(side, [])
        if not pages:
            return {"orderFilledEvents": []}
        return {"orderFilledEvents": pages.pop(0)}


def _buy_row(idx: int, asset_id: str) -> dict:
    return {
        "id": f"tx-{idx}_hash-{idx}",
        "transactionHash": "0x" + str(idx).rjust(64, "0"),
        "timestamp": str(1_700_000_000 + idx),
        "orderHash": "0x" + str(idx).rjust(64, "1"),
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": asset_id,
        "makerAmountFilled": "500000",
        "takerAmountFilled": "1000000",
        "fee": "0",
        "side": "buy",
        "price": "0.5",
    }


def _sell_row(idx: int, asset_id: str) -> dict:
    return {
        "id": f"tx-{idx}_hash-{idx}",
        "transactionHash": "0x" + str(idx).rjust(64, "0"),
        "timestamp": str(1_700_000_000 + idx),
        "orderHash": "0x" + str(idx).rjust(64, "1"),
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": asset_id,
        "takerAssetId": "0",
        "makerAmountFilled": "1000000",
        "takerAmountFilled": "300000",
        "fee": "0",
        "side": "sell",
        "price": "0.3",
    }


@pytest.mark.asyncio
async def test_paginator_returns_empty_on_no_rows():
    client = _FakeSubgraphClient(pages_by_query={})
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(
            client=client, asset_ids=["100"], page_size=2
        )
    ]
    assert out == []
    sides = {c["side"] for c in client.calls}
    assert sides == {"maker", "taker"}


@pytest.mark.asyncio
async def test_paginator_yields_union_of_buy_and_sell_passes():
    buys = [_buy_row(i, "100") for i in range(3)]
    sells = [_sell_row(10 + i, "100") for i in range(2)]
    client = _FakeSubgraphClient(pages_by_query={
        "maker": [buys],
        "taker": [sells],
    })
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(
            client=client, asset_ids=["100"], page_size=10
        )
    ]
    assert len(out) == 5
    assert all(ev.maker_asset_id == 0 for ev, _ in out[:3])
    assert all(ev.taker_asset_id == 0 for ev, _ in out[3:])


@pytest.mark.asyncio
async def test_paginator_advances_cursor_per_side():
    buys = [_buy_row(i, "100") for i in range(5)]
    client = _FakeSubgraphClient(pages_by_query={
        "maker": [buys[0:2], buys[2:4], buys[4:5]],
        "taker": [],
    })
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(
            client=client, asset_ids=["100"], page_size=2
        )
    ]
    assert len(out) == 5
    maker_calls = [c for c in client.calls if c["side"] == "maker"]
    assert len(maker_calls) == 3
    assert maker_calls[0]["variables"]["cursor"] == ""
    assert maker_calls[1]["variables"]["cursor"] == "tx-1_hash-1"
    assert maker_calls[2]["variables"]["cursor"] == "tx-3_hash-3"


@pytest.mark.asyncio
async def test_paginator_rejects_invalid_page_size():
    client = _FakeSubgraphClient(pages_by_query={})
    with pytest.raises(ValueError):
        async for _ in iter_v1_market_trades(
            client=client, asset_ids=["100"], page_size=0
        ):
            pass


@pytest.mark.asyncio
async def test_paginator_short_circuits_on_empty_asset_ids():
    client = _FakeSubgraphClient(pages_by_query={})
    out = [
        x
        async for x in iter_v1_market_trades(
            client=client, asset_ids=[], page_size=10
        )
    ]
    assert out == []
    assert client.calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v -k paginator`
Expected: FAIL — `iter_v1_market_trades` does not exist.

- [ ] **Step 3: Implement the paginator and iterator**

Append to `src/pscanner/corpus/subgraph_ingest_v1.py`:

```python
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any

from pscanner.poly.subgraph import SubgraphClient

_MAX_PAGE_SIZE = 1000

_V1_QUERY_MAKER_SIDE = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee
  }
}
"""

_V1_QUERY_TAKER_SIDE = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee
  }
}
"""


async def _paginate_v1_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from one V1 query (maker or taker side).

    Adapter exceptions (`ValueError` from `subgraph_v1_row_to_event`)
    are propagated — the orchestrator catches them per-row with counted
    skipping.
    """
    cursor = ""
    while True:
        result = await client.query(
            graphql,
            {"assets": list(asset_ids), "cursor": cursor, "first": page_size},
        )
        rows: list[dict[str, Any]] = result.get("orderFilledEvents") or []
        if not rows:
            return
        for row in rows:
            event = subgraph_v1_row_to_event(row)
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            return
        cursor = str(rows[-1]["id"])


async def iter_v1_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every V1 fill involving any of `asset_ids`.

    Two paginated passes: one on `makerAssetId_in` (catches BUY rows),
    one on `takerAssetId_in` (catches SELL rows). The two should be
    disjoint for valid CTF↔USDC fills; the downstream
    ``CorpusTradesRepo.insert_batch`` `INSERT OR IGNORE` absorbs any
    accidental overlap.

    Empty ``asset_ids`` short-circuits to an empty iterator (no query).
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return
    async for ev, ts in _paginate_v1_side(
        client=client,
        graphql=_V1_QUERY_MAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
    async for ev, ts in _paginate_v1_side(
        client=client,
        graphql=_V1_QUERY_TAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
```

- [ ] **Step 4: Run all paginator tests**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v`
Expected: All parser + paginator tests PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py && uv run ty check src/pscanner/corpus/subgraph_ingest_v1.py`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py
git commit -m "$(cat <<'EOF'
feat(corpus): V1 subgraph two-pass paginator + iterator (#193)

V1's schema has no _or operator, so per-market we paginate two queries:
makerAssetId_in catches BUY rows, takerAssetId_in catches SELL rows.
The iterator emits the union; downstream INSERT OR IGNORE absorbs any
accidental overlap.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: V1 orchestrator — queue load, per-market drain, sentinel write

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest_v1.py`
- Modify: `tests/corpus/test_subgraph_ingest_v1.py`

- [ ] **Step 1: Write the failing orchestrator tests**

Append to `tests/corpus/test_subgraph_ingest_v1.py`:

```python
import sqlite3
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    AssetIndexEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
)
from pscanner.corpus.subgraph_ingest_v1 import (
    V1SubgraphRunSummary,
    run_v1_subgraph_backfill,
)


def _seed_v1_pending_market(conn: sqlite3.Connection, condition_id: str, asset_id: str) -> None:
    market = CorpusMarket(
        condition_id=condition_id,
        event_slug="test-event",
        category=None,
        closed_at=1_700_000_000,
        total_volume_usd=50_000.0,
        enumerated_at=1_700_000_000,
        market_slug="test-market",
    )
    CorpusMarketsRepo(conn).insert_pending(market)
    conn.execute(
        "UPDATE corpus_markets SET v1_history_pending = 1 WHERE condition_id = ?",
        (condition_id,),
    )
    AssetIndexRepo(conn).upsert(
        AssetIndexEntry(asset_id=asset_id, condition_id=condition_id, outcome_side="YES", outcome_index=0)
    )
    conn.commit()


@pytest.mark.asyncio
async def test_orchestrator_drains_one_market_and_stamps_sentinel(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cid = "0x" + "a" * 64
    aid = "1234567890"
    _seed_v1_pending_market(conn, cid, aid)
    rows = [_buy_row(0, aid), _buy_row(1, aid)]
    client = _FakeSubgraphClient(pages_by_query={"maker": [rows], "taker": []})
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
    )
    assert summary.markets_processed == 1
    assert summary.markets_zero_events == 0
    assert summary.markets_failed == 0
    assert summary.events_decoded == 2
    assert summary.trades_inserted == 2
    row = conn.execute(
        "SELECT onchain_v1_processed_at, v1_history_pending FROM corpus_markets WHERE condition_id = ?",
        (cid,),
    ).fetchone()
    assert row["onchain_v1_processed_at"] == 1_700_000_999
    assert row["v1_history_pending"] == 0


@pytest.mark.asyncio
async def test_orchestrator_does_not_stamp_on_zero_events(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cid = "0x" + "b" * 64
    aid = "9999999999"
    _seed_v1_pending_market(conn, cid, aid)
    client = _FakeSubgraphClient(pages_by_query={})
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
    )
    assert summary.markets_zero_events == 1
    assert summary.markets_processed == 0
    row = conn.execute(
        "SELECT onchain_v1_processed_at, v1_history_pending FROM corpus_markets WHERE condition_id = ?",
        (cid,),
    ).fetchone()
    assert row["onchain_v1_processed_at"] is None
    assert row["v1_history_pending"] == 1


@pytest.mark.asyncio
async def test_orchestrator_skips_market_with_empty_asset_index(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cid = "0x" + "c" * 64
    aid = "5555555555"
    _seed_v1_pending_market(conn, cid, aid)
    conn.execute("DELETE FROM asset_index WHERE condition_id = ?", (cid,))
    conn.commit()
    client = _FakeSubgraphClient(pages_by_query={})
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
    )
    assert summary.markets_processed == 0
    assert summary.markets_failed == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_orchestrator_respects_limit(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    for i in range(3):
        _seed_v1_pending_market(conn, "0x" + chr(0x61 + i) * 64, f"100{i}")
    client = _FakeSubgraphClient(pages_by_query={"maker": [[_buy_row(0, "1000")]], "taker": []})
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=1, now_ts=1_700_000_999
    )
    assert summary.markets_processed + summary.markets_zero_events == 1
```

- [ ] **Step 2: Verify repo API names**

Before running the test, read `src/pscanner/corpus/repos.py` to confirm `AssetIndexEntry`, `AssetIndexRepo.upsert`, `CorpusMarket`, `CorpusMarketsRepo.insert_pending` exist with these exact names. If any differs, update the test imports/calls.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v -k orchestrator`
Expected: FAIL — `run_v1_subgraph_backfill` / `V1SubgraphRunSummary` do not exist.

- [ ] **Step 4: Implement the orchestrator**

Append to `src/pscanner/corpus/subgraph_ingest_v1.py`:

```python
import sqlite3
import time
from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade, CorpusTradesRepo
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True)
class V1SubgraphRunSummary:
    markets_processed: int
    markets_failed: int
    markets_zero_events: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    dups_dropped: int


@dataclass(frozen=True)
class _PendingV1Market:
    condition_id: str
    market_slug: str
    total_volume_usd: float


def _load_pending_v1_markets(
    conn: sqlite3.Connection, *, limit: int | None
) -> list[_PendingV1Market]:
    sql = """
        SELECT condition_id,
               COALESCE(market_slug, '') AS market_slug,
               total_volume_usd
        FROM corpus_markets
        WHERE platform = 'polymarket'
          AND v1_history_pending = 1
          AND onchain_v1_processed_at IS NULL
        ORDER BY total_volume_usd DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [
        _PendingV1Market(
            condition_id=r["condition_id"],
            market_slug=r["market_slug"],
            total_volume_usd=float(r["total_volume_usd"]),
        )
        for r in rows
    ]


def _load_asset_ids_for_market(conn: sqlite3.Connection, condition_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT asset_id FROM asset_index WHERE condition_id = ?",
        (condition_id,),
    ).fetchall()
    return [r["asset_id"] for r in rows]


def _mark_v1_processed(
    conn: sqlite3.Connection, condition_id: str, *, now_ts: int
) -> None:
    conn.execute(
        """
        UPDATE corpus_markets
        SET onchain_v1_processed_at = ?,
            v1_history_pending = 0
        WHERE platform = 'polymarket' AND condition_id = ?
        """,
        (now_ts, condition_id),
    )
    conn.commit()


async def _backfill_one_v1_market(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    condition_id: str,
    page_size: int,
) -> tuple[int, int, int, int, int]:
    asset_ids = _load_asset_ids_for_market(conn, condition_id)
    if not asset_ids:
        _LOG.warning("subgraph.v1.no_asset_index", condition_id=condition_id)
        return 0, 0, 0, 0, 0

    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    events_decoded = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    pending: list[CorpusTrade] = []
    async for event, ts in iter_v1_market_trades(
        client=client,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        events_decoded += 1
        try:
            trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=ts)
        except UnsupportedFill:
            skipped_unsupported += 1
            continue
        except UnresolvableAsset:
            skipped_unresolvable += 1
            continue
        if trade.condition_id != condition_id:
            continue
        pending.append(trade)
    if not pending:
        return events_decoded, 0, skipped_unsupported, skipped_unresolvable, 0
    inserted = trades_repo.insert_batch(pending)
    dups = len(pending) - inserted
    return events_decoded, inserted, skipped_unsupported, skipped_unresolvable, dups


async def run_v1_subgraph_backfill(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    page_size: int = _MAX_PAGE_SIZE,
    limit: int | None = None,
    now_ts: int | None = None,
) -> V1SubgraphRunSummary:
    pending = _load_pending_v1_markets(conn, limit=limit)
    _LOG.info("subgraph.v1.start", markets=len(pending))

    processed = 0
    failed = 0
    zero_events = 0
    total_events = 0
    total_inserted = 0
    total_unsupported = 0
    total_unresolvable = 0
    total_dups = 0

    for i, market in enumerate(pending, start=1):
        try:
            events, inserted, unsup, unres, dups = await _backfill_one_v1_market(
                conn=conn,
                client=client,
                condition_id=market.condition_id,
                page_size=page_size,
            )
        except Exception as exc:
            failed += 1
            _LOG.error(
                "subgraph.v1.market_failed",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                error=str(exc),
            )
            continue
        total_events += events
        total_inserted += inserted
        total_unsupported += unsup
        total_unresolvable += unres
        total_dups += dups
        if inserted == 0:
            zero_events += 1
            _LOG.info(
                "subgraph.v1.zero_events",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                slug=market.market_slug[:50],
            )
            continue
        _mark_v1_processed(conn, market.condition_id, now_ts=now_ts or int(time.time()))
        processed += 1
        _LOG.info(
            "subgraph.v1.market_complete",
            idx=i,
            of=len(pending),
            condition_id=market.condition_id[:14] + "...",
            slug=market.market_slug[:50],
            events_decoded=events,
            trades_inserted=inserted,
            dups_dropped=dups,
        )

    summary = V1SubgraphRunSummary(
        markets_processed=processed,
        markets_failed=failed,
        markets_zero_events=zero_events,
        events_decoded=total_events,
        trades_inserted=total_inserted,
        skipped_unsupported=total_unsupported,
        skipped_unresolvable=total_unresolvable,
        dups_dropped=total_dups,
    )
    _LOG.info("subgraph.v1.run_done", **summary.__dict__)
    return summary
```

- [ ] **Step 5: Run all V1 tests**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v`
Expected: All parser, paginator, and orchestrator tests PASS.

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py && uv run ty check src/pscanner/corpus/subgraph_ingest_v1.py`

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py
git commit -m "$(cat <<'EOF'
feat(corpus): V1 subgraph orchestrator with sentinel hygiene (#193)

Loads v1_history_pending queue, drains each market via the V1
paginator, inserts via the shared event_to_corpus_trade path, and
stamps onchain_v1_processed_at only when >=1 row was actually
inserted. Zero-event drains and asset_index gaps are logged but leave
both v1_history_pending and the sentinel untouched so re-runs are safe.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Dispatcher — V2 then V1 then clear-truncation

**Files:**
- Create: `src/pscanner/corpus/subgraph_dispatch.py`
- Create: `tests/corpus/test_subgraph_dispatch.py`

- [ ] **Step 1: Write the failing dispatcher tests**

```python
# tests/corpus/test_subgraph_dispatch.py
"""Tests for the V1+V2 subgraph dispatcher (issue #193)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.subgraph_dispatch import (
    DispatchedRunSummary,
    run_subgraph_backfill_dispatched,
)


class _NoopClient:
    async def query(self, graphql: str, variables: dict[str, Any]) -> dict:
        return {"orderFilledEvents": []}


@pytest.mark.asyncio
async def test_dispatcher_runs_both_versions_by_default(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    summary = await run_subgraph_backfill_dispatched(
        conn=conn,
        v1_client=_NoopClient(),
        v2_client=_NoopClient(),
        versions=("v2", "v1"),
    )
    assert isinstance(summary, DispatchedRunSummary)
    assert summary.v2_summary is not None
    assert summary.v1_summary is not None


@pytest.mark.asyncio
async def test_dispatcher_skips_v1_when_only_v2_requested(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    summary = await run_subgraph_backfill_dispatched(
        conn=conn,
        v1_client=_NoopClient(),
        v2_client=_NoopClient(),
        versions=("v2",),
    )
    assert summary.v2_summary is not None
    assert summary.v1_summary is None


@pytest.mark.asyncio
async def test_dispatcher_skips_v2_when_only_v1_requested(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    summary = await run_subgraph_backfill_dispatched(
        conn=conn,
        v1_client=_NoopClient(),
        v2_client=_NoopClient(),
        versions=("v1",),
    )
    assert summary.v2_summary is None
    assert summary.v1_summary is not None


@pytest.mark.asyncio
async def test_dispatcher_rejects_unknown_version(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    with pytest.raises(ValueError):
        await run_subgraph_backfill_dispatched(
            conn=conn,
            v1_client=_NoopClient(),
            v2_client=_NoopClient(),
            versions=("v3",),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_subgraph_dispatch.py -v`

- [ ] **Step 3: Implement the dispatcher**

```python
# src/pscanner/corpus/subgraph_dispatch.py
"""V1+V2 subgraph-backfill dispatcher (issue #193)."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import structlog

from pscanner.corpus.subgraph_ingest import (
    SubgraphRunSummary,
    _clear_truncation_flags,
    run_subgraph_backfill,
)
from pscanner.corpus.subgraph_ingest_v1 import V1SubgraphRunSummary, run_v1_subgraph_backfill
from pscanner.poly.subgraph import SubgraphClient

_LOG = structlog.get_logger(__name__)

SubgraphVersion = Literal["v1", "v2"]


@dataclass(frozen=True)
class DispatchedRunSummary:
    v2_summary: SubgraphRunSummary | None
    v1_summary: V1SubgraphRunSummary | None
    truncation_flags_cleared: int


async def run_subgraph_backfill_dispatched(
    *,
    conn: sqlite3.Connection,
    v1_client: SubgraphClient,
    v2_client: SubgraphClient,
    versions: Sequence[SubgraphVersion] = ("v2", "v1"),
    page_size: int = 1000,
    limit: int | None = None,
    truncation_threshold: int = 3000,
) -> DispatchedRunSummary:
    unknown = [v for v in versions if v not in ("v1", "v2")]
    if unknown:
        raise ValueError(f"unknown subgraph versions: {unknown}")

    v2_summary: SubgraphRunSummary | None = None
    v1_summary: V1SubgraphRunSummary | None = None

    for version in versions:
        if version == "v2":
            _LOG.info("subgraph.dispatch.v2_start")
            v2_summary = await run_subgraph_backfill(
                conn=conn,
                client=v2_client,
                page_size=page_size,
                limit=limit,
                truncation_threshold=truncation_threshold,
            )
        elif version == "v1":
            _LOG.info("subgraph.dispatch.v1_start")
            v1_summary = await run_v1_subgraph_backfill(
                conn=conn,
                client=v1_client,
                page_size=page_size,
                limit=limit,
            )

    cleared = _clear_truncation_flags(conn, threshold=truncation_threshold)
    summary = DispatchedRunSummary(
        v2_summary=v2_summary,
        v1_summary=v1_summary,
        truncation_flags_cleared=cleared,
    )
    _LOG.info("subgraph.dispatch.done", cleared=cleared)
    return summary
```

- [ ] **Step 4: Run the dispatcher tests**

Run: `uv run pytest tests/corpus/test_subgraph_dispatch.py -v`

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_dispatch.py tests/corpus/test_subgraph_dispatch.py && uv run ty check src/pscanner/corpus/subgraph_dispatch.py`

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/subgraph_dispatch.py tests/corpus/test_subgraph_dispatch.py
git commit -m "$(cat <<'EOF'
feat(corpus): V1+V2 subgraph backfill dispatcher (#193)

Drives V2 (existing) then V1 (new) backfills, then runs the shared
_clear_truncation_flags. Versions selectable via the `versions`
sequence; unknown values are rejected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CLI surface — version flag, deprecated alias

**Files:**
- Modify: `src/pscanner/corpus/cli.py`
- Modify: `tests/corpus/test_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/corpus/test_cli.py`:

```python
def test_subgraph_backfill_help_lists_version_flags():
    from pscanner.corpus.cli import _build_parser

    parser = _build_parser()
    sub_actions = [a for a in parser._actions if hasattr(a, "choices")]
    assert sub_actions, "no subparser found"
    subparsers = sub_actions[0].choices
    sg = subparsers["subgraph-backfill"]
    help_text = sg.format_help()
    assert "--subgraph-version" in help_text
    assert "--v1-subgraph-id" in help_text
    assert "--v2-subgraph-id" in help_text
    assert "--subgraph-id" in help_text  # deprecated alias preserved


def test_subgraph_backfill_subgraph_id_alias_maps_to_v2():
    from pscanner.corpus.cli import _build_parser, _resolve_subgraph_flags

    parser = _build_parser()
    args = parser.parse_args(
        ["subgraph-backfill", "--subgraph-id", "deprecated-id-value"]
    )
    resolved = _resolve_subgraph_flags(args)
    assert resolved.v2_subgraph_id == "deprecated-id-value"
```

If `_build_parser` doesn't exist as a module-level function, refactor argparse setup into one. Follow whatever pattern the existing CLI tests use.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_cli.py -v -k subgraph`

- [ ] **Step 3: Add the new CLI flags**

In `src/pscanner/corpus/cli.py`, near the top with the other constants (around line 65), add:

```python
_DEFAULT_V1_SUBGRAPH_ID = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
```

In the `subgraph-backfill` subparser block (around lines 149–186), replace the single `--subgraph-id` flag with this group (keep `--rpm`, `--page-size`, `--limit` unchanged):

```python
    sg.add_argument(
        "--subgraph-version",
        type=str,
        default="both",
        choices=("v1", "v2", "both"),
        help="Which subgraph(s) to backfill from (default: both).",
    )
    sg.add_argument(
        "--v2-subgraph-id",
        type=str,
        default=_DEFAULT_SUBGRAPH_ID,
        help=(
            "V2 subgraph deployment id (default: verified Polymarket "
            "Orderbook subgraph)."
        ),
    )
    sg.add_argument(
        "--v1-subgraph-id",
        type=str,
        default=_DEFAULT_V1_SUBGRAPH_ID,
        help=(
            "V1 subgraph deployment id (default: frozen Polymarket V1 "
            "Orderbook subgraph)."
        ),
    )
    sg.add_argument(
        "--subgraph-id",
        type=str,
        default=None,
        help=(
            "DEPRECATED: alias for --v2-subgraph-id. Will be removed in "
            "a future release."
        ),
    )
```

Add the `_resolve_subgraph_flags` helper near other helpers:

```python
@dataclass(frozen=True)
class _ResolvedSubgraphFlags:
    """Materialized subgraph flags after deprecation-alias resolution."""

    v2_subgraph_id: str
    v1_subgraph_id: str
    versions: tuple[str, ...]


def _resolve_subgraph_flags(args: argparse.Namespace) -> _ResolvedSubgraphFlags:
    """Apply the `--subgraph-id` deprecation alias and version selection."""
    v2_id = args.v2_subgraph_id
    if args.subgraph_id is not None:
        _log.warning(
            "subgraph.cli.deprecated_flag",
            flag="--subgraph-id",
            replacement="--v2-subgraph-id",
        )
        v2_id = args.subgraph_id
    if args.subgraph_version == "both":
        versions: tuple[str, ...] = ("v2", "v1")
    else:
        versions = (args.subgraph_version,)
    return _ResolvedSubgraphFlags(
        v2_subgraph_id=v2_id,
        v1_subgraph_id=args.v1_subgraph_id,
        versions=versions,
    )
```

Make sure `from dataclasses import dataclass` and `import argparse` are imported.

- [ ] **Step 4: Rewire `_cmd_subgraph_backfill` to use the dispatcher**

Replace the existing `_cmd_subgraph_backfill` body (around line 627–654) with:

```python
async def _cmd_subgraph_backfill(args: argparse.Namespace) -> int:
    """Run the subgraph-driven per-market backfill (V1+V2 dispatcher)."""
    from pscanner.corpus.subgraph_dispatch import run_subgraph_backfill_dispatched

    api_key = args.api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        raise SystemExit("subgraph-backfill requires --api-key or $GRAPH_API_KEY")
    resolved = _resolve_subgraph_flags(args)
    v2_url = _GATEWAY_URL_TEMPLATE.format(api_key=api_key, subgraph_id=resolved.v2_subgraph_id)
    v1_url = _GATEWAY_URL_TEMPLATE.format(api_key=api_key, subgraph_id=resolved.v1_subgraph_id)
    conn = init_corpus_db(Path(args.db))
    try:
        async with (
            SubgraphClient(url=v2_url, rpm=args.rpm) as v2_client,
            SubgraphClient(url=v1_url, rpm=args.rpm) as v1_client,
        ):
            summary = await run_subgraph_backfill_dispatched(
                conn=conn,
                v1_client=v1_client,
                v2_client=v2_client,
                versions=resolved.versions,
                page_size=args.page_size,
                limit=args.limit,
            )
        _log.info(
            "subgraph.cli_summary",
            v2_markets_processed=(summary.v2_summary.markets_processed if summary.v2_summary else None),
            v2_trades_inserted=(summary.v2_summary.trades_inserted if summary.v2_summary else None),
            v1_markets_processed=(summary.v1_summary.markets_processed if summary.v1_summary else None),
            v1_trades_inserted=(summary.v1_summary.trades_inserted if summary.v1_summary else None),
            v1_markets_zero_events=(summary.v1_summary.markets_zero_events if summary.v1_summary else None),
            v1_dups_dropped=(summary.v1_summary.dups_dropped if summary.v1_summary else None),
            truncation_flags_cleared=summary.truncation_flags_cleared,
        )
        return 0
    finally:
        conn.close()
```

- [ ] **Step 5: Run the CLI tests**

Run: `uv run pytest tests/corpus/test_cli.py -v -k subgraph`

- [ ] **Step 6: Run the full corpus suite to confirm no regressions**

Run: `uv run pytest tests/corpus -v`

- [ ] **Step 7: Lint + type-check the whole CLI file**

Run: `uv run ruff check src/pscanner/corpus/cli.py && uv run ruff format --check src/pscanner/corpus/cli.py && uv run ty check src/pscanner/corpus/cli.py`

- [ ] **Step 8: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "$(cat <<'EOF'
feat(corpus): wire V1+V2 dispatcher into subgraph-backfill CLI (#193)

Adds --subgraph-version, --v1-subgraph-id, --v2-subgraph-id flags.
Preserves --subgraph-id as a deprecated alias for --v2-subgraph-id;
using it emits one subgraph.cli.deprecated_flag warning.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Hybrid-market integration test

**Files:**
- Modify: `tests/corpus/test_subgraph_ingest_v1.py`

- [ ] **Step 1: Write the integration test**

Append:

```python
@pytest.mark.asyncio
async def test_hybrid_market_sets_both_sentinels(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cid = "0x" + "f" * 64
    aid = "7777777777"
    _seed_v1_pending_market(conn, cid, aid)
    conn.execute(
        "UPDATE corpus_markets SET onchain_processed_at = ? WHERE condition_id = ?",
        (1_600_000_000, cid),
    )
    conn.commit()

    rows = [_buy_row(0, aid)]
    client = _FakeSubgraphClient(pages_by_query={"maker": [rows], "taker": []})
    await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
    )

    row = conn.execute(
        """
        SELECT onchain_processed_at, onchain_v1_processed_at, v1_history_pending
        FROM corpus_markets WHERE condition_id = ?
        """,
        (cid,),
    ).fetchone()
    assert row["onchain_processed_at"] == 1_600_000_000
    assert row["onchain_v1_processed_at"] == 1_700_000_999
    assert row["v1_history_pending"] == 0
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py::test_hybrid_market_sets_both_sentinels -v`

- [ ] **Step 3: Commit**

```bash
git add tests/corpus/test_subgraph_ingest_v1.py
git commit -m "$(cat <<'EOF'
test(corpus): hybrid-market V1+V2 sentinel hygiene (#193)

Confirms a V2-already-processed market accepts a V1 pass without
clobbering its V2 sentinel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Production smoke run + CLAUDE.md update

- [ ] **Step 1: Pre-flight check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`

- [ ] **Step 2: V1-only smoke, small limit**

```bash
GRAPH_API_KEY=9e5231bfb63603ff576b3b0ce1b58913 uv run pscanner corpus subgraph-backfill \
    --db /home/macph/projects/polymarketScanner/data/corpus.sqlite3 \
    --subgraph-version v1 \
    --rpm 50 \
    --limit 5 2>&1 | tee /tmp/v1-smoke.log
```

- [ ] **Step 3: Verify the production DB shows the sentinel + cleared flag**

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('/home/macph/projects/polymarketScanner/data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
rows = conn.execute('''
  SELECT condition_id, onchain_v1_processed_at, v1_history_pending, market_slug
  FROM corpus_markets
  WHERE onchain_v1_processed_at IS NOT NULL
  ORDER BY onchain_v1_processed_at DESC LIMIT 10
''').fetchall()
for r in rows: print(dict(r))
"
```

- [ ] **Step 4: Spot-check a processed market's trade-count delta**

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('/home/macph/projects/polymarketScanner/data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
cid = '<paste condition_id>'
n = conn.execute('SELECT COUNT(*) FROM corpus_trades WHERE condition_id = ?', (cid,)).fetchone()[0]
min_ts = conn.execute('SELECT MIN(ts) FROM corpus_trades WHERE condition_id = ?', (cid,)).fetchone()[0]
print(f'trade_count={n} oldest_ts={min_ts}')
"
```

- [ ] **Step 5: Update CLAUDE.md**

Edit `CLAUDE.md` to:
- Correct the V1 schema description: the V1 subgraph still serves the *pre-#151* `OrderFilledEvent` schema (flat addresses, `makerAssetId` + `takerAssetId`, same amount fields). The "re-pushed with an entirely different schema" claim was wrong.
- Move the entry currently under "Tracked work in flight" → #193 into "Polymarket API quirks" as a shipped-state note.

- [ ] **Step 6: Commit the doc update**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: record V1 subgraph adapter shipped (#193)

Correct the V1 schema description in CLAUDE.md (V1 still serves the
pre-#151 OrderFilledEvent shape, not a re-pushed different schema).
Move V1 adapter notes from "Tracked work in flight" into the Polymarket
API quirks section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Run the full smoke on the production V1 queue (operator-optional)**

```bash
GRAPH_API_KEY=9e5231bfb63603ff576b3b0ce1b58913 uv run pscanner corpus subgraph-backfill \
    --db /home/macph/projects/polymarketScanner/data/corpus.sqlite3 \
    --subgraph-version v1 \
    --rpm 50 2>&1 | tee /tmp/v1-full.log
```

---

## Self-Review

**Spec coverage:**
- Stage 0 investigation → Task 1 ✓ (done)
- Stage 1 amount-equality verification + fixture → Task 2 ✓
- `onchain_v1_processed_at` column migration → Task 3 ✓
- V1 adapter parser → Task 4 ✓
- V1 two-pass paginator + iterator → Task 5 ✓
- V1 orchestrator with sentinel hygiene → Task 6 ✓
- Dispatcher → Task 7 ✓
- CLI flags + deprecated alias → Task 8 ✓
- Hybrid-market integration test → Task 9 ✓
- Production smoke + doc update → Task 10 ✓

**Placeholder scan:** No "TBD" / "TODO" — all code shown for code steps.

**Type consistency:**
- `V1SubgraphRunSummary` field set is used identically across Tasks 6, 7, and 8.
- `subgraph_v1_row_to_event(row)` signature consistent across Tasks 4, 5, and 6.
- `iter_v1_market_trades(*, client, asset_ids, page_size)` consistent across Tasks 5 and 6.
- `run_v1_subgraph_backfill(*, conn, client, page_size, limit, now_ts)` consistent across Tasks 6, 7, and 9.
- `_FakeSubgraphClient(pages_by_query: dict[str, list[list[dict]]])` consistent across Tasks 5, 6, and 9.

---

## Revision history

- 2026-05-26 v1: assumed V1 had a re-pushed `OrderFill` schema.
- 2026-05-26 v2 (current): rewritten after Task 1's investigation
  proved the V1 subgraph still emits the *pre-#151* `OrderFilledEvent`
  shape. Dropped `marketId="0"` cohort scope; replaced
  `price × size / 1e6` unit derivation with direct amount field reads;
  switched paginator from one `marketId_in` query to two
  `makerAssetId_in` / `takerAssetId_in` queries.
