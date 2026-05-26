# V1 Subgraph Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a V1-subgraph adapter that fills `corpus_trades` for the 2,769 markets carrying `corpus_markets.v1_history_pending = 1`, then wire a dispatcher that drives the V1 and V2 backfills independently from one `pscanner corpus subgraph-backfill` invocation.

**Architecture:** Two pre-implementation scripts (investigation + overlap-window verification) lock the V1 schema semantics and economic-units mapping into a committed fixture. A new `subgraph_ingest_v1.py` module mirrors the V2 module's shape (paginator + orchestrator) and emits the existing `OrderFilledEvent` dataclass so the downstream `event_to_corpus_trade` path is shared with V2 unchanged. A thin `subgraph_dispatch.py` runs V2 first, then V1, then the shared `_clear_truncation_flags`. The new `onchain_v1_processed_at` column is the V1-side sentinel.

**Tech Stack:** Python 3.13 + uv + ruff + ty + pytest · `pscanner.poly.subgraph.SubgraphClient` for GraphQL · `sqlite3` + `pscanner.corpus.db.init_corpus_db` for storage · `structlog` for events.

**Spec reference:** `docs/superpowers/specs/2026-05-26-issue-193-v1-subgraph-adapter-design.md`

**Recycled scaffolding (read but rewrite the adapter):** `git show a809378^:src/pscanner/corpus/subgraph_ingest.py` — pre-#151 module. Paginator, orchestrator, and CLI shape lift cleanly; the adapter and query string must be rewritten for the *current* V1 schema (`orderFills` entity, `marketId` filter, BigInt `price`×`size`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `scripts/investigate_v1_schema.py` | Create | Stage 0: pull ~1000 V1 rows, group by `marketId` format, report recovery rate for each cohort. |
| `scripts/v1_investigation_report.md` | Create | Stage 0: markdown report from the script's output. Committed evidence. |
| `scripts/verify_v1_units.py` | Create | Stage 1: pick an overlap-window market, reconcile V1 vs V2 row-by-row, emit `tests/corpus/fixtures/v1_v2_overlap.json`. |
| `tests/corpus/fixtures/v1_v2_overlap.json` | Create | Committed ground-truth fixture: ~5 paired V1+V2 rows for the same fills. Frozen post Stage 1. |
| `src/pscanner/corpus/db.py` | Modify | Add `onchain_v1_processed_at INTEGER` to `_SCHEMA_STATEMENTS` and `_MIGRATIONS`. |
| `src/pscanner/corpus/subgraph_ingest_v1.py` | Create | V1 GraphQL query string, `subgraph_v1_row_to_event` adapter, `_paginate_v1`, `iter_v1_market_trades`, `_load_pending_v1_markets`, `_mark_v1_processed`, `_backfill_one_v1_market`, `run_v1_subgraph_backfill`, `V1SubgraphRunSummary`. |
| `src/pscanner/corpus/subgraph_dispatch.py` | Create | `run_subgraph_backfill_dispatched(...)` — runs V2 first, then V1, then `_clear_truncation_flags`. |
| `src/pscanner/corpus/cli.py` | Modify | Add `--subgraph-version`, `--v1-subgraph-id`, `--v2-subgraph-id` flags; preserve `--subgraph-id` as a deprecated alias that emits `subgraph.cli.deprecated_flag`. Replace the direct `run_subgraph_backfill` call with `run_subgraph_backfill_dispatched`. |
| `tests/corpus/test_subgraph_ingest_v1.py` | Create | Adapter unit tests (against the fixture), orchestrator sentinel-hygiene tests, hybrid-market integration. |
| `tests/corpus/test_subgraph_dispatch.py` | Create | Dispatcher routing tests + CLI flag tests. |
| `CLAUDE.md` | Modify | Append the V1 adapter to "Polymarket API quirks" (drop the "tracked work in flight" entry for #193). |

---

## Task 1: Stage 0 — V1 schema investigation script

**Files:**
- Create: `scripts/investigate_v1_schema.py`
- Create: `scripts/v1_investigation_report.md` (output)

This is a research task, not TDD. The script's *output* — the report — is what informs whether to file a follow-up for `marketId="0"` recovery.

- [ ] **Step 1: Write the investigation script**

```python
# scripts/investigate_v1_schema.py
"""Stage 0 of issue #193: probe the V1 subgraph's schema reality.

Pulls ~1000 recent rows from the V1 Orderbook subgraph
(`7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`), groups them by
`marketId` format, and reports per-cohort recovery rate against the
local `asset_index`. The script is single-shot and produces
`scripts/v1_investigation_report.md` as its only side effect.

Run:
    GRAPH_API_KEY=... uv run python scripts/investigate_v1_schema.py
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.poly.subgraph import SubgraphClient

_V1_SUBGRAPH_ID = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
_GATEWAY = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{id}"
_PROBE_QUERY = """
query($first: Int!) {
  orderFills(first: $first, orderBy: timestamp, orderDirection: desc) {
    id transactionHash timestamp
    marketId outcomeIndex
    maker taker price size fee
    order { id marketId outcomeIndex side }
  }
}
"""


def _format_marketid(value: str) -> str:
    if value == "0":
        return "zero"
    if value.startswith("0-"):
        return "zero-prefix"
    return "bare-decimal"


async def _probe(api_key: str, db_path: Path, sample_size: int) -> str:
    url = _GATEWAY.format(key=api_key, id=_V1_SUBGRAPH_ID)
    async with SubgraphClient(url=url, rpm=60) as client:
        result = await client.query(_PROBE_QUERY, {"first": sample_size})
    rows: list[dict] = result.get("orderFills") or []
    if not rows:
        return f"# V1 schema probe\n\nNo rows returned. Subgraph may be frozen or empty.\n"

    by_format: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_format[_format_marketid(row["marketId"])].append(row)

    conn = init_corpus_db(db_path)
    try:
        known_asset_ids: set[str] = {
            r["asset_id"]
            for r in conn.execute("SELECT asset_id FROM asset_index").fetchall()
        }
    finally:
        conn.close()

    lines: list[str] = [
        "# V1 schema investigation report",
        "",
        f"Subgraph: `{_V1_SUBGRAPH_ID}` (V1, frozen)",
        f"Sample size: {len(rows)} rows (most recent, desc by timestamp)",
        "",
        "## marketId format distribution",
        "",
        "| format | count | pct |",
        "|---|---|---|",
    ]
    for fmt, bucket in sorted(by_format.items(), key=lambda kv: -len(kv[1])):
        pct = 100.0 * len(bucket) / len(rows)
        lines.append(f"| {fmt} | {len(bucket)} | {pct:.1f}% |")
    lines.append("")

    for fmt in ("bare-decimal", "zero-prefix"):
        bucket = by_format.get(fmt, [])
        if not bucket:
            continue
        hits = 0
        for row in bucket:
            mid = row["marketId"]
            asset_id = mid[2:] if mid.startswith("0-") else mid
            if asset_id in known_asset_ids:
                hits += 1
        pct = 100.0 * hits / len(bucket) if bucket else 0.0
        lines += [
            f"## `{fmt}` cohort: asset_index recovery rate",
            "",
            f"- Rows in cohort: {len(bucket)}",
            f"- Resolved via `asset_index`: {hits} ({pct:.1f}%)",
            "",
        ]

    zero_bucket = by_format.get("zero", [])
    if zero_bucket:
        lines += [
            "## `marketId=\"0\"` cohort: Order-parent inspection",
            "",
            "Sample of 10 rows' `order.marketId` values (raw, not resolved):",
            "",
        ]
        for row in zero_bucket[:10]:
            order = row.get("order") or {}
            lines.append(f"- tx={row['transactionHash'][:14]}… order.marketId={order.get('marketId')!r}")
        lines.append("")
        # Quick attempt: do the order.marketId values map to asset_index?
        order_hits = 0
        for row in zero_bucket:
            order = row.get("order") or {}
            omid = str(order.get("marketId") or "")
            asset_id = omid[2:] if omid.startswith("0-") else omid
            if asset_id and asset_id in known_asset_ids:
                order_hits += 1
        opct = 100.0 * order_hits / len(zero_bucket) if zero_bucket else 0.0
        lines += [
            f"`order.marketId` resolves to `asset_index` for {order_hits}/{len(zero_bucket)} ({opct:.1f}%).",
            "",
            "**Decision:** if recovery rate > 50%, file a follow-up issue "
            "to extend Stage 2 with a third query pass via `order.marketId`. "
            "If < 50%, accept the gap and document.",
            "",
        ]

    return "\n".join(lines)


async def _amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="V1 subgraph schema probe (issue #193, Stage 0)")
    parser.add_argument("--db", default="data/corpus.sqlite3")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default="scripts/v1_investigation_report.md")
    args = parser.parse_args(argv)
    api_key = args.api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        sys.stderr.write("error: --api-key or $GRAPH_API_KEY required\n")
        return 2
    report = await _probe(api_key, Path(args.db), args.sample_size)
    Path(args.output).write_text(report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain(sys.argv[1:])))
```

- [ ] **Step 2: Run the script against the production corpus**

Run: `GRAPH_API_KEY=... uv run python scripts/investigate_v1_schema.py --sample-size 1000`
Expected: `wrote scripts/v1_investigation_report.md`. No exceptions.

- [ ] **Step 3: Review the report and decide on `marketId="0"` follow-up**

Read the report. If `order.marketId` resolves above 50% recovery in the zero cohort, file a follow-up GitHub issue titled `V1 adapter: recover marketId="0" rows via Order parent reference (follow-up to #193)`. If below 50%, document the gap inline in the spec and move on.

The Stage 2 adapter does NOT query the `marketId="0"` cohort regardless of this decision.

- [ ] **Step 4: Commit**

```bash
git add scripts/investigate_v1_schema.py scripts/v1_investigation_report.md
git commit -m "$(cat <<'EOF'
feat(corpus): V1 schema investigation script (#193, Stage 0)

Probe the V1 subgraph's marketId format distribution and report
per-cohort recovery rate against the local asset_index. Output drives
the decision on whether marketId="0" recovery is feasible as a
follow-up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Stage 1 — Overlap-window verification script + fixture

**Files:**
- Create: `scripts/verify_v1_units.py`
- Create: `tests/corpus/fixtures/v1_v2_overlap.json` (script output, committed)

Stage 1 proves that `V1.price * V1.size / 1e6 == V2.collateralAmount / 1e6` and locks in the `outcomeIndex + order.side → maker_asset_id / taker_asset_id` mapping. The committed fixture is what every adapter unit test in later tasks asserts against.

- [ ] **Step 1: Identify an overlap-window candidate market**

Run this SQL to pick a candidate (one that traded during 2026-04-03 → 2026-04-28 in V2, AND existed in V1):

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
row = conn.execute('''
  SELECT m.condition_id, m.market_slug, COUNT(t.tx_hash) AS trades, MIN(t.ts) AS first_ts, MAX(t.ts) AS last_ts
  FROM corpus_markets m JOIN corpus_trades t USING (condition_id)
  WHERE m.platform = 'polymarket'
    AND m.v1_history_pending = 1
    AND m.onchain_processed_at IS NOT NULL
  GROUP BY m.condition_id
  HAVING first_ts < 1775220779 AND last_ts > 1775220779
  ORDER BY trades DESC LIMIT 1
''').fetchone()
print(dict(row) if row else 'no overlap-window market found')
"
```

Expected: one `condition_id` printed. Record the value; the script will receive it as `--condition-id`.

- [ ] **Step 2: Write the verification script**

```python
# scripts/verify_v1_units.py
"""Stage 1 of issue #193: verify V1 economic units against V2.

Reconciles V1 `orderFills` and V2 `orderFilledEvents` for one shared
condition_id, row-by-row on (transactionHash, maker). Emits the matched
pairs as `tests/corpus/fixtures/v1_v2_overlap.json` so the V1 adapter
unit tests have ground truth.

Run:
    GRAPH_API_KEY=... uv run python scripts/verify_v1_units.py \\
        --condition-id <hex> --db data/corpus.sqlite3
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

_V1_Q = """
query($ids: [String!]!, $first: Int!) {
  orderFills(
    where: { marketId_in: $ids }
    first: $first orderBy: timestamp orderDirection: asc
  ) {
    id transactionHash timestamp
    marketId outcomeIndex
    maker taker price size fee
    order { id marketId outcomeIndex side }
  }
}
"""

_V2_Q = """
query($ids: [String!]!, $first: Int!) {
  orderFilledEvents(
    where: { market_in: $ids }
    first: $first orderBy: timestamp orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker { id } taker { id } market { id }
    tokenId side makerAmountFilled takerAmountFilled fee
  }
}
"""


async def _amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="V1/V2 overlap reconciliation (issue #193, Stage 1)")
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--db", default="data/corpus.sqlite3")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--output",
        default="tests/corpus/fixtures/v1_v2_overlap.json",
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
    # V1 candidates: bare and "0-"-prefixed forms.
    v1_ids = asset_ids + [f"0-{aid}" for aid in asset_ids]

    async with (
        SubgraphClient(url=_GATEWAY.format(key=api_key, id=_V1), rpm=60) as v1,
        SubgraphClient(url=_GATEWAY.format(key=api_key, id=_V2), rpm=60) as v2,
    ):
        v1_rows = (await v1.query(_V1_Q, {"ids": v1_ids, "first": args.per_side})).get("orderFills") or []
        v2_rows = (await v2.query(_V2_Q, {"ids": asset_ids, "first": args.per_side})).get("orderFilledEvents") or []

    by_tx_v2: dict[tuple[str, str], dict] = {}
    for r in v2_rows:
        by_tx_v2[(r["transactionHash"].lower(), r["maker"]["id"].lower())] = r

    matched: list[dict] = []
    for r in v1_rows:
        key = (r["transactionHash"].lower(), r["maker"].lower())
        if key in by_tx_v2:
            matched.append({"v1": r, "v2": by_tx_v2[key]})

    # Compact down to ~5 representative pairs spanning both outcomes & both sides.
    keep: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for pair in matched:
        v1, v2 = pair["v1"], pair["v2"]
        key = (int(v1["outcomeIndex"]), int(v1["order"]["side"]))
        if key not in seen:
            seen.add(key)
            keep.append(pair)
        if len(keep) >= 8:
            break

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

- [ ] **Step 3: Run the verification script with the candidate condition_id from Step 1**

Run: `GRAPH_API_KEY=... uv run python scripts/verify_v1_units.py --condition-id <hex from Step 1>`
Expected: `wrote tests/corpus/fixtures/v1_v2_overlap.json: N matched pairs from M candidates` where `N >= 4` (at least 4 paired rows covering the outcome × side cross-product, ideally all 4 combinations).

If `N < 4`: pick a different candidate market in Step 1 (re-run the SQL with `ORDER BY trades DESC LIMIT 5` and try each). If no candidate yields ≥4 pairs, the spec's "overlap-window verification" approach is invalidated; STOP and reopen the spec.

- [ ] **Step 4: Manually verify one pair by hand and lock the formula**

Open `tests/corpus/fixtures/v1_v2_overlap.json` and pick the first pair. Compute by hand:
- `usdc_amount_v1 = int(v1.price) * int(v1.size) / 1_000_000`
- `usdc_amount_v2 = int(v2.makerAmountFilled)` (if v2.side==0/BUY) or `int(v2.takerAmountFilled)` (if v2.side==1/SELL)

If they match within $0.01 (1e4 base units), the units formula is `price * size / 1e6`. Annotate the JSON file's top-level with a `"verified_at": "<ISO date>"` field so the test fixture is dated.

If they DON'T match, the spec is wrong. Re-open the design — do not proceed to Task 3.

- [ ] **Step 5: Commit the script and fixture**

```bash
git add scripts/verify_v1_units.py tests/corpus/fixtures/v1_v2_overlap.json
git commit -m "$(cat <<'EOF'
feat(corpus): V1/V2 overlap reconciliation + ground-truth fixture (#193, Stage 1)

Locks the V1 economic-units formula (price * size / 1e6 == V2
collateralAmount) and the (outcomeIndex, order.side) -> maker/taker
asset_id mapping via a hand-verified fixture. Downstream adapter unit
tests assert against this file.

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

Add to `tests/corpus/test_db.py` (or create if absent):

```python
def test_corpus_markets_has_onchain_v1_processed_at_column(tmp_path):
    from pathlib import Path
    from pscanner.corpus.db import init_corpus_db

    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(corpus_markets)").fetchall()}
        assert "onchain_v1_processed_at" in cols
    finally:
        conn.close()


def test_corpus_markets_migration_is_idempotent_for_v1_column(tmp_path):
    from pathlib import Path
    from pscanner.corpus.db import init_corpus_db

    db = tmp_path / "corpus.sqlite3"
    init_corpus_db(db).close()
    # Re-run; must not raise.
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

In `src/pscanner/corpus/db.py`, append to `_MIGRATIONS` (after the `v1_history_pending` entry — at the bottom of the tuple, around line 342):

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
- Create: `src/pscanner/corpus/subgraph_ingest_v1.py` (parser only — paginator and orchestrator land in later tasks)
- Create: `tests/corpus/test_subgraph_ingest_v1.py`

The adapter receives one `orderFills` row and produces an `OrderFilledEvent`. The unit test asserts against `tests/corpus/fixtures/v1_v2_overlap.json` so the (outcomeIndex, order.side) → maker/taker mapping is locked to ground truth.

- [ ] **Step 1: Write the failing test that uses the overlap fixture**

```python
# tests/corpus/test_subgraph_ingest_v1.py
"""Tests for the V1 subgraph adapter (issue #193)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pscanner.corpus.subgraph_ingest_v1 import (
    UnsupportedMarketIdFormat,
    subgraph_v1_row_to_event,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "v1_v2_overlap.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text())


def test_parser_matches_v2_amounts_on_overlap_fixture():
    data = _load_fixture()
    asset_index: dict[str, str] = data["asset_index"]
    for pair in data["pairs"]:
        v1, v2 = pair["v1"], pair["v2"]
        event = subgraph_v1_row_to_event(v1, asset_index=asset_index)

        # V2 has explicit makerAmountFilled / takerAmountFilled / tokenId / side.
        # Our derived (maker_asset_id, taker_asset_id, making, taking) must
        # match V2's exactly when the underlying fill is the same.
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


def test_parser_resolves_bare_marketid():
    row = {
        "id": "0xdead-1",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "marketId": "12345",
        "outcomeIndex": "0",
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "price": "500000",          # 0.5
        "size": "2000000",          # 2.0 contracts
        "fee": "0",
        "order": {"id": "0x1", "marketId": "12345", "outcomeIndex": "0", "side": "0"},
    }
    asset_index = {"12345": "YES"}
    event = subgraph_v1_row_to_event(row, asset_index=asset_index)
    # Maker BUY (side=0): USDC -> CTF. making is USDC base units; taking is CTF base units.
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 12345
    # USDC base units = price * size / 1e6 = 500000 * 2000000 / 1e6 = 1_000_000 (= $1.00).
    assert event.making == 1_000_000
    assert event.taking == 2_000_000


def test_parser_strips_zero_prefix_marketid():
    row = {
        "id": "0xdead-2",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "marketId": "0-67890",
        "outcomeIndex": "1",
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "price": "300000",  # 0.3
        "size": "5000000",  # 5.0 contracts
        "fee": "0",
        "order": {"id": "0x2", "marketId": "0-67890", "outcomeIndex": "1", "side": "1"},
    }
    asset_index = {"67890": "NO"}
    event = subgraph_v1_row_to_event(row, asset_index=asset_index)
    # Maker SELL (side=1): CTF -> USDC.
    assert event.maker_asset_id == 67890
    assert event.taker_asset_id == 0
    assert event.making == 5_000_000   # CTF base units
    assert event.taking == 1_500_000   # USDC base units (0.3 * 5)


def test_parser_rejects_zero_marketid():
    row = {
        "id": "0xdead-3",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "marketId": "0",
        "outcomeIndex": "0",
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "price": "500000",
        "size": "2000000",
        "fee": "0",
        "order": {"id": "0x3", "marketId": "0", "outcomeIndex": "0", "side": "0"},
    }
    with pytest.raises(UnsupportedMarketIdFormat):
        subgraph_v1_row_to_event(row, asset_index={})


def test_parser_rejects_unknown_asset_id():
    row = {
        "id": "0xdead-4",
        "transactionHash": "0x" + "a" * 64,
        "timestamp": "1770000000",
        "marketId": "99999",
        "outcomeIndex": "0",
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "price": "500000",
        "size": "2000000",
        "fee": "0",
        "order": {"id": "0x4", "marketId": "99999", "outcomeIndex": "0", "side": "0"},
    }
    # Empty asset_index — the adapter should raise so the orchestrator can count it.
    with pytest.raises(KeyError):
        subgraph_v1_row_to_event(row, asset_index={})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the adapter**

```python
# src/pscanner/corpus/subgraph_ingest_v1.py
"""V1 subgraph adapter for `corpus_trades` backfill (issue #193).

The V1 Polymarket Orderbook subgraph (`7fu2DWYK…`) is frozen at
2026-04-28 and uses an `OrderFill` entity (different from V2's
`OrderFilledEvent`). This module owns the V1-specific query, paginator,
and orchestrator. The adapter emits the same `OrderFilledEvent`
dataclass that V2 produces so the downstream `event_to_corpus_trade`
insert path is shared verbatim.

Fields and conventions verified against `tests/corpus/fixtures/v1_v2_overlap.json`.
"""

from __future__ import annotations

from collections.abc import Mapping

from pscanner.poly.onchain import OrderFilledEvent


class UnsupportedMarketIdFormat(Exception):  # noqa: N818 — name follows local conv
    """Raised when a V1 row's `marketId` is the unrecoverable sentinel `"0"`.

    The Stage 2 query filters these server-side, but the adapter still
    guards against the case in defence of an upstream query change.
    """


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


def _strip_zero_prefix(market_id: str) -> str:
    return market_id[2:] if market_id.startswith("0-") else market_id


def subgraph_v1_row_to_event(
    row: Mapping[str, object],
    *,
    asset_index: Mapping[str, str],
) -> OrderFilledEvent:
    """Adapt one V1 `orderFills` row to the existing `OrderFilledEvent`.

    Args:
        row: One element of the GraphQL ``orderFills`` list.
        asset_index: ``{asset_id_decimal_string: outcome_side}`` mapping for
            the market the row belongs to. The adapter resolves the row's
            ``marketId`` (bare or ``"0-"``-prefixed) against this and
            raises ``KeyError`` if not found.

    Returns:
        ``OrderFilledEvent`` (block_number=0, log_index=0, order_hash=row.id).

    Raises:
        KeyError: marketId resolves to an asset_id not in ``asset_index``,
            or a required key is missing.
        UnsupportedMarketIdFormat: ``marketId == "0"`` (the unrecoverable
            sentinel cohort; Stage 0 may file a follow-up).
        ValueError: a numeric field is not parseable, a string field has
            the wrong type, or ``order.side`` is not 0 or 1.
    """
    market_id_raw = _parse_str("marketId", row["marketId"])
    if market_id_raw == "0":
        raise UnsupportedMarketIdFormat("marketId='0' is the unrecoverable cohort")
    asset_id_str = _strip_zero_prefix(market_id_raw)
    if asset_id_str not in asset_index:
        raise KeyError(asset_id_str)

    order_raw = row.get("order")
    if not isinstance(order_raw, Mapping):
        raise ValueError(f"order must be an object, got {type(order_raw).__name__}")
    side = _parse_int("order.side", order_raw["side"])

    price = _parse_int("price", row["price"])
    size = _parse_int("size", row["size"])
    # USDC base units = price * size / 1e6 (verified via Stage 1 fixture).
    # Integer division is safe: Stage 1 confirmed price*size is a multiple of 1e6.
    usdc_base_units = price * size // 1_000_000
    ctf_base_units = size

    token_id = int(asset_id_str)
    if side == 0:
        # Maker BUY: gave USDC, took CTF.
        maker_asset_id = 0
        taker_asset_id = token_id
        making = usdc_base_units
        taking = ctf_base_units
    elif side == 1:
        # Maker SELL: gave CTF, took USDC.
        maker_asset_id = token_id
        taker_asset_id = 0
        making = ctf_base_units
        taking = usdc_base_units
    else:
        raise ValueError(f"unexpected order.side: {side}")

    return OrderFilledEvent(
        order_hash=_parse_str("id", row["id"]),
        maker=_parse_str("maker", row["maker"]).lower(),
        taker=_parse_str("taker", row["taker"]).lower(),
        maker_asset_id=maker_asset_id,
        taker_asset_id=taker_asset_id,
        making=making,
        taking=taking,
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
Expected: No errors. Fix any warnings before committing.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest_v1.py tests/corpus/test_subgraph_ingest_v1.py
git commit -m "$(cat <<'EOF'
feat(corpus): V1 subgraph row -> OrderFilledEvent adapter (#193)

Parses V1 orderFills into the existing OrderFilledEvent dataclass so
the downstream event_to_corpus_trade insert path is shared with V2.
Maker-POV BUY/SELL derivation from (outcomeIndex, order.side) is
locked to the Stage 1 v1_v2_overlap.json ground-truth fixture.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: V1 paginator + iterator

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest_v1.py`
- Modify: `tests/corpus/test_subgraph_ingest_v1.py`

- [ ] **Step 1: Write the failing paginator tests**

Append to `tests/corpus/test_subgraph_ingest_v1.py`:

```python
import pytest

from pscanner.corpus.subgraph_ingest_v1 import iter_v1_market_trades


class _FakeSubgraphClient:
    """Records every query() invocation and yields canned responses in order."""

    def __init__(self, pages: list[list[dict]]) -> None:
        self._pages = pages
        self.calls: list[dict] = []

    async def query(self, graphql: str, variables: dict) -> dict:
        self.calls.append({"graphql": graphql, "variables": dict(variables)})
        if not self._pages:
            return {"orderFills": []}
        return {"orderFills": self._pages.pop(0)}


def _row(idx: int, asset_id: str, *, outcome: int = 0, side: int = 0) -> dict:
    return {
        "id": f"0xdead-{idx}",
        "transactionHash": "0x" + str(idx).rjust(64, "0"),
        "timestamp": str(1_700_000_000 + idx),
        "marketId": asset_id,
        "outcomeIndex": str(outcome),
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "price": "500000",
        "size": "1000000",
        "fee": "0",
        "order": {"id": f"0x{idx}", "marketId": asset_id, "outcomeIndex": str(outcome), "side": str(side)},
    }


@pytest.mark.asyncio
async def test_paginator_returns_empty_on_no_rows():
    client = _FakeSubgraphClient(pages=[])
    asset_index = {"100": "YES"}
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(
            client=client, asset_ids=["100", "0-100"], asset_index=asset_index, page_size=2
        )
    ]
    assert out == []


@pytest.mark.asyncio
async def test_paginator_advances_id_gt_cursor():
    rows = [_row(i, "100") for i in range(5)]
    # First page has 2 rows (id_gt cursor empty), second has 2, third has 1 (short page terminates).
    client = _FakeSubgraphClient(pages=[rows[0:2], rows[2:4], rows[4:5]])
    asset_index = {"100": "YES"}
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(
            client=client, asset_ids=["100", "0-100"], asset_index=asset_index, page_size=2
        )
    ]
    assert len(out) == 5
    # Cursor on page 2 should be page 1's last id; page 3's cursor should be page 2's last id.
    assert client.calls[0]["variables"]["cursor"] == ""
    assert client.calls[1]["variables"]["cursor"] == "0xdead-1"
    assert client.calls[2]["variables"]["cursor"] == "0xdead-3"


@pytest.mark.asyncio
async def test_paginator_rejects_invalid_page_size():
    client = _FakeSubgraphClient(pages=[])
    asset_index = {"100": "YES"}
    with pytest.raises(ValueError):
        async for _ in iter_v1_market_trades(
            client=client, asset_ids=["100"], asset_index=asset_index, page_size=0
        ):
            pass


@pytest.mark.asyncio
async def test_paginator_passes_market_ids_to_filter():
    client = _FakeSubgraphClient(pages=[])
    asset_index = {"100": "YES", "200": "NO"}
    out = [
        x
        async for x in iter_v1_market_trades(
            client=client,
            asset_ids=["100", "200", "0-100", "0-200"],
            asset_index=asset_index,
            page_size=100,
        )
    ]
    assert out == []
    assert client.calls[0]["variables"]["assets"] == ["100", "200", "0-100", "0-200"]
```

`pytest-asyncio` should already be wired (other tests use it); if `pyproject.toml`'s pytest config doesn't declare it, run `uv run python -c "import pytest_asyncio"` to confirm. If missing, add to the asyncio test marker (`@pytest.mark.asyncio` on each).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v -k paginator`
Expected: FAIL — `iter_v1_market_trades` does not exist.

- [ ] **Step 3: Implement the paginator and iterator**

Append to `src/pscanner/corpus/subgraph_ingest_v1.py`:

```python
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import Any

from pscanner.poly.subgraph import SubgraphClient

_MAX_PAGE_SIZE = 1000

_V1_QUERY = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFills(
    where: { marketId_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id transactionHash timestamp
    marketId outcomeIndex
    maker taker price size fee
    order { id marketId outcomeIndex side }
  }
}
"""


async def _paginate_v1(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    asset_index: Mapping[str, str],
    page_size: int,
) -> AsyncGenerator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from V1 `orderFills`, paginated by id_gt.

    Args:
        client: Open ``SubgraphClient`` pointed at the V1 endpoint.
        asset_ids: Server-side `marketId_in` allowlist. Caller is
            responsible for assembling the bare and ``0-``-prefixed forms.
        asset_index: ``{asset_id_decimal_string: outcome_side}`` for the
            condition under backfill. Passed through to the adapter.
        page_size: Rows per query (1.._MAX_PAGE_SIZE).

    Yields:
        ``(event, ts)`` tuples. Rows that fail the adapter (e.g.
        ``marketId="0"`` or unknown asset_id) are skipped silently here;
        the orchestrator owns counted skipping for observability.
    """
    cursor = ""
    while True:
        result = await client.query(
            _V1_QUERY,
            {"assets": list(asset_ids), "cursor": cursor, "first": page_size},
        )
        rows: list[dict[str, Any]] = result.get("orderFills") or []
        if not rows:
            return
        for row in rows:
            try:
                event = subgraph_v1_row_to_event(row, asset_index=asset_index)
            except (UnsupportedMarketIdFormat, KeyError, ValueError):
                # Skip silently — the orchestrator does counted skipping.
                continue
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            return
        cursor = str(rows[-1]["id"])


async def iter_v1_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    asset_index: Mapping[str, str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Drive `_paginate_v1` for one market with input validation.

    Empty ``asset_ids`` short-circuits to an empty iterator (no query).
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return
    async for ev, ts in _paginate_v1(
        client=client, asset_ids=asset_ids, asset_index=asset_index, page_size=page_size
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
feat(corpus): V1 subgraph paginator + iterator (#193)

Cursor-paginated GraphQL walker over the V1 orderFills entity. Adapter
exceptions on bad rows are swallowed inside the paginator; counted
skipping moves to the orchestrator in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: V1 orchestrator — queue load, per-market drain, sentinel write

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest_v1.py`
- Modify: `tests/corpus/test_subgraph_ingest_v1.py`

The orchestrator loads the V1-pending queue, drives `iter_v1_market_trades` per market, runs the existing `event_to_corpus_trade` (shared with V2), inserts via `CorpusTradesRepo.insert_batch`, and stamps `onchain_v1_processed_at` only when ≥1 row was inserted.

- [ ] **Step 1: Write the failing orchestrator tests**

Append to `tests/corpus/test_subgraph_ingest_v1.py`:

```python
import sqlite3
import time
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    AssetIndexEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
)
from pscanner.corpus.subgraph_ingest_v1 import V1SubgraphRunSummary, run_v1_subgraph_backfill


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
    rows = [_row(0, aid, outcome=0, side=0), _row(1, aid, outcome=0, side=0)]
    client = _FakeSubgraphClient(pages=[rows])
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
    )
    assert summary == V1SubgraphRunSummary(
        markets_processed=1,
        markets_failed=0,
        markets_zero_events=0,
        events_decoded=2,
        trades_inserted=2,
        skipped_unsupported=0,
        skipped_unresolvable=0,
        dups_dropped=0,
    )
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
    client = _FakeSubgraphClient(pages=[])  # zero-event drain
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
    # Wipe the asset_index entry to simulate the gap.
    conn.execute("DELETE FROM asset_index WHERE condition_id = ?", (cid,))
    conn.commit()
    client = _FakeSubgraphClient(pages=[])
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
    )
    # No insert, no sentinel, no query at all.
    assert summary.markets_processed == 0
    assert summary.markets_failed == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_orchestrator_respects_limit(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    for i in range(3):
        _seed_v1_pending_market(conn, "0x" + chr(0x61 + i) * 64, f"100{i}")
    client = _FakeSubgraphClient(pages=[[_row(0, "1000", outcome=0, side=0)]])
    summary = await run_v1_subgraph_backfill(
        conn=conn, client=client, page_size=1000, limit=1, now_ts=1_700_000_999
    )
    assert summary.markets_processed + summary.markets_zero_events == 1
```

- [ ] **Step 2: Confirm the `AssetIndexRepo` API and `CorpusMarketsRepo` API**

Verify that `AssetIndexRepo` has the right symbols by checking `src/pscanner/corpus/repos.py`. The tests use `AssetIndexEntry`, `AssetIndexRepo.upsert`, and `CorpusMarketsRepo.insert_pending`. If any name differs, update the test imports/calls to match. (Run `uv run pytest tests/corpus/test_subgraph_ingest_v1.py -v -k orchestrator --collect-only` — collection errors will name the missing symbol.)

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
    """Aggregate counts returned by ``run_v1_subgraph_backfill``."""

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
    """Return V1-pending, unprocessed markets ordered by descending volume.

    The order matches the V2 path's heuristic — large markets drain
    first so a partial run is maximally useful.
    """
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


def _load_asset_index_for_market(
    conn: sqlite3.Connection, condition_id: str
) -> Mapping[str, str]:
    rows = conn.execute(
        "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id = ?",
        (condition_id,),
    ).fetchall()
    return {r["asset_id"]: r["outcome_side"] for r in rows}


def _mark_v1_processed(
    conn: sqlite3.Connection, condition_id: str, *, now_ts: int
) -> None:
    """Stamp `onchain_v1_processed_at` AND clear `v1_history_pending`. Atomic."""
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
    """Drain one V1 market.

    Returns:
        ``(events_decoded, trades_inserted, skipped_unsupported,
        skipped_unresolvable, dups_dropped)``.
    """
    asset_index = _load_asset_index_for_market(conn, condition_id)
    if not asset_index:
        _LOG.warning("subgraph.v1.no_asset_index", condition_id=condition_id)
        return 0, 0, 0, 0, 0

    # The V1 server-side filter accepts both bare and "0-"-prefixed forms;
    # submit both so we catch every row variant the issue documented.
    market_ids = list(asset_index.keys()) + [f"0-{aid}" for aid in asset_index.keys()]

    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    events_decoded = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    pending: list[CorpusTrade] = []
    async for event, ts in iter_v1_market_trades(
        client=client,
        asset_ids=market_ids,
        asset_index=asset_index,
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
            # Defensive: asset_index integrity issue, not the adapter's fault.
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
    """Process every V1-pending market via the V1 subgraph.

    Args:
        conn: Open corpus DB connection.
        client: Open ``SubgraphClient`` pointed at the V1 endpoint.
        page_size: GraphQL ``first:`` per query (max 1000).
        limit: Process at most ``N`` markets in this run.
        now_ts: Override ``time.time()`` for the sentinel stamp (testing).

    Returns:
        ``V1SubgraphRunSummary`` with per-class counts.
    """
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
Expected: No errors.

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

The dispatcher runs the existing V2 orchestrator first, then the new V1 orchestrator, then calls the shared `_clear_truncation_flags` once. It does NOT modify the V2 module — it imports `run_subgraph_backfill` and `_clear_truncation_flags` from it.

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
    """Stub that records nothing and returns empty results."""

    async def query(self, graphql: str, variables: dict[str, Any]) -> dict:
        return {"orderFilledEvents": [], "orderFills": []}


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
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the dispatcher**

```python
# src/pscanner/corpus/subgraph_dispatch.py
"""V1+V2 subgraph-backfill dispatcher (issue #193).

Runs the V2 backfill (existing) then the V1 backfill (new), then the
shared truncation-flag clearance. Both versions share the corpus DB
connection but use independent ``SubgraphClient`` instances pointed at
their respective subgraph deployments.
"""

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
    """Result of one dispatched backfill run.

    ``v1_summary``/``v2_summary`` are ``None`` when that version was
    excluded by the ``versions`` argument.
    """

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
    """Run each requested subgraph version's backfill.

    Args:
        conn: Open corpus DB connection.
        v1_client: ``SubgraphClient`` for the V1 endpoint.
        v2_client: ``SubgraphClient`` for the V2 endpoint.
        versions: Ordered list of versions to run. Default ``("v2", "v1")``
            preserves the existing V2-first ordering.
        page_size: GraphQL ``first:`` per query (max 1000), passed to both.
        limit: Process at most ``N`` markets per version in this run.
        truncation_threshold: Trade-count threshold below which
            ``truncated_at_offset_cap`` stays set. Passed to V2 only;
            ``_clear_truncation_flags`` runs once at the end and uses the
            same value.

    Returns:
        ``DispatchedRunSummary`` with per-version summaries and the final
        truncation-clearance count.
    """
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
Expected: All four tests PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_dispatch.py tests/corpus/test_subgraph_dispatch.py && uv run ty check src/pscanner/corpus/subgraph_dispatch.py`
Expected: No errors.

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

Append to `tests/corpus/test_cli.py` (or create a new test file `tests/corpus/test_cli_subgraph_dispatch.py` if `test_cli.py` is unwieldy):

```python
def test_subgraph_backfill_help_lists_version_flags():
    from pscanner.corpus.cli import _build_parser

    parser = _build_parser()
    # argparse subparsers: navigate to `subgraph-backfill`.
    sub_actions = [a for a in parser._actions if hasattr(a, "choices")]
    assert sub_actions, "no subparser found"
    subparsers = sub_actions[0].choices
    sg = subparsers["subgraph-backfill"]
    help_text = sg.format_help()
    assert "--subgraph-version" in help_text
    assert "--v1-subgraph-id" in help_text
    assert "--v2-subgraph-id" in help_text
    assert "--subgraph-id" in help_text  # deprecated alias preserved


def test_subgraph_backfill_subgraph_id_alias_maps_to_v2(caplog, monkeypatch):
    """The deprecated `--subgraph-id` flag must set `args.v2_subgraph_id` and warn."""
    from pscanner.corpus.cli import _build_parser, _resolve_subgraph_flags

    parser = _build_parser()
    args = parser.parse_args(
        ["subgraph-backfill", "--subgraph-id", "deprecated-id-value"]
    )
    # _resolve_subgraph_flags is the canonical place to apply the deprecation.
    resolved = _resolve_subgraph_flags(args)
    assert resolved.v2_subgraph_id == "deprecated-id-value"
```

(If `_build_parser` is not module-level, refactor it out first — argparse parsers are usually trivially extractable. If `tests/corpus/test_cli.py` already imports parser construction differently, follow that local convention.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_cli.py -v -k subgraph`
Expected: FAIL — new flags + `_resolve_subgraph_flags` do not exist.

- [ ] **Step 3: Add the new CLI flags**

In `src/pscanner/corpus/cli.py`, near the top with the other constants (around line 65), add:

```python
_DEFAULT_V1_SUBGRAPH_ID = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
```

In `_build_parser`'s `subgraph-backfill` subparser block (around lines 149–186), replace the single `--subgraph-id` flag with this group (keep `--rpm`, `--page-size`, `--limit` unchanged):

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

Add the `_resolve_subgraph_flags` helper (just after `_build_parser` or near other helpers):

```python
@dataclass(frozen=True)
class _ResolvedSubgraphFlags:
    """Materialized subgraph flags after deprecation-alias resolution."""

    v2_subgraph_id: str
    v1_subgraph_id: str
    versions: tuple[str, ...]


def _resolve_subgraph_flags(args: argparse.Namespace) -> _ResolvedSubgraphFlags:
    """Apply the `--subgraph-id` deprecation alias and version selection.

    Logs `subgraph.cli.deprecated_flag` when the alias is used so the
    operator gets one notice per invocation.
    """
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

Add `from dataclasses import dataclass` and `import argparse` to the imports if not already present.

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
Expected: PASS.

- [ ] **Step 6: Run the full corpus suite to confirm no regressions**

Run: `uv run pytest tests/corpus -v`
Expected: All tests PASS. Investigate any regression before continuing.

- [ ] **Step 7: Lint + type-check the whole CLI file**

Run: `uv run ruff check src/pscanner/corpus/cli.py && uv run ruff format --check src/pscanner/corpus/cli.py && uv run ty check src/pscanner/corpus/cli.py`
Expected: No errors.

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

The most-important integration scenario is a market where V2 already ran and V1 also has data. Confirm that running both leaves the V2 sentinel untouched, sets the V1 sentinel, and clears `v1_history_pending`.

- [ ] **Step 1: Write the integration test**

Append to `tests/corpus/test_subgraph_ingest_v1.py`:

```python
@pytest.mark.asyncio
async def test_hybrid_market_sets_both_sentinels(tmp_path: Path):
    """V2 ran first (sets onchain_processed_at), then V1 runs (sets _v1 column).

    Both sentinels end up populated; v1_history_pending flips to 0.
    """
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    cid = "0x" + "f" * 64
    aid = "7777777777"
    _seed_v1_pending_market(conn, cid, aid)
    # Simulate V2 already ran: stamp onchain_processed_at directly.
    conn.execute(
        "UPDATE corpus_markets SET onchain_processed_at = ? WHERE condition_id = ?",
        (1_600_000_000, cid),
    )
    conn.commit()

    rows = [_row(0, aid, outcome=0, side=0)]
    client = _FakeSubgraphClient(pages=[rows])
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
    assert row["onchain_processed_at"] == 1_600_000_000   # untouched by V1
    assert row["onchain_v1_processed_at"] == 1_700_000_999  # set by V1
    assert row["v1_history_pending"] == 0                  # cleared by V1
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/corpus/test_subgraph_ingest_v1.py::test_hybrid_market_sets_both_sentinels -v`
Expected: PASS.

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

## Task 10: Production smoke run

**Files:**
- None modified. Operator-driven verification.

- [ ] **Step 1: Pre-flight check**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: All green. No warnings.

- [ ] **Step 2: Dry-ish smoke — V1 path only, small limit**

Run:
```bash
GRAPH_API_KEY=... uv run pscanner corpus subgraph-backfill \
    --subgraph-version v1 \
    --rpm 50 \
    --limit 5 2>&1 | tee /tmp/v1-smoke.log
```
Expected: `subgraph.v1.start markets=5`, followed by 5 `market_complete` or `zero_events` events, then `subgraph.cli_summary`. Exit 0.

- [ ] **Step 3: Verify the production DB shows the sentinel + cleared flag**

Run:
```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
rows = conn.execute('''
  SELECT condition_id, onchain_v1_processed_at, v1_history_pending, market_slug
  FROM corpus_markets
  WHERE onchain_v1_processed_at IS NOT NULL
  ORDER BY onchain_v1_processed_at DESC LIMIT 10
''').fetchall()
for r in rows:
    print(dict(r))
"
```
Expected: Up to 5 rows, each with `onchain_v1_processed_at != NULL` and `v1_history_pending = 0`. Markets that produced zero V1 events stay at `v1_history_pending = 1, onchain_v1_processed_at = NULL`.

- [ ] **Step 4: Spot-check trade-count delta on one processed market**

Pick the market with the highest `trades_inserted` from the smoke log. Run:
```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
cid = '<paste the condition_id here>'
n = conn.execute('SELECT COUNT(*) FROM corpus_trades WHERE condition_id = ?', (cid,)).fetchone()[0]
min_ts = conn.execute('SELECT MIN(ts) FROM corpus_trades WHERE condition_id = ?', (cid,)).fetchone()[0]
print(f'trade_count={n} oldest_ts={min_ts}')
"
```
Expected: `oldest_ts < 1775220779` (V2 start), confirming V1 fills landed before the V2 window.

- [ ] **Step 5: Update CLAUDE.md**

Replace the "Tracked work in flight" entry about #193 with a "Polymarket API quirks" entry summarising the V1 adapter shipped state. Concretely: move the relevant lessons (V1 subgraph endpoint, frozen at 2026-04-28, `marketId="0"` skipped) into the quirks section; remove the V1 follow-up bullet from "Tracked work in flight".

- [ ] **Step 6: Commit the doc update**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: record V1 subgraph adapter shipped (#193)

Move V1 adapter notes from "Tracked work in flight" into the
Polymarket API quirks section; document onchain_v1_processed_at
sentinel and marketId="0" skip behaviour.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Run the full smoke on the production V1 queue**

Optional (operator decision — long-running, ~hours depending on queue size):
```bash
GRAPH_API_KEY=... uv run pscanner corpus subgraph-backfill \
    --subgraph-version v1 \
    --rpm 50 2>&1 | tee /tmp/v1-full.log
```
Expected: drains the production V1-pending queue. Check the final `subgraph.cli_summary` event for `v1_trades_inserted` (target: meaningful insertion count; 100K+ rows would not be surprising for the 2,769-market queue).

---

## Self-Review

**Spec coverage:**
- Stage 0 investigation → Task 1 ✓
- Stage 1 overlap-window verification + fixture → Task 2 ✓
- `onchain_v1_processed_at` column migration → Task 3 ✓
- V1 adapter parser → Task 4 ✓
- V1 paginator + iterator → Task 5 ✓
- V1 orchestrator with sentinel hygiene (no stamp on zero rows) → Task 6 ✓
- Dispatcher (V2 → V1 → clear truncation) → Task 7 ✓
- CLI flags + deprecated alias → Task 8 ✓
- Hybrid-market integration test → Task 9 ✓
- Production smoke + doc update → Task 10 ✓
- Error-handling matrix from spec § Error handling: per-row skip via try/except in `_paginate_v1` (Task 5), per-market failure caught in `run_v1_subgraph_backfill`'s outer loop with `markets_failed` increment (Task 6), no-asset-index early-exit (Task 6 test), `INSERT OR IGNORE` dup-counting via `dups = len(pending) - inserted` (Task 6), idempotent schema migration (Task 3) ✓
- Observability events from spec: every event named in the spec's grep-table is emitted in Tasks 5, 6, 7, 8 ✓
- "Server-side filter excludes `marketId="0"`" assertion: covered by the `marketId_in` allowlist construction (Task 6 `_backfill_one_v1_market` uses `list(asset_index.keys()) + [f"0-{aid}" …]` — no `"0"`-only entry). Task 4's `test_parser_rejects_zero_marketid` defends the adapter against a future query-shape change.

**Placeholder scan:** No "TBD" / "TODO" / "similar to" — all code shown.

**Type consistency:**
- `V1SubgraphRunSummary` field set is used identically in Task 6 (test asserts equality), Task 7 (dispatcher's `DispatchedRunSummary.v1_summary` typed as `V1SubgraphRunSummary | None`), and Task 8 (CLI summary extracts fields by name).
- `subgraph_v1_row_to_event(row, *, asset_index)` signature is consistent across Tasks 4, 5, and 6.
- `run_v1_subgraph_backfill(*, conn, client, page_size, limit, now_ts)` is consistent across Tasks 6, 7, and 9.
- `_resolve_subgraph_flags` returns `_ResolvedSubgraphFlags` with `v2_subgraph_id`, `v1_subgraph_id`, `versions` — consumed identically in `_cmd_subgraph_backfill` and the CLI test.

**Scope:** One feature, ten tasks, ~3–5 hours of implementation work (excluding the operator-side smoke run in Task 10 Step 7, which is long-running but not blocking).

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-05-26-issue-193-v1-subgraph-adapter.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
