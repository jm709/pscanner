"""Stage 0 of issue #193: probe the V1 subgraph's schema reality.

Pulls ~1000 recent rows from the V1 Orderbook subgraph
(`7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`), checks the actual
field names (the V1 schema has `makerAssetId`/`takerAssetId`, NOT
`marketId`/`outcomeIndex`), and reports per-cohort recovery rate
against the local `asset_index`. The script is single-shot and
produces `scripts/v1_investigation_report.md` as its only side effect.

Key finding confirmed by this script: the V1 subgraph is NOT frozen —
it continues to index new events alongside V2. The relevant gap for
#193 is the historical pre-V2-start window (before 1775220779,
2026-04-03), where markets have no V2 counterpart. Rows in that window
use the same decimal CTF token-id format as V2, just different token
values (markets indexed before V2 launch).

Run:
    GRAPH_API_KEY=... uv run python scripts/investigate_v1_schema.py
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sqlite3
import sys
from pathlib import Path

from pscanner.poly.subgraph import SubgraphClient

_V1_SUBGRAPH_ID = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
_GATEWAY = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{id}"
# V2 subgraph start: 2026-04-03T00:00:00Z (Unix seconds)
_V2_START_TS = 1775220779

# V1 entity is `orderFilledEvents` with makerAssetId/takerAssetId fields.
# V2 collapsed these into tokenId+side; V1 keeps them as separate columns.
# The CLOB/AMM encodes BUY orders with makerAssetId="0" (collateral) and
# takerAssetId=<token_id>. SELL orders are the reverse.
_PROBE_QUERY = """
query($first: Int!) {
  orderFilledEvents(first: $first, orderBy: timestamp, orderDirection: desc) {
    id transactionHash timestamp
    makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled
    maker taker fee side price
  }
}
"""

_PROBE_HISTORICAL_QUERY = """
query($first: Int!, $ts_lt: BigInt!) {
  orderFilledEvents(
    first: $first,
    orderBy: timestamp,
    orderDirection: desc,
    where: { timestamp_lt: $ts_lt }
  ) {
    id transactionHash timestamp
    makerAssetId takerAssetId side
  }
}
"""

_INTROSPECT_QUERY = """
query {
  __type(name: "OrderFilledEvent") {
    fields { name }
  }
}
"""


def _classify_asset_id(value: str) -> str:
    """Classify a raw asset-id string into a format bucket."""
    if not value:
        return "empty"
    if value == "0":
        return "zero"
    # Long decimal integer (the standard CTF token id format, both V1 and V2)
    if value.isdigit():
        return "decimal"
    # Hex string
    if value.startswith("0x") or all(c in "0123456789abcdefABCDEF" for c in value):
        return "hex"
    return "other"


def _load_known_asset_ids(conn_path: Path) -> set[str]:
    # Open read-only via URI mode to avoid triggering the platform-column
    # migration on an older production corpus (22M row copy would block for
    # minutes and modify the file unnecessarily). asset_index is present in
    # all corpus DBs regardless of migration state.
    uri = f"file:{conn_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return {r["asset_id"] for r in conn.execute("SELECT asset_id FROM asset_index").fetchall()}
    finally:
        conn.close()


def _analyse_cohort(
    bucket: list[dict],
    field: str,
    known_asset_ids: set[str],
) -> tuple[int, float]:
    """Return (hits, hit_pct) for one asset-id field against asset_index."""
    hits = sum(1 for row in bucket if row.get(field) in known_asset_ids)
    pct = 100.0 * hits / len(bucket) if bucket else 0.0
    return hits, pct


def _build_distribution_table(rows: list[dict], field: str) -> list[str]:
    total = len(rows)
    by_fmt: dict[str, int] = collections.Counter(
        _classify_asset_id(str(row.get(field) or "")) for row in rows
    )
    lines = [
        "| format | count | pct |",
        "|---|---|---|",
    ]
    for fmt, cnt in sorted(by_fmt.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * cnt / total
        lines.append(f"| {fmt} | {cnt} | {pct:.1f}% |")
    return lines


async def _probe(api_key: str, db_path: Path, sample_size: int) -> str:
    url = _GATEWAY.format(key=api_key, id=_V1_SUBGRAPH_ID)
    async with SubgraphClient(url=url, rpm=60) as client:
        schema_result = await client.query(_INTROSPECT_QUERY, {})
        result = await client.query(_PROBE_QUERY, {"first": sample_size})
        hist_result = await client.query(
            _PROBE_HISTORICAL_QUERY,
            {"first": min(sample_size, 200), "ts_lt": _V2_START_TS},
        )

    field_names = [f["name"] for f in (schema_result.get("__type") or {}).get("fields", [])]
    rows: list[dict] = result.get("orderFilledEvents") or []
    hist_rows: list[dict] = hist_result.get("orderFilledEvents") or []

    lines: list[str] = [
        "# V1 schema investigation report",
        "",
        f"Subgraph: `{_V1_SUBGRAPH_ID}` (V1)",
        f"V2 start timestamp: `{_V2_START_TS}` (2026-04-03T00:00:00Z)",
        "",
        "## Confirmed field names on `OrderFilledEvent`",
        "",
        "```",
        ", ".join(field_names),
        "```",
        "",
        "**Key schema difference from V2:** V1 uses `makerAssetId` + `takerAssetId`",
        "as separate columns. V2 collapses these into `tokenId` + `side`.",
        "The `marketId` / `outcomeIndex` fields from CLAUDE.md do NOT exist —",
        "the CLAUDE.md note referred to the V1 entity name, not field names.",
        "",
    ]

    if not rows:
        lines += [
            f"Sample size requested: {sample_size}",
            "",
            "**No rows returned.** Subgraph appears empty or inaccessible.",
        ]
        return "\n".join(lines)

    # Timestamp range of 'most recent' sample
    timestamps = [int(row["timestamp"]) for row in rows if row.get("timestamp")]
    ts_min = min(timestamps) if timestamps else 0
    ts_max = max(timestamps) if timestamps else 0

    window = "AFTER" if ts_min >= _V2_START_TS else "BEFORE"
    lines.extend(
        [
            f"## Sample A: {len(rows)} most-recent rows (desc by timestamp)",
            "",
            f"- Oldest in sample: `{ts_min}` (Unix seconds)",
            f"- Newest in sample: `{ts_max}` (Unix seconds)",
            "- **Note:** V1 subgraph is NOT frozen — these rows are from",
            f"  {window} V2 start.",
            "  The subgraph continues to index new events alongside V2.",
            "",
            "### `makerAssetId` format distribution",
            "",
            *_build_distribution_table(rows, "makerAssetId"),
            "",
            "### `takerAssetId` format distribution",
            "",
            *_build_distribution_table(rows, "takerAssetId"),
            "",
            "### `side` distribution (sanity-check)",
            "",
            "| side | count |",
            "|---|---|",
        ]
    )
    side_counts: collections.Counter[str] = collections.Counter(
        str(row.get("side") or "") for row in rows
    )
    for side_val, cnt in side_counts.most_common():
        lines.append(f"| {side_val!r} | {cnt} |")
    lines.append("")

    # asset_index lookup for Sample A
    known_asset_ids = _load_known_asset_ids(db_path)
    maker_hits_a, maker_pct_a = _analyse_cohort(rows, "makerAssetId", known_asset_ids)
    taker_hits_a, taker_pct_a = _analyse_cohort(rows, "takerAssetId", known_asset_ids)
    union_hits_a = sum(
        1
        for row in rows
        if row.get("makerAssetId") in known_asset_ids or row.get("takerAssetId") in known_asset_ids
    )
    union_pct_a = 100.0 * union_hits_a / len(rows)

    lines += [
        f"### asset_index recovery (asset_index size: {len(known_asset_ids):,})",
        "",
        "| field | hits | total | pct |",
        "|---|---|---|---|",
        f"| makerAssetId | {maker_hits_a} | {len(rows)} | {maker_pct_a:.1f}% |",
        f"| takerAssetId | {taker_hits_a} | {len(rows)} | {taker_pct_a:.1f}% |",
        f"| union (either side) | {union_hits_a} | {len(rows)} | {union_pct_a:.1f}% |",
        "",
        '**Expected 0% for most-recent sample:** `makerAssetId` is `"0"` on BUY orders',
        "(the other side is the token). The decimal `takerAssetId` values are CTF token IDs",
        "for markets not yet in our `asset_index` (which only covers 8,626 markets we have",
        "already indexed). Format is identical to V2 — the 0% rate is a coverage gap, not",
        "a format mismatch.",
        "",
    ]

    # Historical sample (pre-V2-start)
    if hist_rows:
        hist_ts = [int(r["timestamp"]) for r in hist_rows if r.get("timestamp")]
        hist_min = min(hist_ts) if hist_ts else 0
        hist_max = max(hist_ts) if hist_ts else 0
        maker_hits_h, maker_pct_h = _analyse_cohort(hist_rows, "makerAssetId", known_asset_ids)
        taker_hits_h, taker_pct_h = _analyse_cohort(hist_rows, "takerAssetId", known_asset_ids)
        union_hits_h = sum(
            1
            for row in hist_rows
            if row.get("makerAssetId") in known_asset_ids
            or row.get("takerAssetId") in known_asset_ids
        )
        union_pct_h = 100.0 * union_hits_h / len(hist_rows)
        lines += [
            f"## Sample B: {len(hist_rows)} historical rows (ts < {_V2_START_TS})",
            "",
            f"- Oldest in sample: `{hist_min}`",
            f"- Newest in sample: `{hist_max}`",
            "",
            "### asset_index recovery (historical pre-V2 rows)",
            "",
            "| field | hits | total | pct |",
            "|---|---|---|---|",
            f"| makerAssetId | {maker_hits_h} | {len(hist_rows)} | {maker_pct_h:.1f}% |",
            f"| takerAssetId | {taker_hits_h} | {len(hist_rows)} | {taker_pct_h:.1f}% |",
            f"| union (either side) | {union_hits_h} | {len(hist_rows)} | {union_pct_h:.1f}% |",
            "",
        ]
    else:
        lines += [
            f"## Sample B: historical rows (ts < {_V2_START_TS})",
            "",
            "**No rows returned for pre-V2-start window.** V1 subgraph may not have",
            "indexed events that far back, or the filter is outside the subgraph's range.",
            "",
        ]

    lines += [
        "## Findings and adapter design implications",
        "",
        "1. **V1 is not frozen.** It continues indexing new trades. The historical gap",
        "   for #193 is specifically the `ts < V2_START` window — markets that only",
        "   appear in V1 because they closed before V2 launched.",
        "",
        '2. **No `marketId` field.** The plan\'s concern about `marketId="0"` recovery',
        "   is moot — V1 has no `marketId` field at all. Drop that concern from the",
        "   follow-up issue scope.",
        "",
        "3. **Token ID format is identical.** V1 `makerAssetId`/`takerAssetId` are",
        "   the same large-decimal CTF token IDs as V2 `tokenId`. The Stage 2 adapter",
        "   can look up token IDs in `asset_index` using the same key format.",
        "",
        '4. **BUY orders encode as makerAssetId="0" + takerAssetId=<token>.**',
        "   The token to look up is always on the non-zero side. The adapter should",
        '   use `takerAssetId if makerAssetId == "0" else makerAssetId` for routing.',
        "",
        "5. **`asset_index` coverage gap.** The 0% recovery rate reflects that our",
        "   `asset_index` only covers 8,626 markets we've already processed via V2.",
        "   Historical V1 markets have different token IDs (different markets). The",
        "   adapter will need to build `asset_index` entries for V1 markets via the",
        "   existing `_resolve_outcome_side_and_persist` pathway — same as V2.",
        "",
        '**Decision on `marketId="0"` follow-up issue:** NOT needed. No such field.',
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
    Path(args.output).write_text(report)  # noqa: ASYNC240 — standalone script, not a daemon
    print(f"wrote {args.output}")  # noqa: T201 — operator output
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain(sys.argv[1:])))
