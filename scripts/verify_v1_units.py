r"""Stage 1 of issue #193: produce adapter ground-truth fixture.

Original goal: reconcile V1 and V2 `orderFilledEvents` row-by-row on
(transactionHash, orderHash).  BLOCKED: V1 and V2 index different Exchange
contracts; no shared (transactionHash, orderHash) pairs exist in the overlap
window (verified 2026-05-26 by checking 200 V1 tx hashes and 200 V1 order
hashes against V2 -- zero intersection).

Revised goal: pull real V1 BUY and SELL rows from the production subgraph and
commit them as `tests/corpus/fixtures/v1_v2_overlap.json` so the V1 adapter
unit tests (Tasks 4-6) have ground truth covering both event shapes.  The
fixture also includes reference V2 rows from the same overlap window so tests
can assert format correspondence without relying on a shared transaction.

Run:
    GRAPH_API_KEY=... uv run python scripts/verify_v1_units.py \
        --condition-id <hex> --db /home/macph/projects/polymarketScanner/data/corpus.sqlite3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

from pscanner.poly.subgraph import SubgraphClient

_V1 = "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"
_V2 = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
_GATEWAY = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{id}"
# April 3 2026 -> April 28 2026 overlap window (Unix seconds).
_OVERLAP_MIN = 1775220779
_OVERLAP_MAX = 1777374040
_MAX_RETRIES = 3
# Spec: "Pick representative samples (>=2 BUY, >=2 SELL)".
_MIN_SAMPLE_PER_SIDE = 2

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

# Reference V2 query -- fetches events from the overlap window from ANY market
# (no asset filter) to demonstrate V2 schema shape alongside V1 rows.
_V2_REF_Q = """
query($tmin: BigInt!, $tmax: BigInt!, $first: Int!) {
  orderFilledEvents(
    where: { timestamp_gte: $tmin, timestamp_lte: $tmax }
    first: $first orderBy: timestamp orderDirection: asc
  ) {
    id transactionHash timestamp orderHash
    maker { id } taker { id } market { id }
    tokenId side makerAmountFilled takerAmountFilled fee
  }
}
"""


def _load_asset_ids(db: str, condition_id: str) -> tuple[list[str], list[dict]]:
    """Load asset_index rows for a condition_id.

    Opens the corpus DB read-only to avoid triggering migrations on a
    production DB that may have an orphaned ``corpus_trades__new`` table from a
    prior interrupted run.

    Args:
        db: Path to the corpus SQLite file.
        condition_id: Polymarket condition ID to look up.

    Returns:
        Tuple of (asset_id list, asset_index row dicts).
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id = ?",
            (condition_id,),
        ).fetchall()
    finally:
        conn.close()
    return [r["asset_id"] for r in rows], [dict(r) for r in rows]


def _filter_v1_sides(
    v1_maker: list[dict],
    v1_taker: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Filter V1 raw query results to CTF<->USDC fills only, with drop diagnostics.

    V1 maker-side query (``makerAssetId_in``) returns all events where the maker
    held a CTF token, including CTF<->CTF split/merge events.  Valid BUY fills
    have ``takerAssetId=="0"`` (USDC).  Mirror logic applies for taker-side.

    Args:
        v1_maker: Raw rows from the ``makerAssetId_in`` query (SELL side).
        v1_taker: Raw rows from the ``takerAssetId_in`` query (BUY side).

    Returns:
        Tuple of (v1_buy_rows, v1_sell_rows) containing only CTF<->USDC fills.
    """
    # BUY: taker held the CTF token; for a valid CTF<->USDC fill the maker pays
    # USDC so makerAssetId must be "0".
    v1_buy_rows: list[dict] = []
    buy_drops = 0
    for r in v1_taker:
        if r.get("makerAssetId") == "0":
            v1_buy_rows.append(r)
        else:
            buy_drops += 1

    # SELL: maker held the CTF token; for a valid CTF<->USDC fill the taker
    # receives USDC so takerAssetId must be "0".
    v1_sell_rows: list[dict] = []
    sell_drops = 0
    for r in v1_maker:
        if r.get("takerAssetId") == "0":
            v1_sell_rows.append(r)
        else:
            sell_drops += 1

    if buy_drops or sell_drops:
        sys.stderr.write(
            f"dropped {buy_drops} split/merge rows from V1 BUY query, "
            f"{sell_drops} from SELL query (these are CTF<->CTF fills, "
            "not CTF<->USDC trades)\n"
        )

    return v1_buy_rows, v1_sell_rows


def _check_v1_sanity(buy_rows: list[dict], sell_rows: list[dict]) -> list[str]:
    """Assert V1 side-encoding invariants hold.

    Args:
        buy_rows: Rows from ``takerAssetId_in`` query (makerAssetId should be
            ``"0"``).
        sell_rows: Rows from ``makerAssetId_in`` query (takerAssetId should be
            ``"0"``).

    Returns:
        List of failure messages (empty on success).
    """
    failures: list[str] = []
    for r in buy_rows:
        if r.get("side") != "buy":
            failures.append(
                f"BUY row side mismatch: expected 'buy' got '{r.get('side')}' "
                f"on {r['transactionHash']}"
            )
    for r in sell_rows:
        if r.get("side") != "sell":
            failures.append(
                f"SELL row side mismatch: expected 'sell' got '{r.get('side')}' "
                f"on {r['transactionHash']}"
            )
    return failures


async def _fetch_rows(
    api_key: str,
    asset_ids: list[str],
    per_side: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Query both subgraphs, returning V1 maker/taker rows and V2 reference rows.

    Retries up to ``_MAX_RETRIES`` times on transient "bad indexers" V1 errors.

    Args:
        api_key: The Graph gateway API key.
        asset_ids: CTF token IDs for the target market.
        per_side: Max rows to fetch per V1 query side.

    Returns:
        Tuple of (v1_maker_rows, v1_taker_rows, v2_ref_rows).

    Raises:
        RuntimeError: On persistent subgraph failure.
    """
    common = {
        "ids": asset_ids,
        "tmin": str(_OVERLAP_MIN),
        "tmax": str(_OVERLAP_MAX),
        "first": per_side,
    }
    ref_common: dict[str, object] = {
        "tmin": str(_OVERLAP_MIN),
        "tmax": str(_OVERLAP_MAX),
        "first": 4,
    }
    v1_url = _GATEWAY.format(key=api_key, id=_V1)
    v2_url = _GATEWAY.format(key=api_key, id=_V2)

    last_exc: RuntimeError | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            async with (
                SubgraphClient(url=v1_url, rpm=60) as v1,
                SubgraphClient(url=v2_url, rpm=60) as v2,
            ):
                maker = (await v1.query(_V1_Q_MAKER, common)).get("orderFilledEvents") or []
                taker = (await v1.query(_V1_Q_TAKER, common)).get("orderFilledEvents") or []
                ref = (await v2.query(_V2_REF_Q, ref_common)).get("orderFilledEvents") or []
            return maker, taker, ref
        except RuntimeError as exc:
            last_exc = exc
            if "bad indexers" in str(exc) and attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(5)
                continue
            break
    msg = f"subgraph query failed after {_MAX_RETRIES} attempts: {last_exc}"
    raise RuntimeError(msg)


async def _amain(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: CLI argument list (excluding the program name).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "V1 adapter ground-truth fixture generator (issue #193, Stage 1). "
            "Cross-subgraph tx-hash verification is impossible -- see module docstring."
        )
    )
    parser.add_argument("--condition-id", required=True)
    parser.add_argument(
        "--db", default="/home/macph/projects/polymarketScanner/data/corpus.sqlite3"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default="tests/corpus/fixtures/v1_v2_overlap.json")
    parser.add_argument("--per-side", type=int, default=4)
    args = parser.parse_args(argv)
    api_key = args.api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        sys.stderr.write("error: --api-key or $GRAPH_API_KEY required\n")
        return 2

    asset_ids, asset_rows = _load_asset_ids(args.db, args.condition_id)
    if not asset_ids:
        sys.stderr.write(f"error: no asset_index rows for {args.condition_id}\n")
        return 3

    try:
        v1_maker, v1_taker, v2_ref = await _fetch_rows(api_key, asset_ids, args.per_side)
    except RuntimeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 5

    v1_buy_rows, v1_sell_rows = _filter_v1_sides(v1_maker, v1_taker)

    if len(v1_sell_rows) < _MIN_SAMPLE_PER_SIDE or len(v1_buy_rows) < _MIN_SAMPLE_PER_SIDE:
        sys.stderr.write(
            f"INSUFFICIENT V1 ROWS: {len(v1_buy_rows)} BUY, "
            f"{len(v1_sell_rows)} SELL rows for {args.condition_id} "
            f"(need >={_MIN_SAMPLE_PER_SIDE} of each)\n"
        )
        return 4

    failures = _check_v1_sanity(v1_buy_rows, v1_sell_rows)
    if failures:
        sys.stderr.write("VERIFICATION FAILED:\n  " + "\n  ".join(failures) + "\n")
        return 4

    out = {
        "condition_id": args.condition_id,
        "asset_index": {r["asset_id"]: r["outcome_side"] for r in asset_rows},
        # NOTE: cross_subgraph_match is False -- V1 and V2 index different
        # Exchange contracts with no shared (transactionHash, orderHash) pairs.
        # v1_buy_rows / v1_sell_rows are real V1 events; v2_reference_rows are
        # real V2 events from the same overlap window (different market).
        # Adapter tests assert shape correspondence, not a shared transaction.
        "cross_subgraph_match": False,
        "v1_buy_rows": v1_buy_rows[:2],
        "v1_sell_rows": v1_sell_rows[:2],
        "v2_reference_rows": v2_ref[:2],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(  # noqa: ASYNC240 -- standalone script, not a daemon
        json.dumps(out, indent=2, sort_keys=True)
    )
    print(  # noqa: T201 -- operator output
        f"wrote {args.output}: {len(v1_buy_rows)} V1 BUY rows, "
        f"{len(v1_sell_rows)} V1 SELL rows, "
        f"{len(v2_ref)} V2 reference rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain(sys.argv[1:])))
