"""Tests for the V1 subgraph adapter (issue #193)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from pscanner.corpus.subgraph_ingest_v1 import (
    _V1_SUBGRAPH_EARLIEST_TS,
    _count_pre_v1_skipped,
    _load_pending_v1_markets,
    subgraph_v1_row_to_event,
)
from pscanner.poly.subgraph import SubgraphClient

_FIXTURE = Path(__file__).parent / "fixtures" / "v1_v2_overlap.json"


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def test_parser_handles_v1_buy_rows_from_production_fixture():
    """V1 BUY rows: makerAssetId='0', takerAssetId=<token>, side='buy'.

    The fixture's `v1_buy_rows` are real production rows from the V1
    Polymarket subgraph (see Task 2's `scripts/verify_v1_units.py`).
    The adapter must produce an OrderFilledEvent where maker_asset_id=0
    and taker_asset_id matches the row's takerAssetId.
    """
    data = _load_fixture()
    assert data["cross_subgraph_match"] is False, (
        "fixture must declare cross_subgraph_match=False — V1/V2 index "
        "different Polygon contracts and share no transactions"
    )
    buys = data["v1_buy_rows"]
    assert len(buys) >= 1, "fixture must contain at least one V1 BUY row"
    for row in buys:
        assert row["makerAssetId"] == "0"
        assert row["side"] == "buy"
        event = subgraph_v1_row_to_event(row)
        assert event.maker_asset_id == 0
        assert event.taker_asset_id == int(row["takerAssetId"])
        assert event.making == int(row["makerAmountFilled"])
        assert event.taking == int(row["takerAmountFilled"])
        assert event.maker == row["maker"].lower()
        assert event.taker == row["taker"].lower()
        assert event.tx_hash == row["transactionHash"]
        assert event.order_hash == row["orderHash"]


def test_parser_handles_v1_sell_rows_from_production_fixture():
    """V1 SELL rows: takerAssetId='0', makerAssetId=<token>, side='sell'."""
    data = _load_fixture()
    sells = data["v1_sell_rows"]
    assert len(sells) >= 1, "fixture must contain at least one V1 SELL row"
    for row in sells:
        assert row["takerAssetId"] == "0"
        assert row["side"] == "sell"
        event = subgraph_v1_row_to_event(row)
        assert event.taker_asset_id == 0
        assert event.maker_asset_id == int(row["makerAssetId"])
        assert event.making == int(row["makerAmountFilled"])
        assert event.taking == int(row["takerAmountFilled"])


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
    assert event.block_number == 0
    assert event.log_index == 0


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
    assert event.block_number == 0
    assert event.log_index == 0


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
    with pytest.raises(ValueError, match="both-zero or both-non-zero"):
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
    with pytest.raises(ValueError, match="both-zero or both-non-zero"):
        subgraph_v1_row_to_event(row)


# ---------------------------------------------------------------------------
# Paginator + iterator tests
# ---------------------------------------------------------------------------


from pscanner.corpus.subgraph_ingest_v1 import iter_v1_market_trades  # noqa: E402


class _FakeSubgraphClient(SubgraphClient):
    """Records every query() invocation and yields canned responses in order.

    Subclasses SubgraphClient (without calling super().__init__()) so it
    satisfies the parameter type at call sites. Never opens a real httpx
    client.

    `pages_by_query` is keyed by 'maker' (queries containing
    'makerAssetId_in') or 'taker' (queries containing 'takerAssetId_in').
    """

    def __init__(self, pages_by_query: dict[str, list[list[dict[str, Any]]]]) -> None:
        # NOTE: deliberately not calling super().__init__() — we don't want
        # to construct the underlying RateLimitedHttpClient. ty accepts the
        # subclass as a valid SubgraphClient regardless of init state.
        self._pages = pages_by_query
        self.calls: list[dict[str, Any]] = []

    async def query(self, graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        side = "maker" if "makerAssetId_in" in graphql else "taker"
        self.calls.append({"side": side, "variables": dict(variables)})
        pages = self._pages.get(side, [])
        if not pages:
            return {"orderFilledEvents": []}
        return {"orderFilledEvents": pages.pop(0)}


def _buy_row(idx: int, asset_id: str) -> dict[str, str]:
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


def _sell_row(idx: int, asset_id: str) -> dict[str, str]:
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
        async for ev, ts in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=2)
    ]
    assert out == []
    sides = {c["side"] for c in client.calls}
    assert sides == {"maker", "taker"}


@pytest.mark.asyncio
async def test_paginator_yields_union_of_buy_and_sell_passes():
    buys = [_buy_row(i, "100") for i in range(3)]
    sells = [_sell_row(10 + i, "100") for i in range(2)]
    client = _FakeSubgraphClient(
        pages_by_query={
            "maker": [sells],  # maker query catches SELL rows (maker holds CTF)
            "taker": [buys],  # taker query catches BUY rows (taker holds CTF)
        }
    )
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=10)
    ]
    assert len(out) == 5


@pytest.mark.asyncio
async def test_paginator_advances_cursor_per_side():
    buys = [_buy_row(i, "100") for i in range(5)]
    client = _FakeSubgraphClient(
        pages_by_query={
            "maker": [],
            "taker": [buys[0:2], buys[2:4], buys[4:5]],
        }
    )
    out = [
        (ev, ts)
        async for ev, ts in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=2)
    ]
    assert len(out) == 5
    taker_calls = [c for c in client.calls if c["side"] == "taker"]
    assert len(taker_calls) == 3
    assert taker_calls[0]["variables"]["cursor"] == ""
    assert taker_calls[1]["variables"]["cursor"] == "tx-1_hash-1"
    assert taker_calls[2]["variables"]["cursor"] == "tx-3_hash-3"


@pytest.mark.asyncio
async def test_paginator_rejects_invalid_page_size():
    client = _FakeSubgraphClient(pages_by_query={})
    with pytest.raises(ValueError, match="page_size"):
        async for _ in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=0):
            pass
    with pytest.raises(ValueError, match="page_size"):
        async for _ in iter_v1_market_trades(client=client, asset_ids=["100"], page_size=1001):
            pass


@pytest.mark.asyncio
async def test_paginator_short_circuits_on_empty_asset_ids():
    client = _FakeSubgraphClient(pages_by_query={})
    out = [x async for x in iter_v1_market_trades(client=client, asset_ids=[], page_size=10)]
    assert out == []
    assert client.calls == []


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

import sqlite3  # noqa: E402

import pytest  # noqa: E402 — already imported above but needed for clarity

from pscanner.corpus.db import init_corpus_db  # noqa: E402
from pscanner.corpus.repos import (  # noqa: E402
    AssetEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
)
from pscanner.corpus.subgraph_ingest_v1 import run_v1_subgraph_backfill  # noqa: E402


def _fat_buy_row(idx: int, asset_id: str) -> dict[str, str]:
    """BUY row with $20 USDC notional — clears the $10 insert floor."""
    return {
        "id": f"tx-{idx}_hash-{idx}",
        "transactionHash": "0x" + str(idx).rjust(64, "0"),
        "timestamp": str(1_700_000_000 + idx),
        "orderHash": "0x" + str(idx).rjust(64, "1"),
        "maker": "0x" + "b" * 40,
        "taker": "0x" + "c" * 40,
        "makerAssetId": "0",
        "takerAssetId": asset_id,
        "makerAmountFilled": "20000000",  # $20 USDC in 6-decimal base units
        "takerAmountFilled": "40000000",  # 40 CTF shares → price = 0.5
        "fee": "0",
        "side": "buy",
        "price": "0.5",
    }


def _seed_v1_pending_market(
    conn: sqlite3.Connection,
    condition_id: str,
    asset_id: str,
    *,
    closed_at: int = 1_760_000_000,  # 2025-10-09, post-V1-earliest (1744013119)
) -> None:
    market = CorpusMarket(
        condition_id=condition_id,
        event_slug="test-event",
        category=None,
        closed_at=closed_at,
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
        AssetEntry(
            asset_id=asset_id,
            condition_id=condition_id,
            outcome_side="YES",
            outcome_index=0,
        )
    )
    conn.commit()


@pytest.mark.asyncio
async def test_orchestrator_drains_one_market_and_stamps_sentinel(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        cid = "0x" + "a" * 64
        aid = "1234567890"
        _seed_v1_pending_market(conn, cid, aid)
        rows = [_fat_buy_row(0, aid), _fat_buy_row(1, aid)]
        client = _FakeSubgraphClient(pages_by_query={"maker": [], "taker": [rows]})
        summary = await run_v1_subgraph_backfill(
            conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
        )
        assert summary.markets_processed == 1
        assert summary.markets_no_new_trades == 0
        assert summary.markets_failed == 0
        assert summary.events_decoded == 2
        assert summary.trades_inserted == 2
        row = conn.execute(
            "SELECT onchain_v1_processed_at, v1_history_pending FROM corpus_markets"
            " WHERE condition_id = ?",
            (cid,),
        ).fetchone()
        assert row["onchain_v1_processed_at"] == 1_700_000_999
        assert row["v1_history_pending"] == 0
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_orchestrator_does_not_stamp_on_zero_events(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        cid = "0x" + "b" * 64
        aid = "9999999999"
        _seed_v1_pending_market(conn, cid, aid)
        client = _FakeSubgraphClient(pages_by_query={})
        summary = await run_v1_subgraph_backfill(
            conn=conn, client=client, page_size=1000, limit=None, now_ts=1_700_000_999
        )
        assert summary.markets_no_new_trades == 1
        assert summary.markets_processed == 0
        row = conn.execute(
            "SELECT onchain_v1_processed_at, v1_history_pending FROM corpus_markets"
            " WHERE condition_id = ?",
            (cid,),
        ).fetchone()
        assert row["onchain_v1_processed_at"] is None
        assert row["v1_history_pending"] == 1
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_orchestrator_skips_market_with_empty_asset_index(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
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
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_orchestrator_respects_limit(tmp_path: Path):
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        for i in range(3):
            _seed_v1_pending_market(conn, "0x" + chr(0x61 + i) * 64, f"100{i}")
        client = _FakeSubgraphClient(
            pages_by_query={"maker": [], "taker": [[_fat_buy_row(0, "1000")]]}
        )
        summary = await run_v1_subgraph_backfill(
            conn=conn, client=client, page_size=1000, limit=1, now_ts=1_700_000_999
        )
        assert summary.markets_processed + summary.markets_no_new_trades == 1
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_hybrid_market_sets_both_sentinels(tmp_path: Path):
    """V2 ran first (sets onchain_processed_at), then V1 runs (sets _v1 column).

    Both sentinels end up populated; v1_history_pending flips to 0; V2's
    sentinel value is unchanged.
    """
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        cid = "0x" + "f" * 64
        aid = "7777777777"
        _seed_v1_pending_market(conn, cid, aid)
        # Simulate V2 already having processed this market.
        conn.execute(
            "UPDATE corpus_markets SET onchain_processed_at = ? WHERE condition_id = ?",
            (1_600_000_000, cid),
        )
        conn.commit()

        rows = [_fat_buy_row(0, aid)]
        client = _FakeSubgraphClient(pages_by_query={"maker": [], "taker": [rows]})
        await run_v1_subgraph_backfill(
            conn=conn,
            client=client,
            page_size=1000,
            limit=None,
            now_ts=1_700_000_999,
        )

        row = conn.execute(
            """
            SELECT onchain_processed_at, onchain_v1_processed_at, v1_history_pending
            FROM corpus_markets WHERE condition_id = ?
            """,
            (cid,),
        ).fetchone()
        assert row["onchain_processed_at"] == 1_600_000_000  # untouched by V1
        assert row["onchain_v1_processed_at"] == 1_700_000_999  # set by V1
        assert row["v1_history_pending"] == 0  # cleared by V1
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_orchestrator_skips_pre_v1_coverage_markets(tmp_path: Path) -> None:
    """Markets with closed_at < V1 earliest are excluded from the queue (#197).

    The V1 subgraph's earliest indexed event is at 1744013119 (2025-04-07).
    Markets that closed before that have no V1 history to serve and were
    wasting ~60-90 sec each on empty `id_gt` scans before this filter.
    """
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        # 1: pre-V1 market — should be filtered out.
        _seed_v1_pending_market(
            conn, "0x" + "a" * 64, "100", closed_at=_V1_SUBGRAPH_EARLIEST_TS - 1
        )
        # 2: at V1 earliest — should be included (boundary).
        _seed_v1_pending_market(
            conn, "0x" + "b" * 64, "200", closed_at=_V1_SUBGRAPH_EARLIEST_TS
        )
        # 3: post-V1 market — should be included.
        _seed_v1_pending_market(
            conn, "0x" + "c" * 64, "300", closed_at=_V1_SUBGRAPH_EARLIEST_TS + 86400
        )

        pending = _load_pending_v1_markets(conn, limit=None)
        cids = {m.condition_id for m in pending}
        assert cids == {"0x" + "b" * 64, "0x" + "c" * 64}
        assert _count_pre_v1_skipped(conn) == 1
    finally:
        conn.close()
