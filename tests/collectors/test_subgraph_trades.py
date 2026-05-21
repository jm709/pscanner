"""Unit tests for SubgraphTradeCollector internals (#152)."""

from __future__ import annotations

from pscanner.collectors.subgraph_trades import (
    DETECTOR_TAG,
    SUBGRAPH_ID,
    _build_where_clause,
    _compute_copy_direction,
    _serialize_where_inline,
)


def test_compute_copy_direction_maker_buy() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=0, watchlist={"0xaa"}
    )
    assert direction == "BUY"


def test_compute_copy_direction_taker_sell_is_buy() -> None:
    # watchlist == taker AND side == 1 -> taker bought (hit a sell order).
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=1, watchlist={"0xbb"}
    )
    assert direction == "BUY"


def test_compute_copy_direction_maker_sell_is_skip() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=1, watchlist={"0xaa"}
    )
    assert direction == "SKIP"


def test_compute_copy_direction_taker_buy_is_skip() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=0, watchlist={"0xbb"}
    )
    assert direction == "SKIP"


def test_compute_copy_direction_neither_side_watched() -> None:
    direction = _compute_copy_direction(
        maker="0xAA", taker="0xBB", side=0, watchlist={"0xcc"}
    )
    assert direction == "SKIP"


def test_serialize_where_inline_renders_object_literal() -> None:
    rendered = _serialize_where_inline(
        {"timestamp_gte": "100", "maker_in": ["0xaa", "0xbb"]}
    )
    # GraphQL object literals do NOT quote keys.
    assert rendered == '{timestamp_gte:"100",maker_in:["0xaa","0xbb"]}'


def test_build_where_clause_repeats_timestamp_inside_or() -> None:
    where = _build_where_clause(["0xaa", "0xbb"], 1_700_000_000)
    assert where == {
        "or": [
            {"timestamp_gte": "1700000000", "maker_in": ["0xaa", "0xbb"]},
            {"timestamp_gte": "1700000000", "taker_in": ["0xaa", "0xbb"]},
        ],
    }


def test_constants_exposed_for_callers() -> None:
    assert DETECTOR_TAG == "subgraph_copy"
    assert SUBGRAPH_ID == "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
