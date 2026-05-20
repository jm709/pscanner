"""Unit tests for scripts/watch_subgraph_copy.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "watch_subgraph_copy.py"
_spec = importlib.util.spec_from_file_location("watch_subgraph_copy", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
watch_subgraph_copy = importlib.util.module_from_spec(_spec)
sys.modules["watch_subgraph_copy"] = watch_subgraph_copy
_spec.loader.exec_module(watch_subgraph_copy)


# A wallet on our watchlist, two non-watchlisted counterparties for clarity.
_WATCH = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_OTHER1 = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_OTHER2 = "0xcccccccccccccccccccccccccccccccccccccccc"


@pytest.mark.parametrize(
    ("maker", "taker", "side", "expected"),
    [
        # maker=watchlist, side=0 (BUY)  -> maker accumulates -> COPY BUY
        (_WATCH, _OTHER1, 0, "BUY"),
        # maker=watchlist, side=1 (SELL) -> maker reduces -> SKIP
        (_WATCH, _OTHER1, 1, "SKIP"),
        # taker=watchlist, side=0 (taker hit a buy order -> sold) -> taker reduces -> SKIP
        (_OTHER1, _WATCH, 0, "SKIP"),
        # taker=watchlist, side=1 (taker hit a sell order -> bought) -> taker accumulates -> BUY
        (_OTHER1, _WATCH, 1, "BUY"),
        # Neither on the watchlist (shouldn't happen at the call site, but defensive)
        (_OTHER1, _OTHER2, 0, "SKIP"),
        (_OTHER1, _OTHER2, 1, "SKIP"),
    ],
)
def test_compute_copy_direction(maker: str, taker: str, side: int, expected: str) -> None:
    watchlist = {_WATCH}
    result = watch_subgraph_copy._compute_copy_direction(maker, taker, side, watchlist)
    assert result == expected


def test_build_where_clause_emits_or_with_per_branch_timestamp() -> None:
    addrs = ["0xaaa", "0xbbb"]
    last_seen_ts = 1779225600
    where = watch_subgraph_copy._build_where_clause(addrs, last_seen_ts)

    # Top-level must NOT have timestamp_gte alongside `or` — TheGraph rejects that.
    assert "timestamp_gte" not in where
    assert "or" in where
    branches = where["or"]
    assert len(branches) == 2
    # Each branch carries the timestamp filter and one of maker/taker filters.
    maker_branches = [b for b in branches if "maker_in" in b]
    taker_branches = [b for b in branches if "taker_in" in b]
    assert len(maker_branches) == 1
    assert len(taker_branches) == 1
    assert maker_branches[0]["maker_in"] == addrs
    assert maker_branches[0]["timestamp_gte"] == str(last_seen_ts)
    assert taker_branches[0]["taker_in"] == addrs
    assert taker_branches[0]["timestamp_gte"] == str(last_seen_ts)
