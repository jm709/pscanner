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
