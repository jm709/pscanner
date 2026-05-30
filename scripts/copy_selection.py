"""Causal copy-trading wallet selection precompute (DuckDB-backed).

Qualifies wallets at >= min_resolved resolved trades, ranks by causal
(no-lookahead) edge, freezes a global top-K copy set per rebalance
boundary, and emits the selected trades + their resolutions as event
rows for scripts.backtest_copy_sizing's Simulator.

Spec: docs/superpowers/specs/2026-05-30-causal-copy-selection-design.md
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class KPolicy:
    """Top-K policy for the copy set. Exactly one field is set."""

    top_k: int | None = None
    capital_per_wallet: float | None = None
    top_frac: float | None = None


def resolve_k(policy: KPolicy, *, bankroll: float, qualified_count: int) -> int:
    """Return the top-K cut for one rebalance boundary.

    Args:
        policy: which sizing rule to apply.
        bankroll: constant starting bankroll (USD).
        qualified_count: number of qualified wallets at this boundary.
    """
    if policy.top_k is not None:
        return policy.top_k
    if policy.capital_per_wallet is not None:
        return max(0, int(bankroll // policy.capital_per_wallet))
    if policy.top_frac is not None:
        return math.ceil(policy.top_frac * qualified_count)
    raise ValueError("KPolicy has no mode set")
