"""Tests for the causal copy-selection precompute."""

from __future__ import annotations

import pytest

from scripts.copy_selection import KPolicy, resolve_k


def test_resolve_k_fixed_count() -> None:
    assert resolve_k(KPolicy(top_k=25), bankroll=10_000.0, qualified_count=500) == 25


def test_resolve_k_capital_per_wallet_floors() -> None:
    # 10_000 / 750 = 13.33 -> 13
    assert resolve_k(KPolicy(capital_per_wallet=750.0), bankroll=10_000.0,
                     qualified_count=500) == 13


def test_resolve_k_top_frac_ceils_against_qualified() -> None:
    # ceil(0.1 * 95) = 10
    assert resolve_k(KPolicy(top_frac=0.1), bankroll=10_000.0, qualified_count=95) == 10


def test_resolve_k_top_frac_zero_qualified_is_zero() -> None:
    assert resolve_k(KPolicy(top_frac=0.1), bankroll=10_000.0, qualified_count=0) == 0


def test_resolve_k_no_mode_raises() -> None:
    with pytest.raises(ValueError, match="no mode"):
        resolve_k(KPolicy(), bankroll=10_000.0, qualified_count=10)
