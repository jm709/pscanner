"""Tests for ml.metrics."""

from __future__ import annotations

import numpy as np

from pscanner.ml.metrics import per_decile_edge_breakdown, realized_edge_metric


def test_returns_minus_one_below_n_min() -> None:
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.9, 0.9])
    implied = np.array([0.5, 0.5, 0.5])
    assert realized_edge_metric(y, p, implied, n_min=10) == -1.0


def test_mean_realized_edge_over_taken_bets() -> None:
    # Three bets total. Take only the two where p > implied.
    y = np.array([1, 0, 1])
    p = np.array([0.7, 0.3, 0.8])
    implied = np.array([0.5, 0.5, 0.5])
    # Taken: indices 0 and 2. realized edges: (1 - 0.5)=0.5, (1 - 0.5)=0.5.
    assert realized_edge_metric(y, p, implied, n_min=2) == 0.5


def test_negative_edge_when_model_wrong() -> None:
    y = np.array([0, 0, 0])
    p = np.array([0.9, 0.9, 0.9])
    implied = np.array([0.5, 0.5, 0.5])
    # Take all three; realized edges all (0 - 0.5)=-0.5.
    assert realized_edge_metric(y, p, implied, n_min=3) == -0.5


def test_no_taken_bets_returns_minus_one() -> None:
    y = np.array([1, 1, 1])
    p = np.array([0.1, 0.2, 0.3])
    implied = np.array([0.5, 0.5, 0.5])
    # Nothing passes p > implied.
    assert realized_edge_metric(y, p, implied, n_min=1) == -1.0


def test_per_decile_breakdown_groups_by_implied_prob() -> None:
    # 20 trades spread across two implied-prob deciles.
    # First 10: implied=0.05 (decile 0). Take all (p=0.9). 7 wins.
    # Next 10: implied=0.55 (decile 5). Take all (p=0.9). 9 wins.
    y = np.array([1] * 7 + [0] * 3 + [1] * 9 + [0])
    p = np.array([0.9] * 20)
    implied = np.array([0.05] * 10 + [0.55] * 10)
    result = per_decile_edge_breakdown(y, p, implied)
    # Decile 0: 7/10 wins, mean edge = 0.7 - 0.05 = 0.65
    assert result["0.0-0.1"]["n"] == 10
    assert result["0.0-0.1"]["mean_edge"] == 0.65
    # Decile 5: 9/10 wins, mean edge = 0.9 - 0.55 = 0.35
    assert result["0.5-0.6"]["n"] == 10
    assert abs(result["0.5-0.6"]["mean_edge"] - 0.35) < 1e-9


def test_per_decile_skips_empty_deciles() -> None:
    y = np.array([1, 1])
    p = np.array([0.9, 0.9])
    implied = np.array([0.05, 0.05])
    result = per_decile_edge_breakdown(y, p, implied)
    assert "0.0-0.1" in result
    assert "0.5-0.6" not in result


def test_per_decile_only_counts_taken_bets() -> None:
    # First bet not taken (p < implied), second taken.
    y = np.array([1, 1])
    p = np.array([0.01, 0.9])
    implied = np.array([0.05, 0.05])
    result = per_decile_edge_breakdown(y, p, implied)
    # Both in decile 0, but only second is taken.
    assert result["0.0-0.1"]["n"] == 1
    assert abs(result["0.0-0.1"]["mean_edge"] - (1 - 0.05)) < 1e-9
