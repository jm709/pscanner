"""Tests for ml.metrics."""

from __future__ import annotations

import numpy as np

from pscanner.ml.metrics import realized_edge_metric


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
