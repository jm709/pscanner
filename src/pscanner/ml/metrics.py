"""Edge-based metrics for copy-trade gate model evaluation.

The optimization target is ``realized_edge_metric`` — mean realized
edge across bets the model would copy. ``per_decile_edge_breakdown``
is a diagnostic stratification by ``implied_prob_at_buy`` decile.
"""

from __future__ import annotations

import numpy as np


def realized_edge_metric(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
    n_min: int = 20,
) -> float:
    """Return mean realized edge over bets the model would copy.

    A bet is "copied" iff ``y_pred_proba > implied_prob`` (the model
    predicts positive expected edge). Realized edge per copied bet is
    ``label_won - implied_prob_at_buy``.

    Args:
        y_true: 1D array of binary labels (``label_won``).
        y_pred_proba: 1D array of model probabilities for ``label_won=1``.
        implied_prob: 1D array of implied probabilities at trade time.
        n_min: Anti-overfit guard. If fewer than ``n_min`` bets pass
            the gate, return ``-1.0`` so trial configurations that
            overfit to a tiny lucky subset are penalized.

    Returns:
        Mean realized edge over taken bets, or ``-1.0`` if too few.
    """
    take = y_pred_proba > implied_prob
    if int(take.sum()) < n_min:
        return -1.0
    return float((y_true[take] - implied_prob[take]).mean())
