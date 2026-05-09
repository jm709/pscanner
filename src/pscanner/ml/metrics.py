"""Edge-based metrics for copy-trade gate model evaluation.

The optimization target is ``realized_edge_metric`` — mean realized
edge across bets the model would copy. ``per_decile_edge_breakdown``
is a diagnostic stratification by ``implied_prob_at_buy`` decile.
"""

from __future__ import annotations

import numpy as np

_LAST_DECILE = 9


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


def per_decile_edge_breakdown(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Stratify realized edge over taken bets by implied-prob decile.

    Diagnostic only — not the optimization target. Reveals whether
    edge is concentrated in cheap-side bets (longshot finder) or
    distributed across the implied-prob range (mispricing detector).

    Args:
        y_true: 1D array of binary labels.
        y_pred_proba: 1D array of model probabilities.
        implied_prob: 1D array of implied probabilities at trade time.

    Returns:
        Mapping from decile label (e.g. ``"0.0-0.1"``) to
        ``{"n": <count>, "mean_edge": <mean_realized_edge>}``. Deciles
        with zero taken bets are omitted.
    """
    take = y_pred_proba > implied_prob
    out: dict[str, dict[str, float]] = {}
    for decile in range(10):
        lo = decile / 10
        hi = (decile + 1) / 10
        if decile < _LAST_DECILE:
            in_decile = (implied_prob >= lo) & (implied_prob < hi)
        else:
            in_decile = (implied_prob >= lo) & (implied_prob <= hi)
        mask = take & in_decile
        n = int(mask.sum())
        if n == 0:
            continue
        mean_edge = float((y_true[mask] - implied_prob[mask]).mean())
        label = f"{lo:.1f}-{hi:.1f}"
        out[label] = {"n": float(n), "mean_edge": mean_edge}
    return out


_VOLUME_BUCKETS_USD: tuple[tuple[str, float, float], ...] = (
    ("<250K", 0.0, 250_000.0),
    ("250K-1M", 250_000.0, 1_000_000.0),
    ("1M-5M", 1_000_000.0, 5_000_000.0),
    ("5M-25M", 5_000_000.0, 25_000_000.0),
    ("25M+", 25_000_000.0, float("inf")),
)


def per_volume_bucket_edge_breakdown(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
    total_volume_usd: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Stratify realized edge over taken bets by market lifetime volume.

    Diagnostic for issue #109. The corpus floor was lowered from $1M to
    $100K for esports; this metric tells us whether the newly-included
    sub-$1M esports cohort carries the same edge as the established $1M+
    cohort or contributes noise.

    Buckets are closed-open (``[lo, hi)``) except the top bucket which is
    closed-closed (``[lo, inf]``). Buckets with zero taken bets are omitted.

    Args:
        y_true: 1D array of binary labels.
        y_pred_proba: 1D array of model probabilities.
        implied_prob: 1D array of implied probabilities at trade time.
        total_volume_usd: 1D array of per-row market lifetime volume.

    Returns:
        Mapping from bucket label (e.g. ``"1M-5M"``) to
        ``{"n": <count>, "mean_edge": <mean_realized_edge>}``.
    """
    take = y_pred_proba > implied_prob
    out: dict[str, dict[str, float]] = {}
    for label, lo, hi in _VOLUME_BUCKETS_USD:
        if hi == float("inf"):
            in_bucket = total_volume_usd >= lo
        else:
            in_bucket = (total_volume_usd >= lo) & (total_volume_usd < hi)
        mask = take & in_bucket
        n = int(mask.sum())
        if n == 0:
            continue
        mean_edge = float((y_true[mask] - implied_prob[mask]).mean())
        out[label] = {"n": float(n), "mean_edge": mean_edge}
    return out
