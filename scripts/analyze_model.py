r"""Diagnostics for a trained copy-trade gate model.

Loads a model artifact directory (``model.json`` + ``preprocessor.json``)
and the corpus, replicates the temporal test split, then prints:

* Test AUC
* Top-N features by xgboost gain
* Top-N features by mean(|SHAP|) — global importance via ``pred_contribs``
* Per-``top_category`` accuracy and realized-edge breakdown

Doesn't pull in the ``shap`` package — uses xgboost's native ``pred_contribs``
which gives the same global mean(|SHAP|) without the extra dependency.

Usage:
    uv run python scripts/analyze_model.py \
        --model models/2026-05-03b-copy_trade_gate-real_temporal \
        --db data/corpus.sqlite3
"""

# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from pscanner.ml.preprocessing import OneHotEncoder
from pscanner.ml.streaming import open_dataset

_BINARY_DECISION_THRESHOLD = 0.5


def _auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Area under ROC curve via the Mann-Whitney U statistic.

    Handles ties on prediction value by averaging ranks (matches sklearn's
    `roc_auc_score` to >=4 decimal places on smooth predictors).
    """
    n_pos = int((y_true == 1).sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_pred_proba, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_pred_proba) + 1, dtype=np.float64)
    # average ranks within ties on the prediction
    sorted_pred = y_pred_proba[order]
    i = 0
    n = len(sorted_pred)
    while i < n:
        j = i
        while j + 1 < n and sorted_pred[j + 1] == sorted_pred[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[y_true == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _print_feature_importance_gain(booster: xgb.Booster, top_k: int) -> None:
    """Print top-K features by xgboost native gain importance."""
    scores = booster.get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    print(f"\nTop {top_k} features by gain:")
    for i, (feat, score) in enumerate(ranked, 1):
        print(f"  {i:>2}. {feat:<50} gain={score:>10.3f}")


def _print_shap_importance(
    booster: xgb.Booster,
    dmatrix: xgb.DMatrix,
    feature_names: list[str],
    top_k: int,
) -> None:
    """Print top-K features by mean(|SHAP|).

    Uses xgboost's `pred_contribs=True` to get per-row, per-feature
    SHAP-style contributions without the `shap` package. Last column of
    the returned array is the global bias term, dropped here.
    """
    contribs = booster.predict(dmatrix, pred_contribs=True)
    contribs_features = contribs[:, :-1]
    mean_abs = np.abs(contribs_features).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    print(f"\nTop {top_k} features by mean(|SHAP|):")
    for i, idx in enumerate(order, 1):
        print(f"  {i:>2}. {feature_names[idx]:<50} mean|shap|={mean_abs[idx]:>10.5f}")


def _print_per_category_breakdown(
    top_categories: list[str],
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    implied_prob: np.ndarray,
) -> None:
    """Print accuracy and realized edge stratified by top_category."""
    cat_array = np.array(top_categories, dtype=object)
    unique_cats = sorted({c for c in top_categories if c is not None})
    print("\nPer-category breakdown:")
    print(
        f"  {'category':<20} {'n':>10} {'won_rate':>10} {'accuracy':>10} "
        f"{'n_taken':>10} {'edge':>10}"
    )
    for cat in unique_cats:
        mask = cat_array == cat
        n = int(mask.sum())
        if n == 0:
            continue
        won_rate = float(y_true[mask].mean())
        pred_class = (y_pred_proba[mask] >= _BINARY_DECISION_THRESHOLD).astype(int)
        accuracy = float((pred_class == y_true[mask]).mean())
        # realized edge over the subset's taken bets
        take = y_pred_proba[mask] > implied_prob[mask]
        n_taken = int(take.sum())
        if n_taken > 0:
            edge = float((y_true[mask][take] - implied_prob[mask][take]).mean())
            edge_str = f"{edge:>10.4f}"
        else:
            edge_str = f"{'-':>10}"
        print(f"  {cat:<20} {n:>10,} {won_rate:>10.4f} {accuracy:>10.4f} {n_taken:>10,} {edge_str}")


def analyze(model_dir: Path, db_path: Path, top_k: int, platform: str = "polymarket") -> None:
    """End-to-end analysis: replicate test split, predict, print diagnostics."""
    print(f"Loading model from {model_dir}")
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))
    encoder_payload = json.loads((model_dir / "preprocessor.json").read_text())
    encoder = OneHotEncoder.from_json({"levels": encoder_payload["encoder"]["levels"]})

    print(f"Loading corpus from {db_path} (platform={platform})")
    with open_dataset(db_path, platform=platform) as ds:
        if ds.encoder is None:
            raise RuntimeError("open_dataset did not fit the encoder")
        # Sanity-check: encoder fit on this corpus should match the
        # encoder serialized into preprocessor.json. A mismatch implies
        # the corpus drifted since the model was trained.
        if ds.encoder.levels != encoder.levels:
            print(
                "WARN: encoder levels in corpus differ from preprocessor.json — "
                "model may be stale relative to the current corpus."
            )
        feature_cols = list(ds.feature_names)
        test = ds.materialize_test()

    x_test = test.x
    y_test = test.y
    implied_test = test.implied_prob
    top_categories = test.top_categories.tolist()
    print(f"Test split: {x_test.shape[0]:,} rows, {x_test.shape[1]} columns")
    print(f"Feature matrix: {x_test.shape}")

    dtest = xgb.DMatrix(x_test, feature_names=feature_cols)
    p_test = booster.predict(dtest)

    # Headline metrics
    auc = _auc(y_test, p_test)
    pred_class = (p_test >= _BINARY_DECISION_THRESHOLD).astype(int)
    accuracy = float((pred_class == y_test).mean())
    take = p_test > implied_test
    edge = (
        float((y_test[take] - implied_test[take]).mean()) if int(take.sum()) > 0 else float("nan")
    )

    print("\n=== Headline ===")
    print(f"  test rows:        {len(y_test):,}")
    print(f"  test won_rate:    {float(y_test.mean()):.4f}")
    print(f"  test accuracy:    {accuracy:.4f}")
    print(f"  test AUC:         {auc:.4f}")
    print(f"  bets taken:       {int(take.sum()):,}  ({take.mean() * 100:.1f}% of test)")
    print(f"  test edge:        {edge:.4f}")

    _print_feature_importance_gain(booster, top_k=top_k)
    _print_shap_importance(booster, dtest, feature_names=feature_cols, top_k=top_k)
    _print_per_category_breakdown(
        top_categories=top_categories,
        y_true=y_test,
        y_pred_proba=p_test,
        implied_prob=implied_test,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the per-run artifact directory (contains model.json, preprocessor.json)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/corpus.sqlite3",
        help="Path to the corpus SQLite database",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top features to show in importance breakdowns",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["polymarket", "kalshi", "manifold"],
        default="polymarket",
        help="Filter to rows with this platform tag (matches `pscanner ml train --platform`).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: parse args, run analysis."""
    args = _parse_args()
    analyze(Path(args.model), Path(args.db), top_k=args.top_k, platform=args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
