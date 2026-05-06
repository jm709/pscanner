"""One-off capture of eager-path metrics for the streaming-vs-eager parity test.

Runs ``run_study`` with the EAGER pipeline (load_dataset + temporal_split)
on a deterministic synthetic fixture. Saves test_edge / test_accuracy /
test_logloss to tests/ml/data/eager_baseline.json.

This script + the JSON file get deleted in the same commit that deletes
load_dataset (Task 14). Until then, the JSON is the regression baseline
the streaming path must match within ≤ 0.001 absolute.

Run:
    uv run python -m tests.ml._capture_eager_baseline
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from pscanner.ml.training import run_study
from tests.ml.conftest import _make_synthetic_examples


def main() -> None:
    np.random.seed(42)
    df = _make_synthetic_examples(n_markets=30, rows_per_market=20, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        run_study(
            df=df,
            output_dir=out,
            n_trials=3,
            n_jobs=1,
            n_min=5,
            seed=42,
        )
        metrics = json.loads((out / "metrics.json").read_text())

    baseline = {
        "fixture": {"n_markets": 30, "rows_per_market": 20, "seed": 3},
        "study": {"n_trials": 3, "n_jobs": 1, "n_min": 5, "seed": 42},
        "test_edge": metrics["test_edge"],
        "test_accuracy": metrics["test_accuracy"],
        "test_logloss": metrics["test_logloss"],
    }
    target = Path(__file__).parent / "data" / "eager_baseline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
