"""argparse handler for ``pscanner ml {train}``.

Mirrors the structure of ``pscanner.corpus.cli`` but synchronous —
training has no network I/O.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import structlog

from pscanner.ml.preprocessing import load_dataset
from pscanner.ml.training import _rss_mb, run_study

_log = structlog.get_logger(__name__)


def build_ml_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ml subcommand group."""
    parser = argparse.ArgumentParser(prog="pscanner ml")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train an XGBoost copy-trade gate model")
    train.add_argument("--n-trials", type=int, default=100, help="Optuna trial budget")
    train.add_argument("--seed", type=int, default=42, help="RNG seed")
    train.add_argument(
        "--n-min", type=int, default=20, help="Min copied bets for the edge metric guard"
    )
    train.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help=(
            "Parallel Optuna trials. Each trial allocates ~0.5 GB of XGBoost "
            "scratch (predict buffers, OpenMP) on top of the shared DMatrix. "
            "On the dev host (~7.6 GB) 2-3 is the safe default; raise after "
            "verifying headroom with `free -h` mid-run."
        ),
    )
    train.add_argument(
        "--db",
        type=str,
        default="data/corpus.sqlite3",
        help="Path to the corpus SQLite database",
    )
    train.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Per-run artifact directory (default: models/<YYYY-MM-DD>-copy_trade_gate)",
    )
    train.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="XGBoost device. ``cuda`` requires an NVIDIA GPU visible to the process.",
    )
    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    """Run the training pipeline end-to-end."""
    db_path = Path(args.db)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        today = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
        output_dir = Path("models") / f"{today}-copy_trade_gate"
    df = load_dataset(db_path)
    _log.info(
        "ml.dataset_loaded",
        rows=df.height,
        cols=len(df.columns),
        output_dir=str(output_dir),
        rss_mb=_rss_mb(),
    )
    run_study(
        df=df,
        output_dir=output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        n_min=args.n_min,
        seed=args.seed,
        device=args.device,
    )
    return 0


_HANDLERS = {"train": _cmd_train}


def run_ml_command(argv: list[str]) -> int:
    """Parse ``argv`` (excluding the leading ``ml``) and dispatch."""
    parser = build_ml_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS[args.command]
    return handler(args)
