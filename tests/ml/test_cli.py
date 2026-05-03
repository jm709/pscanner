"""Tests for the pscanner ml CLI parser."""

from __future__ import annotations

from pscanner.ml.cli import build_ml_parser


def test_train_subcommand_defaults() -> None:
    parser = build_ml_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.n_trials == 100
    assert args.seed == 42
    assert args.n_min == 20
    assert args.n_jobs == 2
    assert args.db == "data/corpus.sqlite3"


def test_train_subcommand_overrides() -> None:
    parser = build_ml_parser()
    args = parser.parse_args(
        [
            "train",
            "--n-trials",
            "5",
            "--seed",
            "7",
            "--n-min",
            "1",
            "--n-jobs",
            "2",
            "--db",
            "./scratch/x.sqlite3",
            "--output-dir",
            "./scratch/out",
        ]
    )
    assert args.n_trials == 5
    assert args.seed == 7
    assert args.n_min == 1
    assert args.n_jobs == 2
    assert args.db == "./scratch/x.sqlite3"
    assert args.output_dir == "./scratch/out"
