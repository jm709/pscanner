"""Tests for the pscanner ml CLI parser."""

from __future__ import annotations

import sqlite3

import pytest

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


def test_train_parser_accepts_platform_flag() -> None:
    parser = build_ml_parser()
    args = parser.parse_args(["train", "--platform", "kalshi", "--db", "x"])
    assert args.platform == "kalshi"


def test_train_parser_default_platform_is_polymarket() -> None:
    parser = build_ml_parser()
    args = parser.parse_args(["train", "--db", "x"])
    assert args.platform == "polymarket"


def test_train_parser_rejects_unknown_platform() -> None:
    parser = build_ml_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--platform", "ftx", "--db", "x"])


def test_make_synthetic_examples_db_accepts_platform_kwarg(
    make_synthetic_examples_db,  # type: ignore[no-untyped-def]
) -> None:
    db_path = make_synthetic_examples_db(n_markets=2, rows_per_market=2, seed=0, platform="kalshi")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT DISTINCT platform FROM training_examples").fetchall()
    finally:
        conn.close()
    assert rows == [("kalshi",)]
