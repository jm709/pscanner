"""Tests for the `pscanner corpus` CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pscanner.corpus.cli import build_corpus_parser, run_corpus_command


def test_parser_recognises_three_commands() -> None:
    parser = build_corpus_parser()
    assert parser.parse_args(["backfill"]).command == "backfill"
    assert parser.parse_args(["refresh"]).command == "refresh"
    assert parser.parse_args(["build-features"]).command == "build-features"


def test_parser_supports_rebuild_flag() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--rebuild"])
    assert args.rebuild is True


@pytest.mark.asyncio
async def test_backfill_command_smokes(tmp_path: Path) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    fake_enumerate = AsyncMock(return_value=0)
    fake_drain = AsyncMock(return_value=0)
    fake_data_cm = MagicMock()
    fake_data_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
    fake_data_cm.__aexit__ = AsyncMock(return_value=None)
    fake_gamma_cm = MagicMock()
    fake_gamma_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
    fake_gamma_cm.__aexit__ = AsyncMock(return_value=None)
    with (
        patch("pscanner.corpus.cli.enumerate_closed_markets", fake_enumerate),
        patch("pscanner.corpus.cli._drain_pending", fake_drain),
        patch("pscanner.corpus.cli._make_data_client", return_value=fake_data_cm),
        patch("pscanner.corpus.cli._make_gamma_client", return_value=fake_gamma_cm),
    ):
        rc = await run_corpus_command(["backfill", "--db", str(db_path)])
    assert rc == 0
    fake_enumerate.assert_awaited()
    fake_drain.assert_awaited()


@pytest.mark.asyncio
async def test_build_features_command_smokes(tmp_path: Path) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    rc = await run_corpus_command(["build-features", "--db", str(db_path)])
    assert rc == 0
