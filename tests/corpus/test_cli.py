"""Tests for the `pscanner corpus` CLI commands."""

from __future__ import annotations

import sqlite3 as _sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from pscanner.corpus import cli as corpus_cli
from pscanner.corpus.cli import (
    _DEFAULT_SUBGRAPH_ID,
    _HANDLERS,
    _register_missing_polymarket_resolutions,
    _resolve_subgraph_flags,
    build_corpus_parser,
    run_corpus_command,
)
from pscanner.corpus.db import apply_read_pragmas, init_corpus_db
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo
from pscanner.corpus.subgraph_dispatch import DispatchedRunSummary
from pscanner.corpus.subgraph_ingest import SubgraphRunSummary
from pscanner.corpus.subgraph_ingest_v1 import V1SubgraphRunSummary
from pscanner.poly.data import DataClient
from pscanner.poly.gamma import GammaClient


def test_parser_recognises_all_subcommands() -> None:
    parser = build_corpus_parser()
    assert parser.parse_args(["backfill"]).command == "backfill"
    assert parser.parse_args(["refresh"]).command == "refresh"
    assert parser.parse_args(["build-features"]).command == "build-features"
    assert (
        parser.parse_args(["subgraph-backfill", "--api-key", "k", "--subgraph-id", "abc"]).command
        == "subgraph-backfill"
    )


def test_parser_supports_rebuild_flag() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--rebuild"])
    assert args.rebuild is True


def _fake_client_factory() -> object:
    """Return a side-effect callable that builds AsyncMock clients per-class.

    Patches the module-private ``_build_client`` seam so ``_client_ctx`` can
    construct and ``await client.aclose()`` on the returned mocks without
    further configuration.
    """
    fakes: dict[type, AsyncMock] = {GammaClient: AsyncMock(), DataClient: AsyncMock()}

    def build(client_cls: type, _rpm: int) -> AsyncMock:
        return fakes[client_cls]

    return build


@pytest.mark.asyncio
async def test_backfill_command_smokes(tmp_path: Path) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    fake_enumerate = AsyncMock(return_value=0)
    fake_drain = AsyncMock(return_value=0)
    with (
        patch("pscanner.corpus.cli.enumerate_closed_markets", fake_enumerate),
        patch("pscanner.corpus.cli._drain_pending", fake_drain),
        patch("pscanner.corpus.cli._build_client", side_effect=_fake_client_factory()),
    ):
        rc = await run_corpus_command(["backfill", "--db", str(db_path)])
    assert rc == 0
    fake_enumerate.assert_awaited()
    fake_drain.assert_awaited()


@pytest.mark.asyncio
async def test_backfill_command_registers_resolutions(tmp_path: Path) -> None:
    """`pscanner corpus backfill` registers resolutions before exiting (#115)."""
    db_path = tmp_path / "corpus.sqlite3"
    fake_enumerate = AsyncMock(return_value=0)
    fake_drain = AsyncMock(return_value=0)
    fake_register = AsyncMock(return_value=0)
    with (
        patch("pscanner.corpus.cli.enumerate_closed_markets", fake_enumerate),
        patch("pscanner.corpus.cli._drain_pending", fake_drain),
        patch(
            "pscanner.corpus.cli._register_missing_polymarket_resolutions",
            fake_register,
        ),
        patch("pscanner.corpus.cli._build_client", side_effect=_fake_client_factory()),
    ):
        rc = await run_corpus_command(["backfill", "--db", str(db_path)])
    assert rc == 0
    fake_enumerate.assert_awaited()
    fake_drain.assert_awaited()
    fake_register.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_features_command_smokes(tmp_path: Path) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    rc = await run_corpus_command(["build-features", "--db", str(db_path)])
    assert rc == 0


@pytest.mark.asyncio
async def test_subgraph_backfill_subcommand_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pscanner corpus subgraph-backfill` calls the V1+V2 dispatcher."""
    db_path = tmp_path / "c.sqlite3"

    captured: dict[str, object] = {}

    _fake_v2_summary = SubgraphRunSummary(0, 0, 0, 0, 0, 0, 0)
    _fake_v1_summary = V1SubgraphRunSummary(0, 0, 0, 0, 0, 0, 0, 0)

    async def fake_dispatched(  # type: ignore[no-untyped-def]
        *, conn, v1_client, v2_client, versions, page_size, limit, **kwargs
    ):
        captured["v2_url"] = v2_client.url
        captured["v1_url"] = v1_client.url
        captured["v2_rpm"] = v2_client.rpm
        captured["page_size"] = page_size
        captured["limit"] = limit
        captured["versions"] = versions
        return DispatchedRunSummary(
            v2_summary=_fake_v2_summary,
            v1_summary=_fake_v1_summary,
            truncation_flags_cleared=0,
        )

    monkeypatch.setattr(corpus_cli, "run_subgraph_backfill_dispatched", fake_dispatched)

    rc = await corpus_cli.run_corpus_command(
        [
            "subgraph-backfill",
            "--db",
            str(db_path),
            "--api-key",
            "test-key",
            "--subgraph-id",  # deprecated alias — should map to V2 URL
            "abc123",
            "--rpm",
            "120",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    assert "test-key" in str(captured["v2_url"])
    assert "abc123" in str(captured["v2_url"])
    assert captured["v2_rpm"] == 120
    assert captured["limit"] == 5
    assert captured["page_size"] == 1000  # default _DEFAULT_SUBGRAPH_PAGE_SIZE


@pytest.mark.asyncio
async def test_subgraph_backfill_missing_api_key_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GRAPH_API_KEY"):
        await corpus_cli.run_corpus_command(
            [
                "subgraph-backfill",
                "--db",
                str(tmp_path / "c.sqlite3"),
                "--subgraph-id",
                "abc",
            ]
        )


def test_default_subgraph_id_matches_polymarket_orderbook() -> None:
    """The pinned default must be the current Polymarket Orderbook subgraph id."""
    assert _DEFAULT_SUBGRAPH_ID == "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"


def test_backfill_parser_accepts_platform_manifold() -> None:
    """`pscanner corpus backfill --platform manifold` parses correctly."""
    parser = build_corpus_parser()
    args = parser.parse_args(["backfill", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_backfill_parser_default_platform_is_polymarket() -> None:
    """Backfill's default platform is polymarket (preserves existing behavior)."""
    parser = build_corpus_parser()
    args = parser.parse_args(["backfill"])
    assert args.platform == "polymarket"


def test_backfill_parser_rejects_unknown_platform() -> None:
    """An unknown platform name fails argparse."""
    parser = build_corpus_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["backfill", "--platform", "ftx"])


def test_backfill_parser_accepts_platform_kalshi() -> None:
    """`pscanner corpus backfill --platform kalshi` parses correctly."""
    parser = build_corpus_parser()
    args = parser.parse_args(["backfill", "--platform", "kalshi"])
    assert args.platform == "kalshi"


def test_backfill_parser_default_platform_is_still_polymarket() -> None:
    """Adding kalshi to choices doesn't change the default."""
    parser = build_corpus_parser()
    args = parser.parse_args(["backfill"])
    assert args.platform == "polymarket"


def test_refresh_parser_accepts_platform_manifold() -> None:
    """`pscanner corpus refresh --platform manifold` parses correctly."""
    parser = build_corpus_parser()
    args = parser.parse_args(["refresh", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_refresh_parser_default_platform_is_polymarket() -> None:
    """Refresh's default platform is polymarket (preserves existing behavior)."""
    parser = build_corpus_parser()
    args = parser.parse_args(["refresh"])
    assert args.platform == "polymarket"


def test_refresh_parser_rejects_unknown_platform() -> None:
    """An unknown platform name fails argparse."""
    parser = build_corpus_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["refresh", "--platform", "ftx"])


def test_refresh_parser_accepts_platform_kalshi() -> None:
    """`pscanner corpus refresh --platform kalshi` parses correctly."""
    parser = build_corpus_parser()
    args = parser.parse_args(["refresh", "--platform", "kalshi"])
    assert args.platform == "kalshi"


def test_build_features_parser_accepts_platform_manifold() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_build_features_parser_default_platform_is_polymarket() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features"])
    assert args.platform == "polymarket"


def test_apply_read_pragmas_sets_expected_values(tmp_path: Path) -> None:
    """``apply_read_pragmas`` configures cache, mmap, temp_store, and query_only."""
    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()

    conn = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        apply_read_pragmas(conn)
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -4000000
        # mmap_size is clamped by SQLITE_MAX_MMAP_SIZE at compile time (often
        # ~2 GB on Linux builds). We requested 8 GB; assert it is at least
        # 1 GB to confirm the PRAGMA fired and wasn't silently ignored.
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] >= 2**30
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_register_missing_polymarket_resolutions_calls_gamma_for_each_missing(
    tmp_path: Path,
) -> None:
    """Helper finds complete markets with no resolution and calls record_resolutions (#115)."""
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    try:
        markets_repo = CorpusMarketsRepo(conn)
        # Two complete markets with valid slugs.
        markets_repo.insert_pending(
            CorpusMarket(
                condition_id="0xabc1",
                event_slug="event-one",
                category="esports",
                closed_at=1_700_000_000,
                enumerated_at=1_700_000_000,
                total_volume_usd=50_000.0,
                market_slug="market-one",
            )
        )
        markets_repo.mark_complete("0xabc1", completed_at=1_700_000_500, truncated=False)
        markets_repo.insert_pending(
            CorpusMarket(
                condition_id="0xabc2",
                event_slug="event-two",
                category="esports",
                closed_at=1_700_001_000,
                enumerated_at=1_700_001_000,
                total_volume_usd=80_000.0,
                market_slug="market-two",
            )
        )
        markets_repo.mark_complete("0xabc2", completed_at=1_700_001_500, truncated=False)

        # determine_outcome_yes_won reads market.outcome_prices (list[float]).
        # index 0 = YES price >= 0.99 → YES won.
        fake_market = MagicMock()
        fake_market.outcome_prices = [1.0, 0.0]

        fake_gamma = MagicMock()
        fake_gamma.get_market_by_slug = AsyncMock(return_value=fake_market)

        written = await _register_missing_polymarket_resolutions(
            conn=conn,
            gamma=fake_gamma,
            now_ts=1_700_002_000,
        )
        assert written == 2
        rows = conn.execute(
            "SELECT condition_id FROM market_resolutions WHERE platform = 'polymarket'"
        ).fetchall()
        assert {next(iter(r)) for r in rows} == {"0xabc1", "0xabc2"}
    finally:
        conn.close()


def test_corpus_parser_supports_backfill_gamma_tags() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["backfill-gamma-tags", "--db", "x.sqlite3", "--limit", "100"])
    assert args.command == "backfill-gamma-tags"
    assert args.db == "x.sqlite3"
    assert args.limit == 100


def test_backfill_gamma_tags_handler_registered() -> None:
    assert "backfill-gamma-tags" in _HANDLERS


def test_backfill_gamma_tags_default_rpm_is_50() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["backfill-gamma-tags", "--db", "x.sqlite3"])
    assert args.rpm == 50


def test_subgraph_backfill_help_lists_version_flags() -> None:
    parser = build_corpus_parser()
    sub_actions = [a for a in parser._actions if hasattr(a, "_name_parser_map")]
    assert sub_actions, "no subparser found"
    subparsers = sub_actions[0].choices
    sg = subparsers["subgraph-backfill"]
    help_text = sg.format_help()
    assert "--subgraph-version" in help_text
    assert "--v1-subgraph-id" in help_text
    assert "--v2-subgraph-id" in help_text
    assert "--subgraph-id" in help_text  # deprecated alias preserved


def test_subgraph_backfill_subgraph_id_alias_maps_to_v2() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["subgraph-backfill", "--subgraph-id", "deprecated-id-value"])
    with capture_logs() as logs:
        resolved = _resolve_subgraph_flags(args)
    assert resolved.v2_subgraph_id == "deprecated-id-value"
    assert any(
        entry.get("event") == "subgraph.cli.deprecated_flag"
        and entry.get("flag") == "--subgraph-id"
        for entry in logs
    ), f"deprecation warning not emitted; saw events: {[e.get('event') for e in logs]}"


def test_subgraph_backfill_no_deprecation_warning_when_alias_unused() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["subgraph-backfill", "--v2-subgraph-id", "explicit-v2"])
    with capture_logs() as logs:
        resolved = _resolve_subgraph_flags(args)
    assert resolved.v2_subgraph_id == "explicit-v2"
    assert not any(
        entry.get("event") == "subgraph.cli.deprecated_flag" for entry in logs
    )


def test_subgraph_backfill_default_version_is_both() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["subgraph-backfill"])
    resolved = _resolve_subgraph_flags(args)
    assert resolved.versions == ("v2", "v1")


def test_subgraph_backfill_version_v1_only() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["subgraph-backfill", "--subgraph-version", "v1"])
    resolved = _resolve_subgraph_flags(args)
    assert resolved.versions == ("v1",)
