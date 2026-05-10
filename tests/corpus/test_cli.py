"""Tests for the `pscanner corpus` CLI commands."""

from __future__ import annotations

import argparse
import json as _json
import sqlite3 as _sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from pscanner.corpus import cli as corpus_cli
from pscanner.corpus.cli import (
    _DEFAULT_SUBGRAPH_ID,
    _cmd_build_features,
    _register_missing_polymarket_resolutions,
    build_corpus_parser,
    run_corpus_command,
)
from pscanner.corpus.db import apply_read_pragmas, init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo, CorpusMarket, CorpusMarketsRepo
from pscanner.corpus.subgraph_ingest import SubgraphRunSummary


def test_parser_recognises_all_subcommands() -> None:
    parser = build_corpus_parser()
    assert parser.parse_args(["backfill"]).command == "backfill"
    assert parser.parse_args(["refresh"]).command == "refresh"
    assert parser.parse_args(["build-features"]).command == "build-features"
    assert parser.parse_args(["onchain-backfill"]).command == "onchain-backfill"
    assert (
        parser.parse_args(["subgraph-backfill", "--api-key", "k", "--subgraph-id", "abc"]).command
        == "subgraph-backfill"
    )


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


def test_corpus_parser_has_onchain_backfill_subcommand() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(
        [
            "onchain-backfill",
            "--from-block",
            "100",
            "--to-block",
            "200",
            "--rpc-url",
            "https://x.test/",
            "--chunk-size",
            "50",
        ]
    )
    assert args.command == "onchain-backfill"
    assert args.from_block == 100
    assert args.to_block == 200
    assert args.rpc_url == "https://x.test/"
    assert args.chunk_size == 50


@pytest.mark.asyncio
@respx.mock
async def test_run_corpus_command_onchain_backfill_inserts_trade(
    tmp_path: Path,
) -> None:
    """End-to-end: CLI handler runs the orchestrator against a mocked RPC."""
    db = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db)
    AssetIndexRepo(conn).upsert(
        AssetEntry(
            asset_id="42",
            condition_id="0xCONDITION",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    conn.close()

    # Maker buying CTF: maker_asset=0 (USDC), taker_asset=42 (CTF), making=20e6 (USDC),
    # taking=40e6 (CTF tokens). Price = 20/40 = 0.50.
    log_data = (
        (0).to_bytes(32, "big")
        + (42).to_bytes(32, "big")
        + (20_000_000).to_bytes(32, "big")
        + (40_000_000).to_bytes(32, "big")
        + (0).to_bytes(32, "big")
    )
    log = {
        "data": "0x" + log_data.hex(),
        "topics": [
            "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6",
            "0x" + "00" * 32,
            "0x" + "00" * 12 + "11" * 20,
            "0x" + "00" * 12 + "22" * 20,
        ],
        "transactionHash": "0x" + "ab" * 32,
        "blockNumber": "0xa",
        "logIndex": "0x0",
    }

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x14"})
        if method == "eth_getLogs":
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            to_b = int(params["toBlock"], 16)
            if from_b <= 10 <= to_b:
                return httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log]}
                )
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": []})
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"timestamp": "0x65f0a000"},
            },
        )

    respx.post("https://example-rpc.test/").mock(side_effect=_route)

    rc = await run_corpus_command(
        [
            "onchain-backfill",
            "--db",
            str(db),
            "--from-block",
            "0",
            "--to-block",
            "20",
            "--chunk-size",
            "20",
            "--rpc-url",
            "https://example-rpc.test/",
        ]
    )
    assert rc == 0

    verify_conn = _sqlite3.connect(str(db))
    verify_conn.row_factory = _sqlite3.Row
    try:
        rows = verify_conn.execute("SELECT bs, price FROM corpus_trades").fetchall()
    finally:
        verify_conn.close()
    assert len(rows) == 1
    assert rows[0]["bs"] == "BUY"
    assert rows[0]["price"] == pytest.approx(0.50)


@pytest.mark.asyncio
async def test_subgraph_backfill_subcommand_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pscanner corpus subgraph-backfill --db ... --api-key X` dispatches the handler."""
    db_path = tmp_path / "c.sqlite3"

    captured: dict[str, object] = {}

    async def fake_run(*, conn, client, page_size, limit, truncation_threshold=3000):  # type: ignore[no-untyped-def]
        captured["conn"] = conn
        captured["client_url"] = client.url
        captured["client_rpm"] = client.rpm
        captured["page_size"] = page_size
        captured["limit"] = limit
        captured["truncation_threshold"] = truncation_threshold
        return SubgraphRunSummary(0, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(corpus_cli, "run_subgraph_backfill", fake_run)

    rc = await corpus_cli.run_corpus_command(
        [
            "subgraph-backfill",
            "--db",
            str(db_path),
            "--api-key",
            "test-key",
            "--subgraph-id",
            "abc123",
            "--rpm",
            "120",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    assert "test-key" in str(captured["client_url"])
    assert "abc123" in str(captured["client_url"])
    assert captured["client_rpm"] == 120
    assert captured["limit"] == 5
    assert captured["page_size"] == 1000  # default _DEFAULT_SUBGRAPH_PAGE_SIZE
    assert captured["truncation_threshold"] == 3000  # orchestrator default


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
    """The pinned default must be the verified Polymarket Orderbook subgraph id."""
    assert _DEFAULT_SUBGRAPH_ID == "7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"


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


@pytest.mark.asyncio
async def test_cli_build_features_uses_separate_read_connection(tmp_path: Path) -> None:
    """The CLI opens a read-only connection for the chronological cursor (#110)."""
    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()

    args = argparse.Namespace(
        db=str(db_path),
        rebuild=False,
        platform="polymarket",
    )

    seen_uris: list[str] = []
    real_connect = _sqlite3.connect

    def _spy_connect(database: object, *cargs: object, **ckwargs: object) -> object:
        if ckwargs.get("uri") and "mode=ro" in str(database):
            seen_uris.append(str(database))
        return real_connect(database, *cargs, **ckwargs)  # type: ignore[arg-type]

    with patch("pscanner.corpus.cli.sqlite3.connect", side_effect=_spy_connect):
        rc = await _cmd_build_features(args)
    assert rc == 0
    assert seen_uris, "_cmd_build_features must open a read-only connection"
    assert any("mode=ro" in u for u in seen_uris)


@pytest.mark.asyncio
async def test_cli_build_features_applies_read_pragmas(tmp_path: Path) -> None:
    """The CLI tunes the read connection with Path A PRAGMAs (#114)."""
    db_path = tmp_path / "corpus.sqlite3"
    init_corpus_db(db_path).close()

    pragmas_applied: list[str] = []
    real_apply = apply_read_pragmas

    def _spy_apply(conn: object) -> None:
        pragmas_applied.append("called")
        real_apply(conn)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    args = argparse.Namespace(
        db=str(db_path),
        rebuild=False,
        platform="polymarket",
    )
    with patch("pscanner.corpus.cli.apply_read_pragmas", side_effect=_spy_apply):
        rc = await _cmd_build_features(args)
    assert rc == 0
    assert pragmas_applied, "apply_read_pragmas must be called by _cmd_build_features"


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
