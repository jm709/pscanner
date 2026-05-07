"""Tests for the `pscanner corpus` CLI commands."""

from __future__ import annotations

import json as _json
import sqlite3 as _sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from pscanner.corpus import cli as corpus_cli
from pscanner.corpus.cli import _DEFAULT_SUBGRAPH_ID, build_corpus_parser, run_corpus_command
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo
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


def test_build_features_parser_accepts_platform_manifold() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features", "--platform", "manifold"])
    assert args.platform == "manifold"


def test_build_features_parser_default_platform_is_polymarket() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(["build-features"])
    assert args.platform == "polymarket"
