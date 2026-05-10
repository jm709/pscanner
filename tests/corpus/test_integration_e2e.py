"""End-to-end smoke for `pscanner corpus` against respx-mocked APIs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import respx
from httpx import Response

from pscanner.corpus.cli import run_corpus_command


def _events_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "evt1",
            "title": "Event 1",
            "slug": "evt1",
            "active": False,
            "closed": True,
            "tags": [{"label": "Crypto", "slug": "crypto"}],
            "markets": [
                {
                    "id": "cond1",
                    "conditionId": "cond1",
                    "question": "?",
                    "slug": "cond1-slug",
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["1.0","0.0"]',
                    "volume": 2_000_000.0,
                    "active": False,
                    "closed": True,
                }
            ],
        }
    ]


def _market_by_slug_payload() -> list[dict[str, Any]]:
    """Gamma /markets?slug=cond1-slug response: YES won (price=1.0)."""
    return [
        {
            "id": "cond1",
            "conditionId": "cond1",
            "question": "?",
            "slug": "cond1-slug",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["1.0","0.0"]',
            "volume": 2_000_000.0,
            "active": False,
            "closed": True,
        }
    ]


def _trades_page() -> list[dict[str, Any]]:
    return [
        {
            "transactionHash": "0xa",
            "asset": "asset1",
            "proxyWallet": "0xWALLET",
            "conditionId": "cond1",
            "outcome": "Yes",
            "side": "BUY",
            "price": 0.4,
            "size": 100.0,
            "timestamp": 1_000,
        }
    ]


@pytest.mark.asyncio
async def test_corpus_backfill_then_build_features_e2e(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    with respx.mock(assert_all_called=False) as rx:
        rx.get("https://gamma-api.polymarket.com/events").mock(
            side_effect=lambda req: Response(
                200,
                json=_events_payload() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        rx.get("https://data-api.polymarket.com/trades").mock(
            side_effect=lambda req: Response(
                200,
                json=_trades_page() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        rx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=Response(200, json=_market_by_slug_payload()),
        )

        rc_backfill = await run_corpus_command(["backfill", "--db", str(db)])
        assert rc_backfill == 0

    # Backfill auto-registers resolutions (#115); no manual seeding needed.

    with respx.mock(assert_all_called=False):
        rc_build = await run_corpus_command(["build-features", "--db", str(db)])
        assert rc_build == 0

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        trades = conn.execute("SELECT COUNT(*) AS c FROM corpus_trades").fetchone()["c"]
        examples = conn.execute("SELECT COUNT(*) AS c FROM training_examples").fetchone()["c"]
        label = conn.execute("SELECT label_won FROM training_examples").fetchone()["label_won"]
    finally:
        conn.close()
    assert trades == 1
    assert examples == 1
    assert label == 1


@pytest.mark.asyncio
async def test_corpus_build_features_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite3"
    with respx.mock(assert_all_called=False) as rx:
        rx.get("https://gamma-api.polymarket.com/events").mock(
            side_effect=lambda req: Response(
                200,
                json=_events_payload() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        rx.get("https://data-api.polymarket.com/trades").mock(
            side_effect=lambda req: Response(
                200,
                json=_trades_page() if int(req.url.params.get("offset") or "0") == 0 else [],
            )
        )
        rx.get("https://gamma-api.polymarket.com/markets").mock(
            return_value=Response(200, json=_market_by_slug_payload()),
        )
        await run_corpus_command(["backfill", "--db", str(db)])
    # Backfill auto-registers resolutions (#115); no manual seeding needed.
    await run_corpus_command(["build-features", "--db", str(db)])
    await run_corpus_command(["build-features", "--db", str(db)])

    conn = sqlite3.connect(str(db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    finally:
        conn.close()
    assert count == 1  # second build-features did NOT add a duplicate
