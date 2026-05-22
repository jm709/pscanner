"""End-to-end CLI integration test for `pscanner corpus backfill-outcome-side`."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus import cli as cli_mod
from pscanner.corpus.db import init_corpus_db
from pscanner.poly.models import Market


def _make_market(*, slug: str, tokens: tuple[str, str]) -> Market:
    return Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": slug,
            "outcomes": ["Cavaliers", "Knicks"],
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": list(tokens),
            "active": True,
            "closed": False,
        }
    )


def _seed_buggy_market(conn: sqlite3.Connection, condition_id: str) -> None:
    conn.execute(
        "INSERT INTO corpus_markets (platform, condition_id, event_slug, market_slug, "
        " category, closed_at, enumerated_at, total_volume_usd, backfill_state) "
        "VALUES ('polymarket', ?, 'evt', ?, 'sports', 1, 1, 0, 'complete')",
        (condition_id, f"slug-{condition_id}"),
    )
    for asset_id in (f"y-{condition_id}", f"n-{condition_id}"):
        conn.execute(
            "INSERT INTO asset_index "
            "(platform, asset_id, condition_id, outcome_side, outcome_index) "
            "VALUES ('polymarket', ?, ?, 'NO', 1)",
            (asset_id, condition_id),
        )
    conn.execute(
        "INSERT INTO corpus_trades (platform, tx_hash, asset_id, wallet_address, "
        " condition_id, outcome_side, bs, price, size, notional_usd, ts) "
        "VALUES ('polymarket', ?, ?, '0xW', ?, 'NO', 'BUY', 0.5, 100.0, 50.0, 1)",
        (f"0xtx-{condition_id}", f"y-{condition_id}", condition_id),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_cli_backfill_outcome_side_end_to_end(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    _seed_buggy_market(conn, "cond1")
    conn.close()

    # Stub the gamma + data context managers used by cli.py.
    fake_data = AsyncMock()
    fake_data.get_market_slug_by_condition_id = AsyncMock(return_value="slug-cond1")
    fake_data.aclose = AsyncMock()

    fake_gamma = AsyncMock()
    fake_gamma.get_market_by_slug = AsyncMock(
        return_value=_make_market(slug="slug-cond1", tokens=("y-cond1", "n-cond1"))
    )
    fake_gamma.aclose = AsyncMock()

    @asynccontextmanager
    async def _gamma_cm():
        yield fake_gamma

    @asynccontextmanager
    async def _data_cm():
        yield fake_data

    monkeypatch.setattr(cli_mod, "_make_gamma_client", _gamma_cm)
    monkeypatch.setattr(cli_mod, "_make_data_client", _data_cm)

    args = argparse.Namespace(
        db=str(db_path),
        rpm=50,
        limit=None,
        dry_run=False,
    )
    rc = await cli_mod._cmd_backfill_outcome_side(args)
    assert rc == 0

    conn = sqlite3.connect(db_path)
    sides = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT asset_id, outcome_side FROM asset_index WHERE condition_id='cond1'"
        )
    }
    assert sides == {"y-cond1": "YES", "n-cond1": "NO"}
    trade_side = conn.execute(
        "SELECT outcome_side FROM corpus_trades WHERE tx_hash='0xtx-cond1'"
    ).fetchone()[0]
    assert trade_side == "YES"
    sentinel = conn.execute(
        "SELECT outcome_side_backfilled_at FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()[0]
    assert sentinel is not None
    conn.close()
