"""End-to-end CLI integration test for `pscanner corpus backfill-outcome-side`."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def _install_fake_clients(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slug: str | None,
    market: Market | None,
) -> tuple[MagicMock, MagicMock]:
    """Replace ``GammaClient`` + ``DataClient`` in ``cli_mod`` namespace.

    Returns the two class-mocks so callers can inspect constructor args
    (e.g. ``rpm`` propagation).
    """
    fake_gamma_instance = AsyncMock()
    fake_gamma_instance.get_market_by_slug = AsyncMock(return_value=market)
    fake_gamma_instance.aclose = AsyncMock()
    fake_gamma_class = MagicMock(return_value=fake_gamma_instance)

    fake_data_instance = AsyncMock()
    fake_data_instance.get_market_slug_by_condition_id = AsyncMock(return_value=slug)
    fake_data_instance.aclose = AsyncMock()
    fake_data_class = MagicMock(return_value=fake_data_instance)

    monkeypatch.setattr(cli_mod, "GammaClient", fake_gamma_class)
    monkeypatch.setattr(cli_mod, "DataClient", fake_data_class)
    return fake_gamma_class, fake_data_class


@pytest.mark.asyncio
async def test_cli_backfill_outcome_side_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    _seed_buggy_market(conn, "cond1")
    conn.close()

    _install_fake_clients(
        monkeypatch,
        slug="slug-cond1",
        market=_make_market(slug="slug-cond1", tokens=("y-cond1", "n-cond1")),
    )

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


@pytest.mark.asyncio
async def test_cli_backfill_outcome_side_propagates_rpm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``--rpm`` must reach the client constructors (final-review fix)."""
    db_path = tmp_path / "corpus.sqlite3"
    conn = init_corpus_db(db_path)
    conn.close()

    fake_gamma_class, fake_data_class = _install_fake_clients(
        monkeypatch, slug=None, market=None
    )

    args = argparse.Namespace(
        db=str(db_path),
        rpm=17,
        limit=None,
        dry_run=False,
    )
    rc = await cli_mod._cmd_backfill_outcome_side(args)
    assert rc == 0
    fake_gamma_class.assert_called_once_with(rpm=17)
    fake_data_class.assert_called_once_with(rpm=17)
