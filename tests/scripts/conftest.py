"""Shared fixtures for scripts tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

# trade tuple: (wallet, condition_id, outcome_side, bs, price, notional_usd, ts)
# resolution tuple: (condition_id, outcome_yes_won, resolved_at)
Trade = tuple[str, str, str, str, float, float, int]
Resolution = tuple[str, int, int]


def _build(
    path: Path, trades: list[Trade], resolutions: list[Resolution], *, with_platform: bool
) -> None:
    plat_col = "platform TEXT NOT NULL DEFAULT 'polymarket'," if with_platform else ""
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""
        CREATE TABLE corpus_trades (
          {plat_col}
          tx_hash TEXT NOT NULL, asset_id TEXT NOT NULL,
          wallet_address TEXT NOT NULL, condition_id TEXT NOT NULL,
          outcome_side TEXT NOT NULL, bs TEXT NOT NULL,
          price REAL NOT NULL, size REAL NOT NULL,
          notional_usd REAL NOT NULL, ts INTEGER NOT NULL
        );
        CREATE TABLE market_resolutions (
          {plat_col}
          condition_id TEXT NOT NULL, winning_outcome_index INTEGER NOT NULL,
          outcome_yes_won INTEGER NOT NULL, resolved_at INTEGER NOT NULL,
          source TEXT NOT NULL, recorded_at INTEGER NOT NULL
        );
        """
    )
    prefix = "'polymarket'," if with_platform else ""
    for i, (w, cid, side, bs, price, notional, ts) in enumerate(trades):
        conn.execute(
            f"INSERT INTO corpus_trades VALUES ({prefix}?,?,?,?,?,?,?,?,?,?)",  # noqa: S608
            (f"0xtx{i}", f"asset{i}", w, cid, side, bs, price, notional / price, notional, ts),
        )
    for cid, yes_won, resolved_at in resolutions:
        conn.execute(
            f"INSERT INTO market_resolutions VALUES ({prefix}?,?,?,?,'test',?)",  # noqa: S608
            (cid, 0 if yes_won else 1, yes_won, resolved_at, resolved_at),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def corpus_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a builder: corpus_factory(trades, resolutions, with_platform=True) -> db path."""
    counter = {"n": 0}

    def make(
        trades: list[Trade], resolutions: list[Resolution], *, with_platform: bool = True
    ) -> Path:
        counter["n"] += 1
        db = tmp_path / f"corpus{counter['n']}.sqlite3"
        _build(db, trades, resolutions, with_platform=with_platform)
        return db

    return make
