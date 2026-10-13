"""Tests for the insider-episode I/O script."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.insider_episode_lib import MarketRef, WindowTrade
from scripts.insider_episodes import (
    candidate_wallets,
    extract_window,
    load_registry,
    market_trades,
    open_corpus,
    resolve_market,
)

ANCHOR = 1_000_000
H = 3600

_REGISTRY = """
registry_version: 1
cases:
  - id: case-a
    name: Case A
    confidence: confirmed
    split: dev
    analyzable: true
    news_ts_utc: "2026-01-03T06:00:00Z"
    news_ts_precision: hour
    corpus_markets:
      - market_slug: slug-a
        winning_side: "YES"
        role: primary
      - market_slug: slug-a2
        winning_side: "YES"
        role: sibling
    wallets:
      - address: "  0xABCDEFabcdef0123456789abcdef0123456789AB "
      - address: "0xa72DB1749e9... (TRUNCATED)"
      - address: null
        username: someone
  - id: case-b
    name: Case B
    confidence: suspected
    split: holdout
    analyzable: false
    not_analyzable_reason: below floor
    news_ts_utc: "2025-10-10T09:00:00Z"
    news_ts_precision: minute
    wallets: []
"""


def test_load_registry_splits_public_and_sealed(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(_REGISTRY)
    public, sealed = load_registry(path)

    assert [c.id for c in public] == ["case-a", "case-b"]
    a = public[0]
    assert a.split == "dev"
    assert a.news_ts == 1_767_420_000  # 2026-01-03T06:00:00Z
    assert a.news_precision == "hour"
    assert a.primary is not None
    assert a.primary.slug == "slug-a"
    assert [m.slug for m in a.markets] == ["slug-a", "slug-a2"]
    assert public[1].analyzable is False
    assert public[1].markets == ()

    # sealed: only full 42-char addresses, lowercased and stripped
    assert sealed["case-a"].wallets == ("0xabcdefabcdef0123456789abcdef0123456789ab",)
    assert sealed["case-b"].wallets == ()


def test_public_case_has_no_wallet_field(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(_REGISTRY)
    public, _ = load_registry(path)
    assert not hasattr(public[0], "wallets")


def _window_db(corpus_factory):
    trades = [
        ("w_early", "m1", "YES", "BUY", 0.10, 100.0, ANCHOR - 80 * H),  # outside 72h
        ("w_edge", "m1", "YES", "BUY", 0.10, 100.0, ANCHOR - 72 * H),  # on boundary: out
        ("w_in", "m1", "YES", "BUY", 0.12, 200.0, ANCHOR - 10 * H),
        ("w_in", "m1", "YES", "BUY", 0.15, 300.0, ANCHOR - 5 * H),
        ("w_seller", "m1", "YES", "SELL", 0.20, 50.0, ANCHOR - 4 * H),  # closing, not a buyer
        ("w_no", "m1", "NO", "BUY", 0.85, 400.0, ANCHOR - 3 * H),
        ("w_at", "m1", "YES", "BUY", 0.30, 100.0, ANCHOR),  # inclusive upper bound
        ("w_late", "m1", "YES", "BUY", 0.90, 100.0, ANCHOR + 1),
        ("w_other", "m2", "YES", "BUY", 0.50, 100.0, ANCHOR - 1 * H),
    ]
    resolutions = [("m1", 1, ANCHOR + 3 * H), ("m2", 1, ANCHOR + 9 * H)]
    markets = [
        ("m1", "slug-m1", "geopolitics", 5_000_000.0, ANCHOR + 3 * H, 0),
        ("m2", "slug-m2", "geopolitics", 1_000_000.0, ANCHOR + 9 * H, 0),
    ]
    return corpus_factory(trades, resolutions, markets=markets)


def test_resolve_market_fills_condition_id(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        ref = resolve_market(conn, MarketRef(slug="slug-m1", winning_side="YES", role="primary"))
        assert ref.condition_id == "m1"
        with pytest.raises(LookupError):
            resolve_market(conn, MarketRef(slug="nope", winning_side="YES", role="primary"))
    finally:
        conn.close()


def test_extract_window_bounds_are_open_low_closed_high(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = extract_window(conn, "m1", ANCHOR, 72)
    finally:
        conn.close()
    wallets = [r.wallet for r in rows]
    assert "w_early" not in wallets
    assert "w_edge" not in wallets
    assert "w_late" not in wallets
    assert "w_other" not in wallets
    assert wallets == ["w_in", "w_in", "w_seller", "w_no", "w_at"]
    assert all(isinstance(r, WindowTrade) for r in rows)


def test_candidate_wallets_require_winning_side_buy(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = extract_window(conn, "m1", ANCHOR, 72)
    finally:
        conn.close()
    assert candidate_wallets(rows, "YES") == ["w_in", "w_at"]
    assert candidate_wallets(rows, "NO") == ["w_no"]


def test_market_trades_returns_full_history_in_order(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = market_trades(conn, "m1")
    finally:
        conn.close()
    assert [r.ts for r in rows] == sorted(r.ts for r in rows)
    assert len(rows) == 8
