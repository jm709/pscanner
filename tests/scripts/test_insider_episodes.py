"""Tests for the insider-episode I/O script."""

from __future__ import annotations

from pathlib import Path

from scripts.insider_episodes import load_registry

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
    assert a.primary is not None and a.primary.slug == "slug-a"
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
