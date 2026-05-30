"""Tests for the causal copy-selection precompute."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.copy_selection import KPolicy, has_platform_column, ranked_qualifiers, resolve_k


def test_resolve_k_fixed_count() -> None:
    assert resolve_k(KPolicy(top_k=25), bankroll=10_000.0, qualified_count=500) == 25


def test_resolve_k_capital_per_wallet_floors() -> None:
    # 10_000 / 750 = 13.33 -> 13
    assert resolve_k(KPolicy(capital_per_wallet=750.0), bankroll=10_000.0,
                     qualified_count=500) == 13


def test_resolve_k_top_frac_ceils_against_qualified() -> None:
    # ceil(0.1 * 95) = 10
    assert resolve_k(KPolicy(top_frac=0.1), bankroll=10_000.0, qualified_count=95) == 10


def test_resolve_k_top_frac_zero_qualified_is_zero() -> None:
    assert resolve_k(KPolicy(top_frac=0.1), bankroll=10_000.0, qualified_count=0) == 0


def test_resolve_k_no_mode_raises() -> None:
    with pytest.raises(ValueError, match="no mode"):
        resolve_k(KPolicy(), bankroll=10_000.0, qualified_count=10)


def test_has_platform_column_true(corpus_factory) -> None:
    db: Path = corpus_factory([], [], with_platform=True)
    assert has_platform_column(db) is True


def test_has_platform_column_false(corpus_factory) -> None:
    db: Path = corpus_factory([], [], with_platform=False)
    assert has_platform_column(db) is False


def _brute_edge(trades, resolutions, boundary_ts, *, min_resolved, window_days):
    """Reference: causal mean(won - price) per wallet at one boundary."""
    res = {cid: (yw, rt) for cid, yw, rt in resolutions}
    per_wallet: dict[str, list[tuple[int, float]]] = {}
    for w, cid, side, bs, price, _notional, ts in trades:
        if bs != "BUY" or cid not in res:
            continue
        yw, rt = res[cid]
        if not (ts <= rt and rt < boundary_ts):
            continue
        if window_days and rt < boundary_ts - window_days * 86400:
            continue
        won = 1.0 if ((yw == 1 and side == "YES") or (yw == 0 and side == "NO")) else 0.0
        per_wallet.setdefault(w, []).append((rt, won - price))
    out = {}
    for w, recs in per_wallet.items():
        if len(recs) < min_resolved:
            continue
        edge = sum(d for _, d in recs) / len(recs)
        if edge > 0:
            out[w] = edge
    return out


def test_ranked_qualifiers_matches_bruteforce_lifetime(corpus_factory) -> None:
    # Two wallets: A strongly positive, B negative. min_resolved=2 for a tiny case.
    trades = [
        ("A", "m1", "YES", "BUY", 0.40, 100.0, 10),
        ("A", "m2", "YES", "BUY", 0.30, 100.0, 20),
        ("B", "m3", "YES", "BUY", 0.80, 100.0, 15),
        ("B", "m4", "YES", "BUY", 0.70, 100.0, 25),
    ]
    resolutions = [("m1", 1, 100), ("m2", 1, 110), ("m3", 0, 120), ("m4", 0, 130)]
    boundary = 200
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=0,
        boundaries=[boundary],
    )
    got = {w: edge for (b, w, edge, n, rk, nq) in rows if b == boundary}
    expected = _brute_edge(trades, resolutions, boundary, min_resolved=2, window_days=0)
    assert set(got) == set(expected)  # only A qualifies (positive edge); B excluded
    for w in expected:
        assert got[w] == pytest.approx(expected[w])
    # rank is deterministic and 1-based
    a_row = next(r for r in rows if r[1] == "A")
    assert a_row[4] == 1


def test_ranked_qualifiers_excludes_below_min_resolved(corpus_factory) -> None:
    trades = [("A", "m1", "YES", "BUY", 0.40, 100.0, 10)]  # only 1 resolved
    resolutions = [("m1", 1, 100)]
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket", min_resolved=2, edge_window_days=0, boundaries=[200],
    )
    assert rows == []
