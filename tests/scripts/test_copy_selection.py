"""Tests for the causal copy-selection precompute."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.copy_selection import (
    KPolicy,
    has_platform_column,
    iter_selected_rows,
    ranked_qualifiers,
    resolve_k,
)


def test_resolve_k_fixed_count() -> None:
    assert resolve_k(KPolicy(top_k=25), bankroll=10_000.0, qualified_count=500) == 25


def test_resolve_k_capital_per_wallet_floors() -> None:
    # 10_000 / 750 = 13.33 -> 13
    assert (
        resolve_k(KPolicy(capital_per_wallet=750.0), bankroll=10_000.0, qualified_count=500) == 13
    )


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
        if not (ts <= rt < boundary_ts):
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
        platform="polymarket",
        min_resolved=2,
        edge_window_days=0,
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
        platform="polymarket",
        min_resolved=2,
        edge_window_days=0,
        boundaries=[200],
    )
    assert rows == []


def test_ranked_qualifiers_no_lookahead(corpus_factory) -> None:
    # A's trades resolve at 250/260, AFTER boundary 200 -> A not qualified at 200.
    trades = [
        ("A", "m1", "YES", "BUY", 0.40, 100.0, 10),
        ("A", "m2", "YES", "BUY", 0.30, 100.0, 20),
    ]
    resolutions = [("m1", 1, 250), ("m2", 1, 260)]
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket",
        min_resolved=2,
        edge_window_days=0,
        boundaries=[200],
    )
    assert rows == []  # no resolutions before the boundary


def test_ranked_qualifiers_rolling_window_drops_old_trades(corpus_factory) -> None:
    # window=1 day: at boundary 200 (ts), only trades resolved within [200-86400, 200).
    # Put 2 old resolutions far in the past and 1 recent -> under min_resolved=2 in window.
    day = 86400
    trades = [
        ("A", "m1", "YES", "BUY", 0.40, 100.0, 1),
        ("A", "m2", "YES", "BUY", 0.40, 100.0, 2),
        ("A", "m3", "YES", "BUY", 0.40, 100.0, 3),
    ]
    boundary = 10 * day
    resolutions = [("m1", 1, 1 * day), ("m2", 1, 2 * day), ("m3", 1, boundary - 100)]
    rows = ranked_qualifiers(
        corpus_factory(trades, resolutions),
        platform="polymarket",
        min_resolved=2,
        edge_window_days=1,
        boundaries=[boundary],
    )
    assert rows == []  # only 1 trade inside the 1-day window -> below min_resolved


def test_iter_selected_rows_only_copies_top_k_in_frozen_period(corpus_factory) -> None:
    # Boundaries every 100s from ts=0. A & B both qualify positive by boundary 100;
    # A has higher edge. top_k=1 -> only A copied. Each makes a NEW trade in [100,200).
    trades = [
        # qualifying history (resolves before boundary 100)
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("B", "h3", "YES", "BUY", 0.45, 100.0, 1),
        ("B", "h4", "YES", "BUY", 0.45, 100.0, 2),
        # new trades inside period [100,200)
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
        ("B", "n2", "YES", "BUY", 0.50, 100.0, 160),
    ]
    resolutions = [
        ("h1", 1, 50),
        ("h2", 1, 60),
        ("h3", 1, 50),
        ("h4", 1, 60),
        ("n1", 1, 300),
        ("n2", 1, 300),
    ]
    rows = list(
        iter_selected_rows(
            corpus_factory(trades, resolutions),
            platform="polymarket",
            min_resolved=2,
            edge_window_days=0,
            rebalance_days=None,
            rebalance_seconds=100,
            policy=KPolicy(top_k=1),
            bankroll=10_000.0,
            start_ts=None,
            end_ts=None,
        )
    )
    trade_rows = [r for r in rows if r[0] == "trade"]
    copied_new = {r[2] for r in trade_rows if r[3] in ("n1", "n2")}
    assert copied_new == {"A"}  # only top-1 wallet A copied; B excluded
    # resolutions for copied markets are present and stream is ts-ordered
    assert any(r[0] == "resolution" and r[3] == "n1" for r in rows)
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)


def test_iter_selected_rows_empty_universe(corpus_factory) -> None:
    rows = list(
        iter_selected_rows(
            corpus_factory([], []),
            platform="polymarket",
            min_resolved=20,
            edge_window_days=0,
            rebalance_days=None,
            rebalance_seconds=100,
            policy=KPolicy(top_k=5),
            bankroll=10_000.0,
            start_ts=None,
            end_ts=None,
        )
    )
    assert rows == []


def test_iter_selected_rows_k_larger_than_qualified(corpus_factory) -> None:
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
    ]
    resolutions = [("h1", 1, 50), ("h2", 1, 60), ("n1", 1, 300)]
    rows = list(
        iter_selected_rows(
            corpus_factory(trades, resolutions),
            platform="polymarket",
            min_resolved=2,
            edge_window_days=0,
            rebalance_days=None,
            rebalance_seconds=100,
            policy=KPolicy(top_k=50),
            bankroll=10_000.0,
            start_ts=None,
            end_ts=None,
        )
    )
    copied = {r[2] for r in rows if r[0] == "trade" and r[3] == "n1"}
    assert copied == {"A"}  # only 1 qualifies; K=50 just takes all qualified


def test_iter_selected_rows_no_platform_corpus(corpus_factory) -> None:
    trades = [
        ("A", "h1", "YES", "BUY", 0.30, 100.0, 1),
        ("A", "h2", "YES", "BUY", 0.30, 100.0, 2),
        ("A", "n1", "YES", "BUY", 0.50, 100.0, 150),
    ]
    resolutions = [("h1", 1, 50), ("h2", 1, 60), ("n1", 1, 300)]
    rows = list(
        iter_selected_rows(
            corpus_factory(trades, resolutions, with_platform=False),
            platform="polymarket",
            min_resolved=2,
            edge_window_days=0,
            rebalance_days=None,
            rebalance_seconds=100,
            policy=KPolicy(top_k=5),
            bankroll=10_000.0,
            start_ts=None,
            end_ts=None,
        )
    )
    assert any(r[0] == "trade" and r[3] == "n1" for r in rows)
