"""Tests for SubgraphCopyEvaluator (#152)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import SubgraphCopyEvaluatorConfig
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.store.db import init_db
from pscanner.store.repo import PaperTradesRepo, WatchlistRepo
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator


@pytest.fixture
def conn(tmp_path) -> Iterator[sqlite3.Connection]:
    db = init_db(tmp_path / "pscanner.sqlite3")
    yield db
    db.close()


def _build_alert(*, wallet: str = "0xAA", outcome: str = "Yes", tx: str = "0xt1") -> Alert:
    return Alert(
        detector="subgraph_copy",
        alert_key=f"subgraph:{tx}:{outcome}",
        severity="med",
        title="copy",
        body={
            "source_wallet": wallet,
            "tx_hash": tx,
            "condition_id": "0xcond",
            "outcome": outcome,
            "ts": 1_700_000_000,
        },
        created_at=1_700_000_000,
    )


def _seed_watchlist(conn: sqlite3.Connection, *addrs: str) -> None:
    repo = WatchlistRepo(conn)
    for a in addrs:
        repo.upsert(address=a, source="manual", reason="test")


def _insert_paper_trade(conn: sqlite3.Connection, wallet: str, key: str) -> None:
    repo = PaperTradesRepo(conn)
    repo.insert_entry(
        triggering_alert_key=key,
        triggering_alert_detector="subgraph_copy",
        rule_variant=None,
        source_wallet=wallet,
        condition_id=ConditionId("0xcond"),
        asset_id=AssetId("123"),
        outcome="Yes",
        shares=1.0,
        fill_price=0.5,
        cost_usd=0.5,
        nav_after_usd=1000.0,
        ts=int(time.time()),
    )


def test_accepts_only_subgraph_copy(conn: sqlite3.Connection) -> None:
    cfg = SubgraphCopyEvaluatorConfig(enabled=True)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    assert ev.accepts(_build_alert()) is True

    other = Alert(
        detector="smart_money",
        alert_key="k",
        severity="med",
        title="",
        body={},
        created_at=0,
    )
    assert ev.accepts(other) is False


def test_parse_returns_one_signal_with_metadata(conn: sqlite3.Connection) -> None:
    cfg = SubgraphCopyEvaluatorConfig(enabled=True)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    alert = _build_alert(wallet="0xAA", outcome="Cavaliers", tx="0xtx")
    signals = ev.parse(alert)
    assert len(signals) == 1
    sig = signals[0]
    assert str(sig.condition_id) == "0xcond"
    assert sig.side == "Cavaliers"
    assert sig.rule_variant is None
    assert sig.metadata["wallet"] == "0xAA"
    assert sig.metadata["tx_hash"] == "0xtx"
    assert sig.metadata["ts"] == 1_700_000_000


def test_quality_passes_always_true(conn: sqlite3.Connection) -> None:
    cfg = SubgraphCopyEvaluatorConfig(enabled=True)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert())
    assert ev.quality_passes(sig) is True


def test_size_full_base_when_no_prior_trades(conn: sqlite3.Connection) -> None:
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    assert ev.size(1000.0, sig) == pytest.approx(5.0)


def test_size_full_base_at_target_share(conn: sqlite3.Connection) -> None:
    # 3 wallets watched; target_share = 1/3 = 0.333.
    # 0xAA has 1/3 of trades -> share exactly at target -> multiplier 1.0.
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    _insert_paper_trade(conn, "0xAA", "k1")
    _insert_paper_trade(conn, "0xBB", "k2")
    _insert_paper_trade(conn, "0xCC", "k3")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    assert ev.size(1000.0, sig) == pytest.approx(5.0)


def test_size_decays_above_target_share(conn: sqlite3.Connection) -> None:
    # 3 wallets watched; target_share = 0.333.
    # 0xAA has 3 of 4 (75%); multiplier = min(1, 0.333/0.75) = 0.444.
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    _insert_paper_trade(conn, "0xAA", "k1")
    _insert_paper_trade(conn, "0xAA", "k2")
    _insert_paper_trade(conn, "0xAA", "k3")
    _insert_paper_trade(conn, "0xBB", "k4")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005, min_multiplier=0.10)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    expected = 1000.0 * 0.005 * (1.0 / 3.0) / 0.75
    assert ev.size(1000.0, sig) == pytest.approx(expected)


def test_size_floored_at_min_multiplier(conn: sqlite3.Connection) -> None:
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC", "0xDD", "0xEE")
    # Only 0xAA has trades; share = 1.0.
    _insert_paper_trade(conn, "0xAA", "k1")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005, min_multiplier=0.10)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    # raw = min(1, 0.2/1.0) = 0.2; floor = 0.1 -> raw wins.
    assert ev.size(1000.0, sig) == pytest.approx(1000.0 * 0.005 * 0.2)


def test_size_wallet_lookup_is_case_insensitive(conn: sqlite3.Connection) -> None:
    _seed_watchlist(conn, "0xAA", "0xBB", "0xCC")
    _insert_paper_trade(conn, "0xaa", "k1")
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    # 0xAA share = 1/1 = 1.0, target = 1/3, raw = 0.333, floored at 0.10.
    assert ev.size(1000.0, sig) == pytest.approx(1000.0 * 0.005 * (1.0 / 3.0))


def test_size_empty_watchlist_treats_as_one(conn: sqlite3.Connection) -> None:
    # Defensive: no active watchlist rows; target_share = 1/1 = 1.0.
    # No trades yet => total=0 => multiplier 1.0 => base size.
    cfg = SubgraphCopyEvaluatorConfig(enabled=True, position_fraction=0.005)
    ev = SubgraphCopyEvaluator(
        config=cfg,
        watchlist_repo=WatchlistRepo(conn),
        paper_trades=PaperTradesRepo(conn),
    )
    [sig] = ev.parse(_build_alert(wallet="0xAA"))
    assert ev.size(1000.0, sig) == pytest.approx(5.0)
