"""Unit tests for PaperTradesRepo.summary_by_pred_bucket (#106)."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, cast

import pytest

from pscanner.alerts.models import Alert, DetectorName, Severity
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, PaperTradesRepo


@pytest.fixture
def conn(tmp_path):  # type: ignore[no-untyped-def]
    db_path = tmp_path / "daemon.sqlite3"
    c = init_db(db_path)
    yield c
    c.close()


def _book_gate_alert(
    conn: sqlite3.Connection,
    *,
    key: str,
    pred: float,
    resolved_pnl: float | None = None,
    ts: int | None = None,
) -> None:
    """Insert an alert and an entry; if ``resolved_pnl`` given, also an exit."""
    ts = ts or int(time.time())
    alert = Alert(
        detector=cast(DetectorName, "gate_buy"),
        alert_key=key,
        severity=cast(Severity, "med"),
        title=f"t-{key}",
        body={"pred": pred, "side": "YES", "implied_prob_at_buy": pred - 0.1},
        created_at=ts,
    )
    AlertsRepo(conn).insert_if_new(alert)
    paper = PaperTradesRepo(conn)
    paper.insert_entry(
        triggering_alert_key=key,
        triggering_alert_detector="gate_buy",
        rule_variant=None,
        source_wallet=None,
        condition_id=cast(Any, f"0xc-{key}"),
        asset_id=cast(Any, f"0xa-{key}"),
        outcome="YES",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=1000.0,
        ts=ts,
    )
    if resolved_pnl is not None:
        # Exit row: cost_usd carries the proceeds, so realized_pnl = exit.cost - entry.cost
        proceeds = 5.0 + resolved_pnl
        # Need the entry's trade_id to set parent_trade_id
        entry_id = conn.execute(
            "SELECT trade_id FROM paper_trades WHERE triggering_alert_key=?", (key,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO paper_trades (
                triggering_alert_key, triggering_alert_detector, rule_variant,
                source_wallet, condition_id, asset_id, outcome, shares,
                fill_price, cost_usd, nav_after_usd, ts, trade_kind, parent_trade_id
            ) VALUES (?, 'gate_buy', NULL, NULL, ?, ?, 'YES', 10.0, 1.0, ?, 1000.0, ?, 'exit', ?)
            """,
            (key, f"0xc-{key}", f"0xa-{key}", proceeds, ts + 100, entry_id),
        )
        conn.commit()


def test_pred_bucket_summary_returns_one_row_per_active_bucket(
    conn: sqlite3.Connection,
) -> None:
    """Buckets with no entries are omitted."""
    _book_gate_alert(conn, key="A", pred=0.55)  # 0.5-0.6
    _book_gate_alert(conn, key="B", pred=0.85)  # 0.8-0.9

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    labels = [r.bucket_label for r in rows]
    assert labels == ["0.5-0.6", "0.8-0.9"]


def test_pred_bucket_open_and_resolved_counts(conn: sqlite3.Connection) -> None:
    """Open vs resolved counts split correctly within a bucket."""
    _book_gate_alert(conn, key="open", pred=0.75)
    _book_gate_alert(conn, key="won", pred=0.75, resolved_pnl=+5.0)  # full payout
    _book_gate_alert(conn, key="lost", pred=0.75, resolved_pnl=-5.0)  # zero payout

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()
    assert len(rows) == 1
    bucket = rows[0]
    assert bucket.bucket_label == "0.7-0.8"
    assert bucket.open_count == 1
    assert bucket.resolved_count == 2
    assert bucket.win_rate == pytest.approx(0.5)
    assert bucket.realized_pnl == pytest.approx(0.0)


def test_pred_bucket_excludes_non_gate_buy(conn: sqlite3.Connection) -> None:
    """Only ``triggering_alert_detector = 'gate_buy'`` is bucketed."""
    AlertsRepo(conn).insert_if_new(
        Alert(
            detector=cast(DetectorName, "smart_money"),
            alert_key="sm-1",
            severity=cast(Severity, "med"),
            title="t",
            body={"foo": "bar"},
            created_at=int(time.time()),
        ),
    )
    PaperTradesRepo(conn).insert_entry(
        triggering_alert_key="sm-1",
        triggering_alert_detector="smart_money",
        rule_variant=None,
        source_wallet=None,
        condition_id=cast(Any, "0xc"),
        asset_id=cast(Any, "0xa"),
        outcome="YES",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=1000.0,
        ts=int(time.time()),
    )

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    assert rows == []


def test_pred_bucket_pred_one_lands_in_top_bucket(conn: sqlite3.Connection) -> None:
    """``pred = 1.0`` exactly is bucketed into ``0.9-1.0`` (closed upper bound)."""
    _book_gate_alert(conn, key="exact-one", pred=1.0)

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    assert len(rows) == 1
    assert rows[0].bucket_label == "0.9-1.0"
    assert rows[0].open_count == 1


def test_pred_bucket_below_floor_omitted(conn: sqlite3.Connection) -> None:
    """``pred < 0.5`` does not appear (matches the new ``min_pred`` floor).

    Defensive: the detector should never emit such alerts post-#106, but if
    a stale model artifact does, the bucket query still tolerates them by
    filtering them out.
    """
    _book_gate_alert(conn, key="too-low", pred=0.3)

    rows = PaperTradesRepo(conn).summary_by_pred_bucket()

    assert rows == []
