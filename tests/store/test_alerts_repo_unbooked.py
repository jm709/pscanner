"""Unit tests for AlertsRepo.fetch_unbooked_since (#105)."""

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


def _make_alert(*, key: str, ts: int, detector: str = "gate_buy") -> Alert:
    return Alert(
        detector=cast(DetectorName, detector),
        alert_key=key,
        severity=cast(Severity, "med"),
        title=f"t-{key}",
        body={"foo": "bar"},
        created_at=ts,
    )


def test_fetch_unbooked_since_returns_alerts_without_entry(conn: sqlite3.Connection) -> None:
    """An alert with no paper_trades entry inside the window is returned."""
    alerts = AlertsRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="recent-1", ts=now - 60))
    alerts.insert_if_new(_make_alert(key="recent-2", ts=now - 120))

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert sorted(keys) == ["recent-1", "recent-2"]


def test_fetch_unbooked_since_excludes_alerts_with_entry(conn: sqlite3.Connection) -> None:
    """An alert that already has a paper_trades entry row is excluded."""
    alerts = AlertsRepo(conn)
    paper = PaperTradesRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="booked", ts=now - 60))
    alerts.insert_if_new(_make_alert(key="unbooked", ts=now - 60))
    paper.insert_entry(
        triggering_alert_key="booked",
        triggering_alert_detector="gate_buy",
        rule_variant=None,
        source_wallet=None,
        condition_id=cast(Any, "0xc"),
        asset_id=cast(Any, "0xa"),
        outcome="YES",
        shares=10.0,
        fill_price=0.5,
        cost_usd=5.0,
        nav_after_usd=1000.0,
        ts=now - 50,
    )

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert keys == ["unbooked"]


def test_fetch_unbooked_since_excludes_alerts_outside_window(
    conn: sqlite3.Connection,
) -> None:
    """An alert older than the cutoff is excluded even if unbooked."""
    alerts = AlertsRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="too-old", ts=now - 3600))
    alerts.insert_if_new(_make_alert(key="in-window", ts=now - 60))

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert keys == ["in-window"]


def test_fetch_unbooked_since_returns_oldest_first(conn: sqlite3.Connection) -> None:
    """Replay processes alerts in chronological order, not newest-first."""
    alerts = AlertsRepo(conn)
    now = int(time.time())
    alerts.insert_if_new(_make_alert(key="newer", ts=now - 60))
    alerts.insert_if_new(_make_alert(key="older", ts=now - 200))

    result = alerts.fetch_unbooked_since(min_created_at=now - 300)

    keys = [a.alert_key for a in result]
    assert keys == ["older", "newer"]
