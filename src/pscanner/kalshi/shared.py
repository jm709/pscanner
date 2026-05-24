"""Shared helpers for the Kalshi platform integration."""

from __future__ import annotations

from datetime import UTC, datetime


def iso_to_epoch(iso: str, *, fallback: int) -> int:
    """Parse an ISO 8601 datetime string to epoch seconds.

    Returns ``fallback`` if the input is empty or unparseable. Kalshi wire
    format is ``"2026-05-04T12:00:00Z"``; ``datetime.fromisoformat`` handles
    the trailing ``Z`` since Python 3.11.
    """
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())
