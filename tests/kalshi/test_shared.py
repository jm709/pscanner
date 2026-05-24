"""Tests for `pscanner.kalshi.shared`."""

from __future__ import annotations

from pscanner.kalshi.shared import iso_to_epoch


def test_iso_to_epoch_parses_zulu_timestamp() -> None:
    """Standard Kalshi wire format with trailing Z."""
    assert iso_to_epoch("2026-05-04T12:00:00Z", fallback=0) == 1777896000


def test_iso_to_epoch_returns_fallback_on_empty() -> None:
    assert iso_to_epoch("", fallback=42) == 42


def test_iso_to_epoch_returns_fallback_on_malformed() -> None:
    assert iso_to_epoch("not a date", fallback=99) == 99


def test_iso_to_epoch_treats_naive_as_utc() -> None:
    """ISO without a timezone is treated as UTC, not local."""
    naive = iso_to_epoch("2026-05-04T12:00:00", fallback=0)
    zulu = iso_to_epoch("2026-05-04T12:00:00Z", fallback=0)
    assert naive == zulu
