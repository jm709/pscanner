"""Tests for ``pscanner.manifold.ids`` NewType wrappers."""

from __future__ import annotations

from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId


def test_market_id_is_str_at_runtime() -> None:
    mid = ManifoldMarketId("kjPkT2HECV")
    assert isinstance(mid, str)
    assert mid == "kjPkT2HECV"


def test_user_id_is_str_at_runtime() -> None:
    uid = ManifoldUserId("igi2zGXsfxYPgB0DJTXVJVmwCOr2")
    assert isinstance(uid, str)
    assert uid == "igi2zGXsfxYPgB0DJTXVJVmwCOr2"


def test_ids_are_distinct_types() -> None:
    """ManifoldMarketId and ManifoldUserId do not share an identity at runtime."""
    mid = ManifoldMarketId("same-value")
    uid = ManifoldUserId("same-value")
    # Both are plain str at runtime — equality holds.
    assert mid == uid
    # But their constructors are distinct callables (NewType wrappers).
    assert ManifoldMarketId is not ManifoldUserId
