"""Tests for ``pscanner.poly.ids`` NewType wrappers."""

from __future__ import annotations

from pscanner.poly.ids import WalletAddress


def test_wallet_address_is_str_at_runtime() -> None:
    addr = WalletAddress("0xabc")
    assert isinstance(addr, str)
    assert addr == "0xabc"
