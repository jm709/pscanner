"""Tests for the feature-projection registry (#145)."""

from __future__ import annotations


def test_module_imports() -> None:
    """The module loads without errors."""
    from pscanner.corpus import feature_projection  # noqa: F401,PLC0415
