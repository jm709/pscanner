"""Pin the ``DetectorName`` Literal contents so detectors can't drift unseen."""

from __future__ import annotations

from typing import get_args

from pscanner.alerts.models import DetectorName


def test_detector_name_literal_contains_gate_buy() -> None:
    assert "gate_buy" in get_args(DetectorName)
