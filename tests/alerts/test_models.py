"""Pin the ``DetectorName`` Literal contents so detectors can't drift unseen."""

from __future__ import annotations

from typing import get_args

from pscanner.alerts.models import DetectorName


def test_detector_name_literal_contains_subgraph_copy() -> None:
    assert "subgraph_copy" in get_args(DetectorName)


def test_detector_name_literal_only_subgraph_copy() -> None:
    """After the detector cleanup wave, ``subgraph_copy`` is the only
    surviving emitter — keep this pin so re-adding a detector forces a
    deliberate update."""
    assert set(get_args(DetectorName)) == {"subgraph_copy"}
