"""Alert dataclass shared by every detector and the alert sink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Severity = Literal["low", "med", "high"]
SEVERITY_RANK: dict[Severity, int] = {"low": 0, "med": 1, "high": 2}
"""Numeric ranking for ``Severity`` to enable comparisons in evaluator quality gates."""

DetectorName = Literal[
    "smart_money",
    "mispricing",
    "monotone",
    "whales",
    "convergence",
    "velocity",
    "cluster",
    "move_attribution",
]


@dataclass(frozen=True, slots=True)
class Alert:
    """A detector output destined for SQLite + the terminal renderer.

    Attributes:
        detector: The :class:`DetectorName` that produced this alert.
        alert_key: Idempotency key — the primary key in the ``alerts`` table.
            Detectors are responsible for choosing a key that collapses
            duplicates within their natural cadence (e.g. a daily snapshot).
        severity: Triage hint for the renderer (``low``, ``med``, ``high``).
        title: Short human-readable headline.
        body: Detector-specific structured payload (must be JSON-serialisable).
        created_at: Unix timestamp (seconds) when the alert was generated.
    """

    detector: DetectorName
    alert_key: str
    severity: Severity
    title: str
    body: dict[str, Any]
    created_at: int
