"""Per-detector :class:`SignalEvaluator` implementations.

PaperTrader walks a list of evaluators on each alert; the first one whose
``accepts`` returns ``True`` runs the parse → quality → size pipeline.
"""

from pscanner.strategies.evaluators.protocol import (
    ParsedSignal,
    SignalEvaluator,
)

__all__ = ["ParsedSignal", "SignalEvaluator"]
