"""Per-detector :class:`SignalEvaluator` implementations.

PaperTrader walks a list of evaluators on each alert; the first one whose
``accepts`` returns ``True`` runs the parse → quality → size pipeline.
"""

from pscanner.strategies.evaluators.mispricing import MispricingEvaluator
from pscanner.strategies.evaluators.monotone import MonotoneEvaluator
from pscanner.strategies.evaluators.move_attribution import MoveAttributionEvaluator
from pscanner.strategies.evaluators.protocol import (
    ParsedSignal,
    SignalEvaluator,
)
from pscanner.strategies.evaluators.smart_money import SmartMoneyEvaluator
from pscanner.strategies.evaluators.velocity import VelocityEvaluator

__all__ = [
    "MispricingEvaluator",
    "MonotoneEvaluator",
    "MoveAttributionEvaluator",
    "ParsedSignal",
    "SignalEvaluator",
    "SmartMoneyEvaluator",
    "VelocityEvaluator",
]
