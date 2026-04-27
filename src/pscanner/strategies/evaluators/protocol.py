"""``SignalEvaluator`` Protocol — contract every per-detector evaluator implements.

PaperTrader fans every alert through a list of evaluators. The first one
whose ``accepts`` returns ``True`` parses the alert into one or more
``ParsedSignal`` instances; each signal is independently quality-gated,
sized, and booked as a paper_trade row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pscanner.alerts.models import Alert
from pscanner.poly.ids import ConditionId


@dataclass(frozen=True, slots=True)
class ParsedSignal:
    """One tradeable direction extracted from a single alert.

    Attributes:
        condition_id: Market identifier the entry will be booked against.
        side: Outcome name (e.g. ``"yes"``, ``"Trump"``) used by
            ``PaperTrader._resolve_outcome`` to look up the asset_id and
            fill price via :class:`MarketCacheRepo`.
        rule_variant: ``"follow"``/``"fade"`` for velocity twin-trades;
            ``None`` for single-entry sources.
        metadata: Pass-through bag of fields each evaluator may stash for
            its own ``quality_passes`` (e.g. SmartMoney stores ``wallet``).
    """

    condition_id: ConditionId
    side: str
    rule_variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalEvaluator(Protocol):
    """Contract for one per-detector trade-signal extractor.

    Each method is called by ``PaperTrader.evaluate`` in order:

    1. ``accepts(alert)`` — does this evaluator handle this alert's detector?
    2. ``parse(alert)`` — extract zero-or-more :class:`ParsedSignal` instances
       (zero on body-shape mismatch; one for single-entry sources; two for
       velocity twin-trades).
    3. ``quality_passes(parsed)`` — per-signal quality gate.
    4. ``size(bankroll, parsed)`` — return cost in USD. Bankroll is
       ``starting_bankroll_usd`` (constant), not running NAV — sizing is
       independent of cumulative PnL by design.
    """

    def accepts(self, alert: Alert) -> bool:
        """Return ``True`` iff this evaluator handles ``alert.detector``."""
        ...

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Extract one or more :class:`ParsedSignal` from ``alert``.

        Returns ``[]`` on body-shape mismatch (treated as a soft failure).
        """
        ...

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """Per-signal quality gate; ``False`` skips this signal."""
        ...

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return USD cost for this signal. ``bankroll`` is constant per run."""
        ...
