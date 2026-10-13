"""Pure logic for the insider-episode catalogue (no I/O).

Dataclasses, the frozen scorer (spec Amendment 1), ranking, the blind-test
comparator, the automatic price-move anchor, control matching, and report
rendering. Everything here is deterministic and testable without a corpus.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field


class HoldoutLockedError(RuntimeError):
    """Raised when holdout cases are requested before the scorer is frozen."""


@dataclass(frozen=True, slots=True)
class MarketRef:
    """One corpus market attached to a case."""

    slug: str
    winning_side: str
    role: str
    condition_id: str | None = None


@dataclass(frozen=True, slots=True)
class PublicCase:
    """The part of a registry case every pipeline stage may see."""

    id: str
    name: str
    split: str
    confidence: str
    analyzable: bool
    news_ts: int
    news_precision: str
    markets: tuple[MarketRef, ...] = field(default=())

    @property
    def primary(self) -> MarketRef | None:
        """First market whose role starts with ``primary``, if any."""
        for m in self.markets:
            if m.role.startswith("primary"):
                return m
        return None


@dataclass(frozen=True, slots=True)
class SealedCase:
    """Published wallet addresses. Read only by the blind-test comparator."""

    id: str
    wallets: tuple[str, ...]


def select_cases(cases: list[PublicCase], split: str, *, scorer_frozen: bool) -> list[PublicCase]:
    """Filter cases by split, refusing holdout access until the scorer is frozen.

    Args:
        cases: All public cases from the registry.
        split: ``dev``, ``holdout`` or ``all``.
        scorer_frozen: Operator attestation that spec Amendment 1 is signed off.

    Returns:
        The selected cases in registry order.

    Raises:
        HoldoutLockedError: ``split`` includes holdout and ``scorer_frozen`` is false.
        ValueError: ``split`` is not one of the three accepted values.
    """
    if split not in {"dev", "holdout", "all"}:
        msg = f"split must be dev, holdout or all, got {split!r}"
        raise ValueError(msg)
    if split != "dev" and not scorer_frozen:
        msg = (
            "holdout cases are locked until the scorer is frozen; "
            "pass --scorer-frozen only after spec Amendment 1 is signed off"
        )
        raise HoldoutLockedError(msg)
    if split == "all":
        return list(cases)
    return [c for c in cases if c.split == split]


@dataclass(frozen=True, slots=True)
class WindowTrade:
    """One corpus trade inside an analysis window."""

    wallet: str
    ts: int
    outcome_side: str
    bs: str
    price: float
    notional: float
