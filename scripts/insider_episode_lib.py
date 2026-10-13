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


@dataclass(frozen=True, slots=True)
class HourBucket:
    """One hour of side-normalized trading on a market."""

    hour_start: int
    side_prob_vwap: float
    notional: float
    n_trades: int


def side_prob(trade: WindowTrade, winning_side: str) -> float:
    """Implied probability of ``winning_side`` from one trade's price."""
    return trade.price if trade.outcome_side == winning_side else 1.0 - trade.price


def hourly_buckets(trades: list[WindowTrade], winning_side: str) -> list[HourBucket]:
    """Volume-weighted hourly side probability, only for hours with trades, ascending."""
    acc: dict[int, list[float]] = {}
    for t in trades:
        hour = (t.ts // 3600) * 3600
        slot = acc.setdefault(hour, [0.0, 0.0, 0.0])
        slot[0] += side_prob(t, winning_side) * t.notional
        slot[1] += t.notional
        slot[2] += 1
    out: list[HourBucket] = []
    for hour in sorted(acc):
        num, den, n = acc[hour]
        if den <= 0:
            continue
        out.append(
            HourBucket(hour_start=hour, side_prob_vwap=num / den, notional=den, n_trades=int(n))
        )
    return out


def auto_anchor(
    buckets: list[HourBucket],
    resolved_at: int,
    *,
    min_move: float = 0.10,
    horizon_hours: int = 24,
) -> int | None:
    """Start hour of the largest forward move toward the winner before resolution.

    For each bucket ``b`` strictly before ``resolved_at``, the forward move is
    ``max(price over (b, b + horizon])`` minus ``price[b]``, using only buckets
    also strictly before ``resolved_at``. Among buckets tied for the largest
    move the latest is returned (the last hour still at the pre-move level, so a
    flat plateau before the rise does not pull the anchor earlier); ``None`` if
    that move is below ``min_move``.
    """
    pre = [b for b in buckets if b.hour_start < resolved_at]
    best_move = 0.0
    best_start: int | None = None
    horizon = horizon_hours * 3600
    for i, b in enumerate(pre):
        later = [x.side_prob_vwap for x in pre[i + 1 :] if x.hour_start <= b.hour_start + horizon]
        if not later:
            continue
        move = max(later) - b.side_prob_vwap
        if move >= best_move and move > 0:
            best_move = move
            best_start = b.hour_start
    if best_start is None or best_move < min_move:
        return None
    return best_start
