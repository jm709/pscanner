"""Pure logic for the insider-episode catalogue (no I/O).

Dataclasses, the frozen scorer (spec Amendment 1), ranking, the blind-test
comparator, the automatic price-move anchor, control matching, and report
rendering. Everything here is deterministic and testable without a corpus.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True, slots=True)
class WalletFeatures:
    """Scorer inputs for one candidate wallet. Nothing here post-dates the anchor."""

    wallet: str
    n_w: float
    n_buys: int
    first_buy_ts: int
    last_buy_ts: int
    flip: bool
    breadth: int
    n_life: float
    corpus_first_ts: int


def window_features(
    trades: list[WindowTrade], wallet: str, winning_side: str
) -> tuple[float, int, int, int, bool]:
    """Window-only features: ``(n_w, n_buys, first_buy_ts, last_buy_ts, flip)``.

    ``flip`` is true when the wallet SOLD the opposing outcome inside the
    window strictly before its first winning-side BUY.
    """
    buys = [
        t for t in trades if t.wallet == wallet and t.bs == "BUY" and t.outcome_side == winning_side
    ]
    if not buys:
        return 0.0, 0, 0, 0, False
    first_buy = min(t.ts for t in buys)
    last_buy = max(t.ts for t in buys)
    flip = any(
        t.wallet == wallet
        and t.bs == "SELL"
        and t.outcome_side != winning_side
        and t.ts < first_buy
        for t in trades
    )
    return sum(t.notional for t in buys), len(buys), first_buy, last_buy, flip


@dataclass(frozen=True, slots=True)
class Provenance:
    """Wallet age signal from the Polymarket data-api."""

    wallet: str
    first_activity_ts: int | None
    n_events: int | None


_FRESH_PLATEAU_DAYS = 10.0
_FRESH_DECAY_DAYS = 60.0
_FRESH_FLOOR = 0.1
_CONC_BLEND = 0.5
_FLIP_BONUS = 0.5
_SECONDS_PER_DAY = 86_400


def age_days(f: WalletFeatures, p: Provenance | None) -> float:
    """Wallet age at its first window buy, in days; data-api first, corpus fallback."""
    if p is not None and p.first_activity_ts is not None:
        first = p.first_activity_ts
    else:
        first = f.corpus_first_ts
    return max(0.0, (f.first_buy_ts - first) / _SECONDS_PER_DAY)


def freshness(age: float) -> float:
    """1 up to the plateau, then linear decay to a floor (spec Amendment 1)."""
    if age <= _FRESH_PLATEAU_DAYS:
        return 1.0
    return max(_FRESH_FLOOR, 1.0 - (age - _FRESH_PLATEAU_DAYS) / _FRESH_DECAY_DAYS)


def score(f: WalletFeatures, p: Provenance | None) -> float:
    """The frozen composite from spec Amendment 1."""
    conc = min(f.n_w / f.n_life, 1.0) if f.n_life > 0 else 1.0
    base = math.log10(1.0 + f.n_w) * freshness(age_days(f, p)) * (_CONC_BLEND + _CONC_BLEND * conc)
    return base / math.log10(2.0 + f.breadth) + (_FLIP_BONUS if f.flip else 0.0)


@dataclass(frozen=True, slots=True)
class ScoredWallet:
    """A candidate with its score and 1-based rank within its market."""

    features: WalletFeatures
    provenance: Provenance | None
    age: float
    score: float
    rank: int


def rank_wallets(feats: list[WalletFeatures], prov: dict[str, Provenance]) -> list[ScoredWallet]:
    """Score and order candidates: score desc, notional desc, earliest buy, wallet."""
    scored = [
        (f, prov.get(f.wallet), age_days(f, prov.get(f.wallet)), score(f, prov.get(f.wallet)))
        for f in feats
    ]
    scored.sort(key=lambda t: (-t[3], -t[0].n_w, t[0].first_buy_ts, t[0].wallet))
    return [
        ScoredWallet(features=f, provenance=p, age=a, score=s, rank=i)
        for i, (f, p, a, s) in enumerate(scored, 1)
    ]


@dataclass(frozen=True, slots=True)
class BlindResult:
    """Where the published wallets landed in a ranking they were hidden from."""

    case_id: str
    anchor_kind: str
    n_candidates: int
    ranks: dict[str, int | None]
    best_rank: int | None
    passed: bool | None


def compare_published(
    case_id: str,
    anchor_kind: str,
    ranked: list[ScoredWallet],
    sealed: SealedCase,
    *,
    pass_rank: int = 10,
) -> BlindResult:
    """Look up each published wallet's rank. ``passed`` is None if none are candidates."""
    by_wallet = {r.features.wallet.lower(): r.rank for r in ranked}
    ranks = {w.lower(): by_wallet.get(w.lower()) for w in sealed.wallets}
    found = [r for r in ranks.values() if r is not None]
    best = min(found) if found else None
    return BlindResult(
        case_id=case_id,
        anchor_kind=anchor_kind,
        n_candidates=len(ranked),
        ranks=ranks,
        best_rank=best,
        passed=None if best is None else best <= pass_rank,
    )


_CONTROL_VOLUME_FACTOR = 3.0
_CONTROL_ERA_SECONDS = 180 * _SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class MarketRow:
    """A resolved corpus market, for control matching and coverage diagnosis."""

    condition_id: str
    slug: str
    category: str | None
    volume: float
    resolved_at: int
    yes_won: int
    truncated: bool


def match_controls(
    case: MarketRow, universe: list[MarketRow], excluded: set[str], *, k: int = 5
) -> list[MarketRow]:
    """Up to ``k`` control markets matched on category, volume band, era, and side."""
    lo, hi = case.volume / _CONTROL_VOLUME_FACTOR, case.volume * _CONTROL_VOLUME_FACTOR
    pool = [
        m
        for m in universe
        if m.condition_id != case.condition_id
        and m.condition_id not in excluded
        and m.category == case.category
        and lo <= m.volume <= hi
        and abs(m.resolved_at - case.resolved_at) <= _CONTROL_ERA_SECONDS
        and m.yes_won == case.yes_won
    ]
    pool.sort(key=lambda m: (abs(math.log(m.volume / case.volume)), m.slug))
    return pool[:k]


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """Why a published wallet is absent from a case's candidate ranking."""

    case_id: str
    wallet: str
    reason: str
    detail: str
