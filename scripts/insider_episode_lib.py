"""Pure logic for the insider-episode catalogue (no I/O).

Dataclasses, the frozen scorer (spec Amendment 1), ranking, the blind-test
comparator, the automatic price-move anchor, control matching, and report
rendering. Everything here is deterministic and testable without a corpus.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
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
    """Published wallet addresses. Read only by the blind-test comparator and coverage audit.

    ``unpublished`` holds labels (usernames, truncated prefixes) of published
    wallets that have no full address and so can never be ranked.
    """

    id: str
    wallets: tuple[str, ...]
    unpublished: tuple[str, ...] = ()


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
        limit = b.hour_start + horizon
        peak: float | None = None
        for x in pre[i + 1 :]:
            if x.hour_start > limit:
                break
            peak = x.side_prob_vwap if peak is None else max(peak, x.side_prob_vwap)
        if peak is None:
            continue
        move = peak - b.side_prob_vwap
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
    truncated: bool = False


_FRESH_PLATEAU_DAYS = 10.0
_FRESH_DECAY_DAYS = 60.0
_FRESH_FLOOR = 0.1
_CONC_BLEND = 0.5
_FLIP_BONUS = 0.5
_SECONDS_PER_DAY = 86_400


def age_days(f: WalletFeatures, p: Provenance | None) -> float:
    """Wallet age at its first window buy, in days.

    Both the data-api first activity and the corpus first trade are lower
    bounds on the wallet's true first activity (the API walk may be truncated
    to recent events), so the earlier of the two is used.
    """
    first = f.corpus_first_ts
    if p is not None and p.first_activity_ts is not None:
        first = min(first, p.first_activity_ts)
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
    market_slug: str = ""
    in_sample: bool = False


def compare_published(
    case_id: str,
    anchor_kind: str,
    ranked: list[ScoredWallet],
    sealed: SealedCase,
    *,
    market_slug: str = "",
    in_sample: bool = False,
    pass_rank: int = 10,
) -> BlindResult:
    """Look up each published wallet's rank. ``passed`` is None if none are candidates.

    ``in_sample`` marks dev-split cases, whose wallets informed the scorer design.
    """
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
        market_slug=market_slug,
        in_sample=in_sample,
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
    event_slug: str = ""


def sibling_exclusions(universe: list[MarketRow], registry_cids: set[str]) -> set[str]:
    """Registry markets plus every market sharing an ``event_slug`` with one of them.

    Siblings of a case event (date ladders, adjacent deadlines) carry the same
    news and the same insiders, so they must never serve as controls.
    """
    slugs = {m.event_slug for m in universe if m.condition_id in registry_cids and m.event_slug}
    out = set(registry_cids)
    out.update(m.condition_id for m in universe if m.event_slug and m.event_slug in slugs)
    return out


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


@dataclass(frozen=True, slots=True)
class MarketResult:
    """Ranking on one case market under one anchor."""

    ref: MarketRef
    anchor_kind: str
    anchor_ts: int
    ranked: list[ScoredWallet]
    n_window_trades: int
    window_volume: float


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Ranking on one control market under its auto anchor."""

    market: MarketRow
    anchor_ts: int | None
    ranked: list[ScoredWallet]


@dataclass(frozen=True, slots=True)
class CaseReport:
    """Everything the reporter needs for one case."""

    case: PublicCase
    results: list[MarketResult]
    controls: list[ControlResult]
    blind: list[BlindResult]
    auto_anchor_ts: int | None


_FEATURES = ("age", "log10_n_w", "breadth", "concentration", "flip")
_TOP_N = 10
_MIN_VARIANCE_N = 2


def _concentration(f: WalletFeatures) -> float:
    return min(1.0, f.n_w / f.n_life) if f.n_life else 1.0


def _feature_values(ranked: list[ScoredWallet], name: str) -> list[float]:
    top = ranked[:_TOP_N]
    if name == "age":
        return [r.age for r in top]
    if name == "log10_n_w":
        return [math.log10(1.0 + r.features.n_w) for r in top]
    if name == "breadth":
        return [float(r.features.breadth) for r in top]
    if name == "concentration":
        return [_concentration(r.features) for r in top]
    return [1.0 if r.features.flip else 0.0 for r in top]


def _cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < _MIN_VARIANCE_N or len(b) < _MIN_VARIANCE_N:
        return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / pooled


def fingerprint_table(
    primary: list[ScoredWallet], controls: list[ControlResult]
) -> list[tuple[str, float, float, float]]:
    """Top-10 feature means for the case versus pooled control top-10s, with Cohen's d."""
    if not controls or not primary:
        return []
    rows: list[tuple[str, float, float, float]] = []
    for name in _FEATURES:
        case_vals = _feature_values(primary, name)
        ctl_vals = [v for c in controls for v in _feature_values(c.ranked, name)]
        if not case_vals or not ctl_vals:
            continue
        rows.append(
            (
                name,
                statistics.mean(case_vals),
                statistics.mean(ctl_vals),
                _cohen_d(case_vals, ctl_vals),
            )
        )
    return rows


def _iso(ts: int | None) -> str:
    if ts is None:
        return "n/a"
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def _ranked_table(ranked: list[ScoredWallet], anchor: int, limit: int = 25) -> list[str]:
    lines = [
        "| Rank | Wallet | Score | Age (d) | Window $ | Buys | First buy | Last buy |"
        " Markets | Conc | Flip |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|",
    ]
    for r in ranked[:limit]:
        f = r.features
        first_h = (f.first_buy_ts - anchor) / 3600
        last_h = (f.last_buy_ts - anchor) / 3600
        lines.append(
            f"| {r.rank} | `{f.wallet}` | {r.score:.3f} | {r.age:.1f} | {f.n_w:,.0f} | {f.n_buys} |"
            f" {first_h:+.1f}h | {last_h:+.1f}h | {f.breadth} | {_concentration(f):.2f} |"
            f" {'yes' if f.flip else ''} |"
        )
    return lines


def _primary_news_ranking(r: CaseReport) -> list[ScoredWallet]:
    for x in r.results:
        if x.ref.role.startswith("primary") and x.anchor_kind == "news":
            return x.ranked
    return []


def _render_controls(r: CaseReport) -> list[str]:
    out = ["## Controls", ""]
    if not r.controls:
        return [*out, "No controls matched; fingerprint unavailable.", ""]
    for ctl in r.controls:
        out.append(
            f"- `{ctl.market.slug}` · anchor {_iso(ctl.anchor_ts)} · {len(ctl.ranked)} candidates"
        )
    out += [
        "",
        "| Feature | Case top-10 mean | Control top-10 mean | Cohen's d |",
        "|---|---:|---:|---:|",
    ]
    for name, a, b, d in fingerprint_table(_primary_news_ranking(r), r.controls):
        out.append(f"| {name} | {a:.3f} | {b:.3f} | {d:+.2f} |")
    return [*out, ""]


_MIN_CANDIDATES_FOR_MEDIAN = 10
_TOP_FLAGS = (5, 10, 25)


def _rank_text(rank: int | None, n: int) -> str:
    if rank is None:
        return "not a candidate"
    pct = 100.0 * rank / n if n else 0.0
    tops = [f"top {k}" for k in _TOP_FLAGS if rank <= k]
    flag = f"; {', '.join(tops)}" if tops else ""
    return f"{rank} / {n} (top {pct:.1f}%{flag})"


def _render_blind(blind: list[BlindResult]) -> list[str]:
    label = " (in-sample: dev split)" if all(b.in_sample for b in blind) else ""
    out = [f"## Blind test{label}", ""]
    for b in blind:
        if b.passed is None:
            verdict = "no published wallet among candidates"
        else:
            verdict = "PASS" if b.passed else "FAIL"
        out.append(
            f"- `{b.market_slug}` · {b.anchor_kind} anchor: {verdict}; {b.n_candidates} candidates"
        )
        for w, rank in b.ranks.items():
            out.append(f"  - `{w}`: {_rank_text(rank, b.n_candidates)}")
    return [*out, ""]


def render_case_report(r: CaseReport) -> str:
    """Markdown report for one case."""
    c = r.case
    gap = "n/a" if r.auto_anchor_ts is None else f"{(r.auto_anchor_ts - c.news_ts) / 3600:+.1f}h"
    out = [
        f"# {c.name}",
        "",
        f"Case `{c.id}` · {c.confidence} · split {c.split}",
        "",
        f"News anchor: {_iso(c.news_ts)} (precision: {c.news_precision})",
        f"Auto anchor: {_iso(r.auto_anchor_ts)} (gap to news: {gap})",
        "",
    ]
    for res in r.results:
        out += [
            f"## {res.ref.slug} ({res.ref.role}, {res.anchor_kind} anchor)",
            "",
            f"{res.n_window_trades} window trades, ${res.window_volume:,.0f} volume,"
            f" {len(res.ranked)} candidates",
            "",
            *_ranked_table(res.ranked, res.anchor_ts),
            "",
        ]
    out += _render_controls(r)
    if r.blind:
        out += _render_blind(r.blind)
    return "\n".join(out)


def _fmt_rank(best: int | None) -> str:
    return "—" if best is None else str(best)


def case_best_rank(r: CaseReport, kind: str) -> tuple[int | None, list[str]]:
    """Best published-wallet rank across the case's primary markets under ``kind``.

    Markets with fewer than ten candidates are excluded from the best rank
    and returned as the second element so the summary can say so.
    """
    best: int | None = None
    small: list[str] = []
    for b in r.blind:
        if b.anchor_kind != kind or b.best_rank is None:
            continue
        if b.n_candidates < _MIN_CANDIDATES_FOR_MEDIAN:
            small.append(b.market_slug)
            continue
        best = b.best_rank if best is None else min(best, b.best_rank)
    return best, small


def _median_lines(reports: list[CaseReport], kind: str) -> list[str]:
    holdout: list[int] = []
    everyone: list[int] = []
    for r in reports:
        best, _ = case_best_rank(r, kind)
        if best is None:
            continue
        everyone.append(best)
        if r.case.split == "holdout":
            holdout.append(best)
    med_h = f"{statistics.median(holdout):.0f}" if holdout else "n/a"
    lines = [f"Median best rank ({kind} anchor, holdout, {len(holdout)} testable cases): {med_h}"]
    if len(holdout) < _LOO_THRESHOLD:
        med_a = f"{statistics.median(everyone):.0f}" if everyone else "n/a"
        lines.append(
            f"  Fewer than three testable holdout cases: leave-one-out fallback reports the median"
            f" over all testable cases, {len(everyone)}: {med_a}. Dev cases are in-sample"
            " (their wallets informed the scorer design) and the scorer has no fitted"
            " parameters, so this is weaker than a true held-out result."
        )
    small = [(r.case.id, s) for r in reports for s in case_best_rank(r, kind)[1]]
    if small:
        lines.append(
            "  Excluded from the median (fewer than 10 candidates): "
            + ", ".join(f"{cid} `{s}`" for cid, s in small)
        )
    return lines


_LOO_THRESHOLD = 3


def render_summary(reports: list[CaseReport], coverage: list[CoverageRow]) -> str:
    """Cross-case summary: blind-test medians, anchor gaps, and the coverage audit."""
    out = ["# Insider-episode catalogue: summary", ""]
    for kind in ("news", "auto"):
        out += _median_lines(reports, kind)
    gaps = sorted(
        (r.auto_anchor_ts - r.case.news_ts) / 3600 for r in reports if r.auto_anchor_ts is not None
    )
    gap_text = ", ".join(f"{g:+.1f}" for g in gaps) if gaps else "n/a"
    out += ["", f"Auto-anchor gap to news, hours (n={len(gaps)}): {gap_text}", ""]
    out += [
        "## Cases",
        "",
        "| Case | Split | Tier | Candidates (primary) | Blind (news) | Blind (auto) |",
        "|---|---|---|---:|---|---|",
    ]
    for r in reports:
        n = len(_primary_news_ranking(r))
        bn, _ = case_best_rank(r, "news")
        ba, _ = case_best_rank(r, "auto")
        tag = " (in-sample)" if r.case.split != "holdout" and bn is not None else ""
        out.append(
            f"| {r.case.id} | {r.case.split} | {r.case.confidence} | {n} |"
            f" {_fmt_rank(bn)}{tag} | {_fmt_rank(ba)} |"
        )
    out += [
        "",
        "## Published wallets not in corpus",
        "",
        "| Case | Wallet | Reason | Detail |",
        "|---|---|---|---|",
    ]
    for cv in coverage:
        out.append(f"| {cv.case_id} | `{cv.wallet}` | {cv.reason} | {cv.detail} |")
    return "\n".join(out) + "\n"


def _csv_row(
    case_id: str, slug: str, r: ScoredWallet, *, is_control: bool, kind: str, anchor: int | None
) -> dict[str, object]:
    f = r.features
    return {
        "case_id": case_id,
        "market_slug": slug,
        "is_control": is_control,
        "anchor_kind": kind,
        "anchor_ts": anchor,
        "rank": r.rank,
        "wallet": f.wallet,
        "score": round(r.score, 6),
        "age_days": round(r.age, 3),
        "n_w": f.n_w,
        "n_buys": f.n_buys,
        "first_buy_ts": f.first_buy_ts,
        "last_buy_ts": f.last_buy_ts,
        "breadth": f.breadth,
        "n_life": f.n_life,
        "concentration": _concentration(f),
        "flip": f.flip,
        "first_activity_ts": None if r.provenance is None else r.provenance.first_activity_ts,
    }


def csv_rows(reports: list[CaseReport]) -> list[dict[str, object]]:
    """Flat rows for every scored wallet across cases and controls."""
    rows: list[dict[str, object]] = []
    for rep in reports:
        for res in rep.results:
            rows += [
                _csv_row(
                    rep.case.id,
                    res.ref.slug,
                    r,
                    is_control=False,
                    kind=res.anchor_kind,
                    anchor=res.anchor_ts,
                )
                for r in res.ranked
            ]
        for ctl in rep.controls:
            rows += [
                _csv_row(
                    rep.case.id,
                    ctl.market.slug,
                    r,
                    is_control=True,
                    kind="auto",
                    anchor=ctl.anchor_ts,
                )
                for r in ctl.ranked
            ]
    return rows
