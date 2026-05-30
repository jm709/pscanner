"""Backtest harness for copy-trading sizing schemes.

Reads the daemon's current watchlist, pulls every position-increasing
(BUY) trade by those wallets from `corpus_trades`, joins with
`market_resolutions`, and walks the chronological event stream
through four sizing schemes. Outputs a markdown report comparing
their realized PnL, ROI, win rate, drawdown, and quarterly P&L.

Spec: docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md
"""
# ruff: noqa: T201  # script prints diagnostics to stdout by design

from __future__ import annotations

import argparse
import csv as _csv
import datetime as dt
import math
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import duckdb

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import copy_selection  # noqa: E402  (import after sys.path bootstrap)

# Default top-K when no copy policy is given. 25 matches the live daemon's
# watchlist target order of magnitude (anti-concentration sizing divides by it).
_DEFAULT_TOP_K = 25


@dataclass(frozen=True)
class Trade:
    """One BUY trade by a watchlist wallet."""

    wallet: str
    condition_id: str
    outcome_side: str  # "YES" or "NO"
    price: float
    notional_usd: float
    ts: int


class EqualWeight:
    """cost = bankroll * position_fraction. No state, no scaling."""

    name = "equal_weight"

    def __init__(self, *, position_fraction: float) -> None:
        """Initialize equal-weight sizing scheme."""
        self._position_fraction = position_fraction

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Return constant position cost regardless of trade details."""
        del trade
        return bankroll * self._position_fraction

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Record resolution outcome (no-op for stateless scheme)."""
        del trade, payout


class ConcentrationCapped:
    """Production sizing: cost = bankroll * position_fraction * mult.

    Mirrors ``pscanner.strategies.evaluators.subgraph_copy
    .SubgraphCopyEvaluator._concentration_multiplier``. The running
    count is incremented AFTER the size is computed so the first trade
    per wallet always gets a clean 1.0 multiplier.
    """

    name = "concentration_capped"

    def __init__(
        self,
        *,
        position_fraction: float,
        min_multiplier: float,
        watchlist_size: int,
    ) -> None:
        """Initialize concentration-capped sizing scheme."""
        self._position_fraction = position_fraction
        self._min_multiplier = min_multiplier
        self._watchlist_size = max(1, watchlist_size)
        self._counts: dict[str, int] = {}

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Size the trade, decaying the multiplier as wallet share rises."""
        total = sum(self._counts.values())
        share = self._counts.get(trade.wallet, 0) / total if total else 0.0
        target_share = 1.0 / self._watchlist_size
        raw = min(1.0, target_share / max(share, target_share))
        mult = max(raw, self._min_multiplier)
        self._counts[trade.wallet] = self._counts.get(trade.wallet, 0) + 1
        return bankroll * self._position_fraction * mult

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Record resolution outcome (no-op; concentration ignores outcomes)."""
        del trade, payout


class FollowSeedSize:
    """cost = min(trade.notional_usd * scale_factor, max_cost_per_trade)."""

    name = "follow_seed_size"

    def __init__(self, *, scale_factor: float, max_cost_per_trade: float) -> None:
        """Initialize follow-seed-size sizing scheme."""
        self._scale_factor = scale_factor
        self._max_cost = max_cost_per_trade

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Return notional * scale, capped at max_cost_per_trade."""
        del bankroll
        return min(trade.notional_usd * self._scale_factor, self._max_cost)

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Record resolution outcome (no-op; stateless scheme)."""
        del trade, payout


class EdgeWeightedCausal:
    """Edge-weighted sizing with strict no-look-ahead.

    ``compute`` uses ONLY trades passed via :meth:`observe_resolution`
    BEFORE this call. The event-walk caller is responsible for
    processing resolutions in ``ts`` order before any subsequent trade
    with ``ts >= resolution.ts``.
    """

    name = "edge_weighted_causal"

    def __init__(
        self,
        *,
        position_fraction: float,
        edge_scale: float,
        min_multiplier: float,
        max_multiplier: float,
        min_trades_for_edge: int,
    ) -> None:
        """Initialize edge-weighted causal sizing scheme."""
        self._position_fraction = position_fraction
        self._edge_scale = edge_scale
        self._min_multiplier = min_multiplier
        self._max_multiplier = max_multiplier
        self._min_trades_for_edge = min_trades_for_edge
        self._resolved: dict[str, list[tuple[float, float]]] = {}

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Size based on wallet's rolling realized edge from prior resolved trades."""
        prior = self._resolved.get(trade.wallet, [])
        if len(prior) < self._min_trades_for_edge:
            mult = 1.0
        else:
            edge = sum(p - imp for p, imp in prior) / len(prior)
            raw = 1.0 + self._edge_scale * edge
            mult = max(self._min_multiplier, min(self._max_multiplier, raw))
        return bankroll * self._position_fraction * mult

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Append (payout, fill price) to this wallet's resolved-trade history."""
        self._resolved.setdefault(trade.wallet, []).append((payout, trade.price))


@dataclass(frozen=True)
class Resolution:
    """A market resolution event from ``market_resolutions``."""

    condition_id: str
    winning_side: str  # "YES" or "NO"
    resolved_at: int


@dataclass(frozen=True)
class TradeEvent:
    """A chronologically-ordered trade event in the simulator stream."""

    kind: Literal["trade"]
    ts: int
    trade: Trade


@dataclass(frozen=True)
class ResolutionEvent:
    """A chronologically-ordered resolution event in the simulator stream."""

    kind: Literal["resolution"]
    ts: int
    resolution: Resolution


@dataclass
class OpenPos:
    """One open paper position held by a sizing scheme."""

    trade_id: int
    wallet: str
    condition_id: str
    outcome_side: str
    shares: float
    cost: float
    ts: int
    price: float


@dataclass
class ResolvedTradeRecord:
    """A position closed by a resolution; carries the final PnL."""

    open_pos: OpenPos
    payout: float
    proceeds: float
    pnl: float
    resolved_at: int


@dataclass
class BacktestState:
    """Per-scheme mutable simulator state."""

    open_positions: dict[int, OpenPos]
    resolved_trades: list[ResolvedTradeRecord]
    cumulative_pnl: float
    nav_series: list[tuple[int, float]]
    open_cost: float = 0.0
    skipped_trades: int = 0


class SizingScheme(Protocol):
    """Protocol every sizing scheme satisfies."""

    name: str

    def compute(self, trade: Trade, bankroll: float) -> float:
        """Return the cost (USD) to size this trade at."""
        ...

    def observe_resolution(self, trade: Trade, payout: float) -> None:
        """Notify the scheme of a resolved trade's payout."""
        ...


class Simulator:
    """Walks an event stream once, dispatching to per-scheme state."""

    def __init__(
        self,
        *,
        schemes: Sequence[SizingScheme],
        bankroll: float,
        enforce_capacity: bool = False,
        max_open_exposure_usd: float | None = None,
    ) -> None:
        """Initialize per-scheme :class:`BacktestState`.

        ``enforce_capacity`` (issue #204 gate #1) refuses a trade when the
        scheme's free capital (``bankroll + cumulative_pnl - open_cost``)
        cannot cover its cost. ``max_open_exposure_usd`` (gate #3) refuses a
        trade when it would push the scheme's open-position cost over the cap.
        Both default off, preserving the edge-measurement mode where every
        scheme processes the identical trade stream.
        """
        self._schemes: Sequence[SizingScheme] = schemes
        self._bankroll = bankroll
        self._enforce_capacity = enforce_capacity
        self._max_open_exposure_usd = max_open_exposure_usd
        self._states: dict[str, BacktestState] = {
            s.name: BacktestState(
                open_positions={},
                resolved_trades=[],
                cumulative_pnl=0.0,
                nav_series=[],
            )
            for s in schemes
        }
        self._next_trade_id = 0

    def state_for(self, scheme: SizingScheme) -> BacktestState:
        """Return the mutable :class:`BacktestState` owned by ``scheme``."""
        return self._states[scheme.name]

    def _can_open(self, state: BacktestState, cost: float) -> bool:
        """Return whether ``cost`` clears both capacity gates for ``state``."""
        if self._enforce_capacity:
            available = self._bankroll + state.cumulative_pnl - state.open_cost
            if available < cost:
                return False
        return (
            self._max_open_exposure_usd is None
            or state.open_cost + cost <= self._max_open_exposure_usd
        )

    def on_trade(self, trade: Trade) -> None:
        """Size the trade and open a position for every scheme that can fund it.

        ``scheme.compute`` is called before the gate, so a skipped trade still
        triggers any compute side effects (e.g. ``ConcentrationCapped``'s
        per-wallet counter). The cost is needed to evaluate the gate, so this
        is unavoidable without a peek/commit split in the scheme protocol.
        """
        for scheme in self._schemes:
            state = self._states[scheme.name]
            cost = scheme.compute(trade, self._bankroll)
            if not self._can_open(state, cost):
                state.skipped_trades += 1
                continue
            shares = cost / trade.price if trade.price > 0 else 0.0
            state.open_positions[self._next_trade_id] = OpenPos(
                trade_id=self._next_trade_id,
                wallet=trade.wallet,
                condition_id=trade.condition_id,
                outcome_side=trade.outcome_side,
                shares=shares,
                cost=cost,
                ts=trade.ts,
                price=trade.price,
            )
            state.open_cost += cost
        self._next_trade_id += 1

    def on_resolution(self, resolution: Resolution) -> None:
        """Close every open position whose market matches, per scheme."""
        for scheme in self._schemes:
            state = self._states[scheme.name]
            closing = [
                tid
                for tid, pos in state.open_positions.items()
                if pos.condition_id == resolution.condition_id
            ]
            for tid in closing:
                pos = state.open_positions.pop(tid)
                state.open_cost -= pos.cost
                payout = 1.0 if resolution.winning_side == pos.outcome_side else 0.0
                proceeds = pos.shares * payout
                pnl = proceeds - pos.cost
                state.resolved_trades.append(
                    ResolvedTradeRecord(
                        open_pos=pos,
                        payout=payout,
                        proceeds=proceeds,
                        pnl=pnl,
                        resolved_at=resolution.resolved_at,
                    )
                )
                state.cumulative_pnl += pnl
                scheme.observe_resolution(
                    Trade(
                        wallet=pos.wallet,
                        condition_id=pos.condition_id,
                        outcome_side=pos.outcome_side,
                        price=pos.price,
                        notional_usd=pos.cost,
                        ts=pos.ts,
                    ),
                    payout=payout,
                )
                state.nav_series.append((resolution.resolved_at, state.cumulative_pnl))


def load_watchlist(daemon_db: Path) -> set[str]:
    """Return the set of active watchlist addresses (lowercase)."""
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{daemon_db}' AS daemon (TYPE sqlite)")
    rows = con.execute(
        "SELECT LOWER(address) FROM daemon.wallet_watchlist WHERE active = 1"
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


def _rows_to_events(rows: Iterable[tuple]) -> Iterator[TradeEvent | ResolutionEvent]:
    """Convert event-row tuples into TradeEvent / ResolutionEvent objects.

    Row shape: (kind, ts, wallet, condition_id, outcome_side, price,
    notional_usd, outcome_yes_won). This shape mirrors the SELECT in
    ``copy_selection._stream_selected`` (and ``load_event_stream`` below)
    and MUST stay in sync with both — they all feed this converter.
    """
    for kind, ts, wallet, cid, side, price, notional, yes_won in rows:
        if kind == "trade":
            yield TradeEvent(
                kind="trade",
                ts=int(ts),
                trade=Trade(
                    wallet=str(wallet),
                    condition_id=str(cid),
                    outcome_side=str(side),
                    price=float(price),
                    notional_usd=float(notional),
                    ts=int(ts),
                ),
            )
        else:
            yield ResolutionEvent(
                kind="resolution",
                ts=int(ts),
                resolution=Resolution(
                    condition_id=str(cid),
                    winning_side="YES" if int(yes_won) == 1 else "NO",
                    resolved_at=int(ts),
                ),
            )


def load_event_stream(
    corpus_db: Path,
    *,
    watchlist: set[str],
    platform: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Iterator[TradeEvent | ResolutionEvent]:
    """Yield :class:`TradeEvent` + :class:`ResolutionEvent` in ``ts`` order.

    A trade is included only when its direction is ``BUY``, its
    ``wallet_address`` is in ``watchlist``, and the trade's market has
    a ``market_resolutions`` row for ``platform``. A resolution is
    included only when at least one of its market's BUY trades passes
    the trade filter above.
    """
    if not watchlist:
        return
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{corpus_db}' AS c (TYPE sqlite)")
    con.execute("CREATE TEMP TABLE wl(addr TEXT)")
    con.executemany("INSERT INTO wl VALUES (?)", [(a,) for a in watchlist])
    ts_filter = ""
    params: list[int] = []
    if start_ts is not None:
        ts_filter += " AND t.ts >= ?"
        params.append(start_ts)
    if end_ts is not None:
        ts_filter += " AND t.ts < ?"
        params.append(end_ts)
    # ts_filter is composed of fixed string literals, not user input. The actual
    # bound values flow via the `?` placeholders below.
    query = f"""
        WITH eligible_trades AS (
          SELECT
            t.wallet_address AS wallet,
            t.condition_id AS condition_id,
            t.outcome_side AS outcome_side,
            t.price AS price,
            t.notional_usd AS notional_usd,
            t.ts AS ts
          FROM c.corpus_trades t
          JOIN c.market_resolutions r
            ON r.platform = t.platform AND r.condition_id = t.condition_id
          WHERE t.platform = ?
            AND t.bs = 'BUY'
            AND LOWER(t.wallet_address) IN (SELECT addr FROM wl)
            {ts_filter}
        ),
        eligible_resolutions AS (
          SELECT
            r.condition_id AS condition_id,
            r.outcome_yes_won AS outcome_yes_won,
            r.resolved_at AS ts
          FROM c.market_resolutions r
          WHERE r.platform = ?
            AND r.condition_id IN (SELECT DISTINCT condition_id FROM eligible_trades)
        )
        SELECT
          'trade' AS kind, ts, wallet, condition_id, outcome_side,
          price, notional_usd, NULL AS outcome_yes_won
        FROM eligible_trades
        UNION ALL
        SELECT
          'resolution' AS kind, ts, NULL, condition_id, NULL,
          NULL, NULL, outcome_yes_won
        FROM eligible_resolutions
        ORDER BY ts ASC
    """  # noqa: S608
    rows = con.execute(query, [platform, *params, platform]).fetchall()
    con.close()
    yield from _rows_to_events(rows)


def _running_max_drawdown(nav_series: list[tuple[int, float]]) -> tuple[float, int]:
    """Return ``(max_drawdown, drawdown_duration_seconds)`` for a NAV series."""
    if not nav_series:
        return 0.0, 0
    peak = nav_series[0][1]
    peak_ts = nav_series[0][0]
    max_dd = 0.0
    max_dd_duration = 0
    for ts, nav in nav_series:
        if nav > peak:
            peak = nav
            peak_ts = ts
        dd = nav - peak
        if dd < max_dd:
            max_dd = dd
            max_dd_duration = ts - peak_ts
    return max_dd, max_dd_duration


_MIN_OBS_FOR_SHARPE = 2
_SECONDS_PER_DAY = 86_400
_TRADING_DAYS_PER_YEAR = 365


def _sharpe_like(nav_series: list[tuple[int, float]]) -> float:
    """Annualized daily-PnL Sharpe-like ratio; ``0.0`` on insufficient data."""
    if len(nav_series) < _MIN_OBS_FOR_SHARPE:
        return 0.0
    by_day: dict[int, float] = {}
    prev_nav = 0.0
    for ts, nav in nav_series:
        day = ts // _SECONDS_PER_DAY
        by_day[day] = nav - prev_nav + by_day.get(day, 0.0)
        prev_nav = nav
    daily = list(by_day.values())
    if len(daily) < _MIN_OBS_FOR_SHARPE:
        return 0.0
    mean = sum(daily) / len(daily)
    var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var) * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _quarter_label(ts: int) -> str:
    """Return the ``YYYY-Q#`` label for a unix timestamp (UTC)."""
    d = dt.datetime.fromtimestamp(ts, tz=dt.UTC)
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def _render_headline_table(sim: Simulator, schemes: Sequence[SizingScheme]) -> str:
    """Return the headline-metrics markdown table (one row per scheme)."""
    rows = [
        "| Scheme | Trades | Cost | Proceeds | PnL | ROI | Win rate"
        " | Avg cost/trade | Unresolved | Skipped |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for scheme in schemes:
        state = sim.state_for(scheme)
        n = len(state.resolved_trades)
        cost = sum(r.open_pos.cost for r in state.resolved_trades)
        proceeds = sum(r.proceeds for r in state.resolved_trades)
        pnl = proceeds - cost
        roi = pnl / cost if cost else 0.0
        wins = sum(1 for r in state.resolved_trades if r.payout > 0)
        win_rate = wins / n if n else 0.0
        avg_cost = cost / n if n else 0.0
        unresolved = len(state.open_positions)
        rows.append(
            f"| {scheme.name} | {n:,} | ${cost:,.2f} | ${proceeds:,.2f}"
            f" | {'+' if pnl >= 0 else ''}${pnl:,.2f}"
            f" | {roi * 100:+.2f}% | {win_rate * 100:.2f}%"
            f" | ${avg_cost:,.2f} | {unresolved} | {state.skipped_trades:,} |\n"
        )
    return "".join(rows)


def _render_risk_table(sim: Simulator, schemes: Sequence[SizingScheme]) -> str:
    """Return the risk-metrics markdown table (one row per scheme)."""
    rows = [
        "| Scheme | Max DD | DD duration (days) | Sharpe-like | Worst trade | Best trade |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for scheme in schemes:
        state = sim.state_for(scheme)
        max_dd, dd_dur_sec = _running_max_drawdown(state.nav_series)
        sharpe = _sharpe_like(state.nav_series)
        pnls = [r.pnl for r in state.resolved_trades]
        worst = min(pnls) if pnls else 0.0
        best = max(pnls) if pnls else 0.0
        rows.append(
            f"| {scheme.name} | ${max_dd:,.2f}"
            f" | {dd_dur_sec / _SECONDS_PER_DAY:.1f}"
            f" | {sharpe:.2f} | ${worst:,.2f} | ${best:,.2f} |\n"
        )
    return "".join(rows)


def _render_quarterly_grid(sim: Simulator, schemes: Sequence[SizingScheme]) -> str:
    """Return the quarterly-PnL markdown grid (one row per scheme)."""
    all_quarters: set[str] = set()
    quarter_pnl: dict[str, dict[str, float]] = {s.name: {} for s in schemes}
    for scheme in schemes:
        state = sim.state_for(scheme)
        for r in state.resolved_trades:
            q = _quarter_label(r.resolved_at)
            all_quarters.add(q)
            quarter_pnl[scheme.name][q] = quarter_pnl[scheme.name].get(q, 0.0) + r.pnl
    sorted_q = sorted(all_quarters)
    if not sorted_q:
        return "(no resolved trades)\n"
    rows = [
        "| Scheme | " + " | ".join(sorted_q) + " |\n",
        "|---|" + "|".join("---:" for _ in sorted_q) + "|\n",
    ]
    for scheme in schemes:
        cells = " | ".join(f"${quarter_pnl[scheme.name].get(q, 0.0):+,.0f}" for q in sorted_q)
        rows.append(f"| {scheme.name} | {cells} |\n")
    return "".join(rows)


def _render_top_contributors(sim: Simulator, schemes: Sequence[SizingScheme]) -> str:
    """Return the top-wallet contributors block for the best-PnL scheme."""
    best_scheme = max(
        schemes,
        key=lambda s: sum(r.pnl for r in sim.state_for(s).resolved_trades),
        default=None,
    )
    if best_scheme is None:
        return ""
    state = sim.state_for(best_scheme)
    by_wallet: dict[str, tuple[int, float, float]] = {}
    for r in state.resolved_trades:
        n_w, c_w, p_w = by_wallet.get(r.open_pos.wallet, (0, 0.0, 0.0))
        by_wallet[r.open_pos.wallet] = (n_w + 1, c_w + r.open_pos.cost, p_w + r.pnl)
    ranked = sorted(by_wallet.items(), key=lambda kv: kv[1][2], reverse=True)[:10]
    rows = [
        f"Scheme: **{best_scheme.name}**\n\n",
        "| Wallet | Copies | Total cost | PnL |\n",
        "|---|---:|---:|---:|\n",
    ]
    for wallet, (n_w, c_w, p_w) in ranked:
        rows.append(f"| `{wallet}` | {n_w} | ${c_w:,.2f} | ${p_w:+,.2f} |\n")
    return "".join(rows)


def _capacity_note(enforce_capacity: bool, max_open_exposure_usd: float | None) -> str:
    """Return the one-line capacity-mode header note."""
    enforcement = "ON" if enforce_capacity else "OFF"
    cap = f"${max_open_exposure_usd:,.2f}" if max_open_exposure_usd is not None else "none"
    return f"Capacity enforcement: {enforcement} | Max open exposure: {cap}\n"


def render_report(
    sim: Simulator,
    *,
    schemes: Sequence[SizingScheme],
    bankroll: float,
    enforce_capacity: bool = False,
    max_open_exposure_usd: float | None = None,
) -> str:
    """Render the backtest result as a multi-section markdown report."""
    return (
        "# Backtest: copy-trading sizing comparison\n"
        f"Bankroll: ${bankroll:,.2f}\n"
        + _capacity_note(enforce_capacity, max_open_exposure_usd)
        + "\n## Headline\n"
        + _render_headline_table(sim, schemes)
        + "\n## Risk\n"
        + _render_risk_table(sim, schemes)
        + "\n## Quarterly PnL\n"
        + _render_quarterly_grid(sim, schemes)
        + "\n## Top contributors (best-PnL scheme)\n"
        + _render_top_contributors(sim, schemes)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with the documented defaults."""
    p = argparse.ArgumentParser(
        description=(
            "Backtest copy-trading sizing schemes against the historical"
            " trade stream of the daemon's current watchlist."
        )
    )
    p.add_argument("--db", default="data/corpus.sqlite3")
    p.add_argument("--watchlist-db", default="data/pscanner.sqlite3")
    p.add_argument("--starting-bankroll-usd", type=float, default=10_000.0)
    p.add_argument("--position-fraction", type=float, default=0.01)
    p.add_argument("--min-multiplier", type=float, default=0.10)
    p.add_argument("--scale-factor", type=float, default=0.01)
    p.add_argument("--max-cost-per-trade", type=float, default=1_000.0)
    p.add_argument("--edge-scale", type=float, default=5.0)
    p.add_argument("--max-multiplier", type=float, default=3.0)
    p.add_argument("--min-trades-for-edge", type=int, default=10)
    p.add_argument("--platform", default="polymarket")
    p.add_argument("--start-ts", type=int, default=None)
    p.add_argument("--end-ts", type=int, default=None)
    p.add_argument(
        "--enforce-capacity",
        action="store_true",
        help=(
            "Refuse a trade when a scheme's free capital"
            " (bankroll + realized PnL - open-position cost) cannot fund it."
            " Off by default (edge-measurement mode)."
        ),
    )
    exposure = p.add_mutually_exclusive_group()
    exposure.add_argument(
        "--max-open-exposure-usd",
        type=float,
        default=None,
        help="Refuse a trade that would push open-position cost over this USD cap.",
    )
    exposure.add_argument(
        "--max-open-exposure-frac",
        type=float,
        default=None,
        help="Same cap as --max-open-exposure-usd, expressed as a fraction of bankroll.",
    )
    p.add_argument(
        "--causal-select",
        action="store_true",
        help="Causally qualify+rank+select wallets from the corpus (ignores --watchlist-db).",
    )
    p.add_argument(
        "--min-resolved",
        type=int,
        default=20,
        help="Qualification: min resolved trades within the edge window.",
    )
    p.add_argument(
        "--edge-window",
        type=int,
        default=0,
        help="Rolling edge window in days; 0 = lifetime.",
    )
    p.add_argument(
        "--rebalance-days",
        type=int,
        default=14,
        help="Days between top-K copy-set recomputes.",
    )
    copy_policy = p.add_mutually_exclusive_group()
    copy_policy.add_argument(
        "--copy-top-k",
        type=int,
        default=None,
        help="Copy the top N wallets by causal edge.",
    )
    copy_policy.add_argument(
        "--copy-capital-per-wallet",
        type=float,
        default=None,
        help="K = floor(bankroll / C).",
    )
    copy_policy.add_argument(
        "--copy-top-frac",
        type=float,
        default=None,
        help="Copy the top X fraction of qualified wallets.",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Optional path to dump per-trade per-scheme rows.",
    )
    return p


def _write_csv(sim: Simulator, schemes: Sequence[SizingScheme], path: str) -> None:
    """Dump per-trade per-scheme rows to ``path`` for downstream analysis."""
    with Path(path).open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(
            [
                "scheme",
                "wallet",
                "condition_id",
                "outcome_side",
                "price",
                "shares",
                "cost",
                "payout",
                "proceeds",
                "pnl",
                "trade_ts",
                "resolved_at",
            ]
        )
        for scheme in schemes:
            state = sim.state_for(scheme)
            for r in state.resolved_trades:
                pos = r.open_pos
                w.writerow(
                    [
                        scheme.name,
                        pos.wallet,
                        pos.condition_id,
                        pos.outcome_side,
                        f"{pos.price}",
                        f"{pos.shares}",
                        f"{pos.cost}",
                        f"{r.payout}",
                        f"{r.proceeds}",
                        f"{r.pnl}",
                        pos.ts,
                        r.resolved_at,
                    ]
                )


def _build_schemes(args: argparse.Namespace, watchlist_size: int) -> list[SizingScheme]:
    """Instantiate the four sizing schemes from CLI args."""
    return [
        EqualWeight(position_fraction=args.position_fraction),
        ConcentrationCapped(
            position_fraction=args.position_fraction,
            min_multiplier=args.min_multiplier,
            watchlist_size=watchlist_size,
        ),
        FollowSeedSize(
            scale_factor=args.scale_factor,
            max_cost_per_trade=args.max_cost_per_trade,
        ),
        EdgeWeightedCausal(
            position_fraction=args.position_fraction,
            edge_scale=args.edge_scale,
            min_multiplier=max(args.min_multiplier, 0.25),
            max_multiplier=args.max_multiplier,
            min_trades_for_edge=args.min_trades_for_edge,
        ),
    ]


def _resolve_policy(args: argparse.Namespace) -> copy_selection.KPolicy:
    """Build the KPolicy from CLI args, defaulting to top-k _DEFAULT_TOP_K."""
    if args.copy_capital_per_wallet is not None:
        return copy_selection.KPolicy(capital_per_wallet=args.copy_capital_per_wallet)
    if args.copy_top_frac is not None:
        return copy_selection.KPolicy(top_frac=args.copy_top_frac)
    return copy_selection.KPolicy(
        top_k=args.copy_top_k if args.copy_top_k is not None else _DEFAULT_TOP_K
    )


def _watchlist_size_for_policy(policy: copy_selection.KPolicy, bankroll: float) -> int:
    """Anti-concentration watchlist size implied by a resolved copy policy."""
    if policy.top_k is not None:
        return policy.top_k
    if policy.capital_per_wallet is not None:
        return max(1, int(bankroll // policy.capital_per_wallet))
    # top_frac: K varies per boundary; use the default as an anti-concentration proxy.
    return _DEFAULT_TOP_K


def main(argv: list[str] | None = None) -> int:
    """Run the backtest end-to-end against the corpus and print the report."""
    args = build_parser().parse_args(argv)
    watchlist: set[str] = set()
    policy: copy_selection.KPolicy | None = None
    if args.causal_select:
        policy = _resolve_policy(args)
        wl_size = _watchlist_size_for_policy(policy, args.starting_bankroll_usd)
    else:
        watchlist = load_watchlist(Path(args.watchlist_db))
        if not watchlist:
            print("Watchlist is empty; nothing to backtest.", file=sys.stderr)
            return 1
        wl_size = len(watchlist)
    schemes = _build_schemes(args, wl_size)
    max_exposure = args.max_open_exposure_usd
    if args.max_open_exposure_frac is not None:
        max_exposure = args.max_open_exposure_frac * args.starting_bankroll_usd
    sim = Simulator(
        schemes=schemes,
        bankroll=args.starting_bankroll_usd,
        enforce_capacity=args.enforce_capacity,
        max_open_exposure_usd=max_exposure,
    )
    if args.causal_select:
        assert policy is not None  # noqa: S101  # set in the causal branch above
        events = _rows_to_events(
            copy_selection.iter_selected_rows(
                Path(args.db),
                platform=args.platform,
                min_resolved=args.min_resolved,
                edge_window_days=args.edge_window,
                rebalance_days=args.rebalance_days,
                policy=policy,
                bankroll=args.starting_bankroll_usd,
                start_ts=args.start_ts,
                end_ts=args.end_ts,
            )
        )
        selection_header = (
            f"Causal selection: min_resolved={args.min_resolved}, "
            f"edge_window={args.edge_window}d, rebalance_days={args.rebalance_days}, "
            f"policy={policy}\n"
        )
    else:
        events = load_event_stream(
            Path(args.db),
            watchlist=watchlist,
            platform=args.platform,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
        )
        selection_header = ""
    for event in events:
        if isinstance(event, TradeEvent):
            sim.on_trade(event.trade)
        else:
            sim.on_resolution(event.resolution)
    print(
        selection_header
        + render_report(
            sim,
            schemes=schemes,
            bankroll=args.starting_bankroll_usd,
            enforce_capacity=args.enforce_capacity,
            max_open_exposure_usd=max_exposure,
        )
    )
    if args.csv:
        _write_csv(sim, schemes, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
