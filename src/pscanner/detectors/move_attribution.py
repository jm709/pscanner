"""Move-attribution detector — bootstraps cluster candidates from market moves.

Subscribes to ``AlertSink``. When an upstream alert (velocity / mispricing /
convergence) names a market, fetches recent trades on that market, walks
back to the start of the burst, tests for a coordinated burst, and emits
``cluster.candidate`` plus upserts the contributors into ``wallet_watchlist``
so the existing :class:`ClusterDetector` can verify them on its next sweep.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

from pscanner.config import MoveAttributionConfig

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BurstHit:
    """One coordinated-burst hit on a single ``(outcome, side, bucket)``."""

    outcome: str
    side: str
    bucket_ts: int
    wallets: tuple[str, ...]
    n_trades: int
    median_size: float
    cv: float


def _bucket_trades(
    trades: Iterable[dict[str, Any]],
    *,
    bucket_seconds: int,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    """Group trades by ``(outcome, side, ts // bucket_seconds)``.

    Drops rows whose ``outcome`` / ``side`` aren't strings or whose
    ``timestamp`` isn't an int — those are malformed and can't be bucketed.
    """
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for t in trades:
        outcome = t.get("outcome")
        side = t.get("side")
        ts = t.get("timestamp")
        if not isinstance(outcome, str) or not isinstance(side, str):
            continue
        if not isinstance(ts, int):
            continue
        bucket_ts = (ts // bucket_seconds) * bucket_seconds
        buckets.setdefault((outcome, side, bucket_ts), []).append(t)
    return buckets


def _evaluate_bucket(
    key: tuple[str, str, int],
    bucket_trades: list[dict[str, Any]],
    *,
    cfg: MoveAttributionConfig,
) -> BurstHit | None:
    """Test one bucket against the burst threshold; return a hit or ``None``.

    Returns ``None`` when distinct-wallet count is below ``min_burst_wallets``,
    when fewer than ``min_burst_wallets`` trades have positive size, or when
    the size CV exceeds ``max_burst_size_cv``.
    """
    outcome, side, bucket_ts = key
    wallet_to_trade: dict[str, dict[str, Any]] = {}
    for t in bucket_trades:
        wallet = t.get("proxyWallet")
        if isinstance(wallet, str) and wallet not in wallet_to_trade:
            wallet_to_trade[wallet] = t
    if len(wallet_to_trade) < cfg.min_burst_wallets:
        return None
    positive_sizes = [
        float(t.get("size") or 0.0)
        for t in wallet_to_trade.values()
        if float(t.get("size") or 0.0) > 0
    ]
    if len(positive_sizes) < cfg.min_burst_wallets:
        return None
    mean = statistics.fmean(positive_sizes)
    if mean <= 0:
        return None
    cv = statistics.pstdev(positive_sizes) / mean
    if cv > cfg.max_burst_size_cv:
        return None
    median_size = statistics.median(positive_sizes)
    kept = _select_contributors(
        wallet_to_trade,
        median_size=median_size,
        limit=cfg.max_contributors_per_burst,
        bucket_ts=bucket_ts,
        outcome=outcome,
        side=side,
    )
    return BurstHit(
        outcome=outcome,
        side=side,
        bucket_ts=bucket_ts,
        wallets=tuple(sorted(kept)),
        n_trades=len(bucket_trades),
        median_size=median_size,
        cv=cv,
    )


def _select_contributors(
    wallet_to_trade: dict[str, dict[str, Any]],
    *,
    median_size: float,
    limit: int,
    bucket_ts: int,
    outcome: str,
    side: str,
) -> list[str]:
    """Keep up to ``limit`` wallets closest to ``median_size``; warn if truncated."""
    ranked = sorted(
        wallet_to_trade.items(),
        key=lambda kv: abs(float(kv[1].get("size") or 0.0) - median_size),
    )
    kept = [w for w, _t in ranked[:limit]]
    if len(wallet_to_trade) > limit:
        _LOG.warning(
            "move_attribution.contributors_truncated",
            bucket_ts=bucket_ts,
            outcome=outcome,
            side=side,
            n_total=len(wallet_to_trade),
            n_kept=limit,
        )
    return kept


def _detect_burst(
    trades: Iterable[dict[str, Any]],
    *,
    cfg: MoveAttributionConfig,
) -> list[BurstHit]:
    """Return one ``BurstHit`` per bucket meeting the threshold.

    Buckets ``(outcome, side, ts // burst_bucket_seconds)``. A bucket fires
    when distinct-wallet count >= ``min_burst_wallets`` and trade-size CV
    (``pstdev / mean``) <= ``max_burst_size_cv``. Up to
    ``max_burst_hits_per_alert`` hits returned (sorted by wallet count desc).
    Each hit's contributor list is truncated to
    ``max_contributors_per_burst`` wallets closest to the bucket's median size.
    """
    buckets = _bucket_trades(trades, bucket_seconds=cfg.burst_bucket_seconds)
    hits: list[BurstHit] = []
    for key, bucket_trades in buckets.items():
        hit = _evaluate_bucket(key, bucket_trades, cfg=cfg)
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda h: len(h.wallets), reverse=True)
    if len(hits) > cfg.max_burst_hits_per_alert:
        _LOG.warning(
            "move_attribution.hits_truncated",
            n_total=len(hits),
            n_kept=cfg.max_burst_hits_per_alert,
        )
        hits = hits[: cfg.max_burst_hits_per_alert]
    return hits


__all__ = ["BurstHit", "_detect_burst"]
