"""Move-attribution detector — bootstraps cluster candidates from market moves.

Subscribes to ``AlertSink``. When an upstream alert (velocity / mispricing /
convergence) names a market, fetches recent trades on that market, walks
back to the start of the burst, tests for a coordinated burst, and emits
``cluster.candidate`` plus upserts the contributors into ``wallet_watchlist``
so the existing :class:`ClusterDetector` can verify them on its next sweep.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import MoveAttributionConfig
from pscanner.poly.data import DataClient
from pscanner.store.repo import WatchlistRepo

_LOG = structlog.get_logger(__name__)

# Two consecutive sub-threshold windows must agree before we declare the burst
# over — one isolated lull is not enough.
_REQUIRED_QUIESCENT_WINDOWS = 2


@dataclass(frozen=True, slots=True)
class BurstHit:
    """One coordinated-burst hit on a single ``(outcome, side, bucket)``."""

    outcome: str
    side: str
    bucket_ts: int
    wallets: tuple[str, ...]
    n_trades: int  # raw bucket trade count, pre-wallet-dedup (n_trades >= len(wallets))
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
    assert mean > 0, "follows from the positive_sizes filter above"  # noqa: S101
    cv = statistics.pstdev(positive_sizes) / mean
    if cv > cfg.max_burst_size_cv:
        return None
    median_size = statistics.median(positive_sizes)
    positive_wallet_to_trade: dict[str, dict[str, Any]] = {
        w: t for w, t in wallet_to_trade.items() if float(t.get("size") or 0.0) > 0
    }
    kept = _select_contributors(
        positive_wallet_to_trade,
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


def _quiescence_threshold(*, n_trades: int, cfg: MoveAttributionConfig) -> float:
    """Per-check-window trade-count threshold below which a window is quiescent."""
    baseline_minutes = max(cfg.lookback_seconds_baseline / 60.0, 1.0)
    baseline_rate = n_trades / baseline_minutes
    return (baseline_rate * cfg.backwalk_multiplier) * (cfg.backwalk_check_window_seconds / 60.0)


def _walk_back_to_burst_start(
    timestamps: list[int],
    *,
    alert_ts: int,
    floor_ts: int,
    threshold: float,
    cfg: MoveAttributionConfig,
) -> int:
    """Step back in ``check_window_seconds`` chunks; return the burst-start ts.

    A window is "quiescent" when its trade count is below ``threshold``. Two
    consecutive quiescent windows stop the walk; otherwise the walk hits
    ``floor_ts``.
    """
    consecutive_quiescent = 0
    cursor = alert_ts
    while cursor > floor_ts:
        window_lo = max(cursor - cfg.backwalk_check_window_seconds, floor_ts)
        in_window = sum(1 for t in timestamps if window_lo <= t < cursor)
        if in_window < threshold:
            consecutive_quiescent += 1
            if consecutive_quiescent >= _REQUIRED_QUIESCENT_WINDOWS:
                return cursor
        else:
            consecutive_quiescent = 0
        cursor = window_lo
    return floor_ts


async def _backwalk(
    client: DataClient,
    *,
    condition_id: str,
    alert_ts: int,
    cfg: MoveAttributionConfig,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Walk back from ``alert_ts`` to the start of the coordinated burst.

    Fetches ``cfg.lookback_seconds_baseline`` (default 24h) of trades, computes
    a baseline trade-rate, then walks back in
    ``cfg.backwalk_check_window_seconds`` (default 300s) steps. The walk stops
    when the trailing-window rate drops below
    ``baseline_rate * cfg.backwalk_multiplier`` for two consecutive windows.
    Hard-capped at ``cfg.max_backwalk_seconds``.

    The 24h fetch is the only API call this function makes — ``burst_trades``
    is sliced from the same list used to compute the baseline.

    Args:
        client: ``DataClient`` to query ``/trades`` against.
        condition_id: Market condition_id whose trades drive the backwalk.
        alert_ts: Anchor timestamp (``until_ts`` of the returned window).
        cfg: Detector config carrying baseline / multiplier / cap knobs.

    Returns:
        ``(since_ts, until_ts, burst_trades)`` where ``burst_trades`` is the
        slice of the fetched trades with ``since_ts <= ts <= until_ts``.
    """
    until_ts = alert_ts
    floor_ts = alert_ts - cfg.max_backwalk_seconds
    baseline_start = alert_ts - cfg.lookback_seconds_baseline
    all_trades = await client.get_market_trades(
        condition_id, since_ts=baseline_start, until_ts=alert_ts
    )
    if not all_trades:
        return floor_ts, until_ts, []
    timestamps = sorted(
        (int(t["timestamp"]) for t in all_trades if isinstance(t.get("timestamp"), int)),
        reverse=True,
    )
    threshold = _quiescence_threshold(n_trades=len(all_trades), cfg=cfg)
    since_ts = _walk_back_to_burst_start(
        timestamps,
        alert_ts=alert_ts,
        floor_ts=floor_ts,
        threshold=threshold,
        cfg=cfg,
    )
    burst_trades = [
        t
        for t in all_trades
        if isinstance(t.get("timestamp"), int) and since_ts <= t["timestamp"] <= until_ts
    ]
    return since_ts, until_ts, burst_trades


class MoveAttributionDetector:
    """Alert-driven detector that bootstraps cluster candidates."""

    name = "move_attribution"

    def __init__(
        self,
        *,
        config: MoveAttributionConfig,
        data_client: DataClient,
        watchlist_repo: WatchlistRepo,
    ) -> None:
        """Bind helpers and the watchlist write target.

        Args:
            config: Section of the root config controlling thresholds.
            data_client: Used for the single ``/trades?market=`` paginated
                fetch on the hot path.
            watchlist_repo: Used to upsert each contributor wallet.
        """
        self._config = config
        self._data_client = data_client
        self._watchlist_repo = watchlist_repo
        self._sink: AlertSink | None = None
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def run(self, sink: AlertSink) -> None:
        """Park forever — this detector is alert-driven, not periodic."""
        if self._sink is None:
            self._sink = sink
        await asyncio.Event().wait()

    def handle_alert_sync(self, alert: Alert) -> None:
        """Subscriber callback fanned out by ``AlertSink.emit``.

        Spawns ``evaluate(alert)`` as a tracked task so it isn't garbage
        collected mid-flight. No-ops if there is no running event loop.
        """
        if alert.detector not in self._config.trigger_detectors:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("move_attribution.no_event_loop", alert_key=alert.alert_key)
            return
        task = loop.create_task(self.evaluate(alert))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate(self, alert: Alert) -> None:
        """Run the full pipeline for one triggering alert."""
        condition_id = alert.body.get("condition_id") if isinstance(alert.body, dict) else None
        if not isinstance(condition_id, str):
            _LOG.debug("move_attribution.no_market", alert_key=alert.alert_key)
            return
        burst_trades = await self._fetch_burst_trades(alert, condition_id)
        if burst_trades is None:
            return
        hits = _detect_burst(burst_trades, cfg=self._config)
        if not hits:
            return
        await self._emit_and_watchlist(alert, condition_id, hits)

    async def _fetch_burst_trades(
        self,
        alert: Alert,
        condition_id: str,
    ) -> list[dict[str, Any]] | None:
        """Run ``_backwalk`` and swallow API errors. Returns ``None`` on failure."""
        try:
            _, _, burst_trades = await _backwalk(
                self._data_client,
                condition_id=condition_id,
                alert_ts=alert.created_at,
                cfg=self._config,
            )
        except Exception:
            _LOG.warning(
                "move_attribution.fetch_failed",
                alert_key=alert.alert_key,
                condition_id=condition_id,
                exc_info=True,
            )
            return None
        return burst_trades

    async def _emit_and_watchlist(
        self,
        triggering: Alert,
        condition_id: str,
        hits: list[BurstHit],
    ) -> None:
        """Emit one ``cluster.candidate`` per hit, watchlist contributors."""
        sink = self._sink
        if sink is None:
            _LOG.warning("move_attribution.no_sink", alert_key=triggering.alert_key)
            return
        reason = f"cluster.candidate-{triggering.alert_key}"
        for hit in hits:
            await self._emit_alert_for_hit(sink, triggering, condition_id, hit)
            self._watchlist_contributors(triggering, hit, reason)

    async def _emit_alert_for_hit(
        self,
        sink: AlertSink,
        triggering: Alert,
        condition_id: str,
        hit: BurstHit,
    ) -> None:
        """Emit a single ``cluster.candidate`` alert for one burst hit."""
        alert_key = f"cluster.candidate:{condition_id}:{hit.outcome}:{hit.side}:{hit.bucket_ts}"
        await sink.emit(
            Alert(
                detector="move_attribution",
                alert_key=alert_key,
                severity="med",
                title=(
                    f"cluster candidate: {len(hit.wallets)} wallets, {hit.outcome} {hit.side} burst"
                ),
                body={
                    "condition_id": condition_id,
                    "outcome": hit.outcome,
                    "side": hit.side,
                    "bucket_ts": hit.bucket_ts,
                    "n_wallets": len(hit.wallets),
                    "n_trades": hit.n_trades,
                    "median_size": hit.median_size,
                    "cv": hit.cv,
                    "triggering_alert_key": triggering.alert_key,
                },
                created_at=triggering.created_at,
            )
        )

    def _watchlist_contributors(
        self,
        triggering: Alert,
        hit: BurstHit,
        reason: str,
    ) -> None:
        """Upsert each contributor wallet, swallowing per-row failures."""
        for wallet in hit.wallets:
            try:
                self._watchlist_repo.upsert(
                    address=wallet,
                    source="cluster.candidate",
                    reason=reason,
                )
            except Exception:
                _LOG.warning(
                    "move_attribution.watchlist_upsert_failed",
                    alert_key=triggering.alert_key,
                    address=wallet,
                    exc_info=True,
                )

    async def aclose(self) -> None:
        """Wait for any in-flight evaluation tasks to finish (test helper)."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*self._pending_tasks, return_exceptions=True)


__all__ = ["BurstHit", "MoveAttributionDetector", "_backwalk", "_detect_burst"]
