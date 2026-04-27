"""Coordinated-wallet cluster detector — issue #28.

Auto-detects groups of wallets that look coordinated (e.g. wash-trading
farms, multi-account Sybil rings) and emits two alert flavours:

* ``cluster.discovered`` — one-shot, fired the first time a candidate
  cluster scores at or above ``discovery_score_threshold``.
* ``cluster.active`` — fired by the trade-callback path when several
  members of a known cluster hit the same condition_id within a short
  window.

Discovery is a periodic scan (cadence ``scan_interval_seconds``) over the
``wallet_first_seen`` cache. Active monitoring is callback-driven: the
detector subscribes to ``TradeCollector.subscribe_new_trade`` via
``handle_trade_sync`` (mirroring whales / convergence). Because the
detector has both a periodic ``run`` AND a per-trade callback, it does
not inherit from :class:`TradeDrivenDetector` (whose ``run`` simply
parks); instead it implements the trade-callback plumbing inline.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
import time
from collections.abc import Iterable

import structlog

from pscanner.alerts.models import Alert, Severity
from pscanner.alerts.sink import AlertSink
from pscanner.config import ClusterConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import (
    CachedMarket,
    MarketCacheRepo,
    WalletCluster,
    WalletClusterMembersRepo,
    WalletClustersRepo,
    WalletFirstSeen,
    WalletFirstSeenRepo,
    WalletTrade,
    WalletTradesRepo,
)
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)
_BEHAVIOR_FARMER_LO = 0.5
_BEHAVIOR_FARMER_HI = 0.8
_BEHAVIOR_TRADER_HI = 0.9
_RECENT_TRADE_LIMIT = 500
_MIN_WALLETS_PER_SHARED_MARKET = 3
_HIGH_SEVERITY_DISCOVERED_SCORE = 7
_MIN_TRADES_FOR_CV = 2


class ClusterDetector:
    """Detector that emits alerts for coordinated wallet clusters."""

    name = "cluster"

    def __init__(
        self,
        *,
        config: ClusterConfig,
        wallet_first_seen: WalletFirstSeenRepo,
        trades_repo: WalletTradesRepo,
        market_cache: MarketCacheRepo,
        clusters_repo: WalletClustersRepo,
        members_repo: WalletClusterMembersRepo,
        clock: Clock | None = None,
    ) -> None:
        """Build the detector with its config and persistence dependencies.

        Args:
            config: Cluster-specific thresholds and cadences.
            wallet_first_seen: Source of cached wallet first-activity rows
                (used as the population for discovery scans).
            trades_repo: Read access to ``wallet_trades`` rows for shared
                market discovery and active monitoring.
            market_cache: Resolves ``condition_id`` to ``CachedMarket`` for
                liquidity / volume thresholding and alert title enrichment.
            clusters_repo: Persists detected ``wallet_clusters`` rows.
            members_repo: Persists ``wallet_cluster_members`` rows.
            clock: Injectable :class:`Clock`. Defaults to :class:`RealClock`.
        """
        self._config = config
        self._first_seen = wallet_first_seen
        self._trades = trades_repo
        self._market_cache = market_cache
        self._clusters = clusters_repo
        self._members = members_repo
        self._clock: Clock = clock if clock is not None else RealClock()
        self._sink: AlertSink | None = None
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def run(self, sink: AlertSink) -> None:
        """Periodic discovery loop — scans for new clusters on a fixed cadence.

        Active monitoring is callback-driven via
        ``TradeCollector.subscribe_new_trade``; this loop only owns the
        periodic discovery sweep.

        Args:
            sink: Shared alert sink used by both discovery and active paths.
        """
        if self._sink is None:
            self._sink = sink
        while True:
            try:
                await self.discovery_scan(sink)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("cluster.discovery_failed")
            await self._clock.sleep(self._config.scan_interval_seconds)

    def handle_trade_sync(self, trade: WalletTrade) -> None:
        """Sync entry called by the trade collector callback.

        Spawns ``evaluate_active(trade)`` as an async task and tracks it so
        it isn't garbage collected mid-flight. No-ops if there is no running
        event loop (e.g. test setup that hasn't started one yet).

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug("cluster.no_event_loop", tx=trade.transaction_hash)
            return
        task = loop.create_task(self.evaluate_active(trade))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def evaluate_active(self, trade: WalletTrade) -> None:
        """Emit ``cluster.active`` when several cluster members hit one market.

        Args:
            trade: Newly-inserted ``WalletTrade`` row.
        """
        if self._sink is None:
            _LOG.warning("cluster.no_sink", tx=trade.transaction_hash)
            return
        cluster_id = self._members.cluster_for_wallet(trade.wallet)
        if cluster_id is None:
            return
        members = set(self._members.members_of(cluster_id))
        if not members:
            return
        since = trade.timestamp - self._config.active_window_seconds
        recent = self._trades.distinct_wallets_for_condition(
            trade.condition_id,
            since=since,
        )
        active = recent & members
        if len(active) < self._config.active_min_members:
            return
        cached = self._market_cache.get_by_condition_id(trade.condition_id)
        alert = self._build_active_alert(trade, cluster_id, members, active, cached)
        if await self._sink.emit(alert):
            self._clusters.update_last_active(cluster_id, int(self._clock.now()))

    async def discovery_scan(self, sink: AlertSink) -> int:
        """Scan ``wallet_first_seen`` for new candidate clusters.

        Args:
            sink: Shared alert sink used to publish newly-discovered clusters.

        Returns:
            Count of newly-detected clusters whose alert was emitted.
        """
        if self._sink is None:
            self._sink = sink
        recent = self._first_seen.list_recent(within=self._config.discovery_lookback_days)
        if len(recent) < self._config.min_cluster_size:
            return 0
        seen_cluster_ids: set[str] = set()
        new_count = 0
        # Path 1: existing creation-window partition.
        for group in self._iter_candidate_groups(recent):
            if await self._consider_group(group, seen_cluster_ids, sink):
                new_count += 1
        # Path 2: co-occurrence partition (organic clusters via shared
        # obscure-market overlap). _consider_group's _cluster_id_for SHA256
        # dedupe ensures clusters surfaced by both paths emit exactly once.
        for group in self._iter_co_trade_groups(recent):
            if await self._consider_group(group, seen_cluster_ids, sink):
                new_count += 1
        return new_count

    async def _consider_group(
        self,
        group: list[WalletFirstSeen],
        seen_cluster_ids: set[str],
        sink: AlertSink,
    ) -> bool:
        """Evaluate one candidate group; return True iff a new alert emitted."""
        cluster_id = _cluster_id_for(group)
        if cluster_id in seen_cluster_ids:
            return False
        seen_cluster_ids.add(cluster_id)
        if self._clusters.get(cluster_id) is not None:
            return False
        scored = self._score_candidate(group)
        if scored is None:
            return False
        score, shared_markets, behavior_tag = scored
        if score < self._config.discovery_score_threshold:
            return False
        return await self._emit_discovered(
            group=group,
            cluster_id=cluster_id,
            score=score,
            shared_markets=shared_markets,
            behavior_tag=behavior_tag,
            sink=sink,
        )

    def _iter_candidate_groups(
        self,
        recent: list[WalletFirstSeen],
    ) -> Iterable[list[WalletFirstSeen]]:
        """Yield maximal creation-clustered groups.

        Wallets are pre-sorted by ``first_activity_at`` ascending. We greedy-
        partition the timeline into "waves": a new wave starts when the gap
        from the previous wallet exceeds ``creation_window_seconds``, and the
        wave keeps extending as long as the next wallet is within window of
        the *previous* wallet (chained, not anchored to the wave start). This
        avoids the combinatorial explosion of overlapping subgroups while
        still preserving the spec's "wallets created in clusters" intent.
        Groups smaller than ``min_cluster_size`` are skipped.
        """
        sorted_recent = [r for r in recent if r.first_activity_at is not None]
        sorted_recent.sort(key=lambda r: r.first_activity_at or 0)
        window = self._config.creation_window_seconds
        if not sorted_recent:
            return
        wave: list[WalletFirstSeen] = [sorted_recent[0]]
        for current in sorted_recent[1:]:
            prev_at = wave[-1].first_activity_at or 0
            cur_at = current.first_activity_at or 0
            if cur_at - prev_at <= window:
                wave.append(current)
                continue
            if len(wave) >= self._config.min_cluster_size:
                yield wave
            wave = [current]
        if len(wave) >= self._config.min_cluster_size:
            yield wave

    def _iter_co_trade_groups(
        self,
        recent: list[WalletFirstSeen],
    ) -> Iterable[list[WalletFirstSeen]]:
        """Yield candidate groups derived from shared-obscure-market overlap.

        For each pair of wallets in ``recent``, count their shared obscure
        markets. Wallets connected by edges of >= ``min_shared_markets``
        shared markets form connected components; each component of size
        in ``[min_cluster_size, max_co_trade_group_size]`` is yielded.

        This path is independent of creation timestamps — it discovers
        clusters that grew organically over time. Existing scoring
        (B/C/D + the refactored Signal A) is applied per component.
        """
        if len(recent) < self._config.min_cluster_size:
            return

        obscure_markets = self._build_obscure_markets_index(recent)
        adjacency = self._build_cotrade_adjacency(obscure_markets)
        yield from self._yield_components(recent, adjacency)

    def _build_obscure_markets_index(
        self,
        recent: list[WalletFirstSeen],
    ) -> dict[str, set[ConditionId]]:
        """Per-wallet set of obscure-market condition_ids.

        Per-wallet ``recent_for_wallet`` failures are isolated — that
        wallet's set is treated as empty (no edges incident on it). Other
        wallets unaffected.
        """
        index: dict[str, set[ConditionId]] = {}
        for wallet in recent:
            try:
                trades = self._trades.recent_for_wallet(
                    wallet.address,
                    limit=_RECENT_TRADE_LIMIT,
                )
            except Exception:
                _LOG.warning(
                    "cluster.cotrade_trades_failed",
                    wallet=wallet.address,
                    exc_info=True,
                )
                index[wallet.address] = set()
                continue
            obscure: set[ConditionId] = set()
            for trade in trades:
                cached = self._market_cache.get_by_condition_id(trade.condition_id)
                if cached is None:
                    continue
                if not self._is_obscure_market(cached):
                    continue
                obscure.add(trade.condition_id)
            index[wallet.address] = obscure
        return index

    def _build_cotrade_adjacency(
        self,
        obscure_markets: dict[str, set[ConditionId]],
    ) -> dict[str, set[str]]:
        """Build undirected adjacency: edge iff shared obscure markets >= threshold."""
        adjacency: dict[str, set[str]] = {addr: set() for addr in obscure_markets}
        addrs = sorted(adjacency)
        threshold = self._config.min_shared_markets
        for i, a in enumerate(addrs):
            ma = obscure_markets[a]
            if not ma:
                continue
            for b in addrs[i + 1 :]:
                mb = obscure_markets[b]
                if len(ma & mb) >= threshold:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        return adjacency

    def _yield_components(
        self,
        recent: list[WalletFirstSeen],
        adjacency: dict[str, set[str]],
    ) -> Iterable[list[WalletFirstSeen]]:
        """BFS the adjacency graph and yield components within size bounds."""
        by_address = {w.address: w for w in recent}
        visited: set[str] = set()
        for start in sorted(adjacency):
            if start in visited:
                continue
            component_addrs = self._bfs_component(start, adjacency)
            visited |= component_addrs
            if len(component_addrs) < self._config.min_cluster_size:
                continue
            cap = self._config.max_co_trade_group_size
            if len(component_addrs) > cap:
                _LOG.warning(
                    "cluster.cotrade_group_truncated",
                    original_size=len(component_addrs),
                    keep=cap,
                )
                kept = sorted(component_addrs)[:cap]
                yield [by_address[a] for a in kept]
            else:
                yield [by_address[a] for a in sorted(component_addrs)]

    @staticmethod
    def _bfs_component(
        start: str,
        adjacency: dict[str, set[str]],
    ) -> set[str]:
        """Return the set of addresses reachable from ``start`` (inclusive)."""
        component: set[str] = {start}
        queue: list[str] = [start]
        while queue:
            current = queue.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        return component

    def _compute_creation_cohesion_score(self, group: list[WalletFirstSeen]) -> int:
        """Score creation-timestamp cohesion for a candidate group.

        Returns 2 if the group's non-NULL ``first_activity_at`` timestamps
        all fit within ``creation_window_seconds``, else 0. Wallets with
        NULL ``first_activity_at`` are excluded from the comparison. If the
        remaining non-NULL set is below ``min_cluster_size``, returns 0 —
        we can't conclude creation clustering with too few timestamps.
        """
        timestamps = [w.first_activity_at for w in group if w.first_activity_at is not None]
        if len(timestamps) < self._config.min_cluster_size:
            return 0
        if max(timestamps) - min(timestamps) <= self._config.creation_window_seconds:
            return 2
        return 0

    def _score_candidate(
        self,
        group: list[WalletFirstSeen],
    ) -> tuple[int, list[CachedMarket], str | None] | None:
        """Score a candidate group and return ``(score, shared_markets, tag)``.

        Returns ``None`` when there is no usable shared-market overlap to
        score — that's an early exit, not a real candidate.
        """
        wallets = [w.address for w in group]
        score = 0
        # Signal A — wallet creation clustering. For Path 1 (creation-window
        # partition) this always returns 2 by construction; for Path 2
        # (co-occurrence — added in a later task) it is a real bonus when
        # the discovered cluster also has tight creation timestamps.
        score += self._compute_creation_cohesion_score(group)
        shared = self._find_shared_obscure_markets(wallets)
        if len(shared) >= self._config.min_shared_markets:
            score += 2
        # Signals C and D inspect cluster trades on shared markets.
        cluster_trades_by_market = self._collect_cluster_trades(wallets, shared)
        if self._has_size_correlation(cluster_trades_by_market):
            score += 1
        if self._has_direction_correlation(cluster_trades_by_market):
            score += 2
        all_trades = [t for trades in cluster_trades_by_market.values() for t in trades]
        behavior_tag = _behavior_tag_for(all_trades)
        return score, shared, behavior_tag

    def _find_shared_obscure_markets(self, wallets: list[str]) -> list[CachedMarket]:
        """Return obscure markets traded by ≥3 distinct cluster wallets.

        "Obscure" means below ``max_shared_market_liquidity_usd`` AND below
        ``max_shared_market_volume_usd``. Markets missing from the cache or
        with NULL liquidity / volume are conservatively excluded.
        """
        traders_per_market: dict[ConditionId, set[str]] = {}
        for wallet in wallets:
            for trade in self._trades.recent_for_wallet(wallet, limit=_RECENT_TRADE_LIMIT):
                traders_per_market.setdefault(trade.condition_id, set()).add(wallet)
        shared: list[CachedMarket] = []
        for condition_id, traders in traders_per_market.items():
            if len(traders) < _MIN_WALLETS_PER_SHARED_MARKET:
                continue
            cached = self._market_cache.get_by_condition_id(condition_id)
            if cached is None:
                continue
            if not self._is_obscure_market(cached):
                continue
            shared.append(cached)
        return shared

    def _is_obscure_market(self, cached: CachedMarket) -> bool:
        """Return whether ``cached`` clears both obscurity thresholds."""
        if cached.liquidity_usd is None or cached.volume_usd is None:
            return False
        if cached.liquidity_usd > self._config.max_shared_market_liquidity_usd:
            return False
        return cached.volume_usd <= self._config.max_shared_market_volume_usd

    def _collect_cluster_trades(
        self,
        wallets: list[str],
        shared: list[CachedMarket],
    ) -> dict[ConditionId, list[WalletTrade]]:
        """Group cluster trades by ``condition_id`` for the shared markets."""
        shared_ids = {m.condition_id for m in shared if m.condition_id is not None}
        out: dict[ConditionId, list[WalletTrade]] = {cid: [] for cid in shared_ids}
        for wallet in wallets:
            for trade in self._trades.recent_for_wallet(wallet, limit=_RECENT_TRADE_LIMIT):
                if trade.condition_id in shared_ids:
                    out[trade.condition_id].append(trade)
        return out

    def _has_size_correlation(
        self,
        trades_by_market: dict[ConditionId, list[WalletTrade]],
    ) -> bool:
        """Return whether any shared market shows low size dispersion."""
        for trades in trades_by_market.values():
            sizes = [t.size for t in trades if t.size > 0]
            if len(sizes) < _MIN_TRADES_FOR_CV:
                continue
            mean = statistics.fmean(sizes)
            if mean <= 0:
                continue
            stdev = statistics.pstdev(sizes)
            cv = stdev / mean
            if cv < self._config.max_trade_size_cv:
                return True
        return False

    def _has_direction_correlation(
        self,
        trades_by_market: dict[ConditionId, list[WalletTrade]],
    ) -> bool:
        """Return whether any shared market shows synchronized buys.

        For each shared market, group trades into buckets of width
        ``direction_window_seconds`` and count distinct wallets that bought
        the same outcome (asset_id+side pair) within one bucket. If any
        bucket reaches ``min_direction_correlation_count``, the signal fires.
        """
        threshold = self._config.min_direction_correlation_count
        bucket = self._config.direction_window_seconds
        for trades in trades_by_market.values():
            if self._market_has_direction_burst(trades, bucket=bucket, threshold=threshold):
                return True
        return False

    @staticmethod
    def _market_has_direction_burst(
        trades: list[WalletTrade],
        *,
        bucket: int,
        threshold: int,
    ) -> bool:
        """Return whether ``trades`` contain a synchronized direction burst."""
        # Key: (asset_id, side, bucket_index) -> distinct wallets
        groups: dict[tuple[str, str, int], set[str]] = {}
        for trade in trades:
            idx = trade.timestamp // bucket if bucket > 0 else 0
            key = (str(trade.asset_id), trade.side, idx)
            groups.setdefault(key, set()).add(trade.wallet)
        return any(len(wallets) >= threshold for wallets in groups.values())

    async def _emit_discovered(
        self,
        *,
        group: list[WalletFirstSeen],
        cluster_id: str,
        score: int,
        shared_markets: list[CachedMarket],
        behavior_tag: str | None,
        sink: AlertSink,
    ) -> bool:
        """Persist the cluster and emit a ``cluster.discovered`` alert."""
        now = int(self._clock.now())
        timestamps = [w.first_activity_at for w in group if w.first_activity_at is not None]
        first_at = min(timestamps) if timestamps else now
        last_at = max(timestamps) if timestamps else now
        cluster = WalletCluster(
            cluster_id=cluster_id,
            member_count=len(group),
            first_member_created_at=first_at,
            last_member_created_at=last_at,
            shared_market_count=len(shared_markets),
            behavior_tag=behavior_tag,
            detection_score=score,
            first_detected_at=now,
            last_active_at=now,
        )
        self._clusters.upsert(cluster)
        for wallet in sorted(w.address for w in group):
            self._members.add_member(cluster_id, wallet)
        alert = _build_discovered_alert(cluster, shared_markets, group)
        return await sink.emit(alert)

    def _build_active_alert(
        self,
        trade: WalletTrade,
        cluster_id: str,
        members: set[str],
        active: set[str],
        cached: CachedMarket | None,
    ) -> Alert:
        """Construct the ``cluster.active`` alert payload."""
        now = int(self._clock.now())
        day = time.strftime("%Y%m%d", time.gmtime(now))
        alert_key = f"cluster.active:{cluster_id}:{trade.condition_id}:{day}"
        severity: Severity = "high" if 2 * len(active) >= len(members) else "med"
        title_target = (cached.title if cached is not None else None) or trade.condition_id
        title = f"cluster active on {title_target}: {len(active)}/{len(members)} members"
        body: dict[str, object] = {
            "cluster_id": cluster_id,
            "condition_id": trade.condition_id,
            "market_title": cached.title if cached is not None else None,
            "active_members": len(active),
            "total_member_count": len(members),
            "window_seconds": self._config.active_window_seconds,
            "first_member_trade_at": trade.timestamp,
        }
        return Alert(
            detector="cluster",
            alert_key=alert_key,
            severity=severity,
            title=title,
            body=body,
            created_at=now,
        )


def _build_discovered_alert(
    cluster: WalletCluster,
    shared_markets: list[CachedMarket],
    group: list[WalletFirstSeen],
) -> Alert:
    """Construct the ``cluster.discovered`` alert payload."""
    severity: Severity = (
        "high" if cluster.detection_score >= _HIGH_SEVERITY_DISCOVERED_SCORE else "med"
    )
    titles = [m.title for m in shared_markets if m.title]
    body: dict[str, object] = {
        "cluster_id": cluster.cluster_id,
        "member_count": cluster.member_count,
        "creation_window_seconds": cluster.last_member_created_at - cluster.first_member_created_at,
        "shared_markets": titles,
        "behavior_tag": cluster.behavior_tag,
        "detection_score": cluster.detection_score,
        "first_seen_active": cluster.first_detected_at,
    }
    title = f"new cluster: {cluster.member_count} wallets, score {cluster.detection_score}"
    alert_key = f"cluster.discovered:{cluster.cluster_id}"
    del group  # currently unused — wallets are addressable via the repo
    return Alert(
        detector="cluster",
        alert_key=alert_key,
        severity=severity,
        title=title,
        body=body,
        created_at=cluster.first_detected_at,
    )


def _cluster_id_for(group: Iterable[WalletFirstSeen]) -> str:
    """Compute a deterministic cluster_id from a group of wallets.

    The id is the SHA-256 hex digest of the cluster's sorted lowercase
    addresses joined by ``|``. Stable across runs so re-detections collapse
    onto the same row, which is the dedupe contract.
    """
    addresses = sorted({w.address.lower() for w in group})
    digest = hashlib.sha256("|".join(addresses).encode("utf-8")).hexdigest()
    return f"cluster-{digest[:16]}"


def _behavior_tag_for(trades: list[WalletTrade]) -> str | None:
    """Return ``"farmer"`` / ``"trader"`` / ``"mixed"`` from buy/sell mix."""
    if not trades:
        return None
    buys = sum(1 for t in trades if t.side.upper() == "BUY")
    total = len(trades)
    if total == 0:
        return None
    ratio = buys / total
    if math.isnan(ratio):
        return None
    if _BEHAVIOR_FARMER_LO <= ratio <= _BEHAVIOR_FARMER_HI:
        return "farmer"
    if ratio > _BEHAVIOR_TRADER_HI:
        return "trader"
    return "mixed"


__all__ = ["ClusterDetector"]
