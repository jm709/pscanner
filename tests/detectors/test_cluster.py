"""Tests for ``ClusterDetector`` (issue #28).

Exercise discovery scoring + active monitoring with synthesised collaborators.
All collaborators are stubbed in-process — no network, no SQLite. We follow
the same shape as ``test_convergence.py`` and ``test_whales.py``: tiny stub
classes for each repo, a ``CapturingSink``, and per-test fixture builders.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from pscanner.alerts.models import Alert
from pscanner.config import ClusterConfig
from pscanner.detectors.cluster import (
    ClusterDetector,
    _build_discovered_alert,
    _cluster_id_for,
)
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.store.repo import (
    CachedMarket,
    WalletCluster,
    WalletFirstSeen,
    WalletTrade,
)
from pscanner.util.clock import FakeClock

_NOW = 1_700_000_000
_OBSCURE_LIQUIDITY = 1000.0
_OBSCURE_VOLUME = 5000.0


def _trade(
    *,
    wallet: str,
    condition_id: str,
    timestamp: int = _NOW,
    side: str = "BUY",
    size: float = 100.0,
    asset_id: str = "asset-1",
    transaction_hash: str | None = None,
) -> WalletTrade:
    """Build a synthetic ``WalletTrade``."""
    tx = transaction_hash or f"0xtx-{wallet}-{condition_id}-{timestamp}-{side}"
    return WalletTrade(
        transaction_hash=tx,
        asset_id=AssetId(asset_id),
        side=side,
        wallet=wallet,
        condition_id=ConditionId(condition_id),
        size=size,
        price=0.5,
        usd_value=size * 0.5,
        status="CONFIRMED",
        source="activity_api",
        timestamp=timestamp,
        recorded_at=timestamp,
    )


def _first_seen(wallet: str, *, first_at: int, total_trades: int = 10) -> WalletFirstSeen:
    """Build a synthetic ``WalletFirstSeen`` row."""
    return WalletFirstSeen(
        address=wallet,
        first_activity_at=first_at,
        total_trades=total_trades,
        cached_at=first_at,
    )


def _cached_market(
    *,
    condition_id: str,
    title: str = "obscure market",
    liquidity: float | None = _OBSCURE_LIQUIDITY,
    volume: float | None = _OBSCURE_VOLUME,
) -> CachedMarket:
    """Build a ``CachedMarket`` with default obscure thresholds."""
    return CachedMarket(
        market_id=MarketId(f"mkt-{condition_id}"),
        event_id=None,
        title=title,
        liquidity_usd=liquidity,
        volume_usd=volume,
        outcome_prices=[],
        active=True,
        cached_at=_NOW,
        condition_id=ConditionId(condition_id),
        event_slug=None,
    )


class StubFirstSeenRepo:
    """In-memory ``WalletFirstSeenRepo.list_recent`` stub."""

    def __init__(self, rows: Iterable[WalletFirstSeen]) -> None:
        self._rows = list(rows)
        self.calls: list[int] = []

    def list_recent(self, *, within: int) -> list[WalletFirstSeen]:
        self.calls.append(within)
        return list(self._rows)


class StubTradesRepo:
    """In-memory ``WalletTradesRepo`` stub."""

    def __init__(
        self,
        *,
        per_wallet: dict[str, list[WalletTrade]] | None = None,
        distinct_for_condition: dict[str, set[str]] | None = None,
    ) -> None:
        self._per_wallet = dict(per_wallet or {})
        self._distinct = dict(distinct_for_condition or {})

    def recent_for_wallet(self, wallet: str, *, limit: int = 100) -> list[WalletTrade]:
        del limit
        return list(self._per_wallet.get(wallet, []))

    def distinct_wallets_for_condition(
        self,
        condition_id: str,
        *,
        since: int,
    ) -> set[str]:
        del since
        return set(self._distinct.get(condition_id, set()))


class StubMarketCache:
    """In-memory ``MarketCacheRepo`` keyed by condition_id."""

    def __init__(self, by_condition: dict[str, CachedMarket] | None = None) -> None:
        self._by_condition = dict(by_condition or {})

    def get_by_condition_id(self, condition_id: str) -> CachedMarket | None:
        return self._by_condition.get(condition_id)


class StubClustersRepo:
    """In-memory ``WalletClustersRepo`` stub."""

    def __init__(self, prepop: dict[str, WalletCluster] | None = None) -> None:
        self._rows: dict[str, WalletCluster] = dict(prepop or {})
        self.last_active_calls: list[tuple[str, int]] = []

    def upsert(self, cluster: WalletCluster) -> None:
        self._rows[cluster.cluster_id] = cluster

    def get(self, cluster_id: str) -> WalletCluster | None:
        return self._rows.get(cluster_id)

    def update_last_active(self, cluster_id: str, ts: int) -> None:
        self.last_active_calls.append((cluster_id, ts))
        if cluster_id in self._rows:
            existing = self._rows[cluster_id]
            self._rows[cluster_id] = WalletCluster(
                cluster_id=existing.cluster_id,
                member_count=existing.member_count,
                first_member_created_at=existing.first_member_created_at,
                last_member_created_at=existing.last_member_created_at,
                shared_market_count=existing.shared_market_count,
                behavior_tag=existing.behavior_tag,
                detection_score=existing.detection_score,
                first_detected_at=existing.first_detected_at,
                last_active_at=ts,
            )


class StubMembersRepo:
    """In-memory ``WalletClusterMembersRepo`` stub."""

    def __init__(self, prepop: dict[str, list[str]] | None = None) -> None:
        self._members: dict[str, list[str]] = {k: list(v) for k, v in (prepop or {}).items()}

    def add_member(self, cluster_id: str, wallet: str) -> None:
        bucket = self._members.setdefault(cluster_id, [])
        if wallet not in bucket:
            bucket.append(wallet)

    def members_of(self, cluster_id: str) -> list[str]:
        return list(self._members.get(cluster_id, []))

    def cluster_for_wallet(self, wallet: str) -> str | None:
        for cluster_id, wallets in self._members.items():
            if wallet in wallets:
                return cluster_id
        return None


class CapturingSink:
    """Collects every alert ``emit`` is called with."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    async def emit(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


def _make_detector(
    *,
    config: ClusterConfig | None = None,
    first_seen: StubFirstSeenRepo | None = None,
    trades: StubTradesRepo | None = None,
    market_cache: StubMarketCache | None = None,
    clusters: StubClustersRepo | None = None,
    members: StubMembersRepo | None = None,
    sink: CapturingSink | None = None,
    clock: FakeClock | None = None,
) -> ClusterDetector:
    """Build a ``ClusterDetector`` wired to stubs."""
    detector = ClusterDetector(
        config=config or ClusterConfig(),
        wallet_first_seen=first_seen or StubFirstSeenRepo([]),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        trades_repo=trades or StubTradesRepo(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        market_cache=market_cache or StubMarketCache(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        clusters_repo=clusters or StubClustersRepo(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        members_repo=members or StubMembersRepo(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        clock=clock or FakeClock(start=float(_NOW)),
    )
    if sink is not None:
        detector._sink = sink  # type: ignore[assignment]  # ty:ignore[invalid-assignment]
    return detector


def _build_cluster_with_overlap(
    *,
    n_wallets: int,
    n_shared_markets: int,
    creation_window: int = 3600,
    same_size: bool = False,
    same_direction: bool = False,
    farmer_ratio: float = 0.6,
) -> tuple[
    list[WalletFirstSeen],
    dict[str, list[WalletTrade]],
    dict[str, CachedMarket],
]:
    """Build a synthetic cluster of ``n_wallets`` sharing ``n_shared_markets``.

    ``same_size`` makes every trade the same size (Signal C). ``same_direction``
    times trades into one direction window (Signal D). ``farmer_ratio`` sets
    the buy fraction across all trades for the behavior tag.
    """
    wallets = [f"0xw{i}" for i in range(n_wallets)]
    first_seen_rows = [
        _first_seen(w, first_at=_NOW + i * (creation_window // max(n_wallets - 1, 1)))
        for i, w in enumerate(wallets)
    ]
    conditions = [f"cond-{i}" for i in range(n_shared_markets)]
    cached_by_cond = {cid: _cached_market(condition_id=cid) for cid in conditions}
    trades_per_wallet: dict[str, list[WalletTrade]] = {w: [] for w in wallets}
    base_size = 100.0
    sequence = 0
    for cond_idx, cond in enumerate(conditions):
        for wallet_idx, wallet in enumerate(wallets):
            size = base_size if same_size else base_size * (1 + wallet_idx * 0.5)
            ts = _NOW + cond_idx * 10000
            if same_direction:
                ts = _NOW + cond_idx * 10000  # all wallets same bucket
            else:
                ts = _NOW + cond_idx * 10000 + wallet_idx * 100000
            target_buy_count = round(farmer_ratio * len(wallets))
            side = "BUY" if wallet_idx < target_buy_count else "SELL"
            trades_per_wallet[wallet].append(
                _trade(
                    wallet=wallet,
                    condition_id=cond,
                    timestamp=ts,
                    side=side,
                    size=size,
                    asset_id=f"asset-{cond_idx}",
                    transaction_hash=f"0xtx-{sequence}",
                )
            )
            sequence += 1
    return first_seen_rows, trades_per_wallet, cached_by_cond


# ---------------------------------------------------------------------------
# Discovery scan tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_emits_when_nine_wallets_share_obscure_markets() -> None:
    rows, per_wallet, cached = _build_cluster_with_overlap(
        n_wallets=9,
        n_shared_markets=3,
        creation_window=3600,
        same_size=True,
        same_direction=True,
        farmer_ratio=0.6,
    )
    sink = CapturingSink()
    detector = _make_detector(
        first_seen=StubFirstSeenRepo(rows),
        trades=StubTradesRepo(per_wallet=per_wallet),
        market_cache=StubMarketCache(cached),
        sink=sink,
    )

    new_count = await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert new_count == 1
    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.detector == "cluster"
    assert alert.alert_key.startswith("cluster.discovered:")
    assert alert.body["member_count"] == 9
    assert alert.body["detection_score"] >= 5
    assert alert.severity in {"high", "med"}


@pytest.mark.asyncio
async def test_discovery_skips_when_creation_window_violated() -> None:
    """Three wallets created over 7 days never qualify under a 24h window."""
    rows = [_first_seen(f"0xw{i}", first_at=_NOW + i * 3 * 86400) for i in range(3)]
    sink = CapturingSink()
    detector = _make_detector(first_seen=StubFirstSeenRepo(rows), sink=sink)

    new_count = await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert new_count == 0
    assert sink.alerts == []


@pytest.mark.asyncio
async def test_discovery_skips_when_shared_markets_are_popular() -> None:
    """Markets above the liquidity floor don't count as obscure overlap."""
    rows, per_wallet, _ = _build_cluster_with_overlap(
        n_wallets=4,
        n_shared_markets=3,
        same_size=True,
        same_direction=True,
    )
    popular_cached: dict[str, CachedMarket] = {
        str(cid): _cached_market(condition_id=str(cid), liquidity=999_999_999.0, volume=999_999.0)
        for cid in {t.condition_id for ts in per_wallet.values() for t in ts}
    }
    sink = CapturingSink()
    detector = _make_detector(
        first_seen=StubFirstSeenRepo(rows),
        trades=StubTradesRepo(per_wallet=per_wallet),
        market_cache=StubMarketCache(popular_cached),
        sink=sink,
    )

    new_count = await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert new_count == 0
    assert sink.alerts == []


@pytest.mark.asyncio
async def test_discovery_minimum_cluster_below_threshold_does_not_alert() -> None:
    """3 wallets on 3 obscure shared markets but no size-CV/direction match.

    Score: A=2 (creation), B=2 (overlap) = 4. Below default threshold of 5.
    """
    rows = [_first_seen(f"0xw{i}", first_at=_NOW + i * 60) for i in range(3)]
    conditions = [f"cond-{i}" for i in range(3)]
    cached = {cid: _cached_market(condition_id=cid) for cid in conditions}
    per_wallet: dict[str, list[WalletTrade]] = {w.address: [] for w in rows}
    # Wide size dispersion AND direction trades scattered across many windows.
    for cond_idx, cond in enumerate(conditions):
        for wallet_idx, wallet_row in enumerate(rows):
            per_wallet[wallet_row.address].append(
                _trade(
                    wallet=wallet_row.address,
                    condition_id=cond,
                    timestamp=_NOW + cond_idx * 10000 + wallet_idx * 100000,
                    side="BUY" if wallet_idx % 2 == 0 else "SELL",
                    size=10.0 * (1 + wallet_idx * 5),
                    transaction_hash=f"0xtx-{cond_idx}-{wallet_idx}",
                )
            )
    sink = CapturingSink()
    detector = _make_detector(
        first_seen=StubFirstSeenRepo(rows),
        trades=StubTradesRepo(per_wallet=per_wallet),
        market_cache=StubMarketCache(cached),
        sink=sink,
    )

    new_count = await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert new_count == 0
    assert sink.alerts == []


@pytest.mark.asyncio
async def test_discovery_behavior_tag_farmer_when_buy_ratio_in_band() -> None:
    """70% buys → behavior_tag farmer."""
    rows, per_wallet, cached = _build_cluster_with_overlap(
        n_wallets=10,
        n_shared_markets=3,
        same_size=True,
        same_direction=True,
        farmer_ratio=0.7,
    )
    sink = CapturingSink()
    detector = _make_detector(
        first_seen=StubFirstSeenRepo(rows),
        trades=StubTradesRepo(per_wallet=per_wallet),
        market_cache=StubMarketCache(cached),
        sink=sink,
    )

    await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert len(sink.alerts) == 1
    assert sink.alerts[0].body["behavior_tag"] == "farmer"


@pytest.mark.asyncio
async def test_discovery_behavior_tag_trader_when_buy_ratio_above_ninety() -> None:
    """95% buys → behavior_tag trader."""
    rows, per_wallet, cached = _build_cluster_with_overlap(
        n_wallets=20,
        n_shared_markets=3,
        same_size=True,
        same_direction=True,
        farmer_ratio=0.95,
    )
    sink = CapturingSink()
    detector = _make_detector(
        first_seen=StubFirstSeenRepo(rows),
        trades=StubTradesRepo(per_wallet=per_wallet),
        market_cache=StubMarketCache(cached),
        sink=sink,
    )

    await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert len(sink.alerts) == 1
    assert sink.alerts[0].body["behavior_tag"] == "trader"


@pytest.mark.asyncio
async def test_discovery_skips_already_known_cluster() -> None:
    """Re-detecting an existing cluster does not re-emit."""
    rows, per_wallet, cached = _build_cluster_with_overlap(
        n_wallets=5,
        n_shared_markets=3,
        same_size=True,
        same_direction=True,
    )
    # Pre-seed clusters repo so .get returns a row for the deterministic id.
    cid = _cluster_id_for(rows)
    clusters = StubClustersRepo(
        prepop={
            cid: WalletCluster(
                cluster_id=cid,
                member_count=5,
                first_member_created_at=_NOW,
                last_member_created_at=_NOW,
                shared_market_count=3,
                behavior_tag="farmer",
                detection_score=7,
                first_detected_at=_NOW,
                last_active_at=_NOW,
            )
        }
    )
    sink = CapturingSink()
    detector = _make_detector(
        first_seen=StubFirstSeenRepo(rows),
        trades=StubTradesRepo(per_wallet=per_wallet),
        market_cache=StubMarketCache(cached),
        clusters=clusters,
        sink=sink,
    )

    new_count = await detector.discovery_scan(sink)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    assert new_count == 0
    assert sink.alerts == []


# ---------------------------------------------------------------------------
# Active monitoring tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_emits_high_severity_when_majority_active() -> None:
    """3 of 4 cluster members trade the same condition within the window.

    active_count=3, member_count=4, 2*3 >= 4 → high severity.
    """
    cluster_id = "cluster-x"
    members = ["0xa", "0xb", "0xc", "0xd"]
    trade = _trade(wallet="0xa", condition_id="cond-1", timestamp=_NOW)
    members_repo = StubMembersRepo({cluster_id: members})
    trades_repo = StubTradesRepo(
        distinct_for_condition={"cond-1": {"0xa", "0xb", "0xc"}},
    )
    market_cache = StubMarketCache(
        {"cond-1": _cached_market(condition_id="cond-1", title="cool market")}
    )
    sink = CapturingSink()
    clusters = StubClustersRepo(
        prepop={
            cluster_id: WalletCluster(
                cluster_id=cluster_id,
                member_count=4,
                first_member_created_at=_NOW,
                last_member_created_at=_NOW,
                shared_market_count=3,
                behavior_tag=None,
                detection_score=6,
                first_detected_at=_NOW,
                last_active_at=_NOW,
            )
        }
    )
    detector = _make_detector(
        members=members_repo,
        trades=trades_repo,
        market_cache=market_cache,
        clusters=clusters,
        sink=sink,
    )

    await detector.evaluate_active(trade)

    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.severity == "high"
    assert alert.body["cluster_id"] == cluster_id
    assert alert.body["active_members"] == 3
    assert alert.body["total_member_count"] == 4
    assert alert.body["market_title"] == "cool market"
    assert clusters.last_active_calls == [(cluster_id, _NOW)]


@pytest.mark.asyncio
async def test_active_skips_when_trader_not_in_cluster() -> None:
    """Trade by a wallet outside any cluster is a silent no-op."""
    sink = CapturingSink()
    detector = _make_detector(sink=sink)

    await detector.evaluate_active(_trade(wallet="0xstranger", condition_id="cond-1"))

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_active_skips_when_only_one_member_active() -> None:
    """active_count=1 → below the default min_active_members=2 → no alert."""
    cluster_id = "cluster-x"
    members = ["0xa", "0xb", "0xc"]
    trade = _trade(wallet="0xa", condition_id="cond-1")
    members_repo = StubMembersRepo({cluster_id: members})
    trades_repo = StubTradesRepo(distinct_for_condition={"cond-1": {"0xa"}})
    sink = CapturingSink()
    detector = _make_detector(
        members=members_repo,
        trades=trades_repo,
        sink=sink,
    )

    await detector.evaluate_active(trade)

    assert sink.alerts == []


@pytest.mark.asyncio
async def test_active_alert_key_collapses_same_day_refire() -> None:
    """Two triggers on the same day for one (cluster, condition) share the alert key.

    Validates idempotency when ``AlertSink`` dedupes via ``insert_if_new`` —
    the detector emits both, but the key is identical so the sink rejects #2.
    The CapturingSink stub doesn't dedupe — it just records — so we instead
    assert that both alerts share the same alert_key (the AlertsRepo
    integration test in the store layer already verifies dedupe behavior).
    """
    cluster_id = "cluster-x"
    members = ["0xa", "0xb"]
    members_repo = StubMembersRepo({cluster_id: members})
    trades_repo = StubTradesRepo(distinct_for_condition={"cond-1": {"0xa", "0xb"}})
    sink = CapturingSink()
    detector = _make_detector(
        members=members_repo,
        trades=trades_repo,
        sink=sink,
    )

    trade_a = _trade(wallet="0xa", condition_id="cond-1", timestamp=_NOW)
    trade_b = _trade(
        wallet="0xa",
        condition_id="cond-1",
        timestamp=_NOW + 30,
        transaction_hash="0xtx-second",
    )

    await detector.evaluate_active(trade_a)
    await detector.evaluate_active(trade_b)

    assert len(sink.alerts) == 2
    assert sink.alerts[0].alert_key == sink.alerts[1].alert_key


@pytest.mark.asyncio
async def test_evaluate_active_no_sink_does_not_raise() -> None:
    """Without a wired sink, evaluate_active short-circuits silently."""
    detector = _make_detector()
    detector._sink = None

    await detector.evaluate_active(_trade(wallet="0xa", condition_id="cond-1"))


def test_handle_trade_sync_no_running_loop_is_noop() -> None:
    """With no running loop, ``handle_trade_sync`` returns silently."""
    detector = _make_detector()
    detector.handle_trade_sync(_trade(wallet="0xa", condition_id="cond-1"))
    assert detector._pending_tasks == set()


@pytest.mark.asyncio
async def test_run_loop_invokes_discovery_then_sleeps() -> None:
    """``run`` calls ``discovery_scan`` then sleeps; cancel terminates cleanly."""
    sink = CapturingSink()
    detector = _make_detector(sink=sink)

    task = asyncio.create_task(detector.run(sink))  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    # Yield enough times for run() to reach its first sleep call.
    for _ in range(8):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_alert_severity_threshold_high_at_seven() -> None:
    """detection_score >= 7 → high severity in discovered alert."""
    cluster = WalletCluster(
        cluster_id="cluster-x",
        member_count=5,
        first_member_created_at=_NOW,
        last_member_created_at=_NOW + 1000,
        shared_market_count=3,
        behavior_tag="farmer",
        detection_score=7,
        first_detected_at=_NOW,
        last_active_at=_NOW,
    )
    alert = _build_discovered_alert(cluster, [], [])
    assert alert.severity == "high"


def test_alert_severity_threshold_med_at_five() -> None:
    """detection_score in [5, 7) → med severity."""
    cluster = WalletCluster(
        cluster_id="cluster-x",
        member_count=5,
        first_member_created_at=_NOW,
        last_member_created_at=_NOW + 1000,
        shared_market_count=3,
        behavior_tag="farmer",
        detection_score=5,
        first_detected_at=_NOW,
        last_active_at=_NOW,
    )
    alert = _build_discovered_alert(cluster, [], [])
    assert alert.severity == "med"
