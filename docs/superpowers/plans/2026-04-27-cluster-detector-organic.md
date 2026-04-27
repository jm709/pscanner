# Cluster Detector Organic Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel co-occurrence-based candidate-group source to `ClusterDetector` so it can discover Cavill-shaped organic clusters (wallets created weeks apart that share obscure markets), refactoring Signal A from "always +2 because the partition is creation-clustered" to "+2 if the actual timestamps cluster, else 0."

**Architecture:** New `_iter_co_trade_groups` helper builds per-wallet sets of obscure-market condition_ids, computes pairwise shared-market counts, finds connected components on the resulting graph, and yields each component of size in `[min_cluster_size, max_co_trade_group_size]`. `discovery_scan` runs both paths sequentially. The existing `_cluster_id_for` SHA256-of-addresses dedupe ensures both paths discovering the same wallet set produce one alert. Existing scoring helpers (`_find_shared_obscure_markets`, `_collect_cluster_trades`, etc.) remain untouched.

**Tech Stack:** Python 3.13, sqlite3, structlog, pytest. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-04-27-cluster-detector-organic-design.md`

---

## File Structure

**Modify:**
- `src/pscanner/config.py` — add `max_co_trade_group_size` field on `ClusterConfig`
- `src/pscanner/detectors/cluster.py` — add `_iter_co_trade_groups`, add `_compute_creation_cohesion_score`, change `_score_candidate` to use the new helper, modify `discovery_scan` to iterate both paths
- `tests/detectors/test_cluster.py` — unit tests for the two new helpers, end-to-end tests for organic discovery, dedup, and overlap. The existing `StubFirstSeenRepo`, `StubTradesRepo`, `StubMarketCache`, `StubClustersRepo`, `StubMembersRepo` patterns are reused.
- `tests/test_config.py` — assert the new field's default value

No new files.

## Task ordering

Sequential — all tasks touch the same two source files (`cluster.py` and `test_cluster.py`). T1 is small and could go first or alongside the others; the rest must run in order.

| # | Task | Touches |
|---|------|---------|
| 1 | Add `max_co_trade_group_size` config field | `config.py`, `tests/test_config.py` |
| 2 | Refactor Signal A — extract `_compute_creation_cohesion_score`, change `_score_candidate` | `cluster.py`, `tests/detectors/test_cluster.py` |
| 3 | Add `_iter_co_trade_groups` helper (with unit tests) | `cluster.py`, `tests/detectors/test_cluster.py` |
| 4 | Wire `_iter_co_trade_groups` into `discovery_scan` + end-to-end tests | `cluster.py`, `tests/detectors/test_cluster.py` |

---

## Task 1: Add `max_co_trade_group_size` config field

Adds the safety cap field to `ClusterConfig`. Defaulted at 100. Used by Task 3 to truncate runaway components.

**Files:**
- Modify: `src/pscanner/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1.1: Locate the ClusterConfig defaults test**

```bash
grep -n "ClusterConfig\|cluster_defaults\|test_cluster_config" tests/test_config.py 2>/dev/null
```

If no existing test covers cluster defaults explicitly, look for a generic "Config defaults" test that asserts on cluster fields. The new test below is additive and self-contained either way.

- [ ] **Step 1.2: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_cluster_max_co_trade_group_size_default() -> None:
    from pscanner.config import ClusterConfig

    cfg = ClusterConfig()
    assert cfg.max_co_trade_group_size == 100
```

- [ ] **Step 1.3: Run, verify it fails**

```bash
uv run pytest tests/test_config.py -v -k max_co_trade_group_size
```

Expected: FAIL — `AttributeError: 'ClusterConfig' object has no attribute 'max_co_trade_group_size'`.

- [ ] **Step 1.4: Add the field**

In `src/pscanner/config.py`, find the `ClusterConfig` class. Append to its field list (the existing class extends `_Section` and uses pydantic field defaults — match that style):

```python
class ClusterConfig(_Section):
    """Thresholds + cadence for the coordinated-wallet cluster detector.

    ... (existing docstring) ...
    """

    # ... (existing fields) ...

    # Safety cap for the co-occurrence candidate-group path. Connected
    # components above this size are truncated deterministically to prevent
    # a misconfigured obscurity gate from producing runaway candidates.
    max_co_trade_group_size: int = 100
```

(Add the field to the bottom of `ClusterConfig`, preserving alphabetical / semantic ordering with neighbors. Don't add it to the docstring — the inline comment above the field is enough.)

- [ ] **Step 1.5: Run, verify it passes**

```bash
uv run pytest tests/test_config.py -v -k max_co_trade_group_size
```

Expected: PASS.

- [ ] **Step 1.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/config.py tests/test_config.py
uv run ruff format --check src/pscanner/config.py tests/test_config.py
uv run ty check src/pscanner/config.py
```

Expected: all clean.

- [ ] **Step 1.7: Commit**

```bash
git add src/pscanner/config.py tests/test_config.py
git commit -m "feat(config): add ClusterConfig.max_co_trade_group_size

Safety cap (default 100) for the upcoming co-occurrence candidate-group
path in ClusterDetector. Bounds connected-component size so a
misconfigured obscurity gate cannot produce a runaway 1000-wallet
candidate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Refactor Signal A via `_compute_creation_cohesion_score`

Extract Signal A scoring into a helper that reads actual timestamps. Current code awards a fixed +2 in `_score_candidate` whenever `len(group) >= min_cluster_size`. The fixed +2 is correct for the creation-window path because that path's groups are creation-clustered by construction. After Task 3 adds a co-occurrence path that produces groups of arbitrarily-spread timestamps, Signal A must compute cohesion from the actual data.

**Behavioral compatibility:** the existing creation-window path's groups all have `max-min <= creation_window_seconds` (greedy partition guarantees it), so the new helper returns 2 for any group from Path 1. Behavior is preserved.

**Files:**
- Modify: `src/pscanner/detectors/cluster.py`
- Modify: `tests/detectors/test_cluster.py`

- [ ] **Step 2.1: Write failing unit tests for the helper**

Append to `tests/detectors/test_cluster.py`:

```python
def test_compute_creation_cohesion_all_within_window_returns_2() -> None:
    detector = _build_detector()
    group = [
        _first_seen("0xa", first_at=_NOW),
        _first_seen("0xb", first_at=_NOW + 1000),
        _first_seen("0xc", first_at=_NOW + 5000),
    ]
    # Default creation_window_seconds=86400, so 5000s spread is within.
    assert detector._compute_creation_cohesion_score(group) == 2


def test_compute_creation_cohesion_spread_beyond_window_returns_0() -> None:
    detector = _build_detector()
    group = [
        _first_seen("0xa", first_at=_NOW),
        _first_seen("0xb", first_at=_NOW + 86_401),  # 1s past 24h
        _first_seen("0xc", first_at=_NOW + 100_000),
    ]
    assert detector._compute_creation_cohesion_score(group) == 0


def test_compute_creation_cohesion_all_null_returns_0() -> None:
    detector = _build_detector()
    group = [
        WalletFirstSeen(address="0xa", first_activity_at=None, total_trades=0, cached_at=_NOW),
        WalletFirstSeen(address="0xb", first_activity_at=None, total_trades=0, cached_at=_NOW),
        WalletFirstSeen(address="0xc", first_activity_at=None, total_trades=0, cached_at=_NOW),
    ]
    assert detector._compute_creation_cohesion_score(group) == 0


def test_compute_creation_cohesion_mixed_null_passing() -> None:
    """Some NULL, remaining ≥ min_cluster_size, all within window → 2."""
    detector = _build_detector()
    group = [
        _first_seen("0xa", first_at=_NOW),
        _first_seen("0xb", first_at=_NOW + 100),
        _first_seen("0xc", first_at=_NOW + 200),
        WalletFirstSeen(address="0xd", first_activity_at=None, total_trades=0, cached_at=_NOW),
    ]
    assert detector._compute_creation_cohesion_score(group) == 2


def test_compute_creation_cohesion_mixed_null_below_min_returns_0() -> None:
    """Some NULL, remaining < min_cluster_size → 0 (cannot conclude clustering)."""
    detector = _build_detector()
    # min_cluster_size default is 3; one non-NULL wallet, three NULL.
    group = [
        _first_seen("0xa", first_at=_NOW),
        WalletFirstSeen(address="0xb", first_activity_at=None, total_trades=0, cached_at=_NOW),
        WalletFirstSeen(address="0xc", first_activity_at=None, total_trades=0, cached_at=_NOW),
        WalletFirstSeen(address="0xd", first_activity_at=None, total_trades=0, cached_at=_NOW),
    ]
    assert detector._compute_creation_cohesion_score(group) == 0
```

You'll need a `_build_detector()` helper. If one doesn't already exist, add it near the top of the test file (after the existing `_NOW` / helper constants):

```python
def _build_detector(
    *,
    config: ClusterConfig | None = None,
    first_seen: StubFirstSeenRepo | None = None,
    trades: StubTradesRepo | None = None,
    market_cache: StubMarketCache | None = None,
    clusters: StubClustersRepo | None = None,
    members: StubMembersRepo | None = None,
) -> ClusterDetector:
    """Construct a ClusterDetector with sensible test defaults."""
    return ClusterDetector(
        config=config if config is not None else ClusterConfig(),
        wallet_first_seen=first_seen if first_seen is not None else StubFirstSeenRepo([]),
        trades_repo=trades if trades is not None else StubTradesRepo(),
        market_cache=market_cache if market_cache is not None else StubMarketCache(),
        clusters_repo=clusters if clusters is not None else StubClustersRepo(),
        members_repo=members if members is not None else StubMembersRepo(),
        clock=FakeClock(start_time=float(_NOW)),
    )
```

(If the test file already constructs `ClusterDetector` inline, lifting that into `_build_detector` is a small refactor. Apply only if needed; otherwise inline construction in the new tests is fine.)

- [ ] **Step 2.2: Run, verify they fail**

```bash
uv run pytest tests/detectors/test_cluster.py -v -k creation_cohesion
```

Expected: FAIL — `AttributeError: 'ClusterDetector' object has no attribute '_compute_creation_cohesion_score'`.

- [ ] **Step 2.3: Add the helper to `ClusterDetector`**

In `src/pscanner/detectors/cluster.py`, find the `ClusterDetector` class. Add the method (placement: anywhere reasonable; near `_score_candidate` makes the relationship visible):

```python
    def _compute_creation_cohesion_score(self, group: list[WalletFirstSeen]) -> int:
        """Return 2 if the group's creation timestamps cluster within
        ``creation_window_seconds``, else 0.

        Wallets with NULL ``first_activity_at`` are excluded from the
        comparison. If the remaining set is below ``min_cluster_size``,
        returns 0 — we can't conclude creation clustering with too few
        non-NULL timestamps.
        """
        timestamps = [
            w.first_activity_at for w in group if w.first_activity_at is not None
        ]
        if len(timestamps) < self._config.min_cluster_size:
            return 0
        if max(timestamps) - min(timestamps) <= self._config.creation_window_seconds:
            return 2
        return 0
```

- [ ] **Step 2.4: Refactor `_score_candidate` to use the helper**

Find `_score_candidate` in `cluster.py`. The current Signal A code looks like:

```python
        score = 0
        # Signal A — wallet creation clustering (group already passed min size).
        if len(group) >= self._config.min_cluster_size:
            score += 2
```

Replace with:

```python
        score = 0
        # Signal A — wallet creation clustering. For Path 1 (creation-window
        # partition) this always returns 2 by construction; for Path 2
        # (co-occurrence) it is a real bonus when the discovered cluster
        # also has tight creation timestamps.
        score += self._compute_creation_cohesion_score(group)
```

- [ ] **Step 2.5: Run the helper tests, verify they pass**

```bash
uv run pytest tests/detectors/test_cluster.py -v -k creation_cohesion
```

Expected: 5 passed.

- [ ] **Step 2.6: Run the full cluster-detector test suite as a regression check**

```bash
uv run pytest tests/detectors/test_cluster.py -q
```

Expected: all existing tests still pass. Specifically the score=7 test for Cavill-shape (verify Signal A still contributes +2 when all timestamps are within `creation_window_seconds`).

- [ ] **Step 2.7: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
uv run ruff format --check src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
uv run ty check src/pscanner/detectors/cluster.py
```

- [ ] **Step 2.8: Commit**

```bash
git add src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
git commit -m "refactor(cluster): Signal A reads actual timestamps via _compute_creation_cohesion_score

Path 1 (creation-window partition) keeps awarding +2 because its groups
are clustered by construction. The behavioral change is invisible today —
this prepares for Task 3's co-occurrence path, where Path 2 groups can
have arbitrarily-spread timestamps and Signal A becomes a real bonus.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add `_iter_co_trade_groups` helper

Adds the new candidate-group source. Pure helper — does not yet wire into `discovery_scan` (that's Task 4). Tested in isolation against stubs.

**Files:**
- Modify: `src/pscanner/detectors/cluster.py`
- Modify: `tests/detectors/test_cluster.py`

- [ ] **Step 3.1: Write the unit tests**

Append to `tests/detectors/test_cluster.py`. The existing test file already imports `WalletFirstSeen`, `WalletTrade`, `CachedMarket`, etc. and has `StubTradesRepo` and `StubMarketCache` available.

```python
# Helpers used across the new co-trade tests. Place near the other helpers.

def _obscure_market(condition_id: str) -> CachedMarket:
    """Build a CachedMarket that passes _is_obscure_market (low liquidity AND volume)."""
    return _cached_market(
        condition_id=condition_id,
        liquidity=_OBSCURE_LIQUIDITY,
        volume=_OBSCURE_VOLUME,
    )


def _high_volume_market(condition_id: str) -> CachedMarket:
    """Build a CachedMarket above the obscurity caps (liquidity > $50k OR volume > $1M)."""
    return _cached_market(
        condition_id=condition_id,
        liquidity=100_000.0,   # > max_shared_market_liquidity_usd default 50k
        volume=2_000_000.0,    # > max_shared_market_volume_usd default 1M
    )


def _wallet_trades_for_markets(wallet: str, condition_ids: list[str]) -> list[WalletTrade]:
    """Build one BUY trade per condition_id for a wallet."""
    return [
        _trade(wallet=wallet, condition_id=cid, asset_id=f"{cid}-asset")
        for cid in condition_ids
    ]


def test_co_trade_groups_happy_path() -> None:
    """4 wallets each trading the same 3 obscure markets → 1 component of 4."""
    markets = ["0xm1", "0xm2", "0xm3"]
    wallets = ["0xa", "0xb", "0xc", "0xd"]
    trades = StubTradesRepo(per_wallet={
        w: _wallet_trades_for_markets(w, markets) for w in wallets
    })
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [_first_seen(w, first_at=_NOW + i * 86_400 * 30) for i, w in enumerate(wallets)]

    groups = list(detector._iter_co_trade_groups(recent))

    assert len(groups) == 1
    assert sorted(w.address for w in groups[0]) == sorted(wallets)


def test_co_trade_groups_below_threshold() -> None:
    """4 wallets sharing 2 obscure markets — below default min_shared_markets=3."""
    markets = ["0xm1", "0xm2"]
    wallets = ["0xa", "0xb", "0xc", "0xd"]
    trades = StubTradesRepo(per_wallet={
        w: _wallet_trades_for_markets(w, markets) for w in wallets
    })
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [_first_seen(w, first_at=_NOW) for w in wallets]

    assert list(detector._iter_co_trade_groups(recent)) == []


def test_co_trade_groups_transitive_closure() -> None:
    """A-B share 3 markets, B-C share 3 markets, A-C share 0 → ABC one component."""
    # Markets only A and B share
    ab_markets = ["0xab1", "0xab2", "0xab3"]
    # Markets only B and C share
    bc_markets = ["0xbc1", "0xbc2", "0xbc3"]
    trades = StubTradesRepo(per_wallet={
        "0xa": _wallet_trades_for_markets("0xa", ab_markets),
        "0xb": _wallet_trades_for_markets("0xb", ab_markets + bc_markets),
        "0xc": _wallet_trades_for_markets("0xc", bc_markets),
    })
    market_cache = StubMarketCache({
        m: _obscure_market(m) for m in ab_markets + bc_markets
    })
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [_first_seen(w, first_at=_NOW) for w in ["0xa", "0xb", "0xc"]]

    groups = list(detector._iter_co_trade_groups(recent))

    assert len(groups) == 1
    assert sorted(w.address for w in groups[0]) == ["0xa", "0xb", "0xc"]


def test_co_trade_groups_two_disjoint_clusters() -> None:
    """6 wallets: ABC trade markets X, DEF trade markets Y → 2 components."""
    x_markets = ["0xx1", "0xx2", "0xx3"]
    y_markets = ["0xy1", "0xy2", "0xy3"]
    trades = StubTradesRepo(per_wallet={
        "0xa": _wallet_trades_for_markets("0xa", x_markets),
        "0xb": _wallet_trades_for_markets("0xb", x_markets),
        "0xc": _wallet_trades_for_markets("0xc", x_markets),
        "0xd": _wallet_trades_for_markets("0xd", y_markets),
        "0xe": _wallet_trades_for_markets("0xe", y_markets),
        "0xf": _wallet_trades_for_markets("0xf", y_markets),
    })
    market_cache = StubMarketCache({
        m: _obscure_market(m) for m in x_markets + y_markets
    })
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [_first_seen(w, first_at=_NOW) for w in ["0xa", "0xb", "0xc", "0xd", "0xe", "0xf"]]

    groups = list(detector._iter_co_trade_groups(recent))

    component_addrs = sorted(sorted(w.address for w in g) for g in groups)
    assert component_addrs == [["0xa", "0xb", "0xc"], ["0xd", "0xe", "0xf"]]


def test_co_trade_groups_skip_components_below_min_size() -> None:
    """A-B share 5 markets but no third connected wallet → component size 2 < 3 → skipped."""
    markets = ["0xm1", "0xm2", "0xm3", "0xm4", "0xm5"]
    trades = StubTradesRepo(per_wallet={
        "0xa": _wallet_trades_for_markets("0xa", markets),
        "0xb": _wallet_trades_for_markets("0xb", markets),
    })
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [_first_seen(w, first_at=_NOW) for w in ["0xa", "0xb"]]

    assert list(detector._iter_co_trade_groups(recent)) == []


def test_co_trade_groups_isolated_wallets_not_yielded() -> None:
    """ABC connected via shared markets; D and E trade nothing → only ABC yielded."""
    markets = ["0xm1", "0xm2", "0xm3"]
    trades = StubTradesRepo(per_wallet={
        "0xa": _wallet_trades_for_markets("0xa", markets),
        "0xb": _wallet_trades_for_markets("0xb", markets),
        "0xc": _wallet_trades_for_markets("0xc", markets),
        # D and E have no trades.
    })
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [
        _first_seen(w, first_at=_NOW) for w in ["0xa", "0xb", "0xc", "0xd", "0xe"]
    ]

    groups = list(detector._iter_co_trade_groups(recent))

    assert len(groups) == 1
    assert sorted(w.address for w in groups[0]) == ["0xa", "0xb", "0xc"]


def test_co_trade_groups_excludes_non_obscure_markets() -> None:
    """4 wallets sharing 5 high-volume (non-obscure) markets → 0 components."""
    markets = ["0xm1", "0xm2", "0xm3", "0xm4", "0xm5"]
    trades = StubTradesRepo(per_wallet={
        w: _wallet_trades_for_markets(w, markets)
        for w in ["0xa", "0xb", "0xc", "0xd"]
    })
    # All markets are HIGH-VOLUME (above obscurity caps).
    market_cache = StubMarketCache({m: _high_volume_market(m) for m in markets})
    detector = _build_detector(trades=trades, market_cache=market_cache)
    recent = [_first_seen(w, first_at=_NOW) for w in ["0xa", "0xb", "0xc", "0xd"]]

    assert list(detector._iter_co_trade_groups(recent)) == []


def test_co_trade_groups_size_cap_truncates() -> None:
    """200 wallets all sharing 3 obscure markets → 1 component truncated to 100."""
    markets = ["0xm1", "0xm2", "0xm3"]
    n = 200
    addrs = [f"0x{i:04x}" for i in range(n)]
    trades = StubTradesRepo(per_wallet={
        a: _wallet_trades_for_markets(a, markets) for a in addrs
    })
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    cfg = ClusterConfig(max_co_trade_group_size=100)
    detector = _build_detector(config=cfg, trades=trades, market_cache=market_cache)
    recent = [_first_seen(a, first_at=_NOW) for a in addrs]

    groups = list(detector._iter_co_trade_groups(recent))

    assert len(groups) == 1
    assert len(groups[0]) == 100
    # Truncation is deterministic by sorted address.
    expected_kept = sorted(addrs)[:100]
    assert sorted(w.address for w in groups[0]) == expected_kept
```

- [ ] **Step 3.2: Run, verify they fail**

```bash
uv run pytest tests/detectors/test_cluster.py -v -k co_trade_groups
```

Expected: FAIL — `AttributeError: 'ClusterDetector' object has no attribute '_iter_co_trade_groups'`.

- [ ] **Step 3.3: Implement the helper**

In `src/pscanner/detectors/cluster.py`, place the new method on `ClusterDetector` near `_iter_candidate_groups` for visual proximity. Adjust imports if needed (no new imports expected — `set`, `dict`, etc. are stdlib).

```python
    def _iter_co_trade_groups(
        self,
        recent: list[WalletFirstSeen],
    ) -> Iterable[list[WalletFirstSeen]]:
        """Yield candidate groups derived from shared-obscure-market overlap.

        For each pair of wallets in ``recent``, count their shared obscure
        markets. Wallets connected by edges of ≥ ``min_shared_markets``
        shared markets form connected components; each component of size
        in ``[min_cluster_size, max_co_trade_group_size]`` is yielded.

        This path is independent of creation timestamps — it discovers
        clusters that grew organically over time. Existing scoring
        (B/C/D + the refactored Signal A) is applied per component.
        """
        if len(recent) < self._config.min_cluster_size:
            return

        # 1. Per-wallet obscure-market sets (function-local; not shared
        #    across scans).
        obscure_markets = self._build_obscure_markets_index(recent)

        # 2. Build adjacency list: edge if pairwise shared count ≥ threshold.
        adjacency: dict[str, set[str]] = {w.address: set() for w in recent}
        addrs = sorted(adjacency)
        threshold = self._config.min_shared_markets
        for i, a in enumerate(addrs):
            ma = obscure_markets[a]
            if not ma:
                continue
            for b in addrs[i + 1:]:
                mb = obscure_markets[b]
                if len(ma & mb) >= threshold:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        # 3. Connected components via BFS over the address graph.
        by_address = {w.address: w for w in recent}
        visited: set[str] = set()
        for start in addrs:
            if start in visited:
                continue
            component_addrs = self._bfs_component(start, adjacency)
            visited |= component_addrs
            if len(component_addrs) < self._config.min_cluster_size:
                continue
            if len(component_addrs) > self._config.max_co_trade_group_size:
                _LOG.warning(
                    "cluster.cotrade_group_truncated",
                    original_size=len(component_addrs),
                    keep=self._config.max_co_trade_group_size,
                )
                kept = sorted(component_addrs)[: self._config.max_co_trade_group_size]
                yield [by_address[a] for a in kept]
            else:
                yield [by_address[a] for a in sorted(component_addrs)]

    def _build_obscure_markets_index(
        self,
        recent: list[WalletFirstSeen],
    ) -> dict[str, set[ConditionId]]:
        """Per-wallet set of obscure-market condition_ids.

        Per-wallet ``recent_for_wallet`` failures and missing market_cache
        rows are isolated — that wallet's set is treated as empty (no
        edges incident on it). Other wallets unaffected.
        """
        index: dict[str, set[ConditionId]] = {}
        for wallet in recent:
            try:
                trades = self._trades.recent_for_wallet(
                    wallet.address, limit=_RECENT_TRADE_LIMIT,
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
```

- [ ] **Step 3.4: Run the new tests, verify they pass**

```bash
uv run pytest tests/detectors/test_cluster.py -v -k co_trade_groups
```

Expected: 8 passed.

- [ ] **Step 3.5: Run the full detector test suite for regressions**

```bash
uv run pytest tests/detectors/test_cluster.py -q
```

Expected: all green. The new helper isn't wired into `discovery_scan` yet, so existing tests are unaffected.

- [ ] **Step 3.6: Lint / format / type-check**

```bash
uv run ruff check src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
uv run ruff format --check src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
uv run ty check src/pscanner/detectors/cluster.py
```

Expected: all clean. If `ruff` flags `_iter_co_trade_groups`'s complexity (>8), extract a helper for the pairwise-edge construction step.

- [ ] **Step 3.7: Commit**

```bash
git add src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
git commit -m "feat(cluster): add _iter_co_trade_groups for organic discovery

Pure helper that yields candidate groups via shared-obscure-market
overlap with connected-components consolidation. Not yet wired into
discovery_scan (next task).

Algorithm: per-wallet obscure-market sets → pairwise shared-market
counts → adjacency where shared ≥ min_shared_markets → BFS components
of size in [min_cluster_size, max_co_trade_group_size]. Per-wallet
trade-fetch failures and missing cache rows are isolated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire into `discovery_scan` + end-to-end tests

The final task. `discovery_scan` runs both candidate-group sources sequentially. The existing `_consider_group` + `_cluster_id_for` dedupe keep both paths honest.

**Files:**
- Modify: `src/pscanner/detectors/cluster.py`
- Modify: `tests/detectors/test_cluster.py`

- [ ] **Step 4.1: Locate the existing `discovery_scan`**

```bash
grep -n "def discovery_scan\|_iter_candidate_groups\|_consider_group" src/pscanner/detectors/cluster.py
```

The existing implementation has the pattern:
```python
async def discovery_scan(self, sink: AlertSink) -> int:
    # ... pre-checks ...
    seen_cluster_ids: set[str] = set()
    new_count = 0
    for group in self._iter_candidate_groups(recent):
        if await self._consider_group(group, seen_cluster_ids, sink):
            new_count += 1
    return new_count
```

(Exact line numbers may differ — locate by inspection.)

- [ ] **Step 4.2: Write the failing end-to-end tests**

Append to `tests/detectors/test_cluster.py`. The existing test suite has helpers `_trade`, `_first_seen`, `_cached_market`, `_obscure_market`, plus all five Stub repos. Reuse them.

```python
# Helpers for end-to-end scoring scenarios.

def _trades_with_synchronized_buy(
    wallets: list[str],
    condition_ids: list[str],
    *,
    bucket_ts: int,
    asset_id_per_market: dict[str, str] | None = None,
) -> dict[str, list[WalletTrade]]:
    """For each (wallet, condition_id) pair, emit one BUY trade with the
    same asset_id and timestamp inside ``direction_window_seconds`` so
    Signal D can fire."""
    asset_map = asset_id_per_market or {c: f"{c}-asset" for c in condition_ids}
    out: dict[str, list[WalletTrade]] = {}
    for w in wallets:
        per_wallet: list[WalletTrade] = []
        for cid in condition_ids:
            per_wallet.append(_trade(
                wallet=w,
                condition_id=cid,
                asset_id=asset_map[cid],
                timestamp=bucket_ts,
                side="BUY",
                size=100.0,
            ))
        out[w] = per_wallet
    return out


@pytest.mark.asyncio
async def test_discovery_scan_organic_cluster_emits_via_co_trade_path() -> None:
    """5 wallets created weeks apart, share 4 obscure markets, low size CV,
    synchronized BUY direction → score 5 from B/C/D, fires cluster.discovered."""
    markets = ["0xobs1", "0xobs2", "0xobs3", "0xobs4"]
    wallets = ["0xa", "0xb", "0xc", "0xd", "0xe"]
    bucket_ts = _NOW + 10_000
    per_wallet = _trades_with_synchronized_buy(wallets, markets, bucket_ts=bucket_ts)
    trades = StubTradesRepo(
        per_wallet=per_wallet,
        distinct_for_condition={cid: set(wallets) for cid in markets},
    )
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    # Spread wallets' first_activity_at across 60 days — beyond 24h window.
    recent = [
        _first_seen(w, first_at=_NOW + i * 86_400 * 12)
        for i, w in enumerate(wallets)
    ]
    sink = CapturingSink()
    detector = _build_detector(
        first_seen=StubFirstSeenRepo(recent),
        trades=trades,
        market_cache=market_cache,
    )

    new_count = await detector.discovery_scan(sink)

    assert new_count == 1
    assert len(sink.alerts) == 1
    body = sink.alerts[0].body
    assert body["member_count"] == 5
    # Signal A (cohesion) returns 0 for this group; B+C+D = 5 hits threshold.
    assert body["detection_score"] == 5


@pytest.mark.asyncio
async def test_discovery_scan_cavill_shape_still_emits_via_creation_path() -> None:
    """Regression: 9 wallets within 24h, share 3 obscure markets, low CV,
    synchronized BUY → existing path still produces score=7 alert."""
    markets = ["0xobs1", "0xobs2", "0xobs3"]
    wallets = [f"0x{i:02x}" for i in range(9)]
    bucket_ts = _NOW + 10_000
    per_wallet = _trades_with_synchronized_buy(wallets, markets, bucket_ts=bucket_ts)
    trades = StubTradesRepo(
        per_wallet=per_wallet,
        distinct_for_condition={cid: set(wallets) for cid in markets},
    )
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    # All within 24h.
    recent = [_first_seen(w, first_at=_NOW + i * 60) for i, w in enumerate(wallets)]
    sink = CapturingSink()
    detector = _build_detector(
        first_seen=StubFirstSeenRepo(recent),
        trades=trades,
        market_cache=market_cache,
    )

    new_count = await detector.discovery_scan(sink)

    assert new_count == 1
    assert sink.alerts[0].body["detection_score"] == 7
    assert sink.alerts[0].body["member_count"] == 9


@pytest.mark.asyncio
async def test_discovery_scan_dedupes_when_both_paths_find_same_cluster() -> None:
    """Cluster that satisfies BOTH path 1 and path 2 emits exactly once."""
    # 9 wallets within 24h AND share 4 obscure markets.
    markets = ["0xobs1", "0xobs2", "0xobs3", "0xobs4"]
    wallets = [f"0x{i:02x}" for i in range(9)]
    bucket_ts = _NOW + 10_000
    per_wallet = _trades_with_synchronized_buy(wallets, markets, bucket_ts=bucket_ts)
    trades = StubTradesRepo(
        per_wallet=per_wallet,
        distinct_for_condition={cid: set(wallets) for cid in markets},
    )
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    recent = [_first_seen(w, first_at=_NOW + i * 60) for i, w in enumerate(wallets)]
    clusters_repo = StubClustersRepo()
    sink = CapturingSink()
    detector = _build_detector(
        first_seen=StubFirstSeenRepo(recent),
        trades=trades,
        market_cache=market_cache,
        clusters=clusters_repo,
    )

    new_count = await detector.discovery_scan(sink)

    assert new_count == 1
    assert len(sink.alerts) == 1
    # Only one wallet_clusters row written.
    assert len(clusters_repo._rows) == 1


@pytest.mark.asyncio
async def test_discovery_scan_organic_below_threshold_does_not_emit() -> None:
    """5 wallets share obscure markets but no synchronized direction → score=3 < 5."""
    markets = ["0xobs1", "0xobs2", "0xobs3"]
    wallets = ["0xa", "0xb", "0xc", "0xd", "0xe"]
    # Each wallet trades the same markets but at DIFFERENT timestamps,
    # so Signal D cannot find a 600s bucket with ≥3 wallets.
    per_wallet: dict[str, list[WalletTrade]] = {}
    for i, w in enumerate(wallets):
        per_wallet[w] = [
            _trade(
                wallet=w,
                condition_id=cid,
                asset_id=f"{cid}-asset",
                timestamp=_NOW + i * 100_000,  # spread across hours
                size=100.0,
                side="BUY",
            )
            for cid in markets
        ]
    trades = StubTradesRepo(per_wallet=per_wallet)
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    recent = [
        _first_seen(w, first_at=_NOW + i * 86_400 * 12)
        for i, w in enumerate(wallets)
    ]
    sink = CapturingSink()
    detector = _build_detector(
        first_seen=StubFirstSeenRepo(recent),
        trades=trades,
        market_cache=market_cache,
    )

    new_count = await detector.discovery_scan(sink)

    assert new_count == 0
    assert sink.alerts == []


@pytest.mark.asyncio
async def test_discovery_scan_high_volume_markets_dont_trigger_co_trade() -> None:
    """8 wallets co-trade 5 high-volume markets → obscurity gate excludes,
    no candidate group formed by the new path.

    Confirms this work doesn't accidentally flag the volume-farming cluster.
    """
    markets = ["0xhi1", "0xhi2", "0xhi3", "0xhi4", "0xhi5"]
    wallets = [f"0x{i:02x}" for i in range(8)]
    bucket_ts = _NOW + 10_000
    per_wallet = _trades_with_synchronized_buy(wallets, markets, bucket_ts=bucket_ts)
    trades = StubTradesRepo(per_wallet=per_wallet)
    # All markets HIGH-VOLUME.
    market_cache = StubMarketCache({m: _high_volume_market(m) for m in markets})
    recent = [
        _first_seen(w, first_at=_NOW + i * 86_400 * 5)
        for i, w in enumerate(wallets)
    ]
    sink = CapturingSink()
    detector = _build_detector(
        first_seen=StubFirstSeenRepo(recent),
        trades=trades,
        market_cache=market_cache,
    )

    new_count = await detector.discovery_scan(sink)

    assert new_count == 0
    assert sink.alerts == []


@pytest.mark.asyncio
async def test_discovery_scan_organic_with_null_first_activity() -> None:
    """5 wallets connected via shared markets; one has NULL first_activity_at.
    Signal A computed on the 4 non-NULL wallets → +2 if clustered, else 0.
    Signals B/C/D fire normally → group still scores ≥ threshold."""
    markets = ["0xobs1", "0xobs2", "0xobs3", "0xobs4"]
    wallets = ["0xa", "0xb", "0xc", "0xd", "0xe"]
    bucket_ts = _NOW + 10_000
    per_wallet = _trades_with_synchronized_buy(wallets, markets, bucket_ts=bucket_ts)
    trades = StubTradesRepo(
        per_wallet=per_wallet,
        distinct_for_condition={cid: set(wallets) for cid in markets},
    )
    market_cache = StubMarketCache({m: _obscure_market(m) for m in markets})
    # 4 wallets within 24h (cohesion fires); 1 with NULL first_activity_at.
    recent = [
        _first_seen("0xa", first_at=_NOW),
        _first_seen("0xb", first_at=_NOW + 100),
        _first_seen("0xc", first_at=_NOW + 200),
        _first_seen("0xd", first_at=_NOW + 300),
        WalletFirstSeen(
            address="0xe", first_activity_at=None, total_trades=10, cached_at=_NOW,
        ),
    ]
    sink = CapturingSink()
    detector = _build_detector(
        first_seen=StubFirstSeenRepo(recent),
        trades=trades,
        market_cache=market_cache,
    )

    new_count = await detector.discovery_scan(sink)

    assert new_count == 1
    body = sink.alerts[0].body
    assert body["member_count"] == 5
    # 4 non-NULL within 24h → cohesion +2; B+C+D = 5 → total 7.
    assert body["detection_score"] == 7
```

Note: `CapturingSink` is the existing test-utility class in `test_cluster.py`. If your file uses a different captured-sink shape, adapt accordingly — the key is to assert on `sink.alerts` after `await discovery_scan(sink)`.

- [ ] **Step 4.3: Run, verify they fail**

```bash
uv run pytest tests/detectors/test_cluster.py -v -k "organic_cluster or organic_below_threshold or organic_with_null or high_volume_markets or dedupes_when_both_paths or cavill_shape_still"
```

Expected: most fail. The Cavill regression test should still pass even before wiring (it goes through Path 1). The organic-discovery tests fail because Path 2 isn't yet integrated.

- [ ] **Step 4.4: Wire `_iter_co_trade_groups` into `discovery_scan`**

In `src/pscanner/detectors/cluster.py`, find `discovery_scan`. Add the second loop after the existing one. Final shape:

```python
    async def discovery_scan(self, sink: AlertSink) -> int:
        """Run both candidate-group paths and score each unique cluster once."""
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
        # Path 2: new co-occurrence partition (organic clusters).
        for group in self._iter_co_trade_groups(recent):
            if await self._consider_group(group, seen_cluster_ids, sink):
                new_count += 1
        return new_count
```

(Preserve any existing pre-check / docstring / logging the current implementation has. The change is purely additive — one new for-loop after the existing one.)

- [ ] **Step 4.5: Run the new tests, verify they pass**

```bash
uv run pytest tests/detectors/test_cluster.py -v -k "organic_cluster or organic_below_threshold or organic_with_null or high_volume_markets or dedupes_when_both_paths or cavill_shape_still"
```

Expected: 6 passed.

- [ ] **Step 4.6: Run the full detector test suite for regressions**

```bash
uv run pytest tests/detectors/test_cluster.py -q
```

Expected: all green. The original Cavill regression test (whichever name it had pre-this-work) continues to pass because Path 1 still produces its group and `_compute_creation_cohesion_score` still returns 2 by construction.

- [ ] **Step 4.7: Run the entire test suite as a final regression check**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 4.8: Lint / format / type-check the entire repo**

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Expected: all clean.

- [ ] **Step 4.9: Commit**

```bash
git add src/pscanner/detectors/cluster.py tests/detectors/test_cluster.py
git commit -m "feat(cluster): wire co-occurrence path into discovery_scan

discovery_scan now runs both _iter_candidate_groups (creation-window) and
_iter_co_trade_groups (shared obscure markets) sequentially. Existing
_consider_group + _cluster_id_for SHA256 dedupe ensures the same wallet
set discovered by both paths produces one alert.

End-to-end tests cover:
- Cavill-shape detected via Path 1 (regression, score=7)
- Organic cluster (5 wallets, weeks-apart creation) detected via Path 2 (score=5)
- Same cluster discovered by both paths emits exactly one alert
- High-volume markets correctly excluded by obscurity gate
- Wallet with NULL first_activity_at handled via cohesion-with-non-NULL filter
- Group below threshold (no synchronized direction) does not emit

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (notes to the executor)

After all 4 tasks complete, verify the spec is fully covered:

1. **Spec coverage** (`docs/superpowers/specs/2026-04-27-cluster-detector-organic-design.md`):
   - "K-shared-obscure-markets candidate-group derivation" — Task 3
   - "Connected components consolidation" — Task 3 (`_bfs_component`)
   - "Signal A refactor to compute creation cohesion from real timestamps" — Task 2
   - "Add a parallel path; existing creation-window path stays untouched" — Task 4
   - "`max_co_trade_group_size` config field, default 100" — Task 1
   - "Cluster ID dedup via existing `_cluster_id_for`" — Task 4 (via `_consider_group` → `seen_cluster_ids`)
   - "No new schema, no new repos, no scheduler changes" — confirmed; only `cluster.py` + `config.py` + tests modified

2. **Type consistency:**
   - `_compute_creation_cohesion_score(group: list[WalletFirstSeen]) -> int` — used in Task 2 (definition) and unchanged through Task 4.
   - `_iter_co_trade_groups(recent: list[WalletFirstSeen]) -> Iterable[list[WalletFirstSeen]]` — same shape as `_iter_candidate_groups` so `_consider_group` accepts both.
   - `_build_obscure_markets_index(recent: list[WalletFirstSeen]) -> dict[str, set[ConditionId]]` — declared in Task 3.
   - `_bfs_component(start: str, adjacency: dict[str, set[str]]) -> set[str]` — static method, declared in Task 3.

3. **No placeholders** — every code block contains real code; commit messages are specific.

4. **Commit cadence** — 4 commits, one per task.

---

## Out-of-plan follow-ups (not blocking)

- **Volume-farming cluster detection.** Separate signal-tuning problem. The
  obscurity gate (Signal B) excludes high-volume markets the volume-farming
  cluster targets, and Signal D's burst detection misses their independent
  pacing. Could be addressed by a future "sustained co-occurrence" signal
  or a per-cluster-type obscurity threshold. Out of scope here.
- **Cluster-detector candidate population is still `wallet_first_seen`.**
  Discovering siblings of an unwatched cluster requires the
  move-attribution / cluster-expansion paths, not the cluster detector.
  Already noted in earlier work.
- **`_iter_co_trade_groups` performance at scale.** Pairwise `O(n²)` scan
  is fine for `wallet_first_seen` ≤ 1000. If population grows past several
  thousand, consider an inverted index (market → traders) and BFS from
  market hub nodes instead of all-pairs.
- **Group-truncation policy.** Current behavior truncates by sorted address
  (first N). For real clusters at scale, ranking truncation by something
  like "most-shared-markets-with-other-members" might preserve more signal.
  Defer until truncation actually fires in production.
