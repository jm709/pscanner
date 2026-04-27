# Cluster detector — organic discovery design

Date: 2026-04-27
Status: design — pending implementation plan

## Problem

`ClusterDetector` was designed around the Cavill cluster's signature: 9
wallets created within a 24-hour batch deployment. `_iter_candidate_groups`
greedy-partitions `wallet_first_seen` into "waves" by `first_activity_at`,
chained — a new wave starts when the gap between consecutive wallets exceeds
`creation_window_seconds` (default 86400 = 24h). Waves smaller than
`min_cluster_size` are dropped. Signals B/C/D only score on whatever waves
this partition produces.

This works for batch-deployed clusters. It misses organically-grown clusters
where wallets are added gradually over weeks or months. The 722-wallet
volume-farming cluster identified during the 2026-04-27 investigation has
creation timestamps spanning 73 days (Feb 13 → Apr 27 2026) — Signal A's
gate blocks it from ever forming a candidate group, so signals B/C/D never
get to run.

## Goal

Add a second candidate-group source that discovers wallet groups via
shared-obscure-market overlap, regardless of when those wallets were
created. Existing creation-window path stays untouched (Cavill stays
detected). Existing scoring signals (B, C, D) are reused. Signal A is
refactored from "always +2 because the candidate group came from a
creation-clustered partition" to "+2 if the actual timestamps are
clustered" — so the existing path keeps awarding +2, and the new path
awards +2 only when the discovered organic cluster also happens to have
creation cohesion.

This work targets organic clusters with the same fingerprint as Cavill —
wallets that share obscure markets, trade at similar sizes (low CV), and
exhibit synchronized direction. It does **not** target the volume-farming
cluster — that operation has separate gaps (high-volume markets that fail
the obscurity gate, no synchronized bursts that would fire Signal D), which
are signal-tuning problems out of scope here.

## Decisions (locked during brainstorming)

1. **Scope**: detect Cavill-shaped organic clusters. Volume-farming-cluster
   detection is a separate signal-tuning problem deferred to a later
   iteration.
2. **Candidate-group derivation for the new path**: K-shared-obscure-markets
   with connected-components consolidation. K reuses the existing
   `min_shared_markets` config field.
3. **Integration**: add a parallel path. Keep the existing creation-window
   path intact. Both paths feed the same `_consider_group` scoring pipeline.
   Existing `_cluster_id_for` SHA256 dedupes when both paths discover the
   same wallet set.
4. **Signal A**: refactor from unconditional +2 to "+2 if actual timestamps
   are clustered within `creation_window_seconds`, else 0". Existing path's
   groups are creation-clustered by construction, so behavior is unchanged
   for Path 1.
5. **No new config except a safety cap**: `max_co_trade_group_size` (default
   100) prevents a misconfigured obscurity gate from producing a runaway
   1000-wallet candidate.

## Architecture

`ClusterDetector.discovery_scan` runs two sequential candidate-group
sources, each feeding the same scoring pipeline:

```
seen = set()
count = 0

for group in _iter_candidate_groups(recent):       # existing creation-window path
    if _consider_group(group, seen, sink): count += 1

for group in _iter_co_trade_groups(recent):        # new co-occurrence path
    if _consider_group(group, seen, sink): count += 1

return count
```

`_cluster_id_for` (SHA256 of sorted addresses) dedupes — a Cavill-shaped
cluster discovered by both paths produces one alert. The `seen` set ensures
each cluster is scored once per scan.

Reuses (zero new infrastructure):
- `WalletFirstSeenRepo.list_recent` — same input universe as existing path
- `WalletTradesRepo.recent_for_wallet` — same trade lookup
- `MarketCacheRepo.get_by_condition_id` + `_is_obscure_market` — same
  obscurity gate
- Existing `_consider_group`, `_score_candidate`, `_emit_discovered`,
  `_cluster_id_for`

## Components

### `_iter_co_trade_groups`

```python
def _iter_co_trade_groups(
    self,
    recent: list[WalletFirstSeen],
) -> Iterable[list[WalletFirstSeen]]:
    """Yield candidate groups derived from shared-obscure-market overlap.

    1. Build per-wallet obscure-market sets (locally; not shared across scans).
    2. Pairwise edge construction: edge if shared-market count ≥ min_shared_markets.
    3. Connected components via BFS.
    4. Yield each component of size [min_cluster_size, max_co_trade_group_size].
    """
```

The function owns its own per-scan cache. Existing scoring helpers
(`_find_shared_obscure_markets`, `_collect_cluster_trades`) are unchanged
— they re-query `trades_repo` per group during scoring. The redundant
per-wallet trade queries are bounded by `_RECENT_TRADE_LIMIT * group_size`
and acceptable for current data volume.

Algorithm:

```
1. Per-wallet obscure-market sets (function-local cache):
   obscure_markets: dict[str, set[ConditionId]] = {}
   for each wallet in recent:
     trades = trades_repo.recent_for_wallet(wallet, limit=_RECENT_TRADE_LIMIT)
     obscure_markets[wallet.address] = {
       trade.condition_id for trade in trades
       if market_cache.get_by_condition_id(trade.condition_id) passes _is_obscure_market
     }

2. Pairwise edge construction:
   for each (w1, w2) pair, w1 < w2 lex:
     shared_count = |obscure_markets[w1] ∩ obscure_markets[w2]|
     if shared_count >= cfg.min_shared_markets:
       add edge w1 ↔ w2

3. Connected components via BFS:
   visited = set()
   for w in recent (sorted by address):
     if w.address in visited: continue
     component = bfs(w, adjacency)
     visited |= {x.address for x in component}
     if len(component) < cfg.min_cluster_size: continue
     if len(component) > cfg.max_co_trade_group_size:
       _LOG.warning("cluster.cotrade_group_truncated", original_size=len(component))
       component = sorted(component, key=lambda w: w.address)[:cfg.max_co_trade_group_size]
     yield component
```

### `_compute_creation_cohesion_score`

```python
def _compute_creation_cohesion_score(self, group: list[WalletFirstSeen]) -> int:
    """Return 2 if the group's creation timestamps fall within
    creation_window_seconds, else 0.

    Wallets with NULL first_activity_at are excluded from the comparison;
    if the remaining set is below min_cluster_size, returns 0.
    """
    timestamps = [w.first_activity_at for w in group if w.first_activity_at is not None]
    if len(timestamps) < self._config.min_cluster_size:
        return 0
    if max(timestamps) - min(timestamps) <= self._config.creation_window_seconds:
        return 2
    return 0
```

### `_score_candidate` change

Replace:
```python
if len(group) >= self._config.min_cluster_size:
    score += 2  # Signal A
```

With:
```python
score += self._compute_creation_cohesion_score(group)
```

The behavioral effect on Path 1 is **zero**: groups produced by
`_iter_candidate_groups` always have `max-min <= creation_window_seconds`
by construction, so cohesion always returns 2. The change only matters for
groups produced by Path 2.

### Config additions

```python
class ClusterConfig(_Section):
    ...
    max_co_trade_group_size: int = 100
```

No other new fields. The new path reuses `min_shared_markets`,
`min_cluster_size`, `max_shared_market_liquidity_usd`,
`max_shared_market_volume_usd`.

### File-level changes

- Modify: `src/pscanner/detectors/cluster.py` — add `_iter_co_trade_groups`,
  add `_compute_creation_cohesion_score`, change `_score_candidate` to use
  the new helper, modify `discovery_scan` to iterate both paths.
- Modify: `src/pscanner/config.py` — add `max_co_trade_group_size` field on
  `ClusterConfig`.
- Modify: `tests/detectors/test_cluster.py` — unit tests for the two new
  helpers, end-to-end test for organic-cluster discovery, regression test
  that Cavill-style detection still works, dedup test for both paths
  finding the same cluster.

No new repos, no new schema, no scheduler changes. The detector is already
wired and the scan cadence is unchanged.

## Data flow

```
discovery_scan(sink)                                 [periodic, default every 1h]
    recent = WalletFirstSeenRepo.list_recent(within=lookback_days)
    if len(recent) < min_cluster_size: return 0
    seen: set[str] = set()
    count = 0

    # ── Path 1: existing creation-window partition ─────────────────────────
    for group in _iter_candidate_groups(recent):
         _consider_group(group, seen, sink)
              ├── cluster_id = _cluster_id_for(group)
              ├── if cluster_id in seen → skip
              ├── if clusters_repo.get(cluster_id) is not None → skip (already known)
              ├── score, shared, behavior_tag = _score_candidate(group)
              │     ├── _compute_creation_cohesion_score(group)   → +2 (always for this path)
              │     ├── _find_shared_obscure_markets(wallets)     → Signal B (+2 if ≥3)
              │     ├── _has_size_correlation(...)                → Signal C (+1)
              │     └── _has_direction_correlation(...)           → Signal D (+2)
              ├── if score < discovery_score_threshold → skip
              └── _emit_discovered(...) → cluster.discovered alert + wallet_clusters row
        count += 1 if emitted

    # ── Path 2: new co-occurrence partition ────────────────────────────────
    for group in _iter_co_trade_groups(recent):
         (same _consider_group pipeline as above; same dedupe via seen + cluster_id)
        count += 1 if emitted

    return count
```

### Worked examples

**Cavill-shaped batch (existing path wins, new path dedupes silently):**
9 wallets created within 24h, share 3+ obscure markets, low size CV,
synchronized direction.
- Path 1: one wave of 9 → `_consider_group` → score=7 (A=2, B=2, C=1, D=2)
  → emits.
- Path 2: same 9 wallets share ≥3 obscure markets → 1 component → same
  cluster_id → already in `seen` → skip silently.
- Net: 1 alert. Existing behavior unchanged.

**Organic Cavill-shaped (new path wins):** 5 wallets created weeks apart,
share 4 obscure markets, low size CV, synchronized direction.
- Path 1: each wallet in its own wave (gaps > `creation_window_seconds`) →
  no candidates yielded.
- Path 2: 5 wallets connected via shared-market edges → 1 component of 5 →
  score = 0 (A) + 2 (B) + 1 (C) + 2 (D) = 5 = threshold → emits.
- Net: 1 alert. Organic cluster discovered.

**False-positive guard (random retail co-traders):** 8 wallets co-trading
3 popular sports markets with varied sizes and unsynchronized direction.
- Path 1: not creation-clustered → no group.
- Path 2: high-volume markets fail `_is_obscure_market` → empty
  obscure-market sets → no edges → no group.
- Net: 0 alerts. Obscurity gate prevents false positives.

### Consistency notes

- `wallet_first_seen` updates concurrent with the scan: `list_recent` reads
  once at the start; subsequent `_ensure_first_seen` writes don't affect
  this scan. Picked up next cycle.
- `market_cache` updates concurrent with the scan: each
  `_is_obscure_market` call reads independently. A market that flips
  active→closed mid-scan might get treated differently across the two
  paths. Acceptable — threshold-based scoring is robust to small
  membership shifts.
- Cluster ID dedup is what makes the two-path design safe. The hash is
  over the sorted address set — any path producing the same wallet group
  produces the same ID.

## Error handling

Same fail-soft posture as the rest of the detector — every helper is
wrapped at boundaries that could fail; nothing tears down the periodic
scan loop.

| Failure | Response |
|---|---|
| `WalletFirstSeenRepo.list_recent` raises | Existing top-level `try/except` in `run()` catches; logs `cluster.discovery_failed`; loop sleeps and retries. |
| `recent` empty or below `min_cluster_size` | Existing early return — both paths skipped. |
| All wallets have empty obscure-market sets | New path yields no components. Existing path still runs. |
| Wallet has `first_activity_at = NULL` | `_compute_creation_cohesion_score` filters out NULL entries. If non-NULL count drops below `min_cluster_size`, returns 0. Group still scored on B/C/D. |
| `WalletTradesRepo.recent_for_wallet` raises | Per-wallet try/except in the new path. On failure: log `cluster.cotrade_trades_failed` WARN, treat that wallet's `obscure_markets` as empty (no edges incident on it). Other wallets unaffected. |
| `MarketCacheRepo.get_by_condition_id` raises | Per-call try/except — treat the market as non-obscure. One bad cache row never blocks the rest. |
| Connected component above `max_co_trade_group_size` | Truncate deterministically by sorted address (first N). Log `cluster.cotrade_group_truncated` WARN with original size. |
| Two paths produce the same cluster_id | `seen` dedupes silently. |
| Two paths produce overlapping but not identical groups | Both scored independently; both emit if both pass threshold. Cluster IDs differ; both rows land in `wallet_clusters`. Operators get visibility into both groupings. |

## Testing

Three layers, reusing existing fixtures (`tmp_db`, `fake_clock`, the
existing stub-repo patterns from `tests/detectors/test_cluster.py`).

### Unit tests

`_compute_creation_cohesion_score`:
- All within window → returns 2
- Spread beyond window → returns 0
- All NULL → returns 0
- Some NULL, remaining ≥ `min_cluster_size`, all within window → returns 2
- Some NULL, remaining < `min_cluster_size` → returns 0

`_iter_co_trade_groups`:
- happy: 4 wallets each trading the same 3 obscure markets → 1 component of 4
- below threshold: 4 wallets sharing 2 obscure markets (< default 3) → 0
- transitive closure: A-B share 3, B-C share 3, A-C share 0 → ABC one component
- two disjoint clusters: 6 wallets in two triangles each sharing 3 markets → 2 components
- below `min_cluster_size`: 2 wallets sharing 5 markets → 0 (component size too small)
- mixed: some wallets connected, others isolated → only connected components yielded
- non-obscure markets excluded: 4 wallets sharing 5 high-volume markets → 0 (filtered)
- group size cap: 200 wallets all sharing 3 obscure markets → 1 truncated to 100, WARN logged

### Detector-level tests (end-to-end with real repos against `tmp_db`)

**Regression on creation-window path** (canonical Cavill-shape) —
9 wallets created within 24h, share 3 obscure markets, low size CV,
synchronized direction → score=7 → fires `cluster.discovered`. Asserts the
existing test's exact outcome continues to hold.

**Organic cluster discovery** — 5 wallets with `first_activity_at` spread
weeks apart, share 4 obscure markets, low size CV, synchronized direction.
Path 1 yields nothing; Path 2 yields one 5-wallet component → score=5 →
fires.

**Both paths discover the same cluster (dedup)** — 9 wallets created within
24h AND share 3 obscure markets. Asserts exactly one alert and one
`wallet_clusters` row.

**Both paths discover overlapping but distinct groups** — 7 wallets where
wallets[0..3] are creation-clustered (Path 1 finds them) and wallets[2..6]
share obscure markets (Path 2 finds them). Asserts two cluster IDs, two
rows, two alerts.

**Negative: organic group below threshold** — 5 wallets sharing 3 obscure
markets, low size CV, but no synchronized direction → score=3 < 5 → no
alert.

**Negative: high-volume sports markets** — 8 wallets co-trading 5
high-volume markets. Asserts the obscurity gate excludes those markets so
the new path produces no component. Confirms this work doesn't
accidentally flag the volume-farming cluster (out of scope).

**Wallet with NULL `first_activity_at`** — Group of 5 connected via shared
markets, one wallet has NULL timestamp. Score uses 4 non-NULL wallets for
Signal A; +2 if those are clustered. Score = 7. Fires.

**Per-wallet `recent_for_wallet` failure** — Mock to raise for one wallet
only. Other wallets' connections unaffected. Verifies isolation.

**Group size cap** — 200 wallets all sharing 3 obscure markets. New path
yields 1 component truncated to 100 wallets; WARN logged.

### Integration smoke test

Wires `WalletFirstSeenRepo`, `WalletTradesRepo`, `MarketCacheRepo`,
`WalletClustersRepo`, `WalletClusterMembersRepo` against `tmp_db` with a
real `AlertSink`. Seeds an organic cluster. Drives `discovery_scan` once.
Asserts:
- One alert in `alerts` with `detector="cluster"`,
  `alert_key` starting `cluster.discovered:`
- One row in `wallet_clusters` with the expected member count and score
- Members in `wallet_cluster_members` matches the seeded set

### What I'm NOT testing

- Multi-scan persistence/dedup — already covered by existing
  `clusters_repo.get(cluster_id)` pre-check.
- Internal BFS algorithm details — tested via observable behavior in the
  disjoint-clusters case.
- Performance — `wallet_first_seen` size is bounded by project usage; the
  `O(n²)` pairwise scan is well within budget.

### Test gotchas (per CLAUDE.md)

- Stub repos for existing detector tests keep working — the new path uses
  the same repo interfaces.
- `pyproject.toml`'s `filterwarnings = ["error"]` — none of the new code
  touches async; no fixtures to clean up beyond what's already established.

Estimated test count: ~13 unit + ~7 detector + 1 integration ≈ 21 tests.

## Out of scope (deferrable)

- **Volume-farming cluster detection.** Separate signal-tuning problem:
  Signal B's obscurity gate excludes high-volume markets that the
  volume-farming cluster targets, and Signal D's burst-detection misses
  the cluster's independent pacing. Could be addressed by a future
  "sustained co-occurrence" signal or by relaxing the obscurity gate per
  the cluster's profile.
- **Per-strategy thresholds.** The threshold model could be tuned per
  detection path (e.g. Path 2 could require a higher threshold than 5 to
  avoid noise from broader candidate generation). Not justified by current
  data; revisit if Path 2 produces too many false positives.
- **Graph community detection algorithms** (Louvain, label propagation).
  Connected-components is sufficient for current data volume. If
  `wallet_first_seen` grows past low thousands and we see meaningful
  cluster overlap (multiple operations sharing wallets), revisit.
- **Backfilling `wallet_first_seen` for unwatched wallets.** The detector
  still operates only on watched wallets. Discovering siblings of an
  already-watched cluster requires the move-attribution / cluster-expansion
  paths, not the cluster detector itself.
