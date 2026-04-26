# Cavill cluster — investigation log

Living record of what's been found about the Cavill (Polymarket maker-rebate
farming) wallet cluster across sessions. Append new findings; don't rewrite.

## Status (2026-04-26)

- **Original 9 wallets** (CLAUDE.md): manually discovered 2026-04-25, auto-confirmed
  by cluster detector with `discovery_lookback_days=365`, score=7, tag=`mixed`.
  Created Feb 20-21 2026 (7 of 9 within a 38-min window on Feb 20).
- **Extended scan (this session, 2026-04-26): cluster is ≥190 wallets, not 9.**
  181 additional wallets surfaced via co-trade + fingerprint match; 99.5% of those
  were created in the same Feb 20-21 window.

## How the 181 were found (2026-04-26)

1. Pulled all `wallet_trades` rows for the 9 known wallets from local DB
   (290 trades across 120 condition_ids). 13 markets had ≥3 cluster wallets
   co-trading or ≥4 total cluster trades — the "high-coordination" subset.
2. For each of those 13 markets, paginated `data-api.polymarket.com/trades?market=`
   within ±1h of the cluster's per-market trade window. Collected 1001 trade rows
   from 250 distinct wallets (9 cluster + 241 non-cluster).
3. Filtered to wallets that co-traded ≥2 of the 13 markets OR ≥4 trades total →
   182 candidates.
4. Verified each candidate's `first_activity_ts` via `DataClient.get_first_activity_timestamp`.
   181/182 fell inside the 2026-02-20/21 window. Only outlier: `0x65c68419…`
   (created 2026-03-20).

The Feb 20-21 24-hour window is so narrow that landing 181/182 there is
essentially a confirmation, not a coincidence.

## Cluster fingerprint (validated against the original 9, n=290 trades)

| metric              | cluster value | claim in CLAUDE.md |
|---------------------|---------------|---------------------|
| sell rate           | 33.4%         | 35-40%              |
| price ≥ 0.95 share  | 56.2%         | 57%                 |
| usd_value < $100    | 52.1% (dust)  | bimodal             |
| usd_value $500-999  | 47.2% (main)  | bimodal             |

Top candidates' main-play trades (the ones we filter on) collapse to ~100% BUY
at price ≥0.95 in the $500-999 band, because we filter to high-coordination
markets where the cluster only places main-play trades. The dust is spread
across the long-tail markets we didn't scan.

## Wallet creation-time histogram (181 newly-found, 10-min UTC buckets)

```
2026-02-20 13:00    7  #######
2026-02-20 13:10   29  #############################
2026-02-20 13:20   17  #################
2026-02-20 13:30   20  ####################
2026-02-20 13:40   15  ###############
2026-02-20 13:50   18  ##################
2026-02-20 14:00   21  #####################
2026-02-20 14:10    6  ######
2026-02-21 05:40    1  #
2026-02-21 05:50   14  ##############
2026-02-21 06:00    8  ########
2026-02-21 06:10    3  ###
2026-02-21 07:10    4  ####
2026-02-21 07:20    8  ########
2026-02-21 07:30    5  #####
2026-02-21 13:00    3  ###
2026-02-21 13:10    2  ##
```

Pattern: 3 distinct deployment batches —
- **Batch A**: 2026-02-20 13:00-14:10 UTC, ~133 wallets in ~70 minutes
- **Batch B**: 2026-02-21 05:40-07:30 UTC, ~43 wallets in ~110 minutes
- **Batch C**: 2026-02-21 13:00-13:10 UTC, ~5 wallets

The original 9 fell mostly in Batch A.

## Market overlap distribution (181 newly-found wallets)

| markets co-traded w/ cluster | wallets |
|------------------------------|--------:|
| 7                            |       1 |
| 6                            |      15 |
| 5                            |      36 |
| 4                            |      73 |
| 3                            |      39 |
| 2                            |      17 |

## Artifacts

- `/tmp/cavill_extended_cluster.txt` — 181 newly-found addresses (one per line)
- `/tmp/cavill_candidates_verified.json` — full per-wallet stats
- `/tmp/cavill_markets.json` — the 120 condition_ids the original 9 traded
- `/tmp/cluster_market_trades.json` — 1001 rows fetched from /trades for the
  13 high-coordination markets

## Open questions / follow-ups

- Cluster detector under-counted by ~95%. Likely cause: candidate set is bootstrapped
  only from already-tracked wallets (`wallet_first_seen`), so siblings of a watched
  wallet are invisible until they themselves get watched. **Need to verify by reading
  the detector source.**
- The 13 markets I scanned are a subset of 120 the cluster touched. Long-tail
  markets may surface still more sibling wallets — possibly older deployment batches.
- Cluster's actual size is unknown; 190 is a lower bound from a subset of markets.
- Should add the 181 to the watchlist (reason `cavill-cluster-extended-feb2026`)
  so the trade collector picks up their ongoing activity.
- CLAUDE.md still says "9-wallet coordinated operation" — now obsolete.

## Re-add full extended cluster after a DB reset

```bash
while read a; do uv run pscanner watch "$a" --reason cavill-cluster-extended-feb2026; done < /tmp/cavill_extended_cluster.txt
```

## Cluster detector walkthrough (2026-04-26)

### What the detector does today

`src/pscanner/detectors/cluster.py`. Two paths:

**Discovery path** (`discovery_scan`, cadence `scan_interval_seconds=3600s`):
1. `recent = self._first_seen.list_recent(within=discovery_lookback_days)` — pulls
   wallets from the `wallet_first_seen` table, default lookback 30 days.
2. `_iter_candidate_groups`: greedy-partitions the recent set into "waves" by
   `first_activity_at`, chained — a new wave starts when the gap to the previous
   wallet exceeds `creation_window_seconds=86400` (24h). Waves smaller than
   `min_cluster_size=3` are dropped.
3. `_score_candidate` per wave, summing 4 signals:
   - **A: creation clustering** — already passed (group ≥ min size) → +2
   - **B: niche-market overlap** — `_find_shared_obscure_markets`: scans each
     wallet's `wallet_trades.recent_for_wallet(limit=500)`, counts markets where
     ≥ 3 cluster wallets traded AND market is "obscure" (liquidity ≤ $50K AND
     volume ≤ $1M, both required, neither NULL). If ≥ `min_shared_markets=3` → +2
   - **C: trade-size correlation** — for any shared market, pstdev/mean (CV) of
     trade sizes < `max_trade_size_cv=0.3` → +1
   - **D: direction correlation** — bucketing trades into
     `direction_window_seconds=600s` windows by (asset_id, side); if ≥
     `min_direction_correlation_count=3` distinct wallets fall in one bucket → +2
4. Score ≥ `discovery_score_threshold=5` emits `cluster.discovered`. The Cavill
   detection scored 7 (all four signals fired).

**Active path** (`evaluate_active`, fired by `TradeCollector.subscribe_new_trade`):
- Looks up the wallet's cluster_id; if it has one, counts how many of that
  cluster's members touched the same condition_id in the last
  `active_window_seconds=300s`. If ≥ `active_min_members=2` → emits
  `cluster.active`.

### Suspected flaws

**1. The candidate population is exactly the watched wallets — discovery is
structurally limited to wallets we already know about.**

Trace:
- `discovery_scan` reads from `WalletFirstSeenRepo.list_recent(...)`.
- `wallet_first_seen` rows are written ONLY by `TradeCollector._ensure_first_seen`
  (`src/pscanner/collectors/trades.py:188-212`).
- That helper is called from `_poll_wallet`, which is called for every wallet in
  `self._registry.addresses()` (`trades.py:134-137`).
- The registry contains: manually-watched wallets (`pscanner watch`), smart-money
  tracked wallets, whale-alert wallets, and similar — i.e. wallets we've already
  surfaced through some other path.

So the detector is a **coordination verifier**, not a coordination discoverer. It
can confirm that a set of *already-watched* wallets are coordinated. It cannot
find sibling wallets you haven't already brought into the watchlist.

This explains the 9 vs 190 gap. We brought the original 9 in by hand. Their
siblings never got watched, so they never got rows in `wallet_first_seen`, so
the detector never considered them — even though they trade the same niche
markets, in the same windows, with the same size pattern.

**2. Signal B's "obscure market" bar drops markets the cluster targets.**

`_is_obscure_market` requires both `liquidity_usd ≤ $50K` AND `volume_usd ≤ $1M`,
and excludes markets with NULL liquidity/volume. The cluster's high-coordination
markets are niche but ACTIVE — Henry Cavill James Bond, Ferran Torres top scorer,
Manchester United 2nd place EPL — these can have volume well over $1M once a
real betting flow shows up. If they exceed the volume cap, signal B fails even
though the structural overlap is exactly what we want to catch.

**3. `_RECENT_TRADE_LIMIT = 500` is a module constant, not config.**

A wallet with > 500 lifetime trades silently has its older history dropped from
both `_find_shared_obscure_markets` and `_collect_cluster_trades`. The Cavill 9
weren't affected (23-37 trades each), but high-volume cluster members would be.

**4. `_MIN_WALLETS_PER_SHARED_MARKET = 3` is also a hardcoded constant.**

A cluster of 2 wallets can't pass signal B regardless of overlap.

**5. Active monitoring depends on `subscribe_new_trade`, which only fires for
watched wallets.**

Same root cause as #1 — sibling cluster members trading the same market won't
trigger `cluster.active` because we never see their trades.

### What "fixed" would look like

The cleanest fix is a **discovery-expansion pass**: when a candidate cluster
hits `discovery_score_threshold`, before emitting, fetch
`/trades?market=<cond>` for each shared market within the cluster's active
window, collect all distinct counterparties, fingerprint-match them, and verify
their first-activity falls inside `creation_window_seconds`. This is exactly
the manual procedure that surfaced 181 siblings in this session.

Cheaper partial fixes:
- Promote `_RECENT_TRADE_LIMIT` and `_MIN_WALLETS_PER_SHARED_MARKET` to
  `ClusterConfig` so they're tunable.
- Reconsider signal B's volume cap, or relax it when liquidity is well below cap.
- After confirming a cluster, auto-watchlist sibling co-traders (rate-limited)
  so the next cycle's signal B / D / active path actually sees them.

## Discovery is the actual bottleneck (2026-04-26)

The Cavill cluster was found because John was investigating an unrelated
market move surfaced by another signal, looked at the trade list behind the
move, and noticed a pattern of similarly-sized trades across many wallets.
The cluster detector then *verified* the 9 wallets he manually flagged.

That means **the discovery step happened in the human-in-the-loop chain, not
in any detector**. There is no detector in pscanner today that:

1. Picks up an alerting market move from another detector,
2. Attributes the move's volume to its top trade contributors,
3. Asks "do these contributors look fingerprint-similar?"

That's the missing primitive. Without it, every new cluster we find requires
the same human-in-the-loop investigation — and we'll only find clusters that
happened to be active during a market move we noticed.

The cluster expansion script is the *second* half of this pipeline (verify
+ expand a known seed). The *first* half — turning a market move into a
candidate seed — is still manual.

## Tools

- `scripts/expand_cluster.py` — given seed wallets (`--wallet …` or
  `--cluster-id …`), pulls the seed's high-coordination markets, fingerprints
  every counterparty, verifies their first-activity, and writes
  `data/cluster_expansion_<digest>.json`. Validated against the Cavill seed:
  182 candidates, 99.5% in the seed creation window. Use `--skip-verify` to
  skip the per-candidate first-activity lookup (~6 min saved at rpm=50, but
  loses the cluster-window classification).
