# Volume-farming cluster — investigation log

Living record of a 722+ wallet operation discovered 2026-04-27 during paper-
trading soak analysis. Distinct from the [Cavill cluster](cavill-cluster-investigation.md):
different strategy (dust-bet entry-and-hold vs. $500-999 maker-rebate),
different temporal pattern (gradual ramp Feb-Apr 2026 vs. 24h Feb-20-21
batch), different scale (~4× larger), and **dramatically unprofitable**.

## Status (2026-04-27)

- 722+ wallets identified by signature filter on a 9,346-candidate expansion.
- 40% cross-market hit rate on a 3-market sample → coordinated market
  selection (not independent retail).
- **Aggregate ROI on top-30 sample: −23.8%** ($6.35M invested, $1.51M lost).
- **Win rate: 7.1%** — they systematically buy outcomes that resolve against
  them.
- Almost certainly a Polymarket points / airdrop farming operation.

## How the cluster was found (2026-04-27)

1. The new `MoveAttributionDetector` auto-watchlisted 20 wallets across 4
   `cluster.candidate` events during the Apr 26-27 daemon runs.
2. Data exploration of the 20 cluster.candidate seeds showed a strikingly
   different fingerprint from Cavill (96% sub-$10 trades, 9% sell rate,
   creation dates spread Feb-Apr 2026).
3. `scripts/expand_cluster.py` ran against the 20 seeds and produced 9,346
   candidates passing the loose ≥2-markets-or-≥4-trades gate.
4. Re-scoring with the operation's distinctive signature (sell rate <20%,
   median trade <$10, ≥5 markets co-traded, ≥10 trades) reduced to **722
   strict candidates**.
5. Top-50 first-activity verification confirmed the temporal pattern:
   72% created in week 17 (Apr 21-27 2026), 96% in April overall.
6. Spot-check on 3 high-overlap markets (NHL Oilers/Ducks, WTA Sabalenka/Osaka,
   Russia-Ukraine ceasefire) confirmed coordination signature: 40%
   cross-market hit rate.
7. Settled-positions P&L pull on top-30 strict candidates revealed −$1.51M
   aggregate cashPnL with 7.1% win rate.

## Operation profile

### Scale (top-30 sample of 722 strict candidates)

| metric | value |
|---|---:|
| settled positions | 11,158 |
| total invested | $6,352,441 |
| total cashPnL | −$1,512,813 |
| ROI | −23.8% |
| win rate | 7.1% |
| profitable wallets | 4 of 29 (+$234 total) |
| unprofitable wallets | 25 of 29 (−$1.51M total) |

### Worst single-wallet losses

| wallet | positions | invested | cashPnL | win rate |
|---|---:|---:|---:|---:|
| 0x84ad9c5c | 2,754 | $4.51M | −$1.19M | 1% |
| 0x0b450a6b | 1,120 | $511k | −$118k | 2% |
| 0x5471604e | 799 | $381k | −$86k | 1% |
| 0x32ccd901 | 68 | $406k | −$35k | 50% |
| 0xd154d336 | 389 | $120k | −$28k | 3% |

`0x84ad9c5c` is the operation flagship: created 2026-04-27 12:29 UTC, generated
**11,373 trades** in ~2 hours from creation, lost $1.19M at 1% win rate.

### Fingerprint (vs. Cavill)

| dimension | Cavill (Feb 2026) | volume-farming cluster |
|---|---|---|
| operation type | $500-999 maker-rebate harvest on niche markets | dust-bet entry-and-hold across many markets |
| sell rate | 33% | ~9% (mostly buying) |
| trade-size mode | $500-999 main + sub-$100 dust | sub-$10 dust only (96%) |
| temporal | tight 24h batch (Feb 20-21) | gradual ramp Feb-Apr, accelerating in April |
| wallets known | ~190 | 722+ (strict) |
| trades / wallet | ~32 | 70-11,000 (variance is huge) |
| likely intent | maker rebates (mixed-direction trading) | points / airdrop farming (unidirectional) |
| profitability | not measured | **−23.8% ROI**, 7.1% WR |

### Cross-market overlap (3-market sample)

Sampled three markets where ≥9 of the original 20 cluster.candidate seeds
co-traded:

- **NHL Oilers vs Ducks** — 3,500 total trades, **180 of 965 wallets** in strict
  set, accounting for 39% of trade volume.
- **WTA Sabalenka vs Osaka** — 3,500 total trades, **177 of 1,013 wallets** in
  strict set, accounting for 32% of volume.
- **Russia-Ukraine Ceasefire (sustained 200+ trade market)** — 2,074 total
  trades, **104 of 389 wallets** in strict set, accounting for 45% of volume.

Of the 308 strict-set wallets seen across these three markets:

| wallets in… | count |
|---|---:|
| 1 of 3 markets | 184 |
| 2 of 3 markets | 95 |
| **3 of 3 markets** | **29** |

40% cross-market hit rate. With 3 random markets sampled from a universe of
thousands, this is far above what independent random hits would predict.

### Within-market timing

Trades **NOT** time-coordinated bursts. Each wallet's trades on a single
market span 60-120 minutes — sustained entry, not a synchronized pump.
Suggests independent pacing per wallet running the same selection logic.

## Why are they doing this?

Plausible motives, ranked by likelihood:

1. **Polymarket points / airdrop farming.** Most likely — accumulate
   trader-volume metrics, never crystallize losses by selling, hold to
   resolution. For this to make sense, expected airdrop value must exceed
   ~$50k per wallet (avg loss across the top losers). Operation accelerating
   in week 17 suggests an incentive deadline.
2. **Bot training data / market microstructure research.** Possible but at
   $1.5M+ aggregate loss it would be an extremely expensive academic
   exercise.
3. **Profit-seeking strategy.** Dismissable — 7.1% win rate with 24%
   drawdown is not a viable strategy.

The structural pattern of how they lose (buying YES on outcomes that resolve
NO, or buying tail outcomes at low prices) is consistent across the sample.

## Why the cluster detector missed them

`ClusterDetector.discovery_scan` requires Signal A (creation-time
clustering within `creation_window_seconds`, default 86400 = 24h). These
wallets were created over a 73-day span (Feb 13 → Apr 27 2026), with no
24-hour batch boundary. The detector's candidate-grouping greedy partition
(`_iter_candidate_groups`) yields only contiguous waves; this organically-
growing operation never forms one. The B/C/D signals (shared markets,
size correlation, direction correlation) would all fire if the candidate
group existed, but they never run because Signal A is the gate.

This is a structural limitation worth filing as a follow-up: `ClusterDetector`
needs a path to detect organically-growing clusters, not just batch
deployments.

## Methodology (reproducible)

1. **Pull the cluster.candidate seeds:**
   ```bash
   uv run python -c "
   import sqlite3
   con = sqlite3.connect('data/pscanner.sqlite3')
   for r in con.execute(\"SELECT address FROM wallet_watchlist WHERE source='cluster.candidate'\"):
       print(r[0])
   " > /tmp/seeds.txt
   ```

2. **Run cluster expansion:**
   ```bash
   flags=""; while read -r a; do flags="$flags --wallet $a"; done < /tmp/seeds.txt
   uv run python scripts/expand_cluster.py $flags --skip-verify --out /tmp/expansion.json
   ```

3. **Re-score with the operation's signature filter:**
   - `sell_rate < 0.20`
   - `median_usd < 10.0`
   - `n_markets >= 5`
   - `n_trades >= 10`

4. **Verify creation timestamps (top-N):**
   ```python
   from pscanner.poly.data import DataClient
   # call get_first_activity_timestamp(wallet) per candidate
   ```

5. **Confirm coordination via cross-market overlap:**
   - Pick 3 high-overlap markets from the seed-market list.
   - Pull `/trades?market=` for each.
   - Count strict-set wallets that appear in ≥2 of 3 markets.
   - >25% cross-market hit rate is strong evidence of coordination.

6. **Profitability check:**
   - For top-N strict candidates, call `data-api /positions?user=X&closed=true`.
   - Sum `cashPnl` and `totalBought` per wallet.
   - Aggregate across the sample. ROI < 0 with WR < 30% confirms
     non-profit-seeking.

## Artifacts (local, not committed)

- `/tmp/new-cluster-seeds.txt` — 20 cluster.candidate seed addresses
- `/tmp/new-cluster-expansion.json` — 9,346 raw candidates from expand_cluster
- `/tmp/new-cluster-strict.json` — 722 strict candidates after signature filter
- `/tmp/new-cluster-top50-verified.json` — top-50 with first-activity timestamps
- `/tmp/cluster-pnl.json` — top-30 settled-position aggregates

These are throwaway artifacts of one investigation session; the methodology
is the primary artifact.

## Open follow-ups

- **`ClusterDetector` Signal A limitation.** ✅ Resolved 2026-04-27 — the
  organic-discovery path landed in `feat/cluster-organic` and rediscovered
  the Cavill cluster automatically (creation spread 86,496s = 24h 1m 36s,
  just past the gate, found via co-occurrence path).
- **Polymarket points-farming exploit.** This is real-world product
  intelligence; the operator is investing $1.5M+ to game whatever incentive
  Polymarket is running. Worth flagging upstream if there's a reporting
  channel.
- **Smart-money detector hardening.** None of these 722 wallets meet the
  `weighted_edge > 0` filter on `tracked_wallets`, so paper-trading copy-
  trade is unaffected. But the smart-money detector should never include any
  of them as "smart" — verify the filter chain holds for high-volume
  negative-edge wallets specifically.
- **Watch the top-30 wallets.** Adding them to the watchlist would let the
  daemon track their ongoing activity and confirm whether the operation
  scales further or stops on a deadline. The 4 already-watchlisted seeds
  (`source=cluster.candidate`) are a small subset.

---

## Update 2026-04-28: A second-generation farming cluster ("Magic / long-shot")

During a 2h paper-trading-expansion smoke run, the `MoveAttributionDetector`
auto-watchlisted **24 cluster-candidate wallets** triggered by velocity
alerts on tail markets — most prominently the Orlando Magic NBA Finals
market (YES = $0.0095). 10 of those came from a single velocity burst on
the Magic market.

`scripts/expand_cluster.py` on the 10 Magic seeds produced **881
candidates**, narrowed to **17 strict** by the same signature filter
(sell_rate<20%, ≥5 markets, ≥10 trades). Investigation results:

### Scale (17 strict candidates)

| metric | value |
|---|---:|
| settled positions | 5,096 |
| total invested | $1,128,478 |
| total cashPnl | −$390,549 |
| **ROI** | **−34.6%** |
| win rate | 30.7% |
| profitable wallets | 4 of 17 (+$1,269) |
| unprofitable wallets | 13 of 17 (−$391,818) |

### Strategy fingerprint (concrete: influenz.eth at −56.5%)

Inspecting one representative strict candidate (`0xe8dd7741…`,
"influenz.eth"):

- 500 of 500 settled positions are buys of **YES tokens only** (no NO buys).
- 479 of 500 entries at avg_price < $0.05 (i.e. buying outcomes the
  market thought had <5% probability).
- 14.6% of those tail bets resolved YES; 85.4% lost the entire entry.
- Even winners only realized ~1.4× of invested capital (many are
  intermediate sales, not full $1 redemptions). Aggregate −56.5% ROI.

This is a **long-shot tail-buying strategy**, not a maker-rebate harvest
(Cavill) and not a uniform dust spray (volume-farming Feb-Apr). The
fingerprint name "near-certainty arbitrage" — initially used because the
expansion script's `hi_price_rate` field showed 0.78 — was misleading.
That field came from the seeds' burst trades on a near-certainty market;
the strict candidates' actual full trade history is overwhelmingly tail
buying.

### Creation pattern

Span: 50 days (Mar 8 → Apr 27 2026), accelerating heavily in last 4 days.

| date | wallets created |
|---|---:|
| 2026-03-08 to 2026-03-12 | 3 |
| 2026-04-20 to 2026-04-26 | 8 |
| 2026-04-27 (single day) | 6 |

Day-of-week skew: 8 of 17 (47%) created Monday; 0 created Tuesday.
Anomalous vs uniform ~14%/day. The 10-wallet seed batch (separately
created Apr 24-27) extends this to 27 wallets in 4 days, all converging
on a small basket of hot retail markets (BTC dip, Iran ops, Iran
diplomacy, Magic NBA).

### ENS verification

Real on-chain ENS reverse-resolution against api.ensideas.com: **0 of 17
strict candidates have a real ENS record**. Polymarket allows display
names that include `.eth` (like "influenz.eth") even without owning the
ENS record. The 1-of-881 `.eth` count earlier was self-set Polymarket
display names, not on-chain reverse records.

### Comparison to prior clusters

| operation | wallets | strategy | win rate | ROI |
|---|---:|---|---:|---:|
| Cavill (Feb 2026) | 9 | maker-rebate harvest, $500-999 chunks | not measured | not measured |
| volume-farming (Feb-Apr) | 722 | dust-bet (sub-$10) all prices | 7.1% | −23.8% |
| **Magic / long-shot (Mar-Apr)** | **17 strict + tail of ~700** | **YES at <$0.05 on tails** | **30.7%** | **−34.6%** |

All three end-state losses fit the same airdrop-credit hypothesis: cost
of generating volume for token-launch eligibility. This Magic generation
is a fresh wave with a different sub-strategy.

### Methodology refinement

For future cluster investigations on freshly-spawned wallets, note:

1. The local `wallet_trades` table won't have data for newly-watchlisted
   wallets until `TradeCollector` polls them (~5 min). Pre-populate via
   `DataClient.get_activity` if you want to run `expand_cluster.py`
   immediately.
2. The expansion script's `hi_price_rate` field measures TRADE prices in
   the seed-overlap window, NOT the candidate's general behavior. To
   distinguish "near-certainty" from "tail-buying" candidates, fetch
   `/positions?user=X&closed=true` and look at `avg_price` distribution.
3. **The "near-certainty arbitrage" thesis is empirically dead** for these
   wallets. May still be a profitable strategy for someone else, but the
   wallets running it via long-shot tail-buying lose money systematically.

### Files

- `/tmp/magic-cluster-expansion.json` — 881 candidates from the 10 seeds.
- `/tmp/magic-cluster-sample-first-activity.json` — 127-wallet stratified
  first-activity sample (60% created April 2026, 47% Mondays for strict).
- `/tmp/magic-strict-investigation.json` — per-wallet PnL, ENS, creation
  for the 17 strict candidates.
