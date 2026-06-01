# Insider-wallet discovery: case-control fingerprinting

Date: 2026-06-01
Status: Design — pending implementation plan
Related: `scripts/wallet_edge_leaderboard.py`, `scripts/copy_selection.py`,
`scripts/check_polymarket_wallet_pnl.py`,
`docs/superpowers/specs/2026-05-30-causal-copy-selection-design.md`,
`volume-farming-cluster-investigation.md`

## Problem

The `--causal-select` copy backtest (#207) is structurally a **persistence
filter**: a wallet is copyable only after `min_resolved >= 20` resolved trades
*and* a positive edge over its own past. A true insider is the negative of that
profile — **few trades, short active lifespan, on a narrow market subset, no
prior track record** — so they never clear the gate, and by the time their bets
resolve as wins the rebalance that would qualify them happens after they have
already stopped trading. The backtest can only discover wallets with a
*repeatable, self-demonstrated* edge (market-makers, persistent sharps).
Insiders are invisible by construction.

A wallet that trades once and quits is also literally **un-copyable** — you
cannot follow someone who is gone. So the prize is not copying *that* wallet; it
is learning the **trade-time fingerprint** of insider behavior early enough to
act on the *next* one (phase 2, a live signal). This spec covers phase 1:
**discover the cohort and extract the fingerprint**, with a built-in causal
forward-test so the result is not hindsight.

## Why naive ranking fails

"Low trade count + high edge + high PnL" as a *selection rule* selects on the
outcome. That bucket is dominated by two non-insider populations:

- **Luck**: a wallet with 2 resolved buys at price 0.10 that both win shows edge
  ≈ 0.90; at low N this is statistically indistinguishable from a coin flip.
- **Farmers**: the documented **magic/long-shot cluster** (buy YES sub-$0.05 on
  tails, modest trade counts) runs **−34.6% ROI**; the **volume-farming
  cluster** −23.8%. Low-N + longshot space is *dominated by losers*.

So discovery needs a **control group** and **trade-time-observable**
discriminators. A feature is a fingerprint only if it separates winners from
matched losers — "buys longshots" appears in *both* and is discarded.

## Goals

- Discover the hit-and-run winner cohort from the corpus (cases defined by
  realized PnL, not by a luck score — stays true to "find the winners first").
- Find **trade-time-observable** features that separate cases from matched
  controls. These features are the phase-2 detector candidates.
- Causally forward-test the surviving features on a held-out later time window,
  to confirm the fingerprint predicts wins out-of-sample (not just in-sample).
- Ship one DuckDB-backed research script, run against the desktop corpus.

## Non-goals

- No live detector in this phase (phase 2, separate spec, only if the forward-
  test shows edge).
- No change to the daemon, the copy backtest, or the ML pipeline.
- No per-wallet *proof* of skill — at N≈1 that is statistically impossible. The
  evidence lives at the **cohort** level and in **non-outcome corroborators**
  (conviction size, market-moved-after-entry), not in per-wallet significance.

## Two separate axes (not one score)

A single large bet on a fairly-priced market (0.30–0.50) that wins is the
archetypal conviction-insider, yet its luck-aware improbability is low
(z ≈ 1.0–1.5). Folding size into the luck score would discard exactly that
wallet. So size and luck are kept as **independent lenses**, and neither is the
case gate:

- **Improbability z** — one-sided test of "won more than entry prices implied."
  Expected wins under the null = `SUM(side_prob_at_entry)`; observed =
  `SUM(won)`; `z = (observed − expected) / sqrt(SUM(p·(1−p)))`. Rewards
  cheap-and-won; near-zero for fair-odds wins. A *lens*, never a gate.
- **Conviction** — `max_bet_usd` and `max_bet_usd / total_notional_usd`
  (a single bet that is most of the wallet's lifetime stake).

## Cohort definitions

All cohorts gated to the **hit-and-run shape**:

- `1 <= n_resolved_buys <= 10` (tunable `--max-trades`)
- `active_lifespan_days <= 30` (tunable `--max-lifespan-days`; last_ts − first_ts)
- `n_distinct_markets` small (reported; not gated initially)

Within that shape:

- **Cases**: top cash-PnL and positive edge (`mean(won − price) > 0`).
- **Controls**: negative/zero PnL, **matched** to cases on
  `(n_resolved_buys bucket, entry-era quarter)` so trade-count and market
  regime are held roughly constant. Match by stratified sampling to a target
  control:case ratio (default 3:1).

Cases are defined by **outcome**; the analysis value is the *contrast* with
controls of the same shape, which strips out "what all low-N wallets do."

## Per-wallet aggregate (DuckDB stage 0)

Read-only attach (`ATTACH '...' AS s (TYPE sqlite, READONLY)`), causal resolved-
buy fact table identical in spirit to `copy_selection._build_rb`
(`bs='BUY' AND ts <= resolved_at`, `won` from `outcome_yes_won` × `outcome_side`).
Per `wallet_address`:

| column | definition |
|---|---|
| `n_resolved_buys` | `COUNT(*)` over resolved buys |
| `n_distinct_markets` | `COUNT(DISTINCT condition_id)` |
| `first_ts`, `last_ts` | activity span |
| `active_lifespan_days` | `(last_ts − first_ts) / 86400` |
| `total_notional_usd` | `SUM(notional_usd)` |
| `mean_bet_usd`, `max_bet_usd` | bet-size moments |
| `mean_edge` | `AVG(won − price)` |
| `cash_pnl_usd` | `SUM(CASE won THEN notional*(1−price)/price ELSE −notional END)` |
| `mean_entry_price` | `AVG(price)` (side-normalized; see below) |
| `improbability_z` | Poisson-binomial z over `(won, side_prob)` |

`platform`-column portability handled the way `wallet_edge_leaderboard.py` does
(detect, conditionally drop the predicate). Corpus is Polymarket-only here; the
guard is for forward-compat.

## Trade-time feature set (compared cases vs controls)

All observable at the moment of the trade (the phase-2 detector inputs):

1. **improbability_z** (above).
2. **conviction**: `max_bet_usd`, `max_bet_usd / total_notional_usd`.
3. **entry-price distribution**: mean/median side-normalized entry price.
4. **market-moved-after-entry** (the key non-outcome tell): for each trade on
   market `m`, side `s`, entry price `p0` at `t0` resolving at `R`, compute the
   side-normalized implied probability of the wallet's side from *other*
   `corpus_trades` on `m` in the window `(t0, min(t0 + 7d, R − 24h)]`, and
   report `drift = windowed_side_prob − p0`. The trailing 24h before `R` is
   excluded so resolution-driven price snapping is not mistaken for foresight.
   Per wallet: mean `drift` across its trades. Early-informed buyers show large
   positive drift *before* resolution; longshot gamblers show ≈ 0 until a
   surprise. This catches the conviction-at-fair-odds insider the luck score
   misses.
5. **time-to-resolution**: `mean((R − t0)/86400)` — catalyst proximity.
6. **fresh-wallet**: is the wallet's first resolved buy ≈ its first-ever activity
   in the corpus (no prior history).
7. **category mix**: `primary_category` distribution of traded markets.

`side_prob` normalization: for a YES buy, side probability = `price`; for a NO
buy, = `1 − price`. All price-direction features are computed in the wallet's-
side probability space so a 0.20 YES buy and an 0.80 NO buy on the same market
state are treated identically.

### Discrimination report

For each numeric feature: case vs control mean/median, a standardized mean
difference (Cohen's d), and a Mann–Whitney U p-value. For categorical features:
case vs control share with a chi-square contribution. Surviving features =
those with meaningful separation (|d| and p threshold reported, not hard-coded
into a gate). Output ranked by |d| so the fingerprint is obvious at a glance.

## Causal forward-test (anti-hindsight bridge to phase 2)

Split the corpus by time at a cutoff `T` (default: 70th percentile of
`resolved_at`). Derive the fingerprint (which features separate cases/controls,
and their direction) on the **pre-`T`** cohort only. Then, on **post-`T`**
trades, score every wallet's trades at trade-time by the pre-`T` fingerprint
(a simple weighted rule or threshold on the surviving features — **no use of any
post-`T` outcome**) and measure the **forward realized edge** of the trades the
fingerprint would have flagged, vs the post-`T` base rate. Positive forward edge
here is the green light for phase 2; near-zero means the cases were luck and the
detector should not be built.

## CLI surface

`scripts/find_insider_wallets.py` (DuckDB-backed, mirrors `wallet_edge_leaderboard.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--db PATH` | `data/corpus.sqlite3` | corpus path |
| `--max-trades` | 10 | hit-and-run shape: max resolved buys |
| `--max-lifespan-days` | 30 | hit-and-run shape: max active span |
| `--control-ratio` | 3 | controls per case in the matched sample |
| `--top-n` | 100 | ranked case wallets to print |
| `--forward-cutoff-pct` | 70 | percentile of `resolved_at` for the time split |
| `--drift-window-days` | 7 | post-entry price-drift window |
| `--csv PATH` | — | write the per-wallet case/control table |

## Output

1. **Cohort summary**: case/control counts, shape gates, match ratio achieved.
2. **Discrimination report**: feature table ranked by |Cohen's d| (the fingerprint).
3. **Ranked case list**: top-N case wallets with all aggregate columns + the
   feature values, for manual inspection / cluster expansion.
4. **Forward-test result**: flagged-trade forward edge vs base rate, with n.

## Performance

DuckDB holds the heavy intermediates; Python receives small aggregated frames.
`memory_limit='3GB'`, `temp_directory='data/duckdb_spill'`, read-only attach.
Expect well under the leaderboard's ~17s full-sweep on the 16M-row corpus for
the aggregate; the market-moved-after window join is the costliest step
(per-trade self-join on `corpus_trades` keyed by `condition_id`, bounded by the
small case+control trade set, not the full corpus).

## Testing (TDD, `tests/scripts/`)

Synthetic corpus DBs via the existing `_make_corpus_db` / `corpus_factory`
fixture, hand-computed expecteds.

- **Improbability z**: single 0.05-win → high z; single 0.50-win → z ≈ 1;
  matches the hand-computed Poisson-binomial.
- **Cash PnL**: won/lost rows produce the `notional*(1−price)/price` / `−notional`
  values exactly.
- **Side normalization**: a 0.20 YES buy and an 0.80 NO buy on the same market
  yield identical side-prob and drift.
- **Shape gate**: 10 trades / 30 days included, 11 / 31 excluded.
- **Matching**: controls drawn at the requested ratio, stratified on
  count-bucket × era; degrades gracefully when too few controls exist.
- **Market-moved-after**: a market that drifts toward the wallet pre-resolution
  yields positive drift; one that snaps only in the final 24h yields ≈ 0
  (trailing-window exclusion works).
- **Forward-test no-lookahead**: fingerprint derivation uses no post-`T` row;
  scoring uses no post-`T` outcome.
- **Edge cases**: empty universe, zero controls, all-cases corpus.
- **Platform portability**: runs with and without the `platform` column.

## Follow-up (out of scope here)

- **On-chain cross-check** of the top discovered wallets via
  `check_polymarket_wallet_pnl.py` (`DataClient.get_settled_positions`,
  paginated `/v1/closed-positions`) — corpus edge ≠ on-chain cash PnL, and the
  corpus only carries resolved markets that were backfilled. Run after the
  ranked case list exists to confirm the top wallets are real on-chain winners,
  not corpus artifacts.
- **Cluster expansion** of confirmed insiders via `scripts/expand_cluster.py`
  fingerprint matching (co-trader overlap), if the cohort looks coordinated.
- **Phase 2 live detector** — only if the forward-test shows positive edge.

## Tunable parameters (for sweeps)

`--max-trades`, `--max-lifespan-days`, `--control-ratio`, `--drift-window-days`,
`--forward-cutoff-pct`. The improbability z and cash-PnL definitions are fixed.
