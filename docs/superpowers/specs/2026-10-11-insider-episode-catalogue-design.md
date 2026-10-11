# Insider-episode catalogue: documented cases, blind re-identification

Date: 2026-10-11
Status: Design — approved in conversation, pending written review
Related: `scripts/find_insider_wallets.py` (#209),
`docs/superpowers/specs/2026-06-01-insider-wallet-discovery-design.md`,
`scripts/expand_cluster.py` (#210), `docs/claude/wallet-clusters.md`

## Problem

The June 2026 insider screen (#209) defined its cases by outcome and trade
count: wallets with at most 10 resolved buys and a 30-day lifespan that ended
with positive PnL. That produced 395,937 "cases" out of 643,907 shape-matching
wallets, a cohort dominated by variance. Its strongest separators (entry price,
improbability) restated the outcome. The one real find, the Khamenei thesis
wallet `0xbad457dc`, had 262 trades on a single market over five weeks and
would never have passed the shape gate. Wallet-centric, outcome-labelled
discovery does not find insiders.

Publicly documented Polymarket insider episodes were found a different way:
start from a **market** whose resolution hinged on a discrete news event, look
at who was positioned just **before** the news, and check **wallet provenance**
(age, funding, prior history). This spec rebuilds the investigation on that
footing, using the documented cases as ground truth and holding part of them
out so the result is falsifiable.

## Goals

1. **Catalogue** every publicly documented Polymarket insider episode that the
   corpus can see, with the news timestamp, the market(s), and any published
   wallet addresses, in a committed registry.
2. **Blind re-identification.** On held-out cases with published addresses, a
   scorer that never sees those addresses must rank them near the top of the
   pre-news winning-side buyers.
3. **Fingerprint characterization.** For every analyzable case, describe how the
   pre-news window differs from matched control markets, feature by feature.
4. **Anchor calibration.** Measure how far an automatically detected price-move
   anchor lands from the hand-pinned news timestamp, to decide whether the
   structural-proxy follow-up can run without manual timestamps.
5. **Coverage audit.** Record every published wallet the corpus cannot see and
   why.

## Non-goals

- No live detector. That is a later phase, gated on goal 2 succeeding.
- No Polygon chain data (funding source, USDC deposit timing). That is a
  shortlist-only follow-up after this phase.
- No structural-proxy scan over undocumented markets. That is the next phase
  and consumes this phase's CSV output.
- No changes to the daemon, the copy backtest, or the ML pipeline.
- No per-wallet proof of insider status. The evidence is a rank on a blind
  test plus a contrast with controls, not an accusation.

## Decisions made in brainstorming

| Question | Decision |
|---|---|
| Primary outcome | Historical catalogue first; identification is the natural expansion |
| Ground truth | Publicly documented cases. Structural proxies deferred to a follow-up |
| Enrichment depth | Polymarket data-api only (wallet age, activity). Polygon deferred |
| Success criterion | Both: blind re-identification on holdout + fingerprint vs controls |
| Anchor method | Hand-pinned news timestamp as primary; automatic price-jump anchor run alongside |
| Feature design | Derived from an exploratory profile of dev-split insider wallets, not from intuition |
| Published wallets missing from corpus | Reported in a dedicated table with a diagnosed reason, never scored as a miss |

## The case registry

`docs/research/insider-cases.yaml`, hand-curated, version-controlled. It is the
only place hindsight enters the system.

```yaml
split_rule: "sha256(name) parity, stratified on has_wallets; assigned at commit <hash>"
cases:
  - name: maduro-capture-2026-01
    confidence: alleged          # confirmed | alleged | suspected
    split: holdout               # dev | holdout; assigned once, never edited
    markets:
      - slug: maduro-out-by-january-31-2026-318
        condition_id: "0x..."    # resolved by the loader if omitted
    winning_side: YES
    news_ts_utc: "2026-01-03T06:40:00Z"
    news_ts_precision: minute    # minute | hour | day
    news_source: https://...
    suspicious_window: "bets placed 6-30 hours before the strike"
    wallets:                     # hidden from the scorer; read only by the comparator
      - address: "0x..."
        source: https://...
        reported_stake_usd: 32000
        reported_profit_usd: 400000
    detection_method: fresh wallet, single market, exchange-funded
    sources:
      - url: https://...
        published: 2026-01-05
    notes: >
      Free text. How the case was originally spotted, Polymarket's response,
      later linkage to other wallets.
    analyzable: true             # set by the loader: market present in corpus
```

Rules:

- `split` is assigned by a deterministic rule recorded in the file, stratified
  so both halves contain cases with published wallets. It is assigned before
  any wallet is examined and never changed afterwards. The commit hash at
  assignment is recorded.
- Holdout cases are not opened, queried, or inspected until the scorer is
  frozen in the spec amendment (see "Scorer").
- Cases whose markets fall below the corpus volume floor stay in the file with
  `analyzable: false`, so the coverage audit can count them.
- The first version is compiled by web research covering November 2022 to May
  2026 and vetted by the operator before any analysis runs. Target 10-20
  cases.

## Pipeline

One new script, `scripts/insider_episodes.py`, DuckDB-backed via read-only
`ATTACH`, with the pure logic in `scripts/insider_episode_lib.py` so it is
testable without a corpus. Five stages with plain-data interfaces.

### 1. Case loader

Reads the registry, resolves slugs to `condition_id` against `corpus_markets`,
sets `analyzable`, and splits each case into two records: the **public
record** (markets, side, anchors) handed to every downstream stage, and the
**sealed record** (published wallets) handed only to the comparator. No other
stage receives the sealed record.

### 2. Episode extractor

Input: one `condition_id`, one anchor timestamp, one window length (default
72 h, flag `--window-hours`). Output: every `corpus_trades` row in
`(anchor − window, anchor]`, plus the market's full trade-derived price path.

Two anchors per case:

- **News anchor**: `news_ts_utc` from the registry.
- **Auto anchor**: the start of the largest sustained move toward the winning
  side before resolution. Defined on the side-normalized trade price series
  resampled to 1-hour buckets (volume-weighted mean): the earliest bucket `b`
  such that the move from `price[b]` to `max(price[b .. b+24h])` is the largest
  such 24-hour forward move in the market's life, with a minimum move of 0.10.
  Returns none if no bucket clears the minimum. Computed from prices alone.

Controls get only the auto anchor.

### 3. Provenance enricher

For every distinct wallet that bought the winning side inside a window (cases
and controls), fetch from the Polymarket data-api:

- `first_activity_ts` via `DataClient.get_first_activity_timestamp`
- lifetime trade count and distinct-market count from paginated
  `DataClient.get_activity` (newest-first, page cap as in `expand_cluster.py`)

Results are cached in `data/insider_provenance.sqlite3`, table
`wallet_provenance(wallet PRIMARY KEY, first_activity_ts, n_trades,
n_markets, fetched_at, status)`. A failed fetch writes no row (no partial
rows), so re-runs retry it. Requests are rate-limited and tolerate 408/5xx the
way #210 made `expand_cluster.py` tolerate them. The cache is desktop-local and
not committed.

### 4. Scorer

Pure function: one wallet's window trades + provenance + window context →
feature vector + composite rank. **Its feature set is deliberately not fixed
in this spec.** It is derived from the exploratory profile (below) and written
into this document as an amendment, with operator sign-off, before
implementation. The scorer receives no field derived from the sealed record;
a test enforces this.

Feature families the profile will examine (the amendment selects and weights
from these, and may add others the profile reveals):

- Provenance: wallet age at first window trade, lifetime trade count, lifetime
  distinct markets, whether the window contains the wallet's first-ever trade.
- Concentration: window notional as a share of the wallet's lifetime notional;
  share of the wallet's window notional on this one market.
- Size relative to context: wallet's window notional as a share of total
  winning-side window volume; rank among window buyers.
- Timing: hours before anchor of first and last buy; whether buying continued
  across multiple sessions or was a single burst.
- Price: side-normalized mean entry price; price at first buy relative to the
  window's opening price.
- Post-entry drift (from #209): move toward the wallet's side after entry,
  excluding the final 24 h before resolution.

### 5. Reporter

Writes:

- `docs/research/insider-episodes/<case-name>.md`: timeline, both anchors and
  their gap, top 25 ranked wallets with features, control comparison, and the
  blind-test result when the case is holdout.
- `docs/research/insider-episodes/SUMMARY.md`: cross-case fingerprint table,
  blind-test median rank, anchor-gap distribution, and the
  **published-wallets-not-in-corpus** table.
- `docs/research/insider-episodes/scored_wallets.csv`: every scored wallet
  across cases and controls, with features, anchor type, and
  `is_control`. Input for the structural-proxy follow-up.

Data flow is strictly forward. Resolution outcomes are used only to pick the
winning side. The auto anchor sees prices only. The comparator is the only
consumer of the sealed record.

## Controls

Per case, three to five control markets drawn automatically from
`corpus_markets` ∩ `market_resolutions`:

- same `category`
- `total_volume_usd` within a factor of 3 of the case market
- `resolved_at` within 180 days of the case market
- same winning side
- not in the registry (any case, any split)

Selection is deterministic (ordered by volume distance, then slug). Fewer than
three matches is reported, not an error. Controls answer "what does a normal
pre-move window look like"; a case feature counts as a fingerprint only when
the case's value lies outside the control distribution (reported as the
case's percentile within the pooled control wallets, and as Cohen's d for
window-level aggregates).

## Blind-test protocol

For each **holdout** case with at least one published wallet:

1. The scorer ranks every winning-side buyer in the window, without the sealed
   record.
2. The comparator looks up each published wallet's rank and percentile and
   whether it is in the top 5, 10, and 25.
3. A case **passes** if at least one published wallet ranks in the top 10.
4. The headline metric is the **median best-rank** across holdout cases,
   always reported next to the number of candidates per window.
5. Steps 1-4 are repeated under the auto anchor to measure the cost of
   automation.

Exclusions and edge cases:

- A published wallet with no trades in the corpus window is reported in the
  not-in-corpus table with a diagnosed reason (see below) and is **not** a
  miss.
- A case with fewer than 10 winning-side buyers in the window is reported but
  excluded from the median.
- Dev cases are run through the same path and reported, labelled in-sample.
- If fewer than three holdout cases are testable, the report falls back to
  leave-one-out across all testable cases and says so prominently.

## Coverage audit: published wallets not in corpus

For every published wallet in the registry (both splits, since this is
descriptive), the script determines one of:

| reason | how it is determined |
|---|---|
| `market_below_volume_floor` | case market absent from `corpus_markets` |
| `no_trades_in_market` | wallet has no `corpus_trades` on the market at all |
| `outside_window` | wallet traded the market but not inside the news window; report its first/last trade offset from the anchor |
| `below_notional_floor` | wallet's reported stake is under the corpus's $10 per-trade floor (inferred; flagged as inference) |
| `backfill_truncated` | `corpus_markets.truncated_at_offset_cap = 1` for the market |

This table is a result in its own right: it measures how much known insider
activity the corpus can see.

## Exploratory profile (precedes the scorer)

A throwaway probe, not part of the script, run only on **dev** cases:

- Pull each published dev wallet's complete corpus trade history and its
  data-api activity.
- Pull 5-10 random winning-side buyers from the same window as a crowd
  baseline.
- Lay them side by side on every feature family listed under "Scorer".
- Output: `docs/research/insider-episodes/dev-profile.md`, a short document of
  what the insiders actually looked like versus the crowd. No modelling.

The scorer amendment is written from this document and signed off before
implementation begins. If the profile shows the feature families above miss
what distinguishes the dev wallets, the amendment adds what it needs.

## CLI surface

`scripts/insider_episodes.py`

| Flag | Default | Meaning |
|---|---|---|
| `--db PATH` | `data/corpus.sqlite3` | corpus path |
| `--registry PATH` | `docs/research/insider-cases.yaml` | case registry |
| `--out-dir PATH` | `docs/research/insider-episodes` | report directory |
| `--window-hours` | 72 | pre-anchor window |
| `--controls-per-case` | 5 | target control count |
| `--split {dev,holdout,all}` | `dev` | which cases to run; `holdout` refuses to run until `--scorer-frozen` is passed |
| `--scorer-frozen` | off | operator attestation that the scorer amendment is signed off |
| `--skip-provenance` | off | run with cached provenance only (offline) |
| `--provenance-rpm` | 60 | data-api rate limit |

## Testing

`tests/scripts/test_insider_episode_lib.py` and
`tests/scripts/test_insider_episodes.py`, using the synthetic-corpus fixtures
already in `tests/scripts/conftest.py`. Behaviour, not implementation:

- **Scorer / comparator**: hand-computed ranks and percentiles; a published
  wallet ranked 1 of 12 passes, 11 of 12 fails; ties resolved
  deterministically.
- **Blind isolation**: the scorer's input dataclass has no field that can hold
  the sealed record; a test constructs a case with published wallets and
  asserts the scorer's inputs are identical whether or not the sealed record
  is populated.
- **Episode extractor**: inclusive/exclusive window boundaries; winning-side
  filter; a wallet with buys on both sides contributes only its winning-side
  buys.
- **Auto anchor**: synthetic path with one clear jump returns the bucket
  before it; a flat path returns none; a jump inside the final 24 h before
  resolution is still eligible (the exclusion applies to drift, not anchors).
- **Control matcher**: respects category, volume band, era, side; excludes
  registry markets of either split; returns fewer than requested without
  error; deterministic order.
- **Coverage audit**: each of the five reasons is triggered by a fixture.
- **Provenance cache**: hit returns cached row with no API call; miss fetches
  and stores; API failure stores nothing.
- **Registry loader**: refuses a holdout run without `--scorer-frozen`; splits
  the record into public and sealed parts; flags a slug absent from the
  corpus as not analyzable.
- Corpus fixtures use notional ≥ $10 (the repo's insert floor).

## Performance

Per-case extraction touches one market's trades through
`idx_corpus_trades_market_ts`; fast. Provenance is the bottleneck: one
`first_activity` call plus a few `activity` pages per distinct window buyer,
across roughly 15 cases × 6 markets (case + controls), likely 2,000-5,000
wallets, at 60 rpm. The cache makes every re-run free. Runs on the desktop
against the 73 GB corpus; DuckDB `memory_limit` set to 8 GB per the #209
follow-up.

## Sequencing

1. Registry committed from the research draft, split assigned, operator vets.
2. Exploratory profile on dev cases → `dev-profile.md`.
3. Scorer amendment appended to this spec, operator signs off.
4. Implementation plan (writing-plans), then the script, dev run, then the
   holdout run with `--scorer-frozen`.

## Follow-ups (out of scope)

- **Structural-proxy scan**: apply the frozen scorer with the auto anchor to
  every corpus market whose resolution hinged on a discrete announcement,
  seeded from `scored_wallets.csv` and the anchor-gap result.
- **Polygon enrichment**: USDC deposit timing and funding-source address for
  the top-ranked wallets only; enables funding-source clustering.
- **Live detector**: only if the holdout blind test passes.

## Amendment 1: scorer (2026-10-11, signed off by operator)

Derived from `docs/research/insider-episodes/dev-profile.md`. Freezes the
scorer before any holdout case is opened.

### Inputs per candidate wallet

A candidate is any wallet with at least one BUY of the winning outcome token
on the case's **primary market** inside `(anchor − window, anchor]`.

| symbol | definition | source |
|---|---|---|
| `N_w` | sum of winning-side BUY notional (USD) in the window on the primary market | corpus |
| `age_d` | days from the wallet's first activity to its first window buy; first activity = data-api `first_activity_ts`, falling back to the wallet's first `corpus_trades` row when the API has no answer | data-api, corpus |
| `B` | lifetime distinct `condition_id`s traded, from the corpus, counted up to the anchor | corpus |
| `N_life` | lifetime corpus notional up to the anchor | corpus |
| `flip` | 1 if the wallet SOLD the opposing outcome token on the primary market inside the window, at a time earlier than its first winning-side buy in the window; else 0 | corpus |

Lifetime quantities are computed with `ts <= anchor` so the scorer sees
nothing after the anchor.

### Score

```
fresh(age_d) = 1                          if age_d <= 10
             = max(0.1, 1 - (age_d - 10)/60)   otherwise   (floors at 0.1 at 64+ days)
conc         = N_w / N_life                (in (0, 1]; 1 when the window is the wallet's whole life)
score        = log10(1 + N_w) * fresh(age_d) * (0.5 + 0.5*conc) / log10(2 + B) + 0.5 * flip
```

Rationale, from the profile: notional in log space so clip-accumulators and
single-order bettors compare on scale, not count; freshness as a multiplier
that leaves wallets under ten days untouched and decays rather than cuts, so
the nine-day-old dev wallet is not thrown away; concentration blended at half
weight so a fresh wallet whose whole life is this window scores up to 2×
a diversified one; breadth in the denominator so market-makers with hundreds
of markets fall below fresh single-market wallets at equal notional; the flip
bonus is additive and sized so that, among wallets of similar notional, a
flipper ranks above non-flippers but a $200 flipper does not outrank a
$50,000 fresh buyer.

### Ranking and ties

Candidates are ordered by `score` descending, then by `N_w` descending, then
by earliest first window buy, then by wallet address. The comparator's
published-wallet rank is the 1-based position in this order.

### Excluded on purpose

Entry price, post-entry drift, single-bet size, conviction fraction, and
anything observed after the anchor. Secondary markets are reported in the
per-case report with the same per-wallet features but are never pooled into
the primary-market rank.

### No tuning after this point

The constants above (10-day plateau, 60-day decay, 0.1 floor, 0.5 blend,
0.5 flip bonus) are fixed. The holdout run uses them unchanged. Any later
change is a new amendment and invalidates the holdout result.
