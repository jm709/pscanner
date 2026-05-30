# Causal wallet selection for the copy-trading backtest

Date: 2026-05-30
Status: Design — pending implementation plan
Related: `scripts/backtest_copy_sizing.py` (#203, #204), `scripts/wallet_edge_leaderboard.py`,
`docs/superpowers/specs/2026-05-28-backtest-copy-sizing-design.md`

## Problem

The current `backtest_copy_sizing.py` backtests a **fixed watchlist** of wallets. The
2026-05-30 run loaded the 1,790 wallets surfaced by `wallet_edge_leaderboard.py` and
copied each from its **first** trade. Two flaws make those results un-deployable:

1. **In-sample / circular.** The wallets were selected by their *full-corpus* edge, then
   copied over that same corpus. The headline +30–49% ROI is a restatement of "we picked
   the winners," not a forward-looking estimate.
2. **Copy-from-trade-1.** At a wallet's first trade its edge is unknown, yet every sizing
   scheme copied it immediately. Combined with the capacity gates (#204), >97% of signals
   were skipped because 1,790 wallets out-signal a $10K bankroll by a wide margin.

This design adds a **causal qualify → rank → select** layer so the backtest resembles live
performance: a wallet is only copied after it has proven an edge over its *own past*
trades, and only the current best wallets are copied, sized to what the bankroll can fund.

## Goals

- Decide *causally* (no lookahead) which wallets to copy at each point in time.
- Never copy a wallet before it has `min_resolved` resolved trades (applies to all four
  sizing schemes uniformly).
- Copy only the globally top-K wallets by causal edge, with K tunable, re-evaluated on a
  cadence. Positive causal edge is a hard floor.
- Keep the four sizing schemes and the #204 capacity gates unchanged underneath.
- Run in minutes / sub-GB on the 18.8M-row corpus.

## Non-goals

- No change to the daemon's live trader (its constant-bankroll design is intentional).
- No selling / mid-position exits — positions are held to resolution, as today.
- No NAV-driven sizing (bankroll stays constant; out of scope per #204).
- The existing watchlist mode is retained, not replaced — it answers a different question
  ("backtest this specific set"). The new mode is additive, gated behind `--causal-select`.

## Architecture: three-layer copy pipeline

```
corpus (all wallets with >=20 lifetime resolved BUY trades)
        │
        ▼
┌──────────────────────────────┐
│ 1. Selection  (DuckDB)       │  WHO to copy
│    qualify -> rank -> freeze  │  → a (boundary_ts, wallet) copy-set table
└──────────────┬───────────────┘
               │ selected trades + their resolutions, ts-ordered
               ▼
┌──────────────────────────────┐
│ 2. SizingScheme  (Python)    │  HOW MUCH  (4 schemes, unchanged)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 3. CapacityGate  (Python)    │  CAN we afford it  (#204, unchanged)
└──────────────────────────────┘
```

The selector is **scheme-independent** (a wallet's edge is a property of its own trades),
so one selection feeds all four schemes identically. Idea #1 ("don't copy from trade 1")
is enforced once, in layer 1, for every scheme.

## Layer 1 — DuckDB selection precompute

All of qualification, edge, ranking, and the top-K freeze are window/aggregate SQL. Python
never holds per-wallet history or an all-wallet pending book. Measured against the corpus:
universe build + per-boundary edge (lifetime) ~36s, rolling-window ~11s, copy-set →
**~146K selected trades** (a ~100× reduction from 15.5M resolved-buys).

### 1a. Resolved-buy fact table (causal)

```sql
CREATE TABLE rb AS
SELECT t.wallet_address AS wallet, t.condition_id, t.price, t.ts,
       t.outcome_side, r.resolved_at,
       CASE WHEN (r.outcome_yes_won = 1 AND t.outcome_side = 'YES')
              OR (r.outcome_yes_won = 0 AND t.outcome_side = 'NO')
            THEN 1 ELSE 0 END AS won
FROM s.corpus_trades t
JOIN s.market_resolutions r ON r.condition_id = t.condition_id
WHERE t.bs = 'BUY' AND t.ts <= r.resolved_at;   -- no-lookahead guard #1
```

Universe = wallets with `COUNT(*) >= min_resolved` rows in `rb` (lifetime). This is a
count filter (not outcome-based) and a necessary condition for qualifying under any window,
so it never drops a wallet that could later qualify.

### 1b. Per-boundary edge + qualification + positive-edge floor

```sql
CREATE TABLE rebalance AS
  SELECT range AS boundary_ts FROM range(:lo, :hi, :rebalance_days * 86400);

CREATE TABLE wallet_edge_at_boundary AS
SELECT b.boundary_ts, rb.wallet,
       COUNT(*)               AS n_resolved,
       AVG(rb.won - rb.price) AS edge
FROM rebalance b
JOIN rb
  ON rb.resolved_at < b.boundary_ts                                  -- no-lookahead guard #2
 AND (:edge_window = 0
      OR rb.resolved_at >= b.boundary_ts - :edge_window * 86400)     -- rolling window
GROUP BY b.boundary_ts, rb.wallet
HAVING COUNT(*) >= :min_resolved                                     -- qualification
   AND AVG(rb.won - rb.price) > 0;                                   -- positive-edge floor
```

The `edge_window` rolling variant is a double-bounded range join (no per-wallet history
needed). Qualification (`COUNT >= min_resolved`) is measured **within the same window** as
the edge, so a wallet is never ranked on fewer than `min_resolved` samples, and a wallet
with 19 recent + 50 old trades does not qualify under a rolling window.

### 1c. Top-K freeze

```sql
CREATE TABLE copyset AS
SELECT boundary_ts, wallet FROM (
  SELECT boundary_ts, wallet, edge, n_resolved,
         ROW_NUMBER() OVER (PARTITION BY boundary_ts
                            ORDER BY edge DESC, wallet ASC) AS rk   -- deterministic tiebreak
  FROM wallet_edge_at_boundary)
WHERE rk <= :k_for_boundary;
```

`k_for_boundary` per policy (mutually exclusive; bankroll is constant, so all three are
SQL-expressible):
- `--copy-top-k N`        → `k = N`
- `--copy-capital-per-wallet C` → `k = floor(bankroll / C)`
- `--copy-top-frac X`     → `k = ceil(X * count(qualified at that boundary))`

Because of the positive-edge floor, **positive edge is a hard floor and K is an upper
bound**: if fewer than K wallets have positive causal edge at a boundary, only those are
copied — never a negative-edge wallet to fill a quota.

### 1d. Selected-trade stream (the only rows Python sees)

```sql
SELECT t.*
FROM s.corpus_trades t
JOIN copyset cs
  ON cs.wallet = t.wallet_address
 AND t.ts >= cs.boundary_ts
 AND t.ts <  cs.boundary_ts + :rebalance_days * 86400     -- this period's frozen set
WHERE t.bs = 'BUY'
ORDER BY t.ts;
```

Freeze membership gates on **trade ts**, not resolution time: a wallet in period B's set is
copied for *all* its trades in `[B, B+N)`, including ones on markets resolving before
`B+N`. Resolutions for these markets are unioned into the stream (as today) so the Simulator
can close positions.

## Layer 2/3 — what stays in Python (unchanged)

The `Simulator` event-walk over the selected stream, the four `SizingScheme`s, and the #204
capacity gates (`_can_open`, `open_cost`, `skipped_trades`, `cumulative_pnl`) are inherently
sequential and stay exactly as they are. The new mode only changes *which* events enter the
walk and adds a "not selected" accounting at the report layer.

## Correctness guards

1. **No lookahead (load-bearing, two guards):** `t.ts <= r.resolved_at` when building `rb`,
   **and** `rb.resolved_at < boundary_ts` when computing each boundary's edge. Both required.
2. **Positive-edge floor is causal:** the `> 0` test is on the per-boundary causal edge, not
   a lifetime/final edge — a hot-then-cold wallet is still copied during its positive periods.
3. **Freeze semantics:** selection membership keys on trade ts and the period's `boundary_ts`,
   never on `resolved_at`.
4. **Deterministic tiebreak:** `ORDER BY edge DESC, wallet ASC` so runs reproduce.
5. **Platform-column portability:** the corpus may or may not carry a `platform` column
   (the laptop's 24.6 GB corpus does not; the desktop's 69 GB does). The new mode detects it
   the way `wallet_edge_leaderboard.py` does (`_has_platform_column`) and conditionally drops
   the `platform = ?` predicate, rather than hardcoding it.

## CLI surface

| Flag | Default | Meaning |
|---|---|---|
| `--causal-select` | off | activate qualify→rank→freeze; ignores `--watchlist-db` |
| `--min-resolved` | 20 | qualification threshold (resolved trades within the edge window) |
| `--edge-window` | 0 | rolling edge window in days; `0` = lifetime |
| `--rebalance-days` | 14 | cadence for recomputing/freezing the top-K set |
| `--copy-top-k` | 25 (default policy) | copy-set policy: fixed N wallets |
| `--copy-capital-per-wallet` | — | copy-set policy: K = floor(bankroll / C) |
| `--copy-top-frac` | — | copy-set policy: top X% of qualified |

The three `--copy-*` flags are a mutually-exclusive group; if none is given with
`--causal-select`, default to `--copy-top-k 25`. All existing flags (`--enforce-capacity`,
`--max-open-exposure-usd/-frac`, scheme params, `--start-ts/--end-ts`, `--csv`, `--db`)
compose unchanged.

## Report additions

- A header block: universe size, # rebalances, K policy/value, `edge_window`, `rebalance_days`,
  and the qualified-wallet count range across boundaries.
- A new **"Not selected"** counter per scheme in the Headline table, distinct from the
  capacity **"Skipped"** counter, so the selector's filtering is visible separately from the
  capacity gate's.
- Headline / Risk / Quarterly / Top-contributors tables work as-is on the booked subset.

## Performance

DuckDB holds the heavy intermediates; Python gets a ~146K-row sorted result set.
- Read-only attach: `ATTACH '...' AS s (TYPE sqlite, READONLY)`.
- `memory_limit='3GB'`, `temp_directory='data/duckdb_spill'`.
- Stream the selected trades with `fetchmany(100_000)` (never `fetchall` of the raw stream).
- `__slots__` on `OpenPos` / `BacktestState` for free per-object savings.
- Materialize `rb` once and reuse across boundary queries.
- Expected end-to-end: **~3–4 min wall, <500 MB Python RSS** on the 18.8M-row corpus.

## Testing (TDD, `tests/scripts/`)

Tiny synthetic corpus DBs via the existing `_make_corpus_db` helper, hand-computed expecteds.

- **No-lookahead (highest value):** a wallet whose strong trades resolve *after* boundary B
  is not qualified/ranked on them at B.
- **Cross-validation:** DuckDB per-boundary edge vs a brute-force pure-Python recomputation
  agree on every `(boundary, wallet)`.
- **Qualification gate:** 19 → excluded, 20 → included; counted *within the window*.
- **Positive-edge floor:** a qualified but non-positive-edge wallet is never selected; K is an
  upper bound when fewer than K wallets are positive.
- **Lifetime vs rolling:** same wallet, different `--edge-window`, different result.
- **Top-K freeze + tiebreak:** only top-K; deterministic ties; membership frozen across a period.
- **Three K-modes:** fixed-k, capital-per-wallet, top-frac.
- **Freeze membership:** copied for all trades in `[B, B+N)`, incl. markets resolving before `B+N`.
- **Platform detection:** precompute runs on a corpus with and without the `platform` column.
- **Integration:** full `--causal-select` run → only selected wallets booked; sizing + #204
  gates compose; "Not selected" counter correct.
- **Edge cases:** empty universe, zero qualified, K > qualified count, `--start-ts/--end-ts`.

## Tunable parameters (for backtest sweeps)

`--min-resolved`, `--edge-window`, `--rebalance-days`, and the K policy/value are all knobs
to sweep. The positive-edge floor is fixed (not a knob) — copying negative-edge wallets is
never desired.
