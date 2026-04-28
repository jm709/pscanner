# Paper-trading expansion (multi-signal `SignalEvaluator`) — Design

**Date:** 2026-04-27
**Status:** approved, awaiting implementation plan
**Prior work:** initial paper-trading lives in `src/pscanner/strategies/paper_trader.py`
(spec `docs/superpowers/specs/2026-04-27-paper-trading-design.md`).

## Motivation

Paper-trading today only books trades from `smart_money` alerts (~240/day). The
2h smoke surfaced four other detector streams that emit signal during the same
run window:

| detector | alerts in 2h | tradeable from body? |
|---|---:|---|
| smart_money | 20 | yes (current baseline) |
| move_attribution | 11 | yes (`condition_id` + `outcome` + `side`) |
| velocity | 294 | partially (signed `change_pct`, no explicit side; needs follow/fade rule) |
| mispricing | 73 | no — requires detector enrichment |
| convergence | 8 | no — meta-signal |
| whales | 0 | no — `side` semantics ambiguous, low frequency |
| cluster | 1 | no — meta-signal |

We want to (1) book trades from `move_attribution`, `velocity`, and `mispricing`
in addition to `smart_money`, (2) handle velocity's directional ambiguity by
running both follow and fade simultaneously so we can learn which works where,
(3) enrich the mispricing detector to make it tradeable, (4) build the
architecture so adding a future signal source is one new class plus one config
block, and (5) **maximize data collection by decoupling bet sizing from
realized PnL** — bets stay constant size regardless of cumulative losses, and
trades keep booking even when the cost-basis NAV goes negative. This is a
research configuration; no real bankroll constraint is modeled. The goal is
labeled-outcome data per `(source, rule_variant)` pair, not realistic PnL
simulation.

## Goal

Refactor `PaperTrader` to a thin orchestrator that fans alerts to per-detector
`SignalEvaluator` classes. Ship four evaluators (smart_money, move_attribution,
velocity, mispricing). Enrich the mispricing detector. Add per-source PnL
tracking columns and CLI breakdown.

## Non-goals

- Pacing or back-pressure changes — the recently-shipped `WorkerSink` already
  decouples the alert hot path; this spec adds new alert sources but does not
  alter the emission pipeline.
- Whales / convergence / cluster / move_attribution-as-meta — out of scope per
  research findings.
- Position-size adaptation from realized PnL — the architecture supports it
  (size lives on the Evaluator), but v1 uses static per-source fractions.
- Aggregate open-exposure cap — deferred. Worth revisiting once we observe
  resolution cadence in production.
- Convergence as a signal — out of scope per research findings (no direction
  in body); revisit only with detector enrichment.

## Architecture

`PaperTrader.evaluate(alert)` today hardcodes the smart-money body shape: parse
`(wallet, condition_id, side)`, look up `weighted_edge`, resolve outcome, size
by a single `position_fraction`, insert one entry. Every step is smart_money
specific. Sizing reads `compute_cost_basis_nav` and gates on `nav <= 0`.

The expansion pluralizes the pipeline AND switches to constant sizing. We
introduce a `SignalEvaluator` Protocol with four methods:

```python
def accepts(alert: Alert) -> bool
def parse(alert: Alert) -> list[ParsedSignal]
def quality_passes(parsed: ParsedSignal) -> bool
def size(bankroll: float, parsed: ParsedSignal) -> float
```

The `size` method receives `starting_bankroll_usd` (a config constant), not
the running cost-basis NAV. Bet size is therefore constant across the run:
$10 (or $2.50 per velocity side) regardless of cumulative wins or losses.
The `if nav <= 0` gate is removed — trades keep booking even if realized PnL
goes deeply negative. We still compute cost-basis NAV once per alert and write
it to `paper_trades.nav_after_usd` for analysis (it can be negative; that's
informative, not blocking).

Four concrete implementations live in their own files:

- `SmartMoneyEvaluator`
- `MoveAttributionEvaluator`
- `VelocityEvaluator` — `parse` returns **two** `ParsedSignal`s
  (`rule_variant="follow"`, `rule_variant="fade"`)
- `MispricingEvaluator`

`PaperTrader` becomes a thin orchestrator: on each alert it walks
`self._evaluators`, picks the first whose `accepts(alert)` returns True, runs
the pipeline, and inserts one paper_trades row per `ParsedSignal`.

Evaluator construction is gated by per-source config flags
(`paper_trading.evaluators.<name>.enabled`); a disabled source is simply not
constructed.

Three properties fall out of this architecture:

1. **Adding a future signal source** is one new class + one config block —
   zero changes to PaperTrader.
2. **Disabling a misbehaving source** is a one-config-bool flip; no redeploy.
3. **Per-rule analysis** (`SELECT triggering_alert_detector, rule_variant,
   AVG(...) GROUP BY 1, 2`) is a single SQL query against the new columns.

## Components

### New: `src/pscanner/strategies/evaluators/protocol.py` (~30 LOC)

`SignalEvaluator` Protocol with the four methods above. Plus `ParsedSignal`
dataclass:

```python
@dataclass(frozen=True, slots=True)
class ParsedSignal:
    condition_id: ConditionId
    asset_id_or_outcome_name: str  # outcome name or asset_id, depending on detector
    side: str  # outcome name as written in market_cache; used for cache lookup
    fill_price_hint: float | None  # used to skip `_resolve_outcome` when set
    rule_variant: str | None  # "follow"/"fade" for velocity; None otherwise
    metadata: dict[str, Any]  # pass-through for downstream logs
```

### New: `src/pscanner/strategies/evaluators/smart_money.py` (~80 LOC)

Moves today's parse/qualify/size logic out of `PaperTrader` unchanged.
- `accepts`: `alert.detector == "smart_money"`.
- `parse`: extracts `(wallet, condition_id, side)`. Returns 1 ParsedSignal.
- `quality_passes`: `tracked_wallets[wallet].weighted_edge >
  config.min_weighted_edge` (default 0.0).
- `size`: `bankroll * config.position_fraction` (default 0.01).

### New: `src/pscanner/strategies/evaluators/move_attribution.py` (~60 LOC)

- `accepts`: `alert.detector == "move_attribution"`.
- `parse`: extracts `condition_id` + `outcome` (the actual outcome name —
  e.g., "Anastasia Potapova"). The `side` field in the body ("BUY"/"SELL")
  is a taker-action modifier, not part of the cache lookup; ignore it.
  Returns 1 ParsedSignal where `side = outcome`.
- `quality_passes`: `severity ∈ {med, high}` AND `n_wallets >=
  config.min_wallets` (default 3).
- `size`: `bankroll * config.position_fraction` (default 0.01).

### New: `src/pscanner/strategies/evaluators/velocity.py` (~90 LOC)

- `accepts`: `alert.detector == "velocity"`.
- `quality_passes`: `severity == "high"` AND (NOT `consolidation`).
- `parse`:
  - Resolve the alert's `asset_id` to its outcome name and the **opposing**
    outcome via `market_cache.get_by_condition_id`. Return one ParsedSignal
    with `rule_variant="follow"` and `side=alert_outcome` and one with
    `rule_variant="fade"` and `side=opposing_outcome`.
  - If market_cache misses (no opposing-side asset_id known), return only
    the follow side.
- `size`: `bankroll * config.position_fraction` per ParsedSignal (default
  0.0025 per side, so a pair totals 0.5% of starting bankroll = $5 on
  $1000).

### New: `src/pscanner/strategies/evaluators/mispricing.py` (~60 LOC)

- `accepts`: `alert.detector == "mispricing"`.
- `parse`: reads the enriched fields `target_condition_id`, `target_side`,
  `target_current_price`, `target_fair_price` from the body. Returns 1
  ParsedSignal. If any field is missing (legacy alert), returns `[]` and
  logs at debug.
- `quality_passes`: `abs(target_fair_price - target_current_price) >=
  config.min_edge_dollars` (default 0.05).
- `size`: `bankroll * config.position_fraction` (default 0.01).

### New: `src/pscanner/strategies/evaluators/__init__.py` (~10 LOC)

Re-exports the five public names.

### Modified: `src/pscanner/strategies/paper_trader.py` (~ -80 / +50 LOC net)

- Strip out source-specific logic from `evaluate`. New `__init__` accepts
  `evaluators: list[SignalEvaluator]` and removes the smart-money-specific
  fields (those move into `SmartMoneyEvaluator`'s ctor).
- New `evaluate` body: iterate `self._evaluators`, find first acceptor, run
  the pipeline, insert one paper_trades row per ParsedSignal.
- Remove the `if alert.detector != "smart_money"` early-return; replaced by
  the evaluator loop.
- Failure isolation: per-evaluator pipeline wrapped in `try/except Exception`,
  logging `paper_trader.evaluator_failed` with `detector`, `evaluator`,
  `alert_key`, `exc_info=True`. CancelledError re-raised.

### Modified: `src/pscanner/detectors/mispricing.py` (~+30 LOC)

In `_build_alert`, after computing `deviation`:

1. Compute proportional fair prices: for each market in `event.markets`,
   `fair_yes_price[i] = current_yes_price[i] / sum_of_yes_prices`.
2. Pick the market with the largest absolute `current - fair` deviation.
3. If `current > fair`, the YES leg is over-priced → `target_side="NO"`,
   with current/fair flipped: `target_current_price = 1 - current_yes`,
   `target_fair_price = 1 - fair_yes`.
4. Otherwise (`current < fair`) → `target_side="YES"`,
   `target_current_price = current_yes`, `target_fair_price = fair_yes`.
5. Add `target_condition_id`, `target_side`, `target_current_price`,
   `target_fair_price` to the body.

Existing `event_id`, `event_title`, `deviation`, `markets[]` fields stay
untouched.

### Modified: `src/pscanner/config.py` (~+60 LOC)

New section `EvaluatorsConfig` with one sub-block per Evaluator:

```python
class SmartMoneyEvaluatorConfig(_Section):
    enabled: bool = True
    position_fraction: float = 0.01
    min_weighted_edge: float = 0.0

class MoveAttributionEvaluatorConfig(_Section):
    enabled: bool = True
    position_fraction: float = 0.01
    min_severity: Literal["low", "med", "high"] = "med"
    min_wallets: int = 3

class VelocityEvaluatorConfig(_Section):
    enabled: bool = True
    position_fraction: float = 0.0025  # per-entry (twin pair = 0.5% total)
    min_severity: Literal["low", "med", "high"] = "high"
    allow_consolidation: bool = False

class MispricingEvaluatorConfig(_Section):
    enabled: bool = True
    position_fraction: float = 0.01
    min_edge_dollars: float = 0.05

class EvaluatorsConfig(_Section):
    smart_money: SmartMoneyEvaluatorConfig = Field(...)
    move_attribution: MoveAttributionEvaluatorConfig = Field(...)
    velocity: VelocityEvaluatorConfig = Field(...)
    mispricing: MispricingEvaluatorConfig = Field(...)
```

`PaperTradingConfig` keeps `enabled`, `starting_bankroll_usd`,
`min_position_cost_usd`, `resolver_scan_interval_seconds`. Today's
`position_fraction` and `min_weighted_edge` fields are **removed**
(per the global "replace, don't deprecate" rule) — their semantics move
into `evaluators.smart_money.position_fraction` /
`evaluators.smart_money.min_weighted_edge`. Any operator config that
sets the old keys at the `paper_trading` level will fail pydantic
validation at boot (`extra="forbid"` is set on Config). The existing
deployment uses defaults only, so no live config files break, but the
removal is called out here for clarity.

`PaperTradingConfig.evaluators: EvaluatorsConfig` is added.

### Modified: `src/pscanner/store/db.py` (~+10 LOC)

Add to the `_MIGRATIONS` tuple:

```python
"ALTER TABLE paper_trades ADD COLUMN triggering_alert_detector TEXT",
"ALTER TABLE paper_trades ADD COLUMN rule_variant TEXT",
"UPDATE paper_trades SET triggering_alert_detector = 'smart_money' "
  "WHERE triggering_alert_detector IS NULL AND trade_kind = 'entry'",
```

The existing migrator's `OperationalError` swallow handles re-runs idempotently.

Update the unique-on-entry index from
`UNIQUE INDEX idx_paper_trades_alert_key ON paper_trades(triggering_alert_key) WHERE trade_kind='entry'`
to
`UNIQUE INDEX idx_paper_trades_alert_key ON paper_trades(triggering_alert_key, COALESCE(rule_variant, '')) WHERE trade_kind='entry'`.

The `COALESCE(rule_variant, '')` is critical: SQLite UNIQUE indexes treat
NULLs as distinct from each other by default (e.g., two rows
`('smart:X', NULL)` would both be permitted). With non-twin sources
(smart_money, move_attribution, mispricing) inserting `rule_variant=NULL`,
the naive index would let the same alert double-book. COALESCE folds NULL
to the empty string so per-`triggering_alert_key` uniqueness holds for
non-twin sources, while velocity's `('vel:X', 'follow')` and
`('vel:X', 'fade')` remain distinct. SQLite supports expression indexes
since 3.9.

The migration is a `DROP INDEX IF EXISTS` + `CREATE UNIQUE INDEX` pair
inside `_apply_migrations`; the existing `OperationalError` swallow handles
re-runs idempotently.

### Modified: `src/pscanner/store/repo.py` (~+15 LOC)

`PaperTradesRepo.insert_entry` accepts `triggering_alert_detector: str` and
`rule_variant: str | None`. Both written to the new columns.

`OpenPaperPosition` gains the same two fields. Read paths
(`list_open_positions`, `summary_stats`) read them through.

New helper `summary_by_source(starting_bankroll: float) -> list[SourceSummary]`
returning per-`(detector, rule_variant)` aggregates: total entries, resolved
count, realized PnL, win rate, open count.

### Modified: `src/pscanner/cli.py` (~+15 LOC)

`paper status` output gains a per-source breakdown table after the existing
aggregate stats. Format matches existing tables; one row per
`(detector, rule_variant)` with non-zero count.

### Modified: `src/pscanner/scheduler.py` (~+25 LOC)

`_attach_paper_trader` (or wherever `PaperTrader` is constructed) now:

1. Builds the four Evaluator instances, each gated by its `enabled` flag.
2. Wires their dependencies (e.g., `SmartMoneyEvaluator` needs the
   `tracked_wallets` repo; others need only the alert + market_cache via
   PaperTrader's existing `_resolve_outcome`).
3. Constructs `PaperTrader(evaluators=[...])` in fixed order: smart_money,
   move_attribution, mispricing, velocity. Order matters because
   `accepts(alert)` returns first-match — though there's no overlap today.

## Data flow

**Hot path (alert → entry):**

```
AlertSink.emit(alert)
        │
        ├─ alerts_repo.insert_if_new(alert)
        ├─ renderer.push(alert)
        └─ subscribers (sync)
                  │
                  └─ PaperTrader.handle_alert_sync(alert)
                              │
                              └─ loop.create_task(self.evaluate(alert))
                                          │
                                          ▼
                       PaperTrader.evaluate:
                         for ev in self._evaluators:
                           if ev.accepts(alert):
                             try:
                               parsed_list = ev.parse(alert)
                               bankroll = config.starting_bankroll_usd  # constant
                               nav = paper_trades.compute_cost_basis_nav(...) # for ledger
                               for parsed in parsed_list:
                                 if not ev.quality_passes(parsed): continue
                                 resolved = await self._resolve_outcome(...)
                                 if resolved is None: continue
                                 cost = ev.size(bankroll, parsed)
                                 if not _size_valid(cost, fill_price): continue
                                 paper_trades.insert_entry(
                                   ...,
                                   nav_after_usd=nav,  # informational, may be negative
                                   triggering_alert_detector=alert.detector,
                                   rule_variant=parsed.rule_variant,
                                 )
                             except asyncio.CancelledError:
                               raise
                             except Exception:
                               _LOG.warning("paper_trader.evaluator_failed", ...)
                             break
```

Three properties:

1. **Constant sizing.** Bet size is `starting_bankroll_usd * position_fraction`
   on every entry — independent of realized PnL. NAV is still read once per
   alert and written to `nav_after_usd` for analysis but plays no role in
   sizing or in any go/no-go gate.
2. **Per-ParsedSignal independence.** If quality_passes / resolve / size
   rejects one of velocity's two signals, the other still books.
3. **Order matters in the evaluator list.** First-match `accepts`. Scheduler
   builds the list in fixed order: smart_money, move_attribution, mispricing,
   velocity.

**Mispricing detector body emission (the enrichment):**

```
mispricing._build_alert(event_summary, deviation):
  prices_sum = sum(m.yes_price for m in markets)
  for m in markets:
    m.fair_price = m.yes_price / prices_sum   # proportional rebalancing
    m.deviation_extra = abs(m.yes_price - m.fair_price)
  target = max(markets, key=lambda m: m.deviation_extra)

  if target.yes_price > target.fair_price:
    target_side = "NO"
    target_current = 1.0 - target.yes_price
    target_fair = 1.0 - target.fair_price
  else:
    target_side = "YES"
    target_current = target.yes_price
    target_fair = target.fair_price

  body = {
    ... existing fields ...,
    "target_condition_id": target.condition_id,
    "target_side": target_side,
    "target_current_price": target_current,
    "target_fair_price": target_fair,
  }
```

Concrete example: market YES prices `[0.4, 0.5, 0.5]`, sum 1.4.
Fair prices: `[0.286, 0.357, 0.357]`. The two 0.5 markets are equally
over-priced (deviation 0.143); detector picks the first (existing tiebreak
behavior). Target: `side="NO"`, `current=0.5`, `fair=0.643` — 14.3¢ edge,
above the default 5¢ threshold.

**Resolution / exit (existing `PaperResolver`):**

Polls `data-api /positions?user=X&closed=true` periodically. Books exits for
resolved positions agnostic to source. The new
`triggering_alert_detector` / `rule_variant` columns ride on the entry row;
exits join via `parent_trade_id`.

**Per-source PnL query (after a few weeks of data):**

```sql
SELECT
  e.triggering_alert_detector AS source,
  e.rule_variant AS variant,
  COUNT(*) AS resolved_n,
  SUM(x.cost_usd - e.cost_usd) AS realized_pnl,
  AVG(CASE WHEN x.cost_usd > e.cost_usd THEN 1.0 ELSE 0.0 END) AS win_rate
FROM paper_trades e
JOIN paper_trades x ON x.parent_trade_id = e.trade_id
WHERE e.trade_kind = 'entry' AND x.trade_kind = 'exit'
GROUP BY 1, 2
ORDER BY realized_pnl DESC;
```

## Sizing summary

| source | per-entry fraction | per-entry $ on $1000 | per-alert $ exposure |
|---|---:|---:|---:|
| smart_money | 1.0% | $10 | $10 (1 entry) |
| move_attribution | 1.0% | $10 | $10 (1 entry) |
| mispricing | 1.0% | $10 | $10 (1 entry) |
| velocity | 0.25% | $2.50 | $5 (2 entries: follow + fade) |

`position_fraction` is per-entry, not per-alert, and applied against
`starting_bankroll_usd` (a constant), not against running NAV. Bet sizes
therefore stay fixed regardless of cumulative wins or losses — by design,
since the goal is maximum data collection per source. Velocity's smaller
fraction reflects that each side of the pair is a smaller bet; pair total
$5 stays below the other signals' $10. Future per-bucket sizing
(severity-tiered, change_pct-bucketed, etc.) is a method change inside the
relevant Evaluator's `size`.

## Error handling

| Condition | Behavior |
|---|---|
| Evaluator's `parse` raises | `paper_trader.evaluator_failed` warning with exc_info; alert dropped; loop survives. |
| Body-shape mismatch (missing field) | `parse` returns `[]`; debug log `paper_trader.bad_body`. Most common error path. |
| `_resolve_outcome` returns None (cache miss + gamma fallback fail) | ParsedSignal skipped at debug; remaining ParsedSignals from same alert continue. |
| `_size_trade` returns None (cost < min, or fill_price ∉ (0,1)) | ParsedSignal skipped at debug. |
| Cost-basis NAV is negative | Trades continue booking; `nav_after_usd` is written as-is (potentially negative). The `bankroll_exhausted` gate that existed before this spec is removed — research configuration assumes infinite money for data collection. |
| Duplicate alert (replay) | Existing UNIQUE-on-entry index catches it; `_insert_entry` swallows IntegrityError with debug log `paper_trader.duplicate_alert`. Index now keyed on `(triggering_alert_key, rule_variant)` so velocity's two entries don't collide. |
| Mispricing alert lacks `target_*` fields (stale/legacy) | `MispricingEvaluator.parse` returns `[]` and logs at debug. Forward-compatible. |
| Backfill UPDATE fails | Existing `_apply_migrations` swallows `OperationalError`; rows stay NULL and surface in an `unknown` source bucket. |
| Misconfigured `position_fraction` (negative, etc.) | Pydantic raises at boot. Standard config-error path. |

## Testing

### Per-Evaluator unit tests

`tests/strategies/evaluators/test_smart_money.py` (~150 LOC):
- `accepts` only `smart_money`.
- `parse` extracts `(wallet, condition_id, side)`; returns 1 ParsedSignal.
- `quality_passes` consults tracked_wallets.weighted_edge; rejects None /
  below threshold.
- `size` returns `nav * position_fraction`.
- Malformed body (missing wallet) → `parse` returns `[]`.

`tests/strategies/evaluators/test_move_attribution.py` (~120 LOC):
- `accepts` only `move_attribution`.
- `parse` extracts `condition_id` + `outcome`; ignores `side` taker action.
- `quality_passes`: severity gate + n_wallets gate. Test each independently.

`tests/strategies/evaluators/test_velocity.py` (~150 LOC):
- `accepts` only `velocity`.
- `parse` returns 2 ParsedSignals (follow + fade) with distinct
  `rule_variant` values.
- The fade ParsedSignal's `side` is the opposing outcome of the alert's
  asset_id, resolved via market_cache.
- `quality_passes`: `severity == "high"` AND `consolidation == False`.
  Test each gate.
- `size` returns per-side fraction; pair total is 2× per-side.
- If market_cache misses opposing-side asset, parse returns just the follow
  side (single-element list, not a tuple of one + None).

`tests/strategies/evaluators/test_mispricing.py` (~120 LOC):
- `accepts` only `mispricing`.
- `parse` extracts the four `target_*` fields; returns 1 ParsedSignal.
- Body without `target_*` → returns `[]` and logs at debug.
- `quality_passes`: edge magnitude gate.
- `size` returns `nav * position_fraction`.

### Mispricing detector enrichment test

`tests/detectors/test_mispricing.py` (extend):
- `deviation > 0` case: most-overpriced YES leg → `target_side="NO"` with
  flipped current/fair.
- `deviation < 0` case: most-underpriced YES leg → `target_side="YES"`.
- Proportional rebalancing math: `[0.4, 0.5, 0.5]` → fair `[0.286, 0.357,
  0.357]` → most-extreme is the 0.5 markets (deviation 0.143); first-by-tiebreak
  selection.

### PaperTrader refactor test

`tests/strategies/test_paper_trader.py` (refactor existing):
- Construct PaperTrader with `evaluators=[stub_a, stub_b]`.
- `evaluate` walks the list and picks the first acceptor.
- Alerts no Evaluator accepts are silently ignored.
- Stub evaluator raising `RuntimeError` during `parse` is caught (verify
  `paper_trader.evaluator_failed` log) and PaperTrader survives.
- Existing tests for outcome resolution / sizing / NAV-exhausted /
  duplicate-alert idempotence stay (orchestrator responsibilities).

### DB schema test

`tests/store/test_paper_trades_repo.py` (extend):
- New columns round-trip on insert + read.
- `OpenPaperPosition` carries them through.
- Backfill migration test: pre-populate row without new columns via raw
  SQL, apply migrations, assert `triggering_alert_detector = 'smart_money'`.
- Index update test:
  - Two velocity entries with same `triggering_alert_key`, distinct
    `rule_variant` → both succeed.
  - Two with same `(triggering_alert_key, rule_variant)` → second raises
    `IntegrityError`.

### Config test

`tests/test_config.py` (extend):
- `EvaluatorsConfig()` defaults match spec values.
- Each sub-block defaults are correct.
- `Config().paper_trading.evaluators.velocity.position_fraction == 0.0025`.

### CLI test

`tests/test_cli.py` (extend):
- Pre-seed `paper_trades` with rows from 4 different
  `triggering_alert_detector` values.
- Run `paper status`.
- Assert all 4 source names appear in the output with correct counts.

### Integration smoke (manual, post-merge)

1h soak with all four evaluators enabled. Confirm:
- `paper_trades` rows appear with `triggering_alert_detector` populated for
  each source.
- Zero `paper_trader.evaluator_failed` exceptions in the log.
- Velocity-pair entries appear in pairs (same `triggering_alert_key`,
  distinct `rule_variant`).
- `paper status` per-source breakdown shows all four sources represented.

## Post-deployment observations (2026-04-27 / 28 smoke runs)

After merging the spec to main, the daemon was run twice with paper
trading enabled and all four evaluators on default settings: a 2h soak
(2026-04-27) and a 6h soak (2026-04-28). Aggregate observations:

### Activity volume

| period | alerts | paper entries | exits |
|---|---:|---:|---:|
| 2h smoke | 372 | 230 | 0 |
| 6h smoke | 603 | 395 | 0 |

Per-source entry distribution from the 6h run is representative:

| source | entries | share |
|---|---:|---:|
| velocity follow | 149 | 38% |
| velocity fade | 149 | 38% |
| mispricing | 71 | 18% |
| smart_money | 13 | 3% |
| move_attribution | 13 | 3% |

Velocity dominates volume (76% combined), exactly as the alert-frequency
research predicted. Smart_money's gate (`tracked_wallets.weighted_edge >
0`) is selective enough that only the single qualified wallet
`0x2005d16a84…` (87.9% historical WR, $7.6M leaderboard PnL) sources its
trades; all 13 smart_money entries are from that one wallet's signal
stream over 8h combined.

### Resolution latency: 0 of 625 open positions resolved across 8h

Across both runs (8h combined runtime), **zero markets that the paper
trader bet on had resolved**. The aggregate state after 8h is 625 open
positions, $0 realized PnL, no win-rate signal yet.

This validates the spec's "constant sizing, drop the bankroll_exhausted
gate" decision: with no PnL feedback in 8h, an adaptive sizing scheme
would have kept booking at the starting bankroll regardless. Resolution
data will start landing over the next days/weeks as bet markets close;
useful per-rule signal probably needs 2-4 weeks of accumulated trades.

### Velocity twin-trade pairing health: perfect

In the 6h run, every high-severity velocity alert produced both a
follow and a fade entry (149 + 149 = 298 entries from 149 alerts).
No cache misses, no malformed bodies. The `MarketCacheRepo`
opposing-outcome lookup is healthy at production scale.

### Worker sink: validated

| metric | 2h | 6h |
|---|---:|---:|
| `worker_sink.stats` events | 119 | 359 |
| `worker_sink.queue_full` | 0 | 0 |
| max queue depth observed | 25 | 17 |
| blocking emit count | 0 | 0 |

Default `velocity_worker_maxsize=4096` is massively over-provisioned
for current load. Could be safely tightened to `512` or `1024` if
memory pressure ever became a concern.

### `paper_trader.market_cache_backfilled` is a real path used in production

19 events in 2h, 25 in 6h. Mispricing alerts on markets that aren't in
the local cache yet (Abraham Accords, Waymo, Russia-Ukraine ceasefire,
etc.) trigger `_resolve_outcome`'s gamma fallback, which discovers the
market slug from a recent trade and refreshes the cache row. This is the
spec'd cache-miss path working as designed; without it, mispricing entries
on fresh markets would be skipped.

## New follow-up surfaced by the 6h run

**`tick_stream.subscriber_queue_full` regression in long runs.** The 2h
smoke had 0 of these warnings; the 6h smoke had 2,708, all concentrated
in a single 1h 42m window (08:22-10:05 UTC) at ~28 drops/minute sustained.
This is **not** a cold-start issue (it began 4.5h into the run) and **not**
worker-sink backpressure (worker sink had max queue depth 17 throughout).

The bottleneck is upstream of `WorkerSink`: between
`BroadcastTickStream.publish` (called from `MarketTickCollector.snapshot_once`
once per cadence per asset, in tight loops) and the velocity detector's
synchronous consume-loop work in `evaluate()` — record-tick + window math
+ market-cache lookup. The `WorkerSink` decouples `await sink.emit(...)`
(the DB-write hot path) from the consume loop, but the *non-emit*
per-tick work is still synchronous; if many markets tick at once and
several trip velocity's threshold simultaneously, the consumer can fall
behind.

Mitigations to consider, in increasing order of cost:

1. **Bump `BroadcastTickStream` per-subscriber queue size** from 1024 to
   4096+. Cheap; doesn't fix the root cause.
2. **Pace `snapshot_once` publisher.** Insert `await asyncio.sleep(0)`
   between per-asset `await self._tick_stream.publish(event)` calls so the
   consumer gets interleaved scheduling. Already noted as a non-goal in
   the original WorkerSink spec; would help here.
3. **Decouple velocity's record from its evaluate.** Move the per-tick
   record + window-math into a separate consumer task with its own queue,
   leaving the tick-consume loop pure-CPU bookkeeping. Larger refactor.

Not blocking. Velocity tick drops are isolated to that one detector and
do not affect paper-trading or other evaluators. File as a separate
follow-up if/when the regression compounds with more tick subscribers
(e.g., depth-shock, spread-widening).

## Other out-of-scope follow-ups (not blocking)

- **Per-bucket sizing inside Evaluators.** Once we have a few weeks of
  resolution data, the per-detector `size` method can grow data-driven
  rules (e.g., velocity sized by `change_pct` bucket). Architecture
  supports this; data does not yet.
- **Aggregate open-exposure cap.** Deferred per user direction. Worth
  revisiting once we observe resolution cadence.
- **More signal sources.** `WhaleClusterEvaluator`, `ConvergenceEvaluator`,
  etc. once those detectors emit tradeable signal. New file + new config
  block, no PaperTrader edits.
- **Velocity rule winners.** After accumulating resolved pairs, query
  `(category, severity, change_pct_bucket) → winning_rule_variant` to
  refine velocity's `parse` (e.g., only fade in sports, only follow in
  thesis markets).
- **`win_rate=0.0` ambiguity for unresolved buckets.** When
  `resolved_count=0` for a source, `win_rate` reports `0.0%` —
  indistinguishable from "all losses." Render as `-` instead of `0.0%`
  in the CLI breakdown when no resolutions yet. Trivial polish.
- **`paper_trades.source_wallet` aggregate panel.** The aggregate
  per-wallet PnL panel in `paper status` shows a single "no wallet"
  bucket aggregating all non-smart_money entries (382 of 395 in the 6h
  run). Group these under "(no wallet)" or merge with the per-source
  breakdown table.
