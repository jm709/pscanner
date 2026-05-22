# Design — `SubgraphTradeCollector` daemon promotion (#152)

Date: 2026-05-21
Issue: [#152](https://github.com/jm709/polymarketScanner/issues/152)
Predecessors: #153 (research script), #154 (V2 subgraph migration), #158 (token resolver)

## Problem

`scripts/watch_subgraph_copy.py` shipped in #153 as a standalone research script.
It polls the Polymarket V2 subgraph for trades by watchlisted wallets and books
paper copies into `paper_trades`. The script works end-to-end on the desktop;
the desktop is currently running it manually to gather copy-trade data on a
small watchlist.

For live operation the script needs to graduate into the daemon so it benefits
from the daemon's lifecycle, supervision, observability, and alert/evaluator
pipeline. Issue #152 calls this the `SubgraphTradeCollector` promotion.

The script today bypasses the daemon's `AlertSink` and writes directly to
`paper_trades`. That's pragmatic but isolates the path from:

- the `alerts` table (no row for an operator to inspect)
- `pscanner paper status` per-source breakdown (works only because the
  detector tag is set, but no upstream alert context)
- `replay_unbooked()` on daemon restart (issue #105)
- the renderer

The promotion fixes this by routing through `AlertSink` + a new evaluator.

## Goals

1. Run the subgraph-watch polling loop inside the daemon process, supervised
   by `Scanner._supervise_collector`.
2. Emit a `subgraph_copy` alert per copy-eligible event; route booking through
   a new `SubgraphCopyEvaluator`.
3. Replace runaway-concentration risk (one chatty wallet booking 100% of
   trades) with adaptive per-wallet sizing.
4. Coexist with the `/activity` REST `TradeCollector` — both collectors run
   simultaneously and write to disjoint tables (wallet_trades vs alerts).
5. Preserve the script's working semantics (subgraph query shape, watermark
   pagination, copy-direction table) verbatim — only the wiring changes.

## Non-goals

- Replacing or modifying `TradeCollector` (`/activity` polling). It keeps
  populating `wallet_trades` for the cluster detector and counterparty
  observations.
- Implementing the WSS RPC alternative path (#152 out-of-scope section).
- Hot-reloading model artifacts or watchlist size (operator restarts daemon).
- Negative caching for tokens gamma can't resolve (deferred per #157).
- Per-wallet quality weighting via `tracked_wallets.weighted_edge`. On the
  desktop today 0 of 5 active watchlist wallets have `tracked_wallets` rows;
  watchlist membership IS the quality bar. Anti-concentration sizing replaces
  quality weighting as the operator's exposure-management knob.

## Architecture overview

```
SubgraphTradeCollector (new, pscanner.collectors.subgraph_trades)
  │  every poll_interval_seconds:
  │    1. read active watchlist (WatchlistRepo)
  │    2. query subgraph orderFilledEvents since last_seen_ts (server-filtered)
  │    3. for each watchlist BUY-direction event:
  │         resolve token (token_resolver + gamma fallback)
  │         lookup fill_price (MarketCacheRepo / MarketTicksRepo)
  │         emit Alert(detector="subgraph_copy", body=...)
  │    4. persist new last_seen_ts to subgraph_watch_state table
  │
  ▼
AlertSink → alerts table (idempotent on alert_key)
  │
  ▼
SubgraphCopyEvaluator (new, pscanner.strategies.evaluators.subgraph_copy)
  │  on each alert:
  │    1. parse body → ParsedSignal
  │    2. compute concentration_multiplier(source_wallet, watchlist_size)
  │    3. size = bankroll * base_fraction * multiplier
  │    4. PaperTradesRepo.insert_entry(...)
  │
  ▼
paper_trades (triggering_alert_detector='subgraph_copy')
```

Five touch-points in the existing codebase:

1. `pscanner.alerts.models.DetectorName` — add `"subgraph_copy"` literal
   (the renderer's per-detector buffer follows automatically via
   `get_args(DetectorName)`)
2. `pscanner.config` — add `SubgraphTradeCollectorConfig` +
   `SubgraphCopyEvaluatorConfig`, wire into `Config` + `EvaluatorsConfig`
3. `pscanner.scheduler.Scanner._build_collectors` /
   `_build_paper_evaluators` — construct conditionally on `enabled`
4. `pscanner.store.db._SCHEMA_STATEMENTS` — new `subgraph_watch_state` table

## `SubgraphTradeCollector` internals

Implements the `Collector` protocol (`name: str`, `async run(stop_event)`).

### Constructor

```python
class SubgraphTradeCollector:
    name: str = "subgraph_trades"

    def __init__(
        self, *,
        config: SubgraphTradeCollectorConfig,
        subgraph_client: SubgraphClient,
        gamma_client: GammaClient,
        watchlist_repo: WatchlistRepo,
        asset_index: AssetIndexRepo,       # corpus DB connection
        market_cache: MarketCacheRepo,
        market_ticks: MarketTicksRepo,
        sink: IAlertSink,
        state_repo: SubgraphWatchStateRepo,
        clock: Clock | None = None,
    ) -> None: ...
```

The script's free functions (`_compute_copy_direction`, `_build_where_clause`,
`_serialize_where_inline`, `_fetch_events_since`) move in mostly verbatim as
private static / module-level helpers. `_resolve_event_booking` becomes
`_build_alert_body` and returns a JSON-serializable dict rather than a
`_BookingParams` dataclass — and the collector no longer calls
`lookup_fill_price`; price and asset_id are resolved by `PaperTrader`
downstream.

### One poll cycle

```python
async def _poll_once(self) -> None:
    addrs = sorted({e.address.lower() for e in self._watchlist.list_active()})
    if not addrs:
        _LOG.warning("subgraph_trades.empty_watchlist")
        return
    last_seen = self._state.get_last_seen_ts() or self._cold_start_ts()
    events, indexer_ts = await _fetch_events_since(
        self._subgraph, addrs=addrs, last_seen_ts=last_seen
    )
    self._warn_on_lag(indexer_ts)
    new_last_seen = last_seen
    for ev in events:
        new_last_seen = max(new_last_seen, int(ev["timestamp"]))
        body = await self._build_alert_body(ev, set(addrs))
        if body is None:
            continue
        alert = Alert(
            detector="subgraph_copy",
            alert_key=f"subgraph:{ev['transactionHash']}:{body['outcome']}",
            severity="med",
            title=f"copy {body['source_wallet'][:14]}.. {body['outcome']}"
                  f" @ {body['fill_price']:.3f}",
            body=body,
            created_at=int(ev["timestamp"]),
        )
        await self._sink.emit(alert)
    self._state.set_last_seen_ts(new_last_seen)
```

### Alert body contract

```python
{
    "source_wallet": "0x...",   # watchlisted wallet whose position increased
    "tx_hash": "0x...",
    "condition_id": "0x...",
    "outcome": "Cavaliers",     # outcome name (becomes ParsedSignal.side)
    "ts": 1747...,
}
```

The collector resolves `tokenId` → `outcome_name` via the existing
`token_resolver` (which also upserts `market_cache` on first-sighting). It
does NOT pre-compute `fill_price` or `asset_id` — `PaperTrader._resolve_outcome`
re-derives both from `MarketCacheRepo` + `MarketTicksRepo` at booking time
(same code path that smart_money, velocity, etc. all use). This matches the
project convention; the script bypassed it only because the script bypassed
PaperTrader.

### Lifecycle

```python
async def run(self, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await self._poll_once()
        except Exception:
            _LOG.exception("subgraph_trades.cycle_failed")
        if await _wait_or_stop(stop_event, self._config.poll_interval_seconds):
            return
```

Restart-on-crash is handled by the existing `Scanner._supervise_collector`.

### Preflight

`Scanner.preflight()` gains: if `subgraph_trades.enabled=true`, refuse to start
when `GRAPH_API_KEY` env var is missing. Mirrors the existing gate-model
preflight refusal.

### Cold-start

If `subgraph_watch_state` has no row (fresh daemon, or table just migrated),
the collector defaults to `now() - config.cold_start_lookback_seconds`.
Default `0` ⇒ start from `now()`, ignore history. Operator can set
e.g. `3600` to backfill the last hour on first daemon start.

## `SubgraphCopyEvaluator` internals

Implements `SignalEvaluator` (in `pscanner.strategies.evaluators.protocol`).
Lives at `pscanner.strategies.evaluators.subgraph_copy.SubgraphCopyEvaluator`.

### Protocol surface

```python
class SubgraphCopyEvaluator:
    def __init__(
        self, *,
        config: SubgraphCopyEvaluatorConfig,
        watchlist_repo: WatchlistRepo,
        paper_trades: PaperTradesRepo,
    ) -> None: ...

    def accepts(self, alert: Alert) -> bool:
        return alert.detector == "subgraph_copy"

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        body = alert.body
        return [ParsedSignal(
            condition_id=ConditionId(body["condition_id"]),
            side=body["outcome"],          # PaperTrader uses this as outcome name
            rule_variant=None,
            metadata={
                "wallet": body["source_wallet"],   # picked up by _insert_entry
                "tx_hash": body["tx_hash"],
                "ts": body["ts"],
            },
        )]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        return True  # quality gating happens at watchlist admission

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        base = bankroll * self._config.position_fraction
        wallet = parsed.metadata.get("wallet", "")
        multiplier = self._concentration_multiplier(wallet)
        return base * multiplier
```

The protocol signatures (`accepts`, `parse`, `quality_passes`, `size`) match
`pscanner.strategies.evaluators.protocol.SignalEvaluator` exactly.
`PaperTrader._insert_entry` reads `parsed.metadata["wallet"]` to populate
`paper_trades.source_wallet` — same convention as the smart_money evaluator.

`parse` returns exactly one `ParsedSignal` per alert (no twin trades).

`quality_passes` is a no-op for v1 — gating happens at watchlist-admission
time outside the daemon. The method exists to satisfy the protocol and gives
a hook for future per-wallet quality decay detection.

### Sizing math

```python
def _concentration_multiplier(self, wallet: str) -> float:
    counts = self._paper_trades.count_by_source_wallet(
        detector="subgraph_copy"
    )
    total = sum(counts.values())
    if total == 0:
        return 1.0
    wallet_count = counts.get(wallet.lower(), 0)
    share = wallet_count / total
    active_n = max(1, len(self._watchlist.list_active()))
    target_share = 1.0 / active_n
    raw = min(1.0, target_share / max(share, target_share))
    return max(raw, self._config.min_multiplier)
```

Two state lookups per alert; both are O(active_watchlist_size). Worked
examples at `base_fraction=0.005`, `min_multiplier=0.10`,
`active_n=30` (so `target_share=0.033`):

| wallet share | multiplier | size at bankroll=$1000 |
|---|---|---|
| 0% (first ever trade) | 1.0 | $5.00 |
| 3.3% (equal flow) | 1.0 | $5.00 |
| 10% | 0.33 | $1.67 |
| 25% | 0.13 | $0.67 |
| 50% | 0.10 (floored) | $0.50 |
| 100% (single chatty wallet) | 0.10 | $0.50 |

The 100% row matches the desktop's actual state today (102 of 102 trades from
one wallet, `0x6a67...`). Under this evaluator the same wallet would size at
`min_multiplier × base = $0.50` rather than the current $5.00.

### New repo method

`PaperTradesRepo.count_by_source_wallet(*, detector: str) -> dict[str, int]`:

```python
def count_by_source_wallet(self, *, detector: str) -> dict[str, int]:
    rows = self._conn.execute(
        """
        SELECT source_wallet, COUNT(*)
        FROM paper_trades
        WHERE triggering_alert_detector = ?
        GROUP BY source_wallet
        """,
        (detector,),
    ).fetchall()
    return {row[0]: int(row[1]) for row in rows}
```

## Schema additions

In `pscanner.store.db._SCHEMA_STATEMENTS`:

```sql
CREATE TABLE IF NOT EXISTS subgraph_watch_state (
  key TEXT PRIMARY KEY,
  last_seen_ts INTEGER NOT NULL
);
```

One row, key = `'default'`. New `SubgraphWatchStateRepo` with:

```python
class SubgraphWatchStateRepo:
    def get_last_seen_ts(self) -> int | None: ...
    def set_last_seen_ts(self, ts: int) -> None: ...
```

Additive migration via `IF NOT EXISTS` — no `_MIGRATIONS` entry needed.

## Config additions

```python
class SubgraphTradeCollectorConfig(_Section):
    enabled: bool = False
    subgraph_id: str = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
    poll_interval_seconds: float = 10.0
    rpm: int = 60
    page_size: int = 1000
    cold_start_lookback_seconds: int = 0
    indexer_lag_warn_seconds: int = 60
    indexer_lag_error_seconds: int = 600


class SubgraphCopyEvaluatorConfig(_Section):
    enabled: bool = False
    position_fraction: float = 0.005
    min_multiplier: float = 0.10
```

Wired at `Config.subgraph_trades = SubgraphTradeCollectorConfig(...)` and
`EvaluatorsConfig.subgraph_copy = SubgraphCopyEvaluatorConfig(...)`.

Note: `SubgraphCopyEvaluatorConfig` does NOT carry `min_position_cost_usd`
— the existing `PaperTradingConfig.min_position_cost_usd` already applies
(see `PaperTrader._run_pipeline`, which short-circuits when the evaluator's
`size()` returns less than the global minimum).

## Scheduler wiring

```python
# _build_collectors:
if self._config.subgraph_trades.enabled:
    subgraph_url = (
        f"https://gateway.thegraph.com/api/{os.environ['GRAPH_API_KEY']}"
        f"/subgraphs/id/{self._config.subgraph_trades.subgraph_id}"
    )
    collectors["subgraph_trades"] = SubgraphTradeCollector(
        config=self._config.subgraph_trades,
        subgraph_client=SubgraphClient(url=subgraph_url, rpm=self._config.subgraph_trades.rpm),
        gamma_client=self._clients.gamma_client,
        watchlist_repo=self._watchlist_repo,
        asset_index=AssetIndexRepo(self._corpus_conn),
        market_cache=self._market_cache_repo,
        market_ticks=self._ticks_repo,
        sink=self._sink,
        state_repo=SubgraphWatchStateRepo(self._db),
        clock=self._clock,
    )

# _build_paper_evaluators:
if cfg.subgraph_copy.enabled:
    evaluators.append(SubgraphCopyEvaluator(
        config=cfg.subgraph_copy,
        watchlist_repo=self._watchlist_repo,
        paper_trades=paper_trades_repo,
    ))
```

### Corpus DB connection

`Scanner` currently holds only the daemon DB. The collector needs read/write
on `data/corpus.sqlite3` so the token resolver can upsert `asset_index` on
gamma-fallback hits. Add `self._corpus_conn = init_corpus_db(corpus_path)`
to `Scanner.__init__`, close in `aclose()`. Opened only when
`subgraph_trades.enabled=true` to avoid pinning the corpus file for daemons
that don't use it.

## Renderer

`TerminalRenderer` enumerates per-detector ring buffers via
`get_args(DetectorName)` — adding `"subgraph_copy"` to the Literal is enough
for the renderer to allocate a buffer and surface incoming alerts. No
per-detector format string needs editing. The alert's `title` field already
carries the human-readable copy line (e.g.
`"copy 0x6a678ca367.. Cavaliers @ 0.420"`).

## Decommissioning the script

`scripts/watch_subgraph_copy.py` stays during the rollout for parity
comparison. It gets deleted in a follow-up commit once the daemon path is
live-validated against the desktop's 5-wallet watchlist for at least one
24-hour window. The reusable helpers
(`pscanner.strategies.paper_trader.lookup_fill_price`,
`pscanner.poly.token_resolver.resolve_token`) already live outside the
script and stay where they are.

## Smoke plan

**All smoke runs on the desktop** — the laptop's corpus DB is stale and the
watchlist is empty. The laptop runs only unit tests + linters.

1. Rsync code to the desktop.
2. Restart the daemon with `subgraph_trades.enabled=true` +
   `evaluators.subgraph_copy.enabled=true`.
3. Confirm:
   - `subgraph_watch_state` row appears in `data/pscanner.sqlite3`.
   - `alerts` table grows `subgraph_copy` rows.
   - `paper_trades(triggering_alert_detector='subgraph_copy')` continues
     growing.
   - The single chatty wallet (`0x6a67...`) sizes at `min_multiplier × base
     = $0.50` once its share clears `target_share = 1/5 = 0.20`. Today
     it's at 100% share, so first booking under the new path should be
     `$0.50`, not `$5.00`.
   - `pscanner paper status` per-source breakdown shows `subgraph_copy`
     line correctly.
4. Diff the first 50 booked rows against what
   `scripts/watch_subgraph_copy.py --once` would book in the same window
   (run the script in dry-run mode in parallel and compare alert_keys).

## Test surface

- `tests/collectors/test_subgraph_trades.py` (~12 tests) — collector poll
  cycle, alert emission, checkpoint, copy-direction table, indexer-lag
  warnings, gamma-fallback resolution path, empty-watchlist short-circuit.
- `tests/strategies/evaluators/test_subgraph_copy.py` (~10 tests) — parse,
  sizing under varying shares (0%, target_share, 50%, 100%), `min_multiplier`
  floor, cold-start (total=0), watchlist-resize behavior.
- `tests/store/test_subgraph_watch_state_repo.py` (~3 tests) — get/set
  roundtrip, missing-row returns `None`.
- Extend `tests/store/test_paper_trades_repo.py` with `count_by_source_wallet`
  cases (~3 tests).
- Extend `tests/test_scheduler.py` with one wiring test that constructs a
  `Scanner` with both `subgraph_trades` and `subgraph_copy` enabled and
  verifies the collector + evaluator are present.

Integration test (#163's request) lands as part of this work: a real-SQLite
fixture that runs one collector poll + alert emit + evaluator booking
end-to-end against in-memory clients. Goes in
`tests/collectors/test_subgraph_trades_integration.py`.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Gamma rate-limit when many tokens miss locally on cold start | Token resolver caches in both DBs on first hit; subsequent cycles are local-only. Existing gamma rpm budget (50/min) is shared with smart-money and refresh paths; if subgraph adds material gamma traffic, raise `gamma_rpm` in config. |
| Daemon restart loses in-flight trades | `replay_unbooked()` in `PaperTrader` covers this for free once we route through `AlertSink`. Issue #105. |
| Subgraph indexer lag silently masks dropped events | Existing `subgraph_watch.indexer_lag` warn (60s) / error (600s) logs already implemented in the script, ported into the collector. |
| Concentration multiplier computed per alert is too expensive | `count_by_source_wallet` returns ≤ active_watchlist_size rows. On the desktop today that's 5 rows. A 10× growth (50 wallets) is still trivial. |
| One wallet's first ever trade gets `multiplier=1.0` even if it's later proven chatty | This is intentional. The multiplier only adapts after observation. Operator can manually reset by deleting that wallet's `paper_trades` rows, or just wait — the curve self-corrects within ~1/`target_share` trades. |

## Open follow-ups (out of scope for this issue)

- Quality decay detection: automatically demote watchlist wallets whose
  recent `paper_trades` PnL turns negative. Deferred until we have enough
  resolved subgraph_copy trades to compute it.
- Hot reload of `subgraph_trades.enabled` or watchlist size without restart.
- Negative caching for tokens gamma can't resolve (deferred per #157).
- WSS RPC alternative path for sub-second latency (deferred per #152
  out-of-scope).
