# Alert-emit offload (`WorkerSink`) — Design

**Date:** 2026-04-27
**Status:** approved, awaiting implementation plan

## Motivation

The 1h smoke (2026-04-27) recorded **498 `tick_stream.subscriber_queue_full`
warnings** over 83,450 ticks (~0.6% drop rate). Investigation traced the
backpressure to two causes:

1. **Bursty publisher.** `MarketTickCollector.snapshot_once` (`ticks.py:390-419`)
   walks every subscribed asset and calls `await self._tick_stream.publish(event)`
   in a tight loop. Bursts of up to ~1000 events back-to-back at startup
   (`ticks.snapshot_complete assets=1000 inserted=1000`) and ~120 events per
   steady-state cadence.

2. **Synchronous drain in the consumer.** Velocity (`velocity.py:97-103`) does
   `async for tick in stream.subscribe(): await self.evaluate(tick, sink)`.
   Inside `evaluate`, alert emissions run inline — `await sink.emit(alert)`
   does a SQLite `insert_if_new`, terminal-renderer push, and synchronous
   subscriber callbacks (e.g., `PaperTrader.handle_alert_sync`). Per-tick
   total time spikes whenever an alert fires.

When a burst arrives during one of these slow ticks, velocity's per-subscriber
queue (default `maxsize=1024`) fills and `BroadcastTickStream.publish` drops
events for that subscriber.

Adding more tick-driven detectors (depth-shock, spread-widening, etc.) would
multiply the surface area of the same problem — each new subscriber gets its
own 1024-deep queue and faces the same burst-vs-drain mismatch. Fixing it
once with a reusable primitive is cheaper than fixing it N times.

## Goal

Offload the slow alert-emit path off velocity's per-tick hot loop using a
reusable, per-detector primitive. Validate with `worker_sink.stats`
instrumentation that the worker is decoupling the path; validate end-to-end
with a 1h smoke showing zero `tick_stream.subscriber_queue_full` warnings.

## Non-goals

- Fix the bursty publisher in `snapshot_once`. Pacing the publisher is a
  separate, cheaper change. We're addressing the consumer side because the
  backpressure pattern repeats per subscriber and the publisher fix doesn't.
- Profile per-tick latency inside velocity. Possibly useful follow-up if
  `worker_sink.stats` shows blocking emits remain high after this work.
- Wrap polling-driven detectors (cluster, smart_money, convergence, etc.).
  They don't share the burst pattern — opt them in later only if a benefit
  is demonstrated.
- Preserve the dedupe `bool` return through the worker. Caller analysis
  (below) shows no tick-driven detector uses it.

## Architecture

Today, every alert path runs synchronously inside the detector that emits:

```
detector.evaluate(...) → await sink.emit(alert)
                                 │
                                 ├─ alerts_repo.insert_if_new(alert)   (SQLite write)
                                 ├─ renderer.push(alert)                (terminal print)
                                 ├─ subscriber callbacks (sync)         (paper_trader, ...)
                                 └─ structlog "alert.emitted"
```

The fix introduces one indirection on the hot path. A new `IAlertSink`
Protocol captures the public surface tick-driven detectors actually depend
on (`async def emit(alert) -> bool`). `AlertSink` (today's concrete class)
is unchanged and structurally implements `IAlertSink`. A new `WorkerSink`
also implements `IAlertSink` but defers the work: `emit()` puts the alert
on an `asyncio.Queue` and returns immediately; a long-lived drain task
pulls alerts off the queue and invokes the wrapped real sink.

The Scanner constructs a `WorkerSink(real_sink, maxsize=N, name=...)` per
tick-driven detector and injects that. Each `WorkerSink` owns its own
queue and its own drain task — per-detector isolation, so a slow alert
subscriber affecting one detector's drain doesn't backpressure another.

**Key consequences:**

- Detectors accept `sink: IAlertSink`, not `AlertSink`. The wrap is
  invisible at the call site (`await self._sink.emit(alert)` is unchanged).
- `WorkerSink.emit()` always returns `True` (enqueue acknowledged). The
  authoritative dedupe `bool` is decided later by the inner sink. This is
  a no-op today: the only callers using the bool are in `cluster.py`, which
  is polling-driven and stays on the raw `AlertSink`.
- `AlertSink.subscribe()` (sync subscriber callbacks like
  `PaperTrader.handle_alert_sync`) stays on the concrete `AlertSink` — it
  fires when the inner sink runs, off the hot path.
- Lifecycle (`start`/`aclose`) is owned by Scanner alongside other
  long-lived tasks; Scanner ordering guarantees the drain task is alive
  before any detector's `run()` is started.

## Components

**New file: `src/pscanner/alerts/protocol.py`** (~15 LOC)
- `IAlertSink` `Protocol` with one method: `async def emit(self, alert: Alert) -> bool`.

**New file: `src/pscanner/alerts/worker_sink.py`** (~110 LOC)
- `WorkerSink` class:
  - `__init__(inner: IAlertSink, *, maxsize: int, name: str, stats_interval_seconds: int = 60, clock: Clock | None = None)`.
  - Owns `_queue: asyncio.Queue[Alert]`, `_drain_task`, `_stats_task`.
  - Counters: `_drained_since_last_log`, `_blocking_emits_since_last_log`, `_depth_max_since_last_log`.
  - `_closing: bool` flag.
- `async def emit(self, alert: Alert) -> bool`:
  - When `self._closing` is True: log `worker_sink.closed_drop` and
    return `False` without enqueueing.
  - Try `put_nowait`. On `QueueFull`: log `worker_sink.queue_full` (one
    per occurrence), increment `_blocking_emits_since_last_log`, then
    `await self._queue.put(alert)`.
  - After any successful enqueue, update
    `_depth_max_since_last_log = max(_depth_max_since_last_log, queue.qsize())`.
  - Return `True` on successful enqueue.
- `async def start(self) -> None` — launch drain + stats tasks.
- `async def aclose(self) -> None` — set `_closing`, drain remaining queue
  with a 5s timeout via `Clock`. Log `worker_sink.shutdown_drain_timeout`
  with `remaining` count if the timeout fires. Cancel stats task and await
  both.
- `_drain_loop` — `while True: alert = await self._queue.get()`; call
  `await self._inner.emit(alert)` inside `try/except`. On exception except
  `CancelledError`, log `worker_sink.drain_failed` at exception level,
  drop the alert, continue. `_drained_since_last_log` increments only on
  successful inner emit.
- `_stats_loop` — `while True: await self._clock.sleep(stats_interval)`;
  emit one structlog `worker_sink.stats` event with `name`,
  `queue_depth_max`, `blocking_emit_count`, `drain_count`. Reset counters.
  Wrap the `_LOG.info` call in `try/except Exception` so a logging glitch
  doesn't kill the stats task.

**Modified: `src/pscanner/alerts/sink.py`**
- No code change. `AlertSink` already structurally satisfies `IAlertSink`;
  `ty check` confirms.

**Modified: `src/pscanner/detectors/velocity.py`**
- Velocity receives the sink via `run(self, sink)` and forwards it to
  `evaluate(self, tick, sink)` — both annotations change from
  `sink: AlertSink` → `sink: IAlertSink`. Internal `await sink.emit(...)`
  call site unchanged.

**Not modified (v1):**
- `cluster.py`, `smart_money.py`, `convergence.py`, `mispricing.py`,
  `move_attribution.py`, `whales.py` keep the raw `AlertSink` — none of
  them are tick-driven, none experience burst pressure.

**Modified: `src/pscanner/scheduler.py`**
- `_maybe_attach_velocity_detector` constructs
  `WorkerSink(self._sink, maxsize=cfg.velocity_worker_maxsize, name="velocity")`
  and stores it on the detector entry (sibling dict, attribute, or small
  detector-spec record — implementation-time choice). `_supervise_detector`
  passes the per-detector sink (worker-wrapped or raw) into
  `await run_fn(sink)`. `await worker.start()` runs before the supervised
  task; `await worker.aclose()` runs in `Scheduler.aclose()` after detector
  cancellation.

**Modified: `src/pscanner/config.py`**
- Add fields under an existing or new `WorkerSinkConfig`:
  - `velocity_worker_maxsize: int = 4096`
  - `worker_stats_interval_seconds: int = 60`

**Total new code:** ~125 LOC across two new files; ~30 LOC of edits across
three existing files.

## Data flow

**Hot path (per tick):**

```
ticks._snapshot_loop → snapshot_once → _maybe_publish → BroadcastTickStream.publish
                                                              │
                                              per-subscriber asyncio.Queue (1024-deep)
                                                              │
velocity._tick_stream.subscribe() ── async for ──→ velocity.evaluate(tick, sink)
                                                              │
                                                              ├─ record + window math (CPU only)
                                                              └─ on threshold trip:
                                                                   await sink.emit(alert)   ← IAlertSink
                                                                              │
                                                                  WorkerSink.emit
                                                                              │
                                                          ┌───────── put_nowait ─────────┐
                                                       success                   QueueFull
                                                          │                            │
                                                          │             log "worker_sink.queue_full"
                                                          │                            │
                                                          │                     await queue.put
                                                          │                            │
                                                          └────── return True ─────────┘
                                                                              │
                                                  ─── velocity loops back to next tick ──
```

In the steady state with a non-full queue, `put_nowait` is constant-time
and velocity returns to its `async for` loop in microseconds.

**Drain path (one task per `WorkerSink`):**

```
WorkerSink._drain_loop:
    while True:
        alert = await self._queue.get()
        try:
            await self._inner.emit(alert)   # AlertSink — DB write, renderer, subscribers
            self._drained_since_last_log += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOG.exception("worker_sink.drain_failed", name=self._name, alert_key=alert.alert_key)
```

**Stats path (one task per `WorkerSink`):**

```
WorkerSink._stats_loop:
    while True:
        await self._clock.sleep(stats_interval)
        try:
            _LOG.info(
                "worker_sink.stats",
                name=self._name,
                queue_depth_max=self._depth_max_since_last_log,
                blocking_emit_count=self._blocking_emits_since_last_log,
                drain_count=self._drained_since_last_log,
            )
        except Exception:
            pass
        # reset counters
```

## Lifecycle

- Scanner calls `await worker.start()` **before** any
  `detector.run()` — guarantees drain task is alive before the first
  `emit`.
- Scanner shutdown sequence calls `await worker.aclose()`:
  1. Set `_closing = True`. Subsequent `emit` calls log
     `worker_sink.closed_drop` and return without enqueueing.
  2. `await asyncio.wait_for(self._queue.join(), timeout=5.0)`. If timeout,
     log `worker_sink.shutdown_drain_timeout` with `remaining=qsize()`.
  3. Cancel `_drain_task` and `_stats_task`, await both with
     `suppress(CancelledError)`.
- Note: alerts already in flight inside a detector's `evaluate()` when
  shutdown cancels are still lost — same as today's behavior. Not in scope
  to recover.

## Backpressure escalation

1. **Steady state (queue not full):** `emit` returns immediately;
   `_blocking_emits_since_last_log` stays 0; `worker_sink.stats` shows
   `blocking_emit_count=0` every interval.
2. **First fill:** `worker_sink.queue_full` warning emitted (one per
   occurrence). `_blocking_emits_since_last_log` increments. `await put`
   blocks the caller (velocity) until the drain pulls one off. Velocity's
   tick-stream subscriber queue starts to fill while velocity is blocked.
3. **Sustained fills:** stats events surface elevated
   `blocking_emit_count`. Operator sees the signal and either bumps
   `velocity_worker_maxsize` in config or investigates why the inner sink
   is slow (e.g., DB contention, slow subscriber callback).
4. **Operator escape hatch:** existing `tick_stream.subscriber_queue_full`
   continues to fire if the velocity tick subscriber falls behind for
   reasons unrelated to alert emission. Distinct event names make the two
   conditions independently observable.

## Error handling

| Condition | Behavior |
|---|---|
| Inner sink raises during drain | `worker_sink.drain_failed` at exception level; drop alert; continue draining. `_drained` does not increment for failed drains. |
| Drain task crashes (truly unexpected) | Surfaced via Scanner's existing `task.add_done_callback` watchdog. Process exits; supervisor restarts. |
| Stats `_LOG.info` fails | `try/except Exception` swallows; counters reset; loop continues. |
| Cancel mid-`emit()` while awaiting full queue's `put` | Alert is lost. Documented; no recovery (matches today's cancel-mid-emit behavior). |
| `aclose()` with pending queue | Drain with 5s timeout; remaining count logged on timeout. |
| `emit()` after `aclose()` | Log `worker_sink.closed_drop`; return False without enqueueing. |
| `Clock.sleep` interrupted | `CancelledError` propagates to caller (drain or stats task), letting Scanner shutdown finish. |

## Caller analysis

Verified call sites for `sink.emit(...)`:

| File | Line | Use of return value |
|---|---|---|
| `velocity.py` | 147 | discarded |
| `whales.py` | 157 | discarded |
| `mispricing.py` | 175 | discarded |
| `smart_money.py` | 400 | discarded |
| `convergence.py` | 122 | discarded |
| `move_attribution.py` | 387 | discarded |
| `cluster.py` | 164 | `if await self._sink.emit(alert):` |
| `cluster.py` | 549 | `return await sink.emit(alert)` |

Both bool-using callers are in `cluster.py`, which is polling-driven and
keeps the raw `AlertSink`. The v1 `WorkerSink` adoptee (velocity)
discards the return. `WorkerSink.emit` always returning `True` is a
zero-behavior-change choice.

## Testing

**File: `tests/alerts/test_worker_sink.py`** (new — ~250 LOC)

All tests use `FakeClock` (`pscanner.util.clock`) and a stub `IAlertSink`
that records calls. No real SQLite needed.

1. **emit returns immediately on non-full queue.** Inner sink not yet
   called when `emit` returns.
2. **drain delivers alert.** Advance clock; inner sink received the alert.
3. **FIFO order preserved.** Emit 50 alerts; inner sink sees 50 in order.
4. **inner sink failure logs and continues.** Stub raises on alert #3;
   alerts {1,2,4,5} delivered; one `worker_sink.drain_failed` log; drain
   alive.
5. **queue-full path: warn + block.** `maxsize=2`; hold inner via
   `asyncio.Event`; emit 3; assert third is pending; one
   `worker_sink.queue_full` log; release inner; all 3 delivered.
6. **stats event fires on cadence.** Emit 5; advance by `stats_interval`;
   one `worker_sink.stats` event with `drain_count >= 5`,
   `blocking_emit_count == 0`.
7. **stats counters reset between intervals.** Emit, advance, assert;
   emit more, advance, assert second event has fresh counts.
8. **aclose drains pending then exits.** Emit 5 (some still queued);
   `aclose()`; all 5 reach inner before return.
9. **aclose blocks new emits.** After `aclose`, `emit` logs
   `worker_sink.closed_drop`, returns `False`, does not enqueue, does
   not raise.
10. **aclose timeout when inner stalls.** Stall inner; queue 3; `aclose`
    with 5s timeout via `FakeClock`; `worker_sink.shutdown_drain_timeout`
    with `remaining=3` log; function returns.

**Integration touch:**

- `tests/detectors/test_velocity.py` — should continue passing unchanged.
  Velocity's signature changes from `sink: AlertSink` → `sink: IAlertSink`
  but the existing fixture sink (already a stub) structurally satisfies
  both. Only update if `ty check` flags the annotation.
- Any Scanner wiring tests gain one assertion: a Scanner constructed with
  `velocity_worker_maxsize=N` builds a `WorkerSink` and starts the drain
  task. (Look for the existing wiring test pattern at implementation time;
  may not exist yet, in which case add one minimal smoke.)

**Smoke validation (post-implementation):**

Re-run a 1h smoke with the same shape as the 2026-04-27 baseline:

| metric | baseline | target |
|---|---:|---:|
| `tick_stream.subscriber_queue_full` | 498/h | 0 (≤5/h acceptable) |
| `worker_sink.stats` events | n/a | every 60s, `drain_count` ≈ velocity alert volume |
| `worker_sink.queue_full` | n/a | 0 under default `maxsize=4096` |
| velocity alert count | 253 | unchanged ±10% |

Alert counts on non-velocity detectors (cluster, smart_money, mispricing,
etc.) must be unchanged — they're not on the new path.

## Out of scope (separate spec when needed)

- **Pace `snapshot_once` publisher.** Inserting `await asyncio.sleep(0)`
  between per-asset publishes is a one-line change that addresses the
  burst at the source. Worth doing if `WorkerSink` adoption shows
  per-detector pressure persists, or as a complementary optimization. Not
  bundled here because it's orthogonal to the per-detector consumer-side
  primitive.
- **Profile per-tick velocity work.** If `worker_sink.stats` shows
  significant `blocking_emit_count` after this lands, profiling
  `evaluate()` is the next step (record_ms / window_ms / sink_emit_ms
  histograms).
- **Wrap polling detectors.** None show burst-driven backpressure today.
  Revisit only if a polling detector starts emitting alerts at burst
  rates.
- **Per-detector configurable `stats_interval`.** Single shared default is
  enough; per-detector tuning is YAGNI until a real diverge case appears.
