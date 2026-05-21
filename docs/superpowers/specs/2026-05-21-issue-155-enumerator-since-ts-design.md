# Issue #155: Plumb `since_ts` through the corpus enumerator

**Status:** design
**Issue:** [#155](https://github.com/jm709/pscanner/issues/155)
**Date:** 2026-05-21

## Problem

`pscanner corpus refresh` reads `corpus_state['last_gamma_sweep_ts']` and
passes it as `since_ts` to `enumerate_closed_markets`. The enumerator
ignores it (`del since_ts  # not yet used` at
`src/pscanner/corpus/enumerator.py:112`) and re-walks the entire gamma
closed-events catalog from offset 0 on every refresh.

Gamma's `/events` pagination caps at offset 10,100 (returns 422). Whether
a newly-closed market lands in the corpus on the next refresh depends on
gamma's default response order and how many older closed events sit in
front of it. Empirically: a 14-second refresh on a 21-day gap inserts
zero new markets, while 47 V2-era closed markets that did make it in
arrived via prior runs at a slow trickle.

Net effect: the V2 era (cutover 2026-04-28) is under-covered in the
corpus. The asset_index population that issue #155 calls for is
mechanically correct but starved of input rows.

## Decision

Add an `end_date_min` query parameter on gamma `/events` and forward
`since_ts` through to it. Server-side filter; pagination cap stops
biting because the filtered set is small.

### Probed filter support

Confirmed against `https://gamma-api.polymarket.com/events`:

| Param | Result |
|---|---|
| `end_date_min=2026-05-10T00:00:00Z` | **Filters as expected** — returns events with `endDate >= 2026-05-10`. Both ISO 8601 and unix integer accepted. |
| `endDateMin=...` (camelCase) | Silently ignored. |
| `close_date_min=...`, `closeStartTimestamp=...`, `closedAtMin=...` | Silently ignored. |
| `order=endDate&ascending=false` | Returned 10/10 events with `endDate=2028-01-01T05:00:00Z` placeholder — unusable. |
| `order=createdAt&ascending=false` | Filters by creation date, wrong axis. |

`end_date_min` (snake_case) is the only viable server-side filter
for this purpose.

## Components touched

- `src/pscanner/poly/gamma.py` — add `end_date_min: int | None = None`
  kwarg to `GammaClient.list_events` and `GammaClient.iter_events`.
  When non-`None`, format as ISO 8601 with `Z` suffix
  (`"2026-05-12T18:32:11Z"`) and add to the query-params dict.
- `src/pscanner/corpus/enumerator.py` — remove
  `del since_ts  # not yet used`. Pass `end_date_min=since_ts` to
  `gamma.iter_events(...)`.

No CLI surface change. The refresh path
(`_run_polymarket_refresh` in `src/pscanner/corpus/cli.py`) already
reads `last_gamma_sweep_ts` and passes it as `since_ts`.

## Data flow

1. `_run_polymarket_refresh` reads `last_gamma_sweep_ts` from
   `corpus_state` (unix int, written at the end of the previous
   refresh as `int(time.time())`).
2. Passes to `enumerate_closed_markets(..., since_ts=...)`.
3. Enumerator forwards as `end_date_min=since_ts` to
   `gamma.iter_events(...)`.
4. `iter_events` formats the int as ISO 8601 and adds
   `end_date_min=2026-05-12T18:32:11Z` to the `/events` query string.
5. Gamma returns only events whose `endDate >= since_ts`.

Fresh-corpus path: `state.get_int("last_gamma_sweep_ts")` returns
`None`. Enumerator receives `since_ts=None`. GammaClient sees
`end_date_min=None` and omits the param. Existing pre-fix behaviour
preserved.

## Error handling and edge cases

| Input | Behaviour |
|---|---|
| `since_ts=None` | No filter sent. Full-catalog scan (existing behaviour). Used on fresh corpora. |
| `since_ts=0` | Formatted as `1970-01-01T00:00:00Z`. Effective no-op filter. Safe. |
| `since_ts > now` | Gamma returns 0 events. Refresh completes with 0 inserts. Operator can reset `corpus_state['last_gamma_sweep_ts']` to unstick. Logged as `corpus.enumerated inserted=0` (existing). No new defensive clamp. |

### Documented gap (out of scope for v1)

Markets whose scheduled `endDate` is before `since_ts` but whose actual
UMA resolution happens after `since_ts` (late-resolved disputes) will
not be picked up. The current `del since_ts` behaviour catches them on
every refresh via the full re-scan; after this fix, they remain missed
until manual intervention.

This is intentional. The smoke run on 2026-05-21 surfaced a dozen
such markets as `resolution_disputed` warnings — already a small
minority. If the gap becomes measurable we can add a fixed slack
window (`end_date_min = since_ts - 7d`) or switch the watermark to
`MAX(corpus_markets.closed_at)`. Tracked as a future-revisit concern,
not blocking on issue #155.

## Testing

Three layers:

- **`GammaClient` unit tests** (`tests/poly/test_gamma.py`) —
  `list_events(end_date_min=N)` produces a URL containing
  `end_date_min=<iso>`. `list_events(end_date_min=None)` does not
  include the param at all. `iter_events` propagates the kwarg into
  `list_events`. Format: ISO 8601 with `Z` suffix, no microseconds.
- **Enumerator unit tests** (`tests/corpus/test_enumerator.py`) — when
  `enumerate_closed_markets` is called with `since_ts=N`, the mocked
  `GammaClient.iter_events` is invoked with `end_date_min=N`. When
  called with `since_ts=None`, `end_date_min` is `None`.
- **Live smoke on the desktop** — after merge: reset
  `corpus_state['last_gamma_sweep_ts']` to a known-old value (e.g.
  the timestamp of the second-most-recent enumeration in the existing
  corpus), run `pscanner corpus refresh`, assert at least one new
  market inserted and that the `data_client.get` debug log shows
  `end_date_min` in the query string.

## Out of scope (explicit)

- Volume gate changes. `VOLUME_GATE_BY_CATEGORY_USD` stays as is.
- Pagination order changes. Current order is unchanged; combined with
  the filter, the offset cap should no longer bite for incremental
  refreshes.
- `list_markets` / `iter_markets` plumbing. No current caller needs it.
- A `pscanner corpus backfill --since DATE` CLI flag.
- Late-resolve recovery (see *Documented gap* above).

## References

- Issue: [#155](https://github.com/jm709/pscanner/issues/155)
- Related: [#151](https://github.com/jm709/pscanner/issues/151) (subgraph migration)
- Related: [#152](https://github.com/jm709/pscanner/issues/152) (SubgraphTradeCollector daemon promotion)
- Implementation file: `src/pscanner/corpus/enumerator.py`
- Caller: `src/pscanner/corpus/cli.py` (`_run_polymarket_refresh`)
- Gamma client: `src/pscanner/poly/gamma.py`
