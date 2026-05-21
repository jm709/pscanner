# Issue #155: Plumb `since_ts` Through the Corpus Enumerator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pscanner corpus refresh` skip already-enumerated closed events by passing `since_ts` to gamma as a server-side `end_date_min` filter, so V2-era and other recently-closed markets actually land in the corpus on a fresh refresh.

**Architecture:** Add `end_date_min: int | None = None` to `GammaClient.list_events` and `GammaClient.iter_events`; format as ISO 8601 with `Z` suffix at the HTTP boundary; remove the `del since_ts` no-op in `enumerate_closed_markets` and forward the value to `iter_events`. No CLI surface change; the refresh path already reads `last_gamma_sweep_ts` and supplies `since_ts`.

**Tech Stack:** Python 3.13, httpx, pydantic, pytest (async). Test fakes via `unittest.mock.AsyncMock` / `MagicMock`. Production lint/type gates: `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`.

**Spec:** `docs/superpowers/specs/2026-05-21-issue-155-enumerator-since-ts-design.md`

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/pscanner/poly/gamma.py` | modify | Add `end_date_min` kwarg to `list_events` + `iter_events`; format as ISO 8601 + Z at the params boundary. |
| `src/pscanner/corpus/enumerator.py` | modify | Remove `del since_ts`; forward to `gamma.iter_events(end_date_min=since_ts, ...)`. |
| `tests/poly/test_gamma.py` | extend | Two new tests: param appears in URL when non-None; absent when None. Plus one `iter_events`-level test confirming propagation. |
| `tests/corpus/test_enumerator.py` | extend | Two new tests: `since_ts=N` → `end_date_min=N` reaches gamma; `since_ts=None` → `end_date_min=None`. |

---

## Task 1: `list_events` accepts and forwards `end_date_min`

**Files:**
- Modify: `src/pscanner/poly/gamma.py:42-68` (the `list_events` method)
- Test: `tests/poly/test_gamma.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/poly/test_gamma.py` (after the existing `test_list_events_passes_filter_params` at line 86-95):

```python
async def test_list_events_passes_end_date_min_as_iso_z() -> None:
    """``end_date_min`` (unix int) is formatted as ISO 8601 with Z suffix."""
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    # 1_778_976_000 == 2026-05-17T00:00:00Z (midnight UTC).
    await client.list_events(end_date_min=1_778_976_000)

    http.get.assert_awaited_once_with(
        "/events",
        params={
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": 0,
            "end_date_min": "2026-05-17T00:00:00Z",
        },
    )


async def test_list_events_omits_end_date_min_when_none() -> None:
    """``end_date_min=None`` (the default) must not appear in the URL params."""
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    await client.list_events(end_date_min=None)

    http.get.assert_awaited_once_with(
        "/events",
        params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/poly/test_gamma.py::test_list_events_passes_end_date_min_as_iso_z tests/poly/test_gamma.py::test_list_events_omits_end_date_min_when_none -v`

Expected: FAIL — `TypeError: list_events() got an unexpected keyword argument 'end_date_min'`.

- [ ] **Step 3: Add the kwarg + ISO formatting helper**

In `src/pscanner/poly/gamma.py`, modify the `list_events` method. The existing signature ends at line 49 with `offset: int = 0,`. Add `end_date_min: int | None = None,` as the last keyword-only param. In the body (currently `params = {...}` at line 61), add the filter conditionally.

The full updated `list_events` should read:

```python
    async def list_events(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        end_date_min: int | None = None,
    ) -> list[Event]:
        """Fetch one page of events matching the filters.

        Args:
            active: Restrict to currently-active events.
            closed: Include closed events.
            limit: Page size (server-capped at 500).
            offset: Pagination offset.
            end_date_min: Filter to events whose ``endDate >= end_date_min``
                (unix seconds). When ``None`` (default) no filter is sent.
                Used by the corpus refresh path to skip events already
                enumerated on prior runs.

        Returns:
            A list of validated ``Event`` models (possibly empty).
        """
        params: dict[str, Any] = {
            "active": _bool_param(active),
            "closed": _bool_param(closed),
            "limit": limit,
            "offset": offset,
        }
        if end_date_min is not None:
            params["end_date_min"] = _format_end_date_min(end_date_min)
        payload = await self._http.get("/events", params=params)
        return _parse_list(payload, Event)
```

Then add the formatter helper near the top of the module (after the `_bool_param` helper, around line 17):

```python
def _format_end_date_min(unix_seconds: int) -> str:
    """Render a unix timestamp as ISO 8601 with ``Z`` suffix (UTC, no microseconds).

    Gamma's ``/events?end_date_min=`` accepts both ISO and unix; we use ISO
    for readable URLs in logs and error messages.
    """
    return datetime.fromtimestamp(unix_seconds, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
```

Add the missing imports at the top of the module (after the existing `from typing import Any`):

```python
from datetime import UTC, datetime
```

Note the `params: dict[str, Any]` annotation on the existing dict literal — required because the new `end_date_min` entry is `str` while `limit` / `offset` are `int`. Annotate as `dict[str, Any]` to satisfy `ty`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/poly/test_gamma.py -v`

Expected: All `test_list_events_*` tests pass (existing + the two new ones). 17 total tests in this file should still pass.

- [ ] **Step 5: Run lint + type checks**

Run: `uv run ruff check src/pscanner/poly/gamma.py tests/poly/test_gamma.py && uv run ruff format --check src/pscanner/poly/gamma.py tests/poly/test_gamma.py && uv run ty check src/pscanner/poly/gamma.py`

Expected: All pass. If `ruff format --check` fails, run `uv run ruff format src/pscanner/poly/gamma.py` and re-check.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/poly/gamma.py tests/poly/test_gamma.py
git commit -m "$(cat <<'EOF'
feat(poly): add end_date_min filter to GammaClient.list_events (#155)

Forward an optional unix timestamp as the gamma /events `end_date_min` query
param, formatted as ISO 8601 with Z suffix. Used by the corpus enumerator to
skip events already seen on prior refreshes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `iter_events` propagates `end_date_min` to every page

**Files:**
- Modify: `src/pscanner/poly/gamma.py:70-101` (the `iter_events` method)
- Test: `tests/poly/test_gamma.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/poly/test_gamma.py`:

```python
async def test_iter_events_forwards_end_date_min_to_every_page() -> None:
    """``iter_events`` must pass ``end_date_min`` through to every ``list_events`` call."""
    fixture = _load_event_fixture()
    # Two pages so we verify the param is on BOTH list_events calls.
    page1 = [fixture, fixture]
    page2 = [fixture]  # short page → terminates after yielding
    http = _mock_http_pages(page1, page2)
    client = GammaClient(http=http)

    collected = [
        event async for event in client.iter_events(page_size=2, end_date_min=1_778_976_000)
    ]

    assert len(collected) == 3
    assert http.get.await_count == 2
    for call in http.get.await_args_list:
        assert call.kwargs["params"]["end_date_min"] == "2026-05-17T00:00:00Z"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/poly/test_gamma.py::test_iter_events_forwards_end_date_min_to_every_page -v`

Expected: FAIL — `TypeError: iter_events() got an unexpected keyword argument 'end_date_min'`.

- [ ] **Step 3: Add the kwarg to `iter_events`**

Modify `iter_events` in `src/pscanner/poly/gamma.py`. Add the kwarg and forward to `list_events`. The full updated method:

```python
    async def iter_events(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        page_size: int = 100,
        end_date_min: int | None = None,
    ) -> AsyncIterator[Event]:
        """Async-iterate every event matching the filters across all pages.

        Args:
            active: Restrict to currently-active events.
            closed: Include closed events.
            page_size: Page size sent to the server per request.
            end_date_min: Filter to events whose ``endDate >= end_date_min``
                (unix seconds). Forwarded to every ``list_events`` page call.

        Yields:
            Each ``Event`` exactly once until the catalogue is exhausted.
        """
        offset = 0
        while True:
            page = await self.list_events(
                active=active,
                closed=closed,
                limit=page_size,
                offset=offset,
                end_date_min=end_date_min,
            )
            if not page:
                return
            for event in page:
                yield event
            if len(page) < page_size:
                return
            offset += page_size
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/poly/test_gamma.py -v`

Expected: All tests pass.

- [ ] **Step 5: Run lint + type checks**

Run: `uv run ruff check src/pscanner/poly/gamma.py tests/poly/test_gamma.py && uv run ruff format --check src/pscanner/poly/gamma.py tests/poly/test_gamma.py && uv run ty check src/pscanner/poly/gamma.py`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/poly/gamma.py tests/poly/test_gamma.py
git commit -m "$(cat <<'EOF'
feat(poly): forward end_date_min through GammaClient.iter_events (#155)

Pass the new gamma /events `end_date_min` filter to every page call so the
filter applies across the full async iteration, not just the first page.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Enumerator forwards `since_ts` to gamma

**Files:**
- Modify: `src/pscanner/corpus/enumerator.py:87-128` (the `enumerate_closed_markets` function — specifically the `del since_ts` no-op at line 112 and the `gamma.iter_events(...)` call at line 115)
- Test: `tests/corpus/test_enumerator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/corpus/test_enumerator.py` (after the existing tests; preserve all existing imports and helpers):

```python
@pytest.mark.asyncio
async def test_enumerate_forwards_since_ts_as_end_date_min(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """``enumerate_closed_markets(since_ts=N)`` must call ``iter_events(end_date_min=N)``."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    captured_kwargs: dict[str, object] = {}

    def _capture_iter_events(**kwargs: object) -> AsyncIterator[Event]:
        captured_kwargs.update(kwargs)
        return _async_events([])

    stub = MagicMock()
    stub.iter_events = _capture_iter_events

    await enumerate_closed_markets(
        gamma=stub, repo=repo, now_ts=1_000, since_ts=1_779_000_000
    )

    assert captured_kwargs.get("end_date_min") == 1_779_000_000


@pytest.mark.asyncio
async def test_enumerate_passes_none_when_since_ts_is_none(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Fresh-corpus path: ``since_ts=None`` must yield ``end_date_min=None``."""
    repo = CorpusMarketsRepo(tmp_corpus_db)
    captured_kwargs: dict[str, object] = {}

    def _capture_iter_events(**kwargs: object) -> AsyncIterator[Event]:
        captured_kwargs.update(kwargs)
        return _async_events([])

    stub = MagicMock()
    stub.iter_events = _capture_iter_events

    await enumerate_closed_markets(
        gamma=stub, repo=repo, now_ts=1_000, since_ts=None
    )

    assert captured_kwargs.get("end_date_min") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/corpus/test_enumerator.py::test_enumerate_forwards_since_ts_as_end_date_min tests/corpus/test_enumerator.py::test_enumerate_passes_none_when_since_ts_is_none -v`

Expected: FAIL — assertion error on `captured_kwargs.get("end_date_min")` because the current enumerator does `del since_ts` and calls `gamma.iter_events(active=False, closed=True, page_size=100)` with no `end_date_min`.

- [ ] **Step 3: Remove the `del since_ts` and forward the value**

Modify `src/pscanner/corpus/enumerator.py`. Replace the body of `enumerate_closed_markets` (lines 112-128 in the current source) so that:

1. The `del since_ts  # not yet used; gamma /events doesn't expose a precise close ts` line is removed.
2. The `iter_events` call passes `end_date_min=since_ts`.
3. The docstring's `Args.since_ts` description is updated to reflect that it's now used.

The full updated function (only the body and docstring change; signature is unchanged):

```python
async def enumerate_closed_markets(
    *,
    gamma: GammaClient,
    repo: CorpusMarketsRepo,
    now_ts: int,
    since_ts: int | None,
) -> int:
    """Walk gamma closed events; insert qualifying markets as ``pending``.

    A ``5xx`` from gamma during pagination is treated as the end of the
    catalog and logged at warn-level. Polymarket's ``/events`` endpoint
    returns ``500`` past a deep offset (mirroring the documented
    ``400`` cap on ``/trades``), so this lets enumeration finish cleanly
    on whatever pages did succeed rather than aborting the whole run.

    Args:
        gamma: Gamma client with ``iter_events``.
        repo: Markets repo to insert into.
        now_ts: Unix seconds at enumeration time (recorded on rows).
        since_ts: Lower bound (unix seconds) on event ``endDate``.
            Forwarded to gamma as ``end_date_min`` so the server elides
            already-seen events. ``None`` (fresh corpus or reset state)
            disables the filter and walks the full closed catalog.

    Returns:
        Count of markets actually inserted (excluding duplicates).
    """
    inserted = 0
    try:
        async for event in gamma.iter_events(
            active=False,
            closed=True,
            page_size=100,
            end_date_min=since_ts,
        ):
            for corpus in _qualifying_markets(event, now_ts):
                inserted += repo.insert_pending(corpus)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status != _DEEP_OFFSET_STATUS and status < _HTTP_SERVER_ERROR_FLOOR:
            raise
        _log.warning(
            "corpus.enumerate_pagination_capped",
            status=status,
            url=str(exc.request.url),
        )
    _log.info("corpus.enumerated", inserted=inserted, since_ts=since_ts)
    return inserted
```

Note: the `_log.info("corpus.enumerated", ...)` call now also carries `since_ts` for operator visibility into which filter was applied. Existing log keys are preserved.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/corpus/test_enumerator.py -v`

Expected: All `test_enumerate_*` tests pass — the existing 8 tests (which pass `since_ts=None`) plus the 2 new ones.

- [ ] **Step 5: Run lint + type checks**

Run: `uv run ruff check src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py && uv run ruff format --check src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py && uv run ty check src/pscanner/corpus/enumerator.py`

Expected: All pass.

- [ ] **Step 6: Run the full project lint + type + tests gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`

Expected: All pass. This is the project's standard verification gate (per CLAUDE.md `Quick verify`). It catches any incidental fallout from the change.

- [ ] **Step 7: Commit**

```bash
git add src/pscanner/corpus/enumerator.py tests/corpus/test_enumerator.py
git commit -m "$(cat <<'EOF'
feat(corpus): forward since_ts to gamma as end_date_min (#155)

Remove the `del since_ts` no-op in enumerate_closed_markets and forward the
value to gamma.iter_events as the new end_date_min filter. The refresh path
now elides already-enumerated events server-side, so V2-era closes that fell
behind gamma's 10,100-offset pagination cap will land in the corpus on the
next refresh.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Live verification on the desktop

**Files:** None modified. Operational verification only.

The desktop already has the corpus at `~/projects/polymarketscanner/pscanner/data/corpus.sqlite3` and the up-to-date branch (per LOCAL_NOTES.md SSH idioms). After Tasks 1-3 are pushed, this task pulls them onto the desktop and runs a smoke that:

1. Captures the current `last_gamma_sweep_ts` value.
2. Resets `last_gamma_sweep_ts` to a known-old value (the second-most-recent enumeration timestamp from `corpus_markets`).
3. Runs `pscanner corpus refresh`.
4. Asserts at least one new market was inserted.
5. Confirms the gamma query string included `end_date_min` by checking debug logs.
6. Restores the original `last_gamma_sweep_ts` if no new markets were inserted (so a no-op rerun on next refresh is preserved); otherwise leaves it at the new sweep time (refresh sets it as part of its success path).

**Append-only:** the smoke writes new rows via `INSERT OR IGNORE` and updates one `corpus_state` row. No DELETEs, no DROPs.

- [ ] **Step 1: Push the branch and pull on desktop**

From the laptop:

```bash
git push -u origin feat/issue-155-enumerator-since-ts
```

Then on the desktop (single SSH command):

```bash
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner \
   && export PATH="$HOME/.local/bin:$PATH" \
   && git fetch origin \
   && git checkout feat/issue-155-enumerator-since-ts \
   && git pull --ff-only origin feat/issue-155-enumerator-since-ts \
   && uv sync 2>&1 | tail -5 \
   && git log -1 --oneline'
```

Expected: prints the commit hash of `feat(corpus): forward since_ts to gamma as end_date_min (#155)`.

- [ ] **Step 2: Write the smoke script locally and copy to the desktop**

Create `/tmp/smoke_since_ts.py` on the laptop:

```python
"""Smoke for issue #155: verify enumerator now forwards since_ts to gamma.

Resets corpus_state['last_gamma_sweep_ts'] to a known-old value, runs the
refresh, asserts at least one new market lands. Append-only: writes go via
INSERT OR IGNORE on corpus_markets, plus one UPDATE on corpus_state.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import CorpusStateRepo

_DB = Path("data/corpus.sqlite3")


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


async def main() -> int:
    if not _DB.exists():
        print(f"ERROR: {_DB} not found", file=sys.stderr)
        return 1

    conn = init_corpus_db(_DB)
    state = CorpusStateRepo(conn)

    # Step A: snapshot pre-state
    before_markets = _scalar(conn, "SELECT COUNT(*) FROM corpus_markets")
    original_since = state.get_int("last_gamma_sweep_ts")
    print(f"before: corpus_markets={before_markets:,}")
    print(f"before: last_gamma_sweep_ts={original_since} "
          f"({datetime.fromtimestamp(original_since, tz=UTC).isoformat() if original_since else '-'})")

    # Step B: pick a known-old value: the 2nd-most-recent enumerated_at
    row = conn.execute(
        "SELECT DISTINCT enumerated_at FROM corpus_markets "
        "ORDER BY enumerated_at DESC LIMIT 2"
    ).fetchall()
    if len(row) < 2:
        print("ERROR: corpus has fewer than 2 distinct enumerated_at values")
        return 1
    target_since = int(row[1][0])
    now_ts = int(time.time())
    print(f"setting last_gamma_sweep_ts -> {target_since} "
          f"({datetime.fromtimestamp(target_since, tz=UTC).isoformat()})")
    state.set("last_gamma_sweep_ts", str(target_since), updated_at=now_ts)
    conn.close()

    # Step C: run refresh in-process via the CLI entry
    # We invoke the CLI directly because that's the codepath the operator uses.
    import subprocess
    print()
    print("=== running: pscanner corpus refresh ===")
    proc = subprocess.run(
        ["uv", "run", "pscanner", "corpus", "refresh"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("stdout:", proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
    print("stderr:", proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
    if proc.returncode != 0:
        print(f"ERROR: refresh exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    # Step D: post-state
    conn = init_corpus_db(_DB)
    after_markets = _scalar(conn, "SELECT COUNT(*) FROM corpus_markets")
    state = CorpusStateRepo(conn)
    new_since = state.get_int("last_gamma_sweep_ts")
    delta = after_markets - before_markets
    print()
    print(f"after: corpus_markets={after_markets:,}  (delta +{delta:,})")
    print(f"after: last_gamma_sweep_ts={new_since} "
          f"({datetime.fromtimestamp(new_since, tz=UTC).isoformat() if new_since else '-'})")

    # Verdict
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if delta > 0:
        print(f"PASS: refresh inserted {delta:,} new markets after the fix.")
        result = 0
    else:
        print("FAIL: refresh still inserted 0 markets. Restoring original sweep_ts.")
        if original_since is not None:
            state.set("last_gamma_sweep_ts", str(original_since), updated_at=now_ts)
        result = 1

    # Sanity: also report the gamma log line containing end_date_min, if present.
    # The structlog output goes to stderr; both proc.stdout and proc.stderr
    # were already printed above. Grep them defensively here.
    full_log = (proc.stdout or "") + (proc.stderr or "")
    if "end_date_min" in full_log:
        print("OK: 'end_date_min' visible in refresh logs (filter was applied).")
    else:
        print("NOTE: 'end_date_min' not in captured logs — logs may have been filtered "
              "to non-debug. The behavioural assertion (delta>0) is the load-bearing check.")
    conn.close()
    return result


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Then copy it to the desktop:

```bash
scp -P 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /tmp/smoke_since_ts.py \
  macph@10.0.0.143:/tmp/smoke_since_ts.py
```

- [ ] **Step 3: Run the smoke on the desktop**

```bash
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner \
   && export PATH="$HOME/.local/bin:$PATH" \
   && uv run python /tmp/smoke_since_ts.py'
```

Expected:
- `corpus_markets` delta > 0 (at least one new market inserted; on a 12-day-old since_ts this should be modest — single to low double digits, given the volume gate).
- `VERDICT: PASS`.
- The refresh log contains `corpus.enumerated inserted=N since_ts=...` where N > 0 and `since_ts` is the target value we set.
- `last_gamma_sweep_ts` updated to current time (this is the refresh's normal end-of-run behaviour).

If the smoke fails (delta=0), do not commit anything. Investigate by:
1. Re-running the refresh with explicit debug logging: `LOG_LEVEL=DEBUG uv run pscanner corpus refresh 2>&1 | grep -E 'enumerated|end_date_min'`
2. Manually probing gamma to confirm `end_date_min` still works (the probe in the spec):
   ```bash
   curl -s 'https://gamma-api.polymarket.com/events?closed=true&limit=3&end_date_min=2026-05-10T00:00:00Z' | python -m json.tool | head -20
   ```
3. Re-reading `src/pscanner/corpus/enumerator.py` to confirm `since_ts` is forwarded.

- [ ] **Step 4: Push and open the PR**

After the smoke PASSes, from the laptop:

```bash
git push -u origin feat/issue-155-enumerator-since-ts
gh pr create --title "feat(corpus): plumb since_ts through enumerator to gamma /events (#155)" --body "$(cat <<'EOF'
## Summary

- Adds an `end_date_min` query param on gamma `/events`, exposed as a kwarg on `GammaClient.list_events` and `GammaClient.iter_events`.
- Removes the `del since_ts  # not yet used` no-op in `enumerate_closed_markets` and forwards the value through.
- The refresh path (`_run_polymarket_refresh`) now skips events already enumerated on prior runs, so V2-era closes that fell behind gamma's 10,100-offset pagination cap will land in the corpus on the next refresh.

Spec: `docs/superpowers/specs/2026-05-21-issue-155-enumerator-since-ts-design.md`
Plan: `docs/superpowers/plans/2026-05-21-issue-155-enumerator-since-ts.md`

## Test plan

- [x] Unit: `tests/poly/test_gamma.py` — three new tests cover the kwarg present, kwarg absent, and propagation across pages.
- [x] Unit: `tests/corpus/test_enumerator.py` — two new tests verify `since_ts → end_date_min` forwarding.
- [x] Project gate: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q` clean.
- [x] Live smoke on desktop corpus (59 GB / 4.7k markets) — refresh with stale `last_gamma_sweep_ts` inserted N>0 new markets.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL is returned. CI runs the same gates locally executed.

---

## Done criteria

- [ ] Three commits on `feat/issue-155-enumerator-since-ts`:
  1. `feat(poly): add end_date_min filter to GammaClient.list_events (#155)`
  2. `feat(poly): forward end_date_min through GammaClient.iter_events (#155)`
  3. `feat(corpus): forward since_ts to gamma as end_date_min (#155)`
- [ ] All project lint / type / test gates pass.
- [ ] Live smoke on the desktop inserts > 0 new markets (proves the fix works against real gamma).
- [ ] PR opened and linked to issue #155.
