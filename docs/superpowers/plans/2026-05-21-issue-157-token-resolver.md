# Issue #157: Just-in-time token resolver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve subgraph `tokenId`s on live open markets by falling back to gamma `/markets?clob_token_ids=<id>` when the local `asset_index` / `market_cache` lookup misses. Upserts both tables on a gamma hit so the next sighting is local-only.

**Architecture:** New `pscanner.poly.token_resolver` module with one async function `resolve_token(...)` and one `ResolvedToken` dataclass. The script's existing sync `_resolve_token` is replaced by an `await` into the new helper. Negative caching (gamma returns nothing) is out of scope — log + skip.

**Tech Stack:** Python 3.13, pytest async, httpx (already in tree), gamma's REST surface verified live during PR #155.

**Spec:** Issue [#157](https://github.com/jm709/pscanner/issues/157).

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/pscanner/poly/gamma.py` | modify | Add `clob_token_ids: AssetId \| None = None` kwarg to `GammaClient.list_markets` (mirrors the `end_date_min` pattern from #155 on `list_events`). |
| `tests/poly/test_gamma.py` | extend | Two tests: kwarg appears in URL when set; absent when None. |
| `src/pscanner/poly/token_resolver.py` | **create** | Async `resolve_token(...)` + `ResolvedToken` dataclass. Falls back to gamma on local miss, upserts both tables, returns the resolved tuple. |
| `tests/poly/test_token_resolver.py` | **create** | Local-hit, gamma-fallback-hit, gamma-empty, gamma-indexing-drift cases. |
| `scripts/watch_subgraph_copy.py` | modify | Delete the inline `_resolve_token` + `_ResolvedToken`; import from the new module. Make `_pre_flight` async (one keyword change at definition + one at call site). Inject a `GammaClient` instance into `_run_one_cycle` so the resolver can use it. |

---

## Task 1: Extend `GammaClient.list_markets` with `clob_token_ids`

**Files:**
- Modify: `src/pscanner/poly/gamma.py` (the `list_markets` method)
- Test: `tests/poly/test_gamma.py`

Mirrors the `end_date_min` pattern from PR #155 on `list_events` — same shape, same convention. Needed so the resolver in Task 2 can call a public method instead of poking `gamma._http.get(...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/poly/test_gamma.py`:

```python
async def test_list_markets_passes_clob_token_ids() -> None:
    """``clob_token_ids`` (string) appears in the URL params when set."""
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    await client.list_markets(clob_token_ids="12345")

    http.get.assert_awaited_once_with(
        "/markets",
        params={
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": 0,
            "clob_token_ids": "12345",
        },
    )


async def test_list_markets_omits_clob_token_ids_when_none() -> None:
    """``clob_token_ids=None`` (the default) must not appear in the URL params."""
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    await client.list_markets(clob_token_ids=None)

    http.get.assert_awaited_once_with(
        "/markets",
        params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
    )
```

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/poly/test_gamma.py::test_list_markets_passes_clob_token_ids tests/poly/test_gamma.py::test_list_markets_omits_clob_token_ids_when_none -v`

Expected: FAIL — `TypeError: list_markets() got an unexpected keyword argument 'clob_token_ids'`.

- [ ] **Step 3: Add the kwarg**

In `src/pscanner/poly/gamma.py`, modify `list_markets` (around line 145). The new method body:

```python
    async def list_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        clob_token_ids: str | None = None,
    ) -> list[Market]:
        """Fetch one page of markets matching the filters.

        Args:
            active: Restrict to currently-active markets.
            closed: Include closed markets.
            limit: Page size (server-capped).
            offset: Pagination offset.
            clob_token_ids: Filter to markets containing this CTF token id
                (decimal string). When ``None`` (default) no filter is sent.
                Used by the just-in-time token resolver to look up a market
                by one of its outcome tokens.

        Returns:
            A list of validated ``Market`` models (possibly empty).
        """
        params: dict[str, Any] = {
            "active": _bool_param(active),
            "closed": _bool_param(closed),
            "limit": limit,
            "offset": offset,
        }
        if clob_token_ids is not None:
            params["clob_token_ids"] = clob_token_ids
        payload = await self._http.get("/markets", params=params)
        return _parse_list(payload, Market)
```

If `params` was previously not annotated as `dict[str, Any]`, add that annotation now (same reason as in #155 — mixing `str` / `int` values).

- [ ] **Step 4: Run tests to verify they pass**

`uv run pytest tests/poly/test_gamma.py -v`

Expected: All tests pass.

- [ ] **Step 5: Lint + type checks**

`uv run ruff check src/pscanner/poly/gamma.py tests/poly/test_gamma.py && uv run ruff format --check src/pscanner/poly/gamma.py tests/poly/test_gamma.py && uv run ty check src/pscanner/poly/gamma.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/poly/gamma.py tests/poly/test_gamma.py
git commit -m "$(cat <<'EOF'
feat(poly): add clob_token_ids filter to GammaClient.list_markets (#157)

Mirror the end_date_min kwarg pattern from #155: optional kwarg, formatted
into the /markets query string only when set. Used by the just-in-time
token resolver (also in #157) to look up a market by one of its outcome
tokens.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `pscanner.poly.token_resolver` (TDD)

**Files:**
- Create: `src/pscanner/poly/token_resolver.py`
- Test: `tests/poly/test_token_resolver.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/poly/test_token_resolver.py`:

```python
"""Tests for ``pscanner.poly.token_resolver.resolve_token``."""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.models import Market
from pscanner.poly.token_resolver import resolve_token
from pscanner.store.db import init_db
from pscanner.store.repo import MarketCacheRepo

_CID = "0x6622b219b4b0796651304f85f279b18797edd2a818be471ecd82dd2d7d8d4ac7"
_TOKEN_YES = "52361899659273746688310711154377414983286902715574343626305571157053514347311"
_TOKEN_NO = "42619408858710407426561352918056373492522898259895604902369103847017159614170"


def _gamma_market_payload(*, condition_id: str = _CID) -> dict[str, Any]:
    """Build a gamma /markets payload matching the real wire shape."""
    return {
        "id": "2289432",
        "conditionId": condition_id,
        "question": "Cavaliers vs. Knicks",
        "slug": "nba-cle-nyk-2026-05-21",
        "outcomes": '["Cavaliers", "Knicks"]',
        "outcomePrices": '["0.355", "0.645"]',
        "clobTokenIds": f'["{_TOKEN_YES}", "{_TOKEN_NO}"]',
        "active": True,
        "closed": False,
    }


@pytest.fixture()
def corpus_conn(tmp_path: Any) -> sqlite3.Connection:
    return init_corpus_db(tmp_path / "corpus.sqlite3")


@pytest.fixture()
def daemon_conn(tmp_path: Any) -> sqlite3.Connection:
    return init_db(tmp_path / "pscanner.sqlite3")


def _make_gamma_returning(markets: list[Market]) -> AsyncMock:
    """Build an ``AsyncMock`` for ``GammaClient`` whose ``list_markets`` returns ``markets``."""
    gamma = AsyncMock(spec=GammaClient)
    gamma.list_markets.return_value = markets
    return gamma


@pytest.mark.asyncio
async def test_resolve_token_local_hit_no_gamma_call(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Local hit on both tables: gamma must not be touched."""
    asset_index = AssetIndexRepo(corpus_conn)
    asset_index.upsert(
        AssetEntry(
            asset_id=_TOKEN_NO,
            condition_id=_CID,
            outcome_side="NO",
            outcome_index=1,
        )
    )
    market_cache = MarketCacheRepo(daemon_conn)
    market_cache.upsert(Market.model_validate(_gamma_market_payload()))
    gamma = _make_gamma_returning([])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    assert result is not None
    assert result.condition_id == ConditionId(_CID)
    assert result.asset_id == AssetId(_TOKEN_NO)
    assert result.outcome_name == "Knicks"
    assert result.outcome_index == 1
    gamma.list_markets.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_token_gamma_fallback_populates_both_tables(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Local miss + gamma hit: both tables upserted, resolver returns correct tuple."""
    asset_index = AssetIndexRepo(corpus_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    gamma_market = Market.model_validate(_gamma_market_payload())
    gamma = _make_gamma_returning([gamma_market])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    # Resolver returned the right tuple.
    assert result is not None
    assert result.condition_id == ConditionId(_CID)
    assert result.outcome_name == "Knicks"
    assert result.outcome_index == 1

    # Asset_index now has both YES (idx 0) and NO (idx 1) rows.
    yes_entry = asset_index.get(_TOKEN_YES)
    no_entry = asset_index.get(_TOKEN_NO)
    assert yes_entry is not None and yes_entry.outcome_side == "YES" and yes_entry.outcome_index == 0
    assert no_entry is not None and no_entry.outcome_side == "NO" and no_entry.outcome_index == 1

    # Market_cache has the market.
    cached = market_cache.get_by_condition_id(ConditionId(_CID))
    assert cached is not None
    assert cached.condition_id == ConditionId(_CID)
    assert cached.outcomes == ["Cavaliers", "Knicks"]

    # Exactly one gamma call with the queried token id.
    gamma.list_markets.assert_awaited_once_with(clob_token_ids=_TOKEN_NO, limit=5)


@pytest.mark.asyncio
async def test_resolve_token_gamma_empty_returns_none(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Local miss + gamma 0-results: returns None, no DB writes."""
    asset_index = AssetIndexRepo(corpus_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    gamma = _make_gamma_returning([])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    assert result is None
    # No DB writes.
    assert asset_index.get(_TOKEN_NO) is None
    assert market_cache.get_by_condition_id(ConditionId(_CID)) is None


@pytest.mark.asyncio
async def test_resolve_token_gamma_payload_missing_token_returns_none(
    corpus_conn: sqlite3.Connection, daemon_conn: sqlite3.Connection
) -> None:
    """Defensive: gamma returns a Market whose clob_token_ids don't include the queried token.

    Would indicate a gamma indexing inconsistency. Resolver returns None and does NOT write
    an incorrect mapping into asset_index for the queried token.
    """
    asset_index = AssetIndexRepo(corpus_conn)
    market_cache = MarketCacheRepo(daemon_conn)
    payload = _gamma_market_payload()
    payload["clobTokenIds"] = '["1111", "2222"]'  # neither matches _TOKEN_NO
    gamma = _make_gamma_returning([Market.model_validate(payload)])

    result = await resolve_token(
        token_id=AssetId(_TOKEN_NO),
        asset_index=asset_index,
        market_cache=market_cache,
        gamma=gamma,
    )

    assert result is None
    assert asset_index.get(_TOKEN_NO) is None
```

- [ ] **Step 2: Run tests to verify they fail**

`uv run pytest tests/poly/test_token_resolver.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'pscanner.poly.token_resolver'`.

- [ ] **Step 3: Create `src/pscanner/poly/token_resolver.py`**

```python
"""Just-in-time resolver: tokenId → ResolvedToken via local tables + gamma fallback.

Local-first: try ``AssetIndexRepo.get(tokenId)`` + ``MarketCacheRepo.get_by_condition_id``.
On miss, call gamma ``/markets?clob_token_ids=<tokenId>`` exactly once, upsert both
tables, then resolve from the just-written rows.

Used by:
  - ``scripts/watch_subgraph_copy.py`` (the subgraph copy-trader)
  - ``pscanner.collectors.subgraph_trades.SubgraphTradeCollector`` (issue #152, pending)

Out of scope: negative caching for tokens gamma can't resolve. If the daemon sees
the same unresolvable token on every poll, the log volume signals a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import AssetEntry, AssetIndexRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId, ConditionId
from pscanner.poly.models import Market
from pscanner.store.repo import MarketCacheRepo

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ResolvedToken:
    """Result of resolving a tokenId to its market + outcome position.

    ``condition_id`` is the on-chain conditional-token id; ``outcome_index`` is
    the position in the market's parallel ``outcomes`` / ``clob_token_ids``
    arrays (0 = YES-equivalent, 1 = NO-equivalent per Polymarket convention).
    """

    condition_id: ConditionId
    asset_id: AssetId
    outcome_name: str
    outcome_index: int


async def resolve_token(
    *,
    token_id: AssetId,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
    gamma: GammaClient,
) -> ResolvedToken | None:
    """Resolve ``token_id`` to a ``ResolvedToken``, falling back to gamma on local miss.

    Lookup order:
        1. ``AssetIndexRepo.get(token_id)`` for ``(condition_id, outcome_side)``.
        2. ``MarketCacheRepo.get_by_condition_id`` for the cached ``Market``.
        3. If either misses, call gamma ``/markets?clob_token_ids=<token_id>``,
           upsert both tables, then re-resolve from the freshly-written rows.

    Args:
        token_id: Subgraph ``tokenId`` (== Polymarket CTF position id, decimal string).
        asset_index: Corpus-side asset_id → condition_id index.
        market_cache: Daemon-side market metadata cache.
        gamma: Gamma client for the on-miss fallback.

    Returns:
        ``ResolvedToken`` on success. ``None`` if gamma returns 0 markets for the
        token id, or if gamma returns a market whose ``clob_token_ids`` don't
        include the queried token (defensive — would indicate gamma indexing drift).
    """
    cached_market = _try_local(token_id, asset_index, market_cache)
    if cached_market is not None:
        return cached_market

    market = await _fetch_market_from_gamma(token_id, gamma)
    if market is None:
        return None

    return _upsert_and_resolve(token_id, market, asset_index, market_cache)


def _try_local(
    token_id: AssetId,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
) -> ResolvedToken | None:
    """Local lookup. Returns None on any miss (asset_index OR market_cache OR position check)."""
    entry = asset_index.get(token_id)
    if entry is None:
        return None
    condition_id = ConditionId(entry.condition_id)
    cached = market_cache.get_by_condition_id(condition_id)
    if cached is None:
        return None
    try:
        idx = cached.asset_ids.index(AssetId(token_id))
    except ValueError:
        return None
    return ResolvedToken(
        condition_id=condition_id,
        asset_id=AssetId(token_id),
        outcome_name=cached.outcomes[idx],
        outcome_index=idx,
    )


async def _fetch_market_from_gamma(token_id: AssetId, gamma: GammaClient) -> Market | None:
    """One-shot gamma fetch by ``clob_token_ids``. Returns the first match or None."""
    matches = await gamma.list_markets(clob_token_ids=str(token_id), limit=5)
    if not matches:
        return None
    # Take the first match. Gamma returns at most a handful of markets per
    # clob_token_ids query; in practice it's always 1.
    return matches[0]


def _upsert_and_resolve(
    token_id: AssetId,
    market: Market,
    asset_index: AssetIndexRepo,
    market_cache: MarketCacheRepo,
) -> ResolvedToken | None:
    """Write the gamma Market into both tables, then return the resolved tuple.

    Returns None if the queried ``token_id`` is not present in
    ``market.clob_token_ids`` (defensive — gamma indexing inconsistency).
    """
    if market.condition_id is None:
        _LOG.warning("token_resolver.gamma_market_missing_condition_id", token_id=token_id)
        return None
    # Write market_cache first so any concurrent reader sees a complete row.
    market_cache.upsert(market)
    # Persist both sides of the binary market into asset_index. Polymarket's
    # convention is outcomes[0] = YES-equivalent (idx 0), outcomes[1] = NO (idx 1).
    for idx, asset_id in enumerate(market.clob_token_ids):
        side = "YES" if idx == 0 else "NO"
        asset_index.upsert(
            AssetEntry(
                asset_id=str(asset_id),
                condition_id=str(market.condition_id),
                outcome_side=side,
                outcome_index=idx,
            )
        )
    # Resolve from the just-written rows.
    try:
        target_idx = [str(a) for a in market.clob_token_ids].index(str(token_id))
    except ValueError:
        _LOG.warning(
            "token_resolver.gamma_market_missing_token",
            token_id=token_id,
            condition_id=str(market.condition_id),
            clob_token_ids=[str(a) for a in market.clob_token_ids],
        )
        return None
    return ResolvedToken(
        condition_id=ConditionId(str(market.condition_id)),
        asset_id=AssetId(str(token_id)),
        outcome_name=market.outcomes[target_idx],
        outcome_index=target_idx,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

`uv run pytest tests/poly/test_token_resolver.py -v`

Expected: 4/4 pass.

- [ ] **Step 5: Lint + type checks**

`uv run ruff check src/pscanner/poly/token_resolver.py tests/poly/test_token_resolver.py && uv run ruff format --check src/pscanner/poly/token_resolver.py tests/poly/test_token_resolver.py && uv run ty check src/pscanner/poly/token_resolver.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/pscanner/poly/token_resolver.py tests/poly/test_token_resolver.py
git commit -m "$(cat <<'EOF'
feat(poly): add just-in-time token_resolver with gamma fallback (#157)

New resolve_token helper: tries local asset_index + market_cache lookup
first, falls back to gamma /markets?clob_token_ids= on miss. On gamma hit
upserts both tables so the next sighting is local-only. Returns None on
gamma empty or gamma-indexing-drift defensively.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire the resolver into `scripts/watch_subgraph_copy.py`

**Files:**
- Modify: `scripts/watch_subgraph_copy.py`
- Test: `tests/scripts/test_watch_subgraph_copy.py` (existing — confirm still green)

- [ ] **Step 1: Delete the inline `_ResolvedToken` dataclass and inline `_resolve_token`**

In `scripts/watch_subgraph_copy.py` remove:

- The `_ResolvedToken` dataclass (around lines 225-232)
- The `_resolve_token` function (around lines 235-276)

Replace with this import at the top of the imports block:

```python
from pscanner.poly.token_resolver import ResolvedToken, resolve_token
```

(`ResolvedToken` is imported for any annotation that still uses the type; verify and remove the import if unused after refactor.)

- [ ] **Step 2: Inject `gamma` into `_run_one_cycle` and `_pre_flight`**

Find the `_run_one_cycle` signature (around line 309). Add `gamma: GammaClient` as a keyword arg.

Then `_pre_flight` (around line 392, currently sync). Change to `async def` and add `gamma: GammaClient` as a keyword arg.

- [ ] **Step 3: Replace the resolution callsite with the async helper**

In `_pre_flight`, around line 419, change:

```python
resolved = _resolve_token(ev["tokenId"], asset_index, market_cache)
```

to:

```python
resolved = await resolve_token(
    token_id=AssetId(ev["tokenId"]),
    asset_index=asset_index,
    market_cache=market_cache,
    gamma=gamma,
)
```

Verify `AssetId` is already imported at the top of the file (it is — line 47).

- [ ] **Step 4: Make `_pre_flight` callers `await` it**

Inside `_run_one_cycle`, find the `_pre_flight(...)` call. Change to `await _pre_flight(...)` and add `gamma=gamma` to the kwarg list.

- [ ] **Step 5: Build a `GammaClient` in `main()` and pass it through**

Find `main()` (around line 513). Inside the async-context-manager setup, construct a `GammaClient` and pass it into `_run_one_cycle(...)`. Use the existing pattern from elsewhere in the codebase — e.g., `pscanner.corpus.cli._make_gamma_client` if it exists, or just `GammaClient()` directly.

Verify `GammaClient` is imported (line 45). If not, add `from pscanner.poly.gamma import GammaClient`.

Construct as:

```python
async with GammaClient() as gamma:
    ...
```

(Verify `GammaClient` supports `async with` — if not, use try/finally with `gamma.aclose()`.)

Looking at `pscanner.poly.gamma.py`, `GammaClient` has an `aclose()` method but the project uses it via try/finally typically. Match the existing pattern in the file. The simplest correct pattern:

```python
gamma = GammaClient(rpm=50)
try:
    # ... existing main loop, with gamma passed to _run_one_cycle
finally:
    await gamma.aclose()
```

- [ ] **Step 6: Run the script's existing tests**

`uv run pytest tests/scripts/test_watch_subgraph_copy.py -v`

Expected: 8/8 still pass (or whatever the current count is). If any fail because `_resolve_token` no longer exists, update the test imports to use the new module's symbols (`resolve_token` / `ResolvedToken`).

- [ ] **Step 7: Run the full project gate**

`uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`

Expected: clean (modulo pre-existing unrelated WIP file format violations — leave those alone).

- [ ] **Step 8: Commit**

```bash
git add scripts/watch_subgraph_copy.py tests/scripts/test_watch_subgraph_copy.py
git commit -m "$(cat <<'EOF'
feat(scripts): wire token_resolver into watch_subgraph_copy (#157)

Replace the inline sync _resolve_token + _ResolvedToken with the new
async pscanner.poly.token_resolver.resolve_token. _pre_flight becomes
async; _run_one_cycle takes a GammaClient and forwards it to the
resolver for on-miss fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Desktop live smoke

**Files:** none modified. Operational verification.

- [ ] **Step 1: Push branch + pull on desktop**

```bash
git push -u origin feat/issue-157-token-resolver
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner \
   && export PATH="$HOME/.local/bin:$PATH" \
   && git fetch origin \
   && git checkout feat/issue-157-token-resolver \
   && git pull --ff-only origin feat/issue-157-token-resolver \
   && uv sync 2>&1 | tail -3 \
   && git log -1 --oneline'
```

- [ ] **Step 2: Pre-flight state snapshot**

```bash
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner && export PATH="$HOME/.local/bin:$PATH" && uv run python -c "
import sqlite3
print(\"asset_index:\", sqlite3.connect(\"data/corpus.sqlite3\").execute(\"SELECT COUNT(*) FROM asset_index\").fetchone()[0])
print(\"market_cache:\", sqlite3.connect(\"data/pscanner.sqlite3\").execute(\"SELECT COUNT(*) FROM market_cache\").fetchone()[0])
print(\"paper_trades subgraph_copy:\", sqlite3.connect(\"data/pscanner.sqlite3\").execute(\"SELECT COUNT(*) FROM paper_trades WHERE triggering_alert_detector=\x27subgraph_copy\x27\").fetchone()[0])
"'
```

Capture the three numbers.

- [ ] **Step 3: Re-add the Cavs/Knicks trader and run the smoke**

The earlier 2026-05-21 session's E2E smoke used `0x6a678ca367432d28cb6aa55bb6bc57d8f53bdf6f`. Re-add it temporarily:

```bash
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner && export PATH="$HOME/.local/bin:$PATH" \
   && uv run pscanner watch 0x6a678ca367432d28cb6aa55bb6bc57d8f53bdf6f --reason "issue-157-resolver-smoke"'
```

Then run with a 2h window (the wallet's recent activity is within that):

```bash
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner && export PATH="$HOME/.local/bin:$PATH" && set -a && source .env && set +a \
   && uv run python scripts/watch_subgraph_copy.py --once --since-hours 2 2>&1 | tail -40'
```

Pass criteria:
- `events_copied > 0` (at least one paper trade booked)
- **Most importantly**: the script should resolve tokenIds that were NOT pre-seeded into asset_index/market_cache — i.e. the previously-failing `tokenid_unresolved_*` warnings should be replaced by gamma fallback calls (visible by lower skip count vs the pre-fix baseline, and by paper_trades growing on tokens the wallet has been trading recently beyond just Cavs/Knicks).

If the smoke passes, the fix is validated end-to-end.

- [ ] **Step 4: Snap post-state, unwatch the smoke wallet, open the PR**

```bash
ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null macph@10.0.0.143 \
  'cd ~/projects/polymarketscanner/pscanner && export PATH="$HOME/.local/bin:$PATH" && uv run python -c "
import sqlite3
print(\"asset_index:\", sqlite3.connect(\"data/corpus.sqlite3\").execute(\"SELECT COUNT(*) FROM asset_index\").fetchone()[0])
print(\"market_cache:\", sqlite3.connect(\"data/pscanner.sqlite3\").execute(\"SELECT COUNT(*) FROM market_cache\").fetchone()[0])
print(\"paper_trades subgraph_copy:\", sqlite3.connect(\"data/pscanner.sqlite3\").execute(\"SELECT COUNT(*) FROM paper_trades WHERE triggering_alert_detector=\x27subgraph_copy\x27\").fetchone()[0])
"
   uv run pscanner unwatch 0x6a678ca367432d28cb6aa55bb6bc57d8f53bdf6f'
```

Open the PR:

```bash
gh pr create --title "feat(poly): just-in-time token_resolver with gamma fallback (#157)" --body "$(cat <<'EOF'
## Summary

- New `pscanner.poly.token_resolver.resolve_token` helper resolves subgraph `tokenId`s by trying local lookups first and falling back to gamma `/markets?clob_token_ids=` on a miss. On gamma hit, upserts both `asset_index` (corpus DB) and `market_cache` (daemon DB).
- `scripts/watch_subgraph_copy.py` rewired to use the new helper; its inline `_resolve_token` deleted. `_pre_flight` is now `async`.
- Unblocks #152 (`SubgraphTradeCollector` daemon promotion).

Spec: issue #157.
Plan: `docs/superpowers/plans/2026-05-21-issue-157-token-resolver.md`.

## Test plan

- [x] Unit: `tests/poly/test_token_resolver.py` — 4 tests covering local hit, gamma fallback, gamma empty, gamma indexing drift.
- [x] Existing tests in `tests/scripts/test_watch_subgraph_copy.py` still pass (1 module-import update).
- [x] Project gate: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q` clean.
- [x] Desktop live smoke: ran against the 5-wallet watchlist + Cavs/Knicks trader. Confirmed previously-failing token resolutions now succeed via gamma fallback, new rows landed in `paper_trades(triggering_alert_detector='subgraph_copy')`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Done criteria

- [ ] Four commits on `feat/issue-157-token-resolver`:
  1. Plan commit (the doc itself)
  2. `feat(poly): add clob_token_ids filter to GammaClient.list_markets (#157)`
  3. `feat(poly): add just-in-time token_resolver with gamma fallback (#157)`
  4. `feat(scripts): wire token_resolver into watch_subgraph_copy (#157)`
- [ ] All project lint / type / test gates pass.
- [ ] Desktop live smoke confirms the gamma fallback resolves previously-failing tokens.
- [ ] PR opened and linked to issue #157.
