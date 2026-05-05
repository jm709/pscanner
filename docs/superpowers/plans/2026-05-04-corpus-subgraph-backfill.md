# Corpus subgraph backfill — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `eth_getLogs` corpus ingest with GraphQL queries against Polymarket's public Orderbook subgraph on The Graph. Server-side filtering by `makerAssetId_in` / `takerAssetId_in` (using the local `asset_index` table) cuts per-market data volume by ~100×, eliminates the firehose-discard waste, and brings full-corpus backfill from "days" to "tens of minutes."

**Architecture:** A new `pscanner.poly.subgraph` module exposes an async GraphQL client (`SubgraphClient`) with one `query()` method, a token-bucket rate limiter, and 429/5xx backoff (mirroring `pscanner.poly.onchain_rpc`). A new `pscanner.corpus.subgraph_ingest` module composes the client with a `subgraph_row_to_event` adapter, the **unchanged** `event_to_corpus_trade` from Phase 2, and `CorpusTradesRepo` to drive a per-market resumable orchestrator. A new `pscanner corpus subgraph-backfill` subcommand wires the CLI. The deletion of the `eth_getLogs` path is **deferred to a follow-up commit** after live validation lands — the `onchain-backfill` and `onchain-backfill-targeted` subcommands stay alongside the new path during the transition.

**Tech Stack:** Python 3.13, `httpx` async, `tenacity` retries, `sqlite3`, `respx` for HTTP mocking. No new third-party dependencies. The existing `OrderFilledEvent` dataclass, `decode_order_filled` decoder, `event_to_corpus_trade`, `AssetIndexRepo`, `CorpusTradesRepo`, and `clear_truncation_flags` are all reused unchanged. The on-chain `OrderFilledEvent` dataclass already exposes everything the adapter needs except `block_number` / `log_index`, which we set to 0 sentinels (subgraph payloads don't include these fields and `event_to_corpus_trade` doesn't read them).

---

## Schema & endpoint findings

Confirmed against `https://github.com/Polymarket/polymarket-subgraph/blob/main/orderbook-subgraph/schema.graphql`:

```graphql
type OrderFilledEvent @entity {
  id: ID!                       # transactionHash + orderHash (string concat — id_gt cursor is monotone)
  transactionHash: Bytes!
  timestamp: BigInt!            # Unix seconds — used directly, no eth_getBlockByNumber needed
  orderHash: Bytes!
  maker: String!
  taker: String!
  makerAssetId: String!         # uint256 as string — "0" for USDC, otherwise CTF token id
  takerAssetId: String!
  makerAmountFilled: BigInt!    # 6-decimal integer (USDC or CTF amount)
  takerAmountFilled: BigInt!
  fee: BigInt!
}
```

**Critical schema deltas vs. the issue draft:**
1. `OrderFilledEvent` has **no `marketHash` / `condition` / `conditionId` field**. The asset-id ↔ condition-id mapping lives on a sibling `MarketData` entity. We don't need it: our local `asset_index` table already has 8K+ entries from REST trades. Filter by `makerAssetId_in` and `takerAssetId_in` (two paginations per market — one for sells from maker POV, one for buys).
2. The same subgraph indexes both the original `Exchange` contract and `NegRiskExchange`, both writing into the same `OrderFilledEvent` entity. **Negative-Risk markets are auto-covered**, contradicting the issue's "out of scope" note. After validation, we expect the residual truncated-market count to be near zero, not "Negative-Risk only."
3. No `blockNumber` field — the subgraph stores `timestamp` directly. The adapter sets `block_number=0` and `log_index=0` on the synthesised `OrderFilledEvent`; `event_to_corpus_trade` doesn't read those fields.

**Endpoint:** Public Gateway requires a free Graph Studio API key. URL pattern:

```
https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}
```

Subgraph deployment id is **TBD by Task 1** — confirm via Graph Explorer (`https://thegraph.com/explorer/?search=Polymarket+Orderbook`). The id `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY` mentioned in the issue may be the Activity subgraph, not the Orderbook one. Free-tier budget is 100K queries/month; expected full-corpus run consumes ~10K-15K queries (well under).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `src/pscanner/poly/subgraph.py` | create | `SubgraphClient`: async GraphQL POST with token-bucket rate limiting and tenacity 429/5xx backoff |
| `tests/poly/test_subgraph.py` | create | Client tests against `respx`-mocked gateway |
| `src/pscanner/corpus/subgraph_ingest.py` | create | `subgraph_row_to_event` adapter, `iter_market_trades` paginator, `run_subgraph_backfill` orchestrator |
| `tests/corpus/test_subgraph_ingest.py` | create | Adapter unit tests + paginator tests + end-to-end orchestrator test against mocked gateway |
| `src/pscanner/corpus/cli.py` | modify | Add `subgraph-backfill` subparser + `_cmd_subgraph_backfill` handler |
| `tests/corpus/test_cli.py` | modify | CLI smoke for `corpus subgraph-backfill` |
| `CLAUDE.md` | modify | Document the subgraph as the canonical corpus data path; flag eth_getLogs path as "to be deleted in follow-up" |

**Deferred to a follow-up PR (after live validation):** delete `src/pscanner/poly/onchain_rpc.py`, `iter_order_filled_logs` in `src/pscanner/poly/onchain_ingest.py`, `src/pscanner/corpus/onchain_backfill.py`, `src/pscanner/corpus/onchain_targeted.py`, and the two onchain CLI subparsers. The decoder (`decode_order_filled`, `OrderFilledEvent`) and `AssetIndexRepo` stay.

---

## Out of scope

- Daemon-side incremental sync via subgraph polling (subgraph is ~1-2 min behind head; the live `/activity` REST collector is fine).
- Deleting the `eth_getLogs` path — separate follow-up commit after live validation succeeds.
- Rebuilding `training_examples` and ML retrain — captured in the post-implementation checklist; not a code change.
- Discovery of new condition_ids via `MarketData` entity — we trust the existing `asset_index` population.

---

## Task 1: Verify subgraph deployment id and gateway behaviour

**Files:**
- Modify: `docs/superpowers/plans/2026-05-04-corpus-subgraph-backfill.md` (this file — update the "Schema & endpoint findings" section with confirmed values)

This task has no code, but downstream tasks hard-code a default subgraph id and depend on the gateway URL pattern. It must run first.

- [ ] **Step 1.1: Locate the Polymarket Orderbook subgraph in Graph Explorer**

Visit `https://thegraph.com/explorer/?search=Polymarket+Orderbook`. Find the subgraph titled "Polymarket Orderbook" (or similar). Record its deployment id (32-char alphanumeric like `7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY`).

- [ ] **Step 1.2: Create a Graph Studio API key**

Sign in to `https://thegraph.com/studio/` (GitHub or wallet auth), generate a free-tier API key. Save to `~/.config/pscanner/graph_api_key` (chmod 600) so the CLI can read it without it landing in shell history.

- [ ] **Step 1.3: Smoke-test one query via curl**

```bash
API_KEY=$(cat ~/.config/pscanner/graph_api_key)
SUBGRAPH_ID=<id from step 1.1>
curl -s -X POST \
  -H "Content-Type: application/json" \
  --data '{"query": "{ orderFilledEvents(first: 1, orderBy: id) { id timestamp makerAssetId } }"}' \
  "https://gateway.thegraph.com/api/${API_KEY}/subgraphs/id/${SUBGRAPH_ID}"
```

Expected: `{"data":{"orderFilledEvents":[{"id":"0x...","timestamp":"...","makerAssetId":"..."}]}}`. Document the actual response shape (specifically: are `BigInt` fields returned as JSON numbers or strings? GraphQL convention is strings to preserve precision; verify).

- [ ] **Step 1.4: Update this plan's "Schema & endpoint findings" section**

Replace the "TBD by Task 1" placeholder with the confirmed subgraph id. Note any field-rename surprises (e.g. if the live schema has drifted from the GitHub copy). If the BigInt returns numbers (not strings), update Task 4's parsing logic accordingly.

- [ ] **Step 1.5: Commit**

```bash
git add docs/superpowers/plans/2026-05-04-corpus-subgraph-backfill.md
git commit -m "docs(corpus): record verified subgraph id + endpoint shape for Phase 3"
```

---

## Task 2: `SubgraphClient` — async GraphQL POST with rate limiting

**Files:**
- Create: `src/pscanner/poly/subgraph.py`
- Create: `tests/poly/test_subgraph.py`

The client mirrors `pscanner.poly.onchain_rpc.OnchainRpcClient` in shape: lazy `httpx.AsyncClient`, internal `_TokenBucket`, tenacity backoff on retryable status codes, async context manager. The single public method is `query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]` returning the `data` payload (GraphQL errors are raised as `RuntimeError`).

- [ ] **Step 2.1: Write failing test — basic query returns `data` payload**

```python
# tests/poly/test_subgraph.py

from __future__ import annotations

import httpx
import pytest
import respx

from pscanner.poly.subgraph import SubgraphClient

_URL = "https://gateway.example.test/api/key/subgraphs/id/abc"


@pytest.fixture
def client() -> SubgraphClient:
    return SubgraphClient(url=_URL, rpm=600, timeout_seconds=5.0)


@respx.mock
async def test_query_returns_data_payload(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"orderFilledEvents": [{"id": "0xabc"}]}},
        )
    )
    try:
        result = await client.query(
            "query Q($x: String!) { orderFilledEvents(where: {id: $x}) { id } }",
            {"x": "0xabc"},
        )
    finally:
        await client.aclose()
    assert result == {"orderFilledEvents": [{"id": "0xabc"}]}
```

- [ ] **Step 2.2: Run test, confirm `ModuleNotFoundError`**

Run: `uv run pytest tests/poly/test_subgraph.py::test_query_returns_data_payload -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pscanner.poly.subgraph'`.

- [ ] **Step 2.3: Implement minimal `SubgraphClient`**

Create `src/pscanner/poly/subgraph.py`:

```python
"""Async GraphQL client for The Graph's hosted gateway.

Mirrors the rate-limit + retry shape of ``pscanner.poly.onchain_rpc``.
The single public surface is ``query(graphql, variables)`` returning the
``data`` payload; GraphQL ``errors`` arrays surface as RuntimeError.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_LOG = structlog.get_logger(__name__)
_USER_AGENT = "pscanner/0.1"

_STATUS_TOO_MANY_REQUESTS = 429
_RETRYABLE_STATUS = frozenset({_STATUS_TOO_MANY_REQUESTS, 502, 503, 504})
_RETRYABLE_TRANSPORT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
_MAX_ATTEMPTS = 5
_BACKOFF_MIN_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0


class _TokenBucket:
    """Async token bucket: capacity tokens, refilled at ``rate`` per second."""

    def __init__(self, *, capacity: int, rate_per_second: float) -> None:
        self._capacity = float(capacity)
        self._rate = rate_per_second
        self._tokens = float(capacity)
        self._last_refill = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            while True:
                now = loop.time()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                    self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                await asyncio.sleep(deficit / self._rate)


class _RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"retryable status {response.status_code}")
        self.response = response


def _parse_retry_after(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(tz=UTC)).total_seconds())


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableStatusError) or isinstance(exc, _RETRYABLE_TRANSPORT_EXC)


def _before_sleep_log(retry_state: RetryCallState) -> None:
    outcome = retry_state.outcome
    if outcome is None:
        return
    exc = outcome.exception()
    if not isinstance(exc, _RetryableStatusError):
        return
    response = exc.response
    _LOG.warning(
        "subgraph_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        retry_after=response.headers.get("Retry-After"),
    )


class SubgraphClient:
    """Async GraphQL client targeting a single subgraph endpoint."""

    def __init__(self, *, url: str, rpm: int, timeout_seconds: float = 30.0) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.url = url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._bucket: _TokenBucket | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_ready(self) -> tuple[httpx.AsyncClient, _TokenBucket]:
        if self._closed:
            raise RuntimeError("SubgraphClient is closed")
        if self._client is not None and self._bucket is not None:
            return self._client, self._bucket
        async with self._init_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Content-Type": "application/json",
                    },
                )
            if self._bucket is None:
                self._bucket = _TokenBucket(
                    capacity=self.rpm,
                    rate_per_second=self.rpm / 60.0,
                )
            return self._client, self._bucket

    async def query(self, graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one GraphQL query, returning the ``data`` payload.

        Raises:
            RuntimeError: If the response contains a non-empty ``errors`` array.
            httpx.HTTPStatusError: On non-2xx after retries are exhausted.
        """
        client, bucket = await self._ensure_ready()
        body = {"query": graphql, "variables": dict(variables)}
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=1.0,
                min=_BACKOFF_MIN_SECONDS,
                max=_BACKOFF_MAX_SECONDS,
            ),
            before_sleep=_before_sleep_log,
            reraise=True,
        )
        response: httpx.Response | None = None
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._send_once(client, bucket, body)
        except _RetryableStatusError as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover
        if response is None:  # pragma: no cover
            raise RuntimeError("retry loop produced no response")
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL errors: {payload['errors']}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"GraphQL response missing 'data' object: {payload!r}")
        return data

    async def _send_once(
        self,
        client: httpx.AsyncClient,
        bucket: _TokenBucket,
        body: dict[str, Any],
    ) -> httpx.Response:
        await bucket.acquire()
        response = await client.post(self.url, json=body)
        if response.status_code in _RETRYABLE_STATUS:
            if response.status_code == _STATUS_TOO_MANY_REQUESTS:
                raw = response.headers.get("Retry-After")
                if raw is not None:
                    wait = _parse_retry_after(raw)
                    if wait is not None and wait > 0:
                        await asyncio.sleep(wait)
            raise _RetryableStatusError(response)
        response.raise_for_status()
        return response

    async def aclose(self) -> None:
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> Self:
        await self._ensure_ready()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
```

- [ ] **Step 2.4: Re-run test, confirm pass**

Run: `uv run pytest tests/poly/test_subgraph.py::test_query_returns_data_payload -v`
Expected: PASS.

- [ ] **Step 2.5: Add backoff + GraphQL-error tests**

Append to `tests/poly/test_subgraph.py`:

```python
@respx.mock
async def test_429_then_200_succeeds(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rl"}),
            httpx.Response(200, json={"data": {"x": 1}}),
        ]
    )
    try:
        result = await client.query("{ x }", {})
    finally:
        await client.aclose()
    assert result == {"x": 1}


@respx.mock
async def test_persistent_503_raises(client: SubgraphClient) -> None:
    route = respx.post(_URL).mock(return_value=httpx.Response(503, json={"err": "down"}))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.query("{ x }", {})
    finally:
        await client.aclose()
    assert route.call_count == 5


@respx.mock
async def test_graphql_errors_surface_as_runtime_error(client: SubgraphClient) -> None:
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json={"errors": [{"message": "bad query"}]}
        )
    )
    try:
        with pytest.raises(RuntimeError, match="GraphQL errors"):
            await client.query("{ broken }", {})
    finally:
        await client.aclose()


@respx.mock
async def test_query_sends_variables_in_body(client: SubgraphClient) -> None:
    captured: list[dict[str, object]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.append(_json.loads(request.read()))
        return httpx.Response(200, json={"data": {"ok": True}})

    respx.post(_URL).mock(side_effect=_capture)
    try:
        await client.query("query Q($a: Int!) { ok }", {"a": 7})
    finally:
        await client.aclose()
    assert captured[0]["query"] == "query Q($a: Int!) { ok }"
    assert captured[0]["variables"] == {"a": 7}
```

- [ ] **Step 2.6: Run all client tests**

Run: `uv run pytest tests/poly/test_subgraph.py -v`
Expected: 4 PASS.

- [ ] **Step 2.7: Lint, type-check**

Run: `uv run ruff check src/pscanner/poly/subgraph.py tests/poly/test_subgraph.py && uv run ruff format --check src/pscanner/poly/subgraph.py tests/poly/test_subgraph.py && uv run ty check src/pscanner/poly/subgraph.py tests/poly/test_subgraph.py`
Expected: clean.

- [ ] **Step 2.8: Commit**

```bash
git add src/pscanner/poly/subgraph.py tests/poly/test_subgraph.py
git commit -m "feat(poly): SubgraphClient for The Graph gateway with backoff"
```

---

## Task 3: `subgraph_row_to_event` adapter

**Files:**
- Create: `src/pscanner/corpus/subgraph_ingest.py` (initial scaffold — adapter only)
- Create: `tests/corpus/test_subgraph_ingest.py` (initial scaffold — adapter tests only)

Maps a single GraphQL `OrderFilledEvent` row to the existing `pscanner.poly.onchain.OrderFilledEvent` dataclass. Sets `block_number=0`, `log_index=0` (subgraph payloads omit these; `event_to_corpus_trade` doesn't read them). Parses `BigInt` strings to `int`; lowercases addresses for downstream consistency.

- [ ] **Step 3.1: Write failing test for the adapter**

```python
# tests/corpus/test_subgraph_ingest.py

from __future__ import annotations

import pytest

from pscanner.corpus.subgraph_ingest import subgraph_row_to_event


def test_subgraph_row_to_event_parses_buy_side_row() -> None:
    """Maker BUY: maker gives USDC ('0'), taker gives CTF token."""
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0xMaker_Address_NOT_LowerCased",
        "taker": "0x" + "22" * 20,
        "makerAssetId": "0",
        "takerAssetId": "222",
        "makerAmountFilled": "20000000",
        "takerAmountFilled": "40000000",
        "fee": "0",
    }
    event = subgraph_row_to_event(row)
    assert event.tx_hash == "0xee" * 32
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 222
    assert event.making == 20_000_000
    assert event.taking == 40_000_000
    assert event.fee == 0
    assert event.block_number == 0
    assert event.log_index == 0
    # event_to_corpus_trade lowercases the maker; the dataclass preserves whatever's passed in
    assert event.maker == "0xMaker_Address_NOT_LowerCased"


def test_subgraph_row_to_event_rejects_missing_field() -> None:
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        # orderHash deliberately missing
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": "0",
        "takerAssetId": "222",
        "makerAmountFilled": "1",
        "takerAmountFilled": "1",
        "fee": "0",
    }
    with pytest.raises(KeyError, match="orderHash"):
        subgraph_row_to_event(row)


def test_subgraph_row_to_event_rejects_non_numeric_amount() -> None:
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": "0",
        "takerAssetId": "222",
        "makerAmountFilled": "not-a-number",
        "takerAmountFilled": "40000000",
        "fee": "0",
    }
    with pytest.raises(ValueError):
        subgraph_row_to_event(row)
```

- [ ] **Step 3.2: Run test, confirm `ModuleNotFoundError`**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3.3: Create the module with the adapter**

Create `src/pscanner/corpus/subgraph_ingest.py`:

```python
"""Subgraph-driven backfill of `corpus_trades` (Phase 3).

Adapter, paginator, and orchestrator that replace the eth_getLogs path
in ``pscanner.corpus.onchain_targeted``. Reuses the Phase 2 decoder
output type (``OrderFilledEvent``) and ``event_to_corpus_trade`` so the
maker-POV BUY/SELL semantics stay identical.
"""

from __future__ import annotations

from collections.abc import Mapping

from pscanner.poly.onchain import OrderFilledEvent

_REQUIRED_KEYS = (
    "transactionHash",
    "timestamp",
    "orderHash",
    "maker",
    "taker",
    "makerAssetId",
    "takerAssetId",
    "makerAmountFilled",
    "takerAmountFilled",
    "fee",
)


def subgraph_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one GraphQL ``OrderFilledEvent`` row to a Phase 2 dataclass.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list. Must
            carry every key in ``_REQUIRED_KEYS``.

    Returns:
        ``OrderFilledEvent`` with ``block_number=0`` and ``log_index=0``
        (subgraph payloads do not include these; downstream
        ``event_to_corpus_trade`` does not read them).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable as int.
    """
    for key in _REQUIRED_KEYS:
        if key not in row:
            raise KeyError(key)

    def as_int(key: str) -> int:
        raw = row[key]
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            return int(raw)
        raise ValueError(f"{key} must be int or str, got {type(raw).__name__}")

    def as_str(key: str) -> str:
        raw = row[key]
        if not isinstance(raw, str):
            raise ValueError(f"{key} must be str, got {type(raw).__name__}")
        return raw

    return OrderFilledEvent(
        order_hash=as_str("orderHash"),
        maker=as_str("maker"),
        taker=as_str("taker"),
        maker_asset_id=as_int("makerAssetId"),
        taker_asset_id=as_int("takerAssetId"),
        making=as_int("makerAmountFilled"),
        taking=as_int("takerAmountFilled"),
        fee=as_int("fee"),
        tx_hash=as_str("transactionHash"),
        block_number=0,
        log_index=0,
    )
```

- [ ] **Step 3.4: Re-run adapter tests**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -v`
Expected: 3 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py
git commit -m "feat(corpus): subgraph_row_to_event adapter for Phase 3"
```

---

## Task 4: `iter_market_trades` paginator

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest.py`
- Modify: `tests/corpus/test_subgraph_ingest.py`

Yields decoded `OrderFilledEvent` rows for one condition_id by running **two** paginated queries in sequence: one filtered by `makerAssetId_in: $assets` (yields SELLs from maker POV — maker gave a CTF token), one by `takerAssetId_in: $assets` (yields BUYs — maker received a CTF token). For each side, paginate using `id_gt: $cursor` and `first: 1000` until empty. Yield `(event, ts)` tuples — the same shape as the eth_getLogs paginator — so the orchestrator below mirrors `run_targeted_backfill.backfill_market`'s loop body.

**Why two queries instead of one:** `OrderFilledEvent` has no `marketHash` filter (see "Schema & endpoint findings"). The Graph supports an `or:` operator since v0.30, but a single `or:` query forces server-side union evaluation and is reportedly slower than two cursor-paginated streams under high cardinality. Two queries also keep the cursor state per-side, which is simpler to reason about.

- [ ] **Step 4.1: Write failing test — paginator yields events from both sides, paginates correctly**

Append to `tests/corpus/test_subgraph_ingest.py`:

```python
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

from pscanner.corpus.subgraph_ingest import iter_market_trades


async def test_iter_market_trades_paginates_both_sides() -> None:
    """Paginator runs maker-side then taker-side, yields decoded events."""
    side_responses = {
        # Maker-side page 1 (full page → another page expected)
        ("makerAssetId_in", ""): [
            _row(id_="0x01_a", maker_asset="111", taker_asset="0", making=1_000_000, taking=2_000_000),
        ],
        # Maker-side page 2 (smaller than 'first' → done)
        ("makerAssetId_in", "0x01_a"): [],
        # Taker-side page 1
        ("takerAssetId_in", ""): [
            _row(id_="0x02_b", maker_asset="0", taker_asset="111", making=1_000_000, taking=2_000_000),
        ],
        ("takerAssetId_in", "0x02_b"): [],
    }

    async def fake_query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        side = "makerAssetId_in" if "makerAssetId_in" in graphql else "takerAssetId_in"
        cursor = variables.get("cursor", "")
        rows = side_responses[(side, cursor)]
        return {"orderFilledEvents": rows}

    client = AsyncMock()
    client.query.side_effect = fake_query

    yielded = []
    async for event, ts in iter_market_trades(
        client=client,
        asset_ids=["111"],
        page_size=1,
    ):
        yielded.append((event, ts))

    assert len(yielded) == 2
    # Maker-side first
    assert yielded[0][0].maker_asset_id == 111
    assert yielded[0][0].taker_asset_id == 0
    # Then taker-side
    assert yielded[1][0].maker_asset_id == 0
    assert yielded[1][0].taker_asset_id == 111
    # Timestamps preserved
    assert yielded[0][1] == 1_700_000_000


def _row(*, id_: str, maker_asset: str, taker_asset: str, making: int, taking: int) -> dict[str, str]:
    return {
        "id": id_,
        "transactionHash": "0x" + "ee" * 32,
        "timestamp": "1700000000",
        "orderHash": "0x" + "ab" * 32,
        "maker": "0x" + "11" * 20,
        "taker": "0x" + "22" * 20,
        "makerAssetId": maker_asset,
        "takerAssetId": taker_asset,
        "makerAmountFilled": str(making),
        "takerAmountFilled": str(taking),
        "fee": "0",
    }
```

- [ ] **Step 4.2: Run test, confirm `ImportError`**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py::test_iter_market_trades_paginates_both_sides -v`
Expected: FAIL — `ImportError: cannot import name 'iter_market_trades'`.

- [ ] **Step 4.3: Implement `iter_market_trades`**

Append to `src/pscanner/corpus/subgraph_ingest.py`:

```python
from collections.abc import AsyncIterator, Sequence

from pscanner.poly.subgraph import SubgraphClient

# The Graph's hard cap on a single page of results.
_MAX_PAGE_SIZE = 1000

_TRADES_QUERY_MAKER_SIDE = """
query MarketTradesMakerSide($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId_in: $assets, id_gt: $cursor }
    orderBy: id
    first: $first
  ) {
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }
}
""".strip()

_TRADES_QUERY_TAKER_SIDE = """
query MarketTradesTakerSide($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { takerAssetId_in: $assets, id_gt: $cursor }
    orderBy: id
    first: $first
  ) {
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }
}
""".strip()


async def _paginate_side(
    *,
    client: SubgraphClient,
    graphql: str,
    asset_ids: Sequence[str],
    page_size: int,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield decoded events from one side of the filter, paginated by id_gt."""
    cursor = ""
    while True:
        result = await client.query(
            graphql,
            {"assets": list(asset_ids), "cursor": cursor, "first": page_size},
        )
        rows = result.get("orderFilledEvents")
        if not rows:
            return
        for row in rows:
            event = subgraph_row_to_event(row)
            ts = int(str(row["timestamp"]))
            yield event, ts
        if len(rows) < page_size:
            # Short page — guaranteed last by ordering invariant
            return
        cursor = str(rows[-1]["id"])


async def iter_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every `OrderFilledEvent` whose maker- or taker-side asset is in `asset_ids`.

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as strings) that belong to one condition.
        page_size: Rows per query, capped at 1000 by The Graph.

    Yields:
        ``(event, ts)`` tuples — same shape as ``iter_order_filled_logs``
        from Phase 2 so the orchestrator can mirror its loop body.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY_MAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY_TAKER_SIDE,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
```

- [ ] **Step 4.4: Re-run paginator test**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py::test_iter_market_trades_paginates_both_sides -v`
Expected: PASS.

- [ ] **Step 4.5: Add edge-case test — empty `asset_ids` yields nothing without querying**

Append to `tests/corpus/test_subgraph_ingest.py`:

```python
async def test_iter_market_trades_empty_asset_ids_skips_query() -> None:
    client = AsyncMock()
    yielded = []
    async for ev, ts in iter_market_trades(client=client, asset_ids=[], page_size=10):
        yielded.append((ev, ts))
    assert yielded == []
    client.query.assert_not_called()
```

- [ ] **Step 4.6: Add edge-case test — short first page exits cleanly**

```python
async def test_iter_market_trades_short_first_page_exits_without_second_query() -> None:
    """When the first page is shorter than page_size, no further query runs for that side."""
    side_calls: dict[str, int] = {"maker": 0, "taker": 0}

    async def fake_query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        side = "maker" if "makerAssetId_in" in graphql else "taker"
        side_calls[side] += 1
        if side == "maker":
            return {"orderFilledEvents": [_row(
                id_="0x01_a", maker_asset="111", taker_asset="0",
                making=1_000_000, taking=2_000_000,
            )]}
        return {"orderFilledEvents": []}

    client = AsyncMock()
    client.query.side_effect = fake_query

    yielded = []
    async for ev, ts in iter_market_trades(client=client, asset_ids=["111"], page_size=100):
        yielded.append((ev, ts))

    assert len(yielded) == 1
    # Maker side: 1 query (short page → no follow-up). Taker side: 1 query (empty).
    assert side_calls == {"maker": 1, "taker": 1}
```

- [ ] **Step 4.7: Run all paginator tests + lint/type-check**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -v && uv run ruff check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py && uv run ty check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py`
Expected: all PASS, clean.

- [ ] **Step 4.8: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py
git commit -m "feat(corpus): iter_market_trades paginator over subgraph"
```

---

## Task 5: `run_subgraph_backfill` orchestrator

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest.py`
- Modify: `tests/corpus/test_subgraph_ingest.py`

Drives one CLI invocation: load truncated + unprocessed markets, for each load asset ids from `asset_index`, run the paginator, decode events through `event_to_corpus_trade`, batch-insert via `CorpusTradesRepo`, mark `corpus_markets.onchain_processed_at`, and after the run call `clear_truncation_flags`. Mirrors `run_targeted_backfill` in `pscanner.corpus.onchain_targeted` so the CLI handlers stay symmetric.

- [ ] **Step 5.1: Write failing end-to-end test**

Append to `tests/corpus/test_subgraph_ingest.py`:

```python
import json as _json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTrade,
    CorpusTradesRepo,
)
from pscanner.corpus.subgraph_ingest import (
    SubgraphRunSummary,
    run_subgraph_backfill,
)
from pscanner.poly.subgraph import SubgraphClient

_GATEWAY_URL = "https://gateway.example.test/api/k/subgraphs/id/abc"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Corpus DB seeded with one truncated market and its asset_index entries."""
    db = init_corpus_db(tmp_path / "c.sqlite3")
    try:
        markets = CorpusMarketsRepo(db)
        markets.insert_pending(
            CorpusMarket(
                condition_id="0xMARKET_A",
                event_slug="some-event",
                category=None,
                closed_at=1_700_001_000,
                total_volume_usd=42_000.0,
                enumerated_at=1_700_000_000,
                market_slug="some-market",
            )
        )
        db.execute(
            """
            UPDATE corpus_markets
            SET truncated_at_offset_cap = 1, backfill_state = 'complete'
            WHERE condition_id = ?
            """,
            ("0xMARKET_A",),
        )
        db.commit()
        # Asset index: YES=111, NO=222 for this market
        AssetIndexRepo(db).upsert(
            AssetEntry(asset_id="111", condition_id="0xMARKET_A", outcome_side="YES", outcome_index=0)
        )
        AssetIndexRepo(db).upsert(
            AssetEntry(asset_id="222", condition_id="0xMARKET_A", outcome_side="NO", outcome_index=1)
        )
        # Seed two existing trades so onchain_trades_count math is meaningful
        CorpusTradesRepo(db).insert_batch(
            [
                CorpusTrade(
                    tx_hash="0x" + "aa" * 32, asset_id="111", wallet_address="0x" + "11" * 20,
                    condition_id="0xMARKET_A", outcome_side="YES", bs="BUY",
                    price=0.5, size=20.0, notional_usd=10.0, ts=1_700_000_500,
                ),
                CorpusTrade(
                    tx_hash="0x" + "bb" * 32, asset_id="111", wallet_address="0x" + "11" * 20,
                    condition_id="0xMARKET_A", outcome_side="YES", bs="BUY",
                    price=0.5, size=20.0, notional_usd=10.0, ts=1_700_000_900,
                ),
            ]
        )
        yield db
    finally:
        db.close()


@respx.mock
async def test_run_subgraph_backfill_processes_pending_market(
    conn: sqlite3.Connection,
) -> None:
    """End-to-end: one pending market → 1 trade inserted → market marked processed."""

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        side = "maker" if "makerAssetId_in" in body["query"] else "taker"
        cursor = body["variables"]["cursor"]
        if side == "maker" and cursor == "":
            # SELL from maker POV: maker gives CTF (asset 111), taker gives USDC
            return httpx.Response(200, json={"data": {"orderFilledEvents": [
                {
                    "id": "0xtx1_0xord1",
                    "transactionHash": "0x" + "ee" * 32,
                    "timestamp": "1700001500",
                    "orderHash": "0x" + "ab" * 32,
                    "maker": "0x" + "ff" * 20,
                    "taker": "0x" + "22" * 20,
                    "makerAssetId": "111",
                    "takerAssetId": "0",
                    "makerAmountFilled": "40000000",
                    "takerAmountFilled": "20000000",
                    "fee": "0",
                }
            ]}})
        # Every other query (next page maker, both pages taker) returns empty
        return httpx.Response(200, json={"data": {"orderFilledEvents": []}})

    respx.post(_GATEWAY_URL).mock(side_effect=_route)

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client)
    finally:
        await client.aclose()

    assert isinstance(summary, SubgraphRunSummary)
    assert summary.markets_processed == 1
    assert summary.markets_failed == 0
    assert summary.trades_inserted == 1

    rows = conn.execute(
        """
        SELECT bs, asset_id, wallet_address, ts FROM corpus_trades
        WHERE condition_id = '0xMARKET_A' AND tx_hash = ?
        """,
        ("0x" + "ee" * 32,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["bs"] == "SELL"  # maker gave CTF → SELL from maker POV
    assert rows[0]["asset_id"] == "111"
    assert rows[0]["wallet_address"] == "0x" + "ff" * 20
    assert rows[0]["ts"] == 1_700_001_500

    row = conn.execute(
        "SELECT onchain_processed_at, onchain_trades_count, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id = '0xMARKET_A'"
    ).fetchone()
    assert row["onchain_processed_at"] is not None
    assert row["onchain_processed_at"] <= int(time.time())
    assert row["onchain_trades_count"] == 3  # 2 seeded + 1 inserted
    assert row["truncated_at_offset_cap"] == 1  # below 3000 threshold


@respx.mock
async def test_run_subgraph_backfill_skips_already_processed_markets(
    conn: sqlite3.Connection,
) -> None:
    """Markets with onchain_processed_at set must be skipped on subsequent runs."""
    conn.execute(
        "UPDATE corpus_markets SET onchain_processed_at = ? WHERE condition_id = ?",
        (int(time.time()) - 60, "0xMARKET_A"),
    )
    conn.commit()

    route = respx.post(_GATEWAY_URL).mock(
        return_value=httpx.Response(200, json={"data": {"orderFilledEvents": []}})
    )

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client)
    finally:
        await client.aclose()

    assert summary.markets_processed == 0
    assert route.call_count == 0  # no markets pending → no queries fired


@respx.mock
async def test_run_subgraph_backfill_respects_limit(
    conn: sqlite3.Connection,
) -> None:
    """`limit=N` processes at most N markets even if more are pending."""
    # Add a second pending market
    CorpusMarketsRepo(conn).insert_pending(
        CorpusMarket(
            condition_id="0xMARKET_B",
            event_slug="event-b",
            category=None,
            closed_at=1_700_001_000,
            total_volume_usd=10_000.0,
            enumerated_at=1_700_000_000,
            market_slug="market-b",
        )
    )
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1, backfill_state = 'complete' "
        "WHERE condition_id = ?",
        ("0xMARKET_B",),
    )
    conn.commit()
    AssetIndexRepo(conn).upsert(
        AssetEntry(asset_id="333", condition_id="0xMARKET_B", outcome_side="YES", outcome_index=0)
    )

    respx.post(_GATEWAY_URL).mock(
        return_value=httpx.Response(200, json={"data": {"orderFilledEvents": []}})
    )

    client = SubgraphClient(url=_GATEWAY_URL, rpm=600)
    try:
        summary = await run_subgraph_backfill(conn=conn, client=client, limit=1)
    finally:
        await client.aclose()

    assert summary.markets_processed == 1
```

- [ ] **Step 5.2: Run, confirm `ImportError`**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py::test_run_subgraph_backfill_processes_pending_market -v`
Expected: FAIL — `ImportError: cannot import name 'run_subgraph_backfill'`.

- [ ] **Step 5.3: Implement orchestrator + summary dataclass**

Append to `src/pscanner/corpus/subgraph_ingest.py`:

```python
import sqlite3
import time
from dataclasses import dataclass

import structlog

from pscanner.corpus.onchain_backfill import clear_truncation_flags
from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade, CorpusTradesRepo
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SubgraphRunSummary:
    """Aggregate counts returned by ``run_subgraph_backfill``."""

    markets_processed: int
    markets_failed: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    truncation_flags_cleared: int


@dataclass(frozen=True)
class _PendingMarket:
    condition_id: str
    market_slug: str
    total_volume_usd: float


def _load_pending_markets(
    conn: sqlite3.Connection, *, limit: int | None
) -> list[_PendingMarket]:
    sql = """
        SELECT condition_id,
               COALESCE(market_slug, '') AS market_slug,
               total_volume_usd
        FROM corpus_markets
        WHERE truncated_at_offset_cap = 1
          AND onchain_processed_at IS NULL
        ORDER BY total_volume_usd DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [
        _PendingMarket(
            condition_id=r["condition_id"],
            market_slug=r["market_slug"],
            total_volume_usd=float(r["total_volume_usd"]),
        )
        for r in rows
    ]


def _load_market_asset_ids(conn: sqlite3.Connection, condition_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT asset_id FROM asset_index WHERE condition_id = ?", (condition_id,)
    ).fetchall()
    return [row["asset_id"] for row in rows]


def _mark_processed(
    conn: sqlite3.Connection,
    condition_id: str,
    *,
    now_ts: int,
) -> int:
    """Persist post-backfill state; returns the new on-chain trade count."""
    count = int(
        conn.execute(
            "SELECT COUNT(*) FROM corpus_trades WHERE condition_id = ?", (condition_id,)
        ).fetchone()[0]
    )
    conn.execute(
        """
        UPDATE corpus_markets
        SET onchain_processed_at = ?,
            onchain_trades_count = ?
        WHERE condition_id = ?
        """,
        (now_ts, count, condition_id),
    )
    conn.commit()
    return count


async def _backfill_one_market(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    condition_id: str,
    page_size: int,
) -> tuple[int, int, int, int]:
    """Returns (events_decoded, trades_inserted, skipped_unsupported, skipped_unresolvable)."""
    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    asset_ids = _load_market_asset_ids(conn, condition_id)
    if not asset_ids:
        return 0, 0, 0, 0

    events_decoded = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    pending: list[CorpusTrade] = []

    async for event, ts in iter_market_trades(
        client=client, asset_ids=asset_ids, page_size=page_size
    ):
        events_decoded += 1
        try:
            trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=ts)
        except UnsupportedFill:
            skipped_unsupported += 1
            continue
        except UnresolvableAsset:
            skipped_unresolvable += 1
            continue
        if trade.condition_id != condition_id:
            # Defensive: shouldn't happen since the asset filter is by this condition's
            # assets, but the asset_index could in principle be stale. Drop silently.
            continue
        pending.append(trade)

    inserted = trades_repo.insert_batch(pending) if pending else 0
    return events_decoded, inserted, skipped_unsupported, skipped_unresolvable


async def run_subgraph_backfill(
    *,
    conn: sqlite3.Connection,
    client: SubgraphClient,
    page_size: int = _MAX_PAGE_SIZE,
    limit: int | None = None,
) -> SubgraphRunSummary:
    """Process every truncated, unprocessed market via the subgraph.

    Args:
        conn: Open corpus DB connection.
        client: Open ``SubgraphClient``.
        page_size: GraphQL ``first:`` per query (max 1000).
        limit: Process at most ``N`` markets in this run.
    """
    pending = _load_pending_markets(conn, limit=limit)
    _LOG.info("subgraph.start", markets=len(pending))

    processed = 0
    failed = 0
    total_events = 0
    total_inserted = 0
    total_unsupported = 0
    total_unresolvable = 0

    for i, market in enumerate(pending, start=1):
        try:
            events, inserted, unsup, unres = await _backfill_one_market(
                conn=conn,
                client=client,
                condition_id=market.condition_id,
                page_size=page_size,
            )
            total_events += events
            total_inserted += inserted
            total_unsupported += unsup
            total_unresolvable += unres
            count = _mark_processed(conn, market.condition_id, now_ts=int(time.time()))
            processed += 1
            _LOG.info(
                "subgraph.market_done",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id[:14] + "...",
                slug=market.market_slug[:50],
                events_decoded=events,
                trades_inserted=inserted,
                trade_count=count,
            )
        except Exception as exc:
            failed += 1
            _LOG.error(
                "subgraph.market_failed",
                idx=i,
                of=len(pending),
                condition_id=market.condition_id,
                error=str(exc),
            )

    cleared = clear_truncation_flags(conn=conn) if processed > 0 else 0

    summary = SubgraphRunSummary(
        markets_processed=processed,
        markets_failed=failed,
        events_decoded=total_events,
        trades_inserted=total_inserted,
        skipped_unsupported=total_unsupported,
        skipped_unresolvable=total_unresolvable,
        truncation_flags_cleared=cleared,
    )
    _LOG.info("subgraph.run_done", **summary.__dict__)
    return summary
```

- [ ] **Step 5.4: Re-run orchestrator tests**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -v`
Expected: all PASS.

- [ ] **Step 5.5: Lint, type-check the module**

Run: `uv run ruff check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py && uv run ruff format --check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py && uv run ty check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py`
Expected: clean.

- [ ] **Step 5.6: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py
git commit -m "feat(corpus): run_subgraph_backfill orchestrator with resume + limit"
```

---

## Task 6: Wire `pscanner corpus subgraph-backfill` CLI

**Files:**
- Modify: `src/pscanner/corpus/cli.py`
- Modify: `tests/corpus/test_cli.py`

Add the subparser, handler, and `_HANDLERS` entry. The handler reads `GRAPH_API_KEY` from env if `--api-key` isn't passed; constructs the gateway URL from `--subgraph-id` (default = the value confirmed in Task 1).

- [ ] **Step 6.1: Write failing CLI smoke test**

Append to `tests/corpus/test_cli.py`:

```python
async def test_subgraph_backfill_subcommand_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pscanner corpus subgraph-backfill --db ... --api-key X` dispatches the handler."""
    from pscanner.corpus import cli as corpus_cli

    db_path = tmp_path / "c.sqlite3"

    captured: dict[str, object] = {}

    async def fake_run(*, conn, client, page_size, limit):  # type: ignore[no-untyped-def]
        captured["conn"] = conn
        captured["client_url"] = client.url
        captured["client_rpm"] = client.rpm
        captured["page_size"] = page_size
        captured["limit"] = limit
        from pscanner.corpus.subgraph_ingest import SubgraphRunSummary

        return SubgraphRunSummary(0, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(corpus_cli, "run_subgraph_backfill", fake_run)

    rc = await corpus_cli.run_corpus_command(
        [
            "subgraph-backfill",
            "--db",
            str(db_path),
            "--api-key",
            "test-key",
            "--subgraph-id",
            "abc123",
            "--rpm",
            "120",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    assert "test-key" in str(captured["client_url"])
    assert "abc123" in str(captured["client_url"])
    assert captured["client_rpm"] == 120
    assert captured["limit"] == 5
```

- [ ] **Step 6.2: Run, confirm `argparse` failure**

Run: `uv run pytest tests/corpus/test_cli.py::test_subgraph_backfill_subcommand_dispatches -v`
Expected: FAIL — argparse rejects unknown subcommand.

- [ ] **Step 6.3: Add subparser, handler, and import in `src/pscanner/corpus/cli.py`**

At the imports near the top, append:

```python
import os

from pscanner.corpus.subgraph_ingest import run_subgraph_backfill
from pscanner.poly.subgraph import SubgraphClient
```

Below the existing `_DEFAULT_*` constants, add:

```python
_DEFAULT_SUBGRAPH_RPM = 600
_DEFAULT_SUBGRAPH_PAGE_SIZE = 1000
# TODO(plan task 1): replace with verified id once Graph Explorer lookup completes.
_DEFAULT_SUBGRAPH_ID = "REPLACE_AFTER_TASK_1"
_GATEWAY_URL_TEMPLATE = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"
```

In `build_corpus_parser`, add a new subparser block after the `ot = sub.add_parser("onchain-backfill-targeted", ...)` block, **before** `return parser`:

```python
sg = sub.add_parser(
    "subgraph-backfill",
    help=(
        "Per-market subgraph-driven backfill of truncated markets (resumable). "
        "Replaces eth_getLogs path with GraphQL queries against The Graph."
    ),
)
_add_db_arg(sg)
sg.add_argument(
    "--api-key",
    type=str,
    default=None,
    help="Graph Studio API key. Falls back to $GRAPH_API_KEY.",
)
sg.add_argument(
    "--subgraph-id",
    type=str,
    default=_DEFAULT_SUBGRAPH_ID,
    help=f"Subgraph deployment id (default: {_DEFAULT_SUBGRAPH_ID}).",
)
sg.add_argument(
    "--rpm",
    type=int,
    default=_DEFAULT_SUBGRAPH_RPM,
    help=f"Subgraph queries per minute (default: {_DEFAULT_SUBGRAPH_RPM}).",
)
sg.add_argument(
    "--page-size",
    type=int,
    default=_DEFAULT_SUBGRAPH_PAGE_SIZE,
    help=f"Rows per query, max 1000 (default: {_DEFAULT_SUBGRAPH_PAGE_SIZE}).",
)
sg.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Process at most N markets in this run (default: no limit).",
)
```

After `_cmd_onchain_backfill_targeted`, add the new handler:

```python
async def _cmd_subgraph_backfill(args: argparse.Namespace) -> int:
    """Run the subgraph-driven per-market backfill."""
    api_key = args.api_key or os.environ.get("GRAPH_API_KEY")
    if not api_key:
        raise SystemExit(
            "subgraph-backfill requires --api-key or $GRAPH_API_KEY"
        )
    if args.subgraph_id == _DEFAULT_SUBGRAPH_ID:
        raise SystemExit(
            "subgraph-backfill requires --subgraph-id (the placeholder default has not been "
            "replaced — see plan Task 1)."
        )
    url = _GATEWAY_URL_TEMPLATE.format(api_key=api_key, subgraph_id=args.subgraph_id)
    conn = init_corpus_db(Path(args.db))
    try:
        async with SubgraphClient(url=url, rpm=args.rpm) as client:
            summary = await run_subgraph_backfill(
                conn=conn,
                client=client,
                page_size=args.page_size,
                limit=args.limit,
            )
        _log.info(
            "subgraph.cli_summary",
            markets_processed=summary.markets_processed,
            markets_failed=summary.markets_failed,
            events_decoded=summary.events_decoded,
            trades_inserted=summary.trades_inserted,
            skipped_unsupported=summary.skipped_unsupported,
            skipped_unresolvable=summary.skipped_unresolvable,
            truncation_flags_cleared=summary.truncation_flags_cleared,
        )
        return 0
    finally:
        conn.close()
```

In `_HANDLERS`, add the entry:

```python
_HANDLERS = {
    "backfill": _cmd_backfill,
    "refresh": _cmd_refresh,
    "build-features": _cmd_build_features,
    "onchain-backfill": _cmd_onchain_backfill,
    "onchain-backfill-targeted": _cmd_onchain_backfill_targeted,
    "subgraph-backfill": _cmd_subgraph_backfill,
}
```

- [ ] **Step 6.4: Re-run CLI test**

Run: `uv run pytest tests/corpus/test_cli.py::test_subgraph_backfill_subcommand_dispatches -v`
Expected: PASS.

- [ ] **Step 6.5: Add a `--api-key` missing-env failure test**

Append to `tests/corpus/test_cli.py`:

```python
async def test_subgraph_backfill_missing_api_key_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GRAPH_API_KEY", raising=False)
    from pscanner.corpus import cli as corpus_cli

    with pytest.raises(SystemExit, match="GRAPH_API_KEY"):
        await corpus_cli.run_corpus_command(
            [
                "subgraph-backfill",
                "--db",
                str(tmp_path / "c.sqlite3"),
                "--subgraph-id",
                "abc",
            ]
        )
```

- [ ] **Step 6.6: Extend the parser-recognises-all-subcommands test**

In `tests/corpus/test_cli.py`, the existing `test_parser_recognises_all_subcommands` test asserts each known subcommand parses. Append the new one:

```python
    assert parser.parse_args(
        ["subgraph-backfill", "--api-key", "k", "--subgraph-id", "abc"]
    ).command == "subgraph-backfill"
```

(Note: `--api-key` and `--subgraph-id` are required as positional-style by the handler's runtime guard, but argparse accepts the parse with their defaults since they're declared `default=None` / `default=_DEFAULT_SUBGRAPH_ID`. The explicit values here just exercise the round-trip.)

- [ ] **Step 6.7: Run all CLI tests + lint/type-check**

Run: `uv run pytest tests/corpus/test_cli.py -v && uv run ruff check src/pscanner/corpus/cli.py tests/corpus/test_cli.py && uv run ty check src/pscanner/corpus/cli.py tests/corpus/test_cli.py`
Expected: clean.

- [ ] **Step 6.8: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(cli): pscanner corpus subgraph-backfill subcommand"
```

---

## Task 7: Replace the `_DEFAULT_SUBGRAPH_ID` placeholder

**Files:**
- Modify: `src/pscanner/corpus/cli.py`

After Task 1 has confirmed the actual subgraph deployment id, replace the placeholder so the CLI doesn't `SystemExit` on the default.

- [ ] **Step 7.1: Replace the placeholder constant**

In `src/pscanner/corpus/cli.py`, change:

```python
_DEFAULT_SUBGRAPH_ID = "REPLACE_AFTER_TASK_1"
```

to the verified id from Task 1's documentation update, e.g.:

```python
_DEFAULT_SUBGRAPH_ID = "<verified-id>"
```

- [ ] **Step 7.2: Remove the matching `SystemExit` guard**

In `_cmd_subgraph_backfill`, delete the `if args.subgraph_id == _DEFAULT_SUBGRAPH_ID:` block.

- [ ] **Step 7.3: Update the placeholder-failure test**

In `tests/corpus/test_cli.py`, the `test_subgraph_backfill_missing_api_key_exits` test still passes a `--subgraph-id`, so it's unaffected. No further test changes needed unless a placeholder-rejection test was added (it wasn't).

- [ ] **Step 7.4: Run all corpus tests**

Run: `uv run pytest tests/corpus/ -v`
Expected: all PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/pscanner/corpus/cli.py
git commit -m "feat(cli): pin verified Polymarket Orderbook subgraph id as default"
```

---

## Task 8: Live validation against real subgraph

**Files:** none — operational task. Captures findings in CLAUDE.md (Task 9).

- [ ] **Step 8.1: Pick the validation market**

Use the same NFL market the Phase 2 smoke used. From the Phase 2 plan: `0xf6a2fb22f50d...` (full condition_id documented in `docs/superpowers/plans/2026-05-03-onchain-trades-phase2.md`). Confirm it's still in `corpus_markets WHERE truncated_at_offset_cap = 1`:

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
row = conn.execute(
    \"SELECT condition_id, market_slug, truncated_at_offset_cap, onchain_processed_at, "
    \"(SELECT COUNT(*) FROM corpus_trades WHERE condition_id = m.condition_id) AS n_trades \"
    \"FROM corpus_markets m WHERE condition_id LIKE '0xf6a2fb22f50d%'\"
).fetchone()
print(dict(row) if row else 'not found')
"
```

If the Phase 2 smoke already processed it and cleared the flag, pick another truncated market (highest `total_volume_usd` from `WHERE truncated_at_offset_cap = 1 AND onchain_processed_at IS NULL`).

- [ ] **Step 8.2: Reset the chosen market to "truncated, unprocessed"**

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
conn.execute(
    'UPDATE corpus_markets SET onchain_processed_at = NULL, truncated_at_offset_cap = 1 WHERE condition_id = ?',
    ('<chosen condition_id>',)
)
conn.commit()
"
```

- [ ] **Step 8.3: Run subgraph backfill against just that market**

```bash
GRAPH_API_KEY=$(cat ~/.config/pscanner/graph_api_key) \
  uv run pscanner corpus subgraph-backfill --limit 1 2>&1 | tee /tmp/subgraph-smoke.log
```

Expected: structlog output ending with `subgraph.cli_summary markets_processed=1 markets_failed=0 trades_inserted=<N>` where `N` >= 0.

- [ ] **Step 8.4: Compare against the REST baseline**

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
conn.row_factory = sqlite3.Row
n = conn.execute(
    'SELECT COUNT(*) AS n FROM corpus_trades WHERE condition_id = ?',
    ('<chosen condition_id>',)
).fetchone()['n']
print(f'corpus_trades for market: {n}')
"
```

Expected: substantially more rows than the pre-run count (which should have been ≤3000 — the offset cap). Document the before/after numbers.

- [ ] **Step 8.5: Spot-check a few subgraph rows match REST**

Pick 3 random `tx_hash` values from `corpus_trades` for the chosen market and confirm they appear in the Polymarket data API's `/trades?condition=<id>&user=<wallet>` response (the same baseline used in Phase 2 validation). Note: the subgraph captures direct-EOA fills that the REST `/activity` endpoint misses, so a few "subgraph-only" rows are expected and *not* a regression.

- [ ] **Step 8.6: Run full backfill (no limit)**

```bash
GRAPH_API_KEY=$(cat ~/.config/pscanner/graph_api_key) \
  uv run pscanner corpus subgraph-backfill 2>&1 | tee /tmp/subgraph-full.log
```

Expected runtime: tens of minutes for ~2,570 markets at ~600 q/min.

Verify post-run:

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('data/corpus.sqlite3')
n = conn.execute('SELECT COUNT(*) FROM corpus_markets WHERE truncated_at_offset_cap = 1').fetchone()[0]
print(f'still-truncated markets: {n}')
"
```

Expected: small residual (likely Negative-Risk markets the subgraph indexes but our `asset_index` doesn't have entries for, OR markets where fewer than 3000 total trades ever happened so the threshold can't clear).

- [ ] **Step 8.7: Rebuild features + retrain the ML model**

```bash
uv run pscanner corpus build-features --rebuild
# Then on the desktop training box (per LOCAL_NOTES.md):
uv run pscanner ml train --device cuda --n-jobs 1
```

Compare `mean test edge` against the post-#40 baseline (4.05%). Record the new number for the issue's DoD.

This step **does not block the plan PR** — it's the validation that gates the follow-up "delete eth_getLogs" PR.

---

## Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 9.1: Update the "CLI surface" section**

Find the line beginning `- pscanner corpus onchain-backfill ...` and add immediately after it (preserving the existing line):

```markdown
- `pscanner corpus subgraph-backfill [--api-key KEY] [--subgraph-id ID] [--rpm N] [--page-size N] [--limit N]` — preferred replacement for `onchain-backfill-targeted`. Queries Polymarket's Orderbook subgraph on The Graph, filtered by `makerAssetId_in` / `takerAssetId_in` from local `asset_index`. Resumable via `corpus_markets.onchain_processed_at`. Free-tier API key (~100K queries/month) covers full corpus runs comfortably.
```

- [ ] **Step 9.2: Update the "Polymarket API quirks" section**

Find the bullet starting `` - `/trades` and `/activity` REST cap at offset=3000 `` and append after the existing on-chain note:

```markdown
- **Phase 3 (subgraph) supersedes the eth_getLogs corpus path.** `pscanner corpus subgraph-backfill` queries `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/{ID}` (Orderbook subgraph; both `Exchange` and `NegRiskExchange` write to the same `OrderFilledEvent` entity, so neg-risk markets are auto-covered). The `eth_getLogs` paths (`onchain-backfill`, `onchain-backfill-targeted`) stay during the transition but will be deleted in a follow-up commit once the subgraph path is fully validated. The decoder (`pscanner.poly.onchain.decode_order_filled`) and `AssetIndexRepo` survive the deletion — they're still useful for low-level event inspection.
```

- [ ] **Step 9.3: Run the full quick-verify gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
Expected: clean across all 800+ tests.

- [ ] **Step 9.4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): point at subgraph-backfill as preferred corpus path"
```

---

## Post-implementation (separate PR, after live validation lands)

When Task 8 has confirmed the subgraph path works end-to-end and ML retrain shows the expected lift, file a follow-up that deletes:

- `src/pscanner/poly/onchain_rpc.py` + `tests/poly/test_onchain_rpc.py`
- `iter_order_filled_logs` and `IngestRunSummary` in `src/pscanner/poly/onchain_ingest.py` (keep `event_to_corpus_trade` and the exception classes)
- `src/pscanner/corpus/onchain_backfill.py` + `tests/corpus/test_onchain_backfill.py`
- `src/pscanner/corpus/onchain_targeted.py` + `tests/corpus/test_onchain_targeted.py`
- `onchain-backfill` and `onchain-backfill-targeted` subparsers/handlers in `src/pscanner/corpus/cli.py` + their CLI tests
- The `_DEFAULT_RPC_URL` / `_DEFAULT_FROM_BLOCK` / etc. constants in `cli.py` that only the deleted commands reference

Keep:
- `src/pscanner/poly/onchain.py` (decoder, dataclass, `CTF_EXCHANGE_ADDRESS`, `ORDER_FILLED_TOPIC0`) — useful for ad-hoc on-chain inspection
- `clear_truncation_flags` — moved out of `onchain_backfill.py` into a standalone module (`src/pscanner/corpus/truncation.py` is a reasonable home), since `subgraph_ingest.py` calls it
- `event_to_corpus_trade`, `UnsupportedFill`, `UnresolvableAsset` from `onchain_ingest.py` — `subgraph_ingest.py` uses them
- `AssetIndexRepo` and the `asset_index` table

The `clear_truncation_flags` move is a refactor that should land **with** the deletion PR, not before — moving it earlier adds churn for no benefit.

---

## Definition of done (this PR)

- [ ] `pscanner corpus subgraph-backfill` runs end-to-end against the real corpus.
- [ ] `SELECT COUNT(*) FROM corpus_markets WHERE truncated_at_offset_cap = 1` drops to a small residual after a full run.
- [ ] All quick-verify checks pass: `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q`
- [ ] CLAUDE.md documents the new path.
- [ ] Test count is up by ~10-12 (one per task that adds tests).
- [ ] The eth_getLogs path is **still in place** — its deletion is the follow-up PR's job.
