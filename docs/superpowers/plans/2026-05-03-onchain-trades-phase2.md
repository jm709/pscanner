# On-chain Trades — Phase 2: `eth_getLogs` Backfill + CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `eth_getLogs` against Polygon mainnet, walk every `OrderFilled` event on the Polymarket CTF Exchange contract from deployment to head, decode via the Phase 1 module, resolve assets via `AssetIndexRepo`, and write into `corpus_trades`. Result: zero markets flagged `truncated_at_offset_cap=1` for those whose offset cap was hit, and a measurable lift on ML test edge.

**Architecture:** A new `pscanner.poly.onchain_rpc` module exposes a thin async JSON-RPC client (`OnchainRpcClient`) with `get_block_number`, `get_logs`, and `get_block_timestamp`. A new `pscanner.poly.onchain_ingest` module composes the client with the Phase 1 decoder and `AssetIndexRepo` to produce `CorpusTrade` rows, inserted via `CorpusTradesRepo`. A `pscanner corpus onchain-backfill` subcommand drives the ingest, persists progress in `corpus_state['onchain_last_block']`, and at the end clears `truncated_at_offset_cap` flags whose markets are now whole.

**Tech Stack:** Python 3.13, `httpx` async, `tenacity` retries (mirroring `pscanner.poly.http`'s pattern), `sqlite3`, `respx` for HTTP mocking in tests. No new third-party dependencies. Decoder + `AssetIndexRepo` both come from Phase 1 and stay untouched.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/pscanner/poly/onchain_rpc.py` | create | Async JSON-RPC client (`get_block_number`, `get_logs`, `get_block_timestamp` with LRU cache, 429/5xx backoff) |
| `tests/poly/test_onchain_rpc.py` | create | Tests for the RPC client against `respx`-mocked endpoints |
| `src/pscanner/poly/onchain_ingest.py` | create | Pure functions: `event_to_corpus_trade`, `iter_order_filled_logs` block-range paginator, run summary |
| `tests/poly/test_onchain_ingest.py` | create | Tests for event→trade conversion + paginator (uses `respx`) |
| `src/pscanner/corpus/onchain_backfill.py` | create | Orchestration: cursor reads/writes, chunk loop, trade insert, truncation flag clearance |
| `tests/corpus/test_onchain_backfill.py` | create | End-to-end test (mocked RPC) against `tmp_corpus_db` |
| `src/pscanner/corpus/cli.py` | modify | Add `onchain-backfill` subparser + `_cmd_onchain_backfill` handler |
| `tests/corpus/test_cli.py` | modify | Add CLI smoke for `corpus onchain-backfill` |
| `src/pscanner/corpus/db.py` | modify | Migration: add `corpus_markets.onchain_trades_count INTEGER` column |
| `tests/corpus/test_db.py` | modify | Migration idempotency assertion for new column |
| `CLAUDE.md` | modify | Document on-chain ingest as a first-class data path; record contract deployment block |

## What's already in place from Phase 1

- `pscanner.poly.onchain.CTF_EXCHANGE_ADDRESS` — `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- `pscanner.poly.onchain.ORDER_FILLED_TOPIC0`
- `pscanner.poly.onchain.OrderFilledEvent` (frozen dataclass) + `decode_order_filled(log)`
- `pscanner.corpus.repos.AssetIndexRepo.get(asset_id) -> AssetEntry | None` populated from `corpus_trades` (~8,626 entries)
- `corpus_state` key/value table + `CorpusStateRepo`
- `corpus_markets.truncated_at_offset_cap` column

## Out of scope

- Negative-Risk adapter contract (separate event source — file follow-up issue if residual is meaningful)
- Daemon-side incremental sync via `eth_getLogs` (Phase 3)
- WebSocket subscription (Phase 3+)
- Rebuilding `training_examples` and ML retrain (separate workflow, not a code change — covered in §Post-implementation)

---

## Task 1: Migration — add `corpus_markets.onchain_trades_count` column

**Files:**
- Modify: `src/pscanner/corpus/db.py`
- Modify: `tests/corpus/test_db.py`

The truncation-flag clearance step (Task 11) needs a per-market on-chain trade count to compare against the in-DB count. We persist it on `corpus_markets` for resumability (if the run crashes mid-clearance) and observability (DB-readable).

- [ ] **Step 1.1: Write failing test for the new column**

```python
# tests/corpus/test_db.py — append at the bottom

def test_corpus_markets_has_onchain_trades_count_column() -> None:
    """Phase 2 migration: corpus_markets.onchain_trades_count is present and nullable."""
    conn = init_corpus_db(Path(":memory:"))
    try:
        cols = {row["name"]: row for row in conn.execute("PRAGMA table_info(corpus_markets)")}
        assert "onchain_trades_count" in cols
        assert cols["onchain_trades_count"]["type"].upper() == "INTEGER"
        assert cols["onchain_trades_count"]["notnull"] == 0
    finally:
        conn.close()
```

- [ ] **Step 1.2: Run test, confirm failure**

Run: `uv run pytest tests/corpus/test_db.py::test_corpus_markets_has_onchain_trades_count_column -v`
Expected: FAIL — `KeyError: 'onchain_trades_count'`.

- [ ] **Step 1.3: Add the migration statement**

Append to the `_MIGRATIONS` tuple in `src/pscanner/corpus/db.py`:

```python
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE corpus_markets ADD COLUMN market_slug TEXT",
    "DROP INDEX IF EXISTS idx_corpus_trades_ts",
    "ALTER TABLE corpus_markets ADD COLUMN onchain_trades_count INTEGER",
)
```

(Existing `_apply_migrations` already swallows `duplicate column name` so re-running is a no-op.)

- [ ] **Step 1.4: Run test, confirm pass**

Run: `uv run pytest tests/corpus/test_db.py::test_corpus_markets_has_onchain_trades_count_column -v`
Expected: PASS.

- [ ] **Step 1.5: Run full corpus test suite to confirm no regression**

Run: `uv run pytest tests/corpus/ -q`
Expected: PASS.

- [ ] **Step 1.6: Commit**

```bash
git add src/pscanner/corpus/db.py tests/corpus/test_db.py
git commit -m "feat(corpus): add onchain_trades_count column for truncation clearance"
```

---

## Task 2: `OnchainRpcClient` — `get_block_number`

**Files:**
- Create: `src/pscanner/poly/onchain_rpc.py`
- Create: `tests/poly/test_onchain_rpc.py`

The RPC client wraps a single `httpx.AsyncClient` for JSON-RPC POSTs. We start with `eth_blockNumber` to lock in the request shape, response parsing, and a closeable async lifecycle.

- [ ] **Step 2.1: Write failing test for `get_block_number`**

```python
# tests/poly/test_onchain_rpc.py
"""Tests for `pscanner.poly.onchain_rpc` — Polygon JSON-RPC client."""

from __future__ import annotations

import httpx
import pytest
import respx

from pscanner.poly.onchain_rpc import OnchainRpcClient

_RPC_URL = "https://example-rpc.test/"


@pytest.fixture
def client() -> OnchainRpcClient:
    """A fresh RPC client per test (high rpm to neutralise rate limiting)."""
    return OnchainRpcClient(rpc_url=_RPC_URL, rpm=600, timeout_seconds=5.0)


@respx.mock
async def test_get_block_number_returns_int(client: OnchainRpcClient) -> None:
    respx.post(_RPC_URL).mock(
        return_value=httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "result": "0x1f4abcd"}
        ),
    )
    try:
        head = await client.get_block_number()
    finally:
        await client.aclose()
    assert head == 0x1F4ABCD
```

- [ ] **Step 2.2: Run, confirm failure**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v`
Expected: FAIL — `ModuleNotFoundError: pscanner.poly.onchain_rpc`.

- [ ] **Step 2.3: Implement minimal client**

```python
# src/pscanner/poly/onchain_rpc.py
"""Async JSON-RPC client for Polygon mainnet (eth_* methods).

Targets any EVM-compatible RPC endpoint. Default is Polygon Foundation's
public RPC (`https://polygon-rpc.com/`), free and unauthenticated.
Override via constructor for Alchemy or other providers.

Mirrors the rate-limiting + retry pattern in `pscanner.poly.http` but
speaks JSON-RPC POST instead of REST GET.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Self

import httpx
import structlog

_LOG = structlog.get_logger(__name__)
_USER_AGENT = "pscanner/0.1"


class OnchainRpcClient:
    """Async JSON-RPC client for `eth_*` calls against any Polygon RPC.

    Long-lived: open once, reuse across an ingest run, close on shutdown.
    The httpx client is created lazily on first use so construction is cheap.
    """

    def __init__(
        self,
        *,
        rpc_url: str,
        rpm: int,
        timeout_seconds: float = 30.0,
    ) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        self.rpc_url = rpc_url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False
        self._next_id = 1

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("OnchainRpcClient is closed")
        if self._client is not None:
            return self._client
        async with self._init_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout_seconds),
                    headers={"User-Agent": _USER_AGENT, "Content-Type": "application/json"},
                )
            return self._client

    async def _call(self, method: str, params: list[Any]) -> Any:
        client = await self._ensure_client()
        request_id = self._next_id
        self._next_id += 1
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = await client.post(self.rpc_url, json=body)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"RPC error from {method}: {payload['error']}")
        return payload["result"]

    async def get_block_number(self) -> int:
        """Return the current Polygon head block number."""
        result = await self._call("eth_blockNumber", [])
        return int(result, 16)

    async def aclose(self) -> None:
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> Self:
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
```

- [ ] **Step 2.4: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_rpc.py::test_get_block_number_returns_int -v`
Expected: PASS.

- [ ] **Step 2.5: Lint + types**

Run: `uv run ruff check src/pscanner/poly/onchain_rpc.py tests/poly/test_onchain_rpc.py && uv run ty check src/pscanner/poly/onchain_rpc.py`
Expected: clean.

- [ ] **Step 2.6: Commit**

```bash
git add src/pscanner/poly/onchain_rpc.py tests/poly/test_onchain_rpc.py
git commit -m "feat(poly): OnchainRpcClient with get_block_number"
```

---

## Task 3: `OnchainRpcClient.get_logs`

**Files:**
- Modify: `src/pscanner/poly/onchain_rpc.py`
- Modify: `tests/poly/test_onchain_rpc.py`

`eth_getLogs` is the load-bearing call: takes `address`, `topics`, `fromBlock`, `toBlock`, returns a list of raw log dicts. Block bounds passed as hex strings (`"0x..."`).

- [ ] **Step 3.1: Write failing test**

```python
# tests/poly/test_onchain_rpc.py — append

@respx.mock
async def test_get_logs_passes_hex_block_bounds(client: OnchainRpcClient) -> None:
    captured: list[dict[str, object]] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        body = request.read()
        import json as _json

        captured.append(_json.loads(body))
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "result": []}
        )

    respx.post(_RPC_URL).mock(side_effect=_capture)
    try:
        logs = await client.get_logs(
            address="0xabc",
            topics=["0xdeadbeef"],
            from_block=100,
            to_block=200,
        )
    finally:
        await client.aclose()

    assert logs == []
    assert captured[0]["method"] == "eth_getLogs"
    params = captured[0]["params"][0]
    assert params["address"] == "0xabc"
    assert params["topics"] == ["0xdeadbeef"]
    assert params["fromBlock"] == "0x64"  # 100
    assert params["toBlock"] == "0xc8"   # 200


@respx.mock
async def test_get_logs_returns_payload(client: OnchainRpcClient) -> None:
    log = {
        "address": "0xabc",
        "topics": ["0xdeadbeef"],
        "data": "0x" + "00" * 256,
        "transactionHash": "0x" + "ab" * 32,
        "blockNumber": "0x10",
        "logIndex": "0x0",
    }
    respx.post(_RPC_URL).mock(
        return_value=httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "result": [log]}
        ),
    )
    try:
        logs = await client.get_logs(
            address="0xabc", topics=["0xdeadbeef"], from_block=16, to_block=16
        )
    finally:
        await client.aclose()

    assert len(logs) == 1
    assert logs[0]["transactionHash"] == "0x" + "ab" * 32
```

- [ ] **Step 3.2: Run, confirm failure**

Run: `uv run pytest tests/poly/test_onchain_rpc.py::test_get_logs_passes_hex_block_bounds -v`
Expected: FAIL — `AttributeError: 'OnchainRpcClient' object has no attribute 'get_logs'`.

- [ ] **Step 3.3: Implement `get_logs`**

Add to `OnchainRpcClient`:

```python
async def get_logs(
    self,
    *,
    address: str,
    topics: list[str],
    from_block: int,
    to_block: int,
) -> list[dict[str, Any]]:
    """Fetch logs matching `address` and `topics` between two block bounds.

    Args:
        address: Contract address (lowercase or checksummed; RPC accepts both).
        topics: Topic filter; `topics[0]` is the event signature hash.
        from_block: First block in the inclusive range.
        to_block: Last block in the inclusive range.

    Returns:
        List of raw log dicts as returned by the RPC.

    Raises:
        RuntimeError: If the RPC returns a JSON-RPC error.
        httpx.HTTPStatusError: On non-2xx HTTP status.
    """
    params = [
        {
            "address": address,
            "topics": topics,
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
        }
    ]
    result = await self._call("eth_getLogs", params)
    if not isinstance(result, list):
        raise RuntimeError(f"eth_getLogs returned non-list result: {result!r}")
    return result
```

- [ ] **Step 3.4: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v -k get_logs`
Expected: 2 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/pscanner/poly/onchain_rpc.py tests/poly/test_onchain_rpc.py
git commit -m "feat(poly): OnchainRpcClient.get_logs with hex block bounds"
```

---

## Task 4: `OnchainRpcClient.get_block_timestamp` with LRU cache

**Files:**
- Modify: `src/pscanner/poly/onchain_rpc.py`
- Modify: `tests/poly/test_onchain_rpc.py`

A chunk of 5,000 blocks at 2s/block ≈ 10,000 seconds of activity. Decoded events reference blocks redundantly — cache `block_number → timestamp` to avoid duplicate RPC calls. Use a small functools-style dict-with-cap LRU (no third-party dep).

- [ ] **Step 4.1: Write failing test**

```python
# tests/poly/test_onchain_rpc.py — append

@respx.mock
async def test_get_block_timestamp_returns_int_from_hex(
    client: OnchainRpcClient,
) -> None:
    respx.post(_RPC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"timestamp": "0x65f0a000", "number": "0x10"},
            },
        ),
    )
    try:
        ts = await client.get_block_timestamp(16)
    finally:
        await client.aclose()
    assert ts == 0x65F0A000


@respx.mock
async def test_get_block_timestamp_caches_repeat_calls(
    client: OnchainRpcClient,
) -> None:
    route = respx.post(_RPC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"timestamp": "0x100", "number": "0xa"},
            },
        ),
    )
    try:
        ts_a = await client.get_block_timestamp(10)
        ts_b = await client.get_block_timestamp(10)
    finally:
        await client.aclose()

    assert ts_a == ts_b == 0x100
    assert route.call_count == 1, "second call should hit the cache"
```

- [ ] **Step 4.2: Run, confirm failure**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v -k timestamp`
Expected: FAIL — `AttributeError: 'OnchainRpcClient' object has no attribute 'get_block_timestamp'`.

- [ ] **Step 4.3: Implement timestamp lookup with capped LRU**

Add to `OnchainRpcClient`:

```python
# Module-level constant
_BLOCK_TIMESTAMP_CACHE_SIZE = 4096
```

And inside `__init__`, after the existing fields:

```python
self._ts_cache: dict[int, int] = {}
```

Add the method:

```python
async def get_block_timestamp(self, block_number: int) -> int:
    """Return the Unix-second timestamp of the given Polygon block.

    Caches the (block_number → timestamp) mapping in-memory; capped to
    `_BLOCK_TIMESTAMP_CACHE_SIZE` entries. When the cap is hit we evict
    the oldest insertion (FIFO is fine — Polygon walk is forward-monotonic
    block-wise so older blocks rarely re-appear).
    """
    cached = self._ts_cache.get(block_number)
    if cached is not None:
        return cached
    result = await self._call(
        "eth_getBlockByNumber", [hex(block_number), False]
    )
    if not isinstance(result, dict) or "timestamp" not in result:
        raise RuntimeError(
            f"eth_getBlockByNumber({block_number}) returned malformed payload: {result!r}"
        )
    ts = int(result["timestamp"], 16)
    if len(self._ts_cache) >= _BLOCK_TIMESTAMP_CACHE_SIZE:
        # Evict oldest insertion — dict preserves insertion order in 3.7+.
        oldest = next(iter(self._ts_cache))
        del self._ts_cache[oldest]
    self._ts_cache[block_number] = ts
    return ts
```

- [ ] **Step 4.4: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v -k timestamp`
Expected: 2 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/pscanner/poly/onchain_rpc.py tests/poly/test_onchain_rpc.py
git commit -m "feat(poly): get_block_timestamp with bounded LRU cache"
```

---

## Task 5: 429 / 5xx backoff on the RPC client

**Files:**
- Modify: `src/pscanner/poly/onchain_rpc.py`
- Modify: `tests/poly/test_onchain_rpc.py`

Polygon Foundation rate-limits anonymous traffic. Mirror `pscanner.poly.http`'s tenacity-based retry pattern (exponential backoff with `Retry-After` honouring) but adapted for POST. The decision to copy structure rather than refactor `PolyHttpClient` to handle POST: REST and JSON-RPC have different success-detection (JSON-RPC's "error" key vs. REST's HTTP status alone), so a shared abstraction would muddy both.

- [ ] **Step 5.1: Write failing test for retry on 429**

```python
# tests/poly/test_onchain_rpc.py — append

@respx.mock
async def test_429_with_retry_after_zero_retries_and_succeeds(
    client: OnchainRpcClient,
) -> None:
    respx.post(_RPC_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}, json={"err": "rl"}),
            httpx.Response(
                200, json={"jsonrpc": "2.0", "id": 1, "result": "0x1"}
            ),
        ],
    )
    try:
        head = await client.get_block_number()
    finally:
        await client.aclose()
    assert head == 1


@respx.mock
async def test_persistent_503_raises_after_max_attempts(
    client: OnchainRpcClient,
) -> None:
    route = respx.post(_RPC_URL).mock(
        return_value=httpx.Response(503, json={"err": "down"}),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_block_number()
    finally:
        await client.aclose()
    assert route.call_count == 5  # _MAX_ATTEMPTS in pscanner.poly.http
```

- [ ] **Step 5.2: Run, confirm failure**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v -k "retry_after or persistent_503"`
Expected: FAIL — second response on `429` test never observed (client raises on first 429).

- [ ] **Step 5.3: Add retry plumbing to `_call`**

Top-of-file imports:

```python
from email.utils import parsedate_to_datetime
from datetime import UTC, datetime

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
```

Module-level constants (mirroring `pscanner.poly.http`):

```python
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


class _RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"retryable status {response.status_code}")
        self.response = response


def _parse_retry_after(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        seconds = float(stripped)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)
    try:
        when = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(tz=UTC)).total_seconds())


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RetryableStatusError):
        return True
    return isinstance(exc, _RETRYABLE_TRANSPORT_EXC)


def _before_sleep_log(retry_state: RetryCallState) -> None:
    outcome = retry_state.outcome
    if outcome is None:
        return
    exc = outcome.exception()
    if not isinstance(exc, _RetryableStatusError):
        return
    response = exc.response
    _LOG.warning(
        "polygon_rpc_retry",
        attempt=retry_state.attempt_number,
        status_code=response.status_code,
        retry_after=response.headers.get("Retry-After"),
    )
```

Replace `_call` body:

```python
async def _call(self, method: str, params: list[Any]) -> Any:
    client = await self._ensure_client()
    request_id = self._next_id
    self._next_id += 1
    body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}

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
                response = await self._send_once(client, body)
    except _RetryableStatusError as exc:
        exc.response.raise_for_status()
        raise
    if response is None:
        raise RuntimeError("retry loop produced no response")
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"RPC error from {method}: {payload['error']}")
    return payload["result"]


async def _send_once(
    self, client: httpx.AsyncClient, body: dict[str, Any]
) -> httpx.Response:
    response = await client.post(self.rpc_url, json=body)
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
```

- [ ] **Step 5.4: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v -k "retry_after or persistent_503"`
Expected: 2 PASS.

- [ ] **Step 5.5: Run full RPC suite**

Run: `uv run pytest tests/poly/test_onchain_rpc.py -v`
Expected: all PASS.

- [ ] **Step 5.6: Commit**

```bash
git add src/pscanner/poly/onchain_rpc.py tests/poly/test_onchain_rpc.py
git commit -m "feat(poly): tenacity retries on 429/5xx for OnchainRpcClient"
```

---

## Task 6: Event → CorpusTrade conversion

**Files:**
- Create: `src/pscanner/poly/onchain_ingest.py`
- Create: `tests/poly/test_onchain_ingest.py`

Convert one decoded `OrderFilledEvent` into one `CorpusTrade`. The CTF Exchange settles CTF tokens against USDC, so for any non-merge/split fill exactly **one** of `maker_asset_id` / `taker_asset_id` is `0` (USDC) and the other is the CTF position id. The wallet whose action we record is the **taker** (the order initiator), with `side = "BUY"` if they gave USDC (`taker_asset_id == 0`) or `"SELL"` if they gave the CTF token. This matches the Polymarket REST `/trades` semantics where the row's `proxyWallet` is the trader's address and `side` is from their POV.

USDC and CTF tokens both have **6 decimals on Polygon mainnet**. Price (USDC per token) = `usdc_amount / token_amount`. Notional USD = `usdc_amount`.

Edge cases to handle defensively:
- Both asset IDs zero → impossible (USDC swap with itself); log + skip.
- Both asset IDs non-zero → same-condition split/merge (Negative-Risk-adjacent); skip with debug log.
- Asset ID not in `AssetIndexRepo` → log + skip (counted in run summary).

- [ ] **Step 6.1: Write failing test for the happy-path BUY conversion**

```python
# tests/poly/test_onchain_ingest.py
"""Tests for `pscanner.poly.onchain_ingest` — event→trade conversion + paginator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import OrderFilledEvent
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)


@pytest.fixture
def asset_repo(tmp_path: Path) -> AssetIndexRepo:
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    repo = AssetIndexRepo(conn)
    repo.upsert(
        AssetEntry(
            asset_id="123456789",
            condition_id="0xCONDITION",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    return repo


def _ev(
    *,
    maker: str = "0x" + "11" * 20,
    taker: str = "0x" + "22" * 20,
    maker_asset_id: int = 0,
    taker_asset_id: int = 123_456_789,
    making: int = 700_000,  # 0.70 USDC (6 decimals)
    taking: int = 1_000_000,  # 1.0 CTF (6 decimals)
) -> OrderFilledEvent:
    return OrderFilledEvent(
        order_hash="0x" + "ab" * 32,
        maker=maker,
        taker=taker,
        maker_asset_id=maker_asset_id,
        taker_asset_id=taker_asset_id,
        making=making,
        taking=taking,
        fee=0,
        tx_hash="0x" + "cd" * 32,
        block_number=42,
        log_index=0,
    )


def test_event_to_corpus_trade_buy_taker_gives_usdc(
    asset_repo: AssetIndexRepo,
) -> None:
    """Taker giving USDC for CTF tokens is a BUY from the taker's POV."""
    event = _ev(maker_asset_id=123_456_789, taker_asset_id=0,
                making=1_000_000, taking=700_000)
    # maker is GIVING the CTF token (the resting limit) → taker is the BUYER.
    trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=1_700_000_000)
    assert isinstance(trade, CorpusTrade)
    assert trade.tx_hash == "0x" + "cd" * 32
    assert trade.asset_id == "123456789"
    assert trade.condition_id == "0xCONDITION"
    assert trade.outcome_side == "YES"
    assert trade.wallet_address == "0x" + "22" * 20  # taker
    assert trade.bs == "BUY"
    assert trade.price == pytest.approx(0.70)
    assert trade.size == pytest.approx(1.0)
    assert trade.notional_usd == pytest.approx(0.70)
    assert trade.ts == 1_700_000_000
```

- [ ] **Step 6.2: Run, confirm failure**

Run: `uv run pytest tests/poly/test_onchain_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: pscanner.poly.onchain_ingest`.

- [ ] **Step 6.3: Implement minimal `event_to_corpus_trade`**

```python
# src/pscanner/poly/onchain_ingest.py
"""Convert decoded `OrderFilled` events to `CorpusTrade` rows.

Pure functions and the block-range paginator. The orchestration loop
that drives state mutations on `corpus_markets` lives in
`pscanner.corpus.onchain_backfill`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import OrderFilledEvent

_LOG = structlog.get_logger(__name__)

# USDC and Polymarket CTF tokens both use 6 decimals on Polygon mainnet.
_DECIMALS = 1_000_000.0


class UnsupportedFill(Exception):
    """Raised when an OrderFilled event does not represent a CTF↔USDC swap."""


class UnresolvableAsset(Exception):
    """Raised when neither asset id is known to `AssetIndexRepo`."""


def event_to_corpus_trade(
    event: OrderFilledEvent,
    *,
    asset_repo: AssetIndexRepo,
    ts: int,
) -> CorpusTrade:
    """Convert one `OrderFilledEvent` to a `CorpusTrade` from the taker's POV.

    Args:
        event: Decoded event payload.
        asset_repo: Lookup for `(asset_id → condition_id, outcome_side)`.
        ts: Block timestamp in Unix seconds.

    Returns:
        A `CorpusTrade` row. Caller inserts via `CorpusTradesRepo`.

    Raises:
        UnsupportedFill: Both asset ids zero, or both non-zero (split/merge).
        UnresolvableAsset: The CTF asset id is not in `asset_repo`.
    """
    maker_id, taker_id = event.maker_asset_id, event.taker_asset_id
    maker_is_usdc = maker_id == 0
    taker_is_usdc = taker_id == 0
    if maker_is_usdc == taker_is_usdc:
        raise UnsupportedFill(
            f"both-zero or both-non-zero asset ids: maker={maker_id}, taker={taker_id}"
        )

    # Taker initiated; their side determines BUY/SELL.
    if taker_is_usdc:
        # Taker gave USDC, received CTF token → taker BUY.
        bs = "BUY"
        usdc_amount = event.taking
        ctf_amount = event.making
        ctf_asset_id = maker_id
    else:
        # Taker gave CTF token, received USDC → taker SELL.
        bs = "SELL"
        usdc_amount = event.making
        ctf_amount = event.taking
        ctf_asset_id = taker_id

    asset_id_str = str(ctf_asset_id)
    entry = asset_repo.get(asset_id_str)
    if entry is None:
        raise UnresolvableAsset(asset_id_str)

    if ctf_amount == 0:
        # Defensive: zero-size fill (shouldn't reach the contract, but guard).
        raise UnsupportedFill(f"zero ctf amount in fill {event.tx_hash}")

    price = usdc_amount / ctf_amount
    size = ctf_amount / _DECIMALS
    notional_usd = usdc_amount / _DECIMALS

    return CorpusTrade(
        tx_hash=event.tx_hash,
        asset_id=asset_id_str,
        wallet_address=event.taker.lower(),
        condition_id=entry.condition_id,
        outcome_side=entry.outcome_side,
        bs=bs,
        price=price,
        size=size,
        notional_usd=notional_usd,
        ts=ts,
    )
```

- [ ] **Step 6.4: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_ingest.py::test_event_to_corpus_trade_buy_taker_gives_usdc -v`
Expected: PASS.

- [ ] **Step 6.5: Add SELL-side test**

```python
# tests/poly/test_onchain_ingest.py — append

def test_event_to_corpus_trade_sell_taker_gives_ctf(
    asset_repo: AssetIndexRepo,
) -> None:
    """Taker giving CTF tokens for USDC is a SELL from the taker's POV."""
    event = _ev(
        maker_asset_id=0,
        taker_asset_id=123_456_789,
        making=420_000,    # 0.42 USDC the maker gives
        taking=1_000_000,  # 1.0 CTF the taker gives
    )
    trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=1_700_000_000)
    assert trade.bs == "SELL"
    assert trade.wallet_address == "0x" + "22" * 20  # taker
    assert trade.price == pytest.approx(0.42)
    assert trade.size == pytest.approx(1.0)
    assert trade.notional_usd == pytest.approx(0.42)
```

- [ ] **Step 6.6: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_ingest.py -v`
Expected: 2 PASS.

- [ ] **Step 6.7: Add edge-case tests**

```python
# tests/poly/test_onchain_ingest.py — append

def test_event_to_corpus_trade_raises_when_both_assets_zero(
    asset_repo: AssetIndexRepo,
) -> None:
    event = _ev(maker_asset_id=0, taker_asset_id=0)
    with pytest.raises(UnsupportedFill, match="both-zero"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)


def test_event_to_corpus_trade_raises_when_both_assets_nonzero(
    asset_repo: AssetIndexRepo,
) -> None:
    event = _ev(maker_asset_id=42, taker_asset_id=99)
    with pytest.raises(UnsupportedFill, match="both-zero or both-non-zero"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)


def test_event_to_corpus_trade_raises_when_asset_unknown(
    asset_repo: AssetIndexRepo,
) -> None:
    event = _ev(maker_asset_id=0, taker_asset_id=999_999_999)  # not in repo
    with pytest.raises(UnresolvableAsset, match="999999999"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)
```

- [ ] **Step 6.8: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_ingest.py -v`
Expected: 5 PASS.

- [ ] **Step 6.9: Commit**

```bash
git add src/pscanner/poly/onchain_ingest.py tests/poly/test_onchain_ingest.py
git commit -m "feat(poly): event_to_corpus_trade with USDC/CTF resolution"
```

---

## Task 7: `iter_order_filled_logs` block-range paginator

**Files:**
- Modify: `src/pscanner/poly/onchain_ingest.py`
- Modify: `tests/poly/test_onchain_ingest.py`

Wraps the RPC client and yields decoded events in chunk-sized block ranges. Handles Polygon RPC's per-call block-range limits by chunking. Each yield carries the block timestamp (fetched lazily once per unique block).

- [ ] **Step 7.1: Write failing test**

```python
# tests/poly/test_onchain_ingest.py — append

import httpx
import respx

from pscanner.poly.onchain import CTF_EXCHANGE_ADDRESS, ORDER_FILLED_TOPIC0
from pscanner.poly.onchain_ingest import iter_order_filled_logs
from pscanner.poly.onchain_rpc import OnchainRpcClient


def _synthetic_log(
    *,
    block_number: int,
    log_index: int,
    tx_hash: str,
    maker_asset_id: int,
    taker_asset_id: int,
    making: int,
    taking: int,
    maker: str = "0x" + "11" * 20,
    taker: str = "0x" + "22" * 20,
) -> dict[str, object]:
    parts = [
        bytes(32),  # order_hash
        bytes(12) + bytes.fromhex(maker[2:]),
        bytes(12) + bytes.fromhex(taker[2:]),
        maker_asset_id.to_bytes(32, "big"),
        taker_asset_id.to_bytes(32, "big"),
        making.to_bytes(32, "big"),
        taking.to_bytes(32, "big"),
        (0).to_bytes(32, "big"),
    ]
    return {
        "data": "0x" + b"".join(parts).hex(),
        "topics": [ORDER_FILLED_TOPIC0],
        "transactionHash": tx_hash,
        "blockNumber": hex(block_number),
        "logIndex": hex(log_index),
    }


@respx.mock
async def test_iter_order_filled_logs_chunks_and_yields() -> None:
    rpc_url = "https://example-rpc.test/"

    log_a = _synthetic_log(
        block_number=100,
        log_index=0,
        tx_hash="0x" + "aa" * 32,
        maker_asset_id=0,
        taker_asset_id=42,
        making=500_000,
        taking=1_000_000,
    )
    log_b = _synthetic_log(
        block_number=200,
        log_index=1,
        tx_hash="0x" + "bb" * 32,
        maker_asset_id=42,
        taker_asset_id=0,
        making=1_000_000,
        taking=500_000,
    )

    # eth_blockNumber, eth_getLogs[0..199], eth_getLogs[200..299], eth_getBlockByNumber x2
    posts: list[dict[str, object]] = []

    def _route(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.read())
        posts.append(body)
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x12c"}  # 300
            )
        if method == "eth_getLogs":
            params = body["params"][0]
            from_b = int(params["fromBlock"], 16)
            if from_b == 0:
                return httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log_a]}
                )
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log_b]}
            )
        if method == "eth_getBlockByNumber":
            block = int(body["params"][0], 16)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": hex(1_700_000_000 + block)},
                },
            )
        raise AssertionError(f"unexpected method: {method}")

    respx.post(rpc_url).mock(side_effect=_route)

    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    yielded: list[tuple[int, int]] = []
    try:
        async for event, ts in iter_order_filled_logs(
            rpc=client,
            from_block=0,
            to_block=300,
            chunk_size=200,
        ):
            yielded.append((event.block_number, ts))
    finally:
        await client.aclose()

    assert yielded == [(100, 1_700_000_100), (200, 1_700_000_200)]
    methods = [p["method"] for p in posts]
    # Two get_logs calls, one per chunk; two block timestamp lookups.
    assert methods.count("eth_getLogs") == 2
    assert methods.count("eth_getBlockByNumber") == 2
```

- [ ] **Step 7.2: Run, confirm failure**

Run: `uv run pytest tests/poly/test_onchain_ingest.py::test_iter_order_filled_logs_chunks_and_yields -v`
Expected: FAIL — `iter_order_filled_logs` undefined.

- [ ] **Step 7.3: Implement the paginator**

Append to `src/pscanner/poly/onchain_ingest.py`:

```python
from pscanner.poly.onchain import (
    CTF_EXCHANGE_ADDRESS,
    ORDER_FILLED_TOPIC0,
    decode_order_filled,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient


@dataclass(frozen=True)
class IngestRunSummary:
    """Per-run accumulator returned by the orchestrator."""

    chunks_processed: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    last_block: int


async def iter_order_filled_logs(
    *,
    rpc: OnchainRpcClient,
    from_block: int,
    to_block: int,
    chunk_size: int = 5_000,
) -> AsyncIterator[tuple["OrderFilledEvent", int]]:
    """Yield decoded `OrderFilled` events from `from_block`..`to_block` inclusive.

    Each yielded tuple is `(event, block_timestamp)`. Walks in fixed-size
    chunks; pre-fetches block timestamps via the RPC client's cache.

    Args:
        rpc: Async RPC client (caller owns lifecycle).
        from_block: Inclusive start block.
        to_block: Inclusive end block.
        chunk_size: Blocks per `eth_getLogs` call (default 5,000).

    Yields:
        `(OrderFilledEvent, unix_timestamp_seconds)` in `(blockNumber, logIndex)`
        order within each chunk.
    """
    if from_block > to_block:
        return
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    cursor = from_block
    while cursor <= to_block:
        chunk_end = min(cursor + chunk_size - 1, to_block)
        logs = await rpc.get_logs(
            address=CTF_EXCHANGE_ADDRESS,
            topics=[ORDER_FILLED_TOPIC0],
            from_block=cursor,
            to_block=chunk_end,
        )
        for log in logs:
            event = decode_order_filled(log)
            ts = await rpc.get_block_timestamp(event.block_number)
            yield event, ts
        cursor = chunk_end + 1
```

- [ ] **Step 7.4: Run, confirm pass**

Run: `uv run pytest tests/poly/test_onchain_ingest.py::test_iter_order_filled_logs_chunks_and_yields -v`
Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/pscanner/poly/onchain_ingest.py tests/poly/test_onchain_ingest.py
git commit -m "feat(poly): iter_order_filled_logs block-range paginator"
```

---

## Task 8: `run_onchain_backfill` orchestrator

**Files:**
- Create: `src/pscanner/corpus/onchain_backfill.py`
- Create: `tests/corpus/test_onchain_backfill.py`

The orchestrator wires `iter_order_filled_logs` to `CorpusTradesRepo.insert_batch`, persists the cursor in `corpus_state['onchain_last_block']`, and produces an `IngestRunSummary`. It batches inserts at chunk boundaries (commit on each chunk, so a kill mid-run doesn't lose the chunk's progress).

- [ ] **Step 8.1: Write failing end-to-end test against an in-memory corpus**

```python
# tests/corpus/test_onchain_backfill.py
"""End-to-end test for `pscanner.corpus.onchain_backfill`."""

from __future__ import annotations

import json as _json
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.onchain_backfill import run_onchain_backfill
from pscanner.corpus.repos import (
    AssetEntry,
    AssetIndexRepo,
    CorpusStateRepo,
    CorpusTradesRepo,
)
from pscanner.poly.onchain import ORDER_FILLED_TOPIC0
from pscanner.poly.onchain_rpc import OnchainRpcClient


def _synthetic_log(
    *,
    block_number: int,
    log_index: int,
    tx_hash: str,
    maker_asset_id: int,
    taker_asset_id: int,
    making: int,
    taking: int,
    maker: str = "0x" + "11" * 20,
    taker: str = "0x" + "22" * 20,
) -> dict[str, object]:
    parts = [
        bytes(32),
        bytes(12) + bytes.fromhex(maker[2:]),
        bytes(12) + bytes.fromhex(taker[2:]),
        maker_asset_id.to_bytes(32, "big"),
        taker_asset_id.to_bytes(32, "big"),
        making.to_bytes(32, "big"),
        taking.to_bytes(32, "big"),
        (0).to_bytes(32, "big"),
    ]
    return {
        "data": "0x" + b"".join(parts).hex(),
        "topics": [ORDER_FILLED_TOPIC0],
        "transactionHash": tx_hash,
        "blockNumber": hex(block_number),
        "logIndex": hex(log_index),
    }


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = init_corpus_db(tmp_path / "corpus.sqlite3")
    AssetIndexRepo(db).upsert(
        AssetEntry(
            asset_id="42",
            condition_id="0xCONDITION",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    return db


@respx.mock
async def test_run_onchain_backfill_inserts_trades_and_advances_cursor(
    conn: sqlite3.Connection,
) -> None:
    rpc_url = "https://example-rpc.test/"
    log = _synthetic_log(
        block_number=150,
        log_index=0,
        tx_hash="0x" + "aa" * 32,
        maker_asset_id=42,
        taker_asset_id=0,
        # taker gave 0.50 USDC for 1.0 CTF (BUY @ 0.50)
        making=1_000_000,
        taking=500_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0xc8"}  # 200
            )
        if method == "eth_getLogs":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log]}
            )
        if method == "eth_getBlockByNumber":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": "0x65f0a000"},
                },
            )
        raise AssertionError(method)

    respx.post(rpc_url).mock(side_effect=_route)

    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        summary = await run_onchain_backfill(
            conn=conn,
            rpc=client,
            from_block=0,
            to_block=200,
            chunk_size=100,
        )
    finally:
        await client.aclose()

    assert summary.events_decoded == 1
    assert summary.trades_inserted == 1
    assert summary.skipped_unresolvable == 0
    assert summary.last_block == 200

    rows = conn.execute(
        "SELECT bs, price, notional_usd, wallet_address FROM corpus_trades"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["bs"] == "BUY"
    assert rows[0]["price"] == pytest.approx(0.50)
    assert rows[0]["notional_usd"] == pytest.approx(0.50)
    assert rows[0]["wallet_address"] == "0x" + "22" * 20

    # cursor advanced
    cursor = CorpusStateRepo(conn).get_int("onchain_last_block")
    assert cursor == 200
```

- [ ] **Step 8.2: Run, confirm failure**

Run: `uv run pytest tests/corpus/test_onchain_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError: pscanner.corpus.onchain_backfill`.

- [ ] **Step 8.3: Implement the orchestrator**

```python
# src/pscanner/corpus/onchain_backfill.py
"""Orchestrate on-chain `OrderFilled` ingest into `corpus_trades`.

Composes `iter_order_filled_logs` with `event_to_corpus_trade` and writes
through `CorpusTradesRepo`. Tracks the chunk cursor in
`corpus_state['onchain_last_block']` so partial runs are resumable.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import structlog

from pscanner.corpus.repos import (
    AssetIndexRepo,
    CorpusStateRepo,
    CorpusTradesRepo,
)
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
    iter_order_filled_logs,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient

_LOG = structlog.get_logger(__name__)
_STATE_KEY = "onchain_last_block"


@dataclass(frozen=True)
class IngestRunSummary:
    chunks_processed: int
    events_decoded: int
    trades_inserted: int
    skipped_unsupported: int
    skipped_unresolvable: int
    last_block: int


async def run_onchain_backfill(
    *,
    conn: sqlite3.Connection,
    rpc: OnchainRpcClient,
    from_block: int,
    to_block: int,
    chunk_size: int = 5_000,
) -> IngestRunSummary:
    """Walk [from_block, to_block], decode events, insert trades, persist cursor.

    Args:
        conn: Open corpus DB connection (must have the corpus schema applied).
        rpc: Async RPC client (caller owns lifecycle).
        from_block: First block (inclusive).
        to_block: Last block (inclusive).
        chunk_size: Blocks per `eth_getLogs` call.

    Returns:
        `IngestRunSummary` with per-run counts.
    """
    asset_repo = AssetIndexRepo(conn)
    trades_repo = CorpusTradesRepo(conn)
    state_repo = CorpusStateRepo(conn)

    events_decoded = 0
    trades_inserted = 0
    skipped_unsupported = 0
    skipped_unresolvable = 0
    last_block = from_block - 1
    chunks_processed = 0
    pending: list = []
    chunk_boundary = from_block + chunk_size - 1

    async for event, ts in iter_order_filled_logs(
        rpc=rpc,
        from_block=from_block,
        to_block=to_block,
        chunk_size=chunk_size,
    ):
        events_decoded += 1
        # Flush + advance cursor when a chunk boundary is crossed.
        while event.block_number > chunk_boundary:
            inserted_count = trades_repo.insert_batch(pending)
            trades_inserted += inserted_count
            pending = []
            last_block = chunk_boundary
            state_repo.set(_STATE_KEY, str(last_block), updated_at=int(time.time()))
            chunks_processed += 1
            chunk_boundary = min(chunk_boundary + chunk_size, to_block)

        try:
            trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=ts)
        except UnsupportedFill as exc:
            skipped_unsupported += 1
            _LOG.debug("onchain.skip_unsupported", reason=str(exc))
            continue
        except UnresolvableAsset as exc:
            skipped_unresolvable += 1
            _LOG.debug("onchain.skip_unresolvable", asset_id=str(exc))
            continue
        pending.append(trade)

    if pending:
        trades_inserted += trades_repo.insert_batch(pending)
    last_block = to_block
    state_repo.set(_STATE_KEY, str(last_block), updated_at=int(time.time()))
    chunks_processed += 1

    summary = IngestRunSummary(
        chunks_processed=chunks_processed,
        events_decoded=events_decoded,
        trades_inserted=trades_inserted,
        skipped_unsupported=skipped_unsupported,
        skipped_unresolvable=skipped_unresolvable,
        last_block=last_block,
    )
    _LOG.info(
        "onchain.backfill_done",
        chunks=summary.chunks_processed,
        events=summary.events_decoded,
        inserted=summary.trades_inserted,
        skipped_unsupported=summary.skipped_unsupported,
        skipped_unresolvable=summary.skipped_unresolvable,
        last_block=summary.last_block,
    )
    return summary
```

- [ ] **Step 8.4: Run, confirm pass**

Run: `uv run pytest tests/corpus/test_onchain_backfill.py -v`
Expected: PASS.

- [ ] **Step 8.5: Add unresolvable-asset test**

```python
# tests/corpus/test_onchain_backfill.py — append

@respx.mock
async def test_run_onchain_backfill_skips_unknown_asset(
    conn: sqlite3.Connection,
) -> None:
    """Asset id absent from `asset_index` → skipped, counted, no insert."""
    rpc_url = "https://example-rpc.test/"
    log = _synthetic_log(
        block_number=10,
        log_index=0,
        tx_hash="0x" + "ee" * 32,
        maker_asset_id=999_999_999,  # not in fixture's asset_index
        taker_asset_id=0,
        making=1_000_000,
        taking=500_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x32"}
            )
        if method == "eth_getLogs":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log]}
            )
        if method == "eth_getBlockByNumber":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": "0x65f0a000"},
                },
            )
        raise AssertionError(method)

    respx.post(rpc_url).mock(side_effect=_route)
    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        summary = await run_onchain_backfill(
            conn=conn, rpc=client, from_block=0, to_block=50, chunk_size=50
        )
    finally:
        await client.aclose()

    assert summary.events_decoded == 1
    assert summary.trades_inserted == 0
    assert summary.skipped_unresolvable == 1
    assert conn.execute("SELECT COUNT(*) FROM corpus_trades").fetchone()[0] == 0
```

- [ ] **Step 8.6: Run, confirm pass**

Run: `uv run pytest tests/corpus/test_onchain_backfill.py -v`
Expected: 2 PASS.

- [ ] **Step 8.7: Add resumability test**

```python
# tests/corpus/test_onchain_backfill.py — append

@respx.mock
async def test_run_onchain_backfill_is_idempotent(
    conn: sqlite3.Connection,
) -> None:
    """Re-running with the same range produces no new inserts (UNIQUE bounces)."""
    rpc_url = "https://example-rpc.test/"
    log = _synthetic_log(
        block_number=10,
        log_index=0,
        tx_hash="0x" + "ff" * 32,
        maker_asset_id=42,
        taker_asset_id=0,
        making=1_000_000,
        taking=500_000,
    )

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x32"}
            )
        if method == "eth_getLogs":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log]}
            )
        if method == "eth_getBlockByNumber":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"timestamp": "0x65f0a000"},
                },
            )
        raise AssertionError(method)

    respx.post(rpc_url).mock(side_effect=_route)
    client = OnchainRpcClient(rpc_url=rpc_url, rpm=600)
    try:
        first = await run_onchain_backfill(
            conn=conn, rpc=client, from_block=0, to_block=50, chunk_size=50
        )
        second = await run_onchain_backfill(
            conn=conn, rpc=client, from_block=0, to_block=50, chunk_size=50
        )
    finally:
        await client.aclose()

    assert first.trades_inserted == 1
    assert second.events_decoded == 1
    assert second.trades_inserted == 0  # UNIQUE constraint bounced the duplicate
    assert conn.execute("SELECT COUNT(*) FROM corpus_trades").fetchone()[0] == 1
```

- [ ] **Step 8.8: Run, confirm pass**

Run: `uv run pytest tests/corpus/test_onchain_backfill.py -v`
Expected: 3 PASS.

- [ ] **Step 8.9: Commit**

```bash
git add src/pscanner/corpus/onchain_backfill.py tests/corpus/test_onchain_backfill.py
git commit -m "feat(corpus): on-chain backfill orchestrator with cursor + summary"
```

---

## Task 9: Truncation flag clearance

**Files:**
- Modify: `src/pscanner/corpus/onchain_backfill.py`
- Modify: `tests/corpus/test_onchain_backfill.py`

After ingest, walk every market with `truncated_at_offset_cap = 1`, count its rows in `corpus_trades`, persist that count on `corpus_markets.onchain_trades_count`, and clear the flag if the count is now ≥ 3000 (the REST cap). For markets where on-chain returned _fewer_ trades than REST captured, leave the flag — this signals the market may live on the Negative-Risk adapter contract (out of scope).

- [ ] **Step 9.1: Write failing test**

```python
# tests/corpus/test_onchain_backfill.py — append

from pscanner.corpus.onchain_backfill import clear_truncation_flags
from pscanner.corpus.repos import CorpusMarket, CorpusMarketsRepo


def test_clear_truncation_flags_clears_market_above_threshold(
    conn: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(conn)
    cid = "0xMARKET_BIG"
    markets.insert_pending(
        CorpusMarket(
            condition_id=cid,
            event_slug="evt",
            category="politics",
            closed_at=1_700_000_000,
            total_volume_usd=2_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="some-slug",
        )
    )
    # Force the truncation flag to 1.
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1 WHERE condition_id = ?",
        (cid,),
    )
    # Seed 3500 trades to mimic a post-onchain enriched state.
    conn.executemany(
        """
        INSERT INTO corpus_trades (
          tx_hash, asset_id, wallet_address, condition_id, outcome_side,
          bs, price, size, notional_usd, ts
        ) VALUES (?, ?, ?, ?, 'YES', 'BUY', 0.5, 100.0, 50.0, ?)
        """,
        [
            (f"0xtx{i:04x}", "42", f"0x{i:040x}", cid, 1_700_000_000 + i)
            for i in range(3500)
        ],
    )
    conn.commit()

    cleared = clear_truncation_flags(conn=conn, threshold=3000)
    assert cleared == 1

    row = conn.execute(
        "SELECT truncated_at_offset_cap, onchain_trades_count "
        "FROM corpus_markets WHERE condition_id = ?",
        (cid,),
    ).fetchone()
    assert row["truncated_at_offset_cap"] == 0
    assert row["onchain_trades_count"] == 3500


def test_clear_truncation_flags_skips_below_threshold(
    conn: sqlite3.Connection,
) -> None:
    """Markets with corpus_trades count < threshold keep the flag set."""
    markets = CorpusMarketsRepo(conn)
    cid = "0xMARKET_SHORT"
    markets.insert_pending(
        CorpusMarket(
            condition_id=cid,
            event_slug="evt",
            category="politics",
            closed_at=1_700_000_000,
            total_volume_usd=2_000_000.0,
            enumerated_at=1_700_000_000,
            market_slug="some-slug",
        )
    )
    conn.execute(
        "UPDATE corpus_markets SET truncated_at_offset_cap = 1 WHERE condition_id = ?",
        (cid,),
    )
    cleared = clear_truncation_flags(conn=conn, threshold=3000)
    assert cleared == 0
    row = conn.execute(
        "SELECT truncated_at_offset_cap, onchain_trades_count "
        "FROM corpus_markets WHERE condition_id = ?",
        (cid,),
    ).fetchone()
    assert row["truncated_at_offset_cap"] == 1
    assert row["onchain_trades_count"] == 0  # the count is recorded regardless
```

- [ ] **Step 9.2: Run, confirm failure**

Run: `uv run pytest tests/corpus/test_onchain_backfill.py -v -k truncation`
Expected: FAIL — `clear_truncation_flags` undefined.

- [ ] **Step 9.3: Implement clearance helper**

Append to `src/pscanner/corpus/onchain_backfill.py`:

```python
def clear_truncation_flags(*, conn: sqlite3.Connection, threshold: int = 3000) -> int:
    """Refresh `corpus_markets.onchain_trades_count` and clear truncation flags.

    For every market where `truncated_at_offset_cap = 1`, count its rows in
    `corpus_trades`, persist that as `onchain_trades_count`, and clear the
    truncation flag iff the count is at or above `threshold` (default
    3000 = the REST `/trades` offset cap).

    Args:
        conn: Open corpus DB connection.
        threshold: Minimum `corpus_trades` row count required to clear the flag.

    Returns:
        Number of markets whose flag was cleared this call.
    """
    rows = conn.execute(
        """
        SELECT m.condition_id, COUNT(t.tx_hash) AS row_count
        FROM corpus_markets m
        LEFT JOIN corpus_trades t USING (condition_id)
        WHERE m.truncated_at_offset_cap = 1
        GROUP BY m.condition_id
        """
    ).fetchall()
    cleared = 0
    for row in rows:
        cid = row["condition_id"]
        count = int(row["row_count"])
        new_flag = 0 if count >= threshold else 1
        conn.execute(
            """
            UPDATE corpus_markets
            SET onchain_trades_count = ?,
                truncated_at_offset_cap = ?
            WHERE condition_id = ?
            """,
            (count, new_flag, cid),
        )
        if new_flag == 0:
            cleared += 1
    conn.commit()
    _LOG.info(
        "onchain.truncation_clearance_done",
        markets_examined=len(rows),
        cleared=cleared,
        threshold=threshold,
    )
    return cleared
```

- [ ] **Step 9.4: Run, confirm pass**

Run: `uv run pytest tests/corpus/test_onchain_backfill.py -v -k truncation`
Expected: 2 PASS.

- [ ] **Step 9.5: Commit**

```bash
git add src/pscanner/corpus/onchain_backfill.py tests/corpus/test_onchain_backfill.py
git commit -m "feat(corpus): clear truncated_at_offset_cap after on-chain ingest"
```

---

## Task 10: CLI command `pscanner corpus onchain-backfill`

**Files:**
- Modify: `src/pscanner/corpus/cli.py`
- Modify: `tests/corpus/test_cli.py`

Surface the orchestrator as a subcommand with the flags listed in the issue: `--from-block`, `--to-block`, `--rpc-url`, `--chunk-size`, `--max-blocks`. Resumes from `corpus_state['onchain_last_block']` when `--from-block` is omitted, and from the contract deployment block on first run.

The Polymarket CTF Exchange was deployed on Polygon at block **31_478_500** (Jan 19, 2023 — verified during plan authoring against the contract address; the implementer should re-confirm via PolygonScan and write the verified value as `_DEFAULT_FROM_BLOCK`).

- [ ] **Step 10.1: Write failing CLI test**

```python
# tests/corpus/test_cli.py — append

import json as _json

import httpx
import respx

from pscanner.corpus.cli import build_corpus_parser, run_corpus_command


def test_corpus_parser_has_onchain_backfill_subcommand() -> None:
    parser = build_corpus_parser()
    args = parser.parse_args(
        [
            "onchain-backfill",
            "--from-block",
            "100",
            "--to-block",
            "200",
            "--rpc-url",
            "https://x.test/",
            "--chunk-size",
            "50",
        ]
    )
    assert args.command == "onchain-backfill"
    assert args.from_block == 100
    assert args.to_block == 200
    assert args.rpc_url == "https://x.test/"
    assert args.chunk_size == 50


@respx.mock
async def test_run_corpus_command_onchain_backfill_inserts_trade(
    tmp_path: object,
) -> None:
    """End-to-end: CLI handler runs the orchestrator against a mocked RPC."""
    from pathlib import Path

    from pscanner.corpus.db import init_corpus_db
    from pscanner.corpus.repos import AssetEntry, AssetIndexRepo

    db = tmp_path / "corpus.sqlite3"  # type: ignore[operator]
    conn = init_corpus_db(Path(str(db)))
    AssetIndexRepo(conn).upsert(
        AssetEntry(
            asset_id="42",
            condition_id="0xCONDITION",
            outcome_side="YES",
            outcome_index=0,
        )
    )
    conn.close()

    log_data = (
        bytes(32)
        + bytes(12) + bytes.fromhex("11" * 20)
        + bytes(12) + bytes.fromhex("22" * 20)
        + (42).to_bytes(32, "big")
        + (0).to_bytes(32, "big")
        + (1_000_000).to_bytes(32, "big")
        + (500_000).to_bytes(32, "big")
        + (0).to_bytes(32, "big")
    )
    log = {
        "data": "0x" + log_data.hex(),
        "topics": ["0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"],
        "transactionHash": "0x" + "ab" * 32,
        "blockNumber": "0xa",
        "logIndex": "0x0",
    }

    def _route(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.read())
        method = body["method"]
        if method == "eth_blockNumber":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": "0x14"}
            )
        if method == "eth_getLogs":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": [log]}
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"timestamp": "0x65f0a000"},
            },
        )

    respx.post("https://example-rpc.test/").mock(side_effect=_route)

    rc = await run_corpus_command(
        [
            "onchain-backfill",
            "--db",
            str(db),
            "--from-block",
            "0",
            "--to-block",
            "20",
            "--chunk-size",
            "20",
            "--rpc-url",
            "https://example-rpc.test/",
        ]
    )
    assert rc == 0

    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(db))
    conn.row_factory = _sqlite3.Row
    try:
        rows = conn.execute("SELECT bs, price FROM corpus_trades").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["bs"] == "BUY"
    assert rows[0]["price"] == pytest.approx(0.50)
```

- [ ] **Step 10.2: Run, confirm failure**

Run: `uv run pytest tests/corpus/test_cli.py -v -k onchain`
Expected: FAIL on the parser test (`onchain-backfill` subcommand undefined).

- [ ] **Step 10.3: Add the subparser + handler**

Edit `src/pscanner/corpus/cli.py`:

Add imports near the top:

```python
from pscanner.corpus.onchain_backfill import (
    IngestRunSummary,
    clear_truncation_flags,
    run_onchain_backfill,
)
from pscanner.poly.onchain_rpc import OnchainRpcClient
```

Add module constants:

```python
_DEFAULT_RPC_URL = "https://polygon-rpc.com/"
_DEFAULT_FROM_BLOCK = 31_478_500  # CTF Exchange deployment (verify on PolygonScan)
_DEFAULT_CHUNK_SIZE = 5_000
_DEFAULT_MAX_BLOCKS = 1_000_000
```

In `build_corpus_parser`, append after the `bf` parser block:

```python
ob = sub.add_parser(
    "onchain-backfill",
    help="Walk CTF Exchange OrderFilled events and write to corpus_trades",
)
_add_db_arg(ob)
ob.add_argument(
    "--from-block",
    type=int,
    default=None,
    help=(
        "First block (inclusive). Default: corpus_state['onchain_last_block'] + 1, "
        f"or {_DEFAULT_FROM_BLOCK} on first run."
    ),
)
ob.add_argument(
    "--to-block",
    type=int,
    default=None,
    help="Last block (inclusive). Default: current Polygon head.",
)
ob.add_argument(
    "--rpc-url",
    type=str,
    default=_DEFAULT_RPC_URL,
    help=f"Polygon RPC endpoint (default: {_DEFAULT_RPC_URL})",
)
ob.add_argument(
    "--chunk-size",
    type=int,
    default=_DEFAULT_CHUNK_SIZE,
    help=f"Blocks per eth_getLogs call (default: {_DEFAULT_CHUNK_SIZE})",
)
ob.add_argument(
    "--max-blocks",
    type=int,
    default=_DEFAULT_MAX_BLOCKS,
    help=f"Safety cap per run (default: {_DEFAULT_MAX_BLOCKS})",
)
ob.add_argument(
    "--rpm",
    type=int,
    default=600,
    help="RPC requests per minute ceiling (default: 600)",
)
```

Add the handler:

```python
async def _cmd_onchain_backfill(args: argparse.Namespace) -> int:
    """Walk on-chain `OrderFilled` events into corpus_trades."""
    conn = init_corpus_db(Path(args.db))
    try:
        state = CorpusStateRepo(conn)
        cursor = state.get_int("onchain_last_block")
        from_block: int = (
            args.from_block
            if args.from_block is not None
            else (cursor + 1 if cursor is not None else _DEFAULT_FROM_BLOCK)
        )
        async with OnchainRpcClient(rpc_url=args.rpc_url, rpm=args.rpm) as rpc:
            to_block: int = (
                args.to_block if args.to_block is not None else await rpc.get_block_number()
            )
            if to_block < from_block:
                _log.info("onchain.nothing_to_do", from_block=from_block, to_block=to_block)
                return 0
            capped_to = min(to_block, from_block + args.max_blocks - 1)
            if capped_to < to_block:
                _log.info(
                    "onchain.capped_to_block",
                    requested_to=to_block,
                    capped_to=capped_to,
                    max_blocks=args.max_blocks,
                )
            summary: IngestRunSummary = await run_onchain_backfill(
                conn=conn,
                rpc=rpc,
                from_block=from_block,
                to_block=capped_to,
                chunk_size=args.chunk_size,
            )
        cleared = clear_truncation_flags(conn=conn)
        _log.info(
            "onchain.run_summary",
            chunks=summary.chunks_processed,
            events=summary.events_decoded,
            inserted=summary.trades_inserted,
            skipped_unsupported=summary.skipped_unsupported,
            skipped_unresolvable=summary.skipped_unresolvable,
            last_block=summary.last_block,
            truncation_flags_cleared=cleared,
        )
        return 0
    finally:
        conn.close()
```

Wire it into `_HANDLERS`:

```python
_HANDLERS = {
    "backfill": _cmd_backfill,
    "refresh": _cmd_refresh,
    "build-features": _cmd_build_features,
    "onchain-backfill": _cmd_onchain_backfill,
}
```

- [ ] **Step 10.4: Run, confirm pass**

Run: `uv run pytest tests/corpus/test_cli.py -v -k onchain`
Expected: 2 PASS.

- [ ] **Step 10.5: Commit**

```bash
git add src/pscanner/corpus/cli.py tests/corpus/test_cli.py
git commit -m "feat(cli): pscanner corpus onchain-backfill subcommand"
```

---

## Task 11: Real-RPC smoke test (manual)

**Files:** none — this is a manual verification step before the full historical sync.

The plan author has not validated `_DEFAULT_FROM_BLOCK` against PolygonScan, and the `event_to_corpus_trade` "taker is the trader" assumption deserves confirmation against REST-collected data on a known overlapping market. Both checks happen here.

- [ ] **Step 11.1: Verify the contract deployment block**

Open https://polygonscan.com/address/0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E in a browser, scroll to "Contract Creation" on the contract tab, and read the deployment block. Update `_DEFAULT_FROM_BLOCK` in `src/pscanner/corpus/cli.py` if it differs from `31_478_500`. Commit:

```bash
git add src/pscanner/corpus/cli.py
git commit -m "chore(cli): pin verified CTF Exchange deployment block"
```

(Skip the commit if the value was already correct.)

- [ ] **Step 11.2: Pick a known-overlap market**

Run against the live corpus DB:

```sql
SELECT m.condition_id, m.market_slug, m.trades_pulled_count, m.closed_at
FROM corpus_markets m
WHERE m.backfill_state = 'complete'
  AND m.truncated_at_offset_cap = 0
  AND m.trades_pulled_count BETWEEN 50 AND 200
ORDER BY m.closed_at DESC
LIMIT 5;
```

(Use `sqlite3 data/corpus.sqlite3 -header -column "<query>"` outside the Python process; the daemon should be stopped.)

Pick one row. Note its `condition_id`, `closed_at`, and `trades_pulled_count`. Capture the trade timestamp window:

```sql
SELECT MIN(ts), MAX(ts) FROM corpus_trades WHERE condition_id = '<chosen-cid>';
```

- [ ] **Step 11.3: Resolve timestamp window to a Polygon block range**

Use a public block-time lookup (`https://polygonscan.com/block/countdown` or `eth_getBlockByNumber` calls scoped by binary search on block number). Record the `[from_block, to_block]` that brackets the trade window with ~10K block of slack on each side.

- [ ] **Step 11.4: Run the on-chain backfill against a clone of the corpus**

```bash
cp data/corpus.sqlite3 data/corpus.smoke.sqlite3
uv run pscanner corpus onchain-backfill \
  --db data/corpus.smoke.sqlite3 \
  --from-block <FROM> \
  --to-block <TO> \
  --chunk-size 1000
```

Expected log lines: `onchain.backfill_done` with `events > 0` and `inserted >= 0`.

- [ ] **Step 11.5: Compare against REST-collected trades on the same market**

```sql
-- in data/corpus.smoke.sqlite3
SELECT COUNT(*), SUM(CASE WHEN bs='BUY' THEN 1 ELSE 0 END) AS buys
FROM corpus_trades
WHERE condition_id = '<chosen-cid>';
```

- The total count should be **at least** the original `trades_pulled_count` (UNIQUE constraint dedups REST rows; new on-chain rows for counterparty wallets only appear if their tx_hash+asset_id+wallet doesn't already exist).
- The buy/sell ratio should roughly match the original (some drift is expected because REST records the trader's POV, on-chain records the taker's POV — they should overlap heavily).

If buy/sell counts are wildly skewed (>3×), the "taker is the trader" assumption is wrong. **Stop here and revisit Task 6** before running the full historical sync. The fix is to swap maker/taker in `event_to_corpus_trade` and re-run this test.

- [ ] **Step 11.6: Clean up the smoke clone**

```bash
trash data/corpus.smoke.sqlite3
```

---

## Task 12: Full historical sync + truncation clearance

**Files:** none — manual operation.

- [ ] **Step 12.1: Stop the daemon if it's running**

```bash
pgrep -af "pscanner run" && echo "daemon is running — stop it first"
```

- [ ] **Step 12.2: Snapshot the corpus DB**

```bash
cp data/corpus.sqlite3 data/corpus.sqlite3.pre-onchain
```

- [ ] **Step 12.3: Run the full sync**

```bash
uv run pscanner corpus onchain-backfill \
  --rpc-url https://polygon-rpc.com/ \
  --max-blocks 50000000 \
  --chunk-size 5000
```

Expected runtime: 30-60 min on the public RPC (faster with Alchemy free tier — set `--rpc-url https://polygon-mainnet.g.alchemy.com/v2/<KEY>`). Watch for `polygon_rpc_retry` warnings — sustained 429s suggest dropping `--rpm`.

- [ ] **Step 12.4: Verify truncation flags**

```sql
SELECT COUNT(*) FROM corpus_markets WHERE truncated_at_offset_cap = 1;
```

Expected: `< 50` (residual from Negative-Risk markets).

- [ ] **Step 12.5: Verify median trades-per-market on the previously-truncated set**

Before the run (against `data/corpus.sqlite3.pre-onchain`):

```sql
SELECT COUNT(*) AS n, AVG(trades_pulled_count) AS mean_trades
FROM corpus_markets WHERE truncated_at_offset_cap = 1;
```

After the run (against `data/corpus.sqlite3`), join with `corpus_trades`:

```sql
WITH counts AS (
  SELECT m.condition_id, COUNT(t.tx_hash) AS post_count
  FROM corpus_markets m
  LEFT JOIN corpus_trades t USING (condition_id)
  WHERE m.condition_id IN (
    SELECT condition_id FROM corpus_markets WHERE truncated_at_offset_cap = 1
       OR onchain_trades_count > 3000
  )
  GROUP BY m.condition_id
)
SELECT COUNT(*) AS n, AVG(post_count) AS mean_post
FROM counts;
```

Expected: `mean_post` substantially exceeds `1883` (the prior median from the issue).

- [ ] **Step 12.6: Rebuild training_examples**

```bash
uv run pscanner corpus build-features --rebuild
```

(`--rebuild` truncates and recreates the table because the per-trade context features depend on the full trade history that just changed.)

---

## Task 13: ML retrain + edge comparison

**Files:** none — runs against the desktop training box (see `LOCAL_NOTES.md`).

- [ ] **Step 13.1: Sync the corpus to the training box**

(Per the `wsl2-tailscale-derp` memory, copy via `netsh portproxy` not Tailscale; check `LOCAL_NOTES.md` for the exact command.)

- [ ] **Step 13.2: Train**

```bash
uv run pscanner ml train --device cuda --n-jobs 1 --output models/<DATE>-onchain-baseline
```

Expected runtime: ~6.5 min on the RTX 3070.

- [ ] **Step 13.3: Compare edge vs the post-#40 baseline**

Use `scripts/analyze_model.py` to compare against `models/2026-05-03b-copy_trade_gate-real_temporal/` (the issue's reference baseline at 4.05% test edge). Capture:
- Test AUC (was 0.5xxx)
- Test edge (was 4.05%)
- Per-category edge — sports was 11.04%; expect lift on sports + politics where pre-trade history was thin.

If edge **decreased**, the new pre-trade rows likely include events whose timestamps land in feature-leakage windows. Investigate: does any new trade's `ts` fall inside the test fold while the corresponding market is in the train fold? Check via the `temporal_split` derivation in `pscanner.corpus.examples`.

- [ ] **Step 13.4: Record the result**

Append a one-line entry to `CLAUDE.md`'s "Open follow-ups" section noting the new edge number and the date the on-chain backfill ran.

---

## Task 14: Documentation updates

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 14.1: Add on-chain ingest as a first-class data path**

In the Polymarket API quirks section, replace the existing on-chain note (currently: "Phase 1 of #42 landed... Phase 2 (RPC client + CLI) is queued.") with:

```markdown
- `/trades` and `/activity` REST cap at `offset=3000` (server: `"max historical activity offset of 3000 exceeded"`, newest-first sort). On-chain ingest fills the gap: `pscanner corpus onchain-backfill` walks Polygon's CTF Exchange (`0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`) via `eth_getLogs` and decodes `OrderFilled` events into `corpus_trades`. Resumable via `corpus_state['onchain_last_block']`. Default RPC is `https://polygon-rpc.com/`; pass `--rpc-url` for Alchemy or other providers.
```

In the CLI surface section, append:

```markdown
- `pscanner corpus onchain-backfill [--from-block N] [--to-block N] [--rpc-url URL]` — walk Polygon CTF Exchange OrderFilled events into `corpus_trades`. Resumes from `corpus_state['onchain_last_block']` when `--from-block` is omitted.
```

- [ ] **Step 14.2: Verify CLAUDE.md still parses cleanly**

```bash
uv run ruff format --check CLAUDE.md  # ruff doesn't format markdown but this is a sanity-check the file isn't truncated
wc -l CLAUDE.md
```

- [ ] **Step 14.3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: on-chain ingest is a first-class corpus data path"
```

---

## Task 15: Final verification

**Files:** none — checklist run.

- [ ] **Step 15.1: Full repo verification**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
```

Expected: all green.

- [ ] **Step 15.2: Manual smoke**

```bash
uv run pscanner corpus onchain-backfill --help
```

Expected: subcommand help text lists every flag from Task 10.

- [ ] **Step 15.3: Confirm Definition of Done**

Cross-check against the issue's DoD:
- [ ] `pscanner corpus onchain-backfill` runs end-to-end and is resumable (Task 8 idempotency test + Task 12 manual run)
- [ ] `SELECT COUNT(*) FROM corpus_markets WHERE truncated_at_offset_cap = 1` < 50 (Task 12.4)
- [ ] Median trades-per-market on the previously-truncated set rises substantially above 1,883 (Task 12.5)
- [ ] ML retrain reports new test-edge numbers vs the post-#40 baseline (Task 13.3)
- [ ] `CLAUDE.md` updated (Task 14)
- [ ] Follow-up issue filed if Negative-Risk coverage is meaningful (see "Post-implementation" below)

---

## Post-implementation

If `clear_truncation_flags` left a meaningful residual (say, > 10 markets where on-chain returned <3000 logs), file a follow-up issue titled "On-chain ingest: Negative-Risk adapter coverage". The follow-up should:

1. List the residual `condition_id`s and confirm via PolygonScan whether each is a Negative-Risk market (typically multi-outcome events like "Which team wins league X").
2. Identify the Negative-Risk adapter contract address (per Polymarket's docs).
3. Repeat Task 7's paginator pattern against the new contract address with whatever its match event is.
