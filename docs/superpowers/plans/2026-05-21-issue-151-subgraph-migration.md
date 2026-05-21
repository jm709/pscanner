# Subgraph migration to current Polymarket Orderbook subgraph — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `pscanner corpus subgraph-backfill` from the stale subgraph `7fu2DWYK...` to the current real-time subgraph `B9mm21D...`, producing `corpus_trades` rows byte-identical to the old path so the backfill is fully idempotent against the existing corpus.

**Architecture:** Contained to `src/pscanner/corpus/subgraph_ingest.py` (rewrite the GraphQL query + row parser, collapse two side-queries into one), a one-line default change in `src/pscanner/corpus/cli.py`, test-fixture updates in `tests/corpus/test_subgraph_ingest.py`, and CLAUDE.md docs touch-up. Public function signatures unchanged; `OrderFilledEvent` dataclass unchanged; downstream `event_to_corpus_trade` unchanged.

**Tech Stack:** Python 3.13, `httpx` via existing `SubgraphClient`, `pytest` + `respx` for tests.

**Spec:** `docs/superpowers/specs/2026-05-21-issue-151-subgraph-migration-design.md`

---

## File structure

**Modified files:**

- `src/pscanner/corpus/subgraph_ingest.py` — rewrite `_REQUIRED_KEYS`, `subgraph_row_to_event`, `_TRADES_QUERY_*` constants; simplify `iter_market_trades` body to make one paginated call instead of two
- `src/pscanner/corpus/cli.py` — flip the default value of the `--subgraph-id` flag
- `tests/corpus/test_subgraph_ingest.py` — rewrite 13 existing test fixtures to the new schema shape, add 2 new side-mapping unit tests
- `CLAUDE.md` — resolve the "pending #151" forward references

**Unchanged:**

- `src/pscanner/poly/subgraph.py` — generic GraphQL client, schema-agnostic
- `src/pscanner/poly/onchain.py` (`OrderFilledEvent` dataclass) — internal field names stay the same
- `src/pscanner/poly/onchain_ingest.py` (`event_to_corpus_trade`) — reads only the dataclass, unaffected
- `src/pscanner/corpus/db.py`, `repos.py` — no schema changes
- All downstream consumers of `OrderFilledEvent`

---

## Task 1: Migrate `subgraph_row_to_event` parser

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest.py` (lines ~30-115, the `_REQUIRED_KEYS` constant + `subgraph_row_to_event` function)
- Modify: `tests/corpus/test_subgraph_ingest.py` (lines ~35-148, the 5 parser unit tests at module top)

- [ ] **Step 1: Confirm baseline tests pass**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q`
Expected: PASS (13 tests). If any fail before we start, stop and investigate.

- [ ] **Step 2: Update `_REQUIRED_KEYS` to the new schema's top-level field names**

In `src/pscanner/corpus/subgraph_ingest.py`, replace the `_REQUIRED_KEYS` tuple (around line 30):

```python
_REQUIRED_KEYS = (
    "id",  # consumed by _paginate_side cursor logic, not by the adapter
    "transactionHash",
    "timestamp",  # consumed by iter_market_trades, not by the adapter
    "orderHash",
    "maker",  # value is the nested {"id": "0x..."} object, see _parse_account_id below
    "taker",  # same
    "tokenId",
    "side",
    "makerAmountFilled",
    "takerAmountFilled",
    "fee",
)
```

- [ ] **Step 3: Add a nested-account-id parser helper**

In `src/pscanner/corpus/subgraph_ingest.py`, add this helper alongside the other `_parse_*` helpers (near line 46):

```python
def _parse_account_id(key: str, raw: object) -> str:
    """Extract ``.id`` from a nested ``Account`` object.

    The new subgraph schema returns ``maker`` and ``taker`` as nested
    objects ``{"id": "0x..."}``. The old schema returned them as bare
    Bytes strings. This helper unwraps and validates.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a nested object, got {type(raw).__name__}")
    inner = raw.get("id")
    if not isinstance(inner, str):
        raise ValueError(f"{key}.id must be str, got {type(inner).__name__}")
    return inner
```

- [ ] **Step 4: Rewrite `subgraph_row_to_event` to use the new schema**

Replace the function body (around line 65). Keep the same signature and the same `OrderFilledEvent` return type:

```python
def subgraph_row_to_event(row: Mapping[str, object]) -> OrderFilledEvent:
    """Adapt one GraphQL ``OrderFilledEvent`` row to the on-chain dataclass.

    The new subgraph schema collapses ``makerAssetId`` / ``takerAssetId``
    into ``tokenId`` (= ``Market.id``, the conditional token traded) +
    ``side`` (Int: 0=BUY, 1=SELL, indicating the maker's order direction).
    The amount fields ``makerAmountFilled`` / ``takerAmountFilled`` follow
    the same maker/taker convention as the old schema and flow through
    directly.

    Args:
        row: One element of the GraphQL ``orderFilledEvents`` list. Must
            carry every key in ``_REQUIRED_KEYS``; ``maker`` and ``taker``
            are nested ``Account`` objects with an ``id`` field.

    Returns:
        ``OrderFilledEvent`` with ``block_number=0`` and ``log_index=0``
        (subgraph payloads do not include these; downstream
        ``event_to_corpus_trade`` does not read those fields).

    Raises:
        KeyError: A required key is missing.
        ValueError: A numeric field is not parseable, a string field has
            the wrong type, or ``side`` is not 0 or 1.
    """
    for key in _REQUIRED_KEYS:
        if key not in row:
            raise KeyError(key)

    def as_int(key: str) -> int:
        return _parse_int_field(key, row[key])

    def as_str(key: str) -> str:
        return _parse_str_field(key, row[key])

    side = as_int("side")
    token_id = as_int("tokenId")
    if side == 0:
        # Maker placed a BUY order: gave USDC, took conditional tokens.
        maker_asset_id = 0
        taker_asset_id = token_id
    elif side == 1:
        # Maker placed a SELL order: gave conditional tokens, took USDC.
        maker_asset_id = token_id
        taker_asset_id = 0
    else:
        raise ValueError(f"unexpected side: {side}")

    return OrderFilledEvent(
        order_hash=as_str("orderHash"),
        maker=_parse_account_id("maker", row["maker"]),
        taker=_parse_account_id("taker", row["taker"]),
        maker_asset_id=maker_asset_id,
        taker_asset_id=taker_asset_id,
        making=as_int("makerAmountFilled"),
        taking=as_int("takerAmountFilled"),
        fee=as_int("fee"),
        tx_hash=as_str("transactionHash"),
        block_number=0,
        log_index=0,
    )
```

- [ ] **Step 5: Run parser tests, expect failures from old fixtures**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py::test_subgraph_row_to_event_parses_buy_side_row tests/corpus/test_subgraph_ingest.py::test_subgraph_row_to_event_parses_sell_side_row tests/corpus/test_subgraph_ingest.py::test_subgraph_row_to_event_rejects_missing_field tests/corpus/test_subgraph_ingest.py::test_subgraph_row_to_event_rejects_non_numeric_amount tests/corpus/test_subgraph_ingest.py::test_subgraph_row_to_event_accepts_int_values_for_bigints -q`
Expected: All 5 FAIL because their fixtures use the old schema's flat `makerAssetId` / `takerAssetId` keys, which we just removed from `_REQUIRED_KEYS`.

- [ ] **Step 6: Update `test_subgraph_row_to_event_parses_buy_side_row` fixture**

In `tests/corpus/test_subgraph_ingest.py`, replace the test body:

```python
def test_subgraph_row_to_event_parses_buy_side_row() -> None:
    """Maker BUY (side=0): maker gives USDC, taker gives CTF token."""
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": {"id": "0xMaker_Address_NOT_LowerCased"},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "222",
        "side": "0",
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
    # event_to_corpus_trade lowercases the maker downstream; the dataclass
    # preserves whatever's passed in here.
    assert event.maker == "0xMaker_Address_NOT_LowerCased"
```

- [ ] **Step 7: Update `test_subgraph_row_to_event_parses_sell_side_row` fixture**

In `tests/corpus/test_subgraph_ingest.py`, replace that test body:

```python
def test_subgraph_row_to_event_parses_sell_side_row() -> None:
    """Maker SELL (side=1): maker gives CTF token, taker gives USDC."""
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "111",
        "side": "1",
        "makerAmountFilled": "40000000",
        "takerAmountFilled": "20000000",
        "fee": "0",
    }
    event = subgraph_row_to_event(row)
    assert event.maker_asset_id == 111
    assert event.taker_asset_id == 0
    assert event.making == 40_000_000
    assert event.taking == 20_000_000
```

- [ ] **Step 8: Update `test_subgraph_row_to_event_rejects_missing_field` fixture**

In `tests/corpus/test_subgraph_ingest.py`, replace that test body:

```python
def test_subgraph_row_to_event_rejects_missing_field() -> None:
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        # orderHash deliberately missing
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "222",
        "side": "0",
        "makerAmountFilled": "1",
        "takerAmountFilled": "1",
        "fee": "0",
    }
    with pytest.raises(KeyError, match="orderHash"):
        subgraph_row_to_event(row)
```

- [ ] **Step 9: Update `test_subgraph_row_to_event_rejects_non_numeric_amount` fixture**

In `tests/corpus/test_subgraph_ingest.py`, replace that test body:

```python
def test_subgraph_row_to_event_rejects_non_numeric_amount() -> None:
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": "1700001234",
        "orderHash": "0x" + "ab" * 32,
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "222",
        "side": "0",
        "makerAmountFilled": "not-a-number",
        "takerAmountFilled": "1",
        "fee": "0",
    }
    with pytest.raises(ValueError, match="makerAmountFilled"):
        subgraph_row_to_event(row)
```

- [ ] **Step 10: Update `test_subgraph_row_to_event_accepts_int_values_for_bigints` fixture**

In `tests/corpus/test_subgraph_ingest.py`, replace that test body:

```python
def test_subgraph_row_to_event_accepts_int_values_for_bigints() -> None:
    """BigInt fields may be returned as native ints rather than strings."""
    row = {
        "id": "0xtx_0xorder",
        "transactionHash": "0xee" * 32,
        "timestamp": 1700001234,
        "orderHash": "0x" + "ab" * 32,
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": 222,
        "side": 0,
        "makerAmountFilled": 20_000_000,
        "takerAmountFilled": 40_000_000,
        "fee": 0,
    }
    event = subgraph_row_to_event(row)
    assert event.making == 20_000_000
    assert event.taking == 40_000_000
```

- [ ] **Step 11: Add the two new side-mapping pin-down tests**

In `tests/corpus/test_subgraph_ingest.py`, add these two tests after the existing parser tests (after the test at line ~148):

```python
def test_subgraph_row_to_event_buy_side_maps_assets_correctly() -> None:
    """Pin down: side=0 ⇒ maker_asset_id=0, taker_asset_id=tokenId."""
    row = {
        "id": "0xtx",
        "transactionHash": "0x" + "aa" * 32,
        "timestamp": "1",
        "orderHash": "0x" + "bb" * 32,
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "999",
        "side": "0",
        "makerAmountFilled": "100",
        "takerAmountFilled": "200",
        "fee": "0",
    }
    event = subgraph_row_to_event(row)
    assert event.maker_asset_id == 0
    assert event.taker_asset_id == 999
    assert event.making == 100
    assert event.taking == 200


def test_subgraph_row_to_event_sell_side_maps_assets_correctly() -> None:
    """Pin down: side=1 ⇒ maker_asset_id=tokenId, taker_asset_id=0."""
    row = {
        "id": "0xtx",
        "transactionHash": "0x" + "aa" * 32,
        "timestamp": "1",
        "orderHash": "0x" + "bb" * 32,
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "999",
        "side": "1",
        "makerAmountFilled": "100",
        "takerAmountFilled": "200",
        "fee": "0",
    }
    event = subgraph_row_to_event(row)
    assert event.maker_asset_id == 999
    assert event.taker_asset_id == 0
    assert event.making == 100
    assert event.taking == 200


def test_subgraph_row_to_event_rejects_invalid_side() -> None:
    row = {
        "id": "0xtx",
        "transactionHash": "0x" + "aa" * 32,
        "timestamp": "1",
        "orderHash": "0x" + "bb" * 32,
        "maker": {"id": "0x" + "11" * 20},
        "taker": {"id": "0x" + "22" * 20},
        "tokenId": "999",
        "side": "2",  # invalid
        "makerAmountFilled": "100",
        "takerAmountFilled": "200",
        "fee": "0",
    }
    with pytest.raises(ValueError, match="unexpected side: 2"):
        subgraph_row_to_event(row)
```

- [ ] **Step 12: Run all parser tests, expect PASS**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q -k "subgraph_row_to_event"`
Expected: 8 passed (5 updated existing + 3 new).

- [ ] **Step 13: Lint + type check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py && uv run ty check src/pscanner/corpus/subgraph_ingest.py`
Expected: clean.

- [ ] **Step 14: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py
git commit -m "$(cat <<'EOF'
refactor(corpus): rewrite subgraph_row_to_event for new schema (#151)

The new Polymarket Orderbook subgraph collapses makerAssetId/takerAssetId
into tokenId + side; maker/taker are nested Account entities. Derive the
two asset-id fields from side+tokenId; pass makerAmountFilled and
takerAmountFilled through unchanged (same convention).

OrderFilledEvent dataclass stays unchanged so downstream event_to_corpus_trade
is invariant — corpus_trades rows remain byte-identical to the old path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Collapse the two side-queries into a single query

**Files:**
- Modify: `src/pscanner/corpus/subgraph_ingest.py` (`_TRADES_QUERY_MAKER_SIDE` + `_TRADES_QUERY_TAKER_SIDE` constants around lines 119-145; `iter_market_trades` around lines 184-232)
- Modify: `tests/corpus/test_subgraph_ingest.py` (4 paginator tests around lines 150-247)

- [ ] **Step 1: Replace the two query constants with a single new-schema query**

In `src/pscanner/corpus/subgraph_ingest.py`, replace both `_TRADES_QUERY_MAKER_SIDE` and `_TRADES_QUERY_TAKER_SIDE` constants with one new constant. Delete the old two; add this above `_paginate_side`:

```python
# Single query — the new subgraph's market_in filter catches every fill
# involving any of the listed tokens (maker side or taker side), so the
# old maker/taker two-query split is no longer needed.
_TRADES_QUERY = """
query($assets: [String!]!, $cursor: String!, $first: Int!) {
  orderFilledEvents(
    where: { market_in: $assets, id_gt: $cursor }
    first: $first
    orderBy: id
    orderDirection: asc
  ) {
    id
    orderHash
    transactionHash
    timestamp
    maker { id }
    taker { id }
    market { id }
    tokenId
    side
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
"""
```

- [ ] **Step 2: Simplify `iter_market_trades` to call `_paginate_side` once**

In `src/pscanner/corpus/subgraph_ingest.py`, replace the `iter_market_trades` function body (around line 184). Keep its signature and docstring intent the same:

```python
async def iter_market_trades(
    *,
    client: SubgraphClient,
    asset_ids: Sequence[str],
    page_size: int = _MAX_PAGE_SIZE,
) -> AsyncIterator[tuple[OrderFilledEvent, int]]:
    """Yield every ``OrderFilledEvent`` involving any asset in ``asset_ids``.

    Uses the new subgraph's ``market_in`` filter so a single paginated
    query catches every fill on the listed tokens (no maker/taker split
    needed). Cursor-paginated via ``id_gt`` so restarts are safe (no
    duplicates on resume, only forward progress).

    Args:
        client: Open ``SubgraphClient``.
        asset_ids: CTF token ids (as decimal strings) belonging to one condition.
            Pass both YES and NO token ids for a binary market.
        page_size: Rows per query, capped at ``_MAX_PAGE_SIZE`` (1000) by
            The Graph. Reduce for lower memory pressure during tests.

    Yields:
        ``(event, ts)`` tuples.

    Raises:
        ValueError: ``page_size`` is out of the ``1.._MAX_PAGE_SIZE`` range.
    """
    if page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{_MAX_PAGE_SIZE}, got {page_size}")
    if not asset_ids:
        return

    async for ev, ts in _paginate_side(
        client=client,
        graphql=_TRADES_QUERY,
        asset_ids=asset_ids,
        page_size=page_size,
    ):
        yield ev, ts
```

- [ ] **Step 3: Run paginator tests, expect failures**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q -k "iter_market_trades"`
Expected: At least 2 of the 4 paginator tests FAIL because their fixtures key on the now-deleted `_TRADES_QUERY_MAKER_SIDE` / `_TRADES_QUERY_TAKER_SIDE` constants, and because the `test_iter_market_trades_paginates_both_sides` test asserts behavior that no longer applies (the two-query pattern).

- [ ] **Step 4: Update `test_iter_market_trades_paginates_both_sides` — rename and simplify**

In `tests/corpus/test_subgraph_ingest.py`, replace that test with a single-query version. Rename to reflect the new behavior:

```python
async def test_iter_market_trades_paginates_single_query() -> None:
    """Paginator runs a single market_in query, paginating by id_gt."""
    # Two-page response: page 1 is a full page (triggers a follow-up query),
    # page 2 is short (terminates pagination).
    page1 = [
        _row(side="0", token_id=111, maker_amt=10_000_000, taker_amt=20_000_000, row_id="0xa"),
        _row(side="0", token_id=111, maker_amt=10_000_000, taker_amt=20_000_000, row_id="0xb"),
    ]
    page2 = [
        _row(side="1", token_id=111, maker_amt=20_000_000, taker_amt=10_000_000, row_id="0xc"),
    ]
    call_count = {"n": 0}

    async def fake_query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"orderFilledEvents": page1}
        return {"orderFilledEvents": page2}

    client = AsyncMock()
    client.query.side_effect = fake_query

    yielded: list[tuple[OrderFilledEvent, int]] = []
    async for event, ts in iter_market_trades(
        client=client,
        asset_ids=["111"],
        page_size=2,
    ):
        yielded.append((event, ts))

    assert call_count["n"] == 2  # one full page + one terminator
    assert len(yielded) == 3
    # First two are BUY (side=0), third is SELL (side=1)
    assert yielded[0][0].maker_asset_id == 0
    assert yielded[0][0].taker_asset_id == 111
    assert yielded[2][0].maker_asset_id == 111
    assert yielded[2][0].taker_asset_id == 0
```

The `_row(...)` helper at the top of the test file currently builds old-schema rows. Update it in the next step.

- [ ] **Step 5: Update the `_row` test-fixture helper to emit new-schema rows**

Find the existing `_row` helper at the top of `tests/corpus/test_subgraph_ingest.py` (search for `def _row(`). Replace it:

```python
def _row(
    *,
    side: str = "0",
    token_id: int = 111,
    maker_amt: int = 10_000_000,
    taker_amt: int = 20_000_000,
    row_id: str = "0xrow_default",
    tx_hash: str | None = None,
    maker_addr: str = "0x" + "11" * 20,
    taker_addr: str = "0x" + "22" * 20,
    fee: str = "0",
) -> dict[str, Any]:
    """Build a new-schema OrderFilledEvent row for tests."""
    return {
        "id": row_id,
        "transactionHash": tx_hash or ("0x" + "ab" * 32),
        "timestamp": "1700000000",
        "orderHash": "0x" + "ee" * 32,
        "maker": {"id": maker_addr},
        "taker": {"id": taker_addr},
        "tokenId": str(token_id),
        "side": side,
        "makerAmountFilled": str(maker_amt),
        "takerAmountFilled": str(taker_amt),
        "fee": fee,
    }
```

Inspect the existing call sites (`_row(...)`) and check they all still pass the right kwargs. If any existing call site used keyword args like `maker_asset_id=` or `taker_asset_id=`, replace them: `maker_asset_id=0, taker_asset_id=111` becomes `side="0", token_id=111`; `maker_asset_id=111, taker_asset_id=0` becomes `side="1", token_id=111`.

- [ ] **Step 6: Update `test_iter_market_trades_empty_asset_ids_skips_query`**

This test currently calls `iter_market_trades(client=client, asset_ids=[], page_size=10)`. It should still pass unchanged — the early-return on empty asset_ids logic stays the same. Verify by re-running:

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py::test_iter_market_trades_empty_asset_ids_skips_query -q`
Expected: PASS without any changes.

- [ ] **Step 7: Update `test_iter_market_trades_short_first_page_exits_without_second_query`**

Find this test in `tests/corpus/test_subgraph_ingest.py`. Its old logic tracked maker-side vs taker-side calls. Replace its body with a simpler single-query version:

```python
async def test_iter_market_trades_short_first_page_exits_without_query() -> None:
    """When the first page is shorter than page_size, no further query runs."""
    call_count = {"n": 0}

    async def fake_query(graphql: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        call_count["n"] += 1
        # Single short page (only 1 row when page_size is 2) → terminate.
        return {"orderFilledEvents": [
            _row(side="0", token_id=111, maker_amt=10_000_000, taker_amt=20_000_000, row_id="0xa"),
        ]}

    client = AsyncMock()
    client.query.side_effect = fake_query

    yielded: list[tuple[OrderFilledEvent, int]] = []
    async for ev, ts in iter_market_trades(client=client, asset_ids=["111"], page_size=2):
        yielded.append((ev, ts))

    assert call_count["n"] == 1
    assert len(yielded) == 1
```

(Note the test name shouldn't say "without_second_query" anymore — `exits_without_query` better reflects that there's no two-query split now. Rename the test function accordingly.)

- [ ] **Step 8: Update `test_iter_market_trades_rejects_invalid_page_size`**

Find this test (parametrized over invalid page sizes). It tests `ValueError` from the `page_size` validator. The validator code didn't change, so this test should still pass. Verify:

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q -k "rejects_invalid_page_size"`
Expected: PASS without any changes.

- [ ] **Step 9: Run all paginator tests, expect PASS**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q -k "iter_market_trades"`
Expected: 4 passed.

- [ ] **Step 10: Lint + type check**

Run: `uv run ruff check src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py && uv run ty check src/pscanner/corpus/subgraph_ingest.py`
Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add src/pscanner/corpus/subgraph_ingest.py tests/corpus/test_subgraph_ingest.py
git commit -m "$(cat <<'EOF'
refactor(corpus): collapse maker/taker subgraph queries into one (#151)

The new schema's market_in filter catches every fill involving any
listed token (maker side or taker side) in a single paginated query.
Drops _TRADES_QUERY_MAKER_SIDE + _TRADES_QUERY_TAKER_SIDE for one
_TRADES_QUERY constant; simplifies iter_market_trades to a single
_paginate_side call. iter_market_trades public signature unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update the integration-test fixtures

**Files:**
- Modify: `tests/corpus/test_subgraph_ingest.py` (4 `run_subgraph_backfill` integration tests starting around line 349)

- [ ] **Step 1: Run the integration tests to see what's failing**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q -k "run_subgraph_backfill"`
Expected: All 4 FAIL because they use the old `_row()` helper signature with `maker_asset_id=` / `taker_asset_id=` kwargs that no longer exist.

- [ ] **Step 2: Update `test_run_subgraph_backfill_processes_pending_market`**

Find this test in `tests/corpus/test_subgraph_ingest.py`. Find every `_row(...)` call within the test body and update the keyword args:

- `_row(maker_asset_id=0, taker_asset_id=111, ...)` becomes `_row(side="0", token_id=111, ...)`
- `_row(maker_asset_id=111, taker_asset_id=0, ...)` becomes `_row(side="1", token_id=111, ...)`

Keep all `maker_amt=` / `taker_amt=` / `tx_hash=` / `row_id=` kwargs unchanged. The downstream assertions on the resulting `corpus_trades` rows should still pass identically because `event_to_corpus_trade` produces the same output from equivalent input.

- [ ] **Step 3: Update `test_run_subgraph_backfill_skips_already_processed_markets`**

Same migration as Step 2 — find `_row(...)` calls, update `maker_asset_id` / `taker_asset_id` kwargs to `side` / `token_id`.

- [ ] **Step 4: Update `test_run_subgraph_backfill_respects_limit`**

Same migration.

- [ ] **Step 5: Update `test_run_subgraph_backfill_records_market_failure_and_continues`**

Same migration.

- [ ] **Step 6: Run all integration tests, expect PASS**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q -k "run_subgraph_backfill"`
Expected: 4 passed.

- [ ] **Step 7: Run the full subgraph_ingest test file**

Run: `uv run pytest tests/corpus/test_subgraph_ingest.py -q`
Expected: All passing — 13 originals + 3 added in Task 1 = 16 tests.

- [ ] **Step 8: Run the broader test suite to confirm no surrounding regressions**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 9: Lint + type check the test file**

Run: `uv run ruff check tests/corpus/test_subgraph_ingest.py && uv run ty check tests/corpus/test_subgraph_ingest.py`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add tests/corpus/test_subgraph_ingest.py
git commit -m "$(cat <<'EOF'
test(corpus): migrate run_subgraph_backfill integration tests (#151)

Update the 4 run_subgraph_backfill integration tests' _row() fixtures
to the new schema's side+token_id kwargs. Downstream assertions on
corpus_trades unchanged — event_to_corpus_trade produces byte-identical
rows from equivalent input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Flip the CLI default + update CLAUDE.md

**Files:**
- Modify: `src/pscanner/corpus/cli.py` (around line 264 — the `subgraph-backfill` parser's `--subgraph-id` flag default)
- Modify: `CLAUDE.md` (two places where the old/new subgraph IDs are discussed)

- [ ] **Step 1: Find the existing `--subgraph-id` default in the CLI**

Run: `grep -n "7fu2DWYK\|--subgraph-id" src/pscanner/corpus/cli.py`
Expected: locates the default argument value (a single line declaring `default="7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY"` or similar) inside the `subgraph-backfill` subparser definition.

- [ ] **Step 2: Replace the default value**

In `src/pscanner/corpus/cli.py`, change the line that sets the default to:

```python
default="B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR",
```

(The argparse `add_argument` call shape and help text stay the same. If there's a module-level constant carrying the old ID, replace it in place — searching for `7fu2DWYK` finds every reference.)

- [ ] **Step 3: Verify CLI parsing still works**

Run: `uv run pscanner corpus subgraph-backfill --help 2>&1 | grep subgraph-id`
Expected: line shows the new default ID.

- [ ] **Step 4: Update CLAUDE.md — main Polymarket section bullet**

In `/home/macph/projects/polymarketScanner/CLAUDE.md`, find the bullet currently starting `- **Phase 3 (subgraph) supersedes the eth_getLogs corpus path for backfill.**` (around line 25). Replace its body to reflect that the new subgraph is now the default:

```markdown
- **Phase 3 (subgraph) supersedes the eth_getLogs corpus path for backfill.** `pscanner corpus subgraph-backfill` queries `https://gateway.thegraph.com/api/{KEY}/subgraphs/id/B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR` (current Polymarket Orderbook subgraph; PR #151 swapped this in after the prior `7fu2DWYK...` indexer stopped emitting `OrderFilledEvent` rows on 2026-04-28). Both the `Exchange` and `NegRiskExchange` contracts write into the same `OrderFilledEvent` entity, so neg-risk markets are auto-covered. The `eth_getLogs` paths (`onchain-backfill`, `onchain-backfill-targeted`) stay during the transition but will be deleted in a follow-up commit. The decoder (`pscanner.poly.onchain.decode_order_filled`) and `AssetIndexRepo` survive the deletion — they're still useful for low-level event inspection. **Schema notes:** the new schema collapses `makerAssetId` / `takerAssetId` into `tokenId` (= `Market.id`) + `side` (Int 0=BUY / 1=SELL); the parser derives the asset-id pair from `tokenId` + `side`. `makerAmountFilled` / `takerAmountFilled` flow through unchanged. Filter via `market_in: [<asset_ids>]` (one query covers maker + taker sides). Subgraph stores `timestamp` directly. The old subgraph's earliest event was `1744013119` (2025-04-07) — pre-cutoff markets still need the eth_getLogs path. Live validation: the 2026-05-05 run inserted 15.87M trade rows across 2,528 markets in 10h 30m on the old subgraph.
```

- [ ] **Step 5: Update CLAUDE.md — CLI reference for `subgraph-backfill`**

In `/home/macph/projects/polymarketScanner/CLAUDE.md`, find the bullet currently starting `- \`pscanner corpus subgraph-backfill [--api-key KEY] [--subgraph-id ID]` (around line 122). Replace its body:

```markdown
- `pscanner corpus subgraph-backfill [--api-key KEY] [--subgraph-id ID] [--rpm N] [--page-size N] [--limit N]` — preferred replacement for `onchain-backfill-targeted`. Queries Polymarket's Orderbook subgraph on The Graph (default `B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR`), filtered server-side by `market_in` against the local `asset_index`. Resumable via `corpus_markets.onchain_processed_at`. Free-tier API key (~100K queries/month) covers full corpus runs comfortably. Requires `$GRAPH_API_KEY` (or `--api-key`). Pass `--subgraph-id 7fu2DWYK93ePfzB24c2wrP94S3x4LGHUrQxphhoEypyY` to query the deprecated pre-2026-04-28 subgraph — note the schema differs, so the parser will fail. Use the eth_getLogs path for pre-cutoff backfill instead.
```

- [ ] **Step 6: Confirm no other stale references remain in CLAUDE.md**

Run: `grep -n "7fu2DWYK\|#151" /home/macph/projects/polymarketScanner/CLAUDE.md`
Expected: no `#151` forward references should remain (it's now landed). Any residual `7fu2DWYK...` references should be in deprecation context only.

- [ ] **Step 7: Lint check**

Run: `uv run ruff check src/pscanner/corpus/cli.py`
Expected: clean.

- [ ] **Step 8: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/pscanner/corpus/cli.py CLAUDE.md
git commit -m "$(cat <<'EOF'
feat(corpus): default --subgraph-id to the current subgraph (#151)

Flip the corpus subgraph-backfill CLI's default subgraph ID from the
stale 7fu2DWYK... (indexing froze 2026-04-28) to the current B9mm21D...
that this PR's parser targets. Operators who pass --subgraph-id
explicitly are unaffected. CLAUDE.md updated with the migrated schema
notes; resolves the prior "#151 will obsolete these" forward references.

Closes #151.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Manual validation gate (one-market parity diff)

**Files:**
- No code changes. This is a one-time validation step before merging.

- [ ] **Step 1: Pick a market fully backfilled before 2026-04-28**

Run this query against the corpus DB to find a candidate market:

```bash
uv run python -c "
import sqlite3
c = sqlite3.connect('file:data/corpus.sqlite3?mode=ro', uri=True)
c.row_factory = sqlite3.Row
rows = c.execute(\"\"\"
  SELECT cm.condition_id, cm.onchain_processed_at, COUNT(ct.tx_hash) AS n_trades
  FROM corpus_markets cm
  JOIN corpus_trades ct USING (condition_id, platform)
  WHERE cm.onchain_processed_at IS NOT NULL
    AND cm.onchain_processed_at < 1714262400  -- 2026-04-28 unix timestamp
  GROUP BY cm.condition_id
  HAVING n_trades BETWEEN 50 AND 500
  ORDER BY cm.onchain_processed_at DESC
  LIMIT 5
\"\"\").fetchall()
for r in rows:
    print(dict(r))
"
```

Expected: at least one candidate with 50-500 trades (small enough to diff fast, large enough to be representative). Note the chosen `condition_id`.

- [ ] **Step 2: Snapshot the current `corpus_trades` for that condition**

```bash
CONDITION_ID="<the chosen condition_id from step 1>"
uv run python -c "
import sqlite3, sys
cid = sys.argv[1]
c = sqlite3.connect('data/corpus.sqlite3')
c.row_factory = sqlite3.Row
rows = c.execute(
    'SELECT platform, tx_hash, asset_id, wallet_address, condition_id, outcome_side, bs, price, size, notional_usd, ts '
    'FROM corpus_trades WHERE condition_id = ? ORDER BY tx_hash, asset_id, wallet_address',
    (cid,)
).fetchall()
import csv
with open('/tmp/parity_old.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(rows[0].keys() if rows else [])
    for r in rows:
        w.writerow(list(r))
print('wrote', len(rows), 'rows to /tmp/parity_old.csv')
" "$CONDITION_ID"
```

Expected: writes N rows to `/tmp/parity_old.csv`.

- [ ] **Step 3: Reset that market's `onchain_processed_at` so the backfill re-runs**

```bash
uv run python -c "
import sqlite3, sys
cid = sys.argv[1]
c = sqlite3.connect('data/corpus.sqlite3')
c.execute('UPDATE corpus_markets SET onchain_processed_at = NULL WHERE condition_id = ?', (cid,))
c.commit()
print('reset', cid)
" "$CONDITION_ID"
```

- [ ] **Step 4: Delete the existing `corpus_trades` for that condition (so the re-backfill writes fresh)**

```bash
uv run python -c "
import sqlite3, sys
cid = sys.argv[1]
c = sqlite3.connect('data/corpus.sqlite3')
n = c.execute('DELETE FROM corpus_trades WHERE condition_id = ?', (cid,)).rowcount
c.commit()
print('deleted', n, 'rows for', cid)
" "$CONDITION_ID"
```

- [ ] **Step 5: Re-run the subgraph backfill against just that market**

```bash
set -a; source .env; set +a
uv run pscanner corpus subgraph-backfill --limit 1
```

Expected: completes successfully, ingests the trades for that condition. Output should report something like `markets processed: 1, trades inserted: N` where N matches or exceeds the original count in `/tmp/parity_old.csv` (may exceed by trades that landed after 2026-04-28 but before today).

- [ ] **Step 6: Snapshot the new `corpus_trades` for that condition**

Filter the new snapshot to only trades whose `ts` is ≤ the maximum ts in the old snapshot (so we're comparing apples to apples; trades from the gap window won't have an old-side counterpart):

```bash
uv run python -c "
import sqlite3, sys, csv
cid = sys.argv[1]
# Find max ts in the old snapshot
with open('/tmp/parity_old.csv') as fh:
    reader = csv.DictReader(fh)
    rows_old = list(reader)
max_ts_old = max(int(r['ts']) for r in rows_old) if rows_old else 0
print('max ts in old snapshot:', max_ts_old)

c = sqlite3.connect('data/corpus.sqlite3')
c.row_factory = sqlite3.Row
rows = c.execute(
    'SELECT platform, tx_hash, asset_id, wallet_address, condition_id, outcome_side, bs, price, size, notional_usd, ts '
    'FROM corpus_trades WHERE condition_id = ? AND ts <= ? '
    'ORDER BY tx_hash, asset_id, wallet_address',
    (cid, max_ts_old)
).fetchall()
with open('/tmp/parity_new.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(rows[0].keys() if rows else [])
    for r in rows:
        w.writerow(list(r))
print('wrote', len(rows), 'rows to /tmp/parity_new.csv')
" "$CONDITION_ID"
```

- [ ] **Step 7: Diff the two snapshots**

```bash
diff /tmp/parity_old.csv /tmp/parity_new.csv
echo "exit=$?"
```

Expected: `exit=0` (no diff). The two files should be byte-identical.

If `exit=1`, the diff output shows which rows differ. The most likely failure modes:
- **Sign/direction error on `bs`** — the side mapping in Task 1 Step 4 is wrong; check whether `side=0` really maps to maker BUY.
- **Off-by-decimal-conversion in `price` or `notional_usd`** — should never happen if `makerAmountFilled`/`takerAmountFilled` are used directly (which they are per the design).
- **Mismatched `wallet_address` casing** — should not happen because `event_to_corpus_trade` lowercases.

- [ ] **Step 8: If diff is clean, kick off the larger backfill**

If `diff` returned 0, the migration is validated. Run the larger backfill (skip on the laptop if memory-tight; do this on the desktop):

```bash
set -a; source .env; set +a
nohup uv run pscanner corpus subgraph-backfill > /tmp/subgraph_backfill_run.log 2>&1 &
disown
```

Watch progress: `tail -f /tmp/subgraph_backfill_run.log`. CLAUDE.md's prior measurement was ~10h 30m for 15.87M rows on the old subgraph — expect similar or faster.

- [ ] **Step 9: After backfill, confirm the gap is filled**

Once the larger backfill completes, the corpus should now contain trades from 2026-04-28 onward. Sanity-check:

```bash
uv run python -c "
import sqlite3, datetime as dt
c = sqlite3.connect('file:data/corpus.sqlite3?mode=ro', uri=True)
max_ts = c.execute('SELECT MAX(ts) FROM corpus_trades').fetchone()[0]
print('max ts in corpus_trades:', max_ts, '=', dt.datetime.fromtimestamp(max_ts).isoformat())
print('trades since 2026-04-28:', c.execute(\"SELECT COUNT(*) FROM corpus_trades WHERE ts >= 1714262400\").fetchone()[0])
"
```

Expected: `max ts` near the current date; `trades since 2026-04-28` is non-zero.

- [ ] **Step 10: Mark #151 done**

No commit needed for this validation step. Update issue #151 with a comment documenting the parity diff result + the larger backfill outcome.

---

## Self-review

**Spec coverage:**

- ✅ Architecture (Architecture section) — Tasks 1+2+4 cover the file changes
- ✅ Schema mapping & parity invariant — Task 1 implements the side-mapping reconstruction; Tasks 1 Step 11 + Task 5 pin it down with unit tests and the validation gate
- ✅ Query strategy (single query, market_in filter) — Task 2
- ✅ Pagination (id_gt cursor on `id`) — preserved in the query in Task 2 Step 1; `_paginate_side` unchanged
- ✅ Error handling (KeyError on missing field, ValueError on bad side/amount) — Task 1 Step 4
- ✅ Testing (5 existing parser tests updated, 3 new tests, 4 paginator tests updated, 4 integration tests updated) — Tasks 1, 2, 3
- ✅ Validation gate — Task 5
- ✅ CLI / docs touch-ups — Task 4

**Placeholders:** No TBD/TODO/etc. Every step contains either the actual code to paste, the exact command to run, or both.

**Type consistency:** `subgraph_row_to_event` signature unchanged. `_TRADES_QUERY` name introduced in Task 2 Step 1 and referenced in Task 2 Step 2 — same name. `_row(...)` helper signature defined in Task 2 Step 5 and referenced in Task 3 Steps 2-5 — same kwargs.

**Out-of-scope (correctly deferred):**

- Deletion of eth_getLogs paths — separate cleanup, CLAUDE.md flags
- `SubgraphTradeCollector` daemon collector — #152, depends on this shipping
- Operator-facing migration helper for the old subgraph ID — none planned; operators get a parser failure with a clear error message
