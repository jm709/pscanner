# Insider-Episode Catalogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/insider_episodes.py`, which reads the committed case registry, ranks pre-news winning-side buyers on each case market with the frozen scorer, compares them against matched control markets, blind-tests the published wallets on holdout cases, and writes per-case and summary reports.

**Architecture:** Two files. `scripts/insider_episode_lib.py` holds every pure function and dataclass (scoring, ranking, comparator, auto-anchor, control matching, report rendering) and imports nothing from the corpus or the network. `scripts/insider_episodes.py` holds the I/O: registry loading, read-only SQLite queries against the corpus, the provenance cache, the data-api adapter, and the CLI. The corpus is queried with the stdlib `sqlite3` module in read-only mode, not DuckDB: every query here is an indexed point or range lookup (`idx_corpus_trades_market_ts`, `idx_corpus_trades_wallet_ts`) and the exploratory probe ran the same shapes in seconds. The spec's "DuckDB-backed" line is superseded by this plan for that reason.

**Tech Stack:** Python 3.13, stdlib `sqlite3`, `pyyaml==6.0.3` (new direct dependency; already in `uv.lock` transitively), existing `pscanner.poly.data.DataClient`, pytest with the `corpus_factory` fixture in `tests/scripts/conftest.py`.

**Spec:** `docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md` (including Amendment 1, the frozen scorer).

## Global Constraints

- Python 3.13, `uv`. Verify with `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q` before every commit; zero new diagnostics.
- Line length 100, Google docstrings on public functions, cyclomatic complexity ≤ 8, ≤ 5 positional params, absolute imports only.
- Scripts bootstrap `sys.path` with the repo root exactly as `scripts/find_insider_wallets.py` does, so `from scripts.insider_episode_lib import ...` works both under pytest and as `python scripts/insider_episodes.py`.
- The scorer constants are fixed by Amendment 1: freshness plateau 10 days, decay 60 days, floor 0.1, concentration blend 0.5, flip bonus 0.5. Do not add flags for them.
- Nothing after the anchor may reach the scorer: every lifetime query filters `ts <= anchor`.
- The scorer and ranker never receive the sealed record. The comparator is the only consumer.
- Holdout cases are refused unless `--scorer-frozen` is passed.
- Corpus test fixtures use `notional_usd >= 10`.
- Test files follow `tests/scripts/test_*.py`; `filterwarnings = error`, so close every sqlite connection.
- The provenance cache lives in `data/` and is never committed. Reports go to `docs/research/insider-episodes/`.

## Review Focus

1. A registry wallet address with mixed case or surrounding whitespace must still match the lowercase corpus address; the loader lowercases and strips, and the comparator compares lowercase. Test in Task 1 and Task 6.
2. A wallet whose only window activity is a SELL of the winning side (closing a position) must not become a candidate; candidates require a BUY of the winning outcome token. Test in Task 2.
3. A wallet with zero lifetime notional before the window cannot exist (its window buy counts), but a wallet whose lifetime notional equals its window notional must give concentration exactly 1.0, not a division error or > 1. Test in Task 6.
4. A case whose primary market resolved NO (the January 2026 negative example) must rank NO buyers and report the published YES-buying wallets under the `losing_side_only` reason, not crash or score them. Test in Task 8.
5. A control search that finds zero matches must still produce a case report that says "no controls", with the fingerprint section marked unavailable. Test in Task 9.

---

### Task 1: Dependency, registry loader, split guard

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Create: `scripts/insider_episode_lib.py`
- Create: `scripts/insider_episodes.py`
- Test: `tests/scripts/test_insider_episode_lib.py`
- Test: `tests/scripts/test_insider_episodes.py`

**Interfaces:**
- Produces:
  - `insider_episode_lib.MarketRef(slug: str, winning_side: str, role: str, condition_id: str | None = None)`
  - `insider_episode_lib.PublicCase(id, name, split, confidence, analyzable, news_ts, news_precision, markets: tuple[MarketRef, ...])` with property `primary -> MarketRef | None` (first market whose `role` starts with `"primary"`)
  - `insider_episode_lib.SealedCase(id: str, wallets: tuple[str, ...])` (lowercased 42-char addresses only)
  - `insider_episode_lib.select_cases(cases: list[PublicCase], split: str, *, scorer_frozen: bool) -> list[PublicCase]` raising `HoldoutLockedError` when holdout is requested without the flag
  - `insider_episodes.load_registry(path: Path) -> tuple[list[PublicCase], dict[str, SealedCase]]`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"pyyaml==6.0.3",` after `"psutil==5.9.8",` in `dependencies`. Run `uv lock` and confirm `uv.lock` changes only by promoting pyyaml to a direct dependency.

- [ ] **Step 2: Write the failing tests for the pure types and the guard**

`tests/scripts/test_insider_episode_lib.py`:

```python
"""Tests for the pure insider-episode library."""

from __future__ import annotations

import pytest

from scripts.insider_episode_lib import (
    HoldoutLockedError,
    MarketRef,
    PublicCase,
    select_cases,
)


def _case(cid: str, split: str, *, markets: tuple[MarketRef, ...] = ()) -> PublicCase:
    return PublicCase(
        id=cid,
        name=cid,
        split=split,
        confidence="alleged",
        analyzable=True,
        news_ts=1_700_000_000,
        news_precision="minute",
        markets=markets,
    )


def test_primary_market_is_first_primary_role() -> None:
    case = _case(
        "c",
        "dev",
        markets=(
            MarketRef(slug="s-sibling", winning_side="YES", role="sibling"),
            MarketRef(slug="s-primary", winning_side="YES", role="primary (main)"),
            MarketRef(slug="s-primary-2", winning_side="YES", role="primary"),
        ),
    )
    assert case.primary is not None
    assert case.primary.slug == "s-primary"


def test_primary_market_none_when_no_primary_role() -> None:
    case = _case("c", "dev", markets=(MarketRef(slug="s", winning_side="YES", role="sibling"),))
    assert case.primary is None


def test_select_cases_dev_only_by_default() -> None:
    cases = [_case("a", "dev"), _case("b", "holdout")]
    assert [c.id for c in select_cases(cases, "dev", scorer_frozen=False)] == ["a"]


def test_select_cases_holdout_refused_without_frozen_flag() -> None:
    cases = [_case("a", "dev"), _case("b", "holdout")]
    with pytest.raises(HoldoutLockedError):
        select_cases(cases, "holdout", scorer_frozen=False)
    with pytest.raises(HoldoutLockedError):
        select_cases(cases, "all", scorer_frozen=False)


def test_select_cases_holdout_allowed_with_frozen_flag() -> None:
    cases = [_case("a", "dev"), _case("b", "holdout")]
    assert [c.id for c in select_cases(cases, "holdout", scorer_frozen=True)] == ["b"]
    assert [c.id for c in select_cases(cases, "all", scorer_frozen=True)] == ["a", "b"]
```

`tests/scripts/test_insider_episodes.py`:

```python
"""Tests for the insider-episode I/O script."""

from __future__ import annotations

from pathlib import Path

from scripts.insider_episodes import load_registry

_REGISTRY = """
registry_version: 1
cases:
  - id: case-a
    name: Case A
    confidence: confirmed
    split: dev
    analyzable: true
    news_ts_utc: "2026-01-03T06:00:00Z"
    news_ts_precision: hour
    corpus_markets:
      - market_slug: slug-a
        winning_side: "YES"
        role: primary
      - market_slug: slug-a2
        winning_side: "YES"
        role: sibling
    wallets:
      - address: "  0xABCDEFabcdef0123456789abcdef0123456789AB "
      - address: "0xa72DB1749e9... (TRUNCATED)"
      - address: null
        username: someone
  - id: case-b
    name: Case B
    confidence: suspected
    split: holdout
    analyzable: false
    not_analyzable_reason: below floor
    news_ts_utc: "2025-10-10T09:00:00Z"
    news_ts_precision: minute
    wallets: []
"""


def test_load_registry_splits_public_and_sealed(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(_REGISTRY)
    public, sealed = load_registry(path)

    assert [c.id for c in public] == ["case-a", "case-b"]
    a = public[0]
    assert a.split == "dev"
    assert a.news_ts == 1_767_420_000  # 2026-01-03T06:00:00Z
    assert a.news_precision == "hour"
    assert a.primary is not None and a.primary.slug == "slug-a"
    assert [m.slug for m in a.markets] == ["slug-a", "slug-a2"]
    assert public[1].analyzable is False
    assert public[1].markets == ()

    # sealed: only full 42-char addresses, lowercased and stripped
    assert sealed["case-a"].wallets == ("0xabcdefabcdef0123456789abcdef0123456789ab",)
    assert sealed["case-b"].wallets == ()


def test_public_case_has_no_wallet_field(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(_REGISTRY)
    public, _ = load_registry(path)
    assert not hasattr(public[0], "wallets")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.insider_episode_lib'`.

- [ ] **Step 4: Write the library types and the guard**

`scripts/insider_episode_lib.py`:

```python
"""Pure logic for the insider-episode catalogue (no I/O).

Dataclasses, the frozen scorer (spec Amendment 1), ranking, the blind-test
comparator, the automatic price-move anchor, control matching, and report
rendering. Everything here is deterministic and testable without a corpus.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field


class HoldoutLockedError(RuntimeError):
    """Raised when holdout cases are requested before the scorer is frozen."""


@dataclass(frozen=True, slots=True)
class MarketRef:
    """One corpus market attached to a case."""

    slug: str
    winning_side: str
    role: str
    condition_id: str | None = None


@dataclass(frozen=True, slots=True)
class PublicCase:
    """The part of a registry case every pipeline stage may see."""

    id: str
    name: str
    split: str
    confidence: str
    analyzable: bool
    news_ts: int
    news_precision: str
    markets: tuple[MarketRef, ...] = field(default=())

    @property
    def primary(self) -> MarketRef | None:
        """First market whose role starts with ``primary``, if any."""
        for m in self.markets:
            if m.role.startswith("primary"):
                return m
        return None


@dataclass(frozen=True, slots=True)
class SealedCase:
    """Published wallet addresses. Read only by the blind-test comparator."""

    id: str
    wallets: tuple[str, ...]


def select_cases(
    cases: list[PublicCase], split: str, *, scorer_frozen: bool
) -> list[PublicCase]:
    """Filter cases by split, refusing holdout access until the scorer is frozen.

    Args:
        cases: All public cases from the registry.
        split: ``dev``, ``holdout`` or ``all``.
        scorer_frozen: Operator attestation that spec Amendment 1 is signed off.

    Returns:
        The selected cases in registry order.

    Raises:
        HoldoutLockedError: ``split`` includes holdout and ``scorer_frozen`` is false.
        ValueError: ``split`` is not one of the three accepted values.
    """
    if split not in {"dev", "holdout", "all"}:
        msg = f"split must be dev, holdout or all, got {split!r}"
        raise ValueError(msg)
    if split != "dev" and not scorer_frozen:
        msg = (
            "holdout cases are locked until the scorer is frozen; "
            "pass --scorer-frozen only after spec Amendment 1 is signed off"
        )
        raise HoldoutLockedError(msg)
    if split == "all":
        return list(cases)
    return [c for c in cases if c.split == split]
```

`scripts/insider_episodes.py` (loader only for now; later tasks extend it):

```python
"""Insider-episode catalogue: rank pre-news buyers on documented case markets.

Reads ``docs/research/insider-cases.yaml``, extracts the pre-anchor window on
each case market from the corpus, enriches buyers with data-api provenance,
scores them with the frozen scorer, compares against matched control markets,
blind-tests published wallets on holdout cases, and writes markdown reports.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""
# ruff: noqa: T201  # script prints progress to stdout by design

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.insider_episode_lib import (  # noqa: E402  (after sys.path bootstrap)
    MarketRef,
    PublicCase,
    SealedCase,
)

_FULL_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")


def _parse_ts(value: str) -> int:
    """Parse an ISO-8601 UTC timestamp like ``2026-01-03T06:00:00Z`` to unix seconds."""
    return int(dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _published_wallets(raw: list[dict[str, Any]]) -> tuple[str, ...]:
    """Keep only full 42-character addresses, lowercased and stripped."""
    out: list[str] = []
    for w in raw:
        addr = w.get("address")
        if not isinstance(addr, str):
            continue
        addr = addr.strip().lower()
        if _FULL_ADDRESS.match(addr):
            out.append(addr)
    return tuple(out)


def _market_refs(raw: list[dict[str, Any]]) -> tuple[MarketRef, ...]:
    return tuple(
        MarketRef(
            slug=str(m["market_slug"]),
            winning_side=str(m["winning_side"]).upper(),
            role=str(m.get("role", "")),
        )
        for m in raw
    )


def load_registry(path: Path) -> tuple[list[PublicCase], dict[str, SealedCase]]:
    """Load the case registry and split it into public and sealed records.

    Args:
        path: Path to ``insider-cases.yaml``.

    Returns:
        ``(public_cases, sealed_by_case_id)``. Only the comparator may read the
        sealed mapping.
    """
    doc = yaml.safe_load(path.read_text())
    public: list[PublicCase] = []
    sealed: dict[str, SealedCase] = {}
    for raw in doc["cases"]:
        cid = str(raw["id"])
        public.append(
            PublicCase(
                id=cid,
                name=str(raw.get("name", cid)),
                split=str(raw["split"]),
                confidence=str(raw["confidence"]),
                analyzable=bool(raw.get("analyzable", False)),
                news_ts=_parse_ts(str(raw["news_ts_utc"])),
                news_precision=str(raw.get("news_ts_precision", "day")),
                markets=_market_refs(raw.get("corpus_markets") or []),
            )
        )
        sealed[cid] = SealedCase(id=cid, wallets=_published_wallets(raw.get("wallets") or []))
    return public, sealed
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py -q`
Expected: 7 passed.

- [ ] **Step 6: Lint, type-check, commit**

Run: `uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check`
Expected: clean (ty may report the pre-existing baseline count on `main`; zero new).

```bash
git add pyproject.toml uv.lock scripts/insider_episode_lib.py scripts/insider_episodes.py tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py
git commit -m "feat(insider): registry loader with public/sealed split and holdout guard"
```

---

### Task 2: Corpus fixture extension and episode extractor

**Files:**
- Modify: `tests/scripts/conftest.py`
- Modify: `scripts/insider_episode_lib.py` (add `WindowTrade`)
- Modify: `scripts/insider_episodes.py` (add `open_corpus`, `resolve_market`, `extract_window`, `market_trades`)
- Test: `tests/scripts/test_insider_episodes.py`

**Interfaces:**
- Consumes: `MarketRef`, `PublicCase` from Task 1.
- Produces:
  - `insider_episode_lib.WindowTrade(wallet: str, ts: int, outcome_side: str, bs: str, price: float, notional: float)`
  - `insider_episodes.open_corpus(path: Path) -> sqlite3.Connection` (read-only URI, `row_factory = sqlite3.Row`)
  - `insider_episodes.resolve_market(conn, ref: MarketRef) -> MarketRef` (fills `condition_id`; raises `LookupError` if the slug is absent)
  - `insider_episodes.extract_window(conn, condition_id: str, anchor: int, window_hours: int) -> list[WindowTrade]` — every trade on the market with `anchor - window_hours*3600 < ts <= anchor`, all sides and both BUY/SELL, ordered by ts
  - `insider_episodes.market_trades(conn, condition_id: str) -> list[WindowTrade]` — the market's whole trade history ordered by ts (for the auto anchor)
  - `insider_episodes.candidate_wallets(trades: list[WindowTrade], winning_side: str) -> list[str]` — distinct wallets with at least one BUY of `winning_side`, in order of first such buy
- Fixture: `corpus_factory(trades, resolutions, markets=None, with_platform=True)` where `markets` is a list of `(condition_id, market_slug, category, total_volume_usd, closed_at, truncated_at_offset_cap)`

- [ ] **Step 1: Extend the fixture**

In `tests/scripts/conftest.py`, add after the `Resolution` alias:

```python
# market tuple: (condition_id, market_slug, category, total_volume_usd, closed_at, truncated)
Market = tuple[str, str, str, float, int, int]
```

Change `_build`'s signature to `(path, trades, resolutions, markets, *, with_platform)` and add to the `executescript` string, after the `market_resolutions` DDL:

```sql
        CREATE TABLE corpus_markets (
          {plat_col}
          condition_id TEXT NOT NULL, event_slug TEXT NOT NULL, category TEXT,
          closed_at INTEGER NOT NULL, total_volume_usd REAL NOT NULL,
          backfill_state TEXT NOT NULL, truncated_at_offset_cap INTEGER NOT NULL DEFAULT 0,
          enumerated_at INTEGER NOT NULL, market_slug TEXT
        );
```

and after the resolutions insert loop:

```python
    for cid, slug, category, volume, closed_at, truncated in markets:
        conn.execute(
            f"INSERT INTO corpus_markets VALUES ({prefix}?,?,?,?,?,'complete',?,?,?)",  # noqa: S608
            (cid, slug, category, closed_at, volume, truncated, closed_at, slug),
        )
```

Change the fixture's inner `make` to accept `markets: list[Market] | None = None` as a keyword and pass `markets or []` to `_build`. Existing callers pass no `markets` and keep working.

- [ ] **Step 2: Write the failing tests**

Append to `tests/scripts/test_insider_episodes.py`:

```python
from scripts.insider_episode_lib import MarketRef, WindowTrade  # noqa: E402
from scripts.insider_episodes import (  # noqa: E402
    candidate_wallets,
    extract_window,
    market_trades,
    open_corpus,
    resolve_market,
)

ANCHOR = 1_000_000
H = 3600


def _window_db(corpus_factory):
    trades = [
        ("w_early", "m1", "YES", "BUY", 0.10, 100.0, ANCHOR - 80 * H),  # outside 72h
        ("w_edge", "m1", "YES", "BUY", 0.10, 100.0, ANCHOR - 72 * H),  # exactly on boundary: out
        ("w_in", "m1", "YES", "BUY", 0.12, 200.0, ANCHOR - 10 * H),
        ("w_in", "m1", "YES", "BUY", 0.15, 300.0, ANCHOR - 5 * H),
        ("w_seller", "m1", "YES", "SELL", 0.20, 50.0, ANCHOR - 4 * H),  # closing, not a buyer
        ("w_no", "m1", "NO", "BUY", 0.85, 400.0, ANCHOR - 3 * H),
        ("w_at", "m1", "YES", "BUY", 0.30, 100.0, ANCHOR),  # inclusive upper bound
        ("w_late", "m1", "YES", "BUY", 0.90, 100.0, ANCHOR + 1),
        ("w_other", "m2", "YES", "BUY", 0.50, 100.0, ANCHOR - 1 * H),
    ]
    resolutions = [("m1", 1, ANCHOR + 3 * H), ("m2", 1, ANCHOR + 9 * H)]
    markets = [
        ("m1", "slug-m1", "geopolitics", 5_000_000.0, ANCHOR + 3 * H, 0),
        ("m2", "slug-m2", "geopolitics", 1_000_000.0, ANCHOR + 9 * H, 0),
    ]
    return corpus_factory(trades, resolutions, markets=markets)


def test_resolve_market_fills_condition_id(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        ref = resolve_market(conn, MarketRef(slug="slug-m1", winning_side="YES", role="primary"))
        assert ref.condition_id == "m1"
        with pytest.raises(LookupError):
            resolve_market(conn, MarketRef(slug="nope", winning_side="YES", role="primary"))
    finally:
        conn.close()


def test_extract_window_bounds_are_open_low_closed_high(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = extract_window(conn, "m1", ANCHOR, 72)
    finally:
        conn.close()
    wallets = [r.wallet for r in rows]
    assert "w_early" not in wallets
    assert "w_edge" not in wallets
    assert "w_late" not in wallets
    assert "w_other" not in wallets
    assert wallets == ["w_in", "w_in", "w_seller", "w_no", "w_at"]
    assert all(isinstance(r, WindowTrade) for r in rows)


def test_candidate_wallets_require_winning_side_buy(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = extract_window(conn, "m1", ANCHOR, 72)
    finally:
        conn.close()
    assert candidate_wallets(rows, "YES") == ["w_in", "w_at"]
    assert candidate_wallets(rows, "NO") == ["w_no"]


def test_market_trades_returns_full_history_in_order(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = market_trades(conn, "m1")
    finally:
        conn.close()
    assert [r.ts for r in rows] == sorted(r.ts for r in rows)
    assert len(rows) == 8
```

Add `import pytest` at the top of the test file.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'WindowTrade'`.

- [ ] **Step 4: Implement**

Append to `scripts/insider_episode_lib.py`:

```python
@dataclass(frozen=True, slots=True)
class WindowTrade:
    """One corpus trade inside an analysis window."""

    wallet: str
    ts: int
    outcome_side: str
    bs: str
    price: float
    notional: float
```

Append to `scripts/insider_episodes.py` (add `import sqlite3` and extend the lib import with `WindowTrade`):

```python
def open_corpus(path: Path) -> sqlite3.Connection:
    """Open the corpus read-only. Never writes; safe against a running backfill."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_market(conn: sqlite3.Connection, ref: MarketRef) -> MarketRef:
    """Return ``ref`` with ``condition_id`` filled from ``corpus_markets``.

    Raises:
        LookupError: the slug is not in the corpus.
    """
    row = conn.execute(
        "SELECT condition_id FROM corpus_markets WHERE market_slug = ?", (ref.slug,)
    ).fetchone()
    if row is None:
        msg = f"market slug not in corpus: {ref.slug}"
        raise LookupError(msg)
    return MarketRef(
        slug=ref.slug,
        winning_side=ref.winning_side,
        role=ref.role,
        condition_id=str(row["condition_id"]),
    )


_TRADE_COLS = "wallet_address, ts, outcome_side, bs, price, notional_usd"


def _to_trades(rows: list[sqlite3.Row]) -> list[WindowTrade]:
    return [
        WindowTrade(
            wallet=str(r["wallet_address"]),
            ts=int(r["ts"]),
            outcome_side=str(r["outcome_side"]),
            bs=str(r["bs"]),
            price=float(r["price"]),
            notional=float(r["notional_usd"]),
        )
        for r in rows
    ]


def extract_window(
    conn: sqlite3.Connection, condition_id: str, anchor: int, window_hours: int
) -> list[WindowTrade]:
    """Every trade on ``condition_id`` with ``anchor - window < ts <= anchor``."""
    lo = anchor - window_hours * 3600
    rows = conn.execute(
        f"SELECT {_TRADE_COLS} FROM corpus_trades "  # noqa: S608  (constant column list)
        "WHERE condition_id = ? AND ts > ? AND ts <= ? ORDER BY ts, rowid",
        (condition_id, lo, anchor),
    ).fetchall()
    return _to_trades(rows)


def market_trades(conn: sqlite3.Connection, condition_id: str) -> list[WindowTrade]:
    """The market's whole trade history, oldest first (for the auto anchor)."""
    rows = conn.execute(
        f"SELECT {_TRADE_COLS} FROM corpus_trades WHERE condition_id = ? ORDER BY ts, rowid",  # noqa: S608
        (condition_id,),
    ).fetchall()
    return _to_trades(rows)


def candidate_wallets(trades: list[WindowTrade], winning_side: str) -> list[str]:
    """Distinct wallets with at least one BUY of ``winning_side``, by first such buy."""
    seen: dict[str, None] = {}
    for t in trades:
        if t.bs == "BUY" and t.outcome_side == winning_side and t.wallet not in seen:
            seen[t.wallet] = None
    return list(seen)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/ -q`
Expected: all pass, including the pre-existing `test_find_insider_wallets.py` and `test_copy_selection.py` (the fixture change is additive).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add tests/scripts/conftest.py scripts/insider_episode_lib.py scripts/insider_episodes.py tests/scripts/test_insider_episodes.py
git commit -m "feat(insider): corpus fixture markets table and episode window extractor"
```

---

### Task 3: Automatic price-move anchor

**Files:**
- Modify: `scripts/insider_episode_lib.py`
- Test: `tests/scripts/test_insider_episode_lib.py`

**Interfaces:**
- Consumes: `WindowTrade`.
- Produces:
  - `insider_episode_lib.HourBucket(hour_start: int, side_prob_vwap: float, notional: float, n_trades: int)`
  - `insider_episode_lib.hourly_buckets(trades: list[WindowTrade], winning_side: str) -> list[HourBucket]` (side-normalized: YES-side trade price for YES winner, `1 - price` otherwise; buckets only for hours with trades; ascending)
  - `insider_episode_lib.auto_anchor(buckets: list[HourBucket], resolved_at: int, *, min_move: float = 0.10, horizon_hours: int = 24) -> int | None` — returns the `hour_start` of the earliest bucket from which the largest forward move within `horizon_hours` occurs, ignoring buckets at or after `resolved_at`; `None` if the largest move is under `min_move`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episode_lib.py`:

```python
from scripts.insider_episode_lib import HourBucket, WindowTrade, auto_anchor, hourly_buckets  # noqa: E402

H = 3600


def _t(ts: int, side: str, price: float, notional: float = 100.0, wallet: str = "w") -> WindowTrade:
    return WindowTrade(wallet=wallet, ts=ts, outcome_side=side, bs="BUY", price=price, notional=notional)


def test_hourly_buckets_side_normalizes_and_volume_weights() -> None:
    trades = [
        _t(10 * H + 5, "YES", 0.20, notional=100.0),
        _t(10 * H + 50, "NO", 0.70, notional=300.0),  # YES-side prob 0.30
        _t(12 * H, "YES", 0.50),
    ]
    b = hourly_buckets(trades, "YES")
    assert [x.hour_start for x in b] == [10 * H, 12 * H]
    assert b[0].n_trades == 2
    assert b[0].notional == 400.0
    assert abs(b[0].side_prob_vwap - (0.20 * 100 + 0.30 * 300) / 400) < 1e-9


def test_hourly_buckets_for_no_winner_flips_prices() -> None:
    b = hourly_buckets([_t(0, "YES", 0.80)], "NO")
    assert abs(b[0].side_prob_vwap - 0.20) < 1e-9


def _flat(n: int, p: float) -> list[HourBucket]:
    return [HourBucket(hour_start=i * H, side_prob_vwap=p, notional=100.0, n_trades=1) for i in range(n)]


def test_auto_anchor_finds_start_of_largest_jump() -> None:
    buckets = _flat(40, 0.10)
    # jump begins at hour 30: 0.10 -> 0.45 over hours 30..33
    for i, p in zip(range(30, 34), (0.10, 0.25, 0.40, 0.45), strict=True):
        buckets[i] = HourBucket(hour_start=i * H, side_prob_vwap=p, notional=100.0, n_trades=1)
    assert auto_anchor(buckets, resolved_at=100 * H) == 30 * H


def test_auto_anchor_none_when_move_below_minimum() -> None:
    buckets = _flat(40, 0.10)
    buckets[20] = HourBucket(hour_start=20 * H, side_prob_vwap=0.15, notional=100.0, n_trades=1)
    assert auto_anchor(buckets, resolved_at=100 * H) is None


def test_auto_anchor_ignores_buckets_at_or_after_resolution() -> None:
    buckets = _flat(10, 0.10)
    buckets[9] = HourBucket(hour_start=9 * H, side_prob_vwap=0.99, notional=100.0, n_trades=1)
    # resolution at hour 9: the 0.99 bucket is the resolution snap, not a pre-move
    assert auto_anchor(buckets, resolved_at=9 * H) is None


def test_auto_anchor_empty_input() -> None:
    assert auto_anchor([], resolved_at=1) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py -q`
Expected: FAIL with `ImportError: cannot import name 'HourBucket'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py`:

```python
@dataclass(frozen=True, slots=True)
class HourBucket:
    """One hour of side-normalized trading on a market."""

    hour_start: int
    side_prob_vwap: float
    notional: float
    n_trades: int


def side_prob(trade: WindowTrade, winning_side: str) -> float:
    """Implied probability of ``winning_side`` from one trade's price."""
    return trade.price if trade.outcome_side == winning_side else 1.0 - trade.price


def hourly_buckets(trades: list[WindowTrade], winning_side: str) -> list[HourBucket]:
    """Volume-weighted hourly side probability, only for hours with trades, ascending."""
    acc: dict[int, list[float]] = {}
    for t in trades:
        hour = (t.ts // 3600) * 3600
        slot = acc.setdefault(hour, [0.0, 0.0, 0.0])
        slot[0] += side_prob(t, winning_side) * t.notional
        slot[1] += t.notional
        slot[2] += 1
    out: list[HourBucket] = []
    for hour in sorted(acc):
        num, den, n = acc[hour]
        if den <= 0:
            continue
        out.append(HourBucket(hour_start=hour, side_prob_vwap=num / den, notional=den, n_trades=int(n)))
    return out


def auto_anchor(
    buckets: list[HourBucket],
    resolved_at: int,
    *,
    min_move: float = 0.10,
    horizon_hours: int = 24,
) -> int | None:
    """Start hour of the largest forward move toward the winner before resolution.

    For each bucket ``b`` strictly before ``resolved_at``, the forward move is
    ``max(price over (b, b + horizon])`` minus ``price[b]``, using only buckets
    also strictly before ``resolved_at``. The earliest bucket achieving the
    largest move is returned; ``None`` if that move is below ``min_move``.
    """
    pre = [b for b in buckets if b.hour_start < resolved_at]
    best_move = 0.0
    best_start: int | None = None
    horizon = horizon_hours * 3600
    for i, b in enumerate(pre):
        later = [x.side_prob_vwap for x in pre[i + 1 :] if x.hour_start <= b.hour_start + horizon]
        if not later:
            continue
        move = max(later) - b.side_prob_vwap
        if move > best_move:
            best_move = move
            best_start = b.hour_start
    if best_start is None or best_move < min_move:
        return None
    return best_start
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py tests/scripts/test_insider_episode_lib.py
git commit -m "feat(insider): hourly side-prob buckets and automatic price-move anchor"
```

---

### Task 4: Per-wallet features from the corpus

**Files:**
- Modify: `scripts/insider_episode_lib.py` (add `WalletFeatures`, `window_features`)
- Modify: `scripts/insider_episodes.py` (add `lifetime_stats`, `build_features`)
- Test: `tests/scripts/test_insider_episode_lib.py`, `tests/scripts/test_insider_episodes.py`

**Interfaces:**
- Consumes: `WindowTrade`, `open_corpus`, `extract_window`, `candidate_wallets`.
- Produces:
  - `insider_episode_lib.WalletFeatures(wallet, n_w: float, n_buys: int, first_buy_ts: int, last_buy_ts: int, flip: bool, breadth: int, n_life: float, corpus_first_ts: int)`
  - `insider_episode_lib.window_features(trades: list[WindowTrade], wallet: str, winning_side: str) -> tuple[float, int, int, int, bool]` = `(n_w, n_buys, first_buy_ts, last_buy_ts, flip)`; pure
  - `insider_episodes.lifetime_stats(conn, wallet: str, anchor: int) -> tuple[int, float, int]` = `(breadth, n_life, corpus_first_ts)` with every row filtered `ts <= anchor`
  - `insider_episodes.build_features(conn, trades, wallets: list[str], winning_side: str, anchor: int) -> list[WalletFeatures]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episode_lib.py`:

```python
from scripts.insider_episode_lib import window_features  # noqa: E402


def _tr(wallet: str, ts: int, side: str, bs: str, price: float, notional: float) -> WindowTrade:
    return WindowTrade(wallet=wallet, ts=ts, outcome_side=side, bs=bs, price=price, notional=notional)


def test_window_features_sums_winning_buys_only() -> None:
    trades = [
        _tr("a", 100, "YES", "BUY", 0.10, 1000.0),
        _tr("a", 200, "NO", "BUY", 0.80, 500.0),  # wrong side, ignored for n_w
        _tr("a", 300, "YES", "SELL", 0.30, 200.0),  # sell, ignored
        _tr("a", 400, "YES", "BUY", 0.20, 2000.0),
        _tr("b", 150, "YES", "BUY", 0.10, 50.0),
    ]
    n_w, n_buys, first, last, flip = window_features(trades, "a", "YES")
    assert n_w == 3000.0
    assert n_buys == 2
    assert (first, last) == (100, 400)
    assert flip is False


def test_window_features_flip_requires_opposing_sell_before_first_buy() -> None:
    flipper = [
        _tr("f", 100, "NO", "SELL", 0.90, 500.0),
        _tr("f", 200, "YES", "BUY", 0.15, 1000.0),
    ]
    assert window_features(flipper, "f", "YES")[4] is True
    late_sell = [
        _tr("g", 100, "YES", "BUY", 0.15, 1000.0),
        _tr("g", 200, "NO", "SELL", 0.90, 500.0),  # after the buy: not a flip
    ]
    assert window_features(late_sell, "g", "YES")[4] is False
```

Append to `tests/scripts/test_insider_episodes.py`:

```python
from scripts.insider_episodes import build_features, lifetime_stats  # noqa: E402


def test_lifetime_stats_filtered_to_anchor(corpus_factory) -> None:
    trades = [
        ("w", "m1", "YES", "BUY", 0.10, 100.0, 500),
        ("w", "m2", "YES", "BUY", 0.10, 200.0, 600),
        ("w", "m3", "YES", "BUY", 0.10, 400.0, 2000),  # after anchor: excluded
    ]
    db = corpus_factory(trades, [("m1", 1, 3000)])
    conn = open_corpus(db)
    try:
        breadth, n_life, first_ts = lifetime_stats(conn, "w", anchor=1000)
    finally:
        conn.close()
    assert (breadth, n_life, first_ts) == (2, 300.0, 500)


def test_build_features_joins_window_and_lifetime(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        trades = extract_window(conn, "m1", ANCHOR, 72)
        feats = {f.wallet: f for f in build_features(conn, trades, ["w_in", "w_at"], "YES", ANCHOR)}
    finally:
        conn.close()
    w_in = feats["w_in"]
    assert w_in.n_w == 500.0
    assert w_in.n_buys == 2
    assert w_in.breadth == 1
    assert w_in.n_life == 500.0
    assert w_in.corpus_first_ts == ANCHOR - 10 * H
    assert w_in.flip is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'window_features'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py`:

```python
@dataclass(frozen=True, slots=True)
class WalletFeatures:
    """Scorer inputs for one candidate wallet. Nothing here post-dates the anchor."""

    wallet: str
    n_w: float
    n_buys: int
    first_buy_ts: int
    last_buy_ts: int
    flip: bool
    breadth: int
    n_life: float
    corpus_first_ts: int


def window_features(
    trades: list[WindowTrade], wallet: str, winning_side: str
) -> tuple[float, int, int, int, bool]:
    """Window-only features: ``(n_w, n_buys, first_buy_ts, last_buy_ts, flip)``.

    ``flip`` is true when the wallet SOLD the opposing outcome inside the
    window strictly before its first winning-side BUY.
    """
    buys = [t for t in trades if t.wallet == wallet and t.bs == "BUY" and t.outcome_side == winning_side]
    if not buys:
        return 0.0, 0, 0, 0, False
    first_buy = min(t.ts for t in buys)
    last_buy = max(t.ts for t in buys)
    flip = any(
        t.wallet == wallet and t.bs == "SELL" and t.outcome_side != winning_side and t.ts < first_buy
        for t in trades
    )
    return sum(t.notional for t in buys), len(buys), first_buy, last_buy, flip
```

Append to `scripts/insider_episodes.py` (extend lib import with `WalletFeatures, window_features`):

```python
def lifetime_stats(conn: sqlite3.Connection, wallet: str, anchor: int) -> tuple[int, float, int]:
    """``(distinct markets, total notional, first trade ts)`` for ``wallet`` with ``ts <= anchor``."""
    row = conn.execute(
        "SELECT count(DISTINCT condition_id), coalesce(sum(notional_usd), 0), min(ts) "
        "FROM corpus_trades WHERE wallet_address = ? AND ts <= ?",
        (wallet, anchor),
    ).fetchone()
    return int(row[0]), float(row[1]), int(row[2])


def build_features(
    conn: sqlite3.Connection,
    trades: list[WindowTrade],
    wallets: list[str],
    winning_side: str,
    anchor: int,
) -> list[WalletFeatures]:
    """Join window features with anchor-bounded lifetime stats for each candidate."""
    out: list[WalletFeatures] = []
    for w in wallets:
        n_w, n_buys, first_buy, last_buy, flip = window_features(trades, w, winning_side)
        breadth, n_life, first_ts = lifetime_stats(conn, w, anchor)
        out.append(
            WalletFeatures(
                wallet=w,
                n_w=n_w,
                n_buys=n_buys,
                first_buy_ts=first_buy,
                last_buy_ts=last_buy,
                flip=flip,
                breadth=breadth,
                n_life=n_life,
                corpus_first_ts=first_ts,
            )
        )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py scripts/insider_episodes.py tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py
git commit -m "feat(insider): per-wallet window and anchor-bounded lifetime features"
```

---

### Task 5: Provenance cache and data-api adapter

**Files:**
- Modify: `scripts/insider_episode_lib.py` (add `Provenance`)
- Modify: `scripts/insider_episodes.py` (add `ProvenanceCache`, `enrich_provenance`, `fetch_provenance_api`)
- Test: `tests/scripts/test_insider_episodes.py`

**Interfaces:**
- Produces:
  - `insider_episode_lib.Provenance(wallet: str, first_activity_ts: int | None, n_events: int | None)`
  - `insider_episodes.ProvenanceCache(path: Path)` with `get(wallet) -> Provenance | None`, `put(p: Provenance) -> None`, `close()`; table `wallet_provenance(wallet TEXT PRIMARY KEY, first_activity_ts INTEGER, n_events INTEGER, fetched_at INTEGER NOT NULL)`
  - `insider_episodes.enrich_provenance(cache, wallets: list[str], fetch: Callable[[str], Awaitable[Provenance]]) -> dict[str, Provenance]` (async; cache hit skips `fetch`; a `fetch` that raises `httpx.HTTPError` or `ValueError` is logged, leaves no row, and the wallet is omitted from the result)
  - `insider_episodes.fetch_provenance_api(client: DataClient, wallet: str) -> Provenance` (async; pages `client.get_activity(wallet, limit=500, offset=k*500)` until a short page or offset 10,000; `first_activity_ts` = min `timestamp`, `n_events` = count)

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episodes.py`:

```python
import httpx  # noqa: E402

from scripts.insider_episode_lib import Provenance  # noqa: E402
from scripts.insider_episodes import (  # noqa: E402
    ProvenanceCache,
    enrich_provenance,
    fetch_provenance_api,
)


async def test_enrich_provenance_uses_cache_and_skips_failures(tmp_path: Path) -> None:
    cache = ProvenanceCache(tmp_path / "prov.sqlite3")
    cache.put(Provenance(wallet="0xcached", first_activity_ts=111, n_events=5))
    calls: list[str] = []

    async def fetch(wallet: str) -> Provenance:
        calls.append(wallet)
        if wallet == "0xbad":
            request = httpx.Request("GET", "https://data-api.polymarket.com/activity")
            raise httpx.HTTPStatusError("408", request=request, response=httpx.Response(408, request=request))
        return Provenance(wallet=wallet, first_activity_ts=222, n_events=1)

    try:
        out = await enrich_provenance(cache, ["0xcached", "0xnew", "0xbad"], fetch)
        assert calls == ["0xnew", "0xbad"]
        assert out["0xcached"].first_activity_ts == 111
        assert out["0xnew"].first_activity_ts == 222
        assert "0xbad" not in out
        assert cache.get("0xnew") is not None
        assert cache.get("0xbad") is None  # failure leaves no row
    finally:
        cache.close()


async def test_fetch_provenance_api_pages_and_takes_min_timestamp() -> None:
    class _Client:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        async def get_activity(self, address: str, *, limit: int = 500, offset: int = 0, type: str | None = None):  # noqa: A002
            self.offsets.append(offset)
            if offset == 0:
                return [{"timestamp": 900 + i, "type": "TRADE"} for i in range(500)]
            return [{"timestamp": 50, "type": "TRADE"}, {"type": "REWARD"}]  # short page ends paging

    client = _Client()
    p = await fetch_provenance_api(client, "0xabc")  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    assert client.offsets == [0, 500]
    assert p.first_activity_ts == 50
    assert p.n_events == 502


async def test_fetch_provenance_api_empty_wallet() -> None:
    class _Client:
        async def get_activity(self, address: str, *, limit: int = 500, offset: int = 0, type: str | None = None):  # noqa: A002
            return []

    p = await fetch_provenance_api(_Client(), "0xnone")  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    assert p == Provenance(wallet="0xnone", first_activity_ts=None, n_events=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'Provenance'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py`:

```python
@dataclass(frozen=True, slots=True)
class Provenance:
    """Wallet age signal from the Polymarket data-api."""

    wallet: str
    first_activity_ts: int | None
    n_events: int | None
```

Append to `scripts/insider_episodes.py` (add imports: `import time`, `from collections.abc import Awaitable, Callable`, `import httpx`, `from pscanner.poly.data import DataClient`; extend lib import with `Provenance`):

```python
_ACTIVITY_PAGE = 500
_ACTIVITY_OFFSET_CAP = 10_000


class ProvenanceCache:
    """SQLite cache of data-api provenance keyed by wallet. Failed fetches are never stored."""

    def __init__(self, path: Path) -> None:
        """Open (and create) the cache at ``path``."""
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS wallet_provenance ("
            " wallet TEXT PRIMARY KEY, first_activity_ts INTEGER, n_events INTEGER,"
            " fetched_at INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get(self, wallet: str) -> Provenance | None:
        """Return the cached row for ``wallet`` or ``None``."""
        row = self._conn.execute(
            "SELECT first_activity_ts, n_events FROM wallet_provenance WHERE wallet = ?", (wallet,)
        ).fetchone()
        if row is None:
            return None
        return Provenance(wallet=wallet, first_activity_ts=row[0], n_events=row[1])

    def put(self, p: Provenance) -> None:
        """Insert or replace ``p``."""
        self._conn.execute(
            "INSERT OR REPLACE INTO wallet_provenance VALUES (?,?,?,?)",
            (p.wallet, p.first_activity_ts, p.n_events, int(time.time())),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()


async def enrich_provenance(
    cache: ProvenanceCache,
    wallets: list[str],
    fetch: Callable[[str], Awaitable[Provenance]],
) -> dict[str, Provenance]:
    """Resolve provenance for ``wallets`` through the cache, fetching misses.

    A fetch that raises ``httpx.HTTPError`` or ``ValueError`` is printed and
    skipped; the wallet is absent from the result and nothing is cached, so a
    later run retries it.
    """
    out: dict[str, Provenance] = {}
    for i, w in enumerate(wallets, 1):
        hit = cache.get(w)
        if hit is not None:
            out[w] = hit
            continue
        try:
            p = await fetch(w)
        except (httpx.HTTPError, ValueError) as exc:
            print(f"[provenance {i}/{len(wallets)}] {w} ERR {exc}")
            continue
        cache.put(p)
        out[w] = p
    return out


async def fetch_provenance_api(client: DataClient, wallet: str) -> Provenance:
    """Page ``/activity`` for ``wallet`` and return first-activity ts and event count."""
    offset = 0
    n = 0
    earliest: int | None = None
    while offset < _ACTIVITY_OFFSET_CAP:
        page = await client.get_activity(wallet, limit=_ACTIVITY_PAGE, offset=offset)
        for e in page:
            n += 1
            ts = e.get("timestamp")
            if isinstance(ts, int) and (earliest is None or ts < earliest):
                earliest = ts
        if len(page) < _ACTIVITY_PAGE:
            break
        offset += _ACTIVITY_PAGE
    return Provenance(wallet=wallet, first_activity_ts=earliest, n_events=n)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episodes.py -q`
Expected: all pass. If `filterwarnings=error` flags an unclosed sqlite connection, confirm every test closes the cache in `finally`.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py scripts/insider_episodes.py tests/scripts/test_insider_episodes.py
git commit -m "feat(insider): provenance cache and data-api activity adapter"
```

---

### Task 6: Frozen scorer, ranking, and blind-test comparator

**Files:**
- Modify: `scripts/insider_episode_lib.py`
- Test: `tests/scripts/test_insider_episode_lib.py`

**Interfaces:**
- Consumes: `WalletFeatures`, `Provenance`, `SealedCase`.
- Produces:
  - `insider_episode_lib.age_days(f: WalletFeatures, p: Provenance | None) -> float` — `(first_buy_ts − first_activity)/86400` where `first_activity` is `p.first_activity_ts` if present else `f.corpus_first_ts`; never below 0
  - `insider_episode_lib.freshness(age: float) -> float` — 1 if `age <= 10` else `max(0.1, 1 − (age − 10)/60)`
  - `insider_episode_lib.score(f: WalletFeatures, p: Provenance | None) -> float` — Amendment 1 formula
  - `insider_episode_lib.ScoredWallet(features: WalletFeatures, provenance: Provenance | None, age: float, score: float, rank: int)`
  - `insider_episode_lib.rank_wallets(feats: list[WalletFeatures], prov: dict[str, Provenance]) -> list[ScoredWallet]` — ordered by score desc, `n_w` desc, `first_buy_ts` asc, wallet asc; rank is 1-based position
  - `insider_episode_lib.BlindResult(case_id, anchor_kind: str, n_candidates: int, ranks: dict[str, int | None], best_rank: int | None, passed: bool | None)` — `passed` is `None` when no published wallet is a candidate
  - `insider_episode_lib.compare_published(case_id, anchor_kind, ranked: list[ScoredWallet], sealed: SealedCase, *, pass_rank: int = 10) -> BlindResult`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episode_lib.py`:

```python
import math  # noqa: E402

from scripts.insider_episode_lib import (  # noqa: E402
    Provenance,
    SealedCase,
    WalletFeatures,
    age_days,
    compare_published,
    freshness,
    rank_wallets,
    score,
)

DAY = 86_400


def _f(
    wallet: str,
    *,
    n_w: float,
    breadth: int = 1,
    n_life: float | None = None,
    flip: bool = False,
    first_buy_ts: int = 100 * DAY,
    corpus_first_ts: int = 99 * DAY,
) -> WalletFeatures:
    return WalletFeatures(
        wallet=wallet,
        n_w=n_w,
        n_buys=1,
        first_buy_ts=first_buy_ts,
        last_buy_ts=first_buy_ts,
        flip=flip,
        breadth=breadth,
        n_life=n_w if n_life is None else n_life,
        corpus_first_ts=corpus_first_ts,
    )


def test_freshness_plateau_decay_floor() -> None:
    assert freshness(0) == 1.0
    assert freshness(10) == 1.0
    assert math.isclose(freshness(40), 0.5)
    assert math.isclose(freshness(64), 0.1)
    assert freshness(400) == 0.1


def test_age_days_prefers_api_then_corpus_and_never_negative() -> None:
    f = _f("w", n_w=100.0, first_buy_ts=100 * DAY, corpus_first_ts=98 * DAY)
    assert math.isclose(age_days(f, Provenance("w", first_activity_ts=90 * DAY, n_events=1)), 10.0)
    assert math.isclose(age_days(f, Provenance("w", first_activity_ts=None, n_events=0)), 2.0)
    assert math.isclose(age_days(f, None), 2.0)
    api_after_buy = Provenance("w", first_activity_ts=101 * DAY, n_events=1)
    assert age_days(f, api_after_buy) == 0.0


def test_score_matches_amendment_formula() -> None:
    f = _f("w", n_w=9_999.0, breadth=8, n_life=19_998.0, flip=True)  # conc = 0.5
    p = Provenance("w", first_activity_ts=100 * DAY - 40 * DAY, n_events=1)  # age 40 -> fresh 0.5
    expected = math.log10(1 + 9_999.0) * 0.5 * (0.5 + 0.5 * 0.5) / math.log10(2 + 8) + 0.5
    assert math.isclose(score(f, p), expected)


def test_score_concentration_exactly_one_when_window_is_whole_life() -> None:
    f = _f("w", n_w=1_000.0, n_life=1_000.0)
    assert math.isclose(score(f, None), math.log10(1_001.0) * 1.0 * 1.0 / math.log10(3))


def test_rank_wallets_order_and_tiebreaks() -> None:
    feats = [
        _f("0xbb", n_w=1_000.0),
        _f("0xaa", n_w=1_000.0),  # identical score and n_w and first_buy: wallet asc wins
        _f("0xmm", n_w=50_000.0, breadth=900, n_life=5_000_000.0),  # market-maker, low score
        _f("0xff", n_w=200.0, flip=True),
    ]
    ranked = rank_wallets(feats, {})
    order = [r.features.wallet for r in ranked]
    assert order[:2] == ["0xaa", "0xbb"]
    assert [r.rank for r in ranked] == [1, 2, 3, 4]
    assert order.index("0xmm") > order.index("0xaa")


def test_rank_wallets_fresh_single_market_beats_market_maker_at_equal_notional() -> None:
    fresh = _f("0xfresh", n_w=20_000.0, breadth=1, n_life=20_000.0)
    mm = _f("0xmm", n_w=20_000.0, breadth=800, n_life=30_000_000.0, corpus_first_ts=0)
    ranked = rank_wallets([mm, fresh], {})
    assert ranked[0].features.wallet == "0xfresh"


def test_compare_published_ranks_and_pass() -> None:
    feats = [_f(f"0x{i:02d}", n_w=float(1_000 - i)) for i in range(15)]
    ranked = rank_wallets(feats, {})
    sealed = SealedCase(id="c", wallets=("0x03", "0x12", "0xzz"))
    res = compare_published("c", "news", ranked, sealed)
    assert res.n_candidates == 15
    assert res.ranks == {"0x03": 4, "0x12": 13, "0xzz": None}
    assert res.best_rank == 4
    assert res.passed is True
    res2 = compare_published("c", "news", ranked, SealedCase(id="c", wallets=("0x12",)))
    assert res2.passed is False
    res3 = compare_published("c", "news", ranked, SealedCase(id="c", wallets=("0xzz",)))
    assert res3.best_rank is None and res3.passed is None


def test_compare_published_is_case_insensitive_on_addresses() -> None:
    ranked = rank_wallets([_f("0xabc", n_w=1.0)], {})
    res = compare_published("c", "news", ranked, SealedCase(id="c", wallets=("0xABC",)))
    assert res.ranks == {"0xabc": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py -q`
Expected: FAIL with `ImportError: cannot import name 'age_days'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py` (add `import math` at the top):

```python
_FRESH_PLATEAU_DAYS = 10.0
_FRESH_DECAY_DAYS = 60.0
_FRESH_FLOOR = 0.1
_CONC_BLEND = 0.5
_FLIP_BONUS = 0.5
_SECONDS_PER_DAY = 86_400


def age_days(f: WalletFeatures, p: Provenance | None) -> float:
    """Wallet age at its first window buy, in days; data-api first, corpus fallback."""
    first = p.first_activity_ts if p is not None and p.first_activity_ts is not None else f.corpus_first_ts
    return max(0.0, (f.first_buy_ts - first) / _SECONDS_PER_DAY)


def freshness(age: float) -> float:
    """1 up to the plateau, then linear decay to a floor (spec Amendment 1)."""
    if age <= _FRESH_PLATEAU_DAYS:
        return 1.0
    return max(_FRESH_FLOOR, 1.0 - (age - _FRESH_PLATEAU_DAYS) / _FRESH_DECAY_DAYS)


def score(f: WalletFeatures, p: Provenance | None) -> float:
    """The frozen composite from spec Amendment 1."""
    conc = f.n_w / f.n_life if f.n_life > 0 else 1.0
    conc = min(conc, 1.0)
    base = math.log10(1.0 + f.n_w) * freshness(age_days(f, p)) * (_CONC_BLEND + _CONC_BLEND * conc)
    return base / math.log10(2.0 + f.breadth) + (_FLIP_BONUS if f.flip else 0.0)


@dataclass(frozen=True, slots=True)
class ScoredWallet:
    """A candidate with its score and 1-based rank within its market."""

    features: WalletFeatures
    provenance: Provenance | None
    age: float
    score: float
    rank: int


def rank_wallets(feats: list[WalletFeatures], prov: dict[str, Provenance]) -> list[ScoredWallet]:
    """Score and order candidates: score desc, notional desc, earliest buy, wallet."""
    scored = [
        (f, prov.get(f.wallet), age_days(f, prov.get(f.wallet)), score(f, prov.get(f.wallet)))
        for f in feats
    ]
    scored.sort(key=lambda t: (-t[3], -t[0].n_w, t[0].first_buy_ts, t[0].wallet))
    return [
        ScoredWallet(features=f, provenance=p, age=a, score=s, rank=i)
        for i, (f, p, a, s) in enumerate(scored, 1)
    ]


@dataclass(frozen=True, slots=True)
class BlindResult:
    """Where the published wallets landed in a ranking they were hidden from."""

    case_id: str
    anchor_kind: str
    n_candidates: int
    ranks: dict[str, int | None]
    best_rank: int | None
    passed: bool | None


def compare_published(
    case_id: str,
    anchor_kind: str,
    ranked: list[ScoredWallet],
    sealed: SealedCase,
    *,
    pass_rank: int = 10,
) -> BlindResult:
    """Look up each published wallet's rank. ``passed`` is None if none are candidates."""
    by_wallet = {r.features.wallet.lower(): r.rank for r in ranked}
    ranks = {w.lower(): by_wallet.get(w.lower()) for w in sealed.wallets}
    found = [r for r in ranks.values() if r is not None]
    best = min(found) if found else None
    return BlindResult(
        case_id=case_id,
        anchor_kind=anchor_kind,
        n_candidates=len(ranked),
        ranks=ranks,
        best_rank=best,
        passed=None if best is None else best <= pass_rank,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py tests/scripts/test_insider_episode_lib.py
git commit -m "feat(insider): frozen scorer, ranking, and blind-test comparator"
```

---

### Task 7: Control-market matcher

**Files:**
- Modify: `scripts/insider_episode_lib.py` (add `MarketRow`, `match_controls`)
- Modify: `scripts/insider_episodes.py` (add `load_market_rows`)
- Test: `tests/scripts/test_insider_episode_lib.py`, `tests/scripts/test_insider_episodes.py`

**Interfaces:**
- Produces:
  - `insider_episode_lib.MarketRow(condition_id, slug, category: str | None, volume: float, resolved_at: int, yes_won: int, truncated: bool)`
  - `insider_episode_lib.match_controls(case: MarketRow, universe: list[MarketRow], excluded: set[str], *, k: int = 5) -> list[MarketRow]` — same category, `case.volume/3 <= volume <= case.volume*3`, `|resolved_at − case.resolved_at| <= 180 days`, same `yes_won`, `condition_id` not in `excluded` and not the case; ordered by `|log(volume/case.volume)|` then slug; first `k`
  - `insider_episodes.load_market_rows(conn) -> list[MarketRow]` — every row of `corpus_markets JOIN market_resolutions`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episode_lib.py`:

```python
from scripts.insider_episode_lib import MarketRow, match_controls  # noqa: E402


def _m(cid: str, *, cat: str = "geo", vol: float = 10e6, res: int = 1_000 * DAY, yes: int = 1) -> MarketRow:
    return MarketRow(condition_id=cid, slug=f"slug-{cid}", category=cat, volume=vol, resolved_at=res, yes_won=yes, truncated=False)


def test_match_controls_filters_and_orders() -> None:
    case = _m("case")
    universe = [
        case,
        _m("c_close", vol=11e6),
        _m("c_far", vol=29e6),
        _m("c_too_big", vol=31e6),
        _m("c_too_small", vol=3e6),
        _m("c_cat", cat="sports"),
        _m("c_era", res=1_000 * DAY + 181 * DAY),
        _m("c_side", yes=0),
        _m("c_registry"),
        _m("c_exact", vol=10e6),
    ]
    out = match_controls(case, universe, excluded={"c_registry"}, k=5)
    assert [m.condition_id for m in out] == ["c_exact", "c_close", "c_far"]


def test_match_controls_respects_k_and_is_deterministic() -> None:
    case = _m("case")
    universe = [case] + [_m(f"c{i}", vol=10e6) for i in range(9)]
    out = match_controls(case, universe, excluded=set(), k=3)
    assert [m.condition_id for m in out] == ["c0", "c1", "c2"]


def test_match_controls_empty_when_nothing_matches() -> None:
    case = _m("case")
    assert match_controls(case, [case], excluded=set()) == []
```

Append to `tests/scripts/test_insider_episodes.py`:

```python
from scripts.insider_episodes import load_market_rows  # noqa: E402


def test_load_market_rows_joins_resolutions(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        rows = {r.condition_id: r for r in load_market_rows(conn)}
    finally:
        conn.close()
    assert rows["m1"].slug == "slug-m1"
    assert rows["m1"].category == "geopolitics"
    assert rows["m1"].yes_won == 1
    assert rows["m1"].resolved_at == ANCHOR + 3 * H
    assert rows["m1"].truncated is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'MarketRow'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py`:

```python
_CONTROL_VOLUME_FACTOR = 3.0
_CONTROL_ERA_SECONDS = 180 * _SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class MarketRow:
    """A resolved corpus market, for control matching and coverage diagnosis."""

    condition_id: str
    slug: str
    category: str | None
    volume: float
    resolved_at: int
    yes_won: int
    truncated: bool


def match_controls(
    case: MarketRow, universe: list[MarketRow], excluded: set[str], *, k: int = 5
) -> list[MarketRow]:
    """Up to ``k`` control markets matched on category, volume band, era, and side."""
    lo, hi = case.volume / _CONTROL_VOLUME_FACTOR, case.volume * _CONTROL_VOLUME_FACTOR
    pool = [
        m
        for m in universe
        if m.condition_id != case.condition_id
        and m.condition_id not in excluded
        and m.category == case.category
        and lo <= m.volume <= hi
        and abs(m.resolved_at - case.resolved_at) <= _CONTROL_ERA_SECONDS
        and m.yes_won == case.yes_won
    ]
    pool.sort(key=lambda m: (abs(math.log(m.volume / case.volume)), m.slug))
    return pool[:k]
```

Append to `scripts/insider_episodes.py` (extend lib import with `MarketRow`):

```python
def load_market_rows(conn: sqlite3.Connection) -> list[MarketRow]:
    """Every resolved corpus market (the control universe)."""
    rows = conn.execute(
        "SELECT m.condition_id, m.market_slug, m.category, m.total_volume_usd,"
        " r.resolved_at, r.outcome_yes_won, m.truncated_at_offset_cap"
        " FROM corpus_markets m JOIN market_resolutions r ON r.condition_id = m.condition_id"
    ).fetchall()
    return [
        MarketRow(
            condition_id=str(r[0]),
            slug=str(r[1] or ""),
            category=None if r[2] is None else str(r[2]),
            volume=float(r[3]),
            resolved_at=int(r[4]),
            yes_won=int(r[5]),
            truncated=bool(r[6]),
        )
        for r in rows
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py scripts/insider_episodes.py tests/scripts/test_insider_episode_lib.py tests/scripts/test_insider_episodes.py
git commit -m "feat(insider): control-market matcher"
```

---

### Task 8: Coverage audit (published wallets not in corpus)

**Files:**
- Modify: `scripts/insider_episode_lib.py` (add `MissingReason`, `CoverageRow`)
- Modify: `scripts/insider_episodes.py` (add `diagnose_missing`)
- Test: `tests/scripts/test_insider_episodes.py`

**Interfaces:**
- Produces:
  - `insider_episode_lib.CoverageRow(case_id, wallet, reason: str, detail: str)` with `reason` one of `market_below_volume_floor`, `no_trades_in_market`, `outside_window`, `losing_side_only`, `below_notional_floor`, `backfill_truncated`
  - `insider_episodes.diagnose_missing(conn, case: PublicCase, wallet: str, anchor: int, window_hours: int, market_row: MarketRow | None) -> CoverageRow | None` — `None` when the wallet is a candidate on the primary market (so it appears in the ranking instead)

Decision order: primary market absent from corpus → `market_below_volume_floor`; no trades on the market at all → `backfill_truncated` if `market_row.truncated` else `no_trades_in_market`; trades on the market but none inside the window → `outside_window` with detail "first +Xh / last +Yh relative to anchor"; trades in window but no BUY of the winning side → `losing_side_only`; a winning-side BUY in window with notional under 10 is impossible in the corpus, so `below_notional_floor` is reported only when the market has trades, none by the wallet, and the registry stake (passed as `detail` by the caller when known) is under $10. Keep `below_notional_floor` out of this function's branches: the CLI emits it when `no_trades_in_market` coincides with a registry `stake_usd < 10`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episodes.py`:

```python
from scripts.insider_episode_lib import CoverageRow, MarketRow, PublicCase  # noqa: E402
from scripts.insider_episodes import diagnose_missing  # noqa: E402


def _pcase(slug: str, side: str = "YES", cid: str | None = "m1") -> PublicCase:
    return PublicCase(
        id="c", name="c", split="dev", confidence="alleged", analyzable=cid is not None,
        news_ts=ANCHOR, news_precision="minute",
        markets=(MarketRef(slug=slug, winning_side=side, role="primary", condition_id=cid),),
    )


def _mrow(truncated: bool = False) -> MarketRow:
    return MarketRow("m1", "slug-m1", "geopolitics", 5e6, ANCHOR + 3 * H, 1, truncated)


def test_diagnose_missing_reasons(corpus_factory) -> None:
    conn = open_corpus(_window_db(corpus_factory))
    try:
        # candidate: not missing
        assert diagnose_missing(conn, _pcase("slug-m1"), "w_in", ANCHOR, 72, _mrow()) is None
        # no trades at all on the market
        r = diagnose_missing(conn, _pcase("slug-m1"), "w_ghost", ANCHOR, 72, _mrow())
        assert r is not None and r.reason == "no_trades_in_market"
        # same, but the market's backfill was truncated
        r = diagnose_missing(conn, _pcase("slug-m1"), "w_ghost", ANCHOR, 72, _mrow(truncated=True))
        assert r is not None and r.reason == "backfill_truncated"
        # traded the market, but outside the window
        r = diagnose_missing(conn, _pcase("slug-m1"), "w_early", ANCHOR, 72, _mrow())
        assert r is not None and r.reason == "outside_window" and "-80.0h" in r.detail
        # in window, but only on the losing side
        r = diagnose_missing(conn, _pcase("slug-m1"), "w_no", ANCHOR, 72, _mrow())
        assert r is not None and r.reason == "losing_side_only"
        # market not in corpus
        r = diagnose_missing(conn, _pcase("gone", cid=None), "w_in", ANCHOR, 72, None)
        assert r is not None and r.reason == "market_below_volume_floor"
    finally:
        conn.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'CoverageRow'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py`:

```python
@dataclass(frozen=True, slots=True)
class CoverageRow:
    """Why a published wallet is absent from a case's candidate ranking."""

    case_id: str
    wallet: str
    reason: str
    detail: str
```

Append to `scripts/insider_episodes.py` (extend lib import with `CoverageRow`):

```python
def diagnose_missing(
    conn: sqlite3.Connection,
    case: PublicCase,
    wallet: str,
    anchor: int,
    window_hours: int,
    market_row: MarketRow | None,
) -> CoverageRow | None:
    """Explain why ``wallet`` is not a candidate on the case's primary market, or ``None`` if it is."""
    primary = case.primary
    if primary is None or primary.condition_id is None or market_row is None:
        return CoverageRow(case.id, wallet, "market_below_volume_floor", "primary market absent")
    rows = conn.execute(
        "SELECT ts, outcome_side, bs FROM corpus_trades WHERE wallet_address = ? AND condition_id = ?"
        " ORDER BY ts",
        (wallet, primary.condition_id),
    ).fetchall()
    if not rows:
        reason = "backfill_truncated" if market_row.truncated else "no_trades_in_market"
        return CoverageRow(case.id, wallet, reason, "no corpus trades on the primary market")
    lo = anchor - window_hours * 3600
    in_window = [r for r in rows if lo < int(r[0]) <= anchor]
    if not in_window:
        first_h = (int(rows[0][0]) - anchor) / 3600
        last_h = (int(rows[-1][0]) - anchor) / 3600
        return CoverageRow(
            case.id, wallet, "outside_window", f"first {first_h:+.1f}h / last {last_h:+.1f}h vs anchor"
        )
    if any(str(r[2]) == "BUY" and str(r[1]) == primary.winning_side for r in in_window):
        return None
    return CoverageRow(case.id, wallet, "losing_side_only", f"{len(in_window)} window trades, none buying {primary.winning_side}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episodes.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py scripts/insider_episodes.py tests/scripts/test_insider_episodes.py
git commit -m "feat(insider): coverage audit for published wallets absent from the ranking"
```

---

### Task 9: Report rendering (pure)

**Files:**
- Modify: `scripts/insider_episode_lib.py`
- Test: `tests/scripts/test_insider_episode_lib.py`

**Interfaces:**
- Consumes: `ScoredWallet`, `BlindResult`, `MarketRow`, `CoverageRow`, `PublicCase`, `HourBucket`.
- Produces:
  - `insider_episode_lib.MarketResult(ref: MarketRef, anchor_kind: str, anchor_ts: int, ranked: list[ScoredWallet], n_window_trades: int, window_volume: float)`
  - `insider_episode_lib.ControlResult(market: MarketRow, anchor_ts: int | None, ranked: list[ScoredWallet])`
  - `insider_episode_lib.CaseReport(case: PublicCase, results: list[MarketResult], controls: list[ControlResult], blind: list[BlindResult], auto_anchor_ts: int | None)`
  - `insider_episode_lib.fingerprint_table(primary: list[ScoredWallet], controls: list[ControlResult]) -> list[tuple[str, float, float, float]]` — rows `(feature, case_top10_mean, control_top10_mean, cohen_d)` for features `age`, `log10_n_w`, `breadth`, `concentration`, `flip`; top-10 of each ranking; `cohen_d` is 0.0 when a pooled std is zero; empty list when there are no controls
  - `insider_episode_lib.render_case_report(r: CaseReport) -> str`
  - `insider_episode_lib.render_summary(reports: list[CaseReport], coverage: list[CoverageRow]) -> str`
  - `insider_episode_lib.csv_rows(reports: list[CaseReport]) -> list[dict[str, object]]` — one dict per scored wallet across cases and controls with keys `case_id, market_slug, is_control, anchor_kind, anchor_ts, rank, wallet, score, age_days, n_w, n_buys, first_buy_ts, last_buy_ts, breadth, n_life, concentration, flip, first_activity_ts`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_insider_episode_lib.py`:

```python
from scripts.insider_episode_lib import (  # noqa: E402
    BlindResult,
    CaseReport,
    ControlResult,
    CoverageRow,
    MarketResult,
    csv_rows,
    fingerprint_table,
    render_case_report,
    render_summary,
)


def _ranked(prefix: str, n: int, *, n_w: float = 1_000.0) -> list:
    return rank_wallets([_f(f"0x{prefix}{i:02d}", n_w=n_w - i) for i in range(n)], {})


def _report(*, with_controls: bool, holdout: bool) -> CaseReport:
    case = PublicCase(
        id="case-x", name="Case X", split="holdout" if holdout else "dev", confidence="alleged",
        analyzable=True, news_ts=1_700_000_000, news_precision="minute",
        markets=(MarketRef("slug-x", "YES", "primary", "cidx"),),
    )
    primary = MarketResult(case.markets[0], "news", 1_700_000_000, _ranked("p", 12), 40, 12_345.0)
    controls = (
        [ControlResult(_m("ctl1"), 999, _ranked("c", 12, n_w=100.0))] if with_controls else []
    )
    blind = [BlindResult("case-x", "news", 12, {"0xp03": 4}, 4, True)] if holdout else []
    return CaseReport(case, [primary], controls, blind, auto_anchor_ts=1_700_000_000 - 7_200)


def test_fingerprint_table_empty_without_controls() -> None:
    r = _report(with_controls=False, holdout=False)
    assert fingerprint_table(r.results[0].ranked, r.controls) == []


def test_fingerprint_table_has_five_features_with_controls() -> None:
    r = _report(with_controls=True, holdout=False)
    rows = fingerprint_table(r.results[0].ranked, r.controls)
    assert [x[0] for x in rows] == ["age", "log10_n_w", "breadth", "concentration", "flip"]
    log_row = rows[1]
    assert log_row[1] > log_row[2]  # case top-10 has larger notional than control top-10


def test_render_case_report_mentions_no_controls() -> None:
    text = render_case_report(_report(with_controls=False, holdout=False))
    assert "# Case X" in text
    assert "no controls" in text.lower()
    assert "Blind test" not in text  # dev case


def test_render_case_report_holdout_includes_blind_result() -> None:
    text = render_case_report(_report(with_controls=True, holdout=True))
    assert "Blind test" in text
    assert "0xp03" in text and "4 / 12" in text
    assert "PASS" in text


def test_render_summary_and_csv_rows() -> None:
    reports = [_report(with_controls=True, holdout=True), _report(with_controls=False, holdout=False)]
    cov = [CoverageRow("case-x", "0xmissing", "outside_window", "first -90.0h")]
    text = render_summary(reports, cov)
    assert "Median best rank" in text
    assert "0xmissing" in text and "outside_window" in text
    rows = csv_rows(reports)
    assert len(rows) == 12 + 12 + 12  # primary + control (holdout report) + primary (dev report)
    assert {r["is_control"] for r in rows} == {False, True}
    assert set(rows[0]) >= {"case_id", "market_slug", "rank", "wallet", "score", "age_days", "flip"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py -q`
Expected: FAIL with `ImportError: cannot import name 'BlindResult'` is already present from Task 6, so expect `cannot import name 'CaseReport'`.

- [ ] **Step 3: Implement**

Append to `scripts/insider_episode_lib.py` (add `import statistics` and `import datetime as dt` at the top):

```python
@dataclass(frozen=True, slots=True)
class MarketResult:
    """Ranking on one case market under one anchor."""

    ref: MarketRef
    anchor_kind: str
    anchor_ts: int
    ranked: list[ScoredWallet]
    n_window_trades: int
    window_volume: float


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Ranking on one control market under its auto anchor (None if no anchor was found)."""

    market: MarketRow
    anchor_ts: int | None
    ranked: list[ScoredWallet]


@dataclass(frozen=True, slots=True)
class CaseReport:
    """Everything the reporter needs for one case."""

    case: PublicCase
    results: list[MarketResult]
    controls: list[ControlResult]
    blind: list[BlindResult]
    auto_anchor_ts: int | None


_FEATURES = ("age", "log10_n_w", "breadth", "concentration", "flip")
_TOP_N = 10


def _feature_values(ranked: list[ScoredWallet], name: str) -> list[float]:
    top = ranked[:_TOP_N]
    if name == "age":
        return [r.age for r in top]
    if name == "log10_n_w":
        return [math.log10(1.0 + r.features.n_w) for r in top]
    if name == "breadth":
        return [float(r.features.breadth) for r in top]
    if name == "concentration":
        return [min(1.0, r.features.n_w / r.features.n_life) if r.features.n_life else 1.0 for r in top]
    return [1.0 if r.features.flip else 0.0 for r in top]


def _cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / pooled


def fingerprint_table(
    primary: list[ScoredWallet], controls: list[ControlResult]
) -> list[tuple[str, float, float, float]]:
    """Top-10 feature means for the case versus pooled control top-10s, with Cohen's d."""
    if not controls or not primary:
        return []
    rows: list[tuple[str, float, float, float]] = []
    for name in _FEATURES:
        case_vals = _feature_values(primary, name)
        ctl_vals = [v for c in controls for v in _feature_values(c.ranked, name)]
        if not case_vals or not ctl_vals:
            continue
        rows.append((name, statistics.mean(case_vals), statistics.mean(ctl_vals), _cohen_d(case_vals, ctl_vals)))
    return rows


def _iso(ts: int | None) -> str:
    if ts is None:
        return "n/a"
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def _ranked_table(ranked: list[ScoredWallet], anchor: int, limit: int = 25) -> list[str]:
    lines = [
        "| Rank | Wallet | Score | Age (d) | Window $ | Buys | First buy | Last buy | Markets | Conc | Flip |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|",
    ]
    for r in ranked[:limit]:
        f = r.features
        conc = min(1.0, f.n_w / f.n_life) if f.n_life else 1.0
        lines.append(
            f"| {r.rank} | `{f.wallet}` | {r.score:.3f} | {r.age:.1f} | {f.n_w:,.0f} | {f.n_buys} |"
            f" {(f.first_buy_ts - anchor) / 3600:+.1f}h | {(f.last_buy_ts - anchor) / 3600:+.1f}h |"
            f" {f.breadth} | {conc:.2f} | {'yes' if f.flip else ''} |"
        )
    return lines


def render_case_report(r: CaseReport) -> str:
    """Markdown report for one case."""
    c = r.case
    out = [f"# {c.name}", "", f"Case `{c.id}` · {c.confidence} · split {c.split}", ""]
    out += [f"News anchor: {_iso(c.news_ts)} (precision: {c.news_precision})"]
    gap = "n/a" if r.auto_anchor_ts is None else f"{(r.auto_anchor_ts - c.news_ts) / 3600:+.1f}h"
    out += [f"Auto anchor: {_iso(r.auto_anchor_ts)} (gap to news: {gap})", ""]
    for res in r.results:
        out += [f"## {res.ref.slug} ({res.ref.role}, {res.anchor_kind} anchor)", ""]
        out += [f"{res.n_window_trades} window trades, ${res.window_volume:,.0f} volume, {len(res.ranked)} candidates", ""]
        out += _ranked_table(res.ranked, res.anchor_ts) + [""]
    out += ["## Controls", ""]
    if not r.controls:
        out += ["No controls matched; fingerprint unavailable.", ""]
    else:
        for ctl in r.controls:
            out += [f"- `{ctl.market.slug}` · anchor {_iso(ctl.anchor_ts)} · {len(ctl.ranked)} candidates"]
        primary = next((x.ranked for x in r.results if x.ref.role.startswith("primary") and x.anchor_kind == "news"), [])
        out += ["", "| Feature | Case top-10 mean | Control top-10 mean | Cohen's d |", "|---|---:|---:|---:|"]
        for name, a, b, d in fingerprint_table(primary, r.controls):
            out += [f"| {name} | {a:.3f} | {b:.3f} | {d:+.2f} |"]
        out += [""]
    if r.blind:
        out += ["## Blind test", ""]
        for b in r.blind:
            verdict = "PASS" if b.passed else ("FAIL" if b.passed is False else "no published wallet among candidates")
            out += [f"- {b.anchor_kind} anchor: {verdict}; {b.n_candidates} candidates"]
            for w, rank in b.ranks.items():
                out += [f"  - `{w}`: {'not a candidate' if rank is None else f'{rank} / {b.n_candidates}'}"]
        out += [""]
    return "\n".join(out)


def render_summary(reports: list[CaseReport], coverage: list[CoverageRow]) -> str:
    """Cross-case summary: blind-test medians, anchor gaps, and the coverage audit."""
    out = ["# Insider-episode catalogue: summary", ""]
    for kind in ("news", "auto"):
        best = [b.best_rank for r in reports for b in r.blind if b.anchor_kind == kind and b.best_rank is not None]
        med = f"{statistics.median(best):.0f}" if best else "n/a"
        out += [f"Median best rank ({kind} anchor, {len(best)} testable holdout cases): {med}"]
    gaps = [(r.auto_anchor_ts - r.case.news_ts) / 3600 for r in reports if r.auto_anchor_ts is not None]
    out += ["", f"Auto-anchor gap to news, hours (n={len(gaps)}): "
            + (", ".join(f"{g:+.1f}" for g in sorted(gaps)) if gaps else "n/a"), ""]
    out += ["## Cases", "", "| Case | Split | Tier | Candidates (primary) | Blind (news) | Blind (auto) |", "|---|---|---|---:|---|---|"]
    for r in reports:
        prim = next((x for x in r.results if x.ref.role.startswith("primary") and x.anchor_kind == "news"), None)
        n = len(prim.ranked) if prim else 0
        bn = next((b for b in r.blind if b.anchor_kind == "news"), None)
        ba = next((b for b in r.blind if b.anchor_kind == "auto"), None)
        fmt = lambda b: "—" if b is None else ("n/a" if b.best_rank is None else f"{b.best_rank}")  # noqa: E731
        out += [f"| {r.case.id} | {r.case.split} | {r.case.confidence} | {n} | {fmt(bn)} | {fmt(ba)} |"]
    out += ["", "## Published wallets not in corpus", "", "| Case | Wallet | Reason | Detail |", "|---|---|---|---|"]
    for cv in coverage:
        out += [f"| {cv.case_id} | `{cv.wallet}` | {cv.reason} | {cv.detail} |"]
    return "\n".join(out) + "\n"


def csv_rows(reports: list[CaseReport]) -> list[dict[str, object]]:
    """Flat rows for every scored wallet across cases and controls."""
    rows: list[dict[str, object]] = []

    def _emit(case_id: str, slug: str, is_control: bool, kind: str, anchor: int | None, ranked: list[ScoredWallet]) -> None:
        for r in ranked:
            f = r.features
            rows.append({
                "case_id": case_id, "market_slug": slug, "is_control": is_control, "anchor_kind": kind,
                "anchor_ts": anchor, "rank": r.rank, "wallet": f.wallet, "score": round(r.score, 6),
                "age_days": round(r.age, 3), "n_w": f.n_w, "n_buys": f.n_buys,
                "first_buy_ts": f.first_buy_ts, "last_buy_ts": f.last_buy_ts, "breadth": f.breadth,
                "n_life": f.n_life, "concentration": min(1.0, f.n_w / f.n_life) if f.n_life else 1.0,
                "flip": f.flip,
                "first_activity_ts": None if r.provenance is None else r.provenance.first_activity_ts,
            })

    for rep in reports:
        for res in rep.results:
            _emit(rep.case.id, res.ref.slug, False, res.anchor_kind, res.anchor_ts, res.ranked)
        for ctl in rep.controls:
            _emit(rep.case.id, ctl.market.slug, True, "auto", ctl.anchor_ts, ctl.ranked)
    return rows
```

If `ruff` flags `_emit` for more than five positional parameters, convert its last four to keyword-only (`*, is_control, kind, anchor, ranked`) and update the two call sites accordingly.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_insider_episode_lib.py -q`
Expected: all pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check scripts tests && uv run ruff format scripts tests && uv run ty check
git add scripts/insider_episode_lib.py tests/scripts/test_insider_episode_lib.py
git commit -m "feat(insider): case and summary report rendering, fingerprint table, csv rows"
```

---

### Task 10: CLI orchestration and subprocess smoke

**Files:**
- Modify: `scripts/insider_episodes.py` (add `run_case`, `run_controls`, `_parse_args`, `main`)
- Test: `tests/scripts/test_insider_episodes.py`
- Modify: `docs/claude/operator-scripts.md` (one bullet)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `insider_episodes.run_market(conn, ref: MarketRef, anchor_kind: str, anchor: int, window_hours: int, prov: dict[str, Provenance]) -> MarketResult`
  - `insider_episodes.collect_candidates(conn, cases: list[PublicCase], window_hours: int, universe: list[MarketRow], controls_per_case: int) -> set[str]` — every wallet that will need provenance (cases under both anchors, plus controls), so one enrichment pass precedes scoring
  - `insider_episodes.run_case(conn, case, sealed: SealedCase | None, *, window_hours, universe, excluded, controls_per_case, prov) -> CaseReport`
  - `insider_episodes.main(argv: list[str] | None = None) -> int`

Flow in `main`: parse args → `load_registry` → `select_cases` (catch `HoldoutLockedError`, print, return 2) → `open_corpus` → resolve each analyzable case's markets (`LookupError` → mark the market unresolved and skip it; a case with no resolved primary becomes a coverage row for each published wallet with `market_below_volume_floor`) → `load_market_rows` → compute each case's auto anchor from `market_trades(primary)` and `hourly_buckets`/`auto_anchor` → `collect_candidates` → provenance (`--skip-provenance` uses cache only; otherwise `DataClient(rpm=args.provenance_rpm)` with `fetch_provenance_api`, closed in `finally`) → `run_case` per case → coverage audit for every published wallet of every selected case (`diagnose_missing`, plus the `below_notional_floor` upgrade when the registry wallet has `stake_usd < 10`; read `stake_usd` straight from the YAML in a small helper `_registry_stakes(path) -> dict[str, dict[str, float]]`) → write `<out_dir>/<case-id>.md`, `<out_dir>/SUMMARY.md`, `<out_dir>/scored_wallets.csv` → print a one-line summary per case → return 0.

`run_case` builds: news-anchor `MarketResult` for every resolved market of the case (primary first), an auto-anchor `MarketResult` for the primary only (when an auto anchor exists), `ControlResult`s for `match_controls(primary_row, universe, excluded, k=controls_per_case)` each under its own auto anchor (skip a control whose auto anchor is `None`), and for holdout cases with a sealed record, `compare_published` for the primary under each anchor kind. `excluded` is the set of every registry market condition id across all splits.

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/scripts/test_insider_episodes.py`:

```python
import subprocess  # noqa: E402
import sys  # noqa: E402

from scripts.insider_episodes import main  # noqa: E402

_E2E_REGISTRY = """
registry_version: 1
cases:
  - id: dev-case
    name: Dev Case
    confidence: alleged
    split: dev
    analyzable: true
    news_ts_utc: "1970-01-12T13:46:40Z"   # == 1_000_000
    news_ts_precision: minute
    corpus_markets:
      - market_slug: slug-m1
        winning_side: "YES"
        role: primary
    wallets:
      - address: "0x00000000000000000000000000000000000000aa"
        stake_usd: 5
  - id: hold-case
    name: Hold Case
    confidence: confirmed
    split: holdout
    analyzable: true
    news_ts_utc: "1970-01-12T13:46:40Z"
    news_ts_precision: minute
    corpus_markets:
      - market_slug: slug-m2
        winning_side: "YES"
        role: primary
    wallets:
      - address: "0x00000000000000000000000000000000000000bb"
"""


def _e2e_db(corpus_factory) -> Path:
    bb = "0x00000000000000000000000000000000000000bb"
    trades = [
        # m1: a crowd; published 0x..aa never trades m1 (coverage: no_trades + stake<10 -> below_notional_floor)
        ("w1", "m1", "YES", "BUY", 0.10, 500.0, ANCHOR - 10 * H),
        ("w2", "m1", "YES", "BUY", 0.10, 100.0, ANCHOR - 9 * H),
        ("w3", "m1", "YES", "BUY", 0.12, 50.0, ANCHOR - 8 * H),
        # m2: published 0x..bb is the biggest fresh buyer -> rank 1
        (bb, "m2", "YES", "BUY", 0.10, 9_000.0, ANCHOR - 20 * H),
        ("mm", "m2", "YES", "BUY", 0.10, 9_500.0, ANCHOR - 30 * H),
        ("mm", "ctl", "YES", "BUY", 0.50, 10.0, ANCHOR - 400 * 24 * H),  # old, broad wallet
        ("z", "m2", "YES", "BUY", 0.10, 20.0, ANCHOR - 1 * H),
        # ctl: a control market in the same category/volume band/era with a price jump
        ("c1", "ctl", "YES", "BUY", 0.10, 100.0, ANCHOR - 50 * H),
        ("c2", "ctl", "YES", "BUY", 0.40, 100.0, ANCHOR - 40 * H),
        ("c3", "ctl", "YES", "BUY", 0.45, 100.0, ANCHOR - 30 * H),
    ]
    resolutions = [("m1", 1, ANCHOR + H), ("m2", 1, ANCHOR + H), ("ctl", 1, ANCHOR + 2 * H)]
    markets = [
        ("m1", "slug-m1", "geopolitics", 5e6, ANCHOR + H, 0),
        ("m2", "slug-m2", "geopolitics", 5e6, ANCHOR + H, 0),
        ("ctl", "slug-ctl", "geopolitics", 6e6, ANCHOR + 2 * H, 0),
    ]
    return corpus_factory(trades, resolutions, markets=markets)


def test_main_dev_run_writes_reports_and_coverage(tmp_path: Path, corpus_factory) -> None:
    db = _e2e_db(corpus_factory)
    reg = tmp_path / "reg.yaml"
    reg.write_text(_E2E_REGISTRY)
    out = tmp_path / "out"
    rc = main([
        "--db", str(db), "--registry", str(reg), "--out-dir", str(out),
        "--provenance-cache", str(tmp_path / "prov.sqlite3"), "--skip-provenance",
    ])
    assert rc == 0
    assert (out / "dev-case.md").exists()
    assert not (out / "hold-case.md").exists()
    summary = (out / "SUMMARY.md").read_text()
    assert "below_notional_floor" in summary
    assert (out / "scored_wallets.csv").read_text().count("\n") >= 4  # header + 3 m1 candidates


def test_main_refuses_holdout_without_frozen_flag(tmp_path: Path, corpus_factory) -> None:
    db = _e2e_db(corpus_factory)
    reg = tmp_path / "reg.yaml"
    reg.write_text(_E2E_REGISTRY)
    rc = main(["--db", str(db), "--registry", str(reg), "--out-dir", str(tmp_path / "o"),
               "--provenance-cache", str(tmp_path / "p.sqlite3"), "--skip-provenance", "--split", "holdout"])
    assert rc == 2


def test_main_holdout_blind_test_passes_on_fixture(tmp_path: Path, corpus_factory) -> None:
    db = _e2e_db(corpus_factory)
    reg = tmp_path / "reg.yaml"
    reg.write_text(_E2E_REGISTRY)
    out = tmp_path / "out"
    rc = main(["--db", str(db), "--registry", str(reg), "--out-dir", str(out),
               "--provenance-cache", str(tmp_path / "p.sqlite3"), "--skip-provenance",
               "--split", "holdout", "--scorer-frozen"])
    assert rc == 0
    text = (out / "hold-case.md").read_text()
    assert "Blind test" in text and "PASS" in text
    assert "slug-ctl" in text  # control matched


def test_script_runs_as_subprocess(tmp_path: Path, corpus_factory) -> None:
    db = _e2e_db(corpus_factory)
    reg = tmp_path / "reg.yaml"
    reg.write_text(_E2E_REGISTRY)
    proc = subprocess.run(
        [sys.executable, "scripts/insider_episodes.py", "--db", str(db), "--registry", str(reg),
         "--out-dir", str(tmp_path / "o"), "--provenance-cache", str(tmp_path / "p.sqlite3"),
         "--skip-provenance"],
        capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_insider_episodes.py -q`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement the orchestration**

Append to `scripts/insider_episodes.py` (add imports: `import argparse`, `import asyncio`, `import csv`; extend lib import with `BlindResult, CaseReport, ControlResult, HoldoutLockedError, MarketResult, ScoredWallet, auto_anchor, compare_published, csv_rows, hourly_buckets, match_controls, rank_wallets, render_case_report, render_summary, select_cases`):

```python
def run_market(
    conn: sqlite3.Connection,
    ref: MarketRef,
    anchor_kind: str,
    anchor: int,
    window_hours: int,
    prov: dict[str, Provenance],
) -> MarketResult:
    """Extract, featurize, and rank one market under one anchor."""
    assert ref.condition_id is not None  # noqa: S101  (resolved upstream)
    trades = extract_window(conn, ref.condition_id, anchor, window_hours)
    wallets = candidate_wallets(trades, ref.winning_side)
    feats = build_features(conn, trades, wallets, ref.winning_side, anchor)
    return MarketResult(
        ref=ref,
        anchor_kind=anchor_kind,
        anchor_ts=anchor,
        ranked=rank_wallets(feats, prov),
        n_window_trades=len(trades),
        window_volume=sum(t.notional for t in trades),
    )


def _auto_anchor_for(conn: sqlite3.Connection, condition_id: str, side: str, resolved_at: int) -> int | None:
    return auto_anchor(hourly_buckets(market_trades(conn, condition_id), side), resolved_at)


def _controls_for(
    primary_row: MarketRow, universe: list[MarketRow], excluded: set[str], k: int
) -> list[MarketRow]:
    return match_controls(primary_row, universe, excluded, k=k)


def collect_candidates(
    conn: sqlite3.Connection,
    cases: list[PublicCase],
    window_hours: int,
    universe: list[MarketRow],
    *,
    excluded: set[str],
    controls_per_case: int,
    auto_anchors: dict[str, int | None],
) -> set[str]:
    """Every wallet that scoring will touch, so provenance is fetched once up front."""
    rows = {m.condition_id: m for m in universe}
    out: set[str] = set()
    for case in cases:
        for ref in case.markets:
            if ref.condition_id is None:
                continue
            out.update(candidate_wallets(extract_window(conn, ref.condition_id, case.news_ts, window_hours), ref.winning_side))
        primary = case.primary
        if primary is None or primary.condition_id is None:
            continue
        aa = auto_anchors.get(case.id)
        if aa is not None:
            out.update(candidate_wallets(extract_window(conn, primary.condition_id, aa, window_hours), primary.winning_side))
        prow = rows.get(primary.condition_id)
        if prow is None:
            continue
        for ctl in _controls_for(prow, universe, excluded, controls_per_case):
            ctl_anchor = _auto_anchor_for(conn, ctl.condition_id, "YES" if ctl.yes_won else "NO", ctl.resolved_at)
            if ctl_anchor is not None:
                out.update(candidate_wallets(extract_window(conn, ctl.condition_id, ctl_anchor, window_hours), "YES" if ctl.yes_won else "NO"))
    return out


def run_case(
    conn: sqlite3.Connection,
    case: PublicCase,
    sealed: SealedCase | None,
    *,
    window_hours: int,
    universe: list[MarketRow],
    excluded: set[str],
    controls_per_case: int,
    prov: dict[str, Provenance],
    auto_anchor_ts: int | None,
) -> CaseReport:
    """Rank every resolved market of ``case``, its controls, and run the blind test if holdout."""
    rows = {m.condition_id: m for m in universe}
    results = [
        run_market(conn, ref, "news", case.news_ts, window_hours, prov)
        for ref in case.markets
        if ref.condition_id is not None
    ]
    controls: list[ControlResult] = []
    blind: list[BlindResult] = []
    primary = case.primary
    if primary is not None and primary.condition_id is not None:
        if auto_anchor_ts is not None:
            results.append(run_market(conn, primary, "auto", auto_anchor_ts, window_hours, prov))
        prow = rows.get(primary.condition_id)
        if prow is not None:
            for ctl in _controls_for(prow, universe, excluded, controls_per_case):
                side = "YES" if ctl.yes_won else "NO"
                ctl_anchor = _auto_anchor_for(conn, ctl.condition_id, side, ctl.resolved_at)
                if ctl_anchor is None:
                    continue
                ref = MarketRef(slug=ctl.slug, winning_side=side, role="control", condition_id=ctl.condition_id)
                controls.append(ControlResult(ctl, ctl_anchor, run_market(conn, ref, "auto", ctl_anchor, window_hours, prov).ranked))
        if case.split == "holdout" and sealed is not None and sealed.wallets:
            for res in results:
                if res.ref.condition_id == primary.condition_id:
                    blind.append(compare_published(case.id, res.anchor_kind, res.ranked, sealed))
    return CaseReport(case=case, results=results, controls=controls, blind=blind, auto_anchor_ts=auto_anchor_ts)


def _registry_stakes(path: Path) -> dict[str, dict[str, float]]:
    """``{case_id: {wallet: stake_usd}}`` for wallets whose stake is recorded in the registry."""
    doc = yaml.safe_load(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for raw in doc["cases"]:
        for w in raw.get("wallets") or []:
            addr, stake = w.get("address"), w.get("stake_usd")
            if isinstance(addr, str) and isinstance(stake, (int, float)):
                out.setdefault(str(raw["id"]), {})[addr.strip().lower()] = float(stake)
    return out


def _resolve_case(conn: sqlite3.Connection, case: PublicCase) -> PublicCase:
    resolved: list[MarketRef] = []
    for ref in case.markets:
        try:
            resolved.append(resolve_market(conn, ref))
        except LookupError as exc:
            print(f"[{case.id}] {exc}; skipping market")
    return PublicCase(
        id=case.id, name=case.name, split=case.split, confidence=case.confidence,
        analyzable=case.analyzable, news_ts=case.news_ts, news_precision=case.news_precision,
        markets=tuple(resolved),
    )


def _coverage(
    conn: sqlite3.Connection,
    cases: list[PublicCase],
    sealed: dict[str, SealedCase],
    stakes: dict[str, dict[str, float]],
    *,
    window_hours: int,
    rows: dict[str, MarketRow],
) -> list[CoverageRow]:
    out: list[CoverageRow] = []
    for case in cases:
        primary = case.primary
        prow = rows.get(primary.condition_id) if primary and primary.condition_id else None
        for w in sealed.get(case.id, SealedCase(case.id, ())).wallets:
            row = diagnose_missing(conn, case, w, case.news_ts, window_hours, prow)
            if row is None:
                continue
            if row.reason == "no_trades_in_market" and stakes.get(case.id, {}).get(w, 1e9) < 10:
                row = CoverageRow(row.case_id, row.wallet, "below_notional_floor", "registry stake under the $10 corpus floor (inferred)")
            out.append(row)
    return out


async def _provenance(args: argparse.Namespace, wallets: list[str]) -> dict[str, Provenance]:
    cache = ProvenanceCache(Path(args.provenance_cache))
    try:
        if args.skip_provenance:
            return {w: p for w in wallets if (p := cache.get(w)) is not None}
        client = DataClient(rpm=args.provenance_rpm)
        try:
            return await enrich_provenance(cache, wallets, lambda w: fetch_provenance_api(client, w))
        finally:
            await client.aclose()
    finally:
        cache.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/corpus.sqlite3")
    p.add_argument("--registry", default="docs/research/insider-cases.yaml")
    p.add_argument("--out-dir", default="docs/research/insider-episodes")
    p.add_argument("--window-hours", type=int, default=72)
    p.add_argument("--controls-per-case", type=int, default=5)
    p.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    p.add_argument("--scorer-frozen", action="store_true", help="attest spec Amendment 1 is signed off")
    p.add_argument("--skip-provenance", action="store_true", help="use cached provenance only")
    p.add_argument("--provenance-cache", default="data/insider_provenance.sqlite3")
    p.add_argument("--provenance-rpm", type=int, default=60)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    public, sealed = load_registry(Path(args.registry))
    try:
        selected = select_cases(public, args.split, scorer_frozen=args.scorer_frozen)
    except HoldoutLockedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    conn = open_corpus(Path(args.db))
    try:
        cases = [_resolve_case(conn, c) for c in selected if c.analyzable]
        universe = load_market_rows(conn)
        rows = {m.condition_id: m for m in universe}
        excluded = {m.condition_id for c in (_resolve_case(conn, x) for x in public if x.analyzable) for m in c.markets if m.condition_id}
        auto_anchors = {
            c.id: _auto_anchor_for(conn, c.primary.condition_id, c.primary.winning_side, rows[c.primary.condition_id].resolved_at)
            for c in cases
            if c.primary and c.primary.condition_id and c.primary.condition_id in rows
        }
        wanted = sorted(collect_candidates(conn, cases, args.window_hours, universe, excluded=excluded,
                                           controls_per_case=args.controls_per_case, auto_anchors=auto_anchors))
        print(f"{len(cases)} cases, {len(wanted)} wallets need provenance")
        prov = asyncio.run(_provenance(args, wanted))
        reports = [
            run_case(conn, c, sealed.get(c.id), window_hours=args.window_hours, universe=universe, excluded=excluded,
                     controls_per_case=args.controls_per_case, prov=prov, auto_anchor_ts=auto_anchors.get(c.id))
            for c in cases
        ]
        coverage = _coverage(conn, [_resolve_case(conn, c) for c in selected], sealed, _registry_stakes(Path(args.registry)),
                             window_hours=args.window_hours, rows=rows)
    finally:
        conn.close()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in reports:
        (out_dir / f"{r.case.id}.md").write_text(render_case_report(r))
        prim = next((x for x in r.results if x.ref.role.startswith("primary") and x.anchor_kind == "news"), None)
        print(f"[{r.case.id}] {len(prim.ranked) if prim else 0} candidates, {len(r.controls)} controls, blind={[b.best_rank for b in r.blind]}")
    (out_dir / "SUMMARY.md").write_text(render_summary(reports, coverage))
    rows_csv = csv_rows(reports)
    with (out_dir / "scored_wallets.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0]) if rows_csv else ["case_id"])
        writer.writeheader()
        writer.writerows(rows_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`main` as written will exceed the complexity limit; split the body between `main` and two helpers, `_prepare(conn, public, selected, args) -> tuple[list[PublicCase], list[MarketRow], set[str], dict[str, int | None]]` and `_write_outputs(out_dir, reports, coverage) -> None`, keeping behaviour identical. Run `uv run ruff check scripts/insider_episodes.py` and keep extracting until C901 is clean.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/ -q`
Expected: all pass. The subprocess test proves the `sys.path` bootstrap works outside pytest.

- [ ] **Step 5: Document the script**

In `docs/claude/operator-scripts.md`, add a bullet after the `find_insider_wallets.py` entry:

```markdown
- **`insider_episodes.py`** — market-first insider catalogue (spec `2026-10-11-insider-episode-catalogue-design.md`). Reads `docs/research/insider-cases.yaml`, ranks pre-news winning-side buyers on each case market with the frozen scorer (freshness × log notional × concentration ÷ breadth + flip), matches control markets, and blind-tests published wallets on holdout cases. `--split holdout` refuses to run without `--scorer-frozen`. Provenance (data-api first activity) is cached in `data/insider_provenance.sqlite3`; `--skip-provenance` runs offline from the cache. Plain `sqlite3` read-only, indexed lookups only; a full dev run on the desktop corpus is minutes, dominated by data-api paging at 60 rpm.
```

- [ ] **Step 6: Lint, type-check, full test run, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q
git add scripts/insider_episodes.py tests/scripts/test_insider_episodes.py docs/claude/operator-scripts.md
git commit -m "feat(insider): insider_episodes CLI with controls, coverage audit, and reports"
```

---

### Task 11: Dev run on the desktop corpus

**Files:**
- Create: `docs/research/insider-episodes/<dev-case-id>.md` for each dev case (script output)
- Create: `docs/research/insider-episodes/SUMMARY.md`, `docs/research/insider-episodes/scored_wallets.csv`

No code. The script runs where the corpus is (see `LOCAL_NOTES.md`; LAN path `ssh -p 2222 macph@10.0.0.143`, project at `~/projects/polymarketscanner/pscanner`). The data-api is reachable from the desktop.

- [ ] **Step 1: Sync the branch to the desktop**

```bash
git push -u origin feat/insider-episode-catalogue
ssh -p 2222 macph@10.0.0.143 'cd ~/projects/polymarketscanner/pscanner && git fetch origin && git checkout feat/insider-episode-catalogue && git pull --ff-only && export PATH="$HOME/.local/bin:$PATH" && uv sync'
```

- [ ] **Step 2: Run the dev split with provenance**

```bash
ssh -p 2222 macph@10.0.0.143 'cd ~/projects/polymarketscanner/pscanner && export PATH="$HOME/.local/bin:$PATH" && nohup uv run python scripts/insider_episodes.py --split dev > /tmp/insider_dev.log 2>&1 < /dev/null & disown'
ssh -p 2222 macph@10.0.0.143 'tail -f /tmp/insider_dev.log'
```

Expected: a `N cases, M wallets need provenance` line, per-wallet provenance progress for misses, then one `[case-id] ... candidates` line per dev case and exit 0. Provenance at 60 rpm for a few thousand wallets takes 30 to 90 minutes on the first run; later runs hit the cache.

- [ ] **Step 3: Pull the reports back and sanity-check them against the profile**

```bash
rsync -avh -e "ssh -p 2222" macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/docs/research/insider-episodes/ docs/research/insider-episodes/
```

Open `israel-strikes-iran-2025-06.md` and `us-israel-strikes-iran-2026-02-28.md`. The published dev wallets are not hidden on dev cases, so check by eye: `0x0afc7ce5` should be in the top 5 on `israel-military-action-against-iran-before-july`; `0x1caa6a7a` and `0x45647817` in the top 10 on the Feb 28 primary. If they are not, the implementation disagrees with the probe and the discrepancy must be found before the holdout run; do not touch the scorer constants.

- [ ] **Step 4: Commit the dev reports**

```bash
git add docs/research/insider-episodes/
git commit -m "docs(insider): dev-split catalogue run"
```

---

### Task 12: Holdout run (the blind test)

Run exactly once with the frozen scorer. If anything about the pipeline has to change after this run, the result is recorded as-is and a rerun is a new amendment.

- [ ] **Step 1: Confirm the scorer file is unchanged since Amendment 1**

```bash
git log --oneline -- scripts/insider_episode_lib.py
git diff f2657f5 -- docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
```

Expected: no edits to the amendment section.

- [ ] **Step 2: Run all cases with the frozen flag on the desktop**

```bash
ssh -p 2222 macph@10.0.0.143 'cd ~/projects/polymarketscanner/pscanner && git pull --ff-only && export PATH="$HOME/.local/bin:$PATH" && nohup uv run python scripts/insider_episodes.py --split all --scorer-frozen > /tmp/insider_all.log 2>&1 < /dev/null & disown'
```

- [ ] **Step 3: Pull reports, read `SUMMARY.md`, commit**

```bash
rsync -avh -e "ssh -p 2222" macph@10.0.0.143:/home/macph/projects/polymarketscanner/pscanner/docs/research/insider-episodes/ docs/research/insider-episodes/
git add docs/research/insider-episodes/
git commit -m "docs(insider): holdout blind test and full catalogue run"
```

Report to the operator: the Maduro blind result under both anchors (rank and candidate count), the median best rank line, the anchor-gap list, and the not-in-corpus table, verbatim from `SUMMARY.md`. State plainly whether the single testable holdout case passed.

---

## Self-review notes

- Spec coverage: registry (T1), extractor and two anchors (T2, T3), provenance cache with no partial rows (T5), scorer per Amendment 1 (T6), controls (T7), blind protocol with pass rule and `None` when absent (T6, T10), coverage audit with all five spec reasons plus `losing_side_only` (T8), reporter outputs and CSV (T9, T10), holdout guard (T1, T10), performance notes (T10 docs), sequencing (T11, T12). The spec's `--provenance-rpm` default of 60 and `--window-hours` 72 are kept. The spec's DuckDB line is superseded by the plan header.
- Review Focus items 1 to 5 have tests in T1/T6, T2, T6, T8, and T9 respectively.
- Type consistency: `MarketRef.condition_id` is `str | None` from T1 onward and asserted non-null in `run_market`; `Provenance.first_activity_ts` is `int | None` everywhere; `BlindResult.passed` is `bool | None`.
