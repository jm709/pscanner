"""Subgraph-driven copy-trade collector (#152).

Polls the Polymarket V2 subgraph for ``orderFilledEvents`` involving any
watchlisted wallet and emits a ``subgraph_copy`` :class:`Alert` for every
position-increasing trade. Booking happens downstream via
:class:`pscanner.strategies.evaluators.subgraph_copy.SubgraphCopyEvaluator`.

Lifecycle is owned by the daemon scheduler — restart-on-crash, shared
``stop_event`` for clean shutdown.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Final

import structlog

from pscanner.alerts.models import Alert
from pscanner.alerts.protocol import IAlertSink
from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import SubgraphTradeCollectorConfig
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.gamma import GammaClient
from pscanner.poly.ids import AssetId
from pscanner.poly.subgraph import SubgraphClient
from pscanner.poly.token_resolver import resolve_token
from pscanner.store.repo import (
    MarketCacheRepo,
    SubgraphWatchStateRepo,
)
from pscanner.util.clock import Clock, RealClock

_LOG = structlog.get_logger(__name__)

SUBGRAPH_ID: Final[str] = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
DETECTOR_TAG: Final[str] = "subgraph_copy"
ALERT_KEY_PREFIX: Final[str] = "subgraph"
PAGE_SIZE: Final[int] = 1000

_GRAPHQL_QUERY: Final[str] = f"""
{{
  orderFilledEvents(
    where: $where
    first: {PAGE_SIZE}
    orderBy: timestamp
    orderDirection: asc
  ) {{
    transactionHash
    timestamp
    maker {{ id }}
    taker {{ id }}
    market {{ id }}
    tokenId
    side
    price
    size
  }}
  _meta {{ block {{ number timestamp }} }}
}}
"""


def _compute_copy_direction(maker: str, taker: str, side: int, watchlist: set[str]) -> str:
    """Return ``"BUY"`` iff the watchlist wallet's position increases.

    Subgraph ``side``: 0=BUY, 1=SELL on the order's direction.
    Maker placed the resting order; taker hit it from the opposite side.

    - watchlist == maker AND side == 0 -> maker accumulates -> BUY
    - watchlist == maker AND side == 1 -> maker reduces      -> SKIP
    - watchlist == taker AND side == 0 -> taker sold         -> SKIP
    - watchlist == taker AND side == 1 -> taker bought       -> BUY
    """
    maker_l = maker.lower()
    taker_l = taker.lower()
    if maker_l in watchlist and side == 0:
        return "BUY"
    if taker_l in watchlist and side == 1:
        return "BUY"
    return "SKIP"


def _serialize_where_inline(where: dict[str, Any]) -> str:
    """Render ``where:`` as a GraphQL object literal (NOT JSON).

    GraphQL object literals do not quote keys. We hand-emit a minimal
    serializer to avoid pulling in a full GraphQL client.
    """

    def render(v: Any) -> str:
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ",".join(render(x) for x in v) + "]"
        if isinstance(v, dict):
            inner = ",".join(f"{k}:{render(val)}" for k, val in v.items())
            return "{" + inner + "}"
        raise TypeError(f"unsupported where value: {v!r}")

    return render(where)


def _build_where_clause(addrs: list[str], last_seen_ts: int) -> dict[str, Any]:
    """Build the ``where:`` argument for ``orderFilledEvents``.

    TheGraph rejects ``or`` mixed with same-level column filters, so the
    timestamp predicate must be repeated inside each ``or`` branch.
    ``timestamp_gte`` (not ``_gt``) plus a within-cycle ``tx_hash`` dedupe
    gives strict no-loss boundary behaviour.
    """
    ts_str = str(last_seen_ts)
    return {
        "or": [
            {"timestamp_gte": ts_str, "maker_in": addrs},
            {"timestamp_gte": ts_str, "taker_in": addrs},
        ],
    }


async def _fetch_events_since(
    client: SubgraphClient,
    *,
    addrs: list[str],
    last_seen_ts: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Drain the subgraph for all events newer than ``last_seen_ts``.

    Watermark pagination: each page advances ``ts`` to the most recent event
    seen. Loop terminates when a page returns fewer than ``PAGE_SIZE`` events.
    Within-cycle ``tx_hash`` dedupe catches boundary events re-fetched by
    ``timestamp_gte``.

    Returns the unique events (asc ``ts`` order) and the indexer's
    ``_meta.block.timestamp`` from the last page (used by the caller for lag
    detection).
    """
    events: list[dict[str, Any]] = []
    seen_tx: set[str] = set()
    ts = last_seen_ts
    indexer_ts: int | None = None
    while True:
        where = _build_where_clause(addrs, ts)
        graphql = _GRAPHQL_QUERY.replace("$where", _serialize_where_inline(where))
        data = await client.query(graphql, {})
        page = data.get("orderFilledEvents") or []
        for e in page:
            tx = e["transactionHash"]
            if tx in seen_tx:
                continue
            seen_tx.add(tx)
            events.append(e)
        meta_block = (data.get("_meta") or {}).get("block") or {}
        meta_ts_raw = meta_block.get("timestamp")
        if meta_ts_raw is not None:
            indexer_ts = int(meta_ts_raw)
        if len(page) < PAGE_SIZE:
            break
        ts = max(int(e["timestamp"]) for e in page)
    return events, indexer_ts


async def _wait_or_stop(stop_event: asyncio.Event, seconds: float) -> bool:
    """Wait up to ``seconds`` for ``stop_event``. Return True if it was set."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


class SubgraphTradeCollector:
    """Polls the V2 subgraph for watchlist trades and emits ``subgraph_copy`` alerts."""

    name: str = "subgraph_trades"

    def __init__(
        self,
        *,
        config: SubgraphTradeCollectorConfig,
        subgraph_client: SubgraphClient,
        gamma_client: GammaClient,
        watchlist: WatchlistRegistry,
        asset_index: AssetIndexRepo,
        market_cache: MarketCacheRepo,
        sink: IAlertSink,
        state_repo: SubgraphWatchStateRepo,
        clock: Clock | None = None,
    ) -> None:
        """Wire the collector to its inputs and outputs.

        Args:
            config: Polling cadence + cold-start + lag thresholds.
            subgraph_client: GraphQL client (already RPM-budgeted).
            gamma_client: Used by ``resolve_token`` on local-cache miss.
            watchlist: Active address set (thread-safe registry).
            asset_index: Corpus-DB asset_id -> condition_id mapping (read/write).
            market_cache: Daemon-DB market metadata (read/write).
            sink: Where to emit ``subgraph_copy`` alerts.
            state_repo: Watermark persistence.
            clock: Injectable clock for tests; defaults to :class:`RealClock`.
        """
        self._config = config
        self._subgraph = subgraph_client
        self._gamma = gamma_client
        self._watchlist = watchlist
        self._asset_index = asset_index
        self._market_cache = market_cache
        self._sink = sink
        self._state = state_repo
        self._clock = clock if clock is not None else RealClock()

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run the polling loop until ``stop_event`` is set.

        Per-cycle exceptions are logged and swallowed so a transient upstream
        hiccup does not kill the loop.
        """
        while not stop_event.is_set():
            try:
                await self._poll_once()
            except Exception:
                _LOG.exception("subgraph_trades.cycle_failed")
            if await _wait_or_stop(stop_event, self._config.poll_interval_seconds):
                return

    async def _poll_once(self) -> None:
        """Run a single poll cycle: fetch, emit alerts, persist watermark."""
        addrs = sorted({a.lower() for a in self._watchlist.addresses()})
        if not addrs:
            _LOG.warning("subgraph_trades.empty_watchlist")
            return
        last_seen = self._state.get_last_seen_ts()
        if last_seen is None:
            last_seen = int(time.time()) - int(self._config.cold_start_lookback_seconds)
        _LOG.info(
            "subgraph_trades.poll_start",
            addrs=len(addrs),
            last_seen_ts=last_seen,
        )
        events, indexer_ts = await _fetch_events_since(
            self._subgraph,
            addrs=addrs,
            last_seen_ts=last_seen,
        )
        self._warn_on_lag(indexer_ts)
        new_last_seen = last_seen
        watchlist_set = set(addrs)
        emitted = 0
        for ev in events:
            new_last_seen = max(new_last_seen, int(ev["timestamp"]))
            body = await self._build_alert_body(ev, watchlist_set)
            if body is None:
                continue
            alert = Alert(
                detector="subgraph_copy",
                alert_key=f"{ALERT_KEY_PREFIX}:{ev['transactionHash']}:{body['outcome']}",
                severity="med",
                title=(
                    f"copy {body['source_wallet'][:14]}.. {body['outcome']}"
                    f" @ {float(ev.get('price', 0.0)):.3f}"
                ),
                body=body,
                created_at=int(ev["timestamp"]),
            )
            await self._sink.emit(alert)
            emitted += 1
        self._state.set_last_seen_ts(new_last_seen)
        _LOG.info(
            "subgraph_trades.poll_done",
            events_seen=len(events),
            emitted=emitted,
            new_last_seen_ts=new_last_seen,
        )

    async def _build_alert_body(
        self, ev: dict[str, Any], watchlist: set[str]
    ) -> dict[str, Any] | None:
        """Resolve the event into an alert body, or ``None`` to skip."""
        maker = ev["maker"]["id"]
        taker = ev["taker"]["id"]
        side = int(ev["side"])
        if _compute_copy_direction(maker, taker, side, watchlist) != "BUY":
            return None
        source_wallet = maker if maker.lower() in watchlist else taker
        try:
            resolved = await resolve_token(
                token_id=AssetId(ev["tokenId"]),
                asset_index=self._asset_index,
                market_cache=self._market_cache,
                gamma=self._gamma,
            )
        except Exception:
            _LOG.exception("subgraph_trades.resolve_failed", token_id=ev.get("tokenId"))
            return None
        if resolved is None:
            _LOG.warning("subgraph_trades.token_unresolved", token_id=ev.get("tokenId"))
            return None
        return {
            "source_wallet": source_wallet,
            "tx_hash": ev["transactionHash"],
            "condition_id": str(resolved.condition_id),
            "outcome": resolved.outcome_name,
            "ts": int(ev["timestamp"]),
        }

    def _warn_on_lag(self, indexer_ts: int | None) -> None:
        """Emit warn/error logs when the subgraph indexer lags the head."""
        if indexer_ts is None:
            return
        lag = int(time.time()) - indexer_ts
        if lag >= self._config.indexer_lag_error_seconds:
            _LOG.error("subgraph_trades.indexer_lag", lag_seconds=lag)
        elif lag >= self._config.indexer_lag_warn_seconds:
            _LOG.warning("subgraph_trades.indexer_lag", lag_seconds=lag)
