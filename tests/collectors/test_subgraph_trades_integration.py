"""End-to-end smoke test: poll-cycle -> alert -> paper_trades booking.

Exercises the full ``SubgraphTradeCollector`` -> ``AlertSink`` ->
``PaperTrader`` -> ``SubgraphCopyEvaluator`` -> ``paper_trades`` path against
on-disk SQLite via the production ``init_db`` / ``init_corpus_db`` helpers.
Only the subgraph + gamma + data clients are mocked. Catches
connection-permission regressions like the ``mode=ro`` bug fixed in commit
``0f54e56`` that unit tests with ``MagicMock`` repos cannot see.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pscanner.alerts.sink import AlertSink
from pscanner.collectors import subgraph_trades as subgraph_trades_mod
from pscanner.collectors.subgraph_trades import SubgraphTradeCollector
from pscanner.collectors.watchlist import WatchlistRegistry
from pscanner.config import (
    PaperTradingConfig,
    SubgraphCopyEvaluatorConfig,
    SubgraphTradeCollectorConfig,
)
from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetIndexRepo
from pscanner.poly.ids import AssetId, ConditionId, MarketId
from pscanner.poly.models import Market
from pscanner.poly.token_resolver import ResolvedToken
from pscanner.store.db import init_db
from pscanner.store.repo import (
    AlertsRepo,
    MarketCacheRepo,
    PaperTradesRepo,
    SubgraphWatchStateRepo,
    WatchlistRepo,
)
from pscanner.strategies.evaluators.subgraph_copy import SubgraphCopyEvaluator
from pscanner.strategies.paper_trader import PaperTrader
from pscanner.util.clock import FakeClock


@pytest.mark.asyncio
async def test_full_cycle_books_paper_trade(tmp_path: Path) -> None:
    """Drive the collector once and assert a ``paper_trades`` entry row lands."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        daemon = init_db(tmp_path / "pscanner.sqlite3")
        corpus = init_corpus_db(tmp_path / "corpus.sqlite3")

        # Seed watchlist with one wallet so the collector has a non-empty set.
        WatchlistRepo(daemon).upsert(address="0xaa", source="manual", reason="t")

        # Pre-seed market_cache so PaperTrader._resolve_outcome maps
        # condition_id + outcome -> asset_id without needing gamma backfill.
        market = Market(
            id=MarketId("m1"),
            question="ignored",
            slug="seed-slug",
            condition_id=ConditionId("0xcond"),
            outcomes=["Yes", "No"],
            outcome_prices=[0.5, 0.5],
            clob_token_ids=[AssetId("token-yes"), AssetId("token-no")],
            active=True,
            closed=False,
        )
        MarketCacheRepo(daemon).upsert(market)

        sink = AlertSink(AlertsRepo(daemon))
        paper_trades = PaperTradesRepo(daemon)
        paper_trader = PaperTrader(
            config=PaperTradingConfig(
                enabled=True,
                starting_bankroll_usd=1000.0,
            ),
            evaluators=[
                SubgraphCopyEvaluator(
                    config=SubgraphCopyEvaluatorConfig(enabled=True),
                    watchlist_repo=WatchlistRepo(daemon),
                    paper_trades=paper_trades,
                ),
            ],
            market_cache=MarketCacheRepo(daemon),
            paper_trades=paper_trades,
            market_ticks=MagicMock(latest_for_asset=MagicMock(return_value=None)),
            data_client=MagicMock(),
            gamma_client=MagicMock(),
            alerts_repo=AlertsRepo(daemon),
        )
        sink.subscribe(paper_trader.handle_alert_sync)

        # Fake subgraph: one BUY event by the watchlisted maker wallet.
        sub_client = MagicMock()
        sub_client.query = AsyncMock(
            return_value={
                "orderFilledEvents": [
                    {
                        "transactionHash": "0xtxabc",
                        "timestamp": "1700000100",
                        "maker": {"id": "0xAA"},
                        "taker": {"id": "0xBB"},
                        "market": {"id": "token-yes"},
                        "tokenId": "token-yes",
                        "side": 0,
                        "price": "0.5",
                        "size": "1.0",
                    },
                ],
                "_meta": {"block": {"number": "1", "timestamp": "1700000100"}},
            },
        )

        # Stub resolve_token so the test does not need a real gamma response.
        async def fake_resolve(
            *,
            token_id: AssetId,
            asset_index: AssetIndexRepo,
            market_cache: MarketCacheRepo,
            gamma: object,
        ) -> ResolvedToken:
            del asset_index, market_cache, gamma
            return ResolvedToken(
                condition_id=ConditionId("0xcond"),
                asset_id=AssetId(str(token_id)),
                outcome_name="Yes",
                outcome_index=0,
            )

        monkeypatch.setattr(subgraph_trades_mod, "resolve_token", fake_resolve)

        collector = SubgraphTradeCollector(
            config=SubgraphTradeCollectorConfig(enabled=True),
            subgraph_client=sub_client,
            gamma_client=MagicMock(),
            watchlist=WatchlistRegistry(WatchlistRepo(daemon)),
            asset_index=AssetIndexRepo(corpus),
            market_cache=MarketCacheRepo(daemon),
            sink=sink,
            state_repo=SubgraphWatchStateRepo(daemon),
            clock=FakeClock(),
        )

        await collector._poll_once()
        # handle_alert_sync schedules an asyncio task via create_task; drain it.
        await paper_trader.aclose()

        rows = list(
            daemon.execute(
                "SELECT triggering_alert_detector, source_wallet, outcome "
                "FROM paper_trades WHERE trade_kind = 'entry'",
            ),
        )
        assert len(rows) == 1
        detector, wallet, outcome = rows[0]
        assert detector == "subgraph_copy"
        assert wallet is not None
        assert wallet.lower() == "0xaa"
        assert outcome == "Yes"

        daemon.close()
        corpus.close()
    finally:
        monkeypatch.undo()
