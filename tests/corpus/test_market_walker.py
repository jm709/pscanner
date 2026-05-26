"""Tests for the per-market trade walker."""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.corpus.market_walker import walk_market
from pscanner.corpus.repos import (
    AssetIndexRepo,
    CorpusMarket,
    CorpusMarketsRepo,
    CorpusTradesRepo,
)
from pscanner.poly.models import Market


def _trade_dict(**overrides: Any) -> dict[str, Any]:
    base = {
        "transactionHash": "0xa",
        "asset": "asset1",
        "proxyWallet": "0xWALLET",
        "conditionId": "cond1",
        "outcome": "Yes",
        "side": "BUY",
        "price": 0.5,
        "size": 100.0,
        "timestamp": 1_000,
    }
    base.update(overrides)
    return base


def _make_market(
    *,
    condition_id: str = "cond1",
    outcomes: tuple[str, str] = ("Yes", "No"),
    clob_token_ids: tuple[str, str] = ("asset1", "asset2"),
) -> Market:
    """Build a minimal Market with two outcomes for outcome_side-index tests."""
    return Market.model_validate(
        {
            "id": "m1",
            "question": "q",
            "slug": "slug-" + condition_id,
            "conditionId": condition_id,
            "outcomes": list(outcomes),
            "outcomePrices": ["0.5", "0.5"],
            "clobTokenIds": list(clob_token_ids),
            "active": True,
            "closed": False,
        }
    )


def _fake_gamma_returning(market: Market | None) -> AsyncMock:
    """Build an AsyncMock GammaClient that returns ``market`` from get_market_by_slug."""
    gamma = AsyncMock()
    gamma.get_market_by_slug = AsyncMock(return_value=market)
    return gamma


def _fake_data_with_slug(
    *, pages: list[list[dict[str, Any]]], slug: str = "slug-cond1"
) -> AsyncMock:
    """Build an AsyncMock DataClient: slug resolves + trade pages drain in order."""
    data = AsyncMock()
    data.get_market_slug_by_condition_id = AsyncMock(return_value=slug)
    data._fetch_market_trades_page = AsyncMock(side_effect=[*pages, []])
    return data


def _seed_market(repo: CorpusMarketsRepo, condition_id: str) -> None:
    repo.insert_pending(
        CorpusMarket(
            condition_id=condition_id,
            event_slug="evt",
            category="crypto",
            closed_at=2_000,
            total_volume_usd=50_000.0,
            enumerated_at=500,
            market_slug="slug-" + condition_id,
        )
    )


@pytest.mark.asyncio
async def test_walk_inserts_trades_and_marks_complete(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")

    fake_data = _fake_data_with_slug(
        pages=[[_trade_dict(transactionHash="0xa", price=0.5, size=100.0)]],
    )
    fake_gamma = _fake_gamma_returning(_make_market())
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute(
        "SELECT backfill_state, trades_pulled_count, truncated_at_offset_cap "
        "FROM corpus_markets WHERE condition_id='cond1'"
    ).fetchone()
    assert rows["backfill_state"] == "complete"
    assert rows["trades_pulled_count"] == 1
    assert rows["truncated_at_offset_cap"] == 0
    trade_count = tmp_corpus_db.execute("SELECT COUNT(*) AS c FROM corpus_trades").fetchone()["c"]
    assert trade_count == 1


@pytest.mark.asyncio
async def test_walk_normalizes_wallet_lowercases(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = _fake_data_with_slug(pages=[[_trade_dict(proxyWallet="0xMIXED")]])
    fake_gamma = _fake_gamma_returning(_make_market())
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute("SELECT wallet_address FROM corpus_trades").fetchone()
    assert row["wallet_address"] == "0xmixed"


@pytest.mark.asyncio
async def test_walk_filters_below_notional_floor(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = _fake_data_with_slug(
        pages=[
            [
                _trade_dict(transactionHash="0xbig", price=0.5, size=100.0),
                _trade_dict(transactionHash="0xsmall", price=0.05, size=1.0),
            ],
        ],
    )
    fake_gamma = _fake_gamma_returning(_make_market())
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute("SELECT tx_hash FROM corpus_trades").fetchall()
    assert [r["tx_hash"] for r in rows] == ["0xbig"]


@pytest.mark.asyncio
async def test_walk_truncates_at_offset_cap(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = AsyncMock()
    fake_data.get_market_slug_by_condition_id = AsyncMock(return_value="slug-cond1")
    full_page = [_trade_dict(transactionHash=f"0x{i}") for i in range(500)]

    async def _fetch(condition_id: str, *, offset: int) -> list[dict[str, Any]]:
        del condition_id
        if offset >= 3000:
            return []
        return full_page

    fake_data._fetch_market_trades_page = AsyncMock(side_effect=_fetch)
    fake_gamma = _fake_gamma_returning(_make_market())
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute(
        "SELECT backfill_state, truncated_at_offset_cap FROM corpus_markets"
    ).fetchone()
    assert row["backfill_state"] == "complete"
    assert row["truncated_at_offset_cap"] == 1


@pytest.mark.asyncio
async def test_walk_resolves_team_name_outcomes_via_clob_token_ids(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """#159 regression: sports markets used to collapse both legs to NO.

    Now each trade's ``outcome_side`` is derived from the position of its
    ``asset`` in ``clob_token_ids`` (parallel to ``outcomes``).
    """
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = _fake_data_with_slug(
        pages=[
            [
                _trade_dict(
                    transactionHash="0xcavs",
                    asset="token-cavs",
                    outcome="Cavaliers",
                    price=0.5,
                    size=100.0,
                ),
                _trade_dict(
                    transactionHash="0xknicks",
                    asset="token-knicks",
                    outcome="Knicks",
                    price=0.5,
                    size=100.0,
                ),
            ],
        ],
    )
    fake_gamma = _fake_gamma_returning(
        _make_market(
            outcomes=("Cavaliers", "Knicks"),
            clob_token_ids=("token-cavs", "token-knicks"),
        )
    )
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute(
        "SELECT tx_hash, asset_id, outcome_side FROM corpus_trades ORDER BY tx_hash"
    ).fetchall()
    sides = {r["asset_id"]: r["outcome_side"] for r in rows}
    assert sides == {"token-cavs": "YES", "token-knicks": "NO"}


@pytest.mark.asyncio
async def test_walk_falls_back_to_outcome_name_on_gamma_failure(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """When gamma can't resolve the market, the walk continues with the
    legacy outcome-name heuristic and still completes successfully."""
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = _fake_data_with_slug(
        pages=[[_trade_dict(transactionHash="0xa", outcome="Yes")]],
    )
    fake_gamma = _fake_gamma_returning(None)  # market not found in gamma
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute(
        "SELECT outcome_side FROM corpus_trades WHERE tx_hash='0xa'"
    ).fetchone()
    assert row["outcome_side"] == "YES"  # outcome.lower() == "yes" fallback path


@pytest.mark.asyncio
async def test_walk_populates_asset_index(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Regression: walk_market must upsert asset_index for both legs.

    Previously walk_market built outcome_side_by_asset_id for parsing but
    never wrote it; markets ingested after the last manual run of
    AssetIndexRepo.backfill_from_corpus_trades had empty asset_index,
    breaking subgraph backfill's _load_market_asset_ids.
    """
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = _fake_data_with_slug(pages=[[_trade_dict(transactionHash="0xa")]])
    fake_gamma = _fake_gamma_returning(
        _make_market(clob_token_ids=("token-yes", "token-no")),
    )
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    rows = tmp_corpus_db.execute(
        "SELECT asset_id, outcome_side, outcome_index FROM asset_index "
        "WHERE condition_id='cond1' ORDER BY outcome_index"
    ).fetchall()
    assert [(r["asset_id"], r["outcome_side"], r["outcome_index"]) for r in rows] == [
        ("token-yes", "YES", 0),
        ("token-no", "NO", 1),
    ]


@pytest.mark.asyncio
async def test_walk_skips_resolution_for_non_binary_market(
    tmp_corpus_db: sqlite3.Connection,
) -> None:
    """Markets with non-binary ``clob_token_ids`` skip the resolver and
    fall back to the legacy heuristic (preserves prior behavior)."""
    markets = CorpusMarketsRepo(tmp_corpus_db)
    trades = CorpusTradesRepo(tmp_corpus_db)
    assets = AssetIndexRepo(tmp_corpus_db)
    _seed_market(markets, "cond1")
    fake_data = _fake_data_with_slug(
        pages=[[_trade_dict(transactionHash="0xa", outcome="OptionA", asset="token-a")]],
    )
    fake_gamma = _fake_gamma_returning(
        Market.model_validate(
            {
                "id": "m1",
                "question": "q",
                "slug": "slug-cond1",
                "conditionId": "cond1",
                "outcomes": ["OptionA", "OptionB", "OptionC"],
                "outcomePrices": ["0.33", "0.33", "0.34"],
                "clobTokenIds": ["token-a", "token-b", "token-c"],
                "active": True,
                "closed": False,
            }
        )
    )
    await walk_market(
        condition_id="cond1",
        data=fake_data,
        gamma=fake_gamma,
        markets_repo=markets,
        trades_repo=trades,
        asset_repo=assets,
        now_ts=1_500,
    )
    row = tmp_corpus_db.execute(
        "SELECT outcome_side FROM corpus_trades WHERE tx_hash='0xa'"
    ).fetchone()
    # 3-outcome market not resolved; falls back to "OptionA".lower() != "yes" → NO.
    assert row["outcome_side"] == "NO"
