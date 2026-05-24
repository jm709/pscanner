"""Tests for `pscanner.poly.onchain_ingest` — event→trade conversion."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pscanner.corpus.db import init_corpus_db
from pscanner.corpus.repos import AssetEntry, AssetIndexRepo, CorpusTrade
from pscanner.poly.onchain import CTF_EXCHANGE_ADDRESS, OrderFilledEvent
from pscanner.poly.onchain_ingest import (
    UnresolvableAsset,
    UnsupportedFill,
    event_to_corpus_trade,
)


@pytest.fixture
def asset_repo(tmp_path: Path) -> Iterator[AssetIndexRepo]:
    conn = init_corpus_db(tmp_path / "corpus.sqlite3")
    try:
        repo = AssetIndexRepo(conn)
        repo.upsert(
            AssetEntry(
                asset_id="123456789",
                condition_id="0xCONDITION",
                outcome_side="YES",
                outcome_index=0,
            )
        )
        yield repo
    finally:
        conn.close()


def _ev(
    *,
    maker: str = "0x" + "11" * 20,
    taker: str = "0x" + "22" * 20,
    maker_asset_id: int = 0,
    taker_asset_id: int = 123_456_789,
    making: int = 700_000,
    taking: int = 1_000_000,
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


def test_event_to_corpus_trade_buy_maker_gives_usdc(
    asset_repo: AssetIndexRepo,
) -> None:
    """Maker giving USDC for CTF tokens is a BUY from the maker's POV."""
    event = _ev(
        maker_asset_id=0,
        taker_asset_id=123_456_789,
        making=700_000,  # 0.70 USDC the maker gives
        taking=1_000_000,  # 1.0 CTF the maker receives
    )
    trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=1_700_000_000)
    assert isinstance(trade, CorpusTrade)
    assert trade.tx_hash == "0x" + "cd" * 32
    assert trade.asset_id == "123456789"
    assert trade.condition_id == "0xCONDITION"
    assert trade.outcome_side == "YES"
    assert trade.wallet_address == "0x" + "11" * 20  # maker
    assert trade.bs == "BUY"
    assert trade.price == pytest.approx(0.70)
    assert trade.size == pytest.approx(1.0)
    assert trade.notional_usd == pytest.approx(0.70)
    assert trade.ts == 1_700_000_000


def test_event_to_corpus_trade_sell_maker_gives_ctf(
    asset_repo: AssetIndexRepo,
) -> None:
    """Maker giving CTF tokens for USDC is a SELL from the maker's POV."""
    event = _ev(
        maker_asset_id=123_456_789,
        taker_asset_id=0,
        making=1_000_000,  # 1.0 CTF the maker gives
        taking=420_000,  # 0.42 USDC the maker receives
    )
    trade = event_to_corpus_trade(event, asset_repo=asset_repo, ts=1_700_000_000)
    assert trade.bs == "SELL"
    assert trade.wallet_address == "0x" + "11" * 20  # maker
    assert trade.price == pytest.approx(0.42)
    assert trade.size == pytest.approx(1.0)
    assert trade.notional_usd == pytest.approx(0.42)


def test_event_to_corpus_trade_skips_exchange_as_maker(
    asset_repo: AssetIndexRepo,
) -> None:
    """When the maker is the exchange contract itself, skip — it's a merge."""
    event = _ev(
        maker=CTF_EXCHANGE_ADDRESS,
        maker_asset_id=0,
        taker_asset_id=123_456_789,
        making=500_000,
        taking=1_000_000,
    )
    with pytest.raises(UnsupportedFill, match="maker is exchange contract"):
        event_to_corpus_trade(event, asset_repo=asset_repo, ts=0)


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
