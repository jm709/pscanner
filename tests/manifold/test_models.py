"""Tests for ``pscanner.manifold.models`` pydantic round-trip."""

from __future__ import annotations

import pytest

from pscanner.manifold.models import ManifoldBet, ManifoldMarket, ManifoldUser

_BINARY_MARKET = {
    "id": "kjPkT2HECV",
    "creatorId": "igi2zGXsfxYPgB0DJTXVJVmwCOr2",
    "question": "Will AI surpass human performance on MATH by 2026?",
    "outcomeType": "BINARY",
    "mechanism": "cpmm-1",
    "prob": 0.62,
    "volume": 15000.0,
    "totalLiquidity": 1200.0,
    "isResolved": False,
    "resolutionTime": None,
    "closeTime": 1_800_000_000_000,
    "url": "https://manifold.markets/alice/will-ai-surpass",
    "slug": "will-ai-surpass",
}

_MULTIPLE_CHOICE_MARKET = {
    "id": "MCMktABC123",
    "creatorId": "userXYZ",
    "question": "Who wins the 2026 World Cup?",
    "outcomeType": "MULTIPLE_CHOICE",
    "mechanism": "cpmm-multi-1",
    "volume": 5000.0,
    "totalLiquidity": 800.0,
    "isResolved": False,
    # No prob — multi-outcome markets don't have a single prob
}

_BET = {
    "id": "bet-abc-123",
    "userId": "igi2zGXsfxYPgB0DJTXVJVmwCOr2",
    "contractId": "kjPkT2HECV",
    "outcome": "YES",
    "amount": 25.0,
    "probBefore": 0.60,
    "probAfter": 0.62,
    "createdTime": 1_714_000_000_000,
    "isFilled": True,
    "isCancelled": False,
    "limitProb": None,
    "shares": 40.0,
}

_LIMIT_BET = {
    **_BET,
    "id": "bet-limit-456",
    "isFilled": False,
    "limitProb": 0.55,
}

_USER = {
    "id": "igi2zGXsfxYPgB0DJTXVJVmwCOr2",
    "username": "alice",
    "name": "Alice Wonderland",
    "createdTime": 1_700_000_000_000,
    "balance": 500.0,
    "avatarUrl": "https://example.com/avatar.png",
}


def test_binary_market_parses_all_fields() -> None:
    market = ManifoldMarket.model_validate(_BINARY_MARKET)
    assert market.id == "kjPkT2HECV"
    assert market.creator_id == "igi2zGXsfxYPgB0DJTXVJVmwCOr2"
    assert market.question == "Will AI surpass human performance on MATH by 2026?"
    assert market.outcome_type == "BINARY"
    assert market.mechanism == "cpmm-1"
    assert market.prob == pytest.approx(0.62)
    assert market.volume == 15000.0
    assert market.total_liquidity == 1200.0
    assert market.is_resolved is False
    assert market.resolution_time is None
    assert market.close_time == 1_800_000_000_000


def test_binary_market_is_binary_property() -> None:
    market = ManifoldMarket.model_validate(_BINARY_MARKET)
    assert market.is_binary is True


def test_multiple_choice_market_is_not_binary() -> None:
    market = ManifoldMarket.model_validate(_MULTIPLE_CHOICE_MARKET)
    assert market.is_binary is False
    assert market.prob is None


def test_market_extra_fields_ignored() -> None:
    """Unknown API fields must not raise ValidationError."""
    payload = {**_BINARY_MARKET, "unknownFutureField": "surprise", "anotherOne": 42}
    market = ManifoldMarket.model_validate(payload)
    assert market.id == "kjPkT2HECV"


def test_market_roundtrip_via_json() -> None:
    market = ManifoldMarket.model_validate(_BINARY_MARKET)
    json_str = market.model_dump_json(by_alias=True)
    market2 = ManifoldMarket.model_validate_json(json_str)
    assert market2.id == market.id
    assert market2.prob == market.prob


def test_bet_parses_market_order() -> None:
    bet = ManifoldBet.model_validate(_BET)
    assert bet.id == "bet-abc-123"
    assert bet.user_id == "igi2zGXsfxYPgB0DJTXVJVmwCOr2"
    assert bet.contract_id == "kjPkT2HECV"
    assert bet.outcome == "YES"
    assert bet.amount == 25.0
    assert bet.prob_before == pytest.approx(0.60)
    assert bet.prob_after == pytest.approx(0.62)
    assert bet.is_filled is True
    assert bet.is_cancelled is False
    assert bet.limit_prob is None


def test_bet_parses_limit_order() -> None:
    bet = ManifoldBet.model_validate(_LIMIT_BET)
    assert bet.limit_prob == pytest.approx(0.55)
    assert bet.is_filled is False


def test_bet_extra_fields_ignored() -> None:
    payload = {**_BET, "fees": {"creatorFee": 0.1, "platformFee": 0.05}}
    bet = ManifoldBet.model_validate(payload)
    assert bet.id == "bet-abc-123"


def test_bet_roundtrip_via_json() -> None:
    bet = ManifoldBet.model_validate(_BET)
    json_str = bet.model_dump_json(by_alias=True)
    bet2 = ManifoldBet.model_validate_json(json_str)
    assert bet2.id == bet.id
    assert bet2.amount == bet.amount


def test_user_parses_all_fields() -> None:
    user = ManifoldUser.model_validate(_USER)
    assert user.id == "igi2zGXsfxYPgB0DJTXVJVmwCOr2"
    assert user.username == "alice"
    assert user.name == "Alice Wonderland"
    assert user.created_time == 1_700_000_000_000
    assert user.balance == 500.0
    assert user.avatar_url == "https://example.com/avatar.png"


def test_user_extra_fields_ignored() -> None:
    payload = {**_USER, "profitCached": {"weekly": 100.0}}
    user = ManifoldUser.model_validate(payload)
    assert user.username == "alice"


def test_user_roundtrip_via_json() -> None:
    user = ManifoldUser.model_validate(_USER)
    json_str = user.model_dump_json(by_alias=True)
    user2 = ManifoldUser.model_validate_json(json_str)
    assert user2.id == user.id
    assert user2.username == user.username
