"""Tests for ``GammaClient`` against a mocked ``PolyHttpClient``.

The Wave 2 ``http-client`` agent owns the real HTTP plumbing; here we mock the
``PolyHttpClient.get`` coroutine and verify only that ``GammaClient`` issues
the right paths/params and validates the JSON correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pscanner.poly import gamma as gamma_module
from pscanner.poly.gamma import GammaClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import Event, Market

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _load_event_fixture() -> dict[str, Any]:
    return json.loads((_FIXTURE_DIR / "event.json").read_text())


def _load_market_fixture() -> dict[str, Any]:
    return json.loads((_FIXTURE_DIR / "market.json").read_text())


def _mock_http_returning(payload: Any) -> AsyncMock:
    """Build a mock ``PolyHttpClient`` whose ``get`` always returns ``payload``."""
    mock = AsyncMock(spec=PolyHttpClient)
    mock.get.return_value = payload
    return mock


def _mock_http_pages(*pages: Any) -> AsyncMock:
    """Build a mock ``PolyHttpClient`` whose ``get`` yields ``pages`` in order."""
    mock = AsyncMock(spec=PolyHttpClient)
    mock.get.side_effect = list(pages)
    return mock


async def test_list_events_parses_fixture_and_calls_correct_path() -> None:
    fixture = _load_event_fixture()
    http = _mock_http_returning([fixture])
    client = GammaClient(http=http)

    events = await client.list_events()

    assert len(events) == 1
    assert isinstance(events[0], Event)
    assert events[0].id == "16167"
    assert events[0].title.startswith("MicroStrategy")
    http.get.assert_awaited_once_with(
        "/events",
        params={"active": "true", "closed": "false", "limit": 100, "offset": 0},
    )


async def test_list_events_passes_filter_params() -> None:
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    await client.list_events(active=False, closed=True, limit=25, offset=50)

    http.get.assert_awaited_once_with(
        "/events",
        params={"active": "false", "closed": "true", "limit": 25, "offset": 50},
    )


async def test_list_events_rejects_non_array_payload() -> None:
    http = _mock_http_returning({"not": "a list"})
    client = GammaClient(http=http)

    with pytest.raises(TypeError, match="expected JSON array"):
        await client.list_events()


async def test_iter_events_paginates_until_short_page() -> None:
    fixture = _load_event_fixture()
    page1 = [fixture, fixture]
    page2 = [fixture]  # short page → terminates after yielding
    http = _mock_http_pages(page1, page2)
    client = GammaClient(http=http)

    collected = [event async for event in client.iter_events(page_size=2)]

    assert len(collected) == 3
    assert all(isinstance(e, Event) for e in collected)
    assert http.get.await_count == 2
    second_call = http.get.await_args_list[1]
    assert second_call.kwargs["params"]["offset"] == 2


async def test_iter_events_terminates_on_empty_page() -> None:
    fixture = _load_event_fixture()
    page1 = [fixture, fixture]
    http = _mock_http_pages(page1, [])
    client = GammaClient(http=http)

    collected = [event async for event in client.iter_events(page_size=2)]

    assert len(collected) == 2
    assert http.get.await_count == 2


async def test_iter_events_handles_immediately_empty_catalogue() -> None:
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    collected = [event async for event in client.iter_events(page_size=10)]

    assert collected == []
    assert http.get.await_count == 1


async def test_get_event_returns_single_event() -> None:
    fixture = _load_event_fixture()
    http = _mock_http_returning(fixture)
    client = GammaClient(http=http)

    event = await client.get_event("16167")

    assert isinstance(event, Event)
    assert event.id == "16167"
    http.get.assert_awaited_once_with("/events/16167")


async def test_get_event_rejects_array_payload() -> None:
    http = _mock_http_returning([{"foo": "bar"}])
    client = GammaClient(http=http)

    with pytest.raises(TypeError, match="expected JSON object"):
        await client.get_event("16167")


async def test_list_markets_parses_outcome_prices_json_string() -> None:
    market_payload = {
        "id": "abc",
        "question": "Will it?",
        "slug": "will-it",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.5","0.5"]',
        "clobTokenIds": '["1","2"]',
    }
    http = _mock_http_returning([market_payload])
    client = GammaClient(http=http)

    markets = await client.list_markets()

    assert len(markets) == 1
    market = markets[0]
    assert isinstance(market, Market)
    assert market.outcomes == ["Yes", "No"]
    assert market.outcome_prices == [0.5, 0.5]
    assert market.clob_token_ids == ["1", "2"]


async def test_list_markets_parses_real_fixture() -> None:
    fixture = _load_market_fixture()
    http = _mock_http_returning([fixture])
    client = GammaClient(http=http)

    markets = await client.list_markets(active=True, closed=False, limit=10, offset=0)

    assert len(markets) == 1
    assert markets[0].outcome_prices == [0.525, 0.475]
    http.get.assert_awaited_once_with(
        "/markets",
        params={"active": "true", "closed": "false", "limit": 10, "offset": 0},
    )


async def test_iter_markets_paginates() -> None:
    fixture = _load_market_fixture()
    http = _mock_http_pages([fixture, fixture], [fixture], [])
    client = GammaClient(http=http)

    collected = [m async for m in client.iter_markets(page_size=2)]

    assert len(collected) == 3
    assert all(isinstance(m, Market) for m in collected)


async def test_aclose_does_not_close_borrowed_http() -> None:
    http = _mock_http_returning([])
    client = GammaClient(http=http)

    await client.aclose()

    http.aclose.assert_not_called()


async def test_aclose_closes_owned_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``http`` is None the client owns its ``PolyHttpClient`` and closes it."""
    owned = AsyncMock(spec=PolyHttpClient)
    captured: dict[str, Any] = {}

    def fake_factory(*, base_url: str, rpm: int) -> AsyncMock:
        captured["base_url"] = base_url
        captured["rpm"] = rpm
        return owned

    monkeypatch.setattr(gamma_module, "PolyHttpClient", fake_factory)

    client = GammaClient(rpm=42)

    assert captured == {"base_url": "https://gamma-api.polymarket.com", "rpm": 42}

    await client.aclose()

    owned.aclose.assert_awaited_once()
