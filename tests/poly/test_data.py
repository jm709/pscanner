"""Tests for ``pscanner.poly.data.DataClient``.

The data API and leaderboard live on different hosts, so :class:`DataClient`
always owns at least one underlying :class:`PolyHttpClient` instance. The
sister-wave ``http-client`` agent owns ``PolyHttpClient``'s implementation, so
these tests stub it out to keep this module under isolated test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import httpx
import pytest

from pscanner.poly import data as data_module
from pscanner.poly.data import DataClient
from pscanner.poly.http import PolyHttpClient
from pscanner.poly.models import ClosedPosition, LeaderboardEntry, Position


def _http_400_error(*, message: str = "offset too large") -> httpx.HTTPStatusError:
    """Build a synthetic 400 ``HTTPStatusError`` for side_effect injection."""
    request = httpx.Request("GET", "https://data-api.polymarket.com/")
    response = httpx.Response(400, json={"error": message}, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> Any:
    """Read a JSON fixture from ``tests/fixtures/``."""
    return json.loads((_FIXTURE_DIR / name).read_text())


class _FakePolyHttpClient:
    """Minimal stand-in for ``PolyHttpClient`` exposing the methods we use.

    Each instance records the ``base_url`` it was built with so tests can
    assert how :class:`DataClient` routes calls between the data and
    leaderboard hosts. ``get`` and ``aclose`` are :class:`AsyncMock` so call
    arguments and counts are introspectable.
    """

    def __init__(self, *, base_url: str, rpm: int, timeout_seconds: float = 30.0) -> None:
        """Capture constructor args and seed mock methods."""
        self.base_url = base_url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self.get = AsyncMock()
        self.aclose = AsyncMock()


@pytest.fixture
def fake_http_factory(monkeypatch: pytest.MonkeyPatch) -> list[_FakePolyHttpClient]:
    """Patch ``PolyHttpClient`` in ``data`` module and capture every instance."""
    instances: list[_FakePolyHttpClient] = []

    def _build(*args: Any, **kwargs: Any) -> _FakePolyHttpClient:
        client = _FakePolyHttpClient(*args, **kwargs)
        instances.append(client)
        return client

    monkeypatch.setattr(data_module, "PolyHttpClient", _build)
    return instances


def _client_for_host(instances: list[_FakePolyHttpClient], host: str) -> _FakePolyHttpClient:
    """Return the captured fake client whose base URL matches ``host``."""
    for client in instances:
        if host in client.base_url:
            return client
    msg = f"no fake client built for host containing {host!r}"
    raise AssertionError(msg)


def test_init_owns_both_clients_when_http_is_none(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    DataClient(rpm=42)

    assert len(fake_http_factory) == 2
    base_urls = {c.base_url for c in fake_http_factory}
    assert "https://data-api.polymarket.com" in base_urls
    assert "https://lb-api.polymarket.com" in base_urls
    assert all(c.rpm == 42 for c in fake_http_factory)


def test_init_reuses_passed_http_for_data_api(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    shared = _FakePolyHttpClient(base_url="https://data-api.polymarket.com", rpm=10)

    client = DataClient(http=cast("PolyHttpClient", shared), rpm=99)

    # Only the leaderboard client is constructed — the data client is reused.
    assert len(fake_http_factory) == 1
    lb = fake_http_factory[0]
    assert "lb-api.polymarket.com" in lb.base_url
    assert lb.rpm == 99
    assert client._data_http is shared


async def test_get_positions_parses_fixture(
    fake_http_factory: list[_FakePolyHttpClient],
    sample_position_json: list[dict[str, Any]],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = sample_position_json

    positions = await client.get_positions(
        "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee",
        size_threshold=1.5,
    )

    data_http.get.assert_awaited_once_with(
        "/positions",
        params={"user": "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee", "sizeThreshold": 1.5},
    )
    assert len(positions) == len(sample_position_json)
    assert all(isinstance(p, Position) for p in positions)
    assert positions[0].size == sample_position_json[0]["size"]


async def test_get_positions_rejects_non_list_response(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = {"error": "oops"}

    with pytest.raises(TypeError, match="expected list"):
        await client.get_positions("0xabc")


async def test_get_closed_positions_parses_fixture_and_computes_won(
    fake_http_factory: list[_FakePolyHttpClient],
    sample_closed_positions_json: list[dict[str, Any]],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = sample_closed_positions_json

    closed = await client.get_closed_positions("0xabc", limit=25)

    data_http.get.assert_awaited_once_with(
        "/v1/closed-positions",
        params={"user": "0xabc", "limit": 25},
    )
    assert len(closed) == len(sample_closed_positions_json)
    assert all(isinstance(c, ClosedPosition) for c in closed)
    # Every fixture row has realizedPnl > 0 → won is True.
    assert all(c.won for c in closed)


async def test_get_closed_positions_won_is_false_when_pnl_negative(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = [
        {
            "proxyWallet": "0xabc",
            "asset": "1",
            "conditionId": "0xc",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "avgPrice": 0.5,
            "realizedPnl": -100.0,
            "redeemable": False,
        },
    ]

    closed = await client.get_closed_positions("0xabc")

    assert closed[0].won is False


async def test_get_settled_positions_hits_positions_endpoint_with_closed_filter(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """``get_settled_positions`` must call ``/positions?closed=true`` not the legacy v1 path."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    # Realistic mix of wins, losses, and a zero-pnl artifact (split/merge/convert).
    data_http.get.return_value = [
        {
            "proxyWallet": "0xabc",
            "asset": "1",
            "conditionId": "0xc1",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "size": 100.0,
            "avgPrice": 0.4,
            "currentValue": 0.0,
            "realizedPnl": 60.0,
            "redeemable": True,
        },
        {
            "proxyWallet": "0xabc",
            "asset": "2",
            "conditionId": "0xc2",
            "outcome": "No",
            "outcomeIndex": 1,
            "size": 50.0,
            "avgPrice": 0.6,
            "currentValue": 0.0,
            "realizedPnl": -30.0,
            "redeemable": False,
        },
        {
            "proxyWallet": "0xabc",
            "asset": "3",
            "conditionId": "0xc3",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "size": 25.0,
            "avgPrice": 0.5,
            "currentValue": 0.0,
            "realizedPnl": 0.0,
            "redeemable": False,
        },
    ]

    settled = await client.get_settled_positions("0xabc", limit=500)

    data_http.get.assert_awaited_once_with(
        "/positions",
        params={"user": "0xabc", "closed": "true", "limit": 500},
    )
    assert len(settled) == 3
    assert all(isinstance(c, ClosedPosition) for c in settled)
    # Field roundtrip on the mixed payload.
    assert settled[0].won is True
    assert settled[1].won is False
    assert settled[2].won is False
    assert settled[0].avg_price == pytest.approx(0.4)
    assert settled[1].realized_pnl == pytest.approx(-30.0)
    assert settled[2].realized_pnl == pytest.approx(0.0)


async def test_get_settled_positions_rejects_non_list_response(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = {"error": "oops"}

    with pytest.raises(TypeError, match="expected list"):
        await client.get_settled_positions("0xabc")


async def test_get_activity_parses_fixture(
    fake_http_factory: list[_FakePolyHttpClient],
    sample_activity_json: list[dict[str, Any]],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = sample_activity_json

    events = await client.get_activity("0xabc", limit=50)

    data_http.get.assert_awaited_once_with(
        "/activity",
        params={"user": "0xabc", "limit": 50},
    )
    assert len(events) == len(sample_activity_json)
    assert events[0]["type"] == "TRADE"


async def test_get_activity_passes_type_filter(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = []

    await client.get_activity("0xabc", limit=10, type="TRADE")

    data_http.get.assert_awaited_once_with(
        "/activity",
        params={"user": "0xabc", "limit": 10, "type": "TRADE"},
    )


async def test_get_activity_forwards_offset(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A non-zero ``offset`` is forwarded as the ``offset`` query param."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = []

    await client.get_activity("0xabc", limit=200, offset=400)

    data_http.get.assert_awaited_once_with(
        "/activity",
        params={"user": "0xabc", "limit": 200, "offset": 400},
    )


async def test_get_activity_omits_offset_when_zero(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """``offset=0`` is omitted from the query string for backward compatibility."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = []

    await client.get_activity("0xabc", limit=200, offset=0)

    data_http.get.assert_awaited_once_with(
        "/activity",
        params={"user": "0xabc", "limit": 200},
    )


async def test_get_first_activity_timestamp_returns_smallest(
    fake_http_factory: list[_FakePolyHttpClient],
    sample_activity_json: list[dict[str, Any]],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    # Single short page → the smallest timestamp on the page is the answer.
    data_http.get.return_value = sample_activity_json

    earliest = await client.get_first_activity_timestamp("0xabc")

    expected = min(item["timestamp"] for item in sample_activity_json)
    assert earliest == expected


async def test_get_first_activity_timestamp_returns_none_for_empty(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = []

    earliest = await client.get_first_activity_timestamp("0xabc")

    assert earliest is None


async def test_get_first_activity_timestamp_pages_until_short_page(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    full_page = [{"timestamp": 2_000_000_000 + i} for i in range(500)]
    short_page = [{"timestamp": 100}, {"timestamp": 200}]
    data_http.get.side_effect = [full_page, short_page]

    earliest = await client.get_first_activity_timestamp("0xabc")

    assert earliest == 100
    assert data_http.get.await_count == 2
    second_call_kwargs = data_http.get.await_args_list[1].kwargs
    assert second_call_kwargs["params"]["offset"] == 500


async def test_get_leaderboard_parses_fixture_and_injects_period(
    fake_http_factory: list[_FakePolyHttpClient],
    sample_leaderboard_json: list[dict[str, Any]],
) -> None:
    client = DataClient()
    lb_http = _client_for_host(fake_http_factory, "lb-api")
    # Strip ``period`` so we are sure it is the client that injects it.
    wire = [{k: v for k, v in row.items() if k != "period"} for row in sample_leaderboard_json]
    lb_http.get.return_value = wire

    entries = await client.get_leaderboard(period="all", limit=50)

    lb_http.get.assert_awaited_once_with("/profit", params={"window": "all", "limit": 50})
    assert len(entries) == len(sample_leaderboard_json)
    assert all(isinstance(e, LeaderboardEntry) for e in entries)
    assert all(e.period == "all" for e in entries)


@pytest.mark.parametrize(
    ("period", "expected_window"),
    [("day", "1d"), ("week", "7d"), ("all", "all")],
)
async def test_get_leaderboard_maps_period_to_wire_window(
    fake_http_factory: list[_FakePolyHttpClient],
    period: Literal["day", "week", "all"],
    expected_window: str,
) -> None:
    client = DataClient()
    lb_http = _client_for_host(fake_http_factory, "lb-api")
    lb_http.get.return_value = []

    await client.get_leaderboard(period=period, limit=5)

    lb_http.get.assert_awaited_once_with(
        "/profit",
        params={"window": expected_window, "limit": 5},
    )


async def test_aclose_closes_owned_clients(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    lb_http = _client_for_host(fake_http_factory, "lb-api")

    await client.aclose()

    data_http.aclose.assert_awaited_once_with()
    lb_http.aclose.assert_awaited_once_with()


async def test_aclose_does_not_close_passed_in_data_http(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    shared = _FakePolyHttpClient(base_url="https://data-api.polymarket.com", rpm=10)
    client = DataClient(http=cast("PolyHttpClient", shared))
    lb_http = _client_for_host(fake_http_factory, "lb-api")

    await client.aclose()

    shared.aclose.assert_not_awaited()
    lb_http.aclose.assert_awaited_once_with()


async def test_aclose_is_idempotent(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    lb_http = _client_for_host(fake_http_factory, "lb-api")

    await client.aclose()
    await client.aclose()

    assert data_http.aclose.await_count == 1
    assert lb_http.aclose.await_count == 1


async def test_get_market_trades_filters_by_window(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = [
        {
            "proxyWallet": "0xa",
            "timestamp": 1500,
            "size": 10.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        },
        {
            "proxyWallet": "0xb",
            "timestamp": 1200,
            "size": 20.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        },
        {
            "proxyWallet": "0xc",
            "timestamp": 900,
            "size": 30.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        },
    ]

    out = await client.get_market_trades(
        condition_id="0xabc",
        since_ts=1000,
        until_ts=1600,
    )

    assert {t["proxyWallet"] for t in out} == {"0xa", "0xb"}
    data_http.get.assert_awaited_once_with(
        "/trades",
        params={"market": "0xabc", "limit": 500, "offset": 0},
    )


async def test_get_market_trades_paginates_until_below_window(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """Stops paginating as soon as the newest timestamp on a page is below the window."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    page1 = [
        {
            "proxyWallet": f"0x{i}",
            "timestamp": 2000 - i,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        }
        for i in range(500)
    ]
    page2 = [
        {
            "proxyWallet": "0x_old",
            "timestamp": 100,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        },
    ]
    data_http.get.side_effect = [page1, page2]

    out = await client.get_market_trades(
        condition_id="0xabc",
        since_ts=1000,
        until_ts=3000,
    )

    assert data_http.get.await_count == 2
    assert len(out) == 500
    assert "0x_old" not in {t["proxyWallet"] for t in out}


async def test_get_market_trades_stops_at_page_cap(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A market that always returns full in-window pages is hard-capped at 30 pages."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    full_page = [
        {
            "proxyWallet": f"0x{i}",
            "timestamp": 2_000_000_000,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        }
        for i in range(500)
    ]
    data_http.get.return_value = full_page

    out = await client.get_market_trades(
        condition_id="0xabc",
        since_ts=0,
        until_ts=3_000_000_000,
    )

    assert data_http.get.await_count == 30
    assert len(out) == 30 * 500


async def test_get_market_slug_by_condition_id_returns_slug(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """``/trades?market=&limit=1`` slug field is surfaced unchanged."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = [
        {
            "proxyWallet": "0xa",
            "timestamp": 1,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
            "slug": "nhl-edm-ana-2026-04-26",
            "conditionId": "0xabc",
        },
    ]

    slug = await client.get_market_slug_by_condition_id("0xabc")

    assert slug == "nhl-edm-ana-2026-04-26"
    data_http.get.assert_awaited_once_with(
        "/trades",
        params={"market": "0xabc", "limit": 1},
    )


async def test_get_market_slug_by_condition_id_returns_none_when_empty(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """An empty page (no trades on this market) returns ``None``."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = []

    assert await client.get_market_slug_by_condition_id("0xabc") is None


async def test_get_market_slug_by_condition_id_returns_none_when_slug_missing(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A trade row that doesn't carry a ``slug`` field returns ``None``."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = [
        {
            "proxyWallet": "0xa",
            "timestamp": 1,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
            "conditionId": "0xabc",
        },
    ]

    assert await client.get_market_slug_by_condition_id("0xabc") is None


async def test_get_market_slug_by_condition_id_returns_none_when_slug_blank(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A trade row whose ``slug`` is the empty string returns ``None``."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = [
        {
            "proxyWallet": "0xa",
            "timestamp": 1,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
            "slug": "",
            "conditionId": "0xabc",
        },
    ]

    assert await client.get_market_slug_by_condition_id("0xabc") is None


async def test_get_market_slug_by_condition_id_returns_none_when_first_item_not_dict(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A malformed first item (not a dict) returns ``None`` rather than raising."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.return_value = ["malformed-string-instead-of-dict"]

    assert await client.get_market_slug_by_condition_id("0xabc") is None


async def test_get_first_activity_timestamp_swallows_offset_cap_400(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A 400 at offset>=3000 must terminate pagination cleanly (Polymarket cap)."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    # Pages 0..5 (offsets 0, 500, ..., 2500) return full 500-row pages; page at
    # offset=3000 raises the offset-cap 400 the API actually emits.
    full_page = [{"timestamp": 1_700_000_000 + i} for i in range(500)]
    data_http.get.side_effect = [*([full_page] * 6), _http_400_error()]

    earliest = await client.get_first_activity_timestamp("0xa")

    assert earliest == 1_700_000_000
    assert data_http.get.await_count == 7
    last_call_kwargs = data_http.get.await_args_list[-1].kwargs
    assert last_call_kwargs["params"]["offset"] == 3000


async def test_get_first_activity_timestamp_propagates_400_at_offset_zero(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A 400 at offset=0 should propagate (real validation error, not the cap)."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.side_effect = _http_400_error(message="bad user id")

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_first_activity_timestamp("0xbad")


async def test_get_market_trades_swallows_offset_cap_400(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A 400 at offset>=3000 in /trades pagination must break cleanly."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    full_page = [
        {
            "proxyWallet": f"0x{i}",
            "timestamp": 2_000_000_000,
            "size": 1.0,
            "price": 0.5,
            "side": "BUY",
            "outcome": "Yes",
        }
        for i in range(500)
    ]
    data_http.get.side_effect = [*([full_page] * 6), _http_400_error()]

    out = await client.get_market_trades(
        condition_id="0xabc",
        since_ts=0,
        until_ts=3_000_000_000,
    )

    # 6 successful pages * 500 rows = 3000 trades, then 400 at offset=3000.
    assert len(out) == 6 * 500
    assert data_http.get.await_count == 7
    last_call_kwargs = data_http.get.await_args_list[-1].kwargs
    assert last_call_kwargs["params"]["offset"] == 3000


async def test_get_market_trades_propagates_400_at_offset_zero(
    fake_http_factory: list[_FakePolyHttpClient],
) -> None:
    """A 400 at offset=0 should propagate (not the offset cap)."""
    client = DataClient()
    data_http = _client_for_host(fake_http_factory, "data-api")
    data_http.get.side_effect = _http_400_error(message="bad market")

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_market_trades(
            condition_id="0xbad",
            since_ts=0,
            until_ts=3_000_000_000,
        )
