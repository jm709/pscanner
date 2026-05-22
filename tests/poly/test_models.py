"""Defensive parsing tests for `pscanner.poly.models` (#161)."""

from __future__ import annotations

from pscanner.poly.models import Market


def _payload(**overrides: object) -> dict[str, object]:
    """Return a minimum-viable gamma `/markets` payload, optionally overridden."""
    base: dict[str, object] = {
        "id": "m1",
        "question": "ignored",
        "slug": "seed",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.5","0.5"]',
        "clobTokenIds": '["yes-token","no-token"]',
        "active": True,
        "closed": False,
    }
    base.update(overrides)
    return base


def test_outcome_prices_literal_null_string_parses_as_empty() -> None:
    """Gamma occasionally returns ``outcomePrices='null'`` for stale markets.

    Regression: pre-#161, this raised ``ValidationError`` and aborted any
    refresh that hit the payload mid-page. Now treated as an empty list.
    """
    market = Market.model_validate(_payload(outcomePrices="null"))
    assert market.outcome_prices == []


def test_outcomes_literal_null_string_parses_as_empty() -> None:
    """Same defensive contract for the sibling ``outcomes`` field."""
    market = Market.model_validate(_payload(outcomes="null"))
    assert market.outcomes == []
