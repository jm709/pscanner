"""Tests for the feature-projection registry (#145)."""

from __future__ import annotations

import dataclasses

import pytest

from pscanner.corpus import feature_projection as fp
from pscanner.corpus.features import (
    MarketMetadata,
    Trade,
    empty_market_state,
    empty_wallet_state,
)


def test_module_imports() -> None:
    """The module loads without errors."""
    from pscanner.corpus import feature_projection  # noqa: F401,PLC0415


def test_constants_are_load_bearing() -> None:
    """Constants exported by the module match what compute_features uses."""
    assert fp.CONFIDENCE_N_MIN == 20
    assert fp.HIGH_QUALITY_WIN_RATE_THRESHOLD == 0.55
    assert fp.SECONDS_PER_DAY == 86_400


def test_known_categories_covers_taxonomy() -> None:
    """All categories in the taxonomy are listed exactly once."""
    assert len(fp.KNOWN_CATEGORIES) == len(set(fp.KNOWN_CATEGORIES))
    # Sanity check: every entry is lowercase and non-empty
    for cat in fp.KNOWN_CATEGORIES:
        assert cat
        assert cat == cat.lower()


def test_feature_formula_is_frozen() -> None:
    """FeatureFormula instances reject mutation."""
    formula = fp.FeatureFormula(
        name="x",
        dtype="float",
        nullable=False,
        py=lambda _i: 1.0,
        sql="1.0",
    )

    try:
        formula.name = "y"  # ty:ignore[invalid-assignment]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("FeatureFormula should be frozen")


def test_feature_inputs_holds_state() -> None:
    """FeatureInputs bundles the four input types."""
    trade = Trade(
        tx_hash="t",
        asset_id="a",
        wallet_address="w",
        condition_id="c",
        outcome_side="YES",
        bs="BUY",
        price=0.5,
        size=100.0,
        notional_usd=50.0,
        ts=1_700_000_000,
        category="sports",
    )
    inputs = fp.FeatureInputs(
        wallet=empty_wallet_state(first_seen_ts=trade.ts),
        market=empty_market_state(market_age_start_ts=trade.ts),
        meta=MarketMetadata(condition_id="c", category="sports", closed_at=0, opened_at=0),
        trade=trade,
    )
    assert inputs.trade.tx_hash == "t"
    assert inputs.wallet.first_seen_ts == trade.ts


def test_render_sql_fragment_substitutes_known_placeholder() -> None:
    """render_sql_fragment swaps {scope.field} with the bound column ref."""
    template = "{w.prior_wins} > 0"
    rendered = fp.render_sql_fragment(template)
    assert rendered == "wa.prior_wins_w > 0"


def test_render_sql_fragment_raises_on_unbound_placeholder() -> None:
    """An unbound placeholder raises KeyError, not silently leaks braces."""
    with pytest.raises(KeyError, match="bogus"):
        fp.render_sql_fragment("{w.bogus_field} > 0")


def test_render_sql_fragment_passes_through_literal_braces_when_resolved() -> None:
    """All placeholders in a real template resolve."""
    template = (
        "CASE WHEN {w.prior_resolved_buys} > 0 "
        "THEN CAST({w.prior_wins} AS DOUBLE) / {w.prior_resolved_buys} "
        "ELSE NULL END"
    )
    rendered = fp.render_sql_fragment(template)
    assert "{" not in rendered
    assert "wa.prior_wins_w" in rendered
    assert "wa.prior_resolved_buys_w" in rendered
