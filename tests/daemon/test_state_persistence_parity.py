"""Column-list ↔ dataclass parity regression for ``_state_persistence``.

The serializers in :mod:`pscanner.daemon._state_persistence` are driven
by hand-written column tuples (``WALLET_STATE_COLUMNS`` /
``MARKET_STATE_COLUMNS``). If a new field is added to ``WalletState`` or
``MarketState`` without a matching column entry, the silent column-drop
would only surface as missing-feature drift in the live daemon, not at
test time.

These tests assert that every dataclass field maps to exactly one column
and vice versa, so adding a field forces a corresponding column update
in the same PR.

If you're hitting these tests after adding a state field:

1. Add the column to the matching ``*_COLUMNS`` tuple in
   ``_state_persistence.py`` (in the correct serialization order).
2. Update ``*_to_row`` and ``*_from_row`` to include the new field.
3. If the field needs a special serialization name (e.g.
   ``_json``-suffixed for collections), add it to ``_JSON_SUFFIX_FIELDS``
   in this test module too.
"""

from __future__ import annotations

import dataclasses

from pscanner.corpus.features import (
    MarketState,
    WalletState,
    empty_market_state,
    empty_wallet_state,
)
from pscanner.daemon._state_persistence import (
    MARKET_STATE_COLUMNS,
    WALLET_STATE_COLUMNS,
    market_state_to_row,
    wallet_state_to_row,
)

# Columns added by the persist layer that don't correspond to a
# WalletState field (the wallet address and the per-wallet pending-buy
# queue, both owned by the provider rather than the state dataclass).
_WALLET_PERSIST_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "wallet_address",
        "unresolved_buys_json",
    }
)

# Columns added by the persist layer that don't correspond to a
# MarketState field (the market identifier and the per-market trader
# set, both owned by the provider rather than the state dataclass).
_MARKET_PERSIST_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "condition_id",
        "traders_json",
    }
)

# Fields whose column name takes a ``_json`` suffix because they're
# serialized as JSON arrays/objects (collections that don't round-trip
# through SQLite's primitive types).
_WALLET_JSON_SUFFIX_FIELDS: frozenset[str] = frozenset(
    {
        "recent_30d_trades",
        "category_counts",
    }
)
_MARKET_JSON_SUFFIX_FIELDS: frozenset[str] = frozenset(
    {
        "recent_prices",
    }
)


def _expected_columns(
    dataclass_type: type,
    *,
    persist_only: frozenset[str],
    json_suffix_fields: frozenset[str],
) -> frozenset[str]:
    """Derive the column-name set a serializer should produce for a dataclass."""
    from_fields = {
        f"{f.name}_json" if f.name in json_suffix_fields else f.name
        for f in dataclasses.fields(dataclass_type)
    }
    return frozenset(persist_only | from_fields)


def test_wallet_state_columns_cover_every_wallet_state_field() -> None:
    """A new WalletState field without a column update must fail this test."""
    expected = _expected_columns(
        WalletState,
        persist_only=_WALLET_PERSIST_ONLY_COLUMNS,
        json_suffix_fields=_WALLET_JSON_SUFFIX_FIELDS,
    )
    actual = frozenset(WALLET_STATE_COLUMNS)
    missing_columns = expected - actual
    unexpected_columns = actual - expected
    assert not missing_columns, (
        f"WALLET_STATE_COLUMNS is missing entries derived from WalletState: "
        f"{sorted(missing_columns)}. Add them in _state_persistence.py and "
        f"update wallet_state_to_row / wallet_state_from_row to cover the new field."
    )
    assert not unexpected_columns, (
        f"WALLET_STATE_COLUMNS has entries with no WalletState counterpart: "
        f"{sorted(unexpected_columns)}. Either add the field to WalletState, "
        f"remove the column, or list it in _WALLET_PERSIST_ONLY_COLUMNS in this test."
    )


def test_market_state_columns_cover_every_market_state_field() -> None:
    """A new MarketState field without a column update must fail this test."""
    expected = _expected_columns(
        MarketState,
        persist_only=_MARKET_PERSIST_ONLY_COLUMNS,
        json_suffix_fields=_MARKET_JSON_SUFFIX_FIELDS,
    )
    actual = frozenset(MARKET_STATE_COLUMNS)
    missing_columns = expected - actual
    unexpected_columns = actual - expected
    assert not missing_columns, (
        f"MARKET_STATE_COLUMNS is missing entries derived from MarketState: "
        f"{sorted(missing_columns)}. Add them in _state_persistence.py and "
        f"update market_state_to_row / market_state_from_row to cover the new field."
    )
    assert not unexpected_columns, (
        f"MARKET_STATE_COLUMNS has entries with no MarketState counterpart: "
        f"{sorted(unexpected_columns)}. Either add the field to MarketState, "
        f"remove the column, or list it in _MARKET_PERSIST_ONLY_COLUMNS in this test."
    )


def test_wallet_state_row_tuple_length_matches_column_tuple() -> None:
    """to_row must produce a tuple whose length matches the column tuple."""
    row = wallet_state_to_row(
        "0xtest",
        empty_wallet_state(first_seen_ts=0),
        unresolved_buys_json="[]",
    )
    assert len(row) == len(WALLET_STATE_COLUMNS), (
        f"wallet_state_to_row produced {len(row)} fields but WALLET_STATE_COLUMNS "
        f"declares {len(WALLET_STATE_COLUMNS)}. The tuple builder must stay in "
        f"sync with the column list."
    )


def test_market_state_row_tuple_length_matches_column_tuple() -> None:
    """to_row must produce a tuple whose length matches the column tuple."""
    row = market_state_to_row(
        "0xtest",
        empty_market_state(market_age_start_ts=0),
        traders=(),
    )
    assert len(row) == len(MARKET_STATE_COLUMNS), (
        f"market_state_to_row produced {len(row)} fields but MARKET_STATE_COLUMNS "
        f"declares {len(MARKET_STATE_COLUMNS)}. The tuple builder must stay in "
        f"sync with the column list."
    )
