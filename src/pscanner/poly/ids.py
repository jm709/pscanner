"""Typed identifier wrappers for Polymarket entities.

Polymarket exposes five distinct id flavors that all serialize as
strings on the wire. Mixing them produces silent bugs (e.g. looking up
``event_tag_cache`` with a numeric event_id when it actually stores
slugs). These ``NewType`` wrappers give ``ty`` enough structure to
catch those bugs at type-check time without any runtime cost.

Conventions:
- ``MarketId``: gamma's numeric market id (e.g. ``"540817"``).
- ``ConditionId``: on-chain condition id (0x-prefixed 66-char hex).
- ``AssetId``: CLOB token id (78-digit decimal string).
- ``EventId``: gamma's numeric event id (e.g. ``"16167"``).
- ``EventSlug``: gamma's URL slug (e.g. ``"trump-2024-election"``).
"""

from __future__ import annotations

from typing import NewType

MarketId = NewType("MarketId", str)
ConditionId = NewType("ConditionId", str)
AssetId = NewType("AssetId", str)
EventId = NewType("EventId", str)
EventSlug = NewType("EventSlug", str)
