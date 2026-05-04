"""Typed identifier wrappers for Kalshi entities.

Kalshi and Polymarket both use string identifiers, but they are structurally
and semantically incompatible:

- Kalshi tickers are human-readable slugs (e.g. ``"KXELONMARS-99"``).
- Polymarket ``ConditionId`` values are 0x-prefixed 66-character hex strings.
- Polymarket ``AssetId`` values are 78-digit decimal strings.

Mixing these across platforms produces silent bugs — a Kalshi ticker passed
to a Polymarket DB query would silently produce empty results or corrupt rows.
These ``NewType`` wrappers give ``ty`` enough structure to catch such bugs at
type-check time without any runtime cost.

Per the multi-platform RFC (Decision 2), each platform keeps its own ``ids``
module with no shared supertype. Cross-platform boundaries are mediated by
normalized dataclasses whose fields are plain ``str``.

Conventions:
- ``KalshiMarketTicker``: per-market ticker (e.g. ``"KXELONMARS-99"``).
- ``KalshiEventTicker``: event-level ticker grouping related markets
  (e.g. ``"KXELONMARS-99"`` — on simple events the event ticker equals the
  market ticker; on multi-outcome events it is shared across legs).
- ``KalshiSeriesTicker``: series ticker grouping related events
  (e.g. ``"KXELONMARS"``).
"""

from __future__ import annotations

from typing import NewType

KalshiMarketTicker = NewType("KalshiMarketTicker", str)
KalshiEventTicker = NewType("KalshiEventTicker", str)
KalshiSeriesTicker = NewType("KalshiSeriesTicker", str)
