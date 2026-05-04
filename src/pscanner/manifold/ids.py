"""Typed identifier wrappers for Manifold Markets entities.

Manifold Markets uses hash-based string IDs (not numeric or hex-prefixed). They
are URL-friendly opaque strings, e.g. ``"abc123XYZ"``.  Mixing them with
Polymarket or Kalshi identifiers would produce silent query bugs — these
``NewType`` wrappers give ``ty`` enough information to catch such mistakes at
type-check time with zero runtime cost.

Conventions:
- ``ManifoldMarketId``: Manifold's opaque hash market ID (e.g. ``"kjPkT2HECV"``).
  This is the ``id`` field on contract objects, NOT the slug.
- ``ManifoldUserId``: Manifold's opaque hash user ID (e.g. ``"igi2zGXsfxYPgB0DJTXVJVmwCOr2"``).
  Distinct from the username (a human-readable handle).

Do not alias these to ``pscanner.poly.ids`` types — a Manifold hash ID passed
to a Polymarket hex-condition-ID query would silently corrupt results.
"""

from __future__ import annotations

from typing import NewType

ManifoldMarketId = NewType("ManifoldMarketId", str)
ManifoldUserId = NewType("ManifoldUserId", str)
