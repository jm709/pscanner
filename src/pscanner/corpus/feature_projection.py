"""Single source of truth for the ML feature projection (#145).

See ``docs/superpowers/plans/2026-05-19-issue-145-feature-projection.md`` for
the architectural rationale. In short: the same FeatureRow is computed by
three code paths (Python streaming via ``compute_features``, live daemon via
``LiveHistoryProvider``, DuckDB batch via ``_final_join_to_v2``). This module
holds the canonical definitions and is consumed by all three.
"""

from __future__ import annotations
