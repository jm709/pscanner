"""Load-bearing parity test for FeatureRow ↔ TrainingExample field sets.

Every column in :class:`TrainingExample` is either:

- an identity column (set per-row from the source ``Trade``: ``tx_hash``,
  ``asset_id``, ``wallet_address``, ``condition_id``, ``trade_ts``,
  ``built_at``, ``platform``),
- ``label_won`` (the supervised target, derived from the resolution),
- or a feature column that mirrors a same-named field on
  :class:`FeatureRow`.

:data:`_FEATURE_ONLY_FIELDS` in ``examples.py`` drives the column copy via
``**asdict(features)``; if a future FeatureRow field doesn't have a
matching TrainingExample column the row would be silently inserted with
that column as NULL, which is the silent-feature-loss footgun this test
guards against.

See ``refactor-plan-corpus-ml.md`` T3.22.
"""

from __future__ import annotations

from dataclasses import fields

from pscanner.corpus.examples import _FEATURE_ONLY_FIELDS
from pscanner.corpus.features import FeatureRow
from pscanner.corpus.repos import TrainingExample

# Columns on TrainingExample that are NOT computed from FeatureRow; they
# come from the source Trade, the orchestrator (``now_ts``, ``platform``),
# or the resolution (``label_won``).
_IDENTITY_COLUMNS = frozenset(
    {
        "tx_hash",
        "asset_id",
        "wallet_address",
        "condition_id",
        "trade_ts",
        "built_at",
        "platform",
        "label_won",
    }
)

# Fields on FeatureRow that are NOT persisted to training_examples. The
# multi-label ``cat_*`` indicators DO persist; the only compute-only
# transient is ``market_categories`` (see feature_projection.py:
# ``project_to_sql=False``).
_COMPUTE_ONLY_FEATUREROW_FIELDS = frozenset({"market_categories"})


def test_feature_only_fields_match_featurerow_minus_compute_only() -> None:
    """Every persisted FeatureRow field has a TrainingExample column."""
    featurerow_fields = {f.name for f in fields(FeatureRow)}
    assert featurerow_fields - _COMPUTE_ONLY_FEATUREROW_FIELDS == _FEATURE_ONLY_FIELDS


def test_feature_only_fields_match_trainingexample_minus_identity() -> None:
    """Every TrainingExample column comes from either FeatureRow or identity."""
    trainingexample_fields = {f.name for f in fields(TrainingExample)}
    assert trainingexample_fields - _IDENTITY_COLUMNS == _FEATURE_ONLY_FIELDS


def test_no_overlap_between_identity_and_feature_columns() -> None:
    """A TrainingExample column can't be both identity and a FeatureRow mirror."""
    assert _IDENTITY_COLUMNS.isdisjoint(_FEATURE_ONLY_FIELDS)
