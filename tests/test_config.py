"""Tests for ``pscanner.config`` typed sections."""

from __future__ import annotations

from pscanner.config import Config, MoveAttributionConfig


def test_move_attribution_defaults() -> None:
    cfg = MoveAttributionConfig()
    assert cfg.enabled is True
    assert cfg.trigger_detectors == ("velocity", "mispricing", "convergence")
    assert cfg.lookback_seconds_baseline == 86400
    assert cfg.backwalk_multiplier == 3.0
    assert cfg.backwalk_check_window_seconds == 300
    assert cfg.max_backwalk_seconds == 7200
    assert cfg.burst_bucket_seconds == 60
    assert cfg.min_burst_wallets == 4
    assert cfg.max_burst_size_cv == 0.4
    assert cfg.max_burst_hits_per_alert == 5
    assert cfg.max_contributors_per_burst == 50


def test_move_attribution_attached_to_root_config() -> None:
    cfg = Config()
    assert isinstance(cfg.move_attribution, MoveAttributionConfig)
    assert cfg.move_attribution.enabled is True
