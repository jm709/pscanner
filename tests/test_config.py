"""Tests for ``pscanner.config`` typed sections."""

from __future__ import annotations

from pscanner.config import (
    ClusterConfig,
    Config,
    MoveAttributionConfig,
    PaperTradingConfig,
    WorkerSinkConfig,
)


def test_move_attribution_defaults() -> None:
    cfg = MoveAttributionConfig()
    assert cfg.enabled is True
    assert cfg.trigger_detectors == ("velocity", "convergence")
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


def test_paper_trading_defaults() -> None:
    cfg = PaperTradingConfig()
    assert cfg.enabled is False  # opt-in
    assert cfg.starting_bankroll_usd == 1000.0
    assert cfg.position_fraction == 0.01
    assert cfg.min_weighted_edge == 0.0
    assert cfg.min_position_cost_usd == 0.50
    assert cfg.resolver_scan_interval_seconds == 300.0


def test_paper_trading_attached_to_root_config() -> None:
    cfg = Config()
    assert isinstance(cfg.paper_trading, PaperTradingConfig)
    assert cfg.paper_trading.enabled is False


def test_cluster_max_co_trade_group_size_default() -> None:
    cfg = ClusterConfig()
    assert cfg.max_co_trade_group_size == 100


def test_worker_sink_config_defaults() -> None:
    cfg = WorkerSinkConfig()
    assert cfg.velocity_maxsize == 4096
    assert cfg.stats_interval_seconds == 60

    root = Config()
    assert root.worker_sink == cfg
